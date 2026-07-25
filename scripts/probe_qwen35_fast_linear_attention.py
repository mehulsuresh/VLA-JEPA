#!/usr/bin/env python3
"""Fail-closed import and CUDA-kernel probe for Qwen3.5 fast linear attention."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any


DEFAULT_FLA_VERSION = "0.5.1"
DEFAULT_CAUSAL_CONV1D_VERSION = "1.6.2.post1"
DEFAULT_TRANSFORMERS_VERSION = "5.13.1"
DEFAULT_TILELANG_VERSION = "0.1.9"
DEFAULT_TVM_FFI_VERSION = "0.1.10"
QWEN35_2B_LINEAR_ATTN_SHAPE = {
    "hidden_size": 2048,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
}
REQUIRED_MODELING_BINDINGS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "FusedRMSNormGated",
)
KNOWN_MOGE_OPTIONAL_DEPENDENCY_FAILURES = frozenset(
    {
        "moge 2.0.0 requires gradio, which is not installed.",
        "moge 2.0.0 requires opencv-python, which is not installed.",
    }
)
PIP_CHECK_SUCCESS_LINE = "No broken requirements found."
HOPPER_REFERENCE_SEQUENCE_LENGTH = 65
HOPPER_REFERENCE_FORWARD_MAX_RELATIVE_L2 = 0.10
HOPPER_REFERENCE_FORWARD_MIN_COSINE = 0.995
HOPPER_REFERENCE_GRADIENT_MAX_RELATIVE_L2 = 0.20
HOPPER_REFERENCE_GRADIENT_MIN_COSINE = 0.98


def _parse_compute_capability(value: str) -> tuple[int, int]:
    parts = value.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(
            f"compute capability must have MAJOR.MINOR form, got {value!r}"
        )
    return int(parts[0]), int(parts[1])


def _require_exact_package_version(distribution: str, expected: str) -> str:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution {distribution!r} is not installed") from exc
    if actual != expected:
        raise RuntimeError(
            f"{distribution} version mismatch: expected {expected!r}, found {actual!r}"
        )
    return actual


def _require_modeling_bindings(modeling: ModuleType | Any) -> dict[str, Any]:
    unavailable = [
        name
        for name in REQUIRED_MODELING_BINDINGS
        if not callable(getattr(modeling, name, None))
    ]
    if unavailable:
        raise RuntimeError(
            "Transformers did not bind every Qwen3.5 fast-path implementation: "
            + ", ".join(unavailable)
        )
    return {name: getattr(modeling, name) for name in REQUIRED_MODELING_BINDINGS}


def _cuda_is_available() -> bool:
    torch_module = _import_runtime_module("torch", "torch")
    cuda_module = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda_module, "is_available", None)
    if not callable(is_available):
        raise RuntimeError("torch.cuda.is_available is unavailable or not callable")
    return bool(is_available())


def _resolve_transformers_bindings(
    modeling: ModuleType | Any,
    package_bindings: dict[str, Any],
    *,
    allow_unbound_for_cpu_build: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not allow_unbound_for_cpu_build:
        bindings = _require_modeling_bindings(modeling)
        mismatched = [
            name
            for name in REQUIRED_MODELING_BINDINGS
            if bindings[name] is not package_bindings[name]
        ]
        if mismatched:
            raise RuntimeError(
                "Transformers Qwen3.5 bindings do not reference the installed "
                "fused packages: " + ", ".join(mismatched)
            )
        return bindings, {
            "mode": "strict-transformers-identity",
            "bound": list(REQUIRED_MODELING_BINDINGS),
            "unbound": [],
        }

    if _cuda_is_available():
        raise RuntimeError(
            "--cpu-build-imports-only is valid only when CUDA is unavailable"
        )

    missing = [
        name for name in REQUIRED_MODELING_BINDINGS if not hasattr(modeling, name)
    ]
    if missing:
        raise RuntimeError(
            "Transformers Qwen3.5 module is missing required CPU-build globals: "
            + ", ".join(missing)
        )
    transformers_values = {
        name: getattr(modeling, name) for name in REQUIRED_MODELING_BINDINGS
    }
    mismatched = [
        name
        for name, value in transformers_values.items()
        if value is not None and value is not package_bindings[name]
    ]
    if mismatched:
        raise RuntimeError(
            "Transformers Qwen3.5 CPU-build globals reference unexpected "
            "implementations: " + ", ".join(mismatched)
        )
    bound = [
        name
        for name, value in transformers_values.items()
        if value is package_bindings[name]
    ]
    unbound = [name for name, value in transformers_values.items() if value is None]
    return package_bindings, {
        "mode": "cpu-build-no-cuda",
        "bound": bound,
        "unbound": unbound,
    }


def _import_runtime_module(distribution: str, module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"{distribution} is installed but runtime module {module_name!r} failed to import: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _describe_import(module: ModuleType) -> dict[str, str | None]:
    return {
        "module": module.__name__,
        "path": str(module.__file__) if getattr(module, "__file__", None) else None,
    }


def _validate_pip_check_result(
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    allow_known_moge_optional_dependencies: bool,
) -> dict[str, Any]:
    lines = [
        line.strip()
        for stream in (stdout, stderr)
        for line in stream.splitlines()
        if line.strip()
    ]
    allowed = (
        KNOWN_MOGE_OPTIONAL_DEPENDENCY_FAILURES
        if allow_known_moge_optional_dependencies
        else frozenset()
    )
    ignored = sorted(line for line in lines if line in allowed)
    unexpected = sorted(
        line
        for line in lines
        if line not in allowed and line != PIP_CHECK_SUCCESS_LINE
    )

    if unexpected:
        raise RuntimeError(
            "pip check reported unapproved dependency failures: " + " | ".join(unexpected)
        )
    if returncode == 0:
        if ignored:
            raise RuntimeError(
                "pip check returned success while reporting dependency failures: "
                + " | ".join(ignored)
            )
    elif not ignored:
        raise RuntimeError(
            f"pip check exited {returncode} without an approved diagnostic"
        )

    return {
        "status": "ok",
        "pip_check_returncode": returncode,
        "allowed_failures": ignored,
    }


def _run_python_dependency_check(
    *, allow_known_moge_optional_dependencies: bool
) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        text=True,
        capture_output=True,
    )
    return _validate_pip_check_result(
        result.returncode,
        result.stdout,
        result.stderr,
        allow_known_moge_optional_dependencies=allow_known_moge_optional_dependencies,
    )


def _load_integrations(
    expected_fla_version: str,
    expected_causal_conv1d_version: str,
    expected_transformers_version: str,
    expected_tilelang_version: str,
    expected_tvm_ffi_version: str,
    *,
    allow_unbound_transformers_for_cpu_build: bool = False,
):
    versions = {
        "flash-linear-attention": _require_exact_package_version(
            "flash-linear-attention", expected_fla_version
        ),
        "causal-conv1d": _require_exact_package_version(
            "causal-conv1d", expected_causal_conv1d_version
        ),
        "transformers": _require_exact_package_version(
            "transformers", expected_transformers_version
        ),
        "tilelang": _require_exact_package_version(
            "tilelang", expected_tilelang_version
        ),
        "apache-tvm-ffi": _require_exact_package_version(
            "apache-tvm-ffi", expected_tvm_ffi_version
        ),
    }

    tvm_ffi_module = _import_runtime_module("apache-tvm-ffi", "tvm_ffi")
    tilelang_module = _import_runtime_module("tilelang", "tilelang")
    causal_conv1d_module = _import_runtime_module("causal-conv1d", "causal_conv1d")
    fla_modules_module = _import_runtime_module("flash-linear-attention", "fla.modules")
    fla_ops_module = _import_runtime_module(
        "flash-linear-attention", "fla.ops.gated_delta_rule"
    )
    modeling_qwen3_5 = _import_runtime_module(
        "transformers", "transformers.models.qwen3_5.modeling_qwen3_5"
    )
    configuration_qwen3_5 = _import_runtime_module(
        "transformers", "transformers.models.qwen3_5.configuration_qwen3_5"
    )

    causal_conv1d_fn = getattr(causal_conv1d_module, "causal_conv1d_fn", None)
    causal_conv1d_update = getattr(causal_conv1d_module, "causal_conv1d_update", None)
    FusedRMSNormGated = getattr(fla_modules_module, "FusedRMSNormGated", None)
    chunk_gated_delta_rule = getattr(fla_ops_module, "chunk_gated_delta_rule", None)
    fused_recurrent_gated_delta_rule = getattr(
        fla_ops_module, "fused_recurrent_gated_delta_rule", None
    )

    package_bindings = _require_modeling_bindings(
        SimpleNamespace(
            causal_conv1d_fn=causal_conv1d_fn,
            causal_conv1d_update=causal_conv1d_update,
            chunk_gated_delta_rule=chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule,
            FusedRMSNormGated=FusedRMSNormGated,
        )
    )
    Qwen3_5TextConfig = getattr(configuration_qwen3_5, "Qwen3_5TextConfig", None)
    if not isinstance(Qwen3_5TextConfig, type):
        raise RuntimeError(
            "Transformers Qwen3.5 configuration module did not expose Qwen3_5TextConfig"
        )

    bindings, binding_validation = _resolve_transformers_bindings(
        modeling_qwen3_5,
        package_bindings,
        allow_unbound_for_cpu_build=(
            allow_unbound_transformers_for_cpu_build
        ),
    )
    imported_modules = {
        "tilelang": _describe_import(tilelang_module),
        "apache-tvm-ffi": _describe_import(tvm_ffi_module),
        "transformers-qwen3.5-modeling": _describe_import(modeling_qwen3_5),
        "transformers-qwen3.5-configuration": _describe_import(configuration_qwen3_5),
    }
    return (
        versions,
        modeling_qwen3_5,
        Qwen3_5TextConfig,
        bindings,
        binding_validation,
        imported_modules,
    )


def _require_finite(torch: Any, name: str, tensor: Any) -> None:
    if tensor is None:
        raise RuntimeError(f"{name} is missing")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")


def _build_qwen35_2b_linear_attention_layer(
    torch: Any,
    modeling: ModuleType,
    config_cls: type,
    bindings: dict[str, Any],
    device: Any,
):
    config = config_cls(
        vocab_size=248320,
        hidden_size=QWEN35_2B_LINEAR_ATTN_SHAPE["hidden_size"],
        intermediate_size=6144,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        head_dim=256,
        layer_types=["linear_attention"],
        linear_conv_kernel_dim=QWEN35_2B_LINEAR_ATTN_SHAPE["linear_conv_kernel_dim"],
        linear_key_head_dim=QWEN35_2B_LINEAR_ATTN_SHAPE["linear_key_head_dim"],
        linear_value_head_dim=QWEN35_2B_LINEAR_ATTN_SHAPE["linear_value_head_dim"],
        linear_num_key_heads=QWEN35_2B_LINEAR_ATTN_SHAPE["linear_num_key_heads"],
        linear_num_value_heads=QWEN35_2B_LINEAR_ATTN_SHAPE["linear_num_value_heads"],
    )
    layer = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(
        device=device, dtype=torch.bfloat16
    )
    layer.train()

    captured = {
        "causal_conv1d_fn": layer.causal_conv1d_fn,
        "causal_conv1d_update": layer.causal_conv1d_update,
        "chunk_gated_delta_rule": layer.chunk_gated_delta_rule,
        "fused_recurrent_gated_delta_rule": layer.recurrent_gated_delta_rule,
    }
    wrong = [name for name, value in captured.items() if value is not bindings[name]]
    if wrong:
        raise RuntimeError(
            "Qwen3.5 layer captured fallback implementations instead of fused kernels: "
            + ", ".join(wrong)
        )
    if not isinstance(layer.norm, bindings["FusedRMSNormGated"]):
        raise RuntimeError("Qwen3.5 layer did not instantiate FusedRMSNormGated")
    return layer


def _probe_inference_only_kernels(torch: Any, layer: Any, device: Any, batch_size: int) -> None:
    shape = QWEN35_2B_LINEAR_ATTN_SHAPE
    conv_dim = (
        2 * shape["linear_num_key_heads"] * shape["linear_key_head_dim"]
        + shape["linear_num_value_heads"] * shape["linear_value_head_dim"]
    )
    with torch.no_grad():
        update_x = torch.randn(batch_size, conv_dim, device=device, dtype=torch.bfloat16)
        conv_state = torch.zeros(
            batch_size,
            conv_dim,
            shape["linear_conv_kernel_dim"],
            device=device,
            dtype=torch.bfloat16,
        )
        update_out = layer.causal_conv1d_update(
            update_x,
            conv_state,
            layer.conv1d.weight.squeeze(1),
            layer.conv1d.bias,
            layer.activation,
        )
        _require_finite(torch, "causal_conv1d_update output", update_out)

        recurrent_shape = (
            batch_size,
            1,
            shape["linear_num_value_heads"],
            shape["linear_key_head_dim"],
        )
        q = torch.randn(recurrent_shape, device=device, dtype=torch.bfloat16)
        k = torch.randn(recurrent_shape, device=device, dtype=torch.bfloat16)
        v = torch.randn(recurrent_shape, device=device, dtype=torch.bfloat16)
        g = -torch.rand(
            batch_size,
            1,
            shape["linear_num_value_heads"],
            device=device,
            dtype=torch.float32,
        )
        beta = torch.rand(
            batch_size,
            1,
            shape["linear_num_value_heads"],
            device=device,
            dtype=torch.bfloat16,
        )
        recurrent_out, recurrent_state = layer.recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        _require_finite(torch, "fused recurrent output", recurrent_out)
        _require_finite(torch, "fused recurrent state", recurrent_state)


def _probe_training_kernels(
    torch: Any,
    layer: Any,
    device: Any,
    batch_size: int,
    sequence_length: int,
) -> float:
    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        QWEN35_2B_LINEAR_ATTN_SHAPE["hidden_size"],
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.02)
    hidden_states.requires_grad_(True)
    output = layer(hidden_states)
    _require_finite(torch, "Qwen3.5 fused linear-attention output", output)

    loss = output.float().square().mean()
    _require_finite(torch, "Qwen3.5 fused linear-attention loss", loss)
    loss.backward()
    torch.cuda.synchronize(device)

    gradients = {
        "input gradient": hidden_states.grad,
        "causal-conv1d weight gradient": layer.conv1d.weight.grad,
        "QKV projection gradient": layer.in_proj_qkv.weight.grad,
        "delta A_log gradient": layer.A_log.grad,
        "delta dt_bias gradient": layer.dt_bias.grad,
        "fused RMSNorm gradient": layer.norm.weight.grad,
        "output projection gradient": layer.out_proj.weight.grad,
    }
    for name, gradient in gradients.items():
        _require_finite(torch, name, gradient)
    return float(loss.detach().item())


def _pytorch_gated_delta_rule_reference(
    torch: Any,
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
) -> Any:
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise RuntimeError("reference gate expects q/k/v tensors with [B, T, H, D] shape")
    batch_size, sequence_length, key_heads, key_dim = q.shape
    if tuple(v.shape[:2]) != (batch_size, sequence_length):
        raise RuntimeError("reference gate q/k/v batch and sequence dimensions differ")
    value_heads, value_dim = v.shape[2:]
    if value_heads % key_heads:
        raise RuntimeError("reference gate requires value heads divisible by key heads")
    if tuple(g.shape) != (batch_size, sequence_length, value_heads):
        raise RuntimeError("reference gate g shape does not match value heads")
    if tuple(beta.shape) != (batch_size, sequence_length, value_heads):
        raise RuntimeError("reference gate beta shape does not match value heads")

    value_heads_per_key_head = value_heads // key_heads
    q = q.repeat_interleave(value_heads_per_key_head, dim=2)
    k = k.repeat_interleave(value_heads_per_key_head, dim=2)
    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    scale = key_dim**-0.5
    state = torch.zeros(
        batch_size,
        value_heads,
        key_dim,
        value_dim,
        dtype=q.dtype,
        device=q.device,
    )
    outputs = []
    for step in range(sequence_length):
        state = state * torch.exp(g[:, step, :, None, None])
        predicted_value = torch.einsum("bhkv,bhk->bhv", state, k[:, step])
        delta = beta[:, step, :, None] * (v[:, step] - predicted_value)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, step], delta)
        outputs.append(
            scale * torch.einsum("bhkv,bhk->bhv", state, q[:, step])
        )
    return torch.stack(outputs, dim=1)


def _tensor_reference_metrics(torch: Any, actual: Any, reference: Any) -> dict[str, float]:
    actual = actual.detach().float()
    reference = reference.detach().float()
    difference = actual - reference
    actual_norm = float(torch.linalg.vector_norm(actual).item())
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    denominator = max(reference_norm, 1e-12)
    cosine_denominator = max(actual_norm * reference_norm, 1e-24)
    cosine = float(torch.sum(actual * reference).item()) / cosine_denominator
    return {
        "relative_l2": difference_norm / denominator,
        "cosine": cosine,
        "max_abs": float(torch.max(torch.abs(difference)).item()),
        "reference_norm": reference_norm,
    }


def _validate_reference_metrics(
    name: str,
    metrics: dict[str, float],
    *,
    max_relative_l2: float,
    min_cosine: float,
) -> None:
    non_finite = [key for key, value in metrics.items() if not math.isfinite(value)]
    if non_finite:
        raise RuntimeError(
            f"Hopper numerical reference metrics for {name} are non-finite: "
            + ", ".join(non_finite)
        )
    if metrics["relative_l2"] > max_relative_l2 or metrics["cosine"] < min_cosine:
        raise RuntimeError(
            f"Hopper fused gradient/reference mismatch for {name}: "
            f"relative_l2={metrics['relative_l2']:.6g} (max {max_relative_l2}), "
            f"cosine={metrics['cosine']:.6g} (min {min_cosine})"
        )


def _probe_hopper_chunk_gradient_reference(
    torch: Any,
    chunk_gated_delta_rule: Any,
    device: Any,
) -> dict[str, dict[str, float]]:
    # T=65 crosses the fused kernel's 64-token chunk boundary while keeping the
    # pure-PyTorch recurrence small enough for an image/runtime preflight.
    batch_size = 1
    sequence_length = HOPPER_REFERENCE_SEQUENCE_LENGTH
    heads = 2
    key_dim = 8
    value_dim = 8

    torch.manual_seed(20260715)
    torch.cuda.manual_seed_all(20260715)
    q = torch.empty(
        batch_size,
        sequence_length,
        heads,
        key_dim,
        device=device,
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.2).requires_grad_(True)
    k = torch.empty_like(q).normal_(mean=0.0, std=0.2).requires_grad_(True)
    v = torch.empty(
        batch_size,
        sequence_length,
        heads,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.2).requires_grad_(True)
    g = torch.empty(
        batch_size,
        sequence_length,
        heads,
        device=device,
        dtype=torch.float32,
    ).uniform_(-1.5, -0.1).requires_grad_(True)
    beta = torch.empty(
        batch_size,
        sequence_length,
        heads,
        device=device,
        dtype=torch.bfloat16,
    ).uniform_(0.05, 0.95).requires_grad_(True)
    fused_inputs = (q, k, v, g, beta)

    fused_output, _ = chunk_gated_delta_rule(
        *fused_inputs,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    _require_finite(torch, "Hopper fused chunk output", fused_output)
    upstream = torch.linspace(
        -0.5,
        0.5,
        fused_output.numel(),
        device=device,
        dtype=torch.float32,
    ).reshape(fused_output.shape)
    fused_loss = torch.sum(fused_output.float() * upstream)
    fused_gradients = torch.autograd.grad(fused_loss, fused_inputs)

    reference_inputs = tuple(
        tensor.detach().float().requires_grad_(True) for tensor in fused_inputs
    )
    reference_output = _pytorch_gated_delta_rule_reference(
        torch, *reference_inputs
    )
    reference_loss = torch.sum(reference_output * upstream)
    reference_gradients = torch.autograd.grad(reference_loss, reference_inputs)
    torch.cuda.synchronize(device)

    metrics = {
        "output": _tensor_reference_metrics(torch, fused_output, reference_output),
    }
    for name, fused_gradient, reference_gradient in zip(
        ("q_gradient", "k_gradient", "v_gradient", "g_gradient", "beta_gradient"),
        fused_gradients,
        reference_gradients,
        strict=True,
    ):
        _require_finite(torch, f"Hopper fused {name}", fused_gradient)
        _require_finite(torch, f"Hopper reference {name}", reference_gradient)
        metrics[name] = _tensor_reference_metrics(
            torch, fused_gradient, reference_gradient
        )

    _validate_reference_metrics(
        "output",
        metrics["output"],
        max_relative_l2=HOPPER_REFERENCE_FORWARD_MAX_RELATIVE_L2,
        min_cosine=HOPPER_REFERENCE_FORWARD_MIN_COSINE,
    )
    for name in ("q_gradient", "k_gradient", "v_gradient", "g_gradient", "beta_gradient"):
        _validate_reference_metrics(
            name,
            metrics[name],
            max_relative_l2=HOPPER_REFERENCE_GRADIENT_MAX_RELATIVE_L2,
            min_cosine=HOPPER_REFERENCE_GRADIENT_MIN_COSINE,
        )
    return metrics


def _run_kernel_probe(
    modeling: ModuleType,
    config_cls: type,
    bindings: dict[str, Any],
    expected_compute_capability: tuple[int, int],
    device_index: int,
    batch_size: int,
    sequence_length: int,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fused linear-attention kernel probe")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {device_index} is unavailable; device_count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda", device_index)
    actual_capability = tuple(torch.cuda.get_device_capability(device))
    if actual_capability != expected_compute_capability:
        raise RuntimeError(
            "CUDA compute capability mismatch: "
            f"expected {expected_compute_capability}, found {actual_capability}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    if batch_size <= 0 or sequence_length <= 0:
        raise RuntimeError("batch size and sequence length must be positive")

    torch.manual_seed(20260714)
    torch.cuda.manual_seed_all(20260714)
    with torch.cuda.device(device):
        layer = _build_qwen35_2b_linear_attention_layer(
            torch, modeling, config_cls, bindings, device
        )
        _probe_inference_only_kernels(torch, layer, device, batch_size)
        loss = _probe_training_kernels(
            torch, layer, device, batch_size, sequence_length
        )
        hopper_reference = _probe_hopper_chunk_gradient_reference(
            torch, bindings["chunk_gated_delta_rule"], device
        )

    if not math.isfinite(loss):
        raise RuntimeError(f"kernel probe returned non-finite loss {loss}")
    return {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": f"{actual_capability[0]}.{actual_capability[1]}",
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "dtype": "bfloat16",
        "loss": loss,
        "hopper_chunk_gradient_reference": hopper_reference,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--imports-only",
        action="store_true",
        help=(
            "import every pinned runtime, validate Transformers Qwen3.5 fused bindings, "
            "and skip CUDA kernels"
        ),
    )
    mode.add_argument(
        "--cpu-build-imports-only",
        action="store_true",
        help=(
            "for a CUDA-less Docker build only, import and validate every pinned "
            "runtime and fused package function while permitting Transformers' "
            "CUDA-gated Qwen3.5 globals to remain unbound"
        ),
    )
    mode.add_argument(
        "--check-python-dependencies",
        action="store_true",
        help="run pip check through the fail-closed Docker dependency policy",
    )
    parser.add_argument(
        "--allow-known-moge-optional-deps",
        action="store_true",
        help=(
            "only in dependency-check mode, permit the exact missing gradio and "
            "opencv-python diagnostics from the pinned MoGe package"
        ),
    )
    parser.add_argument(
        "--expected-fla-version",
        default=DEFAULT_FLA_VERSION,
    )
    parser.add_argument(
        "--expected-causal-conv1d-version",
        default=DEFAULT_CAUSAL_CONV1D_VERSION,
    )
    parser.add_argument(
        "--expected-transformers-version",
        default=DEFAULT_TRANSFORMERS_VERSION,
    )
    parser.add_argument(
        "--expected-tilelang-version",
        default=DEFAULT_TILELANG_VERSION,
    )
    parser.add_argument(
        "--expected-tvm-ffi-version",
        default=DEFAULT_TVM_FFI_VERSION,
    )
    parser.add_argument(
        "--expected-compute-capability",
        default=(9, 0),
        type=_parse_compute_capability,
        metavar="MAJOR.MINOR",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.allow_known_moge_optional_deps and not args.check_python_dependencies:
        parser.error(
            "--allow-known-moge-optional-deps requires --check-python-dependencies"
        )
    if args.check_python_dependencies:
        print(
            json.dumps(
                _run_python_dependency_check(
                    allow_known_moge_optional_dependencies=(
                        args.allow_known_moge_optional_deps
                    )
                ),
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    (
        versions,
        modeling,
        config_cls,
        bindings,
        binding_validation,
        imported_modules,
    ) = _load_integrations(
        args.expected_fla_version,
        args.expected_causal_conv1d_version,
        args.expected_transformers_version,
        args.expected_tilelang_version,
        args.expected_tvm_ffi_version,
        allow_unbound_transformers_for_cpu_build=(
            args.cpu_build_imports_only
        ),
    )
    if args.cpu_build_imports_only:
        mode_name = "cpu-build-imports-only"
    elif args.imports_only:
        mode_name = "imports-only"
    else:
        mode_name = "sm90-kernels"
    result: dict[str, Any] = {
        "status": "ok",
        "mode": mode_name,
        "versions": versions,
        "runtime_imports": imported_modules,
        "transformers_bindings": binding_validation["bound"],
        "transformers_binding_validation": binding_validation,
    }
    if not args.imports_only and not args.cpu_build_imports_only:
        result["kernel_probe"] = _run_kernel_probe(
            modeling,
            config_cls,
            bindings,
            args.expected_compute_capability,
            args.device_index,
            args.batch_size,
            args.sequence_length,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
