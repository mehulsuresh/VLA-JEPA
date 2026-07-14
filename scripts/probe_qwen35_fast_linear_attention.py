#!/usr/bin/env python3
"""Fail-closed import and CUDA-kernel probe for Qwen3.5 fast linear attention."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
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


def _load_integrations(
    expected_fla_version: str,
    expected_causal_conv1d_version: str,
    expected_transformers_version: str,
    expected_tilelang_version: str,
    expected_tvm_ffi_version: str,
    *,
    require_transformers_bindings: bool,
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

    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
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
    if not require_transformers_bindings:
        return versions, None, None, package_bindings

    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    bindings = _require_modeling_bindings(modeling_qwen3_5)
    mismatched = [
        name for name in REQUIRED_MODELING_BINDINGS if bindings[name] is not package_bindings[name]
    ]
    if mismatched:
        raise RuntimeError(
            "Transformers Qwen3.5 bindings do not reference the installed fused packages: "
            + ", ".join(mismatched)
        )
    return versions, modeling_qwen3_5, Qwen3_5TextConfig, bindings


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
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports-only",
        action="store_true",
        help="verify exact package versions and package fast-path symbols without launching CUDA kernels",
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
    args = _build_parser().parse_args()
    versions, modeling, config_cls, bindings = _load_integrations(
        args.expected_fla_version,
        args.expected_causal_conv1d_version,
        args.expected_transformers_version,
        args.expected_tilelang_version,
        args.expected_tvm_ffi_version,
        require_transformers_bindings=not args.imports_only,
    )
    result: dict[str, Any] = {
        "status": "ok",
        "mode": "imports-only" if args.imports_only else "sm90-kernels",
        "versions": versions,
        "transformers_bindings": list(bindings),
    }
    if not args.imports_only:
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
