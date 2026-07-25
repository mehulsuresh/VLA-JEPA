#!/usr/bin/env python3
"""Human-facing, config-only H100x8 training control.

This module deliberately accepts no training hyperparameter overrides.  It
validates and prints the complete YAML contract, then gives Accelerate a
resolved temporary YAML as its only training argument.  The only runtime
inputs are run identity and an optional checkpoint to resume.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from starVLA.action_representation import (
    Q01_Q99_UNCLIPPED,
    REALMAN_18D_ACTION_CONTRACT,
    load_openpi_realman_union_statistics,
    split_manifest_sha256_without_statistics_binding,
)
from starVLA.canonical_contract import (
    CANONICAL_EVAL_SELECTION_ALGORITHM,
    canonical_action_sidecar_variant,
    canonical_adapter_contract_sha256,
)
from starVLA.eval_sampling_policy import (
    derive_episode_holdout_sampling_plan,
    validate_holdout_sampling_policy,
)
from starVLA.holdout_selection_contract import (
    build_realman_holdout_selection_contract,
    configured_realman_holdout_seed_text,
    holdout_selection_contract_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
CHECKPOINT_RE = re.compile(r"steps_([0-9]+)")
HELPER_REPOSITORY_RE = re.compile(r"[a-z][a-z0-9_-]*")
CONTAINER_IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
AUTHORITATIVE_SEMANTIC_ENV_VARS = frozenset(
    {
        "ACCELERATE_CONFIG_FILE",
        "ACCELERATE_DISTRIBUTED_TYPE",
        "ACCELERATE_DYNAMO_BACKEND",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_NUM_MACHINES",
        "ACCELERATE_NUM_PROCESSES",
        "ACCELERATE_USE_DEEPSPEED",
        "CONFIG_YAML",
        "DATALOADER_NUM_WORKERS",
        "DATALOADER_PERSISTENT_WORKERS",
        "DATALOADER_PREFETCH_FACTOR",
        "DATALOADER_TIMEOUT_SECONDS",
        "DDP_BUCKET_CAP_MB",
        "DDP_GRADIENT_AS_BUCKET_VIEW",
        "DDP_STATIC_GRAPH",
        "EPOCHS",
        "EVAL_INTERVAL",
        "FIND_UNUSED_PARAMETERS",
        "GLOO_SOCKET_IFNAME",
        "LOGGING_FREQUENCY",
        "MAIN_PROCESS_PORT",
        "MAX_TRAIN_STEPS",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NUM_PROCESSES",
        "NUM_WARMUP_STEPS",
        "PER_DEVICE_BATCH_SIZE",
        "RUN_ID",
        "SAVE_INTERVAL",
        "STARVLA_ALLOW_COMPILE_WITH_DEEPSPEED",
        "STARVLA_ALLOW_TORCH_COMPILE",
        "STARVLA_DATASET_TIMING",
        "STARVLA_DATASET_TIMING_EVERY",
        "STARVLA_DATASET_TIMING_SLOW_SECONDS",
        "STARVLA_DEEPSPEED_STAGE",
        "STARVLA_DETAILED_TIMING",
        "STARVLA_DETAILED_TIMING_FREQUENCY",
        "STARVLA_DISABLE_FLASH_ATTN_WORLD_MODEL",
        "STARVLA_DISABLE_FLASH_ATTN_PROMOTION",
        "STARVLA_DISABLE_TORCH_COMPILE",
        "STARVLA_ENABLE_FLASH_ATTN_WORLD_MODEL",
        "STARVLA_USE_DEEPSPEED",
        "TORCH_COMPILE_DISABLE",
        "TORCHDYNAMO_DISABLE",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "VLA_JEPA_DISABLE_AUTOGRAD_MULTITHREADING",
        "VLA_JEPA_MAIN_TORCH_INTEROP_THREADS",
        "VLA_JEPA_MAIN_TORCH_THREADS",
        "VIDEO_BACKEND",
        "VIDEO_BACKEND_NUM_THREADS",
    }
)

REALMAN_LEROBOT_PROFILE = "realman_lerobot"
LIBERO_LEROBOT_PROFILE = "libero_lerobot"
CANONICAL_GCS_PROFILE = "canonical_gcs"
SUPPORTED_DATASET_PROFILES = {
    REALMAN_LEROBOT_PROFILE,
    LIBERO_LEROBOT_PROFILE,
    CANONICAL_GCS_PROFILE,
}
CHECKPOINT_HANDOFF_SMOKE_CONTRACT = {
    "schema": "realman-checkpoint-handoff-smoke-stage-v1",
    "scope": "checkpoint_handoff_only",
    "model_quality_claim_allowed": False,
    "expected_global_batch_rows": 128,
    "expected_optimizer_steps": 1,
}
PRODUCTION_HANDOFF_VALIDATION_CONTRACT = {
    "schema": "realman-production-handoff-validation-stage-v1",
    "scope": "checkpoint_handoff_only",
    "model_quality_claim_allowed": False,
    "production_frozen_view_required": True,
}


class PlanError(RuntimeError):
    """A fail-closed training-plan validation error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_remote_identity(url: str) -> str:
    """Return the repository identity used for safe origin comparison.

    GitHub serves the same HTTPS repository URL with or without the
    conventional ``.git`` suffix, and existing checkouts may record either
    spelling.  Normalize only that suffix (plus a trailing slash) so setup
    remains strict about the actual host and repository path.
    """

    normalized = url.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def _is_checkpoint_handoff_validation(
    payload: Mapping[str, Any],
) -> bool:
    smoke = payload.get("checkpoint_handoff_smoke")
    if (
        isinstance(smoke, Mapping)
        and dict(smoke) == CHECKPOINT_HANDOFF_SMOKE_CONTRACT
    ):
        return True
    production = payload.get("production_handoff_validation")
    if not isinstance(production, Mapping):
        return False
    expected_steps = production.get("expected_optimizer_steps")
    if (
        isinstance(expected_steps, bool)
        or not isinstance(expected_steps, int)
        or expected_steps <= 0
    ):
        return False
    required = {
        **PRODUCTION_HANDOFF_VALIDATION_CONTRACT,
        "expected_optimizer_steps": expected_steps,
    }
    return (
        dict(production) == required
        and _get(payload, "trainer.max_train_steps") == expected_steps
    )


def _plain(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig,)):
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return cfg


def _get(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise PlanError(f"required config setting is missing: {path}")
        value = value[part]
    return value


def _require_type(payload: Mapping[str, Any], path: str, expected: type) -> Any:
    value = _get(payload, path)
    if expected is int:
        valid = type(value) is int
    elif expected is bool:
        valid = type(value) is bool
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise PlanError(
            f"{path} must be {expected.__name__}, got {type(value).__name__}: {value!r}"
        )
    return value


def _require_nonempty_string(payload: Mapping[str, Any], path: str) -> str:
    value = _require_type(payload, path, str)
    if not value.strip():
        raise PlanError(f"{path} must not be empty")
    return value


def _resolve_repo_path(value: str, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise PlanError(f"{field} must resolve inside the repository: {resolved}")
    return resolved


def _resolve_extended_config_path(config_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        relative_candidate = config_path.parent / candidate
        candidate = (
            relative_candidate
            if relative_candidate.exists()
            else REPO_ROOT / candidate
        )
    resolved = candidate.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise PlanError(
            f"extended config must resolve inside the repository: {resolved}"
        )
    return resolved


def _load_config(
    config_path: Path,
    *,
    _chain: tuple[Path, ...] = (),
) -> tuple[DictConfig, dict[str, Any]]:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file() or config_path.is_symlink():
        raise PlanError(f"config must be a regular non-symlink file: {config_path}")
    if config_path in _chain:
        cycle = " -> ".join(str(path) for path in (*_chain, config_path))
        raise PlanError(f"config extends cycle detected: {cycle}")
    try:
        local_cfg = OmegaConf.load(config_path)
        local_payload = _plain(local_cfg)
    except Exception as exc:
        raise PlanError(f"could not resolve config {config_path}: {exc}") from exc
    if not isinstance(local_payload, dict):
        raise PlanError("config root must be a mapping")
    raw_extends = local_payload.pop("extends", None)
    if raw_extends is None:
        cfg = local_cfg
    else:
        extend_values = (
            [raw_extends] if isinstance(raw_extends, str) else raw_extends
        )
        if (
            not isinstance(extend_values, list)
            or not extend_values
            or not all(
                isinstance(value, str) and value.strip()
                for value in extend_values
            )
        ):
            raise PlanError("top-level extends must be a string or non-empty string list")
        layers: list[DictConfig] = []
        for value in extend_values:
            base_path = _resolve_extended_config_path(config_path, value)
            base_cfg, _ = _load_config(
                base_path,
                _chain=(*_chain, config_path),
            )
            layers.append(base_cfg)
        layers.append(OmegaConf.create(local_payload))
        cfg = OmegaConf.merge(*layers)
    try:
        payload = _plain(cfg)
    except Exception as exc:
        raise PlanError(f"could not resolve composed config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlanError("resolved config root must be a mapping")
    return cfg, payload


def _config_contract_bytes(
    config_path: Path,
    cfg: DictConfig,
) -> bytes:
    """Return the exact bytes used to bind a reviewed config contract."""

    raw = OmegaConf.load(config_path)
    if not isinstance(raw, DictConfig) or raw.get("extends") is None:
        return config_path.read_bytes()
    return OmegaConf.to_yaml(
        cfg,
        resolve=True,
        sort_keys=True,
    ).encode("utf-8")


def _config_contract_sha256(config_path: Path, cfg: DictConfig) -> str:
    """Hash direct configs byte-for-byte and composed configs after resolution."""

    return hashlib.sha256(
        _config_contract_bytes(config_path, cfg)
    ).hexdigest()


def _validate_runtime(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _get(payload, "runtime")
    if not isinstance(runtime, Mapping):
        raise PlanError("runtime must be a mapping")
    exact = {
        "runtime.schema_version": 1,
        "runtime.platform": "h100x8",
        "runtime.expected_gpu_count": 8,
        "runtime.num_processes": 8,
        "runtime.num_machines": 1,
        "runtime.mixed_precision": "bf16",
        "runtime.dynamo_backend": "no",
        "runtime.use_deepspeed": False,
        "runtime.torch_compile_environment": "disabled",
        "runtime.disable_autograd_multithreading": True,
        "runtime.tokenizers_parallelism": False,
    }
    for path, expected in exact.items():
        actual = _get(payload, path)
        if actual != expected or type(actual) is not type(expected):
            raise PlanError(f"{path} must be explicitly {expected!r}, got {actual!r}")
    image = _require_nonempty_string(payload, "runtime.container_image")
    build_keys = {
        "DOCKERFILE",
        "BASE_IMAGE",
        "TORCH_INDEX_URL",
        "PYTHON_VERSION",
        "INSTALL_DEEPSPEED",
        "INSTALL_MOGE",
        "INSTALL_FLASH_ATTN",
        "FLASH_ATTN_SPEC",
        "FLASH_ATTN_CUDA_ARCH_LIST",
        "FLASH_ATTN_MAX_JOBS",
        "FLASH_ATTN_NVCC_THREADS",
        "INSTALL_FAST_LINEAR_ATTN",
        "FAST_LINEAR_ATTN_SPEC",
        "CAUSAL_CONV1D_SPEC",
        "FAST_LINEAR_ATTN_TRANSFORMERS_SPEC",
        "FAST_LINEAR_ATTN_TILELANG_SPEC",
        "FAST_LINEAR_ATTN_TVM_FFI_SPEC",
        "FAST_LINEAR_ATTN_CUDA_ARCH_LIST",
        "FAST_LINEAR_ATTN_MAX_JOBS",
    }
    build_arguments = _get(payload, "runtime.container_build.arguments")
    if not isinstance(build_arguments, Mapping):
        raise PlanError("runtime.container_build.arguments must be a mapping")
    missing = sorted(build_keys - set(build_arguments))
    extra = sorted(set(build_arguments) - build_keys)
    if missing or extra:
        raise PlanError(
            "runtime.container_build.arguments must be complete and contain no "
            f"unknown settings: missing={missing}, extra={extra}"
        )
    build = dict(build_arguments)
    for key, value in build.items():
        if type(value) is not str or not value:
            raise PlanError(
                f"runtime.container_build.arguments.{key} must be a non-empty string"
            )
    h100_build_requirements = {
        "PYTHON_VERSION": "3.13",
        "INSTALL_DEEPSPEED": "0",
        "INSTALL_MOGE": "1",
        "INSTALL_FLASH_ATTN": "1",
        "FLASH_ATTN_CUDA_ARCH_LIST": "9.0",
        "INSTALL_FAST_LINEAR_ATTN": "1",
        "FAST_LINEAR_ATTN_CUDA_ARCH_LIST": "9.0",
    }
    for key, expected in h100_build_requirements.items():
        if build[key] != expected:
            raise PlanError(
                f"runtime.container_build.arguments.{key} must be {expected!r} "
                "for this H100 profile"
            )
    dockerfile = _resolve_repo_path(
        build["DOCKERFILE"], field="runtime.container_build.arguments.DOCKERFILE"
    )
    if not dockerfile.is_file():
        raise PlanError(f"configured Dockerfile is missing: {dockerfile}")
    if not build["BASE_IMAGE"].startswith("nvidia/cuda:"):
        raise PlanError("runtime container BASE_IMAGE must be a pinned NVIDIA CUDA image")
    if not build["TORCH_INDEX_URL"].startswith("https://download.pytorch.org/whl/cu"):
        raise PlanError("runtime container TORCH_INDEX_URL must select an explicit CUDA wheel index")
    pinned_specs = {
        "FLASH_ATTN_SPEC": r"flash-attn==[A-Za-z0-9][A-Za-z0-9._+-]*",
        "FAST_LINEAR_ATTN_SPEC": r"flash-linear-attention(?:\[cuda\])?==[A-Za-z0-9][A-Za-z0-9._+-]*",
        "CAUSAL_CONV1D_SPEC": r"causal-conv1d==[A-Za-z0-9][A-Za-z0-9._+-]*",
        "FAST_LINEAR_ATTN_TRANSFORMERS_SPEC": r"transformers==[A-Za-z0-9][A-Za-z0-9._+-]*",
        "FAST_LINEAR_ATTN_TILELANG_SPEC": r"tilelang==[A-Za-z0-9][A-Za-z0-9._+-]*",
        "FAST_LINEAR_ATTN_TVM_FFI_SPEC": r"apache-tvm-ffi==[A-Za-z0-9][A-Za-z0-9._+-]*",
    }
    for key, pattern in pinned_specs.items():
        if re.fullmatch(pattern, build[key]) is None:
            raise PlanError(
                f"runtime.container_build.arguments.{key} must be exactly version-pinned"
            )
    for key in (
        "FLASH_ATTN_MAX_JOBS",
        "FLASH_ATTN_NVCC_THREADS",
        "FAST_LINEAR_ATTN_MAX_JOBS",
    ):
        if re.fullmatch(r"[1-9][0-9]*", build[key]) is None:
            raise PlanError(
                f"runtime.container_build.arguments.{key} must be a positive integer string"
            )
    scratch_root = Path(
        _require_nonempty_string(payload, "runtime.scratch_root")
    ).expanduser()
    if not scratch_root.is_absolute():
        raise PlanError("runtime.scratch_root must be absolute")
    gpu_name = _require_nonempty_string(
        payload, "runtime.expected_gpu_name_contains"
    )
    capability = _get(payload, "runtime.expected_compute_capability")
    if capability != [9, 0]:
        raise PlanError(
            "runtime.expected_compute_capability must be explicitly [9, 0] for H100"
        )
    port = _require_type(payload, "runtime.main_process_port", int)
    if not 1024 <= port <= 65535:
        raise PlanError("runtime.main_process_port must be between 1024 and 65535")
    network = _require_nonempty_string(payload, "runtime.network_interface")
    main_torch_threads = _require_type(
        payload,
        "runtime.main_torch_threads",
        int,
    )
    main_torch_interop_threads = _require_type(
        payload,
        "runtime.main_torch_interop_threads",
        int,
    )
    if main_torch_threads <= 0 or main_torch_interop_threads <= 0:
        raise PlanError(
            "runtime.main_torch_threads and "
            "runtime.main_torch_interop_threads must be positive integers"
        )
    pytorch_cuda_alloc_conf = _require_nonempty_string(
        payload,
        "runtime.pytorch_cuda_alloc_conf",
    )
    require_clean = _require_type(payload, "runtime.require_clean_git", bool)
    provenance_launcher = _resolve_repo_path(
        _require_nonempty_string(payload, "runtime.provenance_launcher"),
        field="runtime.provenance_launcher",
    )
    if not provenance_launcher.is_file():
        raise PlanError(f"provenance launcher is missing: {provenance_launcher}")
    helper_payload = _get(payload, "runtime.helper_repositories")
    if not isinstance(helper_payload, Mapping) or not helper_payload:
        raise PlanError(
            "runtime.helper_repositories must be a non-empty mapping"
        )
    missing_model_helpers = {"moge", "vjepa2"} - set(helper_payload)
    if missing_model_helpers:
        raise PlanError(
            "runtime.helper_repositories is missing model dependencies: "
            f"{sorted(missing_model_helpers)}"
        )
    helper_repositories: dict[str, dict[str, str]] = {}
    for name in sorted(helper_payload):
        if HELPER_REPOSITORY_RE.fullmatch(str(name)) is None:
            raise PlanError(
                "runtime.helper_repositories names must match "
                f"{HELPER_REPOSITORY_RE.pattern!r}: {name!r}"
            )
        entry = helper_payload[name]
        if not isinstance(entry, Mapping):
            raise PlanError(f"runtime.helper_repositories.{name} must be a mapping")
        path = Path(
            _require_nonempty_string(
                payload, f"runtime.helper_repositories.{name}.path"
            )
        ).expanduser()
        if not path.is_absolute() or not path.is_relative_to(scratch_root):
            raise PlanError(
                f"runtime.helper_repositories.{name}.path must be absolute and "
                f"inside runtime.scratch_root: {path}"
            )
        url = _require_nonempty_string(
            payload, f"runtime.helper_repositories.{name}.url"
        )
        if not url.startswith("https://") or not url.endswith(".git"):
            raise PlanError(
                f"runtime.helper_repositories.{name}.url must be an HTTPS Git URL"
            )
        commit = _require_nonempty_string(
            payload, f"runtime.helper_repositories.{name}.commit"
        ).lower()
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise PlanError(
                f"runtime.helper_repositories.{name}.commit must be a full 40-character SHA"
            )
        helper_repositories[name] = {
            "path": str(path),
            "url": url,
            "commit": commit,
        }
    return {
        "container_image": image,
        "container_build": build,
        "scratch_root": str(scratch_root),
        "expected_gpu_count": 8,
        "expected_gpu_name_contains": gpu_name,
        "expected_compute_capability": (9, 0),
        "num_processes": 8,
        "num_machines": 1,
        "mixed_precision": "bf16",
        "dynamo_backend": "no",
        "main_process_port": port,
        "use_deepspeed": False,
        "torch_compile_environment": "disabled",
        "main_torch_threads": main_torch_threads,
        "main_torch_interop_threads": main_torch_interop_threads,
        "disable_autograd_multithreading": True,
        "pytorch_cuda_alloc_conf": pytorch_cuda_alloc_conf,
        "tokenizers_parallelism": False,
        "network_interface": network,
        "require_clean_git": require_clean,
        "provenance_launcher": provenance_launcher,
        "helper_repositories": helper_repositories,
    }


def _validate_training_contract(
    payload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    allow_transport_resume: bool = False,
) -> dict[str, Any]:
    """Validate common training settings and one explicit dataset-family contract.

    Dataset semantics are intentionally selected from the authoritative YAML,
    never from launcher flags.  The discriminator is the configured loader plus
    its representation/dimensions.  Unsupported or ambiguous combinations fail
    closed instead of falling through to RealMan assumptions.
    """

    required_strings = (
        "run_id",
        "run_root_dir",
        "framework.name",
        "framework.qwenvl.base_vlm",
        "framework.qwenvl.attn_implementation",
        "framework.vj2_model.predictor_attention_backend",
        "framework.action_model.action_model_type",
        "datasets.vla_data.dataset_py",
        "datasets.vla_data.action_type",
        "trainer.best_metric_name",
        "trainer.best_metric_mode",
        "trainer.lr_scheduler_type",
        "trainer.optimizer.name",
        "trainer.mixed_precision",
    )
    for path in required_strings:
        _require_nonempty_string(payload, path)
    required_booleans = (
        "framework.qwenvl.strict_attn_implementation",
        "framework.qwenvl.enable_fast_linear_attention",
        "framework.qwenvl.strict_fast_linear_attention",
        "framework.qwenvl.strict_full_trainable",
        "framework.qwenvl.lora.enabled",
        "framework.action_model.rtc_training.enabled",
        "datasets.vla_data.append_subtask_to_prompt",
        "datasets.vla_data.persistent_workers",
        "datasets.vla_data.pin_memory",
        "datasets.vla_data.drop_last",
        "trainer.eval_before_train",
        "trainer.allow_training_stream_eval",
        "trainer.checkpoint_eval_include_full_epoch_boundaries",
        "trainer.checkpoint_eval_milestones_only",
        "trainer.detailed_timing_logging",
        "trainer.enable_mixed_precision_training",
        "trainer.find_unused_parameters",
        "trainer.ddp_gradient_as_bucket_view",
        "trainer.ddp_static_graph",
        "trainer.compile_qwen_model",
        "trainer.compile_action_model",
        "trainer.compile_vj_predictor",
        "trainer.compile_vj_encoder",
        "trainer.compile_full_model",
        "trainer.allow_compile_with_deepspeed",
        "trainer.use_rabc",
        "trainer.is_resume",
        "trainer.resume_load_optimizer_state",
        "trainer.eval_only",
        "trainer.save_final_model",
        "trainer.enable_force_checkpoint_file",
    )
    for path in required_booleans:
        _require_type(payload, path, bool)
    required_integers = (
        "framework.action_model.action_dim",
        "framework.action_model.state_dim",
        "framework.action_model.action_horizon",
        "framework.action_model.future_action_window_size",
        "framework.action_model.past_action_window_size",
        "datasets.vla_data.per_device_batch_size",
        "datasets.vla_data.num_workers",
        "datasets.vla_data.prefetch_factor",
        "datasets.vla_data.worker_torch_threads",
        "datasets.vla_data.worker_cv2_threads",
        "trainer.epochs",
        "trainer.save_interval",
        "trainer.eval_interval",
        "trainer.checkpoint_max_to_keep",
        "trainer.logging_frequency",
        "trainer.detailed_timing_frequency",
        "trainer.gradient_accumulation_steps",
        "trainer.repeated_diffusion_steps",
        "trainer.ddp_bucket_cap_mb",
        "trainer.loss_scale.wm_warmup_steps",
    )
    for path in required_integers:
        _require_type(payload, path, int)
    multiprocessing_context = _require_nonempty_string(
        payload,
        "datasets.vla_data.multiprocessing_context",
    )
    if multiprocessing_context not in {"spawn", "forkserver"}:
        raise PlanError(
            "datasets.vla_data.multiprocessing_context must be explicitly "
            "'spawn' or 'forkserver'"
        )

    action = _get(payload, "framework.action_model")
    data = _get(payload, "datasets.vla_data")
    trainer = _get(payload, "trainer")
    scheduler_specific_kwargs = trainer.get("scheduler_specific_kwargs")
    if not isinstance(scheduler_specific_kwargs, Mapping):
        raise PlanError("trainer.scheduler_specific_kwargs must be a mapping")
    if (
        trainer["lr_scheduler_type"] == "cosine_with_min_lr"
        and scheduler_specific_kwargs.get("min_lr") is not None
        and scheduler_specific_kwargs.get("min_lr_rate") is not None
    ):
        raise PlanError(
            "trainer.scheduler_specific_kwargs cannot set both non-null "
            "min_lr and min_lr_rate for cosine_with_min_lr"
        )
    strict_learning_rate_groups = trainer.get(
        "strict_learning_rate_groups",
        False,
    )
    if type(strict_learning_rate_groups) is not bool:
        raise PlanError(
            "trainer.strict_learning_rate_groups must be a boolean"
        )
    for key in (
        "checkpoint_eval_milestone_fractions",
        "checkpoint_eval_milestone_steps",
    ):
        if key not in trainer:
            raise PlanError(
                f"trainer.{key} must be explicitly configured (null is "
                "allowed when the feature is disabled)"
            )
    configured_warmup_steps = trainer.get("num_warmup_steps")
    if isinstance(configured_warmup_steps, bool) or not (
        (
            isinstance(configured_warmup_steps, int)
            and configured_warmup_steps >= 0
        )
        or (
            isinstance(configured_warmup_steps, str)
            and configured_warmup_steps == "auto"
        )
    ):
        raise PlanError(
            "trainer.num_warmup_steps must be a non-negative integer or "
            "the explicit string 'auto'"
        )
    configured_warmup_ratio = trainer.get("warmup_ratio")
    if (
        isinstance(configured_warmup_ratio, bool)
        or not isinstance(configured_warmup_ratio, (int, float))
        or not math.isfinite(float(configured_warmup_ratio))
        or not 0.0 <= float(configured_warmup_ratio) <= 1.0
    ):
        raise PlanError(
            "trainer.warmup_ratio must be a finite number in [0, 1]"
        )
    if (
        isinstance(configured_warmup_steps, int)
        and configured_warmup_steps > 0
        and float(configured_warmup_ratio) != 0.0
    ):
        raise PlanError(
            "trainer.warmup_ratio must be 0 when trainer.num_warmup_steps "
            "owns an exact positive step count"
        )
    if (
        configured_warmup_steps == 0
        and float(configured_warmup_ratio) != 0.0
    ):
        raise PlanError(
            "trainer.num_warmup_steps=0 conflicts with a nonzero "
            "trainer.warmup_ratio; use num_warmup_steps: auto to resolve the "
            "ratio after the exact dataset schedule is known"
        )
    if (
        configured_warmup_steps == "auto"
        and float(configured_warmup_ratio) <= 0.0
    ):
        raise PlanError(
            "trainer.num_warmup_steps=auto requires trainer.warmup_ratio > 0"
        )
    dataset_py = str(data["dataset_py"])
    action_type = str(data["action_type"])
    state_dim = int(action["state_dim"])
    action_dim = int(action["action_dim"])
    action_horizon = int(action["action_horizon"])
    future_window = int(action["future_action_window_size"])
    predictor_attention_backend = _get(
        payload,
        "framework.vj2_model.predictor_attention_backend",
    )
    if predictor_attention_backend not in {"torch_sdpa", "flash_attn"}:
        raise PlanError(
            "framework.vj2_model.predictor_attention_backend must be "
            "'torch_sdpa' or 'flash_attn'"
        )
    for field in (
        "prefetch_factor",
        "worker_torch_threads",
        "worker_cv2_threads",
    ):
        if int(data[field]) <= 0:
            raise PlanError(
                f"datasets.vla_data.{field} must be a positive integer"
            )

    milestone_fractions = trainer.get(
        "checkpoint_eval_milestone_fractions", None
    )
    if milestone_fractions is not None:
        if not isinstance(milestone_fractions, list) or not milestone_fractions:
            raise PlanError(
                "trainer.checkpoint_eval_milestone_fractions must be null or "
                "a non-empty strictly increasing list of values in (0, 1]"
            )
        normalized_fractions: list[float] = []
        for value in milestone_fractions:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                or float(value) > 1.0
            ):
                raise PlanError(
                    "trainer.checkpoint_eval_milestone_fractions must contain "
                    "only finite numeric values in (0, 1]"
                )
            normalized_fractions.append(float(value))
        if any(
            current <= previous
            for previous, current in zip(
                normalized_fractions, normalized_fractions[1:]
            )
        ):
            raise PlanError(
                "trainer.checkpoint_eval_milestone_fractions must be strictly "
                "increasing"
            )
        milestone_fractions = normalized_fractions

    milestone_steps = trainer.get("checkpoint_eval_milestone_steps", None)
    if isinstance(milestone_steps, str):
        if milestone_steps.lower() != "auto":
            raise PlanError(
                "trainer.checkpoint_eval_milestone_steps must be 'auto', null, "
                "or a sorted, unique list of positive integers"
            )
        milestone_steps = "auto"
    elif milestone_steps is not None:
        if not isinstance(milestone_steps, list):
            raise PlanError(
                "trainer.checkpoint_eval_milestone_steps must be 'auto', null, "
                "or a sorted, unique list of positive integers"
            )
        if any(type(step) is not int or step <= 0 for step in milestone_steps):
            raise PlanError(
                "trainer.checkpoint_eval_milestone_steps must contain only "
                "positive integers"
            )
        if milestone_steps != sorted(set(milestone_steps)):
            raise PlanError(
                "trainer.checkpoint_eval_milestone_steps must be sorted and "
                "unique"
            )
        configured_max_steps = trainer.get("max_train_steps", None)
        if (
            type(configured_max_steps) is int
            and configured_max_steps > 0
            and any(step > configured_max_steps for step in milestone_steps)
        ):
            raise PlanError(
                "trainer.checkpoint_eval_milestone_steps cannot exceed "
                f"trainer.max_train_steps={configured_max_steps}"
            )
        milestone_steps = list(milestone_steps)

    include_full_epoch_boundaries = trainer.get(
        "checkpoint_eval_include_full_epoch_boundaries",
        False,
    )
    if type(include_full_epoch_boundaries) is not bool:
        raise PlanError(
            "trainer.checkpoint_eval_include_full_epoch_boundaries must be a "
            "boolean"
        )
    milestone_only = trainer.get(
        "checkpoint_eval_milestones_only", False
    )
    if type(milestone_only) is not bool:
        raise PlanError(
            "trainer.checkpoint_eval_milestones_only must be a boolean"
        )
    if (
        milestone_only
        and milestone_fractions is None
        and not (
            isinstance(milestone_steps, list)
            and bool(milestone_steps)
        )
    ):
        raise PlanError(
            "trainer.checkpoint_eval_milestones_only=true requires "
            "config-owned checkpoint_eval_milestone_fractions or an "
            "explicit non-empty checkpoint_eval_milestone_steps list"
        )

    pretrained_checkpoint = trainer.get("pretrained_checkpoint", None)
    pretrained_checkpoint_sha256 = trainer.get(
        "pretrained_checkpoint_sha256", None
    )
    if (pretrained_checkpoint is None) != (
        pretrained_checkpoint_sha256 is None
    ):
        raise PlanError(
            "trainer.pretrained_checkpoint and "
            "trainer.pretrained_checkpoint_sha256 must be configured together"
        )
    if pretrained_checkpoint_sha256 is not None and (
        not isinstance(pretrained_checkpoint_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", pretrained_checkpoint_sha256)
        is None
    ):
        raise PlanError(
            "trainer.pretrained_checkpoint_sha256 must be a lowercase SHA-256"
        )

    def reject_profile_keys(profile: str, keys: Sequence[str]) -> None:
        configured = sorted(
            key for key in keys if key in data and data.get(key) is not None
        )
        if configured:
            rendered = ", ".join(
                f"datasets.vla_data.{key}" for key in configured
            )
            raise PlanError(
                f"{profile} does not support these dataset-contract keys: "
                f"{rendered}"
            )

    if action_horizon <= 0 or future_window != action_horizon - 1:
        raise PlanError(
            "framework.action_model.future_action_window_size must equal "
            "action_horizon - 1"
        )
    if action["past_action_window_size"] != 0:
        raise PlanError("human H100 profiles require past_action_window_size=0")

    if dataset_py == "canonical_subset_vla":
        dataset_profile = CANONICAL_GCS_PROFILE
    elif dataset_py == "lerobot_datasets":
        data_mix = _require_nonempty_string(payload, "datasets.vla_data.data_mix")
        if (
            action_type == "joint_delta_gripper_absolute"
            and state_dim == 18
            and action_dim == 18
        ):
            dataset_profile = REALMAN_LEROBOT_PROFILE
        elif (
            action_type == "delta_qpos"
            and state_dim == 8
            and action_dim == 7
            and "libero" in data_mix.lower()
        ):
            dataset_profile = LIBERO_LEROBOT_PROFILE
        else:
            raise PlanError(
                "unsupported or ambiguous LeRobot H100 dataset contract: "
                f"data_mix={data_mix!r}, action_type={action_type!r}, "
                f"state_dim={state_dim}, action_dim={action_dim}"
            )
    else:
        raise PlanError(
            f"unsupported H100 dataset loader {dataset_py!r}; supported profiles "
            f"are {sorted(SUPPORTED_DATASET_PROFILES)}"
        )

    representation: str
    artifact_kind: str
    episode_split_manifest: str | None = None
    holdout_episode_count: int | None = None
    holdout_sampling_policy: dict[str, Any] | None = None
    evaluation_observation_count: int | None = None
    canonical_eval_manifest: str | None = None
    canonical_eval_normalization: str | None = None
    canonical_exact_realman_contract = False
    if dataset_profile == REALMAN_LEROBOT_PROFILE:
        reject_profile_keys(
            "RealMan LeRobot",
            (
                "canonical_eval_manifest",
                "canonical_exclude_eval_episodes_from_training",
                "canonical_eval_selection_seed",
                "canonical_eval_candidate_count",
                "canonical_eval_min_episodes_per_shard",
            ),
        )
        for path in (
            "datasets.vla_data.data_root_dir",
            "datasets.vla_data.episode_split_manifest",
            "datasets.vla_data.video_backend",
            "datasets.vla_data.action_delta_anchor",
            "datasets.vla_data.gripper_action_type",
            "datasets.vla_data.state_action_normalization",
            "datasets.vla_data.epoch_sampling_strategy",
        ):
            _require_nonempty_string(payload, path)
        for path in (
            "datasets.vla_data.replace_modality_metadata_with_overrides",
            "datasets.vla_data.use_action_validity_prefix_mask",
            "datasets.vla_data.require_statistics_frame_count",
            "datasets.vla_data.load_all_data_for_training",
            "datasets.vla_data.fail_on_sample_error",
            "datasets.vla_data.shuffle",
        ):
            _require_type(payload, path, bool)
        for path in (
            "datasets.vla_data.eval_num_workers",
            "datasets.vla_data.video_backend_num_threads",
            "datasets.vla_data.action_validity_invalid_run_length",
        ):
            _require_type(payload, path, int)
        if (state_dim, action_dim, action_horizon, future_window) != (18, 18, 50, 49):
            raise PlanError(
                "RealMan LeRobot requires state_dim=18, action_dim=18, "
                "action_horizon=50, and future_action_window_size=49"
            )
        if bool(action["rtc_training"]["enabled"]):
            raise PlanError(
                "the production RealMan H100 profile requires RTC training to "
                "be explicitly disabled"
            )
        expected_representation = {
            "action_type": "joint_delta_gripper_absolute",
            "action_delta_anchor": "chunk_start_state",
            "gripper_action_type": "absolute",
            "state_action_normalization": Q01_Q99_UNCLIPPED,
        }
        for key, expected in expected_representation.items():
            if data[key] != expected:
                raise PlanError(f"datasets.vla_data.{key} must be {expected!r}")
        if data["epoch_sampling_strategy"] != "all_sources_exhaustive":
            raise PlanError(
                "production RealMan H100 requires "
                "datasets.vla_data.epoch_sampling_strategy="
                "'all_sources_exhaustive' so every eligible row from every "
                "configured training source is seen once per logical epoch"
            )
        if not bool(data["fail_on_sample_error"]):
            raise PlanError(
                "production RealMan H100 requires fail_on_sample_error=true; "
                "a corrupt row must not be silently replaced"
            )
        if bool(data["drop_last"]):
            raise PlanError(
                "production RealMan H100 exhaustive epochs require "
                "datasets.vla_data.drop_last=false"
            )
        if bool(data["shuffle"]):
            raise PlanError(
                "production RealMan H100 exhaustive epochs require "
                "datasets.vla_data.shuffle=false because the dataset owns the "
                "deterministic epoch permutation"
            )
        if bool(trainer.get("allow_training_stream_eval", False)):
            raise PlanError(
                "production RealMan H100 exhaustive epochs require "
                "trainer.allow_training_stream_eval=false; training-stream "
                "evaluation would consume a scheduled row without training it"
            )
        if bool(data.get("gpu_video_decode_on_rank", False)) and bool(
            data.get("gpu_video_decode_async_prefetch", True)
        ):
            raise PlanError(
                "production RealMan H100 exact-resume epochs cannot enable "
                "asynchronous rank-video prefetch because its producer can run "
                "ahead of the checkpoint cursor"
            )
        controls = data.get("action_delta_mappings", {}).get(
            "source_controls", {}
        )
        head = data.get("action_delta_mappings", {}).get("source_head", {})
        if controls.get("state_indices") != list(range(16)):
            raise PlanError("source_controls state_indices must be exactly 0..15")
        if controls.get("delta_mask") != (
            [True] * 7 + [False] + [True] * 7 + [False]
        ):
            raise PlanError(
                "source_controls delta mask must keep both grippers absolute"
            )
        if head.get("state_indices") != [16, 17] or head.get("delta_mask") != [
            True,
            True,
        ]:
            raise PlanError(
                "head action mapping must be state indices [16,17] with deltas"
            )
        episode_split_manifest = str(data["episode_split_manifest"])
        try:
            configured_realman_holdout_seed_text(payload)
        except ValueError as exc:
            raise PlanError(str(exc)) from exc
        configured_holdout_count = data.get("holdout_episode_count")
        configured_holdout_policy = data.get("holdout_sampling")
        if (
            configured_holdout_count is not None
            and configured_holdout_policy is not None
        ):
            raise PlanError(
                "Configure either datasets.vla_data.holdout_episode_count or "
                "datasets.vla_data.holdout_sampling, not both"
            )
        if configured_holdout_policy is not None:
            try:
                holdout_sampling_policy = validate_holdout_sampling_policy(
                    configured_holdout_policy
                )
            except ValueError as exc:
                raise PlanError(str(exc)) from exc
            evaluation_observation_count = int(
                holdout_sampling_policy["evaluation_observation_count"]
            )
        if configured_holdout_count is not None:
            if (
                isinstance(configured_holdout_count, bool)
                or not isinstance(configured_holdout_count, int)
                or configured_holdout_count <= 0
            ):
                raise PlanError(
                    "datasets.vla_data.holdout_episode_count must be a positive "
                    "integer when configured"
                )
            holdout_episode_count = int(configured_holdout_count)
        representation = "18-D mixed joint delta / absolute gripper"
        artifact_kind = "realman_episode_split"
    elif dataset_profile == LIBERO_LEROBOT_PROFILE:
        reject_profile_keys(
            "LIBERO LeRobot",
            (
                "episode_split_manifest",
                "holdout_episode_count",
                "holdout_sampling",
                "eval_per_device_batch_size",
                "canonical_eval_manifest",
                "canonical_exclude_eval_episodes_from_training",
                "canonical_eval_selection_seed",
                "canonical_eval_candidate_count",
                "canonical_eval_min_episodes_per_shard",
            ),
        )
        _require_nonempty_string(payload, "datasets.vla_data.data_root_dir")
        _require_type(payload, "datasets.vla_data.load_all_data_for_training", bool)
        video_backend = _require_nonempty_string(
            payload, "datasets.vla_data.video_backend"
        )
        _require_type(payload, "datasets.vla_data.video_backend_num_threads", int)
        if (state_dim, action_dim, action_horizon, future_window) != (8, 7, 7, 6):
            raise PlanError(
                "LIBERO LeRobot requires state_dim=8, action_dim=7, "
                "action_horizon=7, and future_action_window_size=6"
            )
        if video_backend != "pyav":
            raise PlanError("LIBERO H100 video_backend must be explicitly pyav")
        if bool(trainer["eval_before_train"]):
            raise PlanError(
                "LIBERO without an immutable holdout manifest must set "
                "trainer.eval_before_train=false"
            )
        if bool(trainer["allow_training_stream_eval"]):
            raise PlanError(
                "LIBERO H100 must not select checkpoints from shuffled training batches"
            )
        if data.get("episode_split_manifest"):
            raise PlanError(
                "LIBERO episode_split_manifest is not yet a validated launcher "
                "artifact; omit it instead of reusing the RealMan manifest format"
            )
        representation = "8-D state / 7-D LIBERO delta_qpos"
        artifact_kind = "none"
    else:
        reject_profile_keys(
            "canonical GCS",
            (
                "episode_split_manifest",
                "holdout_episode_count",
            ),
        )
        focused_eval_enabled = trainer.get(
            "heldout_focused_eval_enabled",
            None,
        )
        if type(focused_eval_enabled) is not bool:
            raise PlanError(
                "canonical GCS requires an explicit boolean "
                "trainer.heldout_focused_eval_enabled"
            )
        if focused_eval_enabled:
            raise PlanError(
                "canonical GCS constructs only the exact unbiased manifest "
                "heldout loader; set "
                "trainer.heldout_focused_eval_enabled=false"
            )
        if (
            trainer["best_metric_name"]
            != "heldout_eval_normalized_action_mae"
        ):
            raise PlanError(
                "canonical GCS checkpoint selection must use "
                "trainer.best_metric_name="
                "'heldout_eval_normalized_action_mae'"
            )
        for path in (
            "datasets.vla_data.action_delta_anchor",
            "datasets.vla_data.gripper_action_type",
            "datasets.vla_data.dataset_canonicalization_root",
            "datasets.vla_data.manifest_path",
            "datasets.vla_data.adapter_dir",
            "datasets.vla_data.cache_dir",
            "datasets.vla_data.bucket_root",
            "datasets.vla_data.gcs_access_probe_object",
            "datasets.vla_data.sidecar_normalization",
            "datasets.vla_data.canonical_eval_manifest",
        ):
            _require_nonempty_string(payload, path)
        for path in (
            "datasets.vla_data.allow_gcs_download",
            "datasets.vla_data.canonical_exclude_eval_episodes_from_training",
        ):
            _require_type(payload, path, bool)
        canonical_eval_selection_seed = _require_type(
            payload,
            "datasets.vla_data.canonical_eval_selection_seed",
            int,
        )
        canonical_eval_candidate_count = _require_type(
            payload,
            "datasets.vla_data.canonical_eval_candidate_count",
            int,
        )
        canonical_eval_min_episodes_per_shard = _require_type(
            payload,
            "datasets.vla_data.canonical_eval_min_episodes_per_shard",
            int,
        )
        if canonical_eval_selection_seed < 0:
            raise PlanError(
                "datasets.vla_data.canonical_eval_selection_seed must be "
                "non-negative"
            )
        if canonical_eval_candidate_count < 3:
            raise PlanError(
                "datasets.vla_data.canonical_eval_candidate_count must be at "
                "least 3"
            )
        if canonical_eval_min_episodes_per_shard != 2:
            raise PlanError(
                "canonical H100 requires "
                "datasets.vla_data.canonical_eval_min_episodes_per_shard=2 "
                "so finite window caps cannot collapse a shard catalog to "
                "one episode"
            )
        for path in (
            "datasets.vla_data.dataset_ids",
            "datasets.vla_data.adapter_group_ids",
        ):
            values = _get(payload, path)
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value for value in values
            ):
                raise PlanError(f"{path} must be a list of non-empty strings")
        camera_slots = _get(payload, "datasets.vla_data.camera_slots")
        if not isinstance(camera_slots, list) or not camera_slots or not all(
            isinstance(value, str) and value for value in camera_slots
        ):
            raise PlanError(
                "datasets.vla_data.camera_slots must be a non-empty list of strings"
            )
        _require_type(
            payload,
            "datasets.vla_data.enforce_worker_memory_budget",
            bool,
        )
        estimated_worker_memory_gb = _get(
            payload,
            "datasets.vla_data.estimated_worker_memory_gb",
        )
        worker_memory_budget_fraction = _get(
            payload,
            "datasets.vla_data.worker_memory_budget_fraction",
        )
        if (
            isinstance(estimated_worker_memory_gb, bool)
            or not isinstance(estimated_worker_memory_gb, (int, float))
            or not math.isfinite(float(estimated_worker_memory_gb))
            or float(estimated_worker_memory_gb) <= 0.0
        ):
            raise PlanError(
                "datasets.vla_data.estimated_worker_memory_gb must be a "
                "positive finite number"
            )
        if (
            isinstance(worker_memory_budget_fraction, bool)
            or not isinstance(worker_memory_budget_fraction, (int, float))
            or not math.isfinite(float(worker_memory_budget_fraction))
            or not 0.0 < float(worker_memory_budget_fraction) <= 1.0
        ):
            raise PlanError(
                "datasets.vla_data.worker_memory_budget_fraction must be "
                "a finite number in (0, 1]"
            )
        canonical_exact_realman_contract = (
            state_dim,
            action_dim,
            action_horizon,
            future_window,
        ) == (18, 18, 50, 49)
        canonical_legacy_contract = (
            state_dim,
            action_dim,
            action_horizon,
            future_window,
        ) == (53, 49, 50, 49)
        if not (
            canonical_exact_realman_contract or canonical_legacy_contract
        ):
            raise PlanError(
                "canonical GCS requires either the exact RealMan contract "
                "(state_dim=18, action_dim=18, action_horizon=50, "
                "future_action_window_size=49) or the legacy semantic "
                "contract (state_dim=53, action_dim=49, action_horizon=50, "
                "future_action_window_size=49)"
            )
        if action_type != "joint_delta_gripper_absolute":
            raise PlanError(
                "canonical GCS action_type must be "
                "'joint_delta_gripper_absolute'"
            )
        if data["action_delta_anchor"] != "chunk_start_state":
            raise PlanError(
                "canonical GCS action_delta_anchor must be 'chunk_start_state'"
            )
        if data["gripper_action_type"] != "absolute":
            raise PlanError("canonical GCS gripper_action_type must be 'absolute'")
        if canonical_exact_realman_contract:
            for path in (
                "datasets.vla_data.state_action_normalization",
                "datasets.vla_data.normalization_statistics_artifact",
                "datasets.vla_data.normalization_statistics_artifact_sha256",
                "datasets.vla_data.action_representation_contract_sha256",
                "datasets.vla_data.frozen_train_view_manifest",
                "datasets.vla_data.frozen_train_view_manifest_sha256",
                "datasets.vla_data.epoch_sampling_strategy",
            ):
                _require_nonempty_string(payload, path)
            for path in (
                "datasets.vla_data.fail_on_sample_error",
                "datasets.vla_data.shuffle",
                "datasets.vla_data.drop_last",
            ):
                _require_type(payload, path, bool)
            if data["state_action_normalization"] != Q01_Q99_UNCLIPPED:
                raise PlanError(
                    "exact 18-D canonical RealMan requires "
                    "state_action_normalization='q01_q99_unclipped'"
                )
            if data["sidecar_normalization"] != Q01_Q99_UNCLIPPED:
                raise PlanError(
                    "exact 18-D canonical RealMan requires raw canonical "
                    "sidecars plus shared union statistics, expressed as "
                    "sidecar_normalization='q01_q99_unclipped'"
                )
            if (
                data["action_representation_contract_sha256"]
                != REALMAN_18D_ACTION_CONTRACT.sha256()
            ):
                raise PlanError(
                    "exact 18-D canonical RealMan "
                    "action_representation_contract_sha256 does not match "
                    "the versioned RealMan 18-D contract"
                )
            if data["epoch_sampling_strategy"] != "all_sources_exhaustive":
                raise PlanError(
                    "exact 18-D canonical RealMan requires "
                    "epoch_sampling_strategy='all_sources_exhaustive'"
                )
            if not bool(data["fail_on_sample_error"]):
                raise PlanError(
                    "exact 18-D canonical RealMan requires "
                    "fail_on_sample_error=true"
                )
            if bool(data["shuffle"]):
                raise PlanError(
                    "exact 18-D canonical RealMan requires shuffle=false "
                    "because the dataset owns its deterministic epoch "
                    "permutation"
                )
            if bool(data["drop_last"]):
                raise PlanError(
                    "exact 18-D canonical RealMan requires drop_last=false"
                )
            if bool(trainer.get("allow_training_stream_eval", False)):
                raise PlanError(
                    "exact 18-D canonical RealMan requires "
                    "trainer.allow_training_stream_eval=false"
                )
            canonical_eval_normalization = Q01_Q99_UNCLIPPED
            representation = "18-D mixed joint delta / absolute gripper"
        else:
            if (
                data["sidecar_normalization"]
                != "shard_q01_q99_unclipped"
            ):
                raise PlanError(
                    "legacy canonical GCS sidecar_normalization must be "
                    "'shard_q01_q99_unclipped'"
                )
            canonical_eval_normalization = str(
                data["sidecar_normalization"]
            )
            representation = (
                "53-D semantic state / 49-D mixed canonical action"
            )
        if not str(data["bucket_root"]).startswith("gs://"):
            raise PlanError(
                "canonical GCS datasets.vla_data.bucket_root must start with 'gs://'"
            )
        bucket_prefix = str(data["bucket_root"]).rstrip("/") + "/"
        gcs_access_probe_object = str(data["gcs_access_probe_object"])
        if (
            not gcs_access_probe_object.startswith(bucket_prefix)
            or gcs_access_probe_object == bucket_prefix
            or gcs_access_probe_object.endswith("/")
        ):
            raise PlanError(
                "canonical GCS datasets.vla_data.gcs_access_probe_object must "
                "name one exact object below datasets.vla_data.bucket_root"
            )
        if not bool(data["canonical_exclude_eval_episodes_from_training"]):
            raise PlanError(
                "canonical GCS requires "
                "canonical_exclude_eval_episodes_from_training=true"
            )
        canonical_root = str(
            Path(data["dataset_canonicalization_root"]).expanduser()
        )
        helper = runtime["helper_repositories"].get("dataset-canonicalization")
        if helper is None:
            raise PlanError(
                "canonical GCS requires a pinned "
                "runtime.helper_repositories.dataset-canonicalization entry"
            )
        if canonical_root != helper["path"]:
            raise PlanError(
                "datasets.vla_data.dataset_canonicalization_root must match "
                "runtime.helper_repositories.dataset-canonicalization.path"
            )
        canonical_eval_manifest = str(data["canonical_eval_manifest"])
        configured_holdout_policy = data.get("holdout_sampling")
        if configured_holdout_policy is None:
            raise PlanError(
                "canonical GCS requires datasets.vla_data.holdout_sampling so "
                "the heldout episode/window policy is config-owned"
            )
        try:
            holdout_sampling_policy = validate_holdout_sampling_policy(
                configured_holdout_policy
            )
        except ValueError as exc:
            raise PlanError(str(exc)) from exc
        evaluation_observation_count = int(
            holdout_sampling_policy["evaluation_observation_count"]
        )
        artifact_kind = "canonical_eval"

    probability = data.get("subtask_prompt_append_probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise PlanError("subtask_prompt_append_probability must be numeric")
    if not 0.0 <= float(probability) <= 1.0:
        raise PlanError("subtask_prompt_append_probability must be in [0,1]")
    if (
        bool(data["append_subtask_to_prompt"])
        and float(probability) != 0.7
    ):
        raise PlanError(
            "enabled subtask prompt augmentation must explicitly use probability 0.7"
        )
    if trainer["mixed_precision"] != runtime["mixed_precision"]:
        raise PlanError("trainer.mixed_precision and runtime.mixed_precision disagree")
    compile_flags = (
        "compile_qwen_model",
        "compile_action_model",
        "compile_vj_predictor",
        "compile_vj_encoder",
        "compile_full_model",
    )
    if runtime["torch_compile_environment"] == "disabled" and any(
        bool(trainer[name]) for name in compile_flags
    ):
        raise PlanError("runtime disables torch.compile but a trainer compile flag is true")
    if bool(trainer["eval_only"]):
        raise PlanError(
            "source production YAML must describe training, not eval-only "
            "transport state"
        )
    if bool(trainer["is_resume"]) and not allow_transport_resume:
        raise PlanError(
            "source production YAML must describe a fresh train; use the human "
            "resume command for transport state"
        )
    resume_from_checkpoint = trainer.get("resume_from_checkpoint")
    if not allow_transport_resume and resume_from_checkpoint is not None:
        raise PlanError(
            "source production YAML must explicitly set trainer.resume_from_checkpoint: null"
        )
    if allow_transport_resume and (
        not bool(trainer["is_resume"])
        or not isinstance(resume_from_checkpoint, str)
        or not resume_from_checkpoint.strip()
        or not Path(resume_from_checkpoint).is_absolute()
    ):
        raise PlanError(
            "authenticated transport-resume config must set "
            "trainer.is_resume=true and an absolute "
            "trainer.resume_from_checkpoint"
        )
    if not bool(trainer["resume_load_optimizer_state"]):
        raise PlanError("production resume must restore optimizer and scheduler state")
    if not bool(trainer["save_final_model"]):
        raise PlanError("production profile must explicitly save the final model")
    if not bool(trainer["enable_force_checkpoint_file"]):
        raise PlanError(
            "production profile must keep the emergency checkpoint sentinel enabled"
        )
    global_batch = (
        int(data["per_device_batch_size"])
        * int(runtime["num_processes"])
        * int(trainer["gradient_accumulation_steps"])
    )
    if global_batch <= 0:
        raise PlanError("effective global batch must be positive")
    if dataset_profile == REALMAN_LEROBOT_PROFILE:
        if (
            holdout_episode_count is None
            and holdout_sampling_policy is None
        ):
            raise PlanError(
                "RealMan H100 training requires an explicit immutable "
                "holdout episode count or datasets.vla_data.holdout_sampling; "
                "the launcher will not infer it from global batch size"
            )
        if (
            holdout_episode_count is not None
            and global_batch % holdout_episode_count != 0
        ):
            raise PlanError(
                "RealMan effective global batch must be an integer multiple of "
                "datasets.vla_data.holdout_episode_count: "
                f"{global_batch} % {holdout_episode_count} != 0"
            )
        if evaluation_observation_count is None:
            evaluation_observation_count = global_batch
    eval_per_device_batch_size: int | None = None
    if evaluation_observation_count is not None:
        raw_eval_batch = data.get(
            "eval_per_device_batch_size",
            data["per_device_batch_size"],
        )
        if (
            isinstance(raw_eval_batch, bool)
            or not isinstance(raw_eval_batch, int)
            or raw_eval_batch <= 0
        ):
            raise PlanError(
                "datasets.vla_data.eval_per_device_batch_size must be a "
                "positive integer when immutable checkpoint evaluation is enabled"
            )
        eval_per_device_batch_size = int(raw_eval_batch)
        distributed_eval_batch = (
            eval_per_device_batch_size * int(runtime["num_processes"])
        )
        if evaluation_observation_count % distributed_eval_batch != 0:
            raise PlanError(
                "holdout_sampling.evaluation_observation_count must be an "
                "integer multiple of eval_per_device_batch_size * world_size: "
                f"{evaluation_observation_count} % {distributed_eval_batch} != 0"
            )
    helper_repositories = runtime["helper_repositories"]
    moge_repo = str(
        Path(_require_nonempty_string(payload, "framework.depth_teacher_aux.moge_repo_path"))
    )
    vjepa2_repo = str(
        Path(_require_nonempty_string(payload, "framework.vj2_model.hub_repo_or_dir"))
    )
    if moge_repo != helper_repositories["moge"]["path"]:
        raise PlanError(
            "framework.depth_teacher_aux.moge_repo_path must match "
            "runtime.helper_repositories.moge.path"
        )
    if vjepa2_repo != helper_repositories["vjepa2"]["path"]:
        raise PlanError(
            "framework.vj2_model.hub_repo_or_dir must match "
            "runtime.helper_repositories.vjepa2.path"
        )
    scratch_root = Path(str(runtime["scratch_root"]))
    run_root = Path(_require_nonempty_string(payload, "run_root_dir")).expanduser()
    data_root = Path(data.get("data_root_dir") or data["cache_dir"]).expanduser()
    for field, path in (
        ("run_root_dir", run_root),
        (
            "datasets.vla_data."
            + ("cache_dir" if dataset_profile == CANONICAL_GCS_PROFILE else "data_root_dir"),
            data_root,
        ),
    ):
        if not path.is_absolute() or not path.is_relative_to(scratch_root):
            raise PlanError(
                f"{field} must be absolute and inside runtime.scratch_root "
                f"because the human Docker launcher mounts only that data root: {path}"
            )
    if dataset_profile == CANONICAL_GCS_PROFILE:
        canonical_root_path = Path(
            _require_nonempty_string(
                payload, "datasets.vla_data.dataset_canonicalization_root"
            )
        ).expanduser()
        for field in ("manifest_path", "adapter_dir"):
            path = Path(
                _require_nonempty_string(payload, f"datasets.vla_data.{field}")
            ).expanduser()
            if not path.is_absolute() or not path.is_relative_to(
                canonical_root_path
            ):
                raise PlanError(
                    f"datasets.vla_data.{field} must be absolute and inside "
                    "dataset_canonicalization_root"
                )
        eval_manifest_path = Path(
            _require_nonempty_string(
                payload, "datasets.vla_data.canonical_eval_manifest"
            )
        ).expanduser()
        if not eval_manifest_path.is_absolute() or not eval_manifest_path.is_relative_to(
            scratch_root
        ):
            raise PlanError(
                "datasets.vla_data.canonical_eval_manifest must be absolute and "
                "inside runtime.scratch_root"
            )
    return {
        "dataset_profile": dataset_profile,
        "artifact_kind": artifact_kind,
        "run_id_prefix": _require_nonempty_string(payload, "run_id"),
        "run_root_dir": str(run_root),
        "base_vlm": _get(payload, "framework.qwenvl.base_vlm"),
        "attention": _get(payload, "framework.qwenvl.attn_implementation"),
        "world_model_predictor_attention_backend": (
            predictor_attention_backend
        ),
        "fast_linear_attention": _get(
            payload, "framework.qwenvl.enable_fast_linear_attention"
        ),
        "action_model": action["action_model_type"],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "representation": representation,
        "data_root_dir": str(data_root),
        "data_mix": data.get("data_mix")
        or ",".join(str(value) for value in data.get("dataset_ids", [])),
        "episode_split_manifest": episode_split_manifest,
        "holdout_episode_count": holdout_episode_count,
        "holdout_sampling_policy": holdout_sampling_policy,
        "evaluation_observation_count": evaluation_observation_count,
        "eval_per_device_batch_size": eval_per_device_batch_size,
        "canonical_eval_manifest": canonical_eval_manifest,
        "canonical_source_manifest": data.get("manifest_path"),
        "canonical_bucket_root": (
            str(data["bucket_root"])
            if dataset_profile == CANONICAL_GCS_PROFILE
            else None
        ),
        "canonical_gcs_probe_object": (
            str(data["gcs_access_probe_object"])
            if dataset_profile == CANONICAL_GCS_PROFILE
            else None
        ),
        "requires_gcloud": (
            bool(data["allow_gcs_download"])
            if dataset_profile == CANONICAL_GCS_PROFILE
            else False
        ),
        "canonical_eval_selection_seed": (
            canonical_eval_selection_seed
            if dataset_profile == CANONICAL_GCS_PROFILE
            else None
        ),
        "canonical_eval_candidate_count": (
            canonical_eval_candidate_count
            if dataset_profile == CANONICAL_GCS_PROFILE
            else None
        ),
        "canonical_eval_min_episodes_per_shard": (
            canonical_eval_min_episodes_per_shard
            if dataset_profile == CANONICAL_GCS_PROFILE
            else None
        ),
        "canonical_action_type": (
            action_type if dataset_profile == CANONICAL_GCS_PROFILE else None
        ),
        "canonical_sidecar_normalization": (
            data.get("sidecar_normalization")
            if dataset_profile == CANONICAL_GCS_PROFILE
            else None
        ),
        "canonical_eval_normalization": (
            canonical_eval_normalization
            if dataset_profile == CANONICAL_GCS_PROFILE
            else None
        ),
        "canonical_exact_realman_contract": (
            canonical_exact_realman_contract
            if dataset_profile == CANONICAL_GCS_PROFILE
            else False
        ),
        "normalization_statistics_artifact": (
            data.get("normalization_statistics_artifact")
            if canonical_exact_realman_contract
            else None
        ),
        "normalization_statistics_artifact_sha256": (
            data.get("normalization_statistics_artifact_sha256")
            if canonical_exact_realman_contract
            else None
        ),
        "action_representation_contract_sha256": (
            data.get("action_representation_contract_sha256")
            if canonical_exact_realman_contract
            else None
        ),
        "frozen_train_view_manifest": (
            data.get("frozen_train_view_manifest")
            if canonical_exact_realman_contract
            else None
        ),
        "frozen_train_view_manifest_sha256": (
            data.get("frozen_train_view_manifest_sha256")
            if canonical_exact_realman_contract
            else None
        ),
        "canonical_adapter_dir": (
            data.get("adapter_dir")
            if dataset_profile == CANONICAL_GCS_PROFILE
            else None
        ),
        "per_device_batch_size": data["per_device_batch_size"],
        "global_batch_size": global_batch,
        "num_workers": data["num_workers"],
        "video_backend": data.get("video_backend")
        or data.get("video_decode_backend", "canonical-auto"),
        "subtask_prompt_probability": float(probability),
        "epochs": trainer["epochs"],
        "max_train_steps": trainer.get("max_train_steps"),
        "warmup_steps": trainer["num_warmup_steps"],
        "wm_warmup_steps": trainer["loss_scale"]["wm_warmup_steps"],
        "save_interval": trainer["save_interval"],
        "eval_interval": trainer["eval_interval"],
        "checkpoint_eval_milestone_fractions": copy.deepcopy(
            milestone_fractions
        ),
        "checkpoint_eval_milestone_steps": copy.deepcopy(milestone_steps),
        "checkpoint_eval_include_full_epoch_boundaries": (
            include_full_epoch_boundaries
        ),
        "checkpoint_eval_milestones_only": milestone_only,
        "checkpoint_max_to_keep": trainer["checkpoint_max_to_keep"],
        "pretrained_checkpoint": pretrained_checkpoint,
        "pretrained_checkpoint_sha256": pretrained_checkpoint_sha256,
        "save_final_model": trainer["save_final_model"],
        "enable_force_checkpoint_file": trainer["enable_force_checkpoint_file"],
        "learning_rate": copy.deepcopy(trainer.get("learning_rate")),
        "strict_learning_rate_groups": strict_learning_rate_groups,
        "optimizer": copy.deepcopy(trainer.get("optimizer")),
        "scheduler": trainer["lr_scheduler_type"],
        "loss_scale": copy.deepcopy(trainer.get("loss_scale")),
        "repeated_diffusion_steps": trainer["repeated_diffusion_steps"],
        "best_metric_name": trainer["best_metric_name"],
        "best_metric_mode": trainer["best_metric_mode"],
    }


def _resolve_artifact_path(value: str, *, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    if not resolved.is_absolute():  # pragma: no cover - defensive
        raise PlanError(f"{field} must resolve to an absolute path: {resolved}")
    return resolved


def _validate_realman_evaluation_sampling_contract(
    evaluation_sampling: Any,
    *,
    manifest_holdout_episode_count: int,
    expected_observation_count: int,
    expected_allocation: Mapping[str, Any],
    require_explicit_allocation: bool,
    dataset_name: str,
    selected_episode_ids_in_rank_order: Any,
) -> dict[str, Any]:
    """Validate both modern and immutable legacy RealMan window allocations."""

    if not isinstance(evaluation_sampling, Mapping):
        raise PlanError("holdout manifest evaluation_sampling must be an object")
    allocation = evaluation_sampling.get("window_allocation")
    allocation_is_explicit = allocation is not None
    if not allocation_is_explicit:
        if require_explicit_allocation:
            raise PlanError(
                "policy-owned holdout manifests require an explicit balanced "
                "evaluation_sampling.window_allocation contract"
            )
        frames_per_episode = evaluation_sampling.get("frames_per_episode")
        if (
            isinstance(frames_per_episode, bool)
            or not isinstance(frames_per_episode, int)
            or frames_per_episode <= 0
        ):
            raise PlanError(
                "legacy holdout manifests require a positive integer "
                "evaluation_sampling.frames_per_episode"
            )
        inferred_observation_count = (
            manifest_holdout_episode_count * frames_per_episode
        )
        recorded_observation_count = evaluation_sampling.get(
            "observation_count",
            inferred_observation_count,
        )
        if recorded_observation_count != inferred_observation_count:
            raise PlanError(
                "legacy holdout manifest observation_count must equal heldout "
                "episodes * frames_per_episode"
            )
        allocation = {
            "algorithm": "legacy_uniform_per_episode_v1",
            "holdout_episode_count": manifest_holdout_episode_count,
            "base_frames_per_episode": frames_per_episode,
            "extra_window_episode_count": 0,
            "maximum_frames_per_episode": frames_per_episode,
            "extra_window_episode_identities": [],
        }
        observation_count = inferred_observation_count
    else:
        if not isinstance(allocation, Mapping):
            raise PlanError(
                "evaluation_sampling.window_allocation must be an object"
            )
        observation_count = evaluation_sampling.get("observation_count")

    if (
        isinstance(observation_count, bool)
        or not isinstance(observation_count, int)
        or observation_count != expected_observation_count
    ):
        raise PlanError(
            "holdout manifest observation_count does not match the configured "
            f"evaluation count: {observation_count!r} != "
            f"{expected_observation_count}"
        )

    required_integer_fields = (
        ("holdout_episode_count", 1),
        ("base_frames_per_episode", 1),
        ("extra_window_episode_count", 0),
        ("maximum_frames_per_episode", 1),
    )
    normalized: dict[str, Any] = {
        "algorithm": allocation.get("algorithm"),
    }
    for field, minimum in required_integer_fields:
        value = allocation.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise PlanError(
                "holdout manifest window allocation "
                f"{field} must be an integer >= {minimum}"
            )
        normalized[field] = int(value)

    holdout_episode_count = normalized["holdout_episode_count"]
    base_frames_per_episode = normalized["base_frames_per_episode"]
    extra_window_episode_count = normalized["extra_window_episode_count"]
    maximum_frames_per_episode = normalized["maximum_frames_per_episode"]
    if (
        holdout_episode_count != manifest_holdout_episode_count
        or extra_window_episode_count >= holdout_episode_count
        or maximum_frames_per_episode
        != base_frames_per_episode + int(extra_window_episode_count > 0)
        or holdout_episode_count * base_frames_per_episode
        + extra_window_episode_count
        != observation_count
    ):
        raise PlanError(
            "holdout manifest has an invalid balanced q/r window allocation"
        )

    frames_alias = evaluation_sampling.get("frames_per_episode")
    if allocation_is_explicit:
        if extra_window_episode_count > 0:
            if frames_alias is not None:
                raise PlanError(
                    "holdout remainder allocation must omit frames_per_episode"
                )
        elif (
            isinstance(frames_alias, bool)
            or not isinstance(frames_alias, int)
            or frames_alias != base_frames_per_episode
        ):
            raise PlanError(
                "holdout uniform allocation requires frames_per_episode equal "
                "to base_frames_per_episode"
            )

    raw_extra_identities = allocation.get(
        "extra_window_episode_identities",
        [],
    )
    if not isinstance(raw_extra_identities, list):
        raise PlanError(
            "holdout manifest extra_window_episode_identities must be a list"
        )
    extra_identities: list[tuple[str, int]] = []
    for index, identity in enumerate(raw_extra_identities):
        if (
            not isinstance(identity, Mapping)
            or not isinstance(identity.get("dataset_name"), str)
            or not identity["dataset_name"]
            or isinstance(identity.get("episode_id"), bool)
            or not isinstance(identity.get("episode_id"), int)
        ):
            raise PlanError(
                "holdout manifest has an invalid extra-window episode identity "
                f"at index {index}"
            )
        extra_identities.append(
            (str(identity["dataset_name"]), int(identity["episode_id"]))
        )
    if (
        len(extra_identities) != extra_window_episode_count
        or len(set(extra_identities)) != len(extra_identities)
    ):
        raise PlanError(
            "holdout manifest does not bind exactly one unique episode identity "
            "per extra evaluation window"
        )
    if extra_window_episode_count > 0:
        if not isinstance(selected_episode_ids_in_rank_order, list) or any(
            isinstance(episode_id, bool) or not isinstance(episode_id, int)
            for episode_id in selected_episode_ids_in_rank_order
        ):
            raise PlanError(
                "holdout manifest selected episode rank order must be an "
                "integer list"
            )
        expected_extra_identities = [
            (dataset_name, int(episode_id))
            for episode_id in selected_episode_ids_in_rank_order[
                :extra_window_episode_count
            ]
        ]
        if extra_identities != expected_extra_identities:
            raise PlanError(
                "holdout manifest extra windows are not assigned to the "
                "deterministic leading ranked episodes"
            )

    if allocation_is_explicit:
        normalized["extra_window_episode_identities"] = [
            {
                "dataset_name": dataset_name_value,
                "episode_id": episode_id,
            }
            for dataset_name_value, episode_id in extra_identities
        ]
        allocation_mismatches = {
            key: {
                "manifest": normalized.get(key),
                "config": expected,
            }
            for key, expected in expected_allocation.items()
            if normalized.get(key) != expected
        }
        if allocation_mismatches:
            raise PlanError(
                "holdout manifest window allocation does not match the "
                f"configured contract: {allocation_mismatches}"
            )

    return {
        "observation_count": int(observation_count),
        "holdout_episode_count": int(holdout_episode_count),
        "base_frames_per_episode": int(base_frames_per_episode),
        "extra_window_episode_count": int(extra_window_episode_count),
        "maximum_frames_per_episode": int(maximum_frames_per_episode),
        "window_allocation_algorithm": str(normalized["algorithm"]),
        "extra_window_episode_identities": extra_identities,
        "allocation_is_explicit": allocation_is_explicit,
    }


def _validate_realman_train_statistics_columns(
    statistics: Any,
    *,
    data_cfg: Mapping[str, Any],
    expected_train_frames: int,
    declared_numeric_columns: Any,
) -> None:
    """Validate the exact raw columns consumed by RealMan normalization.

    The loader applies modality overrides before selecting state/action
    statistics.  Checking only the statistics file hash is insufficient: a
    validly hashed sidecar can still omit vector columns when its Parquet
    physical type was not recognized by the artifact generator.
    """

    if not isinstance(statistics, Mapping):
        raise PlanError("train-split statistics payload must be an object")
    overrides = data_cfg.get("modality_metadata_overrides")
    if not isinstance(overrides, Mapping):
        raise PlanError(
            "RealMan training requires datasets.vla_data."
            "modality_metadata_overrides"
        )

    required_widths: dict[str, int] = {}
    for modality in ("state", "action"):
        entries = overrides.get(modality)
        if not isinstance(entries, Mapping) or not entries:
            raise PlanError(
                "RealMan modality metadata overrides must define non-empty "
                f"{modality} entries"
            )
        for subkey, raw_spec in entries.items():
            if not isinstance(raw_spec, Mapping):
                raise PlanError(
                    "RealMan modality metadata override "
                    f"{modality}.{subkey} must be an object"
                )
            original_key = raw_spec.get("original_key")
            end = raw_spec.get("end")
            if not isinstance(original_key, str) or not original_key:
                raise PlanError(
                    "RealMan modality metadata override "
                    f"{modality}.{subkey}.original_key must be non-empty"
                )
            if isinstance(end, bool) or not isinstance(end, int) or end <= 0:
                raise PlanError(
                    "RealMan modality metadata override "
                    f"{modality}.{subkey}.end must be a positive integer"
                )
            required_widths[original_key] = max(
                required_widths.get(original_key, 0),
                end,
            )

    declared = (
        set(declared_numeric_columns)
        if isinstance(declared_numeric_columns, list)
        and all(isinstance(value, str) for value in declared_numeric_columns)
        else set()
    )
    missing_declared = sorted(set(required_widths) - declared)
    missing_payload = sorted(set(required_widths) - set(statistics))
    if missing_declared or missing_payload:
        raise PlanError(
            "train-split statistics omit required RealMan normalization "
            "columns: "
            f"manifest_missing={missing_declared}, payload_missing={missing_payload}"
        )

    for key, required_width in sorted(required_widths.items()):
        stat = statistics[key]
        if not isinstance(stat, Mapping):
            raise PlanError(
                f"train-split statistics column {key!r} must be an object"
            )
        count = stat.get("count")
        if (
            not isinstance(count, list)
            or len(count) != 1
            or isinstance(count[0], bool)
            or not isinstance(count[0], int)
            or count[0] != expected_train_frames
        ):
            raise PlanError(
                f"train-split statistics column {key!r} count must equal "
                f"{expected_train_frames}"
            )
        for field in ("min", "max", "mean", "std", "q01", "q99"):
            values = stat.get(field)
            if not isinstance(values, list) or len(values) < required_width:
                raise PlanError(
                    f"train-split statistics column {key!r} {field} must "
                    f"contain at least {required_width} values"
                )
            selected = values[:required_width]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in selected
            ):
                raise PlanError(
                    f"train-split statistics column {key!r} {field} contains "
                    "non-finite or non-numeric values"
                )


def _validate_realman_manifest(
    payload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = _resolve_repo_path(
        str(contract["episode_split_manifest"]),
        field="datasets.vla_data.episode_split_manifest",
    )
    if not manifest_path.is_file():
        raise PlanError(f"holdout manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"holdout manifest is invalid: {exc}") from exc
    derivation = _get(manifest, "selection.holdout_count_derivation")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise PlanError("holdout manifest must bind exactly one dataset")
    entry = datasets[0]
    if not isinstance(entry, Mapping):
        raise PlanError("holdout manifest dataset binding must be an object")
    data_cfg = _get(payload, "datasets.vla_data")
    configured_dataset_root = data_cfg.get("data_root_dir")
    if (
        not isinstance(configured_dataset_root, str)
        or not configured_dataset_root
    ):
        raise PlanError(
            "RealMan holdout selection requires a configured data_root_dir"
        )
    configured_dataset_name = Path(
        configured_dataset_root.rstrip("/")
    ).name
    if entry.get("dataset_name") != configured_dataset_name:
        raise PlanError(
            "holdout manifest dataset_name does not match data_root_dir: "
            f"{entry.get('dataset_name')!r} != {configured_dataset_name!r}"
        )
    try:
        selection_contract = build_realman_holdout_selection_contract(
            payload,
            world_size=int(runtime["num_processes"]),
            dataset_name=configured_dataset_name,
        )
        selection_contract_sha256 = (
            holdout_selection_contract_sha256(selection_contract)
        )
    except (TypeError, ValueError) as exc:
        raise PlanError(
            f"RealMan holdout selection contract is invalid: {exc}"
        ) from exc
    expected_derivation = {
        "launcher_sha256": _sha256(Path(runtime["provenance_launcher"])),
        "holdout_selection_contract_schema": selection_contract["schema"],
        "holdout_selection_contract_sha256": selection_contract_sha256,
        "holdout_selection_contract": selection_contract,
        "world_size": runtime["num_processes"],
        "per_device_batch_size": contract["per_device_batch_size"],
        "gradient_accumulation_steps": _get(
            payload, "trainer.gradient_accumulation_steps"
        ),
        "effective_global_batch_size": contract["global_batch_size"],
    }
    holdout_sampling_policy = contract.get("holdout_sampling_policy")
    if holdout_sampling_policy is not None:
        try:
            holdout_sampling_plan = derive_episode_holdout_sampling_plan(
                total_episode_count=int(entry["full_episode_count"]),
                evaluation_observation_count=int(
                    contract["evaluation_observation_count"]
                ),
                policy=holdout_sampling_policy,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanError(
                f"holdout sampling policy cannot resolve this dataset: {exc}"
            ) from exc
        expected_derivation.update(
            {
                "holdout_episode_count": holdout_sampling_plan[
                    "holdout_episode_count"
                ],
                "evaluation_observation_count": int(
                    contract["evaluation_observation_count"]
                ),
                "evaluation_frames_per_episode": (
                    holdout_sampling_plan["base_frames_per_episode"]
                    if holdout_sampling_plan[
                        "extra_window_episode_count"
                    ]
                    == 0
                    else None
                ),
                "base_frames_per_episode": holdout_sampling_plan[
                    "base_frames_per_episode"
                ],
                "extra_window_episode_count": holdout_sampling_plan[
                    "extra_window_episode_count"
                ],
                "maximum_frames_per_episode": holdout_sampling_plan[
                    "maximum_frames_per_episode"
                ],
                "window_allocation_algorithm": holdout_sampling_plan[
                    "window_allocation_algorithm"
                ],
                "holdout_sampling_policy": dict(holdout_sampling_policy),
                "holdout_sampling_plan": holdout_sampling_plan,
            }
        )
    else:
        expected_derivation.update(
            {
                "holdout_episode_count": contract["holdout_episode_count"],
                "evaluation_frames_per_episode": (
                    int(contract["evaluation_observation_count"])
                    // int(contract["holdout_episode_count"])
                ),
            }
        )
    mismatches = {
        key: {"manifest": derivation.get(key), "config": value}
        for key, value in expected_derivation.items()
        if derivation.get(key) != value
    }
    if _is_checkpoint_handoff_validation(payload):
        # A handoff-only validation makes no model-quality claim and does not
        # rebuild or reselect the frozen holdout. Permit only the provenance
        # launcher's byte hash to drift as human-launch plumbing evolves; the
        # exact selection contract, episode/window allocation, manifest
        # bytes, statistics, and source-view bindings remain mandatory below.
        mismatches.pop("launcher_sha256", None)
    if mismatches:
        raise PlanError(
            "holdout manifest does not match this config/runtime; run the explicit "
            f"prepare command and commit the artifacts: {mismatches}"
        )
    holdout_episode_count = entry.get("holdout_episode_count")
    if (
        isinstance(holdout_episode_count, bool)
        or not isinstance(holdout_episode_count, int)
        or holdout_episode_count <= 0
    ):
        raise PlanError("holdout manifest has an invalid holdout episode count")
    dataset_name = entry.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise PlanError("holdout manifest dataset_name must be a non-empty string")
    expected_observation_count = int(contract["evaluation_observation_count"])
    selected_rank_order = manifest.get("selection", {}).get(
        "selected_episode_ids_in_rank_order",
        [],
    )
    if holdout_sampling_policy is not None:
        required_extra_count = int(
            holdout_sampling_plan["extra_window_episode_count"]
        )
        if (
            not isinstance(selected_rank_order, list)
            or len(selected_rank_order) < required_extra_count
            or any(
                isinstance(episode_id, bool)
                or not isinstance(episode_id, int)
                for episode_id in selected_rank_order
            )
        ):
            raise PlanError(
                "holdout manifest selected episode rank order must be a "
                "complete integer list"
            )
        expected_extra_identities = [
            {
                "dataset_name": dataset_name,
                "episode_id": episode_id,
            }
            for episode_id in selected_rank_order[
                :required_extra_count
            ]
        ]
        expected_allocation = {
            "algorithm": holdout_sampling_plan[
                "window_allocation_algorithm"
            ],
            "holdout_episode_count": holdout_sampling_plan[
                "holdout_episode_count"
            ],
            "base_frames_per_episode": holdout_sampling_plan[
                "base_frames_per_episode"
            ],
            "extra_window_episode_count": holdout_sampling_plan[
                "extra_window_episode_count"
            ],
            "maximum_frames_per_episode": holdout_sampling_plan[
                "maximum_frames_per_episode"
            ],
            "extra_window_episode_identities": expected_extra_identities,
        }
    else:
        expected_holdout_episode_count = contract.get(
            "holdout_episode_count"
        )
        if (
            isinstance(expected_holdout_episode_count, bool)
            or not isinstance(expected_holdout_episode_count, int)
            or expected_holdout_episode_count <= 0
            or holdout_episode_count != expected_holdout_episode_count
            or expected_observation_count % expected_holdout_episode_count
            != 0
        ):
            raise PlanError(
                "legacy holdout manifest episode/window counts do not match "
                "the configured RealMan evaluation contract"
            )
        expected_base_frames_per_episode = (
            expected_observation_count // expected_holdout_episode_count
        )
        expected_allocation = {
            "algorithm": "uniform_per_episode_v1",
            "holdout_episode_count": expected_holdout_episode_count,
            "base_frames_per_episode": expected_base_frames_per_episode,
            "extra_window_episode_count": 0,
            "maximum_frames_per_episode": expected_base_frames_per_episode,
            "extra_window_episode_identities": [],
        }
    sampling_contract = _validate_realman_evaluation_sampling_contract(
        manifest.get("evaluation_sampling"),
        manifest_holdout_episode_count=int(holdout_episode_count),
        expected_observation_count=expected_observation_count,
        expected_allocation=expected_allocation,
        require_explicit_allocation=holdout_sampling_policy is not None,
        dataset_name=dataset_name,
        selected_episode_ids_in_rank_order=selected_rank_order,
    )
    observation_count = sampling_contract["observation_count"]
    base_frames_per_episode = sampling_contract["base_frames_per_episode"]
    extra_window_episode_count = sampling_contract[
        "extra_window_episode_count"
    ]
    train_stats = entry.get("train_statistics", {})
    train_stats_path = manifest_path.parent / str(train_stats.get("path", ""))
    if not train_stats_path.is_file() or train_stats.get("sha256") != _sha256(
        train_stats_path
    ):
        raise PlanError("train-split statistics file/hash binding is invalid")
    try:
        train_statistics_payload = json.loads(
            train_stats_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(
            f"train-split statistics payload is invalid: {exc}"
        ) from exc
    _validate_realman_train_statistics_columns(
        train_statistics_payload,
        data_cfg=data_cfg,
        expected_train_frames=int(entry.get("train_frame_count")),
        declared_numeric_columns=train_stats.get("numeric_columns"),
    )
    action_stats = entry.get("action_representation_statistics", {})
    action_stats_path = manifest_path.parent / str(action_stats.get("path", ""))
    if not action_stats_path.is_file() or action_stats.get("sha256") != _sha256(
        action_stats_path
    ):
        raise PlanError("18-D action statistics file/hash binding is invalid")
    sidecar = json.loads(action_stats_path.read_text(encoding="utf-8"))
    if action_stats.get("contract_sha256") != REALMAN_18D_ACTION_CONTRACT.sha256():
        raise PlanError("manifest action contract hash is not the exact RealMan 18-D contract")
    if sidecar.get("contract_sha256") != REALMAN_18D_ACTION_CONTRACT.sha256():
        raise PlanError("statistics action contract hash is invalid")
    if action_stats.get("normalization") != Q01_Q99_UNCLIPPED:
        raise PlanError("manifest action normalization is not q01_q99_unclipped")
    stable_hash = split_manifest_sha256_without_statistics_binding(manifest)
    if (
        sidecar.get("provenance", {}).get(
            "split_manifest_sha256_without_statistics_binding"
        )
        != stable_hash
    ):
        raise PlanError("action statistics provenance does not bind this immutable split")
    report_path = manifest_path.with_name(f"{manifest_path.stem}_report.json")
    if not report_path.is_file():
        raise PlanError(f"holdout report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") != _sha256(manifest_path):
        raise PlanError("holdout report does not bind the manifest bytes")
    expected_report_selection_contract = {
        "schema": selection_contract["schema"],
        "sha256": selection_contract_sha256,
        "contract": selection_contract,
    }
    if (
        report.get("selection", {}).get("holdout_selection_contract")
        != expected_report_selection_contract
    ):
        raise PlanError(
            "holdout report does not bind the reviewed holdout selection "
            "contract"
        )
    action_report = report.get("action_representation_statistics", {})
    if action_report.get("sha256") != _sha256(action_stats_path):
        raise PlanError("holdout report does not bind the action statistics bytes")
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "holdout_episode_count": entry.get("holdout_episode_count"),
        "evaluation_observation_count": observation_count,
        "base_frames_per_episode": base_frames_per_episode,
        "extra_window_episode_count": extra_window_episode_count,
        "evaluation_frames_per_episode": (
            base_frames_per_episode
            if extra_window_episode_count == 0
            else None
        ),
        "train_episode_count": entry.get("train_episode_count"),
        "train_frame_count": entry.get("train_frame_count"),
        "train_statistics_sha256": _sha256(train_stats_path),
        "action_statistics_sha256": _sha256(action_stats_path),
        "action_contract_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
    }


def _validate_canonical_manifest(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = _resolve_artifact_path(
        str(contract["canonical_eval_manifest"]),
        field="datasets.vla_data.canonical_eval_manifest",
    )
    source_path = _resolve_artifact_path(
        str(contract["canonical_source_manifest"]),
        field="datasets.vla_data.manifest_path",
    )
    if not manifest_path.is_file():
        raise PlanError(f"canonical eval manifest is missing: {manifest_path}")
    if not source_path.is_file():
        raise PlanError(f"canonical source manifest is missing: {source_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"canonical eval manifest is invalid: {exc}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise PlanError("canonical eval manifest must be a schema_version=1 object")
    if manifest.get("purpose") != "heldout":
        raise PlanError("canonical eval manifest purpose must be exactly 'heldout'")
    source_sha256 = _sha256(source_path)
    if manifest.get("source_manifest_sha256") != source_sha256:
        raise PlanError(
            "canonical eval manifest does not bind the configured canonical source manifest"
        )
    shared_statistics: dict[str, Any] | None = None
    shared_statistics_path: Path | None = None
    frozen_view_path: Path | None = None
    if bool(contract.get("canonical_exact_realman_contract", False)):
        shared_statistics_path = _resolve_artifact_path(
            str(contract["normalization_statistics_artifact"]),
            field=(
                "datasets.vla_data.normalization_statistics_artifact"
            ),
        )
        if not shared_statistics_path.is_file():
            raise PlanError(
                "canonical shared normalization statistics are missing: "
                f"{shared_statistics_path}"
            )
        try:
            shared_statistics = load_openpi_realman_union_statistics(
                shared_statistics_path,
                str(
                    contract[
                        "normalization_statistics_artifact_sha256"
                    ]
                ),
            )
        except (OSError, ValueError) as exc:
            raise PlanError(
                "canonical shared normalization statistics are invalid: "
                f"{exc}"
            ) from exc
        frozen_view_path = _resolve_artifact_path(
            str(contract["frozen_train_view_manifest"]),
            field="datasets.vla_data.frozen_train_view_manifest",
        )
        if not frozen_view_path.is_file():
            raise PlanError(
                "canonical frozen training-view manifest is missing: "
                f"{frozen_view_path}"
            )
        actual_view_sha256 = _sha256(frozen_view_path)
        if (
            actual_view_sha256
            != contract["frozen_train_view_manifest_sha256"]
        ):
            raise PlanError(
                "canonical frozen training-view manifest SHA-256 mismatch: "
                f"expected "
                f"{contract['frozen_train_view_manifest_sha256']}, "
                f"found {actual_view_sha256}"
            )
    data = _get(payload, "datasets.vla_data")
    if not isinstance(data, Mapping):
        raise PlanError("datasets.vla_data must be a mapping")
    try:
        adapter_contract_sha256 = canonical_adapter_contract_sha256(
            data,
            local_repo_root=REPO_ROOT,
        )
        action_sidecar_variant = canonical_action_sidecar_variant(
            data,
            action_horizon=int(contract["action_horizon"]),
            canonical_eval_manifest_sha256=None,
            exclude_eval_episodes_from_training=False,
            adapter_contract_sha256=adapter_contract_sha256,
            local_repo_root=REPO_ROOT,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise PlanError(
            f"canonical adapter semantics contract is invalid: {exc}"
        ) from exc
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise PlanError(
            "canonical eval manifest requires a selection contract"
        )
    selection_integer_minimums = {
        "seed": 0,
        "window_count": 1,
        "holdout_episode_count": 1,
        "base_frames_per_episode": 1,
        "extra_window_episode_count": 0,
        "maximum_frames_per_episode": 1,
        "candidate_count": 3,
        "action_horizon": 1,
        "action_dim": 1,
        "configured_episode_count": 1,
    }
    for field, minimum in selection_integer_minimums.items():
        value = selection.get(field)
        if type(value) is not int or value < minimum:
            raise PlanError(
                "canonical eval manifest selection requires integer "
                f"{field}>={minimum}, got {value!r}"
            )
    for field in (
        "algorithm",
        "action_type",
        "normalization",
        "adapter_contract_sha256",
        "action_sidecar_variant",
        "configured_episode_catalog_sha256",
        "window_allocation_algorithm",
    ):
        value = selection.get(field)
        if not isinstance(value, str) or not value:
            raise PlanError(
                "canonical eval manifest selection requires non-empty string "
                f"{field}"
            )
    holdout_sampling_policy = contract.get("holdout_sampling_policy")
    if holdout_sampling_policy is None:
        raise PlanError(
            "canonical checkpoint evaluation requires a config-owned "
            "holdout_sampling policy"
        )
    try:
        holdout_sampling_plan = derive_episode_holdout_sampling_plan(
            total_episode_count=int(selection["configured_episode_count"]),
            evaluation_observation_count=int(
                contract["evaluation_observation_count"]
            ),
            policy=holdout_sampling_policy,
        )
    except (TypeError, ValueError) as exc:
        raise PlanError(
            f"canonical holdout sampling policy cannot resolve this catalog: {exc}"
        ) from exc
    extra_window_episode_count = int(
        holdout_sampling_plan["extra_window_episode_count"]
    )
    base_frames_per_episode = int(
        holdout_sampling_plan["base_frames_per_episode"]
    )
    frames_per_episode_alias = selection.get("frames_per_episode")
    if extra_window_episode_count == 0:
        if (
            type(frames_per_episode_alias) is not int
            or frames_per_episode_alias != base_frames_per_episode
        ):
            raise PlanError(
                "canonical eval manifest uniform allocation requires integer "
                "frames_per_episode equal to base_frames_per_episode"
            )
    elif frames_per_episode_alias is not None:
        raise PlanError(
            "canonical eval manifest remainder allocation must omit "
            "frames_per_episode; use the explicit base/extra fields"
        )
    expected_selection = {
        "algorithm": CANONICAL_EVAL_SELECTION_ALGORITHM,
        "seed": int(contract["canonical_eval_selection_seed"]),
        "window_count": int(contract["evaluation_observation_count"]),
        "holdout_episode_count": int(
            holdout_sampling_plan["holdout_episode_count"]
        ),
        "base_frames_per_episode": base_frames_per_episode,
        "extra_window_episode_count": extra_window_episode_count,
        "maximum_frames_per_episode": int(
            holdout_sampling_plan["maximum_frames_per_episode"]
        ),
        "window_allocation_algorithm": str(
            holdout_sampling_plan["window_allocation_algorithm"]
        ),
        "holdout_sampling_policy": dict(holdout_sampling_policy),
        "holdout_sampling_plan": holdout_sampling_plan,
        "candidate_count": int(contract["canonical_eval_candidate_count"]),
        "action_horizon": int(contract["action_horizon"]),
        "action_dim": int(contract["action_dim"]),
        "action_type": str(contract["canonical_action_type"]),
        "normalization": str(
            contract.get(
                "canonical_eval_normalization",
                contract["canonical_sidecar_normalization"],
            )
        ),
        "adapter_contract_sha256": adapter_contract_sha256,
        "action_sidecar_variant": action_sidecar_variant,
    }
    if extra_window_episode_count == 0:
        expected_selection["frames_per_episode"] = (
            base_frames_per_episode
        )
    selection_mismatches = {
        field: {
            "manifest": selection.get(field),
            "config": expected,
        }
        for field, expected in expected_selection.items()
        if selection.get(field) != expected
    }
    if selection_mismatches:
        raise PlanError(
            "canonical eval manifest selection contract does not match this "
            f"config/runtime: {selection_mismatches}"
        )
    if re.fullmatch(
        r"[0-9a-f]{16}",
        str(selection["action_sidecar_variant"]),
    ) is None:
        raise PlanError(
            "canonical eval manifest selection action_sidecar_variant must be "
            "a 16-character lowercase SHA-256 prefix"
        )
    for field in (
        "adapter_contract_sha256",
        "configured_episode_catalog_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(selection[field])) is None:
            raise PlanError(
                f"canonical eval manifest selection {field} must be "
                "lowercase SHA-256"
            )
    windows = manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise PlanError("canonical eval manifest must contain at least one window")
    if len(windows) != int(contract["evaluation_observation_count"]):
        raise PlanError(
            "canonical eval manifest must contain exactly the configured "
            "evaluation observation count: "
            f"windows={len(windows)}, "
            f"expected={contract['evaluation_observation_count']}"
        )
    required_strings = ("dataset_id", "sid", "revision", "data_file")
    identities: list[tuple[Any, ...]] = []
    episode_identities: list[tuple[Any, ...]] = []
    allowed_datasets = set(
        str(value) for value in _get(payload, "datasets.vla_data.dataset_ids")
    )
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise PlanError(f"canonical eval window {index} must be an object")
        invalid_strings = [
            key
            for key in required_strings
            if not isinstance(window.get(key), str) or not window[key]
        ]
        if invalid_strings:
            raise PlanError(
                f"canonical eval window {index} has invalid fields: {invalid_strings}"
            )
        if allowed_datasets and window["dataset_id"] not in allowed_datasets:
            raise PlanError(
                f"canonical eval window {index} selects unconfigured dataset "
                f"{window['dataset_id']!r}"
            )
        for key in ("episode_index", "base_index"):
            value = window.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PlanError(
                    f"canonical eval window {index} requires non-negative integer {key}"
                )
        episode_identity = (
            window["dataset_id"],
            window["sid"],
            window["revision"],
            window["data_file"],
            window["episode_index"],
        )
        episode_identities.append(episode_identity)
        identities.append((*episode_identity, window["base_index"]))
    if len(set(identities)) != len(identities):
        raise PlanError("canonical eval manifest contains duplicate windows")
    extra_identities_raw = selection.get("extra_window_episode_identities")
    if not isinstance(extra_identities_raw, list):
        raise PlanError(
            "canonical eval selection extra_window_episode_identities must be a list"
        )
    extra_identities: list[tuple[Any, ...]] = []
    for index, raw_identity in enumerate(extra_identities_raw):
        if (
            not isinstance(raw_identity, list)
            or len(raw_identity) != 5
            or any(
                not isinstance(value, str) or not value
                for value in raw_identity[:4]
            )
            or isinstance(raw_identity[4], bool)
            or not isinstance(raw_identity[4], int)
            or raw_identity[4] < 0
        ):
            raise PlanError(
                "canonical eval selection has invalid extra episode identity "
                f"{index}: {raw_identity!r}"
            )
        extra_identities.append(tuple(raw_identity))
    if (
        len(extra_identities)
        != int(holdout_sampling_plan["extra_window_episode_count"])
        or len(set(extra_identities)) != len(extra_identities)
    ):
        raise PlanError(
            "canonical eval selection extra episode identities do not match "
            "the balanced window allocation"
        )
    per_episode: dict[tuple[Any, ...], int] = {}
    for identity in episode_identities:
        per_episode[identity] = per_episode.get(identity, 0) + 1
    extra_identity_set = set(extra_identities)
    if (
        len(per_episode)
        != int(holdout_sampling_plan["holdout_episode_count"])
        or not extra_identity_set.issubset(per_episode)
        or any(
            count
            != (
                int(holdout_sampling_plan["base_frames_per_episode"])
                + int(identity in extra_identity_set)
            )
            for identity, count in per_episode.items()
        )
    ):
        raise PlanError(
            "canonical eval windows do not implement the declared balanced "
            "episode/window allocation"
        )
    if int(selection["window_count"]) != len(windows):
        raise PlanError(
            "canonical eval manifest selection window_count does not match "
            f"windows: {selection['window_count']} != {len(windows)}"
        )
    return {
        "kind": CANONICAL_GCS_PROFILE,
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "window_count": len(windows),
        "holdout_episode_count": len(per_episode),
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": source_sha256,
        "normalization_statistics_path": (
            None
            if shared_statistics_path is None
            else str(shared_statistics_path)
        ),
        "normalization_statistics_sha256": (
            None
            if shared_statistics_path is None
            else _sha256(shared_statistics_path)
        ),
        "normalization_statistics_contract_sha256": (
            None
            if shared_statistics is None
            else shared_statistics.get("contract_sha256")
        ),
        "frozen_train_view_manifest_path": (
            None if frozen_view_path is None else str(frozen_view_path)
        ),
        "frozen_train_view_manifest_sha256": (
            None if frozen_view_path is None else _sha256(frozen_view_path)
        ),
        "selection": dict(selection),
    }


def _validate_dataset_artifacts(
    payload: Mapping[str, Any],
    runtime: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    profile = contract["dataset_profile"]
    if profile == REALMAN_LEROBOT_PROFILE:
        holdout = _validate_realman_manifest(
            payload, runtime, contract
        )
        return {"kind": profile, **holdout}
    if profile == CANONICAL_GCS_PROFILE:
        return _validate_canonical_manifest(payload, contract)
    if profile == LIBERO_LEROBOT_PROFILE:
        return None
    raise AssertionError(profile)


def resolve_plan(
    config_path: Path,
    *,
    validate_artifacts: bool = True,
    allow_transport_resume: bool = False,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    cfg, payload = _load_config(config_path)
    runtime = _validate_runtime(payload)
    contract = _validate_training_contract(
        payload,
        runtime,
        allow_transport_resume=allow_transport_resume,
    )
    plan = {
        "schema": "starvla-human-h100-training-plan-v1",
        "config_path": str(config_path),
        "config_sha256": _config_contract_sha256(config_path, cfg),
        "runtime": {
            **runtime,
            "provenance_launcher": str(runtime["provenance_launcher"]),
        },
        "training": contract,
        "artifact_validation": {
            "required": contract["artifact_kind"] != "none",
            "status": (
                "not_checked"
                if contract["artifact_kind"] != "none"
                else "not_required"
            ),
        },
    }
    if validate_artifacts:
        artifacts = _validate_dataset_artifacts(
            payload, runtime, contract
        )
        if artifacts is not None:
            plan["holdout"] = artifacts
        plan["artifact_validation"]["status"] = (
            "passed" if artifacts is not None else "not_required"
        )
    return plan


def _print_plan(plan: Mapping[str, Any]) -> None:
    runtime = plan["runtime"]
    training = plan["training"]
    holdout = plan.get("holdout", {})
    lines = [
        ("Config", plan["config_path"]),
        ("Config SHA-256", plan["config_sha256"]),
        ("Container image", runtime["container_image"]),
        ("Container build", runtime["container_build"]),
        ("Scratch root", runtime["scratch_root"]),
        (
            "Hardware",
            f"{runtime['expected_gpu_count']}x {runtime['expected_gpu_name_contains']} "
            f"SM{runtime['expected_compute_capability'][0]}{runtime['expected_compute_capability'][1]}",
        ),
        (
            "Distributed",
            f"{runtime['num_processes']} processes, {runtime['mixed_precision']}, "
            f"DeepSpeed={runtime['use_deepspeed']}, compile={runtime['torch_compile_environment']}",
        ),
        ("Run root", training["run_root_dir"]),
        ("Run ID prefix", training["run_id_prefix"]),
        ("Model", f"{training['base_vlm']} + {training['action_model']}"),
        ("Attention", f"{training['attention']}, FLA={training['fast_linear_attention']}"),
        ("Dataset profile", training["dataset_profile"]),
        (
            "Data-contract artifacts",
            plan["artifact_validation"]["status"],
        ),
        (
            "Policy representation",
            f"state={training['state_dim']}D action={training['action_dim']}D "
            f"AH={training['action_horizon']} {training['representation']}",
        ),
        ("Dataset", f"{training['data_mix']} at {training['data_root_dir']}"),
        (
            "Pinned helper repositories",
            {
                name: f"{entry['commit'][:12]} at {entry['path']}"
                for name, entry in runtime["helper_repositories"].items()
            },
        ),
        (
            "Batch",
            f"{training['per_device_batch_size']}/GPU, global={training['global_batch_size']}",
        ),
        (
            "Checkpoint eval batch",
            (
                "disabled"
                if training["evaluation_observation_count"] is None
                else (
                    f"{training['eval_per_device_batch_size']}/GPU, "
                    f"{training['evaluation_observation_count']} total observations"
                )
            ),
        ),
        ("Workers/video", f"{training['num_workers']}/rank, {training['video_backend']}"),
        ("Subtask prompt probability", training["subtask_prompt_probability"]),
        (
            "Duration",
            f"epochs={training['epochs']}, max_train_steps={training['max_train_steps']}",
        ),
        (
            "Warmup",
            f"optimizer={training['warmup_steps']}, world-model={training['wm_warmup_steps']}",
        ),
        (
            "Checkpoint/eval",
            f"save={training['save_interval']}, eval={training['eval_interval']}, "
            f"milestones={training['checkpoint_eval_milestone_steps']}, "
            f"keep={training['checkpoint_max_to_keep']}",
        ),
        (
            "Completion/recovery",
            f"save_final_model={training['save_final_model']}, "
            f"force_checkpoint_file={training['enable_force_checkpoint_file']}",
        ),
        ("Optimizer", training["optimizer"]),
        ("Learning rates", training["learning_rate"]),
        ("Strict LR groups", training["strict_learning_rate_groups"]),
        ("Loss scales", training["loss_scale"]),
        ("Diffusion repeats", training["repeated_diffusion_steps"]),
        (
            "Best checkpoint metric",
            f"{training['best_metric_name']} ({training['best_metric_mode']})",
        ),
    ]
    if holdout and holdout.get("kind") == REALMAN_LEROBOT_PROFILE:
        lines.extend(
            [
                ("Holdout manifest", holdout["path"]),
                ("Holdout SHA-256", holdout["sha256"]),
                (
                    "Train/holdout episodes",
                    f"{holdout['train_episode_count']} / {holdout['holdout_episode_count']}",
                ),
                ("18-D statistics SHA-256", holdout["action_statistics_sha256"]),
            ]
        )
    elif holdout and holdout.get("kind") == CANONICAL_GCS_PROFILE:
        lines.extend(
            [
                ("Canonical eval manifest", holdout["path"]),
                ("Canonical eval SHA-256", holdout["sha256"]),
                (
                    "Canonical holdout episodes/windows",
                    f"{holdout['holdout_episode_count']} / "
                    f"{holdout['window_count']}",
                ),
                (
                    "Canonical source SHA-256",
                    holdout["source_manifest_sha256"],
                ),
            ]
        )
    elif training["dataset_profile"] == LIBERO_LEROBOT_PROFILE:
        lines.append(
            (
                "Checkpoint evaluation",
                "disabled (no immutable LIBERO holdout manifest configured)",
            )
        )
    if training["dataset_profile"] == CANONICAL_GCS_PROFILE:
        lines.append(
            (
                "Canonical eval catalog floor",
                (
                    f"{training['canonical_eval_min_episodes_per_shard']} "
                    "episodes/shard"
                ),
            )
        )
    print("\nResolved H100x8 training plan")
    print("=" * 31)
    width = max(len(label) for label, _ in lines)
    for label, value in lines:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        print(f"{label:<{width}} : {value}")
    print()


def _check_git(
    config_path: Path,
    required: bool,
    *,
    require_tracked: bool = False,
) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    try:
        config_repo_path = config_path.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        raise PlanError(f"production config must live inside the repository: {config_path}") from exc
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "--error-unmatch",
            str(config_repo_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if (required or require_tracked) and tracked.returncode != 0:
        raise PlanError(f"production config must be tracked by Git: {config_path}")
    if required and status.strip():
        preview = "\n".join(status.splitlines()[:20])
        raise PlanError(
            "runtime.require_clean_git=true but the source checkout is dirty; "
            f"commit the reviewed config/code first:\n{preview}"
        )
    return {"commit": commit, "status": "clean" if not status.strip() else "dirty"}


def _check_files(plan: Mapping[str, Any]) -> None:
    paths = [
        Path(plan["runtime"]["scratch_root"]),
        Path(plan["training"]["data_root_dir"]),
        Path(plan["training"]["run_root_dir"]),
    ]
    for path in paths:
        if not path.exists():
            raise PlanError(f"required mounted path does not exist: {path}")
    for name, entry in plan["runtime"]["helper_repositories"].items():
        path = Path(entry["path"])
        if not (path / ".git").is_dir():
            raise PlanError(
                f"pinned helper repository {name} is missing at {path}; run the setup command"
            )
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        actual = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if status.strip():
            raise PlanError(f"pinned helper repository {name} is dirty: {path}")
        if actual != entry["commit"]:
            raise PlanError(
                f"pinned helper repository {name} is at {actual}, expected {entry['commit']}"
            )
    _required_container_image_identity(plan)


def _required_container_image_identity(
    plan: Mapping[str, Any],
) -> tuple[str, str]:
    """Authenticate the configured tag and host-resolved local Docker Image.Id.

    A tag is intentionally config-owned, but tags are mutable.  The host
    launcher resolves that tag once and starts Docker by the resulting Image.Id
    while forwarding both identities here.  Requiring both values prevents a
    direct Python invocation or a tag retarget between check and container
    creation from weakening launch provenance.
    """

    configured_image = str(plan["runtime"]["container_image"])
    actual_image = os.environ.get("STARVLA_CONTAINER_IMAGE", "").strip()
    if not actual_image:
        raise PlanError(
            "STARVLA_CONTAINER_IMAGE is required; use the human-facing H100 "
            "launcher so the configured image tag is authenticated"
        )
    if actual_image != configured_image:
        raise PlanError(
            f"container image mismatch: config={configured_image!r}, "
            f"host={actual_image!r}"
        )
    image_id = os.environ.get("STARVLA_CONTAINER_IMAGE_ID", "").strip()
    if not image_id:
        raise PlanError(
            "STARVLA_CONTAINER_IMAGE_ID is required; the human-facing H100 "
            "launcher must resolve Docker's immutable local Image.Id"
        )
    if CONTAINER_IMAGE_ID_RE.fullmatch(image_id) is None:
        raise PlanError(
            "STARVLA_CONTAINER_IMAGE_ID must be sha256:<64 lowercase hex>"
        )
    legacy_digest = os.environ.get(
        "STARVLA_CONTAINER_IMAGE_DIGEST", ""
    ).strip()
    if legacy_digest and legacy_digest != image_id:
        raise PlanError(
            "STARVLA_CONTAINER_IMAGE_DIGEST contradicts "
            "STARVLA_CONTAINER_IMAGE_ID"
        )
    return actual_image, image_id


def _check_hardware(plan: Mapping[str, Any]) -> None:
    import torch

    expected_count = int(plan["runtime"]["expected_gpu_count"])
    actual_count = torch.cuda.device_count()
    if actual_count != expected_count:
        raise PlanError(f"expected exactly {expected_count} GPUs, found {actual_count}")
    expected_name = str(plan["runtime"]["expected_gpu_name_contains"])
    expected_capability = tuple(plan["runtime"]["expected_compute_capability"])
    problems: list[str] = []
    for index in range(actual_count):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        if expected_name.lower() not in name.lower():
            problems.append(f"GPU {index} is {name!r}, expected name containing {expected_name!r}")
        if capability != expected_capability:
            problems.append(
                f"GPU {index} capability is {capability}, expected {expected_capability}"
            )
    if problems:
        raise PlanError("hardware contract failed:\n" + "\n".join(problems))


def _discover_default_interface() -> str:
    route_path = Path("/proc/net/route")
    if route_path.is_file():
        for line in route_path.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 4 and fields[1] == "00000000" and int(fields[3], 16) & 2:
                return fields[0]
    raise PlanError(
        "runtime.network_interface=auto but no default-route interface was found"
    )


def _check_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            raise PlanError(f"runtime.main_process_port {port} is already in use") from exc


def _check_canonical_gcs_access(plan: Mapping[str, Any]) -> None:
    """Prove the exact CLI/auth path used by canonical shard downloads works."""

    training = plan["training"]
    if (
        training["dataset_profile"] != CANONICAL_GCS_PROFILE
        or not bool(training["requires_gcloud"])
    ):
        return
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise PlanError(
            "canonical GCS downloads require the gcloud CLI inside the training "
            "container. Install/mount Google Cloud SDK and stage an authenticated "
            "GCLOUD_CONFIG_DIR before setup/check."
        )
    try:
        auth = subprocess.run(
            [
                gcloud,
                "auth",
                "list",
                "--filter=status:ACTIVE",
                "--format=value(account)",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise PlanError(
            "gcloud is installed but its active-account check failed inside the "
            "training container"
        ) from exc
    account = auth.stdout.strip()
    if not account:
        raise PlanError(
            "canonical GCS downloads require an active gcloud account inside the "
            "training container; run `gcloud auth login` (or activate a service "
            "account) using the mounted GCLOUD_CONFIG_DIR"
        )
    bucket_root = str(training["canonical_bucket_root"]).rstrip("/") + "/"
    probe_object = str(training["canonical_gcs_probe_object"])
    try:
        subprocess.run(
            [gcloud, "storage", "ls", probe_object],
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        stderr = (
            str(getattr(exc, "stderr", "") or "").strip().replace("\n", " ")
        )
        detail = f": {stderr[:500]}" if stderr else ""
        raise PlanError(
            "active gcloud credentials cannot read the configured canonical "
            f"probe object {probe_object!r} below {bucket_root!r}{detail}"
        ) from exc
    print(f"Canonical GCS access        : PASS ({account})")


def _run_deep_preflight(config_path: Path, plan: Mapping[str, Any]) -> None:
    cfg, _ = _load_config(config_path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="starvla-h100-preflight-",
        suffix=".yaml",
        dir="/tmp",
    ) as handle:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))
        handle.flush()
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/preflight_runtime.py"),
                "--require-cuda",
                "--require-moge",
                "--config-yaml",
                handle.name,
            ],
            check=True,
            cwd=REPO_ROOT,
        )
        if plan["training"]["fast_linear_attention"]:
            subprocess.run(
                [
                    sys.executable,
                    str(
                        REPO_ROOT
                        / "scripts/probe_qwen35_fast_linear_attention.py"
                    ),
                    "--expected-compute-capability",
                    "9.0",
                ],
                check=True,
                cwd=REPO_ROOT,
            )


def _run_plan_checks(
    plan: Mapping[str, Any],
    *,
    git_config_path: Path,
    deep: bool,
    require_tracked_config: bool = False,
) -> dict[str, Any]:
    _print_plan(plan)
    if require_tracked_config:
        git = _check_git(
            git_config_path.resolve(),
            bool(plan["runtime"]["require_clean_git"]),
            require_tracked=True,
        )
    else:
        git = _check_git(
            git_config_path.resolve(),
            bool(plan["runtime"]["require_clean_git"]),
        )
    _check_files(plan)
    _check_canonical_gcs_access(plan)
    _check_hardware(plan)
    _check_port(int(plan["runtime"]["main_process_port"]))
    if deep:
        _run_deep_preflight(Path(str(plan["config_path"])), plan)
    print(f"Source commit               : {git['commit']}")
    print("H100x8 preflight            : PASS")
    return dict(plan)


def check_plan(config_path: Path, *, deep: bool) -> dict[str, Any]:
    plan = resolve_plan(config_path)
    return _run_plan_checks(
        plan,
        git_config_path=config_path,
        deep=deep,
    )


def check_curriculum_materialized_plan(
    materialized_config_path: Path,
    *,
    source_config_path: Path,
    expected_source_config_sha256: str,
    expected_materialized_config_sha256: str,
    expected_run_id: str,
    expected_pretrained_checkpoint: str | None,
    expected_pretrained_checkpoint_sha256: str | None,
    expected_curriculum_handoff: Mapping[str, Any],
    base_materialized_config_path: Path | None = None,
    expected_base_materialized_config_sha256: str | None = None,
    expected_resume_checkpoint: Path | None = None,
    deep: bool,
) -> dict[str, Any]:
    """Check one external curriculum config without weakening Git provenance.

    Curriculum configs are necessarily written under the run/state root
    because they bind a dynamic run ID and an authenticated predecessor model.
    This narrow entrypoint still checks Git against the tracked source YAML and
    proves that the external YAML differs only in those reviewed fields plus
    immutable handoff provenance.  Arbitrary external configs never reach the
    runtime checks.
    """

    source_config_path = source_config_path.expanduser()
    if source_config_path.is_symlink():
        raise PlanError(
            "tracked curriculum source config must be a regular non-symlink "
            f"file: {source_config_path}"
        )
    source_config_path = source_config_path.resolve()
    materialized_config_path = materialized_config_path.expanduser()
    if materialized_config_path.is_symlink():
        raise PlanError(
            "materialized curriculum config must be a regular non-symlink "
            f"file: {materialized_config_path}"
        )
    materialized_config_path = materialized_config_path.resolve()
    if base_materialized_config_path is not None:
        base_materialized_config_path = (
            base_materialized_config_path.expanduser()
        )
        if base_materialized_config_path.is_symlink():
            raise PlanError(
                "base materialized curriculum config must be a regular "
                "non-symlink file: "
                f"{base_materialized_config_path}"
            )
        base_materialized_config_path = base_materialized_config_path.resolve()
    sha_re = re.compile(r"[0-9a-f]{64}")
    if sha_re.fullmatch(expected_source_config_sha256) is None:
        raise PlanError("expected source config SHA-256 is malformed")
    if sha_re.fullmatch(expected_materialized_config_sha256) is None:
        raise PlanError("expected materialized config SHA-256 is malformed")
    if (base_materialized_config_path is None) != (
        expected_base_materialized_config_sha256 is None
    ):
        raise PlanError(
            "base materialized curriculum config path and SHA-256 must be "
            "supplied together"
        )
    if (
        expected_base_materialized_config_sha256 is not None
        and sha_re.fullmatch(expected_base_materialized_config_sha256) is None
    ):
        raise PlanError("expected base materialized config SHA-256 is malformed")
    if RUN_ID_RE.fullmatch(expected_run_id) is None:
        raise PlanError("expected curriculum stage run ID is malformed")
    if (expected_pretrained_checkpoint is None) != (
        expected_pretrained_checkpoint_sha256 is None
    ):
        raise PlanError(
            "expected predecessor checkpoint path and SHA-256 must be supplied "
            "together"
        )
    if (
        expected_pretrained_checkpoint_sha256 is not None
        and sha_re.fullmatch(expected_pretrained_checkpoint_sha256) is None
    ):
        raise PlanError("expected predecessor checkpoint SHA-256 is malformed")
    if not isinstance(expected_curriculum_handoff, Mapping):
        raise PlanError("expected curriculum handoff must be an object")
    if (base_materialized_config_path is None) != (
        expected_resume_checkpoint is None
    ):
        raise PlanError(
            "an exact expected resume checkpoint is required iff a base "
            "materialized curriculum config is supplied"
        )

    def load_authenticated_materialization(
        path: Path,
        expected_sha256: str,
    ) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise PlanError(
                "materialized curriculum config must be a regular non-symlink "
                f"file: {path}"
            )
        if _sha256(path) != expected_sha256:
            raise PlanError(
                "materialized curriculum config SHA-256 does not match the "
                "launcher-authenticated bytes"
            )
        _, payload = _load_config(path)
        return payload

    source_cfg, source_payload = _load_config(source_config_path)
    actual_source_sha = _config_contract_sha256(
        source_config_path,
        source_cfg,
    )
    if actual_source_sha != expected_source_config_sha256:
        raise PlanError(
            "tracked curriculum source config no longer matches the planned "
            "source contract SHA-256"
        )
    base_path = base_materialized_config_path or materialized_config_path
    base_sha = (
        expected_base_materialized_config_sha256
        or expected_materialized_config_sha256
    )
    base_payload = load_authenticated_materialization(base_path, base_sha)
    handoff = base_payload.get("curriculum_handoff")
    if not isinstance(handoff, Mapping):
        raise PlanError(
            "materialized curriculum config lacks immutable handoff provenance"
        )
    if (
        handoff.get("source_stage_config_path")
        != str(source_config_path)
        or handoff.get("source_stage_config_sha256")
        != expected_source_config_sha256
    ):
        raise PlanError(
            "materialized curriculum handoff does not authenticate its tracked "
            "source config path and contract hash"
        )

    expected_payload = copy.deepcopy(source_payload)
    expected_payload["run_id"] = expected_run_id
    expected_payload["trainer"]["pretrained_checkpoint"] = (
        expected_pretrained_checkpoint
    )
    expected_payload["trainer"]["pretrained_checkpoint_sha256"] = (
        expected_pretrained_checkpoint_sha256
    )
    expected_payload["curriculum_handoff"] = copy.deepcopy(
        dict(expected_curriculum_handoff)
    )
    if base_payload != expected_payload:
        raise PlanError(
            "materialized curriculum config changed settings outside runtime "
            "identity or differs from the independently authenticated "
            "predecessor path/hash and immutable handoff provenance"
        )

    allow_transport_resume = base_materialized_config_path is not None
    if allow_transport_resume:
        materialized_payload = load_authenticated_materialization(
            materialized_config_path,
            expected_materialized_config_sha256,
        )
        expected_resume_payload = copy.deepcopy(base_payload)
        expected_resume_payload["trainer"]["is_resume"] = True
        expected_resume_payload["trainer"]["resume_from_checkpoint"] = str(
            expected_resume_checkpoint
        )
        if materialized_payload != expected_resume_payload:
            raise PlanError(
                "materialized curriculum resume config changed settings "
                "outside the exact authenticated same-stage full-state "
                "checkpoint binding"
            )
    plan = resolve_plan(
        materialized_config_path,
        allow_transport_resume=allow_transport_resume,
    )
    if allow_transport_resume:
        validated_checkpoint, _ = _validate_checkpoint(
            expected_resume_checkpoint,
            int(plan["runtime"]["num_processes"]),
        )
        if validated_checkpoint != expected_resume_checkpoint.expanduser().resolve(
            strict=True
        ):
            raise PlanError(
                "validated resume checkpoint differs from the exact "
                "same-stage checkpoint selected by the curriculum"
            )
    return _run_plan_checks(
        plan,
        git_config_path=source_config_path,
        deep=deep,
        require_tracked_config=True,
    )


def _validate_checkpoint(checkpoint: Path, expected_ranks: int) -> tuple[Path, Path]:
    checkpoint = checkpoint.expanduser()
    if checkpoint.is_symlink():
        raise PlanError(
            f"resume checkpoint must be a regular non-symlink directory: "
            f"{checkpoint}"
        )
    checkpoint = checkpoint.resolve(strict=True)
    match = CHECKPOINT_RE.fullmatch(checkpoint.name)
    if not checkpoint.is_dir() or match is None or checkpoint.parent.name != "checkpoints":
        raise PlanError("resume checkpoint must be a .../<run_id>/checkpoints/steps_N directory")
    required = [
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "trainer_state.json",
        *(f"random_states_{rank}.pkl" for rank in range(expected_ranks)),
    ]
    missing = [
        name
        for name in required
        if (checkpoint / name).is_symlink()
        or not (checkpoint / name).is_file()
    ]
    if missing:
        raise PlanError(
            "resume checkpoint is incomplete or contains symlinked required "
            f"artifacts: {missing}"
        )
    run_dir = checkpoint.parent.parent
    source_config = run_dir / "config.yaml"
    if source_config.is_symlink() or not source_config.is_file():
        raise PlanError(f"resume run is missing immutable config.yaml: {run_dir}")
    return checkpoint, source_config


def _load_immutable_run_config(
    config_path: Path,
) -> tuple[DictConfig, dict[str, Any]]:
    """Load the trainer-frozen YAML/JSON pair and reject provenance drift."""

    config_path = config_path.expanduser()
    if config_path.is_symlink():
        raise PlanError(
            f"immutable run config must be a regular non-symlink file: "
            f"{config_path}"
        )
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise PlanError(
            f"immutable run config must be a regular non-symlink file: "
            f"{config_path}"
        )
    json_path = config_path.with_suffix(".json")
    if json_path.is_symlink() or not json_path.is_file():
        raise PlanError(
            f"immutable run config is missing its regular JSON twin: "
            f"{json_path}"
        )
    cfg = OmegaConf.load(config_path)
    payload = _plain(cfg)
    if not isinstance(payload, dict):
        raise PlanError("immutable run config root must be a mapping")
    try:
        json_payload = json.loads(json_path.read_text(encoding="utf-8"))
        normalized_payload = json.loads(
            json.dumps(payload, allow_nan=False)
        )
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise PlanError(
            f"immutable run config JSON is unreadable: {json_path}"
        ) from exc
    if json_payload != normalized_payload:
        raise PlanError(
            "immutable run config YAML/JSON mismatch; refusing resume: "
            f"{config_path} vs {json_path}"
        )
    return cfg, payload


def _resolved_launch_config(
    source_config: Path,
    plan: Mapping[str, Any],
    *,
    run_id: str | None,
    resume_checkpoint: Path | None,
) -> tuple[Path, str]:
    current_image, current_image_id = _required_container_image_identity(plan)
    if resume_checkpoint is None:
        _, resolved_payload = _load_config(source_config)
        # Freeze source-profile interpolations before assigning a unique runtime
        # run ID.  In particular, a canonical eval manifest prepared for the
        # reviewed profile prefix must not silently move to a nonexistent path.
        cfg = OmegaConf.create(resolved_payload)
        if run_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"{plan['training']['run_id_prefix']}_{timestamp}"
        if RUN_ID_RE.fullmatch(run_id) is None:
            raise PlanError(f"invalid run ID: {run_id!r}")
        if not run_id.startswith(f"{plan['training']['run_id_prefix']}_"):
            raise PlanError(
                f"run ID must start with {plan['training']['run_id_prefix']}_"
            )
        output = Path(plan["training"]["run_root_dir"]) / run_id
        if output.exists():
            raise PlanError(f"fresh run output already exists: {output}")
        cfg.run_id = run_id
        human_launch = {
            "schema": "starvla-human-launch-v2",
            "source_config_path": str(source_config),
            "source_config_sha256": plan["config_sha256"],
            "launcher_path": str(Path(__file__).resolve()),
            "launcher_sha256": _sha256(Path(__file__).resolve()),
            "source_commit": subprocess.run(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip(),
            "container_image": current_image,
            "container_image_id": current_image_id,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        cfg.human_launch = human_launch
        expected_payload = copy.deepcopy(resolved_payload)
        expected_payload["run_id"] = run_id
        expected_payload["human_launch"] = human_launch
        if OmegaConf.to_container(cfg, resolve=True) != expected_payload:
            raise PlanError(
                "fresh launch materialization changed settings outside runtime "
                "identity and immutable launch-provenance metadata"
            )
    else:
        checkpoint, immutable_config = _validate_checkpoint(
            resume_checkpoint, int(plan["runtime"]["num_processes"])
        )
        cfg, immutable_payload = _load_immutable_run_config(
            immutable_config
        )
        run_id = str(cfg.run_id)
        recorded_source_sha = cfg.get("human_launch", {}).get(
            "source_config_sha256"
        )
        if recorded_source_sha != plan["config_sha256"]:
            raise PlanError(
                "resume source profile SHA does not match the run's recorded profile"
            )
        recorded_image = cfg.get("human_launch", {}).get(
            "container_image"
        )
        recorded_image_id = cfg.get("human_launch", {}).get(
            "container_image_id"
        )
        if recorded_image != current_image:
            raise PlanError(
                "current configured container image tag does not match the "
                "immutable run launch provenance"
            )
        if recorded_image_id != current_image_id:
            raise PlanError(
                "current Docker Image.Id does not match the immutable run "
                "launch provenance"
            )
        cfg.trainer.is_resume = True
        cfg.trainer.resume_from_checkpoint = str(checkpoint)
        expected_payload = copy.deepcopy(immutable_payload)
        expected_payload["trainer"]["is_resume"] = True
        expected_payload["trainer"]["resume_from_checkpoint"] = str(checkpoint)
        if OmegaConf.to_container(cfg, resolve=True) != expected_payload:
            raise PlanError(
                "resume launch materialization changed settings outside the "
                "authenticated full-state resume checkpoint binding"
            )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"starvla-{run_id}-",
        suffix=".yaml",
        dir="/tmp",
        delete=False,
    )
    try:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))
    finally:
        handle.close()
    return Path(handle.name), run_id


def _accelerate_command(plan: Mapping[str, Any], resolved_config: Path) -> list[str]:
    runtime = plan["runtime"]
    return [
        "accelerate",
        "launch",
        "--num_processes",
        str(runtime["num_processes"]),
        "--num_machines",
        str(runtime["num_machines"]),
        "--mixed_precision",
        str(runtime["mixed_precision"]),
        "--dynamo_backend",
        str(runtime["dynamo_backend"]),
        "--main_process_port",
        str(runtime["main_process_port"]),
        "./starVLA/training/train_starvla.py",
        "--config_yaml",
        str(resolved_config),
    ]


def launch(
    config_path: Path,
    *,
    run_id: str | None,
    resume_checkpoint: Path | None,
    print_command_only: bool,
) -> None:
    plan = check_plan(config_path, deep=not print_command_only)
    resolved_config, resolved_run_id = _resolved_launch_config(
        config_path.resolve(),
        plan,
        run_id=run_id,
        resume_checkpoint=resume_checkpoint,
    )
    # Parse and validate the exact generated YAML that will reach the trainer,
    # rather than constructing the command/environment from the source plan.
    # `_resolved_launch_config` already proves the only fresh changes are
    # run_id/human_launch and the only resume changes are the two authenticated
    # transport fields.
    materialized_plan = resolve_plan(
        resolved_config,
        allow_transport_resume=resume_checkpoint is not None,
    )
    command = _accelerate_command(materialized_plan, resolved_config)
    print(f"Resolved run ID             : {resolved_run_id}")
    print(f"Resolved launch config      : {resolved_config}")
    print(f"Training command            : {shlex.join(command)}")
    if print_command_only:
        return
    runtime = materialized_plan["runtime"]
    env = os.environ.copy()
    for name in AUTHORITATIVE_SEMANTIC_ENV_VARS:
        env.pop(name, None)
    env["STARVLA_USE_DEEPSPEED"] = "1" if runtime["use_deepspeed"] else "0"
    if runtime["torch_compile_environment"] == "disabled":
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["STARVLA_ALLOW_TORCH_COMPILE"] = "0"
    env["VLA_JEPA_MAIN_TORCH_THREADS"] = str(
        runtime["main_torch_threads"]
    )
    env["VLA_JEPA_MAIN_TORCH_INTEROP_THREADS"] = str(
        runtime["main_torch_interop_threads"]
    )
    env["VLA_JEPA_DISABLE_AUTOGRAD_MULTITHREADING"] = (
        "1" if runtime["disable_autograd_multithreading"] else "0"
    )
    env["PYTORCH_CUDA_ALLOC_CONF"] = str(
        runtime["pytorch_cuda_alloc_conf"]
    )
    env["TOKENIZERS_PARALLELISM"] = (
        "true" if runtime["tokenizers_parallelism"] else "false"
    )
    interface = str(runtime["network_interface"])
    if interface == "auto":
        interface = _discover_default_interface()
    env["NCCL_SOCKET_IFNAME"] = interface
    env["GLOO_SOCKET_IFNAME"] = interface
    print(f"Network interface           : {interface}")
    sys.stdout.flush()
    os.chdir(REPO_ROOT)
    os.execvpe(command[0], command, env)


def prepare(config_path: Path, *, confirmed: bool) -> None:
    if not confirmed:
        raise PlanError(
            "prepare rewrites the configured holdout/statistics artifacts; rerun "
            "with --yes-rebuild-data-contract after reviewing the config"
        )
    plan = resolve_plan(config_path, validate_artifacts=False)
    _print_plan(plan)
    profile = plan["training"]["dataset_profile"]
    if profile == LIBERO_LEROBOT_PROFILE:
        raise PlanError(
            "LIBERO has no launcher-managed immutable holdout artifact to prepare; "
            "review the config, then run check"
        )
    if profile == CANONICAL_GCS_PROFILE:
        manifest = _resolve_artifact_path(
            str(plan["training"]["canonical_eval_manifest"]),
            field="datasets.vla_data.canonical_eval_manifest",
        )
        generator = REPO_ROOT / "scripts/generate_canonical_eval_manifest.py"
        if not generator.is_file():
            raise PlanError(
                f"canonical eval manifest generator is missing: {generator}"
            )
        resolved_cfg, _ = _load_config(config_path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="starvla-canonical-prepare-",
            suffix=".yaml",
            dir="/tmp",
        ) as handle:
            handle.write(OmegaConf.to_yaml(resolved_cfg, resolve=True))
            handle.flush()
            subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    "--config",
                    handle.name,
                    "--output",
                    str(manifest),
                    "--world-size",
                    str(plan["runtime"]["num_processes"]),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
        verified = resolve_plan(config_path, validate_artifacts=True)
        _print_plan(verified)
        print(
            "Canonical eval manifest generated/reused and verified. "
            "Review and commit the profile/artifact contract."
        )
        return
    if profile != REALMAN_LEROBOT_PROFILE:  # pragma: no cover - exhaustive
        raise AssertionError(profile)
    manifest = _resolve_repo_path(
        str(plan["training"]["episode_split_manifest"]),
        field="datasets.vla_data.episode_split_manifest",
    )
    resolved_cfg, _ = _load_config(config_path)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="starvla-realman-prepare-",
        suffix=".yaml",
        dir="/tmp",
    ) as handle:
        # Direct configs are byte-bound exactly as reviewed. Composed configs
        # are bound to the deterministic fully resolved YAML. Use the same
        # shared bytes here so `prepare` never mutates the config identity.
        handle.write(
            _config_contract_bytes(config_path, resolved_cfg)
        )
        handle.flush()
        if _sha256(Path(handle.name)) != plan["config_sha256"]:
            raise PlanError(
                "resolved RealMan prepare config does not match the reviewed "
                "composed config contract"
            )
        subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "deployment/realman/build_magna_internal_holdout.py"
                ),
                "--dataset-root",
                str(plan["training"]["data_root_dir"]),
                "--config",
                handle.name,
                "--launcher",
                str(plan["runtime"]["provenance_launcher"]),
                "--world-size",
                str(plan["runtime"]["num_processes"]),
                "--manifest",
                str(manifest),
                "--overwrite",
            ],
            check=True,
            cwd=REPO_ROOT,
        )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/compute_openpi_realman_stats.py"),
            "--manifest",
            str(manifest),
            "--update-manifest",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    verified = resolve_plan(config_path, validate_artifacts=True)
    _print_plan(verified)
    print("Data contract rebuilt and verified. Review and commit every changed artifact.")


def setup_dependencies(config_path: Path) -> None:
    """Clone or validate every config-pinned helper repository."""

    plan = resolve_plan(config_path, validate_artifacts=False)
    _print_plan(plan)
    _check_canonical_gcs_access(plan)
    for name, entry in plan["runtime"]["helper_repositories"].items():
        path = Path(entry["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not (path / ".git").is_dir():
            if any(path.iterdir()):
                raise PlanError(
                    f"refusing to replace non-Git, non-empty helper path for {name}: {path}"
                )
            path.rmdir()
        if not path.exists():
            path.mkdir()
            subprocess.run(
                ["git", "-C", str(path), "init"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(path), "remote", "add", "origin", entry["url"]],
                check=True,
            )
        origin = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if _git_remote_identity(origin) != _git_remote_identity(
            entry["url"]
        ):
            raise PlanError(
                f"helper repository {name} origin is {origin!r}, expected {entry['url']!r}"
            )
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        if status.strip():
            raise PlanError(f"refusing to alter dirty helper repository {name}: {path}")
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
        )
        actual = head.stdout.strip() if head.returncode == 0 else ""
        if actual != entry["commit"]:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(path),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    entry["commit"],
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(path),
                    "checkout",
                    "--detach",
                    entry["commit"],
                ],
                check=True,
            )
            actual = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        if actual != entry["commit"]:
            raise PlanError(
                f"helper repository {name} checkout mismatch: {actual} != {entry['commit']}"
            )
        print(f"Pinned {name:<20} : {actual} at {path}")
    print("Pinned helper repositories  : PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, validate, prepare, or launch one config-only H100x8 run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "setup", "check", "prepare", "launch"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name == "plan":
            sub.add_argument("--json", action="store_true")
        elif name == "prepare":
            sub.add_argument("--yes-rebuild-data-contract", action="store_true")
        elif name == "launch":
            sub.add_argument("--run-id")
            sub.add_argument("--resume", type=Path)
            sub.add_argument("--print-command", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            # A human must be able to inspect a new canonical plan before its
            # launcher-managed holdout manifest exists.  `check` and `launch`
            # remain strict artifact gates; `prepare` creates/verifies it.
            plan = resolve_plan(args.config, validate_artifacts=False)
            if args.json:
                print(json.dumps(plan, indent=2, default=str, sort_keys=True))
            else:
                _print_plan(plan)
        elif args.command == "setup":
            setup_dependencies(args.config)
        elif args.command == "check":
            check_plan(args.config, deep=True)
        elif args.command == "prepare":
            prepare(
                args.config,
                confirmed=bool(args.yes_rebuild_data_contract),
            )
        elif args.command == "launch":
            launch(
                args.config,
                run_id=args.run_id,
                resume_checkpoint=args.resume,
                print_command_only=bool(args.print_command),
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (PlanError, subprocess.CalledProcessError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
