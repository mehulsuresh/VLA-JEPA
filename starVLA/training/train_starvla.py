# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""
import warnings
warnings.filterwarnings("ignore")

# Standard Library
import argparse
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import math
import json
import os
import random
import shutil
import sys
import logging
import queue
from pathlib import Path
import threading
import traceback
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def configure_runtime_logging() -> None:
    """Reduce noisy import-time probe logs from optional DeepSpeed builders."""
    for logger_name in (
        "deepspeed",
        "deepspeed.accelerator.real_accelerator",
        "deepspeed.ops.op_builder",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    try:
        import distutils.log as distutils_log

        distutils_log.set_threshold(distutils_log.WARN)
    except Exception:
        pass

    try:
        import setuptools._distutils.log as setuptools_distutils_log

        setuptools_distutils_log.set_threshold(setuptools_distutils_log.WARN)
    except Exception:
        pass


configure_runtime_logging()

# Third-Party Libraries
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("VLA_JEPA_MAIN_TORCH_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("VLA_JEPA_MAIN_TORCH_THREADS", "1"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("VLA_JEPA_MAIN_TORCH_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("VLA_JEPA_MAIN_TORCH_THREADS", "1"))
import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.tracking import LoggerType
from accelerate.utils import DistributedDataParallelKwargs, gather_object, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.trainer_tools import is_depth_teacher_aux_missing_key_allowed
from starVLA.training.trainer_utils.trainer_tools import is_depth_teacher_aux_unexpected_key_allowed
from starVLA.model.modules.action_model.rtc_training import rtc_training_probability

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import faulthandler
    import sys

    faulthandler.enable(file=sys.stderr, all_threads=True)
except Exception:
    pass

try:
    _main_torch_threads = max(1, int(os.environ.get("VLA_JEPA_MAIN_TORCH_THREADS", "1")))
    torch.set_num_threads(_main_torch_threads)
except Exception:
    pass

try:
    _main_torch_interop_threads = max(1, int(os.environ.get("VLA_JEPA_MAIN_TORCH_INTEROP_THREADS", "1")))
    torch.set_num_interop_threads(_main_torch_interop_threads)
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

_disable_autograd_mt = os.environ.get("VLA_JEPA_DISABLE_AUTOGRAD_MULTITHREADING", "1").strip().lower()
if _disable_autograd_mt not in {"0", "false", "no", "off"}:
    _set_autograd_mt = getattr(torch.autograd, "set_multithreading_enabled", None)
    if _set_autograd_mt is not None:
        _set_autograd_mt(False)


logger = get_logger(__name__)


def _is_torch_compile_exception(exc: BaseException) -> bool:
    compile_modules = ("torch._dynamo", "torch._inductor", "triton")
    pending = [exc]
    visited = set()
    while pending:
        current = pending.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)
        current_module = type(current).__module__
        current_name = type(current).__name__
        if current_module.startswith(compile_modules):
            return True
        if current_name in {"BackendCompilerFailed", "InductorError", "TorchRuntimeError"}:
            return True
        pending.extend((getattr(current, "__cause__", None), getattr(current, "__context__", None)))
    return False


def _resolve_compile_dynamic(trainer_cfg, key: str, default: bool) -> bool:
    value = trainer_cfg.get(key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _install_compiled_forward(
    module: torch.nn.Module,
    module_name: str,
    compile_mode: str,
    compile_backend: Optional[str],
    dynamic_attempts: list[bool],
    *,
    allow_eager_fallback: bool = True,
) -> None:
    eager_forward = module.forward
    attempted_compile_labels: set[str] = set()
    dynamic_attempts = list(dict.fromkeys(dynamic_attempts))
    active_dynamic: Optional[bool] = None
    active_impl = "uninitialized"
    compiled_forward_cache: dict[bool, Callable] = {}

    compile_backend = None if compile_backend in {None, "", "default", "none", "null"} else str(compile_backend)

    def _compile_forward(dynamic: bool):
        if dynamic not in compiled_forward_cache:
            compile_kwargs = {
                "dynamic": dynamic,
                "mode": compile_mode,
            }
            if compile_backend is not None:
                compile_kwargs["backend"] = compile_backend
            compiled_forward_cache[dynamic] = torch.compile(eager_forward, **compile_kwargs)
        return compiled_forward_cache[dynamic]

    last_compile_exc: Optional[BaseException] = None

    def _log_compile_failure(dynamic: bool, exc: BaseException) -> None:
        compile_label = (
            f"torch.compile(dynamic={dynamic}, mode='{compile_mode}', "
            f"backend='{compile_backend or 'default'}')"
        )
        if compile_label in attempted_compile_labels:
            return
        attempted_compile_labels.add(compile_label)
        logger.warning(
            f"{compile_label} failed for {module_name}; trying next fallback: {exc}"
        )

    def compiled_forward(*args, **kwargs):
        nonlocal active_dynamic, active_impl, last_compile_exc

        if active_impl == "eager":
            return eager_forward(*args, **kwargs)

        if active_impl == "compiled" and active_dynamic is not None:
            try:
                return _compile_forward(active_dynamic)(*args, **kwargs)
            except Exception as exc:
                if not _is_torch_compile_exception(exc):
                    raise
                _log_compile_failure(active_dynamic, exc)
                active_impl = "retry"

        attempted_dynamics = set()
        if active_dynamic is not None:
            attempted_dynamics.add(active_dynamic)

        for dynamic in dynamic_attempts:
            if dynamic in attempted_dynamics:
                continue
            try:
                output = _compile_forward(dynamic)(*args, **kwargs)
                active_dynamic = dynamic
                active_impl = "compiled"
                module._starvla_active_compile_impl = "compiled"
                module._starvla_active_compile_dynamic = dynamic
                logger.info(
                    f"Compiled {module_name} with torch.compile(dynamic={dynamic}, "
                    f"mode='{compile_mode}', backend='{compile_backend or 'default'}')"
                )
                return output
            except Exception as exc:
                if not _is_torch_compile_exception(exc):
                    raise
                last_compile_exc = exc
                _log_compile_failure(dynamic, exc)
                attempted_dynamics.add(dynamic)

        if not allow_eager_fallback:
            raise RuntimeError(
                f"torch.compile failed for {module_name}, and eager fallback is disabled."
            ) from last_compile_exc

        active_dynamic = None
        active_impl = "eager"
        module._starvla_active_compile_impl = "eager"
        module._starvla_active_compile_dynamic = None
        logger.warning(f"Falling back to eager forward for {module_name}")
        return eager_forward(*args, **kwargs)

    module._starvla_eager_forward = eager_forward
    module._starvla_compile_mode = compile_mode
    module._starvla_compile_backend = compile_backend or "default"
    module._starvla_compile_dynamic_attempts = tuple(dynamic_attempts)
    if not allow_eager_fallback:
        strict_dynamic = dynamic_attempts[0]
        compile_kwargs = {
            "dynamic": strict_dynamic,
            "mode": compile_mode,
        }
        if compile_backend is not None:
            compile_kwargs["backend"] = compile_backend
        module._starvla_active_compile_impl = "compiled"
        module._starvla_active_compile_dynamic = strict_dynamic
        module.forward = torch.compile(eager_forward, **compile_kwargs)
        logger.info(
            f"Installed strict compiled forward for {module_name} with "
            f"torch.compile(dynamic={strict_dynamic}, mode='{compile_mode}', "
            f"backend='{compile_backend or 'default'}')"
        )
        return

    module._starvla_active_compile_impl = "uninitialized"
    module._starvla_active_compile_dynamic = None
    module.forward = compiled_forward


def _install_compiled_callable_attr(
    owner,
    attr_name: str,
    target_name: str,
    compile_mode: str,
    compile_backend: Optional[str],
    dynamic_attempts: list[bool],
    *,
    allow_eager_fallback: bool = True,
) -> None:
    eager_callable = getattr(owner, attr_name)
    if not callable(eager_callable):
        raise TypeError(f"`{target_name}` is not callable")

    attempted_compile_labels: set[str] = set()
    dynamic_attempts = list(dict.fromkeys(dynamic_attempts))
    active_dynamic: Optional[bool] = None
    active_impl = "uninitialized"
    compiled_callable_cache: dict[bool, Callable] = {}
    compile_backend = None if compile_backend in {None, "", "default", "none", "null"} else str(compile_backend)

    def _compile_callable(dynamic: bool):
        if dynamic not in compiled_callable_cache:
            compile_kwargs = {
                "dynamic": dynamic,
                "mode": compile_mode,
            }
            if compile_backend is not None:
                compile_kwargs["backend"] = compile_backend
            compiled_callable_cache[dynamic] = torch.compile(eager_callable, **compile_kwargs)
        return compiled_callable_cache[dynamic]

    last_compile_exc: Optional[BaseException] = None

    def _log_compile_failure(dynamic: bool, exc: BaseException) -> None:
        compile_label = (
            f"torch.compile(dynamic={dynamic}, mode='{compile_mode}', "
            f"backend='{compile_backend or 'default'}')"
        )
        if compile_label in attempted_compile_labels:
            return
        attempted_compile_labels.add(compile_label)
        logger.warning(
            f"{compile_label} failed for {target_name}; trying next fallback: {exc}"
        )

    def compiled_callable(*args, **kwargs):
        nonlocal active_dynamic, active_impl, last_compile_exc

        if active_impl == "eager":
            return eager_callable(*args, **kwargs)

        if active_impl == "compiled" and active_dynamic is not None:
            try:
                return _compile_callable(active_dynamic)(*args, **kwargs)
            except Exception as exc:
                if not _is_torch_compile_exception(exc):
                    raise
                _log_compile_failure(active_dynamic, exc)
                active_impl = "retry"

        attempted_dynamics = set()
        if active_dynamic is not None:
            attempted_dynamics.add(active_dynamic)

        for dynamic in dynamic_attempts:
            if dynamic in attempted_dynamics:
                continue
            try:
                output = _compile_callable(dynamic)(*args, **kwargs)
                active_dynamic = dynamic
                active_impl = "compiled"
                logger.info(
                    f"Compiled {target_name} with torch.compile(dynamic={dynamic}, "
                    f"mode='{compile_mode}', backend='{compile_backend or 'default'}')"
                )
                return output
            except Exception as exc:
                if not _is_torch_compile_exception(exc):
                    raise
                last_compile_exc = exc
                _log_compile_failure(dynamic, exc)
                attempted_dynamics.add(dynamic)

        if not allow_eager_fallback:
            raise RuntimeError(
                f"torch.compile failed for {target_name}, and eager fallback is disabled."
            ) from last_compile_exc

        active_dynamic = None
        active_impl = "eager"
        logger.warning(f"Falling back to eager callable for {target_name}")
        return eager_callable(*args, **kwargs)

    if not allow_eager_fallback:
        strict_dynamic = dynamic_attempts[0]
        compile_kwargs = {
            "dynamic": strict_dynamic,
            "mode": compile_mode,
        }
        if compile_backend is not None:
            compile_kwargs["backend"] = compile_backend
        setattr(owner, attr_name, torch.compile(eager_callable, **compile_kwargs))
        logger.info(
            f"Installed strict compiled callable for {target_name} with "
            f"torch.compile(dynamic={strict_dynamic}, mode='{compile_mode}', "
            f"backend='{compile_backend or 'default'}')"
        )
        return

    setattr(owner, attr_name, compiled_callable)


def resolve_trackers(cfg):
    configured_trackers = cfg.get("trackers", [])
    if isinstance(configured_trackers, str):
        configured_trackers = [configured_trackers]

    valid_trackers = {tracker.value for tracker in LoggerType}
    resolved_trackers = []
    for tracker in configured_trackers:
        tracker_name = str(tracker).lower()
        if tracker_name in valid_trackers:
            resolved_trackers.append(tracker_name)
        else:
            logger.warning(f"Ignoring unsupported tracker '{tracker_name}'. Valid Accelerate trackers: {sorted(valid_trackers)}")
    return resolved_trackers


def flatten_tracker_config(config, prefix=""):
    flat = {}
    if isinstance(config, Mapping):
        for key, value in config.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_tracker_config(value, next_prefix))
        return flat

    if isinstance(config, Sequence) and not isinstance(config, (str, bytes, bytearray)):
        if all(isinstance(item, (bool, int, float, str)) for item in config):
            flat[prefix] = json.dumps(list(config))
        else:
            for idx, value in enumerate(config):
                next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                flat.update(flatten_tracker_config(value, next_prefix))
        return flat

    if isinstance(config, np.generic):
        config = config.item()

    if isinstance(config, (bool, int, float, str)):
        flat[prefix] = config
    elif config is None:
        flat[prefix] = "null"
    else:
        flat[prefix] = str(config)
    return flat


def resolve_mixed_precision_mode(cfg) -> str:
    trainer_cfg = cfg.get("trainer", {})
    requested_mode = trainer_cfg.get("mixed_precision", None)
    if isinstance(requested_mode, str) and requested_mode.lower() in {"no", "fp16", "bf16"}:
        return requested_mode.lower()

    if not bool(trainer_cfg.get("enable_mixed_precision_training", False)):
        return "no"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def build_accelerator(cfg) -> Accelerator:
    use_deepspeed = os.environ.get("STARVLA_USE_DEEPSPEED", "0") == "1"
    deepspeed_plugin = DeepSpeedPlugin() if use_deepspeed else None
    mixed_precision = resolve_mixed_precision_mode(cfg)
    trackers = resolve_trackers(cfg)
    project_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    trainer_cfg = cfg.get("trainer", {})
    raw_gradient_accumulation_steps = trainer_cfg.get(
        "gradient_accumulation_steps", 1
    )
    if isinstance(raw_gradient_accumulation_steps, bool):
        raise ValueError(
            "trainer.gradient_accumulation_steps must be a positive integer, "
            f"got {raw_gradient_accumulation_steps!r}."
        )
    try:
        gradient_accumulation_steps = int(raw_gradient_accumulation_steps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "trainer.gradient_accumulation_steps must be a positive integer, "
            f"got {raw_gradient_accumulation_steps!r}."
        ) from exc
    if gradient_accumulation_steps <= 0 or float(
        raw_gradient_accumulation_steps
    ) != float(gradient_accumulation_steps):
        raise ValueError(
            "trainer.gradient_accumulation_steps must be a positive integer, "
            f"got {raw_gradient_accumulation_steps!r}."
        )
    step_scheduler_with_optimizer = bool(trainer_cfg.get("step_scheduler_with_optimizer", False))
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(trainer_cfg.get("find_unused_parameters", True)),
        gradient_as_bucket_view=bool(trainer_cfg.get("ddp_gradient_as_bucket_view", False)),
        static_graph=bool(trainer_cfg.get("ddp_static_graph", False)),
        bucket_cap_mb=int(trainer_cfg.get("ddp_bucket_cap_mb", 25)),
    )
    accelerator = (
        Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            mixed_precision=mixed_precision,
            log_with=trackers or None,
            project_dir=project_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            step_scheduler_with_optimizer=step_scheduler_with_optimizer,
            kwargs_handlers=[ddp_kwargs],
        )
        if use_deepspeed
        else Accelerator(
            mixed_precision=mixed_precision,
            log_with=trackers or None,
            project_dir=project_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            step_scheduler_with_optimizer=step_scheduler_with_optimizer,
            kwargs_handlers=[ddp_kwargs],
        )
    )
    if torch.cuda.is_available():
        # Ensure NCCL collectives run on the process-local device before any early barrier().
        torch.cuda.set_device(accelerator.local_process_index)
    runtime_gradient_accumulation_steps = int(
        accelerator.gradient_accumulation_steps
    )
    if runtime_gradient_accumulation_steps != gradient_accumulation_steps:
        raise RuntimeError(
            "Accelerator gradient accumulation differs from the training config: "
            f"runtime={runtime_gradient_accumulation_steps}, "
            f"config={gradient_accumulation_steps}."
        )
    try:
        OmegaConf.update(
            cfg,
            "trainer._accelerate_distributed_type",
            str(accelerator.distributed_type).lower(),
            force_add=True,
        )
        OmegaConf.update(
            cfg,
            "trainer._accelerate_gradient_accumulation_steps",
            runtime_gradient_accumulation_steps,
            force_add=True,
        )
        OmegaConf.update(
            cfg,
            "trainer._accelerate_num_processes",
            int(accelerator.num_processes),
            force_add=True,
        )
        OmegaConf.update(
            cfg,
            "trainer._accelerate_step_scheduler_with_optimizer",
            step_scheduler_with_optimizer,
            force_add=True,
        )
    except Exception as exc:
        logger.warning(f"Could not record Accelerator runtime state in trainer config: {exc}")
    accelerator.print(accelerator.state)
    return accelerator


def distributed_wait(accelerator: Optional[Accelerator] = None) -> None:
    if dist.is_initialized():
        barrier_kwargs = {}
        if torch.cuda.is_available():
            barrier_kwargs["device_ids"] = [torch.cuda.current_device()]
        dist.barrier(**barrier_kwargs)
        return
    if accelerator is not None:
        accelerator.wait_for_everyone()


def finish_trackers(accelerator: Optional[Accelerator]) -> None:
    if accelerator is None:
        return
    for tracker in getattr(accelerator, "trackers", []):
        try:
            tracker.finish()
        except Exception as exc:
            logger.warning(f"Tracker shutdown failed for `{type(tracker).__name__}`: {exc}")


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


_RESUME_TRANSPORT_CONFIG_PATHS = (
    ("output_dir",),
    ("is_resume",),
    ("resume_from_checkpoint",),
    ("trainer", "is_resume"),
    ("trainer", "resume_from_checkpoint"),
)
_PENDING_RESUME_INVOCATION_YAML: dict[str, bytes] = {}


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace a small provenance file without exposing partial bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / (
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with temporary_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _plain_config(config) -> dict:
    if OmegaConf.is_config(config):
        plain = OmegaConf.to_container(config, resolve=True)
    else:
        plain = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    if not isinstance(plain, dict):
        raise TypeError(f"Expected a mapping config, got {type(plain).__name__}.")
    return plain


def _remove_nested_config_path(config: dict, path: tuple[str, ...]) -> None:
    current = config
    for component in path[:-1]:
        next_value = current.get(component)
        if not isinstance(next_value, dict):
            return
        current = next_value
    current.pop(path[-1], None)


def _canonical_resume_config(config) -> dict:
    canonical = _plain_config(config)
    for path in _RESUME_TRANSPORT_CONFIG_PATHS:
        _remove_nested_config_path(canonical, path)
    return canonical


def _config_difference_summaries(source: Mapping, invocation: Mapping) -> list[str]:
    source_flat = flatten_tracker_config(source)
    invocation_flat = flatten_tracker_config(invocation)
    differences = []
    for key in sorted(set(source_flat) | set(invocation_flat)):
        if key not in source_flat:
            differences.append(f"{key}: missing from source config")
        elif key not in invocation_flat:
            differences.append(f"{key}: missing from resume invocation")
        elif source_flat[key] != invocation_flat[key]:
            differences.append(
                f"{key}: source={source_flat[key]!r} resume={invocation_flat[key]!r}"
            )
        if len(differences) == 8:
            break
    return differences


def _persist_resume_invocation_snapshot(
    output_dir: Path,
    cfg,
    yaml_payload: bytes,
) -> Path:
    trainer_cfg = cfg.get("trainer", {})
    checkpoint = cfg.get("resume_from_checkpoint", None) or trainer_cfg.get(
        "resume_from_checkpoint", None
    )
    label = Path(str(checkpoint)).name if checkpoint else "resume"
    safe_label = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(label)
    )
    invocation_sha256 = hashlib.sha256(yaml_payload).hexdigest()
    snapshot_path = (
        output_dir
        / "resume_invocations"
        / f"{safe_label}-{invocation_sha256[:12]}.yaml"
    )
    if snapshot_path.is_file():
        if snapshot_path.read_bytes() != yaml_payload:
            raise RuntimeError(
                "Resume invocation snapshot is immutable and its content does "
                f"not match the validated invocation: {snapshot_path}"
            )
    else:
        _atomic_write_bytes(snapshot_path, yaml_payload)
    return snapshot_path


def _persist_pending_resume_invocation_snapshot(output_dir: Path, cfg) -> Path:
    key = str(output_dir.expanduser().resolve())
    yaml_payload = _PENDING_RESUME_INVOCATION_YAML.pop(key, None)
    if yaml_payload is None:
        raise RuntimeError(
            "Missing the validated pre-resolution resume invocation snapshot "
            f"for {output_dir}."
        )
    return _persist_resume_invocation_snapshot(output_dir, cfg, yaml_payload)


def _validate_resume_config(source_config_path: Path, cfg) -> None:
    source_config = OmegaConf.load(source_config_path)
    source_json_path = source_config_path.with_suffix(".json")
    if not source_json_path.is_file():
        raise RuntimeError(
            f"Resume output is missing immutable {source_json_path.name}: "
            f"{source_json_path.parent}"
        )
    try:
        source_json = json.loads(source_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Immutable source config JSON is unreadable: {source_json_path}"
        ) from exc
    json_normalized_source = json.loads(
        json.dumps(_plain_config(source_config), allow_nan=False)
    )
    if json_normalized_source != source_json:
        raise RuntimeError(
            "Immutable source config YAML/JSON mismatch; refusing resume: "
            f"{source_config_path} vs {source_json_path}"
        )
    canonical_source = _canonical_resume_config(source_config)
    canonical_invocation = _canonical_resume_config(cfg)
    if canonical_source == canonical_invocation:
        return
    differences = _config_difference_summaries(
        canonical_source,
        canonical_invocation,
    )
    detail = "; ".join(differences) if differences else "unknown difference"
    raise RuntimeError(
        "Resume configuration drift detected outside the permitted resume "
        "transport fields; refusing to overwrite the immutable source config. "
        f"Differences: {detail}"
    )


def setup_directories(cfg) -> Path:
    """Create output directories and preserve immutable run provenance."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        preexisting_entries = (
            list(output_dir.iterdir()) if output_dir.is_dir() else []
        )
        yaml_payload = OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")
        json_payload = (
            json.dumps(_plain_config(cfg), indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        config_yaml_path = output_dir / "config.yaml"
        config_json_path = output_dir / "config.json"
        trainer_cfg = cfg.get("trainer", {})
        is_resume = bool(trainer_cfg.get("is_resume", False))
        eval_only = bool(trainer_cfg.get("eval_only", False))
        pending_key = str(output_dir.expanduser().resolve())
        _PENDING_RESUME_INVOCATION_YAML.pop(pending_key, None)

        if is_resume:
            if config_yaml_path.is_file():
                _validate_resume_config(config_yaml_path, cfg)
            elif preexisting_entries:
                raise RuntimeError(
                    "Resume output directory is missing its immutable source "
                    f"config.yaml: {output_dir}"
                )
        elif preexisting_entries:
            raise RuntimeError(
                "Fresh training output directory is not empty; use a new run_id "
                f"or an explicit validated resume: {output_dir}"
            )

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        if is_resume:
            if not config_yaml_path.is_file():
                # A checkpoint audit or branch into a new RUN_ID has no local
                # source run yet. Its invocation becomes that new output's
                # immutable source config.
                _atomic_write_bytes(config_yaml_path, yaml_payload)
                _atomic_write_bytes(config_json_path, json_payload)
            if eval_only:
                _persist_resume_invocation_snapshot(output_dir, cfg, yaml_payload)
            else:
                _PENDING_RESUME_INVOCATION_YAML[pending_key] = yaml_payload
        else:
            _atomic_write_bytes(config_yaml_path, yaml_payload)
            _atomic_write_bytes(config_json_path, json_payload)

    return output_dir


_TORCH_COMPILE_MODEL_FLAGS = (
    "compile_qwen_model",
    "compile_action_model",
    "compile_vj_predictor",
    "compile_vj_encoder",
    "compile_full_model",
)


def _requested_torch_compile_flags(trainer_cfg) -> list[str]:
    return [flag for flag in _TORCH_COMPILE_MODEL_FLAGS if bool(trainer_cfg.get(flag, False))]


def _training_uses_deepspeed(cfg) -> bool:
    trainer_cfg = cfg.get("trainer", {}) if cfg is not None else {}
    distributed_type = str(trainer_cfg.get("_accelerate_distributed_type", "")).lower()
    if "deepspeed" in distributed_type:
        return True
    if os.environ.get("STARVLA_USE_DEEPSPEED", "0") == "1":
        return True
    accelerate_distributed_type = os.environ.get("ACCELERATE_DISTRIBUTED_TYPE", "")
    if "deepspeed" in accelerate_distributed_type.lower():
        return True
    accelerate_use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "")
    return accelerate_use_deepspeed.lower() in {"1", "true", "yes"}


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    trainer_cfg = cfg.get("trainer", {})
    requested_compile_flags = _requested_torch_compile_flags(trainer_cfg)
    if (
        requested_compile_flags
        and _training_uses_deepspeed(cfg)
        and not bool(trainer_cfg.get("allow_compile_with_deepspeed", False))
    ):
        raise RuntimeError(
            "DeepSpeed training was requested with torch.compile flags enabled "
            f"({', '.join(requested_compile_flags)}). This combination is disabled by default "
            "because compiled module wrappers can interact badly with ZeRO partitioning and "
            "Accelerate prepare order. Set these compile flags to false for production DeepSpeed "
            "runs, or set trainer.allow_compile_with_deepspeed=true only after smoke-testing the "
            "exact ZeRO stage, world size, precision, and freeze policy."
        )

    from starVLA.model.framework import build_framework

    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    compile_mode = trainer_cfg.get("compile_mode", "reduce-overhead")
    compile_backend = trainer_cfg.get("compile_backend", "inductor")
    compile_dynamic = bool(trainer_cfg.get("compile_dynamic", True))
    allow_compile_eager_fallback = not bool(trainer_cfg.get("strict_torch_compile", False))
    if torch.cuda.is_available():
        if trainer_cfg.get("compile_qwen_model", False):
            try:
                qwen_iface = getattr(model, "qwen_vl_interface", None)
                if qwen_iface is None:
                    raise AttributeError("model has no qwen_vl_interface")
                if hasattr(qwen_iface, "prepare_for_compile"):
                    qwen_iface.prepare_for_compile()
                qwen_dynamic = _resolve_compile_dynamic(
                    trainer_cfg,
                    "compile_qwen_model_dynamic",
                    compile_dynamic,
                )
                _install_compiled_forward(
                    qwen_iface.model,
                    "qwen_vl_interface.model",
                    compile_mode=compile_mode,
                    compile_backend=compile_backend,
                    dynamic_attempts=[qwen_dynamic] if not qwen_dynamic else [True, False],
                    allow_eager_fallback=allow_compile_eager_fallback,
                )
                if hasattr(qwen_iface, "forward_features"):
                    _install_compiled_callable_attr(
                        qwen_iface,
                        "forward_features",
                        "qwen_vl_interface.forward_features",
                        compile_mode=compile_mode,
                        compile_backend=compile_backend,
                        dynamic_attempts=[qwen_dynamic] if not qwen_dynamic else [True, False],
                        allow_eager_fallback=allow_compile_eager_fallback,
                    )
            except Exception as exc:
                if not allow_compile_eager_fallback:
                    raise
                logger.warning(f"torch.compile failed for qwen_vl_interface.model, continuing without it: {exc}")

        if trainer_cfg.get("compile_action_model", False):
            try:
                if hasattr(model.action_model, "prepare_for_compile"):
                    model.action_model.prepare_for_compile()
                action_dynamic = _resolve_compile_dynamic(
                    trainer_cfg,
                    "compile_action_model_dynamic",
                    False,
                )
                _install_compiled_forward(
                    model.action_model,
                    "action_model",
                    compile_mode=compile_mode,
                    compile_backend=compile_backend,
                    dynamic_attempts=[action_dynamic] if not action_dynamic else [True, False],
                    allow_eager_fallback=allow_compile_eager_fallback,
                )
            except Exception as exc:
                if not allow_compile_eager_fallback:
                    raise
                logger.warning(f"torch.compile failed for action_model, continuing without it: {exc}")

        if trainer_cfg.get("compile_vj_predictor", False):
            try:
                if hasattr(model.vj_predictor, "prepare_for_compile"):
                    model.vj_predictor.prepare_for_compile()
                vj_dynamic = _resolve_compile_dynamic(
                    trainer_cfg,
                    "compile_vj_predictor_dynamic",
                    compile_dynamic,
                )
                _install_compiled_forward(
                    model.vj_predictor,
                    "vj_predictor",
                    compile_mode=compile_mode,
                    compile_backend=compile_backend,
                    dynamic_attempts=[vj_dynamic] if not vj_dynamic else [True, False],
                    allow_eager_fallback=allow_compile_eager_fallback,
                )
            except Exception as exc:
                if not allow_compile_eager_fallback:
                    raise
                logger.warning(f"torch.compile failed for vj_predictor, continuing without it: {exc}")

        if trainer_cfg.get("compile_vj_encoder", False):
            try:
                if hasattr(model, "prepare_vj_encoder_for_compile"):
                    model.prepare_vj_encoder_for_compile()
                vj_encoder_dynamic = _resolve_compile_dynamic(
                    trainer_cfg,
                    "compile_vj_encoder_dynamic",
                    compile_dynamic,
                )
                vj_encoder_attempts = [vj_encoder_dynamic] if not vj_encoder_dynamic else [True, False]
                _install_compiled_forward(
                    model.vj_encoder,
                    "vj_encoder",
                    compile_mode=compile_mode,
                    compile_backend=compile_backend,
                    dynamic_attempts=vj_encoder_attempts,
                    allow_eager_fallback=allow_compile_eager_fallback,
                )
                if hasattr(model.vj_encoder, "get_vision_features"):
                    _install_compiled_callable_attr(
                        model.vj_encoder,
                        "get_vision_features",
                        "vj_encoder.get_vision_features",
                        compile_mode=compile_mode,
                        compile_backend=compile_backend,
                        dynamic_attempts=vj_encoder_attempts,
                        allow_eager_fallback=allow_compile_eager_fallback,
                    )
            except Exception as exc:
                if not allow_compile_eager_fallback:
                    raise
                logger.warning(f"torch.compile failed for vj_encoder, continuing without it: {exc}")

        if trainer_cfg.get("compile_full_model", False):
            try:
                full_model_dynamic = _resolve_compile_dynamic(
                    trainer_cfg,
                    "compile_full_model_dynamic",
                    False,
                )
                _install_compiled_forward(
                    model,
                    f"{type(model).__name__}.forward",
                    compile_mode=compile_mode,
                    compile_backend=compile_backend,
                    dynamic_attempts=[full_model_dynamic] if not full_model_dynamic else [True, False],
                    allow_eager_fallback=allow_compile_eager_fallback,
                )
            except Exception as exc:
                if not allow_compile_eager_fallback:
                    raise
                logger.warning(f"torch.compile failed for full model forward, continuing without it: {exc}")

    return model


def prepare_data(cfg, accelerator, output_dir, model=None) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    from starVLA.dataloader import build_dataloader

    # VLA data loader
    dataset_py = cfg.datasets.vla_data.dataset_py
    if "data_mix" in cfg.datasets.vla_data:
        logger.info(
            f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}` "
            f"via `{dataset_py}`"
        )
    else:
        data_location = cfg.datasets.vla_data.get(
            "data_root_dir",
            cfg.datasets.vla_data.get("cache_dir", "<unset>"),
        )
        logger.info(
            f"Creating VLA Dataset from `{data_location}` "
            f"via `{dataset_py}`"
        )
    vla_train_dataloader = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.vla_data.dataset_py,
        model=model,
    )

    accelerator.dataloader_config.dispatch_batches = False

    return vla_train_dataloader


def prepare_heldout_eval_data(cfg, accelerator, output_dir, model=None):
    """Build separate manifest-selected checkpoint-evaluation loader views.

    Presence of ``episode_split_manifest`` enables the heldout path.  The same
    config is passed with ``mode="eval"`` so the dataset layer selects complete
    holdout episodes while loading only the manifest-bound train statistics.
    The optional focused view shares only the already-independent eval dataset
    readers and adds one effective global forward batch per evaluation.
    """

    del output_dir  # The dataloader resolves the run output path from ``cfg``.
    vla_dataset_cfg = cfg.datasets.vla_data
    if not vla_dataset_cfg.get("episode_split_manifest", None):
        return None, None
    legacy_underfilled_eval = bool(
        cfg.trainer.get("eval_only_legacy_underfilled_holdout", False)
    )
    if legacy_underfilled_eval:
        if not bool(cfg.trainer.get("eval_only", False)):
            raise ValueError(
                "Legacy underfilled holdout mode is forbidden during training."
            )
        # 95 historical holdouts cannot be evenly divided across eight ranks.
        # Padding would duplicate one episode and invalidate the audit.
        accelerator.dataloader_config.even_batches = False
    from starVLA.dataloader import build_dataloader

    eval_loaders = build_dataloader(
        cfg=cfg,
        dataset_py=vla_dataset_cfg.dataset_py,
        model=model,
        mode="eval",
    )
    accelerator.dataloader_config.dispatch_batches = False
    if isinstance(eval_loaders, tuple):
        if len(eval_loaders) != 2:
            raise RuntimeError(
                "Heldout eval loader builder returned an invalid loader bundle."
            )
        return eval_loaders
    return eval_loaders, None


def _find_sampler(obj, visited: set[int] | None = None):
    if obj is None:
        return None
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    sampler = getattr(obj, "sampler", None)
    if sampler is not None:
        return sampler

    batch_sampler = getattr(obj, "batch_sampler", None)
    if batch_sampler is not None:
        nested_sampler = _find_sampler(batch_sampler, visited)
        if nested_sampler is not None:
            return nested_sampler

    for attr_name in ("dataloader", "base_dataloader"):
        nested_obj = getattr(obj, attr_name, None)
        nested_sampler = _find_sampler(nested_obj, visited)
        if nested_sampler is not None:
            return nested_sampler

    return None


def _collect_dataset_object_ids(obj, visited: set[int] | None = None) -> set[int]:
    """Collect dataset object identities through common loader/wrapper shapes."""

    if obj is None:
        return set()
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return set()
    visited.add(obj_id)
    result: set[int] = set()

    if isinstance(obj, Dataset):
        result.add(obj_id)
    for attr_name in ("dataset", "dataloader", "base_dataloader"):
        nested = getattr(obj, attr_name, None)
        if nested is not None:
            result.update(_collect_dataset_object_ids(nested, visited))
    children = getattr(obj, "datasets", None)
    if children is not None:
        for child in children:
            result.update(_collect_dataset_object_ids(child, visited))
    return result


def _reset_loader_generators(obj, seed: int, visited: set[int] | None = None) -> int:
    """Reset only generators owned by the independent deterministic eval loader."""

    if obj is None:
        return 0
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return 0
    visited.add(obj_id)
    reset_count = 0
    generator = getattr(obj, "generator", None)
    if isinstance(generator, torch.Generator):
        generator.manual_seed(int(seed))
        reset_count += 1
    for attr_name in (
        "dataset",
        "sampler",
        "batch_sampler",
        "dataloader",
        "base_dataloader",
    ):
        nested = getattr(obj, attr_name, None)
        if nested is not None:
            reset_count += _reset_loader_generators(nested, seed, visited)
    return reset_count


def _close_heldout_eval_caches(obj, visited: set[int] | None = None) -> None:
    if obj is None:
        return
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)
    close_eval_caches = getattr(obj, "close_eval_caches", None)
    if callable(close_eval_caches):
        close_eval_caches()
    for attr_name in ("dataset", "dataloader", "base_dataloader"):
        nested = getattr(obj, attr_name, None)
        if nested is not None:
            _close_heldout_eval_caches(nested, visited)


@contextmanager
def _isolated_evaluation_rng(seed: int, device: torch.device):
    """Run stochastic diffusion inference without advancing training RNGs."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        ]
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed(int(seed))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _summarize_batch_structure(batch) -> str:
    if isinstance(batch, torch.Tensor):
        return f"Tensor(shape={tuple(batch.shape)}, dtype={batch.dtype}, device={batch.device})"
    if isinstance(batch, Mapping):
        parts = []
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                parts.append(f"{key}:Tensor{tuple(value.shape)}")
            elif isinstance(value, np.ndarray):
                parts.append(f"{key}:ndarray{value.shape}")
            elif isinstance(value, list):
                parts.append(f"{key}:list(len={len(value)})")
            else:
                parts.append(f"{key}:{type(value).__name__}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        return f"{type(batch).__name__}(len={len(batch)})"
    return type(batch).__name__


def _shutdown_dataloader_iterator(iterator) -> None:
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as exc:
            logger.warning(f"Failed to shut down dataloader iterator `{type(iterator).__name__}`: {exc}")


def _shutdown_dataloader_workers(dataloader) -> None:
    if dataloader is None:
        return
    _shutdown_dataloader_iterator(getattr(dataloader, "_iterator", None))
    try:
        dataloader._iterator = None
    except (AttributeError, TypeError):
        pass


class _RankVideoBatchPrefetcher:
    def __init__(self, trainer: "VLATrainer", queue_size: int):
        self.trainer = trainer
        self.queue_size = max(int(queue_size), 1)
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name="rank-video-prefetch",
            daemon=True,
        )
        self._cuda_stream = None

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        try:
            device = self.trainer.accelerator.device
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.set_device(device)
                self._cuda_stream = torch.cuda.Stream(device=device)

            while not self._stop_event.is_set():
                raw_fetch_start = time.perf_counter()
                raw_batch = self.trainer._get_next_raw_batch()
                raw_fetch_time = time.perf_counter() - raw_fetch_start
                prepare_start = time.perf_counter()
                prepared_batch, ready_event = self.trainer._prepare_prefetched_batch(
                    raw_batch, stream=self._cuda_stream
                )
                prepare_time = time.perf_counter() - prepare_start
                if isinstance(prepared_batch, list) and prepared_batch and isinstance(prepared_batch[0], dict):
                    timing_payload = dict(prepared_batch[0].get("_prefetch_timing", {}))
                    timing_payload["raw_batch_fetch_time"] = raw_fetch_time
                    timing_payload["prefetch_prepare_time"] = prepare_time
                    timing_payload["prefetch_total_time"] = raw_fetch_time + prepare_time
                    prepared_batch[0]["_prefetch_timing"] = timing_payload
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(("batch", prepared_batch, ready_event), timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            try:
                self._queue.put(("error", exc, traceback.format_exc()), timeout=0.5)
            except queue.Full:
                pass

    def next_batch(self):
        item = self._queue.get()
        kind = item[0]
        if kind == "error":
            _, exc, tb = item
            raise RuntimeError(f"Rank video prefetch failed:\n{tb}") from exc

        _, batch, ready_event = item
        if ready_event is not None and torch.cuda.is_available():
            torch.cuda.current_stream(device=self.trainer.accelerator.device).wait_event(ready_event)
        return batch


def _drop_file_cache_best_effort(path: str) -> None:
    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed_flag = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed_flag is None:
        return

    try:
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                try:
                    fd = os.open(file_path, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    file_size = os.fstat(fd).st_size
                    posix_fadvise(fd, 0, file_size, dontneed_flag)
                except OSError:
                    pass
                finally:
                    os.close(fd)
    except Exception as exc:
        logger.warning(f"Unable to release checkpoint file cache for `{path}`: {exc}")


def _make_artifact_tree_host_readable(path: str) -> None:
    """Allow a host-side uploader to read artifacts written by root in Docker."""
    directory_read_mode = 0o055
    file_read_mode = 0o044

    for root, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            directory_path = os.path.join(root, dirname)
            current_mode = os.stat(directory_path).st_mode
            os.chmod(directory_path, current_mode | directory_read_mode)
        for filename in filenames:
            file_path = os.path.join(root, filename)
            current_mode = os.stat(file_path).st_mode
            os.chmod(file_path, current_mode | file_read_mode)

    current_mode = os.stat(path).st_mode
    os.chmod(path, current_mode | directory_read_mode)


def _trim_process_memory_best_effort() -> None:
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


def _raw_dataloader_batches_per_rank(vla_train_dataloader, num_processes: int) -> int:
    raw_batches = len(vla_train_dataloader)
    sampler = _find_sampler(vla_train_dataloader)
    if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
        return raw_batches

    if num_processes <= 1:
        return raw_batches

    drop_last = bool(getattr(vla_train_dataloader, "drop_last", False))
    # Accelerate shards an unprepared dataloader across ranks after this point.
    # Match DataLoaderShard semantics: floor when dropping incomplete batches,
    # otherwise ceil so every rank sees the same number of steps.
    if drop_last:
        return max(1, raw_batches // num_processes)
    return max(1, math.ceil(raw_batches / num_processes))


def _resolved_training_schedule_payload(
    cfg,
    *,
    num_processes: int,
    configured_schedule: Mapping,
) -> dict:
    trainer_cfg = cfg.trainer
    data_cfg = cfg.datasets.vla_data
    per_device_batch_size = int(data_cfg.per_device_batch_size)
    grad_accum = int(trainer_cfg.get("gradient_accumulation_steps", 1))
    max_train_steps = int(trainer_cfg.max_train_steps)
    framework_cfg = cfg.get("framework", {})
    depth_teacher_cfg = framework_cfg.get("depth_teacher_aux", {})
    detach_floor = max(int(depth_teacher_cfg.get("detach_vlm_steps", 0) or 0), 0)
    detach_fraction = float(
        depth_teacher_cfg.get("detach_vlm_fraction", 0.0) or 0.0
    )
    fraction_detach = int(math.ceil(detach_fraction * max_train_steps))
    rtc_cfg = framework_cfg.get("action_model", {}).get("rtc_training", {})
    loss_scale_cfg = trainer_cfg.get("loss_scale", {})
    return _plain_config({
        "schema_version": 1,
        "configured": dict(configured_schedule),
        "resolved": {
            "epochs": int(trainer_cfg.epochs),
            "micro_batches_per_epoch": int(trainer_cfg.micro_batches_per_epoch),
            "steps_per_epoch": int(trainer_cfg.steps_per_epoch),
            "max_train_steps": max_train_steps,
            "num_warmup_steps": int(trainer_cfg.num_warmup_steps),
            "save_interval": int(trainer_cfg.save_interval),
            "eval_interval": int(trainer_cfg.eval_interval),
            "gradient_accumulation_steps": grad_accum,
            "per_device_batch_size": per_device_batch_size,
            "num_processes": int(num_processes),
            "effective_global_batch_size": (
                per_device_batch_size * int(num_processes) * grad_accum
            ),
            "step_scheduler_with_optimizer": bool(
                trainer_cfg.get("step_scheduler_with_optimizer", False)
            ),
            "wm_warmup_steps": int(loss_scale_cfg.get("wm_warmup_steps", 0)),
            "depth_teacher_detach_steps": max(detach_floor, fraction_detach),
            "depth_teacher_detach_steps_floor": detach_floor,
            "depth_teacher_detach_fraction": detach_fraction,
            "rtc_enabled": bool(rtc_cfg.get("enabled", False)),
            "rtc_warmup_steps": int(
                rtc_cfg.get("warmup_steps", rtc_cfg.get("start_step", 0)) or 0
            ),
            "rtc_ramp_steps": int(rtc_cfg.get("ramp_steps", 0) or 0),
        },
    })


def persist_resolved_training_schedule(cfg, schedule: Mapping) -> dict:
    """Write once, then fail closed if a resume resolves a different schedule."""

    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    source_config_path = output_dir / "config.yaml"
    if not source_config_path.is_file():
        raise RuntimeError(
            "Cannot bind the resolved schedule to a missing immutable source "
            f"config: {source_config_path}"
        )
    payload = _plain_config(schedule)
    payload["source_config"] = {
        "path": "config.yaml",
        "sha256": hashlib.sha256(source_config_path.read_bytes()).hexdigest(),
    }
    schedule_path = output_dir / "resolved_training_schedule.json"
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if schedule_path.is_file():
        try:
            existing_payload = json.loads(
                schedule_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Existing resolved training schedule is unreadable: {schedule_path}"
            ) from exc
        if existing_payload != payload:
            differences = _config_difference_summaries(
                existing_payload,
                payload,
            )
            detail = "; ".join(differences) if differences else "unknown difference"
            raise RuntimeError(
                "Resolved training schedule drift detected; refusing to "
                f"overwrite {schedule_path}. Differences: {detail}"
            )
    else:
        _atomic_write_bytes(schedule_path, serialized)
    return {
        "path": "resolved_training_schedule.json",
        "sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
    }


def _validated_resolved_schedule_evidence(
    schedule_path: Path,
    *,
    config_sha256: str,
) -> dict[str, str]:
    try:
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Resolved training schedule provenance is unreadable: {schedule_path}"
        ) from exc
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    resolved = payload.get("resolved") if isinstance(payload, dict) else None
    source_config = (
        payload.get("source_config") if isinstance(payload, dict) else None
    )
    required_resolved_fields = {
        "effective_global_batch_size",
        "eval_interval",
        "max_train_steps",
        "num_warmup_steps",
        "save_interval",
    }
    missing_fields = (
        sorted(required_resolved_fields - set(resolved))
        if isinstance(resolved, dict)
        else sorted(required_resolved_fields)
    )
    if (
        schema_version != 1
        or not isinstance(resolved, dict)
        or not isinstance(source_config, dict)
        or source_config.get("path") != "config.yaml"
        or source_config.get("sha256") != config_sha256
        or missing_fields
    ):
        raise RuntimeError(
            "Resolved training schedule provenance is invalid or is not bound "
            f"to the immutable source config: {schedule_path}; "
            f"missing_resolved_fields={missing_fields}"
        )
    return {
        "path": "resolved_training_schedule.json",
        "sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
    }


def resolve_training_schedule(
    cfg,
    vla_train_dataloader,
    num_processes: int = 1,
) -> dict:
    """Resolve epoch-based training schedule once the dataloader length is known."""
    trainer_cfg = cfg.trainer
    configured_schedule = {
        "epochs": trainer_cfg.get("epochs", None),
        "max_train_steps": trainer_cfg.get("max_train_steps", None),
        "num_warmup_steps": trainer_cfg.get("num_warmup_steps", None),
        "save_interval": trainer_cfg.get("save_interval", None),
        "eval_interval": trainer_cfg.get("eval_interval", None),
    }
    micro_batches_per_epoch = _raw_dataloader_batches_per_rank(vla_train_dataloader, num_processes)
    grad_accum_steps = max(int(trainer_cfg.get("gradient_accumulation_steps", 1)), 1)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / grad_accum_steps)
    trainer_cfg.micro_batches_per_epoch = micro_batches_per_epoch
    trainer_cfg.steps_per_epoch = steps_per_epoch

    max_train_steps_cfg = trainer_cfg.get("max_train_steps", None)
    auto_max_train_steps = (
        max_train_steps_cfg is None
        or (isinstance(max_train_steps_cfg, str) and max_train_steps_cfg.lower() == "auto")
        or (isinstance(max_train_steps_cfg, (int, float)) and int(max_train_steps_cfg) <= 0)
    )
    if auto_max_train_steps:
        trainer_cfg.max_train_steps = int(trainer_cfg.epochs) * steps_per_epoch

    num_warmup_steps_cfg = trainer_cfg.get("num_warmup_steps", None)
    auto_num_warmup_steps = (
        num_warmup_steps_cfg is None
        or (isinstance(num_warmup_steps_cfg, str) and num_warmup_steps_cfg.lower() == "auto")
    )
    if auto_num_warmup_steps:
        warmup_ratio = float(trainer_cfg.get("warmup_ratio", 0.0))
        warmup_steps = int(round(float(trainer_cfg.max_train_steps) * warmup_ratio))
        if warmup_ratio > 0.0 and trainer_cfg.max_train_steps > 0:
            warmup_steps = max(1, warmup_steps)
        trainer_cfg.num_warmup_steps = warmup_steps

    def _resolve_periodic_interval(key: str) -> None:
        interval_cfg = trainer_cfg.get(key, None)
        auto_interval = (
            interval_cfg is None
            or (isinstance(interval_cfg, str) and interval_cfg.lower() in {"auto", "epoch", "auto_epoch"})
            or (isinstance(interval_cfg, (int, float)) and int(interval_cfg) <= 0)
        )
        if auto_interval:
            trainer_cfg[key] = steps_per_epoch

    _resolve_periodic_interval("save_interval")
    _resolve_periodic_interval("eval_interval")

    logger.info(
        "Resolved training schedule: "
        f"epochs={trainer_cfg.epochs}, "
        f"micro_batches_per_epoch={micro_batches_per_epoch}, "
        f"gradient_accumulation_steps={grad_accum_steps}, "
        f"steps_per_epoch={steps_per_epoch}, "
        f"max_train_steps={trainer_cfg.max_train_steps}, "
        f"num_warmup_steps={trainer_cfg.num_warmup_steps}, "
        f"save_interval={trainer_cfg.save_interval}, "
        f"eval_interval={trainer_cfg.eval_interval}"
    )
    return _resolved_training_schedule_payload(
        cfg,
        num_processes=num_processes,
        configured_schedule=configured_schedule,
    )


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer_name = cfg.trainer.optimizer.get("name", "AdamW")
    optimizer_weight_decay = cfg.trainer.optimizer.get(
        "weight_decay",
        cfg.trainer.get("weight_decay", 0.0),
    )
    optimizer_kwargs = dict(
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=optimizer_weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    if optimizer_name == "AdamW8bit":
        from bitsandbytes.optim import AdamW8bit

        optimizer_cls = AdamW8bit
        if cfg.trainer.optimizer.get("fused", False):
            logger.warning("trainer.optimizer.fused=true is ignored for AdamW8bit; fused kernels only apply to torch.optim.AdamW")
    else:
        optimizer_cls = torch.optim.AdamW
    if optimizer_cls is torch.optim.AdamW and torch.cuda.is_available() and cfg.trainer.optimizer.get("fused", True):
        optimizer_kwargs["fused"] = True
    optimizer = optimizer_cls(
        param_groups,
        **optimizer_kwargs,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


def _heldout_loader_report(loader, *, label: str) -> dict | None:
    if loader is None:
        return None
    dataset = getattr(loader, "dataset", loader)
    report_fn = getattr(dataset, "sampling_report", None)
    if not callable(report_fn):
        raise ValueError(
            f"{label} loader dataset must expose its immutable sampling_report()."
        )
    report = report_fn()
    if not isinstance(report, Mapping):
        raise ValueError(f"{label} sampling_report() must return a mapping.")
    return dict(report)


def _heldout_report_subtask_counts(
    report: Mapping | None,
) -> tuple[dict[int, int], dict[int, int]]:
    if report is None:
        return {}, {}
    counts = {
        int(key): int(value)
        for key, value in report.get("subtask_observation_counts", {}).items()
    }
    evaluable = {
        int(key): int(value)
        for key, value in report.get(
            "subtask_evaluable_observation_counts", {}
        ).items()
    }
    return counts, evaluable


def _heldout_report_action_subtask_counts(
    report: Mapping | None,
    *,
    horizon: int,
    field: str,
) -> dict[int, int]:
    if report is None:
        return {}
    by_horizon = report.get(field, {})
    if not isinstance(by_horizon, Mapping):
        raise ValueError(f"Heldout sampling report field {field!r} must be a mapping.")
    raw_counts = by_horizon.get(str(int(horizon)), {})
    if not isinstance(raw_counts, Mapping):
        raise ValueError(
            f"Heldout sampling report {field!r} h{int(horizon)} must be a mapping."
        )
    return {int(key): int(value) for key, value in raw_counts.items()}


def _validate_heldout_report_coverage(
    report: Mapping,
    *,
    expected_observations: int,
    required_subtasks: Sequence[int],
    minimum_per_subtask: int,
    label: str,
    allow_legacy_underfilled: bool = False,
) -> None:
    observations = int(report["observation_count"])
    split_provenance = report.get("episode_split_provenance")
    if not isinstance(split_provenance, list) or not split_provenance:
        raise ValueError(f"{label} lacks episode split/statistics provenance.")
    sha_fields = (
        "manifest_sha256",
        "selected_episode_set_sha256",
        "train_episode_set_sha256",
        "holdout_episode_set_sha256",
        "full_catalog_sha256",
        "train_statistics_sha256",
    )
    for entry in split_provenance:
        if not isinstance(entry, Mapping) or not str(
            entry.get("dataset_name", "")
        ):
            raise ValueError(f"{label} has malformed split provenance.")
        for field in sha_fields:
            value = entry.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"{label} split provenance has invalid {field}."
                )
        if str(entry.get("role", "")).lower() not in {
            "eval",
            "evaluation",
            "val",
            "validation",
            "test",
            "holdout",
        }:
            raise ValueError(f"{label} split provenance is not holdout-role data.")
    if sum(
        int(entry.get("selected_episode_count", -1))
        for entry in split_provenance
    ) != int(expected_observations):
        raise ValueError(
            f"{label} split provenance does not bind the original effective "
            "global batch cardinality."
        )
    if allow_legacy_underfilled:
        legacy = report.get("legacy_underfilled_holdout")
        if (
            report.get("production_valid") is not False
            or report.get("checkpoint_selection_eligible") is not False
            or not isinstance(legacy, Mapping)
            or legacy.get("enabled") is not True
            or legacy.get("no_replacement_no_training_leak") is not True
            or legacy.get("replacement_episode_ids") != []
        ):
            raise ValueError(
                f"{label} legacy audit lacks explicit production-invalid, "
                "no-replacement evidence."
            )
        excluded = legacy.get("excluded_zero_valid_episodes", [])
        if not isinstance(excluded, list) or not excluded:
            raise ValueError(f"{label} legacy audit has no explicit exclusions.")
        if (
            int(legacy.get("original_manifest_observation_count", -1))
            != int(expected_observations)
            or int(legacy.get("evaluated_observation_count", -1))
            != observations
            or observations != int(expected_observations) - len(excluded)
        ):
            raise ValueError(
                f"{label} legacy audit cardinality is not exactly original minus "
                "explicit exclusions."
            )
    elif observations != int(expected_observations):
        raise ValueError(
            f"{label} sampling report must contain exactly one effective global "
            f"training batch: report={observations}, expected={expected_observations}."
        )
    subtask_counts, evaluable_counts = _heldout_report_subtask_counts(report)
    if sum(subtask_counts.values()) != observations:
        raise ValueError(
            f"Every {label} observation must carry one integer subtask label: "
            f"labeled={sum(subtask_counts.values())}, observations={observations}."
        )
    expected_evaluable = int(report["action_evaluable_observation_count"])
    zero_valid_episodes = report.get("zero_valid_action_episodes", [])
    if expected_evaluable != observations or zero_valid_episodes:
        raise ValueError(
            f"{label} must keep all {observations} episode windows action-evaluable; "
            f"evaluable={expected_evaluable}, zero-valid={zero_valid_episodes}."
        )
    if sum(evaluable_counts.values()) != expected_evaluable:
        raise ValueError(
            f"{label} per-subtask evaluable coverage does not match its global "
            "evaluable-observation count."
        )
    if int(minimum_per_subtask) <= 0:
        raise ValueError(f"{label} minimum per-subtask coverage must be positive.")
    insufficient = {
        int(subtask): evaluable_counts.get(int(subtask), 0)
        for subtask in required_subtasks
        if evaluable_counts.get(int(subtask), 0) < int(minimum_per_subtask)
    }
    if insufficient:
        raise ValueError(
            f"{label} has insufficient action-evaluable coverage for required "
            f"subtasks (minimum={minimum_per_subtask}): {insufficient}."
        )
    if required_subtasks:
        action_timestep_counts = _heldout_report_action_subtask_counts(
            report,
            horizon=10,
            field="subtask_action_timestep_counts_by_horizon",
        )
        valid_element_counts = _heldout_report_action_subtask_counts(
            report,
            horizon=10,
            field="subtask_valid_action_element_counts_by_horizon",
        )
        if sum(action_timestep_counts.values()) != observations * 10:
            raise ValueError(
                f"{label} must carry one per-action subtask label at every H10 "
                "target timestep."
            )
        if not valid_element_counts or sum(valid_element_counts.values()) <= 0:
            raise ValueError(
                f"{label} has no H10 valid-action element coverage by target-step "
                "subtask."
            )


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_eval_dataloader=None,
        vla_focused_eval_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_eval_dataloader = vla_eval_dataloader
        self.vla_focused_eval_dataloader = vla_focused_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        (
            self.action_loss_scale,
            self.wm_loss_scale,
            self.wm_loss_scale_initial,
            self.wm_loss_warmup_steps,
            self.depth_teacher_loss_scale,
        ) = self._resolve_loss_scales()
        self.best_metric_name = str(self.config.trainer.get("best_metric_name", "mae_score"))
        self.best_metric_mode = str(self.config.trainer.get("best_metric_mode", "min")).lower()
        self.best_metric_value = None
        self.loaded_checkpoint_path: str | None = None
        self.legacy_underfilled_eval = bool(
            self.config.trainer.get(
                "eval_only_legacy_underfilled_holdout", False
            )
        )
        if self.legacy_underfilled_eval and not bool(
            self.config.trainer.get("eval_only", False)
        ):
            raise ValueError(
                "Legacy underfilled holdout mode cannot initialize a trainer."
            )
        self._warned_missing_best_metric = False
        self._warned_training_stream_eval_disabled = False

        if self.vla_focused_eval_dataloader is not None and self.vla_eval_dataloader is None:
            raise ValueError("Focused heldout eval requires the unbiased heldout loader.")
        if bool(
            self.config.trainer.get("heldout_focused_eval_enabled", False)
        ) != (self.vla_focused_eval_dataloader is not None):
            raise ValueError(
                "trainer.heldout_focused_eval_enabled must exactly match focused "
                "loader construction; refusing to silently omit the coverage gate."
            )
        training_dataset_ids = _collect_dataset_object_ids(
            self.vla_train_dataloader
        )
        for label, loader in (
            ("Heldout evaluation", self.vla_eval_dataloader),
            ("Focused heldout evaluation", self.vla_focused_eval_dataloader),
        ):
            if loader is not None and (
                training_dataset_ids & _collect_dataset_object_ids(loader)
            ):
                raise ValueError(
                    f"{label} must use dataset objects independent of training."
                )

        from starVLA.dataloader.heldout_eval import sampling_seed_from_dataset

        self.heldout_eval_seed = sampling_seed_from_dataset(
            getattr(self.vla_eval_dataloader, "dataset", self.vla_eval_dataloader),
            default=int(self.config.get("seed", 0)),
        )
        self.heldout_focused_eval_seed = sampling_seed_from_dataset(
            getattr(
                self.vla_focused_eval_dataloader,
                "dataset",
                self.vla_focused_eval_dataloader,
            ),
            default=self.heldout_eval_seed,
        )
        self.heldout_eval_sampling_report = _heldout_loader_report(
            self.vla_eval_dataloader,
            label="Heldout eval",
        )
        self.heldout_focused_eval_sampling_report = _heldout_loader_report(
            self.vla_focused_eval_dataloader,
            label="Focused heldout eval",
        )
        self.heldout_eval_subtask_counts, (
            self.heldout_eval_evaluable_subtask_counts
        ) = _heldout_report_subtask_counts(self.heldout_eval_sampling_report)
        self.heldout_focused_eval_subtask_counts, (
            self.heldout_focused_eval_evaluable_subtask_counts
        ) = _heldout_report_subtask_counts(
            self.heldout_focused_eval_sampling_report
        )
        self.heldout_eval_expected_valid_elements = (
            None
            if self.heldout_eval_sampling_report is None
            else int(
                self.heldout_eval_sampling_report["valid_action_element_count"]
            )
        )
        self.heldout_eval_expected_valid_observations = (
            None
            if self.heldout_eval_sampling_report is None
            else int(
                self.heldout_eval_sampling_report[
                    "action_evaluable_observation_count"
                ]
            )
        )
        self.heldout_focused_eval_expected_valid_elements = (
            None
            if self.heldout_focused_eval_sampling_report is None
            else int(
                self.heldout_focused_eval_sampling_report[
                    "valid_action_element_count"
                ]
            )
        )
        self.heldout_focused_eval_expected_valid_observations = (
            None
            if self.heldout_focused_eval_sampling_report is None
            else int(
                self.heldout_focused_eval_sampling_report[
                    "action_evaluable_observation_count"
                ]
            )
        )

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        if self.heldout_eval_sampling_report is not None:
            _validate_heldout_report_coverage(
                self.heldout_eval_sampling_report,
                expected_observations=self.total_batch_size,
                required_subtasks=tuple(
                    int(value)
                    for value in self.config.trainer.get(
                        "heldout_eval_required_subtasks", ()
                    )
                ),
                minimum_per_subtask=int(
                    self.config.trainer.get(
                        "heldout_eval_min_evaluable_observations_per_subtask", 1
                    )
                ),
                label="Heldout eval",
                allow_legacy_underfilled=self.legacy_underfilled_eval,
            )
        if self.heldout_focused_eval_sampling_report is not None:
            if (
                self.heldout_focused_eval_sampling_report.get(
                    "episode_split_provenance"
                )
                != self.heldout_eval_sampling_report.get(
                    "episode_split_provenance"
                )
            ):
                raise ValueError(
                    "Focused and unbiased heldout views must bind the identical "
                    "manifest, episode split, catalog, and train statistics."
                )
            _validate_heldout_report_coverage(
                self.heldout_focused_eval_sampling_report,
                expected_observations=self.total_batch_size,
                required_subtasks=tuple(
                    int(value)
                    for value in self.config.trainer.get(
                        "heldout_focused_eval_required_subtasks",
                        (2, 3, 4, 5, 6, 7),
                    )
                ),
                minimum_per_subtask=int(
                    self.config.trainer.get(
                        "heldout_focused_eval_min_evaluable_observations_per_subtask",
                        1,
                    )
                ),
                label="Focused heldout eval",
                allow_legacy_underfilled=self.legacy_underfilled_eval,
            )
            focused_transition_horizon = int(
                self.config.trainer.get(
                    "heldout_focused_eval_transition_coverage_horizon", 10
                )
            )
            if focused_transition_horizon != 10:
                raise ValueError(
                    "The deterministic focused selector is explicitly H10; "
                    "trainer.heldout_focused_eval_transition_coverage_horizon "
                    "must equal 10."
                )
            minimum_open_to_close = int(
                self.config.trainer.get(
                    "heldout_focused_eval_min_open_to_close_transitions", 1
                )
            )
            minimum_close_to_open = int(
                self.config.trainer.get(
                    "heldout_focused_eval_min_close_to_open_transitions", 1
                )
            )
            minimum_open_to_close_windows = int(
                self.config.trainer.get(
                    "heldout_focused_eval_min_open_to_close_windows", 1
                )
            )
            minimum_close_to_open_windows = int(
                self.config.trainer.get(
                    "heldout_focused_eval_min_close_to_open_windows", 1
                )
            )
            minimum_arm_movement_elements = int(
                self.config.trainer.get(
                    "heldout_focused_eval_min_arm_movement_elements_h10", 1
                )
            )
            minimum_arm_movement_hold_abs = float(
                self.config.trainer.get(
                    "heldout_focused_eval_min_arm_movement_hold_abs_h10",
                    1.0e-12,
                )
            )
            if any(
                value < 0
                for value in (
                    minimum_open_to_close,
                    minimum_close_to_open,
                    minimum_open_to_close_windows,
                    minimum_close_to_open_windows,
                    minimum_arm_movement_elements,
                )
            ) or (
                not math.isfinite(minimum_arm_movement_hold_abs)
                or minimum_arm_movement_hold_abs < 0
            ):
                raise ValueError(
                    "Focused heldout movement and transition minimums must be "
                    "finite and non-negative."
                )
            report_open_to_close = int(
                self.heldout_focused_eval_sampling_report.get(
                    "open_to_close_transition_count_h10", 0
                )
            )
            report_close_to_open = int(
                self.heldout_focused_eval_sampling_report.get(
                    "close_to_open_transition_count_h10", 0
                )
            )
            report_open_to_close_windows = int(
                self.heldout_focused_eval_sampling_report.get(
                    "open_to_close_transition_window_count_h10", 0
                )
            )
            report_close_to_open_windows = int(
                self.heldout_focused_eval_sampling_report.get(
                    "close_to_open_transition_window_count_h10", 0
                )
            )
            report_arm_movement_elements = int(
                self.heldout_focused_eval_sampling_report.get(
                    "arm_movement_element_count_h10", 0
                )
            )
            report_arm_movement_hold_abs = float(
                self.heldout_focused_eval_sampling_report.get(
                    "arm_movement_hold_abs_sum_h10", 0.0
                )
            )
            if (
                report_open_to_close < minimum_open_to_close
                or report_close_to_open < minimum_close_to_open
                or report_open_to_close_windows < minimum_open_to_close_windows
                or report_close_to_open_windows < minimum_close_to_open_windows
                or report_arm_movement_elements < minimum_arm_movement_elements
                or report_arm_movement_hold_abs < minimum_arm_movement_hold_abs
            ):
                raise ValueError(
                    "Focused heldout selection fails configured H10 movement/"
                    "transition coverage: arm_movement_elements="
                    f"{report_arm_movement_elements}/"
                    f"{minimum_arm_movement_elements}, arm_movement_hold_abs="
                    f"{report_arm_movement_hold_abs}/"
                    f"{minimum_arm_movement_hold_abs}, open_to_close_events="
                    f"{report_open_to_close}/"
                    f"{minimum_open_to_close}, close_to_open="
                    f"{report_close_to_open}/{minimum_close_to_open}, "
                    f"open_to_close_windows={report_open_to_close_windows}/"
                    f"{minimum_open_to_close_windows}, close_to_open_windows="
                    f"{report_close_to_open_windows}/"
                    f"{minimum_close_to_open_windows}."
                )
        self.train_start_time = time.perf_counter()
        self.progress_eta_window = max(int(self.config.trainer.get("progress_eta_window", 50)), 1)
        self.progress_eta_warmup_steps = max(int(self.config.trainer.get("progress_eta_warmup_steps", 3)), 0)
        self._recent_wall_step_times = deque(maxlen=self.progress_eta_window)
        self._rank_video_prefetcher: Optional[_RankVideoBatchPrefetcher] = None
        self._data_runtime_shutdown = False
        self._prefetch_model = None
        self._last_prefetch_timing: Optional[dict] = None
        self._rank_timing_keys = (
            ("data_time", "data"),
            ("model_time", "model"),
            ("wall_step_time", "wall"),
            ("raw_batch_fetch_time", "fetch"),
            ("video_tensor_to_cuda_time", "to_cuda"),
            ("video_decode_time", "decode"),
            ("video_postprocess_time", "post"),
            ("qwen_tensor_build_time", "qwen"),
            ("qwen_forward_time", "qfwd"),
            ("depth_teacher_time", "depth"),
            ("vj_encode_time", "vj"),
            ("predictor_action_head_time", "head"),
            ("forward_time", "fwd"),
            ("backward_only_time", "back"),
            ("grad_clip_time", "clip"),
            ("optimizer_step_time", "opt"),
            ("backward_optimizer_time", "bwd"),
        )

    @staticmethod
    def _format_duration(seconds: Optional[float]) -> str:
        if seconds is None or not math.isfinite(seconds):
            return "n/a"
        total_seconds = max(int(round(seconds)), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _estimate_remaining_seconds(self) -> Optional[float]:
        if not self._recent_wall_step_times:
            return None
        remaining_steps = max(int(self.config.trainer.max_train_steps) - self.completed_steps, 0)
        if remaining_steps <= 0:
            return 0.0
        avg_wall_step_time = sum(self._recent_wall_step_times) / len(self._recent_wall_step_times)
        return avg_wall_step_time * remaining_steps

    def _runtime_timing_enabled(self) -> bool:
        return bool(self.config.datasets.vla_data.get("runtime_timing_logging", False))

    def _detailed_timing_enabled(self) -> bool:
        env_value = str(os.environ.get("STARVLA_DETAILED_TIMING", "0")).lower()
        return bool(self.config.trainer.get("detailed_timing_logging", False)) or env_value in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _detailed_timing_frequency(self) -> int:
        env_value = os.environ.get("STARVLA_DETAILED_TIMING_FREQUENCY")
        if env_value is not None:
            try:
                return max(1, int(env_value))
            except ValueError:
                pass
        default_frequency = int(self.config.trainer.get("logging_frequency", 1) or 1)
        return max(1, int(self.config.trainer.get("detailed_timing_frequency", default_frequency)))

    def _collect_rank_timing_stats(self, step_metrics: dict) -> dict[str, dict[str, float]] | None:
        if not self._detailed_timing_enabled():
            return None

        local_values = []
        for metric_key, _ in self._rank_timing_keys:
            metric_value = self._to_scalar(step_metrics.get(metric_key))
            local_values.append(float("nan") if metric_value is None else float(metric_value))

        timing_tensor = torch.tensor(local_values, device=self.accelerator.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(timing_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, timing_tensor)
            timing_matrix = torch.stack(gathered, dim=0).detach().cpu().numpy()
        else:
            timing_matrix = timing_tensor.detach().cpu().numpy()[None, :]

        stats: dict[str, dict[str, float]] = {}
        for column_idx, (_, label) in enumerate(self._rank_timing_keys):
            values = timing_matrix[:, column_idx]
            valid_mask = np.isfinite(values)
            if not valid_mask.any():
                continue
            valid_values = values[valid_mask]
            valid_ranks = np.nonzero(valid_mask)[0]
            max_pos = int(np.argmax(valid_values))
            min_pos = int(np.argmin(valid_values))
            stats[label] = {
                "min": float(valid_values[min_pos]),
                "mean": float(valid_values.mean()),
                "max": float(valid_values[max_pos]),
                "max_rank": int(valid_ranks[max_pos]),
            }
        return stats

    def _log_rank_timing_stats(self, timing_stats: dict[str, dict[str, float]]) -> None:
        if not timing_stats or not self.accelerator.is_main_process:
            return
        primary_labels = (
            "wall",
            "data",
            "model",
            "fwd",
            "back",
            "clip",
            "opt",
            "qfwd",
            "vj",
            "depth",
            "head",
        )
        parts = []
        slowest_label = None
        slowest_max = -1.0
        slowest_rank = 0
        for label in primary_labels:
            stats = timing_stats.get(label)
            if not stats:
                continue
            parts.append(
                f"{label}=max {stats['max']:.3f}s@r{stats['max_rank']} "
                f"mean {stats['mean']:.3f}s min {stats['min']:.3f}s"
            )
            if stats["max"] > slowest_max:
                slowest_label = label
                slowest_max = stats["max"]
                slowest_rank = int(stats["max_rank"])

        if parts:
            logger.info(
                "[rank_timing] "
                f"step={self.completed_steps} slowest={slowest_label}@r{slowest_rank}:{slowest_max:.3f}s | "
                + " | ".join(parts)
            )

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().float().item()
            return None
        if isinstance(value, np.generic):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _prepare_heldout_loaders_with_accelerator(self) -> None:
        for attribute, label in (
            ("vla_eval_dataloader", "unbiased"),
            ("vla_focused_eval_dataloader", "focused"),
        ):
            eval_loader = getattr(self, attribute)
            if eval_loader is None:
                continue
            eval_loader = self.accelerator.prepare(eval_loader)
            setattr(self, attribute, eval_loader)
            expected_eval_microbatches = max(
                1, int(self.config.trainer.get("gradient_accumulation_steps", 1))
            )
            actual_eval_microbatches = len(eval_loader)
            if actual_eval_microbatches != expected_eval_microbatches:
                raise RuntimeError(
                    f"Prepared {label} heldout eval loader must expose one local "
                    "microbatch per gradient-accumulation step: "
                    f"got {actual_eval_microbatches}, expected "
                    f"{expected_eval_microbatches}."
                )
            logger.info(
                f"Prepared {label} heldout eval loader with "
                f"{actual_eval_microbatches} local microbatch(es) per checkpoint eval"
            )

    def prepare_checkpoint_evaluation(self) -> None:
        """Prepare only model + heldout views and load a checkpoint model."""

        initialization_eval = bool(
            self.config.trainer.get(
                "eval_only_untrained_initialization", False
            )
        )
        if self.vla_eval_dataloader is None or self.vla_focused_eval_dataloader is None:
            raise ValueError(
                "trainer.eval_only requires both unbiased and focused heldout views."
            )
        if self.vla_train_dataloader is not None or self.optimizer is not None:
            raise ValueError(
                "Eval-only preparation refuses a training dataloader or optimizer."
            )
        checkpoint_path = (
            self.config.get("resume_from_checkpoint", None)
            or self.config.trainer.get("resume_from_checkpoint", None)
        )
        is_resume = bool(self.config.trainer.get("is_resume", False))
        if initialization_eval:
            if is_resume or checkpoint_path:
                raise ValueError(
                    "Untrained-initialization eval cannot resume from a checkpoint."
                )
        elif not is_resume or not checkpoint_path:
            raise ValueError(
                "Checkpoint eval-only requires trainer.is_resume=true and "
                "resume_from_checkpoint."
            )
        if bool(self.config.trainer.get("resume_load_optimizer_state", True)):
            raise ValueError(
                "trainer.eval_only requires trainer.resume_load_optimizer_state=false."
            )
        self.eval_source_training_config_evidence = (
            self._eval_source_training_config_evidence()
        )

        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)
        if initialization_eval:
            pretrained_checkpoint = self.config.trainer.get(
                "pretrained_checkpoint", None
            )
            if pretrained_checkpoint:
                reload_modules = self.config.trainer.get(
                    "reload_modules", None
                )
                self.model = self.load_pretrained_backbones(
                    self.model,
                    pretrained_checkpoint,
                    reload_modules=reload_modules,
                )
        freeze_modules = self.config.trainer.get("freeze_modules", None)
        self.model = self.freeze_backbones(
            self.model,
            freeze_modules=freeze_modules,
        )
        if hasattr(self.model, "refresh_runtime_caches"):
            self.model.refresh_runtime_caches()
        if hasattr(self.model, "validate_runtime_feature_state"):
            self.model.validate_runtime_feature_state()

        self.model = self.accelerator.prepare(self.model)
        self._prepare_heldout_loaders_with_accelerator()
        self._prefetch_model = self.accelerator.unwrap_model(self.model)
        self._init_checkpointing()
        if initialization_eval:
            if self.loaded_checkpoint_path is not None or self.completed_steps != 0:
                raise RuntimeError(
                    "Untrained-initialization eval unexpectedly loaded a checkpoint."
                )
        elif self.loaded_checkpoint_path is None:
            raise RuntimeError("Eval-only preparation did not load its checkpoint.")

    def evaluate_checkpoint_only(self) -> dict:
        """Evaluate one loaded checkpoint and exit without a training iterator."""

        initialization_eval = bool(
            self.config.trainer.get(
                "eval_only_untrained_initialization", False
            )
        )
        if self.loaded_checkpoint_path is None and not initialization_eval:
            raise RuntimeError("No checkpoint is loaded for eval-only execution.")
        metrics = self.eval_heldout_action_model({})
        metrics["epoch"] = 0.0
        metrics["samples_seen"] = 0.0
        if self.accelerator.is_main_process:
            logger.info(
                f"Eval-only checkpoint metrics at step {self.completed_steps}: {metrics}"
            )
        distributed_wait(self.accelerator)
        return metrics

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)
        if self.accelerator.is_main_process:
            logger.info("Step 0 debug: entered prepare_training")

        trackers = resolve_trackers(self.config)
        if trackers:
            self.accelerator.init_trackers(
                project_name="starvla",
                config=flatten_tracker_config(OmegaConf.to_container(self.config, resolve=True)),
            )

        # load pretrained weights
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        if hasattr(self.model, "refresh_runtime_caches"):
            self.model.refresh_runtime_caches()
        if hasattr(self.model, "validate_depth_teacher_aux_training_state"):
            self.model.validate_depth_teacher_aux_training_state(
                depth_teacher_loss_scale=self.depth_teacher_loss_scale,
            )
        if hasattr(self.model, "validate_runtime_feature_state"):
            self.model.validate_runtime_feature_state()

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        if self.accelerator.is_main_process:
            logger.info("Step 0 debug: calling accelerator.prepare")
        self.model, self.optimizer, self.vla_train_dataloader, self.lr_scheduler = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
            # self.vlm_train_dataloader
        )
        if self.accelerator.is_main_process:
            logger.info("Step 0 debug: accelerator.prepare returned")
        # Prepare each independent view only after model/train objects. Each
        # view is exactly one effective global training batch.
        self._prepare_heldout_loaders_with_accelerator()
        self._validate_prepared_dataloader()
        if self.accelerator.is_main_process:
            logger.info("Step 0 debug: prepared dataloader validated")
        self._prefetch_model = self.accelerator.unwrap_model(self.model)

        #self._init_wandb()
        self._init_checkpointing()
        if self.accelerator.is_main_process:
            logger.info("Step 0 debug: checkpointing initialized")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            import wandb

            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _validate_prepared_dataloader(self):
        if not dist.is_initialized():
            return

        if callable(getattr(self.vla_train_dataloader, "set_epoch", None)):
            logger.info(
                "Prepared VLA dataloader wrapper "
                f"`{type(self.vla_train_dataloader).__name__}` manages epoch seeding"
            )
            return

        sampler = _find_sampler(self.vla_train_dataloader)

        if sampler is None:
            logger.info(
                f"Prepared VLA dataloader wrapper `{type(self.vla_train_dataloader).__name__}` does not expose a sampler directly; relying on Accelerate-managed sharding"
            )
        elif not callable(getattr(sampler, "set_epoch", None)):
            logger.warning(
                "Prepared VLA dataloader exposes a sampler without set_epoch(); verify shuffling behavior across ranks"
            )
        else:
            logger.info(f"Prepared VLA dataloader sampler: {type(sampler).__name__}")

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        self.force_checkpoint_path = os.path.join(self.config.output_dir, "FORCE_CHECKPOINT")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        is_resume = getattr(self.config.trainer, "is_resume", False)
        resume_from_checkpoint = (
            getattr(self.config, "resume_from_checkpoint", None)
            or getattr(self.config.trainer, "resume_from_checkpoint", None)
        )

        # resume training state
        if is_resume and resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

    def _force_checkpoint_requested(self) -> bool:
        if not bool(self.config.trainer.get("enable_force_checkpoint_file", True)):
            return False
        return os.path.exists(getattr(self, "force_checkpoint_path", ""))

    def _clear_force_checkpoint_request(self):
        if self.accelerator.is_main_process:
            try:
                if os.path.exists(self.force_checkpoint_path):
                    os.remove(self.force_checkpoint_path)
            except OSError as exc:
                logger.warning(f"Unable to clear force checkpoint request `{self.force_checkpoint_path}`: {exc}")
        distributed_wait(self.accelerator)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        if not Path(checkpoint_path).is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
        load_optimizer_state = bool(self.config.trainer.get("resume_load_optimizer_state", True))
        if load_optimizer_state:
            self.accelerator.load_state(checkpoint_path)
        else:
            model_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(model_path):
                from safetensors.torch import load_file as load_safetensors_file

                state_dict = load_safetensors_file(model_path, device="cpu")
            else:
                fallback_model_path = os.path.join(checkpoint_path, "pytorch_model.pt")
                if not os.path.exists(fallback_model_path):
                    raise FileNotFoundError(
                        f"Checkpoint `{checkpoint_path}` does not contain `model.safetensors` or `pytorch_model.pt`."
                    )
                state_dict = torch.load(fallback_model_path, map_location="cpu")
            incompatible_keys = self.accelerator.unwrap_model(self.model).load_state_dict(
                state_dict,
                strict=False,
            )
            allowed_missing_keys = {"qwen_vl_interface.model.lm_head.weight"}
            missing_keys = {
                key
                for key in incompatible_keys.missing_keys
                if key not in allowed_missing_keys
                and not is_depth_teacher_aux_missing_key_allowed(self.config, key)
            }
            unexpected_keys = set(incompatible_keys.unexpected_keys)
            unexpected_keys = {
                key
                for key in unexpected_keys
                if not is_depth_teacher_aux_unexpected_key_allowed(self.config, key)
            }
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "Model-only checkpoint load mismatch: "
                    f"missing_keys={sorted(missing_keys)} unexpected_keys={sorted(unexpected_keys)}"
                )
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, "r") as f:
                    trainer_state = json.load(f)
                self.completed_steps = int(trainer_state.get("completed_steps", self.completed_steps))
                best_metric_value = trainer_state.get("best_metric_value", None)
                self.best_metric_value = None if best_metric_value is None else float(best_metric_value)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning(f"Unable to load trainer state from checkpoint `{checkpoint_path}`: {exc}")
        else:
            checkpoint_name = os.path.basename(os.path.normpath(checkpoint_path))
            if checkpoint_name.startswith("steps_"):
                try:
                    self.completed_steps = int(checkpoint_name.split("_", 1)[1])
                except ValueError:
                    logger.warning(f"Unable to parse completed_steps from checkpoint path: {checkpoint_path}")
        # Accelerate restores the scheduler together with the optimizer during
        # a full-state resume. Stepping it again would preserve ``last_epoch``
        # but increment ``_step_count`` a second time, so the resumed scheduler
        # would no longer be state-identical to uninterrupted training. Only a
        # model-only resume needs to reconstruct schedule position from the
        # completed-step counter.
        if (
            not load_optimizer_state
            and self.completed_steps > 0
            and self.lr_scheduler is not None
        ):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.lr_scheduler.step(self.completed_steps)
            except Exception as exc:
                logger.warning(
                    f"Unable to fast-forward lr scheduler to resumed step {self.completed_steps}: {exc}"
                )
        resume_mode = "full_state" if load_optimizer_state else "model_only"
        self.loaded_checkpoint_path = checkpoint_path
        self.accelerator.print(f"Resumed from checkpoint ({resume_mode}): {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        os.makedirs(checkpoint_path, exist_ok=True)
        self.accelerator.save_state(checkpoint_path)
        distributed_wait(self.accelerator)
        if self.accelerator.is_main_process:
            if bool(self.config.trainer.get("save_plain_weights_in_checkpoints", False)):
                # Optional convenience artifact. Disabled by default because gathering and
                # serializing a second full copy of the model can create a large rank-0
                # memory spike during periodic checkpoint saves.
                state_dict = self.accelerator.get_state_dict(self.model)
                torch.save(state_dict, os.path.join(checkpoint_path, "pytorch_model.pt"))

            trainer_state = {
                "completed_steps": self.completed_steps,
                "best_metric_name": self.best_metric_name,
                "best_metric_mode": self.best_metric_mode,
                "best_metric_value": self.best_metric_value,
            }
            with open(os.path.join(checkpoint_path, "trainer_state.json"), "w") as f:
                json.dump(trainer_state, f, indent=2)

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")

            _make_artifact_tree_host_readable(checkpoint_path)

            if bool(self.config.trainer.get("drop_checkpoint_page_cache", True)):
                _drop_file_cache_best_effort(checkpoint_path)

            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            self._prune_old_checkpoints()

        if bool(self.config.trainer.get("trim_process_memory_after_checkpoint", True)):
            _trim_process_memory_best_effort()
        distributed_wait(self.accelerator)

    def _prune_old_checkpoints(self):
        max_to_keep = int(self.config.trainer.get("checkpoint_max_to_keep", 0) or 0)
        if max_to_keep <= 0:
            return

        checkpoints = []
        for name in os.listdir(self.checkpoint_dir):
            if not name.startswith("steps_"):
                continue
            try:
                step = int(name.split("_", 1)[1])
            except ValueError:
                continue
            checkpoints.append((step, os.path.join(self.checkpoint_dir, name)))

        checkpoints.sort(key=lambda item: item[0])
        for step, path in checkpoints[:-max_to_keep]:
            if step == self.completed_steps:
                continue
            try:
                shutil.rmtree(path)
                logger.info(f"Pruned old checkpoint at {path}")
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(f"Unable to prune old checkpoint `{path}`: {exc}")

    def _should_save_checkpoint(self, step_metrics: dict) -> bool:
        if getattr(self, "legacy_underfilled_eval", False):
            raise RuntimeError(
                "Legacy underfilled audit metrics are forbidden from checkpoint "
                "selection."
            )
        if not bool(self.config.trainer.get("save_best_only", False)):
            return True

        metric_value = self._to_scalar(step_metrics.get(self.best_metric_name))
        if metric_value is None:
            if self.accelerator.is_main_process and not self._warned_missing_best_metric:
                logger.warning(
                    f"save_best_only=true but metric `{self.best_metric_name}` is unavailable at step {self.completed_steps}; skipping interval checkpoint"
                )
                self._warned_missing_best_metric = True
            return False

        if self.best_metric_mode not in {"min", "max"}:
            raise ValueError(f"Unsupported best_metric_mode: {self.best_metric_mode}")

        is_better = (
            self.best_metric_value is None
            or (metric_value < self.best_metric_value if self.best_metric_mode == "min" else metric_value > self.best_metric_value)
        )
        if is_better:
            self.best_metric_value = metric_value
        return is_better

    def _log_metrics(self, metrics):
        if self.completed_steps > 0 and self.completed_steps % self.config.trainer.logging_frequency == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                scalar_metrics = {}
                for key, value in metrics.items():
                    scalar_value = self._to_scalar(value)
                    if scalar_value is not None:
                        scalar_metrics[key] = scalar_value

                scalar_metrics["epoch"] = self.completed_steps / max(
                    float(self.config.trainer.get("steps_per_epoch", len(self.vla_train_dataloader))),
                    1.0,
                )
                scalar_metrics["samples_seen"] = self.completed_steps * self.total_batch_size

                compute_step_time = scalar_metrics.get("data_time", 0.0) + scalar_metrics.get("model_time", 0.0)
                if compute_step_time > 0:
                    scalar_metrics["compute_step_time"] = compute_step_time
                    scalar_metrics["compute_samples_per_sec"] = self.total_batch_size / compute_step_time
                    scalar_metrics["compute_steps_per_sec"] = 1.0 / compute_step_time

                wall_step_time = scalar_metrics.get("wall_step_time", 0.0)
                if wall_step_time > 0:
                    scalar_metrics["samples_per_sec"] = self.total_batch_size / wall_step_time
                    scalar_metrics["steps_per_sec"] = 1.0 / wall_step_time
                    if compute_step_time > 0:
                        scalar_metrics["step_overhead_time"] = max(0.0, wall_step_time - compute_step_time)
                elif compute_step_time > 0:
                    scalar_metrics["samples_per_sec"] = self.total_batch_size / compute_step_time
                    scalar_metrics["steps_per_sec"] = 1.0 / compute_step_time

                elapsed = time.perf_counter() - self.train_start_time
                if elapsed > 0:
                    scalar_metrics["avg_samples_per_sec"] = scalar_metrics["samples_seen"] / elapsed

                if torch.cuda.is_available():
                    scalar_metrics["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
                    scalar_metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
                    scalar_metrics["gpu_mem_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    scalar_metrics["gpu_mem_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024 ** 3)

                for group in self.optimizer.param_groups:
                    group_name = group.get("name", "default")
                    scalar_metrics[f"lr_{group_name}"] = float(group["lr"])

                rtc_cfg = self.config.framework.action_model.get("rtc_training", {})
                if bool(rtc_cfg.get("enabled", False)):
                    scalar_metrics["rtc_training_probability"] = rtc_training_probability(
                        rtc_cfg,
                        train_step=self.completed_steps,
                        total_steps=int(self.config.trainer.max_train_steps),
                    )

                if self.accelerator.trackers:
                    self.accelerator.log(scalar_metrics, step=self.completed_steps)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                max_train_steps = max(int(self.config.trainer.max_train_steps), 1)
                progress_pct = 100.0 * self.completed_steps / max_train_steps
                eta_seconds = self._estimate_remaining_seconds()
                avg_wall_step_time = (
                    sum(self._recent_wall_step_times) / len(self._recent_wall_step_times)
                    if self._recent_wall_step_times
                    else None
                )
                progress_parts = [
                    f"Progress {self.completed_steps}/{max_train_steps} ({progress_pct:.2f}%)",
                    f"epoch={scalar_metrics.get('epoch', 0.0):.4f}",
                    f"eta={self._format_duration(eta_seconds)}",
                ]

                for metric_key, label in (
                    ("data_time", "data"),
                    ("model_time", "model"),
                    ("wall_step_time", "wall"),
                ):
                    metric_value = scalar_metrics.get(metric_key, None)
                    if metric_value is not None:
                        progress_parts.append(f"{label}={metric_value:.3f}s")

                if avg_wall_step_time is not None:
                    progress_parts.append(f"avg_wall={avg_wall_step_time:.3f}s")

                if bool(self.config.trainer.get("profile_cuda_memory", False)) and torch.cuda.is_available():
                    progress_parts.extend(
                        [
                            f"gpu_alloc={scalar_metrics.get('gpu_mem_allocated_gb', 0.0):.2f}GiB",
                            f"gpu_reserved={scalar_metrics.get('gpu_mem_reserved_gb', 0.0):.2f}GiB",
                            f"gpu_peak={scalar_metrics.get('gpu_mem_peak_allocated_gb', 0.0):.2f}GiB",
                            f"gpu_peak_reserved={scalar_metrics.get('gpu_mem_peak_reserved_gb', 0.0):.2f}GiB",
                        ]
                    )

                if self._runtime_timing_enabled():
                    for metric_key, label in (
                        ("raw_batch_fetch_time", "fetch"),
                        ("video_tensor_to_cuda_time", "to_cuda"),
                        ("video_decode_time", "decode"),
                        ("video_postprocess_time", "post"),
                        ("qwen_tensor_build_time", "qwen"),
                        ("qwen_forward_time", "qfwd"),
                        ("depth_teacher_time", "depth"),
                        ("vj_encode_time", "vj"),
                        ("predictor_action_head_time", "head"),
                        ("forward_time", "fwd"),
                        ("backward_only_time", "back"),
                        ("grad_clip_time", "clip"),
                        ("optimizer_step_time", "opt"),
                        ("backward_optimizer_time", "bwd"),
                    ):
                        metric_value = scalar_metrics.get(metric_key, None)
                        if metric_value is not None:
                            progress_parts.append(f"{label}={metric_value:.3f}s")

                logger.info(" | ".join(progress_parts))

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vla_epoch_count = 0
        if self._rank_video_prefetch_enabled():
            self._rank_video_prefetcher = _RankVideoBatchPrefetcher(
                self,
                queue_size=self._rank_video_prefetch_queue_size(),
            )
            self._rank_video_prefetcher.start()
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _rank_video_prefetch_enabled(self) -> bool:
        data_cfg = self.config.datasets.vla_data
        return bool(
            data_cfg.get("gpu_video_decode_on_rank", False)
            and data_cfg.get("gpu_video_decode_async_prefetch", True)
        )

    def _rank_video_prefetch_queue_size(self) -> int:
        return max(1, int(self.config.datasets.vla_data.get("gpu_video_decode_prefetch_queue_size", 2)))

    def _get_next_raw_batch(self):
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)
        return batch_vla

    def _prepare_prefetched_batch(self, batch_vla, stream=None):
        if not self._rank_video_prefetch_enabled():
            return batch_vla, None
        model = self._prefetch_model or self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "prepare_rank_prefetched_batch"):
            return batch_vla, None
        return model.prepare_rank_prefetched_batch(batch_vla, stream=stream)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        if self.completed_steps == 0 and self.accelerator.is_main_process:
            logger.info("Step 0 debug: fetching first training batch")
        if self._rank_video_prefetcher is not None:
            batch_vla = self._rank_video_prefetcher.next_batch()
        else:
            batch_vla = self._get_next_raw_batch()
        self._last_prefetch_timing = None
        if isinstance(batch_vla, list) and batch_vla and isinstance(batch_vla[0], dict):
            timing_payload = batch_vla[0].pop("_prefetch_timing", None)
            if isinstance(timing_payload, dict):
                self._last_prefetch_timing = timing_payload
        elif isinstance(batch_vla, dict):
            timing_payload = batch_vla.pop("_prefetch_timing", None)
            if isinstance(timing_payload, dict):
                self._last_prefetch_timing = timing_payload
        if self.completed_steps == 0 and self.accelerator.is_main_process:
            logger.info(f"Step 0 debug: fetched first training batch { _summarize_batch_structure(batch_vla) }")

        return batch_vla

    def _get_next_eval_batch(self):
        """Get an eval batch from the same live training iterator."""
        return self._get_next_batch()

    def _resolve_loss_scales(self):
        loss_scale_cfg = self.config.trainer.get("loss_scale", {})
        action_scale = float(loss_scale_cfg.get("action", loss_scale_cfg.get("vla", 1.0)))
        wm_scale = float(loss_scale_cfg.get("wm", loss_scale_cfg.get("vlm", 0.1)))
        wm_initial_scale = float(loss_scale_cfg.get("wm_initial", loss_scale_cfg.get("wm_start", wm_scale)))
        wm_warmup_steps = max(int(loss_scale_cfg.get("wm_warmup_steps", 0)), 0)
        depth_teacher_cfg = self.config.framework.get("depth_teacher_aux", {})
        if bool(depth_teacher_cfg.get("enabled", False)):
            if depth_teacher_cfg.get("loss_weight", None) is not None:
                raise ValueError(
                    "Depth teacher loss weight must be configured only at "
                    "`trainer.loss_scale.depth_teacher`; remove "
                    "`framework.depth_teacher_aux.loss_weight`."
                )
            if loss_scale_cfg.get("depth_aux", None) is not None:
                raise ValueError(
                    "Depth teacher loss weight must be configured only at "
                    "`trainer.loss_scale.depth_teacher`; remove legacy "
                    "`trainer.loss_scale.depth_aux`."
                )
            if loss_scale_cfg.get("depth_teacher", None) is None:
                raise ValueError(
                    "depth_teacher_aux is enabled but `trainer.loss_scale.depth_teacher` is not set."
                )
            depth_teacher_scale = float(loss_scale_cfg.get("depth_teacher"))
        else:
            depth_teacher_scale = float(loss_scale_cfg.get("depth_teacher", 0.0))
        return action_scale, wm_scale, wm_initial_scale, wm_warmup_steps, depth_teacher_scale

    def _current_wm_loss_scale(self) -> float:
        if self.wm_loss_warmup_steps <= 0:
            return self.wm_loss_scale
        progress = min(max(float(self.completed_steps) / float(self.wm_loss_warmup_steps), 0.0), 1.0)
        return self.wm_loss_scale_initial + (self.wm_loss_scale - self.wm_loss_scale_initial) * progress

    def train(self):
        """execute training loop"""
        if self.accelerator.is_main_process:
            logger.info("Step 0 debug: entered train()")
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()
        self.optimizer.zero_grad(set_to_none=True)
        eval_before_train = bool(
            self.config.trainer.get("eval_before_train", False)
        )
        if eval_before_train and self.completed_steps == 0:
            if self.vla_eval_dataloader is not None:
                baseline_metrics = self.eval_heldout_action_model({})
            else:
                baseline_metrics = self.eval_action_model({})
            if self.accelerator.is_main_process:
                baseline_metrics["epoch"] = 0.0
                baseline_metrics["samples_seen"] = 0.0
                if self.accelerator.trackers:
                    self.accelerator.log(baseline_metrics, step=0)
                logger.info(f"Step 0 Eval Metrics: {baseline_metrics}")
        elif eval_before_train and self.completed_steps > 0:
            if self.accelerator.is_main_process:
                logger.info(
                    "Skipping eval_before_train after resume at completed step "
                    f"{self.completed_steps}; the immutable step-0 baseline "
                    "belongs to the fresh run only."
                )

        # create progress bar
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=min(self.completed_steps, self.config.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_step = time.perf_counter()
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            # Capture this once at the optimization boundary.  Accelerate keeps
            # ``sync_gradients`` false on intermediate accumulation
            # microbatches; periodic evaluation and checkpointing must never
            # re-run merely because ``completed_steps`` still names the last
            # completed optimizer step.
            optimizer_step_completed = bool(self.accelerator.sync_gradients)
            t_end_model = time.perf_counter()

            if optimizer_step_completed:
                progress_bar.update(1)
                self.completed_steps += 1

            eval_due = (
                optimizer_step_completed
                and self.completed_steps > 0
                and self.completed_steps % self.config.trainer.eval_interval == 0
            )
            save_due = (
                optimizer_step_completed
                and self.completed_steps > 0
                and self.completed_steps % self.config.trainer.save_interval == 0
            )
            checkpoint_saved_this_step = False

            # A heldout evaluator is intentionally fail-closed.  At a
            # coincident unconditional save/eval boundary, make the completed
            # optimizer step recoverable first so a decode, inference, metric,
            # or collective failure cannot discard the entire run.  Best-only
            # saving still has to evaluate first because the metric determines
            # whether a checkpoint should exist.
            if (
                eval_due
                and save_due
                and not bool(self.config.trainer.get("save_best_only", False))
            ):
                self._save_checkpoint()
                checkpoint_saved_this_step = True

            # The legacy "eval" path consumes the next shuffled training batch.
            # Keep it available only as an explicitly named diagnostic; it is
            # not held-out validation and must not silently advance the train
            # iterator or select a best checkpoint.
            if eval_due:
                if self.vla_eval_dataloader is not None:
                    eval_start = time.perf_counter()
                    step_metrics = self.eval_heldout_action_model(step_metrics)
                    step_metrics["heldout_eval_time"] = time.perf_counter() - eval_start
                elif bool(self.config.trainer.get("allow_training_stream_eval", False)):
                    step_metrics = self.eval_action_model(step_metrics)
                elif (
                    self.accelerator.is_main_process
                    and not self._warned_training_stream_eval_disabled
                ):
                    logger.warning(
                        "Periodic in-process evaluation is disabled because no held-out "
                        "dataloader is configured. Add an immutable episode_split_manifest "
                        "for one-window-per-episode checkpoint eval. The legacy path consumes "
                        "a shuffled training batch; set "
                        "trainer.allow_training_stream_eval=true only for a clearly "
                        "labeled training-stream diagnostic."
                    )
                    self._warned_training_stream_eval_disabled = True

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model

            # save checkpoint
            if optimizer_step_completed and self.completed_steps > 0:
                force_checkpoint_requested = self._force_checkpoint_requested()
                if save_due:
                    if not checkpoint_saved_this_step:
                        should_save_checkpoint = self._should_save_checkpoint(
                            step_metrics
                        )
                        if should_save_checkpoint or force_checkpoint_requested:
                            if (
                                force_checkpoint_requested
                                and self.accelerator.is_main_process
                            ):
                                logger.info(
                                    f"Force checkpoint requested via `{self.force_checkpoint_path}` at step {self.completed_steps}"
                                )
                            self._save_checkpoint()
                            checkpoint_saved_this_step = True
                    elif (
                        force_checkpoint_requested
                        and self.accelerator.is_main_process
                    ):
                        logger.info(
                            "Force checkpoint request was satisfied by the "
                            f"already-saved step {self.completed_steps} checkpoint"
                        )
                    if force_checkpoint_requested:
                        self._clear_force_checkpoint_request()
                elif force_checkpoint_requested:
                    if self.accelerator.is_main_process:
                        logger.info(
                            f"Force checkpoint requested via `{self.force_checkpoint_path}` at step {self.completed_steps}"
                        )
                    self._save_checkpoint()
                    checkpoint_saved_this_step = True
                    self._clear_force_checkpoint_request()

            step_metrics["wall_step_time"] = time.perf_counter() - t_start_step
            if optimizer_step_completed and self.completed_steps > self.progress_eta_warmup_steps:
                self._recent_wall_step_times.append(step_metrics["wall_step_time"])
            if (
                optimizer_step_completed
                and self.completed_steps > 0
                and self.completed_steps % self._detailed_timing_frequency() == 0
            ):
                self._log_rank_timing_stats(self._collect_rank_timing_stats(step_metrics) or {})

            if self.accelerator.is_local_main_process:
                eta_seconds = self._estimate_remaining_seconds()
                avg_wall_step_time = (
                    sum(self._recent_wall_step_times) / len(self._recent_wall_step_times)
                    if self._recent_wall_step_times
                    else None
                )
                postfix = {
                    "data_times": f"{t_end_data - t_start_data:.3f}",
                    "model_times": f"{t_end_model - t_start_model:.3f}",
                    "wall_time": f"{step_metrics['wall_step_time']:.3f}",
                    "avg_wall": f"{avg_wall_step_time:.3f}" if avg_wall_step_time is not None else "warmup",
                    "eta": self._format_duration(eta_seconds),
                }
                if self._runtime_timing_enabled():
                    timing_keys = (
                        ("raw_batch_fetch_time", "fetch"),
                        ("video_tensor_to_cuda_time", "to_cuda"),
                        ("video_decode_time", "decode"),
                        ("video_postprocess_time", "post"),
                        ("qwen_tensor_build_time", "qwen"),
                        ("qwen_forward_time", "qfwd"),
                        ("depth_teacher_time", "depth"),
                        ("vj_encode_time", "vj"),
                        ("predictor_action_head_time", "head"),
                        ("forward_time", "fwd"),
                        ("backward_only_time", "back"),
                        ("grad_clip_time", "clip"),
                        ("optimizer_step_time", "opt"),
                        ("backward_optimizer_time", "bwd"),
                    )
                    for metric_key, label in timing_keys:
                        metric_value = self._to_scalar(step_metrics.get(metric_key))
                        if metric_value is not None:
                            postfix[label] = f"{metric_value:.3f}"
                progress_bar.set_postfix(postfix)

            self._log_metrics(step_metrics)

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    @staticmethod
    def _eval_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _heldout_indices_from_batch(examples) -> list[int]:
        if isinstance(examples, Mapping):
            values = examples.get("_heldout_eval_index")
            if values is None:
                return []
            return [int(value) for value in VLATrainer._eval_numpy(values).reshape(-1)]
        if isinstance(examples, Sequence) and not isinstance(examples, (str, bytes)):
            indices = []
            for example in examples:
                if not isinstance(example, Mapping) or "_heldout_eval_index" not in example:
                    return []
                indices.append(int(example["_heldout_eval_index"]))
            return indices
        return []

    @staticmethod
    def _without_heldout_metadata(examples):
        if isinstance(examples, Mapping):
            return {
                key: value
                for key, value in examples.items()
                if not str(key).startswith("_heldout_eval_")
            }
        if isinstance(examples, Sequence) and not isinstance(examples, (str, bytes)):
            return [
                {
                    key: value
                    for key, value in example.items()
                    if not str(key).startswith("_heldout_eval_")
                }
                if isinstance(example, Mapping)
                else example
                for example in examples
            ]
        return examples

    @staticmethod
    def _heldout_metadata_array(examples, key: str) -> np.ndarray | None:
        if isinstance(examples, Mapping):
            value = examples.get(key)
            return None if value is None else VLATrainer._eval_numpy(value)
        if isinstance(examples, Sequence) and not isinstance(examples, (str, bytes)):
            if not examples or any(
                not isinstance(example, Mapping) or key not in example
                for example in examples
            ):
                return None
            return np.asarray(
                [VLATrainer._eval_numpy(example[key]) for example in examples]
            )
        return None

    def _eval_action_groups(self, action_dim: int) -> dict[str, tuple[int, ...]]:
        groups: dict[str, tuple[int, ...]] = {
            "all_action": tuple(range(int(action_dim)))
        }
        arm_dimensions = self._eval_arm_dimensions(action_dim)
        if arm_dimensions is not None:
            groups["arm"] = arm_dimensions
        if int(action_dim) in {18, 19, 22}:
            groups["gripper"] = (7, 15)
            if int(action_dim) in {18, 19}:
                groups["head"] = (16, 17)
                if int(action_dim) == 19:
                    groups["lift"] = (18,)
            else:
                groups["base"] = (16, 17, 18)
                groups["head"] = (19, 20)
                groups["lift"] = (21,)
        return groups

    def _eval_arm_dimensions(self, action_dim: int) -> tuple[int, ...] | None:
        configured = self.config.trainer.get(
            "heldout_eval_arm_dimensions",
            None,
        )
        if configured is not None:
            dimensions = tuple(int(value) for value in configured)
        elif int(action_dim) in {18, 19, 22}:
            # RealMan: left 7 arm joints, left gripper, right 7 arm
            # joints, right gripper, then head/base/lift controls.
            dimensions = tuple(range(0, 7)) + tuple(range(8, 15))
        else:
            return None
        if (
            not dimensions
            or len(set(dimensions)) != len(dimensions)
            or min(dimensions) < 0
            or max(dimensions) >= int(action_dim)
        ):
            raise ValueError(
                "trainer.heldout_eval_arm_dimensions must be unique indices "
                f"inside action_dim={action_dim}, got {dimensions}."
            )
        return dimensions

    def _evaluate_action_batches(
        self,
        batches,
        *,
        step_metrics: dict,
        metric_prefix: str,
        expected_observations: int | None = None,
        expected_valid_observations: int | None = None,
        expected_valid_elements: int | None = None,
        require_heldout_indices: bool = False,
        sampling_report: Mapping | None = None,
        evaluation_seed: int | None = None,
        transition_coverage_horizon: int | None = None,
        minimum_open_to_close_transitions: int = 0,
        minimum_close_to_open_transitions: int = 0,
        minimum_open_to_close_windows: int = 0,
        minimum_close_to_open_windows: int = 0,
        minimum_arm_movement_elements: int = 0,
        minimum_arm_movement_hold_abs: float = 0.0,
    ) -> dict:
        """Run deterministic deployment-style diffusion and control diagnostics."""

        infer_model = self.accelerator.unwrap_model(self.model)
        module_modes = [(module, bool(module.training)) for module in infer_model.modules()]
        num_inference_timesteps = int(
            self.config.framework.action_model.get("num_inference_timesteps", 8)
        )
        if num_inference_timesteps <= 0:
            raise ValueError(
                "framework.action_model.num_inference_timesteps must be positive "
                "for heldout checkpoint evaluation."
            )
        movement_threshold = float(
            self.config.trainer.get("heldout_eval_movement_threshold", 0.02)
        )
        if not math.isfinite(movement_threshold) or movement_threshold < 0:
            raise ValueError(
                "trainer.heldout_eval_movement_threshold must be finite and "
                "non-negative."
            )

        requested_prefix_horizons = (1, 5, 10, 20, 50)
        subtask_counts, evaluable_subtask_counts = _heldout_report_subtask_counts(
            sampling_report
        )
        report_action_subtasks: set[int] = set()
        if sampling_report is not None:
            for counts in sampling_report.get(
                "subtask_action_timestep_counts_by_horizon", {}
            ).values():
                if isinstance(counts, Mapping):
                    report_action_subtasks.update(int(key) for key in counts)
        expected_subtasks = tuple(
            sorted(set(subtask_counts) | report_action_subtasks)
        )
        local_heldout_indices: list[int] = []
        action_shape: tuple[int, int] | None = None
        action_groups: dict[str, tuple[int, ...]] = {}
        available_prefix_horizons: tuple[int, ...] = ()
        control_metadata_seen: bool | None = None

        totals = {
            "abs_sum": 0.0,
            "sq_sum": 0.0,
            "element_count": 0.0,
            "observations": 0.0,
            "valid_observations": 0.0,
        }
        group_stats: dict[tuple[int, str], dict[str, float]] = {}
        subtask_stats: dict[tuple[int, int, str], dict[str, float]] = {}
        subtask_observations = {subtask: 0.0 for subtask in expected_subtasks}
        subtask_valid_observations = {
            subtask: 0.0 for subtask in expected_subtasks
        }
        subtask_action_timesteps: dict[tuple[int, int], float] = {}
        gripper_stats: dict[int, dict[str, float]] = {}

        def _new_group_stats() -> dict[str, float]:
            return {
                "policy_abs": 0.0,
                "count": 0.0,
                "hold_abs": 0.0,
                "movement_policy_abs": 0.0,
                "movement_hold_abs": 0.0,
                "movement_count": 0.0,
                "direction_correct": 0.0,
            }

        def _new_gripper_stats() -> dict[str, float]:
            return {
                "target_close": 0.0,
                "target_open": 0.0,
                "predicted_close": 0.0,
                "true_close": 0.0,
                "true_open": 0.0,
                "open_to_close": 0.0,
                "open_to_close_correct": 0.0,
                "close_to_open": 0.0,
                "close_to_open_correct": 0.0,
                "open_to_close_windows": 0.0,
                "close_to_open_windows": 0.0,
            }

        try:
            infer_model.eval()
            with _isolated_evaluation_rng(
                int(
                    getattr(self, "heldout_eval_seed", 0)
                    if evaluation_seed is None
                    else evaluation_seed
                ),
                self.accelerator.device,
            ), torch.inference_mode():
                for raw_examples in batches:
                    heldout_indices = self._heldout_indices_from_batch(raw_examples)
                    if require_heldout_indices and not heldout_indices:
                        raise RuntimeError(
                            "Heldout eval batch is missing deterministic reference indices; "
                            "cannot prove complete, duplicate-free coverage."
                        )
                    local_heldout_indices.extend(heldout_indices)
                    hold_actions = self._heldout_metadata_array(
                        raw_examples, "_heldout_eval_hold_action"
                    )
                    action_midpoints = self._heldout_metadata_array(
                        raw_examples, "_heldout_eval_action_midpoint"
                    )
                    subtask_indices = self._heldout_metadata_array(
                        raw_examples, "_heldout_eval_subtask_index"
                    )
                    action_subtask_indices = self._heldout_metadata_array(
                        raw_examples, "_heldout_eval_action_subtask_indices"
                    )
                    has_control_metadata = all(
                        value is not None
                        for value in (
                            hold_actions,
                            action_midpoints,
                            subtask_indices,
                            action_subtask_indices,
                        )
                    )
                    if require_heldout_indices and not has_control_metadata:
                        raise RuntimeError(
                            "Heldout eval batch is missing current-state hold, action "
                            "midpoint, anchor-subtask, or per-action subtask metadata "
                            "required by control metrics."
                        )
                    if control_metadata_seen is None:
                        control_metadata_seen = has_control_metadata
                    elif control_metadata_seen != has_control_metadata:
                        raise RuntimeError(
                            "Heldout control metadata presence changed between microbatches."
                        )

                    examples = self._without_heldout_metadata(raw_examples)
                    if isinstance(examples, Mapping):
                        actions = self._eval_numpy(examples["action"])
                        action_mask = examples.get("action_mask")
                        action_is_pad = examples.get("action_is_pad")
                        output_dict = infer_model.predict_action(
                            batch=examples,
                            prev_actions=None,
                            prefix_len=0,
                            rtc_config=None,
                            num_inference_timesteps=num_inference_timesteps,
                        )
                    else:
                        if not examples:
                            raise RuntimeError("Evaluation loader produced an empty batch.")
                        actions = np.asarray(
                            [self._eval_numpy(example["action"]) for example in examples]
                        )
                        action_mask = (
                            [self._eval_numpy(example["action_mask"]) for example in examples]
                            if all("action_mask" in example for example in examples)
                            else None
                        )
                        action_is_pad = (
                            [self._eval_numpy(example["action_is_pad"]) for example in examples]
                            if all("action_is_pad" in example for example in examples)
                            else None
                        )
                        state = (
                            [example["state"] for example in examples]
                            if all("state" in example for example in examples)
                            else None
                        )
                        output_dict = infer_model.predict_action(
                            batch=examples,
                            state=state,
                            prev_actions=None,
                            prefix_len=0,
                            rtc_config=None,
                            num_inference_timesteps=num_inference_timesteps,
                        )

                    actions = np.asarray(actions, dtype=np.float32)
                    normalized_actions = np.asarray(
                        self._eval_numpy(output_dict["normalized_actions"]),
                        dtype=np.float32,
                    )
                    if actions.ndim != 3:
                        raise RuntimeError(
                            "Evaluation actions must have shape [batch, horizon, dim], "
                            f"got {actions.shape}."
                        )
                    if normalized_actions.shape != actions.shape:
                        raise RuntimeError(
                            "Evaluation prediction/target shape mismatch: "
                            f"predicted={normalized_actions.shape}, target={actions.shape}."
                        )
                    current_action_shape = (int(actions.shape[1]), int(actions.shape[2]))
                    if action_shape is None:
                        action_shape = current_action_shape
                        action_groups = self._eval_action_groups(actions.shape[2])
                        available_prefix_horizons = tuple(
                            horizon
                            for horizon in requested_prefix_horizons
                            if horizon <= int(actions.shape[1])
                        )
                        for horizon in available_prefix_horizons:
                            for group_name in action_groups:
                                group_stats[(horizon, group_name)] = _new_group_stats()
                                for subtask in expected_subtasks:
                                    subtask_stats[(subtask, horizon, group_name)] = {
                                        "policy_abs": 0.0,
                                        "count": 0.0,
                                    }
                            for subtask in expected_subtasks:
                                subtask_action_timesteps[(subtask, horizon)] = 0.0
                            if "gripper" in action_groups:
                                gripper_stats[horizon] = _new_gripper_stats()
                    elif current_action_shape != action_shape:
                        raise RuntimeError(
                            "Evaluation action shape changed between microbatches: "
                            f"{current_action_shape} != {action_shape}."
                        )

                    metric_mask = np.ones_like(actions, dtype=bool)
                    if action_mask is not None:
                        raw_mask = np.asarray(action_mask, dtype=bool)
                        if raw_mask.shape != actions.shape:
                            raw_mask = np.broadcast_to(raw_mask, actions.shape)
                        metric_mask &= raw_mask
                    if action_is_pad is not None:
                        timestep_mask = ~np.asarray(action_is_pad, dtype=bool)
                        if timestep_mask.ndim == 2:
                            timestep_mask = timestep_mask[..., None]
                        metric_mask &= np.broadcast_to(timestep_mask, actions.shape)

                    nonfinite_valid = metric_mask & (
                        ~np.isfinite(normalized_actions) | ~np.isfinite(actions)
                    )
                    if bool(nonfinite_valid.any()):
                        raise RuntimeError(
                            "Evaluation prediction/target contains non-finite values "
                            "at supervised action elements: "
                            f"count={int(nonfinite_valid.sum())}."
                        )
                    diff = np.zeros_like(normalized_actions, dtype=np.float32)
                    np.subtract(
                        normalized_actions,
                        actions,
                        out=diff,
                        where=metric_mask,
                    )
                    per_observation_counts = metric_mask.reshape(
                        metric_mask.shape[0], -1
                    ).sum(axis=1)
                    valid_observation = per_observation_counts > 0

                    totals["abs_sum"] += float(np.abs(diff).sum())
                    totals["sq_sum"] += float(np.square(diff).sum())
                    totals["element_count"] += float(metric_mask.sum())
                    totals["observations"] += float(actions.shape[0])
                    totals["valid_observations"] += float(valid_observation.sum())

                    if has_control_metadata:
                        hold_actions = np.asarray(hold_actions, dtype=np.float32)
                        action_midpoints = np.asarray(action_midpoints, dtype=np.float32)
                        subtask_indices = np.asarray(subtask_indices).reshape(-1)
                        action_subtask_indices = np.asarray(action_subtask_indices)
                        if hold_actions.ndim == 1 and actions.shape[0] == 1:
                            hold_actions = hold_actions[None, :]
                        if action_midpoints.ndim == 1 and actions.shape[0] == 1:
                            action_midpoints = action_midpoints[None, :]
                        expected_metadata_shape = (actions.shape[0], actions.shape[2])
                        if hold_actions.shape != expected_metadata_shape:
                            raise RuntimeError(
                                "Heldout current-state hold shape mismatch: "
                                f"{hold_actions.shape} != {expected_metadata_shape}."
                            )
                        if action_midpoints.shape != expected_metadata_shape:
                            raise RuntimeError(
                                "Heldout action-midpoint shape mismatch: "
                                f"{action_midpoints.shape} != {expected_metadata_shape}."
                            )
                        if subtask_indices.shape != (actions.shape[0],):
                            raise RuntimeError(
                                "Heldout subtask metadata shape mismatch: "
                                f"{subtask_indices.shape} != {(actions.shape[0],)}."
                            )
                        if action_subtask_indices.ndim == 1 and actions.shape[0] == 1:
                            action_subtask_indices = action_subtask_indices[None, :]
                        expected_action_subtask_shape = actions.shape[:2]
                        if action_subtask_indices.shape != expected_action_subtask_shape:
                            raise RuntimeError(
                                "Heldout per-action subtask metadata shape mismatch: "
                                f"{action_subtask_indices.shape} != "
                                f"{expected_action_subtask_shape}."
                            )
                        if not np.all(np.isfinite(hold_actions)) or not np.all(
                            np.isfinite(action_midpoints)
                        ):
                            raise RuntimeError(
                                "Heldout hold or action-midpoint metadata contains NaN/Inf."
                            )
                        subtasks = np.asarray(
                            [int(value) for value in subtask_indices], dtype=np.int64
                        )
                        action_subtasks = np.asarray(
                            action_subtask_indices, dtype=np.int64
                        )
                        unknown_subtasks = sorted(
                            (
                                set(subtasks.tolist())
                                | set(action_subtasks.reshape(-1).tolist())
                            )
                            - set(expected_subtasks)
                        )
                        if unknown_subtasks:
                            raise RuntimeError(
                                "Heldout eval observed subtasks absent from its immutable "
                                f"sampling report: {unknown_subtasks}."
                            )
                        hold_chunk = np.broadcast_to(
                            hold_actions[:, None, :], actions.shape
                        )
                        target_delta = actions - hold_chunk
                        prediction_delta = normalized_actions - hold_chunk
                        hold_error = np.zeros_like(actions, dtype=np.float32)
                        np.subtract(
                            hold_chunk,
                            actions,
                            out=hold_error,
                            where=metric_mask,
                        )
                        movement_mask = metric_mask & (
                            np.abs(target_delta) >= movement_threshold
                        )
                        direction_correct = target_delta * prediction_delta > 0
                        for subtask in expected_subtasks:
                            selected = subtasks == subtask
                            subtask_observations[subtask] += float(selected.sum())
                            subtask_valid_observations[subtask] += float(
                                (selected & valid_observation).sum()
                            )
                    else:
                        hold_error = None
                        movement_mask = None
                        direction_correct = None
                        subtasks = None
                        action_subtasks = None

                    for horizon in available_prefix_horizons:
                        if has_control_metadata:
                            for subtask in expected_subtasks:
                                subtask_action_timesteps[(subtask, horizon)] += float(
                                    (action_subtasks[:, :horizon] == subtask).sum()
                                )
                        for group_name, dimensions in action_groups.items():
                            dim_array = np.asarray(dimensions, dtype=np.int64)
                            selected_mask = metric_mask[:, :horizon, :][..., dim_array]
                            selected_diff = diff[:, :horizon, :][..., dim_array]
                            stats = group_stats[(horizon, group_name)]
                            stats["policy_abs"] += float(np.abs(selected_diff).sum())
                            stats["count"] += float(selected_mask.sum())
                            if has_control_metadata:
                                selected_hold = hold_error[:, :horizon, :][..., dim_array]
                                selected_movement = movement_mask[:, :horizon, :][
                                    ..., dim_array
                                ]
                                stats["hold_abs"] += float(np.abs(selected_hold).sum())
                                stats["movement_policy_abs"] += float(
                                    np.abs(selected_diff)[selected_movement].sum()
                                )
                                stats["movement_hold_abs"] += float(
                                    np.abs(selected_hold)[selected_movement].sum()
                                )
                                stats["movement_count"] += float(
                                    selected_movement.sum()
                                )
                                stats["direction_correct"] += float(
                                    direction_correct[:, :horizon, :][..., dim_array][
                                        selected_movement
                                    ].sum()
                                )
                                for subtask in expected_subtasks:
                                    timestep_selector = (
                                        action_subtasks[:, :horizon] == subtask
                                    )
                                    stage_mask = selected_mask & timestep_selector[
                                        :, :, None
                                    ]
                                    stage_stats = subtask_stats[
                                        (subtask, horizon, group_name)
                                    ]
                                    stage_stats["policy_abs"] += float(
                                        np.abs(selected_diff)[stage_mask].sum()
                                    )
                                    stage_stats["count"] += float(stage_mask.sum())
                        if has_control_metadata and "gripper" in action_groups:
                            gripper_dimensions = np.asarray(
                                action_groups["gripper"], dtype=np.int64
                            )
                            gripper_mask = metric_mask[:, :horizon, :][
                                ..., gripper_dimensions
                            ]
                            thresholds = action_midpoints[:, None, gripper_dimensions]
                            target_close = (
                                actions[:, :horizon, :][..., gripper_dimensions]
                                < thresholds
                            )
                            predicted_close = (
                                normalized_actions[:, :horizon, :][..., gripper_dimensions]
                                < thresholds
                            )
                            current_close = hold_actions[:, gripper_dimensions] < action_midpoints[
                                :, gripper_dimensions
                            ]
                            previous_close = np.concatenate(
                                (current_close[:, None, :], target_close[:, :-1, :]),
                                axis=1,
                            )
                            previous_valid = np.concatenate(
                                (
                                    np.ones_like(current_close[:, None, :], dtype=bool),
                                    gripper_mask[:, :-1, :],
                                ),
                                axis=1,
                            )
                            transition_valid = gripper_mask & previous_valid
                            open_to_close = (
                                ~previous_close & target_close & transition_valid
                            )
                            close_to_open = (
                                previous_close & ~target_close & transition_valid
                            )
                            stats = gripper_stats[horizon]
                            stats["target_close"] += float(
                                (target_close & gripper_mask).sum()
                            )
                            stats["target_open"] += float(
                                (~target_close & gripper_mask).sum()
                            )
                            stats["predicted_close"] += float(
                                (predicted_close & gripper_mask).sum()
                            )
                            stats["true_close"] += float(
                                (predicted_close & target_close & gripper_mask).sum()
                            )
                            stats["true_open"] += float(
                                (~predicted_close & ~target_close & gripper_mask).sum()
                            )
                            stats["open_to_close"] += float(open_to_close.sum())
                            stats["open_to_close_correct"] += float(
                                (predicted_close & open_to_close).sum()
                            )
                            stats["close_to_open"] += float(close_to_open.sum())
                            stats["close_to_open_correct"] += float(
                                (~predicted_close & close_to_open).sum()
                            )
                            stats["open_to_close_windows"] += float(
                                open_to_close.any(axis=(1, 2)).sum()
                            )
                            stats["close_to_open_windows"] += float(
                                close_to_open.any(axis=(1, 2)).sum()
                            )
        finally:
            for module, was_training in module_modes:
                module.training = was_training

        if action_shape is None:
            raise RuntimeError("Evaluation loader produced no batches.")
        if transition_coverage_horizon is not None and "gripper" not in action_groups:
            raise RuntimeError(
                "Transition coverage was requested for an action layout without "
                "the RealMan gripper actuator group."
            )

        scalar_entries: list[tuple[tuple, float]] = [
            (("total", key), float(value)) for key, value in totals.items()
        ]
        for key in sorted(group_stats):
            for field in sorted(group_stats[key]):
                scalar_entries.append((("group", *key, field), group_stats[key][field]))
        for subtask in expected_subtasks:
            scalar_entries.append(
                (("subtask_observations", subtask), subtask_observations[subtask])
            )
            scalar_entries.append(
                (
                    ("subtask_valid_observations", subtask),
                    subtask_valid_observations[subtask],
                )
            )
        for key in sorted(subtask_stats):
            for field in sorted(subtask_stats[key]):
                scalar_entries.append(
                    (("subtask", *key, field), subtask_stats[key][field])
                )
        for key in sorted(subtask_action_timesteps):
            scalar_entries.append(
                (
                    ("subtask_action_timesteps", *key),
                    subtask_action_timesteps[key],
                )
            )
        for horizon in sorted(gripper_stats):
            for field in sorted(gripper_stats[horizon]):
                scalar_entries.append(
                    (("gripper", horizon, field), gripper_stats[horizon][field])
                )
        local_metrics = torch.tensor(
            [value for _, value in scalar_entries],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        reduced_metrics = self.accelerator.reduce(local_metrics, reduction="sum")
        reduced = {
            key: float(value)
            for (key, _), value in zip(scalar_entries, reduced_metrics.tolist())
        }
        total_abs_sum = reduced[("total", "abs_sum")]
        total_sq_sum = reduced[("total", "sq_sum")]
        total_count = reduced[("total", "element_count")]
        total_observations = reduced[("total", "observations")]
        total_valid_observations = reduced[("total", "valid_observations")]

        if expected_observations is not None:
            if int(total_observations) != int(expected_observations):
                raise RuntimeError(
                    "Heldout checkpoint eval did not visit exactly one effective "
                    f"global batch: got {int(total_observations)} observations, "
                    f"expected {int(expected_observations)}."
                )
            if dist.is_initialized():
                # The frozen pilot audit deliberately has 95 examples over
                # eight ranks (7x12 + 1x11). Tensor all-gather requires equal
                # shapes and would either fail or force duplicate padding.
                # Object gather preserves the exact variable-length index list
                # from each rank so the no-replacement audit remains provable.
                gathered_indices = gather_object(local_heldout_indices)
            else:
                local_index_tensor = torch.tensor(
                    local_heldout_indices,
                    device=self.accelerator.device,
                    dtype=torch.int64,
                )
                gather = getattr(self.accelerator, "gather", None)
                if callable(gather):
                    gathered_indices = (
                        gather(local_index_tensor).detach().cpu().tolist()
                    )
                else:
                    # Unit-test/single-process accelerators may expose only reduce.
                    gathered_indices = local_index_tensor.detach().cpu().tolist()
            expected_indices = list(range(int(expected_observations)))
            if sorted(int(value) for value in gathered_indices) != expected_indices:
                raise RuntimeError(
                    "Heldout checkpoint eval reference coverage contains missing or "
                    "duplicate episode windows."
                )
        if expected_valid_observations is not None and int(
            total_valid_observations
        ) != int(expected_valid_observations):
            raise RuntimeError(
                "Heldout eval action-evaluable observation coverage differs from "
                f"its immutable sampling report: got {int(total_valid_observations)}, "
                f"expected {int(expected_valid_observations)}."
            )
        if expected_valid_elements is not None and int(total_count) != int(
            expected_valid_elements
        ):
            raise RuntimeError(
                "Heldout eval valid action-element coverage differs from its "
                f"immutable sampling report: got {int(total_count)}, expected "
                f"{int(expected_valid_elements)}."
            )

        step_metrics[f"{metric_prefix}_observations"] = float(total_observations)
        step_metrics[f"{metric_prefix}_action_evaluable_observations"] = float(
            total_valid_observations
        )
        step_metrics[f"{metric_prefix}_valid_action_elements"] = float(total_count)
        for horizon in available_prefix_horizons:
            for group_name in action_groups:
                count = reduced[("group", horizon, group_name, "count")]
                policy_abs = reduced[("group", horizon, group_name, "policy_abs")]
                step_metrics[
                    f"{metric_prefix}_valid_{group_name}_elements_h{horizon}"
                ] = count
                if count > 0:
                    step_metrics[
                        f"{metric_prefix}_normalized_{group_name}_mae_h{horizon}"
                    ] = policy_abs / count
                if control_metadata_seen:
                    hold_abs = reduced[("group", horizon, group_name, "hold_abs")]
                    movement_count = reduced[
                        ("group", horizon, group_name, "movement_count")
                    ]
                    movement_policy_abs = reduced[
                        ("group", horizon, group_name, "movement_policy_abs")
                    ]
                    movement_hold_abs = reduced[
                        ("group", horizon, group_name, "movement_hold_abs")
                    ]
                    direction_correct = reduced[
                        ("group", horizon, group_name, "direction_correct")
                    ]
                    if count > 0:
                        hold_mae = hold_abs / count
                        step_metrics[
                            f"{metric_prefix}_current_state_hold_normalized_"
                            f"{group_name}_mae_h{horizon}"
                        ] = hold_mae
                        if hold_abs > 0:
                            step_metrics[
                                f"{metric_prefix}_policy_vs_hold_{group_name}_"
                                f"mae_ratio_h{horizon}"
                            ] = policy_abs / hold_abs
                    step_metrics[
                        f"{metric_prefix}_movement_{group_name}_elements_h{horizon}"
                    ] = movement_count
                    if movement_count > 0:
                        step_metrics[
                            f"{metric_prefix}_normalized_{group_name}_movement_"
                            f"mae_h{horizon}"
                        ] = movement_policy_abs / movement_count
                        step_metrics[
                            f"{metric_prefix}_current_state_hold_normalized_"
                            f"{group_name}_movement_mae_h{horizon}"
                        ] = movement_hold_abs / movement_count
                        if movement_hold_abs > 0:
                            step_metrics[
                                f"{metric_prefix}_policy_vs_hold_{group_name}_"
                                f"movement_mae_ratio_h{horizon}"
                            ] = movement_policy_abs / movement_hold_abs
                        step_metrics[
                            f"{metric_prefix}_{group_name}_movement_direction_"
                            f"accuracy_h{horizon}"
                        ] = direction_correct / movement_count

        if control_metadata_seen:
            for subtask in expected_subtasks:
                observations = reduced[("subtask_observations", subtask)]
                valid_observations = reduced[
                    ("subtask_valid_observations", subtask)
                ]
                expected_count = int(subtask_counts.get(subtask, 0))
                expected_valid_count = int(evaluable_subtask_counts.get(subtask, 0))
                if int(observations) != expected_count or int(
                    valid_observations
                ) != expected_valid_count:
                    raise RuntimeError(
                        "Heldout per-subtask coverage differs from its immutable "
                        f"sampling report for subtask {subtask}: observations="
                        f"{int(observations)}/{expected_count}, evaluable="
                        f"{int(valid_observations)}/{expected_valid_count}."
                    )
                step_metrics[
                    f"{metric_prefix}_subtask_{subtask}_observations"
                ] = observations
                step_metrics[
                    f"{metric_prefix}_subtask_{subtask}_action_evaluable_observations"
                ] = valid_observations
                for horizon in available_prefix_horizons:
                    action_timestep_count = reduced[
                        ("subtask_action_timesteps", subtask, horizon)
                    ]
                    step_metrics[
                        f"{metric_prefix}_subtask_{subtask}_action_timesteps_h{horizon}"
                    ] = action_timestep_count
                    expected_action_timestep_counts = (
                        _heldout_report_action_subtask_counts(
                            sampling_report,
                            horizon=horizon,
                            field="subtask_action_timestep_counts_by_horizon",
                        )
                    )
                    if expected_action_timestep_counts and int(
                        action_timestep_count
                    ) != int(expected_action_timestep_counts.get(subtask, 0)):
                        raise RuntimeError(
                            "Heldout per-action subtask timestep coverage differs "
                            f"from its immutable report at h{horizon}, subtask "
                            f"{subtask}: {int(action_timestep_count)}/"
                            f"{int(expected_action_timestep_counts.get(subtask, 0))}."
                        )
                    for group_name in action_groups:
                        count = reduced[
                            ("subtask", subtask, horizon, group_name, "count")
                        ]
                        policy_abs = reduced[
                            ("subtask", subtask, horizon, group_name, "policy_abs")
                        ]
                        step_metrics[
                            f"{metric_prefix}_subtask_{subtask}_valid_{group_name}_"
                            f"elements_h{horizon}"
                        ] = count
                        if count > 0:
                            step_metrics[
                                f"{metric_prefix}_subtask_{subtask}_normalized_"
                                f"{group_name}_mae_h{horizon}"
                            ] = policy_abs / count
                        if group_name == "all_action":
                            expected_valid_counts = (
                                _heldout_report_action_subtask_counts(
                                    sampling_report,
                                    horizon=horizon,
                                    field=(
                                        "subtask_valid_action_element_counts_by_horizon"
                                    ),
                                )
                            )
                            if expected_valid_counts and int(count) != int(
                                expected_valid_counts.get(subtask, 0)
                            ):
                                raise RuntimeError(
                                    "Heldout per-action subtask valid-element "
                                    "coverage differs from its immutable report at "
                                    f"h{horizon}, subtask {subtask}: {int(count)}/"
                                    f"{int(expected_valid_counts.get(subtask, 0))}."
                                )

            if "gripper" in action_groups:
                for horizon in available_prefix_horizons:
                    values = {
                        field: reduced[("gripper", horizon, field)]
                        for field in gripper_stats[horizon]
                    }
                    for field in (
                        "target_close",
                        "target_open",
                        "open_to_close",
                        "close_to_open",
                    ):
                        step_metrics[
                            f"{metric_prefix}_gripper_{field}_elements_h{horizon}"
                        ] = values[field]
                    step_metrics[
                        f"{metric_prefix}_gripper_open_to_close_windows_h{horizon}"
                    ] = values["open_to_close_windows"]
                    step_metrics[
                        f"{metric_prefix}_gripper_close_to_open_windows_h{horizon}"
                    ] = values["close_to_open_windows"]
                    if values["target_close"] > 0:
                        step_metrics[
                            f"{metric_prefix}_gripper_close_recall_h{horizon}"
                        ] = values["true_close"] / values["target_close"]
                    if values["predicted_close"] > 0:
                        step_metrics[
                            f"{metric_prefix}_gripper_close_precision_h{horizon}"
                        ] = values["true_close"] / values["predicted_close"]
                    if values["target_open"] > 0:
                        step_metrics[
                            f"{metric_prefix}_gripper_open_recall_h{horizon}"
                        ] = values["true_open"] / values["target_open"]
                    if values["target_close"] > 0 and values["target_open"] > 0:
                        step_metrics[
                            f"{metric_prefix}_gripper_balanced_accuracy_h{horizon}"
                        ] = 0.5 * (
                            values["true_close"] / values["target_close"]
                            + values["true_open"] / values["target_open"]
                        )
                    if values["open_to_close"] > 0:
                        step_metrics[
                            f"{metric_prefix}_gripper_open_to_close_recall_h{horizon}"
                        ] = (
                            values["open_to_close_correct"]
                            / values["open_to_close"]
                        )
                    if values["close_to_open"] > 0:
                        step_metrics[
                            f"{metric_prefix}_gripper_close_to_open_recall_h{horizon}"
                        ] = (
                            values["close_to_open_correct"]
                            / values["close_to_open"]
                        )
                    if (
                        values["open_to_close"] > 0
                        and values["close_to_open"] > 0
                    ):
                        open_to_close_recall = (
                            values["open_to_close_correct"]
                            / values["open_to_close"]
                        )
                        close_to_open_recall = (
                            values["close_to_open_correct"]
                            / values["close_to_open"]
                        )
                        transition_balanced_recall = 0.5 * (
                            open_to_close_recall + close_to_open_recall
                        )
                        transition_min_recall = min(
                            open_to_close_recall,
                            close_to_open_recall,
                        )
                        step_metrics[
                            f"{metric_prefix}_gripper_transition_balanced_"
                            f"recall_h{horizon}"
                        ] = transition_balanced_recall
                        step_metrics[
                            f"{metric_prefix}_gripper_transition_min_"
                            f"recall_h{horizon}"
                        ] = transition_min_recall
                        arm_ratio_key = (
                            f"{metric_prefix}_policy_vs_hold_arm_movement_"
                            f"mae_ratio_h{horizon}"
                        )
                        if arm_ratio_key in step_metrics:
                            # Lower is better and non-compensatory: if either
                            # transition direction is ignored, arm motion alone
                            # cannot improve the score below 1. Arm quality only
                            # improves a checkpoint after both grasp and release
                            # recall are non-zero.
                            arm_ratio = max(float(step_metrics[arm_ratio_key]), 0.0)
                            task_success_score = transition_min_recall / (
                                1.0 + arm_ratio
                            )
                            step_metrics[
                                f"{metric_prefix}_task_success_score_h{horizon}"
                            ] = task_success_score
                            step_metrics[
                                f"{metric_prefix}_task_failure_score_h{horizon}"
                            ] = 1.0 - task_success_score

                if (
                    any(
                        value < 0
                        for value in (
                            minimum_open_to_close_transitions,
                            minimum_close_to_open_transitions,
                            minimum_open_to_close_windows,
                            minimum_close_to_open_windows,
                            minimum_arm_movement_elements,
                        )
                    )
                    or not math.isfinite(float(minimum_arm_movement_hold_abs))
                    or minimum_arm_movement_hold_abs < 0
                ):
                    raise ValueError(
                        "Heldout minimum movement/transition coverage values must "
                        "be finite and non-negative."
                    )
                if transition_coverage_horizon is not None:
                    transition_horizon = int(transition_coverage_horizon)
                    if transition_horizon not in available_prefix_horizons:
                        raise RuntimeError(
                            "Configured transition coverage horizon must be one of "
                            f"{available_prefix_horizons}, got {transition_horizon}."
                        )
                    transition_values = {
                        field: reduced[("gripper", transition_horizon, field)]
                        for field in gripper_stats[transition_horizon]
                    }
                    if (
                        transition_values["open_to_close"]
                        < minimum_open_to_close_transitions
                        or transition_values["close_to_open"]
                        < minimum_close_to_open_transitions
                        or transition_values["open_to_close_windows"]
                        < minimum_open_to_close_windows
                        or transition_values["close_to_open_windows"]
                        < minimum_close_to_open_windows
                    ):
                        raise RuntimeError(
                            "Heldout gripper transition coverage is insufficient at "
                            f"h{transition_horizon}: open_to_close_events="
                            f"{int(transition_values['open_to_close'])}/"
                            f"{minimum_open_to_close_transitions}, close_to_open_events="
                            f"{int(transition_values['close_to_open'])}/"
                            f"{minimum_close_to_open_transitions}, "
                            f"open_to_close_windows="
                            f"{int(transition_values['open_to_close_windows'])}/"
                            f"{minimum_open_to_close_windows}, "
                            f"close_to_open_windows="
                            f"{int(transition_values['close_to_open_windows'])}/"
                            f"{minimum_close_to_open_windows}."
                        )
                    arm_movement_count = reduced[
                        ("group", transition_horizon, "arm", "movement_count")
                    ]
                    arm_movement_hold_abs = reduced[
                        (
                            "group",
                            transition_horizon,
                            "arm",
                            "movement_hold_abs",
                        )
                    ]
                    if sampling_report is not None:
                        expected_transition_coverage = {
                            "open_to_close": sampling_report.get(
                                "open_to_close_transition_count_h10"
                            ),
                            "close_to_open": sampling_report.get(
                                "close_to_open_transition_count_h10"
                            ),
                            "open_to_close_windows": sampling_report.get(
                                "open_to_close_transition_window_count_h10"
                            ),
                            "close_to_open_windows": sampling_report.get(
                                "close_to_open_transition_window_count_h10"
                            ),
                        }
                        mismatched_transition_coverage = {
                            key: (int(transition_values[key]), int(expected))
                            for key, expected in expected_transition_coverage.items()
                            if expected is not None
                            and int(transition_values[key]) != int(expected)
                        }
                        if mismatched_transition_coverage:
                            raise RuntimeError(
                                "Heldout runtime transition coverage differs from "
                                "its immutable sampling report: "
                                f"{mismatched_transition_coverage}."
                            )
                        expected_movement_count = sampling_report.get(
                            "arm_movement_element_count_h10"
                        )
                        if expected_movement_count is not None and int(
                            arm_movement_count
                        ) != int(expected_movement_count):
                            raise RuntimeError(
                                "Heldout runtime H10 arm-movement element coverage "
                                "differs from its immutable sampling report: "
                                f"{int(arm_movement_count)}/"
                                f"{int(expected_movement_count)}."
                            )
                        expected_hold_abs = sampling_report.get(
                            "arm_movement_hold_abs_sum_h10"
                        )
                        if expected_hold_abs is not None and not math.isclose(
                            arm_movement_hold_abs,
                            float(expected_hold_abs),
                            rel_tol=1.0e-5,
                            abs_tol=1.0e-6,
                        ):
                            raise RuntimeError(
                                "Heldout runtime H10 arm-movement hold denominator "
                                "differs from its immutable sampling report: "
                                f"{arm_movement_hold_abs}/"
                                f"{float(expected_hold_abs)}."
                            )
                    if (
                        arm_movement_count < minimum_arm_movement_elements
                        or arm_movement_hold_abs < minimum_arm_movement_hold_abs
                    ):
                        raise RuntimeError(
                            "Heldout H10 arm-movement denominator coverage is "
                            "insufficient: elements="
                            f"{int(arm_movement_count)}/"
                            f"{minimum_arm_movement_elements}, hold_abs="
                            f"{arm_movement_hold_abs}/"
                            f"{minimum_arm_movement_hold_abs}."
                        )
                    task_failure_key = (
                        f"{metric_prefix}_task_failure_score_h{transition_horizon}"
                    )
                    if task_failure_key not in step_metrics:
                        raise RuntimeError(
                            "Heldout task failure score is unavailable despite "
                            "configured transition coverage. H10 arm movement/hold "
                            "coverage and both gripper transition directions are "
                            "required."
                        )
        if total_count > 0:
            step_metrics[f"{metric_prefix}_normalized_action_mae"] = (
                total_abs_sum / total_count
            )
            step_metrics[f"{metric_prefix}_normalized_action_rmse"] = (
                total_sq_sum / total_count
            ) ** 0.5
        return step_metrics

    @staticmethod
    def _heldout_report_evidence(report: Mapping, *, label: str) -> dict:
        digest = report.get("window_selection_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(
                f"{label} sampling report lacks a valid window-selection digest."
            )
        evidence_keys = (
            "schema_version",
            "purpose",
            "view",
            "algorithm",
            "seed_sha256",
            "window_selection_sha256",
            "observation_count",
            "action_evaluable_observation_count",
            "valid_action_timestep_count",
            "valid_action_element_count",
            "subtask_observation_counts",
            "subtask_evaluable_observation_counts",
            "open_to_close_transition_count_h10",
            "close_to_open_transition_count_h10",
            "open_to_close_transition_window_count_h10",
            "close_to_open_transition_window_count_h10",
            "arm_movement_element_count_h10",
            "arm_movement_hold_abs_sum_h10",
            "movement_threshold_normalized",
            "focused_subtasks",
            "subtask_action_timestep_counts_by_horizon",
            "subtask_valid_action_element_counts_by_horizon",
            "zero_valid_action_episodes",
            "production_valid",
            "checkpoint_selection_eligible",
            "legacy_underfilled_holdout",
            "episode_split_provenance",
        )
        evidence = {key: report[key] for key in evidence_keys if key in report}
        if "episode_split_provenance" not in evidence:
            raise RuntimeError(
                f"{label} sampling report lacks split/statistics provenance."
            )
        return evidence

    def _eval_source_training_config_evidence(self) -> dict[str, str]:
        """Validate the frozen source-run config used to build an eval model."""

        path_value = self.config.trainer.get(
            "eval_source_training_config_path", None
        )
        expected_sha256 = self.config.trainer.get(
            "eval_source_training_config_sha256", None
        )
        if not path_value or not expected_sha256:
            raise RuntimeError(
                "Eval-only checkpoint audit requires the frozen source run's "
                "config path and SHA-256."
            )
        path = Path(str(path_value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Frozen source training config does not exist: {path}"
            )
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != str(expected_sha256):
            raise RuntimeError(
                "Frozen source training config SHA-256 mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}."
            )
        return {"path": str(path), "sha256": actual_sha256}

    def _persist_heldout_eval_artifact(self, metrics: Mapping) -> Path | None:
        """Atomically persist machine-readable evidence independent of trackers."""

        if not getattr(self.accelerator, "is_main_process", True):
            return None
        output_dir_value = self.config.get("output_dir", None)
        if not output_dir_value:
            # Object-level evaluator tests and ad-hoc probes may have no run dir.
            return None
        if self.heldout_eval_sampling_report is None:
            raise RuntimeError("Cannot persist heldout eval without its sampling report.")

        output_dir = Path(str(output_dir_value)).expanduser().resolve()
        eval_only = bool(self.config.trainer.get("eval_only", False))
        config_path = output_dir / "config.yaml"
        if config_path.is_file():
            config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
            config_identity_path = "config.yaml"
        else:
            serialized_config = OmegaConf.to_yaml(self.config, resolve=True).encode(
                "utf-8"
            )
            config_sha256 = hashlib.sha256(serialized_config).hexdigest()
            config_identity_path = None
        resolved_schedule_path = output_dir / "resolved_training_schedule.json"
        resolved_schedule_evidence = None
        if resolved_schedule_path.is_file():
            resolved_schedule_evidence = _validated_resolved_schedule_evidence(
                resolved_schedule_path,
                config_sha256=config_sha256,
            )
        elif not eval_only:
            raise RuntimeError(
                "Live training heldout evaluation requires immutable resolved "
                "schedule provenance at "
                f"{resolved_schedule_path}."
            )

        def metric_group(prefix: str) -> dict[str, float]:
            result: dict[str, float] = {}
            for key, value in metrics.items():
                if not str(key).startswith(prefix):
                    continue
                scalar = self._to_scalar(value)
                if scalar is not None:
                    result[str(key)] = float(scalar)
            return dict(sorted(result.items()))

        unbiased_metrics = metric_group("heldout_eval_")
        focused_metrics = metric_group("heldout_focused_eval_")
        if not unbiased_metrics:
            raise RuntimeError("Heldout eval artifact has no unbiased metrics.")
        focused_enabled = (
            getattr(self, "vla_focused_eval_dataloader", None) is not None
        )
        if focused_enabled and not focused_metrics:
            raise RuntimeError("Heldout eval artifact has no focused metrics.")

        sampling_reports = {
            "unbiased": self._heldout_report_evidence(
                self.heldout_eval_sampling_report,
                label="Unbiased heldout",
            )
        }
        if focused_enabled:
            if self.heldout_focused_eval_sampling_report is None:
                raise RuntimeError(
                    "Focused eval artifact is missing its sampling report."
                )
            sampling_reports["focused"] = self._heldout_report_evidence(
                self.heldout_focused_eval_sampling_report,
                label="Focused heldout",
            )
        production_valid = all(
            report.get("production_valid") is True
            for report in sampling_reports.values()
        )
        checkpoint_selection_eligible = all(
            report.get("checkpoint_selection_eligible") is True
            for report in sampling_reports.values()
        )
        if getattr(self, "legacy_underfilled_eval", False) and (
            production_valid or checkpoint_selection_eligible
        ):
            raise RuntimeError(
                "Legacy underfilled audit cannot produce a production-valid or "
                "checkpoint-selection-eligible artifact."
            )

        step = int(self.completed_steps)
        source_training_config_evidence = None
        if eval_only:
            source_training_config_evidence = getattr(
                self,
                "eval_source_training_config_evidence",
                None,
            )
            if source_training_config_evidence is None:
                source_training_config_evidence = (
                    self._eval_source_training_config_evidence()
                )
        source_checkpoint = (
            getattr(self, "loaded_checkpoint_path", None)
            if eval_only
            else None
        )
        if source_checkpoint is None and not eval_only:
            # An unconditional save at a coincident save/eval boundary happens
            # before evaluation, so that checkpoint is an exact source. Step-0
            # and eval-only-without-save artifacts must never invent a
            # checkpoints/steps_N path that does not exist.
            live_checkpoint = output_dir / "checkpoints" / f"steps_{step}"
            if live_checkpoint.is_dir():
                source_checkpoint = str(live_checkpoint.resolve())
        checkpoint_relative_path = None
        if source_checkpoint is not None:
            try:
                checkpoint_relative_path = str(
                    Path(source_checkpoint).resolve().relative_to(output_dir)
                )
            except ValueError:
                checkpoint_relative_path = None
        checkpoint_identity: dict[str, object] = {
            "step": step,
            "source_path": source_checkpoint,
            "source_kind": (
                "checkpoint"
                if source_checkpoint
                else (
                    "deterministic_untrained_initialization"
                    if eval_only
                    and bool(
                        self.config.trainer.get(
                            "eval_only_untrained_initialization", False
                        )
                    )
                    else "live_in_memory_model"
                )
            ),
        }
        if source_checkpoint:
            checkpoint_root = Path(source_checkpoint)
            trainer_state_path = checkpoint_root / "trainer_state.json"
            if trainer_state_path.is_file():
                checkpoint_identity["trainer_state_sha256"] = hashlib.sha256(
                    trainer_state_path.read_bytes()
                ).hexdigest()
            for candidate in ("model.safetensors", "pytorch_model.pt"):
                model_path = checkpoint_root / candidate
                if model_path.is_file():
                    model_stat = model_path.stat()
                    checkpoint_identity["model_file"] = candidate
                    checkpoint_identity["model_file_size_bytes"] = int(
                        model_stat.st_size
                    )
                    break
        selection_metric_name = str(
            getattr(
                self,
                "best_metric_name",
                self.config.trainer.get("best_metric_name", "mae_score"),
            )
        )
        selection_metric_mode = str(
            getattr(
                self,
                "best_metric_mode",
                self.config.trainer.get("best_metric_mode", "min"),
            )
        )
        if (
            checkpoint_selection_eligible
            and selection_metric_name not in metrics
        ):
            raise RuntimeError(
                "Production-valid heldout eval is missing its configured "
                f"checkpoint-selection metric {selection_metric_name!r}."
            )
        payload = {
            "schema_version": 1,
            "checkpoint_step": step,
            "checkpoint_relative_path": checkpoint_relative_path,
            "checkpoint": checkpoint_identity,
            "run": {
                "run_id": str(self.config.get("run_id", "")),
                "output_dir": str(output_dir),
                "seed": int(self.config.get("seed", 0)),
                "config_path": config_identity_path,
                "config_sha256": config_sha256,
                "resolved_training_schedule": resolved_schedule_evidence,
                "source_training_config": source_training_config_evidence,
            },
            "sampling_reports": sampling_reports,
            "production_valid": production_valid,
            "checkpoint_selection_eligible": checkpoint_selection_eligible,
            "selection_metric": {
                "name": selection_metric_name,
                "mode": selection_metric_mode,
                "eligible": checkpoint_selection_eligible,
                "value": (
                    float(metrics[selection_metric_name])
                    if selection_metric_name in metrics
                    else None
                ),
            },
            "metrics": {
                "unbiased": unbiased_metrics,
                "focused": focused_metrics,
            },
        }
        artifact_dir = output_dir / "heldout_eval_metrics"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"step_{step:08d}.json"
        serialized_payload = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if artifact_path.is_file():
            if artifact_path.read_bytes() != serialized_payload:
                raise RuntimeError(
                    "Heldout evaluation evidence is immutable and a different "
                    f"payload already exists for step {step}: {artifact_path}"
                )
        else:
            _atomic_write_bytes(artifact_path, serialized_payload)
        _make_artifact_tree_host_readable(artifact_dir)
        return artifact_path

    def eval_heldout_action_model(self, step_metrics: dict = None) -> dict:
        if self.vla_eval_dataloader is None:
            raise RuntimeError("No independent heldout eval dataloader is configured.")
        metrics = step_metrics or {}
        _reset_loader_generators(
            self.vla_eval_dataloader,
            int(self.heldout_eval_seed),
        )
        try:
            metrics = self._evaluate_action_batches(
                self.vla_eval_dataloader,
                step_metrics=metrics,
                metric_prefix="heldout_eval",
                expected_observations=int(
                    self.heldout_eval_sampling_report["observation_count"]
                ),
                expected_valid_observations=int(
                    self.heldout_eval_expected_valid_observations
                ),
                expected_valid_elements=int(self.heldout_eval_expected_valid_elements),
                require_heldout_indices=True,
                sampling_report=self.heldout_eval_sampling_report,
                evaluation_seed=int(self.heldout_eval_seed),
            )
        finally:
            _close_heldout_eval_caches(self.vla_eval_dataloader)
        zero_valid_episodes = self.heldout_eval_sampling_report.get(
            "zero_valid_action_episodes", []
        )
        metrics["heldout_eval_zero_valid_action_episodes"] = float(
            len(zero_valid_episodes)
        )
        if getattr(self, "vla_focused_eval_dataloader", None) is not None:
            _reset_loader_generators(
                self.vla_focused_eval_dataloader,
                int(self.heldout_focused_eval_seed),
            )
            try:
                metrics = self._evaluate_action_batches(
                    self.vla_focused_eval_dataloader,
                    step_metrics=metrics,
                    metric_prefix="heldout_focused_eval",
                    expected_observations=int(
                        self.heldout_focused_eval_sampling_report[
                            "observation_count"
                        ]
                    ),
                    expected_valid_observations=int(
                        self.heldout_focused_eval_expected_valid_observations
                    ),
                    expected_valid_elements=int(
                        self.heldout_focused_eval_expected_valid_elements
                    ),
                    require_heldout_indices=True,
                    sampling_report=self.heldout_focused_eval_sampling_report,
                    evaluation_seed=int(self.heldout_focused_eval_seed),
                    transition_coverage_horizon=int(
                        self.config.trainer.get(
                            "heldout_focused_eval_transition_coverage_horizon", 10
                        )
                    ),
                    minimum_open_to_close_transitions=int(
                        self.config.trainer.get(
                            "heldout_focused_eval_min_open_to_close_transitions", 1
                        )
                    ),
                    minimum_close_to_open_transitions=int(
                        self.config.trainer.get(
                            "heldout_focused_eval_min_close_to_open_transitions", 1
                        )
                    ),
                    minimum_open_to_close_windows=int(
                        self.config.trainer.get(
                            "heldout_focused_eval_min_open_to_close_windows", 1
                        )
                    ),
                    minimum_close_to_open_windows=int(
                        self.config.trainer.get(
                            "heldout_focused_eval_min_close_to_open_windows", 1
                        )
                    ),
                    minimum_arm_movement_elements=int(
                        self.config.trainer.get(
                            "heldout_focused_eval_min_arm_movement_elements_h10", 1
                        )
                    ),
                    minimum_arm_movement_hold_abs=float(
                        self.config.trainer.get(
                            "heldout_focused_eval_min_arm_movement_hold_abs_h10",
                            1.0e-12,
                        )
                    ),
                )
            finally:
                _close_heldout_eval_caches(self.vla_focused_eval_dataloader)
            focused_zero_valid = self.heldout_focused_eval_sampling_report.get(
                "zero_valid_action_episodes", []
            )
            metrics["heldout_focused_eval_zero_valid_action_episodes"] = float(
                len(focused_zero_valid)
            )
        self._persist_heldout_eval_artifact(metrics)
        return metrics

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Run the explicitly enabled legacy training-stream diagnostic."""

        if not bool(self.config.trainer.get("allow_training_stream_eval", False)):
            raise RuntimeError(
                "eval_action_model() reads the live training iterator and is disabled by "
                "default. Set trainer.allow_training_stream_eval=true only for an "
                "explicit training-stream diagnostic."
            )
        return self._evaluate_action_batches(
            [self._get_next_eval_batch()],
            step_metrics=step_metrics or {},
            metric_prefix="train_stream_probe",
        )

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            if resolve_mixed_precision_mode(self.config) == "no":
                logger.info(
                    "  Mixed precision: Accelerator disabled; training uses manual torch.autocast(..., bfloat16) in the train/model path"
                )
            if bool(self.config.framework.get("depth_teacher_aux", {}).get("enabled", False)):
                logger.info(f"  Depth teacher auxiliary loss scale = {self.depth_teacher_loss_scale}")
                depth_teacher_model = self.accelerator.unwrap_model(self.model)
                detach_steps = (
                    depth_teacher_model.resolve_depth_teacher_detach_steps()
                    if hasattr(depth_teacher_model, "resolve_depth_teacher_detach_steps")
                    else int(self.config.framework.depth_teacher_aux.get("detach_vlm_steps", 0) or 0)
                )
                logger.info(
                    "  Depth teacher VLM detach warmup steps = "
                    f"{detach_steps}"
                )
            rtc_cfg = self.config.framework.action_model.get("rtc_training", {})
            if bool(rtc_cfg.get("enabled", False)):
                max_steps = int(self.config.trainer.max_train_steps)
                warmup_steps = int(rtc_cfg.get("warmup_steps", rtc_cfg.get("start_step", 0)) or 0)
                ramp_steps = int(rtc_cfg.get("ramp_steps", 0) or 0)
                probe_steps = sorted({0, max(warmup_steps - 1, 0), warmup_steps, warmup_steps + ramp_steps - 1})
                schedule = ", ".join(
                    f"step {step}: {rtc_training_probability(rtc_cfg, train_step=step, total_steps=max_steps):.4f}"
                    for step in probe_steps
                    if step >= 0
                )
                logger.info(
                    "  RTC training enabled: "
                    f"target_prob={float(rtc_cfg.get('rtc_prob', 1.0)):.4f}, "
                    f"warmup_steps={warmup_steps}, ramp_steps={ramp_steps}, "
                    f"condition_dit_tokens={bool(rtc_cfg.get('condition_dit_tokens', False))}, "
                    f"schedule=[{schedule}]"
                )
            if self.config.trainer.get("use_rabc", False) and float(self.config.trainer.get("rabc_mistake_weight", 0.0)) <= 0.0:
                logger.warning(
                    "RABC is enabled with rabc_mistake_weight <= 0.0; mistake-labeled samples are excluded from action_loss while still incurring full forward compute"
                )

    def _compute_rabc_weights(self, batch_vla):
        trainer_cfg = self.config.trainer
        if not trainer_cfg.get("use_rabc", False):
            return None, {}

        progress_key = trainer_cfg.get("rabc_progress_key", "rabc_progress_delta")
        fallback_progress_key = trainer_cfg.get("rabc_fallback_progress_key", "rabc_global_progress_delta")
        kappa = float(trainer_cfg.get("rabc_kappa", 0.01))
        epsilon = float(trainer_cfg.get("rabc_epsilon", 1e-6))
        mistake_penalty = float(trainer_cfg.get("rabc_mistake_weight", 0.0))
        valid_state_key = trainer_cfg.get("rabc_valid_state_key", None)
        future_valid_state_key = trainer_cfg.get(
            "rabc_future_valid_state_key",
            f"future_{valid_state_key}" if valid_state_key else None,
        )
        invalid_state_weight = float(trainer_cfg.get("rabc_invalid_state_weight", mistake_penalty))

        def _as_float_tensor(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                tensor = value.to(self.accelerator.device, dtype=torch.float32, non_blocking=True)
            else:
                tensor = torch.as_tensor(value, device=self.accelerator.device, dtype=torch.float32)
            if tensor.ndim > 1 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
            return tensor

        def _as_scalar_float(value, default=None):
            if value is None:
                return default
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return default
                value = value.reshape(-1)[0]
            if isinstance(value, (list, tuple)):
                if not value:
                    return default
                value = value[0]
            if isinstance(value, np.generic):
                value = value.item()
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        if isinstance(batch_vla, dict):
            raw_deltas = batch_vla.get(progress_key, batch_vla.get(fallback_progress_key))
            raw_valid_state = batch_vla.get(valid_state_key) if valid_state_key else None
            raw_future_valid_state = (
                batch_vla.get(future_valid_state_key)
                if future_valid_state_key and isinstance(batch_vla, dict)
                else None
            )
            if raw_deltas is None and raw_valid_state is None:
                return None, {}
            progress_available = raw_deltas is not None
            if progress_available:
                deltas = _as_float_tensor(raw_deltas)
                valid_mask = torch.isfinite(deltas)
                deltas = torch.where(valid_mask, deltas, torch.zeros_like(deltas))
            else:
                valid_reference = _as_float_tensor(raw_valid_state)
                deltas = torch.zeros_like(valid_reference)
                valid_mask = torch.ones_like(deltas, dtype=torch.bool)

            if raw_valid_state is None:
                valid_state_mask = None
            else:
                current_valid_state = _as_float_tensor(raw_valid_state)
                future_valid_state = (
                    _as_float_tensor(raw_future_valid_state)
                    if raw_future_valid_state is not None
                    else current_valid_state
                )
                valid_state_mask = (
                    torch.isfinite(current_valid_state)
                    & torch.isfinite(future_valid_state)
                    & (current_valid_state > 0.5)
                    & (future_valid_state > 0.5)
                )

            current_mistake = batch_vla.get("mistake_label")
            future_mistake = batch_vla.get("future_mistake_label")
            if current_mistake is None and future_mistake is None:
                mistake_mask = torch.zeros_like(deltas, dtype=torch.bool)
            else:
                current_mistake = (
                    _as_float_tensor(current_mistake)
                    if current_mistake is not None
                    else torch.zeros_like(deltas)
                )
                future_mistake = (
                    _as_float_tensor(future_mistake)
                    if future_mistake is not None
                    else torch.zeros_like(deltas)
                )
                # Training datasets standardize mistake_label so positive values mean "is a mistake",
                # even if the raw source uses the inverse convention.
                mistake_mask = torch.maximum(current_mistake, future_mistake) > 0.5
        else:
            deltas = []
            valid_mask = []
            valid_state_mask = []
            mistake_mask = []
            progress_available = False
            mistake_keys = (
                "mistake",
                "mistake_label",
                "is_mistake",
                "failure",
                "error",
            )

            for example in batch_vla:
                delta = example.get(progress_key, example.get(fallback_progress_key))
                if delta is None:
                    deltas.append(0.0)
                    valid_mask.append(bool(valid_state_key and valid_state_key in example))
                else:
                    delta = _as_scalar_float(delta, np.nan)
                    is_valid = not np.isnan(delta)
                    deltas.append(0.0 if not is_valid else delta)
                    valid_mask.append(is_valid)
                    progress_available = progress_available or is_valid

                if valid_state_key:
                    current_valid = _as_scalar_float(example.get(valid_state_key), None)
                    future_valid = _as_scalar_float(
                        example.get(future_valid_state_key, current_valid) if future_valid_state_key else current_valid,
                        current_valid,
                    )
                    if current_valid is None and future_valid is None:
                        valid_state_mask.append(True)
                    else:
                        valid_state_mask.append(
                            current_valid is not None
                            and future_valid is not None
                            and current_valid > 0.5
                            and future_valid > 0.5
                        )

                has_mistake = False
                for key in mistake_keys:
                    if key in example or f"future_{key}" in example:
                        current_val = _as_scalar_float(example.get(key), 0.0)
                        future_val = _as_scalar_float(example.get(f"future_{key}"), 0.0)
                        has_mistake = max(current_val, future_val) > 0.5
                        break
                mistake_mask.append(has_mistake)

            deltas = torch.tensor(deltas, device=self.accelerator.device, dtype=torch.float32)
            valid_mask = torch.tensor(valid_mask, device=self.accelerator.device, dtype=torch.bool)
            valid_state_mask = (
                torch.tensor(valid_state_mask, device=self.accelerator.device, dtype=torch.bool)
                if valid_state_key
                else None
            )
            mistake_mask = torch.tensor(mistake_mask, device=self.accelerator.device, dtype=torch.bool)

        if not valid_mask.any():
            return None, {}

        stats_mask = valid_mask
        if valid_state_mask is not None and (valid_mask & valid_state_mask).any():
            stats_mask = valid_mask & valid_state_mask
        valid_deltas = deltas[stats_mask]

        if progress_available:
            delta_mean = torch.clamp(valid_deltas.mean(), min=0.0)
            delta_std = torch.clamp(valid_deltas.std(unbiased=False), min=epsilon)
            lower_bound = delta_mean - 2 * delta_std
            soft_weights = torch.clamp((deltas - lower_bound) / (4 * delta_std + epsilon), 0.0, 1.0)

            weights = torch.zeros_like(deltas)
            weights = torch.where(deltas > kappa, torch.ones_like(weights), weights)
            moderate_mask = (deltas >= 0.0) & (deltas <= kappa)
            weights = torch.where(moderate_mask, soft_weights, weights)
            weights = torch.where(valid_mask, weights, torch.ones_like(weights))
        else:
            weights = torch.ones_like(deltas)
        if mistake_mask.any():
            weights = torch.where(mistake_mask, torch.full_like(weights, mistake_penalty), weights)
        if valid_state_mask is not None:
            weights = torch.where(valid_state_mask, weights, torch.full_like(weights, invalid_state_weight))

        weights = weights * weights.shape[0] / (weights.sum() + epsilon)
        stats = {
            "rabc_mean_weight": weights.mean().detach(),
            "rabc_valid_ratio": valid_mask.float().mean().detach(),
            "rabc_mean_delta": valid_deltas.mean().detach(),
            "rabc_zero_weight_ratio": (weights <= epsilon).float().mean().detach(),
        }
        if mistake_mask.any():
            stats["rabc_mistake_ratio"] = mistake_mask.float().mean().detach()
        if valid_state_mask is not None:
            stats["rabc_valid_state_ratio"] = valid_state_mask.float().mean().detach()
        return weights, stats

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            rabc_weights, rabc_stats = self._compute_rabc_weights(batch_vla)
            # VLA task forward propagation
            if self.completed_steps == 0 and self.accelerator.is_main_process:
                logger.info("Step 0 debug: starting model.forward")
            forward_start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(
                    batch_vla,
                    rabc_weights=rabc_weights,
                    train_step=self.completed_steps,
                )
                current_wm_loss_scale = self._current_wm_loss_scale()
                weighted_action_loss = output_dict.get("action_loss", 0.0) * self.action_loss_scale
                weighted_wm_loss = output_dict.get("wm_loss", 0.0) * current_wm_loss_scale
                weighted_depth_teacher_loss = (
                    output_dict.get("depth_teacher_loss", 0.0) * self.depth_teacher_loss_scale
                )
                total_loss = weighted_action_loss + weighted_wm_loss + weighted_depth_teacher_loss
            forward_time = time.perf_counter() - forward_start
            if self.completed_steps == 0 and self.accelerator.is_main_process:
                logger.info("Step 0 debug: finished model.forward")

            # VLA backward propagation
            if self.completed_steps == 0 and self.accelerator.is_main_process:
                logger.info("Step 0 debug: starting backward")
            backward_start = time.perf_counter()
            self.accelerator.backward(total_loss)
            backward_only_time = time.perf_counter() - backward_start
            if self.completed_steps == 0 and self.accelerator.is_main_process:
                logger.info("Step 0 debug: finished backward")

            # gradient clipping
            grad_norm = None
            grad_clip_time = 0.0
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                grad_clip_start = time.perf_counter()
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )
                grad_clip_time = time.perf_counter() - grad_clip_start

            # optimizer step
            optimizer_step_time = 0.0
            if self.accelerator.sync_gradients:
                optimizer_step_start = time.perf_counter()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_step_time = time.perf_counter() - optimizer_step_start
            backward_optimizer_time = time.perf_counter() - backward_start
            
            result_dict = {}
            for key, value in output_dict.items():
                result_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
            if "qwen_tensor_build_time" not in result_dict and "qwen_input_build_time" in result_dict:
                # Surface the non-prefetch Qwen build timing under the existing progress-bar key.
                result_dict["qwen_tensor_build_time"] = result_dict["qwen_input_build_time"]
            result_dict["total_loss"] = total_loss.detach() if isinstance(total_loss, torch.Tensor) else total_loss
            result_dict["weighted_action_loss"] = (
                weighted_action_loss.detach()
                if isinstance(weighted_action_loss, torch.Tensor)
                else weighted_action_loss
            )
            result_dict["weighted_wm_loss"] = (
                weighted_wm_loss.detach() if isinstance(weighted_wm_loss, torch.Tensor) else weighted_wm_loss
            )
            result_dict["weighted_depth_teacher_loss"] = (
                weighted_depth_teacher_loss.detach()
                if isinstance(weighted_depth_teacher_loss, torch.Tensor)
                else weighted_depth_teacher_loss
            )
            result_dict["action_loss_scale"] = self.action_loss_scale
            result_dict["wm_loss_scale"] = current_wm_loss_scale
            result_dict["target_wm_loss_scale"] = self.wm_loss_scale
            result_dict["depth_teacher_loss_scale"] = self.depth_teacher_loss_scale
            for key, value in rabc_stats.items():
                result_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
            if grad_norm is not None:
                result_dict["grad_norm"] = grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else grad_norm
            result_dict["forward_time"] = forward_time
            result_dict["backward_only_time"] = backward_only_time
            result_dict["grad_clip_time"] = grad_clip_time
            result_dict["optimizer_step_time"] = optimizer_step_time
            result_dict["backward_optimizer_time"] = backward_optimizer_time
            if self._last_prefetch_timing:
                for key, value in self._last_prefetch_timing.items():
                    result_dict[key] = value
            if bool(self.config.trainer.get("profile_cuda_memory", False)) and torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
                max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                max_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                result_dict["cuda_memory_allocated_gb"] = allocated_gb
                result_dict["cuda_memory_reserved_gb"] = reserved_gb
                result_dict["cuda_max_memory_allocated_gb"] = max_allocated_gb
                result_dict["cuda_max_memory_reserved_gb"] = max_reserved_gb
                log_step = int(self.config.trainer.get("profile_cuda_memory_log_step", 10))
                step_index = self.completed_steps + 1
                if self.accelerator.is_main_process and step_index == log_step:
                    self.accelerator.print(
                        "CUDA memory after step "
                        f"{step_index}: allocated={allocated_gb:.2f} GiB, reserved={reserved_gb:.2f} GiB, "
                        f"max_allocated={max_allocated_gb:.2f} GiB, max_reserved={max_reserved_gb:.2f} GiB"
                    )

        return result_dict

    def _finalize_training(self):
        if bool(self.config.trainer.get("save_final_model", True)):
            state_dict = self.accelerator.get_state_dict(self.model)
            if self.accelerator.is_main_process:
                final_checkpoint = os.path.join(self.config.output_dir, "final_model")
                os.makedirs(final_checkpoint, exist_ok=True)
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
                _make_artifact_tree_host_readable(final_checkpoint)
                if bool(self.config.trainer.get("drop_checkpoint_page_cache", True)):
                    _drop_file_cache_best_effort(final_checkpoint)
                logger.info(f"Training complete. Final model saved at {final_checkpoint}")
            if bool(self.config.trainer.get("trim_process_memory_after_checkpoint", True)):
                _trim_process_memory_best_effort()
        elif self.accelerator.is_main_process:
            logger.info("Training complete. Final model save skipped because trainer.save_final_model=false.")
        self._shutdown_data_runtime()
        distributed_wait(self.accelerator)
        finish_trackers(self.accelerator)

    def _shutdown_data_runtime(self):
        if self._data_runtime_shutdown:
            return
        self._data_runtime_shutdown = True
        if self._rank_video_prefetcher is not None:
            try:
                self._rank_video_prefetcher.close()
            except Exception as exc:
                logger.warning(f"Rank video prefetcher shutdown failed: {exc}")
            self._rank_video_prefetcher = None
        for attr_name in ("vla_iter", "vla_eval_iter"):
            iterator = getattr(self, attr_name, None)
            _shutdown_dataloader_iterator(iterator)
            setattr(self, attr_name, None)
        _shutdown_dataloader_workers(getattr(self, "vla_train_dataloader", None))
        _shutdown_dataloader_workers(getattr(self, "vla_eval_dataloader", None))
        _shutdown_dataloader_workers(
            getattr(self, "vla_focused_eval_dataloader", None)
        )
        gc.collect()


def main(cfg) -> None:
    eval_only = bool(cfg.trainer.get("eval_only", False))
    legacy_underfilled_eval = bool(
        cfg.trainer.get("eval_only_legacy_underfilled_holdout", False)
    )
    if legacy_underfilled_eval and not eval_only:
        raise ValueError(
            "trainer.eval_only_legacy_underfilled_holdout is audit-only and "
            "cannot be used during training or checkpoint selection."
        )
    accelerator = build_accelerator(cfg)
    logger.info(
        "VLA Checkpoint Evaluation :: Warming Up"
        if eval_only
        else "VLA Training :: Warming Up"
    )
    interrupted = False
    trainer = None
    vla_train_dataloader = None
    vla_eval_dataloader = None
    vla_focused_eval_dataloader = None

    try:
        # create output directory and save config
        output_dir = setup_directories(cfg=cfg)
        if eval_only and bool(
            cfg.trainer.get("eval_only_untrained_initialization", False)
        ):
            # Model construction happens before trainer preparation. Pin it so
            # the step-0 reference is reproducible across launches/ranks.
            set_seed(int(cfg.get("seed", 0)))
        # build model
        vla = build_model(cfg)
        # Eval-only deliberately never constructs or reads a shuffled training
        # loader and never creates an optimizer.
        if not eval_only:
            vla_train_dataloader = prepare_data(
                cfg=cfg,
                accelerator=accelerator,
                output_dir=output_dir,
                model=vla,
            )
        (
            vla_eval_dataloader,
            vla_focused_eval_dataloader,
        ) = prepare_heldout_eval_data(
            cfg=cfg,
            accelerator=accelerator,
            output_dir=output_dir,
            model=vla,
        )
        if eval_only:
            optimizer = None
            lr_scheduler = None
        else:
            resolved_training_schedule = resolve_training_schedule(
                cfg=cfg,
                vla_train_dataloader=vla_train_dataloader,
                num_processes=accelerator.num_processes,
            )
            if accelerator.is_main_process:
                persist_resolved_training_schedule(
                    cfg,
                    resolved_training_schedule,
                )
                if bool(cfg.trainer.get("is_resume", False)):
                    _persist_pending_resume_invocation_snapshot(
                        Path(str(cfg.output_dir)).expanduser().resolve(),
                        cfg,
                    )
            distributed_wait(accelerator)
            optimizer, lr_scheduler = setup_optimizer_and_scheduler(
                model=vla, cfg=cfg
            )

        trainer = VLATrainer(
            cfg=cfg,
            model=vla,
            vla_train_dataloader=vla_train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            vla_eval_dataloader=vla_eval_dataloader,
            vla_focused_eval_dataloader=vla_focused_eval_dataloader,
        )

        if eval_only:
            trainer.prepare_checkpoint_evaluation()
            trainer.evaluate_checkpoint_only()
        else:
            trainer.prepare_training()
            trainer.train()

        # And... we're done!
        logger.info("... and that's all, folks!")
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Training interrupted; shutting down distributed workers")
        raise
    finally:
        if trainer is not None:
            try:
                trainer._shutdown_data_runtime()
            except Exception as exc:
                logger.warning(f"Training data shutdown failed: {exc}")
        elif vla_train_dataloader is not None:
            try:
                _shutdown_dataloader_workers(vla_train_dataloader)
                _shutdown_dataloader_workers(vla_eval_dataloader)
                _shutdown_dataloader_workers(vla_focused_eval_dataloader)
            except Exception as exc:
                logger.warning(f"Fallback dataloader shutdown failed: {exc}")
        if dist.is_initialized():
            if not interrupted:
                try:
                    distributed_wait(accelerator)
                except Exception as exc:
                    logger.warning(f"Distributed barrier during shutdown failed: {exc}")
            try:
                dist.destroy_process_group()
            except Exception as exc:
                logger.warning(f"Distributed shutdown failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="scripts/config/vlajepa_robot_ft.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
