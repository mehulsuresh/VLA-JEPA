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
import gc
import math
import json
import os
import sys
import logging
from pathlib import Path
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
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.tracking import LoggerType
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


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
    dynamic_attempts: list[bool],
) -> None:
    eager_forward = module.forward
    attempted_compile_labels: set[str] = set()
    dynamic_attempts = list(dict.fromkeys(dynamic_attempts))
    active_dynamic: Optional[bool] = None
    active_impl = "uninitialized"
    compiled_forward_cache: dict[bool, Callable] = {}

    def _compile_forward(dynamic: bool):
        if dynamic not in compiled_forward_cache:
            compiled_forward_cache[dynamic] = torch.compile(
                eager_forward,
                dynamic=dynamic,
                mode=compile_mode,
            )
        return compiled_forward_cache[dynamic]

    def _log_compile_failure(dynamic: bool, exc: BaseException) -> None:
        compile_label = f"torch.compile(dynamic={dynamic}, mode='{compile_mode}')"
        if compile_label in attempted_compile_labels:
            return
        attempted_compile_labels.add(compile_label)
        logger.warning(
            f"{compile_label} failed for {module_name}; trying next fallback: {exc}"
        )

    def compiled_forward(*args, **kwargs):
        nonlocal active_dynamic, active_impl

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
                    f"Compiled {module_name} with torch.compile(dynamic={dynamic}, mode='{compile_mode}')"
                )
                return output
            except Exception as exc:
                if not _is_torch_compile_exception(exc):
                    raise
                _log_compile_failure(dynamic, exc)
                attempted_dynamics.add(dynamic)

        active_dynamic = None
        active_impl = "eager"
        module._starvla_active_compile_impl = "eager"
        module._starvla_active_compile_dynamic = None
        logger.warning(f"Falling back to eager forward for {module_name}")
        return eager_forward(*args, **kwargs)

    module._starvla_eager_forward = eager_forward
    module._starvla_compile_mode = compile_mode
    module._starvla_compile_dynamic_attempts = tuple(dynamic_attempts)
    module._starvla_active_compile_impl = "uninitialized"
    module._starvla_active_compile_dynamic = None
    module.forward = compiled_forward


def _install_compiled_callable_attr(
    owner,
    attr_name: str,
    target_name: str,
    compile_mode: str,
    dynamic_attempts: list[bool],
) -> None:
    eager_callable = getattr(owner, attr_name)
    if not callable(eager_callable):
        raise TypeError(f"`{target_name}` is not callable")

    attempted_compile_labels: set[str] = set()
    dynamic_attempts = list(dict.fromkeys(dynamic_attempts))
    active_dynamic: Optional[bool] = None
    active_impl = "uninitialized"
    compiled_callable_cache: dict[bool, Callable] = {}

    def _compile_callable(dynamic: bool):
        if dynamic not in compiled_callable_cache:
            compiled_callable_cache[dynamic] = torch.compile(
                eager_callable,
                dynamic=dynamic,
                mode=compile_mode,
            )
        return compiled_callable_cache[dynamic]

    def _log_compile_failure(dynamic: bool, exc: BaseException) -> None:
        compile_label = f"torch.compile(dynamic={dynamic}, mode='{compile_mode}')"
        if compile_label in attempted_compile_labels:
            return
        attempted_compile_labels.add(compile_label)
        logger.warning(
            f"{compile_label} failed for {target_name}; trying next fallback: {exc}"
        )

    def compiled_callable(*args, **kwargs):
        nonlocal active_dynamic, active_impl

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
                    f"Compiled {target_name} with torch.compile(dynamic={dynamic}, mode='{compile_mode}')"
                )
                return output
            except Exception as exc:
                if not _is_torch_compile_exception(exc):
                    raise
                _log_compile_failure(dynamic, exc)
                attempted_dynamics.add(dynamic)

        active_dynamic = None
        active_impl = "eager"
        logger.warning(f"Falling back to eager callable for {target_name}")
        return eager_callable(*args, **kwargs)

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
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(cfg.get("trainer", {}).get("find_unused_parameters", True))
    )
    accelerator = (
        Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            mixed_precision=mixed_precision,
            log_with=trackers or None,
            project_dir=project_dir,
            kwargs_handlers=[ddp_kwargs],
        )
        if use_deepspeed
        else Accelerator(
            mixed_precision=mixed_precision,
            log_with=trackers or None,
            project_dir=project_dir,
            kwargs_handlers=[ddp_kwargs],
        )
    )
    if torch.cuda.is_available():
        # Ensure NCCL collectives run on the process-local device before any early barrier().
        torch.cuda.set_device(accelerator.local_process_index)
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


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    trainer_cfg = cfg.get("trainer", {})
    compile_mode = trainer_cfg.get("compile_mode", "reduce-overhead")
    compile_dynamic = bool(trainer_cfg.get("compile_dynamic", True))
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
                    dynamic_attempts=[qwen_dynamic] if not qwen_dynamic else [True, False],
                )
                if hasattr(qwen_iface, "forward_features"):
                    _install_compiled_callable_attr(
                        qwen_iface,
                        "forward_features",
                        "qwen_vl_interface.forward_features",
                        compile_mode=compile_mode,
                        dynamic_attempts=[qwen_dynamic] if not qwen_dynamic else [True, False],
                    )
            except Exception as exc:
                logger.warning(f"torch.compile failed for qwen_vl_interface.model, continuing without it: {exc}")

        if trainer_cfg.get("compile_action_model", False):
            try:
                action_dynamic = _resolve_compile_dynamic(
                    trainer_cfg,
                    "compile_action_model_dynamic",
                    False,
                )
                _install_compiled_forward(
                    model.action_model,
                    "action_model",
                    compile_mode=compile_mode,
                    dynamic_attempts=[action_dynamic] if not action_dynamic else [True, False],
                )
            except Exception as exc:
                logger.warning(f"torch.compile failed for action_model, continuing without it: {exc}")

        if trainer_cfg.get("compile_vj_predictor", False):
            try:
                vj_dynamic = _resolve_compile_dynamic(
                    trainer_cfg,
                    "compile_vj_predictor_dynamic",
                    compile_dynamic,
                )
                _install_compiled_forward(
                    model.vj_predictor,
                    "vj_predictor",
                    compile_mode=compile_mode,
                    dynamic_attempts=[vj_dynamic] if not vj_dynamic else [True, False],
                )
            except Exception as exc:
                logger.warning(f"torch.compile failed for vj_predictor, continuing without it: {exc}")

        if trainer_cfg.get("compile_vj_encoder", False):
            try:
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
                    dynamic_attempts=vj_encoder_attempts,
                )
                if hasattr(model.vj_encoder, "get_vision_features"):
                    _install_compiled_callable_attr(
                        model.vj_encoder,
                        "get_vision_features",
                        "vj_encoder.get_vision_features",
                        compile_mode=compile_mode,
                        dynamic_attempts=vj_encoder_attempts,
                    )
            except Exception as exc:
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
                    dynamic_attempts=[full_model_dynamic] if not full_model_dynamic else [True, False],
                )
            except Exception as exc:
                logger.warning(f"torch.compile failed for full model forward, continuing without it: {exc}")

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir, model=None) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    dataset_py = cfg.datasets.vla_data.dataset_py
    if "data_mix" in cfg.datasets.vla_data:
        logger.info(
            f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}` "
            f"via `{dataset_py}`"
        )
    else:
        logger.info(
            f"Creating VLA Dataset from `{cfg.datasets.vla_data.data_root_dir}` "
            f"via `{dataset_py}`"
        )
    vla_train_dataloader = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.vla_data.dataset_py,
        model=model,
    )

    accelerator.dataloader_config.dispatch_batches = False

    return vla_train_dataloader


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


def resolve_training_schedule(cfg, vla_train_dataloader, num_processes: int = 1) -> None:
    """Resolve epoch-based training schedule once the dataloader length is known."""
    trainer_cfg = cfg.trainer
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


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.action_loss_scale, self.wm_loss_scale = self._resolve_loss_scales()
        self.best_metric_name = str(self.config.trainer.get("best_metric_name", "mae_score"))
        self.best_metric_mode = str(self.config.trainer.get("best_metric_mode", "min")).lower()
        self.best_metric_value = None
        self._warned_missing_best_metric = False

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.train_start_time = time.perf_counter()
        self.progress_eta_window = max(int(self.config.trainer.get("progress_eta_window", 50)), 1)
        self.progress_eta_warmup_steps = max(int(self.config.trainer.get("progress_eta_warmup_steps", 3)), 0)
        self._recent_wall_step_times = deque(maxlen=self.progress_eta_window)

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

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

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

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            # self.vlm_train_dataloader
        )
        self._validate_prepared_dataloader()

        #self._init_wandb()
        self._init_checkpointing()

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
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        is_resume = getattr(self.config.trainer, "is_resume", False)
        resume_from_checkpoint = (
            getattr(self.config, "resume_from_checkpoint", None)
            or getattr(self.config.trainer, "resume_from_checkpoint", None)
        )

        # resume training state
        if is_resume and resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
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
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        os.makedirs(checkpoint_path, exist_ok=True)
        self.accelerator.save_state(checkpoint_path)
        if self.accelerator.is_main_process:
            # save plain model weights alongside the full accelerator state for convenience
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
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        distributed_wait(self.accelerator)

    def _should_save_checkpoint(self, step_metrics: dict) -> bool:
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

                if self.accelerator.trackers:
                    self.accelerator.log(scalar_metrics, step=self.completed_steps)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                logger.info(f"Step {self.completed_steps}, Metrics: {scalar_metrics}")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vla_eval_iter = iter(self.vla_train_dataloader)
        self.vla_epoch_count = 0
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def _get_next_eval_batch(self):
        """get next evaluation batch without consuming the training iterator"""
        try:
            batch_vla = next(self.vla_eval_iter)
        except StopIteration:
            # Eval shares the loader but must not mutate the training sampler epoch.
            self.vla_eval_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_eval_iter)

        return batch_vla

    def _resolve_loss_scales(self):
        loss_scale_cfg = self.config.trainer.get("loss_scale", {})
        action_scale = float(loss_scale_cfg.get("action", loss_scale_cfg.get("vla", 1.0)))
        wm_scale = float(loss_scale_cfg.get("wm", loss_scale_cfg.get("vlm", 0.1)))
        return action_scale, wm_scale

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()
        self.optimizer.zero_grad(set_to_none=True)
        if bool(self.config.trainer.get("eval_before_train", False)):
            baseline_metrics = self.eval_action_model({})
            if self.accelerator.is_main_process:
                baseline_metrics["epoch"] = 0.0
                baseline_metrics["samples_seen"] = 0.0
                if self.accelerator.trackers:
                    self.accelerator.log(baseline_metrics, step=0)
                logger.info(f"Step 0 Eval Metrics: {baseline_metrics}")

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
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            # evaluate model
            if self.completed_steps > 0 and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                if self._should_save_checkpoint(step_metrics):
                    self._save_checkpoint()

            step_metrics["wall_step_time"] = time.perf_counter() - t_start_step
            if self.accelerator.sync_gradients and self.completed_steps > self.progress_eta_warmup_steps:
                self._recent_wall_step_times.append(step_metrics["wall_step_time"])

            if self.accelerator.is_local_main_process:
                eta_seconds = self._estimate_remaining_seconds()
                avg_wall_step_time = (
                    sum(self._recent_wall_step_times) / len(self._recent_wall_step_times)
                    if self._recent_wall_step_times
                    else None
                )
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                        "wall_time": f"{step_metrics['wall_step_time']:.3f}",
                        "avg_wall": f"{avg_wall_step_time:.3f}" if avg_wall_step_time is not None else "warmup",
                        "eta": self._format_duration(eta_seconds),
                    }
                )

            self._log_metrics(step_metrics)

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        step_metrics = step_metrics or {}

        examples = self._get_next_eval_batch()
        infer_model = self.accelerator.unwrap_model(self.model)
        if isinstance(examples, dict):
            actions = examples["action"].cpu().numpy()
            with torch.no_grad():
                output_dict = infer_model.predict_action(
                    batch=examples,
                    use_ddim=True,
                    num_ddim_steps=20,
                )
        else:
            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]
            actions = [example["action"] for example in examples]
            state = [example["state"] for example in examples] if "state" in examples[0] else None
            with torch.no_grad():
                output_dict = infer_model.predict_action(
                    batch_images=batch_images,
                    instructions=instructions,
                    state=state,
                    use_ddim=True,
                    num_ddim_steps=20,
                )

        actions = np.asarray(actions, dtype=np.float32)
        normalized_actions = np.asarray(output_dict["normalized_actions"], dtype=np.float32)
        diff = normalized_actions - actions
        local_metrics = torch.tensor(
            [
                float(np.abs(diff).sum()),
                float(np.square(diff).sum()),
                float(diff.size),
            ],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        reduced_metrics = self.accelerator.reduce(local_metrics, reduction="sum")

        total_abs_sum, total_sq_sum, total_count = reduced_metrics.tolist()
        if total_count > 0:
            step_metrics["mae_score"] = total_abs_sum / total_count
            step_metrics["norm_l2_per_element"] = (total_sq_sum ** 0.5) / total_count
        return step_metrics

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

        if isinstance(batch_vla, dict):
            raw_deltas = batch_vla.get(progress_key, batch_vla.get(fallback_progress_key))
            if raw_deltas is None:
                return None, {}
            deltas = raw_deltas.to(self.accelerator.device, dtype=torch.float32, non_blocking=True)
            valid_mask = torch.isfinite(deltas)
            deltas = torch.where(valid_mask, deltas, torch.zeros_like(deltas))

            current_mistake = batch_vla.get("mistake_label")
            future_mistake = batch_vla.get("future_mistake_label")
            if current_mistake is None and future_mistake is None:
                mistake_mask = torch.zeros_like(deltas, dtype=torch.bool)
            else:
                current_mistake = (
                    current_mistake.to(self.accelerator.device, dtype=torch.float32, non_blocking=True)
                    if current_mistake is not None
                    else torch.zeros_like(deltas)
                )
                future_mistake = (
                    future_mistake.to(self.accelerator.device, dtype=torch.float32, non_blocking=True)
                    if future_mistake is not None
                    else torch.zeros_like(deltas)
                )
                # Training datasets standardize mistake_label so positive values mean "is a mistake",
                # even if the raw source uses the inverse convention.
                mistake_mask = torch.maximum(current_mistake, future_mistake) > 0.5
        else:
            deltas = []
            valid_mask = []
            mistake_mask = []
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
                    valid_mask.append(False)
                else:
                    delta = float(delta)
                    is_valid = not np.isnan(delta)
                    deltas.append(0.0 if not is_valid else delta)
                    valid_mask.append(is_valid)

                has_mistake = False
                for key in mistake_keys:
                    if key in example or f"future_{key}" in example:
                        current_val = float(example.get(key, 0.0))
                        future_val = float(example.get(f"future_{key}", 0.0))
                        has_mistake = max(current_val, future_val) > 0.5
                        break
                mistake_mask.append(has_mistake)

            deltas = torch.tensor(deltas, device=self.accelerator.device, dtype=torch.float32)
            valid_mask = torch.tensor(valid_mask, device=self.accelerator.device, dtype=torch.bool)
            mistake_mask = torch.tensor(mistake_mask, device=self.accelerator.device, dtype=torch.bool)

        if not valid_mask.any():
            return None, {}

        valid_deltas = deltas[valid_mask]
        delta_mean = torch.clamp(valid_deltas.mean(), min=0.0)
        delta_std = torch.clamp(valid_deltas.std(unbiased=False), min=epsilon)
        lower_bound = delta_mean - 2 * delta_std
        soft_weights = torch.clamp((deltas - lower_bound) / (4 * delta_std + epsilon), 0.0, 1.0)

        weights = torch.zeros_like(deltas)
        weights = torch.where(deltas > kappa, torch.ones_like(weights), weights)
        moderate_mask = (deltas >= 0.0) & (deltas <= kappa)
        weights = torch.where(moderate_mask, soft_weights, weights)
        weights = torch.where(valid_mask, weights, torch.ones_like(weights))
        if mistake_mask.any():
            weights = torch.where(mistake_mask, torch.full_like(weights, mistake_penalty), weights)

        weights = weights * weights.shape[0] / (weights.sum() + epsilon)
        stats = {
            "rabc_mean_weight": weights.mean().detach(),
            "rabc_valid_ratio": valid_mask.float().mean().detach(),
            "rabc_mean_delta": valid_deltas.mean().detach(),
            "rabc_zero_weight_ratio": (weights <= epsilon).float().mean().detach(),
        }
        if mistake_mask.any():
            stats["rabc_mistake_ratio"] = mistake_mask.float().mean().detach()
        return weights, stats

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            rabc_weights, rabc_stats = self._compute_rabc_weights(batch_vla)
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla, rabc_weights=rabc_weights)
                total_loss = (
                    output_dict.get("action_loss", 0.0) * self.action_loss_scale
                    + output_dict.get("wm_loss", 0.0) * self.wm_loss_scale
                )

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # gradient clipping
            grad_norm = None
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )

            # optimizer step
            if self.accelerator.sync_gradients:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            result_dict = {}
            for key, value in output_dict.items():
                result_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
            result_dict["total_loss"] = total_loss.detach() if isinstance(total_loss, torch.Tensor) else total_loss
            for key, value in rabc_stats.items():
                result_dict[key] = value.detach() if isinstance(value, torch.Tensor) else value
            if grad_norm is not None:
                result_dict["grad_norm"] = grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else grad_norm

        return result_dict

    def _finalize_training(self):
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
        self._shutdown_data_runtime()
        distributed_wait(self.accelerator)
        finish_trackers(self.accelerator)

    def _shutdown_data_runtime(self):
        for attr_name in ("vla_iter", "vla_eval_iter"):
            iterator = getattr(self, attr_name, None)
            _shutdown_dataloader_iterator(iterator)
            setattr(self, attr_name, None)
        _shutdown_dataloader_workers(getattr(self, "vla_train_dataloader", None))
        gc.collect()


def main(cfg) -> None:
    accelerator = build_accelerator(cfg)
    logger.info("VLA Training :: Warming Up")
    interrupted = False
    trainer = None

    try:
        # create output directory and save config
        output_dir = setup_directories(cfg=cfg)
        # build model
        vla = build_model(cfg)
        # prepare data
        vla_train_dataloader = prepare_data(
            cfg=cfg,
            accelerator=accelerator,
            output_dir=output_dir,
            model=vla,
        )
        resolve_training_schedule(
            cfg=cfg,
            vla_train_dataloader=vla_train_dataloader,
            num_processes=accelerator.num_processes,
        )

        # set optimizer and scheduler
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

        trainer = VLATrainer(
            cfg=cfg,
            model=vla,
            vla_train_dataloader=vla_train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
        )

        # execute training preparation
        trainer.prepare_training()
        # execute training
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
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
