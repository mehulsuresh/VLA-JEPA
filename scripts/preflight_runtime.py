#!/usr/bin/env python3
"""Runtime preflight checks for VLA-JEPA cloud GPU containers."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path
from typing import Any


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _ok(message: str) -> None:
    print(f"[ok] {message}", flush=True)


def _warn(message: str) -> None:
    print(f"[warn] {message}", flush=True)


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _import(name: str, *, required: bool = True):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        if required:
            _fail(f"Could not import {name}: {exc}")
        _warn(f"Could not import {name}: {exc}")
        return None
    _ok(f"import {name} ({_version(module)})")
    return module


def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    cur = cfg
    for part in dotted.split("."):
        if cur is None:
            return default
        if hasattr(cur, "get"):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _load_config(path: str | None) -> Any:
    if not path:
        return None
    omega = _import("omegaconf")
    cfg_path = Path(path)
    if not cfg_path.exists():
        _fail(f"Config does not exist: {cfg_path}")
    cfg = omega.OmegaConf.load(cfg_path)
    _ok(f"loaded config {cfg_path}")
    return cfg


def _check_torch(*, require_cuda: bool) -> None:
    torch = _import("torch")
    print(f"python={platform.python_version()} executable={sys.executable}", flush=True)
    print(f"torch={torch.__version__} cuda_runtime={getattr(torch.version, 'cuda', None)}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}", flush=True)
    if require_cuda and not torch.cuda.is_available():
        _fail("CUDA is required but torch.cuda.is_available() is false")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            print(
                f"gpu[{index}]={torch.cuda.get_device_name(index)} capability=sm{capability[0]}{capability[1]}",
                flush=True,
            )
        _ok(f"torch CUDA arch list: {torch.cuda.get_arch_list()}")
    if not torch.distributed.is_available():
        _warn("torch.distributed is not available")
    elif not torch.distributed.is_nccl_available():
        _warn("torch.distributed NCCL backend is not available")
    else:
        _ok("torch.distributed NCCL backend available")

    try:
        from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401
    except Exception as exc:
        _fail(f"FlexAttention is not available: {exc}")
    _ok("FlexAttention import")


def _check_config_runtime(cfg: Any) -> tuple[str | None, bool, bool]:
    if cfg is None:
        return None, False, False

    base_vlm = _cfg_get(cfg, "framework.qwenvl.base_vlm")
    attn_impl = str(_cfg_get(cfg, "framework.qwenvl.attn_implementation", "")).lower()
    blockwise_enabled = bool(_cfg_get(cfg, "framework.qwenvl.blockwise_attention.enabled", False))
    depth_enabled = bool(_cfg_get(cfg, "framework.depth_teacher_aux.enabled", False))
    dataset_py = _cfg_get(cfg, "datasets.vla_data.dataset_py")
    video_backend = _cfg_get(cfg, "datasets.vla_data.video_backend")

    print(
        f"config base_vlm={base_vlm} attn_implementation={attn_impl} "
        f"blockwise={blockwise_enabled} depth_teacher={depth_enabled}",
        flush=True,
    )

    if blockwise_enabled and attn_impl not in {"flex_attention", "flex", "flex-attn"}:
        _fail("blockwise_attention.enabled=true requires attn_implementation=flex_attention")

    if dataset_py == "lerobot_datasets":
        if not video_backend:
            _fail("lerobot_datasets config must set datasets.vla_data.video_backend explicitly")
        if str(video_backend).lower() == "pyav":
            _import("av")
        elif str(video_backend).lower() == "decord":
            _import("decord")
        else:
            _warn(f"lerobot_datasets uses video_backend={video_backend}; preflight does not validate it")
    else:
        _import("av")

    return base_vlm, blockwise_enabled, depth_enabled


def _check_qwen(base_vlm: str | None, *, local_files_only: bool, load_weights: bool) -> None:
    if not base_vlm:
        return
    transformers = _import("transformers")
    config = transformers.AutoConfig.from_pretrained(
        base_vlm,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(text_config, "hidden_size", None)
    _ok(f"Qwen config {base_vlm} hidden_size={hidden_size}")
    if load_weights:
        model_cls = getattr(transformers, "AutoModelForCausalLM", None)
        if "Qwen3-VL" in base_vlm:
            from transformers import Qwen3VLForConditionalGeneration as model_cls
        model = model_cls.from_pretrained(
            base_vlm,
            trust_remote_code=True,
            local_files_only=local_files_only,
            device_map="cpu",
        )
        _ok(f"loaded Qwen weights on CPU: {type(model).__name__}")


def _check_moge(cfg: Any, *, required: bool) -> None:
    if cfg is not None:
        repo_path = _cfg_get(cfg, "framework.depth_teacher_aux.moge_repo_path")
        if repo_path:
            repo_path = str(repo_path)
            if Path(repo_path).exists() and repo_path not in sys.path:
                sys.path.insert(0, repo_path)
                _ok(f"added MoGe repo path to sys.path: {repo_path}")
    _import("utils3d", required=required)
    _import("pipeline", required=required)
    _import("moge.model.v2", required=required)


def _check_deepspeed(required: bool) -> None:
    module = _import("deepspeed", required=required)
    if module is not None:
        _ok("DeepSpeed import")
    _import("mpi4py", required=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-yaml")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-deepspeed", action="store_true")
    parser.add_argument("--require-moge", action="store_true")
    parser.add_argument("--load-qwen-config", action="store_true", default=True)
    parser.add_argument("--load-qwen-weights", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    _check_torch(require_cuda=args.require_cuda)
    _import("av")
    _import("transformers")
    _import("accelerate")
    _import("peft")
    _import("qwen_vl_utils")

    cfg = _load_config(args.config_yaml)
    base_vlm, _blockwise_enabled, depth_enabled = _check_config_runtime(cfg)
    if args.load_qwen_config:
        _check_qwen(base_vlm, local_files_only=args.local_files_only, load_weights=args.load_qwen_weights)
    if args.require_deepspeed:
        _check_deepspeed(required=True)
    elif os.environ.get("STARVLA_USE_DEEPSPEED", "0") == "1":
        _check_deepspeed(required=True)
    else:
        _check_deepspeed(required=False)
    if args.require_moge or depth_enabled:
        _check_moge(cfg, required=True)
    else:
        _check_moge(cfg, required=False)

    _ok("preflight complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
