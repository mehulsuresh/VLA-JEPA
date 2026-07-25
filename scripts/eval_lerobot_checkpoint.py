#!/usr/bin/env python3
"""Evaluate a VLA-JEPA checkpoint on a single local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from tqdm.auto import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _dataset_report(dataset_path: Path) -> dict[str, Any]:
    meta_dir = dataset_path / "meta"
    info = _load_json(meta_dir / "info.json")
    video_keys = sorted(
        key
        for key, value in info.get("features", {}).items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    )
    parquet_files = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    video_files = sorted((dataset_path / "videos").glob("chunk-*/*/*.mp4"))
    missing_video_dirs = [
        key
        for key in video_keys
        if not (dataset_path / "videos" / "chunk-000" / key).exists()
    ]
    return {
        "path": str(dataset_path),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes_meta": info.get("total_episodes"),
        "total_frames_meta": info.get("total_frames"),
        "total_videos_meta": info.get("total_videos"),
        "split": info.get("splits", {}),
        "tasks_jsonl": _count_jsonl(meta_dir / "tasks.jsonl"),
        "episodes_jsonl": _count_jsonl(meta_dir / "episodes.jsonl"),
        "episodes_stats_jsonl": _count_jsonl(meta_dir / "episodes_stats.jsonl"),
        "parquet_files": len(parquet_files),
        "video_files": len(video_files),
        "video_keys": video_keys,
        "missing_video_dirs": missing_video_dirs,
    }


def _generated_modality_meta(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features", {})
    state_shape = features.get("observation.state", {}).get("shape", [14])
    action_shape = features.get("action", {}).get("shape", [14])
    state_dim = int(state_shape[0])
    action_dim = int(action_shape[0])
    action_dtype = features.get("action", {}).get("dtype", "float32")
    state_dtype = features.get("observation.state", {}).get("dtype", "float32")

    video_candidates = [
        ("base_view", "observation.images.cam_high"),
        ("left_wrist", "observation.images.cam_left_wrist"),
        ("right_wrist", "observation.images.cam_right_wrist"),
        ("main", "observation.images.main"),
        ("left", "observation.images.left"),
        ("right", "observation.images.right"),
        ("extra", "observation.images.extra"),
    ]
    video_meta = {
        alias: {"original_key": original_key}
        for alias, original_key in video_candidates
        if original_key in features
    }
    if not video_meta:
        video_meta = {
            key.rsplit(".", 1)[-1]: {"original_key": key}
            for key, value in features.items()
            if isinstance(value, dict) and value.get("dtype") == "video"
        }

    return {
        "state": {
            "joints": {
                "start": 0,
                "end": state_dim,
                "original_key": "observation.state",
                "absolute": True,
                "dtype": state_dtype,
            }
        },
        "action": {
            "delta_joints": {
                "start": 0,
                "end": action_dim,
                "original_key": "action",
                "absolute": False,
                "dtype": action_dtype,
            }
        },
        "video": video_meta,
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }


def _prepare_dataset_overlay(source_dataset_path: Path, output_json: Path) -> Path:
    overlay_path = output_json.parent / "dataset_overlay" / source_dataset_path.name
    overlay_meta = overlay_path / "meta"
    overlay_path.mkdir(parents=True, exist_ok=True)
    overlay_meta.mkdir(parents=True, exist_ok=True)

    for dirname in ("data", "videos"):
        src = source_dataset_path / dirname
        dst = overlay_path / dirname
        if dst.is_symlink():
            if dst.resolve() == src.resolve():
                continue
            dst.unlink()
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)

    source_meta = source_dataset_path / "meta"
    for src_file in source_meta.iterdir():
        dst_file = overlay_meta / src_file.name
        if src_file.is_file():
            shutil.copy2(src_file, dst_file)
        elif src_file.is_dir():
            if dst_file.exists():
                shutil.rmtree(dst_file)
            shutil.copytree(src_file, dst_file)

    modality_path = overlay_meta / "modality.json"
    if not modality_path.exists():
        info = _load_json(source_meta / "info.json")
        with modality_path.open("w", encoding="utf-8") as f:
            json.dump(_generated_modality_meta(info), f, indent=4)

    return overlay_path


def _patch_single_dataset_mix(dataset_path: Path, mix_name: str, robot_type: str, lerobot_version: str) -> None:
    dataset_name = dataset_path.name
    mixture = [(dataset_name, 1.0, robot_type, lerobot_version)]

    from starVLA.dataloader.gr00t_lerobot import mixtures as mixture_mod
    import starVLA.dataloader.lerobot_datasets as lerobot_datasets

    mixture_mod.DATASET_NAMED_MIXTURES[mix_name] = mixture
    lerobot_datasets.DATASET_NAMED_MIXTURES[mix_name] = mixture


def _set_if_present(cfg: Any, dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(node, part):
            return
        node = getattr(node, part)
    if hasattr(node, parts[-1]):
        setattr(node, parts[-1], value)


def _prepare_cfg(args: argparse.Namespace, dataset_report: dict[str, Any]) -> Any:
    cfg = OmegaConf.load(args.config_yaml)

    loader_dataset_path = getattr(args, "loader_dataset_path", args.dataset_path)
    cfg.datasets.vla_data.data_root_dir = str(loader_dataset_path.parent)
    cfg.datasets.vla_data.data_mix = args.mix_name
    cfg.datasets.vla_data.shuffle_buffer_size = 0
    cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    cfg.datasets.vla_data.num_workers = args.num_workers
    cfg.datasets.vla_data.pin_memory = False
    cfg.datasets.vla_data.drop_last = False
    cfg.datasets.vla_data.persistent_workers = args.num_workers > 0
    if hasattr(cfg.datasets.vla_data, "prefetch_factor"):
        cfg.datasets.vla_data.prefetch_factor = args.prefetch_factor
    if hasattr(cfg.datasets.vla_data, "shuffle"):
        cfg.datasets.vla_data.shuffle = False

    if hasattr(cfg.datasets.vla_data, "state_history_len"):
        cfg.datasets.vla_data.state_history_len = max(1, int(cfg.datasets.vla_data.state_history_len))

    cfg.trainer.output_dir = str(args.output_dir)
    if hasattr(cfg, "output_dir"):
        cfg.output_dir = str(args.output_dir)
    cfg.trainer.is_resume = False
    cfg.trainer.resume_load_optimizer_state = False
    cfg.trainer.save_final_model = False
    cfg.trainer.eval_before_train = False
    cfg.trainer.enable_mixed_precision_training = False
    _set_if_present(cfg, "trainer.compile_full_model", False)
    _set_if_present(cfg, "trainer.compile_model", False)
    _set_if_present(cfg, "trainer.torch_compile", False)
    _set_if_present(cfg, "trainer.tracker", "none")

    if args.attn_implementation != "config" and hasattr(cfg.framework, "qwenvl"):
        cfg.framework.qwenvl.attn_implementation = args.attn_implementation
        cfg.framework.qwenvl.strict_attn_implementation = False
    if args.disable_fast_linear_attention and hasattr(cfg.framework, "qwenvl"):
        cfg.framework.qwenvl.enable_fast_linear_attention = False
        cfg.framework.qwenvl.strict_fast_linear_attention = False
    if args.disable_blockwise_attention and hasattr(cfg.framework, "qwenvl"):
        if hasattr(cfg.framework.qwenvl, "blockwise_attention"):
            cfg.framework.qwenvl.blockwise_attention.enabled = False

    return cfg


def _checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    safetensors_path = path / "model.safetensors"
    pytorch_path = path / "pytorch_model.pt"
    if safetensors_path.exists():
        return safetensors_path
    if pytorch_path.exists():
        return pytorch_path
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.pt found under {path}")


def _load_checkpoint_into_model(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    ckpt_file = _checkpoint_file(checkpoint)
    if ckpt_file.suffix == ".safetensors":
        state = load_safetensors(str(ckpt_file))
    else:
        state = torch.load(str(ckpt_file), map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    load_result = model.load_state_dict(state, strict=False)
    return {
        "checkpoint_file": str(ckpt_file),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_targets_and_mask(batch: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(batch, dict):
        targets = _to_numpy(batch["action"])
        mask = _to_numpy(batch["action_mask"]) if "action_mask" in batch else None
        if "action_is_pad" in batch:
            pad_keep = (~_to_numpy(batch["action_is_pad"]).astype(bool)).astype(np.float32)
            if pad_keep.ndim == 2:
                pad_keep = pad_keep[..., None]
            mask = pad_keep if mask is None else _to_numpy(mask).astype(np.float32) * pad_keep
        return targets, mask

    targets = np.stack([_to_numpy(sample["action"]) for sample in batch], axis=0)
    mask = None
    if batch and isinstance(batch[0], dict) and "action_mask" in batch[0]:
        mask = np.stack([_to_numpy(sample["action_mask"]) for sample in batch], axis=0)
    if batch and isinstance(batch[0], dict) and "action_is_pad" in batch[0]:
        pad_keep = ~np.stack([_to_numpy(sample["action_is_pad"]).astype(bool) for sample in batch], axis=0)
        pad_keep = pad_keep.astype(np.float32)
        if pad_keep.ndim == 2:
            pad_keep = pad_keep[..., None]
        mask = pad_keep if mask is None else mask.astype(np.float32) * pad_keep
    return targets, mask


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        return int(batch["action"].shape[0])
    return len(batch)


def _predict(model: torch.nn.Module, batch: Any, num_ddim_steps: int) -> np.ndarray:
    kwargs = {"batch": batch, "use_ddim": True, "num_ddim_steps": num_ddim_steps}
    if not isinstance(batch, dict):
        states = [sample.get("state") for sample in batch if isinstance(sample, dict)]
        if states and len(states) == len(batch):
            kwargs["state"] = states
    output = model.predict_action(**kwargs)
    if isinstance(output, dict):
        output = output.get("normalized_actions", output.get("actions"))
    return _to_numpy(output).astype(np.float32)


def _validate_and_eval(args: argparse.Namespace) -> dict[str, Any]:
    source_dataset_path = args.dataset_path
    dataset_report = _dataset_report(source_dataset_path)
    loader_dataset_path = _prepare_dataset_overlay(source_dataset_path, args.output_json)
    args.loader_dataset_path = loader_dataset_path
    dataset_report["loader_overlay_path"] = str(loader_dataset_path)

    robot_type = args.robot_type or str(dataset_report["robot_type"])
    lerobot_version = args.lerobot_version or str(dataset_report["codebase_version"])
    if lerobot_version == "v2.1":
        lerobot_version = "v2.0"
    _patch_single_dataset_mix(loader_dataset_path, args.mix_name, robot_type, lerobot_version)
    cfg = _prepare_cfg(args, dataset_report)

    from accelerate import PartialState
    from starVLA.dataloader import build_dataloader
    from starVLA.training.train_starvla import build_model

    PartialState()
    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = build_model(cfg)
    load_report = _load_checkpoint_into_model(model, args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)
    model.eval()

    dataloader = build_dataloader(cfg, dataset_py=args.dataset_py, model=model)

    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    abs_by_horizon: np.ndarray | None = None
    count_by_horizon: np.ndarray | None = None
    abs_by_dim: np.ndarray | None = None
    count_by_dim: np.ndarray | None = None
    samples = 0
    first_batch_shapes: dict[str, Any] = {}
    errors: list[str] = []

    iterator = iter(dataloader)
    total = args.max_batches if args.max_batches > 0 else None
    with torch.no_grad():
        for batch_idx in tqdm(range(total) if total else iterator, total=total, desc="eval", leave=False):
            if total:
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
            else:
                batch = batch_idx
            try:
                targets, mask = _extract_targets_and_mask(batch)
                preds = _predict(model, batch, args.num_ddim_steps)
                if preds.shape != targets.shape:
                    raise ValueError(f"prediction shape {preds.shape} != target shape {targets.shape}")

                if mask is not None:
                    if mask.shape != targets.shape:
                        mask = np.broadcast_to(mask, targets.shape)
                    mask = mask.astype(bool)
                else:
                    mask = np.ones_like(targets, dtype=bool)

                diff = (preds - targets)[mask]
                abs_sum += float(np.abs(diff).sum())
                sq_sum += float(np.square(diff).sum())
                count += int(diff.size)
                samples += _batch_size(batch)

                abs_full = np.abs(preds - targets)
                mask_full = mask.astype(bool)
                if abs_by_horizon is None:
                    abs_by_horizon = np.zeros(abs_full.shape[1], dtype=np.float64)
                    count_by_horizon = np.zeros(abs_full.shape[1], dtype=np.float64)
                    abs_by_dim = np.zeros(abs_full.shape[2], dtype=np.float64)
                    count_by_dim = np.zeros(abs_full.shape[2], dtype=np.float64)
                abs_by_horizon += (abs_full * mask_full).sum(axis=(0, 2))
                count_by_horizon += mask_full.sum(axis=(0, 2))
                abs_by_dim += (abs_full * mask_full).sum(axis=(0, 1))
                count_by_dim += mask_full.sum(axis=(0, 1))

                if not first_batch_shapes:
                    first_batch_shapes = {
                        "target": list(targets.shape),
                        "prediction": list(preds.shape),
                        "mask": list(mask.shape),
                    }
            except Exception as exc:  # Keep decode/model errors visible in the JSON.
                errors.append(f"batch {len(errors)}: {type(exc).__name__}: {exc}")
                if len(errors) >= args.max_errors:
                    break
                continue

    mae = abs_sum / count if count else math.nan
    rmse = math.sqrt(sq_sum / count) if count else math.nan
    norm_l2_per_element = math.sqrt(sq_sum) / count if count else math.nan
    horizon_mae = None
    dim_mae = None
    if abs_by_horizon is not None and count_by_horizon is not None:
        horizon_mae = np.divide(
            abs_by_horizon,
            count_by_horizon,
            out=np.full_like(abs_by_horizon, np.nan, dtype=np.float64),
            where=count_by_horizon > 0,
        ).tolist()
    if abs_by_dim is not None and count_by_dim is not None:
        dim_mae = np.divide(
            abs_by_dim,
            count_by_dim,
            out=np.full_like(abs_by_dim, np.nan, dtype=np.float64),
            where=count_by_dim > 0,
        ).tolist()
    return {
        "config_yaml": str(args.config_yaml),
        "checkpoint": str(args.checkpoint),
        "dataset": dataset_report,
        "model_load": {
            **load_report,
            "missing_key_count": len(load_report["missing_keys"]),
            "unexpected_key_count": len(load_report["unexpected_keys"]),
            "missing_keys_preview": load_report["missing_keys"][:25],
            "unexpected_keys_preview": load_report["unexpected_keys"][:25],
        },
        "eval": {
            "batches_requested": args.max_batches,
            "samples": samples,
            "elements": count,
            "batch_size": args.batch_size,
            "num_ddim_steps": args.num_ddim_steps,
            "mae_score": mae,
            "rmse": rmse,
            "norm_l2_per_element": norm_l2_per_element,
            "mae_by_horizon": horizon_mae,
            "mae_by_action_dim": dim_mae,
            "first_batch_shapes": first_batch_shapes,
            "errors": errors,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-yaml", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=_repo_root() / "artifacts" / "eval_tmp")
    parser.add_argument("--mix-name", default="__single_eval_dataset__")
    parser.add_argument("--robot-type", default=None)
    parser.add_argument("--lerobot-version", default=None)
    parser.add_argument("--dataset-py", default="lerobot_datasets")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--num-ddim-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["config", "sdpa", "eager", "flash_attention_2", "flex_attention"])
    parser.add_argument("--disable-fast-linear-attention", action="store_true")
    parser.add_argument("--disable-blockwise-attention", action="store_true")
    parser.add_argument("--max-errors", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.resolve()
    args.config_yaml = args.config_yaml.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_json = args.output_json.resolve()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = _validate_and_eval(args)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
