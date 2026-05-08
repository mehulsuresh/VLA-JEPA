#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
from functools import partial

import torch
from omegaconf import OmegaConf

from starVLA.dataloader import _configure_lerobot_worker
from starVLA.dataloader.canonical_subset_dataset import collate_fn, get_vla_dataset


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _maybe_repo_relative(path_value: str | None, repo_root: Path) -> str | None:
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return path.as_posix()
    return (repo_root / path).as_posix()


def _configure_cfg(args: argparse.Namespace) -> Any:
    repo_root = _resolve_repo_root()
    cfg = OmegaConf.load(args.config_yaml)
    data_cfg = cfg.datasets.vla_data
    data_cfg.dataset_ids = [args.dataset_id] if args.dataset_id else []
    data_cfg.exclude_dataset_ids = []
    data_cfg.exclude_sids = []
    data_cfg.exclude_dataset_ids_path = []
    data_cfg.exclude_sids_path = []
    data_cfg.max_shards = args.max_shards
    data_cfg.max_windows = args.max_windows
    data_cfg.max_shards_per_dataset = 0
    data_cfg.max_windows_per_dataset = 0
    data_cfg.shuffle_shards = args.shuffle_shards
    data_cfg.shuffle = False
    data_cfg.shuffle_seed = args.shuffle_seed
    data_cfg.prefetch_metadata_across_ranks = False
    data_cfg.metadata_index_cache = False
    data_cfg.lazy_cache_shards = True
    data_cfg.index_windows_lazily = True
    data_cfg.video_decode_backend = args.video_decode_backend
    data_cfg.reader_cache_size = args.reader_cache_size
    data_cfg.sidecar_cache_size = args.sidecar_cache_size
    data_cfg.slow_sample_log_seconds = args.slow_sample_log_seconds
    data_cfg.manifest_path = _maybe_repo_relative(str(data_cfg.manifest_path), repo_root)
    data_cfg.adapter_dir = _maybe_repo_relative(str(data_cfg.adapter_dir), repo_root)
    data_cfg.dataset_canonicalization_root = _maybe_repo_relative(
        str(data_cfg.dataset_canonicalization_root), repo_root
    )
    data_cfg.cache_dir = _maybe_repo_relative(str(data_cfg.cache_dir), repo_root)
    return cfg


def _summarize_shards(dataset: Any, limit: int) -> None:
    print(
        json.dumps(
            {
                "num_windows": len(dataset),
                "num_shards": len(dataset.shards),
                "exclude_dataset_ids": dataset.exclude_dataset_id_list,
                "reader_cache_size": dataset.reader_cache_size,
                "slow_sample_log_seconds": dataset.slow_sample_log_seconds,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    shard_window_starts: dict[int, int] = {}
    previous_end = 0
    for window_range in getattr(dataset, "_window_ranges", []):
        shard_window_starts.setdefault(window_range.shard_index, previous_end)
        previous_end = window_range.cumulative_end
    for shard_index, shard in enumerate(dataset.shards[:limit]):
        window_start = shard_window_starts.get(shard_index)
        window_end = None
        for range_index in range(len(getattr(dataset, "_window_ranges", [])) - 1, -1, -1):
            window_range = dataset._window_ranges[range_index]
            if window_range.shard_index == shard_index:
                window_end = window_range.cumulative_end
                break
        video_files = {
            slot: sorted(
                {
                    episode.video_paths[slot].relative_to(shard.root).as_posix()
                    for episode in shard.episodes
                    if slot in episode.video_paths
                }
            )[:6]
            for slot in dataset.camera_slots
        }
        print(
            json.dumps(
                {
                    "shard_index": shard_index,
                    "dataset_id": shard.dataset_id,
                    "sid": shard.sid,
                    "data_file": shard.data_relative_path,
                    "episodes": len(shard.episodes),
                    "fps": shard.fps,
                    "window_start": window_start,
                    "window_end": window_end,
                    "video_files": video_files,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _sample_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.indices:
        return [int(value) % dataset_len for value in args.indices.split(",") if value.strip()]
    count = max(args.samples, 0)
    return [(args.start_index + idx * args.sample_stride) % dataset_len for idx in range(count)]


def _run_single_process_probe(dataset: Any, indices: list[int]) -> None:
    print(f"single_process_probe samples={len(indices)}", flush=True)
    for probe_index, dataset_index in enumerate(indices):
        start = time.monotonic()
        sample = dataset[dataset_index]
        elapsed = time.monotonic() - start
        video = sample["video_compact"]
        print(
            json.dumps(
                {
                    "probe_index": probe_index,
                    "dataset_index": int(dataset_index),
                    "elapsed_sec": round(elapsed, 4),
                    "dataset_id": sample.get("dataset_id"),
                    "episode_index": int(sample.get("episode_index", -1)),
                    "frame_index": int(sample.get("frame_index", -1)),
                    "video_shape": list(video.shape),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _run_dataloader_probe(dataset: Any, args: argparse.Namespace) -> None:
    if args.loader_batches <= 0:
        return
    loader_dataset = dataset
    if args.loader_start_index > 0 or args.loader_sample_stride != 1:
        subset_len = args.loader_batches * args.batch_size
        subset_indices = [
            (args.loader_start_index + idx * args.loader_sample_stride) % len(dataset)
            for idx in range(subset_len)
        ]
        loader_dataset = torch.utils.data.Subset(dataset, subset_indices)
    print(
        "dataloader_probe "
        f"batches={args.loader_batches} batch_size={args.batch_size} workers={args.num_workers} "
        f"prefetch_factor={args.prefetch_factor} timeout={args.timeout_seconds} "
        f"pin_memory={args.pin_memory} persistent_workers={args.persistent_workers} "
        f"multiprocessing_context={args.multiprocessing_context} "
        f"loader_start_index={args.loader_start_index} loader_sample_stride={args.loader_sample_stride}",
        flush=True,
    )
    if hasattr(dataset, "_decord_readers"):
        dataset._decord_readers.clear()
    if hasattr(dataset, "_loaded_shards"):
        dataset._loaded_shards.clear()
    loader_kwargs = dict(
        dataset=loader_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["multiprocessing_context"] = args.multiprocessing_context
        if args.timeout_seconds > 0:
            loader_kwargs["timeout"] = args.timeout_seconds
        loader_kwargs["worker_init_fn"] = partial(
            _configure_lerobot_worker,
            torch_threads=1,
            cv2_threads=1,
        )
    loader = torch.utils.data.DataLoader(**loader_kwargs)
    iterator = iter(loader)
    for batch_index in range(args.loader_batches):
        start = time.monotonic()
        batch = next(iterator)
        batch_size = len(batch)
        elapsed = time.monotonic() - start
        dataset_ids = sorted({str(sample.get("dataset_id")) for sample in batch})
        frame_indices = [int(sample.get("frame_index", -1)) for sample in batch[: min(4, len(batch))]]
        print(
            json.dumps(
                {
                    "batch_index": batch_index,
                    "batch_size": batch_size,
                    "elapsed_fetch_sec": round(elapsed, 4),
                    "dataset_ids": dataset_ids,
                    "first_frame_indices": frame_indices,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe canonical dataset video decode behavior without GPUs.")
    parser.add_argument("--config-yaml", default="scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml")
    parser.add_argument("--dataset-id", default="BAAI-DataCube/AgiBotWorld-Beta_G1_task_372_Supermarket_packaging")
    parser.add_argument("--max-shards", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--shuffle-shards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--video-decode-backend", default="decord", choices=["auto", "decord", "pyav", "imageio"])
    parser.add_argument("--reader-cache-size", type=int, default=32)
    parser.add_argument("--sidecar-cache-size", type=int, default=8)
    parser.add_argument("--slow-sample-log-seconds", type=float, default=2.0)
    parser.add_argument("--show-shards", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--indices", default="")
    parser.add_argument("--loader-batches", type=int, default=4)
    parser.add_argument("--loader-start-index", type=int, default=0)
    parser.add_argument("--loader-sample-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=26)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multiprocessing-context", default="spawn")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    cfg = _configure_cfg(args)
    dataset = get_vla_dataset(
        cfg.datasets.vla_data,
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        video_frame_stride=cfg.datasets.vla_data.video_frame_stride,
    )
    _summarize_shards(dataset, args.show_shards)
    indices = _sample_indices(len(dataset), args)
    _run_single_process_probe(dataset, indices)
    _run_dataloader_probe(dataset, args)


if __name__ == "__main__":
    main()
