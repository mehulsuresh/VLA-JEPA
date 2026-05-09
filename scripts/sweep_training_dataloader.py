#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader


def _maybe_set(cfg: Any, key: str, value: Any) -> None:
    if value is not None and value != "":
        cfg[key] = value


def _process_thread_count(pid: int) -> int | None:
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except Exception:
        return None


def _iterator_worker_pids(iterator: Any) -> list[int]:
    workers = getattr(iterator, "_workers", None)
    if not workers:
        return []
    return [int(worker.pid) for worker in workers if getattr(worker, "pid", None) is not None]


def _batch_summary(batch: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_ids = sorted({str(sample.get("dataset_id")) for sample in batch})
    first_samples = []
    for sample in batch[: min(4, len(batch))]:
        first_samples.append(
            {
                "dataset_id": sample.get("dataset_id"),
                "episode_index": int(sample.get("episode_index", -1)),
                "frame_index": int(sample.get("frame_index", -1)),
            }
        )
    return {"dataset_ids": dataset_ids, "first_samples": first_samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterate the VLA training DataLoader without model forward/backward.")
    parser.add_argument("--config-yaml", default="scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml")
    parser.add_argument("--dataset-canonicalization-root", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--metadata-index-cache-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--metadata-prefetch-workers", type=int, default=None)
    parser.add_argument("--data-file-prefetch-shards", type=int, default=None)
    parser.add_argument("--pyav-thread-count", default=None)
    parser.add_argument("--pyav-reader-cache-size", type=int, default=None)
    parser.add_argument("--video-cache-max-gb", type=float, default=None)
    parser.add_argument("--video-cache-prune-interval-downloads", type=int, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enforce-worker-memory-budget", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--multiprocessing-context", default="")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--gcs-download-timeout-seconds", type=int, default=None)
    parser.add_argument("--gcs-download-retries", type=int, default=None)
    parser.add_argument("--gcs-download-retry-backoff-seconds", type=float, default=None)
    parser.add_argument("--progress-interval-batches", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means iterate until DataLoader exhaustion.")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    cfg = OmegaConf.load(args.config_yaml)
    data_cfg = cfg.datasets.vla_data
    _maybe_set(data_cfg, "dataset_canonicalization_root", args.dataset_canonicalization_root)
    _maybe_set(data_cfg, "manifest_path", args.manifest_path)
    _maybe_set(data_cfg, "adapter_dir", args.adapter_dir)
    _maybe_set(data_cfg, "cache_dir", args.cache_dir)
    _maybe_set(data_cfg, "metadata_index_cache_dir", args.metadata_index_cache_dir)
    _maybe_set(data_cfg, "num_workers", args.num_workers)
    _maybe_set(data_cfg, "per_device_batch_size", args.batch_size)
    _maybe_set(data_cfg, "metadata_prefetch_workers", args.metadata_prefetch_workers)
    _maybe_set(data_cfg, "data_file_prefetch_shards", args.data_file_prefetch_shards)
    _maybe_set(data_cfg, "pyav_thread_count", args.pyav_thread_count)
    _maybe_set(data_cfg, "pyav_reader_cache_size", args.pyav_reader_cache_size)
    _maybe_set(data_cfg, "video_cache_max_gb", args.video_cache_max_gb)
    _maybe_set(
        data_cfg,
        "video_cache_prune_interval_downloads",
        args.video_cache_prune_interval_downloads,
    )
    _maybe_set(data_cfg, "multiprocessing_context", args.multiprocessing_context)
    _maybe_set(data_cfg, "prefetch_factor", args.prefetch_factor)
    _maybe_set(data_cfg, "dataloader_timeout_seconds", args.timeout_seconds)
    _maybe_set(data_cfg, "gcs_download_timeout_seconds", args.gcs_download_timeout_seconds)
    _maybe_set(data_cfg, "gcs_download_retries", args.gcs_download_retries)
    _maybe_set(data_cfg, "gcs_download_retry_backoff_seconds", args.gcs_download_retry_backoff_seconds)
    if args.pin_memory is not None:
        data_cfg.pin_memory = args.pin_memory
    if args.persistent_workers is not None:
        data_cfg.persistent_workers = args.persistent_workers
    if args.enforce_worker_memory_budget is not None:
        data_cfg.enforce_worker_memory_budget = args.enforce_worker_memory_budget
    if args.output_dir:
        cfg.output_dir = args.output_dir
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "batch_size": int(data_cfg.per_device_batch_size),
                "cache_dir": str(data_cfg.cache_dir),
                "dataset_py": str(data_cfg.dataset_py),
                "metadata_prefetch_workers": int(data_cfg.get("metadata_prefetch_workers", 1)),
                "enforce_worker_memory_budget": bool(data_cfg.get("enforce_worker_memory_budget", True)),
                "data_file_prefetch_shards": int(data_cfg.get("data_file_prefetch_shards", 0)),
                "gcs_download_retries": int(data_cfg.get("gcs_download_retries", 1)),
                "gcs_download_timeout_seconds": int(data_cfg.get("gcs_download_timeout_seconds", 0)),
                "num_workers": int(data_cfg.num_workers),
                "pin_memory": bool(data_cfg.pin_memory),
                "multiprocessing_context": str(data_cfg.get("multiprocessing_context", "")),
                "prefetch_factor": int(data_cfg.get("prefetch_factor", 0)),
                "pyav_thread_count": str(data_cfg.get("pyav_thread_count", "")),
                "pyav_reader_cache_size": int(data_cfg.get("pyav_reader_cache_size", 0) or 0),
                "timeout_seconds": int(data_cfg.get("dataloader_timeout_seconds", 0)),
                "video_cache_max_gb": float(data_cfg.get("video_cache_max_gb", 0) or 0),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    torch.set_num_threads(max(1, int(data_cfg.get("worker_torch_threads", 1))))
    loader = build_dataloader(cfg, dataset_py=str(data_cfg.dataset_py), model=None)
    print(
        json.dumps(
            {
                "effective_num_workers": int(getattr(loader, "num_workers", 0) or 0),
                "effective_prefetch_factor": int(getattr(loader, "prefetch_factor", 0) or 0),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    expected_batches = len(loader)
    iterator = iter(loader)
    start_total = time.monotonic()
    completed_batches = 0
    while args.max_batches <= 0 or completed_batches < args.max_batches:
        start = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        elapsed = time.monotonic() - start
        completed_batches += 1
        should_log = (
            completed_batches <= 5
            or completed_batches == expected_batches
            or (
                args.progress_interval_batches > 0
                and completed_batches % args.progress_interval_batches == 0
            )
        )
        if should_log:
            worker_thread_counts = {
                str(pid): thread_count
                for pid in _iterator_worker_pids(iterator)
                if (thread_count := _process_thread_count(pid)) is not None
            }
            payload = {
                "batch_index": completed_batches - 1,
                "batch_size": len(batch),
                "completed_batches": completed_batches,
                "elapsed_fetch_sec": round(elapsed, 4),
                "elapsed_total_sec": round(time.monotonic() - start_total, 2),
                "expected_batches": expected_batches,
                "worker_thread_counts": worker_thread_counts,
            }
            payload.update(_batch_summary(batch))
            print(json.dumps(payload, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "completed_batches": completed_batches,
                "elapsed_total_sec": round(time.monotonic() - start_total, 2),
                "expected_batches": expected_batches,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
