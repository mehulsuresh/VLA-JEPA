#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from bisect import bisect_right
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any
from functools import partial
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from starVLA.dataloader import _configure_lerobot_worker
from starVLA.dataloader.canonical_subset_dataset import (
    DEFAULT_QWEN_CAMERA_SLOTS,
    _ensure_relative_path,
    collate_fn,
    get_vla_dataset,
)


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _maybe_repo_relative(path_value: str | None, repo_root: Path) -> str | None:
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return path.as_posix()
    return (repo_root / path).as_posix()


def _audit_camera_slots(value: str | None) -> list[str] | None:
    token = (value or "").strip().lower()
    if token in {"", "config"}:
        return None
    if token == "all":
        return list(DEFAULT_QWEN_CAMERA_SLOTS)
    return [part.strip() for part in token.split(",") if part.strip()]


def _configure_cfg(args: argparse.Namespace) -> Any:
    repo_root = _resolve_repo_root()
    cfg = OmegaConf.load(args.config_yaml)
    data_cfg = cfg.datasets.vla_data
    audit_mode = bool(getattr(args, "audit_corrupt_videos", False))
    data_cfg.dataset_ids = [args.dataset_id] if args.dataset_id else []
    data_cfg.exclude_dataset_ids = []
    data_cfg.exclude_sids = []
    data_cfg.exclude_dataset_ids_path = []
    data_cfg.exclude_sids_path = []
    data_cfg.max_shards = 0 if audit_mode else args.max_shards
    data_cfg.max_windows = 0 if audit_mode else args.max_windows
    data_cfg.max_shards_per_dataset = 0
    data_cfg.max_windows_per_dataset = 0
    data_cfg.shuffle_shards = False if audit_mode else args.shuffle_shards
    data_cfg.shuffle = False
    data_cfg.shuffle_seed = args.shuffle_seed
    data_cfg.prefetch_metadata_across_ranks = args.metadata_prefetch_workers > 1
    data_cfg.metadata_prefetch_workers = args.metadata_prefetch_workers
    if args.metadata_index_cache is not None:
        data_cfg.metadata_index_cache = args.metadata_index_cache
    data_cfg.lazy_cache_shards = True
    data_cfg.index_windows_lazily = True
    data_cfg.video_decode_backend = "pyav" if audit_mode else args.video_decode_backend
    if audit_mode:
        data_cfg.skip_corrupt_videos = False
        data_cfg.max_sample_decode_retries = 0
        data_cfg.pyav_max_missing_frames_for_fill = 0
        data_cfg.pyav_fail_on_decode_error_recovery = True
        data_cfg.pyav_corrupt_warning_limit = 0
        if audit_camera_slots := _audit_camera_slots(getattr(args, "audit_camera_slots", "main,left,right")):
            data_cfg.qwen_camera_slots = audit_camera_slots
            data_cfg.vjepa_camera_slots = audit_camera_slots
        data_cfg.gcs_download_timeout_seconds = max(int(data_cfg.get("gcs_download_timeout_seconds", 900)), 900)
        data_cfg.gcs_download_retries = max(int(data_cfg.get("gcs_download_retries", 3)), 3)
    if args.pyav_thread_count is not None:
        data_cfg.pyav_thread_count = args.pyav_thread_count
    data_cfg.reader_cache_size = args.reader_cache_size
    data_cfg.sidecar_cache_size = args.sidecar_cache_size
    data_cfg.slow_sample_log_seconds = args.slow_sample_log_seconds
    if args.dataset_canonicalization_root:
        data_cfg.dataset_canonicalization_root = args.dataset_canonicalization_root
    if args.manifest_path:
        data_cfg.manifest_path = args.manifest_path
    if args.adapter_dir:
        data_cfg.adapter_dir = args.adapter_dir
    if args.cache_dir:
        data_cfg.cache_dir = args.cache_dir
    if args.metadata_index_cache_dir:
        data_cfg.metadata_index_cache_dir = args.metadata_index_cache_dir
    if args.bucket_root:
        data_cfg.bucket_root = args.bucket_root
    data_cfg.manifest_path = _maybe_repo_relative(str(data_cfg.manifest_path), repo_root)
    data_cfg.adapter_dir = _maybe_repo_relative(str(data_cfg.adapter_dir), repo_root)
    data_cfg.dataset_canonicalization_root = _maybe_repo_relative(
        str(data_cfg.dataset_canonicalization_root), repo_root
    )
    data_cfg.cache_dir = _maybe_repo_relative(str(data_cfg.cache_dir), repo_root)
    if hasattr(data_cfg, "metadata_index_cache_dir"):
        data_cfg.metadata_index_cache_dir = _maybe_repo_relative(str(data_cfg.metadata_index_cache_dir), repo_root)
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
    if args.loader_batches == 0:
        return
    full_sweep = args.loader_batches < 0
    if full_sweep and (args.loader_start_index > 0 or args.loader_sample_stride != 1):
        raise ValueError("--loader-batches -1 requires loader_start_index=0 and loader_sample_stride=1")
    loader_dataset = dataset
    if not full_sweep and (args.loader_start_index > 0 or args.loader_sample_stride != 1):
        subset_len = args.loader_batches * args.batch_size
        subset_indices = [
            (args.loader_start_index + idx * args.loader_sample_stride) % len(dataset)
            for idx in range(subset_len)
        ]
        loader_dataset = torch.utils.data.Subset(dataset, subset_indices)
    expected_batches = len(loader_dataset) // args.batch_size
    print(
        "dataloader_probe "
        f"batches={'all' if full_sweep else args.loader_batches} expected_batches={expected_batches} "
        f"batch_size={args.batch_size} workers={args.num_workers} "
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
    batch_index = 0
    start_total = time.monotonic()
    while full_sweep or batch_index < args.loader_batches:
        start = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch_size = len(batch)
        elapsed = time.monotonic() - start
        should_log = (
            batch_index < 5
            or not full_sweep
            or (args.progress_interval_batches > 0 and (batch_index + 1) % args.progress_interval_batches == 0)
            or (full_sweep and batch_index + 1 == expected_batches)
        )
        if should_log:
            dataset_ids = sorted({str(sample.get("dataset_id")) for sample in batch})
            frame_indices = [int(sample.get("frame_index", -1)) for sample in batch[: min(4, len(batch))]]
            worker_pids = _iterator_worker_pids(iterator)
            worker_thread_counts = {
                str(pid): thread_count
                for pid in worker_pids
                if (thread_count := _process_thread_count(pid)) is not None
            }
            print(
                json.dumps(
                    {
                        "batch_index": batch_index,
                        "batch_size": batch_size,
                        "dataset_ids": dataset_ids,
                        "elapsed_fetch_sec": round(elapsed, 4),
                        "elapsed_total_sec": round(time.monotonic() - start_total, 2),
                        "expected_batches": expected_batches,
                        "first_frame_indices": frame_indices,
                        "worker_thread_counts": worker_thread_counts,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        batch_index += 1
    print(
        json.dumps(
            {
                "completed_batches": batch_index,
                "elapsed_total_sec": round(time.monotonic() - start_total, 2),
                "expected_batches": expected_batches,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_index_component(value: str, prefix: str) -> int | None:
    if not value.startswith(prefix):
        return None
    suffix = value[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _video_chunk_file_indices(relative_path: str) -> dict[str, int | None]:
    parts = Path(relative_path).parts
    chunk_index = _parse_index_component(parts[-2], "chunk-") if len(parts) >= 2 else None
    file_index = _parse_index_component(Path(parts[-1]).stem, "file-") if parts else None
    return {
        "video_chunk_index": chunk_index,
        "video_file_index": file_index,
    }


def _format_decode_error(dataset: Any, exc: Exception) -> str:
    formatter = getattr(dataset, "_format_pyav_error", None)
    if formatter is None:
        return str(exc)
    try:
        return str(formatter(exc))
    except Exception:
        return str(exc)


def _seek_pyav_reader(reader: Any, seek_frame: int) -> None:
    fps = float(reader.fps or 0.0)
    time_base = float(reader.time_base or 0.0)
    start_time = int(reader.start_time or 0)
    if time_base > 0 and fps > 0:
        seek_pts = start_time + int((int(seek_frame) / fps) / time_base)
        try:
            reader.container.seek(max(seek_pts - 2, 0), stream=reader.stream, backward=True, any_frame=False)
            return
        except Exception:
            pass
    reader.container.seek(0, stream=reader.stream, backward=True, any_frame=False)


def _pyav_frame_index(reader: Any, frame: Any, last_decoded_index: int | None) -> int:
    fps = float(reader.fps or 0.0)
    time_base = float(reader.time_base or 0.0)
    start_time = int(reader.start_time or 0)
    stream_start_seconds = float(start_time) * time_base if time_base > 0 else 0.0
    if frame.time is not None and fps > 0:
        return int(round((float(frame.time) - stream_start_seconds) * fps))
    if frame.pts is not None and time_base > 0 and fps > 0:
        return int(round((int(frame.pts) - start_time) * time_base * fps))
    return 0 if last_decoded_index is None else last_decoded_index + 1


def _audit_decode_episode_interval(
    dataset: Any,
    local_path: Path,
    start_frame: int,
    end_frame_exclusive: int,
) -> dict[str, Any]:
    expected_count = max(0, int(end_frame_exclusive) - int(start_frame))
    result: dict[str, Any] = {
        "ok": expected_count == 0,
        "expected_frame_count": expected_count,
        "decoded_frame_count": 0,
        "first_decoded_frame": None,
        "last_decoded_frame": None,
        "first_missing_frame": None,
        "corrupt_video_frame_start": None,
        "corrupt_video_frame_end_exclusive": None,
        "failure_stage": None,
        "error_type": None,
        "error": None,
    }
    if expected_count == 0:
        return result

    reader = None
    expected_next = int(start_frame)
    last_decoded_index: int | None = None
    first_missing_frame: int | None = None
    try:
        reader = dataset._make_pyav_reader(local_path.as_posix())
        _seek_pyav_reader(reader, int(start_frame))
        for frame in reader.container.decode(reader.stream):
            frame_index = _pyav_frame_index(reader, frame, last_decoded_index)
            last_decoded_index = frame_index
            if frame_index < start_frame:
                continue
            if frame_index >= end_frame_exclusive:
                break
            if result["first_decoded_frame"] is None:
                result["first_decoded_frame"] = int(frame_index)
            if frame_index < expected_next:
                continue
            if frame_index > expected_next and first_missing_frame is None:
                first_missing_frame = int(expected_next)
            expected_next = int(frame_index) + 1
            result["decoded_frame_count"] = int(result["decoded_frame_count"]) + 1
            result["last_decoded_frame"] = int(frame_index)
            if expected_next >= end_frame_exclusive:
                break
    except Exception as exc:
        result["failure_stage"] = "pyav_decode"
        result["error_type"] = type(exc).__name__
        result["error"] = _format_decode_error(dataset, exc)
    finally:
        if reader is not None:
            try:
                reader.container.close()
            except Exception:
                pass

    if (
        result["error"] is None
        and first_missing_frame is None
        and result["first_decoded_frame"] == int(start_frame)
        and expected_next >= int(end_frame_exclusive)
    ):
        result["ok"] = True
        return result

    if first_missing_frame is not None:
        corrupt_start = first_missing_frame
    elif result["first_decoded_frame"] is None or int(result["first_decoded_frame"]) > int(start_frame):
        corrupt_start = int(start_frame)
    else:
        last_decoded = result["last_decoded_frame"]
        corrupt_start = int(start_frame) if last_decoded is None else min(int(last_decoded) + 1, int(end_frame_exclusive))
    result["first_missing_frame"] = int(corrupt_start)
    result["corrupt_video_frame_start"] = int(corrupt_start)
    result["corrupt_video_frame_end_exclusive"] = int(end_frame_exclusive)
    if result["failure_stage"] is None:
        result["failure_stage"] = "pyav_incomplete"
    return result


def _add_corrupt_episode(
    corrupt_episodes: dict[tuple[str, str, str, int, str], dict[str, Any]],
    event: dict[str, Any],
) -> None:
    key = (
        str(event["dataset_id"]),
        str(event["sid"]),
        str(event["revision"]),
        int(event["episode_index"]),
        str(event["data_file"]),
    )
    episode = corrupt_episodes.setdefault(
        key,
        {
            "dataset_id": event["dataset_id"],
            "sid": event["sid"],
            "revision": event["revision"],
            "episode_index": event["episode_index"],
            "episode_position_in_shard": event["episode_position_in_shard"],
            "data_file": event["data_file"],
            "task": event["task"],
            "dataset_from_index": event["dataset_from_index"],
            "slots": set(),
            "video_files": set(),
            "corrupt_ranges": [],
        },
    )
    episode["slots"].add(event["slot"])
    episode["video_files"].add(event["video_relative_path"])
    episode["corrupt_ranges"].append(
        {
            "slot": event["slot"],
            "source_key": event["source_key"],
            "video_relative_path": event["video_relative_path"],
            "video_chunk_index": event["video_chunk_index"],
            "video_file_index": event["video_file_index"],
            "episode_frame_start": event["corrupt_episode_frame_start"],
            "episode_frame_end_exclusive": event["corrupt_episode_frame_end_exclusive"],
            "video_frame_start": event["corrupt_video_frame_start"],
            "video_frame_end_exclusive": event["corrupt_video_frame_end_exclusive"],
            "global_dataset_index_start": event["global_dataset_index_start"],
            "global_dataset_index_end_exclusive": event["global_dataset_index_end_exclusive"],
            "failure_stage": event["failure_stage"],
            "error_type": event["error_type"],
            "error": event["error"],
        }
    )


def _serializable_episode_entry(entry: dict[str, Any]) -> dict[str, Any]:
    result = dict(entry)
    result["slots"] = sorted(result["slots"])
    result["video_files"] = sorted(result["video_files"])
    result["corrupt_ranges"] = sorted(
        result["corrupt_ranges"],
        key=lambda item: (
            str(item["video_relative_path"]),
            int(item["episode_frame_start"]),
            str(item["slot"]),
        ),
    )
    return result


def _video_audit_key(sid: str, revision: str, video_relative_path: str) -> str:
    return f"{sid}\t{revision}\t{video_relative_path}"


def _episode_blacklist_key(event: dict[str, Any]) -> str:
    return "\t".join(
        [
            str(event["sid"]),
            str(event["revision"]),
            str(event["data_file"]),
            str(event["episode_index"]),
        ]
    )


def _make_worker_pyav_reader(path_key: str, thread_count: int, thread_type: str) -> Any:
    import av

    try:
        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass
    container = av.open(path_key, mode="r")
    try:
        stream = container.streams.video[0]
        if thread_count > 0:
            try:
                stream.codec_context.thread_count = thread_count
            except Exception:
                pass
        if thread_type and thread_type != "DEFAULT":
            try:
                stream.thread_type = thread_type
            except Exception:
                pass
        return SimpleNamespace(
            container=container,
            stream=stream,
            fps=float(stream.average_rate or stream.base_rate or 30.0),
            time_base=float(stream.time_base or 0.0),
            start_time=int(stream.start_time or 0),
        )
    except Exception:
        container.close()
        raise


def _decode_video_interval_once(
    local_path: str,
    start_frame: int,
    end_frame_exclusive: int,
    *,
    thread_count: int,
    thread_type: str,
) -> dict[str, Any]:
    expected_count = max(0, int(end_frame_exclusive) - int(start_frame))
    result: dict[str, Any] = {
        "ok": expected_count == 0,
        "expected_frame_count": expected_count,
        "decoded_frame_count": 0,
        "first_decoded_frame": None,
        "last_decoded_frame": None,
        "first_missing_frame": None,
        "corrupt_video_frame_start": None,
        "corrupt_video_frame_end_exclusive": None,
        "failure_stage": None,
        "error_type": None,
        "error": None,
    }
    if expected_count == 0:
        return result

    reader = None
    expected_next = int(start_frame)
    last_decoded_index: int | None = None
    first_missing_frame: int | None = None
    try:
        reader = _make_worker_pyav_reader(local_path, thread_count, thread_type)
        _seek_pyav_reader(reader, int(start_frame))
        for frame in reader.container.decode(reader.stream):
            frame_index = _pyav_frame_index(reader, frame, last_decoded_index)
            last_decoded_index = frame_index
            if frame_index < start_frame:
                continue
            if frame_index >= end_frame_exclusive:
                break
            if result["first_decoded_frame"] is None:
                result["first_decoded_frame"] = int(frame_index)
            if frame_index < expected_next:
                continue
            if frame_index > expected_next and first_missing_frame is None:
                first_missing_frame = int(expected_next)
            expected_next = int(frame_index) + 1
            result["decoded_frame_count"] = int(result["decoded_frame_count"]) + 1
            result["last_decoded_frame"] = int(frame_index)
            if expected_next >= end_frame_exclusive:
                break
    except Exception as exc:
        result["failure_stage"] = "pyav_decode"
        result["error_type"] = type(exc).__name__
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if reader is not None:
            try:
                reader.container.close()
            except Exception:
                pass

    if (
        result["error"] is None
        and first_missing_frame is None
        and result["first_decoded_frame"] == int(start_frame)
        and expected_next >= int(end_frame_exclusive)
    ):
        result["ok"] = True
        return result

    if first_missing_frame is not None:
        corrupt_start = first_missing_frame
    elif result["first_decoded_frame"] is None or int(result["first_decoded_frame"]) > int(start_frame):
        corrupt_start = int(start_frame)
    else:
        last_decoded = result["last_decoded_frame"]
        corrupt_start = int(start_frame) if last_decoded is None else min(int(last_decoded) + 1, int(end_frame_exclusive))
    result["first_missing_frame"] = int(corrupt_start)
    result["corrupt_video_frame_start"] = int(corrupt_start)
    result["corrupt_video_frame_end_exclusive"] = int(end_frame_exclusive)
    if result["failure_stage"] is None:
        result["failure_stage"] = "pyav_incomplete"
    return result


def _merge_frame_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _find_recovery_start(
    local_path: str,
    episode_starts: list[int],
    corrupt_start: int,
    union_end: int,
    *,
    thread_count: int,
    thread_type: str,
) -> int | None:
    start_index = bisect_right(episode_starts, int(corrupt_start))
    for candidate in episode_starts[start_index:]:
        if candidate >= union_end:
            break
        probe = _decode_video_interval_once(
            local_path,
            int(candidate),
            min(int(candidate) + 1, int(union_end)),
            thread_count=thread_count,
            thread_type=thread_type,
        )
        if probe["ok"]:
            return int(candidate)
    return None


def _scan_video_bad_ranges(
    local_path: str,
    refs: list[dict[str, Any]],
    *,
    thread_count: int,
    thread_type: str,
) -> list[dict[str, Any]]:
    union_intervals = _merge_frame_intervals(
        [(int(ref["video_frame_start"]), int(ref["video_frame_end_exclusive"])) for ref in refs]
    )
    episode_starts = sorted({int(ref["video_frame_start"]) for ref in refs})
    bad_ranges: list[dict[str, Any]] = []
    for union_start, union_end in union_intervals:
        scan_start = int(union_start)
        while scan_start < union_end:
            result = _decode_video_interval_once(
                local_path,
                scan_start,
                int(union_end),
                thread_count=thread_count,
                thread_type=thread_type,
            )
            if result["ok"]:
                break
            corrupt_start = int(result["corrupt_video_frame_start"] or scan_start)
            recovery_start = None
            if result["failure_stage"] == "pyav_decode":
                recovery_start = _find_recovery_start(
                    local_path,
                    episode_starts,
                    corrupt_start,
                    int(union_end),
                    thread_count=thread_count,
                    thread_type=thread_type,
                )
            corrupt_end = int(recovery_start or union_end)
            if corrupt_end > corrupt_start:
                bad_ranges.append(
                    {
                        **result,
                        "corrupt_video_frame_start": corrupt_start,
                        "corrupt_video_frame_end_exclusive": corrupt_end,
                    }
                )
            if recovery_start is None or recovery_start <= scan_start:
                break
            scan_start = int(recovery_start)
    return bad_ranges


def _map_bad_ranges_to_episode_events(
    job: dict[str, Any],
    local_path: str,
    bad_ranges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for ref in job["episode_refs"]:
        episode_start = int(ref["video_frame_start"])
        episode_end = int(ref["video_frame_end_exclusive"])
        for bad_range in bad_ranges:
            corrupt_video_start = max(int(bad_range["corrupt_video_frame_start"]), episode_start)
            corrupt_video_end = min(int(bad_range["corrupt_video_frame_end_exclusive"]), episode_end)
            if corrupt_video_start >= corrupt_video_end:
                continue
            corrupt_episode_start = corrupt_video_start - episode_start
            corrupt_episode_end = corrupt_video_end - episode_start
            dataset_from_index = ref["dataset_from_index"]
            event = {
                **ref,
                "local_video_path": local_path,
                "ok": False,
                "expected_frame_count": int(ref["episode_length"]),
                "decoded_frame_count": bad_range["decoded_frame_count"],
                "first_decoded_frame": bad_range["first_decoded_frame"],
                "last_decoded_frame": bad_range["last_decoded_frame"],
                "first_missing_frame": bad_range["first_missing_frame"],
                "corrupt_video_frame_start": corrupt_video_start,
                "corrupt_video_frame_end_exclusive": corrupt_video_end,
                "corrupt_episode_frame_start": int(corrupt_episode_start),
                "corrupt_episode_frame_end_exclusive": int(corrupt_episode_end),
                "global_dataset_index_start": (
                    None if dataset_from_index is None else int(dataset_from_index + corrupt_episode_start)
                ),
                "global_dataset_index_end_exclusive": (
                    None if dataset_from_index is None else int(dataset_from_index + corrupt_episode_end)
                ),
                "failure_stage": bad_range["failure_stage"],
                "error_type": bad_range["error_type"],
                "error": bad_range["error"],
            }
            event["blacklist_key"] = _episode_blacklist_key(event)
            events.append(event)
    return events


def _run_video_audit_job(job: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "video_key": job["video_key"],
        "status": "ok",
        "checks": len(job["episode_refs"]),
        "local_video_path": None,
        "corrupt_events": [],
        "fetch_failure": None,
        "elapsed_sec": None,
    }
    try:
        local_path = _ensure_relative_path(
            root=Path(job["root"]),
            gcs_prefix=str(job["gcs_prefix"]),
            relative_path=str(job["video_relative_path"]),
            allow_gcs_download=bool(job["allow_gcs_download"]),
            gcs_timeout_seconds=int(job["gcs_timeout_seconds"]),
            gcs_retries=int(job["gcs_retries"]),
            gcs_retry_backoff_seconds=float(job["gcs_retry_backoff_seconds"]),
        )
        if local_path is None:
            raise FileNotFoundError(job["video_relative_path"])
        result["local_video_path"] = local_path.as_posix()
    except Exception as exc:
        result["status"] = "fetch_failed"
        result["fetch_failure"] = {
            "video_key": job["video_key"],
            "dataset_id": job["dataset_id"],
            "sid": job["sid"],
            "revision": job["revision"],
            "video_relative_path": job["video_relative_path"],
            "affected_episode_camera_checks": len(job["episode_refs"]),
            "failure_stage": "video_fetch",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        result["elapsed_sec"] = round(time.monotonic() - started, 4)
        return result

    bad_ranges = _scan_video_bad_ranges(
        local_path.as_posix(),
        job["episode_refs"],
        thread_count=int(job["pyav_thread_count"]),
        thread_type=str(job["pyav_thread_type"]),
    )
    if bad_ranges:
        result["status"] = "corrupt"
        result["corrupt_events"] = _map_bad_ranges_to_episode_events(job, local_path.as_posix(), bad_ranges)
    result["elapsed_sec"] = round(time.monotonic() - started, 4)
    return result


def _is_gcloud_auth_failure_message(message: str) -> bool:
    auth_markers = (
        "Reauthentication failed",
        "cannot prompt during non-interactive execution",
        "invalid_grant",
        "There was a problem refreshing your current auth tokens",
    )
    return any(marker in message for marker in auth_markers)


def _is_gcloud_auth_failure_result(result: dict[str, Any]) -> bool:
    if result.get("status") != "fetch_failed":
        return False
    fetch_failure = result.get("fetch_failure") or {}
    return _is_gcloud_auth_failure_message(str(fetch_failure.get("error") or result.get("error") or ""))


def _load_video_audit_results(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    results = []
    bad_lines = 0
    with results_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    if bad_lines:
        with results_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "audit_warning": "dropped_invalid_video_audit_result_lines",
                    "path": results_path.as_posix(),
                    "dropped_lines": bad_lines,
                    "kept_results": len(results),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return results


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _local_video_path_from_result(result: dict[str, Any], cache_root: Path | None) -> Path | None:
    path_value = result.get("local_video_path")
    if path_value:
        return Path(str(path_value))
    corrupt_events = result.get("corrupt_events") or []
    if corrupt_events:
        event_path = corrupt_events[0].get("local_video_path")
        if event_path:
            return Path(str(event_path))
    if cache_root is None:
        return None
    video_key = str(result.get("video_key") or "")
    parts = video_key.split("\t", 2)
    if len(parts) != 3:
        return None
    sid, revision, relative_path = parts
    return cache_root / sid / revision / relative_path


def _delete_local_video_cache_file(
    result: dict[str, Any],
    *,
    cache_root: Path | None,
    deletion_log_handle: Any | None,
) -> int:
    if result.get("status") == "fetch_failed":
        return 0
    path = _local_video_path_from_result(result, cache_root)
    if path is None or not path.exists() or not path.is_file():
        return 0
    if path.suffix.lower() != ".mp4":
        return 0
    if cache_root is not None and not _path_is_relative_to(path, cache_root):
        print(
            json.dumps(
                {
                    "audit_warning": "refusing_to_delete_video_outside_cache_root",
                    "path": path.as_posix(),
                    "cache_root": cache_root.as_posix(),
                    "video_key": result.get("video_key"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    try:
        size = path.stat().st_size
        path.unlink()
    except FileNotFoundError:
        return 0
    if deletion_log_handle is not None:
        deletion_log_handle.write(
            "\t".join(
                [
                    str(int(time.time())),
                    str(size),
                    str(result.get("status") or ""),
                    str(result.get("video_key") or ""),
                    path.as_posix(),
                ]
            )
            + "\n"
        )
        deletion_log_handle.flush()
    return int(size)


def _rebuild_audit_outputs_from_results(
    results: list[dict[str, Any]],
    corruptions_path: Path,
    fetch_failures_path: Path,
) -> tuple[dict[tuple[str, str, str, int, str], dict[str, Any]], set[tuple[str, str, str]], int, int]:
    corrupt_episodes: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    corrupt_video_files: set[tuple[str, str, str]] = set()
    corruptions = 0
    fetch_failures = 0
    with (
        corruptions_path.open("w", encoding="utf-8") as corruptions_handle,
        fetch_failures_path.open("w", encoding="utf-8") as fetch_failures_handle,
    ):
        for result in results:
            fetch_failure = result.get("fetch_failure")
            if fetch_failure:
                fetch_failures += 1
                fetch_failures_handle.write(json.dumps(fetch_failure, sort_keys=True) + "\n")
            for event in result.get("corrupt_events", []):
                corruptions += 1
                corruptions_handle.write(json.dumps(event, sort_keys=True) + "\n")
                _add_corrupt_episode(corrupt_episodes, event)
                corrupt_video_files.add((str(event["dataset_id"]), str(event["sid"]), str(event["video_relative_path"])))
    return corrupt_episodes, corrupt_video_files, corruptions, fetch_failures


def _write_corrupt_episode_outputs(
    corrupt_episodes: dict[tuple[str, str, str, int, str], dict[str, Any]],
    cull_episodes_path: Path,
    cull_episode_indices_path: Path,
    corrupt_video_files_path: Path,
    corrupt_video_files: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    serializable_episodes = [
        _serializable_episode_entry(entry)
        for entry in sorted(
            corrupt_episodes.values(),
            key=lambda item: (
                str(item["sid"]),
                str(item["revision"]),
                str(item["data_file"]),
                int(item["episode_index"]),
            ),
        )
    ]
    with cull_episodes_path.open("w", encoding="utf-8") as handle:
        for entry in serializable_episodes:
            entry["blacklist_key"] = "\t".join(
                [str(entry["sid"]), str(entry["revision"]), str(entry["data_file"]), str(entry["episode_index"])]
            )
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    with cull_episode_indices_path.open("w", encoding="utf-8") as handle:
        handle.write("sid\trevision\tdata_file\tepisode_index\tdataset_id\tdataset_from_index\ttask\n")
        for entry in serializable_episodes:
            handle.write(
                "\t".join(
                    [
                        str(entry["sid"]),
                        str(entry["revision"]),
                        str(entry["data_file"]),
                        str(entry["episode_index"]),
                        str(entry["dataset_id"]),
                        "" if entry["dataset_from_index"] is None else str(entry["dataset_from_index"]),
                        str(entry["task"]).replace("\t", " "),
                    ]
                )
                + "\n"
            )
    with corrupt_video_files_path.open("w", encoding="utf-8") as handle:
        handle.write("dataset_id\tsid\tvideo_relative_path\n")
        for dataset_id, sid, video_relative_path in sorted(corrupt_video_files):
            handle.write(f"{dataset_id}\t{sid}\t{video_relative_path}\n")
    return serializable_episodes


def _build_video_audit_jobs(dataset: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    audit_slots = _audit_camera_slots(getattr(args, "audit_camera_slots", "main,left,right"))
    audit_slot_set = set(audit_slots) if audit_slots else None
    jobs: dict[str, dict[str, Any]] = {}
    checks = 0
    stop = False
    for shard_index, shard in enumerate(dataset.shards):
        if stop:
            break
        for episode_position, episode in enumerate(shard.episodes):
            if stop:
                break
            episode_index = int(episode.episode_index if episode.episode_index is not None else episode_position)
            dataset_from_index = int(episode.dataset_from_index) if episode.dataset_from_index is not None else None
            for slot in shard.decode_camera_slots:
                if audit_slot_set is not None and slot not in audit_slot_set:
                    continue
                if slot not in episode.video_paths:
                    continue
                if args.audit_max_episode_checks > 0 and checks >= args.audit_max_episode_checks:
                    stop = True
                    break
                checks += 1
                video_path = episode.video_paths[slot]
                video_relative_path = _safe_relative_path(video_path, shard.root)
                video_indices = _video_chunk_file_indices(video_relative_path)
                video_key = _video_audit_key(str(shard.sid), str(shard.revision), video_relative_path)
                job = jobs.get(video_key)
                if job is None:
                    job = {
                        "video_key": video_key,
                        "dataset_id": str(shard.dataset_id),
                        "sid": str(shard.sid),
                        "revision": str(shard.revision),
                        "root": shard.root.as_posix(),
                        "gcs_prefix": shard.gcs_prefix,
                        "video_relative_path": video_relative_path,
                        "allow_gcs_download": bool(dataset.allow_gcs_download),
                        "gcs_timeout_seconds": int(args.audit_video_fetch_timeout_seconds),
                        "gcs_retries": int(args.audit_video_fetch_retries),
                        "gcs_retry_backoff_seconds": float(args.audit_video_fetch_retry_backoff_seconds),
                        "pyav_thread_count": int(dataset.pyav_thread_count),
                        "pyav_thread_type": str(dataset.pyav_thread_type),
                        "episode_refs": [],
                    }
                    jobs[video_key] = job
                video_start = int(episode.video_base_frames[slot])
                video_end = video_start + int(episode.length)
                ref = {
                    "dataset_id": str(shard.dataset_id),
                    "sid": str(shard.sid),
                    "revision": str(shard.revision),
                    "shard_index": int(shard_index),
                    "data_file": str(shard.data_relative_path),
                    "episode_index": episode_index,
                    "episode_position_in_shard": int(episode_position),
                    "task": str(episode.task),
                    "dataset_from_index": dataset_from_index,
                    "slot": str(slot),
                    "source_key": str(shard.camera_source_keys.get(slot, slot)),
                    "video_relative_path": video_relative_path,
                    "video_frame_start": video_start,
                    "video_frame_end_exclusive": video_end,
                    "episode_length": int(episode.length),
                    **video_indices,
                }
                job["episode_refs"].append(ref)
    return list(jobs.values())


def _run_corrupt_video_audit(dataset: Any, args: argparse.Namespace) -> None:
    output_dir = Path(args.audit_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corruptions_path = output_dir / "corrupted_video_episode_ranges.jsonl"
    fetch_failures_path = output_dir / "video_fetch_failures.jsonl"
    cull_episodes_path = output_dir / "corrupted_episodes.jsonl"
    cull_episode_indices_path = output_dir / "corrupted_episode_indices.tsv"
    corrupt_video_files_path = output_dir / "corrupted_video_files.tsv"
    video_results_path = output_dir / "video_audit_results.jsonl"
    deleted_video_cache_path = output_dir / "deleted_processed_videos.tsv"
    summary_path = output_dir / "summary.json"
    if not args.audit_resume:
        for path in (
            corruptions_path,
            fetch_failures_path,
            cull_episodes_path,
            cull_episode_indices_path,
            corrupt_video_files_path,
            video_results_path,
            deleted_video_cache_path,
            summary_path,
        ):
            path.unlink(missing_ok=True)

    jobs = _build_video_audit_jobs(dataset, args)
    total_checks = sum(len(job["episode_refs"]) for job in jobs)
    completed_results = _load_video_audit_results(video_results_path) if args.audit_resume else []
    completed_video_keys = {str(result["video_key"]) for result in completed_results}
    corrupt_episodes, corrupt_video_files, corruptions, fetch_failures = _rebuild_audit_outputs_from_results(
        completed_results,
        corruptions_path,
        fetch_failures_path,
    )
    completed_checks = sum(int(result.get("checks", 0)) for result in completed_results)
    remaining_jobs = [job for job in jobs if job["video_key"] not in completed_video_keys]
    cache_root = Path(args.cache_dir).expanduser() if args.cache_dir else None
    deleted_video_cache_files = 0
    deleted_video_cache_bytes = 0
    if args.audit_delete_video_cache_after_processing and cache_root is None:
        raise ValueError("--audit-delete-video-cache-after-processing requires --cache-dir for safety")

    print(
        json.dumps(
            {
                "audit": "canonical_pyav_video_corruption",
                "audit_unit": "video_file",
                "camera_slots": _audit_camera_slots(getattr(args, "audit_camera_slots", "main,left,right")),
                "checks": total_checks,
                "completed_checks": completed_checks,
                "video_files": len(jobs),
                "completed_video_files": len(completed_results),
                "remaining_video_files": len(remaining_jobs),
                "datasets": len({shard.dataset_id for shard in dataset.shards}),
                "resume": bool(args.audit_resume),
                "workers": int(args.audit_workers),
                "multiprocessing_context": args.multiprocessing_context or "default",
                "training_skip_disabled": True,
                "delete_video_cache_after_processing": bool(args.audit_delete_video_cache_after_processing),
                "shards": len(dataset.shards),
                "output_dir": output_dir.as_posix(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    start_time = time.monotonic()
    completed_video_count = len(completed_results)

    def _abort_if_gcloud_auth_failed(result: dict[str, Any]) -> None:
        if not args.audit_abort_on_auth_failure or not _is_gcloud_auth_failure_result(result):
            return
        fetch_failure = result.get("fetch_failure") or {}
        print(
            json.dumps(
                {
                    "audit_error": "gcloud_auth_failed",
                    "message": "Aborting without checkpointing this auth failure. Refresh gcloud auth and resume.",
                    "video_key": result.get("video_key"),
                    "dataset_id": fetch_failure.get("dataset_id"),
                    "video_relative_path": fetch_failure.get("video_relative_path"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise RuntimeError("gcloud auth failed during audit; refresh auth and resume")

    def _record_result(
        result: dict[str, Any],
        corruptions_handle: Any,
        fetch_failures_handle: Any,
        deletion_log_handle: Any | None,
    ) -> None:
        nonlocal completed_checks, completed_video_count, corruptions, fetch_failures
        nonlocal deleted_video_cache_files, deleted_video_cache_bytes
        completed_video_count += 1
        completed_checks += int(result.get("checks", 0))
        fetch_failure = result.get("fetch_failure")
        if fetch_failure:
            fetch_failures += 1
            fetch_failures_handle.write(json.dumps(fetch_failure, sort_keys=True) + "\n")
            fetch_failures_handle.flush()
        for event in result.get("corrupt_events", []):
            corruptions += 1
            corruptions_handle.write(json.dumps(event, sort_keys=True) + "\n")
            _add_corrupt_episode(corrupt_episodes, event)
            corrupt_video_files.add((str(event["dataset_id"]), str(event["sid"]), str(event["video_relative_path"])))
        if result.get("corrupt_events"):
            corruptions_handle.flush()
        if args.audit_delete_video_cache_after_processing:
            deleted_bytes = _delete_local_video_cache_file(
                result,
                cache_root=cache_root,
                deletion_log_handle=deletion_log_handle,
            )
            if deleted_bytes:
                deleted_video_cache_files += 1
                deleted_video_cache_bytes += deleted_bytes

    workers = max(1, int(args.audit_workers))
    with (
        video_results_path.open("a", encoding="utf-8") as results_handle,
        corruptions_path.open("a", encoding="utf-8") as corruptions_handle,
        fetch_failures_path.open("a", encoding="utf-8") as fetch_failures_handle,
        deleted_video_cache_path.open("a", encoding="utf-8") as deletion_log_handle,
    ):
        if args.audit_delete_video_cache_after_processing:
            for completed_result in completed_results:
                deleted_bytes = _delete_local_video_cache_file(
                    completed_result,
                    cache_root=cache_root,
                    deletion_log_handle=deletion_log_handle,
                )
                if deleted_bytes:
                    deleted_video_cache_files += 1
                    deleted_video_cache_bytes += deleted_bytes
        if workers == 1:
            for job in remaining_jobs:
                result = _run_video_audit_job(job)
                _abort_if_gcloud_auth_failed(result)
                results_handle.write(json.dumps(result, sort_keys=True) + "\n")
                results_handle.flush()
                _record_result(result, corruptions_handle, fetch_failures_handle, deletion_log_handle)
                if args.audit_progress_interval > 0 and (
                    completed_video_count == 1
                    or completed_video_count % args.audit_progress_interval == 0
                    or completed_video_count == len(jobs)
                ):
                    print(
                        json.dumps(
                            {
                                "audit_progress": completed_checks,
                                "checks": total_checks,
                                "audit_progress_videos": completed_video_count,
                                "video_files": len(jobs),
                                "corruptions": corruptions,
                                "corrupt_episodes": len(corrupt_episodes),
                                "fetch_failures": fetch_failures,
                                "deleted_video_cache_files": deleted_video_cache_files,
                                "deleted_video_cache_gib": round(deleted_video_cache_bytes / 1024 / 1024 / 1024, 2),
                                "elapsed_sec": round(time.monotonic() - start_time, 2),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        else:
            worker_context = mp.get_context(args.multiprocessing_context) if args.multiprocessing_context else None
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=worker_context)
            try:
                futures = {executor.submit(_run_video_audit_job, job): job["video_key"] for job in remaining_jobs}
                for future in as_completed(futures):
                    result = future.result()
                    try:
                        _abort_if_gcloud_auth_failed(result)
                    except Exception:
                        for pending in futures:
                            pending.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise
                    results_handle.write(json.dumps(result, sort_keys=True) + "\n")
                    results_handle.flush()
                    _record_result(result, corruptions_handle, fetch_failures_handle, deletion_log_handle)
                    if args.audit_progress_interval > 0 and (
                        completed_video_count == 1
                        or completed_video_count % args.audit_progress_interval == 0
                        or completed_video_count == len(jobs)
                    ):
                        print(
                            json.dumps(
                                {
                                    "audit_progress": completed_checks,
                                    "checks": total_checks,
                                    "audit_progress_videos": completed_video_count,
                                    "video_files": len(jobs),
                                    "corruptions": corruptions,
                                    "corrupt_episodes": len(corrupt_episodes),
                                    "fetch_failures": fetch_failures,
                                    "deleted_video_cache_files": deleted_video_cache_files,
                                    "deleted_video_cache_gib": round(deleted_video_cache_bytes / 1024 / 1024 / 1024, 2),
                                    "elapsed_sec": round(time.monotonic() - start_time, 2),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

    serializable_episodes = _write_corrupt_episode_outputs(
        corrupt_episodes,
        cull_episodes_path,
        cull_episode_indices_path,
        corrupt_video_files_path,
        corrupt_video_files,
    )

    summary = {
        "checks": completed_checks,
        "total_checks": total_checks,
        "video_files": len(jobs),
        "completed_video_files": completed_video_count,
        "corruptions": corruptions,
        "corrupt_episodes": len(corrupt_episodes),
        "corrupt_video_files": len(corrupt_video_files),
        "fetch_failures": fetch_failures,
        "deleted_video_cache_files": deleted_video_cache_files,
        "deleted_video_cache_gib": round(deleted_video_cache_bytes / 1024 / 1024 / 1024, 2),
        "multiprocessing_context": args.multiprocessing_context or "default",
        "training_skip_disabled": True,
        "elapsed_sec": round(time.monotonic() - start_time, 2),
        "outputs": {
            "corrupted_ranges": corruptions_path.as_posix(),
            "video_fetch_failures": fetch_failures_path.as_posix(),
            "corrupted_episodes": cull_episodes_path.as_posix(),
            "corrupted_episode_indices": cull_episode_indices_path.as_posix(),
            "corrupted_video_files": corrupt_video_files_path.as_posix(),
            "video_audit_results": video_results_path.as_posix(),
            "deleted_video_cache": deleted_video_cache_path.as_posix(),
            "summary": summary_path.as_posix(),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe or audit canonical dataset video decode behavior without GPUs.")
    parser.add_argument("--config-yaml", default="scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml")
    parser.add_argument("--dataset-id", default="", help="Optional single canonical dataset id. Empty means all datasets.")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--shuffle-shards", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--video-decode-backend", default="pyav", choices=["auto", "decord", "pyav", "imageio"])
    parser.add_argument("--pyav-thread-count", default=None)
    parser.add_argument("--reader-cache-size", type=int, default=32)
    parser.add_argument("--sidecar-cache-size", type=int, default=8)
    parser.add_argument("--slow-sample-log-seconds", type=float, default=2.0)
    parser.add_argument("--dataset-canonicalization-root", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--metadata-index-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--metadata-index-cache-dir", default="")
    parser.add_argument("--metadata-prefetch-workers", type=int, default=1)
    parser.add_argument("--bucket-root", default="")
    parser.add_argument("--show-shards", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--indices", default="")
    parser.add_argument("--loader-batches", type=int, default=4)
    parser.add_argument("--loader-start-index", type=int, default=0)
    parser.add_argument("--loader-sample-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=26)
    parser.add_argument("--progress-interval-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multiprocessing-context", default="spawn")
    parser.add_argument(
        "--audit-corrupt-videos",
        action="store_true",
        help="Decode every episode camera interval with PyAV and log corrupt chunks/files/episodes for culling.",
    )
    parser.add_argument("--audit-output-dir", default="artifacts/canonical_decode_audit")
    parser.add_argument("--audit-progress-interval", type=int, default=100)
    parser.add_argument("--audit-workers", type=int, default=8)
    parser.add_argument("--audit-camera-slots", default="main,left,right", help="Comma-separated slots, 'all', or 'config'.")
    parser.add_argument("--audit-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit-video-fetch-timeout-seconds", type=int, default=900)
    parser.add_argument("--audit-video-fetch-retries", type=int, default=1)
    parser.add_argument("--audit-video-fetch-retry-backoff-seconds", type=float, default=0.0)
    parser.add_argument(
        "--audit-abort-on-auth-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort instead of checkpointing gcloud reauth failures so they remain retryable after auth refresh.",
    )
    parser.add_argument(
        "--audit-delete-video-cache-after-processing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After a per-video result is checkpointed, delete the local cached MP4 under --cache-dir.",
    )
    parser.add_argument(
        "--audit-max-episode-checks",
        type=int,
        default=0,
        help="Debug-only cap on episode-camera checks. The default 0 audits all checks.",
    )
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
    if args.audit_corrupt_videos:
        _run_corrupt_video_audit(dataset, args)
        return
    indices = _sample_indices(len(dataset), args)
    _run_single_process_probe(dataset, indices)
    _run_dataloader_probe(dataset, args)


if __name__ == "__main__":
    main()
