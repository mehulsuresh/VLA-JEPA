#!/usr/bin/env python3
"""
Dataset integrity checker for LeRobot-format datasets.

Checks:
  1. All expected parquet files exist and are readable
  2. All expected video files exist and are non-zero size
  3. Frame counts in parquet match metadata total_frames
  4. No NaN/Inf in action or state columns
  5. Video files can be opened (header check via av/decord)

Usage:
    python scripts/check_dataset.py [--data-root <path>] [--workers <n>]
"""

import argparse
import json
import math
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[1] / "playground/Datasets/TROSSEN_SUBTASK_COMBINED"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_meta(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def episode_parquet_path(root: Path, meta: dict, idx: int) -> Path:
    chunk = idx // meta["chunks_size"]
    template = meta["data_path"]
    rel = template.format(episode_chunk=chunk, episode_index=idx)
    return root / rel


def episode_video_paths(root: Path, meta: dict, idx: int) -> list[Path]:
    chunk = idx // meta["chunks_size"]
    template = meta["video_path"]
    paths = []
    for key in meta.get("features", {}):
        feat = meta["features"][key]
        if feat.get("dtype") == "video":
            rel = template.format(episode_chunk=chunk, video_key=key, episode_index=idx)
            paths.append(root / rel)
    return paths


def check_parquet(path: Path) -> tuple[int, list[str]]:
    """Return (frame_count, list_of_errors)."""
    errors = []
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(path))
        df = table.to_pydict()
        n_rows = len(next(iter(df.values())))

        for col in ("action", "observation.state"):
            if col not in df:
                continue
            arr = np.array(df[col], dtype=np.float32)
            if np.isnan(arr).any():
                errors.append(f"NaN in column '{col}'")
            if np.isinf(arr).any():
                errors.append(f"Inf in column '{col}'")
        return n_rows, errors
    except Exception as e:
        return 0, [f"Cannot read parquet: {e}"]


def check_video(path: Path) -> list[str]:
    """Try opening the video container (fast header-only check)."""
    errors = []
    if not path.exists():
        return [f"Missing: {path.name}"]
    if path.stat().st_size == 0:
        return [f"Zero-byte: {path.name}"]
    try:
        import av
        with av.open(str(path)) as container:
            if not container.streams.video:
                errors.append(f"No video stream: {path.name}")
    except Exception as e:
        errors.append(f"Cannot open video {path.name}: {e}")
    return errors


def check_episode(root: Path, meta: dict, idx: int) -> tuple[int, list[str]]:
    all_errors = []

    pq_path = episode_parquet_path(root, meta, idx)
    if not pq_path.exists():
        all_errors.append(f"[ep {idx:06d}] Missing parquet: {pq_path}")
        return 0, all_errors

    n_frames, pq_errors = check_parquet(pq_path)
    for e in pq_errors:
        all_errors.append(f"[ep {idx:06d}] Parquet: {e}")

    for vpath in episode_video_paths(root, meta, idx):
        for e in check_video(vpath):
            all_errors.append(f"[ep {idx:06d}] Video: {e}")

    return n_frames, all_errors


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Check LeRobot dataset integrity")
    parser.add_argument("--data-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel worker threads (default: 16)")
    parser.add_argument("--stop-on-first-error", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    meta = load_meta(root)
    total_episodes = meta["total_episodes"]
    expected_frames = meta["total_frames"]
    total_videos = meta.get("total_videos", "?")

    print(f"Dataset : {root}")
    print(f"Episodes: {total_episodes}  |  Expected frames: {expected_frames:,}  |  Videos: {total_videos}")
    print(f"Workers : {args.workers}")
    print()

    all_errors: list[str] = []
    counted_frames = 0
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(check_episode, root, meta, idx): idx
            for idx in range(total_episodes)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                n_frames, errors = fut.result()
            except Exception as e:
                errors = [f"[ep {idx:06d}] Unexpected exception: {e}"]
                n_frames = 0

            counted_frames += n_frames
            all_errors.extend(errors)
            done += 1

            if done % 100 == 0 or done == total_episodes:
                pct = 100 * done / total_episodes
                print(f"  {done:4d}/{total_episodes}  ({pct:5.1f}%)  errors so far: {len(all_errors)}", flush=True)

            if args.stop_on_first_error and all_errors:
                pool.shutdown(wait=False, cancel_futures=True)
                break

    print()
    print("=" * 60)
    if all_errors:
        print(f"FAILED — {len(all_errors)} error(s) found:\n")
        for e in all_errors:
            print(f"  {e}")
        print()

    frame_ok = counted_frames == expected_frames
    print(f"Frame count : counted={counted_frames:,}  expected={expected_frames:,}  {'✓ OK' if frame_ok else '✗ MISMATCH'}")
    print(f"Total errors: {len(all_errors)}")

    if all_errors or not frame_ok:
        sys.exit(1)
    else:
        print("\nAll checks passed ✓")


if __name__ == "__main__":
    main()
