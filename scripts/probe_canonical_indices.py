#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from starVLA.dataloader.canonical_subset_dataset import get_vla_dataset


def _video_paths_for_window(dataset: Any, window: Any, limit: int = 8) -> dict[str, str]:
    shard = dataset.shards[window.shard_index]
    episode = shard.episodes[window.episode_index]
    paths = {}
    for slot in shard.decode_camera_slots[:limit]:
        try:
            paths[slot] = episode.video_paths[slot].relative_to(shard.root).as_posix()
        except ValueError:
            paths[slot] = episode.video_paths[slot].as_posix()
    return paths


def _sample_rows(dataset: Any, indices: list[int]) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        window = dataset._window_from_index(index) if dataset.index_windows_lazily else dataset.windows[index]
        shard = dataset.shards[window.shard_index]
        episode = shard.episodes[window.episode_index]
        rows.append(
            {
                "index": int(index),
                "shard_index": int(window.shard_index),
                "episode_index": int(window.episode_index),
                "base_index": int(window.base_index),
                "dataset_id": shard.dataset_id,
                "sid": shard.sid,
                "data_file": shard.data_relative_path,
                "episode_length": int(episode.length),
                "videos": _video_paths_for_window(dataset, window),
            }
        )
    return rows


def _decode_batch(dataset: Any, indices: list[int]) -> dict[str, Any]:
    import time

    start = time.monotonic()
    samples = dataset.__getitems__([index % len(dataset) for index in indices])
    elapsed = time.monotonic() - start
    dataset_ids = sorted({str(sample.get("dataset_id")) for sample in samples})
    return {
        "decoded_batch_size": len(samples),
        "decode_elapsed_sec": round(elapsed, 4),
        "dataset_ids": dataset_ids,
        "first_samples": [
            {
                "dataset_id": sample.get("dataset_id"),
                "episode_index": int(sample.get("episode_index", -1)),
                "frame_index": int(sample.get("frame_index", -1)),
            }
            for sample in samples[: min(4, len(samples))]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Map canonical dataset indices to shards/videos without decoding.")
    parser.add_argument("--config-yaml", default="scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml")
    parser.add_argument("--indices", default="", help="Comma-separated dataset indices.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--raw-batch-start", type=int, default=None)
    parser.add_argument("--raw-batch-count", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rank-step", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--ranks", default="")
    parser.add_argument("--samples-per-batch", type=int, default=4)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    data_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(
        data_cfg,
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        video_frame_stride=data_cfg.video_frame_stride,
    )
    batch_size = int(args.batch_size or data_cfg.per_device_batch_size)
    print(
        json.dumps(
            {
                "dataset_len": len(dataset),
                "batch_size": batch_size,
                "num_shards": len(dataset.shards),
                "exclude_sids": dataset.exclude_sid_list,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    indices: list[int] = []
    if args.indices:
        indices.extend(int(value.strip()) for value in args.indices.split(",") if value.strip())
    if args.count > 0:
        indices.extend(args.start_index + offset * args.stride for offset in range(args.count))

    for row in _sample_rows(dataset, [index % len(dataset) for index in indices]):
        print(json.dumps({"kind": "index", **row}, sort_keys=True), flush=True)

    if args.raw_batch_start is not None and args.raw_batch_count > 0:
        for batch_index in range(args.raw_batch_start, args.raw_batch_start + args.raw_batch_count):
            batch_first = batch_index * batch_size
            sample_count = min(max(args.samples_per_batch, 1), batch_size)
            rows = _sample_rows(dataset, [(batch_first + offset) % len(dataset) for offset in range(sample_count)])
            decode_payload = {}
            if args.decode:
                decode_payload = _decode_batch(dataset, [batch_first + offset for offset in range(batch_size)])
            print(
                json.dumps(
                    {
                        "kind": "raw_batch",
                        "raw_batch_index": batch_index,
                        "first_index": batch_first,
                        "samples": rows,
                        **decode_payload,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if args.rank_step is not None:
        ranks = (
            [int(value.strip()) for value in args.ranks.split(",") if value.strip()]
            if args.ranks
            else list(range(args.world_size))
        )
        for rank in ranks:
            raw_batch = args.rank_step * args.world_size + rank
            batch_first = raw_batch * batch_size
            sample_count = min(max(args.samples_per_batch, 1), batch_size)
            rows = _sample_rows(dataset, [(batch_first + offset) % len(dataset) for offset in range(sample_count)])
            decode_payload = {}
            if args.decode:
                decode_payload = _decode_batch(dataset, [batch_first + offset for offset in range(batch_size)])
            print(
                json.dumps(
                    {
                        "kind": "rank_step",
                        "step": args.rank_step,
                        "rank": rank,
                        "raw_batch_index": raw_batch,
                        "first_index": batch_first,
                        "samples": rows,
                        **decode_payload,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
