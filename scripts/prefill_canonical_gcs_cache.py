#!/usr/bin/env python3
"""Warm the canonical GCS subset cache for a training config.

This intentionally uses the repo's canonical_subset_vla dataloader, so it
exercises the same manifest, adapter, GCS cache, and sidecar path as training.
Run it once before multi-rank launch when you want all ranks to start from an
already-warm shared cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.dataloader.canonical_subset_dataset import get_vla_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-yaml", required=True, type=Path)
    parser.add_argument(
        "--touch-sample",
        action="store_true",
        help="Also decode the first sample after caching shard files and sidecars.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    data_cfg = cfg.datasets.vla_data
    if data_cfg.dataset_py != "canonical_subset_vla":
        raise ValueError(f"Expected canonical_subset_vla, got {data_cfg.dataset_py!r}")

    dataset = get_vla_dataset(
        data_cfg,
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        video_frame_stride=data_cfg.get("video_frame_stride", 1),
    )

    print(f"cached canonical subset: windows={len(dataset)} shards={len(dataset.shards)}")
    for shard in dataset.shards:
        print(
            f"- {shard.dataset_id} | {shard.data_relative_path} | episodes={len(shard.episodes)}"
        )

    if args.touch_sample:
        sample = dataset[0]
        print(
            "decoded sample: "
            f"video_compact={sample['video_compact'].shape} "
            f"state={sample['state'].shape} action={sample['action'].shape}"
        )


if __name__ == "__main__":
    main()
