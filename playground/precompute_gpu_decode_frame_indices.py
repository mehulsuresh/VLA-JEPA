#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.datasets import (
    GPU_DECODE_FRAME_INDEX_CACHE_DIRNAME,
    get_gpu_decode_frame_index_cache_path,
)
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def _nearest_frame_indices(frame_timestamps: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(frame_timestamps, target_timestamps, side="left")
    positions = np.clip(positions, 0, len(frame_timestamps) - 1)
    prev_positions = np.clip(positions - 1, 0, len(frame_timestamps) - 1)
    use_prev = np.abs(frame_timestamps[prev_positions] - target_timestamps) <= np.abs(
        frame_timestamps[positions] - target_timestamps
    )
    return np.where(use_prev, prev_positions, positions).astype(np.int32)


def _build_episode_cache(single_dataset, trajectory_id: int) -> dict[str, np.ndarray]:
    trajectory_id = int(trajectory_id)
    trajectory_data = single_dataset.get_trajectory_data(trajectory_id)
    single_dataset.curr_traj_id = trajectory_id
    single_dataset.curr_traj_data = trajectory_data
    episode_timestamps = trajectory_data["timestamp"].to_numpy(dtype=np.float64)

    arrays: dict[str, np.ndarray] = {
        "__length__": np.asarray([len(episode_timestamps)], dtype=np.int32),
        "__trajectory_id__": np.asarray([trajectory_id], dtype=np.int32),
    }

    episode_meta = {}
    if getattr(single_dataset, "_lerobot_version", None) == "v3.0":
        episode_meta = single_dataset.trajectory_ids_to_metadata.get(trajectory_id, {})
    from_timestamps = episode_meta.get("videos/from_timestamps", {})

    for video_key in single_dataset.modality_keys["video"]:
        video_subkey = video_key.replace("video.", "")
        original_key = single_dataset.lerobot_modality_meta.video[video_subkey].original_key
        if original_key is None:
            original_key = video_subkey

        video_path = single_dataset.get_video_path(trajectory_id, video_subkey)
        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        frame_timestamps = np.asarray(
            reader.get_frame_timestamp(range(len(reader)))[:, 0],
            dtype=np.float64,
        )
        target_timestamps = episode_timestamps + float(from_timestamps.get(original_key, 0.0))
        arrays[video_subkey] = _nearest_frame_indices(frame_timestamps, target_timestamps)

    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute per-episode video frame indices for gpu_video_decode_on_rank.")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="/home/mehul_yonduai_com/work/VLA-JEPA/scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_a100x4_weekend_20260404.yaml",
        help="Training config used to construct the LeRobot dataset mixture.",
    )
    parser.add_argument(
        "--max_episodes_per_dataset",
        type=int,
        default=None,
        help="Optional cap for smoke runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite existing cache files.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    if str(cfg.datasets.vla_data.dataset_py) != "lerobot_datasets":
        raise ValueError(
            f"Expected lerobot_datasets config, got {cfg.datasets.vla_data.dataset_py!r}"
        )

    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        video_frame_stride=cfg.datasets.vla_data.get("video_frame_stride", 1),
    )

    for single_dataset in dataset.datasets:
        cache_dir = single_dataset.dataset_path / "meta" / GPU_DECODE_FRAME_INDEX_CACHE_DIRNAME
        cache_dir.mkdir(parents=True, exist_ok=True)
        trajectory_ids = [int(tid) for tid in single_dataset.trajectory_ids]
        if args.max_episodes_per_dataset is not None:
            trajectory_ids = trajectory_ids[: int(args.max_episodes_per_dataset)]

        progress = tqdm(
            trajectory_ids,
            desc=f"Precomputing {single_dataset.dataset_name}",
        )
        for trajectory_id in progress:
            cache_path = get_gpu_decode_frame_index_cache_path(
                single_dataset.dataset_path, trajectory_id
            )
            if cache_path.exists() and not args.force:
                continue
            arrays = _build_episode_cache(single_dataset, trajectory_id)
            np.savez_compressed(cache_path, **arrays)


if __name__ == "__main__":
    main()
