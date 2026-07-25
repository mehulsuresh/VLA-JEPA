from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.gr00t_lerobot import datasets as datasets_module
from starVLA.dataloader.gr00t_lerobot.datasets import (
    GPU_DECODE_FRAME_INDEX_CACHE_DIRNAME,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
)


CAMERA_KEY = "observation.images.head"


def _video_modality_metadata():
    return SimpleNamespace(
        video={"base_view": SimpleNamespace(original_key=CAMERA_KEY)}
    )


def _minimal_video_dataset(tmp_path: Path) -> LeRobotSingleDataset:
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = tmp_path
    dataset._lerobot_version = "v3.0"
    dataset._video_path_pattern = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    dataset._chunk_size = 1000
    dataset._lerobot_modality_meta = _video_modality_metadata()
    dataset._trajectory_ids = np.asarray([42], dtype=np.int64)
    dataset._trajectory_lengths = np.asarray([3], dtype=np.int64)
    dataset._modality_keys = {"video": ["video.base_view"]}
    dataset._delta_indices = {"video.base_view": np.asarray([0], dtype=np.int64)}
    dataset.video_backend = "pyav"
    dataset.video_backend_kwargs = {"num_threads": 1}
    dataset.curr_traj_data = pd.DataFrame(
        {"timestamp": np.asarray([0.0, 0.05, 0.10], dtype=np.float64)}
    )
    dataset.trajectory_ids_to_metadata = {
        42: {
            "data/chunk_index": 0,
            "data/file_index": 2,
            "videos": {
                CAMERA_KEY: {
                    "chunk_index": 3,
                    "file_index": 7,
                    "from_timestamp": 12.5,
                    "to_timestamp": 12.65,
                }
            },
            "videos/from_timestamps": {CAMERA_KEY: 12.5},
        }
    }
    return dataset


def test_lerobot_v3_parser_preserves_per_camera_shard_and_timestamp(tmp_path):
    episodes_dir = tmp_path / "meta/episodes/chunk-000"
    episodes_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [42],
            "length": [3],
            "data/chunk_index": [0],
            "data/file_index": [2],
            "dataset_from_index": [100],
            "dataset_to_index": [103],
            f"videos/{CAMERA_KEY}/chunk_index": [3],
            f"videos/{CAMERA_KEY}/file_index": [7],
            f"videos/{CAMERA_KEY}/from_timestamp": [12.5],
            f"videos/{CAMERA_KEY}/to_timestamp": [12.65],
        }
    ).to_parquet(episodes_dir / "file-000.parquet")

    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = tmp_path
    dataset._lerobot_version = "v3.0"
    dataset._v3_data_file_start_indices = {}
    dataset._video_path_pattern = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    dataset._chunk_size = 1000
    dataset._lerobot_modality_meta = _video_modality_metadata()

    trajectory_ids, trajectory_lengths = dataset._get_trajectories()

    np.testing.assert_array_equal(trajectory_ids, [42])
    np.testing.assert_array_equal(trajectory_lengths, [3])
    assert dataset.trajectory_ids_to_metadata[42]["videos"][CAMERA_KEY] == {
        "chunk_index": 3,
        "file_index": 7,
        "from_timestamp": 12.5,
        "to_timestamp": 12.65,
    }
    assert dataset.get_video_path(42, "base_view") == (
        tmp_path / "videos/observation.images.head/chunk-003/file-007.mp4"
    )


def test_lerobot_v3_compact_cpu_decode_uses_camera_shard_and_offset(
    tmp_path,
    monkeypatch,
):
    dataset = _minimal_video_dataset(tmp_path)
    calls = []

    def fake_get_frames(path, timestamps, **kwargs):
        calls.append((path, np.asarray(timestamps).copy(), kwargs))
        return np.zeros((2, 4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(datasets_module, "get_frames_by_timestamps", fake_get_frames)

    result = dataset.get_video_by_step_indices(
        42,
        "video.base_view",
        np.asarray([0, 2], dtype=np.int64),
    )

    assert result.shape == (2, 4, 4, 3)
    assert calls[0][0] == str(
        tmp_path / "videos/observation.images.head/chunk-003/file-007.mp4"
    )
    np.testing.assert_allclose(calls[0][1], [12.5, 12.6])


def test_lerobot_v3_gpu_decode_spec_uses_camera_shard_and_offset(tmp_path):
    dataset = _minimal_video_dataset(tmp_path)
    mixture = object.__new__(LeRobotMixtureDataset)
    mixture.video_frame_stride = 1
    mixture.video_resolution_size = 384
    mixture.resolution_size = 224
    mixture.video_target_shift_steps = 2

    specs = mixture._build_video_decode_specs(
        dataset,
        trajectory_name=42,
        step=1,
        step_offsets=np.asarray([-1, 0, 1], dtype=np.int64),
    )

    assert specs == [
        {
            "video_path": str(
                tmp_path / "videos/observation.images.head/chunk-003/file-007.mp4"
            ),
            "timestamps": pytest.approx(np.asarray([12.5, 12.55, 12.6])),
        }
    ]


def test_lerobot_v3_missing_camera_binding_fails_closed(tmp_path):
    dataset = _minimal_video_dataset(tmp_path)
    dataset.trajectory_ids_to_metadata[42]["videos"] = {}

    with pytest.raises(KeyError, match="missing camera binding"):
        dataset.get_video_path(42, "base_view")


def test_gpu_decode_cache_version_invalidates_legacy_wrong_path_indices():
    assert GPU_DECODE_FRAME_INDEX_CACHE_DIRNAME == "gpu_decode_frame_indices_v2"
