import json

import numpy as np

from starVLA.dataloader.preprocessed_subtask_dataset import PreprocessedSubtaskVLADataset


def _write_episode(root, labels=None):
    episode_dir = root / "episode_000000"
    episode_dir.mkdir(parents=True)
    metadata = {
        "frame_indices": [0, 1, 2],
        "frame_features": {
            "action": [
                [0.0, 0.1],
                [1.0, 1.1],
                [2.0, 2.1],
            ],
            "observation.state": [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
            ],
        },
        "labels": labels or {},
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_preprocessed_dataset_accepts_missing_rabc_labels(tmp_path):
    _write_episode(tmp_path)

    dataset = PreprocessedSubtaskVLADataset(
        tmp_path,
        action_horizon=2,
        video_horizon=3,
        video_target_shift_steps=1,
        resolution_size=4,
        video_resolution_size=4,
        current_cameras=["observation.images.cam_high"],
        frame_cache_size=0,
    )

    sample = dataset[0]

    assert len(dataset) == 3
    assert sample["task_id"] == 0
    assert sample["future_task_id"] == 0
    assert sample["mistake_label"] == 0.0
    assert sample["future_mistake_label"] == 0.0
    assert np.isnan(sample["global_complexity_to_go"])
    assert np.isnan(sample["local_complexity_to_go"])
    assert np.isnan(sample["rabc_global_progress_delta"])
    assert np.isnan(sample["rabc_progress_delta"])
    assert sample["action"].shape == (2, 2)
    assert sample["state"].shape == (1, 3)
    assert sample["video_compact"].shape == (2, 3, 4, 4, 3)


def test_preprocessed_dataset_statistics_report_optional_label_coverage(tmp_path):
    _write_episode(tmp_path)
    stats_path = tmp_path / "stats" / "dataset_statistics.json"

    dataset = PreprocessedSubtaskVLADataset(
        tmp_path,
        action_horizon=2,
        video_horizon=3,
        video_target_shift_steps=1,
        current_cameras=["observation.images.cam_high"],
        frame_cache_size=0,
    )
    dataset.save_dataset_statistics(stats_path)

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["unique_task_ids"] == []
    assert stats["mistake_positive_count"] == 0
    assert stats["episodes_with_task_labels"] == 0
    assert stats["episodes_with_mistake_labels"] == 0
    assert stats["episodes_with_global_complexity_to_go"] == 0
    assert stats["episodes_with_local_complexity_to_go"] == 0
