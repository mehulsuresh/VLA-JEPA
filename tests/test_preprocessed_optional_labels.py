import json

import numpy as np

from starVLA.dataloader.prompt_labels import (
    append_resolved_label_to_language,
    append_task_id_label_to_language,
)
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


def _write_long_episode(root, *, num_frames, labels=None):
    episode_dir = root / "episode_000000"
    episode_dir.mkdir(parents=True)
    metadata = {
        "frame_indices": list(range(num_frames)),
        "frame_features": {
            "action": [[float(i), float(i) + 0.1] for i in range(num_frames)],
            "observation.state": [
                [float(i), float(i) + 0.1, float(i) + 0.2] for i in range(num_frames)
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
    assert sample["action_is_pad"].tolist() == [False, False]
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


def test_task_id_prompt_label_helper_appends_text_label_once():
    cfg = {
        "append_task_id_to_prompt": True,
        "task_id_prompt_separator": " | ",
        "task_id_label_map": {
            "3": "firmly grasp the item",
        },
    }

    language, label = append_task_id_label_to_language("Put the shirt in the other bin", 3, cfg)
    assert label == "firmly grasp the item"
    assert language == "Put the shirt in the other bin | firmly grasp the item"

    language, label = append_task_id_label_to_language(language, 3, cfg)
    assert label == "firmly grasp the item"
    assert language == "Put the shirt in the other bin | firmly grasp the item"


def test_task_id_prompt_label_helper_respects_append_probability():
    cfg = {
        "append_task_id_to_prompt": True,
        "task_id_prompt_append_probability": 0.0,
        "task_id_prompt_separator": " | ",
        "task_id_label_map": {
            "3": "firmly grasp the item",
        },
    }

    language, label = append_task_id_label_to_language("Put the shirt in the other bin", 3, cfg)

    assert label is None
    assert language == "Put the shirt in the other bin"


def test_resolved_prompt_label_helper_skips_unlabeled_sentinel():
    language, label = append_resolved_label_to_language(
        "Put the chain in the jig",
        "__unlabeled__",
        {
            "append_task_id_to_prompt": True,
            "task_id_prompt_append_probability": 1.0,
        },
    )

    assert label is None
    assert language == "Put the chain in the jig"


def test_resolved_prompt_label_helper_skips_missing_label():
    language, label = append_resolved_label_to_language(
        "Put the chain in the jig",
        None,
        {
            "append_task_id_to_prompt": True,
            "task_id_prompt_append_probability": 1.0,
        },
    )

    assert label is None
    assert language == "Put the chain in the jig"


def test_preprocessed_dataset_appends_task_id_label_to_prompt(tmp_path):
    _write_episode(
        tmp_path,
        labels={
            "subtask_id": [2, 2, 3],
            "mistake_label": [1.0, 1.0, 1.0],
        },
    )

    dataset = PreprocessedSubtaskVLADataset(
        tmp_path,
        action_horizon=2,
        video_horizon=3,
        video_target_shift_steps=1,
        resolution_size=4,
        video_resolution_size=4,
        current_cameras=["observation.images.cam_high"],
        frame_cache_size=0,
        data_cfg={
            "append_task_id_to_prompt": True,
            "task_id_prompt_separator": " | ",
            "task_id_label_map": {
                2: "reach hand inside",
                3: "firmly grasp the item",
            },
        },
    )

    sample = dataset[0]

    assert sample["task_id"] == 2
    assert sample["task_id_label"] == "reach hand inside"
    assert sample["lang"] == "Complete the task successfully. | reach hand inside"


def test_preprocessed_dataset_marks_clamped_tail_actions_as_pad(tmp_path):
    _write_episode(tmp_path)

    dataset = PreprocessedSubtaskVLADataset(
        tmp_path,
        action_horizon=4,
        video_horizon=3,
        video_target_shift_steps=1,
        resolution_size=4,
        video_resolution_size=4,
        current_cameras=["observation.images.cam_high"],
        frame_cache_size=0,
    )

    sample = dataset[2]

    np.testing.assert_allclose(
        sample["action"],
        np.asarray(
            [
                [2.0, 2.1],
                [2.0, 2.1],
                [2.0, 2.1],
                [2.0, 2.1],
            ],
            dtype=np.float32,
        ),
    )
    assert sample["action_is_pad"].tolist() == [False, True, True, True]


def test_preprocessed_dataset_builds_action_validity_prefix_mask(tmp_path):
    _write_long_episode(
        tmp_path,
        num_frames=6,
        labels={
            # Raw preprocessed labels use 1 = ok, so the dataset inverts them
            # internally to 1 = mistake before building the action mask.
            "mistake_label": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
    )

    dataset = PreprocessedSubtaskVLADataset(
        tmp_path,
        action_horizon=6,
        video_horizon=3,
        video_target_shift_steps=1,
        resolution_size=4,
        video_resolution_size=4,
        current_cameras=["observation.images.cam_high"],
        frame_cache_size=0,
        data_cfg={
            "use_action_validity_prefix_mask": True,
            "action_validity_invalid_run_length": 3,
        },
    )

    sample = dataset[0]

    assert sample["action_mask"].shape == (6, 2)
    assert sample["action_mask"].astype(int).tolist() == [
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
