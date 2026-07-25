from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from starVLA.dataloader.canonical_subset_dataset import (
    ACTION_DIM,
    JOINT_DELTA_GRIPPER_ABSOLUTE,
    STATE_DIM,
    CanonicalSubsetVLADataset,
    WindowSpec,
    _RecoverableSampleError,
)
from starVLA.action_representation import (
    CANONICAL_REALMAN_ACTION_SOURCE_INDICES,
    CANONICAL_REALMAN_STATE_SOURCE_INDICES,
    normalize_q01_q99_unclipped,
)


class _IndexOnlyCanonical(CanonicalSubsetVLADataset):
    def __init__(self, size: int, *, seed: int = 917) -> None:
        self.mode = "train"
        self.epoch_sampling_strategy = "all_sources_exhaustive"
        self.epoch_sampling_algorithm_version = (
            "all_sources_exhaustive_affine_v1"
        )
        self.seed = int(seed)
        self.total_windows = int(size)
        self.index_windows_lazily = False
        self.windows = [WindowSpec(0, 0, index) for index in range(size)]
        self.max_sample_decode_retries = 8
        self.skip_corrupt_videos = True
        self.fail_on_sample_error = True
        self.slow_sample_log_seconds = 0.0
        self._bad_video_paths = set()
        self._bad_video_warning_count = 0
        self.pyav_corrupt_warning_limit = 0
        self._initialize_epoch_schedule()

    def _sample_context(self, index: int):
        window = self.windows[int(index)]
        return {
            "window": window,
            "shard": SimpleNamespace(
                root=Path("/tmp"),
                dataset_id="synthetic",
                sid="s0",
                data_relative_path="synthetic.parquet",
            ),
            "episode": SimpleNamespace(length=len(self.windows)),
            "video_frames": {},
        }

    def _sample_from_context(self, context, decoded_frames=None):
        return {
            "window_index": int(context["window"].base_index),
            "epoch": self.current_epoch,
        }


class _FailingCanonical(_IndexOnlyCanonical):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.attempts = 0

    def _sample_context(self, index: int):
        self.attempts += 1
        raise _RecoverableSampleError(f"synthetic failure at {index}")


def _read_loader(loader: DataLoader) -> tuple[list[int], list[int]]:
    window_indices: list[int] = []
    epochs: list[int] = []
    for batch in loader:
        window_indices.extend(
            int(value) for value in batch["window_index"].tolist()
        )
        epochs.extend(int(value) for value in batch["epoch"].tolist())
    return window_indices, epochs


def test_all_sources_epoch_visits_every_canonical_window_once() -> None:
    dataset = _IndexOnlyCanonical(101)

    epoch_zero = [dataset[index]["window_index"] for index in range(len(dataset))]
    dataset.set_epoch(1)
    epoch_one = [dataset[index]["window_index"] for index in range(len(dataset))]

    assert dataset.epoch_sampling_algorithm_version == (
        "all_sources_exhaustive_affine_v1"
    )
    np.testing.assert_array_equal(dataset.dataset_lengths, [101])
    np.testing.assert_array_equal(dataset.primary_dataset_indices, [True])
    np.testing.assert_array_equal(dataset.epoch_dataset_counts, [101])
    np.testing.assert_array_equal(
        dataset._raw_dataset_sampling_weights, [1.0]
    )
    assert sorted(epoch_zero) == list(range(101))
    assert sorted(epoch_one) == list(range(101))
    assert epoch_zero != epoch_one


def test_persistent_spawn_worker_sees_canonical_epoch_change() -> None:
    dataset = _IndexOnlyCanonical(41)
    loader = DataLoader(
        dataset,
        batch_size=7,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    try:
        epoch_zero_indices, epoch_zero_values = _read_loader(loader)
        dataset.set_epoch(2)
        epoch_two_indices, epoch_two_values = _read_loader(loader)
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()

    assert set(epoch_zero_values) == {0}
    assert set(epoch_two_values) == {2}
    assert sorted(epoch_zero_indices) == list(range(41))
    assert sorted(epoch_two_indices) == list(range(41))
    assert epoch_zero_indices != epoch_two_indices


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_shards", 1),
        ("max_shards_per_dataset", 1),
        ("max_windows", 10),
        ("max_windows_per_dataset", 10),
        ("shuffle_shards", True),
        ("sample_stride", 2),
    ),
)
def test_exact_epoch_rejects_window_subsampling(
    field: str,
    value,
) -> None:
    settings = {
        "mode": "train",
        "epoch_sampling_strategy": "all_sources_exhaustive",
        "max_shards": 0,
        "max_shards_per_dataset": 0,
        "max_windows": 0,
        "max_windows_per_dataset": 0,
        "shuffle_shards": False,
        "sample_stride": 1,
    }
    settings[field] = value

    with pytest.raises(ValueError, match=field):
        CanonicalSubsetVLADataset._validate_exact_epoch_settings(**settings)


def test_exact_epoch_decode_error_never_substitutes_another_window() -> None:
    dataset = _FailingCanonical(17)

    with pytest.raises(
        RuntimeError,
        match="retry substitution is forbidden",
    ):
        dataset[3]

    assert dataset.attempts == 1


def test_canonical_shared_statistics_project_and_normalize_18d() -> None:
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.action_type = JOINT_DELTA_GRIPPER_ABSOLUTE
    dataset.action_horizon = 2
    dataset.video_horizon = 1
    dataset.video_frame_stride = 1
    dataset.video_target_shift_steps = 0
    dataset._compact_offsets_cache = None
    dataset.data_cfg = {"append_subtask_to_prompt": False}
    dataset.normalization_statistics = {
        "selected": {
            "state": {"q01": [-1.0] * 18, "q99": [1.0] * 18},
            "action": {"q01": [-1.0] * 18, "q99": [1.0] * 18},
        }
    }

    policy_state = np.linspace(-0.4, 0.4, 18, dtype=np.float32)
    policy_state[16:18] = 0.0
    policy_action = np.stack(
        [policy_state + np.float32(0.2), policy_state + np.float32(0.4)]
    )
    policy_action[:, 7] = [0.25, 0.75]
    policy_action[:, 15] = [0.8, 0.2]
    policy_action[:, 16:18] = 0.0

    state = np.zeros((2, STATE_DIM), dtype=np.float32)
    action = np.zeros((2, ACTION_DIM), dtype=np.float32)
    state_mask = np.zeros_like(state, dtype=bool)
    action_mask = np.zeros_like(action, dtype=bool)
    state_indices = np.asarray(CANONICAL_REALMAN_STATE_SOURCE_INDICES)
    action_indices = np.asarray(CANONICAL_REALMAN_ACTION_SOURCE_INDICES)
    state[:, state_indices] = policy_state
    action[:, action_indices] = policy_action
    state_mask[:, state_indices] = True
    action_mask[:, action_indices] = True
    state_mask[:, state_indices[16:18]] = False
    action_mask[:, action_indices[16:18]] = False

    delta_mask = np.zeros((ACTION_DIM,), dtype=bool)
    delta_mask[action_indices] = True
    delta_mask[action_indices[[7, 15]]] = False
    action_to_state = np.full((ACTION_DIM,), -1, dtype=np.int64)
    for output_index, (action_index, state_index) in enumerate(
        zip(action_indices, state_indices, strict=True)
    ):
        if output_index not in {7, 15}:
            action_to_state[action_index] = state_index

    shard_data = SimpleNamespace(
        state=state,
        state_mask=state_mask,
        action=action,
        action_mask=action_mask,
        action_delta_mask=delta_mask,
        action_to_state_indices=action_to_state,
        timestamp=np.asarray([0.0, 0.05], dtype=np.float32),
        episode_index=np.asarray([3, 3], dtype=np.int64),
        frame_index=np.asarray([0, 1], dtype=np.int64),
        task_index=np.asarray([9, 9], dtype=np.int64),
    )
    shard = SimpleNamespace(
        dataset_id="synthetic_realman",
        vjepa_camera_slots=("main",),
        qwen_camera_slots=("main",),
    )
    episode = SimpleNamespace(
        local_start=0,
        length=2,
        task="move the object",
        subtask_spans=(),
    )
    context = {
        "shard_data": shard_data,
        "shard": shard,
        "episode": episode,
        "row_base": 0,
        "action_rows": np.asarray([0, 1], dtype=np.int64),
        "action_is_pad": np.asarray([False, False]),
        "window": WindowSpec(0, 0, 0),
        "video_frames": {
            "main": (
                Path("/tmp/synthetic.mp4"),
                np.asarray([0], dtype=np.int64),
                Path("/tmp/synthetic.lock"),
            )
        },
        "qwen_frame_positions": {"main": 0},
    }
    dataset._decode_episode_video = lambda _shard, _path, indices, _lock: (
        np.zeros((len(indices), 2, 2, 3), dtype=np.uint8)
    )

    sample = dataset._sample_from_context(context)

    expected_state = normalize_q01_q99_unclipped(
        policy_state[None, :],
        dataset.normalization_statistics["selected"]["state"],
    )
    mixed_action = policy_action.copy()
    mixed_action[:, [*range(0, 7), *range(8, 15)]] -= policy_state[
        [*range(0, 7), *range(8, 15)]
    ]
    expected_action = normalize_q01_q99_unclipped(
        mixed_action,
        dataset.normalization_statistics["selected"]["action"],
    )
    expected_state[:, 16:18] = 0.0
    expected_action[:, 16:18] = 0.0

    assert sample["state"].shape == (1, 18)
    assert sample["action"].shape == (2, 18)
    assert sample["state_mask"].shape == (1, 18)
    assert sample["action_mask"].shape == (2, 18)
    assert not sample["state_mask"][:, 16:18].any()
    assert not sample["action_mask"][:, 16:18].any()
    np.testing.assert_allclose(sample["state"], expected_state, atol=1e-6)
    np.testing.assert_allclose(sample["action"], expected_action, atol=1e-6)
