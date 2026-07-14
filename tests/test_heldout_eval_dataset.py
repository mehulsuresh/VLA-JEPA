import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import BatchSampler, SequentialSampler
from accelerate.data_loader import BatchSamplerShard
from omegaconf import OmegaConf

from starVLA.dataloader import _build_eval_lerobot_data_cfg
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.heldout_eval import (
    DeterministicHeldoutEvalDataset,
    _candidate_control_diagnostics,
    load_evaluation_sampling_contract,
    required_window_offsets,
    validate_global_eval_observation_count,
)


class _SplitSelection(SimpleNamespace):
    def provenance(self):
        return {
            "manifest_path": "/fixture/split.json",
            "manifest_sha256": "1" * 64,
            "role": self.role,
            "selected_episode_count": len(self.selected_episode_ids),
            "selected_episode_set_sha256": "2" * 64,
            "selected_frame_count": 200,
            "train_episode_count": 10,
            "train_episode_set_sha256": "3" * 64,
            "train_frame_count": 1000,
            "holdout_episode_count": len(self.selected_episode_ids),
            "holdout_episode_set_sha256": "2" * 64,
            "full_catalog_sha256": "4" * 64,
            "train_statistics_path": "/fixture/train_stats.json",
            "train_statistics_sha256": "5" * 64,
        }


class _EpisodeDataset:
    dataset_name = "fixture"

    def __init__(self):
        self.trajectory_ids = np.asarray([7, 9], dtype=np.int64)
        self.trajectory_lengths = np.asarray([100, 100], dtype=np.int64)
        self.all_steps = [
            (episode_id, step)
            for episode_id in self.trajectory_ids.tolist()
            for step in range(100)
        ]
        self.modality_keys = {
            "video": ["video.cam"],
            "state": ["state.x"],
            "action": ["action.x"],
            "language": ["language.task"],
        }
        self.delta_indices = {
            "video.cam": np.arange(8, dtype=np.int64) * 10,
            "state.x": np.arange(8, dtype=np.int64) * 10,
            "action.x": np.arange(5, dtype=np.int64),
            "language.task": np.arange(8, dtype=np.int64) * 10,
        }
        self.data_cfg = {
            "use_action_validity_prefix_mask": True,
            "action_validity_label_key": "valid_state",
            "action_validity_positive_is_valid": True,
            "action_validity_invalid_run_length": 3,
        }
        self._episode_split_selection = _SplitSelection(
            role="eval",
            selected_episode_ids=(7, 9),
        )
        self._frames = {
            7: pd.DataFrame({"valid_state": np.ones(100, dtype=np.float32)}),
            # Preserve an all-invalid manifest-random episode in the forward
            # coverage while excluding it from action metrics.
            9: pd.DataFrame({"valid_state": np.zeros(100, dtype=np.float32)}),
        }
        self.curr_traj_data = None
        self.curr_traj_id = None
        self.parquet_cache_close_count = 0

    def get_trajectory_data(self, episode_id):
        self.curr_traj_id = int(episode_id)
        self.curr_traj_data = self._frames[int(episode_id)]
        return self.curr_traj_data

    def close_parquet_cache(self):
        self.parquet_cache_close_count += 1
        self.curr_traj_id = None
        self.curr_traj_data = None


class _FocusedEpisodeDataset(_EpisodeDataset):
    def __init__(self):
        super().__init__()
        self.trajectory_lengths = np.asarray([70, 70], dtype=np.int64)
        self.all_steps = [
            (episode_id, step)
            for episode_id in self.trajectory_ids.tolist()
            for step in range(70)
        ]
        self.modality_keys = {
            "video": ["video.cam"],
            "state": ["state.source"],
            "action": ["action.source_controls"],
            "language": ["language.task"],
        }
        self.delta_indices = {
            "video.cam": np.asarray([0]),
            "state.source": np.asarray([0]),
            "action.source_controls": np.arange(50, dtype=np.int64),
            "language.task": np.asarray([0]),
        }
        self.lerobot_modality_meta = SimpleNamespace(
            action={
                "source_controls": SimpleNamespace(
                    original_key="source.action", start=0, end=18
                )
            },
            state={
                "source": SimpleNamespace(
                    original_key="source.observation.state", start=0, end=19
                )
            },
        )
        self.metadata = SimpleNamespace(
            modalities=SimpleNamespace(
                action={
                    "source_controls": SimpleNamespace(shape=(18,))
                }
            ),
            statistics=SimpleNamespace(
                action={
                    "source_controls": SimpleNamespace(
                        min=np.zeros(18, dtype=np.float32),
                        max=np.ones(18, dtype=np.float32),
                    )
                }
            )
        )
        self.transforms = None
        self._frames = {}
        for episode_id, stage in ((7, 2), (9, 3)):
            actions = np.ones((70, 18), dtype=np.float32)
            states = np.ones((70, 19), dtype=np.float32)
            # Every possible focused candidate near this event sees a genuine
            # open->close transition in its first ten targets.
            actions[8:, [7, 15]] = 0.0
            states[8:, [7, 15]] = 0.0
            self._frames[episode_id] = pd.DataFrame(
                {
                    "valid_state": np.ones(70, dtype=np.float32),
                    "subtask_index": np.full(70, stage, dtype=np.int64),
                    "source.action": list(actions),
                    "source.observation.state": list(states),
                }
            )


def _focused_mixture():
    mixture = object.__new__(LeRobotMixtureDataset)
    mixture.datasets = [_FocusedEpisodeDataset()]
    mixture.video_target_shift_steps = 0
    mixture.video_frame_stride = 1
    mixture.use_action_validity_prefix_mask = True
    mixture.action_validity_invalid_run_length = 3
    return mixture


def _mixture():
    mixture = object.__new__(LeRobotMixtureDataset)
    mixture.datasets = [_EpisodeDataset()]
    mixture.video_target_shift_steps = 2
    mixture.video_frame_stride = 10
    mixture.use_action_validity_prefix_mask = True
    mixture.action_validity_invalid_run_length = 3
    return mixture


def _manifest(tmp_path, *, action_max=4):
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_sampling": {
                    "algorithm": "nonzero_valid_unpadded_uniform_v1",
                    "frames_per_episode": 1,
                    "seed_sha256": "ab" * 32,
                    "observation_mode": "deployment_action_current_qwen_rgb_v1",
                    "evaluation_video_offsets": [0],
                    "action_offset_range_inclusive": [0, action_max],
                    "candidate_policy": (
                        "structurally unpadded frames with at least one valid "
                        "action-mask element"
                    ),
                    "all_invalid_episode_fallback": (
                        "uniform over all structurally unpadded frames; report "
                        "zero valid elements"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_one_deterministic_unpadded_window_per_manifest_episode(tmp_path):
    manifest = _manifest(tmp_path)
    first = DeterministicHeldoutEvalDataset.from_manifest(
        _mixture(), manifest, action_dim=2
    )
    second = DeterministicHeldoutEvalDataset.from_manifest(
        _mixture(), manifest, action_dim=2
    )

    assert len(first) == 2
    assert first.heldout_window_references == second.heldout_window_references
    assert first.heldout_window_digest == second.heldout_window_digest
    assert first.datasets[0].parquet_cache_close_count == 1
    assert {ref.episode_id for ref in first.heldout_window_references} == {7, 9}

    # Eval mutates only its independent view from compact training video to the
    # deployment action-policy's current frame. Action/state/language remain.
    assert required_window_offsets(first, first.datasets[0]).tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert first.video_target_shift_steps == 0
    assert first.datasets[0].delta_indices["video.cam"].tolist() == [0]
    assert first.datasets[0].delta_indices["action.x"].tolist() == list(range(5))
    for reference in first.heldout_window_references:
        assert 0 <= reference.base_index <= 95

    by_episode = {
        reference.episode_id: reference
        for reference in first.heldout_window_references
    }
    assert by_episode[7].valid_action_timesteps == 5
    assert by_episode[7].valid_action_elements == 10
    assert by_episode[9].valid_action_timesteps == 0
    assert by_episode[9].valid_action_elements == 0

    report = first.sampling_report()
    assert report["observation_count"] == 2
    assert report["observation_mode"] == "deployment_action_current_qwen_rgb_v1"
    assert report["evaluation_video_offsets"] == [0]
    assert report["action_evaluable_observation_count"] == 1
    assert report["valid_action_element_count"] == 10
    assert report["zero_valid_action_episodes"] == [
        {
            "dataset_name": "fixture",
            "episode_id": 9,
            "base_index": by_episode[9].base_index,
        }
    ]


def test_eval_sampling_contract_and_effective_batch_count_fail_closed(tmp_path):
    manifest = _manifest(tmp_path)
    contract = load_evaluation_sampling_contract(manifest)
    assert contract.frames_per_episode == 1
    assert contract.seed_sha256 == "ab" * 32
    assert contract.observation_mode == "deployment_action_current_qwen_rgb_v1"
    assert contract.evaluation_video_offsets == (0,)
    assert contract.action_offset_range_inclusive == (0, 4)

    assert validate_global_eval_observation_count(
        holdout_episode_count=96,
        per_device_batch_size=12,
        world_size=8,
        gradient_accumulation_steps=1,
    ) == 96
    assert validate_global_eval_observation_count(
        holdout_episode_count=192,
        per_device_batch_size=12,
        world_size=8,
        gradient_accumulation_steps=2,
    ) == 192
    with pytest.raises(ValueError, match="exactly one effective global"):
        validate_global_eval_observation_count(
            holdout_episode_count=95,
            per_device_batch_size=12,
            world_size=8,
            gradient_accumulation_steps=1,
        )


def test_eval_loader_generator_is_dedicated(tmp_path):
    dataset = DeterministicHeldoutEvalDataset.from_manifest(
        _mixture(), _manifest(tmp_path), action_dim=2
    )
    torch.manual_seed(123)
    global_state = torch.random.get_rng_state().clone()
    generator = dataset.make_torch_generator()
    _ = torch.rand(4, generator=generator)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_legacy_underfilled_audit_excludes_zero_valid_without_replacement(
    tmp_path,
):
    dataset = DeterministicHeldoutEvalDataset.from_manifest(
        _mixture(),
        _manifest(tmp_path),
        action_dim=2,
        legacy_underfilled_eval=True,
        legacy_excluded_zero_valid_episode_ids=(9,),
    )

    assert dataset.original_manifest_observation_count == 2
    assert len(dataset) == 1
    assert [item.episode_id for item in dataset.heldout_window_references] == [7]
    report = dataset.sampling_report()
    assert report["production_valid"] is False
    assert report["checkpoint_selection_eligible"] is False
    assert report["zero_valid_action_episodes"] == []
    assert report["legacy_underfilled_holdout"] == {
        "enabled": True,
        "original_manifest_observation_count": 2,
        "evaluated_observation_count": 1,
        "excluded_zero_valid_episodes": [
            {
                "dataset_name": "fixture",
                "episode_id": 9,
                "base_index": next(
                    item.base_index
                    for item in DeterministicHeldoutEvalDataset.from_manifest(
                        _mixture(), _manifest(tmp_path), action_dim=2
                    ).heldout_window_references
                    if item.episode_id == 9
                ),
                "reason": (
                    "no structurally valid window has a supervised action element"
                ),
            }
        ],
        "replacement_episode_ids": [],
        "no_replacement_no_training_leak": True,
        "reason": (
            "Historical checkpoint audit only: the frozen original holdout "
            "contained zero-supervision episodes."
        ),
    }

    with pytest.raises(ValueError, match="exclude exactly the explicit"):
        DeterministicHeldoutEvalDataset.from_manifest(
            _mixture(),
            _manifest(tmp_path),
            action_dim=2,
            legacy_underfilled_eval=True,
            legacy_excluded_zero_valid_episode_ids=(7,),
        )


def test_legacy_95_view_distributed_sharding_never_pads_or_duplicates():
    source = list(range(95))
    global_batch_sampler = BatchSampler(
        SequentialSampler(source), batch_size=12, drop_last=False
    )
    rank_batches = [
        list(
            BatchSamplerShard(
                global_batch_sampler,
                num_processes=8,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
        )
        for rank in range(8)
    ]

    assert all(len(batches) == 1 for batches in rank_batches)
    visited = [index for batches in rank_batches for batch in batches for index in batch]
    assert sorted(visited) == source
    assert len(visited) == len(set(visited)) == 95


def test_sampling_is_uniform_over_all_nonzero_valid_unpadded_candidates(tmp_path):
    mixture = _mixture()
    alternating = (np.arange(100) % 2 == 0).astype(np.float32)
    mixture.datasets[0]._frames[7] = pd.DataFrame(
        {"valid_state": alternating}
    )
    dataset = DeterministicHeldoutEvalDataset.from_manifest(
        mixture,
        _manifest(tmp_path),
        action_dim=2,
    )

    reference = next(
        item for item in dataset.heldout_window_references if item.episode_id == 7
    )
    assert reference.evaluable_candidate_count == reference.structural_candidate_count
    assert reference.selection_pool_candidate_count == reference.evaluable_candidate_count
    assert reference.valid_action_timesteps in {2, 3}


def test_current_frame_eval_keeps_full_action_horizon_and_accepts_short_episode(
    tmp_path,
):
    mixture = _mixture()
    child = mixture.datasets[0]
    child.delta_indices["action.x"] = np.arange(50, dtype=np.int64)
    child.trajectory_lengths = np.asarray([90, 90], dtype=np.int64)
    child.all_steps = [
        (episode_id, step)
        for episode_id in child.trajectory_ids.tolist()
        for step in range(90)
    ]

    dataset = DeterministicHeldoutEvalDataset.from_manifest(
        mixture,
        _manifest(tmp_path, action_max=49),
        action_dim=2,
    )

    assert dataset.datasets[0].delta_indices["video.cam"].tolist() == [0]
    assert dataset.datasets[0].delta_indices["action.x"].tolist() == list(range(50))
    assert required_window_offsets(dataset, dataset.datasets[0]).tolist() == list(
        range(50)
    )
    assert all(
        0 <= reference.base_index <= 40
        for reference in dataset.heldout_window_references
    )


def test_eval_cache_config_is_independent_and_capped_at_one():
    train_cfg = OmegaConf.create(
        {"lerobot_v3_parquet_cache_size": 5, "video_target_shift_steps": 2}
    )

    eval_cfg = _build_eval_lerobot_data_cfg(train_cfg)

    assert eval_cfg is not train_cfg
    assert eval_cfg.lerobot_v3_parquet_cache_size == 1
    assert train_cfg.lerobot_v3_parquet_cache_size == 5
    assert eval_cfg.video_target_shift_steps == train_cfg.video_target_shift_steps == 2


def test_focused_view_is_deterministic_manifest_bound_and_h10_transition_rich(
    tmp_path,
):
    manifest = _manifest(tmp_path, action_max=49)
    first = DeterministicHeldoutEvalDataset.from_manifest(
        _focused_mixture(),
        manifest,
        action_dim=18,
        focused_subtasks=(2, 3),
    )
    second = DeterministicHeldoutEvalDataset.from_manifest(
        _focused_mixture(),
        manifest,
        action_dim=18,
        focused_subtasks=(2, 3),
    )

    focused = first.make_focused_view()
    second_focused = second.make_focused_view()
    assert focused.heldout_window_references == (
        second_focused.heldout_window_references
    )
    assert focused.heldout_window_digest == second_focused.heldout_window_digest
    assert {item.episode_id for item in focused.heldout_window_references} == {7, 9}
    assert all(
        item.open_to_close_transitions_h10 > 0
        for item in focused.heldout_window_references
    )
    report = focused.sampling_report()
    assert report["view"] == "focused"
    assert report["observation_count"] == 2
    assert report["open_to_close_transition_count_h10"] >= 2
    assert report["open_to_close_transition_window_count_h10"] == 2
    assert report["close_to_open_transition_window_count_h10"] == 0
    assert report["subtask_evaluable_observation_counts"] == {"2": 1, "3": 1}
    assert report["subtask_action_timestep_counts_by_horizon"]["10"] == {
        "2": 10,
        "3": 10,
    }
    assert report["subtask_valid_action_element_counts_by_horizon"]["10"] == {
        "2": 180,
        "3": 180,
    }
    # Creating the focused view does not alter the original unbiased references.
    assert first.heldout_eval_view == "unbiased"
    assert first.heldout_window_references != focused.heldout_window_references


def test_focused_movement_diagnostics_mirror_float16_packed_targets():
    mixture = _focused_mixture()
    dataset = mixture.datasets[0]
    frame = dataset._frames[7]
    actions = np.full((70, 18), 0.1, dtype=np.float32)
    states = np.full((70, 19), 0.1, dtype=np.float32)

    # In float32 this first delta exceeds the threshold, but the target's
    # float16 wire/storage value falls just below it.  The second movement stays
    # above threshold and proves the reported denominator uses its packed value.
    actions[0, 0] = np.float32(0.12001)
    actions[1, 1] = np.float32(0.12345)
    frame["source.action"] = list(actions)
    frame["source.observation.state"] = list(states)

    diagnostic = _candidate_control_diagnostics(
        mixture=mixture,
        dataset=dataset,
        episode_id=7,
        candidates=(0,),
        action_dim=18,
        movement_threshold=0.02,
    )[0]

    packed_moving_target = np.float16(actions[1, 1]).astype(np.float32)
    expected_hold_abs = abs(float(packed_moving_target) - 0.1)
    assert diagnostic.arm_movement_elements_h10 == 1
    assert diagnostic.arm_movement_hold_abs_h10 == pytest.approx(
        expected_hold_abs
    )
