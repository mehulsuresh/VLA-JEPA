import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.dataloader import _build_eval_lerobot_data_cfg
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.dataloader.heldout_eval import (
    DeterministicHeldoutEvalDataset,
    load_evaluation_sampling_contract,
    required_window_offsets,
    validate_global_eval_observation_count,
)


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
        self._episode_split_selection = SimpleNamespace(
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
