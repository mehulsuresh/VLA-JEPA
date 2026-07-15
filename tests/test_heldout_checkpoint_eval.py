import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.training import train_starvla
from starVLA.training.train_starvla import (
    VLATrainer,
    _validate_heldout_report_coverage,
)


class _EvalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.calls = []

    def predict_action(self, **kwargs):
        # Exercise all three process-global RNGs; the trainer must restore them.
        stochastic_probe = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        actions = np.asarray(
            [np.asarray(example["action"], dtype=np.float32) for example in kwargs["batch"]]
        )
        self.calls.append(
            {
                "kwargs": kwargs,
                "stochastic_probe": stochastic_probe,
                "parent_training": self.training,
                "dropout_training": self.dropout.training,
            }
        )
        return {"normalized_actions": np.zeros_like(actions)}


class _Accelerator:
    device = torch.device("cpu")
    is_main_process = True

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def reduce(value, reduction):
        assert reduction == "sum"
        return value

    @staticmethod
    def gather(value):
        return value


class _ClosableEvalLoader(list):
    def __init__(self, batches):
        super().__init__(batches)
        self.close_count = 0

    def close_eval_caches(self):
        self.close_count += 1


def _split_provenance(selected_episode_count):
    return [
        {
            "dataset_name": "fixture",
            "manifest_path": "/fixture/manifest.json",
            "manifest_sha256": "1" * 64,
            "role": "eval",
            "selected_episode_count": selected_episode_count,
            "selected_episode_set_sha256": "2" * 64,
            "selected_frame_count": selected_episode_count * 100,
            "train_episode_count": 10,
            "train_episode_set_sha256": "3" * 64,
            "train_frame_count": 1000,
            "holdout_episode_count": selected_episode_count,
            "holdout_episode_set_sha256": "2" * 64,
            "full_catalog_sha256": "4" * 64,
            "train_statistics_path": "/fixture/train_stats.json",
            "train_statistics_sha256": "5" * 64,
        }
    ]


def _numpy_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _trainer():
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "framework": {"action_model": {"num_inference_timesteps": 8}},
            "trainer": {"allow_training_stream_eval": False},
        }
    )
    trainer.accelerator = _Accelerator()
    trainer.model = _EvalModel()
    trainer.model.train()
    trainer.model.dropout.eval()
    trainer.total_batch_size = 2
    trainer.vla_focused_eval_dataloader = None
    trainer.heldout_eval_seed = 8675309
    trainer.heldout_eval_expected_valid_observations = 1
    trainer.heldout_eval_expected_valid_elements = 4
    trainer.heldout_eval_sampling_report = {
        "observation_count": 2,
        "action_evaluable_observation_count": 1,
        "valid_action_element_count": 4,
        "subtask_observation_counts": {"7": 1, "9": 1},
        "subtask_evaluable_observation_counts": {"7": 1},
        "episode_split_provenance": _split_provenance(2),
        "zero_valid_action_episodes": [
            {"dataset_name": "fixture", "episode_id": 9, "base_index": 50}
        ]
    }
    trainer.heldout_eval_subtask_counts = {7: 1, 9: 1}
    trainer.heldout_eval_evaluable_subtask_counts = {7: 1}
    trainer.vla_eval_dataloader = [
        [
            {
                "action": np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32),
                "action_mask": np.ones((2, 2), dtype=bool),
                "action_is_pad": np.zeros(2, dtype=bool),
                "_heldout_eval_index": 0,
                "_heldout_eval_episode_id": 7,
                "_heldout_eval_hold_action": np.asarray([1.0, 1.0]),
                "_heldout_eval_action_midpoint": np.asarray([0.0, 0.0]),
                "_heldout_eval_subtask_index": 7,
                "_heldout_eval_action_subtask_indices": np.asarray([7, 7]),
                "_heldout_eval_view": "unbiased",
            }
        ],
        [
            {
                "action": np.asarray([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32),
                "action_mask": np.zeros((2, 2), dtype=bool),
                "action_is_pad": np.zeros(2, dtype=bool),
                "_heldout_eval_index": 1,
                "_heldout_eval_episode_id": 9,
                "_heldout_eval_hold_action": np.asarray([1.0, 1.0]),
                "_heldout_eval_action_midpoint": np.asarray([0.0, 0.0]),
                "_heldout_eval_subtask_index": 9,
                "_heldout_eval_action_subtask_indices": np.asarray([9, 9]),
                "_heldout_eval_view": "unbiased",
            }
        ],
    ]
    trainer._get_next_eval_batch = lambda: (_ for _ in ()).throw(
        AssertionError("heldout eval must never read the training iterator")
    )
    return trainer


def test_heldout_eval_is_complete_rng_isolated_and_deployment_shaped():
    trainer = _trainer()
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    metrics = trainer.eval_heldout_action_model({})

    assert random.getstate() == python_state
    assert _numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert len(trainer.model.calls) == 2
    for call in trainer.model.calls:
        assert call["parent_training"] is False
        assert call["dropout_training"] is False
        assert call["kwargs"]["num_inference_timesteps"] == 8
        assert call["kwargs"]["prev_actions"] is None
        assert call["kwargs"]["prefix_len"] == 0
        assert call["kwargs"]["rtc_config"] is None
        assert all(
            not key.startswith("_heldout_eval_")
            for example in call["kwargs"]["batch"]
            for key in example
        )

    assert metrics["heldout_eval_observations"] == 2
    assert metrics["heldout_eval_action_evaluable_observations"] == 1
    assert metrics["heldout_eval_valid_action_elements"] == 4
    assert metrics["heldout_eval_zero_valid_action_episodes"] == 1
    assert metrics["heldout_eval_normalized_action_mae"] == pytest.approx(1.75)
    assert metrics["heldout_eval_normalized_action_rmse"] == pytest.approx(2.5)
    assert metrics["heldout_eval_normalized_all_action_mae_h1"] == pytest.approx(3.5)
    assert trainer.model.training is True
    assert trainer.model.dropout.training is False

    # Re-running the fixed loader uses the same isolated diffusion seed.
    first_probes = [call["stochastic_probe"] for call in trainer.model.calls]
    trainer.eval_heldout_action_model({})
    second_probes = [call["stochastic_probe"] for call in trainer.model.calls[2:]]
    assert second_probes == first_probes


def test_heldout_eval_releases_eval_only_caches_after_each_pass():
    trainer = _trainer()
    trainer.vla_eval_dataloader = _ClosableEvalLoader(trainer.vla_eval_dataloader)

    trainer.eval_heldout_action_model({})

    assert trainer.vla_eval_dataloader.close_count == 1


def test_heldout_eval_rejects_duplicate_or_missing_reference_coverage():
    trainer = _trainer()
    trainer.vla_eval_dataloader[1][0]["_heldout_eval_index"] = 0
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        trainer.eval_heldout_action_model({})


def test_distributed_heldout_index_coverage_uses_variable_length_object_gather(
    monkeypatch,
):
    trainer = _trainer()

    class _NoTensorGatherAccelerator(_Accelerator):
        @staticmethod
        def gather(_value):
            raise AssertionError(
                "variable-length heldout indices must not use tensor gather"
            )

    trainer.accelerator = _NoTensorGatherAccelerator()
    gathered = []

    monkeypatch.setattr(train_starvla.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        train_starvla,
        "gather_object",
        lambda indices: gathered.extend(indices) or list(indices),
    )

    metrics = trainer.eval_heldout_action_model({})

    assert gathered == [0, 1]
    assert metrics["heldout_eval_observations"] == 2


def test_heldout_eval_rejects_valid_element_drift_from_sampling_report():
    trainer = _trainer()
    trainer.heldout_eval_expected_valid_elements = 5
    with pytest.raises(RuntimeError, match="valid action-element coverage"):
        trainer.eval_heldout_action_model({})


def test_effective_eval_view_rejects_zero_supervision_episode():
    report = {
        "observation_count": 2,
        "action_evaluable_observation_count": 1,
        "subtask_observation_counts": {"2": 2},
        "subtask_evaluable_observation_counts": {"2": 1},
        "episode_split_provenance": _split_provenance(2),
        "zero_valid_action_episodes": [
            {"dataset_name": "fixture", "episode_id": 955, "base_index": 0}
        ],
    }

    with pytest.raises(ValueError, match="must keep all 2 episode windows"):
        _validate_heldout_report_coverage(
            report,
            expected_observations=2,
            required_subtasks=(),
            minimum_per_subtask=1,
            label="Heldout eval",
        )


def test_legacy_underfilled_report_is_accepted_only_as_no_replacement_audit():
    report = {
        "observation_count": 95,
        "action_evaluable_observation_count": 95,
        "subtask_observation_counts": {"2": 95},
        "subtask_evaluable_observation_counts": {"2": 95},
        "zero_valid_action_episodes": [],
        "production_valid": False,
        "checkpoint_selection_eligible": False,
        "episode_split_provenance": _split_provenance(96),
        "legacy_underfilled_holdout": {
            "enabled": True,
            "original_manifest_observation_count": 96,
            "evaluated_observation_count": 95,
            "excluded_zero_valid_episodes": [
                {"dataset_name": "magna", "episode_id": 955}
            ],
            "replacement_episode_ids": [],
            "no_replacement_no_training_leak": True,
        },
    }

    _validate_heldout_report_coverage(
        report,
        expected_observations=96,
        required_subtasks=(),
        minimum_per_subtask=1,
        label="Legacy heldout eval",
        allow_legacy_underfilled=True,
    )

    report["legacy_underfilled_holdout"]["replacement_episode_ids"] = [123]
    with pytest.raises(ValueError, match="no-replacement evidence"):
        _validate_heldout_report_coverage(
            report,
            expected_observations=96,
            required_subtasks=(),
            minimum_per_subtask=1,
            label="Legacy heldout eval",
            allow_legacy_underfilled=True,
        )


def test_zero_valid_episode_nonfinite_targets_do_not_poison_metrics():
    trainer = _trainer()
    trainer.vla_eval_dataloader[1][0]["action"][:] = np.nan

    metrics = trainer.eval_heldout_action_model({})

    assert np.isfinite(metrics["heldout_eval_normalized_action_mae"])
    assert metrics["heldout_eval_normalized_action_mae"] == pytest.approx(1.75)
    assert metrics["heldout_eval_valid_action_elements"] == 4


def test_nonfinite_value_at_supervised_element_fails_closed():
    trainer = _trainer()
    trainer.vla_eval_dataloader[0][0]["action"][0, 0] = np.nan

    with pytest.raises(RuntimeError, match="non-finite.*supervised"):
        trainer.eval_heldout_action_model({})


def test_realman_prefix_and_arm_metrics_reuse_the_same_forward():
    trainer = _trainer()
    action_row = np.ones(18, dtype=np.float32)
    arm_dimensions = tuple(range(0, 7)) + tuple(range(8, 15))
    action_row[list(arm_dimensions)] = 2.0
    action = np.repeat(action_row[None, :], 50, axis=0)
    trainer.total_batch_size = 1
    trainer.heldout_eval_expected_valid_observations = 1
    trainer.heldout_eval_expected_valid_elements = 50 * 18
    trainer.heldout_eval_sampling_report = {
        "observation_count": 1,
        "action_evaluable_observation_count": 1,
        "valid_action_element_count": 50 * 18,
        "subtask_observation_counts": {"2": 1},
        "subtask_evaluable_observation_counts": {"2": 1},
        "zero_valid_action_episodes": [],
    }
    trainer.heldout_eval_subtask_counts = {2: 1}
    trainer.heldout_eval_evaluable_subtask_counts = {2: 1}
    trainer.vla_eval_dataloader = [
        [
            {
                "action": action,
                "action_mask": np.ones_like(action, dtype=bool),
                "action_is_pad": np.zeros(50, dtype=bool),
                "_heldout_eval_index": 0,
                "_heldout_eval_hold_action": np.zeros(18, dtype=np.float32),
                "_heldout_eval_action_midpoint": np.full(
                    18, 0.5, dtype=np.float32
                ),
                "_heldout_eval_subtask_index": 2,
                "_heldout_eval_action_subtask_indices": np.full(50, 2),
                "_heldout_eval_view": "unbiased",
            }
        ]
    ]

    metrics = trainer.eval_heldout_action_model({})

    assert len(trainer.model.calls) == 1
    for horizon in (1, 5, 10, 20, 50):
        assert metrics[
            f"heldout_eval_normalized_all_action_mae_h{horizon}"
        ] == pytest.approx(32.0 / 18.0)
        assert metrics[
            f"heldout_eval_normalized_arm_mae_h{horizon}"
        ] == pytest.approx(2.0)
        assert metrics[
            f"heldout_eval_current_state_hold_normalized_arm_mae_h{horizon}"
        ] == pytest.approx(2.0)
        assert metrics[
            f"heldout_eval_policy_vs_hold_arm_mae_ratio_h{horizon}"
        ] == pytest.approx(1.0)
        assert metrics[
            f"heldout_eval_policy_vs_hold_arm_movement_mae_ratio_h{horizon}"
        ] == pytest.approx(1.0)
        assert metrics[
            f"heldout_eval_arm_movement_direction_accuracy_h{horizon}"
        ] == pytest.approx(0.0)
        assert metrics[
            f"heldout_eval_normalized_gripper_mae_h{horizon}"
        ] == pytest.approx(1.0)
        assert metrics[
            f"heldout_eval_subtask_2_normalized_arm_mae_h{horizon}"
        ] == pytest.approx(2.0)
    assert metrics["heldout_eval_subtask_2_observations"] == 1
    assert metrics["heldout_eval_gripper_target_open_elements_h10"] == 20
    assert metrics["heldout_eval_gripper_open_recall_h10"] == pytest.approx(0.0)
    assert metrics["heldout_eval_gripper_close_to_open_elements_h10"] == 2
    assert metrics[
        "heldout_eval_gripper_close_to_open_recall_h10"
    ] == pytest.approx(0.0)


def test_focused_h10_transition_gate_is_explicit_and_fail_closed():
    trainer = _trainer()
    action = np.zeros((50, 18), dtype=np.float32)
    action[1:, [7, 15]] = 1.0
    sample = {
        "action": action,
        "action_mask": np.ones_like(action, dtype=bool),
        "action_is_pad": np.zeros(50, dtype=bool),
        "_heldout_eval_index": 0,
        "_heldout_eval_hold_action": np.ones(18, dtype=np.float32),
        "_heldout_eval_action_midpoint": np.full(18, 0.5, dtype=np.float32),
        "_heldout_eval_subtask_index": 2,
        "_heldout_eval_action_subtask_indices": np.full(50, 2),
        "_heldout_eval_view": "focused",
    }
    report = {
        "observation_count": 1,
        "action_evaluable_observation_count": 1,
        "valid_action_element_count": 50 * 18,
        "subtask_observation_counts": {"2": 1},
        "subtask_evaluable_observation_counts": {"2": 1},
    }

    metrics = trainer._evaluate_action_batches(
        [[sample]],
        step_metrics={},
        metric_prefix="heldout_focused_eval",
        expected_observations=1,
        expected_valid_observations=1,
        expected_valid_elements=50 * 18,
        require_heldout_indices=True,
        sampling_report=report,
        transition_coverage_horizon=10,
        minimum_open_to_close_transitions=2,
        minimum_close_to_open_transitions=2,
    )

    assert metrics[
        "heldout_focused_eval_gripper_open_to_close_elements_h10"
    ] == 2
    assert metrics[
        "heldout_focused_eval_gripper_open_to_close_recall_h10"
    ] == pytest.approx(1.0)
    assert metrics[
        "heldout_focused_eval_gripper_close_to_open_elements_h10"
    ] == 2
    assert metrics[
        "heldout_focused_eval_gripper_close_to_open_recall_h10"
    ] == pytest.approx(0.0)
    assert metrics[
        "heldout_focused_eval_gripper_transition_balanced_recall_h10"
    ] == pytest.approx(0.5)
    assert metrics[
        "heldout_focused_eval_gripper_transition_min_recall_h10"
    ] == pytest.approx(0.0)
    assert metrics[
        "heldout_focused_eval_policy_vs_hold_arm_movement_mae_ratio_h10"
    ] == pytest.approx(0.0)
    assert metrics[
        "heldout_focused_eval_arm_movement_direction_accuracy_h10"
    ] == pytest.approx(1.0)
    assert metrics[
        "heldout_focused_eval_task_failure_score_h10"
    ] == pytest.approx(1.0)
    assert metrics[
        "heldout_focused_eval_gripper_open_to_close_windows_h10"
    ] == 1
    assert metrics[
        "heldout_focused_eval_gripper_close_to_open_windows_h10"
    ] == 1

    with pytest.raises(RuntimeError, match="transition coverage is insufficient.*h10"):
        trainer._evaluate_action_batches(
            [[sample]],
            step_metrics={},
            metric_prefix="heldout_focused_eval",
            expected_observations=1,
            expected_valid_observations=1,
            expected_valid_elements=50 * 18,
            require_heldout_indices=True,
            sampling_report=report,
            transition_coverage_horizon=10,
            minimum_open_to_close_transitions=2,
            minimum_close_to_open_transitions=3,
        )

    with pytest.raises(RuntimeError, match="transition coverage is insufficient.*windows"):
        trainer._evaluate_action_batches(
            [[sample]],
            step_metrics={},
            metric_prefix="heldout_focused_eval",
            expected_observations=1,
            expected_valid_observations=1,
            expected_valid_elements=50 * 18,
            require_heldout_indices=True,
            sampling_report=report,
            transition_coverage_horizon=10,
            minimum_open_to_close_transitions=2,
            minimum_close_to_open_transitions=2,
            minimum_open_to_close_windows=2,
            minimum_close_to_open_windows=1,
        )

    with pytest.raises(RuntimeError, match="arm-movement denominator coverage"):
        trainer._evaluate_action_batches(
            [[sample]],
            step_metrics={},
            metric_prefix="heldout_focused_eval",
            expected_observations=1,
            expected_valid_observations=1,
            expected_valid_elements=50 * 18,
            require_heldout_indices=True,
            sampling_report=report,
            transition_coverage_horizon=10,
            minimum_open_to_close_transitions=2,
            minimum_close_to_open_transitions=2,
            minimum_arm_movement_elements=141,
        )


def test_per_subtask_metrics_use_each_target_step_not_anchor_label():
    trainer = _trainer()
    action = np.asarray(
        [[1.0, 1.0], [1.0, 1.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0]],
        dtype=np.float32,
    )
    sample = {
        "action": action,
        "action_mask": np.ones_like(action, dtype=bool),
        "action_is_pad": np.zeros(5, dtype=bool),
        "_heldout_eval_index": 0,
        "_heldout_eval_hold_action": np.zeros(2, dtype=np.float32),
        "_heldout_eval_action_midpoint": np.zeros(2, dtype=np.float32),
        "_heldout_eval_subtask_index": 2,
        "_heldout_eval_action_subtask_indices": np.asarray([2, 2, 3, 3, 3]),
    }
    report = {
        "observation_count": 1,
        "action_evaluable_observation_count": 1,
        "valid_action_element_count": 10,
        "subtask_observation_counts": {"2": 1},
        "subtask_evaluable_observation_counts": {"2": 1},
        "subtask_action_timestep_counts_by_horizon": {
            "1": {"2": 1},
            "5": {"2": 2, "3": 3},
        },
        "subtask_valid_action_element_counts_by_horizon": {
            "1": {"2": 2},
            "5": {"2": 4, "3": 6},
        },
    }

    metrics = trainer._evaluate_action_batches(
        [[sample]],
        step_metrics={},
        metric_prefix="heldout_eval",
        expected_observations=1,
        expected_valid_observations=1,
        expected_valid_elements=10,
        require_heldout_indices=True,
        sampling_report=report,
    )

    assert metrics["heldout_eval_subtask_2_observations"] == 1
    assert metrics["heldout_eval_subtask_3_observations"] == 0
    assert metrics["heldout_eval_subtask_2_action_timesteps_h5"] == 2
    assert metrics["heldout_eval_subtask_3_action_timesteps_h5"] == 3
    assert metrics["heldout_eval_subtask_2_normalized_all_action_mae_h5"] == 1
    assert metrics["heldout_eval_subtask_3_normalized_all_action_mae_h5"] == 3


def test_task_selection_score_cannot_trade_arm_motion_for_zero_transitions():
    trainer = _trainer()

    class _PerfectArmWrongGripper(_EvalModel):
        def predict_action(self, **kwargs):
            actions = np.asarray(
                [np.asarray(item["action"], dtype=np.float32) for item in kwargs["batch"]]
            )
            prediction = actions.copy()
            prediction[..., [7, 15]] = 1.0 - actions[..., [7, 15]]
            return {"normalized_actions": prediction}

    trainer.model = _PerfectArmWrongGripper()
    action = np.zeros((50, 18), dtype=np.float32)
    action[1:, [7, 15]] = 1.0
    sample = {
        "action": action,
        "action_mask": np.ones_like(action, dtype=bool),
        "action_is_pad": np.zeros(50, dtype=bool),
        "_heldout_eval_index": 0,
        "_heldout_eval_hold_action": np.ones(18, dtype=np.float32),
        "_heldout_eval_action_midpoint": np.full(18, 0.5, dtype=np.float32),
        "_heldout_eval_subtask_index": 2,
        "_heldout_eval_action_subtask_indices": np.full(50, 2),
    }
    report = {
        "observation_count": 1,
        "action_evaluable_observation_count": 1,
        "valid_action_element_count": 50 * 18,
        "subtask_observation_counts": {"2": 1},
        "subtask_evaluable_observation_counts": {"2": 1},
    }

    metrics = trainer._evaluate_action_batches(
        [[sample]],
        step_metrics={},
        metric_prefix="heldout_focused_eval",
        expected_observations=1,
        expected_valid_observations=1,
        expected_valid_elements=50 * 18,
        require_heldout_indices=True,
        sampling_report=report,
        transition_coverage_horizon=10,
        minimum_open_to_close_transitions=2,
        minimum_close_to_open_transitions=2,
    )

    assert metrics[
        "heldout_focused_eval_policy_vs_hold_arm_movement_mae_ratio_h10"
    ] == 0
    assert metrics[
        "heldout_focused_eval_gripper_transition_balanced_recall_h10"
    ] == 0
    assert metrics[
        "heldout_focused_eval_gripper_transition_min_recall_h10"
    ] == 0
    assert metrics["heldout_focused_eval_task_success_score_h10"] == 0
    assert metrics["heldout_focused_eval_task_failure_score_h10"] == 1


def test_eval_only_main_never_constructs_training_loader_or_optimizer(
    monkeypatch, tmp_path
):
    cfg = OmegaConf.create(
        {
            "run_id": "eval-only-fixture",
            "run_root_dir": str(tmp_path),
            "trainer": {"eval_only": True},
        }
    )
    calls = []

    class Accelerator:
        num_processes = 8

    class Trainer:
        def __init__(self, **kwargs):
            assert kwargs["vla_train_dataloader"] is None
            assert kwargs["optimizer"] is None
            assert kwargs["lr_scheduler"] is None
            calls.append("constructed")

        def prepare_checkpoint_evaluation(self):
            calls.append("prepared_eval")

        def evaluate_checkpoint_only(self):
            calls.append("evaluated")

        def train(self):
            raise AssertionError("eval-only must never enter train()")

        def _shutdown_data_runtime(self):
            calls.append("shutdown")

    def setup(cfg):
        cfg.output_dir = str(tmp_path / cfg.run_id)
        return tmp_path / cfg.run_id

    monkeypatch.setattr(train_starvla, "build_accelerator", lambda _cfg: Accelerator())
    monkeypatch.setattr(train_starvla, "setup_directories", setup)
    monkeypatch.setattr(train_starvla, "build_model", lambda _cfg: object())
    monkeypatch.setattr(
        train_starvla,
        "prepare_heldout_eval_data",
        lambda **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        train_starvla,
        "prepare_data",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("eval-only constructed a training loader")
        ),
    )
    monkeypatch.setattr(
        train_starvla,
        "setup_optimizer_and_scheduler",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("eval-only constructed an optimizer")
        ),
    )
    monkeypatch.setattr(train_starvla, "VLATrainer", Trainer)
    monkeypatch.setattr(
        train_starvla,
        "logger",
        SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None),
    )

    train_starvla.main(cfg)

    assert calls == ["constructed", "prepared_eval", "evaluated", "shutdown"]


def test_eval_only_untrained_initialization_can_score_step_zero_without_checkpoint():
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {"trainer": {"eval_only_untrained_initialization": True}}
    )
    trainer.loaded_checkpoint_path = None
    trainer.completed_steps = 0
    trainer.accelerator = SimpleNamespace(
        is_main_process=False,
        wait_for_everyone=lambda: None,
    )
    trainer.eval_heldout_action_model = lambda _metrics: {"heldout_eval_x": 1.0}

    metrics = trainer.evaluate_checkpoint_only()

    assert metrics == {
        "heldout_eval_x": 1.0,
        "epoch": 0.0,
        "samples_seen": 0.0,
    }


def test_checkpoint_eval_runs_two_views_and_atomically_persists_evidence(tmp_path):
    trainer = _trainer()
    trainer.total_batch_size = 1
    trainer.completed_steps = 2500
    trainer.config.output_dir = str(tmp_path)
    trainer.config.run_id = "focused-eval-fixture"
    trainer.config.seed = 42
    trainer.config.trainer.eval_only = True
    config_bytes = b"run_id: focused-eval-fixture\nseed: 42\n"
    (tmp_path / "config.yaml").write_bytes(config_bytes)
    schedule_bytes = (
        json.dumps(
            {
                "schema_version": 1,
                "resolved": {
                    "effective_global_batch_size": 96,
                    "eval_interval": 2500,
                    "max_train_steps": 7500,
                    "num_warmup_steps": 3000,
                    "save_interval": 2500,
                },
                "source_config": {
                    "path": "config.yaml",
                    "sha256": hashlib.sha256(config_bytes).hexdigest(),
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    (tmp_path / "resolved_training_schedule.json").write_bytes(
        schedule_bytes
    )
    trainer.config.trainer.eval_source_training_config_path = str(
        tmp_path / "config.yaml"
    )
    trainer.config.trainer.eval_source_training_config_sha256 = hashlib.sha256(
        config_bytes
    ).hexdigest()
    checkpoint_path = tmp_path / "source/checkpoints/steps_2500"
    checkpoint_path.mkdir(parents=True)
    trainer_state_bytes = b'{"completed_steps": 2500}\n'
    (checkpoint_path / "trainer_state.json").write_bytes(trainer_state_bytes)
    (checkpoint_path / "model.safetensors").write_bytes(b"fixture-model")
    trainer.loaded_checkpoint_path = str(checkpoint_path.resolve())
    trainer.config.trainer.heldout_focused_eval_transition_coverage_horizon = 10
    trainer.config.trainer.heldout_focused_eval_min_open_to_close_transitions = 2
    trainer.config.trainer.heldout_focused_eval_min_close_to_open_transitions = 2
    trainer.config.trainer.best_metric_name = (
        "heldout_focused_eval_task_failure_score_h10"
    )
    trainer.config.trainer.best_metric_mode = "min"
    action = np.zeros((50, 18), dtype=np.float32)
    action[1:, [7, 15]] = 1.0

    def sample(view):
        return {
            "action": action.copy(),
            "action_mask": np.ones_like(action, dtype=bool),
            "action_is_pad": np.zeros(50, dtype=bool),
            "_heldout_eval_index": 0,
            "_heldout_eval_hold_action": np.ones(18, dtype=np.float32),
            "_heldout_eval_action_midpoint": np.full(
                18, 0.5, dtype=np.float32
            ),
            "_heldout_eval_subtask_index": 2,
            "_heldout_eval_action_subtask_indices": np.full(50, 2),
            "_heldout_eval_view": view,
        }

    report = {
        "observation_count": 1,
        "action_evaluable_observation_count": 1,
        "valid_action_element_count": 50 * 18,
        "subtask_observation_counts": {"2": 1},
        "subtask_evaluable_observation_counts": {"2": 1},
        "zero_valid_action_episodes": [],
        "production_valid": True,
        "checkpoint_selection_eligible": True,
        "episode_split_provenance": _split_provenance(1),
    }
    trainer.vla_eval_dataloader = [[sample("unbiased")]]
    trainer.vla_focused_eval_dataloader = [[sample("focused")]]
    trainer.heldout_eval_sampling_report = {
        **report,
        "window_selection_sha256": "a" * 64,
    }
    trainer.heldout_eval_expected_valid_observations = 1
    trainer.heldout_eval_expected_valid_elements = 50 * 18
    trainer.heldout_focused_eval_sampling_report = {
        **report,
        "window_selection_sha256": "b" * 64,
        "open_to_close_transition_count_h10": 2,
        "close_to_open_transition_count_h10": 2,
    }
    trainer.heldout_focused_eval_expected_valid_observations = 1
    trainer.heldout_focused_eval_expected_valid_elements = 50 * 18
    trainer.heldout_focused_eval_seed = trainer.heldout_eval_seed

    metrics = trainer.eval_heldout_action_model({})

    assert len(trainer.model.calls) == 2
    assert "heldout_eval_normalized_action_mae" in metrics
    assert "heldout_focused_eval_normalized_action_mae" in metrics
    assert "heldout_focused_eval_task_failure_score_h10" in metrics
    artifact_path = tmp_path / "heldout_eval_metrics/step_00002500.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_step"] == 2500
    assert payload["checkpoint"] == {
        "step": 2500,
        "source_path": str(checkpoint_path.resolve()),
        "source_kind": "checkpoint",
        "trainer_state_sha256": hashlib.sha256(trainer_state_bytes).hexdigest(),
        "model_file": "model.safetensors",
        "model_file_size_bytes": len(b"fixture-model"),
    }
    assert payload["checkpoint_relative_path"] == (
        "source/checkpoints/steps_2500"
    )
    assert payload["run"] == {
        "run_id": "focused-eval-fixture",
        "output_dir": str(tmp_path.resolve()),
        "seed": 42,
        "config_path": "config.yaml",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "resolved_training_schedule": {
            "path": "resolved_training_schedule.json",
            "sha256": hashlib.sha256(schedule_bytes).hexdigest(),
        },
        "source_training_config": {
            "path": str((tmp_path / "config.yaml").resolve()),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
    }
    assert payload["sampling_reports"]["unbiased"][
        "window_selection_sha256"
    ] == "a" * 64
    assert payload["sampling_reports"]["focused"][
        "window_selection_sha256"
    ] == "b" * 64
    assert payload["selection_metric"] == {
        "name": "heldout_focused_eval_task_failure_score_h10",
        "mode": "min",
        "eligible": True,
        "value": pytest.approx(1.0),
    }
    assert payload["production_valid"] is True
    assert payload["checkpoint_selection_eligible"] is True
    assert payload["metrics"]["unbiased"][
        "heldout_eval_normalized_action_mae"
    ] == pytest.approx(98.0 / (50 * 18))
    assert payload["metrics"]["focused"][
        "heldout_focused_eval_task_failure_score_h10"
    ] == pytest.approx(1.0)
    assert list((tmp_path / "heldout_eval_metrics").glob("*.tmp-*")) == []


def test_legacy_artifact_is_explicitly_production_invalid(tmp_path):
    trainer = _trainer()
    trainer.completed_steps = 2500
    trainer.config.output_dir = str(tmp_path)
    trainer.config.run_id = "legacy-audit"
    trainer.config.trainer.eval_only = True
    trainer.eval_source_training_config_evidence = {
        "path": "/fixture/source/config.yaml",
        "sha256": "c" * 64,
    }
    trainer.legacy_underfilled_eval = True
    trainer.vla_focused_eval_dataloader = object()
    legacy = {
        "enabled": True,
        "original_manifest_observation_count": 96,
        "evaluated_observation_count": 95,
        "excluded_zero_valid_episodes": [
            {"dataset_name": "magna", "episode_id": 955}
        ],
        "replacement_episode_ids": [],
        "no_replacement_no_training_leak": True,
    }
    trainer.heldout_eval_sampling_report = {
        "window_selection_sha256": "a" * 64,
        "production_valid": False,
        "checkpoint_selection_eligible": False,
        "episode_split_provenance": _split_provenance(96),
        "legacy_underfilled_holdout": legacy,
    }
    trainer.heldout_focused_eval_sampling_report = {
        "window_selection_sha256": "b" * 64,
        "production_valid": False,
        "checkpoint_selection_eligible": False,
        "episode_split_provenance": _split_provenance(96),
        "legacy_underfilled_holdout": legacy,
    }

    path = trainer._persist_heldout_eval_artifact(
        {
            "heldout_eval_normalized_action_mae": 0.4,
            "heldout_focused_eval_task_failure_score_h10": 0.7,
        }
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["production_valid"] is False
    assert payload["checkpoint_selection_eligible"] is False
    assert payload["selection_metric"]["eligible"] is False
    assert payload["sampling_reports"]["unbiased"][
        "legacy_underfilled_holdout"
    ]["replacement_episode_ids"] == []


def test_step_zero_artifact_does_not_claim_a_nonexistent_checkpoint(tmp_path):
    trainer = _trainer()
    trainer.completed_steps = 0
    trainer.config.output_dir = str(tmp_path)
    trainer.config.run_id = "step-zero-live-eval"
    trainer.config.trainer.best_metric_name = "heldout_eval_normalized_action_mae"
    config_bytes = b"run_id: step-zero-live-eval\n"
    (tmp_path / "config.yaml").write_bytes(config_bytes)
    (tmp_path / "resolved_training_schedule.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "resolved": {
                    "effective_global_batch_size": 2,
                    "eval_interval": 5,
                    "max_train_steps": 15,
                    "num_warmup_steps": 2250,
                    "save_interval": 5,
                },
                "source_config": {
                    "path": "config.yaml",
                    "sha256": hashlib.sha256(config_bytes).hexdigest(),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    trainer.heldout_eval_sampling_report = {
        "window_selection_sha256": "a" * 64,
        "production_valid": True,
        "checkpoint_selection_eligible": True,
        "episode_split_provenance": _split_provenance(2),
    }

    path = trainer._persist_heldout_eval_artifact(
        {"heldout_eval_normalized_action_mae": 0.4}
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["checkpoint_relative_path"] is None
    assert payload["checkpoint"] == {
        "step": 0,
        "source_path": None,
        "source_kind": "live_in_memory_model",
    }

    original_bytes = path.read_bytes()
    assert trainer._persist_heldout_eval_artifact(
        {"heldout_eval_normalized_action_mae": 0.4}
    ) == path
    assert path.read_bytes() == original_bytes
    with pytest.raises(RuntimeError, match="evidence is immutable"):
        trainer._persist_heldout_eval_artifact(
            {"heldout_eval_normalized_action_mae": 0.5}
        )
    assert path.read_bytes() == original_bytes

    schedule_path = tmp_path / "resolved_training_schedule.json"
    tampered_schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    tampered_schedule["source_config"]["sha256"] = "d" * 64
    schedule_path.write_text(json.dumps(tampered_schedule), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not bound to the immutable source config"):
        trainer._persist_heldout_eval_artifact(
            {"heldout_eval_normalized_action_mae": 0.4}
        )
    assert path.read_bytes() == original_bytes


def test_training_entrypoint_rejects_legacy_underfilled_mode_before_setup():
    cfg = OmegaConf.create(
        {
            "trainer": {
                "eval_only": False,
                "eval_only_legacy_underfilled_holdout": True,
            }
        }
    )
    with pytest.raises(ValueError, match="cannot be used during training"):
        train_starvla.main(cfg)
