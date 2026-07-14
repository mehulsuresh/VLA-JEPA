import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.training.train_starvla import VLATrainer


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
    trainer.heldout_eval_seed = 8675309
    trainer.heldout_eval_expected_valid_observations = 1
    trainer.heldout_eval_expected_valid_elements = 4
    trainer.heldout_eval_sampling_report = {
        "zero_valid_action_episodes": [
            {"dataset_name": "fixture", "episode_id": 9, "base_index": 50}
        ]
    }
    trainer.vla_eval_dataloader = [
        [
            {
                "action": np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32),
                "action_mask": np.ones((2, 2), dtype=bool),
                "action_is_pad": np.zeros(2, dtype=bool),
                "_heldout_eval_index": 0,
                "_heldout_eval_episode_id": 7,
            }
        ],
        [
            {
                "action": np.asarray([[9.0, 9.0], [9.0, 9.0]], dtype=np.float32),
                "action_mask": np.zeros((2, 2), dtype=bool),
                "action_is_pad": np.zeros(2, dtype=bool),
                "_heldout_eval_index": 1,
                "_heldout_eval_episode_id": 9,
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


def test_heldout_eval_rejects_valid_element_drift_from_sampling_report():
    trainer = _trainer()
    trainer.heldout_eval_expected_valid_elements = 5
    with pytest.raises(RuntimeError, match="valid action-element coverage"):
        trainer.eval_heldout_action_model({})


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
    trainer.heldout_eval_sampling_report = {"zero_valid_action_episodes": []}
    trainer.vla_eval_dataloader = [
        [
            {
                "action": action,
                "action_mask": np.ones_like(action, dtype=bool),
                "action_is_pad": np.zeros(50, dtype=bool),
                "_heldout_eval_index": 0,
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
