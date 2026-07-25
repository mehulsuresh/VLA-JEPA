import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.training.train_starvla import VLATrainer


class _EvalProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.calls = []

    def predict_action(self, **kwargs):
        self.calls.append(
            {
                "kwargs": kwargs,
                "parent_training": self.training,
                "dropout_training": self.dropout.training,
            }
        )
        return {
            "normalized_actions": np.zeros((1, 2, 2), dtype=np.float32),
        }


class _Accelerator:
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def reduce(value, reduction):
        assert reduction == "sum"
        return value


def _trainer(*, allow=True):
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "framework": {"action_model": {"num_inference_timesteps": 8}},
            "trainer": {"allow_training_stream_eval": allow},
        }
    )
    trainer.accelerator = _Accelerator()
    trainer.model = _EvalProbeModel()
    # Verify exact restoration rather than a blanket model.train().
    trainer.model.train()
    trainer.model.dropout.eval()
    trainer._get_next_eval_batch = lambda: {
        "action": torch.tensor([[[3.0, 4.0], [9.0, 9.0]]]),
        "action_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, True]]),
    }
    return trainer


def test_training_stream_probe_uses_deployment_steps_no_rtc_and_true_rmse():
    trainer = _trainer()

    metrics = trainer.eval_action_model({})

    call = trainer.model.calls[0]
    assert call["parent_training"] is False
    assert call["dropout_training"] is False
    assert call["kwargs"]["num_inference_timesteps"] == 8
    assert call["kwargs"]["prev_actions"] is None
    assert call["kwargs"]["prefix_len"] == 0
    assert call["kwargs"]["rtc_config"] is None
    assert metrics["train_stream_probe_normalized_action_mae"] == pytest.approx(3.5)
    assert metrics["train_stream_probe_normalized_action_rmse"] == pytest.approx(
        math.sqrt(12.5)
    )
    assert "mae_score" not in metrics
    assert "norm_l2_per_element" not in metrics
    assert trainer.model.training is True
    assert trainer.model.dropout.training is False


def test_training_stream_probe_is_fail_closed_by_default():
    trainer = _trainer(allow=False)
    fetches = 0

    def _unexpected_fetch():
        nonlocal fetches
        fetches += 1
        raise AssertionError("training iterator must not advance")

    trainer._get_next_eval_batch = _unexpected_fetch

    with pytest.raises(RuntimeError, match="disabled by default"):
        trainer.eval_action_model({})
    assert fetches == 0
