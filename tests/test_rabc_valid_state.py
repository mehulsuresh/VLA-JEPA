from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from starVLA.training.train_starvla import VLATrainer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _trainer_for_valid_state_rabc():
    trainer = object.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        trainer={
            "use_rabc": True,
            "rabc_valid_state_key": "valid_state",
            "rabc_future_valid_state_key": "future_valid_state",
            "rabc_invalid_state_weight": 0.25,
            "rabc_mistake_weight": 0.25,
        }
    )
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    return trainer


def test_rabc_uses_valid_state_when_progress_delta_is_absent():
    trainer = _trainer_for_valid_state_rabc()
    batch = [
        {"valid_state": 1, "future_valid_state": 1},
        {"valid_state": 0, "future_valid_state": 0},
        {"valid_state": 1, "future_valid_state": 0},
    ]

    weights, stats = trainer._compute_rabc_weights(batch)

    assert weights is not None
    assert weights[0] > weights[1]
    assert weights[1] == weights[2]
    assert torch.isclose(stats["rabc_valid_state_ratio"], torch.tensor(1.0 / 3.0))


def test_rabc_combines_valid_state_with_progress_delta():
    trainer = _trainer_for_valid_state_rabc()
    batch = [
        {"rabc_progress_delta": 0.2, "valid_state": 1, "future_valid_state": 1},
        {"rabc_progress_delta": 0.2, "valid_state": 0, "future_valid_state": 0},
    ]

    weights, stats = trainer._compute_rabc_weights(batch)

    assert weights is not None
    assert weights[0] > weights[1]
    assert torch.isclose(stats["rabc_valid_state_ratio"], torch.tensor(0.5))


def test_trossen_training_uses_sustained_prefix_mask_instead_of_rabc():
    config = OmegaConf.load(
        REPO_ROOT
        / "scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_5090_lerobot.yaml"
    )

    assert config.datasets.vla_data.use_action_validity_prefix_mask is True
    assert config.datasets.vla_data.action_validity_invalid_run_length == 10
    assert config.trainer.use_rabc is False
