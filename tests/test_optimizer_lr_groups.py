from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import (
    build_param_lr_groups,
)


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Linear(3, 3)
        self.head = torch.nn.Linear(3, 2)


def _config(*, learning_rate: dict, strict: bool = True):
    return OmegaConf.create(
        {
            "trainer": {
                "learning_rate": learning_rate,
                "strict_learning_rate_groups": strict,
                "freeze_modules": "",
            },
            "framework": {"qwenvl": {"lora": {"enabled": False}}},
        }
    )


def test_strict_learning_rate_groups_cover_named_and_base_parameters():
    model = _ToyModel()
    groups = build_param_lr_groups(
        model,
        _config(
            learning_rate={
                "base": 1.0e-5,
                "head": 5.0e-5,
            }
        ),
    )

    assert [group["name"] for group in groups] == ["head", "base"]
    assert [group["lr"] for group in groups] == [5.0e-5, 1.0e-5]
    grouped_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert {
        group["name"]: sum(
            parameter.numel() for parameter in group["params"]
        )
        for group in groups
    } == {
        "head": 8,
        "base": 12,
    }


def test_strict_learning_rate_groups_reject_missing_module_path():
    with pytest.raises(
        ValueError,
        match=r"trainer\.learning_rate\.missing does not resolve",
    ):
        build_param_lr_groups(
            _ToyModel(),
            _config(
                learning_rate={
                    "base": 1.0e-5,
                    "missing": 5.0e-5,
                }
            ),
        )


def test_strict_learning_rate_groups_reject_zero_trainable_parameters():
    model = _ToyModel()
    model.head.requires_grad_(False)

    with pytest.raises(
        ValueError,
        match="head resolved to a module with zero trainable parameters",
    ):
        build_param_lr_groups(
            model,
            _config(
                learning_rate={
                    "base": 1.0e-5,
                    "head": 5.0e-5,
                }
            ),
        )


def test_non_strict_learning_rate_groups_keep_legacy_base_fallback(capsys):
    groups = build_param_lr_groups(
        _ToyModel(),
        _config(
            learning_rate={
                "base": 1.0e-5,
                "missing": 5.0e-5,
            },
            strict=False,
        ),
    )

    assert [group["name"] for group in groups] == ["base"]
    assert "does not resolve to a model module" in capsys.readouterr().out
