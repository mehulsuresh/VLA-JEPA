import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from starVLA.training import train_starvla
from starVLA.training.train_starvla import VLATrainer


class _FakeAccelerator:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.local_process_index = 0
        self.distributed_type = "NO"
        self.num_processes = 8
        self.gradient_accumulation_steps = kwargs.get(
            "gradient_accumulation_steps", 1
        )
        self.state = "fake-accelerator-state"
        _FakeAccelerator.instances.append(self)

    def print(self, *args, **kwargs):
        return None


def _cfg(tmp_path, trainer=None):
    return OmegaConf.create(
        {
            "run_root_dir": str(tmp_path),
            "run_id": "scheduler_test",
            "trackers": [],
            "trainer": {
                "enable_mixed_precision_training": False,
                **(trainer or {}),
            },
        }
    )


def test_accelerator_does_not_auto_step_scheduler_by_default(monkeypatch, tmp_path):
    _FakeAccelerator.instances.clear()
    monkeypatch.setattr(train_starvla, "Accelerator", _FakeAccelerator)
    monkeypatch.setattr(train_starvla.torch.cuda, "is_available", lambda: False)

    cfg = _cfg(tmp_path)
    train_starvla.build_accelerator(cfg)

    assert _FakeAccelerator.instances[-1].kwargs["step_scheduler_with_optimizer"] is False
    assert _FakeAccelerator.instances[-1].kwargs["gradient_accumulation_steps"] == 1
    assert cfg.trainer._accelerate_step_scheduler_with_optimizer is False
    assert cfg.trainer._accelerate_gradient_accumulation_steps == 1


def test_accelerator_scheduler_auto_step_can_be_enabled_explicitly(monkeypatch, tmp_path):
    _FakeAccelerator.instances.clear()
    monkeypatch.setattr(train_starvla, "Accelerator", _FakeAccelerator)
    monkeypatch.setattr(train_starvla.torch.cuda, "is_available", lambda: False)

    cfg = _cfg(tmp_path, {"step_scheduler_with_optimizer": True})
    train_starvla.build_accelerator(cfg)

    assert _FakeAccelerator.instances[-1].kwargs["step_scheduler_with_optimizer"] is True
    assert cfg.trainer._accelerate_step_scheduler_with_optimizer is True


def test_accelerator_uses_configured_gradient_accumulation(monkeypatch, tmp_path):
    _FakeAccelerator.instances.clear()
    monkeypatch.setattr(train_starvla, "Accelerator", _FakeAccelerator)
    monkeypatch.setattr(train_starvla.torch.cuda, "is_available", lambda: False)

    cfg = _cfg(tmp_path, {"gradient_accumulation_steps": 4})
    train_starvla.build_accelerator(cfg)

    accelerator = _FakeAccelerator.instances[-1]
    assert accelerator.kwargs["gradient_accumulation_steps"] == 4
    assert accelerator.gradient_accumulation_steps == 4
    assert cfg.trainer._accelerate_gradient_accumulation_steps == 4


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "not-an-int"])
def test_accelerator_rejects_invalid_gradient_accumulation(
    monkeypatch, tmp_path, value
):
    monkeypatch.setattr(train_starvla, "Accelerator", _FakeAccelerator)
    monkeypatch.setattr(train_starvla.torch.cuda, "is_available", lambda: False)
    cfg = _cfg(tmp_path, {"gradient_accumulation_steps": value})

    with pytest.raises(ValueError, match="must be a positive integer"):
        train_starvla.build_accelerator(cfg)


def _advance_adamw_and_scheduler(model, optimizer, scheduler, steps):
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.125)
        optimizer.step()
        scheduler.step()


def _linear_scheduler(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.05, 1.0 - 0.03 * step),
    )
    return optimizer, scheduler


class _CheckpointLoadingAccelerator:
    def __init__(self, model, optimizer, scheduler, state):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = state
        self.loaded_path = None

    def load_state(self, checkpoint_path):
        self.loaded_path = checkpoint_path
        self.model.load_state_dict(copy.deepcopy(self.state["model"]))
        self.optimizer.load_state_dict(copy.deepcopy(self.state["optimizer"]))
        self.scheduler.load_state_dict(copy.deepcopy(self.state["scheduler"]))

    def unwrap_model(self, _model):
        return self.model

    def print(self, *_args, **_kwargs):
        return None


def _checkpoint_trainer(tmp_path, *, load_optimizer_state, completed_steps=10):
    checkpoint = tmp_path / "steps_10"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"completed_steps": completed_steps}),
        encoding="utf-8",
    )
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {"trainer": {"resume_load_optimizer_state": load_optimizer_state}}
    )
    trainer.completed_steps = 0
    trainer.best_metric_value = None
    return trainer, checkpoint


def test_full_state_resume_preserves_lambdalr_state_and_next_lr(tmp_path):
    torch.manual_seed(7)
    reference_model = torch.nn.Linear(3, 2)
    reference_optimizer, reference_scheduler = _linear_scheduler(reference_model)
    _advance_adamw_and_scheduler(
        reference_model, reference_optimizer, reference_scheduler, 10
    )
    saved_state = {
        "model": copy.deepcopy(reference_model.state_dict()),
        "optimizer": copy.deepcopy(reference_optimizer.state_dict()),
        "scheduler": copy.deepcopy(reference_scheduler.state_dict()),
    }

    resumed_model = torch.nn.Linear(3, 2)
    resumed_optimizer, resumed_scheduler = _linear_scheduler(resumed_model)
    accelerator = _CheckpointLoadingAccelerator(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        saved_state,
    )
    trainer, checkpoint = _checkpoint_trainer(
        tmp_path, load_optimizer_state=True
    )
    trainer.model = resumed_model
    trainer.optimizer = resumed_optimizer
    trainer.lr_scheduler = resumed_scheduler
    trainer.accelerator = accelerator

    trainer._load_checkpoint(checkpoint)

    assert accelerator.loaded_path == str(checkpoint.resolve())
    assert trainer.completed_steps == 10
    assert resumed_scheduler.state_dict() == reference_scheduler.state_dict()
    assert resumed_scheduler.last_epoch == 10
    assert resumed_scheduler._step_count == 11
    assert [group["lr"] for group in resumed_optimizer.param_groups] == [
        group["lr"] for group in reference_optimizer.param_groups
    ]

    _advance_adamw_and_scheduler(
        reference_model, reference_optimizer, reference_scheduler, 1
    )
    _advance_adamw_and_scheduler(
        resumed_model, resumed_optimizer, resumed_scheduler, 1
    )

    assert resumed_scheduler.state_dict() == reference_scheduler.state_dict()
    assert [group["lr"] for group in resumed_optimizer.param_groups] == [
        group["lr"] for group in reference_optimizer.param_groups
    ]
    for resumed_parameter, reference_parameter in zip(
        resumed_model.parameters(), reference_model.parameters(), strict=True
    ):
        torch.testing.assert_close(resumed_parameter, reference_parameter, rtol=0, atol=0)


def test_model_only_resume_still_fast_forwards_fresh_lambdalr(tmp_path):
    torch.manual_seed(11)
    source_model = torch.nn.Linear(3, 2)
    trainer, checkpoint = _checkpoint_trainer(
        tmp_path, load_optimizer_state=False
    )
    torch.save(source_model.state_dict(), checkpoint / "pytorch_model.pt")

    resumed_model = torch.nn.Linear(3, 2)
    resumed_optimizer, resumed_scheduler = _linear_scheduler(resumed_model)
    accelerator = _CheckpointLoadingAccelerator(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        {},
    )
    trainer.model = resumed_model
    trainer.optimizer = resumed_optimizer
    trainer.lr_scheduler = resumed_scheduler
    trainer.accelerator = accelerator

    trainer._load_checkpoint(checkpoint)

    assert accelerator.loaded_path is None
    assert trainer.completed_steps == 10
    assert resumed_scheduler.last_epoch == 10
    assert resumed_scheduler._step_count == 2
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.007)
    for resumed_parameter, source_parameter in zip(
        resumed_model.parameters(), source_model.parameters(), strict=True
    ):
        torch.testing.assert_close(resumed_parameter, source_parameter, rtol=0, atol=0)


def test_eval_only_model_only_resume_accepts_external_checkpoint_without_mutation(
    tmp_path,
):
    source_checkpoint = tmp_path / "external-source" / "steps_10"
    source_checkpoint.mkdir(parents=True)
    (source_checkpoint / "trainer_state.json").write_text(
        json.dumps({"completed_steps": 10}),
        encoding="utf-8",
    )
    torch.manual_seed(13)
    source_model = torch.nn.Linear(3, 2)
    torch.save(source_model.state_dict(), source_checkpoint / "pytorch_model.pt")

    eval_output = tmp_path / "separate-eval-output"
    eval_output.mkdir()
    sentinel = eval_output / "sentinel.txt"
    sentinel.write_text("eval output must remain selection-free\n", encoding="utf-8")

    resumed_model = torch.nn.Linear(3, 2)
    accelerator = _CheckpointLoadingAccelerator(resumed_model, None, None, {})
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "output_dir": str(eval_output),
            "trainer": {
                "eval_only": True,
                "resume_load_optimizer_state": False,
            },
        }
    )
    trainer.checkpoint_dir = str(eval_output / "checkpoints")
    trainer.model = resumed_model
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.accelerator = accelerator
    trainer.completed_steps = 0
    trainer.best_metric_value = None

    trainer._load_checkpoint(source_checkpoint)

    assert accelerator.loaded_path is None
    assert trainer.loaded_checkpoint_path == str(source_checkpoint.resolve())
    assert trainer.completed_steps == 10
    for resumed_parameter, source_parameter in zip(
        resumed_model.parameters(), source_model.parameters(), strict=True
    ):
        torch.testing.assert_close(resumed_parameter, source_parameter, rtol=0, atol=0)
    assert not (eval_output / "best_checkpoint.json").exists()
    assert not Path(trainer.checkpoint_dir).exists()
    assert sentinel.read_text(encoding="utf-8") == (
        "eval output must remain selection-free\n"
    )
