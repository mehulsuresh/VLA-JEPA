import pytest
from omegaconf import OmegaConf

from starVLA.training import train_starvla


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
