import pytest
from omegaconf import OmegaConf

from starVLA.training.train_starvla import (
    _requested_torch_compile_flags,
    _training_uses_deepspeed,
    build_model,
)


def test_deepspeed_compile_guard_raises_before_model_load(monkeypatch):
    monkeypatch.setenv("STARVLA_USE_DEEPSPEED", "1")
    cfg = OmegaConf.create(
        {
            "trainer": {
                "compile_qwen_model": True,
                "compile_action_model": False,
                "compile_vj_predictor": False,
                "compile_vj_encoder": False,
                "compile_full_model": False,
            },
            "framework": {"qwenvl": {"base_vlm": "unused"}},
        }
    )

    with pytest.raises(RuntimeError, match="DeepSpeed training was requested"):
        build_model(cfg)


def test_deepspeed_detection_uses_accelerator_state(monkeypatch):
    monkeypatch.delenv("STARVLA_USE_DEEPSPEED", raising=False)
    monkeypatch.delenv("ACCELERATE_DISTRIBUTED_TYPE", raising=False)
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    cfg = OmegaConf.create({"trainer": {"_accelerate_distributed_type": "DistributedType.DEEPSPEED"}})

    assert _training_uses_deepspeed(cfg)


def test_requested_torch_compile_flags_are_exact():
    cfg = OmegaConf.create(
        {
            "compile_qwen_model": True,
            "compile_action_model": False,
            "compile_vj_predictor": True,
            "compile_vj_encoder": False,
            "compile_full_model": False,
        }
    )

    assert _requested_torch_compile_flags(cfg) == [
        "compile_qwen_model",
        "compile_vj_predictor",
    ]
