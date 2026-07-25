import pytest
from omegaconf import OmegaConf

from starVLA.model.framework.VLA_JEPA import (
    _resolve_vj_predictor_attention_backend,
)
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC


def _make_predictor(*, use_flash_attention: bool):
    return VisionTransformerPredictorAC(
        img_size=(32, 32),
        patch_size=16,
        num_frames=2,
        tubelet_size=1,
        embed_dim=16,
        predictor_embed_dim=32,
        depth=2,
        num_heads=4,
        action_embed_dim=8,
        num_add_tokens=2,
        use_extrinsics=False,
        use_flash_attention=use_flash_attention,
    )


@pytest.mark.parametrize(
    ("backend", "enable_env", "disable_env", "expected_flash"),
    [
        ("torch_sdpa", "1", "0", False),
        ("flash_attn", "0", "1", True),
    ],
)
def test_predictor_attention_backend_is_config_owned_not_environment_owned(
    monkeypatch,
    backend,
    enable_env,
    disable_env,
    expected_flash,
):
    monkeypatch.setenv("STARVLA_ENABLE_FLASH_ATTN_WORLD_MODEL", enable_env)
    monkeypatch.setenv("STARVLA_DISABLE_FLASH_ATTN_WORLD_MODEL", disable_env)
    vj2_model_cfg = OmegaConf.create({"predictor_attention_backend": backend})

    resolved_backend = _resolve_vj_predictor_attention_backend(vj2_model_cfg)
    model = _make_predictor(
        use_flash_attention=(resolved_backend == "flash_attn")
    )

    assert resolved_backend == backend
    assert [
        block.attn.use_flash_attention
        for block in model.predictor_blocks
    ] == [expected_flash, expected_flash]


def test_predictor_attention_backend_rejects_unknown_config_value():
    with pytest.raises(
        ValueError,
        match="predictor_attention_backend must be 'torch_sdpa' or 'flash_attn'",
    ):
        _resolve_vj_predictor_attention_backend(
            OmegaConf.create({"predictor_attention_backend": "ambient"})
        )


def test_predictor_attention_backend_must_be_explicit():
    with pytest.raises(
        ValueError,
        match="predictor_attention_backend is required",
    ):
        _resolve_vj_predictor_attention_backend(OmegaConf.create({}))


def test_inactive_vj_predictor_condition_encoders_are_frozen():
    model = VisionTransformerPredictorAC(
        img_size=(32, 32),
        patch_size=16,
        num_frames=2,
        tubelet_size=1,
        embed_dim=16,
        predictor_embed_dim=32,
        depth=1,
        num_heads=4,
        action_embed_dim=8,
        num_add_tokens=2,
        use_extrinsics=False,
    )

    assert not any(param.requires_grad for param in model.state_encoder.parameters())
    assert not any(param.requires_grad for param in model.extrinsics_encoder.parameters())


def test_vj_predictor_extrinsics_encoder_trains_when_enabled():
    model = VisionTransformerPredictorAC(
        img_size=(32, 32),
        patch_size=16,
        num_frames=2,
        tubelet_size=1,
        embed_dim=16,
        predictor_embed_dim=32,
        depth=1,
        num_heads=4,
        action_embed_dim=8,
        num_add_tokens=3,
        use_extrinsics=True,
    )

    assert not any(param.requires_grad for param in model.state_encoder.parameters())
    assert all(param.requires_grad for param in model.extrinsics_encoder.parameters())
