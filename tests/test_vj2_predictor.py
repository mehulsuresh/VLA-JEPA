from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC


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
