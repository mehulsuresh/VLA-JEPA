import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from starVLA.model.framework.VLA_JEPA import VLA_JEPA
from starVLA.model.modules.geometry_teacher import (
    MoGeGeometryTeacher,
    QueryGeometryTeacherHead,
    query_feature_distillation_loss,
)
from starVLA.training.trainer_utils.trainer_tools import (
    is_depth_teacher_aux_missing_key_allowed,
    is_depth_teacher_aux_unexpected_key_allowed,
)


def test_known_moge_feature_dim_without_loading_weights_on_cpu():
    dims = [1024, 256, 128, 64, 32]
    for level, expected_dim in enumerate(dims):
        teacher = MoGeGeometryTeacher(
            {
                "teacher_model": "Ruicheng/moge-2-vitl-normal",
                "teacher_feature_source": "neck",
                "teacher_feature_level": level,
                "teacher_feature_dim": "auto",
            }
        )
        teacher.initialize(torch.device("cpu"))
        assert teacher.model is None
        assert teacher.feature_dim() == expected_dim


def test_loss_finite_at_zero_init():
    pred = torch.zeros(1, 16, 8, requires_grad=True)
    target = torch.randn(1, 8, 4, 4)
    loss, metrics = query_feature_distillation_loss(
        pred,
        {"features": target},
        {"query_smooth_l1_beta": 1.0},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["depth_teacher_query_smooth_l1_loss"])
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_loss_drives_pred_toward_target():
    torch.manual_seed(0)
    pred = torch.nn.Parameter(torch.zeros(1, 16, 8))
    target = torch.randn(1, 8, 4, 4)
    opt = torch.optim.Adam([pred], lr=5e-2)
    cfg = {"query_smooth_l1_beta": 1.0}

    initial, _ = query_feature_distillation_loss(pred, {"features": target}, cfg)
    for _ in range(200):
        loss, _ = query_feature_distillation_loss(pred, {"features": target}, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final, _ = query_feature_distillation_loss(pred, {"features": target}, cfg)

    assert final.item() < initial.item() * 0.1


def test_query_geometry_head_shape():
    torch.manual_seed(0)
    head = QueryGeometryTeacherHead(
        hidden_size=16,
        output_size=8,
        num_output_tokens=16,
        num_layers=1,
        num_heads=2,
        dim_head=4,
    )
    context = torch.randn(6, 12, 16)
    pred = head(context)

    assert pred.shape == (6, 16, 8)


def test_image_token_gather_alignment():
    class DummyVLA:
        _qwen_image_token_id = 99

        def _qwen_image_merge_size(self):
            return 1

    dummy = DummyVLA()
    input_ids = torch.tensor(
        [
            [0, 99, 99, 1, 2, 99, 99, 3],
            [99, 4, 5, 99, 99, 6, 7, 99],
        ]
    )
    last_hidden = torch.arange(input_ids.numel(), dtype=torch.float32).view(2, 8, 1)
    image_grid_thw = torch.tensor(
        [
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 2],
        ]
    )

    tokens, token_grid_hw = VLA_JEPA._extract_qwen_image_hidden(
        dummy,
        last_hidden=last_hidden,
        qwen_inputs={"input_ids": input_ids, "image_grid_thw": image_grid_thw},
        batch_size=2,
        num_views=2,
    )

    assert token_grid_hw == (1, 2)
    assert tokens.squeeze(-1).tolist() == [[1.0, 2.0], [5.0, 6.0], [8.0, 11.0], [12.0, 15.0]]


def test_image_token_gather_alignment_with_variable_qwen_views():
    class DummyVLA:
        _qwen_image_token_id = 99

        def _qwen_image_merge_size(self):
            return 1

    dummy = DummyVLA()
    input_ids = torch.tensor(
        [
            [99, 99, 1, 99, 99, 2, 99, 99],
            [3, 99, 99, 4, 5, 6, 7, 8],
        ]
    )
    last_hidden = torch.arange(input_ids.numel(), dtype=torch.float32).view(2, 8, 1)
    image_grid_thw = torch.tensor(
        [
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 2],
        ]
    )

    tokens, token_grid_hw = VLA_JEPA._extract_qwen_image_hidden(
        dummy,
        last_hidden=last_hidden,
        qwen_inputs={"input_ids": input_ids, "image_grid_thw": image_grid_thw},
        batch_size=2,
        num_views=3,
        qwen_view_counts=[3, 1],
        qwen_selected_view_indices=[[1, 1, 0], [0, 0, 0]],
    )

    assert token_grid_hw == (1, 2)
    assert tokens.squeeze(-1).tolist() == [
        [3.0, 4.0],
        [3.0, 4.0],
        [0.0, 1.0],
        [9.0, 10.0],
        [9.0, 10.0],
        [9.0, 10.0],
    ]


def test_module_out_channels_prefers_residual_output_dim():
    block = SimpleNamespace(
        neck=SimpleNamespace(
            output_blocks=nn.ModuleList([nn.Identity()]),
            input_blocks=nn.ModuleList([nn.Identity()]),
            res_blocks=nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Sequential(
                            nn.Conv2d(3, 6, kernel_size=1),
                            nn.ReLU(),
                            nn.Conv2d(6, 3, kernel_size=1),
                        )
                    )
                ]
            ),
        )
    )
    teacher = MoGeGeometryTeacher({"teacher_feature_source": "neck", "teacher_feature_level": 0})
    assert teacher._feature_dim_from_model(block) == 3


def test_depth_teacher_aux_checkpoint_key_filters():
    enabled_cfg = {"framework": {"depth_teacher_aux": {"enabled": True}}}
    disabled_cfg = {"framework": {"depth_teacher_aux": {"enabled": False}}}

    assert is_depth_teacher_aux_missing_key_allowed(enabled_cfg, "depth_teacher_aux_head.net.0.weight")
    assert not is_depth_teacher_aux_missing_key_allowed(disabled_cfg, "depth_teacher_aux_head.net.0.weight")
    assert is_depth_teacher_aux_unexpected_key_allowed(disabled_cfg, "depth_teacher_aux_head.net.0.weight")
    assert not is_depth_teacher_aux_unexpected_key_allowed(enabled_cfg, "depth_teacher_aux_head.net.0.weight")
    assert not is_depth_teacher_aux_unexpected_key_allowed(disabled_cfg, "action_model.weight")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not (os.environ.get("STARVLA_MOGE_REPO_PATH") or os.environ.get("MOGE_REPO_PATH")),
    reason="requires CUDA and a local MoGe checkout path",
)
def test_moge_loads_and_feature_shape():
    teacher = MoGeGeometryTeacher(
        {
            "teacher_model": "Ruicheng/moge-2-vitl-normal",
            "teacher_feature_source": "neck",
            "teacher_feature_level": 0,
            "teacher_feature_dim": "auto",
            "frame_value_range": "0_1",
            "input_size": 224,
            "num_tokens": 256,
        }
    )
    teacher.initialize(torch.device("cuda"))
    assert teacher.feature_dim() == 1024
    out = teacher.infer_features(torch.rand(2, 3, 224, 224, device="cuda"))
    assert out["features"].shape == (2, 1024, 16, 16)
