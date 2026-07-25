from types import SimpleNamespace

import torch

from starVLA.model.framework.VLA_JEPA import VLA_JEPA


def _make_model(video_target_shift_steps: int):
    model = object.__new__(VLA_JEPA)
    model.config = SimpleNamespace(
        datasets=SimpleNamespace(
            vla_data={
                "video_target_shift_steps": video_target_shift_steps,
                "qwen_observation_frame_index": "current",
            }
        )
    )
    model.depth_teacher_aux_cfg = {}
    return model


def test_current_observation_frame_is_first_for_forward_only_lerobot_clips():
    model = _make_model(video_target_shift_steps=0)

    assert model._qwen_observation_frame_index(total_frames=8) == 0
    assert model._resolve_video_frame_index("observation", total_frames=8) == 0


def test_current_observation_frame_is_last_context_frame_for_shifted_compact_clips():
    model = _make_model(video_target_shift_steps=2)

    compact = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1, 1)
    context, target = model._split_compact_videos(compact)

    assert context.shape[2] == 6
    assert target.shape[2] == 6
    assert context[0, 0, :, 0, 0, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert target[0, 0, :, 0, 0, 0].tolist() == [2, 3, 4, 5, 6, 7]

    assert model._qwen_observation_frame_index(total_frames=context.shape[2]) == 5
    assert model._resolve_video_frame_index("observation", total_frames=context.shape[2]) == 5

    teacher_frames = model._frames_for_depth_teacher(context, device=torch.device("cpu"))
    assert teacher_frames.shape == (1, 1, 1, 1)
    assert teacher_frames[0, 0, 0, 0].item() == 5


def test_explicit_first_and_last_frame_overrides_are_still_available():
    model = _make_model(video_target_shift_steps=2)

    assert model._resolve_video_frame_index("first", total_frames=6) == 0
    assert model._resolve_video_frame_index("last", total_frames=6) == 5
    assert model._resolve_video_frame_index(-2, total_frames=6) == 4
