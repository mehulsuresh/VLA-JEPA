import torch

from starVLA.model.framework.VLA_JEPA import VLA_JEPA


def _make_vlajepa_with_future_window(future_action_window_size):
    model = object.__new__(VLA_JEPA)
    model.future_action_window_size = future_action_window_size
    return model


def test_training_action_chunk_uses_tail_future_window_plus_current_step():
    model = _make_vlajepa_with_future_window(2)
    actions = torch.arange(2 * 5 * 3).reshape(2, 5, 3)

    chunk = model._slice_training_action_chunk(actions)

    assert chunk.shape == (2, 3, 3)
    assert torch.equal(chunk, actions[:, -3:, :])


def test_training_action_mask_is_chunked_like_actions():
    model = _make_vlajepa_with_future_window(3)
    actions = torch.arange(1 * 6 * 2).reshape(1, 6, 2)
    action_mask = torch.zeros_like(actions)
    action_mask[:, -4:, :] = 1

    action_chunk = model._slice_training_action_chunk(actions)
    mask_chunk = model._slice_training_action_chunk(action_mask)

    assert torch.equal(action_chunk, actions[:, -4:, :])
    assert torch.equal(mask_chunk, action_mask[:, -4:, :])
    assert torch.all(mask_chunk == 1)
