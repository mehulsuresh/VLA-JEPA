import torch

from starVLA.model.modules.action_model.rtc_training import reduce_masked_loss
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


def test_action_is_pad_masks_padded_future_loss():
    model = _make_vlajepa_with_future_window(3)
    actions_target = torch.zeros(1, 4, 2)
    action_is_pad = torch.tensor([[False, False, False, True, True, True]])

    loss_mask = model._build_training_action_loss_mask(
        action_mask=None,
        action_is_pad=action_is_pad,
        actions_target=actions_target,
        device=actions_target.device,
    )

    expected = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    assert torch.equal(loss_mask, expected)

    base_loss = torch.ones_like(actions_target)
    changed_padded_loss = base_loss.clone()
    changed_padded_loss[:, 1:] = 1000.0

    assert torch.equal(
        reduce_masked_loss(base_loss, loss_mask=loss_mask),
        reduce_masked_loss(changed_padded_loss, loss_mask=loss_mask),
    )
