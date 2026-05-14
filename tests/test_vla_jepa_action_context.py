import torch

from starVLA.model.framework.VLA_JEPA import VLA_JEPA
from starVLA.model.modules.vlm.QWen3_5 import _QWen3_5_Interface


def test_action_head_context_keeps_prompt_image_and_embodied_tokens_only():
    model = object.__new__(VLA_JEPA)
    model._action_token_ids_t = torch.tensor([10, 11], dtype=torch.long)
    model._embodied_token_id_t = torch.tensor([20], dtype=torch.long)
    model._geometry_token_ids_t = torch.tensor([30, 31], dtype=torch.long)
    model._qwen_state_token_ids_t = torch.tensor([40], dtype=torch.long)
    model._qwen_image_token_id = 99

    input_ids = torch.tensor(
        [
            [0, 101, 99, 40, 20, 10, 102, 30],
            [201, 99, 202, 40, 20, 10, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    last_hidden = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)

    context, key_keep_mask, key_block_ids = VLA_JEPA._build_action_head_context(
        model,
        last_hidden=last_hidden,
        qwen_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )

    expected_keep = torch.tensor(
        [
            [False, True, True, True, True, False, False, False],
            [True, True, True, True, True, False, False, False],
        ]
    )

    assert context.shape == last_hidden.shape
    assert torch.equal(context[expected_keep], last_hidden[expected_keep])
    assert torch.all(context[~expected_keep] == 0)
    assert torch.equal(key_keep_mask, expected_keep)
    assert key_block_ids.tolist() == [
        [-1, 0, 1, 2, 3, -1, -1, -1],
        [0, 1, 0, 2, 3, -1, -1, -1],
    ]


def test_blockwise_cross_attention_mask_uses_pi0_style_blocks():
    key_keep_mask = torch.tensor([[True, True, True, True, False]])
    key_block_ids = torch.tensor([[0, 1, 2, -1, 0]])
    query_block_ids = torch.tensor([0, 1, 2])

    attention_mask = VLA_JEPA._build_blockwise_cross_attention_mask(
        key_keep_mask=key_keep_mask,
        key_block_ids=key_block_ids,
        query_block_ids=query_block_ids,
        dtype=torch.float32,
    )

    expected_visible = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
        ]
    )
    assert attention_mask.shape == (1, 3, 5)
    assert torch.all(attention_mask[0][expected_visible] == 0)
    assert torch.all(attention_mask[0][~expected_visible] == -10000.0)


def test_action_head_masks_use_embodied_and_noisy_action_blocks():
    model = object.__new__(VLA_JEPA)
    model.config = type(
        "Cfg",
        (),
        {
            "framework": type(
                "Framework",
                (),
                {
                    "action_model": type(
                        "ActionModelCfg",
                        (),
                        {"num_target_vision_tokens": 2},
                    )()
                },
            )()
        },
    )()

    key_keep_mask = torch.tensor([[True, True, True, True]])
    key_block_ids = torch.tensor([[0, 1, 2, 3]])

    encoder_mask = VLA_JEPA._build_action_head_encoder_attention_mask(
        model,
        key_keep_mask=key_keep_mask,
        key_block_ids=key_block_ids,
        action_horizon=3,
        dtype=torch.float32,
    )

    assert encoder_mask.shape == (1, 5, 4)
    assert torch.all(encoder_mask == 0)

    self_mask = VLA_JEPA._build_action_head_self_attention_mask(
        model,
        batch_size=1,
        action_horizon=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_visible = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ]
    )
    assert self_mask.shape == (1, 5, 5)
    assert torch.all(self_mask[0][expected_visible] == 0)
    assert torch.all(self_mask[0][~expected_visible] == -10000.0)


def test_prompt_places_state_and_embodied_queries_before_auxiliary_tokens():
    model = object.__new__(VLA_JEPA)
    model.qwen_state_projector = object()

    prompt = (
        "Your task is {instruction}. Infer from frames {actions} {geometry} "
        "and produce actions {e_actions}."
    )

    reordered = VLA_JEPA._move_action_head_placeholders_before_auxiliary_tokens(
        model,
        prompt,
        has_actions=True,
        has_state=True,
    )

    assert reordered.index("{state}") < reordered.index("{actions}")
    assert reordered.index("{e_actions}") < reordered.index("{actions}")
    assert "{state}" in reordered
    assert reordered.count("{e_actions}") == 1


def test_qwen_prompt_split_places_images_before_state_and_action_slots():
    interface = object.__new__(_QWen3_5_Interface)
    prompt = (
        "Your task is stack the cup. "
        "<|state_0|><|state_1|><|embodied_action|><|action_0|>"
    )

    prefix, suffix, use_interleaved = _QWen3_5_Interface._split_prompt_for_interleaved_images(
        interface,
        prompt,
        prompt_replace_dict={
            "{state}": "<|state_0|><|state_1|>",
            "{e_actions}": "<|embodied_action|>",
            "{actions}": "<|action_0|>",
        },
    )

    assert use_interleaved
    assert prefix == "Your task is stack the cup. "
    assert suffix.startswith("<|state_0|>")
