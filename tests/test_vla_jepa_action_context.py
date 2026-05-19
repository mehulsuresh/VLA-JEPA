import inspect
import torch
import yaml
from pathlib import Path
from types import SimpleNamespace

from starVLA.model.framework.VLA_JEPA import VLA_JEPA
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
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


def test_qwen_blockwise_attention_uses_pi0_prefix_state_action_aux_blocks():
    model = object.__new__(VLA_JEPA)
    model._action_token_ids_t = torch.tensor([10, 11], dtype=torch.long)
    model._embodied_token_id_t = torch.tensor([20], dtype=torch.long)
    model._geometry_token_ids_t = torch.tensor([30], dtype=torch.long)
    model._qwen_state_token_ids_t = torch.tensor([40, 41], dtype=torch.long)

    input_ids = torch.tensor(
        [
            [101, 99, 40, 41, 20, 20, 10, 11, 30, 102, 0],
            [201, 99, 202, 40, 20, 10, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    block_ids = VLA_JEPA._build_qwen_blockwise_attention_block_ids(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )

    assert block_ids.tolist() == [
        [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, -1],
        [0, 0, 0, 1, 2, 3, -1, -1, -1, -1, -1],
    ]

    visible = VLA_JEPA._build_blockwise_visibility_from_block_ids(block_ids)
    assert visible[0, 0, 1]  # prefix is bidirectional internally
    assert visible[0, 1, 0]
    assert visible[0, 3, 0]  # state sees prefix
    assert visible[0, 3, 2]  # state is bidirectional internally
    assert not visible[0, 0, 2]  # prefix cannot see state
    assert visible[0, 4, 3]  # embodied action sees state
    assert visible[0, 5, 4]  # embodied action is bidirectional internally
    assert not visible[0, 3, 4]  # state cannot see embodied action
    assert visible[0, 8, 0]  # aux sees prefix
    assert visible[0, 8, 9]  # aux/trailing suffix is bidirectional internally
    assert not visible[0, 4, 8]  # embodied action cannot see aux


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


def test_qwen3_vl_wrapper_exposes_vla_jepa_contract():
    required_methods = [
        "forward_features",
        "build_qwenvl_inputs",
        "build_qwenvl_inputs_from_frames_tensor",
        "prepare_for_compile",
        "supports_blockwise_attention",
    ]

    for method_name in required_methods:
        assert hasattr(_QWen3_VL_Interface, method_name)


def test_qwen3_vl_respects_explicit_sdpa_attention_backend():
    assert _QWen3_VL_Interface._resolve_attn_implementation("sdpa") == "sdpa"


def test_qwen3_blockwise_attention_uses_flex_block_mask():
    interface = object.__new__(_QWen3_VL_Interface)
    interface.attn_implementation = "flex_attention"
    interface.config = SimpleNamespace(
        framework={"qwenvl": {"blockwise_attention": {"compile_mask": False}}}
    )

    block_mask = _QWen3_VL_Interface._build_blockwise_flex_attention_mask(
        interface,
        block_ids=torch.tensor([[0, 0, 1]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        input_ids=torch.tensor([[101, 102, 103]], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert type(block_mask).__name__ == "BlockMask"
    assert block_mask.shape == (1, 1, 3, 3)


def test_vlm_factory_keeps_qwen35_rollback_route():
    source = inspect.getsource(get_vlm_model)

    assert "Qwen3.5" in source
    assert "_QWen3_5_Interface" in source
    assert "Qwen3-VL" in source
    assert "_QWen3_VL_Interface" in source


def test_vla_jepa_configs_declare_state_tokens_and_ordered_prompts():
    for path in sorted(Path("scripts/config").glob("vlajepa_robot_ft*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        framework_cfg = cfg["framework"]
        assert framework_cfg["name"] == "VLA_JEPA", path
        qwenvl_cfg = framework_cfg["qwenvl"]
        assert qwenvl_cfg["base_vlm"] == "Qwen/Qwen3-VL-2B-Instruct", path
        assert qwenvl_cfg["vl_hidden_dim"] == "auto", path
        assert framework_cfg["action_model"]["diffusion_model_cfg"]["cross_attention_dim"] == "auto", path
        assert framework_cfg["qwen_state"] == {
            "num_tokens": 8,
            "token_template": "<|state_{}|>",
        }, path

        blockwise_enabled = bool(qwenvl_cfg.get("blockwise_attention", {}).get("enabled", False))
        if blockwise_enabled:
            assert qwenvl_cfg["attn_implementation"] == "flex_attention", path
            freeze_modules = cfg["trainer"].get("freeze_modules", "")
            assert "qwen_vl_interface.model" not in freeze_modules, path
            assert not bool(qwenvl_cfg.get("lora", {}).get("enabled", False)), path
        else:
            assert qwenvl_cfg["attn_implementation"] in {"sdpa", "flash_attention_2", "flash_attention_4"}, path

        prompt = cfg["datasets"]["vla_data"]["CoT_prompt"]
        assert "temporal dynamics from frames" not in prompt, path
        assert prompt.index("{state}") < prompt.index("{e_actions}") < prompt.index("{actions}"), path
        if cfg["datasets"]["vla_data"].get("dataset_py") == "lerobot_datasets":
            assert cfg["datasets"]["vla_data"].get("video_backend") == "pyav", path
        if framework_cfg.get("depth_teacher_aux", {}).get("enabled", False):
            assert "{geometry}" in prompt, path
            assert prompt.index("{actions}") < prompt.index("{geometry}"), path
        else:
            assert "{geometry}" not in prompt, path
