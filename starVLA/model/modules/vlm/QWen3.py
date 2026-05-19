# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl_modeling
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.utils import is_flash_attn_2_available

try:
    from transformers.utils import is_flash_attn_4_available
except ImportError:
    def is_flash_attn_4_available() -> bool:
        return False


logger = get_logger(__name__)

IGNORE_INDEX = -100
_ACTION_TOKEN_MIN = 151936
_ACTION_TOKEN_MAX = 153984


class _QWen3_VL_Interface(nn.Module):
    """Wrapper around Qwen3-VL multimodal models."""

    @staticmethod
    def _safe_log(level: str, message: str) -> None:
        try:
            getattr(logger, level)(message)
        except RuntimeError:
            print(message)

    @staticmethod
    def _normalize_attn_implementation_name(requested: Optional[str]) -> str:
        normalized = (requested or "flash_attention_2").strip().lower()
        alias_map = {
            "flash": "flash_attention_2",
            "flash2": "flash_attention_2",
            "flash4": "flash_attention_4",
            "flash_attn": "flash_attention_2",
            "flash_attn_2": "flash_attention_2",
            "flash_attn_4": "flash_attention_4",
            "flash-attn": "flash_attention_2",
            "flash-attn-2": "flash_attention_2",
            "flash-attn-4": "flash_attention_4",
            "flex": "flex_attention",
            "flex-attn": "flex_attention",
        }
        return alias_map.get(normalized, normalized)

    @staticmethod
    def _resolve_attention_head_dim(model_id: str) -> Optional[int]:
        try:
            model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            _QWen3_VL_Interface._safe_log(
                "warning", f"Could not inspect `{model_id}` attention head dimension before loading: {exc}"
            )
            return None

        text_config = getattr(model_config, "text_config", model_config)
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is not None:
            return int(head_dim)
        hidden_size = getattr(text_config, "hidden_size", None)
        num_heads = getattr(text_config, "num_attention_heads", None)
        if hidden_size is None or num_heads in {None, 0}:
            return None
        return int(hidden_size) // int(num_heads)

    @staticmethod
    def _resolve_attn_implementation(
        requested: Optional[str],
        *,
        strict: bool = False,
        max_head_dim: Optional[int] = None,
    ) -> str:
        normalized = _QWen3_VL_Interface._normalize_attn_implementation_name(requested)
        if normalized == "auto":
            normalized = "flash_attention_4" if (
                torch.cuda.is_available() and is_flash_attn_4_available()
            ) else "flash_attention_2"

        if normalized == "flex_attention":
            return normalized

        flash2_available = torch.cuda.is_available() and is_flash_attn_2_available()
        flash4_available = torch.cuda.is_available() and is_flash_attn_4_available()

        if normalized == "flash_attention_2":
            if not torch.cuda.is_available():
                if strict:
                    raise RuntimeError(
                        "Qwen FlashAttention was requested with strict attention enabled, but CUDA is unavailable."
                    )
                _QWen3_VL_Interface._safe_log(
                    "warning", "FlashAttention requested but CUDA is unavailable; falling back to sdpa"
                )
                return "sdpa"
            if not is_flash_attn_2_available():
                if flash4_available and not strict:
                    _QWen3_VL_Interface._safe_log(
                        "warning",
                        "FlashAttention 2 requested but unavailable; using `flash_attention_4` instead",
                    )
                    return "flash_attention_4"
                if strict:
                    raise RuntimeError(
                        "Qwen FlashAttention was requested with strict attention enabled, "
                        "but `flash-attn` is unavailable."
                    )
                _QWen3_VL_Interface._safe_log(
                    "warning", "FlashAttention requested but `flash-attn` is unavailable; falling back to sdpa"
                )
                return "sdpa"
            return "flash_attention_2"

        if normalized == "flash_attention_4":
            if not torch.cuda.is_available():
                if strict:
                    raise RuntimeError(
                        "Qwen FlashAttention 4 was requested with strict attention enabled, but CUDA is unavailable."
                    )
                _QWen3_VL_Interface._safe_log(
                    "warning", "FlashAttention 4 requested but CUDA is unavailable; falling back to sdpa"
                )
                return "sdpa"
            if not is_flash_attn_4_available():
                if is_flash_attn_2_available() and not strict:
                    _QWen3_VL_Interface._safe_log(
                        "warning",
                        "FlashAttention 4 requested but unavailable; using `flash_attention_2` instead",
                    )
                    return "flash_attention_2"
                if strict:
                    raise RuntimeError(
                        "Qwen FlashAttention 4 was requested with strict attention enabled, "
                        "but `flash-attn-4` is unavailable."
                    )
                _QWen3_VL_Interface._safe_log(
                    "warning", "FlashAttention 4 requested but `flash-attn-4` is unavailable; falling back to sdpa"
                )
                return "sdpa"
            return "flash_attention_4"

        return normalized

    @staticmethod
    def _enable_flex_attention_support_if_needed(attn_implementation: str) -> None:
        if attn_implementation != "flex_attention":
            return

        patched_classes = []
        for class_name in (
            "Qwen3VLPreTrainedModel",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLModel",
            "Qwen3VLTextModel",
            "Qwen3VLVisionModel",
        ):
            model_class = getattr(qwen3_vl_modeling, class_name, None)
            if model_class is not None and not bool(getattr(model_class, "_supports_flex_attn", False)):
                setattr(model_class, "_supports_flex_attn", True)
                patched_classes.append(class_name)

        if patched_classes:
            _QWen3_VL_Interface._safe_log(
                "warning",
                "Enabled local Qwen3-VL FlexAttention support flags for "
                f"{', '.join(patched_classes)}. Transformers has the flex kernels wired for this "
                "attention module, but does not currently advertise Qwen3-VL flex support.",
            )

    @staticmethod
    def _normalize_name_list(value) -> Optional[list[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            return items or None
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None

    @staticmethod
    def _resolve_lora_layers(model, lora_config) -> Optional[list[int]]:
        explicit_layers = lora_config.get("layers_to_transform", None)
        if explicit_layers is not None:
            if isinstance(explicit_layers, int):
                return [int(explicit_layers)]
            return [int(layer_idx) for layer_idx in explicit_layers]

        train_last_n_layers = lora_config.get("train_last_n_layers", None)
        if train_last_n_layers is None:
            return None

        train_last_n_layers = int(train_last_n_layers)
        if train_last_n_layers <= 0:
            return []

        qwen_core = getattr(model, "model", None)
        language_model = getattr(qwen_core, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is not None:
            num_layers = len(layers)
        else:
            text_config = getattr(model.config, "text_config", None)
            num_layers = getattr(text_config, "num_hidden_layers", None)
        if num_layers is None:
            raise RuntimeError("Unable to resolve Qwen3-VL language layer count for LoRA targeting")

        start_idx = max(int(num_layers) - train_last_n_layers, 0)
        return list(range(start_idx, int(num_layers)))

    def _maybe_apply_lora(self, model, *, compile_qwen_model: bool):
        qwenvl_config = self.config.framework.get("qwenvl", {})
        lora_config = qwenvl_config.get("lora", {})
        if not bool(lora_config.get("enabled", False)):
            return model

        try:
            from peft import LoraConfig, TaskType
        except ImportError as exc:
            raise ImportError(
                "Qwen LoRA is enabled but `peft` is not installed. Install `peft>=0.18.0`."
            ) from exc

        if not hasattr(model, "add_adapter"):
            raise RuntimeError(
                "Current transformers build does not expose `add_adapter()` on Qwen3-VL. "
                "Install a recent Transformers build with PEFT integration."
            )

        target_modules = self._normalize_name_list(
            lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        )
        layers_to_transform = self._resolve_lora_layers(model, lora_config)
        layers_pattern = lora_config.get("layers_pattern", "layers")
        adapter_name = str(lora_config.get("adapter_name", "default"))

        hf_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_config.get("r", 16)),
            lora_alpha=int(lora_config.get("alpha", lora_config.get("lora_alpha", 32))),
            lora_dropout=float(lora_config.get("dropout", lora_config.get("lora_dropout", 0.05))),
            bias=str(lora_config.get("bias", "none")),
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
        )
        model.add_adapter(hf_lora_config, adapter_name=adapter_name)
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)

        if compile_qwen_model:
            trainable_patterns = ("lora_",)
            for name, param in model.named_parameters():
                if any(pattern in name for pattern in trainable_patterns):
                    param.requires_grad_(True)

        lora_trainable_params = sum(
            param.numel()
            for name, param in model.named_parameters()
            if param.requires_grad and "lora_" in name
        )
        if lora_trainable_params <= 0:
            raise RuntimeError("Qwen LoRA was enabled, but no LoRA parameters are trainable.")

        layer_summary = (
            "all layers"
            if layers_to_transform is None
            else f"layers {layers_to_transform[0]}..{layers_to_transform[-1]}"
            if layers_to_transform
            else "no layers"
        )
        self._safe_log(
            "info",
            "Enabled Qwen3-VL LoRA "
            f"(adapter={adapter_name}, r={hf_lora_config.r}, alpha={hf_lora_config.lora_alpha}, "
            f"dropout={hf_lora_config.lora_dropout}, targets={target_modules}, {layer_summary}, "
            f"trainable_lora_params={lora_trainable_params})",
        )
        return model

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-2B-Instruct")
        requested_attn_implementation = qwenvl_config.get("attn_implementation", "flash_attention_2")
        strict_attn_implementation = bool(qwenvl_config.get("strict_attn_implementation", False))
        max_attention_head_dim = qwenvl_config.get("max_attention_head_dim", None)
        if max_attention_head_dim is None:
            max_attention_head_dim = self._resolve_attention_head_dim(model_id)
        else:
            max_attention_head_dim = int(max_attention_head_dim)
        attn_implementation = self._resolve_attn_implementation(
            requested_attn_implementation,
            strict=strict_attn_implementation,
            max_head_dim=max_attention_head_dim,
        )
        self._enable_flex_attention_support_if_needed(attn_implementation)
        compile_qwen_model = bool(config.get("trainer", {}).get("compile_qwen_model", False))
        device_map = qwenvl_config.get("device_map", None if compile_qwen_model else "cuda")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        if device_map is None and torch.cuda.is_available():
            model = model.to("cuda")
        model = self._maybe_apply_lora(model, compile_qwen_model=compile_qwen_model)
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        self.requested_attn_implementation = requested_attn_implementation
        self.attn_implementation = attn_implementation
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self._compile_prepared = False
        self._chat_wrapper_cache: dict[tuple, tuple[str, ...]] = {}

        self._safe_log(
            "info",
            f"Loaded `{model_id}` with attention backend `{attn_implementation}` "
            f"(requested `{requested_attn_implementation}`)",
        )

    def supports_blockwise_attention(self) -> bool:
        text_config = getattr(self.model.config, "text_config", self.model.config)
        layer_types = getattr(text_config, "layer_types", None)
        return not layer_types or all(str(layer_type) == "full_attention" for layer_type in layer_types)

    def prepare_for_compile(self) -> int:
        if self._compile_prepared:
            return 0

        patched = 0
        qwen_core = getattr(self.model, "model", None)
        if qwen_core is not None and hasattr(qwen_core, "get_vision_position_ids"):
            qwen_core.get_vision_position_ids = torch.compiler.disable(qwen_core.get_vision_position_ids)
            patched += 1

        self._compile_prepared = True
        message = f"Prepared Qwen3-VL for torch.compile by forcing {patched} problematic subpaths to eager mode"
        try:
            logger.info(message)
        except RuntimeError:
            print(message)
        return patched

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def forward_features(
        self,
        token_replacement_embeds: Optional[torch.Tensor] = None,
        token_replacement_mask: Optional[torch.Tensor] = None,
        qwen_blockwise_block_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Feature-extraction path: runs only the base transformer (no LM head, no
        intermediate hidden states stored). Returns last_hidden_state directly.
        """
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = False
        kwargs["output_attentions"] = False
        kwargs["return_dict"] = True
        if (
            token_replacement_embeds is not None
            or token_replacement_mask is not None
            or qwen_blockwise_block_ids is not None
        ):
            return self._forward_features_with_custom_inputs(
                token_replacement_embeds=token_replacement_embeds,
                token_replacement_mask=token_replacement_mask,
                qwen_blockwise_block_ids=qwen_blockwise_block_ids,
                **kwargs,
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base_outputs = self.model.model(**kwargs)
        return base_outputs.last_hidden_state

    def _build_blockwise_flex_attention_mask(
        self,
        *,
        block_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        device: torch.device,
    ):
        if self.attn_implementation != "flex_attention":
            raise RuntimeError(
                "Qwen blockwise attention requires `framework.qwenvl.attn_implementation: flex_attention`; "
                f"loaded `{self.attn_implementation}`."
            )
        try:
            from torch.nn.attention.flex_attention import create_block_mask
        except ImportError as exc:
            raise ImportError(
                "Qwen blockwise attention requires PyTorch FlexAttention "
                "(`torch.nn.attention.flex_attention`)."
            ) from exc

        if block_ids.shape != input_ids.shape:
            raise ValueError(
                f"qwen_blockwise_block_ids shape {tuple(block_ids.shape)} does not match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        block_ids = block_ids.to(device=device, dtype=torch.long)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "Qwen blockwise attention expects a 2D padding attention_mask matching input_ids; "
                    f"got {tuple(attention_mask.shape)}."
                )
            valid_mask = attention_mask.to(device=device, dtype=torch.bool)
            block_ids = block_ids.masked_fill(~valid_mask, -1)

        batch_size, seq_len = block_ids.shape

        def mask_mod(batch_idx, _head_idx, query_idx, key_idx):
            query_block = block_ids[batch_idx, query_idx]
            key_block = block_ids[batch_idx, key_idx]
            return (query_block >= 0) & (key_block >= 0) & (key_block <= query_block)

        blockwise_cfg = self.config.framework.get("qwenvl", {}).get("blockwise_attention", {})
        compile_mask = bool(blockwise_cfg.get("compile_mask", True)) if hasattr(blockwise_cfg, "get") else True
        return create_block_mask(
            mask_mod,
            B=int(batch_size),
            H=None,
            Q_LEN=int(seq_len),
            KV_LEN=int(seq_len),
            device=device,
            _compile=compile_mask,
        )

    def _forward_features_with_custom_inputs(
        self,
        *,
        token_replacement_embeds: Optional[torch.Tensor],
        token_replacement_mask: Optional[torch.Tensor],
        qwen_blockwise_block_ids: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if (token_replacement_embeds is None) ^ (token_replacement_mask is None):
            raise ValueError("token_replacement_embeds and token_replacement_mask must be provided together.")

        input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
        kwargs.pop("inputs_embeds", None)

        qwen_model = self.model.model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

            image_mask = None
            video_mask = None
            deepstack_image_embeds = None
            deepstack_video_embeds = None

            if pixel_values is not None:
                image_outputs = qwen_model.get_image_features(
                    pixel_values,
                    image_grid_thw,
                    return_dict=True,
                )
                image_embeds = image_outputs.pooler_output
                deepstack_image_embeds = image_outputs.deepstack_features
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = qwen_model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_outputs = qwen_model.get_video_features(
                    pixel_values_videos,
                    video_grid_thw,
                    return_dict=True,
                )
                video_embeds = video_outputs.pooler_output
                deepstack_video_embeds = video_outputs.deepstack_features
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = qwen_model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    video_features=video_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            visual_pos_masks = None
            deepstack_visual_embeds = None
            if image_mask is not None and video_mask is not None:
                image_mask = image_mask[..., 0]
                video_mask = video_mask[..., 0]
                visual_pos_masks = image_mask | video_mask
                deepstack_visual_embeds = []
                image_mask_joint = image_mask[visual_pos_masks]
                video_mask_joint = video_mask[visual_pos_masks]
                for image_embed, video_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = image_embed.new_zeros(
                        visual_pos_masks.sum(),
                        image_embed.shape[-1],
                    ).to(image_embed.device)
                    embed_joint[image_mask_joint, :] = image_embed
                    embed_joint[video_mask_joint, :] = video_embed
                    deepstack_visual_embeds.append(embed_joint)
            elif image_mask is not None:
                visual_pos_masks = image_mask[..., 0]
                deepstack_visual_embeds = deepstack_image_embeds
            elif video_mask is not None:
                visual_pos_masks = video_mask[..., 0]
                deepstack_visual_embeds = deepstack_video_embeds

            if token_replacement_embeds is not None:
                replacement_mask = token_replacement_mask.to(device=inputs_embeds.device, dtype=torch.bool)
                replacement_embeds = token_replacement_embeds.to(
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                if replacement_mask.shape != input_ids.shape:
                    raise ValueError(
                        f"token_replacement_mask shape {tuple(replacement_mask.shape)} does not match "
                        f"input_ids shape {tuple(input_ids.shape)}"
                    )
                expected_values = int(replacement_mask.sum().item()) * inputs_embeds.shape[-1]
                if replacement_embeds.numel() != expected_values:
                    raise ValueError(
                        "token_replacement_embeds did not match replacement mask: "
                        f"expected {expected_values} values, got {replacement_embeds.numel()}"
                    )
                expanded_mask = replacement_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, replacement_embeds.reshape(-1))

            if position_ids is None:
                position_ids = qwen_model.compute_3d_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )

            if qwen_blockwise_block_ids is not None:
                attention_mask = self._build_blockwise_flex_attention_mask(
                    block_ids=qwen_blockwise_block_ids,
                    attention_mask=attention_mask,
                    input_ids=input_ids,
                    device=inputs_embeds.device,
                )

            base_outputs = qwen_model.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )
        return base_outputs.last_hidden_state

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model.generate(**kwargs)

    def _render_prompt_text(
        self,
        instruction: str,
        prompt_replace_dict=None,
        prompt_template=None,
    ) -> str:
        if prompt_template is None:
            if "CoT_prompt" in self.config.datasets.vla_data:
                prompt = self.config.datasets.vla_data.get("CoT_prompt", "").replace("{instruction}", instruction)
                if prompt_replace_dict is not None:
                    for key, value in prompt_replace_dict.items():
                        prompt = prompt.replace(key, value)
            else:
                prompt = instruction
        else:
            prompt = prompt_template.replace("{instruction}", instruction)
            if prompt_replace_dict is not None:
                for key, value in prompt_replace_dict.items():
                    prompt = prompt.replace(key, value)
        return prompt

    def _image_token_text(self) -> str:
        return str(getattr(self.processor, "image_token", "<|image_pad|>"))

    def _get_chat_wrapper(self, num_images: int, *, add_generation_prompt: bool) -> tuple[str, str]:
        cache_key = (int(num_images), bool(add_generation_prompt))
        cached = self._chat_wrapper_cache.get(cache_key)
        if cached is not None:
            return cached

        sentinel = "__STARVLA_PROMPT_SENTINEL__"
        content = [{"type": "image", "image": f"dummy_{index}"} for index in range(num_images)]
        content.append({"type": "text", "text": sentinel})
        rendered = self.processor.apply_chat_template(
            [[{"role": "user", "content": content}]],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )[0]
        sentinel_index = rendered.find(sentinel)
        if sentinel_index < 0:
            raise ValueError("Failed to derive cached Qwen chat wrapper")
        wrapper = (
            rendered[:sentinel_index],
            rendered[sentinel_index + len(sentinel):],
        )
        self._chat_wrapper_cache[cache_key] = wrapper
        return wrapper

    def _get_interleaved_chat_wrapper(
        self,
        num_images: int,
        *,
        add_generation_prompt: bool,
    ) -> tuple[str, str, str]:
        cache_key = ("interleaved", int(num_images), bool(add_generation_prompt))
        cached = self._chat_wrapper_cache.get(cache_key)
        if cached is not None:
            return cached

        prefix_sentinel = "__STARVLA_PREFIX_SENTINEL__"
        suffix_sentinel = "__STARVLA_SUFFIX_SENTINEL__"
        content = [{"type": "text", "text": prefix_sentinel}]
        content.extend({"type": "image", "image": f"dummy_{index}"} for index in range(num_images))
        content.append({"type": "text", "text": suffix_sentinel})
        rendered = self.processor.apply_chat_template(
            [[{"role": "user", "content": content}]],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )[0]
        prefix_index = rendered.find(prefix_sentinel)
        suffix_index = rendered.find(suffix_sentinel)
        if prefix_index < 0 or suffix_index < 0 or suffix_index < prefix_index:
            raise ValueError("Failed to derive interleaved Qwen chat wrapper")
        wrapper = (
            rendered[:prefix_index],
            rendered[prefix_index + len(prefix_sentinel):suffix_index],
            rendered[suffix_index + len(suffix_sentinel):],
        )
        self._chat_wrapper_cache[cache_key] = wrapper
        return wrapper

    @staticmethod
    def _leading_special_token(text: str) -> Optional[str]:
        if not text.startswith("<|"):
            return None
        end = text.find("|>")
        if end < 0:
            return None
        return text[: end + 2]

    def _split_prompt_for_interleaved_images(
        self,
        prompt: str,
        prompt_replace_dict=None,
    ) -> tuple[str, str, bool]:
        if not prompt_replace_dict:
            return prompt, "", False

        split_markers = []
        for placeholder in ("{state}", "{e_actions}", "{actions}", "{geometry}"):
            value = prompt_replace_dict.get(placeholder)
            if not value:
                continue
            split_markers.append(value)
            leading = self._leading_special_token(value)
            if leading is not None:
                split_markers.append(leading)

        marker_positions = [
            position
            for marker in split_markers
            if (position := prompt.find(marker)) >= 0
        ]
        if not marker_positions:
            return prompt, "", False

        split_position = min(marker_positions)
        return prompt[:split_position], prompt[split_position:], True

    def build_qwenvl_inputs_from_frames_tensor(
        self,
        frames: torch.Tensor,
        instructions,
        prompt_replace_dict=None,
        prompt_template=None,
    ):
        if frames.ndim != 5:
            raise ValueError(f"Expected frames with shape [B, V, C, H, W], got {tuple(frames.shape)}")

        qwen_device = self.model.device
        frames = frames.to(qwen_device, non_blocking=True)
        if frames.dtype != torch.uint8:
            frames = frames.clamp_(0, 255).round_().to(torch.uint8)

        batch_size, num_views, channels, height, width = frames.shape
        image_processor = self.processor.image_processor
        patch_size = int(image_processor.patch_size)
        merge_size = int(image_processor.merge_size)
        temporal_patch_size = int(image_processor.temporal_patch_size)
        factor = patch_size * merge_size

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=image_processor.size["shortest_edge"],
            max_pixels=image_processor.size["longest_edge"],
        )
        if (resized_height, resized_width) != (height, width):
            frames = F.interpolate(
                frames.reshape(batch_size * num_views, channels, height, width).to(dtype=torch.float32),
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_views, channels, resized_height, resized_width)
            frames = frames.clamp_(0, 255).round_().to(torch.uint8)

        flat_images = frames.reshape(batch_size * num_views, channels, resized_height, resized_width).to(dtype=torch.float32)
        if bool(getattr(image_processor, "do_rescale", True)):
            flat_images = flat_images * float(getattr(image_processor, "rescale_factor", 1.0 / 255.0))
        if bool(getattr(image_processor, "do_normalize", True)):
            mean = torch.as_tensor(
                image_processor.image_mean,
                device=flat_images.device,
                dtype=flat_images.dtype,
            ).view(1, channels, 1, 1)
            std = torch.as_tensor(
                image_processor.image_std,
                device=flat_images.device,
                dtype=flat_images.dtype,
            ).view(1, channels, 1, 1)
            flat_images = (flat_images - mean) / std

        patches = flat_images.unsqueeze(1)
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(
                1,
                temporal_patch_size - patches.shape[1] % temporal_patch_size,
                1,
                1,
                1,
            )
            patches = torch.cat([patches, repeats], dim=1)

        image_batch_size, grid_t_raw, channel = patches.shape[:3]
        grid_t = grid_t_raw // temporal_patch_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        patches = patches.view(
            image_batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixel_values = patches.reshape(
            image_batch_size * grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        image_grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * image_batch_size,
            dtype=torch.long,
            device=flat_images.device,
        )

        num_image_tokens = int((grid_t * grid_h * grid_w) // (merge_size ** 2))
        prompts = [
            self._render_prompt_text(
                instruction=instruction,
                prompt_replace_dict=prompt_replace_dict,
                prompt_template=prompt_template,
            )
            for instruction in instructions
        ]
        image_token = self._image_token_text()
        image_token_run = image_token * num_image_tokens
        rendered_text = []
        for prompt in prompts:
            prompt_prefix, prompt_suffix, use_interleaved = self._split_prompt_for_interleaved_images(
                prompt,
                prompt_replace_dict=prompt_replace_dict,
            )
            if use_interleaved:
                prefix, image_middle, suffix = self._get_interleaved_chat_wrapper(
                    num_views,
                    add_generation_prompt=True,
                )
                rendered = f"{prefix}{prompt_prefix}{image_middle}{prompt_suffix}{suffix}"
            else:
                prefix, suffix = self._get_chat_wrapper(num_views, add_generation_prompt=True)
                rendered = f"{prefix}{prompt}{suffix}"
            rendered_text.append(rendered.replace(image_token, image_token_run))

        tokenizer_kwargs = {
            "padding": True,
            "return_token_type_ids": False,
            "return_tensors": "pt",
        }
        if self.processor.tokenizer.bos_token is not None and rendered_text and rendered_text[0].startswith(self.processor.tokenizer.bos_token):
            tokenizer_kwargs["add_special_tokens"] = False

        text_inputs = self.processor.tokenizer(rendered_text, **tokenizer_kwargs)
        if hasattr(self.processor, "create_mm_token_type_ids"):
            mm_token_type_ids = self.processor.create_mm_token_type_ids(text_inputs["input_ids"].tolist())
            text_inputs["mm_token_type_ids"] = torch.tensor(
                mm_token_type_ids,
                dtype=text_inputs["input_ids"].dtype,
            )

        batch_inputs = {
            **text_inputs,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        return {
            key: value.to(qwen_device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch_inputs.items()
        }

    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        solutions=None,
        prompt_replace_dict=None,
        prompt_template=None,
        **kwargs,
    ):
        if solutions is None:
            prompts = [
                self._render_prompt_text(
                    instruction=instruction,
                    prompt_replace_dict=prompt_replace_dict,
                    prompt_template=prompt_template,
                )
                for instruction in instructions
            ]
            rendered_text = []
            for imgs, prompt in zip(images, prompts):
                prompt_prefix, prompt_suffix, use_interleaved = self._split_prompt_for_interleaved_images(
                    prompt,
                    prompt_replace_dict=prompt_replace_dict,
                )
                if use_interleaved:
                    prefix, image_middle, suffix = self._get_interleaved_chat_wrapper(
                        len(imgs),
                        add_generation_prompt=True,
                    )
                    rendered_text.append(f"{prefix}{prompt_prefix}{image_middle}{prompt_suffix}{suffix}")
                else:
                    prefix, suffix = self._get_chat_wrapper(
                        len(imgs),
                        add_generation_prompt=True,
                    )
                    rendered_text.append(f"{prefix}{prompt}{suffix}")

            batch_inputs = self.processor(
                text=rendered_text,
                images=images,
                text_kwargs={
                    "padding": True,
                    "return_tensors": "pt",
                },
            )

            if self.config.get("trainer", {}).get("channels_last", False):
                if "pixel_values" in batch_inputs and isinstance(batch_inputs["pixel_values"], torch.Tensor):
                    if batch_inputs["pixel_values"].dim() == 4:
                        batch_inputs["pixel_values"] = batch_inputs["pixel_values"].contiguous(
                            memory_format=torch.channels_last
                        )
                if "pixel_values_videos" in batch_inputs and isinstance(batch_inputs["pixel_values_videos"], torch.Tensor):
                    if batch_inputs["pixel_values_videos"].dim() == 5:
                        batch_inputs["pixel_values_videos"] = batch_inputs["pixel_values_videos"].contiguous(
                            memory_format=torch.channels_last_3d
                        )

            return batch_inputs.to(self.model.device)

        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            prompt = self._render_prompt_text(
                instruction=instruction,
                prompt_replace_dict=prompt_replace_dict,
                prompt_template=prompt_template,
            )
            prompt_prefix, prompt_suffix, use_interleaved = self._split_prompt_for_interleaved_images(
                prompt,
                prompt_replace_dict=prompt_replace_dict,
            )
            if use_interleaved:
                content = []
                if prompt_prefix:
                    content.append({"type": "text", "text": prompt_prefix})
                content.extend({"type": "image", "image": img} for img in imgs)
                if prompt_suffix:
                    content.append({"type": "text", "text": prompt_suffix})
            else:
                content = [{"type": "image", "image": img} for img in imgs]
                content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            solution = solutions[len(messages)]
            msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        labels = batch_inputs["input_ids"].clone()
        for index in range(labels.size(0)):
            seq = labels[index]
            mask_seq = (seq >= _ACTION_TOKEN_MIN) & (seq <= _ACTION_TOKEN_MAX)
            nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
            if nonzero_indices.numel() > 0:
                first_action_index = nonzero_indices[0].item()
                seq[:first_action_index] = IGNORE_INDEX
            else:
                seq[:] = IGNORE_INDEX
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        batch_inputs["labels"] = labels

        if self.config.get("trainer", {}).get("channels_last", False):
            if "pixel_values" in batch_inputs and isinstance(batch_inputs["pixel_values"], torch.Tensor):
                if batch_inputs["pixel_values"].dim() == 4:
                    batch_inputs["pixel_values"] = batch_inputs["pixel_values"].contiguous(
                        memory_format=torch.channels_last
                    )
            if "pixel_values_videos" in batch_inputs and isinstance(batch_inputs["pixel_values_videos"], torch.Tensor):
                if batch_inputs["pixel_values_videos"].dim() == 5:
                    batch_inputs["pixel_values_videos"] = batch_inputs["pixel_values_videos"].contiguous(
                        memory_format=torch.channels_last_3d
                    )

        return batch_inputs.to(self.model.device)
