# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import os
import torch
import torch.nn.functional as F
from typing import Optional
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.utils import is_flash_attn_2_available

from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
_ACTION_TOKEN_MIN = 248320
_ACTION_TOKEN_MAX = 260000

import torch.nn as nn


def _enable_partial_fast_linear_attention_logging() -> None:
    has_fla_kernels = all(
        (
            modeling_qwen3_5.chunk_gated_delta_rule,
            modeling_qwen3_5.fused_recurrent_gated_delta_rule,
            modeling_qwen3_5.FusedRMSNormGated,
        )
    )
    has_full_fast_path = has_fla_kernels and all(
        (
            modeling_qwen3_5.causal_conv1d_fn,
            modeling_qwen3_5.causal_conv1d_update,
        )
    )
    if has_fla_kernels and not has_full_fast_path:
        # Upstream only uses `is_fast_path_available` for warning text. Keep the
        # fused delta-rule kernels enabled even when conv1d falls back to torch.
        modeling_qwen3_5.is_fast_path_available = True


class _QWen3_5_Interface(nn.Module):
    """Wrapper around Qwen3.5 multimodal models."""

    @staticmethod
    def _safe_log(level: str, message: str) -> None:
        try:
            getattr(logger, level)(message)
        except RuntimeError:
            print(message)

    @staticmethod
    def _resolve_attn_implementation(requested: Optional[str]) -> str:
        normalized = (requested or "flash_attention_2").strip().lower()
        alias_map = {
            "flash": "flash_attention_2",
            "flash2": "flash_attention_2",
            "flash_attn": "flash_attention_2",
            "flash_attn_2": "flash_attention_2",
            "flash-attn": "flash_attention_2",
            "flash-attn-2": "flash_attention_2",
        }
        normalized = alias_map.get(normalized, normalized)

        if normalized == "auto":
            normalized = "flash_attention_2"

        flash_available = torch.cuda.is_available() and is_flash_attn_2_available()
        prefer_flash = os.getenv("STARVLA_DISABLE_FLASH_ATTN_PROMOTION", "0") != "1"

        if normalized == "sdpa" and flash_available and prefer_flash:
            _QWen3_5_Interface._safe_log(
                "info", "Promoting requested `sdpa` attention to `flash_attention_2` for Qwen3.5"
            )
            normalized = "flash_attention_2"

        if normalized == "flash_attention_2":
            if not torch.cuda.is_available():
                _QWen3_5_Interface._safe_log(
                    "warning", "FlashAttention requested but CUDA is unavailable; falling back to sdpa"
                )
                return "sdpa"
            if not is_flash_attn_2_available():
                _QWen3_5_Interface._safe_log(
                    "warning", "FlashAttention requested but `flash-attn` is unavailable; falling back to sdpa"
                )
                return "sdpa"
            return "flash_attention_2"

        return normalized

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-2B")
        requested_attn_implementation = qwenvl_config.get("attn_implementation", "flash_attention_2")
        attn_implementation = self._resolve_attn_implementation(requested_attn_implementation)
        compile_qwen_model = bool(config.get("trainer", {}).get("compile_qwen_model", False))
        device_map = qwenvl_config.get("device_map", None if compile_qwen_model else "cuda")
        enable_fast_linear_attention = bool(qwenvl_config.get("enable_fast_linear_attention", False))

        if not enable_fast_linear_attention:
            # Force the safe torch path. On this machine the custom causal-conv1d build
            # does not provide a kernel image for the target GPU architecture.
            modeling_qwen3_5.causal_conv1d_fn = None
            modeling_qwen3_5.causal_conv1d_update = None
            modeling_qwen3_5.chunk_gated_delta_rule = None
            modeling_qwen3_5.fused_recurrent_gated_delta_rule = None
            modeling_qwen3_5.FusedRMSNormGated = None
        else:
            _enable_partial_fast_linear_attention_logging()

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        if device_map is None and torch.cuda.is_available():
            model = model.to("cuda")
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        logger.info(
            "Loaded `%s` with attention backend `%s` (requested `%s`)",
            model_id,
            attn_implementation,
            requested_attn_implementation,
        )
        if enable_fast_linear_attention:
            has_fast_delta = all(
                (
                    modeling_qwen3_5.chunk_gated_delta_rule,
                    modeling_qwen3_5.fused_recurrent_gated_delta_rule,
                    modeling_qwen3_5.FusedRMSNormGated,
                )
            )
            if has_fast_delta and modeling_qwen3_5.causal_conv1d_fn is None:
                self._safe_log(
                    "info",
                    "Qwen3.5 fast linear attention is enabled with fused delta kernels; causal-conv1d is unavailable so conv1d stays on the torch fallback path",
                )

        self.model = model
        self.processor = processor
        self.config = config
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self._compile_prepared = False
        self._chat_wrapper_cache: dict[tuple[int, bool], tuple[str, str]] = {}

    def prepare_for_compile(self) -> int:
        """
        Make the Qwen3.5 model compile-friendly enough for torch.compile.

        Two transformer internals currently cause pathological graph breaks/recompiles:
        - `get_vision_position_ids` calls `.item()`
        - each decoder layer's `linear_attn.forward` recompiles on `layer_idx`

        Running those pieces eagerly allows the rest of the model to compile.
        """
        if self._compile_prepared:
            return 0

        patched = 0
        qwen_core = getattr(self.model, "model", None)
        if qwen_core is not None and hasattr(qwen_core, "get_vision_position_ids"):
            qwen_core.get_vision_position_ids = torch.compiler.disable(qwen_core.get_vision_position_ids)
            patched += 1

        language_model = getattr(qwen_core, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is not None:
            for layer in layers:
                if hasattr(layer, "linear_attn"):
                    layer.linear_attn.forward = torch.compiler.disable(layer.linear_attn.forward)
                    patched += 1

        self._compile_prepared = True
        message = (
            f"Prepared Qwen3.5 for torch.compile by forcing {patched} problematic subpaths to eager mode"
        )
        try:
            logger.info(message)
        except RuntimeError:
            print(message)
        return patched

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def forward_features(self, **kwargs) -> torch.Tensor:
        """
        Feature-extraction path: runs only the base transformer (no LM head, no
        intermediate hidden states stored).  Returns last_hidden_state directly.

        Saves compute (skips vocab-size matmul) and memory (no per-layer state tuple).
        """
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = False
        kwargs["output_attentions"] = False
        kwargs["return_dict"] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base_outputs = self.model.model(**kwargs)
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
                    for k, v in prompt_replace_dict.items():
                        prompt = prompt.replace(k, v)
            else:
                prompt = instruction
        else:
            prompt = prompt_template.replace("{instruction}", instruction)
            if prompt_replace_dict is not None:
                for k, v in prompt_replace_dict.items():
                    prompt = prompt.replace(k, v)
        return prompt

    def _get_chat_wrapper(self, num_images: int, *, add_generation_prompt: bool) -> tuple[str, str]:
        cache_key = (int(num_images), bool(add_generation_prompt))
        cached = self._chat_wrapper_cache.get(cache_key)
        if cached is not None:
            return cached

        sentinel = "__STARVLA_PROMPT_SENTINEL__"
        content = [{"type": "image", "image": f"dummy_{i}"} for i in range(num_images)]
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

        B, V, C, H, W = frames.shape
        image_processor = self.processor.image_processor
        patch_size = int(image_processor.patch_size)
        merge_size = int(image_processor.merge_size)
        temporal_patch_size = int(image_processor.temporal_patch_size)
        factor = patch_size * merge_size

        resized_height, resized_width = smart_resize(
            H,
            W,
            factor=factor,
            min_pixels=image_processor.size["shortest_edge"],
            max_pixels=image_processor.size["longest_edge"],
        )
        if (resized_height, resized_width) != (H, W):
            frames = F.interpolate(
                frames.reshape(B * V, C, H, W).to(dtype=torch.float32),
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, V, C, resized_height, resized_width)
            frames = frames.clamp_(0, 255).round_().to(torch.uint8)

        flat_images = frames.reshape(B * V, C, resized_height, resized_width).to(dtype=torch.float32)
        if bool(getattr(image_processor, "do_rescale", True)):
            flat_images = flat_images * float(getattr(image_processor, "rescale_factor", 1.0 / 255.0))
        if bool(getattr(image_processor, "do_normalize", True)):
            mean = torch.as_tensor(image_processor.image_mean, device=flat_images.device, dtype=flat_images.dtype).view(1, C, 1, 1)
            std = torch.as_tensor(image_processor.image_std, device=flat_images.device, dtype=flat_images.dtype).view(1, C, 1, 1)
            flat_images = (flat_images - mean) / std

        patches = flat_images.unsqueeze(1)
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(1, temporal_patch_size - patches.shape[1] % temporal_patch_size, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=1)

        batch_size, grid_t_raw, channel = patches.shape[:3]
        grid_t = grid_t_raw // temporal_patch_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        patches = patches.view(
            batch_size,
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
            batch_size * grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        image_grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * batch_size,
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
        prefix, suffix = self._get_chat_wrapper(V, add_generation_prompt=True)
        rendered_text = [
            f"{prefix}{prompt}{suffix}".replace(self.processor.image_token, self.processor.image_token * num_image_tokens)
            for prompt in prompts
        ]

        tokenizer_kwargs = {
            "padding": True,
            "return_token_type_ids": False,
            "return_tensors": "pt",
        }
        if self.processor.tokenizer.bos_token is not None and rendered_text and rendered_text[0].startswith(self.processor.tokenizer.bos_token):
            tokenizer_kwargs["add_special_tokens"] = False

        text_inputs = self.processor.tokenizer(rendered_text, **tokenizer_kwargs)
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
            content = [{"type": "image", "image": img} for img in imgs]

            if prompt_template is None:
                if "CoT_prompt" in self.config.datasets.vla_data:
                    prompt = self.config.datasets.vla_data.get("CoT_prompt", "").replace("{instruction}", instruction)
                    if prompt_replace_dict is not None:
                        for k, v in prompt_replace_dict.items():
                            prompt = prompt.replace(k, v)
                else:
                    prompt = instruction
            else:
                prompt = prompt_template.replace("{instruction}", instruction)
                if prompt_replace_dict is not None:
                    for k, v in prompt_replace_dict.items():
                        prompt = prompt.replace(k, v)

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
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

        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
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
