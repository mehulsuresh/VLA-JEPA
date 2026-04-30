# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from contextlib import nullcontext
from functools import lru_cache
import importlib
from collections import OrderedDict
from pathlib import Path
import time
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

try:
    import decord

    DECORD_AVAILABLE = True
except ImportError:
    decord = None
    DECORD_AVAILABLE = False

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    torchcodec = None
    TORCHCODEC_AVAILABLE = False


def _clean_vjepa_backbone_key(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


@lru_cache(maxsize=64)
def _get_rank_cpu_video_reader(video_path: str, num_threads: int):
    if not DECORD_AVAILABLE:
        raise ImportError("decord is not available.")
    return decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=num_threads)


def _get_rank_gpu_video_reader(video_path: str, device_index: int, num_threads: int):
    if not DECORD_AVAILABLE:
        raise ImportError("decord is not available.")
    return decord.VideoReader(video_path, ctx=decord.gpu(device_index), num_threads=num_threads)


def _get_rank_torchcodec_video_reader(
    video_path: str,
    device: str,
    num_threads: int,
    *,
    dimension_order: str = "NHWC",
    seek_mode: str = "exact",
):
    if not TORCHCODEC_AVAILABLE:
        raise ImportError("torchcodec is not available.")
    return torchcodec.decoders.VideoDecoder(
        video_path,
        device=device,
        dimension_order=dimension_order,
        num_ffmpeg_threads=max(1, int(num_threads)),
        seek_mode=seek_mode,
    )


@lru_cache(maxsize=256)
def _get_rank_video_frame_timestamps(
    video_path: str,
    device_index: int,
    num_threads: int,
) -> np.ndarray:
    if device_index >= 0:
        reader = _get_rank_gpu_video_reader(video_path, device_index, num_threads)
    else:
        reader = _get_rank_cpu_video_reader(video_path, num_threads)
    return reader.get_frame_timestamp(range(len(reader)))

@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.vj_encoder, self.vj_processor = self._load_vjepa_backbone(self.config.framework.vj2_model)
        self.vj_freeze_encoder = self.config.framework.vj2_model.get("freeze_encoder", True)
        if self.vj_freeze_encoder:
            self.vj_encoder.requires_grad_(False)
            self.vj_encoder.eval()
        if bool(self.config.get("trainer", {}).get("channels_last", False)) and self.config.framework.vj2_model.get("source", "hf") == "torchhub":
            self.vj_encoder = self.vj_encoder.to(memory_format=torch.channels_last_3d)
        self.vj_num_video_views = self.config.framework.vj2_model.get("num_video_views", 2)

        tubelet_size = self._get_vjepa_attr("tubelet_size")
        image_size = self._get_vjepa_attr("image_size")
        hidden_size = self._get_vjepa_attr("hidden_size")

        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames // tubelet_size,
            img_size=((image_size, image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=hidden_size * self.vj_num_video_views,
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
            use_activation_checkpointing=bool(
                self.config.get("trainer", {}).get("enable_gradient_checkpointing", False)
            ),
            use_legacy_rope_bug=bool(
                self.config.framework.vj2_model.get("use_legacy_rope_bug", False)
            ),
        )
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[: self.config.framework.vj2_model.num_frames // tubelet_size - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])

        # Pre-cache token ID tensors (avoids torch.tensor + torch.isin every forward)
        self.register_buffer(
            "_action_token_ids_t",
            torch.tensor(self.action_token_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_embodied_token_id_t",
            torch.tensor([self.embodied_action_token_id], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_img_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_img_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self._qwen_grad_cache: Optional[bool] = None
        self._torchcodec_reader_cache: "OrderedDict[tuple, object]" = OrderedDict()
        self._vj_compile_prepared = False
        repeated_diffusion_steps = 4
        if self.config is not None:
            framework_action_cfg = self.config.get("framework", {}).get("action_model", {})
            trainer_cfg = self.config.get("trainer", {})
            repeated_diffusion_steps = int(
                trainer_cfg.get(
                    "repeated_diffusion_steps",
                    framework_action_cfg.get("repeated_diffusion_steps", 4),
                )
            )
        self._repeated_diffusion_steps = max(repeated_diffusion_steps, 1)

    def prepare_vj_encoder_for_compile(self) -> int:
        """
        Make the frozen V-JEPA encoder more compile-friendly.

        The upstream RoPE attention path builds token-position helpers on the fly.
        Those helper methods tend to be dynamic-shape heavy and a poor target for
        `torch.compile`, so we force just those subpaths back to eager mode while
        leaving the main encoder forward compiled.
        """
        if self._vj_compile_prepared:
            return 0

        patched = 0
        seen_modules: set[int] = set()

        def _disable_bound_method(owner, method_name: str) -> bool:
            method = getattr(owner, method_name, None)
            if method is None or not callable(method):
                return False
            if getattr(method, "_starvla_compile_disabled", False):
                return False
            disabled = torch.compiler.disable(method)
            disabled._starvla_compile_disabled = True
            setattr(owner, method_name, disabled)
            return True

        blocks = getattr(self.vj_encoder, "blocks", None)
        if blocks is not None:
            for blk in blocks:
                attn = getattr(blk, "attn", None)
                if attn is None:
                    continue
                for helper_name in ("_get_frame_pos", "_get_height_pos", "separate_positions"):
                    if _disable_bound_method(attn, helper_name):
                        patched += 1

                attn_module_name = type(attn).__module__
                try:
                    attn_module = importlib.import_module(attn_module_name)
                except Exception:
                    attn_module = None

                if attn_module is None:
                    continue

                module_id = id(attn_module)
                if module_id in seen_modules:
                    continue
                seen_modules.add(module_id)

                rotate_fn = getattr(attn_module, "rotate_queries_or_keys", None)
                if rotate_fn is not None and callable(rotate_fn):
                    if not getattr(rotate_fn, "_starvla_compile_disabled", False):
                        disabled_rotate = torch.compiler.disable(rotate_fn)
                        disabled_rotate._starvla_compile_disabled = True
                        setattr(attn_module, "rotate_queries_or_keys", disabled_rotate)
                        patched += 1

        self._vj_compile_prepared = True
        logger.info(
            "Prepared V-JEPA encoder for torch.compile by forcing %d RoPE helper subpaths to eager mode",
            patched,
        )
        return patched

    def _load_vjepa_backbone(self, vj_cfg):
        source = vj_cfg.get("source", "hf")
        if source == "hf":
            base_encoder = vj_cfg.base_encoder
            encoder = AutoModel.from_pretrained(base_encoder)
            processor = AutoVideoProcessor.from_pretrained(base_encoder)
            return encoder, processor

        if source == "torchhub":
            repo_or_dir = vj_cfg.get("hub_repo_or_dir", "facebookresearch/vjepa2")
            model_name = vj_cfg.get("hub_model_name", "vjepa2_1_vit_large_384")
            preprocessor_name = vj_cfg.get("hub_preprocessor_name", "vjepa2_preprocessor")
            crop_size = vj_cfg.get("crop_size", 384)
            pretrained = vj_cfg.get("pretrained", True)
            checkpoint_url = vj_cfg.get("hub_checkpoint_url", None)
            checkpoint_path = vj_cfg.get("hub_checkpoint_path", None)
            checkpoint_key = vj_cfg.get("hub_checkpoint_key", "ema_encoder")
            repo_or_dir = str(repo_or_dir)
            expanded_repo_path = Path(repo_or_dir).expanduser()
            hub_source = "local" if expanded_repo_path.exists() else "github"
            if hub_source == "local":
                repo_or_dir = str(expanded_repo_path)

            def _hub_load(entrypoint: str, **load_kwargs):
                return torch.hub.load(
                    repo_or_dir,
                    entrypoint,
                    source=hub_source,
                    **load_kwargs,
                )

            manual_checkpoint = checkpoint_url is not None or checkpoint_path is not None

            if hub_source == "github" and torch.distributed.is_available() and torch.distributed.is_initialized():
                cache_sentinel = Path(torch.hub.get_dir()) / f".{repo_or_dir.replace('/', '_')}_ready"
                if torch.distributed.get_rank() == 0:
                    _hub_load(model_name, pretrained=False if manual_checkpoint else pretrained)
                    _hub_load(
                        preprocessor_name,
                        pretrained=pretrained,
                        crop_size=crop_size,
                    )
                    cache_sentinel.write_text("ready\n")
                else:
                    deadline = time.time() + 600
                    while not cache_sentinel.exists():
                        if time.time() > deadline:
                            raise TimeoutError(
                                f"Timed out waiting for torch.hub cache warmup for `{repo_or_dir}`"
                            )
                        time.sleep(1.0)

            loaded = _hub_load(
                model_name,
                pretrained=False if manual_checkpoint else pretrained,
            )
            encoder = loaded[0] if isinstance(loaded, tuple) else loaded
            processor = _hub_load(
                preprocessor_name,
                pretrained=pretrained,
                crop_size=crop_size,
            )

            if manual_checkpoint:
                if checkpoint_path is not None:
                    checkpoint_file = Path(str(checkpoint_path)).expanduser()
                    state_dict = torch.load(checkpoint_file, map_location="cpu")
                else:
                    state_dict = torch.hub.load_state_dict_from_url(str(checkpoint_url), map_location="cpu")

                encoder_state_dict = _clean_vjepa_backbone_key(state_dict[checkpoint_key])
                encoder.load_state_dict(encoder_state_dict, strict=True)
            return encoder, processor

        raise ValueError(f"Unsupported V-JEPA source: {source}")

    def _get_vjepa_attr(self, key: str):
        if hasattr(self.vj_encoder, "config") and hasattr(self.vj_encoder.config, key):
            return getattr(self.vj_encoder.config, key)
        if key == "image_size" and hasattr(self.vj_encoder, "img_height"):
            return getattr(self.vj_encoder, "img_height")
        if key == "hidden_size" and hasattr(self.vj_encoder, "embed_dim"):
            return getattr(self.vj_encoder, "embed_dim")
        if hasattr(self.vj_encoder, key):
            return getattr(self.vj_encoder, key)
        raise AttributeError(f"Unable to resolve `{key}` from V-JEPA encoder")

    def _get_qwen_device(self) -> torch.device:
        return next(self.qwen_vl_interface.model.parameters()).device

    def _qwen_requires_grad(self) -> bool:
        if self._qwen_grad_cache is None:
            self._qwen_grad_cache = any(param.requires_grad for param in self.qwen_vl_interface.model.parameters())
        return self._qwen_grad_cache

    def refresh_runtime_caches(self) -> None:
        self._qwen_grad_cache = any(param.requires_grad for param in self.qwen_vl_interface.model.parameters())

    def _move_qwen_inputs(self, qwen_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        qwen_device = self._get_qwen_device()
        moved = {}
        for key, value in qwen_inputs.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(qwen_device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _resolve_qwen_prompt_args(
        self,
        *,
        has_actions: bool,
        prompt_replace_dict: Optional[dict[str, str]] = None,
        prompt_template: Optional[str] = None,
    ) -> tuple[dict[str, str], str]:
        if prompt_replace_dict is None:
            prompt_replace_dict = {"{actions}": self.replace_prompt}
            if has_actions:
                prompt_replace_dict["{e_actions}"] = self.embodied_replace_prompt

        if prompt_template is None:
            prompt_template = (
                self.config.datasets.vla_data.get("CoT_prompt", "")
                if has_actions
                else self.config.datasets.video_data.get("CoT_prompt", "")
            )

        return prompt_replace_dict, prompt_template

    def _validate_qwen_action_prompt_tokens(
        self,
        qwen_inputs: dict[str, torch.Tensor],
        *,
        has_actions: bool,
        stage: str,
    ) -> None:
        input_ids = qwen_inputs["input_ids"]
        action_token_ids = self._action_token_ids_t.to(input_ids.device)
        embodied_token_id = self._embodied_token_id_t.to(input_ids.device)
        batch_size = int(input_ids.shape[0]) if input_ids.ndim > 1 else 1

        expected_action_count_per_example = (
            self.config.framework.vj2_model.num_frames // self._get_vjepa_attr("tubelet_size") - 1
        ) * self.config.framework.vj2_model.num_action_tokens_per_timestep
        expected_action_count = expected_action_count_per_example * batch_size
        action_count = int(torch.isin(input_ids, action_token_ids).sum().item())
        if action_count != expected_action_count:
            raise RuntimeError(
                f"{stage}: expected {expected_action_count} total action prompt tokens in Qwen "
                f"inputs ({expected_action_count_per_example} per sample across batch_size="
                f"{batch_size}), found {action_count}. This usually means prompt placeholders "
                "were not expanded before building Qwen inputs."
            )

        if has_actions:
            expected_embodied_count_per_example = (
                self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction
            )
            expected_embodied_count = expected_embodied_count_per_example * batch_size
            embodied_count = int(torch.isin(input_ids, embodied_token_id).sum().item())
            if embodied_count != expected_embodied_count:
                raise RuntimeError(
                    f"{stage}: expected {expected_embodied_count} total embodied-action prompt "
                    f"tokens in Qwen inputs ({expected_embodied_count_per_example} per sample "
                    f"across batch_size={batch_size}), found {embodied_count}. This usually means "
                    "prompt placeholders were not expanded before building Qwen inputs."
                )

    def _build_qwen_inputs_from_examples(self, examples: List[dict]) -> dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        has_actions = "action" in examples[0]
        prompt_replace_dict, prompt_template = self._resolve_qwen_prompt_args(has_actions=has_actions)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict=prompt_replace_dict,
            prompt_template=prompt_template,
        )
        self._validate_qwen_action_prompt_tokens(
            qwen_inputs,
            has_actions=has_actions,
            stage="_build_qwen_inputs_from_examples",
        )
        return qwen_inputs

    def _build_qwen_inputs_from_video_tensor(
        self,
        batch_videos: np.ndarray | torch.Tensor,
        instructions: List[str],
        has_actions: bool,
        prompt_replace_dict: Optional[dict[str, str]] = None,
        prompt_template: Optional[str] = None,
        *,
        return_timing: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, float]]:
        qwen_device = self._get_qwen_device()
        if isinstance(batch_videos, np.ndarray):
            frames = torch.from_numpy(np.ascontiguousarray(batch_videos[:, :, -1]))
        else:
            frames = batch_videos[:, :, -1]

        to_cuda_start = time.perf_counter()
        frames = frames.to(qwen_device, dtype=torch.float32, non_blocking=True)
        video_tensor_to_cuda_time = time.perf_counter() - to_cuda_start
        build_start = time.perf_counter()
        B, V, C, H, W = frames.shape
        target_size = int(self.config.datasets.vla_data.get("resolution_size", H))
        if H != target_size or W != target_size:
            frames = F.interpolate(
                frames.reshape(B * V, C, H, W),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, V, C, target_size, target_size)
        frames = frames.clamp_(0, 255).round_().to(torch.uint8)

        prompt_replace_dict, prompt_template = self._resolve_qwen_prompt_args(
            has_actions=has_actions,
            prompt_replace_dict=prompt_replace_dict,
            prompt_template=prompt_template,
        )

        build_from_tensor = getattr(self.qwen_vl_interface, "build_qwenvl_inputs_from_frames_tensor", None)
        if callable(build_from_tensor):
            qwen_inputs = build_from_tensor(
                frames=frames,
                instructions=instructions,
                prompt_replace_dict=prompt_replace_dict,
                prompt_template=prompt_template,
            )
            self._validate_qwen_action_prompt_tokens(
                qwen_inputs,
                has_actions=has_actions,
                stage="_build_qwen_inputs_from_video_tensor[tensor_fast_path]",
            )
            if return_timing:
                return qwen_inputs, {
                    "video_tensor_to_cuda_time": video_tensor_to_cuda_time,
                    "qwen_input_build_time": time.perf_counter() - build_start,
                }
            return qwen_inputs

        image_batches = [
            [frames[b, v].permute(1, 2, 0).contiguous() for v in range(V)]
            for b in range(B)
        ]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=image_batches,
            instructions=instructions,
            prompt_replace_dict=prompt_replace_dict,
            prompt_template=prompt_template,
        )
        self._validate_qwen_action_prompt_tokens(
            qwen_inputs,
            has_actions=has_actions,
            stage="_build_qwen_inputs_from_video_tensor[image_fallback]",
        )
        if return_timing:
            return qwen_inputs, {
                "video_tensor_to_cuda_time": video_tensor_to_cuda_time,
                "qwen_input_build_time": time.perf_counter() - build_start,
            }
        return qwen_inputs

    def _split_compact_videos(
        self,
        batch_compact_videos: np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray | torch.Tensor, Optional[np.ndarray | torch.Tensor]]:
        shift = int(self.config.datasets.vla_data.get("video_target_shift_steps", 0))
        if shift <= 0:
            return batch_compact_videos, None

        total_frames = batch_compact_videos.shape[2]
        context_horizon = total_frames - shift
        if context_horizon <= 0:
            raise ValueError(
                f"Compact video clip has invalid temporal length {total_frames} for shift {shift}"
            )
        return batch_compact_videos[:, :, :context_horizon], batch_compact_videos[:, :, shift:]

    def _gpu_decode_debug_enabled(self) -> bool:
        return bool(self.config.datasets.vla_data.get("debug_gpu_decode_timing", False))

    def _gpu_decode_debug_rank(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _log_gpu_decode_debug(self, message: str) -> None:
        if self._gpu_decode_debug_enabled():
            logger.info(f"[gpu-decode][rank {self._gpu_decode_debug_rank()}] {message}")

    def _gpu_video_decode_backend(self) -> str:
        return str(self.config.datasets.vla_data.get("gpu_video_decode_backend", "decord")).lower()

    def _torchcodec_cuda_backend(self) -> str:
        return str(self.config.datasets.vla_data.get("gpu_video_decode_torchcodec_cuda_backend", "beta")).lower()

    def _torchcodec_seek_mode(self) -> str:
        return str(self.config.datasets.vla_data.get("gpu_video_decode_torchcodec_seek_mode", "exact")).lower()

    def _torchcodec_dimension_order(self) -> str:
        return str(self.config.datasets.vla_data.get("gpu_video_decode_torchcodec_dimension_order", "NCHW")).upper()

    def _torchcodec_fetch_mode(self) -> str:
        return str(self.config.datasets.vla_data.get("gpu_video_decode_torchcodec_fetch_mode", "auto")).lower()

    def _torchcodec_nvdec_cache_capacity(self) -> Optional[int]:
        value = self.config.datasets.vla_data.get("gpu_video_decode_torchcodec_nvdec_cache_capacity", None)
        if value is None:
            return None
        return int(value)

    def _torchcodec_reader_cache_size(self) -> int:
        return max(0, int(self.config.datasets.vla_data.get("gpu_video_decode_torchcodec_reader_cache_size", 32)))

    def _get_or_create_torchcodec_reader(
        self,
        *,
        video_path: str,
        device: str,
        decode_threads: int,
        dimension_order: str,
        seek_mode: str,
        cuda_backend: str,
        debug_enabled: bool,
        spec_index: int,
    ):
        cache_size = self._torchcodec_reader_cache_size()
        cache_key = (
            video_path,
            device,
            int(decode_threads),
            dimension_order,
            seek_mode,
            cuda_backend,
        )
        if cache_size > 0:
            cached = self._torchcodec_reader_cache.get(cache_key)
            if cached is not None:
                self._torchcodec_reader_cache.move_to_end(cache_key)
                if debug_enabled:
                    self._log_gpu_decode_debug(
                        f"spec {spec_index} torchcodec reader cache hit size={len(self._torchcodec_reader_cache)}"
                    )
                return cached

        create_start = time.perf_counter()
        if debug_enabled:
            self._log_gpu_decode_debug(
                f"spec {spec_index} creating torchcodec reader device={device} "
                f"cuda_backend={cuda_backend} seek_mode={seek_mode}"
            )

        if device.startswith("cuda"):
            from torchcodec.decoders import set_cuda_backend

            with set_cuda_backend(cuda_backend):
                reader = _get_rank_torchcodec_video_reader(
                    video_path,
                    device,
                    decode_threads,
                    dimension_order=dimension_order,
                    seek_mode=seek_mode,
                )
        else:
            reader = _get_rank_torchcodec_video_reader(
                video_path,
                device,
                decode_threads,
                dimension_order=dimension_order,
                seek_mode=seek_mode,
            )

        if debug_enabled:
            self._log_gpu_decode_debug(
                f"spec {spec_index} torchcodec reader created elapsed={time.perf_counter() - create_start:.3f}s"
            )

        if cache_size > 0:
            self._torchcodec_reader_cache[cache_key] = reader
            self._torchcodec_reader_cache.move_to_end(cache_key)
            if len(self._torchcodec_reader_cache) > cache_size:
                _, evicted_reader = self._torchcodec_reader_cache.popitem(last=False)
                del evicted_reader
        return reader

    @staticmethod
    def _frame_indices_to_range(frame_indices: list[int], fetch_mode: str) -> Optional[tuple[int, int, int]]:
        if fetch_mode == "indices" or len(frame_indices) < 2:
            return None
        step = int(frame_indices[1] - frame_indices[0])
        if step <= 0:
            return None
        if any((frame_indices[i + 1] - frame_indices[i]) != step for i in range(len(frame_indices) - 1)):
            if fetch_mode == "range":
                raise RuntimeError(
                    f"Requested torchcodec range fetch but indices are not an arithmetic progression: {frame_indices}"
                )
            return None
        return int(frame_indices[0]), int(frame_indices[-1] + step), step

    def _decode_video_specs(
        self,
        examples: List[dict],
        spec_key: str,
        *,
        return_timing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        decode_threads = max(1, int(self.config.datasets.vla_data.get("video_backend_num_threads", 1)))
        target_size = int(
            self.config.datasets.vla_data.get(
                "video_resolution_size",
                self.config.datasets.vla_data.get("resolution_size", 224),
            )
        )
        encoder_device = next(self.vj_encoder.parameters()).device
        decode_backend = self._gpu_video_decode_backend()
        torchcodec_seek_mode = self._torchcodec_seek_mode()
        torchcodec_dimension_order = self._torchcodec_dimension_order()
        torchcodec_fetch_mode = self._torchcodec_fetch_mode()
        torchcodec_cuda_backend = self._torchcodec_cuda_backend()
        torchcodec_nvdec_cache_capacity = self._torchcodec_nvdec_cache_capacity()
        if decode_backend not in {"decord", "torchcodec"}:
            raise ValueError(f"Unsupported gpu_video_decode_backend: {decode_backend}")
        if decode_backend == "decord" and not DECORD_AVAILABLE:
            raise ImportError("CUDA-enabled decord is required for rank-side GPU video decoding.")
        if decode_backend == "torchcodec" and not TORCHCODEC_AVAILABLE:
            raise ImportError("TorchCodec is required for rank-side TorchCodec video decoding.")
        if decode_backend == "torchcodec" and encoder_device.type == "cuda":
            from torchcodec.decoders import set_nvdec_cache_capacity

            if torchcodec_nvdec_cache_capacity is not None:
                set_nvdec_cache_capacity(torchcodec_nvdec_cache_capacity)
        batch_videos = []
        debug_enabled = self._gpu_decode_debug_enabled()
        total_specs = 0
        batch_decode_start = time.perf_counter()
        decode_total_time = 0.0
        postprocess_total_time = 0.0

        if debug_enabled:
            self._log_gpu_decode_debug(
                f"start {spec_key}: examples={len(examples)} target_size={target_size} "
                f"device={encoder_device} decode_threads={decode_threads} backend={decode_backend} "
                f"torchcodec_cuda_backend={torchcodec_cuda_backend if decode_backend == 'torchcodec' else 'n/a'} "
                f"torchcodec_seek_mode={torchcodec_seek_mode if decode_backend == 'torchcodec' else 'n/a'} "
                f"torchcodec_fetch_mode={torchcodec_fetch_mode if decode_backend == 'torchcodec' else 'n/a'} "
                f"torchcodec_dimension_order={torchcodec_dimension_order if decode_backend == 'torchcodec' else 'n/a'}"
            )

        for example_index, example in enumerate(examples):
            view_videos = []
            for view_index, spec in enumerate(example[spec_key]):
                total_specs += 1
                video_path = spec["video_path"]
                spec_decode_start = time.perf_counter()
                if debug_enabled:
                    self._log_gpu_decode_debug(
                        f"spec {total_specs} begin example={example_index} view={view_index} path={Path(video_path).name}"
                    )
                if "frame_indices" in spec:
                    indices_start = time.perf_counter()
                    frame_indices = np.asarray(spec["frame_indices"], dtype=np.int64).tolist()
                    if debug_enabled:
                        self._log_gpu_decode_debug(
                            f"spec {total_specs} using cached frame_indices count={len(frame_indices)} "
                            f"elapsed={time.perf_counter() - indices_start:.3f}s"
                        )
                else:
                    ts_lookup_start = time.perf_counter()
                    timestamps = np.asarray(spec["timestamps"], dtype=np.float64)
                    decode_device_index = int(encoder_device.index or 0) if encoder_device.type == "cuda" else -1
                    if decode_backend == "decord":
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} lookup timestamps count={len(timestamps)}"
                            )
                        frame_ts = _get_rank_video_frame_timestamps(
                            video_path,
                            decode_device_index,
                            decode_threads,
                        )[:, :1]
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} frame timestamp table shape={frame_ts.shape} "
                                f"elapsed={time.perf_counter() - ts_lookup_start:.3f}s"
                            )
                        argmin_start = time.perf_counter()
                        frame_indices = np.abs(frame_ts - timestamps).argmin(axis=0).astype(np.int64).tolist()
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} mapped timestamps->indices elapsed={time.perf_counter() - argmin_start:.3f}s"
                            )
                    else:
                        frame_indices = None
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} using torchcodec timestamp decode count={len(timestamps)} "
                                f"elapsed={time.perf_counter() - ts_lookup_start:.3f}s"
                            )

                if encoder_device.type == "cuda":
                    decode_device_index = int(encoder_device.index or 0)
                    if decode_backend == "decord":
                        create_start = time.perf_counter()
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} creating gpu reader device={decode_device_index}"
                            )
                        reader = _get_rank_gpu_video_reader(
                            video_path, decode_device_index, decode_threads
                        )
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} gpu reader created elapsed={time.perf_counter() - create_start:.3f}s"
                            )
                        get_batch_start = time.perf_counter()
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} gpu get_batch start count={len(frame_indices)}"
                            )
                        frames = reader.get_batch(frame_indices)
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} gpu get_batch returned elapsed={time.perf_counter() - get_batch_start:.3f}s"
                            )
                        dlpack_start = time.perf_counter()
                        frames = torch.utils.dlpack.from_dlpack(frames.to_dlpack())
                        del reader
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} gpu dlpack->torch elapsed={time.perf_counter() - dlpack_start:.3f}s"
                            )
                    else:
                        decode_device = f"cuda:{decode_device_index}"
                        reader = self._get_or_create_torchcodec_reader(
                            video_path=video_path,
                            device=decode_device,
                            decode_threads=decode_threads,
                            dimension_order=torchcodec_dimension_order,
                            seek_mode=torchcodec_seek_mode,
                            cuda_backend=torchcodec_cuda_backend,
                            debug_enabled=debug_enabled,
                            spec_index=total_specs,
                        )
                        get_batch_start = time.perf_counter()
                        frame_range = (
                            self._frame_indices_to_range(frame_indices, torchcodec_fetch_mode)
                            if frame_indices is not None
                            else None
                        )
                        if frame_range is not None:
                            start_idx, stop_idx, step_idx = frame_range
                            if debug_enabled:
                                self._log_gpu_decode_debug(
                                    f"spec {total_specs} torchcodec get_frames_in_range start={start_idx} stop={stop_idx} step={step_idx}"
                                )
                            frame_batch = reader.get_frames_in_range(
                                start=start_idx,
                                stop=stop_idx,
                                step=step_idx,
                            )
                        elif frame_indices is not None:
                            if debug_enabled:
                                self._log_gpu_decode_debug(
                                    f"spec {total_specs} torchcodec get_frames_at start count={len(frame_indices)}"
                                )
                            frame_batch = reader.get_frames_at(indices=frame_indices)
                        else:
                            if debug_enabled:
                                self._log_gpu_decode_debug(
                                    f"spec {total_specs} torchcodec get_frames_played_at start count={len(timestamps)}"
                                )
                            frame_batch = reader.get_frames_played_at(seconds=timestamps.tolist())
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} torchcodec frame fetch returned elapsed={time.perf_counter() - get_batch_start:.3f}s"
                            )
                        frames = frame_batch.data
                        del frame_batch
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} torchcodec tensor ready device={frames.device} dtype={frames.dtype}"
                            )
                else:
                    create_start = time.perf_counter()
                    if debug_enabled:
                        self._log_gpu_decode_debug(
                            f"spec {total_specs} creating cpu reader"
                        )
                    if decode_backend == "decord":
                        reader = _get_rank_cpu_video_reader(video_path, decode_threads)
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} cpu reader created elapsed={time.perf_counter() - create_start:.3f}s"
                            )
                        get_batch_start = time.perf_counter()
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} cpu get_batch start count={len(frame_indices)}"
                            )
                        frames = torch.from_numpy(reader.get_batch(frame_indices).asnumpy())
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} decord cpu get_batch elapsed={time.perf_counter() - get_batch_start:.3f}s"
                            )
                    else:
                        reader = self._get_or_create_torchcodec_reader(
                            video_path=video_path,
                            device="cpu",
                            decode_threads=decode_threads,
                            dimension_order=torchcodec_dimension_order,
                            seek_mode=torchcodec_seek_mode,
                            cuda_backend=torchcodec_cuda_backend,
                            debug_enabled=debug_enabled,
                            spec_index=total_specs,
                        )
                        get_batch_start = time.perf_counter()
                        frame_range = (
                            self._frame_indices_to_range(frame_indices, torchcodec_fetch_mode)
                            if frame_indices is not None
                            else None
                        )
                        if frame_range is not None:
                            start_idx, stop_idx, step_idx = frame_range
                            if debug_enabled:
                                self._log_gpu_decode_debug(
                                    f"spec {total_specs} torchcodec cpu get_frames_in_range start={start_idx} stop={stop_idx} step={step_idx}"
                                )
                            frame_batch = reader.get_frames_in_range(
                                start=start_idx,
                                stop=stop_idx,
                                step=step_idx,
                            )
                        elif frame_indices is not None:
                            if debug_enabled:
                                self._log_gpu_decode_debug(
                                    f"spec {total_specs} torchcodec cpu get_frames_at start count={len(frame_indices)}"
                                )
                            frame_batch = reader.get_frames_at(indices=frame_indices)
                        else:
                            if debug_enabled:
                                self._log_gpu_decode_debug(
                                    f"spec {total_specs} torchcodec cpu get_frames_played_at start count={len(timestamps)}"
                                )
                            frame_batch = reader.get_frames_played_at(seconds=timestamps.tolist())
                        if debug_enabled:
                            self._log_gpu_decode_debug(
                                f"spec {total_specs} torchcodec cpu frame fetch elapsed={time.perf_counter() - get_batch_start:.3f}s"
                            )
                        frames = frame_batch.data.cpu()
                        del frame_batch

                decode_total_time += time.perf_counter() - spec_decode_start
                post_start = time.perf_counter()
                frames = frames.to(encoder_device, non_blocking=True)
                if decode_backend == "torchcodec" and torchcodec_dimension_order == "NCHW":
                    _, _, H, W = frames.shape
                else:
                    frames = frames.permute(0, 3, 1, 2).contiguous()
                    _, _, H, W = frames.shape
                if H != target_size or W != target_size:
                    frames = F.interpolate(
                        frames.to(dtype=torch.float32),
                        size=(target_size, target_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    frames = frames.clamp_(0, 255).round_().to(torch.uint8)
                elif frames.dtype != torch.uint8:
                    frames = frames.to(torch.uint8)
                if debug_enabled:
                    self._log_gpu_decode_debug(
                        f"spec {total_specs} postprocess shape={tuple(frames.shape)} "
                        f"elapsed={time.perf_counter() - post_start:.3f}s"
                    )
                postprocess_total_time += time.perf_counter() - post_start
                view_videos.append(frames)

            if len(view_videos) == 1:
                view_videos.append(view_videos[0].clone())
            batch_videos.append(torch.stack(view_videos, dim=0))

        if debug_enabled:
            self._log_gpu_decode_debug(
                f"done {spec_key}: examples={len(examples)} specs={total_specs} "
                f"elapsed={time.perf_counter() - batch_decode_start:.3f}s"
            )
        batch_tensor = torch.stack(batch_videos, dim=0)
        if return_timing:
            return batch_tensor, {
                "video_decode_time": decode_total_time,
                "video_postprocess_time": postprocess_total_time,
                "video_decode_total_time": time.perf_counter() - batch_decode_start,
                "video_decode_specs": float(total_specs),
            }
        return batch_tensor

    def _extract_training_videos(
        self,
        examples: List[dict] | dict,
    ) -> tuple[np.ndarray | torch.Tensor, Optional[np.ndarray | torch.Tensor]]:
        if isinstance(examples, dict):
            batch_compact_videos = examples.get("video_compact")
            if batch_compact_videos is not None:
                return self._split_compact_videos(batch_compact_videos)
            return examples["video"], examples.get("video_target")

        if "_prefetched_video_compact_batch" in examples[0]:
            return self._split_compact_videos(examples[0]["_prefetched_video_compact_batch"])

        if "_prefetched_video_batch" in examples[0]:
            return examples[0]["_prefetched_video_batch"], None

        if "video_compact_decode_specs" in examples[0]:
            batch_compact_videos = self._decode_video_specs(examples, "video_compact_decode_specs")
            return self._split_compact_videos(batch_compact_videos)

        if "video_decode_specs" in examples[0]:
            return self._decode_video_specs(examples, "video_decode_specs"), None

        if "video_compact" in examples[0]:
            batch_compact_videos = np.stack([example["video_compact"] for example in examples]).transpose(0, 1, 2, 5, 3, 4)
            return self._split_compact_videos(batch_compact_videos)

        batch_videos = np.stack([example["video"] for example in examples]).transpose(0, 1, 2, 5, 3, 4)
        batch_target_videos = None
        if "video_target" in examples[0]:
            batch_target_videos = np.stack([example["video_target"] for example in examples]).transpose(0, 1, 2, 5, 3, 4)
        return batch_videos, batch_target_videos

    def prepare_rank_prefetched_batch(
        self,
        examples: List[dict] | dict,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[List[dict] | dict, Optional[torch.cuda.Event]]:
        if isinstance(examples, dict) or not isinstance(examples, list) or len(examples) == 0:
            return examples, None

        batch_key = None
        spec_key = None
        if "video_compact_decode_specs" in examples[0]:
            batch_key = "_prefetched_video_compact_batch"
            spec_key = "video_compact_decode_specs"
        elif "video_decode_specs" in examples[0]:
            batch_key = "_prefetched_video_batch"
            spec_key = "video_decode_specs"

        if spec_key is None:
            return examples, None

        stream_ctx = (
            torch.cuda.stream(stream)
            if stream is not None and torch.cuda.is_available()
            else nullcontext()
        )
        instructions = [example["lang"] for example in examples]
        has_actions = "action" in examples[0]
        timing_payload = None
        with torch.inference_mode(), stream_ctx:
            decoded_batch, decode_timing = self._decode_video_specs(
                examples,
                spec_key,
                return_timing=True,
            )
            qwen_build_start = time.perf_counter()
            qwen_inputs = self._build_qwen_inputs_from_video_tensor(
                batch_videos=self._split_compact_videos(decoded_batch)[0] if batch_key == "_prefetched_video_compact_batch" else decoded_batch,
                instructions=instructions,
                has_actions=has_actions,
            )
            timing_payload = dict(decode_timing)
            timing_payload["qwen_tensor_build_time"] = time.perf_counter() - qwen_build_start

        prepared_examples = [dict(example) for example in examples]
        for example in prepared_examples:
            example.pop(spec_key, None)
        prepared_examples[0][batch_key] = decoded_batch

        prepared_examples[0]["qwen_inputs"] = qwen_inputs
        if timing_payload is not None:
            prepared_examples[0]["_prefetch_timing"] = timing_payload

        ready_event = None
        if stream is not None and torch.cuda.is_available():
            ready_event = torch.cuda.Event()
            stream.record_event(ready_event)

        return prepared_examples, ready_event

    def _encode_videos(
        self,
        batch_videos: np.ndarray | torch.Tensor,
        device: torch.device,
        *,
        return_timing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """
        Encode multi-view videos into patch tokens.

        Args:
            batch_videos: [B, V, T, C, H, W] uint8/float video tensor.
            device: target device.
        """
        B, V, T, C, H, W = batch_videos.shape
        flat_videos = batch_videos.reshape(B * V, T, C, H, W)
        source = self.config.framework.vj2_model.get("source", "hf")
        encode_start = time.perf_counter()
        video_tensor_to_cuda_time = 0.0

        if source == "hf":
            processed = []
            for i in range(B * V):
                processed.append(
                    self.vj_processor(videos=flat_videos[i], return_tensors="pt")["pixel_values_videos"]
                )
            input_videos = torch.cat(processed, dim=0)
            to_cuda_start = time.perf_counter()
            input_videos = input_videos.to(device, non_blocking=True)
            video_tensor_to_cuda_time += time.perf_counter() - to_cuda_start
            encoded = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
        elif source == "torchhub":
            crop_size = self.config.framework.vj2_model.get("crop_size", 384)
            if H == crop_size and W == crop_size:
                input_videos = flat_videos
                if isinstance(input_videos, np.ndarray):
                    input_videos = torch.from_numpy(np.ascontiguousarray(input_videos))
                to_cuda_start = time.perf_counter()
                input_videos = input_videos.to(device, dtype=torch.float32, non_blocking=True)
                video_tensor_to_cuda_time += time.perf_counter() - to_cuda_start
                input_videos = input_videos.permute(0, 2, 1, 3, 4)
                input_videos.div_(255.0)
                input_videos.sub_(self._img_mean.to(device=device)).div_(self._img_std.to(device=device))
            else:
                if isinstance(flat_videos, torch.Tensor):
                    flat_videos_thwc = flat_videos.permute(0, 1, 3, 4, 2).cpu().numpy()
                else:
                    flat_videos_thwc = flat_videos.transpose(0, 1, 3, 4, 2)
                processed = []
                for i in range(B * V):
                    out = self.vj_processor(flat_videos_thwc[i])
                    if isinstance(out, list):
                        out = out[0]
                    processed.append(out)
                input_videos = torch.stack(processed)
                to_cuda_start = time.perf_counter()
                input_videos = input_videos.to(device, non_blocking=True)
                video_tensor_to_cuda_time += time.perf_counter() - to_cuda_start
            if self.config.get("trainer", {}).get("channels_last", False):
                input_videos = input_videos.contiguous(memory_format=torch.channels_last_3d)
            encoded = self.vj_encoder(input_videos)
        else:
            raise ValueError(f"Unsupported V-JEPA source: {source}")

        merged = torch.cat(torch.chunk(encoded, chunks=V, dim=0), dim=2)
        if return_timing:
            total_encode_time = time.perf_counter() - encode_start
            return merged, {
                "video_tensor_to_cuda_time": video_tensor_to_cuda_time,
                "vj_encode_time": max(0.0, total_encode_time - video_tensor_to_cuda_time),
            }
        return merged

    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def forward(
        self,
        examples: List[dict] = None,
        rabc_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple:
        timing_stats = {
            "video_tensor_to_cuda_time": 0.0,
            "qwen_input_build_time": 0.0,
            "qwen_forward_time": 0.0,
            "vj_encode_time": 0.0,
            "predictor_action_head_time": 0.0,
        }
        batch_videos, batch_target_videos = self._extract_training_videos(examples)
        if isinstance(examples, dict):
            actions = examples.get("action")
            action_mask = examples.get("action_mask")
            state = examples.get("state")
            if "qwen_inputs" in examples:
                qwen_inputs = self._move_qwen_inputs(examples["qwen_inputs"])
            else:
                instructions = list(examples["lang"])
                qwen_inputs, qwen_timing = self._build_qwen_inputs_from_video_tensor(
                    batch_videos=batch_videos,
                    instructions=instructions,
                    has_actions=actions is not None,
                    return_timing=True,
                )
                timing_stats["video_tensor_to_cuda_time"] += qwen_timing["video_tensor_to_cuda_time"]
                timing_stats["qwen_input_build_time"] += qwen_timing["qwen_input_build_time"]
            has_actions = actions is not None
        else:
            actions = [example["action"] for example in examples] if "action" in examples[0] else None
            action_mask = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None
            state = [example["state"] for example in examples] if "state" in examples[0] else None
            if "qwen_inputs" in examples[0]:
                qwen_inputs = self._move_qwen_inputs(examples[0]["qwen_inputs"])
            else:
                instructions = [example["lang"] for example in examples]
                qwen_inputs, qwen_timing = self._build_qwen_inputs_from_video_tensor(
                    batch_videos=batch_videos,
                    instructions=instructions,
                    has_actions=actions is not None,
                    return_timing=True,
                )
                timing_stats["video_tensor_to_cuda_time"] += qwen_timing["video_tensor_to_cuda_time"]
                timing_stats["qwen_input_build_time"] += qwen_timing["qwen_input_build_time"]
            has_actions = actions is not None

        input_ids = qwen_inputs["input_ids"]
        action_token_ids = self._action_token_ids_t.to(input_ids.device)
        embodied_token_id = self._embodied_token_id_t.to(input_ids.device)
        action_indices = torch.isin(input_ids, action_token_ids).nonzero(as_tuple=True)
        embodied_action_indices = torch.isin(input_ids, embodied_token_id).nonzero(as_tuple=True)
        
        qwen_context = nullcontext() if self._qwen_requires_grad() else torch.no_grad()
        qwen_forward_start = time.perf_counter()
        with qwen_context, torch.autocast("cuda", dtype=torch.bfloat16):
            # Use feature-extraction path: skips LM head and avoids storing all
            # intermediate hidden states (saves both compute and memory).
            last_hidden = self.qwen_vl_interface.forward_features(**qwen_inputs)
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)
        timing_stats["qwen_forward_time"] += time.perf_counter() - qwen_forward_start

        # Step 2: JEPA Encoder
        with torch.autocast("cuda", dtype=torch.bfloat16):
            B, V, T, C, H, W = batch_videos.shape
            encoder_device = next(self.vj_encoder.parameters()).device
            encoder_context = torch.no_grad() if self.vj_freeze_encoder else nullcontext()
            with encoder_context:
                input_states, input_encode_timing = self._encode_videos(
                    batch_videos=batch_videos,
                    device=encoder_device,
                    return_timing=True,
                )
                timing_stats["video_tensor_to_cuda_time"] += input_encode_timing["video_tensor_to_cuda_time"]
                timing_stats["vj_encode_time"] += input_encode_timing["vj_encode_time"]
                if batch_target_videos is not None:
                    gt_states, target_encode_timing = self._encode_videos(
                        batch_videos=batch_target_videos,
                        device=encoder_device,
                        return_timing=True,
                    )
                    timing_stats["video_tensor_to_cuda_time"] += target_encode_timing["video_tensor_to_cuda_time"]
                    timing_stats["vj_encode_time"] += target_encode_timing["vj_encode_time"]
                else:
                    video_embeddings = input_states
                    T = T // self._get_vjepa_attr("tubelet_size")
                    input_states = video_embeddings[:, :video_embeddings.shape[1] // T * (T-1), :]
                    gt_states = video_embeddings[:, video_embeddings.shape[1] // T:, :]

        # Step 3: VJ Predictor / Action Head
        predictor_head_start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted_states = self.vj_predictor(
                input_states,
                action_tokens
            )

            teacher_forcing_wm_loss = F.l1_loss(
                predicted_states,
                gt_states,
                reduction="mean"
            )
        
        if not has_actions:
            timing_stats["predictor_action_head_time"] += time.perf_counter() - predictor_head_start
            return {"wm_loss": teacher_forcing_wm_loss, **timing_stats}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if isinstance(actions, torch.Tensor):
                actions = actions.to(device=last_hidden.device, dtype=torch.float32, non_blocking=True)
            else:
                actions = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(
                    device=last_hidden.device, non_blocking=True
                )
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

            repeated_diffusion_steps = self._repeated_diffusion_steps
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)

            action_mask_repeated = None
            if action_mask is not None:
                if isinstance(action_mask, torch.Tensor):
                    action_mask = action_mask.to(device=last_hidden.device, dtype=torch.float32, non_blocking=True)
                else:
                    action_mask = torch.from_numpy(np.asarray(action_mask, dtype=np.float32)).to(
                        device=last_hidden.device, non_blocking=True
                    )
                action_mask_target = action_mask[:, -(self.future_action_window_size + 1) :, :]
                action_mask_repeated = action_mask_target.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                if isinstance(state, torch.Tensor):
                    state = state.to(device=last_hidden.device, dtype=torch.float32, non_blocking=True)
                else:
                    state = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(
                        device=last_hidden.device, non_blocking=True
                    )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            per_sample_action_loss = self.action_model(
                embodied_action_repeated,
                actions_target_repeated,
                state_repeated,
                action_mask=action_mask_repeated,
                reduction="none",
            ).float()
            if rabc_weights is not None:
                expanded_weights = rabc_weights.repeat(repeated_diffusion_steps)
                action_loss = (per_sample_action_loss * expanded_weights).sum() / (expanded_weights.sum() + 1e-6)
            else:
                action_loss = per_sample_action_loss.mean()

        timing_stats["predictor_action_head_time"] += time.perf_counter() - predictor_head_start
        result = {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss, **timing_stats}
        if rabc_weights is not None:
            result["rabc_mean_weight"] = rabc_weights.detach().mean()
        return result

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: Optional[List[List[Image.Image]]] = None,
        instructions: Optional[List[str]] = None,
        state: Optional[np.ndarray] = None,
        batch: Optional[dict | List[dict]] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to predict future actions via flow matching.

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: Natural language task instructions.
            state: Optional proprioceptive state.
            **kwargs: Reserved.

        Returns:
            dict with normalized_actions [B, T, action_dim] and embodied_action_tokens.
        """
        if batch is not None:
            if isinstance(batch, dict):
                qwen_inputs = self._move_qwen_inputs(batch["qwen_inputs"])
                if state is None:
                    state = batch.get("state")
            else:
                if state is None and "state" in batch[0]:
                    state = [example["state"] for example in batch]
                if "qwen_inputs" in batch[0]:
                    qwen_inputs = self._move_qwen_inputs(batch[0]["qwen_inputs"])
                else:
                    instructions = [example["lang"] for example in batch]
                    if (
                        "video_decode_specs" in batch[0]
                        or "video_compact_decode_specs" in batch[0]
                        or "video" in batch[0]
                        or "video_compact" in batch[0]
                    ):
                        batch_videos, _ = self._extract_training_videos(batch)
                        qwen_inputs = self._build_qwen_inputs_from_video_tensor(
                            batch_videos=batch_videos,
                            instructions=instructions,
                            has_actions=False,
                            prompt_replace_dict={
                                "{actions}": self.replace_prompt,
                                "{e_actions}": self.embodied_replace_prompt,
                            },
                            prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
                        )
                    else:
                        batch_images = [example["image"] for example in batch]
                        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                            images=batch_images,
                            instructions=instructions,
                            prompt_replace_dict={
                                "{actions}": self.replace_prompt,
                                "{e_actions}": self.embodied_replace_prompt,
                            },
                        )
        else:
            train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
            if train_obs_image_size:
                batch_images = resize_images(batch_images, target_size=train_obs_image_size)

            prompt_replace_dict, prompt_template = self._resolve_qwen_prompt_args(has_actions=True)
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict=prompt_replace_dict,
                prompt_template=prompt_template,
            )
        self._validate_qwen_action_prompt_tokens(
            qwen_inputs,
            has_actions=True,
            stage="predict_action",
        )

        embodied_action_indices = torch.isin(qwen_inputs["input_ids"], self._embodied_token_id_t).nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self.qwen_vl_interface.forward_features(**qwen_inputs)
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

        if state is not None:
            if isinstance(state, torch.Tensor):
                state = state.to(last_hidden.device, dtype=torch.float32, non_blocking=True)
            else:
                state = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(last_hidden.device, dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state)

        normalized_actions = pred_actions.float().detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}
