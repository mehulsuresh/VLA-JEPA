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
import importlib
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


def _clean_vjepa_backbone_key(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned

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

    def _build_qwen_inputs_from_examples(self, examples: List[dict]) -> dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        has_actions = "action" in examples[0]
        if has_actions:
            return self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
            )
        return self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={"{actions}": self.replace_prompt},
            prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""),
        )

    def _encode_videos(self, batch_videos: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Encode multi-view videos into patch tokens.

        Args:
            batch_videos: [B, V, T, C, H, W] uint8/float video tensor.
            device: target device.
        """
        B, V, T, C, H, W = batch_videos.shape
        flat_videos = batch_videos.reshape(B * V, T, C, H, W)
        source = self.config.framework.vj2_model.get("source", "hf")

        if source == "hf":
            processed = []
            for i in range(B * V):
                processed.append(
                    self.vj_processor(videos=flat_videos[i], return_tensors="pt")["pixel_values_videos"]
                )
            input_videos = torch.cat(processed, dim=0).to(device, non_blocking=True)
            encoded = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
        elif source == "torchhub":
            crop_size = self.config.framework.vj2_model.get("crop_size", 384)
            if H == crop_size and W == crop_size:
                input_videos = flat_videos
                if isinstance(input_videos, np.ndarray):
                    input_videos = torch.from_numpy(np.ascontiguousarray(input_videos))
                input_videos = input_videos.to(device, dtype=torch.float32, non_blocking=True)
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
                input_videos = torch.stack(processed).to(device, non_blocking=True)
            if self.config.get("trainer", {}).get("channels_last", False):
                input_videos = input_videos.contiguous(memory_format=torch.channels_last_3d)
            encoded = self.vj_encoder(input_videos)
        else:
            raise ValueError(f"Unsupported V-JEPA source: {source}")

        return torch.cat(torch.chunk(encoded, chunks=V, dim=0), dim=2)

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
        if isinstance(examples, dict):
            batch_videos = examples["video"]
            batch_target_videos = examples.get("video_target")
            actions = examples.get("action")
            state = examples.get("state")
            qwen_inputs = self._move_qwen_inputs(examples["qwen_inputs"])
            has_actions = actions is not None
        else:
            batch_videos = np.stack([example["video"] for example in examples]).transpose(0, 1, 2, 5, 3, 4)
            batch_target_videos = None
            if "video_target" in examples[0]:
                batch_target_videos = np.stack([example["video_target"] for example in examples]).transpose(0, 1, 2, 5, 3, 4)
            actions = [example["action"] for example in examples] if "action" in examples[0] else None
            state = [example["state"] for example in examples] if "state" in examples[0] else None
            qwen_inputs = self._build_qwen_inputs_from_examples(examples)
            has_actions = actions is not None

        input_ids = qwen_inputs["input_ids"]
        action_token_ids = self._action_token_ids_t.to(input_ids.device)
        embodied_token_id = self._embodied_token_id_t.to(input_ids.device)
        action_indices = torch.isin(input_ids, action_token_ids).nonzero(as_tuple=True)
        embodied_action_indices = torch.isin(input_ids, embodied_token_id).nonzero(as_tuple=True)
        
        qwen_context = nullcontext() if self._qwen_requires_grad() else torch.no_grad()
        with qwen_context, torch.autocast("cuda", dtype=torch.bfloat16):
            # Use feature-extraction path: skips LM head and avoids storing all
            # intermediate hidden states (saves both compute and memory).
            last_hidden = self.qwen_vl_interface.forward_features(**qwen_inputs)
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

        # Step 2: JEPA Encoder
        with torch.autocast("cuda", dtype=torch.bfloat16):
            B, V, T, C, H, W = batch_videos.shape
            encoder_device = next(self.vj_encoder.parameters()).device
            encoder_context = torch.no_grad() if self.vj_freeze_encoder else nullcontext()
            with encoder_context:
                input_states = self._encode_videos(batch_videos=batch_videos, device=encoder_device)
                if batch_target_videos is not None:
                    gt_states = self._encode_videos(batch_videos=batch_target_videos, device=encoder_device)
                else:
                    video_embeddings = input_states
                    T = T // self._get_vjepa_attr("tubelet_size")
                    input_states = video_embeddings[:, :video_embeddings.shape[1] // T * (T-1), :]
                    gt_states = video_embeddings[:, video_embeddings.shape[1] // T:, :]

        # Step 3: VJ Predictor
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
            return {"wm_loss": teacher_forcing_wm_loss}

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
                reduction="none",
            ).float()
            if rabc_weights is not None:
                expanded_weights = rabc_weights.repeat(repeated_diffusion_steps)
                action_loss = (per_sample_action_loss * expanded_weights).sum() / (expanded_weights.sum() + 1e-6)
            else:
                action_loss = per_sample_action_loss.mean()

        result = {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss}
        if rabc_weights is not None:
            result["rabc_mean_weight"] = rabc_weights.detach().mean()
        return result

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: Optional[List[List[Image.Image]]] = None,
        instructions: Optional[List[str]] = None,
        state: Optional[np.ndarray] = None,
        batch: Optional[dict] = None,
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
            qwen_inputs = self._move_qwen_inputs(batch["qwen_inputs"])
            if state is None:
                state = batch.get("state")
        else:
            train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
            if train_obs_image_size:
                batch_images = resize_images(batch_images, target_size=train_obs_image_size)

            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
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
