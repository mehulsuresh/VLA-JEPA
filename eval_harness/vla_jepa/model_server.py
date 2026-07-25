#!/usr/bin/env python3
"""VLA-JEPA adapter for allenai/vla-evaluation-harness.

This file is intentionally standalone: keep harness-specific eval code here
instead of mixing it into the training or robot deployment paths.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


def _add_local_paths() -> None:
    adapter_dir = Path(__file__).resolve().parent
    repo_root = Path(os.environ.get("VLA_JEPA_ROOT", adapter_dir.parents[2])).resolve()
    harness_root = os.environ.get("VLA_EVAL_HARNESS_ROOT")

    for path in (
        repo_root,
        Path(harness_root).resolve() / "src" if harness_root else None,
    ):
        if path is not None and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_add_local_paths()

from deployment.model_server.checkpoint_utils import build_policy_metadata, resolve_policy_checkpoint
from starVLA.model.framework.base_framework import baseframework
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import run_server
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, Observation


logger = logging.getLogger(__name__)


def _as_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [str(item) for item in value]


class VLAJEPAModelServer(PredictModelServer):
    """Batched model server for VLA-JEPA harness evals.

    Benchmark-specific behavior is intentionally restricted to named profiles.
    Add a new profile when a benchmark needs different action conventions
    rather than hiding that behavior in the generic policy path.
    """

    def __init__(
        self,
        checkpoint: str = "",
        *,
        unnorm_key: str | None = None,
        action_norm_mode: str = "auto",
        chunk_size: int | None = None,
        max_batch_size: int = 8,
        max_wait_time: float = 0.05,
        action_ensemble: str = "newest",
        ema_alpha: float = 0.5,
        num_ddim_steps: int | None = None,
        benchmark_profile: str = "libero",
        action_dim: int | None = None,
        image_keys: list[str] | str | None = None,
        include_unlisted_images: bool = True,
        state_keys: list[str] | str | None = None,
        state_dim: int | None = None,
        state_fallback_zeros: bool = False,
        preserve_masked_action_dims: bool | None = None,
        binarize_gripper: bool | None = None,
        gripper_index: int = -1,
        use_bf16: bool = True,
        cuda: int = 0,
        device: str | None = None,
        load_training_backbones: bool = False,
        send_wrist_image: bool = True,
        send_state: bool = True,
        quat_no_antipodal: bool = True,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        if not checkpoint:
            raise ValueError("`checkpoint` is required. Pass --args.checkpoint=/path/to/model.safetensors")

        self.checkpoint = resolve_policy_checkpoint(checkpoint)
        self.device = self._resolve_device(device=device, cuda=cuda)
        self.use_bf16 = bool(use_bf16 and self.device.type == "cuda")
        self.num_ddim_steps = num_ddim_steps
        self.benchmark_profile = self._resolve_benchmark_profile(benchmark_profile)
        self.send_wrist_image = bool(send_wrist_image)
        self.send_state = bool(send_state)
        self.quat_no_antipodal = bool(quat_no_antipodal)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.image_keys = _as_string_list(image_keys)
        self.include_unlisted_images = bool(include_unlisted_images)
        self.state_keys = _as_string_list(state_keys) or ["states", "state"]
        self.state_dim = int(state_dim) if state_dim is not None else None
        self.state_fallback_zeros = bool(state_fallback_zeros)
        self.gripper_index = int(gripper_index)
        self.preserve_masked_action_dims = (
            self.benchmark_profile == "libero"
            if preserve_masked_action_dims is None
            else bool(preserve_masked_action_dims)
        )
        self.binarize_gripper = (
            self.benchmark_profile == "libero"
            if binarize_gripper is None
            else bool(binarize_gripper)
        )

        logger.info("Loading VLA-JEPA checkpoint: %s", self.checkpoint)
        self.model = baseframework.from_pretrained(
            str(self.checkpoint),
            inference_only=not load_training_backbones,
            skip_training_backbones=not load_training_backbones,
        )
        if self.use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device).eval()

        self.metadata = build_policy_metadata(self.model, self.checkpoint)
        self.unnorm_key = self._resolve_unnorm_key(unnorm_key)
        self.action_stats = self.metadata["action_stats_by_key"][self.unnorm_key]
        self.action_norm_mode = self._resolve_action_norm_mode(action_norm_mode)
        self.model_action_dim = self._infer_model_action_dim()
        self.expected_action_dim = self._resolve_expected_action_dim(action_dim)

        resolved_chunk_size = int(chunk_size) if chunk_size is not None else int(self.metadata["action_horizon"])
        if resolved_chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {resolved_chunk_size}")

        super().__init__(
            chunk_size=resolved_chunk_size,
            action_ensemble=action_ensemble,
            ema_alpha=ema_alpha,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
        )
        logger.info(
            "VLA-JEPA server ready: action_horizon=%s chunk_size=%s max_batch_size=%s max_wait_time=%.3f "
            "profile=%s action_dim=%s unnorm_key=%s norm_mode=%s send_wrist=%s send_state=%s",
            self.metadata["action_horizon"],
            self.chunk_size,
            self.max_batch_size,
            self.max_wait_time,
            self.benchmark_profile,
            self.expected_action_dim,
            self.unnorm_key,
            self.action_norm_mode,
            self.send_wrist_image,
            self.send_state,
        )

    @staticmethod
    def _resolve_benchmark_profile(benchmark_profile: str) -> str:
        profile = str(benchmark_profile or "raw").lower().replace("-", "_")
        aliases = {
            "libero_plus": "libero",
            "libero": "libero",
            "raw": "raw",
            "generic": "raw",
        }
        if profile not in aliases:
            raise ValueError(
                f"Unsupported benchmark_profile={benchmark_profile!r}. "
                "Add a named profile before using benchmark-specific conventions."
            )
        return aliases[profile]

    @staticmethod
    def _resolve_device(*, device: str | None, cuda: int) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(cuda))
            return torch.device(f"cuda:{int(cuda)}")
        return torch.device("cpu")

    def _resolve_unnorm_key(self, unnorm_key: str | None) -> str:
        available = list(self.metadata.get("available_unnorm_keys") or [])
        if unnorm_key is None:
            default_key = self.metadata.get("default_unnorm_key")
            if default_key:
                return str(default_key)
            if len(available) == 1:
                return str(available[0])
            raise ValueError(f"Checkpoint has multiple stat keys; pass unnorm_key. Available: {available}")
        if unnorm_key not in available:
            raise ValueError(f"Unknown unnorm_key={unnorm_key!r}; available: {available}")
        return unnorm_key

    def _resolve_action_norm_mode(self, action_norm_mode: str) -> str:
        mode = str(action_norm_mode).lower()
        if mode == "auto":
            mode = str(self.metadata.get("default_action_norm_mode") or "").lower()
            if not mode:
                if "min" in self.action_stats and "max" in self.action_stats:
                    mode = "min_max"
                elif "q01" in self.action_stats and "q99" in self.action_stats:
                    mode = "q99"
        if mode not in {"min_max", "q99", "mean_std"}:
            raise ValueError(
                f"Unsupported action_norm_mode={action_norm_mode!r}; resolved to {mode!r}. "
                "Expected auto|min_max|q99|mean_std."
            )
        return mode

    def _infer_model_action_dim(self) -> int:
        metadata_dim = self.metadata.get("action_dim")
        if metadata_dim is not None:
            return int(metadata_dim)
        for key in ("min", "max", "q01", "q99", "mean", "std"):
            if key in self.action_stats:
                return int(np.asarray(self.action_stats[key]).shape[-1])
        raise ValueError("Could not infer action_dim from checkpoint metadata or action stats")

    def _resolve_expected_action_dim(self, action_dim: int | None) -> int:
        if action_dim is not None:
            expected = int(action_dim)
        elif self.benchmark_profile == "libero":
            expected = 7
        else:
            expected = self.model_action_dim
        if expected != self.model_action_dim:
            raise ValueError(
                f"Checkpoint action_dim={self.model_action_dim} does not match "
                f"benchmark/profile action_dim={expected}."
            )
        return expected

    def get_observation_params(self) -> dict[str, Any]:
        return {
            "send_wrist_image": self.send_wrist_image,
            "send_state": self.send_state,
            "quat_no_antipodal": self.quat_no_antipodal,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        if self.benchmark_profile == "libero":
            return {
                "position": POSITION_DELTA,
                "rotation": ROTATION_AA,
                "gripper": GRIPPER_CLOSE_POS,
            }
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        if self.benchmark_profile == "libero":
            spec: dict[str, DimSpec] = {"agentview": IMAGE_RGB, "language": LANGUAGE}
        else:
            spec = {"image": IMAGE_RGB, "language": LANGUAGE}
        if self.send_wrist_image:
            spec["wrist"] = IMAGE_RGB
        if self.send_state:
            spec["state"] = STATE_EEF_POS_AA_GRIP if self.benchmark_profile == "libero" else RAW
        return spec

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        batch_images: list[list[Image.Image]] = []
        instructions: list[str] = []
        states: list[np.ndarray] = []

        for obs in obs_batch:
            images = self._extract_images(obs)
            batch_images.append(images)
            instructions.append(str(obs.get("task_description", "")))
            if self.send_state:
                states.append(self._extract_state(obs))

        state_batch = np.stack(states, axis=0).astype(np.float32, copy=False) if self.send_state else None
        kwargs: dict[str, Any] = {}
        if self.num_ddim_steps is not None:
            kwargs["num_ddim_steps"] = int(self.num_ddim_steps)

        result = self.model.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            state=state_batch,
            **kwargs,
        )
        normalized = np.asarray(result["normalized_actions"], dtype=np.float32)
        if normalized.ndim != 3 or normalized.shape[0] != len(obs_batch):
            raise RuntimeError(
                f"Expected normalized_actions [B,T,D] with B={len(obs_batch)}, got {normalized.shape}"
            )

        outputs: list[Action] = []
        for sample in normalized:
            actions = self._postprocess_actions(sample)
            outputs.append({"actions": actions.astype(np.float32, copy=False)})
        return outputs

    def _postprocess_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        normalized = np.asarray(normalized_actions, dtype=np.float32).copy()
        if normalized.ndim != 2:
            raise RuntimeError(f"Expected one action chunk [T,D], got {normalized.shape}")
        if normalized.shape[-1] != self.expected_action_dim:
            raise RuntimeError(
                f"Expected action_dim={self.expected_action_dim}, got {normalized.shape[-1]}"
            )

        if self.binarize_gripper and normalized.shape[-1] > 0:
            gripper_idx = self.gripper_index % normalized.shape[-1]
            mask = np.asarray(
                self.action_stats.get("mask", np.ones(self.expected_action_dim, dtype=bool)),
                dtype=bool,
            )
            if gripper_idx < mask.shape[0] and not bool(mask[gripper_idx]):
                normalized[..., gripper_idx] = np.where(normalized[..., gripper_idx] > 0.0, 1.0, -1.0)

        actions = self._unnormalize_actions(normalized)
        if self.benchmark_profile == "libero" and actions.shape[-1] != 7:
            raise RuntimeError(f"LIBERO expects 7D actions; checkpoint produced {actions.shape[-1]}D")
        return actions

    def _unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(normalized_actions, dtype=np.float32), -1.0, 1.0)
        if self.action_norm_mode == "min_max":
            low = np.asarray(self.action_stats["min"], dtype=np.float32)
            high = np.asarray(self.action_stats["max"], dtype=np.float32)
            actions = 0.5 * (clipped + 1.0) * (high - low) + low
        elif self.action_norm_mode == "q99":
            low = np.asarray(self.action_stats["q01"], dtype=np.float32)
            high = np.asarray(self.action_stats["q99"], dtype=np.float32)
            actions = 0.5 * (clipped + 1.0) * (high - low) + low
        elif self.action_norm_mode == "mean_std":
            mean = np.asarray(self.action_stats["mean"], dtype=np.float32)
            std = np.asarray(self.action_stats["std"], dtype=np.float32)
            actions = clipped * std + mean
        else:
            raise ValueError(f"Unsupported action_norm_mode={self.action_norm_mode!r}")

        if self.preserve_masked_action_dims:
            mask = np.asarray(
                self.action_stats.get("mask", np.ones(self.expected_action_dim, dtype=bool)),
                dtype=bool,
            )
            actions = np.where(mask, actions, clipped)
        return actions.astype(np.float32, copy=False)

    def _extract_images(self, obs: Observation) -> list[Image.Image]:
        raw_images = obs.get("images", {})
        if not isinstance(raw_images, dict):
            raw_images = {"agentview": raw_images}

        if self.image_keys is not None:
            ordered_keys = list(self.image_keys)
        elif self.benchmark_profile == "libero":
            ordered_keys = ["agentview"]
            if self.send_wrist_image:
                ordered_keys.append("wrist")
        else:
            ordered_keys = list(raw_images.keys())
        if self.include_unlisted_images:
            ordered_keys.extend(k for k in sorted(raw_images) if k not in ordered_keys)

        images: list[Image.Image] = []
        for key in ordered_keys:
            if key in raw_images:
                images.append(self._to_pil(raw_images[key]))
        if not images:
            raise ValueError(f"Observation did not contain images. Keys: {list(raw_images)}")
        return images

    def _to_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            array = np.asarray(image)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            pil = Image.fromarray(array).convert("RGB")
        if self.image_size is not None and pil.size != (self.image_size[1], self.image_size[0]):
            pil = pil.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
        return pil

    @staticmethod
    def _state_from_obs(obs: Observation, state_keys: list[str]) -> Any:
        for key in state_keys:
            if key in obs:
                return obs[key]
        return None

    def _extract_state(self, obs: Observation) -> np.ndarray:
        state = self._state_from_obs(obs, self.state_keys)
        if state is None:
            if self.state_fallback_zeros and self.state_dim is not None:
                return np.zeros(self.state_dim, dtype=np.float32)
            raise ValueError(
                "Model was configured with send_state=True, but observation has none of "
                f"{self.state_keys}. Set state_fallback_zeros=true only for benchmarks "
                "where missing state is expected."
            )
        array = np.asarray(state, dtype=np.float32).reshape(-1)
        if self.state_dim is not None and array.shape[0] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {array.shape[0]}")
        return array


if __name__ == "__main__":
    run_server(VLAJEPAModelServer)
