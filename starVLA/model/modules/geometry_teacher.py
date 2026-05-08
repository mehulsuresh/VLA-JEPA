from __future__ import annotations

import sys
import os
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_frame_value_range(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if numeric == 1.0:
            return "0_1"
        if numeric == 255.0:
            return "0_255"
        if numeric == 173.0:
            # OmegaConf/YAML can parse an unquoted CLI value like `0_255`
            # as an octal-looking integer. Treat it as the intended uint8 range.
            return "0_255"
    return str(value).lower()


_MOGE2_VITL_NORMAL_NECK_DIMS = (1024, 256, 128, 64, 32)


class DirectGeometryTeacherHead(nn.Module):
    """LingBot-style direct head: project VLM image tokens into teacher feature space."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        head_hidden_multiplier: float = 2.0,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        final_init_std: float = 0.0,
    ):
        super().__init__()
        inner_size = max(int(output_size * head_hidden_multiplier), output_size)
        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.extend(
            [
                nn.Linear(hidden_size, inner_size),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(inner_size, output_size),
            ]
        )
        self.net = nn.Sequential(*layers)
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            if final_init_std <= 0.0:
                nn.init.zeros_(final_layer.weight)
            else:
                nn.init.normal_(final_layer.weight, mean=0.0, std=float(final_init_std))
            nn.init.zeros_(final_layer.bias)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(image_tokens)


class MoGeGeometryTeacher:
    """Frozen MoGe feature teacher kept out of the trainable model state dict."""

    def __init__(self, cfg: Any, logger: Optional[Any] = None):
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self._resolved_frame_value_range: Optional[str] = None
        self._logged_frame_value_range = False
        self._logged_deferred_cpu_load = False

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _insert_moge_path(self) -> None:
        repo_path = (
            _cfg_get(self.cfg, "moge_repo_path", None)
            or os.environ.get("STARVLA_MOGE_REPO_PATH")
            or os.environ.get("MOGE_REPO_PATH")
        )
        if repo_path:
            repo_path = str(Path(str(repo_path)).expanduser())
            if not Path(repo_path).exists():
                raise FileNotFoundError(f"Configured MoGe repo path does not exist: {repo_path}")
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

    def _load_pretrained(self, model_name: str):
        from moge.model.v2 import MoGeModel

        return MoGeModel.from_pretrained(model_name)

    @staticmethod
    def _distributed_rank_world() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
        return 0, 1

    @staticmethod
    def _download_ready_path(model_name: str) -> Path:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
        digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
        return hf_home / "starvla" / "moge" / f".{digest}.ready"

    def _load_pretrained_rank_safe(self, model_name: str):
        model_name_str = str(model_name)
        if Path(model_name_str).expanduser().exists():
            return self._load_pretrained(model_name_str)

        rank, world_size = self._distributed_rank_world()
        if world_size <= 1:
            return self._load_pretrained(model_name_str)

        ready_path = self._download_ready_path(model_name_str)
        timeout_seconds = float(_cfg_get(self.cfg, "download_wait_timeout_seconds", 1800))
        if rank == 0:
            model = self._load_pretrained(model_name_str)
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text(model_name_str, encoding="utf-8")
            return model

        start_time = time.monotonic()
        while not ready_path.exists():
            if time.monotonic() - start_time > timeout_seconds:
                raise TimeoutError(
                    "Timed out waiting for rank 0 to cache MoGe geometry teacher "
                    f"`{model_name_str}` at {ready_path}"
                )
            time.sleep(1.0)
        return self._load_pretrained(model_name_str)

    def initialize(self, device: torch.device) -> None:
        device = torch.device(device)
        if device.type == "cpu" and not bool(_cfg_get(self.cfg, "eager_load_on_cpu", False)):
            if self._feature_dim_without_model() is not None:
                if not self._logged_deferred_cpu_load:
                    self._log(
                        "Deferring MoGe geometry teacher weight load until first non-CPU forward; "
                        "using configured/known metadata for feature dimension."
                    )
                    self._logged_deferred_cpu_load = True
                return
        self._ensure_model(device)

    @staticmethod
    def _module_out_channels(module: nn.Module) -> Optional[int]:
        for child in reversed(list(module.modules())):
            out_channels = getattr(child, "out_channels", None)
            if isinstance(out_channels, int):
                return int(out_channels)
        return None

    def _configured_feature_dim(self) -> Optional[int]:
        configured_feature_dim = _cfg_get(self.cfg, "teacher_feature_dim", None)
        if configured_feature_dim is None:
            return None
        if isinstance(configured_feature_dim, str) and configured_feature_dim.lower() == "auto":
            return None
        return int(configured_feature_dim)

    def _feature_dim_without_model(self) -> Optional[int]:
        configured_feature_dim = self._configured_feature_dim()
        if configured_feature_dim is not None:
            return configured_feature_dim
        model_name = str(_cfg_get(self.cfg, "teacher_model", "Ruicheng/moge-2-vitl-normal")).rstrip("/")
        model_path_name = Path(model_name).name
        feature_source = str(_cfg_get(self.cfg, "teacher_feature_source", "neck")).lower()
        if model_name != "Ruicheng/moge-2-vitl-normal" and model_path_name != "moge-2-vitl-normal":
            return None
        if feature_source == "encoder":
            return _MOGE2_VITL_NORMAL_NECK_DIMS[0]
        if feature_source == "neck":
            feature_level = int(_cfg_get(self.cfg, "teacher_feature_level", 0))
            if 0 <= feature_level < len(_MOGE2_VITL_NORMAL_NECK_DIMS):
                return _MOGE2_VITL_NORMAL_NECK_DIMS[feature_level]
        return None

    def _feature_dim_from_model(self, model: nn.Module) -> int:
        feature_source = str(_cfg_get(self.cfg, "teacher_feature_source", "neck")).lower()
        if feature_source == "encoder":
            projection = model.encoder.output_projections[0]
            feature_dim = getattr(projection, "out_channels", None)
            if not isinstance(feature_dim, int):
                raise RuntimeError("Could not infer MoGe encoder feature dimension")
            return int(feature_dim)
        if feature_source != "neck":
            raise ValueError(f"Unsupported teacher_feature_source: {feature_source}")

        feature_level = int(_cfg_get(self.cfg, "teacher_feature_level", 0))
        output_blocks = getattr(model.neck, "output_blocks", None)
        input_blocks = getattr(model.neck, "input_blocks", None)
        res_blocks = getattr(model.neck, "res_blocks", None)
        try:
            candidates = (output_blocks[feature_level], input_blocks[feature_level], res_blocks[feature_level])
        except (TypeError, IndexError):
            raise ValueError(f"Invalid MoGe neck teacher_feature_level: {feature_level}") from None
        for module in candidates:
            feature_dim = self._module_out_channels(module)
            if feature_dim is not None:
                return feature_dim
        raise RuntimeError(f"Could not infer MoGe neck feature dimension for level {feature_level}")

    def feature_dim(self) -> int:
        if self.model is None:
            feature_dim = self._feature_dim_without_model()
            if feature_dim is None:
                raise RuntimeError(
                    "MoGe geometry teacher feature dimension is `auto`, but it cannot be inferred without "
                    "loading the teacher weights for this model. Set teacher_feature_dim explicitly, use "
                    "Ruicheng/moge-2-vitl-normal, or enable eager_load_on_cpu."
                )
            return feature_dim
        return self._feature_dim_from_model(self.model)

    def _ensure_model(self, device: torch.device) -> nn.Module:
        dtype = torch.float16 if device.type == "cuda" and bool(_cfg_get(self.cfg, "use_fp16", True)) else torch.float32
        if self.model is not None:
            if self.device != device or self.dtype != dtype:
                self.model.to(device=device, dtype=dtype)
                self.device = device
                self.dtype = dtype
            return self.model

        self._insert_moge_path()
        try:
            import moge.model.v2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MoGe geometry teacher is enabled, but `moge.model.v2.MoGeModel` could not be imported. "
                "Install MoGe and its dependencies, set `framework.depth_teacher_aux.moge_repo_path`, "
                "or set STARVLA_MOGE_REPO_PATH/MOGE_REPO_PATH to a local MoGe checkout."
            ) from exc

        model_name = _cfg_get(self.cfg, "teacher_model", "Ruicheng/moge-2-vitl-normal")
        self._log(f"Loading MoGe geometry teacher `{model_name}`")
        model = self._load_pretrained_rank_safe(model_name)

        model.requires_grad_(False)
        model.eval()
        model.to(device=device, dtype=dtype)
        self.model = model
        self.device = device
        self.dtype = dtype
        expected_feature_dim = self._feature_dim_without_model()
        if expected_feature_dim is not None:
            loaded_feature_dim = self._feature_dim_from_model(model)
            if loaded_feature_dim != expected_feature_dim:
                raise RuntimeError(
                    "Loaded MoGe feature dimension does not match configured/known metadata: "
                    f"expected {expected_feature_dim}, loaded {loaded_feature_dim}"
                )
        return model

    def _prepare_images(self, frames: torch.Tensor, teacher: nn.Module) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected frames with shape [N, C, H, W], got {tuple(frames.shape)}")
        input_size = _cfg_get(self.cfg, "input_size", 224)
        frames = frames.to(device=teacher.device, dtype=torch.float32, non_blocking=True)
        if input_size is not None:
            input_size = int(input_size)
            if frames.shape[-2:] != (input_size, input_size):
                frames = F.interpolate(
                    frames,
                    size=(input_size, input_size),
                    mode="bilinear",
                    align_corners=False,
                )

        configured_range = _normalize_frame_value_range(_cfg_get(self.cfg, "frame_value_range", "auto"))
        explicit_range = configured_range in {"0_1", "0-1", "unit", "0_255", "0-255", "uint8"}
        if explicit_range and self._resolved_frame_value_range is not None:
            value_range = self._resolved_frame_value_range
        else:
            frame_min = float(frames.amin().detach().cpu())
            frame_max = float(frames.amax().detach().cpu())
            if configured_range in {"0_1", "0-1", "unit"}:
                if frame_min < -1e-4 or frame_max > 1.0 + 1e-4:
                    raise ValueError(
                        f"depth_teacher_aux expected frames in [0, 1], got min={frame_min:.4f}, max={frame_max:.4f}"
                    )
                value_range = "0_1"
                self._resolved_frame_value_range = value_range
            elif configured_range in {"0_255", "0-255", "uint8"}:
                if frame_min < -1e-3 or frame_max > 255.0 + 1e-3:
                    raise ValueError(
                        f"depth_teacher_aux expected frames in [0, 255], got min={frame_min:.4f}, max={frame_max:.4f}"
                    )
                value_range = "0_255"
                self._resolved_frame_value_range = value_range
            elif configured_range == "auto":
                # Auto must stay per-batch. Caching a first all-black batch as [0, 1]
                # would silently corrupt later uint8 [0, 255] robot frames.
                if frame_min >= -1e-4 and frame_max <= 1.0 + 1e-4:
                    value_range = "0_1"
                elif frame_min >= -1e-3 and frame_max <= 255.0 + 1e-3:
                    value_range = "0_255"
                else:
                    raise ValueError(
                        "depth_teacher_aux could not infer frame value range. "
                        f"Expected [0, 1] or [0, 255], got min={frame_min:.4f}, max={frame_max:.4f}. "
                        "Set `framework.depth_teacher_aux.frame_value_range` explicitly if needed."
                    )
            else:
                raise ValueError(f"Unsupported depth_teacher_aux.frame_value_range: {configured_range}")
            if not self._logged_frame_value_range:
                resolved_label = "[0, 1]" if value_range == "0_1" else "[0, 255]"
                self._log(
                    f"depth_teacher_aux using input frame range {resolved_label} "
                    f"(batch min={frame_min:.4f}, max={frame_max:.4f})"
                )
                self._logged_frame_value_range = True
        if value_range == "0_1":
            images = frames.clamp(0.0, 1.0)
        elif value_range == "0_255":
            images = frames.clamp(0.0, 255.0) / 255.0
        else:
            raise RuntimeError(f"Unsupported cached depth_teacher_aux frame value range: {value_range}")

        return images.to(dtype=teacher.dtype)

    @staticmethod
    def _num_tokens_to_grid(num_tokens: int, height: int, width: int) -> tuple[int, int]:
        aspect_ratio = width / height
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)
        return int(base_h), int(base_w)

    @torch.inference_mode()
    def infer_features(self, frames: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        teacher = self._ensure_model(frames.device)
        images = self._prepare_images(frames, teacher)
        _, _, img_h, img_w = images.shape
        num_tokens = int(_cfg_get(self.cfg, "num_tokens", 256))
        token_h, token_w = self._num_tokens_to_grid(num_tokens, img_h, img_w)

        autocast_ctx = (
            torch.autocast(device_type=images.device.type, enabled=False)
            if images.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_ctx:
            encoder_features, _ = teacher.encoder(images, token_h, token_w, return_class_token=True)
            feature_source = str(_cfg_get(self.cfg, "teacher_feature_source", "neck")).lower()
            if feature_source == "encoder":
                feature_map = encoder_features
            elif feature_source == "neck":
                from moge.utils.geometry_torch import normalized_view_plane_uv

                features = [encoder_features, None, None, None, None]
                aspect_ratio = img_w / img_h
                for level in range(5):
                    uv = normalized_view_plane_uv(
                        width=token_w * 2 ** level,
                        height=token_h * 2 ** level,
                        aspect_ratio=aspect_ratio,
                        dtype=images.dtype,
                        device=images.device,
                    )
                    uv = uv.permute(2, 0, 1).unsqueeze(0).expand(images.shape[0], -1, -1, -1)
                    if features[level] is None:
                        features[level] = uv
                    else:
                        features[level] = torch.cat([features[level], uv], dim=1)
                neck_features = teacher.neck(features)
                feature_level = int(_cfg_get(self.cfg, "teacher_feature_level", 0))
                feature_map = neck_features[feature_level]
            else:
                raise ValueError(f"Unsupported teacher_feature_source: {feature_source}")

        return {"features": feature_map.detach().to(device=frames.device)}


def direct_feature_distillation_loss(
    predictions: torch.Tensor,
    teacher_output: dict[str, torch.Tensor | tuple[int, int]],
    token_grid_hw: tuple[int, int],
    cfg: Any,
    train_step: Optional[int] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """LingBot direct-mode loss: pooled teacher-feature L1 plus feature-similarity L1."""

    if predictions.ndim != 3:
        raise ValueError(f"Expected predictions [N, L, D], got {tuple(predictions.shape)}")
    grid_h, grid_w = int(token_grid_hw[0]), int(token_grid_hw[1])
    expected_tokens = grid_h * grid_w
    if predictions.shape[1] != expected_tokens:
        raise ValueError(
            f"Prediction token count {predictions.shape[1]} does not match Qwen grid {grid_h}x{grid_w}"
        )

    feature_map = teacher_output.get("features")
    if not isinstance(feature_map, torch.Tensor):
        raise RuntimeError("MoGe teacher output did not include feature embeddings")
    if feature_map.ndim != 4:
        raise ValueError(f"Expected teacher features [N, D, H, W], got {tuple(feature_map.shape)}")
    if feature_map.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"Teacher feature batch {feature_map.shape[0]} does not match predictions {predictions.shape[0]}"
        )
    if feature_map.shape[1] != predictions.shape[-1]:
        raise ValueError(
            f"Teacher feature dim {feature_map.shape[1]} does not match direct head dim {predictions.shape[-1]}"
        )

    target = F.adaptive_avg_pool2d(
        feature_map.to(device=predictions.device, dtype=torch.float32),
        (grid_h, grid_w),
    )
    target = target.flatten(2).transpose(1, 2).contiguous()
    pred = predictions.float()

    l1_loss = F.l1_loss(pred, target.detach(), reduction="mean")

    flat_pred = pred.reshape(-1, pred.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    max_tokens = int(_cfg_get(cfg, "similarity_max_tokens", 4096))
    if max_tokens > 0 and flat_pred.shape[0] > max_tokens:
        sample_seed = _cfg_get(cfg, "similarity_sample_seed", None)
        if train_step is not None or sample_seed is not None:
            seed = int(0 if sample_seed is None else sample_seed) + int(0 if train_step is None else train_step)
            generator = torch.Generator(device=flat_pred.device).manual_seed(seed)
            indices = torch.randperm(flat_pred.shape[0], device=flat_pred.device, generator=generator)[:max_tokens]
        else:
            indices = torch.randperm(flat_pred.shape[0], device=flat_pred.device)[:max_tokens]
        flat_pred = flat_pred.index_select(0, indices)
        flat_target = flat_target.index_select(0, indices)

    pred_norm = F.normalize(flat_pred, p=2, dim=-1, eps=1e-6)
    target_norm = F.normalize(flat_target.detach(), p=2, dim=-1, eps=1e-6)
    pred_sim = torch.matmul(pred_norm, pred_norm.transpose(0, 1))
    target_sim = torch.matmul(target_norm, target_norm.transpose(0, 1))
    sim_loss = F.l1_loss(pred_sim, target_sim.detach(), reduction="mean")

    total = (
        float(_cfg_get(cfg, "feature_l1_weight", 1.0)) * l1_loss
        + float(_cfg_get(cfg, "feature_similarity_weight", 1.0)) * sim_loss
    )
    metrics = {
        "depth_teacher_feature_l1_loss": l1_loss.detach(),
        "depth_teacher_feature_similarity_loss": sim_loss.detach(),
    }
    return total, metrics
