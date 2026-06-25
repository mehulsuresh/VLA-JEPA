from __future__ import annotations

import sys
import os
import hashlib
import math
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


_MOGE2_NORMAL_NECK_DIMS_BY_MODEL = {
    "moge-2-vits-normal": (384, 256, 128, 64, 32),
    "moge-2-vitb-normal": (768, 256, 128, 64, 32),
    "moge-2-vitl-normal": (1024, 256, 128, 64, 32),
    "Ruicheng/moge-2-vits-normal": (384, 256, 128, 64, 32),
    "Ruicheng/moge-2-vitb-normal": (768, 256, 128, 64, 32),
    "Ruicheng/moge-2-vitl-normal": (1024, 256, 128, 64, 32),
}


class _GeometryFeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: float = 1.0, dropout: float = 0.0):
        super().__init__()
        inner_dim = max(int(dim * ff_mult), dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GeometryPerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 32, heads: int = 4):
        super().__init__()
        self.scale = float(dim_head) ** -0.5
        self.dim_head = int(dim_head)
        self.heads = int(heads)
        inner_dim = self.dim_head * self.heads

        self.norm_context = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.heads, self.dim_head).transpose(1, 2)

    def forward(self, context: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        context = self.norm_context(context)
        queries = self.norm_queries(queries)

        q = self._reshape_heads(self.to_q(queries))
        kv_input = torch.cat([context, queries], dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        # The sqrt-sqrt scaling mirrors LingBot/OpenFlamingo's fp16-stable attention.
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weights = (q * scale) @ (k * scale).transpose(-2, -1)
        weights = torch.softmax(weights.float(), dim=-1).to(dtype=weights.dtype)
        out = weights @ v
        out = out.transpose(1, 2).reshape(queries.shape[0], queries.shape[1], -1)
        return self.to_out(out)


class QueryGeometryTeacherHead(nn.Module):
    """LingBot-style query head: resample VLM geometry tokens into teacher tokens."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        num_output_tokens: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        dim_head: int = 32,
        ff_mult: float = 1.0,
        dropout: float = 0.0,
        final_init_std: Optional[float] = None,
    ):
        super().__init__()
        self.output_queries = nn.Parameter(
            torch.randn(1, int(num_output_tokens), int(hidden_size)) / (float(hidden_size) ** 0.5)
        )
        self.context_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _GeometryPerceiverAttention(
                            dim=hidden_size,
                            dim_head=int(dim_head),
                            heads=int(num_heads),
                        ),
                        _GeometryFeedForward(
                            dim=hidden_size,
                            ff_mult=float(ff_mult),
                            dropout=float(dropout),
                        ),
                    ]
                )
                for _ in range(int(num_layers))
            ]
        )
        self.proj_out = nn.Linear(hidden_size, output_size)
        self.norm_out = nn.LayerNorm(output_size)
        if final_init_std is not None:
            final_init_std = float(final_init_std)
            if final_init_std <= 0.0:
                nn.init.zeros_(self.proj_out.weight)
            else:
                nn.init.normal_(self.proj_out.weight, mean=0.0, std=final_init_std)
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        context = self.context_proj(context_tokens)
        queries = self.output_queries.repeat(context_tokens.shape[0], 1, 1)
        queries = queries.to(device=context_tokens.device, dtype=context_tokens.dtype)
        queries = self.query_proj(queries)

        for attn, ff in self.layers:
            queries = attn(context, queries) + queries
            queries = ff(queries) + queries

        return self.norm_out(self.proj_out(queries))


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

    @staticmethod
    def _known_moge2_neck_dims(model_name: str) -> Optional[tuple[int, ...]]:
        model_name = str(model_name).rstrip("/")
        candidates = [model_name, Path(model_name).name]
        if model_name.endswith(".pt") or model_name.endswith(".safetensors"):
            candidates.append(Path(model_name).stem)
        for candidate in candidates:
            dims = _MOGE2_NORMAL_NECK_DIMS_BY_MODEL.get(candidate)
            if dims is not None:
                return dims
        return None

    def _feature_dim_without_model(self) -> Optional[int]:
        configured_feature_dim = self._configured_feature_dim()
        if configured_feature_dim is not None:
            return configured_feature_dim
        model_name = str(_cfg_get(self.cfg, "teacher_model", "Ruicheng/moge-2-vitl-normal")).rstrip("/")
        neck_dims = self._known_moge2_neck_dims(model_name)
        if neck_dims is None:
            return None
        feature_source = str(_cfg_get(self.cfg, "teacher_feature_source", "neck")).lower()
        if feature_source == "encoder":
            return neck_dims[0]
        if feature_source == "neck":
            feature_level = int(_cfg_get(self.cfg, "teacher_feature_level", 0))
            if 0 <= feature_level < len(neck_dims):
                return neck_dims[feature_level]
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
                    "a known Ruicheng/moge-2-vits/vitb/vitl-normal model, or enable eager_load_on_cpu."
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


def _tokens_to_grid(num_tokens: int, height: int, width: int) -> tuple[int, int]:
    aspect_ratio = float(width) / max(float(height), 1.0)
    grid_h = max(round((float(num_tokens) / aspect_ratio) ** 0.5), 1)
    grid_w = max(round(float(num_tokens) / float(grid_h)), 1)
    if grid_h * grid_w != int(num_tokens):
        grid_h = max(int(round(float(num_tokens) ** 0.5)), 1)
        grid_w = max(int(num_tokens) // grid_h, 1)
    if grid_h * grid_w != int(num_tokens):
        raise ValueError(f"Cannot map {num_tokens} geometry tokens to a rectangular feature grid")
    return int(grid_h), int(grid_w)


def query_feature_distillation_loss(
    predictions: torch.Tensor,
    teacher_output: dict[str, torch.Tensor | tuple[int, int]],
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """LingBot query-mode loss: SmoothL1 over resampled teacher feature tokens."""

    if predictions.ndim != 3:
        raise ValueError(f"Expected predictions [N, L, D], got {tuple(predictions.shape)}")
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
            f"Teacher feature dim {feature_map.shape[1]} does not match query head dim {predictions.shape[-1]}"
        )

    grid_h, grid_w = _tokens_to_grid(
        int(predictions.shape[1]),
        int(feature_map.shape[-2]),
        int(feature_map.shape[-1]),
    )
    target = F.adaptive_avg_pool2d(
        feature_map.to(device=predictions.device, dtype=torch.float32),
        (grid_h, grid_w),
    )
    target = target.flatten(2).transpose(1, 2).contiguous()
    if target.shape[1] != predictions.shape[1]:
        raise ValueError(
            f"Teacher target token count {target.shape[1]} does not match predictions {predictions.shape[1]}"
        )

    beta = float(_cfg_get(cfg, "query_smooth_l1_beta", 1.0))
    if beta > 0.0:
        loss = F.smooth_l1_loss(predictions.float(), target.detach(), reduction="mean", beta=beta)
    else:
        loss = F.l1_loss(predictions.float(), target.detach(), reduction="mean")
    metrics = {
        "depth_teacher_query_smooth_l1_loss": loss.detach(),
        "depth_teacher_query_tokens": torch.tensor(
            float(predictions.shape[1]),
            device=predictions.device,
            dtype=loss.dtype,
        ),
    }
    return loss, metrics
