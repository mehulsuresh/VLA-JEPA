from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


DEFAULT_CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_YONDU_LEROBOT_ROOT = Path(__file__).resolve().parents[4] / "yondu-trossen-lerobot"
DEFAULT_NORM_MODE = "q99"
DEFAULT_RESAMPLE = Image.Resampling.BICUBIC


def _observation_key(camera_name: str) -> str:
    if camera_name.startswith("observation.images."):
        return camera_name
    return f"observation.images.{camera_name}"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def coerce_rgb_uint8(image: Any) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"Expected an image with 3 dimensions, got shape {array.shape}")

    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image data, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(array.max()) if array.size else 0.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


def resize_for_training(image: Any, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    pil_image = Image.fromarray(coerce_rgb_uint8(image), mode="RGB")
    resized = pil_image.resize((image_size, image_size), resample=DEFAULT_RESAMPLE)
    return np.asarray(resized, dtype=np.uint8)


def extract_ordered_images(
    observation: dict[str, Any],
    camera_order: Iterable[str] = DEFAULT_CAMERA_ORDER,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> list[np.ndarray]:
    images = []
    missing_keys = []
    for camera_name in camera_order:
        obs_key = _observation_key(camera_name)
        if obs_key not in observation:
            missing_keys.append(obs_key)
            continue
        images.append(resize_for_training(observation[obs_key], image_size=image_size))

    if missing_keys:
        raise KeyError(f"Observation is missing required camera keys: {missing_keys}")
    return images


def extract_state_vector(observation: dict[str, Any]) -> np.ndarray:
    if "observation.state" not in observation:
        raise KeyError("Observation is missing `observation.state`.")
    state = _to_numpy(observation["observation.state"]).astype(np.float32, copy=False).reshape(-1)
    return np.ascontiguousarray(state)


def build_policy_payload(
    observation: dict[str, Any],
    instruction: str,
    camera_order: Iterable[str] = DEFAULT_CAMERA_ORDER,
    image_size: int = DEFAULT_IMAGE_SIZE,
    state_stats: dict[str, Any] | None = None,
    state_norm_mode: str | None = None,
) -> dict[str, Any]:
    state = extract_state_vector(observation)
    if state_stats is not None:
        state = continuous_normalize(state, state_stats, mode=state_norm_mode or DEFAULT_NORM_MODE)
    images = extract_ordered_images(observation, camera_order=camera_order, image_size=image_size)
    return {
        "batch_images": [images],
        "instructions": [instruction],
        "state": state[None, None, :],
    }


def resolve_action_stats(metadata: dict[str, Any], unnorm_key: str | None = None) -> dict[str, Any]:
    action_stats_by_key = metadata.get("action_stats_by_key") or {}
    if not action_stats_by_key:
        raise KeyError("Server metadata does not include `action_stats_by_key`.")

    if unnorm_key is None:
        unnorm_key = metadata.get("default_unnorm_key")
    if unnorm_key is None:
        available = sorted(action_stats_by_key)
        raise ValueError(
            "Server exposes multiple normalization keys. Pass `--unnorm-key` from: "
            f"{available}"
        )
    if unnorm_key not in action_stats_by_key:
        raise KeyError(
            f"Normalization key `{unnorm_key}` is not available. "
            f"Choices: {sorted(action_stats_by_key)}"
        )
    return action_stats_by_key[unnorm_key]


def resolve_state_stats(metadata: dict[str, Any], unnorm_key: str | None = None) -> dict[str, Any]:
    state_stats_by_key = metadata.get("state_stats_by_key") or {}
    if not state_stats_by_key:
        raise KeyError("Server metadata does not include `state_stats_by_key`.")

    if unnorm_key is None:
        unnorm_key = metadata.get("default_unnorm_key")
    if unnorm_key is None:
        available = sorted(state_stats_by_key)
        raise ValueError(
            "Server exposes multiple state normalization keys. Pass `--unnorm-key` from: "
            f"{available}"
        )
    if unnorm_key not in state_stats_by_key:
        raise KeyError(
            f"State normalization key `{unnorm_key}` is not available. "
            f"Choices: {sorted(state_stats_by_key)}"
        )
    return state_stats_by_key[unnorm_key]


def _ensure_last_dim(values: np.ndarray, reference_dim: int, label: str) -> None:
    if values.shape[-1] != reference_dim:
        raise ValueError(
            f"{label} dimension mismatch: got last dim {values.shape[-1]}, expected {reference_dim}"
        )


def continuous_normalize(values: np.ndarray, stats: dict[str, Any], *, mode: str = DEFAULT_NORM_MODE) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)

    if mode == "min_max":
        min_value = np.asarray(stats["min"], dtype=np.float32)
        max_value = np.asarray(stats["max"], dtype=np.float32)
        _ensure_last_dim(array, min_value.shape[-1], "Normalization input")
        mask = min_value != max_value
        normalized = np.zeros_like(array, dtype=np.float32)
        normalized[..., mask] = (array[..., mask] - min_value[..., mask]) / (
            max_value[..., mask] - min_value[..., mask]
        )
        normalized[..., mask] = 2.0 * normalized[..., mask] - 1.0
        normalized[..., ~mask] = 0.0
        return normalized.astype(np.float32, copy=False)

    if mode == "q99":
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        _ensure_last_dim(array, q01.shape[-1], "Normalization input")
        mask = q01 != q99
        normalized = np.zeros_like(array, dtype=np.float32)
        normalized[..., mask] = (array[..., mask] - q01[..., mask]) / (
            q99[..., mask] - q01[..., mask]
        )
        normalized[..., mask] = 2.0 * normalized[..., mask] - 1.0
        normalized[..., ~mask] = array[..., ~mask]
        return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)

    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        _ensure_last_dim(array, mean.shape[-1], "Normalization input")
        mask = std != 0
        normalized = np.zeros_like(array, dtype=np.float32)
        normalized[..., mask] = (array[..., mask] - mean[..., mask]) / std[..., mask]
        normalized[..., ~mask] = array[..., ~mask]
        return normalized.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported normalization mode `{mode}`")


def continuous_unnormalize(values: np.ndarray, stats: dict[str, Any], *, mode: str = DEFAULT_NORM_MODE) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)

    if mode == "min_max":
        min_value = np.asarray(stats["min"], dtype=np.float32)
        max_value = np.asarray(stats["max"], dtype=np.float32)
        _ensure_last_dim(array, min_value.shape[-1], "Unnormalization input")
        return ((array + 1.0) / 2.0 * (max_value - min_value) + min_value).astype(np.float32, copy=False)

    if mode == "q99":
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        _ensure_last_dim(array, q01.shape[-1], "Unnormalization input")
        clipped = np.clip(array, -1.0, 1.0)
        mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)
        return np.where(
            mask,
            0.5 * (clipped + 1.0) * (q99 - q01) + q01,
            clipped,
        ).astype(np.float32, copy=False)

    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        _ensure_last_dim(array, mean.shape[-1], "Unnormalization input")
        return (array * std + mean).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported normalization mode `{mode}`")


def continuous_minmax_unnormalize(normalized_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    return continuous_unnormalize(normalized_actions, action_stats, mode="q99")


def resolve_norm_mode(
    metadata: dict[str, Any],
    target: str,
    explicit_mode: str | None = None,
) -> str:
    if explicit_mode is not None and explicit_mode != "auto":
        return explicit_mode

    if target == "action":
        default_mode = metadata.get("default_action_norm_mode")
    elif target == "state":
        default_mode = metadata.get("default_state_norm_mode")
    else:
        raise ValueError(f"Unknown norm target `{target}`")

    if isinstance(default_mode, str) and default_mode:
        return default_mode
    return DEFAULT_NORM_MODE


def compute_action_state_stats_closeness(
    action_stats: dict[str, Any],
    state_stats: dict[str, Any] | None = None,
) -> float | None:
    if state_stats is None:
        return None

    try:
        action_q01 = np.asarray(action_stats["q01"], dtype=np.float32)
        action_q99 = np.asarray(action_stats["q99"], dtype=np.float32)
        state_q01 = np.asarray(state_stats["q01"], dtype=np.float32)
        state_q99 = np.asarray(state_stats["q99"], dtype=np.float32)
    except KeyError:
        return None

    if (
        action_q01.shape != action_q99.shape
        or state_q01.shape != state_q99.shape
        or action_q01.shape != state_q01.shape
    ):
        return None

    scale = np.maximum(np.abs(action_q99 - action_q01), 1.0)
    return float(
        np.mean(
            np.abs(action_q01 - state_q01) / scale
            + np.abs(action_q99 - state_q99) / scale
        )
        / 2.0
    )


def infer_action_mode_from_stats(
    action_stats: dict[str, Any],
    state_stats: dict[str, Any] | None = None,
    *,
    closeness_threshold: float = 0.05,
) -> str:
    closeness = compute_action_state_stats_closeness(action_stats, state_stats)
    if closeness is None:
        return "delta_qpos"
    return "absolute_qpos" if closeness <= closeness_threshold else "delta_qpos"


def compute_absolute_goal(
    current_state: np.ndarray,
    delta_action: np.ndarray,
    *,
    action_scale: float = 1.0,
    delta_clip: float | None = None,
) -> np.ndarray:
    current = np.asarray(current_state, dtype=np.float32).reshape(-1)
    delta = np.asarray(delta_action, dtype=np.float32).reshape(-1)
    if current.shape != delta.shape:
        raise ValueError(f"State/action shape mismatch: {current.shape} vs {delta.shape}")
    if delta_clip is not None:
        delta = np.clip(delta, -float(delta_clip), float(delta_clip))
    return current + float(action_scale) * delta


def compute_absolute_goal_chunk(
    current_state: np.ndarray,
    delta_actions: np.ndarray,
    *,
    chunk_size: int | None = None,
    action_scale: float = 1.0,
    delta_clip: float | None = None,
) -> np.ndarray:
    current = np.asarray(current_state, dtype=np.float32).reshape(-1)
    deltas = np.asarray(delta_actions, dtype=np.float32)
    if deltas.ndim != 2:
        raise ValueError(f"Expected delta action chunk with shape [T, D], got {deltas.shape}")
    if deltas.shape[-1] != current.shape[-1]:
        raise ValueError(f"State/action shape mismatch: {current.shape[-1]} vs {deltas.shape[-1]}")
    if chunk_size is not None:
        deltas = deltas[: int(chunk_size)]
    if delta_clip is not None:
        deltas = np.clip(deltas, -float(delta_clip), float(delta_clip))
    scaled = float(action_scale) * deltas
    return current[None, :] + np.cumsum(scaled, axis=0)


def validate_server_metadata(
    metadata: dict[str, Any],
    *,
    expected_action_type: str = "delta_qpos",
    expected_camera_order: Iterable[str] = DEFAULT_CAMERA_ORDER,
    expected_state_dim: int | None = 14,
    expected_action_dim: int | None = 14,
) -> list[str]:
    warnings = []
    if metadata.get("action_type") != expected_action_type:
        warnings.append(
            f"Server reports action_type={metadata.get('action_type')!r}, expected {expected_action_type!r}."
        )
    if expected_state_dim is not None and metadata.get("state_dim") != expected_state_dim:
        warnings.append(
            f"Server reports state_dim={metadata.get('state_dim')}, expected {expected_state_dim}."
        )
    if expected_action_dim is not None and metadata.get("action_dim") != expected_action_dim:
        warnings.append(
            f"Server reports action_dim={metadata.get('action_dim')}, expected {expected_action_dim}."
        )

    camera_hint = metadata.get("camera_order_hint")
    if camera_hint is not None and list(camera_hint) != list(expected_camera_order):
        warnings.append(
            f"Server camera hint {camera_hint} does not match runner camera order {list(expected_camera_order)}."
        )
    return warnings


def resolve_yondu_root(path: str | Path | None) -> Path:
    if path is None:
        path = DEFAULT_YONDU_LEROBOT_ROOT
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Yondu LeRobot root `{root}` does not exist. "
            "Pass `--yondu-lerobot-root` or set `YONDU_TROSSEN_LEROBOT_ROOT`."
        )
    return root


def add_to_syspath(import_root: Path) -> None:
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)


def build_stationary_robot_config(
    *,
    yondu_lerobot_root: str | Path | None = None,
    max_relative_target: float | None = None,
    mock: bool = False,
    connect_leaders: bool = False,
    camera_order: Iterable[str] = DEFAULT_CAMERA_ORDER,
    left_leader_ip: str | None = None,
    right_leader_ip: str | None = None,
    left_follower_ip: str | None = None,
    right_follower_ip: str | None = None,
    cam_high_serial: int | None = None,
    cam_left_wrist_serial: int | None = None,
    cam_right_wrist_serial: int | None = None,
):
    yondu_root = resolve_yondu_root(yondu_lerobot_root)
    add_to_syspath(yondu_root)

    from lerobot.common.robot_devices.robots.configs import TrossenAIStationaryRobotConfig

    cfg = TrossenAIStationaryRobotConfig(mock=mock)
    cfg.max_relative_target = max_relative_target

    if not connect_leaders:
        cfg.leader_arms = {}

    cfg.cameras = {
        camera_name: deepcopy(cfg.cameras[camera_name])
        for camera_name in camera_order
    }

    if left_leader_ip is not None and "left" in cfg.leader_arms:
        cfg.leader_arms["left"].ip = left_leader_ip
    if right_leader_ip is not None and "right" in cfg.leader_arms:
        cfg.leader_arms["right"].ip = right_leader_ip
    if left_follower_ip is not None:
        cfg.follower_arms["left"].ip = left_follower_ip
    if right_follower_ip is not None:
        cfg.follower_arms["right"].ip = right_follower_ip
    if cam_high_serial is not None:
        cfg.cameras["cam_high"].serial_number = int(cam_high_serial)
    if cam_left_wrist_serial is not None:
        cfg.cameras["cam_left_wrist"].serial_number = int(cam_left_wrist_serial)
    if cam_right_wrist_serial is not None:
        cfg.cameras["cam_right_wrist"].serial_number = int(cam_right_wrist_serial)

    return cfg


def build_rollout_record(
    *,
    step_index: int,
    replan_index: int | None = None,
    chunk_offset: int | None = None,
    action_mode: str,
    instruction: str,
    latency_ms: float,
    current_state: np.ndarray,
    policy_action: np.ndarray,
    goal_action: np.ndarray,
    sent_action: np.ndarray | None = None,
    normalized_chunk: np.ndarray,
) -> dict[str, Any]:
    record = {
        "step_index": int(step_index),
        "replan_index": None if replan_index is None else int(replan_index),
        "chunk_offset": None if chunk_offset is None else int(chunk_offset),
        "action_mode": str(action_mode),
        "instruction": instruction,
        "latency_ms": float(latency_ms),
        "current_state": np.asarray(current_state, dtype=np.float32).tolist(),
        "policy_action": np.asarray(policy_action, dtype=np.float32).tolist(),
        "goal_action": np.asarray(goal_action, dtype=np.float32).tolist(),
        "sent_action": None if sent_action is None else np.asarray(sent_action, dtype=np.float32).tolist(),
        "normalized_chunk": np.asarray(normalized_chunk, dtype=np.float32).tolist(),
    }
    if action_mode == "delta_qpos":
        record["delta_action"] = np.asarray(policy_action, dtype=np.float32).tolist()
    else:
        record["delta_action"] = None
    return record


def save_rollout_record(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
