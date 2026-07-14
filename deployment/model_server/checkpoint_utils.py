from __future__ import annotations

from pathlib import Path
from typing import Any

from deployment.realman.pipeline import (
    DEFAULT_QWEN_FRAME_SIZE,
    QWEN_TENSOR_PAYLOAD_KEY,
    REALMAN_ACTION_DIM,
    REALMAN_ACTION_NAMES,
    REALMAN_CAMERA_ORDER,
    REALMAN_POLICY_ACTION_DIMS,
    REALMAN_STATE_DIM,
    REALMAN_STATE_NAMES,
    realman_omitted_robot_action_indices,
    realman_policy_action_names,
)


_CAMERA_ORDER_HINTS = {
    "trossen_subtask_combined": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    "ogrealman_source_v3": ["head", "wrist_left", "wrist_right"],
    "ogrealman_source_no_base_v3": ["head", "wrist_left", "wrist_right"],
    "ogrealman_source_no_base_human_labelled_cloud_v3": ["head", "wrist_left", "wrist_right"],
    "magna_source_no_base_interventions_v3": ["head", "wrist_left", "wrist_right"],
    "magna_source_no_base_no_lift_interventions_v3": ["head", "wrist_left", "wrist_right"],
    "ogrealman_canonical_v3": ["head", "wrist_left", "wrist_right"],
}


def _cfg_get(obj: Any, *keys: str, default: Any = None) -> Any:
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return default if current is None else current


def _infer_norm_mode_hints(policy) -> dict[str, Any]:
    data_mix = _cfg_get(policy.config, "datasets", "vla_data", "data_mix")
    action_horizon = _cfg_get(policy.config, "framework", "action_model", "action_horizon", default=1)
    video_horizon = _cfg_get(policy.config, "framework", "vj2_model", "num_frames", default=1)

    try:
        from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
        from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform
    except Exception:
        return {}

    mixture_spec = DATASET_NAMED_MIXTURES.get(data_mix)
    if not mixture_spec:
        return {}

    robot_types = {entry[2] for entry in mixture_spec if len(entry) >= 3}
    if len(robot_types) != 1:
        return {}

    robot_type = next(iter(robot_types))
    data_config_cls = ROBOT_TYPE_CONFIG_MAP.get(robot_type)
    if data_config_cls is None:
        return {"robot_type": robot_type}

    try:
        try:
            data_config = data_config_cls(
                observation_indices=list(range(max(int(video_horizon), 1))),
                action_indices=list(range(max(int(action_horizon), 1))),
            )
        except TypeError:
            data_config = data_config_cls()
        transform = data_config.transform()
    except Exception:
        return {"robot_type": robot_type}

    state_norm_modes_by_key: dict[str, str] = {}
    action_norm_modes_by_key: dict[str, str] = {}
    for sub_transform in getattr(transform, "transforms", []):
        if not isinstance(sub_transform, StateActionTransform):
            continue
        for key, mode in sub_transform.normalization_modes.items():
            if key.startswith("state."):
                state_norm_modes_by_key[key] = mode
            elif key.startswith("action."):
                action_norm_modes_by_key[key] = mode

    unique_state_modes = sorted(set(state_norm_modes_by_key.values()))
    unique_action_modes = sorted(set(action_norm_modes_by_key.values()))
    return {
        "robot_type": robot_type,
        "state_norm_modes_by_key": state_norm_modes_by_key,
        "action_norm_modes_by_key": action_norm_modes_by_key,
        "default_state_norm_mode": unique_state_modes[0] if len(unique_state_modes) == 1 else None,
        "default_action_norm_mode": unique_action_modes[0] if len(unique_action_modes) == 1 else None,
    }


def resolve_policy_checkpoint(checkpoint_path: str | Path) -> Path:
    """Resolve a user-supplied checkpoint path to a loadable model artifact."""
    raw_path = Path(checkpoint_path).expanduser().resolve()
    supported_suffixes = {".pt", ".safetensors"}

    if raw_path.is_file():
        if raw_path.suffix not in supported_suffixes:
            raise ValueError(
                f"Expected a `.pt` or `.safetensors` policy artifact, but got `{raw_path}`. "
                "Pass `final_model/`, an interval checkpoint directory, the run root, "
                "or a concrete model artifact."
            )
        return raw_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Checkpoint path `{raw_path}` does not exist.")

    candidates = [
        raw_path / "pytorch_model.pt",
        raw_path / "model.safetensors",
        raw_path / "final_model" / "pytorch_model.pt",
        raw_path / "final_model" / "model.safetensors",
    ]

    recursive_hits = sorted(
        path
        for path in raw_path.glob("**/pytorch_model.pt")
        if len(path.relative_to(raw_path).parts) <= 3
    )
    for hit in recursive_hits:
        if hit not in candidates:
            candidates.append(hit)

    safetensor_hits = sorted(
        path
        for path in raw_path.glob("**/model.safetensors")
        if len(path.relative_to(raw_path).parts) <= 3
    )
    for hit in safetensor_hits:
        if hit not in candidates:
            candidates.append(hit)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve a `.pt` or `.safetensors` model artifact under `{raw_path}`. "
        "Pass the run root, `final_model/`, an interval checkpoint directory, "
        "or the concrete model artifact."
    )


def build_policy_metadata(policy, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    data_mix = _cfg_get(policy.config, "datasets", "vla_data", "data_mix")
    norm_stats = getattr(policy, "norm_stats", {}) or {}
    action_stats_by_key = {
        key: value["action"]
        for key, value in norm_stats.items()
        if isinstance(value, dict) and "action" in value
    }
    state_stats_by_key = {
        key: value["state"]
        for key, value in norm_stats.items()
        if isinstance(value, dict) and "state" in value
    }
    available_unnorm_keys = sorted(action_stats_by_key)
    default_unnorm_key = available_unnorm_keys[0] if len(available_unnorm_keys) == 1 else None

    action_horizon = _cfg_get(policy.config, "framework", "action_model", "action_horizon")
    if action_horizon is None:
        future_window = _cfg_get(policy.config, "framework", "action_model", "future_action_window_size", default=0)
        action_horizon = int(future_window) + 1

    norm_mode_hints = _infer_norm_mode_hints(policy)

    metadata = {
        "env": "real_robot",
        "checkpoint_path": str(checkpoint_path),
        "run_id": _cfg_get(policy.config, "run_id"),
        "framework_name": _cfg_get(policy.config, "framework", "name"),
        "data_mix": data_mix,
        "action_type": _cfg_get(policy.config, "datasets", "vla_data", "action_type"),
        "resolution_size": _cfg_get(policy.config, "datasets", "vla_data", "resolution_size"),
        "video_resolution_size": _cfg_get(policy.config, "datasets", "vla_data", "video_resolution_size"),
        "with_state": bool(_cfg_get(policy.config, "datasets", "vla_data", "with_state", default=False)),
        "action_dim": _cfg_get(policy.config, "framework", "action_model", "action_dim"),
        "state_dim": _cfg_get(policy.config, "framework", "action_model", "state_dim"),
        "action_horizon": action_horizon,
        "future_action_window_size": _cfg_get(policy.config, "framework", "action_model", "future_action_window_size"),
        "num_inference_timesteps": _cfg_get(policy.config, "framework", "action_model", "num_inference_timesteps"),
        "available_unnorm_keys": available_unnorm_keys,
        "default_unnorm_key": default_unnorm_key,
        "action_stats_by_key": action_stats_by_key,
        "state_stats_by_key": state_stats_by_key,
        "camera_order_hint": _CAMERA_ORDER_HINTS.get(data_mix),
        **norm_mode_hints,
    }
    camera_order_hint = tuple(metadata.get("camera_order_hint") or ())
    try:
        action_dim = int(metadata.get("action_dim"))
        state_dim = int(metadata.get("state_dim"))
    except (TypeError, ValueError):
        action_dim = state_dim = -1
    if (
        camera_order_hint == REALMAN_CAMERA_ORDER
        and action_dim in REALMAN_POLICY_ACTION_DIMS
        and state_dim == REALMAN_STATE_DIM
    ):
        omitted_indices = realman_omitted_robot_action_indices(action_dim)
        qwen_frame_size = metadata.get("video_resolution_size") or DEFAULT_QWEN_FRAME_SIZE
        qwen_resolution_size = metadata.get("resolution_size")
        metadata.update(
            {
                "policy_action_names": list(realman_policy_action_names(action_dim)),
                "state_names": list(REALMAN_STATE_NAMES),
                "robot_action_dim": REALMAN_ACTION_DIM,
                "robot_action_names": list(REALMAN_ACTION_NAMES),
                "realman_action_contract": {
                    "version": 1,
                    "policy_action_dim": action_dim,
                    "robot_action_dim": REALMAN_ACTION_DIM,
                    "omitted_robot_action_indices": list(omitted_indices),
                    "base_velocity_source": "policy" if action_dim == REALMAN_ACTION_DIM else "zero",
                    "lift_source": "measured_state" if 21 in omitted_indices else "policy",
                },
                "realman_input_contract": {
                    "version": 1,
                    "payload_key": QWEN_TENSOR_PAYLOAD_KEY,
                    "camera_order": list(REALMAN_CAMERA_ORDER),
                    "frame_shape": [len(REALMAN_CAMERA_ORDER), int(qwen_frame_size), int(qwen_frame_size), 3],
                    "frame_size": int(qwen_frame_size),
                    "frame_dtype": "uint8",
                    "color_space": "RGB",
                    "transport_encoding": "msgpack_ndarray",
                    "client_resize": "opencv_inter_linear",
                    "model_preprocess": "qwen_tensor_fast_path",
                    "model_resolution_size": int(qwen_resolution_size) if qwen_resolution_size is not None else None,
                    "state_shape": [1, 1, state_dim],
                    "state_dtype": "float32",
                    "state_normalized": True,
                },
            }
        )
    return metadata
