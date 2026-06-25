from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from deployment.trossen.pipeline import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NORM_MODE,
    continuous_normalize,
    resize_for_training,
)


REALMAN_CAMERA_ORDER = ("head", "wrist_left", "wrist_right")

REALMAN_ACTION_NAMES = (
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
    "base_linear_x_mps",
    "base_linear_y_mps",
    "base_angular_z_radps",
    "head_joint_1_rad",
    "head_joint_2_rad",
    "lift_height_mm",
)

REALMAN_STATE_NAMES = (
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper_open",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper_open",
    "head_joint_1_rad",
    "head_joint_2_rad",
    "lift_height_mm",
)

REALMAN_ACTION_DIM = len(REALMAN_ACTION_NAMES)
REALMAN_STATE_DIM = len(REALMAN_STATE_NAMES)
REALMAN_POLICY_ACTION_NAMES_NO_BASE = REALMAN_ACTION_NAMES[:16] + REALMAN_ACTION_NAMES[19:22]
REALMAN_POLICY_ACTION_DIM_NO_BASE = len(REALMAN_POLICY_ACTION_NAMES_NO_BASE)
REALMAN_POLICY_ACTION_DIMS = (REALMAN_ACTION_DIM, REALMAN_POLICY_ACTION_DIM_NO_BASE)

_IMAGE_KEY_ALIASES = {
    "head": (
        "observation.images.head",
        "image.head",
        "camera.head",
        "head",
        "base_view",
        "observation.images.base_view",
    ),
    "wrist_left": (
        "observation.images.wrist_left",
        "observation.images.left_wrist",
        "image.wrist_left",
        "image.left_wrist",
        "camera.wrist_left",
        "camera.left_wrist",
        "wrist_left",
        "left_wrist",
    ),
    "wrist_right": (
        "observation.images.wrist_right",
        "observation.images.right_wrist",
        "image.wrist_right",
        "image.right_wrist",
        "camera.wrist_right",
        "camera.right_wrist",
        "wrist_right",
        "right_wrist",
    ),
}

_STATE_VECTOR_KEYS = (
    "source.observation.state",
    "observation.state",
    "state",
)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _lookup_observation_value(observation: dict[str, Any], aliases: Iterable[str]) -> Any:
    for key in aliases:
        if key in observation:
            return observation[key]
    raise KeyError(f"None of the observation keys are present: {list(aliases)}")


def extract_ordered_images(
    observation: dict[str, Any],
    camera_order: Iterable[str] = REALMAN_CAMERA_ORDER,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    missing: list[str] = []
    for camera_name in camera_order:
        aliases = _IMAGE_KEY_ALIASES.get(camera_name, (camera_name,))
        try:
            image = _lookup_observation_value(observation, aliases)
        except KeyError:
            missing.append(camera_name)
            continue
        images.append(resize_for_training(image, image_size=image_size))

    if missing:
        raise KeyError(
            "Observation is missing Realman camera(s) "
            f"{missing}. Expected aliases: "
            f"{ {name: _IMAGE_KEY_ALIASES.get(name, (name,)) for name in missing} }"
        )
    return images


def extract_state_vector(observation: dict[str, Any]) -> np.ndarray:
    for key in _STATE_VECTOR_KEYS:
        if key in observation:
            state = _to_numpy(observation[key]).astype(np.float32, copy=False).reshape(-1)
            if state.shape[0] != REALMAN_STATE_DIM:
                raise ValueError(
                    f"Realman state vector `{key}` has dim {state.shape[0]}, "
                    f"expected {REALMAN_STATE_DIM}."
                )
            return np.ascontiguousarray(state)

    if all(name in observation for name in REALMAN_STATE_NAMES):
        state = np.asarray([observation[name] for name in REALMAN_STATE_NAMES], dtype=np.float32)
        return np.ascontiguousarray(state)

    raise KeyError(
        "Observation is missing Realman state. Provide one of "
        f"{list(_STATE_VECTOR_KEYS)} with dim {REALMAN_STATE_DIM}, or provide all named state fields."
    )


def build_policy_payload(
    observation: dict[str, Any],
    instruction: str,
    camera_order: Iterable[str] = REALMAN_CAMERA_ORDER,
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


def validate_realman_server_metadata(
    metadata: dict[str, Any],
    *,
    expected_action_dim: int | Iterable[int] | None = REALMAN_POLICY_ACTION_DIMS,
    expected_state_dim: int = REALMAN_STATE_DIM,
    expected_action_type: str | None = None,
    expected_camera_order: Iterable[str] = REALMAN_CAMERA_ORDER,
) -> list[str]:
    warnings: list[str] = []
    if expected_action_type is not None and metadata.get("action_type") != expected_action_type:
        warnings.append(
            f"Server reports action_type={metadata.get('action_type')!r}, "
            f"expected {expected_action_type!r}."
        )
    action_dim_ok = True
    if expected_action_dim is not None:
        reported_action_dim = metadata.get("action_dim")
        try:
            reported_action_dim = int(reported_action_dim)
        except (TypeError, ValueError):
            pass
        if isinstance(expected_action_dim, int):
            action_dim_ok = reported_action_dim == expected_action_dim
            expected_action_dim_text = str(expected_action_dim)
        else:
            expected_action_dims = tuple(int(dim) for dim in expected_action_dim)
            action_dim_ok = reported_action_dim in expected_action_dims
            expected_action_dim_text = str(list(expected_action_dims))
    if not action_dim_ok:
        warnings.append(
            f"Server reports action_dim={metadata.get('action_dim')}, expected {expected_action_dim_text}."
        )
    reported_state_dim = metadata.get("state_dim")
    try:
        reported_state_dim = int(reported_state_dim)
    except (TypeError, ValueError):
        pass
    if reported_state_dim != expected_state_dim:
        warnings.append(
            f"Server reports state_dim={metadata.get('state_dim')}, expected {expected_state_dim}."
        )

    camera_hint = metadata.get("camera_order_hint")
    if camera_hint is not None and list(camera_hint) != list(expected_camera_order):
        warnings.append(
            f"Server camera hint {camera_hint} does not match Realman camera order "
            f"{list(expected_camera_order)}."
        )
    return warnings


def expand_policy_action_to_robot_action(action: np.ndarray) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape[-1] == REALMAN_ACTION_DIM:
        return np.ascontiguousarray(array)
    if array.shape[-1] != REALMAN_POLICY_ACTION_DIM_NO_BASE:
        raise ValueError(
            f"Realman policy action dim is {array.shape[-1]}, expected "
            f"{REALMAN_POLICY_ACTION_DIM_NO_BASE} no-base or {REALMAN_ACTION_DIM} legacy."
        )

    expanded = np.zeros((*array.shape[:-1], REALMAN_ACTION_DIM), dtype=np.float32)
    expanded[..., :16] = array[..., :16]
    expanded[..., 19:22] = array[..., 16:19]
    return np.ascontiguousarray(expanded)


def split_action_vector(action: np.ndarray) -> dict[str, Any]:
    vector = expand_policy_action_to_robot_action(action).reshape(-1)
    if vector.shape[0] != REALMAN_ACTION_DIM:
        raise ValueError(f"Realman action dim is {vector.shape[0]}, expected {REALMAN_ACTION_DIM}.")
    return {
        "left_arm_joints": vector[0:7].copy(),
        "left_gripper": float(vector[7]),
        "right_arm_joints": vector[8:15].copy(),
        "right_gripper": float(vector[15]),
        "base_velocity": {
            "linear_x_mps": float(vector[16]),
            "linear_y_mps": float(vector[17]),
            "angular_z_radps": float(vector[18]),
        },
        "head_joints": vector[19:21].copy(),
        "lift_height_mm": float(vector[21]),
        "vector": vector.copy(),
        "names": REALMAN_ACTION_NAMES,
    }


def split_action_chunk(actions: np.ndarray) -> list[dict[str, Any]]:
    chunk = expand_policy_action_to_robot_action(actions)
    if chunk.ndim != 2:
        raise ValueError(f"Expected Realman action chunk [T, {REALMAN_ACTION_DIM}], got {chunk.shape}.")
    return [split_action_vector(action) for action in chunk]


def action_summary(action: np.ndarray) -> dict[str, float]:
    split = split_action_vector(action)
    return {
        "left_arm_l2": float(np.linalg.norm(split["left_arm_joints"])),
        "right_arm_l2": float(np.linalg.norm(split["right_arm_joints"])),
        "left_gripper": float(split["left_gripper"]),
        "right_gripper": float(split["right_gripper"]),
        "base_linear_x_mps": float(split["base_velocity"]["linear_x_mps"]),
        "base_linear_y_mps": float(split["base_velocity"]["linear_y_mps"]),
        "base_angular_z_radps": float(split["base_velocity"]["angular_z_radps"]),
        "head_l2": float(np.linalg.norm(split["head_joints"])),
        "lift_height_mm": float(split["lift_height_mm"]),
    }


def load_observation_npz(path: str | Path) -> dict[str, Any]:
    npz_path = Path(path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def json_safe(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record)) + "\n")
