from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from deployment.trossen.pipeline import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NORM_MODE,
    coerce_rgb_uint8,
    continuous_normalize,
    resize_for_training,
)


REALMAN_CAMERA_ORDER = ("head", "wrist_left", "wrist_right")
DEFAULT_QWEN_FRAME_SIZE = 384
QWEN_TENSOR_PAYLOAD_KEY = "qwen_frames"

MAGNA_DEFAULT_INSTRUCTION = (
    "reach into the bin, lift the chain, put it in the jig, then remove it from "
    "the jig and put it in the other bin"
)

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
REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT = REALMAN_ACTION_NAMES[:16] + REALMAN_ACTION_NAMES[19:21]
REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT = len(REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT)
REALMAN_POLICY_ACTION_DIMS = (
    REALMAN_ACTION_DIM,
    REALMAN_POLICY_ACTION_DIM_NO_BASE,
    REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT,
)


def realman_continuous_unnormalize(
    values: Any,
    stats: Mapping[str, Any],
    *,
    mode: str,
) -> np.ndarray:
    """Apply the exact training action inverse without clipping predictions.

    Values outside the nominal normalized interval remain outside it during
    the affine inverse.  This matches ``Normalizer.inverse`` and the RealMan
    policy server; any robot-side safety handling is a separate post-inverse
    concern.  Keeping this helper RealMan-local avoids inheriting the legacy
    bounded diagnostic inverse used by the Trossen deployment adapter.
    """

    array = np.asarray(values, dtype=np.float32)

    def _stat(name: str) -> np.ndarray:
        value = np.asarray(stats[name], dtype=np.float32)
        if value.ndim != 1:
            raise ValueError(
                f"RealMan action statistic {name!r} must be one-dimensional, "
                f"got {value.shape}."
            )
        if array.ndim == 0 or array.shape[-1] != value.shape[0]:
            raise ValueError(
                "RealMan action inverse dimension mismatch: "
                f"values={array.shape}, {name}={value.shape}."
            )
        return value

    if mode == "min_max":
        minimum = _stat("min")
        maximum = _stat("max")
        if minimum.shape != maximum.shape:
            raise ValueError(
                f"RealMan min/max statistics differ: {minimum.shape} != {maximum.shape}."
            )
        return (0.5 * (array + 1.0) * (maximum - minimum) + minimum).astype(
            np.float32,
            copy=False,
        )
    if mode == "q99":
        q01 = _stat("q01")
        q99 = _stat("q99")
        if q01.shape != q99.shape:
            raise ValueError(
                f"RealMan q01/q99 statistics differ: {q01.shape} != {q99.shape}."
            )
        return (0.5 * (array + 1.0) * (q99 - q01) + q01).astype(
            np.float32,
            copy=False,
        )
    if mode == "mean_std":
        mean = _stat("mean")
        std = _stat("std")
        if mean.shape != std.shape:
            raise ValueError(
                f"RealMan mean/std statistics differ: {mean.shape} != {std.shape}."
            )
        return (array * std + mean).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported RealMan action normalization mode {mode!r}.")


def realman_policy_action_names(action_dim: int) -> tuple[str, ...]:
    """Return the model-side action layout for a supported Realman checkpoint."""
    layouts = {
        REALMAN_ACTION_DIM: REALMAN_ACTION_NAMES,
        REALMAN_POLICY_ACTION_DIM_NO_BASE: REALMAN_POLICY_ACTION_NAMES_NO_BASE,
        REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT: REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT,
    }
    try:
        return layouts[int(action_dim)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported Realman policy action dim {action_dim!r}; expected one of "
            f"{REALMAN_POLICY_ACTION_DIMS}."
        ) from exc


def realman_omitted_robot_action_indices(action_dim: int) -> tuple[int, ...]:
    """Return 22D robot-command indices not predicted by the policy."""
    action_dim = int(action_dim)
    realman_policy_action_names(action_dim)
    if action_dim == REALMAN_ACTION_DIM:
        return ()
    if action_dim == REALMAN_POLICY_ACTION_DIM_NO_BASE:
        return (16, 17, 18)
    return (16, 17, 18, 21)

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


def _coerce_realman_rgb_uint8(image: Any) -> np.ndarray:
    rgb = coerce_rgb_uint8(image)
    if str(getattr(image, "_yondu_color_space", "")).strip().lower() == "bgr":
        rgb = rgb[:, :, ::-1]
    return np.ascontiguousarray(rgb, dtype=np.uint8)


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
        images.append(resize_for_training(_coerce_realman_rgb_uint8(image), image_size=image_size))

    if missing:
        raise KeyError(
            "Observation is missing Realman camera(s) "
            f"{missing}. Expected aliases: "
            f"{ {name: _IMAGE_KEY_ALIASES.get(name, (name,)) for name in missing} }"
        )
    return images


def resize_for_qwen_tensor_path(image: Any, image_size: int = DEFAULT_QWEN_FRAME_SIZE) -> np.ndarray:
    """Match the CPU resize performed by the production training dataloader."""
    import cv2

    size = int(image_size)
    if size <= 0:
        raise ValueError(f"Qwen tensor frame size must be positive, got {image_size!r}.")
    rgb = _coerce_realman_rgb_uint8(image)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def extract_ordered_qwen_frames(
    observation: dict[str, Any],
    camera_order: Iterable[str] = REALMAN_CAMERA_ORDER,
    image_size: int = DEFAULT_QWEN_FRAME_SIZE,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    missing: list[str] = []
    for camera_name in camera_order:
        aliases = _IMAGE_KEY_ALIASES.get(camera_name, (camera_name,))
        try:
            image = _lookup_observation_value(observation, aliases)
        except KeyError:
            missing.append(camera_name)
            continue
        frames.append(resize_for_qwen_tensor_path(image, image_size=image_size))

    if missing:
        raise KeyError(
            "Observation is missing Realman camera(s) "
            f"{missing}. Expected aliases: "
            f"{ {name: _IMAGE_KEY_ALIASES.get(name, (name,)) for name in missing} }"
        )
    return np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)


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
    image_size: int = DEFAULT_QWEN_FRAME_SIZE,
    state_stats: dict[str, Any] | None = None,
    state_norm_mode: str | None = None,
) -> dict[str, Any]:
    state = extract_state_vector(observation)
    if state_stats is not None:
        state = continuous_normalize(state, state_stats, mode=state_norm_mode or DEFAULT_NORM_MODE)
    frames = extract_ordered_qwen_frames(observation, camera_order=camera_order, image_size=image_size)
    return {
        QWEN_TENSOR_PAYLOAD_KEY: [frames],
        "instructions": [instruction],
        "state": np.ascontiguousarray(state[None, None, :], dtype=np.float32),
    }


def resolve_qwen_frame_size(metadata: dict[str, Any], explicit_size: int | None = None) -> int:
    contract = metadata.get("realman_input_contract") or {}
    configured_size = contract.get("frame_size", metadata.get("video_resolution_size"))
    try:
        expected_size = int(configured_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Server metadata does not define a valid Realman Qwen frame size.") from exc
    if expected_size <= 0:
        raise ValueError(f"Server reports invalid Realman Qwen frame size {expected_size}.")

    if explicit_size is not None and int(explicit_size) > 0 and int(explicit_size) != expected_size:
        raise ValueError(
            f"Requested image size {int(explicit_size)} does not match the checkpoint's "
            f"training-aligned Qwen frame size {expected_size}."
        )
    return expected_size


def validate_realman_policy_payload(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    *,
    require_normalized_bounds: bool = True,
) -> None:
    contract = metadata.get("realman_input_contract") or {}
    payload_key = str(contract.get("payload_key") or QWEN_TENSOR_PAYLOAD_KEY)
    if payload_key != QWEN_TENSOR_PAYLOAD_KEY:
        raise ValueError(
            f"Unsupported Realman payload key {payload_key!r}; expected {QWEN_TENSOR_PAYLOAD_KEY!r}."
        )
    if payload_key not in payload:
        raise KeyError(f"Policy payload is missing `{payload_key}`.")

    frame_size = resolve_qwen_frame_size(metadata)
    frames = np.asarray(payload[payload_key])
    expected_shape = (1, len(REALMAN_CAMERA_ORDER), frame_size, frame_size, 3)
    if frames.shape != expected_shape:
        raise ValueError(f"Qwen frame shape is {frames.shape}, expected {expected_shape}.")
    if frames.dtype != np.uint8:
        raise ValueError(f"Qwen frame dtype is {frames.dtype}, expected uint8.")

    instructions = payload.get("instructions")
    if not isinstance(instructions, (list, tuple)) or len(instructions) != 1:
        raise ValueError("Policy payload must contain exactly one instruction.")
    if not isinstance(instructions[0], str) or not instructions[0].strip():
        raise ValueError("Policy instruction must be a non-empty string.")

    state = np.asarray(payload.get("state"))
    expected_state_shape = (1, 1, REALMAN_STATE_DIM)
    if state.shape != expected_state_shape:
        raise ValueError(f"Policy state shape is {state.shape}, expected {expected_state_shape}.")
    if state.dtype != np.float32:
        raise ValueError(f"Policy state dtype is {state.dtype}, expected float32.")
    if not np.all(np.isfinite(state)):
        raise ValueError("Policy state contains NaN or Inf.")
    if require_normalized_bounds and np.any(np.abs(state) > 1.00001):
        raise ValueError("Normalized policy state contains values outside [-1, 1].")


def validate_realman_server_metadata(
    metadata: dict[str, Any],
    *,
    expected_action_dim: int | Iterable[int] | None = REALMAN_POLICY_ACTION_DIMS,
    expected_state_dim: int = REALMAN_STATE_DIM,
    expected_action_type: str | None = "absolute_qpos",
    expected_camera_order: Iterable[str] = REALMAN_CAMERA_ORDER,
    require_input_contract: bool = False,
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

    reported_action_names = metadata.get("policy_action_names")
    if action_dim_ok and reported_action_names is not None:
        expected_names = realman_policy_action_names(int(reported_action_dim))
        if tuple(reported_action_names) != expected_names:
            warnings.append(
                "Server policy_action_names do not match the reported Realman "
                f"action_dim={reported_action_dim}."
            )

    robot_action_dim = metadata.get("robot_action_dim")
    if robot_action_dim is not None and int(robot_action_dim) != REALMAN_ACTION_DIM:
        warnings.append(
            f"Server reports robot_action_dim={robot_action_dim}, expected {REALMAN_ACTION_DIM}."
        )

    camera_hint = metadata.get("camera_order_hint")
    if camera_hint is not None and list(camera_hint) != list(expected_camera_order):
        warnings.append(
            f"Server camera hint {camera_hint} does not match Realman camera order "
            f"{list(expected_camera_order)}."
        )

    input_contract = metadata.get("realman_input_contract")
    if require_input_contract and not isinstance(input_contract, dict):
        warnings.append("Server metadata does not include `realman_input_contract`.")
    elif isinstance(input_contract, dict):
        expected_contract_values = {
            "payload_key": QWEN_TENSOR_PAYLOAD_KEY,
            "camera_order": list(expected_camera_order),
            "frame_dtype": "uint8",
            "color_space": "RGB",
            "transport_encoding": "msgpack_ndarray",
            "client_resize": "opencv_inter_linear",
            "model_preprocess": "qwen_tensor_fast_path",
            "state_shape": [1, 1, expected_state_dim],
            "state_dtype": "float32",
            "state_normalized": True,
        }
        for key, expected in expected_contract_values.items():
            if input_contract.get(key) != expected:
                warnings.append(
                    f"Server realman_input_contract.{key}={input_contract.get(key)!r}, expected {expected!r}."
                )
        expected_frame_size = metadata.get("video_resolution_size")
        if expected_frame_size is not None:
            try:
                frame_size_ok = int(input_contract.get("frame_size")) == int(expected_frame_size)
            except (TypeError, ValueError):
                frame_size_ok = False
            if not frame_size_ok:
                warnings.append(
                    "Server realman_input_contract.frame_size does not match video_resolution_size."
                )
    return warnings


def expand_policy_action_to_robot_action(
    action: np.ndarray,
    *,
    lift_height_mm: float | np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape[-1] == REALMAN_ACTION_DIM:
        return np.ascontiguousarray(array)
    if array.shape[-1] not in {
        REALMAN_POLICY_ACTION_DIM_NO_BASE,
        REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT,
    }:
        raise ValueError(
            f"Realman policy action dim is {array.shape[-1]}, expected "
            f"one of {REALMAN_POLICY_ACTION_DIMS}."
        )

    expanded = np.zeros((*array.shape[:-1], REALMAN_ACTION_DIM), dtype=np.float32)
    expanded[..., :16] = array[..., :16]
    if array.shape[-1] == REALMAN_POLICY_ACTION_DIM_NO_BASE:
        expanded[..., 19:22] = array[..., 16:19]
    else:
        if lift_height_mm is None:
            raise ValueError(
                "An 18D no-base/no-lift policy action requires the current measured "
                "lift_height_mm when expanding to the robot's 22D command."
            )
        expanded[..., 19:21] = array[..., 16:18]
        try:
            expanded[..., 21] = np.asarray(lift_height_mm, dtype=np.float32)
        except ValueError as exc:
            raise ValueError(
                f"lift_height_mm cannot broadcast to action shape {array.shape[:-1]}."
            ) from exc
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
