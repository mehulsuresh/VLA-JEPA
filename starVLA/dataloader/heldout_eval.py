"""Deterministic LeRobot holdout evaluation.

The training mixture deliberately maps an integer index to a pseudo-random
episode and frame.  That is appropriate for training, but it cannot express
the checkpoint-evaluation contract: visit every manifest-selected holdout
episode exactly once at one immutable, structurally valid frame.  This module
adapts a separately constructed ``mode="eval"`` mixture without sharing any
dataset object, iterator, sampler, or RNG with the training loader.  It keeps
that unbiased one-window-per-episode view and can add a separate deterministic
H10 transition/stage-focused diagnostic view from the same manifest episodes.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    apply_transforms_for_present_keys,
)


EVAL_SAMPLING_ALGORITHM = "nonzero_valid_unpadded_uniform_v1"
EVAL_OBSERVATION_MODE = "deployment_action_current_qwen_rgb_v1"
EVAL_CANDIDATE_POLICY = (
    "structurally unpadded frames with at least one valid action-mask element"
)
EVAL_ALL_INVALID_FALLBACK = (
    "uniform over all structurally unpadded frames; report zero valid elements"
)
EVAL_SAMPLING_REPORT_SCHEMA_VERSION = 1
FOCUSED_EVAL_ALGORITHM = "h10_gripper_transition_stage_balanced_v1"
FOCUSED_EVAL_TRANSITION_HORIZON = 10
DEFAULT_FOCUSED_SUBTASKS = (2, 3, 4, 5, 6, 7)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationSamplingContract:
    algorithm: str
    frames_per_episode: int
    seed_sha256: str
    observation_mode: str
    evaluation_video_offsets: tuple[int, ...]
    action_offset_range_inclusive: tuple[int, int]

    @property
    def torch_seed(self) -> int:
        # Keep the seed in torch's portable positive signed-64-bit range.
        return int(self.seed_sha256[:16], 16) % (2**63 - 1)


@dataclass(frozen=True, slots=True)
class HeldoutWindowReference:
    dataset_index: int
    dataset_name: str
    episode_id: int
    base_index: int
    valid_base_index_min: int
    valid_base_index_max: int
    structural_candidate_count: int
    evaluable_candidate_count: int
    selection_pool_candidate_count: int
    valid_action_timesteps: int
    valid_action_elements: int
    anchor_subtask_index: int | None
    action_subtask_indices: tuple[int, ...] = ()
    valid_action_elements_per_timestep: tuple[int, ...] = ()
    open_to_close_transitions_h10: int = 0
    close_to_open_transitions_h10: int = 0
    open_to_close_window_h10: bool = False
    close_to_open_window_h10: bool = False
    arm_movement_elements_h10: int = 0
    arm_movement_hold_abs_h10: float = 0.0


@dataclass(frozen=True, slots=True)
class _CandidateDiagnostic:
    base_index: int
    valid_action_timesteps: int
    valid_action_elements: int
    anchor_subtask_index: int | None
    action_subtask_indices: tuple[int, ...]
    valid_action_elements_per_timestep: tuple[int, ...]
    open_to_close_transitions_h10: int
    close_to_open_transitions_h10: int
    arm_movement_elements_h10: int
    arm_movement_hold_abs_h10: float


def _coerce_optional_integer_label(value: Any) -> int | None:
    """Return a stable integer label, rejecting lossy/non-finite coercions."""

    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    scalar = array.reshape(-1)[0]
    try:
        numeric = float(scalar)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric != int(numeric):
        return None
    return int(numeric)


def _action_subtask_sequence(
    dataset: Any,
    *,
    base_index: int,
    action_offsets: np.ndarray,
) -> tuple[int, ...]:
    if dataset.curr_traj_data is None or "subtask_index" not in (
        dataset.curr_traj_data.columns
    ):
        # Generic heldout fixtures and non-RealMan datasets may not expose stage
        # labels. Production coverage validation requires them explicitly; the
        # selector itself remains usable for the legacy aggregate metrics.
        return ()
    indices = int(base_index) + np.asarray(action_offsets, dtype=np.int64)
    if indices.min() < 0 or indices.max() >= len(dataset.curr_traj_data):
        raise ValueError(
            "Heldout per-action subtask labels escaped the structural window: "
            f"{dataset.dataset_name}/{base_index}."
        )
    labels: list[int] = []
    for value in dataset.curr_traj_data.iloc[indices]["subtask_index"].tolist():
        label = _coerce_optional_integer_label(value)
        if label is None:
            raise ValueError(
                "Heldout eval encountered a missing/non-integer per-action "
                f"subtask label at {dataset.dataset_name}/{base_index}."
            )
        labels.append(label)
    return tuple(labels)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _realman_hold_raw_action(raw_state: np.ndarray, action_dim: int) -> np.ndarray:
    """Project the current 19D RealMan state into an absolute-action hold."""

    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    if state.shape != (19,):
        raise ValueError(
            "Heldout RealMan baseline requires a 19D raw state, "
            f"got {state.shape}."
        )
    if int(action_dim) == 18:
        return state[:18].copy()
    if int(action_dim) == 19:
        return state.copy()
    if int(action_dim) == 22:
        action = np.zeros(22, dtype=np.float32)
        action[:16] = state[:16]
        action[19:22] = state[16:19]
        return action
    raise ValueError(
        "Heldout RealMan control metrics support action_dim 18, 19, or 22, "
        f"got {action_dim}."
    )


def _current_raw_state(dataset: Any, reference: HeldoutWindowReference) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for key in dataset.modality_keys.get("state", ()):
        values = dataset.get_state_or_action(
            int(reference.episode_id),
            "state",
            key,
            int(reference.base_index),
        )
        if isinstance(values, tuple):
            values = values[0]
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] < 1:
            raise ValueError(
                f"Heldout raw state {key!r} must have shape [time, dim], got "
                f"{array.shape}."
            )
        pieces.append(array[0])
    if not pieces:
        raise ValueError("Heldout RealMan baseline requires a state modality.")
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)


def _split_action_matrix(dataset: Any, values: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(values, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2:
        raise ValueError(
            "Heldout action normalization requires [rows, action_dim], "
            f"got {action.shape}."
        )
    payload: dict[str, np.ndarray] = {}
    offset = 0
    for key in dataset.modality_keys.get("action", ()):
        if "." not in key:
            raise ValueError(f"Expected namespaced action key, got {key!r}.")
        subkey = key.split(".", 1)[1]
        metadata = getattr(dataset.metadata.modalities, "action")[subkey]
        shape = tuple(int(value) for value in metadata.shape)
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError(
                f"Heldout action metadata {key!r} must be one-dimensional, got {shape}."
            )
        next_offset = offset + shape[0]
        payload[key] = action[:, offset:next_offset]
        offset = next_offset
    if offset != action.shape[1]:
        raise ValueError(
            "Heldout action metadata dimensions do not match the policy action "
            f"dimension: metadata={offset}, action={action.shape[1]}."
        )
    return payload


def _normalize_action_matrix(dataset: Any, values: np.ndarray) -> np.ndarray:
    payload = _split_action_matrix(dataset, values)
    transformed = apply_transforms_for_present_keys(dataset.transforms, payload)
    pieces: list[np.ndarray] = []
    for key in dataset.modality_keys.get("action", ()):
        if key not in transformed:
            raise ValueError(
                f"Heldout action transform did not return required key {key!r}."
            )
        array = _to_numpy(transformed[key]).astype(np.float32, copy=False)
        if array.ndim != 2:
            raise ValueError(
                f"Heldout normalized action {key!r} must have shape [rows, dim], "
                f"got {array.shape}."
            )
        pieces.append(array)
    row_counts = {piece.shape[0] for piece in pieces}
    if len(row_counts) != 1:
        raise ValueError(
            "Heldout normalized action modalities returned inconsistent row counts."
        )
    return np.ascontiguousarray(np.concatenate(pieces, axis=1), dtype=np.float32)


def _normalize_action_vector(dataset: Any, vector: np.ndarray) -> np.ndarray:
    normalized = _normalize_action_matrix(dataset, vector)
    if normalized.shape[0] != 1:
        raise ValueError(
            "Heldout action-vector normalization unexpectedly returned "
            f"{normalized.shape[0]} rows."
        )
    return normalized[0]


def _raw_action_midpoint(dataset: Any) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for key in dataset.modality_keys.get("action", ()):
        subkey = key.split(".", 1)[1]
        statistics = getattr(dataset.metadata.statistics, "action")[subkey]
        low = np.asarray(statistics.min, dtype=np.float32).reshape(-1)
        high = np.asarray(statistics.max, dtype=np.float32).reshape(-1)
        if low.shape != high.shape or low.size == 0:
            raise ValueError(
                f"Heldout action statistics for {key!r} have incompatible extrema: "
                f"min={low.shape}, max={high.shape}."
            )
        pieces.append(0.5 * (low + high))
    if not pieces:
        raise ValueError("Heldout RealMan metrics require an action modality.")
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)


def load_evaluation_sampling_contract(
    manifest_path: Path | str,
) -> EvaluationSamplingContract:
    """Load and strictly validate the manifest's checkpoint sampling rule."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Episode split manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Episode split manifest is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Episode split manifest root must be a JSON object.")

    sampling = payload.get("evaluation_sampling")
    if not isinstance(sampling, dict):
        raise ValueError(
            "Episode split manifest must contain a top-level evaluation_sampling object."
        )
    algorithm = sampling.get("algorithm")
    if algorithm != EVAL_SAMPLING_ALGORITHM:
        raise ValueError(
            "evaluation_sampling.algorithm must be "
            f"{EVAL_SAMPLING_ALGORITHM!r}, got {algorithm!r}."
        )
    frames_per_episode = sampling.get("frames_per_episode")
    if isinstance(frames_per_episode, bool) or frames_per_episode != 1:
        raise ValueError(
            "In-training checkpoint evaluation requires exactly one frame per "
            f"holdout episode; got frames_per_episode={frames_per_episode!r}."
        )
    seed_sha256 = sampling.get("seed_sha256")
    if not isinstance(seed_sha256, str) or _SHA256_RE.fullmatch(seed_sha256) is None:
        raise ValueError(
            "evaluation_sampling.seed_sha256 must be exactly 64 lowercase hex characters."
        )
    observation_mode = sampling.get("observation_mode")
    if observation_mode != EVAL_OBSERVATION_MODE:
        raise ValueError(
            "evaluation_sampling.observation_mode must be "
            f"{EVAL_OBSERVATION_MODE!r}, got {observation_mode!r}."
        )
    evaluation_video_offsets = sampling.get("evaluation_video_offsets")
    if evaluation_video_offsets != [0]:
        raise ValueError(
            "Deployment-style in-training action eval requires exactly "
            "evaluation_video_offsets=[0]."
        )
    raw_action_range = sampling.get("action_offset_range_inclusive")
    if (
        not isinstance(raw_action_range, list)
        or len(raw_action_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_action_range)
        or int(raw_action_range[0]) > int(raw_action_range[1])
    ):
        raise ValueError(
            "evaluation_sampling.action_offset_range_inclusive must be two "
            "ordered integer offsets."
        )
    if sampling.get("candidate_policy") != EVAL_CANDIDATE_POLICY:
        raise ValueError(
            "evaluation_sampling.candidate_policy does not match the supported "
            "nonzero-valid unpadded sampling contract."
        )
    if sampling.get("all_invalid_episode_fallback") != EVAL_ALL_INVALID_FALLBACK:
        raise ValueError(
            "evaluation_sampling.all_invalid_episode_fallback does not match "
            "the supported zero-metric forwarding contract."
        )
    return EvaluationSamplingContract(
        algorithm=algorithm,
        frames_per_episode=1,
        seed_sha256=seed_sha256,
        observation_mode=observation_mode,
        evaluation_video_offsets=(0,),
        action_offset_range_inclusive=(
            int(raw_action_range[0]),
            int(raw_action_range[1]),
        ),
    )


def _used_modality_offsets(dataset, modality: str) -> np.ndarray:
    """Return offsets actually consumed by the mixture's packed sample.

    Action chunks consume every configured action offset.  State and language
    are fetched at the observation horizon but the mixture packs only their
    first (current) row, so only that first offset constrains a valid window.
    """

    keys = list(dataset.modality_keys.get(modality, ()))
    if not keys:
        return np.empty(0, dtype=np.int64)
    offsets = [np.asarray(dataset.delta_indices[key], dtype=np.int64) for key in keys]
    if modality in {"state", "language"}:
        return np.asarray([int(values[0]) for values in offsets], dtype=np.int64)
    return np.concatenate(offsets)


def configure_deployment_current_frame_observation(
    mixture: LeRobotMixtureDataset,
) -> None:
    """Make the independent eval view match deployed action-policy observation.

    Training's compact V-JEPA clip is auxiliary world-model context.  Action
    deployment sends only the current RGB frame to Qwen, so checkpoint action
    eval must not decode or constrain anchors by that eight-frame clip.
    """

    mixture.video_target_shift_steps = 0
    for dataset in mixture.datasets:
        for video_key in dataset.modality_keys.get("video", ()):
            dataset.delta_indices[video_key] = np.asarray([0], dtype=np.int64)


def _validate_sampling_offsets(
    mixture: LeRobotMixtureDataset,
    contract: EvaluationSamplingContract,
) -> None:
    for dataset in mixture.datasets:
        for video_key in dataset.modality_keys.get("video", ()):
            video_offsets = tuple(
                int(value)
                for value in np.asarray(
                    dataset.delta_indices[video_key], dtype=np.int64
                ).reshape(-1)
            )
            if video_offsets != contract.evaluation_video_offsets:
                raise ValueError(
                    f"Dataset {dataset.dataset_name!r} eval video offsets "
                    f"{video_offsets} do not match manifest "
                    f"{contract.evaluation_video_offsets}."
                )
        action_offsets = _used_modality_offsets(dataset, "action")
        if action_offsets.size == 0:
            raise ValueError(f"Dataset {dataset.dataset_name!r} has no action offsets.")
        action_range = (int(action_offsets.min()), int(action_offsets.max()))
        if action_range != contract.action_offset_range_inclusive:
            raise ValueError(
                f"Dataset {dataset.dataset_name!r} action offset range "
                f"{action_range} does not match manifest "
                f"{contract.action_offset_range_inclusive}."
            )


def required_window_offsets(
    mixture: LeRobotMixtureDataset,
    dataset,
) -> np.ndarray:
    """Offsets that must remain inside an episode for an unpadded eval sample."""

    parts = [
        _used_modality_offsets(dataset, "action"),
        _used_modality_offsets(dataset, "state"),
        _used_modality_offsets(dataset, "language"),
    ]

    video_keys = list(dataset.modality_keys.get("video", ()))
    if video_keys:
        target_shift = int(getattr(mixture, "video_target_shift_steps", 0) or 0)
        if target_shift > 0:
            horizons = {
                len(np.asarray(dataset.delta_indices[key]).reshape(-1))
                for key in video_keys
            }
            if len(horizons) != 1:
                raise ValueError(
                    f"Dataset {dataset.dataset_name!r} has inconsistent video horizons: "
                    f"{sorted(horizons)}."
                )
            video_horizon = next(iter(horizons))
            context_horizon = video_horizon - target_shift
            if context_horizon <= 0:
                raise ValueError(
                    f"Dataset {dataset.dataset_name!r} video horizon {video_horizon} "
                    f"must exceed video_target_shift_steps={target_shift}."
                )
            stride = max(int(getattr(mixture, "video_frame_stride", 1)), 1)
            parts.append(
                np.arange(
                    -(context_horizon - 1),
                    target_shift + 1,
                    dtype=np.int64,
                )
                * stride
            )
        else:
            parts.extend(
                np.asarray(dataset.delta_indices[key], dtype=np.int64)
                for key in video_keys
            )

    nonempty = [values.reshape(-1) for values in parts if values.size]
    if not nonempty:
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} has no configured evaluation modalities."
        )
    return np.unique(np.concatenate(nonempty))


def _episode_candidate_steps(
    *,
    dataset,
    episode_id: int,
    episode_length: int,
    required_offsets: np.ndarray,
) -> tuple[list[int], int, int]:
    valid_min = max(0, -int(required_offsets.min()))
    valid_max = min(
        int(episode_length) - 1,
        int(episode_length) - 1 - int(required_offsets.max()),
    )
    if valid_max < valid_min:
        raise ValueError(
            f"Heldout episode {dataset.dataset_name}/{episode_id} has length "
            f"{episode_length}, but the training context/action offsets "
            f"[{int(required_offsets.min())}, {int(required_offsets.max())}] leave "
            "no structurally valid unpadded evaluation window."
        )

    # ``all_steps`` incorporates any dataset-specific candidate filtering (for
    # example language validity or pause-frame deletion).  Intersecting it with
    # the structural interval avoids silently selecting a padded window.
    candidates = sorted(
        {
            int(base_index)
            for trajectory_id, base_index in dataset.all_steps
            if int(trajectory_id) == int(episode_id)
            and valid_min <= int(base_index) <= valid_max
        }
    )
    if not candidates:
        raise ValueError(
            f"Heldout episode {dataset.dataset_name}/{episode_id} has no valid "
            f"candidate steps in [{valid_min}, {valid_max}]."
        )
    return candidates, valid_min, valid_max


def _candidate_valid_action_counts(
    *,
    mixture: LeRobotMixtureDataset,
    dataset,
    episode_id: int,
    candidates: Sequence[int],
    action_dim: int,
) -> list[tuple[int, int, int, tuple[int, ...]]]:
    """Return aggregate and per-timestep counts using the training mask."""

    action_keys = list(dataset.modality_keys.get("action", ()))
    if not action_keys:
        raise ValueError(f"Dataset {dataset.dataset_name!r} has no action modality.")
    action_offsets = [
        np.asarray(dataset.delta_indices[key], dtype=np.int64).reshape(-1)
        for key in action_keys
    ]
    if any(not np.array_equal(action_offsets[0], values) for values in action_offsets[1:]):
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} action keys have inconsistent horizons."
        )
    action_horizon = int(action_offsets[0].size)
    if action_horizon <= 0:
        raise ValueError(f"Dataset {dataset.dataset_name!r} has an empty action horizon.")

    # Load only the episode parquet.  The training mask helper reads labels from
    # curr_traj_data and never decodes video or builds model image inputs.
    dataset.curr_traj_data = dataset.get_trajectory_data(int(episode_id))
    dummy_action = np.zeros((action_horizon, int(action_dim)), dtype=np.float32)
    no_padding = np.zeros(action_horizon, dtype=bool)
    counts: list[tuple[int, int, int, tuple[int, ...]]] = []
    for step in candidates:
        mask = mixture._build_action_validity_mask(
            dataset,
            step=int(step),
            action=dummy_action,
            action_is_pad=no_padding,
        )
        if mask is None:
            valid_timesteps = action_horizon
            valid_elements = action_horizon * int(action_dim)
            per_timestep = (int(action_dim),) * action_horizon
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != dummy_action.shape:
                raise ValueError(
                    f"Training action-validity mask for {dataset.dataset_name}/{episode_id} "
                    f"has shape {mask.shape}, expected {dummy_action.shape}."
                )
            valid_elements = int(mask.sum())
            valid_timesteps = int(mask.any(axis=1).sum())
            per_timestep = tuple(int(value) for value in mask.sum(axis=1).tolist())
        counts.append(
            (int(step), valid_timesteps, valid_elements, per_timestep)
        )
    return counts


def _raw_episode_modality_matrix(dataset: Any, modality: str) -> np.ndarray:
    """Read one episode's selected raw modality columns without per-step copies."""

    if dataset.curr_traj_data is None:
        raise ValueError("Focused heldout selection requires loaded episode data.")
    pieces: list[np.ndarray] = []
    for namespaced_key in dataset.modality_keys.get(modality, ()):
        if not namespaced_key.startswith(f"{modality}."):
            raise ValueError(
                f"Expected {modality!r} key, got {namespaced_key!r}."
            )
        key = namespaced_key.split(".", 1)[1]
        lerobot_cfg = getattr(dataset.lerobot_modality_meta, modality)[key]
        column = lerobot_cfg.original_key or key
        if column not in dataset.curr_traj_data.columns:
            raise ValueError(
                f"Focused heldout selection cannot find raw column {column!r}."
            )
        values = np.stack(dataset.curr_traj_data[column].to_numpy())
        indices = np.arange(int(lerobot_cfg.start), int(lerobot_cfg.end))
        pieces.append(np.asarray(values[:, indices], dtype=np.float32))
    if not pieces:
        raise ValueError(
            f"Focused heldout selection requires a {modality!r} modality."
        )
    return np.ascontiguousarray(np.concatenate(pieces, axis=1), dtype=np.float32)


def _candidate_control_diagnostics(
    *,
    mixture: LeRobotMixtureDataset,
    dataset: Any,
    episode_id: int,
    candidates: Sequence[int],
    action_dim: int,
    movement_threshold: float,
) -> list[_CandidateDiagnostic]:
    """Inspect labels/actions only; this never decodes an evaluation image."""

    if int(action_dim) not in {18, 19, 22}:
        raise ValueError(
            "Focused transition eval supports RealMan action_dim 18, 19, or 22, "
            f"got {action_dim}."
        )
    action_keys = list(dataset.modality_keys.get("action", ()))
    state_keys = list(dataset.modality_keys.get("state", ()))
    action_offsets = [
        np.asarray(dataset.delta_indices[key], dtype=np.int64).reshape(-1)
        for key in action_keys
    ]
    if not action_offsets or any(
        not np.array_equal(action_offsets[0], values)
        for values in action_offsets[1:]
    ):
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} has missing/inconsistent action offsets."
        )
    if action_offsets[0].size < FOCUSED_EVAL_TRANSITION_HORIZON:
        raise ValueError(
            "Focused transition eval requires an action horizon of at least "
            f"{FOCUSED_EVAL_TRANSITION_HORIZON}."
        )
    state_offsets = [
        np.asarray(dataset.delta_indices[key], dtype=np.int64).reshape(-1)
        for key in state_keys
    ]
    if not state_offsets or any(values.size == 0 for values in state_offsets):
        raise ValueError("Focused transition eval requires current robot state.")
    current_state_offsets = {int(values[0]) for values in state_offsets}
    if len(current_state_offsets) != 1:
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} has inconsistent state offsets."
        )

    dataset.curr_traj_data = dataset.get_trajectory_data(int(episode_id))
    action_matrix = _raw_episode_modality_matrix(dataset, "action")
    state_matrix = _raw_episode_modality_matrix(dataset, "state")
    if action_matrix.shape[1] != int(action_dim):
        raise ValueError(
            "Focused heldout raw action dimension mismatch: "
            f"{action_matrix.shape[1]} != {action_dim}."
        )
    action_midpoint = _raw_action_midpoint(dataset)
    if action_midpoint.shape != (int(action_dim),):
        raise ValueError(
            "Focused heldout action midpoint dimension mismatch: "
            f"{action_midpoint.shape} != {(int(action_dim),)}."
        )
    normalized_action_matrix = _normalize_action_matrix(dataset, action_matrix)
    if normalized_action_matrix.shape != action_matrix.shape:
        raise ValueError(
            "Focused heldout normalized action matrix changed shape: "
            f"{normalized_action_matrix.shape} != {action_matrix.shape}."
        )
    # The mixture packs normalized policy targets as float16 before collation
    # (see LeRobotMixtureDataset.__getitem__).  Movement coverage and its hold
    # denominator must use those exact target values or threshold-adjacent
    # elements can disagree with the runtime evaluator.
    packed_normalized_action_matrix = np.ascontiguousarray(
        normalized_action_matrix.astype(np.float16).astype(np.float32),
        dtype=np.float32,
    )
    raw_hold_matrix = np.stack(
        [
            _realman_hold_raw_action(row, action_dim)
            for row in state_matrix
        ],
        axis=0,
    )
    normalized_hold_matrix = _normalize_action_matrix(dataset, raw_hold_matrix)
    if normalized_hold_matrix.shape != action_matrix.shape:
        raise ValueError(
            "Focused heldout normalized persistence matrix changed shape: "
            f"{normalized_hold_matrix.shape} != {action_matrix.shape}."
        )

    horizon = int(action_offsets[0].size)
    dummy_action = np.zeros((horizon, int(action_dim)), dtype=np.float32)
    no_padding = np.zeros(horizon, dtype=bool)
    gripper_dimensions = np.asarray((7, 15), dtype=np.int64)
    state_offset = next(iter(current_state_offsets))
    diagnostics: list[_CandidateDiagnostic] = []
    for raw_step in candidates:
        step = int(raw_step)
        mask = mixture._build_action_validity_mask(
            dataset,
            step=step,
            action=dummy_action,
            action_is_pad=no_padding,
        )
        if mask is None:
            mask = np.ones_like(dummy_action, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != dummy_action.shape:
                raise ValueError(
                    f"Training action-validity mask for {dataset.dataset_name}/{episode_id} "
                    f"has shape {mask.shape}, expected {dummy_action.shape}."
                )

        target_indices = step + action_offsets[0]
        state_index = step + state_offset
        if (
            target_indices.min() < 0
            or target_indices.max() >= action_matrix.shape[0]
            or state_index < 0
            or state_index >= state_matrix.shape[0]
        ):
            raise ValueError(
                "Focused heldout candidate escaped the structurally valid window: "
                f"{dataset.dataset_name}/{episode_id}/{step}."
            )
        raw_targets = action_matrix[target_indices]
        raw_hold = raw_hold_matrix[state_index]
        normalized_targets = packed_normalized_action_matrix[target_indices]
        normalized_hold = normalized_hold_matrix[state_index]
        limit = FOCUSED_EVAL_TRANSITION_HORIZON
        target_close = (
            raw_targets[:limit, gripper_dimensions]
            < action_midpoint[gripper_dimensions]
        )
        current_close = (
            raw_hold[gripper_dimensions] < action_midpoint[gripper_dimensions]
        )
        previous_close = np.concatenate(
            (current_close[None, :], target_close[:-1, :]), axis=0
        )
        gripper_mask = mask[:limit, gripper_dimensions]
        previous_valid = np.concatenate(
            (np.ones_like(current_close[None, :]), gripper_mask[:-1, :]),
            axis=0,
        )
        transition_valid = gripper_mask & previous_valid
        arm_dimensions = np.asarray(
            tuple(range(0, 7)) + tuple(range(8, 15)), dtype=np.int64
        )
        arm_mask = mask[:limit, arm_dimensions]
        arm_hold_error = np.abs(
            normalized_targets[:limit, arm_dimensions]
            - normalized_hold[arm_dimensions][None, :]
        )
        arm_movement = arm_mask & (arm_hold_error >= float(movement_threshold))
        subtask = None
        if "subtask_index" in dataset.curr_traj_data.columns:
            subtask = _coerce_optional_integer_label(
                dataset.curr_traj_data.iloc[step]["subtask_index"]
            )
        diagnostics.append(
            _CandidateDiagnostic(
                base_index=step,
                valid_action_timesteps=int(mask.any(axis=1).sum()),
                valid_action_elements=int(mask.sum()),
                anchor_subtask_index=subtask,
                action_subtask_indices=_action_subtask_sequence(
                    dataset,
                    base_index=step,
                    action_offsets=action_offsets[0],
                ),
                valid_action_elements_per_timestep=tuple(
                    int(value) for value in mask.sum(axis=1).tolist()
                ),
                open_to_close_transitions_h10=int(
                    ((~previous_close) & target_close & transition_valid).sum()
                ),
                close_to_open_transitions_h10=int(
                    (previous_close & (~target_close) & transition_valid).sum()
                ),
                arm_movement_elements_h10=int(arm_movement.sum()),
                arm_movement_hold_abs_h10=float(
                    arm_hold_error[arm_movement].sum()
                ),
            )
        )
    return diagnostics


def _focused_digest(
    contract: EvaluationSamplingContract,
    *parts: Any,
) -> str:
    return _canonical_sha256(
        {
            "algorithm": FOCUSED_EVAL_ALGORITHM,
            "seed_sha256": contract.seed_sha256,
            "parts": list(parts),
        }
    )


def _uniform_candidate_index(
    *,
    contract: EvaluationSamplingContract,
    dataset_name: str,
    episode_id: int,
    candidate_count: int,
) -> int:
    digest = hashlib.sha256(
        _canonical_json_bytes(
            {
                "algorithm": contract.algorithm,
                "seed_sha256": contract.seed_sha256,
                "dataset_name": str(dataset_name),
                "episode_id": int(episode_id),
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % candidate_count


def select_heldout_window_references(
    mixture: LeRobotMixtureDataset,
    contract: EvaluationSamplingContract,
    *,
    action_dim: int,
) -> tuple[HeldoutWindowReference, ...]:
    """Select one stable, unpadded base index from every heldout episode."""

    references: list[HeldoutWindowReference] = []
    seen_identities: set[tuple[str, int]] = set()
    for dataset_index, dataset in enumerate(mixture.datasets):
        selection = getattr(dataset, "_episode_split_selection", None)
        if selection is None:
            raise ValueError(
                f"Evaluation dataset {dataset.dataset_name!r} was not selected by "
                "an episode split manifest."
            )
        if str(getattr(selection, "role", "")) not in {
            "eval",
            "evaluation",
            "val",
            "validation",
            "test",
            "holdout",
        }:
            raise ValueError(
                f"Evaluation dataset {dataset.dataset_name!r} has non-holdout "
                f"episode split role {getattr(selection, 'role', None)!r}."
            )

        selected_ids = tuple(int(value) for value in dataset.trajectory_ids.tolist())
        manifest_ids = tuple(int(value) for value in selection.selected_episode_ids)
        if set(selected_ids) != set(manifest_ids) or len(selected_ids) != len(manifest_ids):
            raise ValueError(
                f"Evaluation dataset {dataset.dataset_name!r} episode IDs do not "
                "exactly match its manifest-selected holdout IDs."
            )
        lengths_by_id = {
            int(episode_id): int(length)
            for episode_id, length in zip(
                dataset.trajectory_ids.tolist(),
                dataset.trajectory_lengths.tolist(),
            )
        }
        offsets = required_window_offsets(mixture, dataset)
        for episode_id in sorted(selected_ids):
            identity = (str(dataset.dataset_name), int(episode_id))
            if identity in seen_identities:
                raise ValueError(f"Duplicate heldout episode identity {identity!r}.")
            seen_identities.add(identity)
            candidates, valid_min, valid_max = _episode_candidate_steps(
                dataset=dataset,
                episode_id=episode_id,
                episode_length=lengths_by_id[episode_id],
                required_offsets=offsets,
            )
            candidate_counts = _candidate_valid_action_counts(
                mixture=mixture,
                dataset=dataset,
                episode_id=episode_id,
                candidates=candidates,
                action_dim=action_dim,
            )
            evaluable = [entry for entry in candidate_counts if entry[2] > 0]
            # Sample uniformly over every structurally unpadded candidate with
            # at least one supervised action element.  This avoids optimistic
            # max-valid anchor bias.  A manifest-random episode that is entirely
            # invalid falls back to all structural candidates, is still
            # forwarded, and is explicitly excluded from action metrics.
            selection_pool = evaluable if evaluable else candidate_counts
            choice = _uniform_candidate_index(
                contract=contract,
                dataset_name=dataset.dataset_name,
                episode_id=episode_id,
                candidate_count=len(selection_pool),
            )
            (
                selected_step,
                valid_timesteps,
                valid_elements,
                valid_elements_per_timestep,
            ) = selection_pool[choice]
            anchor_subtask_index = None
            if (
                dataset.curr_traj_data is not None
                and "subtask_index" in dataset.curr_traj_data.columns
            ):
                anchor_subtask_index = _coerce_optional_integer_label(
                    dataset.curr_traj_data.iloc[int(selected_step)]["subtask_index"]
                )
            action_key = dataset.modality_keys["action"][0]
            action_subtask_indices = _action_subtask_sequence(
                dataset,
                base_index=int(selected_step),
                action_offsets=np.asarray(
                    dataset.delta_indices[action_key], dtype=np.int64
                ),
            )
            references.append(
                HeldoutWindowReference(
                    dataset_index=int(dataset_index),
                    dataset_name=str(dataset.dataset_name),
                    episode_id=int(episode_id),
                    base_index=int(selected_step),
                    valid_base_index_min=int(valid_min),
                    valid_base_index_max=int(valid_max),
                    structural_candidate_count=len(candidates),
                    evaluable_candidate_count=len(evaluable),
                    selection_pool_candidate_count=len(selection_pool),
                    valid_action_timesteps=int(valid_timesteps),
                    valid_action_elements=int(valid_elements),
                    anchor_subtask_index=anchor_subtask_index,
                    action_subtask_indices=action_subtask_indices,
                    valid_action_elements_per_timestep=tuple(
                        valid_elements_per_timestep
                    ),
                )
            )
    if not references:
        raise ValueError("The manifest-selected holdout contains no episodes.")
    return tuple(
        sorted(
            references,
            key=lambda ref: (ref.dataset_name, ref.episode_id, ref.dataset_index),
        )
    )


def select_focused_heldout_window_references(
    mixture: LeRobotMixtureDataset,
    contract: EvaluationSamplingContract,
    *,
    action_dim: int,
    focused_subtasks: Sequence[int],
    movement_threshold: float,
) -> tuple[HeldoutWindowReference, ...]:
    """Select a second deterministic per-episode view rich in H10 transitions.

    Selection is target-aware but model-independent.  It remains confined to
    the immutable manifest holdout, and therefore cannot change or leak into
    the training iterator.  Episode ordering is hash-derived so stage balancing
    is deterministic rather than dependent on catalog order.
    """

    required_subtasks = tuple(sorted({int(value) for value in focused_subtasks}))
    if not required_subtasks:
        raise ValueError("Focused heldout eval requires at least one subtask label.")
    if not np.isfinite(float(movement_threshold)) or float(movement_threshold) < 0:
        raise ValueError(
            "Focused heldout movement_threshold must be finite and non-negative."
        )

    episode_candidates: list[
        tuple[int, Any, int, int, int, list[_CandidateDiagnostic]]
    ] = []
    seen_identities: set[tuple[str, int]] = set()
    for dataset_index, dataset in enumerate(mixture.datasets):
        selection = getattr(dataset, "_episode_split_selection", None)
        if selection is None or str(getattr(selection, "role", "")) not in {
            "eval",
            "evaluation",
            "val",
            "validation",
            "test",
            "holdout",
        }:
            raise ValueError(
                f"Focused eval dataset {dataset.dataset_name!r} is not manifest holdout."
            )
        selected_ids = tuple(int(value) for value in dataset.trajectory_ids.tolist())
        manifest_ids = tuple(int(value) for value in selection.selected_episode_ids)
        if set(selected_ids) != set(manifest_ids) or len(selected_ids) != len(
            manifest_ids
        ):
            raise ValueError(
                f"Focused eval dataset {dataset.dataset_name!r} does not exactly "
                "match the manifest holdout."
            )
        lengths_by_id = {
            int(episode_id): int(length)
            for episode_id, length in zip(
                dataset.trajectory_ids.tolist(),
                dataset.trajectory_lengths.tolist(),
            )
        }
        offsets = required_window_offsets(mixture, dataset)
        for episode_id in sorted(selected_ids):
            identity = (str(dataset.dataset_name), int(episode_id))
            if identity in seen_identities:
                raise ValueError(f"Duplicate focused heldout identity {identity!r}.")
            seen_identities.add(identity)
            candidates, valid_min, valid_max = _episode_candidate_steps(
                dataset=dataset,
                episode_id=episode_id,
                episode_length=lengths_by_id[episode_id],
                required_offsets=offsets,
            )
            diagnostics = _candidate_control_diagnostics(
                mixture=mixture,
                dataset=dataset,
                episode_id=episode_id,
                candidates=candidates,
                action_dim=action_dim,
                movement_threshold=float(movement_threshold),
            )
            episode_candidates.append(
                (
                    int(dataset_index),
                    dataset,
                    int(episode_id),
                    int(valid_min),
                    int(valid_max),
                    diagnostics,
                )
            )

    if not episode_candidates:
        raise ValueError("The manifest-selected holdout contains no focused episodes.")

    episode_candidates.sort(
        key=lambda item: _focused_digest(
            contract,
            "episode",
            str(item[1].dataset_name),
            int(item[2]),
        )
    )
    stage_counts: Counter[int] = Counter()
    transition_window_counts: Counter[str] = Counter()
    required_set = set(required_subtasks)
    references: list[HeldoutWindowReference] = []
    for (
        dataset_index,
        dataset,
        episode_id,
        valid_min,
        valid_max,
        diagnostics,
    ) in episode_candidates:
        evaluable = [item for item in diagnostics if item.valid_action_elements > 0]
        base_pool = evaluable if evaluable else diagnostics
        open_to_close = [
            item
            for item in base_pool
            if item.open_to_close_transitions_h10 > 0
        ]
        close_to_open = [
            item
            for item in base_pool
            if item.close_to_open_transitions_h10 > 0
        ]
        transition_pool = [
            item
            for item in base_pool
            if item.open_to_close_transitions_h10 > 0
            or item.close_to_open_transitions_h10 > 0
        ]
        selection_pool = transition_pool or base_pool

        def transition_balance(item: _CandidateDiagnostic) -> int:
            directions: list[str] = []
            if item.open_to_close_transitions_h10 > 0:
                directions.append("open_to_close")
            if item.close_to_open_transitions_h10 > 0:
                directions.append("close_to_open")
            if not directions:
                return len(episode_candidates) + 1
            return min(transition_window_counts[direction] for direction in directions)

        selected = min(
            selection_pool,
            key=lambda item: (
                0 if item.anchor_subtask_index in required_set else 1,
                stage_counts[item.anchor_subtask_index]
                if item.anchor_subtask_index is not None
                else len(episode_candidates) + 1,
                transition_balance(item),
                -int(item.arm_movement_elements_h10 > 0),
                -item.arm_movement_elements_h10,
                -item.open_to_close_transitions_h10,
                -item.close_to_open_transitions_h10,
                _focused_digest(
                    contract,
                    "candidate",
                    str(dataset.dataset_name),
                    int(episode_id),
                    int(item.base_index),
                ),
            ),
        )
        if selected.anchor_subtask_index is not None:
            stage_counts[int(selected.anchor_subtask_index)] += 1
        if selected.open_to_close_transitions_h10 > 0:
            transition_window_counts["open_to_close"] += 1
        if selected.close_to_open_transitions_h10 > 0:
            transition_window_counts["close_to_open"] += 1
        references.append(
            HeldoutWindowReference(
                dataset_index=int(dataset_index),
                dataset_name=str(dataset.dataset_name),
                episode_id=int(episode_id),
                base_index=int(selected.base_index),
                valid_base_index_min=int(valid_min),
                valid_base_index_max=int(valid_max),
                structural_candidate_count=len(diagnostics),
                evaluable_candidate_count=len(evaluable),
                selection_pool_candidate_count=len(selection_pool),
                valid_action_timesteps=int(selected.valid_action_timesteps),
                valid_action_elements=int(selected.valid_action_elements),
                anchor_subtask_index=selected.anchor_subtask_index,
                action_subtask_indices=selected.action_subtask_indices,
                valid_action_elements_per_timestep=(
                    selected.valid_action_elements_per_timestep
                ),
                open_to_close_transitions_h10=int(
                    selected.open_to_close_transitions_h10
                ),
                close_to_open_transitions_h10=int(
                    selected.close_to_open_transitions_h10
                ),
                open_to_close_window_h10=(
                    selected.open_to_close_transitions_h10 > 0
                ),
                close_to_open_window_h10=(
                    selected.close_to_open_transitions_h10 > 0
                ),
                arm_movement_elements_h10=int(
                    selected.arm_movement_elements_h10
                ),
                arm_movement_hold_abs_h10=float(
                    selected.arm_movement_hold_abs_h10
                ),
            )
        )
    return tuple(
        sorted(
            references,
            key=lambda ref: (ref.dataset_name, ref.episode_id, ref.dataset_index),
        )
    )


def validate_global_eval_observation_count(
    *,
    holdout_episode_count: int,
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Require exactly one effective global training batch of holdout episodes."""

    values = {
        "holdout_episode_count": holdout_episode_count,
        "per_device_batch_size": per_device_batch_size,
        "world_size": world_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for name, value in values.items():
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    expected = (
        int(per_device_batch_size)
        * int(world_size)
        * int(gradient_accumulation_steps)
    )
    if int(holdout_episode_count) != expected:
        raise ValueError(
            "The manifest holdout must contain exactly one effective global "
            "training batch: "
            f"holdout episodes={int(holdout_episode_count)}, expected={expected} "
            f"({int(per_device_batch_size)} per device * {int(world_size)} ranks * "
            f"{int(gradient_accumulation_steps)} gradient accumulation steps)."
        )
    return expected


class DeterministicHeldoutEvalDataset(LeRobotMixtureDataset):
    """A separate eval mixture whose index is a fixed episode/window reference."""

    def __init__(
        self,
        source: LeRobotMixtureDataset,
        *,
        sampling_contract: EvaluationSamplingContract,
        action_dim: int,
        focused_subtasks: Sequence[int] | None = None,
        movement_threshold: float = 0.02,
        legacy_underfilled_eval: bool = False,
        legacy_excluded_zero_valid_episode_ids: Sequence[int] = (),
    ) -> None:
        if not isinstance(source, LeRobotMixtureDataset):
            raise TypeError(
                "Deterministic heldout evaluation currently supports only "
                f"LeRobotMixtureDataset, got {type(source).__name__}."
            )
        # The source was constructed independently with mode="eval".  Copying
        # its attributes avoids re-running expensive catalog/statistics setup;
        # it does not share with the separately constructed training dataset.
        self.__dict__.update(source.__dict__)
        self.mode = "eval"
        if isinstance(action_dim, bool) or int(action_dim) <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}.")
        self.heldout_eval_action_dim = int(action_dim)
        self.evaluation_sampling_contract = sampling_contract
        self.heldout_eval_view = "unbiased"
        self.legacy_underfilled_eval = bool(legacy_underfilled_eval)
        self.legacy_excluded_zero_valid_episode_ids = tuple(
            sorted(
                {
                    int(value)
                    for value in legacy_excluded_zero_valid_episode_ids
                }
            )
        )
        if (
            self.legacy_excluded_zero_valid_episode_ids
            and not self.legacy_underfilled_eval
        ):
            raise ValueError(
                "Legacy zero-valid episode exclusions require "
                "legacy_underfilled_eval=true."
            )
        self.focused_subtasks = (
            None
            if focused_subtasks is None
            else tuple(sorted({int(value) for value in focused_subtasks}))
        )
        self.heldout_eval_movement_threshold = float(movement_threshold)
        if (
            not np.isfinite(self.heldout_eval_movement_threshold)
            or self.heldout_eval_movement_threshold < 0
        ):
            raise ValueError(
                "Heldout eval movement_threshold must be finite and non-negative."
            )
        configure_deployment_current_frame_observation(self)
        _validate_sampling_offsets(self, sampling_contract)
        unbiased_references = select_heldout_window_references(
            self,
            sampling_contract,
            action_dim=self.heldout_eval_action_dim,
        )
        self.original_manifest_observation_count = len(unbiased_references)
        observed_zero_valid = tuple(
            reference
            for reference in unbiased_references
            if reference.valid_action_elements == 0
        )
        if self.legacy_underfilled_eval:
            observed_ids = tuple(
                sorted(reference.episode_id for reference in observed_zero_valid)
            )
            if observed_ids != self.legacy_excluded_zero_valid_episode_ids:
                raise ValueError(
                    "Legacy underfilled audit must exclude exactly the explicit "
                    "zero-supervision episode IDs with no replacement: observed="
                    f"{observed_ids}, configured="
                    f"{self.legacy_excluded_zero_valid_episode_ids}."
                )
            if not observed_ids:
                raise ValueError(
                    "Legacy underfilled audit requires at least one proven "
                    "zero-supervision episode."
                )
            self.legacy_excluded_zero_valid_episodes = tuple(
                {
                    "dataset_name": reference.dataset_name,
                    "episode_id": reference.episode_id,
                    "base_index": reference.base_index,
                    "reason": (
                        "no structurally valid window has a supervised action element"
                    ),
                }
                for reference in observed_zero_valid
            )
            excluded_identities = {
                (reference.dataset_name, reference.episode_id)
                for reference in observed_zero_valid
            }
            unbiased_references = tuple(
                reference
                for reference in unbiased_references
                if (reference.dataset_name, reference.episode_id)
                not in excluded_identities
            )
        else:
            self.legacy_excluded_zero_valid_episodes = ()
        self.heldout_window_references = unbiased_references
        self.heldout_window_digest = _canonical_sha256(
            {
                "algorithm": sampling_contract.algorithm,
                "observation_mode": sampling_contract.observation_mode,
                "evaluation_video_offsets": list(
                    sampling_contract.evaluation_video_offsets
                ),
                "action_offset_range_inclusive": list(
                    sampling_contract.action_offset_range_inclusive
                ),
                "frames_per_episode": sampling_contract.frames_per_episode,
                "seed_sha256": sampling_contract.seed_sha256,
                "windows": [asdict(reference) for reference in self.heldout_window_references],
            }
        )
        self.focused_window_references = None
        self.focused_window_digest = None
        if self.focused_subtasks is not None:
            focused_references = (
                select_focused_heldout_window_references(
                    self,
                    sampling_contract,
                    action_dim=self.heldout_eval_action_dim,
                    focused_subtasks=self.focused_subtasks,
                    movement_threshold=self.heldout_eval_movement_threshold,
                )
            )
            if self.legacy_underfilled_eval:
                excluded_identities = {
                    (str(item["dataset_name"]), int(item["episode_id"]))
                    for item in self.legacy_excluded_zero_valid_episodes
                }
                focused_references = tuple(
                    reference
                    for reference in focused_references
                    if (reference.dataset_name, reference.episode_id)
                    not in excluded_identities
                )
            if len(focused_references) != len(self.heldout_window_references):
                raise ValueError(
                    "Focused and unbiased heldout views differ in episode cardinality."
                )
            self.focused_window_references = focused_references
            self.focused_window_digest = _canonical_sha256(
                {
                    "algorithm": FOCUSED_EVAL_ALGORITHM,
                    "transition_horizon": FOCUSED_EVAL_TRANSITION_HORIZON,
                    "focused_subtasks": list(self.focused_subtasks),
                    "manifest_seed_sha256": sampling_contract.seed_sha256,
                    "unbiased_window_selection_sha256": self.heldout_window_digest,
                    "windows": [
                        asdict(reference)
                        for reference in self.focused_window_references
                    ],
                }
            )
        self.close_eval_caches()

    @classmethod
    def from_manifest(
        cls,
        source: LeRobotMixtureDataset,
        manifest_path: Path | str,
        *,
        action_dim: int,
        focused_subtasks: Sequence[int] | None = None,
        movement_threshold: float = 0.02,
        legacy_underfilled_eval: bool = False,
        legacy_excluded_zero_valid_episode_ids: Sequence[int] = (),
    ) -> "DeterministicHeldoutEvalDataset":
        return cls(
            source,
            sampling_contract=load_evaluation_sampling_contract(manifest_path),
            action_dim=action_dim,
            focused_subtasks=focused_subtasks,
            movement_threshold=movement_threshold,
            legacy_underfilled_eval=legacy_underfilled_eval,
            legacy_excluded_zero_valid_episode_ids=(
                legacy_excluded_zero_valid_episode_ids
            ),
        )

    def make_focused_view(self) -> "DeterministicHeldoutEvalDataset":
        if self.focused_window_references is None or self.focused_window_digest is None:
            raise ValueError(
                "Focused heldout references were not enabled when the dataset was built."
            )
        focused = copy.copy(self)
        focused.heldout_eval_view = "focused"
        focused.heldout_window_references = self.focused_window_references
        focused.heldout_window_digest = self.focused_window_digest
        return focused

    def __len__(self) -> int:
        return len(self.heldout_window_references)

    def _episode_split_provenance(self) -> list[dict[str, Any]]:
        """Return compact, content-addressed split/statistics evidence."""

        evidence: list[dict[str, Any]] = []
        keys = (
            "manifest_path",
            "manifest_sha256",
            "role",
            "selected_episode_count",
            "selected_episode_set_sha256",
            "selected_frame_count",
            "train_episode_count",
            "train_episode_set_sha256",
            "train_frame_count",
            "holdout_episode_count",
            "holdout_episode_set_sha256",
            "full_catalog_sha256",
            "train_statistics_path",
            "train_statistics_sha256",
        )
        for dataset in self.datasets:
            selection = getattr(dataset, "_episode_split_selection", None)
            if selection is None:
                raise ValueError(
                    f"Heldout dataset {dataset.dataset_name!r} has no split provenance."
                )
            provenance = selection.provenance()
            evidence.append(
                {
                    "dataset_name": str(dataset.dataset_name),
                    **{key: provenance[key] for key in keys},
                }
            )
        return evidence

    def sample_step(self, index: int):
        reference = getattr(self, "_active_heldout_window_reference", None)
        if reference is None:
            reference = self.heldout_window_references[int(index)]
        return (
            self.datasets[reference.dataset_index],
            reference.episode_id,
            reference.base_index,
        )

    def __getitem__(self, index: int) -> dict:
        eval_index = int(index)
        reference = self.heldout_window_references[eval_index]
        sentinel = object()
        previous = getattr(self, "_active_heldout_window_reference", sentinel)
        self._active_heldout_window_reference = reference
        try:
            # The parent has retry handling.  Pinning the active reference makes
            # every retry target the same episode/window instead of its fallback
            # random index, so a corrupt heldout sample fails closed.
            sample = super().__getitem__(eval_index)
        finally:
            if previous is sentinel:
                del self._active_heldout_window_reference
            else:
                self._active_heldout_window_reference = previous
        sample["_heldout_eval_index"] = eval_index
        sample["_heldout_eval_dataset_name"] = reference.dataset_name
        sample["_heldout_eval_episode_id"] = reference.episode_id
        sample["_heldout_eval_base_index"] = reference.base_index
        sample["_heldout_eval_view"] = self.heldout_eval_view
        if reference.anchor_subtask_index is None:
            raise ValueError(
                "Heldout RealMan control metrics require an integer subtask_index "
                f"at {reference.dataset_name}/{reference.episode_id}/"
                f"{reference.base_index}."
            )
        dataset = self.datasets[reference.dataset_index]
        raw_state = _current_raw_state(dataset, reference)
        hold_raw = _realman_hold_raw_action(raw_state, self.heldout_eval_action_dim)
        sample["_heldout_eval_hold_action"] = _normalize_action_vector(
            dataset,
            hold_raw,
        )
        sample["_heldout_eval_action_midpoint"] = _normalize_action_vector(
            dataset,
            _raw_action_midpoint(dataset),
        )
        sample["_heldout_eval_subtask_index"] = reference.anchor_subtask_index
        if len(reference.action_subtask_indices) != sample["action"].shape[0]:
            raise ValueError(
                "Heldout per-action subtask labels do not match the action horizon: "
                f"{len(reference.action_subtask_indices)} != "
                f"{sample['action'].shape[0]}."
            )
        sample["_heldout_eval_action_subtask_indices"] = np.asarray(
            reference.action_subtask_indices,
            dtype=np.int64,
        )
        return sample

    def sampling_report(self) -> dict[str, Any]:
        subtask_observation_counts = Counter(
            int(reference.anchor_subtask_index)
            for reference in self.heldout_window_references
            if reference.anchor_subtask_index is not None
        )
        subtask_evaluable_observation_counts = Counter(
            int(reference.anchor_subtask_index)
            for reference in self.heldout_window_references
            if reference.anchor_subtask_index is not None
            and reference.valid_action_elements > 0
        )
        action_horizon = max(
            (len(reference.action_subtask_indices) for reference in self.heldout_window_references),
            default=0,
        )
        report_horizons = tuple(
            horizon for horizon in (1, 5, 10, 20, 50) if horizon <= action_horizon
        )
        subtask_action_timestep_counts_by_horizon: dict[str, dict[str, int]] = {}
        subtask_valid_action_element_counts_by_horizon: dict[
            str, dict[str, int]
        ] = {}
        for horizon in report_horizons:
            timestep_counts: Counter[int] = Counter()
            valid_element_counts: Counter[int] = Counter()
            for reference in self.heldout_window_references:
                labels = reference.action_subtask_indices[:horizon]
                valid_counts = reference.valid_action_elements_per_timestep[:horizon]
                if len(labels) != horizon or len(valid_counts) != horizon:
                    raise ValueError(
                        "Heldout sampling report cannot prove per-action subtask "
                        f"coverage at h{horizon}; label/mask metadata is incomplete "
                        f"for {reference.dataset_name}/{reference.episode_id}."
                    )
                for label, valid_count in zip(labels, valid_counts):
                    timestep_counts[int(label)] += 1
                    valid_element_counts[int(label)] += int(valid_count)
            subtask_action_timestep_counts_by_horizon[str(horizon)] = {
                str(key): int(value)
                for key, value in sorted(timestep_counts.items())
            }
            subtask_valid_action_element_counts_by_horizon[str(horizon)] = {
                str(key): int(value)
                for key, value in sorted(valid_element_counts.items())
            }
        report = {
            "schema_version": EVAL_SAMPLING_REPORT_SCHEMA_VERSION,
            "purpose": (
                "one_window_per_manifest_holdout_episode_checkpoint_eval"
                if self.heldout_eval_view == "unbiased"
                else "h10_transition_stage_focused_manifest_holdout_checkpoint_eval"
            ),
            "view": self.heldout_eval_view,
            "algorithm": (
                self.evaluation_sampling_contract.algorithm
                if self.heldout_eval_view == "unbiased"
                else FOCUSED_EVAL_ALGORITHM
            ),
            "observation_mode": self.evaluation_sampling_contract.observation_mode,
            "evaluation_video_offsets": list(
                self.evaluation_sampling_contract.evaluation_video_offsets
            ),
            "action_offset_range_inclusive": list(
                self.evaluation_sampling_contract.action_offset_range_inclusive
            ),
            "frames_per_episode": self.evaluation_sampling_contract.frames_per_episode,
            "seed_sha256": self.evaluation_sampling_contract.seed_sha256,
            "observation_count": len(self),
            "action_evaluable_observation_count": sum(
                reference.valid_action_elements > 0
                for reference in self.heldout_window_references
            ),
            "action_dim": self.heldout_eval_action_dim,
            "valid_action_timestep_count": sum(
                reference.valid_action_timesteps
                for reference in self.heldout_window_references
            ),
            "valid_action_element_count": sum(
                reference.valid_action_elements
                for reference in self.heldout_window_references
            ),
            "subtask_observation_counts": {
                str(key): int(value)
                for key, value in sorted(subtask_observation_counts.items())
            },
            "subtask_evaluable_observation_counts": {
                str(key): int(value)
                for key, value in sorted(
                    subtask_evaluable_observation_counts.items()
                )
            },
            "subtask_action_timestep_counts_by_horizon": (
                subtask_action_timestep_counts_by_horizon
            ),
            "subtask_valid_action_element_counts_by_horizon": (
                subtask_valid_action_element_counts_by_horizon
            ),
            "zero_valid_action_episodes": [
                {
                    "dataset_name": reference.dataset_name,
                    "episode_id": reference.episode_id,
                    "base_index": reference.base_index,
                }
                for reference in self.heldout_window_references
                if reference.valid_action_elements == 0
            ],
            "window_selection_sha256": self.heldout_window_digest,
            "episode_split_provenance": self._episode_split_provenance(),
            "production_valid": not self.legacy_underfilled_eval,
            "checkpoint_selection_eligible": not self.legacy_underfilled_eval,
            "windows": [asdict(reference) for reference in self.heldout_window_references],
        }
        if self.legacy_underfilled_eval:
            report["legacy_underfilled_holdout"] = {
                "enabled": True,
                "original_manifest_observation_count": (
                    self.original_manifest_observation_count
                ),
                "evaluated_observation_count": len(self),
                "excluded_zero_valid_episodes": list(
                    self.legacy_excluded_zero_valid_episodes
                ),
                "replacement_episode_ids": [],
                "no_replacement_no_training_leak": True,
                "reason": (
                    "Historical checkpoint audit only: the frozen original "
                    "holdout contained zero-supervision episodes."
                ),
            }
        if self.heldout_eval_view == "focused":
            report.update(
                {
                    "open_to_close_transition_count_h10": sum(
                        reference.open_to_close_transitions_h10
                        for reference in self.heldout_window_references
                    ),
                    "close_to_open_transition_count_h10": sum(
                        reference.close_to_open_transitions_h10
                        for reference in self.heldout_window_references
                    ),
                    "open_to_close_transition_window_count_h10": sum(
                        reference.open_to_close_window_h10
                        for reference in self.heldout_window_references
                    ),
                    "close_to_open_transition_window_count_h10": sum(
                        reference.close_to_open_window_h10
                        for reference in self.heldout_window_references
                    ),
                    "arm_movement_element_count_h10": sum(
                        reference.arm_movement_elements_h10
                        for reference in self.heldout_window_references
                    ),
                    "arm_movement_hold_abs_sum_h10": sum(
                        reference.arm_movement_hold_abs_h10
                        for reference in self.heldout_window_references
                    ),
                    "movement_threshold_normalized": (
                        self.heldout_eval_movement_threshold
                    ),
                    "focused_subtasks": list(self.focused_subtasks or ()),
                }
            )
        return report

    def close_eval_caches(self) -> None:
        """Release eval-only parquet/video readers after selection or a pass."""

        for dataset in self.datasets:
            close_parquet = getattr(dataset, "close_parquet_cache", None)
            if callable(close_parquet):
                close_parquet()
            close_video = getattr(dataset, "close_video_readers", None)
            if callable(close_video):
                close_video()

    def save_sampling_report(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.sampling_report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def make_torch_generator(self) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.evaluation_sampling_contract.torch_seed)
        return generator


def build_heldout_eval_dataset(
    source: LeRobotMixtureDataset,
    *,
    manifest_path: Path | str,
    action_dim: int,
    focused_subtasks: Sequence[int] | None = None,
    movement_threshold: float = 0.02,
    legacy_underfilled_eval: bool = False,
    legacy_excluded_zero_valid_episode_ids: Sequence[int] = (),
) -> DeterministicHeldoutEvalDataset:
    return DeterministicHeldoutEvalDataset.from_manifest(
        source,
        manifest_path,
        action_dim=action_dim,
        focused_subtasks=focused_subtasks,
        movement_threshold=movement_threshold,
        legacy_underfilled_eval=legacy_underfilled_eval,
        legacy_excluded_zero_valid_episode_ids=(
            legacy_excluded_zero_valid_episode_ids
        ),
    )


def sampling_seed_from_dataset(dataset: Any, default: int = 0) -> int:
    """Resolve the dedicated inference seed through common loader wrappers."""

    visited: set[int] = set()
    pending = [dataset]
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        contract = getattr(current, "evaluation_sampling_contract", None)
        if contract is not None:
            return int(contract.torch_seed)
        for attr in ("dataset", "base_dataloader", "dataloader"):
            nested = getattr(current, attr, None)
            if nested is not None:
                pending.append(nested)
    return int(default)
