"""Versioned action-representation contracts shared by data and deployment.

This module intentionally depends only on NumPy and the Python standard
library.  Robot clients and checkpoint metadata validation must be able to use
the exact same conversion code as training without importing the dataloader or
Torch stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def split_manifest_sha256_without_statistics_binding(
    payload: Mapping[str, Any],
) -> str:
    """Hash an immutable split without its circular statistics back-link."""

    stripped = json.loads(json.dumps(payload, allow_nan=False))
    datasets = stripped.get("datasets", [])
    if isinstance(datasets, list):
        for entry in datasets:
            if isinstance(entry, dict):
                entry.pop("action_representation_statistics", None)
    encoded = json.dumps(
        stripped,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


JOINT_DELTA_GRIPPER_ABSOLUTE = "joint_delta_gripper_absolute"
Q01_Q99_UNCLIPPED = "q01_q99_unclipped"
CHUNK_START_STATE = "chunk_start_state"

REALMAN_POLICY_DIM = 18
REALMAN_ACTION_HORIZON = 50
REALMAN_SOURCE_STATE_DIM = 19
# Some canonicalized RealMan captures append two diagnostic torque channels
# after the 19-D control state. They are observations only and are deliberately
# excluded from the policy representation.
REALMAN_DIAGNOSTIC_SOURCE_STATE_DIM = 21
REALMAN_SUPPORTED_SOURCE_STATE_DIMS = (
    REALMAN_SOURCE_STATE_DIM,
    REALMAN_DIAGNOSTIC_SOURCE_STATE_DIM,
)
REALMAN_SOURCE_ACTION_DIM = 22
REALMAN_STATE_SOURCE_INDICES = tuple(range(18))
REALMAN_ACTION_SOURCE_INDICES = tuple(range(16)) + (19, 20)
# Canonical semantic-flat layouts are 53-D state and 49-D action.  Both use
# the same spans for the channels selected by the RealMan policy:
# left_arm_joint=0:7, right_arm_joint=7:14, left_hand=28:34,
# right_hand=34:40, and neck=40:42.
CANONICAL_REALMAN_STATE_DIM = 53
CANONICAL_REALMAN_ACTION_DIM = 49
CANONICAL_REALMAN_STATE_SOURCE_INDICES = (
    *range(0, 7),
    28,
    *range(7, 14),
    34,
    *range(40, 42),
)
CANONICAL_REALMAN_ACTION_SOURCE_INDICES = (
    *range(0, 7),
    28,
    *range(7, 14),
    34,
    *range(40, 42),
)
REALMAN_ABSOLUTE_ACTION_INDICES = (7, 15)
# -1 denotes a native/absolute action channel.  The other entries are indices
# into the selected 18-D policy state.
REALMAN_ACTION_TO_STATE_INDICES = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    -1,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    -1,
    16,
    17,
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ActionRepresentationContract:
    """Complete, serializable contract for mixed absolute/delta actions."""

    schema_version: int
    action_type: str
    delta_anchor: str
    action_horizon: int
    source_state_dim: int
    source_action_dim: int
    state_source_indices: tuple[int, ...]
    action_source_indices: tuple[int, ...]
    state_dim: int
    action_dim: int
    action_to_state_indices: tuple[int, ...]
    absolute_action_indices: tuple[int, ...]
    normalization: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported action representation schema {self.schema_version}; expected 1."
            )
        if self.action_type != JOINT_DELTA_GRIPPER_ABSOLUTE:
            raise ValueError(f"Unsupported action representation {self.action_type!r}.")
        if self.delta_anchor != CHUNK_START_STATE:
            raise ValueError(
                f"Unsupported delta anchor {self.delta_anchor!r}; expected {CHUNK_START_STATE!r}."
            )
        if self.action_horizon <= 0 or self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Action horizon and state/action dimensions must be positive.")
        if self.source_state_dim <= 0 or self.source_action_dim <= 0:
            raise ValueError("Source state/action dimensions must be positive.")
        if len(self.state_source_indices) != self.state_dim:
            raise ValueError("state_source_indices width does not match state_dim.")
        if len(self.action_source_indices) != self.action_dim:
            raise ValueError("action_source_indices width does not match action_dim.")
        if len(set(self.state_source_indices)) != len(self.state_source_indices):
            raise ValueError("state_source_indices must be unique.")
        if len(set(self.action_source_indices)) != len(self.action_source_indices):
            raise ValueError("action_source_indices must be unique.")
        if any(index < 0 or index >= self.source_state_dim for index in self.state_source_indices):
            raise ValueError("state_source_indices contains an out-of-range channel.")
        if any(index < 0 or index >= self.source_action_dim for index in self.action_source_indices):
            raise ValueError("action_source_indices contains an out-of-range channel.")
        if len(self.action_to_state_indices) != self.action_dim:
            raise ValueError(
                "action_to_state_indices width does not match action_dim: "
                f"{len(self.action_to_state_indices)} != {self.action_dim}."
            )
        mapping = np.asarray(self.action_to_state_indices, dtype=np.int64)
        if np.any(mapping < -1) or np.any(mapping >= self.state_dim):
            raise ValueError(
                f"Action-to-state mapping is invalid for state_dim={self.state_dim}: "
                f"{self.action_to_state_indices}."
            )
        absolute = tuple(sorted(set(int(value) for value in self.absolute_action_indices)))
        if absolute != self.absolute_action_indices:
            raise ValueError("absolute_action_indices must be sorted and unique.")
        if any(value < 0 or value >= self.action_dim for value in absolute):
            raise ValueError("absolute_action_indices contains an out-of-range channel.")
        native = tuple(np.flatnonzero(mapping < 0).astype(int).tolist())
        if native != absolute:
            raise ValueError(
                "Every native channel must be explicitly declared absolute and vice versa: "
                f"mapping native={native}, absolute={absolute}."
            )
        if self.normalization != Q01_Q99_UNCLIPPED:
            raise ValueError(
                f"Unsupported action normalization {self.normalization!r}; "
                f"expected {Q01_Q99_UNCLIPPED!r}."
            )

    @property
    def delta_action_indices(self) -> tuple[int, ...]:
        mapping = np.asarray(self.action_to_state_indices, dtype=np.int64)
        return tuple(np.flatnonzero(mapping >= 0).astype(int).tolist())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_type": self.action_type,
            "delta_anchor": self.delta_anchor,
            "action_horizon": self.action_horizon,
            "source_state_dim": self.source_state_dim,
            "source_action_dim": self.source_action_dim,
            "state_source_indices": list(self.state_source_indices),
            "action_source_indices": list(self.action_source_indices),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_to_state_indices": list(self.action_to_state_indices),
            "absolute_action_indices": list(self.absolute_action_indices),
            "normalization": self.normalization,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionRepresentationContract":
        return cls(
            schema_version=int(payload["schema_version"]),
            action_type=str(payload["action_type"]),
            delta_anchor=str(payload["delta_anchor"]),
            action_horizon=int(payload["action_horizon"]),
            source_state_dim=int(payload["source_state_dim"]),
            source_action_dim=int(payload["source_action_dim"]),
            state_source_indices=tuple(int(value) for value in payload["state_source_indices"]),
            action_source_indices=tuple(int(value) for value in payload["action_source_indices"]),
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            action_to_state_indices=tuple(
                int(value) for value in payload["action_to_state_indices"]
            ),
            absolute_action_indices=tuple(
                int(value) for value in payload["absolute_action_indices"]
            ),
            normalization=str(payload["normalization"]),
        )

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest()


REALMAN_18D_ACTION_CONTRACT = ActionRepresentationContract(
    schema_version=1,
    action_type=JOINT_DELTA_GRIPPER_ABSOLUTE,
    delta_anchor=CHUNK_START_STATE,
    action_horizon=REALMAN_ACTION_HORIZON,
    source_state_dim=REALMAN_SOURCE_STATE_DIM,
    source_action_dim=REALMAN_SOURCE_ACTION_DIM,
    state_source_indices=REALMAN_STATE_SOURCE_INDICES,
    action_source_indices=REALMAN_ACTION_SOURCE_INDICES,
    state_dim=REALMAN_POLICY_DIM,
    action_dim=REALMAN_POLICY_DIM,
    action_to_state_indices=REALMAN_ACTION_TO_STATE_INDICES,
    absolute_action_indices=REALMAN_ABSOLUTE_ACTION_INDICES,
    normalization=Q01_Q99_UNCLIPPED,
)


def select_realman_policy_state(source_state: Any) -> np.ndarray:
    array = np.asarray(source_state)
    if array.shape[-1] == REALMAN_POLICY_DIM:
        return np.ascontiguousarray(array, dtype=np.float32)
    if array.shape[-1] not in REALMAN_SUPPORTED_SOURCE_STATE_DIMS:
        raise ValueError(
            "RealMan source state must be already-selected "
            f"{REALMAN_POLICY_DIM}D or one of the supported raw widths "
            f"{REALMAN_SUPPORTED_SOURCE_STATE_DIMS}, got {array.shape}."
        )
    return np.ascontiguousarray(
        array[..., np.asarray(REALMAN_STATE_SOURCE_INDICES, dtype=np.int64)],
        dtype=np.float32,
    )


def select_realman_policy_actions(source_actions: Any) -> np.ndarray:
    array = np.asarray(source_actions)
    if array.shape[-1] == REALMAN_POLICY_DIM:
        return np.ascontiguousarray(array, dtype=np.float32)
    if array.shape[-1] != REALMAN_SOURCE_ACTION_DIM:
        raise ValueError(
            f"RealMan source action must be {REALMAN_SOURCE_ACTION_DIM}D or already "
            f"{REALMAN_POLICY_DIM}D, got {array.shape}."
        )
    return np.ascontiguousarray(
        array[..., np.asarray(REALMAN_ACTION_SOURCE_INDICES, dtype=np.int64)],
        dtype=np.float32,
    )


def _select_canonical_realman_values(
    values: Any,
    *,
    source_dim: int,
    source_indices: Sequence[int],
    label: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != source_dim:
        raise ValueError(
            f"Canonical RealMan {label} must have final width {source_dim}, "
            f"got {array.shape}."
        )
    selected = np.asarray(
        array[..., np.asarray(source_indices, dtype=np.int64)],
        dtype=np.float32,
    )
    if not np.isfinite(selected).all():
        raise ValueError(f"Canonical RealMan {label} contains non-finite values.")
    return np.ascontiguousarray(selected)


def _select_canonical_realman_mask(
    mask: Any,
    *,
    source_dim: int,
    source_indices: Sequence[int],
    label: str,
) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 0 or array.shape[-1] != source_dim:
        raise ValueError(
            f"Canonical RealMan {label} mask must have final width {source_dim}, "
            f"got {array.shape}."
        )
    if array.dtype != np.bool_:
        if not np.isin(array, (0, 1)).all():
            raise ValueError(
                f"Canonical RealMan {label} mask must contain only booleans."
            )
        array = array.astype(bool, copy=False)
    return np.ascontiguousarray(
        array[..., np.asarray(source_indices, dtype=np.int64)],
        dtype=bool,
    )


def select_canonical_realman_policy_state(values: Any) -> np.ndarray:
    """Project canonical semantic-flat 53-D state values to RealMan 18-D."""

    return _select_canonical_realman_values(
        values,
        source_dim=CANONICAL_REALMAN_STATE_DIM,
        source_indices=CANONICAL_REALMAN_STATE_SOURCE_INDICES,
        label="state",
    )


def select_canonical_realman_policy_actions(values: Any) -> np.ndarray:
    """Project canonical semantic-flat 49-D action values to RealMan 18-D."""

    return _select_canonical_realman_values(
        values,
        source_dim=CANONICAL_REALMAN_ACTION_DIM,
        source_indices=CANONICAL_REALMAN_ACTION_SOURCE_INDICES,
        label="action",
    )


def select_canonical_realman_policy_state_mask(mask: Any) -> np.ndarray:
    """Project a canonical 53-D state availability mask to RealMan 18-D."""

    return _select_canonical_realman_mask(
        mask,
        source_dim=CANONICAL_REALMAN_STATE_DIM,
        source_indices=CANONICAL_REALMAN_STATE_SOURCE_INDICES,
        label="state",
    )


def select_canonical_realman_policy_action_mask(mask: Any) -> np.ndarray:
    """Project a canonical 49-D action availability mask to RealMan 18-D."""

    return _select_canonical_realman_mask(
        mask,
        source_dim=CANONICAL_REALMAN_ACTION_DIM,
        source_indices=CANONICAL_REALMAN_ACTION_SOURCE_INDICES,
        label="action",
    )


def _broadcast_anchor(
    actions: np.ndarray,
    current_state: Any,
    contract: ActionRepresentationContract,
) -> tuple[np.ndarray, np.ndarray]:
    if actions.shape[-1] != contract.action_dim:
        raise ValueError(
            f"Action dim {actions.shape[-1]} does not match contract {contract.action_dim}."
        )
    state = np.asarray(current_state, dtype=np.float32)
    if state.shape[-1] != contract.state_dim:
        raise ValueError(
            f"State dim {state.shape[-1]} does not match contract {contract.state_dim}."
        )
    mapping = np.asarray(contract.action_to_state_indices, dtype=np.int64)
    delta_mask = mapping >= 0
    reference = state[..., mapping[delta_mask]]
    while reference.ndim < actions.ndim:
        reference = np.expand_dims(reference, axis=-2)
    return delta_mask, reference


def encode_actions(
    absolute_actions: Any,
    current_state: Any,
    contract: ActionRepresentationContract = REALMAN_18D_ACTION_CONTRACT,
) -> np.ndarray:
    """Convert absolute actions to mixed chunk-start deltas/absolute channels."""

    actions = np.asarray(absolute_actions, dtype=np.float32)
    if actions.ndim == 0:
        raise ValueError("Actions must have a feature dimension.")
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain non-finite values.")
    delta_mask, reference = _broadcast_anchor(actions, current_state, contract)
    converted = actions.copy()
    converted[..., delta_mask] -= reference
    return np.ascontiguousarray(converted)


def decode_actions(
    policy_actions: Any,
    current_state: Any,
    contract: ActionRepresentationContract = REALMAN_18D_ACTION_CONTRACT,
) -> np.ndarray:
    """Restore absolute actions from mixed chunk-start deltas/absolute channels."""

    actions = np.asarray(policy_actions, dtype=np.float32)
    if actions.ndim == 0:
        raise ValueError("Actions must have a feature dimension.")
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain non-finite values.")
    delta_mask, reference = _broadcast_anchor(actions, current_state, contract)
    converted = actions.copy()
    converted[..., delta_mask] += reference
    return np.ascontiguousarray(converted)


def normalize_q01_q99_unclipped(values: Any, statistics: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    q01 = np.asarray(statistics["q01"], dtype=np.float32)
    q99 = np.asarray(statistics["q99"], dtype=np.float32)
    if q01.shape != (array.shape[-1],) or q99.shape != q01.shape:
        raise ValueError(
            f"Quantile statistics do not match values {array.shape}: q01={q01.shape}, q99={q99.shape}."
        )
    return (
        (array - q01) / (q99 - q01 + np.float32(1e-6)) * 2.0 - 1.0
    ).astype(np.float32, copy=False)


def unnormalize_q01_q99(values: Any, statistics: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    q01 = np.asarray(statistics["q01"], dtype=np.float32)
    q99 = np.asarray(statistics["q99"], dtype=np.float32)
    if q01.shape != (array.shape[-1],) or q99.shape != q01.shape:
        raise ValueError(
            f"Quantile statistics do not match values {array.shape}: q01={q01.shape}, q99={q99.shape}."
        )
    return (
        (array + 1.0) * 0.5 * (q99 - q01 + np.float32(1e-6)) + q01
    ).astype(
        np.float32, copy=False
    )


class PiCompatibleRunningStats:
    """OpenPI-compatible streaming moments and 5,000-bin quantiles.

    Update order and batch boundaries are part of OpenPI's histogram behavior,
    so parity callers must feed episodes in the same deterministic order.
    """

    def __init__(self, *, num_quantile_bins: int = 5000) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.mean_of_squares: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None
        self.histograms: list[np.ndarray] | None = None
        self.bin_edges: list[np.ndarray] | None = None
        self.num_quantile_bins = int(num_quantile_bins)
        if self.num_quantile_bins <= 1:
            raise ValueError("num_quantile_bins must be greater than one.")

    def update(self, batch: Any) -> None:
        values = np.asarray(batch)
        if values.ndim < 2:
            raise ValueError(f"Statistics batch must end in a feature dimension: {values.shape}.")
        # Keep the input dtype exactly as OpenPI does.  In particular, its
        # float32 means and expanding histogram bounds are part of the parity
        # contract rather than an opportunity for a higher-precision rewrite.
        values = values.reshape(-1, values.shape[-1])
        if values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("Statistics batch must be non-empty and finite.")
        num_elements, width = values.shape
        if self.count == 0:
            self.mean = np.mean(values, axis=0)
            self.mean_of_squares = np.mean(values**2, axis=0)
            self.minimum = np.min(values, axis=0)
            self.maximum = np.max(values, axis=0)
            self.histograms = [np.zeros(self.num_quantile_bins) for _ in range(width)]
            self.bin_edges = [
                np.linspace(
                    self.minimum[index] - 1e-10,
                    self.maximum[index] + 1e-10,
                    self.num_quantile_bins + 1,
                )
                for index in range(width)
            ]
        else:
            assert self.mean is not None
            if width != self.mean.size:
                raise ValueError(
                    f"Statistics width changed from {self.mean.size} to {width}."
                )
            assert self.minimum is not None and self.maximum is not None
            new_min = np.min(values, axis=0)
            new_max = np.max(values, axis=0)
            bounds_changed = bool(
                np.any(new_min < self.minimum) or np.any(new_max > self.maximum)
            )
            self.minimum = np.minimum(self.minimum, new_min)
            self.maximum = np.maximum(self.maximum, new_max)
            if bounds_changed:
                self._adjust_histograms()

        self.count += num_elements
        batch_mean = np.mean(values, axis=0)
        batch_mean_of_squares = np.mean(values**2, axis=0)
        assert self.mean is not None and self.mean_of_squares is not None
        self.mean += (batch_mean - self.mean) * (num_elements / self.count)
        self.mean_of_squares += (
            batch_mean_of_squares - self.mean_of_squares
        ) * (num_elements / self.count)
        self._update_histograms(values)

    def _adjust_histograms(self) -> None:
        assert self.histograms is not None and self.bin_edges is not None
        assert self.minimum is not None and self.maximum is not None
        for index in range(len(self.histograms)):
            old_edges = self.bin_edges[index]
            new_edges = np.linspace(
                self.minimum[index],
                self.maximum[index],
                self.num_quantile_bins + 1,
            )
            new_histogram, _ = np.histogram(
                old_edges[:-1], bins=new_edges, weights=self.histograms[index]
            )
            self.histograms[index] = new_histogram
            self.bin_edges[index] = new_edges

    def _update_histograms(self, values: np.ndarray) -> None:
        assert self.histograms is not None and self.bin_edges is not None
        for index in range(values.shape[1]):
            histogram, _ = np.histogram(values[:, index], bins=self.bin_edges[index])
            self.histograms[index] += histogram

    def _quantiles(self, quantiles: Sequence[float]) -> list[np.ndarray]:
        if self.count < 2:
            raise ValueError("Cannot compute statistics for fewer than two vectors.")
        assert self.histograms is not None and self.bin_edges is not None
        results = []
        for quantile in quantiles:
            target_count = float(quantile) * self.count
            values = []
            for histogram, edges in zip(self.histograms, self.bin_edges, strict=True):
                index = int(np.searchsorted(np.cumsum(histogram), target_count))
                values.append(edges[index])
            results.append(np.asarray(values, dtype=np.float64))
        return results

    def get_statistics(self) -> dict[str, Any]:
        if self.count < 2:
            raise ValueError("Cannot compute statistics for fewer than two vectors.")
        assert self.mean is not None and self.mean_of_squares is not None
        assert self.minimum is not None and self.maximum is not None
        q01, q99 = self._quantiles((0.01, 0.99))
        variance = self.mean_of_squares - self.mean**2
        return {
            "count": [int(self.count)],
            "mean": self.mean.tolist(),
            "std": np.sqrt(np.maximum(0.0, variance)).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


class PiCompatibleMaskedRunningStats:
    """Per-channel OpenPI statistics for representations with missing channels.

    OpenPI's histogram and moment calculations are channel-independent.  One
    scalar :class:`PiCompatibleRunningStats` per channel therefore preserves
    the exact 5,000-bin algorithm while allowing, for example, RealSource
    samples to omit both head channels instead of treating zero padding as
    observed data.
    """

    def __init__(self, width: int, *, num_quantile_bins: int = 5000) -> None:
        if isinstance(width, bool) or not isinstance(width, (int, np.integer)):
            raise ValueError("Masked statistics width must be an integer.")
        self.width = int(width)
        if self.width <= 0:
            raise ValueError("Masked statistics width must be positive.")
        self.num_quantile_bins = int(num_quantile_bins)
        self._channels = [
            PiCompatibleRunningStats(num_quantile_bins=self.num_quantile_bins)
            for _ in range(self.width)
        ]

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(channel.count for channel in self._channels)

    def update(self, batch: Any, mask: Any | None = None) -> None:
        values = np.asarray(batch)
        if values.ndim < 2 or values.shape[-1] != self.width:
            raise ValueError(
                "Masked statistics batch must end in the configured feature "
                f"width {self.width}, got {values.shape}."
            )
        if mask is None:
            valid = np.ones(values.shape, dtype=bool)
        else:
            valid = np.asarray(mask)
            if valid.shape != values.shape:
                raise ValueError(
                    "Masked statistics values/mask shapes differ: "
                    f"{values.shape} != {valid.shape}."
                )
            if valid.dtype != np.bool_:
                if not np.isin(valid, (0, 1)).all():
                    raise ValueError(
                        "Masked statistics mask must contain only booleans."
                    )
                valid = valid.astype(bool, copy=False)

        flattened_values = values.reshape(-1, self.width)
        flattened_valid = valid.reshape(-1, self.width)
        if flattened_values.shape[0] == 0:
            raise ValueError("Masked statistics batch must be non-empty.")
        if not np.isfinite(flattened_values[flattened_valid]).all():
            raise ValueError(
                "Masked statistics batch contains non-finite observed values."
            )

        for index, channel in enumerate(self._channels):
            observed = flattened_values[flattened_valid[:, index], index]
            if observed.size:
                # Keep the source dtype and episode-level update boundary just
                # like PiCompatibleRunningStats/OpenPI.
                channel.update(observed.reshape(-1, 1))

    def get_statistics(self) -> dict[str, Any]:
        insufficient = [
            index for index, channel in enumerate(self._channels) if channel.count < 2
        ]
        if insufficient:
            raise ValueError(
                "Cannot compute masked statistics; fewer than two observed "
                f"values for channels {insufficient}."
            )
        per_channel = [channel.get_statistics() for channel in self._channels]
        return {
            "count": [int(channel.count) for channel in self._channels],
            **{
                name: [
                    float(statistics[name][0]) for statistics in per_channel
                ]
                for name in ("mean", "std", "min", "max", "q01", "q99")
            },
        }


OPENPI_REALMAN_UNION_POPULATION_SCHEMA = (
    "openpi-realman-18d-union-population-v1"
)
OPENPI_REALMAN_UNION_STATISTICS_SCHEMA = (
    "openpi-realman-18d-union-statistics-v1"
)
OPENPI_REALMAN_UNION_LEDGER_SCHEMA = "openpi-realman-18d-union-ledger-v1"
OPENPI_REALMAN_EPISODE_CONTENT_SCHEMA = "realman-18d-episode-content-v1"
OPENPI_REALMAN_UNION_DEDUP_ALGORITHM = (
    "episode-content-sha256-plus-base-frame-v1"
)
OPENPI_REALMAN_UNION_STATISTIC_NAMES = (
    "count",
    "mean",
    "std",
    "min",
    "max",
    "q01",
    "q99",
)


def deterministic_json_bytes(payload: Any) -> bytes:
    """Return the canonical, newline-terminated bytes used by artifacts."""

    return _canonical_json_bytes(payload) + b"\n"


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest.")
    return value


def _validate_union_statistic(
    value: Any,
    *,
    label: str,
    width: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    if set(value) != set(OPENPI_REALMAN_UNION_STATISTIC_NAMES):
        raise ValueError(
            f"{label} must contain exactly "
            f"{list(OPENPI_REALMAN_UNION_STATISTIC_NAMES)}."
        )
    output: dict[str, Any] = {}
    for name in OPENPI_REALMAN_UNION_STATISTIC_NAMES:
        raw = value[name]
        if not isinstance(raw, list) or len(raw) != width:
            raise ValueError(f"{label}.{name} must have width {width}.")
        if name == "count":
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, np.integer))
                or int(item) < 2
                for item in raw
            ):
                raise ValueError(
                    f"{label}.count must contain integers greater than one."
                )
            output[name] = [int(item) for item in raw]
            continue
        array = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"{label}.{name} contains non-finite values.")
        if name == "std" and np.any(array < 0):
            raise ValueError(f"{label}.std contains a negative value.")
        output[name] = array.tolist()
    minimum = np.asarray(output["min"])
    maximum = np.asarray(output["max"])
    q01 = np.asarray(output["q01"])
    q99 = np.asarray(output["q99"])
    if np.any(minimum > maximum) or np.any(q01 > q99):
        raise ValueError(f"{label} has inconsistent min/max or q01/q99 values.")
    return output


def _slice_union_statistic(
    statistics: Mapping[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    return {
        name: list(statistics[name][start:end])
        for name in OPENPI_REALMAN_UNION_STATISTIC_NAMES
    }


def validate_openpi_realman_union_statistics(
    payload: Any,
) -> dict[str, Any]:
    """Validate and return one immutable 18-D union-statistics artifact."""

    if not isinstance(payload, Mapping):
        raise ValueError("Union statistics artifact root must be a JSON object.")
    required_root = {
        "schema",
        "contract",
        "contract_sha256",
        "normalization",
        "algorithm",
        "population",
        "selected",
        "modalities",
    }
    if set(payload) != required_root:
        raise ValueError(
            "Union statistics artifact must contain exactly "
            f"{sorted(required_root)}; got {sorted(payload)}."
        )
    if payload["schema"] != OPENPI_REALMAN_UNION_STATISTICS_SCHEMA:
        raise ValueError(
            "Unsupported union statistics schema: "
            f"{payload['schema']!r}."
        )
    if payload["contract"] != REALMAN_18D_ACTION_CONTRACT.to_dict():
        raise ValueError(
            "Union statistics action representation contract is not the exact "
            "RealMan 18-D contract."
        )
    if payload["contract_sha256"] != REALMAN_18D_ACTION_CONTRACT.sha256():
        raise ValueError("Union statistics contract SHA-256 is invalid.")
    if payload["normalization"] != Q01_Q99_UNCLIPPED:
        raise ValueError(
            f"Union statistics normalization must be {Q01_Q99_UNCLIPPED!r}."
        )

    algorithm = payload["algorithm"]
    expected_algorithm = {
        "quantile_bins": 5000,
        "update_batch": "one_episode",
        "action_horizon": REALMAN_ACTION_HORIZON,
        "action_padding": "repeat_episode_final_frame",
        "deduplication": OPENPI_REALMAN_UNION_DEDUP_ALGORITHM,
    }
    if algorithm != expected_algorithm:
        raise ValueError(
            "Union statistics algorithm contract is invalid: "
            f"{algorithm!r} != {expected_algorithm!r}."
        )

    population = payload["population"]
    if not isinstance(population, Mapping):
        raise ValueError("Union statistics population must be a JSON object.")
    required_population = {
        "schema",
        "manifest_sha256",
        "ledger_sha256",
        "holdout_manifest_sha256",
        "source_order",
        "sources",
        "candidate_base_frames",
        "unique_base_frames",
        "duplicate_base_frames",
        "holdout_excluded_base_frames",
    }
    if set(population) != required_population:
        raise ValueError(
            "Union statistics population must contain exactly "
            f"{sorted(required_population)}."
        )
    if population["schema"] != OPENPI_REALMAN_UNION_POPULATION_SCHEMA:
        raise ValueError("Union statistics population schema is invalid.")
    for name in (
        "manifest_sha256",
        "ledger_sha256",
        "holdout_manifest_sha256",
    ):
        _require_sha256(population[name], label=f"population.{name}")
    source_order = population["source_order"]
    sources = population["sources"]
    if (
        not isinstance(source_order, list)
        or not source_order
        or any(not isinstance(value, str) or not value for value in source_order)
        or len(source_order) != len(set(source_order))
    ):
        raise ValueError("population.source_order must contain unique source IDs.")
    if (
        not isinstance(sources, list)
        or [source.get("id") if isinstance(source, Mapping) else None for source in sources]
        != source_order
    ):
        raise ValueError(
            "population.sources must be ordered exactly like source_order."
        )
    for source_index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"population.sources[{source_index}] must be an object.")
        for digest_name in ("catalog_sha256", "selected_content_sha256"):
            _require_sha256(
                source.get(digest_name),
                label=f"population.sources[{source_index}].{digest_name}",
            )
    counts = {}
    for name in (
        "candidate_base_frames",
        "unique_base_frames",
        "duplicate_base_frames",
        "holdout_excluded_base_frames",
    ):
        raw = population[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"population.{name} must be a non-negative integer.")
        counts[name] = raw
    if (
        counts["unique_base_frames"]
        + counts["duplicate_base_frames"]
        + counts["holdout_excluded_base_frames"]
        != counts["candidate_base_frames"]
    ):
        raise ValueError("Union population base-frame accounting is inconsistent.")
    if counts["unique_base_frames"] <= 0:
        raise ValueError("Union population has no unique training base frames.")

    selected = payload["selected"]
    if not isinstance(selected, Mapping) or set(selected) != {"state", "action"}:
        raise ValueError("Union statistics selected must contain state and action.")
    state = _validate_union_statistic(
        selected["state"], label="selected.state", width=REALMAN_POLICY_DIM
    )
    action = _validate_union_statistic(
        selected["action"], label="selected.action", width=REALMAN_POLICY_DIM
    )

    modalities = payload["modalities"]
    if not isinstance(modalities, Mapping) or set(modalities) != {"state", "action"}:
        raise ValueError("Union statistics modalities must contain state and action.")
    expected_modalities = {
        "state": {"source": state},
        "action": {
            "source_controls": _slice_union_statistic(action, 0, 16),
            "source_head": _slice_union_statistic(action, 16, 18),
        },
    }
    if modalities != expected_modalities:
        raise ValueError(
            "Union statistics modality views do not exactly match selected stats."
        )
    # Return an ordinary detached dict so downstream callers cannot retain an
    # OmegaConf/custom Mapping with unstable serialization behavior.
    return json.loads(
        deterministic_json_bytes(dict(payload)).decode("utf-8")
    )


def serialize_openpi_realman_union_statistics(payload: Any) -> bytes:
    """Validate and deterministically serialize one union-statistics artifact."""

    validated = validate_openpi_realman_union_statistics(payload)
    return deterministic_json_bytes(validated)


def load_openpi_realman_union_statistics(
    path: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load a byte-bound union-statistics artifact and fail closed on drift."""

    statistics_path = Path(path).expanduser().resolve()
    if not statistics_path.is_file():
        raise FileNotFoundError(
            f"Union statistics artifact does not exist: {statistics_path}"
        )
    _require_sha256(expected_sha256, label="expected_sha256")
    raw = statistics_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Union statistics artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Union statistics artifact is invalid JSON: {statistics_path}: {exc}"
        ) from exc
    return validate_openpi_realman_union_statistics(payload)
