"""Deterministic, episode-level offline evaluation for Realman VLA policies.

This utility is intentionally stricter than a training-loop validation pass:

* episode IDs are selected before frames, so an evaluation split never mixes
  frames from selected and non-selected episodes;
* policy inputs use ``qwen_frames`` at the current training observation,
  normalized state, and the checkpoint prompt builder;
* padding and action-validity masks are combined before any metric is updated;
* optional state-matched image ablation runs clean, repeated-clean, and donor-image
  policy calls with the same local RNG seed and refuses nondeterministic repeats;
* normalized and deployed/raw-space metrics are reported for policy draws,
  ensemble mean/median, current-state hold, repeat-h0, checkpoint dataset mean,
  and normalized-center baselines;
* arm, head, and gripper metrics are kept separate at every action horizon.

Important: a hash-selected subset of the checkpoint's training dataset is not a
true held-out evaluation set.  The report only calls episodes held out from
training when a supplied manifest explicitly records that guarantee.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 5
DEFAULT_PREFIX_HORIZONS = (1, 5, 10, 20, 50)
IMAGE_ABLATION_DETERMINISM_MAX_ABS_TOLERANCE = 1e-6
PAIRED_BOOTSTRAP_RESAMPLES = 10_000
PAIRED_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
PAIRED_BOOTSTRAP_MAX_INDEX_ELEMENTS_PER_BATCH = 1_000_000
PAIRED_BOOTSTRAP_SEED_DOMAIN = (
    "realman_offline_eval/state_matched_image_shuffle/"
    "paired_arm_target_error/v1"
)
REALMAN_DATASET_CAMERA_TO_DEPLOYMENT = {
    "video.base_view": "head",
    "video.left_wrist": "wrist_left",
    "video.right_wrist": "wrist_right",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _stable_digest(*parts: Any) -> bytes:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).digest()


def _stable_seed(*parts: Any) -> int:
    return int.from_bytes(_stable_digest(*parts)[:8], "big", signed=False)


def _paired_bootstrap_improvement(
    *,
    clean_errors: Sequence[float],
    shuffled_errors: Sequence[float],
    eval_seed: int,
    prefix: int,
    num_resamples: int = PAIRED_BOOTSTRAP_RESAMPLES,
    confidence_level: float = PAIRED_BOOTSTRAP_CONFIDENCE_LEVEL,
    max_index_elements_per_batch: int = (
        PAIRED_BOOTSTRAP_MAX_INDEX_ELEMENTS_PER_BATCH
    ),
) -> dict[str, Any]:
    """Bootstrap paired per-observation error improvements.

    One bootstrap unit is one observation's ``(clean, shuffled)`` pair.  The
    paired improvement is ``shuffled_error - clean_error``, so a confidence
    interval wholly above zero is evidence that clean images reduce target
    error.  Policy sampling and bootstrap sampling use separate SHA256 seed
    domains; the bootstrap never consumes or mutates the policy RNG stream.
    """

    clean = np.asarray(clean_errors, dtype=np.float64)
    shuffled = np.asarray(shuffled_errors, dtype=np.float64)
    if clean.ndim != 1 or shuffled.ndim != 1:
        raise ValueError("Paired bootstrap errors must be one-dimensional.")
    if clean.shape != shuffled.shape:
        raise ValueError(
            "Paired bootstrap clean/shuffled populations differ: "
            f"{clean.shape} != {shuffled.shape}."
        )
    if clean.size < 2:
        raise ValueError(
            "Paired bootstrap requires at least two evaluated observations; "
            f"got {clean.size}."
        )
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(shuffled)):
        raise ValueError("Paired bootstrap errors contain NaN or Infinity.")
    resamples = int(num_resamples)
    if resamples < 1:
        raise ValueError("Paired bootstrap num_resamples must be positive.")
    confidence = float(confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("Paired bootstrap confidence_level must be between 0 and 1.")
    prefix_value = int(prefix)
    if prefix_value < 1:
        raise ValueError("Paired bootstrap prefix must be positive.")
    index_element_cap = int(max_index_elements_per_batch)
    if index_element_cap < 1:
        raise ValueError(
            "Paired bootstrap max_index_elements_per_batch must be positive."
        )
    if clean.size > index_element_cap:
        raise ValueError(
            "Paired bootstrap observation population exceeds the fixed sampled-index "
            f"memory cap: {clean.size} observations > {index_element_cap} index "
            "elements. Reduce the evaluation sample plan or increase the audited "
            "implementation cap deliberately."
        )

    improvements = shuffled - clean
    bootstrap_seed = _stable_seed(
        PAIRED_BOOTSTRAP_SEED_DOMAIN,
        int(eval_seed),
        f"arm_raw_mae_h{prefix_value}",
    )
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    bootstrap_medians = np.empty(resamples, dtype=np.float64)
    # Bound peak memory by sampled *elements*, not just resample count.  A
    # 90k-observation all-frame audit therefore uses batches of 11 rows rather
    # than allocating a 4096x90k (~2.95 GB) int64 index matrix.
    batch_size = min(
        resamples,
        max(1, index_element_cap // int(improvements.size)),
    )
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(
            0,
            improvements.size,
            size=(stop - start, improvements.size),
            dtype=np.int64,
        )
        sampled = improvements[indices]
        bootstrap_means[start:stop] = sampled.mean(axis=1)
        bootstrap_medians[start:stop] = np.median(sampled, axis=1)

    alpha_percent = 50.0 * (1.0 - confidence)

    def _estimate_with_interval(
        estimate: float,
        bootstrap_values: np.ndarray,
    ) -> dict[str, Any]:
        lower, upper = np.percentile(
            bootstrap_values,
            [alpha_percent, 100.0 - alpha_percent],
        )
        lower_value = float(lower)
        upper_value = float(upper)
        return {
            "estimate": float(estimate),
            "percentile_confidence_interval": {
                "confidence_level": confidence,
                "lower": lower_value,
                "upper": upper_value,
            },
            "confidence_interval_lower_bound_gt_zero": bool(lower_value > 0.0),
        }

    return {
        "observation_count": int(improvements.size),
        "clean_error": {
            "mean": float(clean.mean()),
            "median": float(np.median(clean)),
        },
        "state_matched_image_shuffle_error": {
            "mean": float(shuffled.mean()),
            "median": float(np.median(shuffled)),
        },
        "paired_improvement": {
            "definition": "state_matched_image_shuffle_error_minus_clean_error",
            "positive_means": "clean_images_reduce_arm_target_error",
            "mean": _estimate_with_interval(
                float(improvements.mean()), bootstrap_means
            ),
            "median": _estimate_with_interval(
                float(np.median(improvements)), bootstrap_medians
            ),
            "positive_observation_count": int(np.sum(improvements > 0.0)),
            "zero_observation_count": int(np.sum(improvements == 0.0)),
            "negative_observation_count": int(np.sum(improvements < 0.0)),
            "positive_observation_fraction": float(np.mean(improvements > 0.0)),
        },
        "bootstrap": {
            "method": "paired_nonparametric_percentile",
            "resampling_unit": "observation_pair",
            "resamples": resamples,
            "seed": int(bootstrap_seed),
            "seed_domain": PAIRED_BOOTSTRAP_SEED_DOMAIN,
            "eval_seed": int(eval_seed),
            "max_index_elements_per_batch": index_element_cap,
            "max_index_bytes_per_batch": int(
                index_element_cap * np.dtype(np.int64).itemsize
            ),
            "resamples_per_batch": int(batch_size),
            "actual_max_index_elements_per_batch": int(
                batch_size * improvements.size
            ),
        },
    }


def build_paired_arm_target_error_bootstrap(
    sample_records: Sequence[Mapping[str, Any]],
    *,
    eval_seed: int,
    num_resamples: int = PAIRED_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Summarize clean-vs-shuffled arm target error as paired observations.

    Each sample must contain the per-prefix metrics produced by
    :func:`build_per_sample_prefix_metrics`.  Missing samples, missing prefixes,
    zero-valid-element masks, and unequal clean/shuffled mask counts are fatal:
    silently dropping any of them would change the paired population.
    """

    records = list(sample_records)
    if not records:
        raise ValueError(
            "Paired arm target-error bootstrap received no evaluated observations."
        )

    clean_name = "policy_ensemble_median"
    shuffled_name = "policy_state_matched_image_shuffle_ensemble_median"
    discovered_prefixes: set[str] = set()
    for record in records:
        errors = record.get("arm_errors_by_prefix")
        if not isinstance(errors, Mapping):
            continue
        for method_name in (clean_name, shuffled_name):
            method_prefixes = errors.get(method_name)
            if isinstance(method_prefixes, Mapping):
                discovered_prefixes.update(str(key) for key in method_prefixes)
    prefix_keys = [
        str(prefix)
        for prefix in DEFAULT_PREFIX_HORIZONS
        if str(prefix) in discovered_prefixes
    ]
    for required_prefix in ("1", "5"):
        if required_prefix not in prefix_keys:
            raise ValueError(
                "Paired arm target-error bootstrap requires both h1 and h5; "
                f"missing h{required_prefix}."
            )

    values: dict[str, dict[str, list[float]]] = {
        prefix: {"clean": [], "shuffled": []} for prefix in prefix_keys
    }
    seen_identities: set[tuple[str, int, int]] = set()
    for record_index, record in enumerate(records):
        try:
            identity = (
                str(record["dataset_name"]),
                int(record["episode_id"]),
                int(record["requested_frame_index"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Paired bootstrap observation {record_index} has no canonical identity."
            ) from exc
        if identity in seen_identities:
            raise ValueError(
                "Paired bootstrap received a duplicate observation identity: "
                f"{identity}."
            )
        seen_identities.add(identity)

        errors = record.get("arm_errors_by_prefix")
        if not isinstance(errors, Mapping):
            raise ValueError(
                f"Paired bootstrap observation {identity} is missing arm errors."
            )
        clean_prefixes = errors.get(clean_name)
        shuffled_prefixes = errors.get(shuffled_name)
        if not isinstance(clean_prefixes, Mapping) or not isinstance(
            shuffled_prefixes, Mapping
        ):
            raise ValueError(
                f"Paired bootstrap observation {identity} is missing one side of the pair."
            )
        if any(
            prefix not in clean_prefixes or prefix not in shuffled_prefixes
            for prefix in prefix_keys
        ):
            raise ValueError(
                f"Paired bootstrap observation {identity} is missing a required prefix."
            )

        for prefix in prefix_keys:
            clean_item = clean_prefixes[prefix]
            shuffled_item = shuffled_prefixes[prefix]
            if not isinstance(clean_item, Mapping) or not isinstance(
                shuffled_item, Mapping
            ):
                raise ValueError(
                    f"Paired bootstrap h{prefix} metrics for {identity} are malformed."
                )
            clean_count = int(clean_item.get("count", 0))
            shuffled_count = int(shuffled_item.get("count", 0))
            if clean_count <= 0 or shuffled_count <= 0:
                raise ValueError(
                    f"Paired bootstrap h{prefix} for {identity} has no valid arm targets."
                )
            if clean_count != shuffled_count:
                raise ValueError(
                    f"Paired bootstrap h{prefix} mask counts differ for {identity}: "
                    f"clean={clean_count}, shuffled={shuffled_count}."
                )
            try:
                clean_error = float(clean_item["raw_mae"])
                shuffled_error = float(shuffled_item["raw_mae"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Paired bootstrap h{prefix} target error is missing for {identity}."
                ) from exc
            if not np.isfinite(clean_error) or not np.isfinite(shuffled_error):
                raise ValueError(
                    f"Paired bootstrap h{prefix} target error is non-finite for {identity}."
                )
            values[prefix]["clean"].append(clean_error)
            values[prefix]["shuffled"].append(shuffled_error)

    by_prefix = {
        prefix: _paired_bootstrap_improvement(
            clean_errors=values[prefix]["clean"],
            shuffled_errors=values[prefix]["shuffled"],
            eval_seed=int(eval_seed),
            prefix=int(prefix),
            num_resamples=int(num_resamples),
        )
        for prefix in prefix_keys
    }
    mean_gate_by_prefix = {
        prefix: bool(
            item["paired_improvement"]["mean"][
                "confidence_interval_lower_bound_gt_zero"
            ]
        )
        for prefix, item in by_prefix.items()
    }
    median_gate_by_prefix = {
        prefix: bool(
            item["paired_improvement"]["median"][
                "confidence_interval_lower_bound_gt_zero"
            ]
        )
        for prefix, item in by_prefix.items()
    }
    return {
        "metric": "arm_raw_mae",
        "aggregation_before_resampling": (
            "mean_absolute_error_over_valid_arm_elements_within_each_observation"
        ),
        "observation_weighting": "one_equal_weight_pair_per_evaluated_observation",
        "by_prefix": by_prefix,
        "confidence_interval_lower_bound_gt_zero_gate": {
            "criterion": (
                "95% paired-bootstrap percentile confidence-interval lower bound > 0"
            ),
            "mean_improvement_by_prefix": mean_gate_by_prefix,
            "median_improvement_by_prefix": median_gate_by_prefix,
            "required_h1_and_h5_mean_improvement_pass": bool(
                mean_gate_by_prefix["1"] and mean_gate_by_prefix["5"]
            ),
            "required_h1_and_h5_median_improvement_pass": bool(
                median_gate_by_prefix["1"] and median_gate_by_prefix["5"]
            ),
        },
    }


@dataclass(frozen=True)
class EpisodeRef:
    dataset_index: int
    dataset_name: str
    episode_id: int
    length: int
    selection_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_index": self.dataset_index,
            "dataset_name": self.dataset_name,
            "episode_id": self.episode_id,
            "length": self.length,
            "selection_digest": self.selection_digest,
        }


@dataclass(frozen=True, order=True)
class PlannedSampleIdentity:
    """Stable identity for one observation in an offline sample plan."""

    dataset_name: str
    episode_id: int
    frame_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "episode_id": self.episode_id,
            "frame_index": self.frame_index,
        }


@dataclass(frozen=True)
class StateMatchedImageDonor:
    """Deterministic image donor selected while matching target robot state."""

    target: PlannedSampleIdentity
    donor: PlannedSampleIdentity
    normalized_state_rms_distance: float

    def as_dict(self) -> dict[str, Any]:
        same_episode = (
            self.target.dataset_name == self.donor.dataset_name
            and self.target.episode_id == self.donor.episode_id
        )
        return {
            "target": self.target.as_dict(),
            "donor": self.donor.as_dict(),
            "normalized_state_rms_distance": self.normalized_state_rms_distance,
            "same_episode": same_episode,
            "same_episode_frame_gap": (
                abs(self.target.frame_index - self.donor.frame_index)
                if same_episode
                else None
            ),
        }


@dataclass(frozen=True)
class PairedImageAblationPrediction:
    """Clean and image-ablated policy draws generated with matched conditions."""

    clean_draws: np.ndarray
    shuffled_draws: np.ndarray
    clean_latency_ms: float
    clean_repeat_latency_ms: float
    shuffled_latency_ms: float
    clean_repeat_max_abs_difference: float
    clean_repeat_mean_abs_difference: float
    target_frames_sha256: str
    donor_frames_sha256: str
    pixel_mean_absolute_difference: float


def build_state_matched_image_donor_map(
    sample_states: Mapping[PlannedSampleIdentity, np.ndarray],
    *,
    min_same_episode_frame_gap: int = 20,
) -> dict[PlannedSampleIdentity, StateMatchedImageDonor]:
    """Match every target to the nearest eligible observation in state space.

    States must be the exact normalized deployment states sent to the policy.
    A donor is never the target itself. Donors in another episode are always
    eligible; donors in the same episode must be separated by at least
    ``min_same_episode_frame_gap`` frames so an adjacent near-duplicate frame
    cannot masquerade as a meaningful vision ablation. Ties are broken by the
    stable sample identity, making the result independent of mapping order.
    """

    minimum_gap = int(min_same_episode_frame_gap)
    if minimum_gap < 1:
        raise ValueError("min_same_episode_frame_gap must be at least 1.")
    if len(sample_states) < 2:
        raise ValueError(
            "State-matched image ablation requires at least two planned observations."
        )

    normalized_states: dict[PlannedSampleIdentity, np.ndarray] = {}
    state_shape: tuple[int, ...] | None = None
    for identity, raw_state in sample_states.items():
        if not isinstance(identity, PlannedSampleIdentity):
            raise TypeError(
                "sample_states keys must be PlannedSampleIdentity instances."
            )
        state = np.asarray(raw_state, dtype=np.float64).reshape(-1)
        if state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError(f"State for {identity} is empty or non-finite.")
        if state_shape is None:
            state_shape = state.shape
        elif state.shape != state_shape:
            raise ValueError(
                f"State shape mismatch for {identity}: {state.shape} != {state_shape}."
            )
        normalized_states[identity] = state

    result: dict[PlannedSampleIdentity, StateMatchedImageDonor] = {}
    identities = sorted(normalized_states)
    for target in identities:
        candidates: list[PlannedSampleIdentity] = []
        for donor in identities:
            if donor == target:
                continue
            same_episode = (
                donor.dataset_name == target.dataset_name
                and donor.episode_id == target.episode_id
            )
            if same_episode and abs(donor.frame_index - target.frame_index) < minimum_gap:
                continue
            candidates.append(donor)
        if not candidates:
            raise ValueError(
                "No eligible image donor for "
                f"{target}. Select another episode or sample same-episode frames at "
                f"least {minimum_gap} steps apart."
            )

        def _score(donor: PlannedSampleIdentity) -> tuple[float, PlannedSampleIdentity]:
            delta = normalized_states[target] - normalized_states[donor]
            distance = float(np.sqrt(np.mean(np.square(delta))))
            return distance, donor

        distance, donor = min((_score(candidate) for candidate in candidates))
        result[target] = StateMatchedImageDonor(
            target=target,
            donor=donor,
            normalized_state_rms_distance=distance,
        )
    return result


def _image_ablation_plan_fingerprint(
    donor_map: Mapping[PlannedSampleIdentity, StateMatchedImageDonor],
) -> str:
    rows = [
        (
            match.target.dataset_name,
            match.target.episode_id,
            match.target.frame_index,
            match.donor.dataset_name,
            match.donor.episode_id,
            match.donor.frame_index,
            match.normalized_state_rms_distance,
        )
        for _, match in sorted(donor_map.items())
    ]
    encoded = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _uint8_array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def predict_clean_and_state_matched_image_shuffle(
    predictor: Any,
    *,
    target_qwen_frames: np.ndarray,
    donor_qwen_frames: np.ndarray,
    instruction: str,
    state: np.ndarray,
    seed: int,
    num_samples: int,
) -> PairedImageAblationPrediction:
    """Run a clean/image-shuffled pair with every non-visual input fixed."""

    target_frames = np.ascontiguousarray(target_qwen_frames)
    donor_frames = np.ascontiguousarray(donor_qwen_frames)
    if target_frames.dtype != np.uint8 or donor_frames.dtype != np.uint8:
        raise TypeError("Paired image ablation requires uint8 RGB frame payloads.")
    if target_frames.shape != donor_frames.shape:
        raise ValueError(
            "Target and donor frame payloads have different shapes: "
            f"{target_frames.shape} != {donor_frames.shape}."
        )
    if np.array_equal(target_frames, donor_frames):
        raise ValueError(
            "The selected donor has a byte-identical image payload; this would not "
            "constitute a vision ablation."
        )
    state_snapshot = np.asarray(state).copy()
    clean_draws, clean_latency_ms = predictor.predict_many(
        qwen_frames=target_frames,
        instruction=instruction,
        state=state,
        seed=seed,
        num_samples=num_samples,
    )
    if not np.array_equal(np.asarray(state), state_snapshot, equal_nan=True):
        raise RuntimeError("Policy predictor mutated the target state during clean inference.")
    clean_repeat_draws, clean_repeat_latency_ms = predictor.predict_many(
        qwen_frames=target_frames,
        instruction=instruction,
        state=state,
        seed=seed,
        num_samples=num_samples,
    )
    if not np.array_equal(np.asarray(state), state_snapshot, equal_nan=True):
        raise RuntimeError(
            "Policy predictor mutated the target state during repeated clean inference."
        )
    shuffled_draws, shuffled_latency_ms = predictor.predict_many(
        qwen_frames=donor_frames,
        instruction=instruction,
        state=state,
        seed=seed,
        num_samples=num_samples,
    )
    if not np.array_equal(np.asarray(state), state_snapshot, equal_nan=True):
        raise RuntimeError("Policy predictor mutated the target state during ablated inference.")
    clean = np.asarray(clean_draws, dtype=np.float32)
    clean_repeat = np.asarray(clean_repeat_draws, dtype=np.float32)
    shuffled = np.asarray(shuffled_draws, dtype=np.float32)
    expected_batch_size = int(num_samples)
    for label, draws in (
        ("clean", clean),
        ("repeated clean", clean_repeat),
        ("image-shuffled", shuffled),
    ):
        if (
            draws.ndim != 3
            or draws.shape[0] != expected_batch_size
            or draws.shape[1] <= 0
            or draws.shape[2] <= 0
        ):
            raise ValueError(
                f"{label} policy draws must be [{expected_batch_size},H,D], got "
                f"{draws.shape}."
            )
        if not np.all(np.isfinite(draws)):
            raise ValueError(f"{label} policy draws contain NaN or Infinity.")
    if clean_repeat.shape != clean.shape or shuffled.shape != clean.shape:
        raise ValueError(
            "Paired policy draw shapes differ: "
            f"clean={clean.shape}, repeated_clean={clean_repeat.shape}, "
            f"image_shuffled={shuffled.shape}."
        )
    repeat_delta = np.abs(clean_repeat - clean)
    repeat_max_abs = float(np.max(repeat_delta))
    repeat_mean_abs = float(repeat_delta.mean())
    if repeat_max_abs > IMAGE_ABLATION_DETERMINISM_MAX_ABS_TOLERANCE:
        raise RuntimeError(
            "Same-image, same-seed policy repeat is not deterministic enough to "
            "attribute clean/shuffled output differences to vision: "
            f"max_abs={repeat_max_abs} exceeds "
            f"{IMAGE_ABLATION_DETERMINISM_MAX_ABS_TOLERANCE}."
        )
    pixel_delta = np.abs(
        target_frames.astype(np.int16) - donor_frames.astype(np.int16)
    )
    return PairedImageAblationPrediction(
        clean_draws=clean,
        shuffled_draws=shuffled,
        clean_latency_ms=float(clean_latency_ms),
        clean_repeat_latency_ms=float(clean_repeat_latency_ms),
        shuffled_latency_ms=float(shuffled_latency_ms),
        clean_repeat_max_abs_difference=repeat_max_abs,
        clean_repeat_mean_abs_difference=repeat_mean_abs,
        target_frames_sha256=_uint8_array_sha256(target_frames),
        donor_frames_sha256=_uint8_array_sha256(donor_frames),
        pixel_mean_absolute_difference=float(pixel_delta.mean()),
    )


def enumerate_episodes(single_datasets: Sequence[Any], *, seed: int) -> list[EpisodeRef]:
    """Enumerate complete episode identities with stable selection scores."""

    refs: list[EpisodeRef] = []
    for dataset_index, dataset in enumerate(single_datasets):
        dataset_name = str(getattr(dataset, "dataset_name", f"dataset_{dataset_index}"))
        episode_ids = np.asarray(dataset.trajectory_ids).reshape(-1)
        lengths = np.asarray(dataset.trajectory_lengths).reshape(-1)
        if episode_ids.shape != lengths.shape:
            raise ValueError(
                f"Episode ID/length shape mismatch for {dataset_name}: "
                f"{episode_ids.shape} versus {lengths.shape}."
            )
        for episode_id, length in zip(episode_ids, lengths, strict=True):
            digest = _stable_digest(seed, dataset_name, int(episode_id)).hex()
            refs.append(
                EpisodeRef(
                    dataset_index=dataset_index,
                    dataset_name=dataset_name,
                    episode_id=int(episode_id),
                    length=int(length),
                    selection_digest=digest,
                )
            )
    return refs


def enumerate_lerobot_episode_root(
    dataset_root: Path,
    *,
    seed: int,
) -> list[EpisodeRef]:
    """Read an episode catalog without constructing its training dataset.

    This is used to prove that an external evaluation catalog is disjoint from
    the checkpoint's full training catalog even when the latter is not the
    active dataset being decoded.
    """

    root = Path(dataset_root).expanduser().resolve()
    dataset_name = root.name
    v3_paths = sorted((root / "meta/episodes").glob("*/*.parquet"))
    rows: list[tuple[int, int]] = []
    if v3_paths:
        import pyarrow.parquet as pq

        for path in v3_paths:
            table = pq.read_table(path, columns=["episode_index", "length"])
            episode_ids = np.asarray(table.column("episode_index").to_numpy()).reshape(-1)
            lengths = np.asarray(table.column("length").to_numpy()).reshape(-1)
            rows.extend(
                (int(episode_id), int(length))
                for episode_id, length in zip(episode_ids, lengths, strict=True)
            )
    else:
        v2_path = root / "meta/episodes.jsonl"
        if not v2_path.is_file():
            raise FileNotFoundError(
                f"Could not find a LeRobot episode catalog under {root}."
            )
        for line in v2_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append((int(row["episode_index"]), int(row["length"])))

    if not rows:
        raise ValueError(f"Training dataset catalog is empty: {root}")
    if len({episode_id for episode_id, _ in rows}) != len(rows):
        raise ValueError(f"Training dataset has duplicate episode IDs: {root}")
    return [
        EpisodeRef(
            dataset_index=0,
            dataset_name=dataset_name,
            episode_id=episode_id,
            length=length,
            selection_digest=_stable_digest(seed, dataset_name, episode_id).hex(),
        )
        for episode_id, length in sorted(rows)
    ]


def select_episode_refs(
    refs: Sequence[EpisodeRef],
    *,
    num_episodes: int,
    explicit_episode_ids: Mapping[str, Iterable[int]] | None = None,
) -> list[EpisodeRef]:
    """Select whole episodes deterministically.

    Hash selection is invariant to source ordering.  Explicit IDs are useful
    with a split manifest produced before training.
    """

    if not refs:
        raise ValueError("Dataset contains no episodes.")
    by_identity = {(ref.dataset_name, ref.episode_id): ref for ref in refs}
    if len(by_identity) != len(refs):
        raise ValueError("Duplicate (dataset_name, episode_id) identities are not supported.")

    if explicit_episode_ids is not None:
        requested = {
            (str(dataset_name), int(episode_id))
            for dataset_name, episode_ids in explicit_episode_ids.items()
            for episode_id in episode_ids
        }
        missing = sorted(requested - set(by_identity))
        if missing:
            raise ValueError(f"Episode manifest references unknown episodes: {missing[:20]}")
        selected = [by_identity[identity] for identity in requested]
        selected.sort(key=lambda ref: (ref.dataset_name, ref.episode_id))
        if num_episodes > 0 and len(selected) > num_episodes:
            selected = sorted(selected, key=lambda ref: ref.selection_digest)[:num_episodes]
        return selected

    if num_episodes <= 0:
        raise ValueError("--num-episodes must be positive without an episode manifest.")
    if num_episodes > len(refs):
        raise ValueError(f"Requested {num_episodes} episodes, but only {len(refs)} exist.")
    return sorted(refs, key=lambda ref: ref.selection_digest)[:num_episodes]


def deterministic_frame_indices(
    episode_length: int,
    *,
    frames_per_episode: int,
    seed: int,
    dataset_name: str,
    episode_id: int,
) -> list[int]:
    """Choose one deterministic frame per equal-width episode stratum.

    ``frames_per_episode=0`` evaluates every frame.  Sampling across the full
    episode intentionally includes end-of-episode windows; their padded suffix
    is removed by :func:`build_valid_action_mask`.
    """

    length = int(episode_length)
    if length <= 0:
        return []
    requested = int(frames_per_episode)
    if requested < 0:
        raise ValueError("frames_per_episode must be non-negative.")
    if requested == 0 or requested >= length:
        return list(range(length))

    rng = np.random.default_rng(_stable_seed(seed, dataset_name, episode_id, "frames"))
    edges = np.linspace(0, length, requested + 1, dtype=np.int64)
    indices: list[int] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        lo = int(lower)
        hi = max(lo + 1, int(upper))
        indices.append(int(rng.integers(lo, hi)))
    return sorted(set(indices))


def extract_training_aligned_qwen_frames(
    sample: Mapping[str, Any],
    *,
    video_target_shift_steps: int,
) -> np.ndarray:
    """Extract the exact current Qwen observation as ``[views,H,W,RGB]``."""

    if "qwen_frames" in sample:
        frames = np.asarray(sample["qwen_frames"])
    elif "video_compact" in sample:
        compact = np.asarray(sample["video_compact"])
        if compact.ndim != 5:
            raise ValueError(f"Expected video_compact [V,T,H,W,C], got {compact.shape}.")
        current_index = compact.shape[1] - int(video_target_shift_steps) - 1
        if current_index < 0 or current_index >= compact.shape[1]:
            raise ValueError(
                f"Invalid current Qwen frame index {current_index} for compact shape "
                f"{compact.shape} and target shift {video_target_shift_steps}."
            )
        frames = compact[:, current_index]
    else:
        raise KeyError("Sample contains neither qwen_frames nor video_compact.")

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected qwen_frames [V,H,W,3], got {frames.shape}.")
    if frames.dtype != np.uint8:
        raise TypeError(f"Expected uint8 qwen_frames, got {frames.dtype}.")
    return np.ascontiguousarray(frames)


def validate_training_aligned_input_contract(
    *,
    qwen_frames: np.ndarray,
    state: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    """Refuse camera/state payloads that differ from the served contract."""

    contract = metadata.get("realman_input_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Policy metadata is missing realman_input_contract.")
    if contract.get("payload_key") != "qwen_frames":
        raise ValueError(
            f"Policy contract requires payload {contract.get('payload_key')!r}, not qwen_frames."
        )
    if contract.get("color_space") != "RGB":
        raise ValueError(
            f"Policy contract color space is {contract.get('color_space')!r}, expected RGB."
        )
    expected_frames = tuple(int(value) for value in contract.get("frame_shape", ()))
    actual_frames = tuple(np.asarray(qwen_frames).shape)
    if actual_frames != expected_frames:
        raise ValueError(
            f"Qwen frame shape {actual_frames} does not match policy contract {expected_frames}."
        )
    expected_state = tuple(int(value) for value in contract.get("state_shape", ()))
    # The sample is unbatched [1,D]; transport adds the outer batch dimension.
    actual_state = (1, *tuple(np.asarray(state).shape))
    if actual_state != expected_state:
        raise ValueError(
            f"State transport shape {actual_state} does not match policy contract {expected_state}."
        )


def validate_dataset_camera_order(
    single_dataset: Any,
    metadata: Mapping[str, Any],
) -> None:
    dataset_keys = [str(value) for value in single_dataset.modality_keys["video"]]
    try:
        semantic_order = [REALMAN_DATASET_CAMERA_TO_DEPLOYMENT[key] for key in dataset_keys]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Realman dataset camera key {exc.args[0]!r}; semantic view order "
            "cannot be proven."
        ) from exc
    contract = metadata.get("realman_input_contract") or {}
    expected = [str(value) for value in contract.get("camera_order", ())]
    if semantic_order != expected:
        raise ValueError(
            "Dataset camera semantics do not match deployment contract: "
            f"dataset={semantic_order}, deployment={expected}."
        )


def build_valid_action_mask(sample: Mapping[str, Any], action_shape: tuple[int, int]) -> np.ndarray:
    """Combine action validity, padding, and finite-target masks."""

    horizon, action_dim = action_shape
    valid = np.ones((horizon, action_dim), dtype=bool)
    if "action_mask" in sample:
        action_mask = np.asarray(sample["action_mask"], dtype=bool)
        if action_mask.shape == (horizon,):
            action_mask = np.broadcast_to(action_mask[:, None], valid.shape)
        elif action_mask.shape != valid.shape:
            raise ValueError(
                f"action_mask shape {action_mask.shape} does not match action {action_shape}."
            )
        valid &= action_mask
    if "action_is_pad" in sample:
        is_pad = np.asarray(sample["action_is_pad"], dtype=bool).reshape(-1)
        if is_pad.shape != (horizon,):
            raise ValueError(
                f"action_is_pad shape {is_pad.shape} does not match horizon {horizon}."
            )
        valid &= ~is_pad[:, None]

    target = np.asarray(sample["action"], dtype=np.float32)
    if target.shape != action_shape:
        raise ValueError(f"Sample action shape changed from {action_shape} to {target.shape}.")
    valid &= np.isfinite(target)
    return valid


def normalize_values(values: Any, stats: Mapping[str, Any], *, mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if mode in {"min_max", "q99"}:
        low_key, high_key = ("min", "max") if mode == "min_max" else ("q01", "q99")
        low = np.asarray(stats[low_key], dtype=np.float32)
        high = np.asarray(stats[high_key], dtype=np.float32)
        denominator = high - low
        active = denominator != 0
        if mode == "q99":
            active &= np.asarray(stats.get("mask", np.ones_like(active)), dtype=bool)
        normalized = np.zeros_like(array, dtype=np.float32)
        normalized[..., active] = (
            2.0 * (array[..., active] - low[active]) / denominator[active] - 1.0
        )
        if mode == "q99":
            normalized[..., ~active] = array[..., ~active]
        return normalized
    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        active = std != 0
        normalized = np.zeros_like(array, dtype=np.float32)
        normalized[..., active] = (array[..., active] - mean[active]) / std[active]
        return normalized
    raise ValueError(f"Unsupported normalization mode {mode!r}.")


def unnormalize_values(
    values: Any,
    stats: Mapping[str, Any],
    *,
    mode: str,
    clip: bool = False,
) -> np.ndarray:
    """Invert training normalization, optionally clipping an explicit diagnostic.

    The default is the exact training/server inverse: normalized predictions are
    not clipped before the affine transform.  ``clip=True`` is retained only for
    deliberately labeled bounded diagnostics and must not be used for rollout-
    comparable policy metrics.
    """

    array = np.asarray(values, dtype=np.float32)
    if mode in {"min_max", "q99"}:
        low_key, high_key = ("min", "max") if mode == "min_max" else ("q01", "q99")
        low = np.asarray(stats[low_key], dtype=np.float32)
        high = np.asarray(stats[high_key], dtype=np.float32)
        inverse_input = np.clip(array, -1.0, 1.0) if clip else array
        raw = 0.5 * (inverse_input + 1.0) * (high - low) + low
        return raw.astype(np.float32, copy=False)
    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return (array * std + mean).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported normalization mode {mode!r}.")


def project_realman_state_to_action(state_raw: Any, *, action_dim: int) -> np.ndarray:
    """Build an absolute-action hold vector from the current 19D state."""

    state = np.asarray(state_raw, dtype=np.float32).reshape(-1)
    if state.shape != (19,):
        raise ValueError(f"Expected Realman state dim 19, got {state.shape}.")
    if action_dim == 18:  # arms/grippers + head, no base or lift
        return state[:18].copy()
    if action_dim == 19:  # arms/grippers + head + lift, no base
        return state.copy()
    if action_dim == 22:
        action = np.zeros(22, dtype=np.float32)
        action[:16] = state[:16]
        action[19:22] = state[16:19]
        return action
    raise ValueError(f"Unsupported Realman policy action dimension {action_dim}.")


def realman_action_groups(action_dim: int) -> dict[str, tuple[int, ...]]:
    if action_dim not in {18, 19, 22}:
        raise ValueError(f"Unsupported Realman policy action dimension {action_dim}.")
    groups: dict[str, tuple[int, ...]] = {
        "all": tuple(range(action_dim)),
        "arm": tuple(range(0, 7)) + tuple(range(8, 15)),
        "gripper": (7, 15),
    }
    if action_dim in {18, 19}:
        groups["head"] = (16, 17)
        if action_dim == 19:
            groups["lift"] = (18,)
    else:
        groups["base"] = (16, 17, 18)
        groups["head"] = (19, 20)
        groups["lift"] = (21,)
    return groups


def _empty_group_stats() -> dict[str, float | int]:
    return {
        "count": 0,
        "normalized_abs_sum": 0.0,
        "normalized_sq_sum": 0.0,
        "raw_abs_sum": 0.0,
        "raw_sq_sum": 0.0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }


def _merge_group_stats(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in destination:
        destination[key] += source[key]


class MetricAccumulator:
    """Mask-aware error and gripper-classification aggregation."""

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        gripper_thresholds: np.ndarray,
    ) -> None:
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.groups = realman_action_groups(self.action_dim)
        thresholds = np.asarray(gripper_thresholds, dtype=np.float32).reshape(-1)
        if thresholds.shape != (self.action_dim,):
            raise ValueError(
                f"Gripper threshold shape {thresholds.shape} does not match action dim {self.action_dim}."
            )
        self.gripper_thresholds = thresholds
        self.aggregate = {name: _empty_group_stats() for name in self.groups}
        self.by_horizon = [
            {name: _empty_group_stats() for name in self.groups} for _ in range(self.horizon)
        ]

    def update(
        self,
        *,
        prediction_normalized: np.ndarray,
        prediction_raw: np.ndarray,
        target_normalized: np.ndarray,
        target_raw: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        expected = (self.horizon, self.action_dim)
        arrays = {
            "prediction_normalized": np.asarray(prediction_normalized, dtype=np.float32),
            "prediction_raw": np.asarray(prediction_raw, dtype=np.float32),
            "target_normalized": np.asarray(target_normalized, dtype=np.float32),
            "target_raw": np.asarray(target_raw, dtype=np.float32),
        }
        for name, array in arrays.items():
            if array.shape != expected:
                raise ValueError(f"{name} shape {array.shape} does not match {expected}.")
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != expected:
            raise ValueError(f"valid_mask shape {valid.shape} does not match {expected}.")
        if not np.all(np.isfinite(arrays["prediction_normalized"])):
            raise ValueError("Policy or baseline produced NaN/Inf normalized actions.")
        if not np.all(np.isfinite(arrays["prediction_raw"])):
            raise ValueError("Policy or baseline produced NaN/Inf raw actions.")

        norm_error = arrays["prediction_normalized"] - arrays["target_normalized"]
        raw_error = arrays["prediction_raw"] - arrays["target_raw"]
        for horizon_index in range(self.horizon):
            for group_name, dimensions in self.groups.items():
                dim_array = np.asarray(dimensions, dtype=np.int64)
                group_valid = valid[horizon_index, dim_array]
                if not np.any(group_valid):
                    continue
                selected_dims = dim_array[group_valid]
                norm_values = norm_error[horizon_index, selected_dims]
                raw_values = raw_error[horizon_index, selected_dims]
                stats = _empty_group_stats()
                stats["count"] = int(selected_dims.size)
                stats["normalized_abs_sum"] = float(np.abs(norm_values).sum(dtype=np.float64))
                stats["normalized_sq_sum"] = float(np.square(norm_values).sum(dtype=np.float64))
                stats["raw_abs_sum"] = float(np.abs(raw_values).sum(dtype=np.float64))
                stats["raw_sq_sum"] = float(np.square(raw_values).sum(dtype=np.float64))
                if group_name == "gripper":
                    threshold = self.gripper_thresholds[selected_dims]
                    predicted_positive = arrays["prediction_raw"][horizon_index, selected_dims] >= threshold
                    target_positive = arrays["target_raw"][horizon_index, selected_dims] >= threshold
                    stats["tp"] = int(np.sum(predicted_positive & target_positive))
                    stats["tn"] = int(np.sum(~predicted_positive & ~target_positive))
                    stats["fp"] = int(np.sum(predicted_positive & ~target_positive))
                    stats["fn"] = int(np.sum(~predicted_positive & target_positive))
                _merge_group_stats(self.aggregate[group_name], stats)
                _merge_group_stats(self.by_horizon[horizon_index][group_name], stats)

    @staticmethod
    def _finalize_group(stats: Mapping[str, Any], *, gripper: bool) -> dict[str, Any]:
        count = int(stats["count"])
        result: dict[str, Any] = {
            "count": count,
            "normalized_mae": None,
            "normalized_rmse": None,
            "raw_mae": None,
            "raw_rmse": None,
        }
        if count:
            result.update(
                {
                    "normalized_mae": float(stats["normalized_abs_sum"] / count),
                    "normalized_rmse": float(np.sqrt(stats["normalized_sq_sum"] / count)),
                    "raw_mae": float(stats["raw_abs_sum"] / count),
                    "raw_rmse": float(np.sqrt(stats["raw_sq_sum"] / count)),
                }
            )
        if gripper:
            tp, tn = int(stats["tp"]), int(stats["tn"])
            fp, fn = int(stats["fp"]), int(stats["fn"])
            positives, negatives = tp + fn, tn + fp
            result["classification"] = {
                "accuracy": float((tp + tn) / count) if count else None,
                "balanced_accuracy": (
                    float(0.5 * (tp / positives + tn / negatives))
                    if positives and negatives
                    else None
                ),
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            }
        return result

    def _finalize_groups(self, groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        return {
            name: self._finalize_group(stats, gripper=name == "gripper")
            for name, stats in groups.items()
        }

    def finalize(self, *, include_horizons: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"aggregate": self._finalize_groups(self.aggregate)}
        if not include_horizons:
            return result
        result["by_horizon"] = [
            {
                "action_index": horizon_index,
                "groups": self._finalize_groups(groups),
            }
            for horizon_index, groups in enumerate(self.by_horizon)
        ]
        prefixes: dict[str, Any] = {}
        for prefix in DEFAULT_PREFIX_HORIZONS:
            if prefix > self.horizon:
                continue
            merged = {name: _empty_group_stats() for name in self.groups}
            for horizon_groups in self.by_horizon[:prefix]:
                for name, stats in horizon_groups.items():
                    _merge_group_stats(merged[name], stats)
            prefixes[str(prefix)] = self._finalize_groups(merged)
        result["prefix_horizons"] = prefixes
        return result


@contextlib.contextmanager
def _seeded_python_random(seed: int):
    state = random.getstate()
    random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(state)


def load_targeted_training_sample(
    mixture_dataset: Any,
    single_dataset: Any,
    *,
    episode_id: int,
    frame_index: int,
    prompt_seed: int,
) -> tuple[dict[str, Any], str]:
    """Run the actual training dataset path for a requested episode/frame.

    The legacy mixture API only exposes randomized indexing.  Temporarily
    replacing ``sample_step`` keeps all decoding, transforms, masks, labels,
    and prompt augmentation in the authoritative dataset implementation while
    making the requested identity explicit.  Evaluation is single-threaded.
    """

    original_sample_step = mixture_dataset.sample_step

    def _sample_step(_: int) -> tuple[Any, int, int]:
        return single_dataset, int(episode_id), int(frame_index)

    mixture_dataset.sample_step = _sample_step
    try:
        with _seeded_python_random(prompt_seed):
            sample = mixture_dataset[0]
        language_key = single_dataset.modality_keys["language"][0]
        deployment_instruction = str(
            single_dataset.get_language(int(episode_id), language_key, int(frame_index))[0]
        )
    finally:
        mixture_dataset.sample_step = original_sample_step

    actual_episode = int(np.asarray(sample.get("episode_index", episode_id)).reshape(-1)[0])
    if actual_episode != int(episode_id):
        raise RuntimeError(
            f"Targeted sample requested episode {episode_id}, but loader returned {actual_episode}."
        )
    return dict(sample), deployment_instruction


def build_subtask_explicit_instruction(
    single_dataset: Any,
    sample: Mapping[str, Any],
    *,
    deployment_instruction: str,
) -> tuple[str, str]:
    """Always append the authoritative local-stage label for a diagnostic.

    This deliberately bypasses the training-time append probability while
    retaining the exact dataset label lookup, ignored-label configuration, and
    separator behavior.  Missing/unlabeled stages are errors rather than being
    silently folded into the global-prompt population.
    """

    from starVLA.dataloader.prompt_labels import (
        _ignored_prompt_labels,
        append_prompt_label,
    )

    if "subtask_index" not in sample:
        raise ValueError(
            "subtask_explicit prompt mode requires sample['subtask_index']."
        )
    resolver = getattr(single_dataset, "_subtask_label_for_index", None)
    if not callable(resolver):
        raise ValueError(
            "subtask_explicit prompt mode requires the dataset's "
            "_subtask_label_for_index resolver."
        )
    label_value = resolver(sample["subtask_index"])
    if label_value is None or not str(label_value).strip():
        raise ValueError(
            "subtask_explicit prompt mode could not resolve a local subtask label "
            f"for index {sample['subtask_index']!r}."
        )
    label = str(label_value).strip()
    data_cfg = single_dataset.data_cfg
    ignored_labels = _ignored_prompt_labels(data_cfg) | {"__unlabeled__"}
    if label.casefold() in ignored_labels:
        raise ValueError(
            "subtask_explicit prompt mode resolved an ignored/unlabeled stage "
            f"{label!r} for index {sample['subtask_index']!r}."
        )
    return append_prompt_label(deployment_instruction, label, data_cfg), label


def extract_authoritative_raw_modality_window(
    single_dataset: Any,
    *,
    modality: str,
    frame_index: int,
) -> np.ndarray:
    """Read the float32 source rows selected by the configured modality slices.

    The training sample intentionally stores state/action tensors as float16 before
    normalization. Raw-unit metrics and deployment-state reconstruction should use
    the original parquet values instead of inverting those quantized tensors.
    """

    frame_table = getattr(single_dataset, "curr_traj_data", None)
    if frame_table is None or len(frame_table) <= 0:
        raise RuntimeError("Targeted dataset sample did not leave episode parquet rows loaded.")
    data_cfg = single_dataset.data_cfg
    overrides = data_cfg.get("modality_metadata_overrides", {})
    modality_overrides = overrides.get(modality, {})
    pieces: list[np.ndarray] = []
    expected_rows: int | None = None
    for modality_key in single_dataset.modality_keys[modality]:
        if "." not in modality_key:
            raise ValueError(
                f"Expected namespaced {modality} modality key, got {modality_key!r}."
            )
        subkey = modality_key.split(".", 1)[1]
        spec = modality_overrides.get(subkey)
        if spec is None:
            raise ValueError(
                f"Missing modality_metadata_overrides.{modality}.{subkey}; cannot "
                "prove raw source layout."
            )
        original_key_value = spec.get("original_key")
        if not isinstance(original_key_value, str) or not original_key_value:
            raise ValueError(
                f"Missing original_key for {modality} modality {modality_key!r}."
            )
        original_key = original_key_value
        start = int(spec.get("start", 0))
        end_value = spec.get("end")
        offsets = np.asarray(single_dataset.delta_indices[modality_key], dtype=np.int64)
        row_indices = np.clip(
            offsets + int(frame_index),
            0,
            len(frame_table) - 1,
        )
        rows = frame_table.iloc[row_indices]
        if original_key not in rows.columns:
            raise KeyError(
                f"Raw source column {original_key!r} is absent for modality {modality_key!r}."
            )
        values = np.stack(
            [np.asarray(value, dtype=np.float32) for value in rows[original_key].tolist()],
            axis=0,
        )
        if values.ndim != 2:
            raise ValueError(
                f"Raw source {original_key!r} for {modality_key!r} must contain "
                f"1D vectors, got stacked shape {values.shape}."
            )
        end = values.shape[1] if end_value is None else int(end_value)
        if start < 0 or end < start or end > values.shape[1]:
            raise ValueError(
                f"Invalid raw slice [{start}:{end}] for {modality_key!r} with source "
                f"dimension {values.shape[1]}."
            )
        piece = values[:, start:end]
        if piece.shape[1] == 0:
            raise ValueError(f"Raw slice for {modality_key!r} selects zero dimensions.")
        if expected_rows is None:
            expected_rows = int(piece.shape[0])
        elif piece.shape[0] != expected_rows:
            raise ValueError(
                f"Raw {modality} pieces have inconsistent horizon lengths: "
                f"{expected_rows} versus {piece.shape[0]}."
            )
        pieces.append(piece)
    if not pieces:
        raise ValueError(f"Dataset has no configured {modality} modalities.")
    return np.ascontiguousarray(np.concatenate(pieces, axis=1), dtype=np.float32)


def _resolve_run_file(checkpoint_path: Path, filename: str) -> Path:
    start = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    for directory in (start, *start.parents):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} above checkpoint {checkpoint_path}.")


def load_episode_manifest(path: Path) -> tuple[dict[str, list[int]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Episode manifest must contain a `datasets` object.")
    parsed: dict[str, list[int]] = {}
    for dataset_name, episode_ids in datasets.items():
        if not isinstance(episode_ids, list):
            raise ValueError(f"Manifest dataset {dataset_name!r} must map to a list of IDs.")
        parsed[str(dataset_name)] = [int(value) for value in episode_ids]
    return parsed, payload


def validate_manifest_frame_plan(
    payload: Mapping[str, Any] | None,
    *,
    selected: Sequence[EpisodeRef],
) -> dict[tuple[str, int], list[int]] | None:
    """Validate and canonicalize an optional manifest-provided frame plan.

    The accepted schema is ``frames: {dataset_name: {episode_id_string:
    [frame_indices...]}}``.  A frame plan is an exact binding to the selected
    episodes: missing or extra datasets/episodes are rejected instead of being
    silently supplemented by deterministic stratum sampling.
    """

    if payload is None or "frames" not in payload:
        return None
    raw_frames = payload["frames"]
    if not isinstance(raw_frames, Mapping):
        raise ValueError("Manifest `frames` must be an object keyed by dataset name.")

    selected_by_identity = {
        (ref.dataset_name, ref.episode_id): ref for ref in selected
    }
    if len(selected_by_identity) != len(selected):
        raise ValueError("Selected episodes contain duplicate identities.")
    expected_datasets = {dataset_name for dataset_name, _ in selected_by_identity}
    provided_datasets = {str(dataset_name) for dataset_name in raw_frames}
    missing_datasets = sorted(expected_datasets - provided_datasets)
    unexpected_datasets = sorted(provided_datasets - expected_datasets)
    if missing_datasets or unexpected_datasets:
        raise ValueError(
            "Manifest frame-plan datasets do not exactly match selected episodes: "
            f"missing={missing_datasets}, unexpected={unexpected_datasets}."
        )

    frame_plan: dict[tuple[str, int], list[int]] = {}
    for raw_dataset_name, raw_episodes in raw_frames.items():
        dataset_name = str(raw_dataset_name)
        if not isinstance(raw_episodes, Mapping):
            raise ValueError(
                f"Manifest frames for dataset {dataset_name!r} must be an object "
                "keyed by episode ID strings."
            )
        for raw_episode_id, raw_indices in raw_episodes.items():
            if not isinstance(raw_episode_id, str):
                raise ValueError(
                    f"Manifest frame-plan episode IDs for {dataset_name!r} must be strings."
                )
            try:
                episode_id = int(raw_episode_id)
            except ValueError as exc:
                raise ValueError(
                    f"Manifest frame-plan episode ID {raw_episode_id!r} for "
                    f"{dataset_name!r} is not an integer string."
                ) from exc
            if str(episode_id) != raw_episode_id:
                raise ValueError(
                    f"Manifest frame-plan episode ID {raw_episode_id!r} for "
                    f"{dataset_name!r} is not in canonical integer-string form."
                )
            identity = (dataset_name, episode_id)
            ref = selected_by_identity.get(identity)
            if ref is None:
                raise ValueError(
                    "Manifest frame plan references an episode that is not selected: "
                    f"{identity}."
                )
            if not isinstance(raw_indices, list):
                raise ValueError(
                    f"Manifest frame plan for {identity} must be a list of frame indices."
                )
            if not raw_indices:
                raise ValueError(
                    f"Manifest frame plan for {identity} must contain at least one frame index."
                )
            indices: list[int] = []
            for raw_index in raw_indices:
                if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
                    raise ValueError(
                        f"Manifest frame index {raw_index!r} for {identity} must be an integer."
                    )
                index = int(raw_index)
                if index < 0 or index >= ref.length:
                    raise ValueError(
                        f"Manifest frame index {index} for {identity} is outside "
                        f"[0, {ref.length})."
                    )
                indices.append(index)
            if len(set(indices)) != len(indices):
                raise ValueError(
                    f"Manifest frame plan for {identity} contains duplicate frame indices."
                )
            frame_plan[identity] = sorted(indices)

    expected_identities = set(selected_by_identity)
    provided_identities = set(frame_plan)
    missing_identities = sorted(expected_identities - provided_identities)
    unexpected_identities = sorted(provided_identities - expected_identities)
    if missing_identities or unexpected_identities:
        raise ValueError(
            "Manifest frame plan does not exactly match selected episodes: "
            f"missing={missing_identities}, unexpected={unexpected_identities}."
        )
    return frame_plan


def validate_manifest_holdout_claim(
    payload: Mapping[str, Any] | None,
    *,
    cfg: Any,
    metadata: Mapping[str, Any],
    config_path: Path,
    episode_catalog_sha256: str,
    selected: Sequence[EpisodeRef],
    training_episode_catalog_sha256: str | None = None,
    training_refs: Sequence[EpisodeRef] | None = None,
    training_source_identity_catalog: Any | None = None,
    evaluation_source_identity_catalog: Any | None = None,
    selected_source_episode_indices: Sequence[int] | None = None,
) -> bool:
    """Verify, rather than trust, a manifest's held-out semantic label."""

    if not payload or not bool(payload.get("excluded_from_training", False)):
        return False
    if bool(payload.get("external_to_training_catalog", False)):
        if not training_episode_catalog_sha256 or training_refs is None:
            raise ValueError(
                "External held-out validation requires --training-dataset-root so the "
                "checkpoint's training catalog can be verified independently."
            )
        if (
            training_source_identity_catalog is None
            or evaluation_source_identity_catalog is None
            or selected_source_episode_indices is None
        ):
            raise ValueError(
                "External held-out validation requires independently enumerated "
                "training/evaluation source-content catalogs."
            )
        expected_bindings = {
            "evaluation_dataset_episode_catalog_sha256": str(
                episode_catalog_sha256
            ),
            "training_dataset_episode_catalog_sha256": str(
                training_episode_catalog_sha256
            ),
            "training_data_mix": str(cfg.datasets.vla_data.get("data_mix", "")),
        }
        for key, expected in expected_bindings.items():
            actual = str(payload.get(key) or "")
            if not expected or actual != expected:
                raise ValueError(
                    f"External held-out manifest binding {key!r} does not match: "
                    f"manifest={actual!r}, expected={expected!r}."
                )

        # Dataset names and local episode IDs are storage locators, so a renamed
        # or reindexed copy of a training episode must not pass as held out.
        from deployment.realman.holdout_identity import validate_holdout_proof

        validate_holdout_proof(
            payload.get("holdout_proof"),
            training_source_identity_catalog,
            evaluation_source_identity_catalog,
            selected_source_episode_indices,
        )
        return True

    if bool(cfg.datasets.vla_data.get("load_all_data_for_training", False)):
        raise ValueError(
            "Manifest claims held-out episodes, but this checkpoint config has "
            "load_all_data_for_training=true. This checkpoint trained on the full catalog."
        )
    expected_bindings = {
        "checkpoint_run_id": str(metadata.get("run_id") or ""),
        "dataset_episode_catalog_sha256": str(episode_catalog_sha256),
        "checkpoint_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    for key, expected in expected_bindings.items():
        actual = str(payload.get(key) or "")
        if not expected or actual != expected:
            raise ValueError(
                f"Held-out manifest binding {key!r} does not match the checkpoint: "
                f"manifest={actual!r}, expected={expected!r}."
            )
    training_datasets = payload.get("training_datasets")
    if not isinstance(training_datasets, Mapping):
        raise ValueError(
            "Held-out manifest must list authoritative `training_datasets` episode IDs."
        )
    training_identities = {
        (str(dataset_name), int(episode_id))
        for dataset_name, episode_ids in training_datasets.items()
        for episode_id in episode_ids
    }
    overlap = sorted(
        (ref.dataset_name, ref.episode_id)
        for ref in selected
        if (ref.dataset_name, ref.episode_id) in training_identities
    )
    if overlap:
        raise ValueError(
            f"Manifest held-out episodes overlap its training split: {overlap[:20]}."
        )
    return True


def _dataset_fingerprint(refs: Sequence[EpisodeRef]) -> str:
    identities = sorted((ref.dataset_name, ref.episode_id, ref.length) for ref in refs)
    encoded = json.dumps(identities, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stats_fingerprint(stats: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(stats), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def merged_dataset_modality_stats(
    mixture_dataset: Any,
    single_dataset: Any,
    *,
    modality: str,
) -> dict[str, list[float]]:
    """Concatenate the exact per-key stats used by dataset transforms."""

    metadata = mixture_dataset.merged_metadata[single_dataset.tag]
    modality_stats = getattr(metadata.statistics, modality)
    keys = list(single_dataset.modality_keys[modality])
    subkeys = [key.split(".", 1)[1] for key in keys]
    stat_names = ("min", "max", "mean", "std", "q01", "q99")
    result: dict[str, list[float]] = {}
    for stat_name in stat_names:
        pieces = []
        for subkey in subkeys:
            stats_object = modality_stats[subkey]
            pieces.append(np.asarray(getattr(stats_object, stat_name), dtype=np.float32).reshape(-1))
        result[stat_name] = np.concatenate(pieces).tolist()
    return result


def assert_checkpoint_dataset_stats_match(
    checkpoint_stats: Mapping[str, Any],
    dataset_stats: Mapping[str, Any],
    *,
    modality: str,
) -> None:
    """Fail when targets were normalized with stats other than the checkpoint's."""

    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        checkpoint_value = np.asarray(checkpoint_stats[stat_name], dtype=np.float32)
        dataset_value = np.asarray(dataset_stats[stat_name], dtype=np.float32)
        if checkpoint_value.shape != dataset_value.shape or not np.allclose(
            checkpoint_value, dataset_value, rtol=1e-6, atol=1e-6
        ):
            max_abs = None
            if checkpoint_value.shape == dataset_value.shape:
                max_abs = float(np.max(np.abs(checkpoint_value - dataset_value)))
            raise ValueError(
                f"Checkpoint and dataset {modality} {stat_name} statistics differ: "
                f"checkpoint_shape={checkpoint_value.shape} dataset_shape={dataset_value.shape} "
                f"max_abs={max_abs}. Offline targets and policy inputs would use different scales."
            )


def _metadata_stats(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    key = metadata.get("default_unnorm_key")
    if not key:
        raise ValueError("Policy metadata does not define one default normalization key.")
    action_stats = (metadata.get("action_stats_by_key") or {}).get(key)
    state_stats = (metadata.get("state_stats_by_key") or {}).get(key)
    if not isinstance(action_stats, dict) or not isinstance(state_stats, dict):
        raise ValueError(f"Policy metadata has incomplete action/state stats for key {key!r}.")
    action_mode = metadata.get("default_action_norm_mode")
    state_mode = metadata.get("default_state_norm_mode")
    if not action_mode or not state_mode:
        raise ValueError("Policy metadata does not define action/state normalization modes.")
    return action_stats, state_stats, str(action_mode), str(state_mode), str(key)


def assert_server_checkpoint_matches(
    requested_checkpoint: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Fail closed when a server-backed audit points at another checkpoint."""

    from deployment.model_server.checkpoint_utils import resolve_policy_checkpoint

    expected = resolve_policy_checkpoint(requested_checkpoint).expanduser().resolve()
    reported_value = metadata.get("checkpoint_path")
    if not isinstance(reported_value, str) or not reported_value.strip():
        raise ValueError(
            "Policy server metadata does not identify its checkpoint; a server-backed "
            "audit cannot prove which model it scored."
        )
    reported = Path(reported_value).expanduser().resolve()
    if reported != expected:
        raise ValueError(
            "Policy server checkpoint mismatch: "
            f"requested={expected}, server_reported={reported}. Refusing to write a "
            "mislabelled checkpoint report."
        )
    return expected


class ServerPredictor:
    def __init__(self, *, host: str, port: int, timeout: float) -> None:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host=host, port=port, timeout=timeout)
        self.metadata = dict(self.client.get_server_metadata())
        self.description = {
            "kind": "websocket_server",
            "host": host,
            "port": int(port),
            "server_precision": self.metadata.get("server_precision"),
            "deterministic_seed_control": False,
        }

    def predict_many(
        self,
        *,
        qwen_frames: np.ndarray,
        instruction: str,
        state: np.ndarray,
        seed: int,
        num_samples: int,
    ) -> tuple[np.ndarray, float]:
        del seed  # The current server protocol does not expose per-request RNG seeds.
        batch_size = int(num_samples)
        if batch_size <= 0:
            raise ValueError("num_samples must be positive.")
        payload = {
            "qwen_frames": np.repeat(
                np.asarray(qwen_frames, dtype=np.uint8)[None, ...], batch_size, axis=0
            ),
            "instructions": [str(instruction)] * batch_size,
            "state": np.repeat(
                np.asarray(state, dtype=np.float32)[None, ...], batch_size, axis=0
            ),
        }
        started = time.perf_counter()
        response = self.client.infer(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not response.get("ok", False):
            raise RuntimeError(f"Policy server inference failed: {response}")
        actions = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != batch_size:
            raise ValueError(
                f"Expected server actions [{batch_size},H,D], got {actions.shape}."
            )
        return actions, latency_ms

    def close(self) -> None:
        self.client.close()


class LocalPredictor:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device_name: str,
        global_bf16_parameter_cast: bool = False,
    ) -> None:
        import torch

        from deployment.model_server.checkpoint_utils import build_policy_metadata, resolve_policy_checkpoint
        from starVLA.model.framework.base_framework import baseframework

        self.torch = torch
        self.resolved_checkpoint = resolve_policy_checkpoint(checkpoint_path)
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device_name!r} requested, but CUDA is unavailable.")
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device.index)
        model = baseframework.from_pretrained(
            str(self.resolved_checkpoint),
            inference_only=True,
            skip_training_backbones=True,
        )
        model = model.to(self.device)
        if global_bf16_parameter_cast:
            if self.device.type != "cuda":
                raise ValueError("Global BF16 parameter casting requires a CUDA device.")
            # Match the historical policy-server --use_bf16 behavior exactly.
            # This intentionally casts the FP32 action head/state projector too.
            model = model.to(torch.bfloat16)
        self.model = model.eval()
        self.metadata = build_policy_metadata(self.model, self.resolved_checkpoint)
        self.description = {
            "kind": "local_checkpoint",
            "device": str(self.device),
            "resolved_checkpoint": str(self.resolved_checkpoint),
            "global_bf16_parameter_cast": bool(global_bf16_parameter_cast),
            "deterministic_seed_control": True,
        }

    def predict_many(
        self,
        *,
        qwen_frames: np.ndarray,
        instruction: str,
        state: np.ndarray,
        seed: int,
        num_samples: int,
    ) -> tuple[np.ndarray, float]:
        torch = self.torch
        batch_size = int(num_samples)
        if batch_size <= 0:
            raise ValueError("num_samples must be positive.")
        torch.manual_seed(int(seed) % (2**63 - 1))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed) % (2**63 - 1))
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.predict_action(
                qwen_frames=[np.asarray(qwen_frames, dtype=np.uint8)] * batch_size,
                instructions=[str(instruction)] * batch_size,
                state=np.repeat(
                    np.asarray(state, dtype=np.float32)[None, ...], batch_size, axis=0
                ),
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        actions = np.asarray(output["normalized_actions"], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != batch_size:
            raise ValueError(
                f"Expected local actions [{batch_size},H,D], got {actions.shape}."
            )
        return actions, latency_ms

    def close(self) -> None:
        return None


def _new_accumulator_set(
    *,
    action_dim: int,
    horizon: int,
    thresholds: np.ndarray,
    include_state_matched_image_shuffle: bool = False,
) -> dict[str, MetricAccumulator]:
    names = [
        "policy",
        "policy_ensemble_mean",
        "policy_ensemble_median",
        "policy_ensemble_median_target_h0_mask",
        "policy_median_h0_repeat",
        "current_state_hold",
        "target_h0_repeat",
        "dataset_action_mean",
        "normalized_center",
    ]
    if include_state_matched_image_shuffle:
        names.extend(
            (
                "policy_state_matched_image_shuffle",
                "policy_state_matched_image_shuffle_ensemble_mean",
                "policy_state_matched_image_shuffle_ensemble_median",
            )
        )
    return {
        name: MetricAccumulator(
            action_dim=action_dim,
            horizon=horizon,
            gripper_thresholds=thresholds,
        )
        for name in names
    }


def policy_ensemble_chunks(policy_draws: Any) -> dict[str, np.ndarray]:
    """Reduce ``[draw,H,D]`` stochastic outputs in normalized action space."""

    draws = np.asarray(policy_draws, dtype=np.float32)
    if draws.ndim != 3 or draws.shape[0] <= 0:
        raise ValueError(f"Expected policy draws [B,H,D] with B>0, got {draws.shape}.")
    mean = np.mean(draws, axis=0, dtype=np.float32)
    median = np.median(draws, axis=0).astype(np.float32, copy=False)
    return {
        "policy_ensemble_mean": mean,
        "policy_ensemble_median": median,
        "policy_median_h0_repeat": np.broadcast_to(median[0], median.shape).copy(),
    }


def _metric_ratio(
    candidate: float | None,
    baseline: float | None,
    *,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    if candidate is None or baseline is None:
        return {"candidate": candidate, "baseline": baseline, "delta": None, "ratio": None}
    candidate_value, baseline_value = float(candidate), float(baseline)
    ratio = candidate_value / baseline_value if baseline_value != 0.0 else None
    return {
        "candidate": candidate_value,
        "baseline": baseline_value,
        "delta": candidate_value - baseline_value,
        "ratio": ratio,
        "relative_improvement": (
            ((ratio - 1.0) if higher_is_better else (1.0 - ratio))
            if ratio is not None
            else None
        ),
    }


def build_horizon_comparison(
    candidate_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build explicit aggregate and per-horizon candidate/baseline deltas."""

    candidate_horizons = candidate_report["by_horizon"]
    baseline_horizons = baseline_report["by_horizon"]
    if len(candidate_horizons) != len(baseline_horizons):
        raise ValueError("Cannot compare reports with different action horizons.")

    def _groups(candidate_groups: Mapping[str, Any], baseline_groups: Mapping[str, Any]):
        if set(candidate_groups) != set(baseline_groups):
            raise ValueError(
                "Cannot compare reports with different metric groups: "
                f"candidate={sorted(candidate_groups)}, baseline={sorted(baseline_groups)}."
            )
        result: dict[str, Any] = {}
        for group_name in candidate_groups:
            candidate_group = candidate_groups[group_name]
            baseline_group = baseline_groups[group_name]
            candidate_count = int(candidate_group["count"])
            baseline_count = int(baseline_group["count"])
            if candidate_count != baseline_count:
                raise ValueError(
                    "Cannot compare metrics accumulated over different populations: "
                    f"group={group_name!r}, candidate_count={candidate_count}, "
                    f"baseline_count={baseline_count}."
                )
            result[group_name] = {
                metric_name: _metric_ratio(
                    candidate_group.get(metric_name), baseline_group.get(metric_name)
                )
                for metric_name in ("normalized_mae", "raw_mae")
            }
            if group_name == "gripper":
                result[group_name]["classification_accuracy"] = _metric_ratio(
                    candidate_group["classification"].get("accuracy"),
                    baseline_group["classification"].get("accuracy"),
                    higher_is_better=True,
                )
        return result

    by_horizon: list[dict[str, Any]] = []
    for candidate_item, baseline_item in zip(
        candidate_horizons, baseline_horizons, strict=True
    ):
        candidate_index = int(candidate_item["action_index"])
        baseline_index = int(baseline_item["action_index"])
        if candidate_index != baseline_index:
            raise ValueError(
                "Cannot compare reports with differently indexed horizons: "
                f"candidate={candidate_index}, baseline={baseline_index}."
            )
        by_horizon.append(
            {
                "action_index": candidate_index,
                "groups": _groups(
                    candidate_item["groups"], baseline_item["groups"]
                ),
            }
        )

    return {
        "aggregate": _groups(
            candidate_report["aggregate"], baseline_report["aggregate"]
        ),
        "by_horizon": by_horizon,
    }


def _update_method(
    accumulator: MetricAccumulator,
    *,
    prediction_normalized: np.ndarray,
    action_stats: Mapping[str, Any],
    action_mode: str,
    target_normalized: np.ndarray,
    target_raw: np.ndarray,
    valid_mask: np.ndarray,
    clip_prediction: bool = False,
) -> None:
    prediction_raw = unnormalize_values(
        prediction_normalized,
        action_stats,
        mode=action_mode,
        clip=clip_prediction,
    )
    accumulator.update(
        prediction_normalized=prediction_normalized,
        prediction_raw=prediction_raw,
        target_normalized=target_normalized,
        target_raw=target_raw,
        valid_mask=valid_mask,
    )


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _sample_plan_fingerprint(
    frame_plan: Mapping[tuple[str, int], Sequence[int]],
) -> str:
    identities = sorted(
        (dataset_name, int(episode_id), int(frame_index))
        for (dataset_name, episode_id), frame_indices in frame_plan.items()
        for frame_index in frame_indices
    )
    encoded = json.dumps(identities, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_cli_headline(
    report: Mapping[str, Any], *, report_path: Path
) -> dict[str, Any]:
    """Build the compact, decision-oriented summary printed after an audit.

    Aggregate horizon error can hide the near-term drift that matters most for
    synchronous Realman rollouts.  Keep the fixed 1/5/10/20/50 prefixes in the
    headline and pair each policy value with the current-state hold baseline.
    """

    metrics = report["metrics"]
    policy_prefixes = metrics["policy_ensemble_median"]["prefix_horizons"]
    hold_prefixes = metrics["current_state_hold"]["prefix_horizons"]
    prefix_comparisons: dict[str, Any] = {}
    for prefix in DEFAULT_PREFIX_HORIZONS:
        key = str(prefix)
        policy_item = policy_prefixes.get(key)
        hold_item = hold_prefixes.get(key)
        policy_mae = (
            policy_item["arm"].get("raw_mae") if policy_item is not None else None
        )
        hold_mae = hold_item["arm"].get("raw_mae") if hold_item is not None else None
        prefix_comparisons[key] = _metric_ratio(policy_mae, hold_mae)

    policy_arm = metrics["policy"]["aggregate"]["arm"]
    ensemble_mean_arm = metrics["policy_ensemble_mean"]["aggregate"]["arm"]
    ensemble_median_arm = metrics["policy_ensemble_median"]["aggregate"]["arm"]
    hold_arm = metrics["current_state_hold"]["aggregate"]["arm"]
    policy_gripper = metrics["policy_ensemble_median"]["aggregate"]["gripper"]
    headline = {
        "semantic_label": report["episode_split"]["semantic_label"],
        "episodes": len(report["episode_split"]["selected_episodes"]),
        "frames": report["sampling"]["evaluated_frames"],
        "sample_plan_sha256": report["sampling"]["sample_plan_sha256"],
        "policy_draw_arm_raw_mae": policy_arm["raw_mae"],
        "ensemble_mean_arm_raw_mae": ensemble_mean_arm["raw_mae"],
        "ensemble_median_arm_raw_mae": ensemble_median_arm["raw_mae"],
        "hold_arm_raw_mae": hold_arm["raw_mae"],
        "ensemble_median_vs_hold_arm_raw_mae_by_prefix": prefix_comparisons,
        "ensemble_median_gripper_accuracy": policy_gripper["classification"][
            "accuracy"
        ],
        "report_path": str(report_path),
    }
    image_ablation = report.get("image_ablation") or {}
    if image_ablation.get("mode") == "state_matched_shuffle":
        shuffled = metrics[
            "policy_state_matched_image_shuffle_ensemble_median"
        ]
        shuffled_prefixes = shuffled["prefix_horizons"]
        clean_vs_shuffled: dict[str, Any] = {}
        for prefix in DEFAULT_PREFIX_HORIZONS:
            key = str(prefix)
            clean_item = policy_prefixes.get(key)
            shuffled_item = shuffled_prefixes.get(key)
            clean_mae = (
                clean_item["arm"].get("raw_mae")
                if clean_item is not None
                else None
            )
            shuffled_mae = (
                shuffled_item["arm"].get("raw_mae")
                if shuffled_item is not None
                else None
            )
            clean_vs_shuffled[key] = _metric_ratio(clean_mae, shuffled_mae)
        output_delta_prefixes = image_ablation[
            "paired_policy_output_delta"
        ]["prefix_horizons"]
        headline.update(
            {
                "ensemble_median_vs_state_matched_image_shuffle_arm_raw_mae_by_prefix": (
                    clean_vs_shuffled
                ),
                "same_seed_image_shuffle_output_delta_arm_raw_mae_h1": (
                    output_delta_prefixes.get("1", {})
                    .get("arm", {})
                    .get("raw_mae")
                ),
                "image_ablation_donor_plan_sha256": image_ablation[
                    "donor_plan_sha256"
                ],
                "paired_arm_target_error_bootstrap_by_prefix": image_ablation[
                    "paired_arm_target_error_bootstrap"
                ]["by_prefix"],
            }
        )
    return headline


def _prefix_error_summary(
    *,
    prediction_normalized: np.ndarray,
    prediction_raw: np.ndarray,
    target_normalized: np.ndarray,
    target_raw: np.ndarray,
    valid_mask: np.ndarray,
    dimensions: Sequence[int],
) -> dict[str, Any]:
    dims = np.asarray(tuple(dimensions), dtype=np.int64)
    result: dict[str, Any] = {}
    for prefix in DEFAULT_PREFIX_HORIZONS:
        if prefix > target_raw.shape[0]:
            continue
        mask = np.asarray(valid_mask[:prefix, dims], dtype=bool)
        count = int(mask.sum())
        item: dict[str, Any] = {
            "count": count,
            "normalized_mae": None,
            "raw_mae": None,
        }
        if count:
            norm_error = np.abs(
                prediction_normalized[:prefix, dims]
                - target_normalized[:prefix, dims]
            )
            raw_error = np.abs(prediction_raw[:prefix, dims] - target_raw[:prefix, dims])
            item["normalized_mae"] = float(norm_error[mask].mean())
            item["raw_mae"] = float(raw_error[mask].mean())
        result[str(prefix)] = item
    return result


def _prefix_gripper_summary(
    *,
    prediction_normalized: np.ndarray,
    prediction_raw: np.ndarray,
    target_normalized: np.ndarray,
    target_raw: np.ndarray,
    valid_mask: np.ndarray,
    dimensions: Sequence[int],
    gripper_thresholds: np.ndarray,
) -> dict[str, Any]:
    """Return per-prefix gripper errors and open/close classification metrics."""

    result = _prefix_error_summary(
        prediction_normalized=prediction_normalized,
        prediction_raw=prediction_raw,
        target_normalized=target_normalized,
        target_raw=target_raw,
        valid_mask=valid_mask,
        dimensions=dimensions,
    )
    dims = np.asarray(tuple(dimensions), dtype=np.int64)
    thresholds = np.asarray(gripper_thresholds, dtype=np.float32).reshape(-1)
    if thresholds.shape != (target_raw.shape[1],):
        raise ValueError(
            f"Gripper threshold shape {thresholds.shape} does not match action dim "
            f"{target_raw.shape[1]}."
        )
    threshold = thresholds[dims][None, :]
    for prefix_text, item in result.items():
        prefix = int(prefix_text)
        mask = np.asarray(valid_mask[:prefix, dims], dtype=bool)
        predicted_positive = prediction_raw[:prefix, dims] >= threshold
        target_positive = target_raw[:prefix, dims] >= threshold
        tp = int(np.sum((predicted_positive & target_positive) & mask))
        tn = int(np.sum((~predicted_positive & ~target_positive) & mask))
        fp = int(np.sum((predicted_positive & ~target_positive) & mask))
        fn = int(np.sum((~predicted_positive & target_positive) & mask))
        count = int(item["count"])
        positives = tp + fn
        negatives = tn + fp
        item["classification"] = {
            "accuracy": float((tp + tn) / count) if count else None,
            "balanced_accuracy": (
                float(0.5 * (tp / positives + tn / negatives))
                if positives and negatives
                else None
            ),
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        }
    return result


def build_per_sample_prefix_metrics(
    *,
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    target_normalized: np.ndarray,
    target_raw: np.ndarray,
    arm_dimensions: Sequence[int],
    gripper_dimensions: Sequence[int],
    gripper_thresholds: np.ndarray,
) -> dict[str, Any]:
    """Build report-ready arm and gripper prefix metrics for one observation."""

    arm_errors = {
        name: _prefix_error_summary(
            prediction_normalized=prediction_norm,
            prediction_raw=prediction_raw,
            target_normalized=target_normalized,
            target_raw=target_raw,
            valid_mask=prediction_mask,
            dimensions=arm_dimensions,
        )
        for name, (prediction_norm, prediction_raw, prediction_mask) in predictions.items()
    }
    gripper_metrics = {
        name: _prefix_gripper_summary(
            prediction_normalized=prediction_norm,
            prediction_raw=prediction_raw,
            target_normalized=target_normalized,
            target_raw=target_raw,
            valid_mask=prediction_mask,
            dimensions=gripper_dimensions,
            gripper_thresholds=gripper_thresholds,
        )
        for name, (prediction_norm, prediction_raw, prediction_mask) in predictions.items()
    }
    return {
        "arm_errors_by_prefix": arm_errors,
        "gripper_metrics_by_prefix": gripper_metrics,
    }


def _motion_from_state_arm_first20(
    *,
    target_raw: np.ndarray,
    hold_raw: np.ndarray,
    valid_mask: np.ndarray,
    arm_dimensions: Sequence[int],
) -> float | None:
    prefix = min(20, int(target_raw.shape[0]))
    dims = np.asarray(tuple(arm_dimensions), dtype=np.int64)
    mask = np.asarray(valid_mask[:prefix, dims], dtype=bool)
    if not np.any(mask):
        return None
    displacement = np.abs(target_raw[:prefix, dims] - hold_raw[dims])
    return float(displacement[mask].mean())


def _motion_bin(value: float | None) -> str:
    if value is None:
        return "no_valid_arm_targets"
    for upper in (0.02, 0.05, 0.1, 0.2):
        if value < upper:
            return f"lt_{upper:g}"
    return "ge_0.2"


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    # Evaluation must remain offline; importing Albumentations otherwise starts
    # a best-effort PyPI version check in some installed releases.
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    image_ablation_mode = str(getattr(args, "image_ablation", "none"))
    if image_ablation_mode not in {"none", "state_matched_shuffle"}:
        raise ValueError(f"Unknown image ablation mode {image_ablation_mode!r}.")
    image_ablation_enabled = image_ablation_mode == "state_matched_shuffle"
    image_ablation_min_gap = int(
        getattr(args, "image_ablation_min_same_episode_frame_gap", 20)
    )
    if image_ablation_enabled and args.backend != "local":
        raise ValueError(
            "--image-ablation state_matched_shuffle requires --backend local because "
            "the server protocol cannot reset request-scoped policy noise."
        )
    if image_ablation_enabled and args.state_input_mode != "deployment":
        raise ValueError(
            "--image-ablation state_matched_shuffle requires --state-input-mode "
            "deployment so donor matching uses the exact checkpoint-normalized state "
            "sent by the robot rollout path."
        )

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else _resolve_run_file(checkpoint_path, "config.yaml")
    )
    cfg = OmegaConf.load(config_path)
    configured_training_dataset_root = Path(
        str(cfg.datasets.vla_data.data_root_dir)
    ).expanduser()
    if args.dataset_root:
        cfg.datasets.vla_data.data_root_dir = str(Path(args.dataset_root).expanduser().resolve())

    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="val",
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=int(args.seed),
        action_horizon=int(cfg.framework.action_model.action_horizon),
        video_horizon=int(cfg.framework.vj2_model.num_frames),
        video_frame_stride=int(cfg.datasets.vla_data.get("video_frame_stride", 1)),
    )
    refs = enumerate_episodes(dataset.datasets, seed=int(args.seed))
    explicit_ids = None
    manifest_payload: dict[str, Any] | None = None
    if args.episode_manifest:
        explicit_ids, manifest_payload = load_episode_manifest(
            Path(args.episode_manifest).expanduser().resolve()
        )
    selected = select_episode_refs(
        refs,
        num_episodes=int(args.num_episodes),
        explicit_episode_ids=explicit_ids,
    )
    explicit_frame_plan = validate_manifest_frame_plan(
        manifest_payload,
        selected=selected,
    )
    frame_plan = (
        explicit_frame_plan
        if explicit_frame_plan is not None
        else {
            (episode_ref.dataset_name, episode_ref.episode_id): deterministic_frame_indices(
                episode_ref.length,
                frames_per_episode=int(args.frames_per_episode),
                seed=int(args.seed),
                dataset_name=episode_ref.dataset_name,
                episode_id=episode_ref.episode_id,
            )
            for episode_ref in selected
        }
    )

    if args.backend == "server":
        predictor: Any = ServerPredictor(
            host=str(args.server_host),
            port=int(args.server_port),
            timeout=float(args.server_timeout),
        )
    else:
        predictor = LocalPredictor(
            checkpoint_path=checkpoint_path,
            device_name=str(args.device),
            global_bf16_parameter_cast=bool(args.global_bf16_parameter_cast),
        )

    metadata = predictor.metadata
    action_stats, state_stats, action_mode, state_mode, norm_key = _metadata_stats(metadata)
    clip_deployment_state = (
        state_mode == "q99"
        if args.clip_deployment_state is None
        else bool(args.clip_deployment_state)
    )
    action_dim = int(metadata["action_dim"])
    state_dim = int(metadata["state_dim"])
    horizon = int(metadata["action_horizon"])
    if action_dim != int(cfg.framework.action_model.action_dim):
        raise ValueError(
            f"Policy action dim {action_dim} != config action dim {cfg.framework.action_model.action_dim}."
        )
    if state_dim != int(cfg.framework.action_model.state_dim):
        raise ValueError(
            f"Policy state dim {state_dim} != config state dim {cfg.framework.action_model.state_dim}."
        )
    if horizon != int(cfg.framework.action_model.action_horizon):
        raise ValueError(
            f"Policy horizon {horizon} != config horizon {cfg.framework.action_model.action_horizon}."
        )
    if action_dim not in {18, 19, 22} or state_dim != 19:
        raise ValueError(
            f"This evaluator requires a Realman 18/19/22D action and 19D state, got "
            f"{action_dim}/{state_dim}."
        )

    dataset_statistics_alignment: list[dict[str, Any]] = []
    for single_dataset in dataset.datasets:
        validate_dataset_camera_order(single_dataset, metadata)
        for modality, checkpoint_values in (
            ("action", action_stats),
            ("state", state_stats),
        ):
            dataset_values = merged_dataset_modality_stats(
                dataset,
                single_dataset,
                modality=modality,
            )
            try:
                assert_checkpoint_dataset_stats_match(
                    checkpoint_values,
                    dataset_values,
                    modality=f"{single_dataset.dataset_name} {modality}",
                )
            except ValueError as exc:
                dataset_statistics_alignment.append(
                    {
                        "dataset_name": single_dataset.dataset_name,
                        "modality": modality,
                        "matches_checkpoint": False,
                        "detail": str(exc),
                        "dataset_stats_sha256": _stats_fingerprint(dataset_values),
                        "checkpoint_stats_sha256": _stats_fingerprint(
                            checkpoint_values
                        ),
                    }
                )
            else:
                dataset_statistics_alignment.append(
                    {
                        "dataset_name": single_dataset.dataset_name,
                        "modality": modality,
                        "matches_checkpoint": True,
                        "detail": None,
                        "dataset_stats_sha256": _stats_fingerprint(dataset_values),
                        "checkpoint_stats_sha256": _stats_fingerprint(
                            checkpoint_values
                        ),
                    }
                )

    if args.state_input_mode == "training" and any(
        not item["matches_checkpoint"]
        for item in dataset_statistics_alignment
        if item["modality"] == "state"
    ):
        raise ValueError(
            "--state-input-mode training would use evaluation-dataset normalization, "
            "which differs from the checkpoint. Use deployment mode so raw state is "
            "normalized with checkpoint statistics."
        )

    action_min = np.asarray(action_stats["min"], dtype=np.float32)
    action_max = np.asarray(action_stats["max"], dtype=np.float32)
    thresholds = 0.5 * (action_min + action_max)
    action_groups = realman_action_groups(action_dim)
    arm_dimensions = action_groups["arm"]
    gripper_dimensions = action_groups["gripper"]
    global_accumulators = _new_accumulator_set(
        action_dim=action_dim,
        horizon=horizon,
        thresholds=thresholds,
        include_state_matched_image_shuffle=image_ablation_enabled,
    )
    paired_image_output_delta = (
        MetricAccumulator(
            action_dim=action_dim,
            horizon=horizon,
            gripper_thresholds=thresholds,
        )
        if image_ablation_enabled
        else None
    )
    dataset_mean_raw = np.asarray(action_stats["mean"], dtype=np.float32)
    dataset_mean_norm = normalize_values(dataset_mean_raw, action_stats, mode=action_mode)
    center_norm = np.zeros(action_dim, dtype=np.float32)
    latency_ms: list[float] = []
    image_ablation_clean_repeat_latency_ms: list[float] = []
    image_ablation_latency_ms: list[float] = []
    image_ablation_clean_repeat_max_abs: list[float] = []
    image_ablation_clean_repeat_mean_abs: list[float] = []
    episode_reports: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    prompt_mode = str(args.prompt_mode)
    if prompt_mode == "explicit" and not args.instruction:
        raise ValueError("--prompt-mode explicit requires --instruction.")

    episode_catalog_sha256 = _dataset_fingerprint(refs)
    training_refs: list[EpisodeRef] | None = None
    training_episode_catalog_sha256: str | None = None
    training_root_for_holdout_proof: Path | None = None
    training_source_identity_catalog: Any | None = None
    evaluation_source_identity_catalog: Any | None = None
    selected_source_episode_indices: list[int] | None = None
    source_content_holdout_proof: dict[str, Any] | None = None
    if manifest_payload and bool(
        manifest_payload.get("external_to_training_catalog", False)
    ):
        training_root_for_holdout_proof = (
            Path(args.training_dataset_root).expanduser().resolve()
            if args.training_dataset_root
            else configured_training_dataset_root.resolve()
        )
        if not training_root_for_holdout_proof.exists():
            raise FileNotFoundError(
                "The checkpoint's configured training dataset is unavailable locally. "
                "Pass --training-dataset-root to validate the external held-out claim: "
                f"{training_root_for_holdout_proof}"
            )
        training_refs = enumerate_lerobot_episode_root(
            training_root_for_holdout_proof,
            seed=int(args.seed),
        )
        training_episode_catalog_sha256 = _dataset_fingerprint(training_refs)
        selected_dataset_names = {ref.dataset_name for ref in selected}
        if len(dataset.datasets) != 1 or len(selected_dataset_names) != 1:
            raise ValueError(
                "External source-content holdout proof currently requires one "
                "evaluation dataset root; mixtures must provide one independently "
                "bound proof per root."
            )
        evaluation_root_for_holdout_proof = Path(
            str(cfg.datasets.vla_data.data_root_dir)
        ).expanduser().resolve()
        from deployment.realman.holdout_identity import (
            enumerate_v3_source_identity_catalog,
        )

        training_source_identity_catalog = enumerate_v3_source_identity_catalog(
            training_root_for_holdout_proof
        )
        evaluation_source_identity_catalog = enumerate_v3_source_identity_catalog(
            evaluation_root_for_holdout_proof
        )
        selected_source_episode_indices = [ref.episode_id for ref in selected]
    manifest_guarantees_holdout = validate_manifest_holdout_claim(
        manifest_payload,
        cfg=cfg,
        metadata=metadata,
        config_path=config_path,
        episode_catalog_sha256=episode_catalog_sha256,
        selected=selected,
        training_episode_catalog_sha256=training_episode_catalog_sha256,
        training_refs=training_refs,
        training_source_identity_catalog=training_source_identity_catalog,
        evaluation_source_identity_catalog=evaluation_source_identity_catalog,
        selected_source_episode_indices=selected_source_episode_indices,
    )
    if manifest_guarantees_holdout and training_source_identity_catalog is not None:
        assert evaluation_source_identity_catalog is not None
        assert selected_source_episode_indices is not None
        proof_payload = manifest_payload.get("holdout_proof") if manifest_payload else None
        source_content_holdout_proof = {
            "proof_kind": (
                proof_payload.get("proof_kind")
                if isinstance(proof_payload, Mapping)
                else None
            ),
            "episode_identity_algorithm": (
                proof_payload.get("episode_identity_algorithm")
                if isinstance(proof_payload, Mapping)
                else None
            ),
            "training": {
                "episode_count": training_source_identity_catalog.episode_count,
                "frame_count": training_source_identity_catalog.frame_count,
                "source_content_catalog_sha256": (
                    training_source_identity_catalog.content_catalog_sha256
                ),
                "source_statistics_catalog_sha256": (
                    training_source_identity_catalog.statistics_catalog_sha256
                ),
            },
            "evaluation": {
                "episode_count": evaluation_source_identity_catalog.episode_count,
                "frame_count": evaluation_source_identity_catalog.frame_count,
                "source_content_catalog_sha256": (
                    evaluation_source_identity_catalog.content_catalog_sha256
                ),
                "source_statistics_catalog_sha256": (
                    evaluation_source_identity_catalog.statistics_catalog_sha256
                ),
            },
            "selected_evaluation_episodes": [
                evaluation_source_identity_catalog.episode(episode_index).manifest_record()
                for episode_index in selected_source_episode_indices
            ],
            "source_trajectory_or_full_state_overlap_count": 0,
        }
    if not manifest_guarantees_holdout:
        warnings.append(
            "Selected episodes are from the configured dataset and are not proven excluded "
            "from checkpoint training. Treat these as an episode-level regression subset, "
            "not a held-out generalization score."
        )
    for alignment in dataset_statistics_alignment:
        if not alignment["matches_checkpoint"]:
            warnings.append(
                "Evaluation-dataset statistics differ from checkpoint statistics for "
                f"{alignment['dataset_name']} {alignment['modality']}. Raw values will "
                "be normalized with checkpoint statistics for policy inputs and targets; "
                "the evaluation dataset's own statistics are used only to validate its "
                "catalog and loader."
            )
    if args.backend == "server":
        assert_server_checkpoint_matches(checkpoint_path, metadata)
        warnings.append(
            "The current WebSocket protocol does not expose request-scoped RNG seeds. "
            "Server-backed samples are suitable for deployment smoke tests, but not for a "
            "deterministic paired checkpoint A/B; use --backend local for that comparison."
        )

    selected_ref_lookup = {
        (episode_ref.dataset_name, episode_ref.episode_id): episode_ref
        for episode_ref in selected
    }
    planned_state_inputs: dict[PlannedSampleIdentity, np.ndarray] = {}
    image_donor_map: dict[PlannedSampleIdentity, StateMatchedImageDonor] = {}
    if image_ablation_enabled:
        for episode_ref in selected:
            single_dataset = dataset.datasets[episode_ref.dataset_index]
            single_dataset.get_trajectory_data(episode_ref.episode_id)
            for frame_index in frame_plan[
                (episode_ref.dataset_name, episode_ref.episode_id)
            ]:
                identity = PlannedSampleIdentity(
                    dataset_name=episode_ref.dataset_name,
                    episode_id=episode_ref.episode_id,
                    frame_index=int(frame_index),
                )
                raw_state_window = extract_authoritative_raw_modality_window(
                    single_dataset,
                    modality="state",
                    frame_index=int(frame_index),
                )
                state_raw = np.asarray(raw_state_window[0], dtype=np.float32)
                if state_raw.shape != (state_dim,):
                    raise ValueError(
                        f"Ablation-plan state shape {state_raw.shape} != {(state_dim,)} "
                        f"for {identity}."
                    )
                state_norm = normalize_values(
                    state_raw[None, :], state_stats, mode=state_mode
                )
                if clip_deployment_state:
                    state_norm = np.clip(state_norm, -1.0, 1.0)
                planned_state_inputs[identity] = np.ascontiguousarray(
                    state_norm.reshape(-1), dtype=np.float32
                )
        image_donor_map = build_state_matched_image_donor_map(
            planned_state_inputs,
            min_same_episode_frame_gap=image_ablation_min_gap,
        )

    total_attempted = 0
    total_evaluated = 0
    try:
        for episode_ref in selected:
            single_dataset = dataset.datasets[episode_ref.dataset_index]
            frame_indices = frame_plan[
                (episode_ref.dataset_name, episode_ref.episode_id)
            ]
            episode_accumulators = _new_accumulator_set(
                action_dim=action_dim,
                horizon=horizon,
                thresholds=thresholds,
                include_state_matched_image_shuffle=image_ablation_enabled,
            )
            episode_evaluated = 0
            episode_errors: list[dict[str, Any]] = []
            for frame_index in frame_indices:
                total_attempted += 1
                target_identity = PlannedSampleIdentity(
                    dataset_name=episode_ref.dataset_name,
                    episode_id=episode_ref.episode_id,
                    frame_index=int(frame_index),
                )
                sample_seed = _stable_seed(
                    args.seed, episode_ref.dataset_name, episode_ref.episode_id, frame_index
                )
                try:
                    sample, deployment_instruction = load_targeted_training_sample(
                        dataset,
                        single_dataset,
                        episode_id=episode_ref.episode_id,
                        frame_index=frame_index,
                        prompt_seed=sample_seed,
                    )
                    loader_target_norm = np.asarray(sample["action"], dtype=np.float32)
                    if loader_target_norm.shape != (horizon, action_dim):
                        raise ValueError(
                            f"Target action shape {loader_target_norm.shape} != {(horizon, action_dim)}."
                        )
                    valid_mask = build_valid_action_mask(sample, loader_target_norm.shape)
                    if not np.any(valid_mask):
                        sample_records.append(
                            {
                                "dataset_name": episode_ref.dataset_name,
                                "episode_id": episode_ref.episode_id,
                                "frame_index": int(frame_index),
                                "status": "no_valid_actions",
                            }
                        )
                        continue
                    qwen_frames = extract_training_aligned_qwen_frames(
                        sample,
                        video_target_shift_steps=int(
                            cfg.datasets.vla_data.get("video_target_shift_steps", 0)
                        ),
                    )
                    training_state_norm = np.asarray(sample["state"], dtype=np.float32)
                    if training_state_norm.shape != (1, state_dim):
                        raise ValueError(
                            f"Expected normalized state [1,{state_dim}], got "
                            f"{training_state_norm.shape}."
                        )
                    raw_state_window = extract_authoritative_raw_modality_window(
                        single_dataset,
                        modality="state",
                        frame_index=frame_index,
                    )
                    state_raw = raw_state_window[0]
                    if state_raw.shape != (state_dim,):
                        raise ValueError(
                            f"Authoritative raw state shape {state_raw.shape} != {(state_dim,)}."
                        )
                    deployment_state_norm_unclipped = normalize_values(
                        state_raw[None, :], state_stats, mode=state_mode
                    )
                    deployment_state_oob_elements = int(
                        np.sum(np.abs(deployment_state_norm_unclipped) > 1.0)
                    )
                    if args.state_input_mode == "training":
                        state_norm = training_state_norm
                    else:
                        state_norm = deployment_state_norm_unclipped
                        if clip_deployment_state:
                            state_norm = np.clip(state_norm, -1.0, 1.0)
                        state_norm = np.ascontiguousarray(state_norm, dtype=np.float32)
                    if image_ablation_enabled:
                        planned_state = planned_state_inputs[target_identity][None, :]
                        if not np.array_equal(state_norm, planned_state):
                            max_abs = float(np.max(np.abs(state_norm - planned_state)))
                            raise RuntimeError(
                                "Ablation donor planning did not use the exact policy state "
                                f"for {target_identity}; max_abs={max_abs}."
                            )
                    validate_training_aligned_input_contract(
                        qwen_frames=qwen_frames,
                        state=state_norm,
                        metadata=metadata,
                    )
                    diagnostic_subtask_label: str | None = None
                    if prompt_mode == "deployment":
                        instruction = deployment_instruction
                    elif prompt_mode == "training_deterministic":
                        instruction = str(sample["lang"])
                    elif prompt_mode == "subtask_explicit":
                        instruction, diagnostic_subtask_label = (
                            build_subtask_explicit_instruction(
                                single_dataset,
                                sample,
                                deployment_instruction=deployment_instruction,
                            )
                        )
                    else:
                        instruction = str(args.instruction)

                    target_raw = extract_authoritative_raw_modality_window(
                        single_dataset,
                        modality="action",
                        frame_index=frame_index,
                    )
                    if target_raw.shape != (horizon, action_dim):
                        raise ValueError(
                            f"Authoritative raw action shape {target_raw.shape} != "
                            f"{(horizon, action_dim)}."
                        )
                    # The evaluation dataset validates/decodes itself using its
                    # own counted statistics, but both targets and policy state
                    # must use the checkpoint's training normalization.
                    target_norm = normalize_values(
                        target_raw,
                        action_stats,
                        mode=action_mode,
                    )
                    hold_raw = project_realman_state_to_action(state_raw, action_dim=action_dim)
                    hold_norm = normalize_values(hold_raw, action_stats, mode=action_mode)
                    baseline_chunks = {
                        "current_state_hold": np.broadcast_to(hold_norm, target_norm.shape).copy(),
                        "target_h0_repeat": np.broadcast_to(
                            target_norm[0], target_norm.shape
                        ).copy(),
                        "dataset_action_mean": np.broadcast_to(dataset_mean_norm, target_norm.shape).copy(),
                        "normalized_center": np.broadcast_to(center_norm, target_norm.shape).copy(),
                    }

                    image_ablation_match: StateMatchedImageDonor | None = None
                    donor_qwen_frames: np.ndarray | None = None
                    if image_ablation_enabled:
                        image_ablation_match = image_donor_map[target_identity]
                        donor_identity = image_ablation_match.donor
                        donor_ref = selected_ref_lookup[
                            (donor_identity.dataset_name, donor_identity.episode_id)
                        ]
                        donor_dataset = dataset.datasets[donor_ref.dataset_index]
                        donor_sample, _ = load_targeted_training_sample(
                            dataset,
                            donor_dataset,
                            episode_id=donor_identity.episode_id,
                            frame_index=donor_identity.frame_index,
                            prompt_seed=_stable_seed(
                                args.seed,
                                donor_identity.dataset_name,
                                donor_identity.episode_id,
                                donor_identity.frame_index,
                                "image_ablation_donor_decode",
                            ),
                        )
                        donor_qwen_frames = extract_training_aligned_qwen_frames(
                            donor_sample,
                            video_target_shift_steps=int(
                                cfg.datasets.vla_data.get("video_target_shift_steps", 0)
                            ),
                        )
                        validate_training_aligned_input_contract(
                            qwen_frames=donor_qwen_frames,
                            state=state_norm,
                            metadata=metadata,
                        )

                    inference_seed = _stable_seed(sample_seed, "policy_batch")
                    paired_ablation: PairedImageAblationPrediction | None = None
                    if image_ablation_enabled:
                        assert donor_qwen_frames is not None
                        paired_ablation = predict_clean_and_state_matched_image_shuffle(
                            predictor,
                            target_qwen_frames=qwen_frames,
                            donor_qwen_frames=donor_qwen_frames,
                            instruction=instruction,
                            state=state_norm,
                            seed=inference_seed,
                            num_samples=int(args.samples_per_observation),
                        )
                        policy_draws = paired_ablation.clean_draws
                        elapsed_ms = paired_ablation.clean_latency_ms
                        shuffled_policy_draws = paired_ablation.shuffled_draws
                        image_ablation_clean_repeat_latency_ms.append(
                            paired_ablation.clean_repeat_latency_ms
                        )
                        image_ablation_latency_ms.append(
                            paired_ablation.shuffled_latency_ms
                        )
                        image_ablation_clean_repeat_max_abs.append(
                            paired_ablation.clean_repeat_max_abs_difference
                        )
                        image_ablation_clean_repeat_mean_abs.append(
                            paired_ablation.clean_repeat_mean_abs_difference
                        )
                    else:
                        policy_draws, elapsed_ms = predictor.predict_many(
                            qwen_frames=qwen_frames,
                            instruction=instruction,
                            state=state_norm,
                            seed=inference_seed,
                            num_samples=int(args.samples_per_observation),
                        )
                        shuffled_policy_draws = None
                    expected_policy_shape = (
                        int(args.samples_per_observation),
                        horizon,
                        action_dim,
                    )
                    if policy_draws.shape != expected_policy_shape:
                        raise ValueError(
                            f"Policy action shape {policy_draws.shape} != "
                            f"{expected_policy_shape}."
                        )
                    if (
                        shuffled_policy_draws is not None
                        and shuffled_policy_draws.shape != expected_policy_shape
                    ):
                        raise ValueError(
                            "Image-shuffled policy action shape "
                            f"{shuffled_policy_draws.shape} != {expected_policy_shape}."
                        )
                    latency_ms.append(elapsed_ms)
                    ensemble_chunks = policy_ensemble_chunks(policy_draws)
                    shuffled_ensemble_chunks = (
                        policy_ensemble_chunks(shuffled_policy_draws)
                        if shuffled_policy_draws is not None
                        else None
                    )
                    repeat_h0_valid = valid_mask & valid_mask[0][None, :]
                    for accumulators in (global_accumulators, episode_accumulators):
                        for policy_norm in policy_draws:
                            _update_method(
                                accumulators["policy"],
                                prediction_normalized=policy_norm,
                                action_stats=action_stats,
                                action_mode=action_mode,
                                target_normalized=target_norm,
                                target_raw=target_raw,
                                valid_mask=valid_mask,
                            )
                        for ensemble_name, ensemble_norm in ensemble_chunks.items():
                            _update_method(
                                accumulators[ensemble_name],
                                prediction_normalized=ensemble_norm,
                                action_stats=action_stats,
                                action_mode=action_mode,
                                target_normalized=target_norm,
                                target_raw=target_raw,
                                valid_mask=valid_mask,
                            )
                        _update_method(
                            accumulators["policy_ensemble_median_target_h0_mask"],
                            prediction_normalized=ensemble_chunks[
                                "policy_ensemble_median"
                            ],
                            action_stats=action_stats,
                            action_mode=action_mode,
                            target_normalized=target_norm,
                            target_raw=target_raw,
                            valid_mask=repeat_h0_valid,
                        )
                        for baseline_name, baseline_norm in baseline_chunks.items():
                            _update_method(
                                accumulators[baseline_name],
                                prediction_normalized=baseline_norm,
                                action_stats=action_stats,
                                action_mode=action_mode,
                                target_normalized=target_norm,
                                target_raw=target_raw,
                                valid_mask=(
                                    repeat_h0_valid
                                    if baseline_name == "target_h0_repeat"
                                    else valid_mask
                                ),
                                clip_prediction=False,
                            )
                        if shuffled_policy_draws is not None:
                            assert shuffled_ensemble_chunks is not None
                            for shuffled_norm in shuffled_policy_draws:
                                _update_method(
                                    accumulators[
                                        "policy_state_matched_image_shuffle"
                                    ],
                                    prediction_normalized=shuffled_norm,
                                    action_stats=action_stats,
                                    action_mode=action_mode,
                                    target_normalized=target_norm,
                                    target_raw=target_raw,
                                    valid_mask=valid_mask,
                                )
                            for source_name, destination_name in (
                                (
                                    "policy_ensemble_mean",
                                    "policy_state_matched_image_shuffle_ensemble_mean",
                                ),
                                (
                                    "policy_ensemble_median",
                                    "policy_state_matched_image_shuffle_ensemble_median",
                                ),
                            ):
                                _update_method(
                                    accumulators[destination_name],
                                    prediction_normalized=shuffled_ensemble_chunks[
                                        source_name
                                    ],
                                    action_stats=action_stats,
                                    action_mode=action_mode,
                                    target_normalized=target_norm,
                                    target_raw=target_raw,
                                    valid_mask=valid_mask,
                                )
                    if shuffled_policy_draws is not None:
                        assert paired_image_output_delta is not None
                        for clean_norm, shuffled_norm in zip(
                            policy_draws, shuffled_policy_draws, strict=True
                        ):
                            clean_raw = unnormalize_values(
                                clean_norm,
                                action_stats,
                                mode=action_mode,
                            )
                            shuffled_raw = unnormalize_values(
                                shuffled_norm,
                                action_stats,
                                mode=action_mode,
                            )
                            paired_image_output_delta.update(
                                prediction_normalized=shuffled_norm,
                                prediction_raw=shuffled_raw,
                                target_normalized=clean_norm,
                                target_raw=clean_raw,
                                valid_mask=valid_mask,
                            )
                    episode_evaluated += 1
                    total_evaluated += 1
                    draw_std = np.std(policy_draws, axis=0)
                    median_norm = ensemble_chunks["policy_ensemble_median"]
                    median_raw = unnormalize_values(
                        median_norm, action_stats, mode=action_mode
                    )
                    policy_draws_raw = unnormalize_values(
                        policy_draws, action_stats, mode=action_mode
                    )
                    shuffled_median_norm: np.ndarray | None = None
                    shuffled_median_raw: np.ndarray | None = None
                    shuffled_policy_draws_raw: np.ndarray | None = None
                    image_ablation_sample_record: dict[str, Any] | None = None
                    if shuffled_policy_draws is not None:
                        assert shuffled_ensemble_chunks is not None
                        assert image_ablation_match is not None
                        assert paired_ablation is not None
                        shuffled_median_norm = shuffled_ensemble_chunks[
                            "policy_ensemble_median"
                        ]
                        shuffled_median_raw = unnormalize_values(
                            shuffled_median_norm,
                            action_stats,
                            mode=action_mode,
                        )
                        shuffled_policy_draws_raw = unnormalize_values(
                            shuffled_policy_draws,
                            action_stats,
                            mode=action_mode,
                        )
                        image_ablation_sample_record = {
                            **image_ablation_match.as_dict(),
                            "paired_nonvisual_inputs_identical": True,
                            "paired_policy_seed": int(inference_seed),
                            "target_frames_sha256": (
                                paired_ablation.target_frames_sha256
                            ),
                            "donor_frames_sha256": (
                                paired_ablation.donor_frames_sha256
                            ),
                            "pixel_mean_absolute_difference": (
                                paired_ablation.pixel_mean_absolute_difference
                            ),
                            "batched_inference_latency_ms": (
                                paired_ablation.shuffled_latency_ms
                            ),
                            "same_image_repeat": {
                                "batched_inference_latency_ms": (
                                    paired_ablation.clean_repeat_latency_ms
                                ),
                                "max_abs_normalized_action_difference": (
                                    paired_ablation.clean_repeat_max_abs_difference
                                ),
                                "mean_abs_normalized_action_difference": (
                                    paired_ablation.clean_repeat_mean_abs_difference
                                ),
                                "max_abs_tolerance": (
                                    IMAGE_ABLATION_DETERMINISM_MAX_ABS_TOLERANCE
                                ),
                            },
                            "paired_output_delta": build_per_sample_prefix_metrics(
                                predictions={
                                    "state_matched_image_shuffle_vs_clean": (
                                        shuffled_median_norm,
                                        shuffled_median_raw,
                                        valid_mask,
                                    )
                                },
                                target_normalized=median_norm,
                                target_raw=median_raw,
                                arm_dimensions=arm_dimensions,
                                gripper_dimensions=gripper_dimensions,
                                gripper_thresholds=thresholds,
                            ),
                        }
                    per_sample_predictions = {
                        "policy_ensemble_median": (
                            median_norm,
                            median_raw,
                            valid_mask,
                        ),
                        "current_state_hold": (
                            baseline_chunks["current_state_hold"],
                            np.broadcast_to(hold_raw, target_raw.shape),
                            valid_mask,
                        ),
                        "target_h0_repeat": (
                            baseline_chunks["target_h0_repeat"],
                            np.broadcast_to(target_raw[0], target_raw.shape),
                            repeat_h0_valid,
                        ),
                        "policy_median_h0_repeat": (
                            ensemble_chunks["policy_median_h0_repeat"],
                            unnormalize_values(
                                ensemble_chunks["policy_median_h0_repeat"],
                                action_stats,
                                mode=action_mode,
                            ),
                            valid_mask,
                        ),
                    }
                    if shuffled_median_norm is not None:
                        assert shuffled_median_raw is not None
                        per_sample_predictions[
                            "policy_state_matched_image_shuffle_ensemble_median"
                        ] = (
                            shuffled_median_norm,
                            shuffled_median_raw,
                            valid_mask,
                        )
                    per_sample_prefix_metrics = build_per_sample_prefix_metrics(
                        predictions=per_sample_predictions,
                        target_normalized=target_norm,
                        target_raw=target_raw,
                        arm_dimensions=arm_dimensions,
                        gripper_dimensions=gripper_dimensions,
                        gripper_thresholds=thresholds,
                    )
                    motion_from_state = _motion_from_state_arm_first20(
                        target_raw=target_raw,
                        hold_raw=hold_raw,
                        valid_mask=valid_mask,
                        arm_dimensions=arm_dimensions,
                    )
                    subtask_value = sample.get("subtask_index")
                    subtask_index = (
                        int(np.asarray(subtask_value).reshape(-1)[0])
                        if subtask_value is not None
                        else None
                    )
                    gripper_prefix = min(10, horizon)
                    gripper_dims_array = np.asarray(
                        tuple(gripper_dimensions), dtype=np.int64
                    )
                    gripper_forecast = {
                        "action_dimensions": gripper_dims_array.tolist(),
                        "thresholds_raw": thresholds[gripper_dims_array].tolist(),
                        "target_raw": target_raw[
                            :gripper_prefix, gripper_dims_array
                        ].tolist(),
                        "policy_ensemble_median_raw": median_raw[
                            :gripper_prefix, gripper_dims_array
                        ].tolist(),
                        "policy_draws_raw": policy_draws_raw[
                            :, :gripper_prefix, gripper_dims_array
                        ].tolist(),
                    }
                    sample_records.append(
                        {
                            "dataset_name": episode_ref.dataset_name,
                            "episode_id": episode_ref.episode_id,
                            "requested_frame_index": int(frame_index),
                            "dataset_frame_index": _json_safe(sample.get("frame_index")),
                            "instruction": instruction,
                            "prompt_had_subtask_suffix": instruction != deployment_instruction,
                            "diagnostic_subtask_label": diagnostic_subtask_label,
                            "state_input_mode": str(args.state_input_mode),
                            "deployment_state_oob_elements_before_clip": (
                                deployment_state_oob_elements
                            ),
                            "sent_vs_training_state_max_abs": float(
                                np.max(np.abs(state_norm - training_state_norm))
                            ),
                            "subtask_index": subtask_index,
                            "valid_mask_kind": (
                                "full" if bool(np.all(valid_mask)) else "partial"
                            ),
                            "valid_action_elements": int(valid_mask.sum()),
                            "valid_timesteps": int(np.any(valid_mask, axis=1).sum()),
                            "policy_draws": int(policy_draws.shape[0]),
                            "batched_inference_latency_ms": elapsed_ms,
                            "normalized_draw_std_mean_valid": float(
                                draw_std[valid_mask].mean()
                            ),
                            "motion_from_state_arm_raw_mae_first20": motion_from_state,
                            "motion_from_state_bin": _motion_bin(motion_from_state),
                            "gripper_forecast_first10": gripper_forecast,
                            "image_ablation": image_ablation_sample_record,
                            **(
                                {
                                    "policy_draws_normalized": policy_draws.tolist(),
                                    "policy_draws_raw": policy_draws_raw.tolist(),
                                    **(
                                        {
                                            "state_matched_image_shuffle_policy_draws_normalized": (
                                                shuffled_policy_draws.tolist()
                                            ),
                                            "state_matched_image_shuffle_policy_draws_raw": (
                                                shuffled_policy_draws_raw.tolist()
                                            ),
                                        }
                                        if shuffled_policy_draws is not None
                                        and shuffled_policy_draws_raw is not None
                                        else {}
                                    ),
                                }
                                if args.store_policy_actions
                                else {}
                            ),
                            **per_sample_prefix_metrics,
                            "status": "evaluated",
                        }
                    )
                except Exception as exc:
                    error = {
                        "frame_index": int(frame_index),
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    episode_errors.append(error)
                    if not args.allow_sample_errors:
                        raise

            episode_reports.append(
                {
                    **episode_ref.as_dict(),
                    "sampled_frame_indices": frame_indices,
                    "frames_evaluated": episode_evaluated,
                    "errors": episode_errors,
                    "metrics": {
                        name: accumulator.finalize(include_horizons=False)
                        for name, accumulator in episode_accumulators.items()
                    },
                }
            )
    finally:
        predictor.close()

    if total_evaluated == 0:
        raise RuntimeError("No samples with valid action targets were evaluated.")
    finalized_metrics = {
        name: accumulator.finalize(include_horizons=True)
        for name, accumulator in global_accumulators.items()
    }
    median_report = finalized_metrics["policy_ensemble_median"]
    comparisons = {
        f"policy_ensemble_median_vs_{baseline_name}": build_horizon_comparison(
            (
                finalized_metrics["policy_ensemble_median_target_h0_mask"]
                if baseline_name == "target_h0_repeat"
                else median_report
            ),
            finalized_metrics[baseline_name],
        )
        for baseline_name in (
            "current_state_hold",
            "target_h0_repeat",
            "policy_median_h0_repeat",
        )
    }
    if image_ablation_enabled:
        if len(sample_records) != total_evaluated:
            raise RuntimeError(
                "Paired image-ablation accounting is inconsistent: "
                f"{len(sample_records)} completed observation records versus "
                f"{total_evaluated} metric updates. Refusing to bootstrap an "
                "ambiguous paired population."
            )
        paired_arm_target_error_bootstrap = (
            build_paired_arm_target_error_bootstrap(
                sample_records,
                eval_seed=int(args.seed),
            )
        )
        comparisons[
            "clean_policy_ensemble_median_vs_state_matched_image_shuffle_ensemble_median"
        ] = build_horizon_comparison(
            median_report,
            finalized_metrics[
                "policy_state_matched_image_shuffle_ensemble_median"
            ],
        )
        assert paired_image_output_delta is not None
        image_ablation_report = {
            "mode": image_ablation_mode,
            "pairing": "clean_and_ablated_in_same_local_run",
            "paired_observations": total_evaluated,
            "donor_plan_sha256": _image_ablation_plan_fingerprint(
                image_donor_map
            ),
            "donor_selection": {
                "distance": "rms_l2_over_checkpoint_normalized_deployment_state",
                "different_episode_candidates": "eligible",
                "minimum_same_episode_frame_gap": image_ablation_min_gap,
                "tie_break": "lexicographic_sample_identity",
                "target_or_adjacent_self_donor_allowed": False,
            },
            "paired_invariants": {
                "target_action_and_valid_mask": "identical",
                "policy_state": "identical_checkpoint_normalized_deployment_state",
                "instruction": "identical",
                "policy_noise": "identical_seed_reset_before_each_local_call",
                "samples_per_observation": "identical",
                "only_changed_policy_input": "qwen_frames",
                "byte_identical_target_and_donor_images_allowed": False,
            },
            "target_metric_comparison_direction": {
                "candidate": "clean policy ensemble median",
                "baseline": "state-matched image-shuffle policy ensemble median",
                "positive_relative_improvement_means": (
                    "clean images reduce target-action error relative to shuffled images"
                ),
            },
            "paired_arm_target_error_bootstrap": (
                paired_arm_target_error_bootstrap
            ),
            "same_image_repeat_determinism": {
                "checked_for_every_paired_observation": True,
                "max_abs_normalized_action_tolerance": (
                    IMAGE_ABLATION_DETERMINISM_MAX_ABS_TOLERANCE
                ),
                "maximum_observed_max_abs_normalized_action_difference": (
                    max(image_ablation_clean_repeat_max_abs)
                    if image_ablation_clean_repeat_max_abs
                    else None
                ),
                "mean_observed_mean_abs_normalized_action_difference": (
                    float(np.mean(image_ablation_clean_repeat_mean_abs))
                    if image_ablation_clean_repeat_mean_abs
                    else None
                ),
                "repeat_latency": _latency_summary(
                    image_ablation_clean_repeat_latency_ms
                ),
            },
            "state_matched_image_shuffle_latency": _latency_summary(
                image_ablation_latency_ms
            ),
            "paired_policy_output_delta": paired_image_output_delta.finalize(
                include_horizons=True
            ),
        }
    else:
        image_ablation_report = {"mode": "none"}
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint_requested": str(checkpoint_path),
        "config_path": str(config_path),
        "backend": predictor.description,
        "policy_contract": {
            "run_id": metadata.get("run_id"),
            "data_mix": metadata.get("data_mix"),
            "action_type": metadata.get("action_type"),
            "action_dim": action_dim,
            "state_dim": state_dim,
            "action_horizon": horizon,
            "camera_order": metadata.get("camera_order_hint"),
            "qwen_input_contract": metadata.get("realman_input_contract"),
            "normalization_key": norm_key,
            "action_normalization_mode": action_mode,
            "state_normalization_mode": state_mode,
            "normalization_stats_sha256": _stats_fingerprint(
                {"action": action_stats, "state": state_stats}
            ),
        },
        "dataset": {
            "root": str(cfg.datasets.vla_data.data_root_dir),
            "data_mix": str(cfg.datasets.vla_data.data_mix),
            "episode_count": len(refs),
            "episode_catalog_sha256": episode_catalog_sha256,
            "statistics_alignment_with_checkpoint": dataset_statistics_alignment,
            "training_dataset_root_for_holdout_proof": (
                str(training_root_for_holdout_proof)
                if training_root_for_holdout_proof is not None
                else None
            ),
            "training_episode_catalog_sha256": training_episode_catalog_sha256,
            "source_content_holdout_proof": source_content_holdout_proof,
        },
        "episode_split": {
            "selection_kind": "manifest" if explicit_ids is not None else "sha256_sorted",
            "seed": int(args.seed),
            "manifest_path": str(Path(args.episode_manifest).resolve()) if args.episode_manifest else None,
            "excluded_from_training": manifest_guarantees_holdout,
            "semantic_label": (
                "held_out_from_training"
                if manifest_guarantees_holdout
                else "deterministic_episode_regression_subset"
            ),
            "selected_episodes": [ref.as_dict() for ref in selected],
        },
        "sampling": {
            "frames_per_episode": int(args.frames_per_episode),
            "frame_selection_kind": (
                "manifest_explicit"
                if explicit_frame_plan is not None
                else "deterministic_stratified"
            ),
            "frames_per_episode_ignored_due_to_manifest_plan": (
                explicit_frame_plan is not None
            ),
            "samples_per_observation": int(args.samples_per_observation),
            "stored_policy_actions": bool(args.store_policy_actions),
            "image_ablation_mode": image_ablation_mode,
            "sample_plan_sha256": _sample_plan_fingerprint(frame_plan),
            "policy_sampling": "single_batched_stochastic_call_per_observation",
            "prompt_mode": prompt_mode,
            "state_input_mode": str(args.state_input_mode),
            "deployment_state_clipped_to_unit_range": bool(
                clip_deployment_state
            ),
            "policy_output_inverse": "training_aligned_affine_without_clipping",
            "policy_outputs_clipped_to_unit_range_for_raw_metrics": False,
            "raw_target_source": "authoritative_float32_parquet_rows",
            "target_normalization_source": "checkpoint_dataset_statistics",
            "target_video_frame": "current_context_frame",
            "video_target_shift_steps": int(
                cfg.datasets.vla_data.get("video_target_shift_steps", 0)
            ),
            "attempted_frames": total_attempted,
            "evaluated_frames": total_evaluated,
        },
        "baseline_definitions": {
            "policy": "All stochastic policy draws, pooled as separate predictions.",
            "policy_ensemble_mean": "Elementwise mean of batched normalized policy draws.",
            "policy_ensemble_median": "Elementwise median of batched normalized policy draws.",
            "policy_ensemble_median_target_h0_mask": (
                "The ensemble-median policy evaluated only where the h0 target and the "
                "compared future target are both valid; this is the paired population for "
                "target_h0_repeat."
            ),
            "policy_median_h0_repeat": (
                "Repeat the ensemble-median policy action at action index 0 over the horizon."
            ),
            "current_state_hold": "Repeat current raw state projected into absolute policy action layout.",
            "target_h0_repeat": (
                "Repeat the dataset target at action index 0; dimensions with an invalid h0 "
                "target are excluded. This oracle-persistence baseline measures future-action "
                "prediction beyond the observed first target."
            ),
            "dataset_action_mean": "Repeat checkpoint dataset-statistics raw action mean.",
            "normalized_center": "Repeat zero in normalized policy action space.",
            **(
                {
                    "policy_state_matched_image_shuffle": (
                        "All same-seed policy draws with only qwen_frames replaced "
                        "by the deterministic nearest-state donor payload."
                    ),
                    "policy_state_matched_image_shuffle_ensemble_mean": (
                        "Elementwise mean of the paired state-matched image-shuffle draws."
                    ),
                    "policy_state_matched_image_shuffle_ensemble_median": (
                        "Elementwise median of the paired state-matched image-shuffle draws."
                    ),
                }
                if image_ablation_enabled
                else {}
            ),
        },
        "batched_inference_latency": _latency_summary(latency_ms),
        "image_ablation": image_ablation_report,
        "metrics": finalized_metrics,
        "comparisons": comparisons,
        "episodes": episode_reports,
        "samples": sample_records,
        "warnings": warnings,
    }
    # Reject NaN/Infinity in the artifact; missing metrics are encoded as null.
    json.dumps(_json_safe(report), allow_nan=False)
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--training-dataset-root",
        type=Path,
        help=(
            "Authoritative checkpoint training catalog. Required to verify an "
            "external held-out manifest when the path saved in config.yaml is not "
            "available on this host."
        ),
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--backend", choices=("local", "server"), default="local")
    parser.add_argument("--device", default="cuda:0", help="Local checkpoint device.")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=10093)
    parser.add_argument("--server-timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        help=(
            "JSON with `datasets: {name: [episode_ids...]}` and optional exact "
            "`frames: {name: {episode_id_string: [indices...]}}`. Set top-level "
            "`excluded_from_training: true` only when the training split enforces it."
        ),
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=16,
        help="Deterministic stratified frames per selected episode; 0 evaluates every frame.",
    )
    parser.add_argument("--samples-per-observation", type=int, default=1)
    parser.add_argument(
        "--global-bf16-parameter-cast",
        action="store_true",
        help=(
            "Diagnostic local-backend mode matching the historical server --use_bf16: "
            "globally cast all model parameters, including the FP32 action/state "
            "modules, to BF16."
        ),
    )
    parser.add_argument(
        "--store-policy-actions",
        action="store_true",
        help="Include exact normalized and raw policy chunks in each sample record.",
    )
    parser.add_argument(
        "--image-ablation",
        choices=("none", "state_matched_shuffle"),
        default="none",
        help=(
            "state_matched_shuffle performs paired local clean/ablated inference. "
            "It resets the same policy seed and holds target state, instruction, "
            "action labels, and masks fixed while replacing only qwen_frames with "
            "those from the nearest checkpoint-normalized-state donor."
        ),
    )
    parser.add_argument(
        "--image-ablation-min-same-episode-frame-gap",
        type=int,
        default=20,
        help=(
            "Minimum target/donor frame separation when both come from the same "
            "episode; other-episode donors are not subject to this temporal gap."
        ),
    )
    parser.add_argument(
        "--state-input-mode",
        choices=("deployment", "training"),
        default="deployment",
        help=(
            "deployment reconstructs float32 raw parquet state and follows the Realman "
            "client normalization path; training reuses the loader's normalized state."
        ),
    )
    parser.add_argument(
        "--clip-deployment-state",
        dest="clip_deployment_state",
        action="store_true",
        help=(
            "Force deployment-mode state clipping. By default the evaluator follows "
            "training semantics: q99 clips; min_max and mean_std do not."
        ),
    )
    parser.add_argument(
        "--no-clip-deployment-state",
        dest="clip_deployment_state",
        action="store_false",
        help="Force deployment-mode normalized state to remain unclipped.",
    )
    parser.set_defaults(clip_deployment_state=None)
    parser.add_argument(
        "--prompt-mode",
        choices=(
            "deployment",
            "training_deterministic",
            "subtask_explicit",
            "explicit",
        ),
        default="deployment",
        help=(
            "deployment uses the task instruction sent by the robot; training_deterministic "
            "uses the seeded training-time subtask augmentation; subtask_explicit always "
            "appends the authoritative current local-stage label with the training "
            "separator and fails on missing/unlabeled stages; explicit uses --instruction."
        ),
    )
    parser.add_argument("--instruction")
    parser.add_argument(
        "--allow-sample-errors",
        action="store_true",
        help="Record decode/sample failures and continue. Default is fail closed to avoid biased metrics.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.samples_per_observation <= 0:
        raise ValueError("--samples-per-observation must be positive.")
    if args.global_bf16_parameter_cast and args.backend != "local":
        raise ValueError("--global-bf16-parameter-cast requires --backend local.")
    if args.image_ablation_min_same_episode_frame_gap < 1:
        raise ValueError(
            "--image-ablation-min-same-episode-frame-gap must be at least 1."
        )
    report = run_audit(args)
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = build_cli_headline(report, report_path=report_path)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
