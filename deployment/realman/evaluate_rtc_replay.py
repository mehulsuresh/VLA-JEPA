"""Sequential, read-only replay evaluation for Realman RTC policies.

This evaluator answers a narrower question than ``audit_checkpoint_offline``:
given one contiguous recorded episode segment, what would be executed by

* a fresh synchronous policy request at every frame (h0),
* a fresh open-loop plan held for a fixed chunk size, and
* an overlapping RTC plan with a fixed inference delay/prefix length?

RTC alignment is intentionally explicit.  A fresh bootstrap request completes
synchronously and executes its first ``delay`` rows.  Thereafter, if an active
plan was anchored at frame ``a`` and a replacement request starts at frame
``t``, the clean prefix sent to the model is
``active_plan[t-a : t-a+delay]``.  That committed prefix executes while the
request is in flight.  The replacement plan is anchored at ``t`` but is not
activated until frame ``t+delay``; execution therefore begins at its action
index ``delay``.  Passing the old plan from index zero would condition on
actions for the wrong recorded frames.

The utility never commands a robot, contacts a policy server, or changes a
checkpoint.  It loads a checkpoint locally in inference mode and reads an
existing dataset.  Inputs follow the deployed Realman contract: three RGB
uint8 Qwen frames, the deployment-normalized 19D state, and one explicit task
instruction supplied on the command line.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from deployment.realman.audit_checkpoint_offline import (
    LocalPredictor,
    _json_safe,
    _metadata_stats,
    _resolve_run_file,
    _stable_seed,
    _stats_fingerprint,
    assert_checkpoint_dataset_stats_match,
    build_valid_action_mask,
    enumerate_episodes,
    extract_authoritative_raw_modality_window,
    extract_training_aligned_qwen_frames,
    load_targeted_training_sample,
    merged_dataset_modality_stats,
    normalize_values,
    realman_action_groups,
    validate_dataset_camera_order,
    validate_training_aligned_input_contract,
)
from deployment.realman.pipeline import realman_continuous_unnormalize


SCHEMA_VERSION = 3
DEFAULT_OPEN_LOOP_CHUNK_SIZES = (5, 10, 20)
DEFAULT_RTC_PREFIX_LENGTHS = (1, 5, 10)
# Must match the live RtcPrefixPlanner acceptance threshold.
DEFAULT_PREFIX_COPY_TOLERANCE = 5e-3


def _plain_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def fresh_plan_inference_seed(
    *, base_seed: int, dataset_name: str, episode_id: int, frame_index: int
) -> int:
    """One canonical seed shared by cached fresh and paired RTC requests."""

    sample_seed = _stable_seed(
        int(base_seed), str(dataset_name), int(episode_id), int(frame_index)
    )
    return _stable_seed(sample_seed, "fresh_plan")


def validate_contiguous_frame_range(
    *, episode_length: int, start_frame: int, num_frames: int
) -> tuple[int, ...]:
    """Return an exact contiguous range, rejecting clipping and empty ranges."""

    length = _plain_positive_int(episode_length, name="episode_length")
    count = _plain_positive_int(num_frames, name="num_frames")
    if isinstance(start_frame, (bool, np.bool_)) or not isinstance(
        start_frame, (int, np.integer)
    ):
        raise ValueError(f"start_frame must be an integer, got {start_frame!r}.")
    start = int(start_frame)
    if start < 0:
        raise ValueError(f"start_frame must be non-negative, got {start}.")
    stop = start + count
    if stop > length:
        raise ValueError(
            f"Replay range [{start}, {stop}) exceeds episode length {length}; "
            "refusing to clip a supposedly contiguous replay."
        )
    return tuple(range(start, stop))


def validate_contiguous_frame_indices(frame_indices: Sequence[int]) -> tuple[int, ...]:
    """Validate an already materialized frame sequence without sorting it."""

    if not frame_indices:
        raise ValueError("Replay frame sequence is empty.")
    canonical: list[int] = []
    for position, value in enumerate(frame_indices):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"Replay frame index at position {position} must be an integer, got {value!r}."
            )
        canonical.append(int(value))
    for previous, current in zip(canonical, canonical[1:]):
        if current != previous + 1:
            raise ValueError(
                "Replay frames must be in original contiguous order; "
                f"found {previous} followed by {current}."
            )
    return tuple(canonical)


def validate_replay_method_configuration(
    *,
    horizon: int,
    num_frames: int,
    open_loop_chunk_sizes: Sequence[int],
    rtc_prefix_lengths: Sequence[int],
    rtc_max_prefix_len: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Canonicalize method sizes and reject configurations that cannot align."""

    action_horizon = _plain_positive_int(horizon, name="horizon")
    replay_length = _plain_positive_int(num_frames, name="num_frames")
    if isinstance(rtc_max_prefix_len, (bool, np.bool_)) or not isinstance(
        rtc_max_prefix_len, (int, np.integer)
    ):
        raise ValueError("rtc_max_prefix_len must be an integer.")
    max_prefix = int(rtc_max_prefix_len)
    if max_prefix < 0:
        raise ValueError("rtc_max_prefix_len must be non-negative.")

    def _canonical(values: Sequence[int], *, label: str) -> tuple[int, ...]:
        parsed = tuple(_plain_positive_int(value, name=label) for value in values)
        if not parsed:
            raise ValueError(f"At least one {label} is required.")
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"Duplicate {label} values are not allowed: {parsed}.")
        return tuple(sorted(parsed))

    chunks = _canonical(open_loop_chunk_sizes, label="open_loop_chunk_size")
    prefixes = _canonical(rtc_prefix_lengths, label="rtc_prefix_len")
    for chunk_size in chunks:
        if chunk_size > action_horizon:
            raise ValueError(
                f"open_loop_chunk_size={chunk_size} exceeds action horizon {action_horizon}."
            )
    for prefix_len in prefixes:
        if prefix_len > max_prefix:
            raise ValueError(
                f"rtc_prefix_len={prefix_len} exceeds checkpoint maximum {max_prefix}."
            )
        if 2 * prefix_len >= action_horizon:
            raise ValueError(
                f"rtc_prefix_len={prefix_len} cannot be repeatedly aligned inside horizon "
                f"{action_horizon}: activation-discontinuity scoring needs prior-plan "
                f"index {2 * prefix_len} to exist."
            )
        if replay_length <= 2 * prefix_len:
            raise ValueError(
                f"num_frames={replay_length} must exceed twice rtc_prefix_len={prefix_len}; "
                "otherwise no conditioned plan suffix is ever executed after bootstrap."
            )
        if replay_length % prefix_len != 0:
            raise ValueError(
                f"num_frames={replay_length} must be divisible by rtc_prefix_len={prefix_len}; "
                "the fixed-delay replay refuses a partial committed prefix at its boundary."
            )
    return chunks, prefixes


def validate_rtc_inference_contract(
    metadata: Mapping[str, Any], *, requested_prefix_lengths: Sequence[int]
) -> dict[str, Any]:
    """Require a checkpoint-advertised prefix RTC contract; never infer one."""

    contract = metadata.get("rtc_inference_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Checkpoint metadata is missing rtc_inference_contract.")
    if contract.get("training_enabled") is not True:
        raise ValueError("Checkpoint metadata says RTC training was not enabled.")
    if contract.get("method") != "prefix":
        raise ValueError(
            f"Unsupported RTC method {contract.get('method')!r}; expected 'prefix'."
        )
    if contract.get("action_space") != "normalized_policy_action":
        raise ValueError(
            "RTC prev_actions must use normalized_policy_action space; checkpoint "
            f"advertises {contract.get('action_space')!r}."
        )
    if contract.get("client_opt_in") is not True:
        raise ValueError("Checkpoint RTC contract does not require explicit client opt-in.")
    max_prefix_value = contract.get("max_prefix_len")
    if isinstance(max_prefix_value, (bool, np.bool_)) or not isinstance(
        max_prefix_value, (int, np.integer)
    ):
        raise ValueError("RTC contract max_prefix_len must be an integer.")
    max_prefix_len = int(max_prefix_value)
    if max_prefix_len <= 0:
        raise ValueError(f"RTC contract max_prefix_len must be positive, got {max_prefix_len}.")
    for requested in requested_prefix_lengths:
        prefix_len = _plain_positive_int(requested, name="rtc_prefix_len")
        if prefix_len > max_prefix_len:
            raise ValueError(
                f"Requested RTC prefix {prefix_len} exceeds checkpoint maximum {max_prefix_len}."
            )
    return dict(contract)


def validate_unit_step_action_offsets(single_dataset: Any, *, horizon: int) -> None:
    """Prove action index k refers to recorded frame t+k for every action key."""

    expected = np.arange(int(horizon), dtype=np.int64)
    action_keys = tuple(single_dataset.modality_keys["action"])
    if not action_keys:
        raise ValueError("Dataset has no action modality keys.")
    for key in action_keys:
        actual = np.asarray(single_dataset.delta_indices[key], dtype=np.int64).reshape(-1)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                f"Action key {key!r} uses offsets {actual.tolist()}, not unit-step "
                f"recorded-frame offsets {expected.tolist()}; sequential plan alignment "
                "cannot be proven."
            )


def validate_returned_frame_index(sample: Mapping[str, Any], *, requested: int) -> None:
    if "frame_index" not in sample:
        raise ValueError("Targeted dataset sample is missing frame_index.")
    reported = np.asarray(sample["frame_index"]).reshape(-1)
    if reported.size != 1:
        raise ValueError(
            f"Targeted dataset frame_index must contain one value, got shape {reported.shape}."
        )
    value = reported[0]
    if isinstance(value, (bool, np.bool_)) or not np.issubdtype(reported.dtype, np.integer):
        raise ValueError(f"Targeted dataset frame_index must be integral, got {value!r}.")
    if int(value) != int(requested):
        raise ValueError(
            f"Requested recorded frame {requested}, but targeted loader returned {int(value)}."
        )


def validate_normalized_plan(
    plan: Any, *, horizon: int, action_dim: int, name: str = "plan"
) -> np.ndarray:
    array = np.asarray(plan, dtype=np.float32)
    expected = (int(horizon), int(action_dim))
    if array.shape != expected:
        raise ValueError(f"{name} shape {array.shape} does not match {expected}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite normalized actions.")
    return np.ascontiguousarray(array, dtype=np.float32)


def align_prior_normalized_prefix(
    prior_plan: Any,
    *,
    prior_anchor_frame: int,
    request_frame: int,
    prefix_len: int,
) -> np.ndarray:
    """Slice an active plan by elapsed recorded frames for an RTC request."""

    plan = np.asarray(prior_plan, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[0] <= 0 or plan.shape[1] <= 0:
        raise ValueError(f"prior_plan must have shape [H,D], got {plan.shape}.")
    if not np.isfinite(plan).all():
        raise ValueError("prior_plan contains non-finite normalized actions.")
    length = _plain_positive_int(prefix_len, name="prefix_len")
    if isinstance(prior_anchor_frame, (bool, np.bool_)) or not isinstance(
        prior_anchor_frame, (int, np.integer)
    ):
        raise ValueError("prior_anchor_frame must be an integer.")
    if isinstance(request_frame, (bool, np.bool_)) or not isinstance(
        request_frame, (int, np.integer)
    ):
        raise ValueError("request_frame must be an integer.")
    elapsed = int(request_frame) - int(prior_anchor_frame)
    if elapsed < 0:
        raise ValueError(
            f"request_frame={request_frame} precedes prior plan anchor {prior_anchor_frame}."
        )
    stop = elapsed + length
    if stop > plan.shape[0]:
        raise ValueError(
            f"Aligned RTC prefix [{elapsed}:{stop}] exceeds prior plan horizon {plan.shape[0]}."
        )
    return np.ascontiguousarray(plan[elapsed:stop], dtype=np.float32)


def align_prior_normalized_plan(
    prior_plan: Any,
    *,
    prior_anchor_frame: int,
    request_frame: int,
) -> np.ndarray:
    """Build the exact full-horizon ``prev_actions`` used by the live client.

    Unexecuted rows are shifted to index zero.  Any unavailable tail is filled
    with the prior plan's final row, matching ``RtcPrefixPlanner.prepare_request``.
    Only the advertised prefix is consumed by the model, but keeping the full
    payload identical makes the offline/live contract directly auditable.
    """

    plan = np.asarray(prior_plan, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[0] <= 0 or plan.shape[1] <= 0:
        raise ValueError(f"prior_plan must have shape [H,D], got {plan.shape}.")
    if not np.isfinite(plan).all():
        raise ValueError("prior_plan contains non-finite normalized actions.")
    if isinstance(prior_anchor_frame, (bool, np.bool_)) or not isinstance(
        prior_anchor_frame, (int, np.integer)
    ):
        raise ValueError("prior_anchor_frame must be an integer.")
    if isinstance(request_frame, (bool, np.bool_)) or not isinstance(
        request_frame, (int, np.integer)
    ):
        raise ValueError("request_frame must be an integer.")
    elapsed = int(request_frame) - int(prior_anchor_frame)
    if elapsed < 0:
        raise ValueError(
            f"request_frame={request_frame} precedes prior plan anchor {prior_anchor_frame}."
        )
    if elapsed >= plan.shape[0]:
        raise ValueError(
            f"request_frame elapsed {elapsed} rows from prior plan horizon {plan.shape[0]}; "
            "no unexecuted action remains."
        )
    remaining = plan.shape[0] - elapsed
    aligned = np.empty_like(plan, dtype=np.float32)
    aligned[:remaining] = plan[elapsed:]
    aligned[remaining:] = plan[-1]
    return np.ascontiguousarray(aligned, dtype=np.float32)


@dataclass(frozen=True)
class ExecutedAction:
    frame_index: int
    plan_anchor_frame: int
    plan_action_index: int
    source: str
    normalized_action: np.ndarray


@dataclass(frozen=True)
class PrefixConsistency:
    request_frame: int
    prior_plan_anchor_frame: int
    prior_plan_elapsed_frames: int
    prefix_len: int
    copy_mae_normalized: float
    copy_max_abs_normalized: float
    activation_next_action_mae_normalized: float | None
    activation_next_arm_mae_normalized: float | None


@dataclass(frozen=True)
class RtcReplayTrace:
    executed: tuple[ExecutedAction, ...]
    prefixes: tuple[PrefixConsistency, ...]
    conditioned_query_count: int


ConditionedPredictor = Callable[[int, np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class ActivationBoundary:
    request_frame: int
    activation_frame: int
    prior_plan_anchor_frame: int
    prior_plan_elapsed_frames: int
    prior_plan_action_index: int
    replacement_plan_anchor_frame: int
    replacement_plan_action_index: int
    action_mae_normalized: float
    arm_mae_normalized: float


@dataclass(frozen=True)
class DelayedUnconditionedReplayTrace:
    executed: tuple[ExecutedAction, ...]
    activation_boundaries: tuple[ActivationBoundary, ...]
    ordinary_no_prefix_query_count: int


def simulate_rtc_overlap(
    *,
    bootstrap_plan: Any,
    start_frame: int,
    num_frames: int,
    prefix_len: int,
    action_dim: int,
    arm_dimensions: Sequence[int],
    predict_conditioned: ConditionedPredictor,
    prefix_copy_tolerance: float = DEFAULT_PREFIX_COPY_TOLERANCE,
) -> RtcReplayTrace:
    """Simulate fixed-delay RTC execution on an exact recorded-frame clock.

    The callback receives ``(request_frame, aligned_prev_actions, prefix_len)``.
    ``aligned_prev_actions`` is the full normalized ``[H,D]`` client payload;
    its first ``prefix_len`` rows are the committed prefix.  The callback must
    return a full normalized ``[H,D]`` conditioned plan.
    """

    length = _plain_positive_int(num_frames, name="num_frames")
    delay = _plain_positive_int(prefix_len, name="prefix_len")
    if length <= 2 * delay:
        raise ValueError(
            "num_frames must exceed twice prefix_len so a conditioned suffix is executed."
        )
    if length % delay != 0:
        raise ValueError(
            "num_frames must be divisible by prefix_len; partial committed prefixes are invalid."
        )
    tolerance = float(prefix_copy_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("prefix_copy_tolerance must be finite and non-negative.")
    active = np.asarray(bootstrap_plan, dtype=np.float32)
    if active.ndim != 2 or active.shape[0] <= 0 or active.shape[1] != int(action_dim):
        raise ValueError(
            f"bootstrap_plan must have shape [H,{action_dim}], got {active.shape}."
        )
    if not np.isfinite(active).all():
        raise ValueError("bootstrap_plan contains non-finite normalized actions.")
    horizon = int(active.shape[0])
    if 2 * delay >= horizon:
        raise ValueError(
            f"prefix_len={delay} cannot score an activation boundary inside horizon {horizon}."
        )
    arm_indices = np.asarray(tuple(int(value) for value in arm_dimensions), dtype=np.int64)
    if arm_indices.size == 0 or np.any(arm_indices < 0) or np.any(arm_indices >= int(action_dim)):
        raise ValueError("arm_dimensions are empty or outside action_dim.")

    active_anchor = int(start_frame)
    active_source = "fresh_bootstrap"
    stop_frame = int(start_frame) + length
    cursor = int(start_frame)
    executed: list[ExecutedAction] = []
    prefix_records: list[PrefixConsistency] = []

    # The client has no prior plan for its first request, so bootstrap inference
    # is synchronous.  Only after these rows have actually executed can the
    # first aligned RTC request start at active_plan[delay:2*delay].
    bootstrap_slice = align_prior_normalized_prefix(
        active,
        prior_anchor_frame=active_anchor,
        request_frame=cursor,
        prefix_len=delay,
    )
    for offset, action in enumerate(bootstrap_slice):
        executed.append(
            ExecutedAction(
                frame_index=cursor + offset,
                plan_anchor_frame=active_anchor,
                plan_action_index=offset,
                source=active_source,
                normalized_action=np.ascontiguousarray(action, dtype=np.float32),
            )
        )
    cursor += delay

    while cursor < stop_frame:
        interval = stop_frame - cursor
        if interval < delay:
            raise RuntimeError(
                "RTC replay reached a partial committed prefix despite divisibility validation."
            )
        interval = delay
        execution_slice = align_prior_normalized_prefix(
            active,
            prior_anchor_frame=active_anchor,
            request_frame=cursor,
            prefix_len=interval,
        )
        first_action_index = cursor - active_anchor

        aligned_prev_actions = align_prior_normalized_plan(
            active,
            prior_anchor_frame=active_anchor,
            request_frame=cursor,
        )
        supplied_prefix = aligned_prev_actions[:delay]
        if not np.array_equal(supplied_prefix, execution_slice):
            raise RuntimeError(
                "Full RTC prev_actions alignment disagrees with the committed execution slice."
            )
        predicted = np.asarray(
            predict_conditioned(cursor, aligned_prev_actions.copy(), delay),
            dtype=np.float32,
        )
        replacement = validate_normalized_plan(
            predicted,
            horizon=horizon,
            action_dim=int(action_dim),
            name=f"RTC plan at frame {cursor}",
        )
        prefix_delta = np.abs(replacement[:delay] - supplied_prefix)
        copy_mae = float(prefix_delta.mean())
        copy_max = float(prefix_delta.max())
        if copy_max > tolerance:
            raise ValueError(
                f"RTC model failed to preserve the supplied normalized prefix at frame "
                f"{cursor}: max_abs={copy_max:.9g} > tolerance={tolerance:.9g}."
            )

        old_next_index = cursor - active_anchor + delay
        next_mae: float | None = None
        next_arm_mae: float | None = None
        if old_next_index < horizon and delay < horizon:
            next_delta = np.abs(replacement[delay] - active[old_next_index])
            next_mae = float(next_delta.mean())
            next_arm_mae = float(next_delta[arm_indices].mean())
        prefix_records.append(
            PrefixConsistency(
                request_frame=cursor,
                prior_plan_anchor_frame=active_anchor,
                prior_plan_elapsed_frames=cursor - active_anchor,
                prefix_len=delay,
                copy_mae_normalized=copy_mae,
                copy_max_abs_normalized=copy_max,
                activation_next_action_mae_normalized=next_mae,
                activation_next_arm_mae_normalized=next_arm_mae,
            )
        )

        for offset, action in enumerate(execution_slice):
            executed.append(
                ExecutedAction(
                    frame_index=cursor + offset,
                    plan_anchor_frame=active_anchor,
                    plan_action_index=first_action_index + offset,
                    source=active_source,
                    normalized_action=np.ascontiguousarray(action, dtype=np.float32),
                )
            )

        cursor += interval
        # Its prefix covered [request, request+delay); the first newly
        # generated action that can execute is replacement[delay].
        active = replacement
        active_anchor = cursor - interval
        active_source = "rtc_conditioned_suffix"

    if len(executed) != length:
        raise RuntimeError(
            f"RTC replay generated {len(executed)} actions for {length} recorded frames."
        )
    expected_frames = tuple(range(int(start_frame), stop_frame))
    actual_frames = tuple(record.frame_index for record in executed)
    if actual_frames != expected_frames:
        raise RuntimeError(
            "RTC replay did not produce exactly one executed action per contiguous frame."
        )
    return RtcReplayTrace(
        executed=tuple(executed),
        prefixes=tuple(prefix_records),
        conditioned_query_count=len(prefix_records),
    )


def simulate_delayed_unconditioned_async(
    *,
    fresh_plans: Mapping[int, np.ndarray],
    start_frame: int,
    num_frames: int,
    delay_frames: int,
    action_dim: int,
    arm_dimensions: Sequence[int],
) -> DelayedUnconditionedReplayTrace:
    """Replay asynchronous delay without RTC prefix conditioning.

    A no-prefix bootstrap plan executes synchronously for ``delay_frames``.
    At every subsequent request frame, an ordinary no-prefix fresh plan for
    that recorded observation is selected from ``fresh_plans`` while the old
    active plan executes its next committed delay-length slice.  When the
    request completes, the replacement is activated at action index ``delay``
    so its action time remains aligned with the now-stale request observation.

    The in-memory cache is the same deterministic, unclipped fresh-plan cache
    used by synchronous h0 and open-loop evaluation, so this baseline adds no
    model calls and changes only the asynchronous execution schedule.
    """

    length = _plain_positive_int(num_frames, name="num_frames")
    delay = _plain_positive_int(delay_frames, name="delay_frames")
    if length <= 2 * delay:
        raise ValueError(
            "num_frames must exceed twice delay_frames so a delayed replacement executes."
        )
    if length % delay != 0:
        raise ValueError(
            "num_frames must be divisible by delay_frames; partial committed slices are invalid."
        )
    arm_indices = np.asarray(tuple(int(value) for value in arm_dimensions), dtype=np.int64)
    if arm_indices.size == 0 or np.any(arm_indices < 0) or np.any(
        arm_indices >= int(action_dim)
    ):
        raise ValueError("arm_dimensions are empty or outside action_dim.")
    if int(start_frame) not in fresh_plans:
        raise ValueError(
            f"Missing cached fresh bootstrap plan for frame {int(start_frame)}."
        )
    active = np.asarray(fresh_plans[int(start_frame)], dtype=np.float32)
    if active.ndim != 2 or active.shape[0] <= 0:
        raise ValueError(
            f"Cached fresh bootstrap plan must have shape [H,D], got {active.shape}."
        )
    horizon = int(active.shape[0])
    active = validate_normalized_plan(
        active,
        horizon=horizon,
        action_dim=int(action_dim),
        name=f"cached fresh plan at frame {int(start_frame)}",
    )
    if 2 * delay >= horizon:
        raise ValueError(
            f"delay_frames={delay} cannot score an activation boundary inside horizon {horizon}."
        )

    start = int(start_frame)
    stop = start + length
    active_anchor = start
    active_source = "fresh_bootstrap"
    executed: list[ExecutedAction] = []
    boundaries: list[ActivationBoundary] = []

    bootstrap_slice = align_prior_normalized_prefix(
        active,
        prior_anchor_frame=active_anchor,
        request_frame=start,
        prefix_len=delay,
    )
    for offset, action in enumerate(bootstrap_slice):
        executed.append(
            ExecutedAction(
                frame_index=start + offset,
                plan_anchor_frame=active_anchor,
                plan_action_index=offset,
                source=active_source,
                normalized_action=np.ascontiguousarray(action, dtype=np.float32),
            )
        )

    cursor = start + delay
    while cursor < stop:
        if cursor not in fresh_plans:
            raise ValueError(
                f"Missing cached ordinary no-prefix plan for async request frame {cursor}."
            )
        replacement = validate_normalized_plan(
            fresh_plans[cursor],
            horizon=horizon,
            action_dim=int(action_dim),
            name=f"cached fresh plan at async request frame {cursor}",
        )
        committed = align_prior_normalized_prefix(
            active,
            prior_anchor_frame=active_anchor,
            request_frame=cursor,
            prefix_len=delay,
        )
        first_action_index = cursor - active_anchor
        old_activation_index = first_action_index + delay
        if old_activation_index >= horizon:
            raise ValueError(
                f"Prior plan anchored at {active_anchor} has no action for delayed activation "
                f"frame {cursor + delay} (index {old_activation_index})."
            )
        replacement_activation_index = delay
        activation_delta = np.abs(
            replacement[replacement_activation_index] - active[old_activation_index]
        )
        boundaries.append(
            ActivationBoundary(
                request_frame=cursor,
                activation_frame=cursor + delay,
                prior_plan_anchor_frame=active_anchor,
                prior_plan_elapsed_frames=first_action_index,
                prior_plan_action_index=old_activation_index,
                replacement_plan_anchor_frame=cursor,
                replacement_plan_action_index=replacement_activation_index,
                action_mae_normalized=float(activation_delta.mean()),
                arm_mae_normalized=float(activation_delta[arm_indices].mean()),
            )
        )
        for offset, action in enumerate(committed):
            executed.append(
                ExecutedAction(
                    frame_index=cursor + offset,
                    plan_anchor_frame=active_anchor,
                    plan_action_index=first_action_index + offset,
                    source=active_source,
                    normalized_action=np.ascontiguousarray(action, dtype=np.float32),
                )
            )

        active = replacement
        active_anchor = cursor
        active_source = "delayed_unconditioned_suffix"
        cursor += delay

    expected_frames = tuple(range(start, stop))
    actual_frames = tuple(record.frame_index for record in executed)
    if actual_frames != expected_frames:
        raise RuntimeError(
            "Delayed-unconditioned replay did not produce exactly one action per contiguous frame."
        )
    expected_query_frames = tuple(range(start + delay, stop, delay))
    actual_query_frames = tuple(item.request_frame for item in boundaries)
    if actual_query_frames != expected_query_frames:
        raise RuntimeError(
            "Delayed-unconditioned async request frames lost fixed-delay alignment."
        )
    return DelayedUnconditionedReplayTrace(
        executed=tuple(executed),
        activation_boundaries=tuple(boundaries),
        ordinary_no_prefix_query_count=len(boundaries),
    )


def build_fresh_h0_trace(
    fresh_plans: Mapping[int, np.ndarray], *, frame_indices: Sequence[int]
) -> tuple[ExecutedAction, ...]:
    frames = validate_contiguous_frame_indices(frame_indices)
    records: list[ExecutedAction] = []
    for frame in frames:
        if frame not in fresh_plans:
            raise ValueError(f"Missing fresh plan for recorded frame {frame}.")
        plan = np.asarray(fresh_plans[frame], dtype=np.float32)
        if plan.ndim != 2 or plan.shape[0] <= 0 or not np.isfinite(plan).all():
            raise ValueError(f"Fresh plan at frame {frame} is invalid: shape={plan.shape}.")
        records.append(
            ExecutedAction(
                frame_index=frame,
                plan_anchor_frame=frame,
                plan_action_index=0,
                source="fresh_synchronous_h0",
                normalized_action=np.ascontiguousarray(plan[0], dtype=np.float32),
            )
        )
    return tuple(records)


def build_open_loop_trace(
    fresh_plans: Mapping[int, np.ndarray],
    *,
    frame_indices: Sequence[int],
    chunk_size: int,
) -> tuple[ExecutedAction, ...]:
    frames = validate_contiguous_frame_indices(frame_indices)
    chunk = _plain_positive_int(chunk_size, name="chunk_size")
    records: list[ExecutedAction] = []
    for chunk_start_position in range(0, len(frames), chunk):
        anchor = frames[chunk_start_position]
        if anchor not in fresh_plans:
            raise ValueError(f"Missing fresh plan for open-loop anchor frame {anchor}.")
        plan = np.asarray(fresh_plans[anchor], dtype=np.float32)
        if plan.ndim != 2 or plan.shape[0] < chunk or not np.isfinite(plan).all():
            raise ValueError(
                f"Open-loop plan at frame {anchor} cannot cover chunk {chunk}: shape={plan.shape}."
            )
        segment = frames[chunk_start_position : chunk_start_position + chunk]
        for offset, frame in enumerate(segment):
            if frame != anchor + offset:
                raise RuntimeError("Open-loop plan/frame alignment lost contiguity.")
            records.append(
                ExecutedAction(
                    frame_index=frame,
                    plan_anchor_frame=anchor,
                    plan_action_index=offset,
                    source=f"open_loop_chunk_{chunk}",
                    normalized_action=np.ascontiguousarray(plan[offset], dtype=np.float32),
                )
            )
    if len(records) != len(frames):
        raise RuntimeError("Open-loop replay did not cover every requested frame.")
    return tuple(records)


class ExecutionMetrics:
    """Raw-space execution metrics with close as the positive gripper class."""

    def __init__(
        self,
        *,
        arm_dimensions: Sequence[int],
        gripper_dimensions: Sequence[int],
        gripper_thresholds: np.ndarray,
    ) -> None:
        self.arm_dimensions = np.asarray(tuple(arm_dimensions), dtype=np.int64)
        self.gripper_dimensions = np.asarray(tuple(gripper_dimensions), dtype=np.int64)
        self.thresholds = np.asarray(gripper_thresholds, dtype=np.float32)
        if self.arm_dimensions.size == 0 or self.gripper_dimensions.size == 0:
            raise ValueError("Arm and gripper dimension sets must both be non-empty.")
        self.frames = 0
        self.arm_abs_sum = 0.0
        self.arm_elements = 0
        self.close_tp = 0
        self.close_fp = 0
        self.close_fn = 0
        self.close_tn = 0

    def update(self, prediction_raw: Any, target_raw: Any) -> dict[str, Any]:
        prediction = np.asarray(prediction_raw, dtype=np.float32).reshape(-1)
        target = np.asarray(target_raw, dtype=np.float32).reshape(-1)
        if prediction.shape != target.shape or prediction.shape != self.thresholds.shape:
            raise ValueError(
                f"Execution prediction/target/threshold shapes differ: "
                f"{prediction.shape}/{target.shape}/{self.thresholds.shape}."
            )
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Execution metric received non-finite raw actions.")
        arm_error = np.abs(
            prediction[self.arm_dimensions] - target[self.arm_dimensions]
        )
        self.arm_abs_sum += float(arm_error.sum())
        self.arm_elements += int(arm_error.size)

        grippers = self.gripper_dimensions
        predicted_close = prediction[grippers] < self.thresholds[grippers]
        target_close = target[grippers] < self.thresholds[grippers]
        self.close_tp += int(np.sum(predicted_close & target_close))
        self.close_fp += int(np.sum(predicted_close & ~target_close))
        self.close_fn += int(np.sum(~predicted_close & target_close))
        self.close_tn += int(np.sum(~predicted_close & ~target_close))
        self.frames += 1
        return {
            "arm_mae_raw": float(arm_error.mean()),
            "target_close_count": int(target_close.sum()),
            "predicted_close_count": int(predicted_close.sum()),
        }

    def finalize(self) -> dict[str, Any]:
        close_targets = self.close_tp + self.close_fn
        close_predictions = self.close_tp + self.close_fp
        total_grippers = self.close_tp + self.close_fp + self.close_fn + self.close_tn
        return {
            "executed_frames": self.frames,
            "arm_elements": self.arm_elements,
            "arm_mae_raw": (
                self.arm_abs_sum / self.arm_elements if self.arm_elements else None
            ),
            "gripper_close_positive_rule": "raw_action < midpoint(min,max)",
            "gripper_close_recall": (
                self.close_tp / close_targets if close_targets else None
            ),
            "gripper_close_precision": (
                self.close_tp / close_predictions if close_predictions else None
            ),
            "gripper_close_accuracy": (
                (self.close_tp + self.close_tn) / total_grippers
                if total_grippers
                else None
            ),
            "gripper_close_confusion": {
                "true_positive_close": self.close_tp,
                "false_positive_close": self.close_fp,
                "false_negative_close": self.close_fn,
                "true_negative_open": self.close_tn,
                "target_close_count": close_targets,
                "predicted_close_count": close_predictions,
            },
        }


@dataclass(frozen=True)
class ReplayObservation:
    frame_index: int
    qwen_frames: np.ndarray
    state_normalized: np.ndarray
    target_action_normalized: np.ndarray
    target_action_raw_window: np.ndarray
    target_action_valid_mask: np.ndarray
    target_action_raw_h0: np.ndarray
    dataset_instruction: str
    state_oob_elements_before_optional_clip: int


def _load_replay_observation(
    *,
    mixture_dataset: Any,
    single_dataset: Any,
    episode_id: int,
    frame_index: int,
    prompt_seed: int,
    video_target_shift_steps: int,
    state_dim: int,
    horizon: int,
    action_dim: int,
    state_stats: Mapping[str, Any],
    state_mode: str,
    clip_state: bool,
    metadata: Mapping[str, Any],
    required_action_dimensions: Sequence[int],
) -> ReplayObservation:
    sample, dataset_instruction = load_targeted_training_sample(
        mixture_dataset,
        single_dataset,
        episode_id=int(episode_id),
        frame_index=int(frame_index),
        prompt_seed=int(prompt_seed),
    )
    validate_returned_frame_index(sample, requested=int(frame_index))
    target_normalized = np.asarray(sample["action"], dtype=np.float32)
    if target_normalized.shape != (int(horizon), int(action_dim)):
        raise ValueError(
            f"Target action at frame {frame_index} has shape {target_normalized.shape}, "
            f"expected {(horizon, action_dim)}."
        )
    valid_mask = build_valid_action_mask(sample, target_normalized.shape)
    required = np.asarray(tuple(required_action_dimensions), dtype=np.int64)
    if required.size == 0 or not bool(np.all(valid_mask[0, required])):
        raise ValueError(
            f"Recorded frame {frame_index} has an invalid h0 target in a scored arm/gripper "
            "dimension; refusing to create a selectively filtered contiguous replay."
        )

    qwen_frames = extract_training_aligned_qwen_frames(
        sample,
        video_target_shift_steps=int(video_target_shift_steps),
    )
    raw_state_window = extract_authoritative_raw_modality_window(
        single_dataset,
        modality="state",
        frame_index=int(frame_index),
    )
    state_raw = np.asarray(raw_state_window[0], dtype=np.float32)
    if state_raw.shape != (int(state_dim),) or not np.isfinite(state_raw).all():
        raise ValueError(
            f"Authoritative state at frame {frame_index} is invalid: shape={state_raw.shape}."
        )
    normalized_unclipped = normalize_values(
        state_raw[None, :], state_stats, mode=state_mode
    )
    oob_count = int(np.sum(np.abs(normalized_unclipped) > 1.0))
    state_normalized = (
        np.clip(normalized_unclipped, -1.0, 1.0)
        if clip_state
        else normalized_unclipped
    )
    state_normalized = np.ascontiguousarray(state_normalized, dtype=np.float32)
    validate_training_aligned_input_contract(
        qwen_frames=qwen_frames,
        state=state_normalized,
        metadata=metadata,
    )

    target_raw_window = extract_authoritative_raw_modality_window(
        single_dataset,
        modality="action",
        frame_index=int(frame_index),
    )
    if target_raw_window.shape != (int(horizon), int(action_dim)):
        raise ValueError(
            f"Authoritative action window at frame {frame_index} has shape "
            f"{target_raw_window.shape}, expected {(horizon, action_dim)}."
        )
    target_h0 = np.ascontiguousarray(target_raw_window[0], dtype=np.float32)
    if not np.isfinite(target_h0).all():
        raise ValueError(f"Authoritative h0 action at frame {frame_index} is non-finite.")
    return ReplayObservation(
        frame_index=int(frame_index),
        qwen_frames=qwen_frames,
        state_normalized=state_normalized,
        target_action_normalized=np.ascontiguousarray(
            target_normalized, dtype=np.float32
        ),
        target_action_raw_window=np.ascontiguousarray(
            target_raw_window, dtype=np.float32
        ),
        target_action_valid_mask=np.ascontiguousarray(valid_mask, dtype=bool),
        target_action_raw_h0=target_h0,
        dataset_instruction=str(dataset_instruction),
        state_oob_elements_before_optional_clip=oob_count,
    )


def _predict_local_plan(
    predictor: LocalPredictor,
    *,
    qwen_frames: np.ndarray,
    instruction: str,
    state_normalized: np.ndarray,
    seed: int,
    horizon: int,
    action_dim: int,
    prev_actions: np.ndarray | None = None,
    prefix_len: int = 0,
) -> tuple[np.ndarray, float]:
    """Run the same local model entry point as the policy server, batch size one."""

    torch = predictor.torch
    torch.manual_seed(int(seed) % (2**63 - 1))
    if predictor.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) % (2**63 - 1))
    kwargs: dict[str, Any] = {
        "qwen_frames": [np.asarray(qwen_frames, dtype=np.uint8)],
        "instructions": [str(instruction)],
        "state": np.asarray(state_normalized, dtype=np.float32)[None, ...],
    }
    if prev_actions is not None:
        prefix = _plain_positive_int(prefix_len, name="prefix_len")
        previous = np.asarray(prev_actions, dtype=np.float32)
        if previous.shape != (int(horizon), int(action_dim)):
            raise ValueError(
                f"prev_actions must be a full normalized plan {(horizon, action_dim)}, "
                f"got {previous.shape}."
            )
        if not np.isfinite(previous).all():
            raise ValueError("prev_actions contains non-finite values.")
        kwargs.update(
            {
                "prev_actions": previous[None, ...],
                "prefix_len": prefix,
                "rtc_config": {"enabled": True, "method": "prefix"},
            }
        )
    elif int(prefix_len) != 0:
        raise ValueError("prefix_len must be zero when prev_actions is absent.")

    started = time.perf_counter()
    with torch.inference_mode():
        output = predictor.model.predict_action(**kwargs)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(output, Mapping) or "normalized_actions" not in output:
        raise ValueError("Policy output is missing normalized_actions.")
    actions = np.asarray(output["normalized_actions"], dtype=np.float32)
    if actions.shape != (1, int(horizon), int(action_dim)):
        raise ValueError(
            f"Policy normalized_actions shape {actions.shape} does not match "
            f"{(1, horizon, action_dim)}."
        )
    plan = validate_normalized_plan(
        actions[0], horizon=int(horizon), action_dim=int(action_dim), name="policy plan"
    )
    return plan, float(latency_ms)


def _metrics_for_trace(
    trace: Sequence[ExecutedAction],
    *,
    target_raw_by_frame: Mapping[int, np.ndarray],
    action_stats: Mapping[str, Any],
    action_mode: str,
    arm_dimensions: Sequence[int],
    gripper_dimensions: Sequence[int],
    gripper_thresholds: np.ndarray,
    include_records: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accumulator = ExecutionMetrics(
        arm_dimensions=arm_dimensions,
        gripper_dimensions=gripper_dimensions,
        gripper_thresholds=gripper_thresholds,
    )
    records: list[dict[str, Any]] = []
    for execution in trace:
        target = target_raw_by_frame.get(execution.frame_index)
        if target is None:
            raise ValueError(f"Missing target action for frame {execution.frame_index}.")
        prediction_raw = realman_continuous_unnormalize(
            execution.normalized_action,
            action_stats,
            mode=action_mode,
        )
        diagnostic = accumulator.update(prediction_raw, target)
        if include_records:
            records.append(
                {
                    "frame_index": execution.frame_index,
                    "plan_anchor_frame": execution.plan_anchor_frame,
                    "plan_action_index": execution.plan_action_index,
                    "source": execution.source,
                    **diagnostic,
                }
            )
    return accumulator.finalize(), records


def _prefix_summary(prefixes: Sequence[PrefixConsistency]) -> dict[str, Any]:
    if not prefixes:
        raise ValueError("RTC replay produced no conditioned prefix queries.")
    copy_mae = np.asarray([item.copy_mae_normalized for item in prefixes], dtype=np.float64)
    copy_max = np.asarray(
        [item.copy_max_abs_normalized for item in prefixes], dtype=np.float64
    )
    next_all = np.asarray(
        [
            item.activation_next_action_mae_normalized
            for item in prefixes
            if item.activation_next_action_mae_normalized is not None
        ],
        dtype=np.float64,
    )
    next_arm = np.asarray(
        [
            item.activation_next_arm_mae_normalized
            for item in prefixes
            if item.activation_next_arm_mae_normalized is not None
        ],
        dtype=np.float64,
    )
    return {
        "conditioned_queries": len(prefixes),
        "copied_prefix_normalized_mae": float(copy_mae.mean()),
        "copied_prefix_normalized_max_abs": float(copy_max.max()),
        "activation_next_action_normalized_mae": (
            float(next_all.mean()) if next_all.size else None
        ),
        "activation_next_arm_normalized_mae": (
            float(next_arm.mean()) if next_arm.size else None
        ),
        "alignment_records": [
            {
                "request_frame": item.request_frame,
                "prior_plan_anchor_frame": item.prior_plan_anchor_frame,
                "prior_plan_elapsed_frames": item.prior_plan_elapsed_frames,
                "prefix_len": item.prefix_len,
                "copied_prior_plan_slice": [
                    item.prior_plan_elapsed_frames,
                    item.prior_plan_elapsed_frames + item.prefix_len,
                ],
                "copy_mae_normalized": item.copy_mae_normalized,
                "copy_max_abs_normalized": item.copy_max_abs_normalized,
                "activation_next_action_mae_normalized": (
                    item.activation_next_action_mae_normalized
                ),
                "activation_next_arm_mae_normalized": (
                    item.activation_next_arm_mae_normalized
                ),
            }
            for item in prefixes
        ],
    }


def _activation_boundary_summary(
    boundaries: Sequence[ActivationBoundary],
) -> dict[str, Any]:
    if not boundaries:
        raise ValueError("Delayed-unconditioned replay produced no activation boundaries.")
    action_mae = np.asarray(
        [item.action_mae_normalized for item in boundaries], dtype=np.float64
    )
    arm_mae = np.asarray(
        [item.arm_mae_normalized for item in boundaries], dtype=np.float64
    )
    return {
        "activation_boundaries": len(boundaries),
        "activation_next_action_normalized_mae": float(action_mae.mean()),
        "activation_next_arm_normalized_mae": float(arm_mae.mean()),
        "records": [
            {
                "request_frame": item.request_frame,
                "activation_frame": item.activation_frame,
                "prior_plan_anchor_frame": item.prior_plan_anchor_frame,
                "prior_plan_elapsed_frames": item.prior_plan_elapsed_frames,
                "prior_plan_action_index": item.prior_plan_action_index,
                "replacement_plan_anchor_frame": item.replacement_plan_anchor_frame,
                "replacement_plan_action_index": item.replacement_plan_action_index,
                "action_mae_normalized": item.action_mae_normalized,
                "arm_mae_normalized": item.arm_mae_normalized,
            }
            for item in boundaries
        ],
    }


def _frame_sequence_summary(frame_indices: Sequence[int]) -> dict[str, Any]:
    frames = validate_contiguous_frame_indices(frame_indices)
    encoded = json.dumps(list(frames), separators=(",", ":")).encode("utf-8")
    return {
        "count": len(frames),
        "start_frame_inclusive": frames[0],
        "stop_frame_exclusive": frames[-1] + 1,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def build_delayed_unconditioned_vs_rtc_comparison(
    *,
    delayed_trace: DelayedUnconditionedReplayTrace,
    rtc_trace: RtcReplayTrace,
    delayed_execution_metrics: Mapping[str, Any],
    rtc_execution_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a paired comparison, refusing unequal action or request frames."""

    delayed_executions = tuple(
        item
        for item in delayed_trace.executed
        if item.source == "delayed_unconditioned_suffix"
    )
    rtc_executions = tuple(
        item for item in rtc_trace.executed if item.source == "rtc_conditioned_suffix"
    )
    delayed_frames = tuple(item.frame_index for item in delayed_executions)
    rtc_frames = tuple(item.frame_index for item in rtc_executions)
    if delayed_frames != rtc_frames:
        raise ValueError(
            "Delayed-unconditioned and RTC replacement actions are not evaluated on "
            f"identical frames: delayed={delayed_frames[:10]} rtc={rtc_frames[:10]}."
        )
    if not delayed_frames:
        raise ValueError("Paired delayed-unconditioned/RTC execution frame set is empty.")

    delayed_boundaries = {
        item.request_frame: item for item in delayed_trace.activation_boundaries
    }
    rtc_boundaries = {item.request_frame: item for item in rtc_trace.prefixes}
    if tuple(delayed_boundaries) != tuple(rtc_boundaries):
        raise ValueError(
            "Delayed-unconditioned and RTC activation requests are not on identical frames: "
            f"delayed={tuple(delayed_boundaries)} rtc={tuple(rtc_boundaries)}."
        )
    paired_boundaries: list[dict[str, Any]] = []
    for request_frame, delayed in delayed_boundaries.items():
        rtc = rtc_boundaries[request_frame]
        if rtc.activation_next_action_mae_normalized is None or (
            rtc.activation_next_arm_mae_normalized is None
        ):
            raise ValueError(
                f"RTC activation discontinuity is unavailable at request frame {request_frame}."
            )
        if delayed.activation_frame != request_frame + rtc.prefix_len:
            raise ValueError(
                f"Activation frame mismatch at request {request_frame}: delayed="
                f"{delayed.activation_frame}, RTC={request_frame + rtc.prefix_len}."
            )
        paired_boundaries.append(
            {
                "request_frame": request_frame,
                "activation_frame": delayed.activation_frame,
                "delayed_unconditioned_action_mae_normalized": (
                    delayed.action_mae_normalized
                ),
                "rtc_action_mae_normalized": (
                    rtc.activation_next_action_mae_normalized
                ),
                "delayed_minus_rtc_action_mae_normalized": (
                    delayed.action_mae_normalized
                    - rtc.activation_next_action_mae_normalized
                ),
                "delayed_unconditioned_arm_mae_normalized": delayed.arm_mae_normalized,
                "rtc_arm_mae_normalized": rtc.activation_next_arm_mae_normalized,
                "delayed_minus_rtc_arm_mae_normalized": (
                    delayed.arm_mae_normalized - rtc.activation_next_arm_mae_normalized
                ),
            }
        )

    def _metric(name: str, metrics: Mapping[str, Any]) -> float | None:
        value = metrics.get(name)
        return None if value is None else float(value)

    delayed_arm = _metric("arm_mae_raw", delayed_execution_metrics)
    rtc_arm = _metric("arm_mae_raw", rtc_execution_metrics)
    if delayed_arm is None or rtc_arm is None:
        raise ValueError("Paired execution arm MAE is missing.")
    delayed_boundary_arm = float(
        np.mean([item.arm_mae_normalized for item in delayed_trace.activation_boundaries])
    )
    rtc_boundary_arm = float(
        np.mean(
            [
                float(item.activation_next_arm_mae_normalized)
                for item in rtc_trace.prefixes
                if item.activation_next_arm_mae_normalized is not None
            ]
        )
    )
    execution_metric_comparison: dict[str, Any] = {
        "delayed_unconditioned_arm_mae_raw": delayed_arm,
        "rtc_arm_mae_raw": rtc_arm,
        "delayed_minus_rtc_arm_mae_raw": delayed_arm - rtc_arm,
        "delayed_div_rtc_arm_mae_raw": delayed_arm / rtc_arm if rtc_arm else None,
    }
    for metric_name in (
        "gripper_close_recall",
        "gripper_close_precision",
        "gripper_close_accuracy",
    ):
        delayed_value = _metric(metric_name, delayed_execution_metrics)
        rtc_value = _metric(metric_name, rtc_execution_metrics)
        execution_metric_comparison[metric_name] = {
            "delayed_unconditioned": delayed_value,
            "rtc": rtc_value,
            "delayed_minus_rtc": (
                delayed_value - rtc_value
                if delayed_value is not None and rtc_value is not None
                else None
            ),
        }
    return {
        "identical_replacement_execution_frames": True,
        "replacement_execution_frames": _frame_sequence_summary(delayed_frames),
        "execution_metrics": execution_metric_comparison,
        "identical_activation_request_frames": True,
        "activation_boundary_discontinuity": {
            "delayed_unconditioned_arm_normalized_mae": delayed_boundary_arm,
            "rtc_arm_normalized_mae": rtc_boundary_arm,
            "delayed_minus_rtc_arm_normalized_mae": (
                delayed_boundary_arm - rtc_boundary_arm
            ),
            "delayed_div_rtc_arm_normalized_mae": (
                delayed_boundary_arm / rtc_boundary_arm if rtc_boundary_arm else None
            ),
            "paired_records": paired_boundaries,
        },
    }


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _episode_ref_for_request(
    refs: Sequence[Any], *, dataset_name: str | None, episode_id: int
) -> Any:
    matches = [
        ref
        for ref in refs
        if ref.episode_id == int(episode_id)
        and (dataset_name is None or ref.dataset_name == str(dataset_name))
    ]
    if not matches:
        scope = f"dataset {dataset_name!r}" if dataset_name else "configured datasets"
        raise ValueError(f"Episode {episode_id} was not found in {scope}.")
    if len(matches) != 1:
        identities = [(ref.dataset_name, ref.episode_id) for ref in matches]
        raise ValueError(
            f"Episode {episode_id} is ambiguous across datasets {identities}; pass --dataset-name."
        )
    return matches[0]


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    """Run one checkpoint against one exact contiguous recorded episode range."""

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    instruction = str(args.instruction).strip()
    if not instruction:
        raise ValueError("--instruction must contain the exact non-empty deployment prompt.")
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else _resolve_run_file(checkpoint_path, "config.yaml")
    )
    cfg = OmegaConf.load(config_path)
    if args.dataset_root:
        cfg.datasets.vla_data.data_root_dir = str(
            Path(args.dataset_root).expanduser().resolve()
        )

    action_horizon = int(cfg.framework.action_model.action_horizon)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="val",
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=int(args.seed),
        action_horizon=action_horizon,
        video_horizon=int(cfg.framework.vj2_model.num_frames),
        video_frame_stride=int(cfg.datasets.vla_data.get("video_frame_stride", 1)),
    )
    refs = enumerate_episodes(dataset.datasets, seed=int(args.seed))
    episode_ref = _episode_ref_for_request(
        refs,
        dataset_name=args.dataset_name,
        episode_id=int(args.episode_id),
    )
    frame_indices = validate_contiguous_frame_range(
        episode_length=int(episode_ref.length),
        start_frame=int(args.start_frame),
        num_frames=int(args.num_frames),
    )
    single_dataset = dataset.datasets[episode_ref.dataset_index]
    validate_unit_step_action_offsets(single_dataset, horizon=action_horizon)

    predictor = LocalPredictor(
        checkpoint_path=checkpoint_path,
        device_name=str(args.device),
    )
    try:
        metadata = predictor.metadata
        action_stats, state_stats, action_mode, state_mode, norm_key = _metadata_stats(
            metadata
        )
        action_dim = int(metadata["action_dim"])
        state_dim = int(metadata["state_dim"])
        horizon = int(metadata["action_horizon"])
        if (action_dim, state_dim, horizon) != (
            int(cfg.framework.action_model.action_dim),
            int(cfg.framework.action_model.state_dim),
            action_horizon,
        ):
            raise ValueError(
                "Checkpoint/config action contract differs: "
                f"checkpoint={(action_dim, state_dim, horizon)} "
                f"config={(int(cfg.framework.action_model.action_dim), int(cfg.framework.action_model.state_dim), action_horizon)}."
            )
        if action_dim not in {18, 19, 22} or state_dim != 19:
            raise ValueError(
                f"Sequential Realman replay requires 18/19/22D actions and 19D state, "
                f"got {action_dim}/{state_dim}."
            )
        requested_prefixes = tuple(int(value) for value in args.rtc_prefix_lengths)
        rtc_contract = validate_rtc_inference_contract(
            metadata, requested_prefix_lengths=requested_prefixes
        )
        chunk_sizes, prefix_lengths = validate_replay_method_configuration(
            horizon=horizon,
            num_frames=len(frame_indices),
            open_loop_chunk_sizes=tuple(int(value) for value in args.open_loop_chunk_sizes),
            rtc_prefix_lengths=requested_prefixes,
            rtc_max_prefix_len=int(rtc_contract["max_prefix_len"]),
        )
        validate_dataset_camera_order(single_dataset, metadata)
        assert_checkpoint_dataset_stats_match(
            action_stats,
            merged_dataset_modality_stats(dataset, single_dataset, modality="action"),
            modality=f"{episode_ref.dataset_name} action",
        )
        assert_checkpoint_dataset_stats_match(
            state_stats,
            merged_dataset_modality_stats(dataset, single_dataset, modality="state"),
            modality=f"{episode_ref.dataset_name} state",
        )

        action_groups = realman_action_groups(action_dim)
        arm_dimensions = action_groups["arm"]
        gripper_dimensions = action_groups["gripper"]
        required_dimensions = tuple(sorted((*arm_dimensions, *gripper_dimensions)))
        action_min = np.asarray(action_stats["min"], dtype=np.float32)
        action_max = np.asarray(action_stats["max"], dtype=np.float32)
        gripper_thresholds = 0.5 * (action_min + action_max)
        clip_state = state_mode == "q99"
        video_target_shift_steps = int(
            cfg.datasets.vla_data.get("video_target_shift_steps", 0)
        )

        rtc_request_frames = {
            request_frame
            for prefix_len in prefix_lengths
            for request_frame in range(
                frame_indices[0] + prefix_len,
                frame_indices[-1] + 1,
                prefix_len,
            )
        }
        fresh_plans: dict[int, np.ndarray] = {}
        target_raw_by_frame: dict[int, np.ndarray] = {}
        rtc_observations: dict[int, ReplayObservation] = {}
        dataset_instructions: set[str] = set()
        state_oob_total = 0
        fresh_latency_ms: list[float] = []

        for frame_index in frame_indices:
            sample_seed = _stable_seed(
                args.seed,
                episode_ref.dataset_name,
                episode_ref.episode_id,
                frame_index,
            )
            observation = _load_replay_observation(
                mixture_dataset=dataset,
                single_dataset=single_dataset,
                episode_id=episode_ref.episode_id,
                frame_index=frame_index,
                prompt_seed=sample_seed,
                video_target_shift_steps=video_target_shift_steps,
                state_dim=state_dim,
                horizon=horizon,
                action_dim=action_dim,
                state_stats=state_stats,
                state_mode=state_mode,
                clip_state=clip_state,
                metadata=metadata,
                required_action_dimensions=required_dimensions,
            )
            plan, latency = _predict_local_plan(
                predictor,
                qwen_frames=observation.qwen_frames,
                instruction=instruction,
                state_normalized=observation.state_normalized,
                seed=fresh_plan_inference_seed(
                    base_seed=int(args.seed),
                    dataset_name=episode_ref.dataset_name,
                    episode_id=episode_ref.episode_id,
                    frame_index=frame_index,
                ),
                horizon=horizon,
                action_dim=action_dim,
            )
            # Preserve the exact model output.  The live RealMan rollout default
            # does not clamp normalized actions before its affine inverse or
            # before retaining a plan for delayed/RTC alignment.
            plan = np.ascontiguousarray(plan, dtype=np.float32)
            fresh_plans[frame_index] = plan
            target_raw_by_frame[frame_index] = observation.target_action_raw_h0
            dataset_instructions.add(observation.dataset_instruction)
            state_oob_total += observation.state_oob_elements_before_optional_clip
            fresh_latency_ms.append(latency)
            if frame_index in rtc_request_frames:
                rtc_observations[frame_index] = observation

        include_records = bool(args.include_frame_records)
        fresh_trace = build_fresh_h0_trace(
            fresh_plans, frame_indices=frame_indices
        )
        fresh_metrics, fresh_records = _metrics_for_trace(
            fresh_trace,
            target_raw_by_frame=target_raw_by_frame,
            action_stats=action_stats,
            action_mode=action_mode,
            arm_dimensions=arm_dimensions,
            gripper_dimensions=gripper_dimensions,
            gripper_thresholds=gripper_thresholds,
            include_records=include_records,
        )

        open_loop_reports: dict[str, Any] = {}
        for chunk_size in chunk_sizes:
            trace = build_open_loop_trace(
                fresh_plans,
                frame_indices=frame_indices,
                chunk_size=chunk_size,
            )
            metrics, records = _metrics_for_trace(
                trace,
                target_raw_by_frame=target_raw_by_frame,
                action_stats=action_stats,
                action_mode=action_mode,
                arm_dimensions=arm_dimensions,
                gripper_dimensions=gripper_dimensions,
                gripper_thresholds=gripper_thresholds,
                include_records=include_records,
            )
            open_loop_reports[str(chunk_size)] = {
                "chunk_size_frames": chunk_size,
                "fresh_plan_queries": (len(frame_indices) + chunk_size - 1) // chunk_size,
                "execution_metrics": metrics,
                "arm_mae_ratio_vs_fresh_sync_h0": (
                    metrics["arm_mae_raw"] / fresh_metrics["arm_mae_raw"]
                    if fresh_metrics["arm_mae_raw"]
                    else None
                ),
                "executions": records if include_records else None,
            }

        rtc_reports: dict[str, Any] = {}
        delayed_unconditioned_reports: dict[str, Any] = {}
        for prefix_len in prefix_lengths:
            rtc_latencies: list[float] = []

            def _conditioned(
                request_frame: int,
                aligned_prev_actions: np.ndarray,
                callback_prefix_len: int,
            ) -> np.ndarray:
                if callback_prefix_len != prefix_len:
                    raise RuntimeError("RTC callback prefix length changed unexpectedly.")
                observation = rtc_observations.get(request_frame)
                if observation is None:
                    raise ValueError(
                        f"No cached exact deployment observation for RTC request frame "
                        f"{request_frame}."
                    )
                previous = validate_normalized_plan(
                    aligned_prev_actions,
                    horizon=horizon,
                    action_dim=action_dim,
                    name=f"aligned RTC prev_actions at frame {request_frame}",
                )
                plan, latency = _predict_local_plan(
                    predictor,
                    qwen_frames=observation.qwen_frames,
                    instruction=instruction,
                    state_normalized=observation.state_normalized,
                    seed=fresh_plan_inference_seed(
                        base_seed=int(args.seed),
                        dataset_name=episode_ref.dataset_name,
                        episode_id=episode_ref.episode_id,
                        frame_index=request_frame,
                    ),
                    horizon=horizon,
                    action_dim=action_dim,
                    prev_actions=previous,
                    prefix_len=prefix_len,
                )
                rtc_latencies.append(latency)
                return np.ascontiguousarray(plan, dtype=np.float32)

            rtc_trace = simulate_rtc_overlap(
                bootstrap_plan=fresh_plans[frame_indices[0]],
                start_frame=frame_indices[0],
                num_frames=len(frame_indices),
                prefix_len=prefix_len,
                action_dim=action_dim,
                arm_dimensions=arm_dimensions,
                predict_conditioned=_conditioned,
                prefix_copy_tolerance=float(args.prefix_copy_tolerance),
            )
            all_metrics, all_records = _metrics_for_trace(
                rtc_trace.executed,
                target_raw_by_frame=target_raw_by_frame,
                action_stats=action_stats,
                action_mode=action_mode,
                arm_dimensions=arm_dimensions,
                gripper_dimensions=gripper_dimensions,
                gripper_thresholds=gripper_thresholds,
                include_records=include_records,
            )
            post_bootstrap = tuple(
                item
                for item in rtc_trace.executed
                if item.source == "rtc_conditioned_suffix"
            )
            post_metrics, post_records = _metrics_for_trace(
                post_bootstrap,
                target_raw_by_frame=target_raw_by_frame,
                action_stats=action_stats,
                action_mode=action_mode,
                arm_dimensions=arm_dimensions,
                gripper_dimensions=gripper_dimensions,
                gripper_thresholds=gripper_thresholds,
                include_records=include_records,
            )
            post_frames = tuple(item.frame_index for item in post_bootstrap)
            fresh_same_frames = tuple(
                item for item in fresh_trace if item.frame_index in set(post_frames)
            )
            fresh_post_metrics, _ = _metrics_for_trace(
                fresh_same_frames,
                target_raw_by_frame=target_raw_by_frame,
                action_stats=action_stats,
                action_mode=action_mode,
                arm_dimensions=arm_dimensions,
                gripper_dimensions=gripper_dimensions,
                gripper_thresholds=gripper_thresholds,
                include_records=False,
            )
            delayed_trace = simulate_delayed_unconditioned_async(
                fresh_plans=fresh_plans,
                start_frame=frame_indices[0],
                num_frames=len(frame_indices),
                delay_frames=prefix_len,
                action_dim=action_dim,
                arm_dimensions=arm_dimensions,
            )
            delayed_all_metrics, delayed_all_records = _metrics_for_trace(
                delayed_trace.executed,
                target_raw_by_frame=target_raw_by_frame,
                action_stats=action_stats,
                action_mode=action_mode,
                arm_dimensions=arm_dimensions,
                gripper_dimensions=gripper_dimensions,
                gripper_thresholds=gripper_thresholds,
                include_records=include_records,
            )
            delayed_replacement = tuple(
                item
                for item in delayed_trace.executed
                if item.source == "delayed_unconditioned_suffix"
            )
            delayed_replacement_metrics, delayed_replacement_records = (
                _metrics_for_trace(
                    delayed_replacement,
                    target_raw_by_frame=target_raw_by_frame,
                    action_stats=action_stats,
                    action_mode=action_mode,
                    arm_dimensions=arm_dimensions,
                    gripper_dimensions=gripper_dimensions,
                    gripper_thresholds=gripper_thresholds,
                    include_records=include_records,
                )
            )
            delayed_vs_rtc = build_delayed_unconditioned_vs_rtc_comparison(
                delayed_trace=delayed_trace,
                rtc_trace=rtc_trace,
                delayed_execution_metrics=delayed_replacement_metrics,
                rtc_execution_metrics=post_metrics,
            )
            rtc_reports[str(prefix_len)] = {
                "prefix_len_and_simulated_delay_frames": prefix_len,
                "bootstrap_definition": (
                    "the first no-prefix request completes synchronously, then its first "
                    "prefix_len rows execute before the first RTC request starts"
                ),
                "conditioned_plan_activation": (
                    "replacement plan is anchored at request frame and starts execution "
                    "at action index prefix_len after the simulated delay"
                ),
                "conditioned_queries": rtc_trace.conditioned_query_count,
                "all_execution_metrics_including_fresh_bootstrap": all_metrics,
                "conditioned_plan_execution_metrics": post_metrics,
                "fresh_sync_h0_on_same_conditioned_plan_frames": fresh_post_metrics,
                "conditioned_plan_arm_mae_ratio_vs_fresh_sync_same_frames": (
                    post_metrics["arm_mae_raw"] / fresh_post_metrics["arm_mae_raw"]
                    if fresh_post_metrics["arm_mae_raw"]
                    else None
                ),
                "prefix_consistency": _prefix_summary(rtc_trace.prefixes),
                "conditioned_inference_latency": _latency_summary(rtc_latencies),
                "executions": all_records if include_records else None,
                "post_bootstrap_executions": post_records if include_records else None,
            }
            delayed_unconditioned_reports[str(prefix_len)] = {
                "delay_frames": prefix_len,
                "ordinary_no_prefix_queries": (
                    delayed_trace.ordinary_no_prefix_query_count
                ),
                "additional_model_calls_beyond_fresh_sync_cache": 0,
                "fresh_plan_cache_seed_definition": (
                    "same deterministic unclipped no-prefix plan used by fresh_sync_h0 at "
                    "each request frame"
                ),
                "rng_seed_paired_with_rtc_conditioned_request": True,
                "bootstrap_definition": (
                    "the first no-prefix plan completes synchronously and executes its "
                    "first delay_frames rows"
                ),
                "async_request_and_activation": (
                    "at each request frame, select the cached ordinary no-prefix plan for "
                    "that observation while executing the old active plan for delay_frames; "
                    "then activate the replacement at output[delay_frames]"
                ),
                "all_execution_metrics_including_fresh_bootstrap": delayed_all_metrics,
                "replacement_plan_execution_metrics": delayed_replacement_metrics,
                "activation_boundary_discontinuity": _activation_boundary_summary(
                    delayed_trace.activation_boundaries
                ),
                "paired_comparison_to_rtc": delayed_vs_rtc,
                "executions": delayed_all_records if include_records else None,
                "replacement_plan_executions": (
                    delayed_replacement_records if include_records else None
                ),
            }
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "evaluation_kind": "read_only_contiguous_recorded_episode_rtc_replay",
            "checkpoint_requested": str(checkpoint_path),
            "checkpoint_resolved": predictor.description.get("resolved_checkpoint"),
            "config_path": str(config_path),
            "backend": predictor.description,
            "safety": {
                "robot_commands_sent": False,
                "policy_server_contacted": False,
                "training_process_contacted": False,
                "dataset_writes": False,
                "checkpoint_writes": False,
            },
            "policy_contract": {
                "action_dim": action_dim,
                "state_dim": state_dim,
                "action_horizon": horizon,
                "qwen_input_contract": metadata.get("realman_input_contract"),
                "rtc_inference_contract": rtc_contract,
                "normalization_key": norm_key,
                "action_normalization_mode": action_mode,
                "state_normalization_mode": state_mode,
                "state_clipped_to_unit_range": clip_state,
                "policy_outputs_clipped_to_unit_range_before_execution_and_rtc_reuse": (
                    False
                ),
                "raw_action_inverse": (
                    "unclipped_training_affine_matches_live_realman_rollout"
                ),
                "rtc_and_delayed_unconditioned_request_rng_seeds_paired": True,
                "normalization_stats_sha256": _stats_fingerprint(
                    {"action": action_stats, "state": state_stats}
                ),
            },
            "replay": {
                "dataset_root": str(cfg.datasets.vla_data.data_root_dir),
                "dataset_name": episode_ref.dataset_name,
                "episode_id": episode_ref.episode_id,
                "episode_length": episode_ref.length,
                "start_frame_inclusive": frame_indices[0],
                "stop_frame_exclusive": frame_indices[-1] + 1,
                "num_contiguous_frames": len(frame_indices),
                "frame_step": 1,
                "action_offsets_verified": list(range(horizon)),
                "video_target_shift_steps": video_target_shift_steps,
                "instruction_sent_exactly": instruction,
                "instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
                "dataset_instructions_seen": sorted(dataset_instructions),
                "qwen_frame_source": "current training-aligned recorded context frame",
                "raw_state_source": "authoritative float32 parquet row",
                "raw_target_source": "authoritative float32 parquet h0 action row",
                "state_oob_elements_before_optional_clip": state_oob_total,
                "fresh_full_normalized_plans_serialized_in_report": False,
            },
            "method_definitions": {
                "fresh_sync_h0": (
                    "fresh no-prefix plan at every recorded frame; execute action index 0"
                ),
                "open_loop": (
                    "fresh no-prefix plan at each chunk boundary; execute indices 0..chunk-1"
                ),
                "rtc_overlap": (
                    "after one synchronous bootstrap prefix, request every prefix_len "
                    "frames using the prior active normalized plan slice shifted by elapsed "
                    "recorded frames; execute that committed slice during inference, then "
                    "activate the replacement suffix beginning at output[prefix_len]"
                ),
                "delayed_unconditioned_async": (
                    "same fixed-delay execution schedule as rtc_overlap, but each replacement "
                    "is an ordinary no-prefix fresh plan cached for the request observation; "
                    "after executing the old committed slice during delay, activate the "
                    "unconditioned replacement at output[delay_frames]"
                ),
            },
            "methods": {
                "fresh_sync_h0": {
                    "fresh_plan_queries": len(frame_indices),
                    "execution_metrics": fresh_metrics,
                    "inference_latency": _latency_summary(fresh_latency_ms),
                    "executions": fresh_records if include_records else None,
                },
                "open_loop": open_loop_reports,
                "rtc_overlap": rtc_reports,
                "delayed_unconditioned_async": delayed_unconditioned_reports,
            },
            "warnings": [
                "These episodes belong to the configured training dataset and are not a "
                "held-out generalization set.",
                "Recorded-state replay measures action agreement under expert observations; "
                "it does not reproduce closed-loop state drift from policy mistakes.",
                "RTC delay is measured in recorded frames, not wall-clock latency. Convert "
                "measured deployment latency to frames before choosing a live prefix length.",
                "Full normalized fresh trajectories are intentionally not serialized. A saved "
                "older report that lacks this method cannot be augmented with the delayed-"
                "unconditioned baseline without rerunning checkpoint inference.",
            ],
        }
        json.dumps(_json_safe(report), allow_nan=False)
        return report
    finally:
        predictor.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument(
        "--instruction",
        required=True,
        help="Exact global task prompt used by the live Realman client.",
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--open-loop-chunk-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_OPEN_LOOP_CHUNK_SIZES),
    )
    parser.add_argument(
        "--rtc-prefix-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_RTC_PREFIX_LENGTHS),
    )
    parser.add_argument(
        "--prefix-copy-tolerance",
        type=float,
        default=DEFAULT_PREFIX_COPY_TOLERANCE,
    )
    parser.add_argument(
        "--include-frame-records",
        action="store_true",
        help="Include compact per-frame alignment/error diagnostics in the JSON report.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    report = run_replay(args)
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "report_path": str(report_path),
        "checkpoint": report["checkpoint_resolved"],
        "replay": report["replay"],
        "fresh_sync_h0": report["methods"]["fresh_sync_h0"]["execution_metrics"],
        "open_loop": {
            key: value["execution_metrics"]
            for key, value in report["methods"]["open_loop"].items()
        },
        "rtc_overlap": {
            key: {
                "conditioned_plan_execution_metrics": value[
                    "conditioned_plan_execution_metrics"
                ],
                "prefix_consistency": {
                    name: metric
                    for name, metric in value["prefix_consistency"].items()
                    if name != "alignment_records"
                },
            }
            for key, value in report["methods"]["rtc_overlap"].items()
        },
        "delayed_unconditioned_async": {
            key: {
                "replacement_plan_execution_metrics": value[
                    "replacement_plan_execution_metrics"
                ],
                "activation_boundary_discontinuity": {
                    name: metric
                    for name, metric in value[
                        "activation_boundary_discontinuity"
                    ].items()
                    if name != "records"
                },
                "paired_comparison_to_rtc": {
                    name: metric
                    for name, metric in value["paired_comparison_to_rtc"].items()
                    if name != "activation_boundary_discontinuity"
                },
            }
            for key, value in report["methods"][
                "delayed_unconditioned_async"
            ].items()
        },
    }
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
