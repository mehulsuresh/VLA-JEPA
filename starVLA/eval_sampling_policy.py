"""Dataset-agnostic episode-holdout sampling policy.

The policy keeps an immutable episode holdout near a configured fraction of
the full episode catalog while still producing an exact configured number of
evaluation observations. Window multiplicity is balanced across episodes;
when division has a remainder, a deterministic ranked subset receives one
additional window.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


DATASET_FRACTION_DIVISOR_POLICY = "dataset_fraction_divisor_v1"


def validate_holdout_sampling_policy(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize a config-owned holdout sampling policy."""

    if not isinstance(value, Mapping):
        raise ValueError("holdout_sampling must be a mapping")
    algorithm = value.get("algorithm")
    if algorithm != DATASET_FRACTION_DIVISOR_POLICY:
        raise ValueError(
            "holdout_sampling.algorithm must be "
            f"{DATASET_FRACTION_DIVISOR_POLICY!r}, got {algorithm!r}"
        )
    minimum_fraction = value.get("minimum_episode_fraction")
    maximum_fraction = value.get("maximum_episode_fraction")
    for name, fraction in (
        ("minimum_episode_fraction", minimum_fraction),
        ("maximum_episode_fraction", maximum_fraction),
    ):
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) < 1.0
        ):
            raise ValueError(
                f"holdout_sampling.{name} must be finite and strictly between 0 and 1"
            )
    if float(minimum_fraction) > float(maximum_fraction):
        raise ValueError(
            "holdout_sampling.minimum_episode_fraction cannot exceed "
            "maximum_episode_fraction"
        )
    episode_count_multiple = value.get("episode_count_multiple")
    if (
        isinstance(episode_count_multiple, bool)
        or not isinstance(episode_count_multiple, int)
        or episode_count_multiple <= 0
    ):
        raise ValueError(
            "holdout_sampling.episode_count_multiple must be a positive integer"
        )
    max_episode_count = value.get("max_episode_count")
    evaluation_observation_count = value.get("evaluation_observation_count")
    for name, count in (
        ("max_episode_count", max_episode_count),
        ("evaluation_observation_count", evaluation_observation_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"holdout_sampling.{name} must be a positive integer"
            )
    if max_episode_count % episode_count_multiple != 0:
        raise ValueError(
            "holdout_sampling.max_episode_count must be a multiple of "
            "episode_count_multiple"
        )
    if evaluation_observation_count < max_episode_count:
        raise ValueError(
            "holdout_sampling.evaluation_observation_count cannot be smaller "
            "than max_episode_count"
        )
    return {
        "algorithm": DATASET_FRACTION_DIVISOR_POLICY,
        "minimum_episode_fraction": float(minimum_fraction),
        "maximum_episode_fraction": float(maximum_fraction),
        "episode_count_multiple": int(episode_count_multiple),
        "max_episode_count": int(max_episode_count),
        "evaluation_observation_count": int(evaluation_observation_count),
    }


def derive_episode_holdout_sampling_plan(
    *,
    total_episode_count: int,
    evaluation_observation_count: int,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve episode/window counts for any finite episode catalog.

    If the configured maximum episode count is no more than the preferred
    maximum fraction, it is used even when this falls below the preferred
    minimum on a very large catalog. Otherwise the minimum target is rounded
    up to ``episode_count_multiple``. Small catalogs may require one recorded
    rounding exception above the maximum percentage.
    """

    if (
        isinstance(total_episode_count, bool)
        or not isinstance(total_episode_count, int)
        or total_episode_count <= 1
    ):
        raise ValueError("total_episode_count must be an integer greater than one")
    if (
        isinstance(evaluation_observation_count, bool)
        or not isinstance(evaluation_observation_count, int)
        or evaluation_observation_count <= 0
    ):
        raise ValueError(
            "evaluation_observation_count must be a positive integer"
        )
    normalized = validate_holdout_sampling_policy(policy)
    if (
        evaluation_observation_count
        != normalized["evaluation_observation_count"]
    ):
        raise ValueError(
            "evaluation_observation_count does not match the configured "
            "holdout sampling policy"
        )
    minimum_target = math.ceil(
        total_episode_count * normalized["minimum_episode_fraction"]
    )
    maximum_target = math.floor(
        total_episode_count * normalized["maximum_episode_fraction"]
    )
    maximum_available = min(
        normalized["max_episode_count"],
        evaluation_observation_count,
        total_episode_count - 1,
    )
    multiple = normalized["episode_count_multiple"]
    maximum_available -= maximum_available % multiple
    if maximum_available < multiple:
        raise ValueError(
            "The episode catalog is too small for one holdout multiple while "
            "retaining training episodes"
        )
    if maximum_available <= maximum_target:
        holdout_episode_count = maximum_available
        fraction_band_exception = (
            None
            if holdout_episode_count >= minimum_target
            else "maximum_episode_cap_below_minimum_fraction"
        )
    else:
        rounded_minimum = (
            math.ceil(minimum_target / multiple) * multiple
        )
        if rounded_minimum <= maximum_target:
            holdout_episode_count = rounded_minimum
            fraction_band_exception = None
        else:
            # The multiple itself is the smallest meaningful holdout. This is
            # the only way to support small catalogs without silently changing
            # the configured episode-count granularity.
            holdout_episode_count = multiple
            fraction_band_exception = "small_dataset_rounding"
    if (
        holdout_episode_count > maximum_available
        or holdout_episode_count >= total_episode_count
    ):
        raise ValueError(
            "The resolved holdout cannot retain a non-empty training complement"
        )
    base_frames_per_episode, extra_window_episode_count = divmod(
        evaluation_observation_count,
        holdout_episode_count,
    )
    if base_frames_per_episode <= 0:
        raise ValueError(
            "Resolved holdout has more episodes than evaluation observations"
        )
    within_preferred_fraction = (
        minimum_target <= holdout_episode_count <= maximum_target
    )
    return {
        **normalized,
        "total_episode_count": int(total_episode_count),
        "evaluation_observation_count": int(evaluation_observation_count),
        "minimum_target_episode_count": int(minimum_target),
        "maximum_target_episode_count": int(maximum_target),
        "holdout_episode_count": int(holdout_episode_count),
        "base_frames_per_episode": int(base_frames_per_episode),
        "extra_window_episode_count": int(extra_window_episode_count),
        "maximum_frames_per_episode": int(
            base_frames_per_episode
            + int(extra_window_episode_count > 0)
        ),
        "window_allocation_algorithm": "balanced_digest_rank_v1",
        "actual_episode_fraction": (
            float(holdout_episode_count) / float(total_episode_count)
        ),
        "within_preferred_fraction": bool(within_preferred_fraction),
        "fraction_band_exception": fraction_band_exception,
    }
