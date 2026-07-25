from __future__ import annotations

import pytest

from starVLA.eval_sampling_policy import (
    DATASET_FRACTION_DIVISOR_POLICY,
    derive_episode_holdout_sampling_plan,
)


POLICY = {
    "algorithm": DATASET_FRACTION_DIVISOR_POLICY,
    "minimum_episode_fraction": 0.05,
    "maximum_episode_fraction": 0.08,
    "episode_count_multiple": 8,
    "max_episode_count": 128,
    "evaluation_observation_count": 128,
}


@pytest.mark.parametrize(
    ("episodes", "heldout", "frames"),
    (
        (100, 8, 16),
        (200, 16, 8),
        (553, 32, 4),
        (1_000, 56, 2),
        (1_700, 128, 1),
        (3_000, 128, 1),
    ),
)
def test_dataset_fraction_policy_resolves_exact_global_batch(
    episodes: int,
    heldout: int,
    frames: int,
):
    result = derive_episode_holdout_sampling_plan(
        total_episode_count=episodes,
        evaluation_observation_count=128,
        policy=POLICY,
    )
    assert result["holdout_episode_count"] == heldout
    assert result["base_frames_per_episode"] == frames
    assert (
        heldout * frames + result["extra_window_episode_count"]
        == 128
    )
    assert heldout % 8 == 0


def test_dataset_fraction_policy_records_divisibility_fallback():
    result = derive_episode_holdout_sampling_plan(
        total_episode_count=700,
        evaluation_observation_count=128,
        policy=POLICY,
    )
    assert result["holdout_episode_count"] == 40
    assert result["within_preferred_fraction"] is True
    assert result["extra_window_episode_count"] == 8


def test_dataset_fraction_policy_records_small_catalog_rounding():
    result = derive_episode_holdout_sampling_plan(
        total_episode_count=50,
        evaluation_observation_count=128,
        policy=POLICY,
    )
    assert result["holdout_episode_count"] == 8
    assert result["fraction_band_exception"] == "small_dataset_rounding"
    assert result["within_preferred_fraction"] is False


def test_dataset_fraction_policy_fails_only_without_train_complement():
    with pytest.raises(ValueError, match="too small"):
        derive_episode_holdout_sampling_plan(
            total_episode_count=8,
            evaluation_observation_count=128,
            policy=POLICY,
        )


def test_dataset_fraction_policy_is_total_for_every_supported_catalog_size():
    for episode_count in range(9, 10_001):
        result = derive_episode_holdout_sampling_plan(
            total_episode_count=episode_count,
            evaluation_observation_count=128,
            policy=POLICY,
        )
        heldout = result["holdout_episode_count"]
        assert heldout % 8 == 0
        assert 0 < heldout < episode_count
        assert heldout <= 128
        assert (
            heldout * result["base_frames_per_episode"]
            + result["extra_window_episode_count"]
            == 128
        )
