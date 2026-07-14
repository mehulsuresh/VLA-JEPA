from __future__ import annotations

import numpy as np
import pytest

from deployment.realman.run_realman_policy import (
    _realman_continuous_unnormalize,
)


@pytest.mark.parametrize(
    ("mode", "stats", "expected"),
    [
        (
            "min_max",
            {"min": [0.0, 10.0], "max": [2.0, 14.0]},
            [[-0.5, 15.0]],
        ),
        (
            "q99",
            {
                "q01": [0.0, 7.0],
                "q99": [2.0, 7.0],
                # Training's inverse is affine and does not branch on this
                # legacy metadata field.
                "mask": [True, False],
            },
            [[-0.5, 7.0]],
        ),
        (
            "mean_std",
            {"mean": [1.0, 2.0], "std": [2.0, 4.0]},
            [[-2.0, 8.0]],
        ),
    ],
)
def test_realman_policy_inverse_matches_training_without_clipping(
    mode: str,
    stats: dict[str, list[float]],
    expected: list[list[float]],
) -> None:
    normalized = np.asarray([[-1.5, 1.5]], dtype=np.float32)

    actual = _realman_continuous_unnormalize(normalized, stats, mode=mode)

    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float32))


def test_realman_policy_inverse_rejects_action_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        _realman_continuous_unnormalize(
            np.zeros((3, 2), dtype=np.float32),
            {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
            mode="min_max",
        )
