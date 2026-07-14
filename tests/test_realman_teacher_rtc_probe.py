from __future__ import annotations

import numpy as np
import pytest

from deployment.realman.evaluate_teacher_rtc_probe import (
    _finalize_regions,
    _new_regions,
    _update_regions_from_normalized_plan,
)


def test_teacher_probe_primary_raw_metrics_preserve_model_extrapolation() -> None:
    regions = _new_regions(prefix_len=1, horizon=2, arm_dimensions=(0, 1))
    prediction = np.asarray(
        [[1.5, -1.5, 0.0], [1.5, -1.5, 0.0]],
        dtype=np.float32,
    )

    _update_regions_from_normalized_plan(
        regions,
        prediction_normalized=prediction,
        target_normalized=np.zeros_like(prediction),
        action_stats={"min": [0.0, 0.0, 0.0], "max": [2.0, 2.0, 2.0]},
        action_mode="min_max",
        target_raw=np.zeros_like(prediction),
        valid_mask=np.ones_like(prediction, dtype=bool),
    )

    report = _finalize_regions(regions)
    # Exact inverse is [2.5, -0.5, 1.0], so arm MAE is 1.5.  The former
    # bounded inverse would report 1.0 for the same model output.
    assert report["h0_reference"]["raw_arm_mae_rad"] == pytest.approx(1.5)
    assert report["first_predicted_row"]["raw_arm_mae_rad"] == pytest.approx(1.5)
