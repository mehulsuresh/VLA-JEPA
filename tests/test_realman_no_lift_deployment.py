import numpy as np
import pytest

from deployment.model_server.server_policy import (
    ActionGuardRetryPolicy,
    _expand_realman_policy_actions,
    _is_realman_data_mix,
)
from deployment.realman.pipeline import (
    REALMAN_ACTION_DIM,
    REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT,
    expand_policy_action_to_robot_action,
)


def test_no_lift_policy_expansion_preserves_measured_lift():
    policy_action = np.arange(REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT, dtype=np.float32)

    expanded = expand_policy_action_to_robot_action(policy_action, lift_height_mm=310.0)

    assert expanded.shape == (REALMAN_ACTION_DIM,)
    np.testing.assert_allclose(expanded[:16], policy_action[:16])
    np.testing.assert_allclose(expanded[16:19], 0.0)
    np.testing.assert_allclose(expanded[19:21], policy_action[16:18])
    assert expanded[21] == pytest.approx(310.0)


def test_no_lift_policy_expansion_rejects_missing_lift_state():
    policy_action = np.zeros((REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT,), dtype=np.float32)

    with pytest.raises(ValueError, match="current measured lift_height_mm"):
        expand_policy_action_to_robot_action(policy_action)


def test_policy_server_recognizes_new_realman_robot_type_and_maps_head_only():
    assert _is_realman_data_mix({"robot_type": "realman_bimanual_source_no_base_no_lift"})
    policy_action = np.arange(REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT, dtype=np.float32)

    expanded = _expand_realman_policy_actions(policy_action)

    np.testing.assert_allclose(expanded[:16], policy_action[:16])
    np.testing.assert_allclose(expanded[16:19], 0.0)
    np.testing.assert_allclose(expanded[19:21], policy_action[16:18])
    assert expanded[21] == 0.0


def test_action_guard_skips_lift_threshold_for_policy_that_does_not_control_lift():
    action_dim = REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT
    metadata = {
        "default_unnorm_key": "realman",
        "default_action_norm_mode": "min_max",
        "action_stats_by_key": {
            "realman": {
                "min": [-1.0] * action_dim,
                "max": [1.0] * action_dim,
            }
        },
    }
    guard = ActionGuardRetryPolicy(
        object(),
        metadata=metadata,
        max_attempts=2,
        first_n=10,
        tail_start=20,
        late_start=30,
        last_n=5,
        min_first_arms_mean=-1.0,
        min_tail_arms_mean=-1.0,
        min_late_arms_mean=-1.0,
        min_last_arms_mean=-1.0,
        min_tail_lift_mean=250.0,
    )

    valid, metrics, reasons = guard._validate_output(
        {"normalized_actions": np.zeros((1, 50, action_dim), dtype=np.float32)}
    )

    assert valid
    assert reasons == []
    assert metrics["lift_is_policy_controlled"] is False
    assert metrics["tail_lift_mean_min"] is None
