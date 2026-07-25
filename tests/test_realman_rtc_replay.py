from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from deployment.realman.evaluate_rtc_replay import (
    ExecutionMetrics,
    ExecutedAction,
    _metrics_for_trace,
    align_prior_normalized_plan,
    align_prior_normalized_prefix,
    build_delayed_unconditioned_vs_rtc_comparison,
    build_fresh_h0_trace,
    build_open_loop_trace,
    fresh_plan_inference_seed,
    simulate_delayed_unconditioned_async,
    simulate_rtc_overlap,
    validate_contiguous_frame_indices,
    validate_contiguous_frame_range,
    validate_replay_method_configuration,
    validate_returned_frame_index,
    validate_rtc_inference_contract,
    validate_normalized_plan,
    validate_unit_step_action_offsets,
)


def _rtc_metadata(**overrides):
    contract = {
        "training_enabled": True,
        "method": "prefix",
        "max_delay_exclusive": 11,
        "max_prefix_len": 10,
        "action_space": "normalized_policy_action",
        "client_opt_in": True,
    }
    contract.update(overrides)
    return {"rtc_inference_contract": contract}


def _plan(base: float, *, horizon: int = 8, action_dim: int = 3) -> np.ndarray:
    indices = np.arange(horizon, dtype=np.float32)[:, None]
    dimensions = np.arange(action_dim, dtype=np.float32)[None, :] / 100.0
    return base + indices + dimensions


def test_fresh_plan_seed_is_canonical_per_recorded_observation() -> None:
    kwargs = {
        "base_seed": 7,
        "dataset_name": "data",
        "episode_id": 12,
        "frame_index": 34,
    }
    assert fresh_plan_inference_seed(**kwargs) == fresh_plan_inference_seed(**kwargs)
    assert fresh_plan_inference_seed(**kwargs) != fresh_plan_inference_seed(
        **{**kwargs, "frame_index": 35}
    )


def test_normalized_policy_plan_preserves_out_of_range_predictions() -> None:
    plan = np.asarray([[1.5, -1.5], [2.0, -2.0]], dtype=np.float32)

    actual = validate_normalized_plan(
        plan,
        horizon=2,
        action_dim=2,
        name="out-of-range model plan",
    )

    np.testing.assert_array_equal(actual, plan)


def test_execution_metrics_use_unclipped_training_affine_inverse() -> None:
    trace = (
        ExecutedAction(
            frame_index=7,
            plan_anchor_frame=7,
            plan_action_index=0,
            source="test",
            normalized_action=np.asarray([1.5, -1.5, 1.5], dtype=np.float32),
        ),
    )

    metrics, records = _metrics_for_trace(
        trace,
        target_raw_by_frame={7: np.zeros(3, dtype=np.float32)},
        action_stats={"min": [0.0, 0.0, 0.0], "max": [2.0, 2.0, 2.0]},
        action_mode="min_max",
        arm_dimensions=(0, 1),
        gripper_dimensions=(2,),
        gripper_thresholds=np.ones(3, dtype=np.float32),
        include_records=True,
    )

    # Exact inverse is [2.5, -0.5, 2.5], so arm MAE is 1.5.  A silently
    # clipped inverse would instead report 1.0 and conceal extrapolation.
    assert metrics["arm_mae_raw"] == pytest.approx(1.5)
    assert records[0]["arm_mae_raw"] == pytest.approx(1.5)


def test_contiguous_replay_range_is_exact_and_never_clipped() -> None:
    assert validate_contiguous_frame_range(
        episode_length=20, start_frame=7, num_frames=4
    ) == (7, 8, 9, 10)
    with pytest.raises(ValueError, match="exceeds episode length"):
        validate_contiguous_frame_range(
            episode_length=10, start_frame=8, num_frames=3
        )
    with pytest.raises(ValueError, match="positive"):
        validate_contiguous_frame_range(
            episode_length=10, start_frame=2, num_frames=0
        )
    with pytest.raises(ValueError, match="non-negative"):
        validate_contiguous_frame_range(
            episode_length=10, start_frame=-1, num_frames=2
        )


def test_contiguous_frame_indices_reject_gaps_reordering_and_boolean_indices() -> None:
    assert validate_contiguous_frame_indices([4, 5, 6]) == (4, 5, 6)
    for frames in ([4, 6], [5, 4], [4, 4]):
        with pytest.raises(ValueError, match="contiguous order"):
            validate_contiguous_frame_indices(frames)
    with pytest.raises(ValueError, match="must be an integer"):
        validate_contiguous_frame_indices([4, True])


def test_method_configuration_accepts_only_executable_aligned_sizes() -> None:
    chunks, prefixes = validate_replay_method_configuration(
        horizon=50,
        num_frames=100,
        open_loop_chunk_sizes=[20, 5, 10],
        rtc_prefix_lengths=[10, 1, 5],
        rtc_max_prefix_len=10,
    )
    assert chunks == (5, 10, 20)
    assert prefixes == (1, 5, 10)

    invalid_cases = [
        ({"open_loop_chunk_sizes": [51]}, "exceeds action horizon"),
        ({"rtc_prefix_lengths": [11]}, "exceeds checkpoint maximum"),
        (
            {"horizon": 10, "rtc_prefix_lengths": [6], "rtc_max_prefix_len": 6},
            "cannot be repeatedly aligned",
        ),
        ({"num_frames": 5, "rtc_prefix_lengths": [5]}, "must exceed"),
        ({"num_frames": 101, "rtc_prefix_lengths": [5]}, "must be divisible"),
        ({"rtc_prefix_lengths": [5, 5]}, "Duplicate"),
        ({"open_loop_chunk_sizes": [True]}, "must be an integer"),
    ]
    defaults = {
        "horizon": 50,
        "num_frames": 100,
        "open_loop_chunk_sizes": [5],
        "rtc_prefix_lengths": [5],
        "rtc_max_prefix_len": 10,
    }
    for overrides, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            validate_replay_method_configuration(**{**defaults, **overrides})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training_enabled": False}, "not enabled"),
        ({"method": "blend"}, "expected 'prefix'"),
        ({"action_space": "raw"}, "normalized_policy_action"),
        ({"client_opt_in": False}, "opt-in"),
        ({"max_prefix_len": True}, "must be an integer"),
        ({"max_prefix_len": 0}, "must be positive"),
    ],
)
def test_rtc_contract_fails_closed_on_unproven_semantics(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_rtc_inference_contract(
            _rtc_metadata(**overrides), requested_prefix_lengths=[1]
        )


def test_rtc_contract_rejects_requested_prefix_past_checkpoint_support() -> None:
    with pytest.raises(ValueError, match="exceeds checkpoint maximum"):
        validate_rtc_inference_contract(
            _rtc_metadata(max_prefix_len=4), requested_prefix_lengths=[5]
        )
    contract = validate_rtc_inference_contract(
        _rtc_metadata(), requested_prefix_lengths=[1, 5, 10]
    )
    assert contract["max_prefix_len"] == 10


def test_unit_step_action_offsets_are_required_for_frame_alignment() -> None:
    aligned = SimpleNamespace(
        modality_keys={"action": ["action.arm", "action.gripper"]},
        delta_indices={
            "action.arm": np.arange(4),
            "action.gripper": np.arange(4),
        },
    )
    validate_unit_step_action_offsets(aligned, horizon=4)

    shifted = SimpleNamespace(
        modality_keys=aligned.modality_keys,
        delta_indices={
            "action.arm": np.array([0, 1, 3, 4]),
            "action.gripper": np.arange(4),
        },
    )
    with pytest.raises(ValueError, match="unit-step recorded-frame offsets"):
        validate_unit_step_action_offsets(shifted, horizon=4)


def test_targeted_loader_frame_identity_is_required() -> None:
    validate_returned_frame_index(
        {"frame_index": np.array([17], dtype=np.int64)}, requested=17
    )
    with pytest.raises(ValueError, match="missing frame_index"):
        validate_returned_frame_index({}, requested=17)
    with pytest.raises(ValueError, match="loader returned 18"):
        validate_returned_frame_index(
            {"frame_index": np.array([18], dtype=np.int64)}, requested=17
        )
    with pytest.raises(ValueError, match="must be integral"):
        validate_returned_frame_index(
            {"frame_index": np.array([17.0], dtype=np.float32)}, requested=17
        )


def test_prior_prefix_alignment_shifts_by_elapsed_recorded_frames() -> None:
    plan = _plan(100.0, horizon=8, action_dim=2)
    prefix = align_prior_normalized_prefix(
        plan,
        prior_anchor_frame=40,
        request_frame=43,
        prefix_len=3,
    )
    np.testing.assert_array_equal(prefix, plan[3:6])
    assert not np.array_equal(prefix, plan[:3])

    with pytest.raises(ValueError, match="precedes prior plan anchor"):
        align_prior_normalized_prefix(
            plan, prior_anchor_frame=40, request_frame=39, prefix_len=1
        )
    with pytest.raises(ValueError, match="exceeds prior plan horizon"):
        align_prior_normalized_prefix(
            plan, prior_anchor_frame=40, request_frame=47, prefix_len=2
        )
    bad = plan.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        align_prior_normalized_prefix(
            bad, prior_anchor_frame=40, request_frame=40, prefix_len=1
        )


def test_full_prior_plan_alignment_shifts_and_holds_final_row_like_client() -> None:
    plan = _plan(10.0, horizon=6, action_dim=2)
    aligned = align_prior_normalized_plan(
        plan,
        prior_anchor_frame=100,
        request_frame=102,
    )
    np.testing.assert_array_equal(aligned[:4], plan[2:])
    np.testing.assert_array_equal(aligned[4:], np.repeat(plan[-1:], 2, axis=0))
    with pytest.raises(ValueError, match="no unexecuted action remains"):
        align_prior_normalized_plan(
            plan,
            prior_anchor_frame=100,
            request_frame=106,
        )


def test_rtc_replay_executes_prior_plan_during_delay_then_conditioned_suffix() -> None:
    bootstrap = _plan(0.0)
    supplied: list[tuple[int, np.ndarray]] = []
    generated: dict[int, np.ndarray] = {}

    def predict(
        request_frame: int, prev_actions: np.ndarray, prefix_len: int
    ) -> np.ndarray:
        supplied.append((request_frame, prev_actions.copy()))
        plan = _plan(float(request_frame * 10))
        plan[:prefix_len] = prev_actions[:prefix_len]
        generated[request_frame] = plan.copy()
        return plan

    trace = simulate_rtc_overlap(
        bootstrap_plan=bootstrap,
        start_frame=100,
        num_frames=8,
        prefix_len=2,
        action_dim=3,
        arm_dimensions=(0, 1),
        predict_conditioned=predict,
    )

    assert [item.frame_index for item in trace.executed] == list(range(100, 108))
    assert [item.plan_anchor_frame for item in trace.executed] == [
        100,
        100,
        100,
        100,
        102,
        102,
        104,
        104,
    ]
    assert [item.plan_action_index for item in trace.executed] == [
        0,
        1,
        2,
        3,
        2,
        3,
        2,
        3,
    ]
    assert [item.source for item in trace.executed[:4]] == [
        "fresh_bootstrap",
        "fresh_bootstrap",
        "fresh_bootstrap",
        "fresh_bootstrap",
    ]
    assert all(
        item.source == "rtc_conditioned_suffix" for item in trace.executed[4:]
    )
    assert [request for request, _ in supplied] == [102, 104, 106]
    # The first RTC request starts only after synchronous bootstrap execution,
    # and must use rows 2:4 rather than already-executed bootstrap rows 0:2.
    np.testing.assert_array_equal(supplied[0][1][:2], bootstrap[2:4])
    np.testing.assert_array_equal(supplied[0][1][:6], bootstrap[2:])
    np.testing.assert_array_equal(
        supplied[0][1][6:], np.repeat(bootstrap[-1:], 2, axis=0)
    )
    np.testing.assert_array_equal(supplied[1][1][:2], generated[102][2:4])
    np.testing.assert_array_equal(supplied[2][1][:2], generated[104][2:4])
    np.testing.assert_array_equal(
        np.stack([item.normalized_action for item in trace.executed[4:6]]),
        generated[102][2:4],
    )
    assert trace.conditioned_query_count == 3
    assert all(item.copy_max_abs_normalized == 0.0 for item in trace.prefixes)
    assert [item.prior_plan_elapsed_frames for item in trace.prefixes] == [2, 2, 2]


def test_rtc_replay_fails_closed_if_model_does_not_copy_prefix() -> None:
    bootstrap = _plan(0.0)

    def bad_predict(_frame: int, _prefix: np.ndarray, _length: int) -> np.ndarray:
        return _plan(100.0)

    with pytest.raises(ValueError, match="failed to preserve"):
        simulate_rtc_overlap(
            bootstrap_plan=bootstrap,
            start_frame=0,
            num_frames=6,
            prefix_len=2,
            action_dim=3,
            arm_dimensions=(0, 1),
            predict_conditioned=bad_predict,
            prefix_copy_tolerance=0.0,
        )


def test_rtc_replay_rejects_prefix_that_exhausts_active_plan() -> None:
    with pytest.raises(ValueError, match="cannot score an activation boundary"):
        simulate_rtc_overlap(
            bootstrap_plan=_plan(0.0, horizon=3),
            start_frame=0,
            num_frames=6,
            prefix_len=2,
            action_dim=3,
            arm_dimensions=(0, 1),
            predict_conditioned=lambda _frame, _prefix, _length: _plan(
                0.0, horizon=3
            ),
        )


def test_delayed_unconditioned_async_uses_cached_request_plan_at_delayed_index() -> None:
    plans = {frame: _plan(float(frame * 10)) for frame in range(100, 108)}
    trace = simulate_delayed_unconditioned_async(
        fresh_plans=plans,
        start_frame=100,
        num_frames=8,
        delay_frames=2,
        action_dim=3,
        arm_dimensions=(0, 1),
    )

    assert [item.frame_index for item in trace.executed] == list(range(100, 108))
    assert [item.plan_anchor_frame for item in trace.executed] == [
        100,
        100,
        100,
        100,
        102,
        102,
        104,
        104,
    ]
    assert [item.plan_action_index for item in trace.executed] == [
        0,
        1,
        2,
        3,
        2,
        3,
        2,
        3,
    ]
    assert all(
        item.source == "fresh_bootstrap" for item in trace.executed[:4]
    )
    assert all(
        item.source == "delayed_unconditioned_suffix"
        for item in trace.executed[4:]
    )
    np.testing.assert_array_equal(
        np.stack([item.normalized_action for item in trace.executed[4:6]]),
        plans[102][2:4],
    )
    assert [item.request_frame for item in trace.activation_boundaries] == [
        102,
        104,
        106,
    ]
    first_boundary = trace.activation_boundaries[0]
    assert first_boundary.activation_frame == 104
    assert first_boundary.prior_plan_action_index == 4
    assert first_boundary.replacement_plan_action_index == 2
    expected = np.abs(plans[102][2] - plans[100][4])
    assert first_boundary.action_mae_normalized == pytest.approx(expected.mean())
    assert first_boundary.arm_mae_normalized == pytest.approx(expected[:2].mean())
    assert trace.ordinary_no_prefix_query_count == 3


def test_delayed_unconditioned_async_fails_closed_on_missing_cached_request_plan() -> None:
    plans = {frame: _plan(float(frame)) for frame in range(8)}
    del plans[2]
    with pytest.raises(ValueError, match="request frame 2"):
        simulate_delayed_unconditioned_async(
            fresh_plans=plans,
            start_frame=0,
            num_frames=8,
            delay_frames=2,
            action_dim=3,
            arm_dimensions=(0, 1),
        )


def test_delayed_vs_rtc_comparison_requires_and_reports_identical_frames() -> None:
    plans = {frame: _plan(float(frame * 10)) for frame in range(100, 110)}
    delayed = simulate_delayed_unconditioned_async(
        fresh_plans=plans,
        start_frame=100,
        num_frames=8,
        delay_frames=2,
        action_dim=3,
        arm_dimensions=(0, 1),
    )

    def conditioned(request_frame, prev_actions, prefix_len):
        replacement = plans[request_frame].copy()
        replacement[:prefix_len] = prev_actions[:prefix_len]
        return replacement

    rtc = simulate_rtc_overlap(
        bootstrap_plan=plans[100],
        start_frame=100,
        num_frames=8,
        prefix_len=2,
        action_dim=3,
        arm_dimensions=(0, 1),
        predict_conditioned=conditioned,
    )
    metric_template = {
        "arm_mae_raw": 0.25,
        "gripper_close_recall": 0.75,
        "gripper_close_precision": 0.5,
        "gripper_close_accuracy": 0.8,
    }
    comparison = build_delayed_unconditioned_vs_rtc_comparison(
        delayed_trace=delayed,
        rtc_trace=rtc,
        delayed_execution_metrics=metric_template,
        rtc_execution_metrics=metric_template,
    )
    assert comparison["identical_replacement_execution_frames"] is True
    assert comparison["replacement_execution_frames"]["count"] == 4
    assert comparison["identical_activation_request_frames"] is True
    assert comparison["execution_metrics"]["delayed_minus_rtc_arm_mae_raw"] == 0.0
    assert comparison["activation_boundary_discontinuity"][
        "delayed_minus_rtc_arm_normalized_mae"
    ] == pytest.approx(0.0)

    longer_rtc = simulate_rtc_overlap(
        bootstrap_plan=plans[100],
        start_frame=100,
        num_frames=10,
        prefix_len=2,
        action_dim=3,
        arm_dimensions=(0, 1),
        predict_conditioned=conditioned,
    )
    with pytest.raises(ValueError, match="not evaluated on identical frames"):
        build_delayed_unconditioned_vs_rtc_comparison(
            delayed_trace=delayed,
            rtc_trace=longer_rtc,
            delayed_execution_metrics=metric_template,
            rtc_execution_metrics=metric_template,
        )


def test_fresh_and_open_loop_traces_use_expected_plan_action_indices() -> None:
    plans = {frame: _plan(float(frame * 10)) for frame in range(10, 16)}
    frames = tuple(range(10, 16))
    fresh = build_fresh_h0_trace(plans, frame_indices=frames)
    assert [item.plan_anchor_frame for item in fresh] == list(frames)
    assert [item.plan_action_index for item in fresh] == [0] * len(frames)

    open_loop = build_open_loop_trace(plans, frame_indices=frames, chunk_size=4)
    assert [item.plan_anchor_frame for item in open_loop] == [10, 10, 10, 10, 14, 14]
    assert [item.plan_action_index for item in open_loop] == [0, 1, 2, 3, 0, 1]
    np.testing.assert_array_equal(open_loop[3].normalized_action, plans[10][3])
    np.testing.assert_array_equal(open_loop[5].normalized_action, plans[14][1])


def test_execution_metrics_report_close_recall_and_precision_with_close_positive() -> None:
    metrics = ExecutionMetrics(
        arm_dimensions=(0, 1),
        gripper_dimensions=(2, 3),
        gripper_thresholds=np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
    )
    # Frame one: left close TP, right open TN.
    metrics.update(
        np.array([1.0, 3.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0, 1.0]),
    )
    # Frame two: left target close missed (FN), right predicted close while open (FP).
    metrics.update(
        np.array([2.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    report = metrics.finalize()
    assert report["arm_mae_raw"] == pytest.approx(1.25)
    assert report["gripper_close_recall"] == pytest.approx(0.5)
    assert report["gripper_close_precision"] == pytest.approx(0.5)
    assert report["gripper_close_accuracy"] == pytest.approx(0.5)
    assert report["gripper_close_confusion"] == {
        "true_positive_close": 1,
        "false_positive_close": 1,
        "false_negative_close": 1,
        "true_negative_open": 1,
        "target_close_count": 2,
        "predicted_close_count": 2,
    }
