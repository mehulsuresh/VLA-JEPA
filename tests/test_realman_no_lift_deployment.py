import numpy as np
import pytest

from deployment.model_server import checkpoint_utils
from deployment.model_server.checkpoint_utils import build_policy_metadata

from deployment.model_server.server_policy import (
    ActionGuardRetryPolicy,
    BatchedMedianActionEnsemblePolicy,
    _configure_policy_ensemble,
    _continuous_unnormalize,
    _expand_realman_policy_actions,
    _is_realman_data_mix,
    build_argparser,
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
            {"q01": [0.0, 7.0], "q99": [2.0, 7.0]},
            [[-0.5, 7.0]],
        ),
    ],
)
def test_server_diagnostic_inverse_matches_training_without_clipping(mode, stats, expected):
    normalized = np.asarray([[-1.5, 1.5]], dtype=np.float32)

    actual = _continuous_unnormalize(normalized, stats, mode=mode)

    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float32))


def test_action_guard_is_opt_in_and_legacy_disable_flag_still_works(monkeypatch):
    monkeypatch.delenv("POLICY_ENABLE_ACTION_GUARD", raising=False)
    monkeypatch.delenv("POLICY_DISABLE_ACTION_GUARD", raising=False)
    parser = build_argparser()

    assert parser.parse_args([]).enable_action_guard is False
    assert parser.parse_args(["--enable_action_guard"]).enable_action_guard is True
    assert parser.parse_args(["--disable_action_guard"]).enable_action_guard is False

    monkeypatch.setenv("POLICY_ENABLE_ACTION_GUARD", "1")
    assert build_argparser().parse_args([]).enable_action_guard is True
    monkeypatch.setenv("POLICY_DISABLE_ACTION_GUARD", "1")
    assert build_argparser().parse_args([]).enable_action_guard is False


def test_policy_ensemble_repeats_training_aligned_inputs_and_returns_median():
    ensemble_size = 4
    draws = np.asarray(
        [
            [[[-0.4, 0.5], [0.1, 0.9]]],
            [[[0.2, 0.1], [0.7, 0.3]]],
            [[[-0.2, 0.3], [0.5, 0.7]]],
            [[[0.6, -0.1], [0.3, 0.5]]],
        ],
        dtype=np.float32,
    ).reshape(ensemble_size, 2, 2)
    embodied = np.arange(ensemble_size * 6, dtype=np.float32).reshape(ensemble_size, 2, 3)

    class RecordingPolicy:
        def __init__(self):
            self.calls = 0
            self.kwargs = None

        def predict_action(self, **kwargs):
            self.calls += 1
            self.kwargs = kwargs
            return {
                "normalized_actions": draws.copy(),
                "embodied_action_tokens": embodied.copy(),
                "per_draw_label": [f"draw-{index}" for index in range(ensemble_size)],
                "constant": "kept",
            }

    policy = RecordingPolicy()
    ensemble = BatchedMedianActionEnsemblePolicy(policy, ensemble_size=ensemble_size)
    qwen_frames = np.stack(
        [
            np.full((3, 4, 3), 11, dtype=np.uint8),
            np.full((3, 4, 3), 22, dtype=np.uint8),
            np.full((3, 4, 3), 33, dtype=np.uint8),
        ],
        axis=0,
    )[None, ...]
    state = np.arange(19, dtype=np.float32).reshape(1, 1, 19)
    prev_actions = np.arange(5 * 18, dtype=np.float32).reshape(1, 5, 18)

    output = ensemble.predict_action(
        qwen_frames=qwen_frames,
        instructions=["pick up the chain"],
        state=state,
        prev_actions=prev_actions,
        prefix_len=5,
    )

    assert policy.calls == 1
    repeated_frames = policy.kwargs["qwen_frames"]
    assert len(repeated_frames) == ensemble_size
    assert all(sample is repeated_frames[0] for sample in repeated_frames)
    for sample in repeated_frames:
        np.testing.assert_array_equal(sample, qwen_frames[0])
        assert [int(view[0, 0, 0]) for view in sample] == [11, 22, 33]
    assert policy.kwargs["instructions"] == ["pick up the chain"] * ensemble_size
    np.testing.assert_array_equal(
        policy.kwargs["state"],
        np.repeat(state, ensemble_size, axis=0),
    )
    np.testing.assert_array_equal(
        policy.kwargs["prev_actions"],
        np.repeat(prev_actions, ensemble_size, axis=0),
    )
    assert policy.kwargs["prefix_len"] == 5

    assert output["normalized_actions"].shape == (1, 2, 2)
    assert output["normalized_actions"].dtype == np.float32
    np.testing.assert_allclose(
        output["normalized_actions"],
        np.median(draws, axis=0, keepdims=True),
    )
    np.testing.assert_array_equal(output["embodied_action_tokens"], embodied[:1])
    assert output["per_draw_label"] == ["draw-0"]
    assert output["constant"] == "kept"
    assert output["policy_ensemble"] == {
        "size": ensemble_size,
        "reducer": "median",
        "normalized_draw_std": pytest.approx(
            float(np.std(draws.astype(np.float64), axis=0).mean())
        ),
    }


def test_policy_ensemble_preserves_legacy_image_view_order_and_identity():
    views = [object(), object(), object()]

    class RecordingPolicy:
        def predict_action(self, **kwargs):
            self.kwargs = kwargs
            return {"normalized_actions": np.zeros((3, 2, 1), dtype=np.float32)}

    policy = RecordingPolicy()
    ensemble = BatchedMedianActionEnsemblePolicy(policy, ensemble_size=3)
    output = ensemble.predict_action(batch_images=[views], instructions=["task"])

    repeated = policy.kwargs["batch_images"]
    assert len(repeated) == 3
    assert len({id(sample) for sample in repeated}) == 3
    assert all(sample == views for sample in repeated)
    assert all(
        repeated[sample_index][view_index] is views[view_index]
        for sample_index in range(3)
        for view_index in range(3)
    )
    assert output["normalized_actions"].shape == (1, 2, 1)


def test_policy_ensemble_size_one_is_exact_noop_and_metadata_is_explicit():
    sentinel = {"normalized_actions": object()}

    class Policy:
        def predict_action(self, *args, **kwargs):
            self.received = (args, kwargs)
            return sentinel

    policy = Policy()
    ensemble = BatchedMedianActionEnsemblePolicy(policy, ensemble_size=1)
    marker = object()

    assert ensemble.predict_action(marker, untouched=marker) is sentinel
    assert policy.received == ((marker,), {"untouched": marker})

    metadata = {}
    assert _configure_policy_ensemble(policy, ensemble_size=1, metadata=metadata) is policy
    assert metadata["policy_ensemble"] == {
        "enabled": False,
        "size": 1,
        "reducer": "identity",
        "model_calls_per_request": 1,
    }

    wrapped = _configure_policy_ensemble(policy, ensemble_size=4, metadata=metadata)
    assert isinstance(wrapped, BatchedMedianActionEnsemblePolicy)
    assert wrapped.policy is policy
    assert metadata["policy_ensemble"] == {
        "enabled": True,
        "size": 4,
        "reducer": "median",
        "model_calls_per_request": 1,
    }


def test_policy_ensemble_cli_and_environment_require_positive_integer(monkeypatch):
    monkeypatch.delenv("POLICY_ENSEMBLE_SIZE", raising=False)
    assert build_argparser().parse_args([]).policy_ensemble_size == 1
    assert build_argparser().parse_args(["--policy-ensemble-size", "4"]).policy_ensemble_size == 4
    assert build_argparser().parse_args(["--policy_ensemble_size", "3"]).policy_ensemble_size == 3

    monkeypatch.setenv("POLICY_ENSEMBLE_SIZE", "5")
    assert build_argparser().parse_args([]).policy_ensemble_size == 5
    monkeypatch.setenv("POLICY_ENSEMBLE_SIZE", "0")
    with pytest.raises(SystemExit):
        build_argparser().parse_args([])
    with pytest.raises(SystemExit):
        build_argparser().parse_args(["--policy-ensemble-size", "-2"])

    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer|must be positive"):
            BatchedMedianActionEnsemblePolicy(object(), ensemble_size=invalid)


def test_policy_ensemble_fails_closed_for_bad_single_observation_inputs():
    class Policy:
        calls = 0

        def predict_action(self, **kwargs):
            self.calls += 1
            return {"normalized_actions": np.zeros((4, 2, 1), dtype=np.float32)}

    frames = np.zeros((1, 3, 4, 4, 3), dtype=np.uint8)
    valid = {"qwen_frames": frames, "instructions": ["task"]}
    bad_cases = [
        ({**valid, "qwen_frames": np.zeros((2, 3, 4, 4, 3), dtype=np.uint8)}, "batch size 1"),
        ({**valid, "qwen_frames": np.zeros((1, 4, 4, 3), dtype=np.uint8)}, "sample shape"),
        ({**valid, "instructions": ["task", "extra"]}, "batch size 1"),
        ({**valid, "state": np.zeros((2, 1, 19), dtype=np.float32)}, "state must have batch size 1"),
        ({**valid, "prev_actions": np.zeros((1, 18), dtype=np.float32)}, "prev_actions must have rank"),
        (
            {**valid, "batch_images": [[object(), object(), object()]]},
            "exactly one of qwen_frames or batch_images",
        ),
        ({**valid, "batch": [{"qwen_frames": frames[0]}]}, "do not accept prebuilt `batch`"),
    ]

    for kwargs, match in bad_cases:
        policy = Policy()
        ensemble = BatchedMedianActionEnsemblePolicy(policy, ensemble_size=4)
        with pytest.raises(ValueError, match=match):
            ensemble.predict_action(**kwargs)
        assert policy.calls == 0


@pytest.mark.parametrize(
    ("bad_output", "match"),
    [
        (None, "must return a dict"),
        ({}, "missing normalized_actions"),
        ({"normalized_actions": np.zeros((1, 2, 1), dtype=np.float32)}, "expected normalized_actions"),
        ({"normalized_actions": np.zeros((4, 0, 1), dtype=np.float32)}, "empty shape"),
        (
            {"normalized_actions": np.full((4, 2, 1), np.nan, dtype=np.float32)},
            "non-finite",
        ),
        (
            {
                "normalized_actions": np.zeros((4, 2, 1), dtype=np.float32),
                "policy_ensemble": {},
            },
            "reserved key",
        ),
    ],
)
def test_policy_ensemble_fails_closed_for_bad_model_output(bad_output, match):
    class Policy:
        def predict_action(self, **kwargs):
            return bad_output

    ensemble = BatchedMedianActionEnsemblePolicy(Policy(), ensemble_size=4)
    with pytest.raises(ValueError, match=match):
        ensemble.predict_action(
            qwen_frames=np.zeros((1, 3, 4, 4, 3), dtype=np.uint8),
            instructions=["task"],
        )


def test_policy_ensemble_composes_inside_action_guard():
    action_dim = REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT

    class Policy:
        calls = 0

        def predict_action(self, **kwargs):
            self.calls += 1
            return {
                "normalized_actions": np.zeros((3, 50, action_dim), dtype=np.float32)
            }

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
    policy = Policy()
    ensemble = _configure_policy_ensemble(policy, ensemble_size=3, metadata=metadata)
    guard = ActionGuardRetryPolicy(
        ensemble,
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
        min_tail_lift_mean=-1.0,
    )

    output = guard.predict_action(
        qwen_frames=np.zeros((1, 3, 4, 4, 3), dtype=np.uint8),
        instructions=["task"],
    )

    assert guard.policy is ensemble
    assert policy.calls == 1
    assert output["normalized_actions"].shape == (1, 50, action_dim)
    assert output["policy_ensemble"]["size"] == 3
    assert output["action_guard"]["accepted"] is True


def test_policy_metadata_describes_18d_realman_expansion_contract(tmp_path):
    class Policy:
        config = {
            "run_id": "magna-test",
            "datasets": {
                "vla_data": {
                    "data_mix": "magna_source_no_base_no_lift_interventions_v3",
                    "action_type": "absolute_qpos",
                    "resolution_size": 224,
                    "video_resolution_size": 384,
                    "with_state": True,
                }
            },
            "framework": {
                "name": "VLA_JEPA",
                "action_model": {
                    "action_dim": 18,
                    "state_dim": 19,
                    "action_horizon": 50,
                    "future_action_window_size": 49,
                    "num_inference_timesteps": 8,
                    "rtc_training": {
                        "enabled": True,
                        "method": "prefix",
                        "max_delay": 11,
                    },
                },
                "vj2_model": {"num_frames": 8},
            },
        }
        norm_stats = {
            "new_embodiment": {
                "action": {"min": [-1.0] * 18, "max": [1.0] * 18},
                "state": {"min": [-1.0] * 19, "max": [1.0] * 19},
            }
        }

    metadata = build_policy_metadata(Policy(), tmp_path / "model.safetensors")

    assert metadata["policy_action_names"][-2:] == ["head_joint_1_rad", "head_joint_2_rad"]
    assert metadata["robot_action_dim"] == 22
    assert metadata["rtc_inference_contract"] == {
        "training_enabled": True,
        "method": "prefix",
        "max_delay_exclusive": 11,
        "max_prefix_len": 10,
        "action_space": "normalized_policy_action",
        "client_opt_in": True,
    }
    assert metadata["realman_action_contract"] == {
        "version": 1,
        "policy_action_dim": 18,
        "robot_action_dim": 22,
        "omitted_robot_action_indices": [16, 17, 18, 21],
        "base_velocity_source": "zero",
        "lift_source": "measured_state",
    }
    assert metadata["realman_input_contract"] == {
        "version": 1,
        "payload_key": "qwen_frames",
        "camera_order": ["head", "wrist_left", "wrist_right"],
        "frame_shape": [3, 384, 384, 3],
        "frame_size": 384,
        "frame_dtype": "uint8",
        "color_space": "RGB",
        "transport_encoding": "msgpack_ndarray",
        "client_resize": "opencv_inter_linear",
        "model_preprocess": "qwen_tensor_fast_path",
        "model_resolution_size": 224,
        "state_shape": [1, 1, 19],
        "state_dtype": "float32",
        "state_normalized": True,
        "state_normalization_mode": "min_max",
        "state_normalization_clip": False,
    }


def test_realman_metadata_refuses_to_guess_normalization_mode(tmp_path, monkeypatch):
    class Policy:
        config = {
            "datasets": {
                "vla_data": {
                    "data_mix": "magna_source_no_base_no_lift_interventions_v3",
                    "action_type": "absolute_qpos",
                    "resolution_size": 224,
                    "video_resolution_size": 384,
                }
            },
            "framework": {
                "action_model": {"action_dim": 18, "state_dim": 19, "action_horizon": 50},
                "vj2_model": {"num_frames": 8},
            },
        }
        norm_stats = {
            "new_embodiment": {
                "action": {"min": [-1.0] * 18, "max": [1.0] * 18},
                "state": {"min": [-1.0] * 19, "max": [1.0] * 19},
            }
        }

    monkeypatch.setattr(checkpoint_utils, "_infer_norm_mode_hints", lambda _policy: {})

    with pytest.raises(RuntimeError, match="Refusing to guess q99"):
        build_policy_metadata(Policy(), tmp_path / "model.safetensors")
