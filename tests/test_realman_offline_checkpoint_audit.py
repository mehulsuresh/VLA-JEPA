from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from deployment.realman.audit_checkpoint_offline import (
    EpisodeRef,
    MetricAccumulator,
    PlannedSampleIdentity,
    _image_ablation_plan_fingerprint,
    _paired_bootstrap_improvement,
    _sample_plan_fingerprint,
    _update_method,
    assert_checkpoint_dataset_stats_match,
    assert_server_checkpoint_matches,
    build_paired_arm_target_error_bootstrap,
    build_cli_headline,
    build_horizon_comparison,
    build_per_sample_prefix_metrics,
    build_state_matched_image_donor_map,
    build_subtask_explicit_instruction,
    build_valid_action_mask,
    deterministic_frame_indices,
    enumerate_episodes,
    extract_authoritative_raw_modality_window,
    extract_training_aligned_qwen_frames,
    normalize_values,
    policy_ensemble_chunks,
    predict_clean_and_state_matched_image_shuffle,
    project_realman_state_to_action,
    select_episode_refs,
    unnormalize_values,
    validate_dataset_camera_order,
    validate_manifest_frame_plan,
    validate_manifest_holdout_claim,
    validate_training_aligned_input_contract,
)


def _stats(dim: int) -> dict[str, list[float]]:
    return {
        "min": [-2.0] * dim,
        "max": [2.0] * dim,
        "mean": [0.25] * dim,
        "std": [0.5] * dim,
        "q01": [-1.0] * dim,
        "q99": [1.0] * dim,
        "mask": [True] * dim,
    }


def test_episode_selection_is_deterministic_and_source_order_invariant() -> None:
    first = SimpleNamespace(
        dataset_name="b",
        trajectory_ids=np.array([9, 3]),
        trajectory_lengths=np.array([90, 30]),
    )
    second = SimpleNamespace(
        dataset_name="a",
        trajectory_ids=np.array([8, 2]),
        trajectory_lengths=np.array([80, 20]),
    )
    refs_forward = enumerate_episodes([first, second], seed=17)
    refs_reverse = enumerate_episodes([second, first], seed=17)

    identities_forward = [
        (ref.dataset_name, ref.episode_id)
        for ref in select_episode_refs(refs_forward, num_episodes=3)
    ]
    identities_reverse = [
        (ref.dataset_name, ref.episode_id)
        for ref in select_episode_refs(refs_reverse, num_episodes=3)
    ]
    assert identities_forward == identities_reverse


def test_manifest_selection_never_adds_frames_from_other_episodes() -> None:
    refs = [
        EpisodeRef(0, "data", episode_id, 100, f"{episode_id:064x}")
        for episode_id in range(5)
    ]
    selected = select_episode_refs(
        refs,
        num_episodes=10,
        explicit_episode_ids={"data": [1, 4]},
    )
    assert [(ref.dataset_name, ref.episode_id) for ref in selected] == [
        ("data", 1),
        ("data", 4),
    ]


def test_deterministic_frame_sampling_covers_strata_and_can_take_whole_episode() -> None:
    sampled = deterministic_frame_indices(
        100,
        frames_per_episode=4,
        seed=42,
        dataset_name="data",
        episode_id=5,
    )
    assert sampled == deterministic_frame_indices(
        100,
        frames_per_episode=4,
        seed=42,
        dataset_name="data",
        episode_id=5,
    )
    assert len(sampled) == 4
    assert all(lower <= value < upper for value, lower, upper in zip(sampled, (0, 25, 50, 75), (25, 50, 75, 100)))
    assert deterministic_frame_indices(
        3,
        frames_per_episode=0,
        seed=1,
        dataset_name="data",
        episode_id=0,
    ) == [0, 1, 2]


def test_state_matched_image_donors_are_nearest_eligible_and_order_invariant() -> None:
    target = PlannedSampleIdentity("data", 1, 0)
    adjacent = PlannedSampleIdentity("data", 1, 5)
    distant = PlannedSampleIdentity("data", 1, 30)
    other_episode = PlannedSampleIdentity("data", 2, 0)
    states = {
        target: np.array([0.0, 0.0], dtype=np.float32),
        adjacent: np.array([0.01, 0.01], dtype=np.float32),
        distant: np.array([0.3, 0.3], dtype=np.float32),
        other_episode: np.array([0.1, 0.1], dtype=np.float32),
    }

    donors = build_state_matched_image_donor_map(
        states, min_same_episode_frame_gap=20
    )
    reversed_donors = build_state_matched_image_donor_map(
        dict(reversed(list(states.items()))), min_same_episode_frame_gap=20
    )

    # The closer frame at +5 is deliberately ineligible; the cross-episode
    # state match wins over the eligible but more distant +30 frame.
    assert donors[target].donor == other_episode
    assert donors[target].normalized_state_rms_distance == pytest.approx(0.1)
    assert donors[other_episode].donor == adjacent
    assert donors == reversed_donors
    assert _image_ablation_plan_fingerprint(donors) == (
        _image_ablation_plan_fingerprint(reversed_donors)
    )


def test_state_matched_image_donor_can_fall_back_to_separated_same_episode() -> None:
    first = PlannedSampleIdentity("data", 7, 10)
    second = PlannedSampleIdentity("data", 7, 35)
    donors = build_state_matched_image_donor_map(
        {
            first: np.zeros(3, dtype=np.float32),
            second: np.ones(3, dtype=np.float32),
        },
        min_same_episode_frame_gap=20,
    )
    assert donors[first].donor == second
    assert donors[first].as_dict()["same_episode_frame_gap"] == 25


def test_state_matched_image_donor_fails_without_effective_candidate() -> None:
    only = PlannedSampleIdentity("data", 1, 0)
    with pytest.raises(ValueError, match="at least two planned observations"):
        build_state_matched_image_donor_map({only: np.zeros(2)})

    adjacent = PlannedSampleIdentity("data", 1, 3)
    with pytest.raises(ValueError, match="No eligible image donor"):
        build_state_matched_image_donor_map(
            {only: np.zeros(2), adjacent: np.ones(2)},
            min_same_episode_frame_gap=20,
        )


def test_paired_image_ablation_changes_only_images_and_resets_same_seed() -> None:
    class _Predictor:
        def __init__(self) -> None:
            self.calls = []

        def predict_many(
            self, *, qwen_frames, instruction, state, seed, num_samples
        ):
            self.calls.append(
                {
                    "frames": np.asarray(qwen_frames).copy(),
                    "instruction": instruction,
                    "state": np.asarray(state).copy(),
                    "state_object_id": id(state),
                    "seed": seed,
                    "num_samples": num_samples,
                }
            )
            value = float(np.asarray(qwen_frames, dtype=np.float32).mean())
            return np.full((num_samples, 2, 3), value, dtype=np.float32), 4.0

    predictor = _Predictor()
    target_frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    donor_frames = np.full_like(target_frames, 10)
    state = np.arange(19, dtype=np.float32)[None, :]
    paired = predict_clean_and_state_matched_image_shuffle(
        predictor,
        target_qwen_frames=target_frames,
        donor_qwen_frames=donor_frames,
        instruction="do the task",
        state=state,
        seed=1234,
        num_samples=2,
    )

    assert len(predictor.calls) == 3
    clean_call, repeated_clean_call, shuffled_call = predictor.calls
    assert {
        call["instruction"] for call in predictor.calls
    } == {"do the task"}
    assert {call["seed"] for call in predictor.calls} == {1234}
    assert {call["num_samples"] for call in predictor.calls} == {2}
    assert {call["state_object_id"] for call in predictor.calls} == {id(state)}
    np.testing.assert_array_equal(clean_call["state"], repeated_clean_call["state"])
    np.testing.assert_array_equal(clean_call["state"], shuffled_call["state"])
    np.testing.assert_array_equal(clean_call["frames"], target_frames)
    np.testing.assert_array_equal(repeated_clean_call["frames"], target_frames)
    np.testing.assert_array_equal(shuffled_call["frames"], donor_frames)
    np.testing.assert_array_equal(paired.clean_draws, np.zeros((2, 2, 3)))
    np.testing.assert_array_equal(paired.shuffled_draws, np.full((2, 2, 3), 10.0))
    assert paired.target_frames_sha256 != paired.donor_frames_sha256
    assert paired.pixel_mean_absolute_difference == pytest.approx(10.0)
    assert paired.clean_repeat_max_abs_difference == 0.0
    assert paired.clean_repeat_mean_abs_difference == 0.0


def test_paired_image_ablation_rejects_identical_or_shape_mismatched_payloads() -> None:
    predictor = SimpleNamespace(predict_many=lambda **_: (np.zeros((1, 1, 1)), 0.0))
    frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    kwargs = {
        "predictor": predictor,
        "target_qwen_frames": frames,
        "instruction": "task",
        "state": np.zeros((1, 19), dtype=np.float32),
        "seed": 1,
        "num_samples": 1,
    }
    with pytest.raises(ValueError, match="byte-identical"):
        predict_clean_and_state_matched_image_shuffle(
            donor_qwen_frames=frames.copy(), **kwargs
        )
    with pytest.raises(ValueError, match="different shapes"):
        predict_clean_and_state_matched_image_shuffle(
            donor_qwen_frames=np.zeros((3, 2, 2, 3), dtype=np.uint8), **kwargs
        )


def test_paired_image_ablation_fails_when_same_seed_repeat_is_nondeterministic() -> None:
    class _NondeterministicPredictor:
        def __init__(self) -> None:
            self.call = 0

        def predict_many(self, **kwargs):
            del kwargs
            self.call += 1
            return np.full((1, 2, 3), self.call * 1e-3, dtype=np.float32), 0.0

    with pytest.raises(RuntimeError, match="Same-image, same-seed"):
        predict_clean_and_state_matched_image_shuffle(
            _NondeterministicPredictor(),
            target_qwen_frames=np.zeros((3, 2, 2, 3), dtype=np.uint8),
            donor_qwen_frames=np.ones((3, 2, 2, 3), dtype=np.uint8),
            instruction="task",
            state=np.zeros((1, 19), dtype=np.float32),
            seed=1,
            num_samples=1,
        )


@pytest.mark.parametrize("bad_kind", ["shape", "nonfinite"])
def test_paired_image_ablation_rejects_malformed_policy_draws(bad_kind) -> None:
    class _MalformedPredictor:
        def __init__(self) -> None:
            self.call = 0

        def predict_many(self, **kwargs):
            del kwargs
            self.call += 1
            if bad_kind == "shape" and self.call == 3:
                return np.zeros((1, 3, 3), dtype=np.float32), 0.0
            output = np.zeros((1, 2, 3), dtype=np.float32)
            if bad_kind == "nonfinite" and self.call == 3:
                output[0, 0, 0] = np.nan
            return output, 0.0

    expected = "draw shapes differ" if bad_kind == "shape" else "NaN or Infinity"
    with pytest.raises(ValueError, match=expected):
        predict_clean_and_state_matched_image_shuffle(
            _MalformedPredictor(),
            target_qwen_frames=np.zeros((3, 2, 2, 3), dtype=np.uint8),
            donor_qwen_frames=np.ones((3, 2, 2, 3), dtype=np.uint8),
            instruction="task",
            state=np.zeros((1, 19), dtype=np.float32),
            seed=1,
            num_samples=1,
        )


def _paired_arm_error_record(
    index: int,
    *,
    clean_h1: float,
    shuffled_h1: float,
    clean_h5: float,
    shuffled_h5: float,
    h1_count: int = 14,
    h5_count: int = 70,
) -> dict:
    def _prefixes(h1: float, h5: float) -> dict:
        return {
            "1": {"count": h1_count, "raw_mae": h1},
            "5": {"count": h5_count, "raw_mae": h5},
        }

    return {
        "dataset_name": "eval",
        "episode_id": index // 2,
        "requested_frame_index": index,
        "arm_errors_by_prefix": {
            "policy_ensemble_median": _prefixes(clean_h1, clean_h5),
            "policy_state_matched_image_shuffle_ensemble_median": _prefixes(
                shuffled_h1, shuffled_h5
            ),
        },
    }


def test_paired_arm_target_error_bootstrap_is_observation_paired_and_deterministic() -> None:
    records = [
        _paired_arm_error_record(
            index,
            clean_h1=1.0,
            shuffled_h1=1.0 + improvement,
            clean_h5=2.0,
            shuffled_h5=2.0 + 0.5 * improvement,
            # Deliberately unequal counts across observations: each observation
            # must still receive one bootstrap vote rather than one per element.
            h1_count=14 * (index + 1),
            h5_count=70 * (index + 1),
        )
        for index, improvement in enumerate((1.0, 2.0, 3.0, 4.0))
    ]

    first = build_paired_arm_target_error_bootstrap(
        records, eval_seed=42, num_resamples=2_000
    )
    repeated = build_paired_arm_target_error_bootstrap(
        records, eval_seed=42, num_resamples=2_000
    )
    assert first == repeated
    assert first["observation_weighting"] == (
        "one_equal_weight_pair_per_evaluated_observation"
    )

    h1 = first["by_prefix"]["1"]
    h5 = first["by_prefix"]["5"]
    assert h1["observation_count"] == 4
    assert h1["paired_improvement"]["mean"]["estimate"] == pytest.approx(2.5)
    assert h1["paired_improvement"]["median"]["estimate"] == pytest.approx(2.5)
    assert h5["paired_improvement"]["mean"]["estimate"] == pytest.approx(1.25)
    for prefix_report in (h1, h5):
        assert prefix_report["paired_improvement"]["mean"][
            "confidence_interval_lower_bound_gt_zero"
        ]
        assert prefix_report["paired_improvement"]["median"][
            "confidence_interval_lower_bound_gt_zero"
        ]
        assert prefix_report["bootstrap"]["resampling_unit"] == "observation_pair"
        assert prefix_report["bootstrap"]["eval_seed"] == 42
        assert prefix_report["bootstrap"]["seed"] != 42
    assert h1["bootstrap"]["seed"] != h5["bootstrap"]["seed"]
    gate = first["confidence_interval_lower_bound_gt_zero_gate"]
    assert gate["required_h1_and_h5_mean_improvement_pass"]
    assert gate["required_h1_and_h5_median_improvement_pass"]


def test_paired_arm_target_error_bootstrap_fails_closed_on_empty_or_mask_mismatch() -> None:
    with pytest.raises(ValueError, match="no evaluated observations"):
        build_paired_arm_target_error_bootstrap([], eval_seed=42)

    zero_mask = [
        _paired_arm_error_record(
            index,
            clean_h1=1.0,
            shuffled_h1=2.0,
            clean_h5=1.0,
            shuffled_h5=2.0,
            h5_count=0 if index == 1 else 70,
        )
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="has no valid arm targets"):
        build_paired_arm_target_error_bootstrap(zero_mask, eval_seed=42)

    mismatched = [
        _paired_arm_error_record(
            index,
            clean_h1=1.0,
            shuffled_h1=2.0,
            clean_h5=1.0,
            shuffled_h5=2.0,
        )
        for index in range(2)
    ]
    mismatched[1]["arm_errors_by_prefix"][
        "policy_state_matched_image_shuffle_ensemble_median"
    ]["5"]["count"] = 69
    with pytest.raises(ValueError, match="mask counts differ"):
        build_paired_arm_target_error_bootstrap(mismatched, eval_seed=42)


def test_paired_bootstrap_caps_sampled_index_elements_per_batch() -> None:
    report = _paired_bootstrap_improvement(
        clean_errors=[1.0, 1.0, 1.0, 1.0],
        shuffled_errors=[2.0, 3.0, 4.0, 5.0],
        eval_seed=42,
        prefix=1,
        num_resamples=5,
        max_index_elements_per_batch=8,
    )
    bootstrap = report["bootstrap"]
    assert bootstrap["max_index_elements_per_batch"] == 8
    assert bootstrap["max_index_bytes_per_batch"] == 64
    assert bootstrap["resamples_per_batch"] == 2
    assert bootstrap["actual_max_index_elements_per_batch"] == 8

    with pytest.raises(ValueError, match="population exceeds"):
        _paired_bootstrap_improvement(
            clean_errors=[1.0, 1.0, 1.0, 1.0],
            shuffled_errors=[2.0, 3.0, 4.0, 5.0],
            eval_seed=42,
            prefix=1,
            num_resamples=5,
            max_index_elements_per_batch=3,
        )


def test_manifest_frame_plan_is_exact_canonical_and_fingerprint_stable() -> None:
    selected = [
        EpisodeRef(0, "late_friday", 1605, 12, "a" * 64),
        EpisodeRef(0, "late_friday", 1607, 20, "b" * 64),
    ]
    payload = {
        "datasets": {"late_friday": [1605, 1607]},
        "frames": {
            "late_friday": {
                "1605": [9, 1, 5],
                "1607": [19, 0],
            }
        },
    }
    frame_plan = validate_manifest_frame_plan(payload, selected=selected)
    assert frame_plan == {
        ("late_friday", 1605): [1, 5, 9],
        ("late_friday", 1607): [0, 19],
    }
    assert _sample_plan_fingerprint(frame_plan) == _sample_plan_fingerprint(
        {
            ("late_friday", 1607): [19, 0],
            ("late_friday", 1605): [5, 9, 1],
        }
    )
    assert validate_manifest_frame_plan(
        {"datasets": {"late_friday": [1605, 1607]}},
        selected=selected,
    ) is None


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ({"late_friday": {"1605": [1]}}, "does not exactly match selected episodes"),
        (
            {"late_friday": {"1605": [1], "1607": [2], "1608": [3]}},
            "not selected",
        ),
        (
            {"late_friday": {"1605": [1], "1607": [2]}, "other": {}},
            "datasets do not exactly match",
        ),
        (
            {"late_friday": {"1605": [1], "1607": [2, 2]}},
            "duplicate frame indices",
        ),
        (
            {"late_friday": {"1605": [1], "1607": [20]}},
            "outside",
        ),
        (
            {"late_friday": {"1605": [1], "1607": [True]}},
            "must be an integer",
        ),
        (
            {"late_friday": {"1605": [1], "1607": []}},
            "at least one frame index",
        ),
        (
            {"late_friday": {"01605": [1], "1607": [2]}},
            "canonical integer-string form",
        ),
    ],
)
def test_manifest_frame_plan_rejects_incomplete_or_invalid_bindings(
    frames, message
) -> None:
    selected = [
        EpisodeRef(0, "late_friday", 1605, 12, "a" * 64),
        EpisodeRef(0, "late_friday", 1607, 20, "b" * 64),
    ]
    with pytest.raises(ValueError, match=message):
        validate_manifest_frame_plan({"frames": frames}, selected=selected)


def test_qwen_frame_extraction_uses_current_context_not_future_frames() -> None:
    video = np.zeros((3, 8, 2, 2, 3), dtype=np.uint8)
    for timestep in range(8):
        video[:, timestep] = timestep
    frames = extract_training_aligned_qwen_frames(
        {"video_compact": video}, video_target_shift_steps=2
    )
    assert frames.shape == (3, 2, 2, 3)
    assert np.all(frames == 5)


def test_training_aligned_contract_checks_transport_shape_and_rgb() -> None:
    metadata = {
        "realman_input_contract": {
            "payload_key": "qwen_frames",
            "color_space": "RGB",
            "frame_shape": [3, 4, 4, 3],
            "state_shape": [1, 1, 19],
        }
    }
    validate_training_aligned_input_contract(
        qwen_frames=np.zeros((3, 4, 4, 3), dtype=np.uint8),
        state=np.zeros((1, 19), dtype=np.float32),
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="frame shape"):
        validate_training_aligned_input_contract(
            qwen_frames=np.zeros((3, 2, 2, 3), dtype=np.uint8),
            state=np.zeros((1, 19), dtype=np.float32),
            metadata=metadata,
        )


def test_dataset_camera_order_is_checked_by_deployment_semantics() -> None:
    metadata = {
        "realman_input_contract": {
            "camera_order": ["head", "wrist_left", "wrist_right"]
        }
    }
    aligned = SimpleNamespace(
        modality_keys={
            "video": [
                "video.base_view",
                "video.left_wrist",
                "video.right_wrist",
            ]
        }
    )
    validate_dataset_camera_order(aligned, metadata)

    swapped = SimpleNamespace(
        modality_keys={
            "video": [
                "video.base_view",
                "video.right_wrist",
                "video.left_wrist",
            ]
        }
    )
    with pytest.raises(ValueError, match="camera semantics"):
        validate_dataset_camera_order(swapped, metadata)


def test_authoritative_raw_modality_extraction_uses_source_rows_slices_and_order() -> None:
    class _Series:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class _SelectedRows:
        def __init__(self, columns, row_indices):
            self._columns = columns
            self._row_indices = np.asarray(row_indices, dtype=np.int64)
            self.columns = tuple(columns)

        def __getitem__(self, key):
            return _Series([self._columns[key][index] for index in self._row_indices])

    class _ILoc:
        def __init__(self, table):
            self._table = table

        def __getitem__(self, row_indices):
            return _SelectedRows(self._table._columns, row_indices)

    class _FrameTable:
        def __init__(self, columns):
            self._columns = columns
            self.iloc = _ILoc(self)

        def __len__(self):
            return len(next(iter(self._columns.values())))

    raw_rows = [
        np.array([10.0, 11.0, 12.0, 999.0], dtype=np.float64),
        np.array([20.0, 21.0, 22.0, 999.0], dtype=np.float64),
    ]
    dataset = SimpleNamespace(
        curr_traj_data=_FrameTable({"observation.robot_action": raw_rows}),
        modality_keys={"action": ["action.arm", "action.gripper"]},
        delta_indices={
            "action.arm": np.array([-1, 0, 1]),
            "action.gripper": np.array([-1, 0, 1]),
        },
        data_cfg={
            "modality_metadata_overrides": {
                "action": {
                    "arm": {
                        "original_key": "observation.robot_action",
                        "start": 0,
                        "end": 2,
                    },
                    "gripper": {
                        "original_key": "observation.robot_action",
                        "start": 2,
                        "end": 3,
                    },
                }
            }
        },
    )
    extracted = extract_authoritative_raw_modality_window(
        dataset, modality="action", frame_index=0
    )
    assert extracted.dtype == np.float32
    np.testing.assert_array_equal(
        extracted,
        np.array(
            [
                [10.0, 11.0, 12.0],
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ],
            dtype=np.float32,
        ),
    )


def test_subtask_explicit_prompt_uses_dataset_label_and_exact_separator() -> None:
    labels = {
        2: "reach into the bin for the chain",
        0: "__unlabeled__",
    }
    dataset = SimpleNamespace(
        data_cfg={
            "task_id_prompt_separator": " <stage> ",
            "subtask_prompt_ignored_labels": ["__unlabeled__"],
        },
        _subtask_label_for_index=lambda value: labels.get(int(np.asarray(value).item())),
    )
    instruction, label = build_subtask_explicit_instruction(
        dataset,
        {"subtask_index": np.asarray(2, dtype=np.int64)},
        deployment_instruction="complete the chain task",
    )
    assert label == "reach into the bin for the chain"
    assert instruction == (
        "complete the chain task <stage> reach into the bin for the chain"
    )

    with pytest.raises(ValueError, match=r"requires sample\['subtask_index'\]"):
        build_subtask_explicit_instruction(
            dataset, {}, deployment_instruction="complete the chain task"
        )
    with pytest.raises(ValueError, match="ignored/unlabeled"):
        build_subtask_explicit_instruction(
            dataset,
            {"subtask_index": 0},
            deployment_instruction="complete the chain task",
        )
    with pytest.raises(ValueError, match="could not resolve"):
        build_subtask_explicit_instruction(
            dataset,
            {"subtask_index": 99},
            deployment_instruction="complete the chain task",
        )


def test_action_mask_combines_dimension_validity_padding_and_finiteness() -> None:
    action = np.zeros((4, 3), dtype=np.float32)
    action[1, 1] = np.nan
    sample = {
        "action": action,
        "action_mask": np.array(
            [[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool
        ),
        "action_is_pad": np.array([False, False, True, True]),
    }
    valid = build_valid_action_mask(sample, action.shape)
    np.testing.assert_array_equal(
        valid,
        np.array(
            [[1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=bool
        ),
    )


def test_metric_accumulator_excludes_masked_suffix_and_reports_gripper_accuracy() -> None:
    horizon, action_dim = 3, 18
    target = np.zeros((horizon, action_dim), dtype=np.float32)
    prediction = np.zeros_like(target)
    target[:, 7] = [1.0, 1.0, 0.0]
    target[:, 15] = [0.0, 1.0, 1.0]
    prediction[:, 7] = [1.0, 0.0, 100.0]
    prediction[:, 15] = [0.0, 1.0, -100.0]
    valid = np.ones_like(target, dtype=bool)
    valid[2] = False

    accumulator = MetricAccumulator(
        action_dim=action_dim,
        horizon=horizon,
        gripper_thresholds=np.full(action_dim, 0.5, dtype=np.float32),
    )
    accumulator.update(
        prediction_normalized=prediction,
        prediction_raw=prediction,
        target_normalized=target,
        target_raw=target,
        valid_mask=valid,
    )
    report = accumulator.finalize()
    gripper = report["aggregate"]["gripper"]
    assert gripper["count"] == 4
    assert gripper["classification"]["accuracy"] == pytest.approx(0.75)
    assert report["by_horizon"][2]["groups"]["gripper"]["count"] == 0
    assert report["by_horizon"][2]["groups"]["gripper"]["raw_mae"] is None


def test_per_sample_prefix_metrics_retain_gripper_mae_and_classification() -> None:
    horizon, action_dim = 50, 18
    target = np.zeros((horizon, action_dim), dtype=np.float32)
    target[:, 7] = 1.0
    prediction = target.copy()
    prediction[0, 7] = 0.0
    valid = np.ones_like(target, dtype=bool)
    predictions = {
        "policy_ensemble_median": (prediction, prediction, valid),
    }

    metrics = build_per_sample_prefix_metrics(
        predictions=predictions,
        target_normalized=target,
        target_raw=target,
        arm_dimensions=tuple(range(7)) + tuple(range(8, 15)),
        gripper_dimensions=(7, 15),
        gripper_thresholds=np.full(action_dim, 0.5, dtype=np.float32),
    )

    assert set(metrics) == {
        "arm_errors_by_prefix",
        "gripper_metrics_by_prefix",
    }
    gripper = metrics["gripper_metrics_by_prefix"]["policy_ensemble_median"]
    assert list(gripper) == ["1", "5", "10", "20", "50"]
    assert gripper["1"]["count"] == 2
    assert gripper["1"]["raw_mae"] == pytest.approx(0.5)
    assert gripper["1"]["classification"]["accuracy"] == pytest.approx(0.5)
    assert gripper["1"]["classification"]["false_negative"] == 1
    assert gripper["5"]["raw_mae"] == pytest.approx(0.1)
    assert gripper["5"]["classification"]["accuracy"] == pytest.approx(0.9)
    assert gripper["50"]["raw_mae"] == pytest.approx(0.01)
    assert gripper["50"]["classification"]["accuracy"] == pytest.approx(0.99)


def test_policy_ensemble_reports_mean_median_and_median_h0_repeat() -> None:
    draws = np.array(
        [
            [[0.0, 1.0], [4.0, 5.0]],
            [[2.0, 3.0], [6.0, 7.0]],
            [[10.0, 9.0], [8.0, 9.0]],
        ],
        dtype=np.float32,
    )
    chunks = policy_ensemble_chunks(draws)
    np.testing.assert_allclose(
        chunks["policy_ensemble_mean"], np.mean(draws, axis=0)
    )
    np.testing.assert_allclose(
        chunks["policy_ensemble_median"], np.median(draws, axis=0)
    )
    np.testing.assert_allclose(
        chunks["policy_median_h0_repeat"], [[2.0, 3.0], [2.0, 3.0]]
    )


def test_horizon_comparison_exposes_raw_mae_delta_and_ratio() -> None:
    candidate = MetricAccumulator(
        action_dim=18,
        horizon=1,
        gripper_thresholds=np.full(18, 0.5, dtype=np.float32),
    )
    baseline = MetricAccumulator(
        action_dim=18,
        horizon=1,
        gripper_thresholds=np.full(18, 0.5, dtype=np.float32),
    )
    target = np.zeros((1, 18), dtype=np.float32)
    valid = np.ones_like(target, dtype=bool)
    for accumulator, error in ((candidate, 1.0), (baseline, 2.0)):
        prediction = np.full_like(target, error)
        accumulator.update(
            prediction_normalized=prediction,
            prediction_raw=prediction,
            target_normalized=target,
            target_raw=target,
            valid_mask=valid,
        )
    comparison = build_horizon_comparison(candidate.finalize(), baseline.finalize())
    arm = comparison["by_horizon"][0]["groups"]["arm"]["raw_mae"]
    assert arm["candidate"] == 1.0
    assert arm["baseline"] == 2.0
    assert arm["delta"] == -1.0
    assert arm["ratio"] == 0.5
    assert arm["relative_improvement"] == 0.5


def test_horizon_comparison_rejects_unpaired_metric_counts() -> None:
    candidate = MetricAccumulator(
        action_dim=18,
        horizon=1,
        gripper_thresholds=np.full(18, 0.5, dtype=np.float32),
    )
    baseline = MetricAccumulator(
        action_dim=18,
        horizon=1,
        gripper_thresholds=np.full(18, 0.5, dtype=np.float32),
    )
    target = np.zeros((1, 18), dtype=np.float32)
    prediction = np.ones_like(target)
    candidate.update(
        prediction_normalized=prediction,
        prediction_raw=prediction,
        target_normalized=target,
        target_raw=target,
        valid_mask=np.ones_like(target, dtype=bool),
    )
    baseline_mask = np.ones_like(target, dtype=bool)
    baseline_mask[0, 0] = False
    baseline.update(
        prediction_normalized=prediction,
        prediction_raw=prediction,
        target_normalized=target,
        target_raw=target,
        valid_mask=baseline_mask,
    )

    with pytest.raises(ValueError, match="different populations"):
        build_horizon_comparison(candidate.finalize(), baseline.finalize())


def test_gripper_accuracy_comparison_uses_higher_is_better_semantics() -> None:
    candidate = MetricAccumulator(
        action_dim=18,
        horizon=1,
        gripper_thresholds=np.full(18, 0.5, dtype=np.float32),
    )
    baseline = MetricAccumulator(
        action_dim=18,
        horizon=1,
        gripper_thresholds=np.full(18, 0.5, dtype=np.float32),
    )
    target = np.zeros((1, 18), dtype=np.float32)
    target[0, 7] = 1.0
    candidate_prediction = target.copy()  # Both grippers classified correctly.
    baseline_prediction = target.copy()
    baseline_prediction[0, 7] = 0.0  # One of two grippers classified incorrectly.
    valid = np.ones_like(target, dtype=bool)
    for accumulator, prediction in (
        (candidate, candidate_prediction),
        (baseline, baseline_prediction),
    ):
        accumulator.update(
            prediction_normalized=prediction,
            prediction_raw=prediction,
            target_normalized=target,
            target_raw=target,
            valid_mask=valid,
        )

    comparison = build_horizon_comparison(
        candidate.finalize(), baseline.finalize()
    )
    accuracy = comparison["aggregate"]["gripper"]["classification_accuracy"]
    assert accuracy["candidate"] == 1.0
    assert accuracy["baseline"] == 0.5
    assert accuracy["delta"] == 0.5
    assert accuracy["ratio"] == 2.0
    assert accuracy["relative_improvement"] == 1.0


@pytest.mark.parametrize("mode", ["min_max", "q99", "mean_std"])
def test_normalization_round_trip(mode: str) -> None:
    stats = _stats(3)
    values = np.array([[-0.5, 0.0, 0.5]], dtype=np.float32)
    normalized = normalize_values(values, stats, mode=mode)
    recovered = unnormalize_values(normalized, stats, mode=mode)
    np.testing.assert_allclose(recovered, values, atol=1e-6)


def test_rollout_comparable_policy_inverse_stays_unclipped() -> None:
    stats = _stats(18)
    raw_target_or_state = np.array(
        [[4.0, -4.0] + [0.0] * 16], dtype=np.float32
    )
    normalized = normalize_values(raw_target_or_state, stats, mode="min_max")
    assert normalized[0, 0] == 2.0
    assert normalized[0, 1] == -2.0
    np.testing.assert_allclose(
        unnormalize_values(normalized, stats, mode="min_max", clip=False),
        raw_target_or_state,
    )
    np.testing.assert_allclose(
        unnormalize_values(normalized, stats, mode="min_max", clip=True)[0, :2],
        [2.0, -2.0],
    )

    accumulator = MetricAccumulator(
        action_dim=18,
        horizon=1,
        gripper_thresholds=np.zeros(18, dtype=np.float32),
    )
    _update_method(
        accumulator,
        prediction_normalized=normalized,
        action_stats=stats,
        action_mode="min_max",
        target_normalized=np.zeros_like(normalized),
        target_raw=np.zeros_like(normalized),
        valid_mask=np.ones_like(normalized, dtype=bool),
    )
    # Rollout-comparable policy metrics use the same unbounded affine inverse as
    # the RealMan server, so extrapolation remains visible in raw-space error.
    assert accumulator.finalize()["aggregate"]["all"]["raw_mae"] == pytest.approx(
        8.0 / 18.0
    )


def test_q99_inverse_matches_training_for_zero_range_even_with_false_mask() -> None:
    stats = {
        "q01": [0.0, 7.0],
        "q99": [2.0, 7.0],
        "mask": [True, False],
    }
    normalized = np.asarray([[-1.5, 1.5]], dtype=np.float32)

    np.testing.assert_allclose(
        unnormalize_values(normalized, stats, mode="q99"),
        [[-0.5, 7.0]],
    )
    np.testing.assert_allclose(
        unnormalize_values(normalized, stats, mode="q99", clip=True),
        [[0.0, 7.0]],
    )


def test_current_state_hold_projection_matches_realman_18d_layout() -> None:
    state = np.arange(19, dtype=np.float32)
    np.testing.assert_array_equal(
        project_realman_state_to_action(state, action_dim=18), state[:18]
    )
    expanded = project_realman_state_to_action(state, action_dim=22)
    np.testing.assert_array_equal(expanded[:16], state[:16])
    np.testing.assert_array_equal(expanded[16:19], np.zeros(3))
    np.testing.assert_array_equal(expanded[19:22], state[16:19])


def test_checkpoint_dataset_stats_mismatch_fails_closed() -> None:
    checkpoint = _stats(2)
    dataset = _stats(2)
    dataset["max"][1] = 3.0
    with pytest.raises(ValueError, match="statistics differ"):
        assert_checkpoint_dataset_stats_match(
            checkpoint, dataset, modality="action"
        )


def test_server_checkpoint_must_exactly_match_requested_artifact(tmp_path) -> None:
    requested_dir = tmp_path / "steps_62500"
    requested_dir.mkdir()
    requested_artifact = requested_dir / "pytorch_model.pt"
    requested_artifact.write_bytes(b"requested")
    other_dir = tmp_path / "steps_57500"
    other_dir.mkdir()
    other_artifact = other_dir / "pytorch_model.pt"
    other_artifact.write_bytes(b"other")

    assert assert_server_checkpoint_matches(
        requested_dir, {"checkpoint_path": str(requested_artifact)}
    ) == requested_artifact.resolve()
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        assert_server_checkpoint_matches(
            requested_dir, {"checkpoint_path": str(other_artifact)}
        )
    with pytest.raises(ValueError, match="does not identify its checkpoint"):
        assert_server_checkpoint_matches(requested_dir, {})


def test_heldout_manifest_refuses_checkpoint_trained_with_load_all_data(
    tmp_path,
) -> None:
    cfg = SimpleNamespace(
        datasets=SimpleNamespace(
            vla_data={"load_all_data_for_training": True}
        )
    )
    with pytest.raises(ValueError, match="load_all_data_for_training=true"):
        validate_manifest_holdout_claim(
            {"excluded_from_training": True},
            cfg=cfg,
            metadata={},
            config_path=tmp_path / "config.yaml",
            episode_catalog_sha256="catalog",
            selected=[],
        )


def test_cli_headline_includes_fixed_arm_prefixes_vs_hold_and_plan_hash(
    tmp_path,
) -> None:
    def _report_for_error(error: float) -> dict:
        accumulator = MetricAccumulator(
            action_dim=18,
            horizon=50,
            gripper_thresholds=np.full(18, 0.5, dtype=np.float32),
        )
        target = np.zeros((50, 18), dtype=np.float32)
        prediction = np.full_like(target, error)
        accumulator.update(
            prediction_normalized=prediction,
            prediction_raw=prediction,
            target_normalized=target,
            target_raw=target,
            valid_mask=np.ones_like(target, dtype=bool),
        )
        return accumulator.finalize()

    policy_report = _report_for_error(1.0)
    hold_report = _report_for_error(2.0)
    report = {
        "episode_split": {
            "semantic_label": "deterministic_episode_regression_subset",
            "selected_episodes": [{"episode_id": 1}],
        },
        "sampling": {
            "evaluated_frames": 5,
            "sample_plan_sha256": "abc123",
        },
        "metrics": {
            "policy": policy_report,
            "policy_ensemble_mean": policy_report,
            "policy_ensemble_median": policy_report,
            "current_state_hold": hold_report,
        },
    }
    report_path = tmp_path / "report.json"
    headline = build_cli_headline(report, report_path=report_path)

    assert headline["sample_plan_sha256"] == "abc123"
    prefixes = headline["ensemble_median_vs_hold_arm_raw_mae_by_prefix"]
    assert list(prefixes) == ["1", "5", "10", "20", "50"]
    for comparison in prefixes.values():
        assert comparison["candidate"] == 1.0
        assert comparison["baseline"] == 2.0
        assert comparison["relative_improvement"] == 0.5


def test_cli_headline_surfaces_state_matched_image_ablation_sensitivity(
    tmp_path,
) -> None:
    def _report_for_error(error: float) -> dict:
        accumulator = MetricAccumulator(
            action_dim=18,
            horizon=50,
            gripper_thresholds=np.full(18, 0.5, dtype=np.float32),
        )
        target = np.zeros((50, 18), dtype=np.float32)
        prediction = np.full_like(target, error)
        accumulator.update(
            prediction_normalized=prediction,
            prediction_raw=prediction,
            target_normalized=target,
            target_raw=target,
            valid_mask=np.ones_like(target, dtype=bool),
        )
        return accumulator.finalize()

    clean = _report_for_error(1.0)
    shuffled = _report_for_error(2.0)
    output_delta = _report_for_error(0.25)
    bootstrap_by_prefix = {
        "1": {
            "paired_improvement": {
                "mean": {
                    "estimate": 1.0,
                    "confidence_interval_lower_bound_gt_zero": True,
                }
            }
        },
        "5": {
            "paired_improvement": {
                "mean": {
                    "estimate": 1.0,
                    "confidence_interval_lower_bound_gt_zero": True,
                }
            }
        },
    }
    report = {
        "episode_split": {
            "semantic_label": "held_out_from_training",
            "selected_episodes": [{"episode_id": 1}],
        },
        "sampling": {"evaluated_frames": 4, "sample_plan_sha256": "samples"},
        "metrics": {
            "policy": clean,
            "policy_ensemble_mean": clean,
            "policy_ensemble_median": clean,
            "policy_state_matched_image_shuffle_ensemble_median": shuffled,
            "current_state_hold": shuffled,
        },
        "image_ablation": {
            "mode": "state_matched_shuffle",
            "donor_plan_sha256": "donors",
            "paired_policy_output_delta": output_delta,
            "paired_arm_target_error_bootstrap": {
                "by_prefix": bootstrap_by_prefix
            },
        },
    }

    headline = build_cli_headline(report, report_path=tmp_path / "report.json")
    h1 = headline[
        "ensemble_median_vs_state_matched_image_shuffle_arm_raw_mae_by_prefix"
    ]["1"]
    assert h1["candidate"] == 1.0
    assert h1["baseline"] == 2.0
    assert h1["relative_improvement"] == 0.5
    assert headline["same_seed_image_shuffle_output_delta_arm_raw_mae_h1"] == 0.25
    assert headline["image_ablation_donor_plan_sha256"] == "donors"
    assert headline["paired_arm_target_error_bootstrap_by_prefix"] == (
        bootstrap_by_prefix
    )
