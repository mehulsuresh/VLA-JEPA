from __future__ import annotations

import json

import numpy as np

from deployment.realman.pipeline import expand_policy_action_to_robot_action
from starVLA.action_representation import (
    REALMAN_18D_ACTION_CONTRACT,
    decode_actions,
    encode_actions,
    normalize_q01_q99_unclipped,
    select_realman_policy_actions,
    select_realman_policy_state,
    split_manifest_sha256_without_statistics_binding,
    unnormalize_q01_q99,
)


def test_realman_source_selection_drops_only_lift_and_base():
    state = np.arange(19, dtype=np.float32)
    action = np.arange(22, dtype=np.float32)

    np.testing.assert_array_equal(select_realman_policy_state(state), state[:18])
    np.testing.assert_array_equal(
        select_realman_policy_actions(action),
        np.concatenate([action[:16], action[19:21]]),
    )


def test_realman_source_selection_drops_lift_and_diagnostic_torques():
    state = np.arange(21, dtype=np.float32)

    np.testing.assert_array_equal(select_realman_policy_state(state), state[:18])


def test_realman_source_selection_rejects_unknown_raw_width():
    with np.testing.assert_raises_regex(ValueError, "supported raw widths"):
        select_realman_policy_state(np.arange(20, dtype=np.float32))


def test_mixed_action_round_trip_and_grippers_remain_absolute():
    rng = np.random.default_rng(7)
    state = rng.normal(size=18).astype(np.float32)
    actions = rng.normal(size=(50, 18)).astype(np.float32)
    actions[:, 7] = np.linspace(0, 1, 50)
    actions[:, 15] = np.linspace(1, 0, 50)

    encoded = encode_actions(actions, state)

    np.testing.assert_array_equal(encoded[:, [7, 15]], actions[:, [7, 15]])
    np.testing.assert_allclose(decode_actions(encoded, state), actions, atol=1e-6)
    assert encoded.shape == (REALMAN_18D_ACTION_CONTRACT.action_horizon, 18)


def test_all_horizon_rows_use_one_chunk_start_anchor():
    state = np.arange(18, dtype=np.float32)
    actions = np.broadcast_to(state, (50, 18)).copy()
    actions[:, 0] += np.arange(50, dtype=np.float32)
    actions[:, [7, 15]] = 0.5

    encoded = encode_actions(actions, state)

    np.testing.assert_array_equal(encoded[:, 0], np.arange(50, dtype=np.float32))
    np.testing.assert_array_equal(encoded[:, [7, 15]], 0.5)


def test_delta_ranges_come_from_paired_values_not_independent_extrema():
    state = np.asarray([10.0, -10.0] + [0.0] * 16, dtype=np.float32)
    actions = np.broadcast_to(state, (50, 18)).copy()
    actions[:, 0] += np.linspace(-0.1, 0.1, 50, dtype=np.float32)
    actions[:, 1] += np.linspace(0.2, -0.2, 50, dtype=np.float32)
    actions[:, [7, 15]] = 1.0

    encoded = encode_actions(actions, state)

    assert np.max(np.abs(encoded[:, 0])) <= 0.100001
    assert np.max(np.abs(encoded[:, 1])) <= 0.200001
    assert np.max(np.abs(encoded[:, :7])) < 1.0


def test_split_hash_excludes_only_the_circular_statistics_binding():
    manifest = {
        "schema_version": 1,
        "datasets": [
            {
                "dataset_name": "magna",
                "train_catalog_sha256": "train-a",
                "action_representation_statistics": {
                    "path": "first.json",
                    "sha256": "a" * 64,
                },
            }
        ],
    }
    rebound = json.loads(json.dumps(manifest))
    rebound["datasets"][0]["action_representation_statistics"] = {
        "path": "second.json",
        "sha256": "b" * 64,
    }
    changed_split = json.loads(json.dumps(manifest))
    changed_split["datasets"][0]["train_catalog_sha256"] = "train-b"

    original_hash = split_manifest_sha256_without_statistics_binding(manifest)

    assert (
        split_manifest_sha256_without_statistics_binding(rebound)
        == original_hash
    )
    assert (
        split_manifest_sha256_without_statistics_binding(changed_split)
        != original_hash
    )


def test_openpi_quantile_normalization_is_unclipped_and_uses_epsilon():
    values = np.asarray([[-2.0, 0.25], [3.0, 0.75]], dtype=np.float32)
    stats = {"q01": [-1.0, 0.0], "q99": [1.0, 1.0]}
    expected = (values - np.asarray(stats["q01"])) / (
        np.asarray(stats["q99"]) - np.asarray(stats["q01"]) + 1e-6
    ) * 2.0 - 1.0

    normalized = normalize_q01_q99_unclipped(values, stats)

    np.testing.assert_allclose(normalized, expected, atol=1e-6)
    assert normalized[0, 0] < -1.0
    assert normalized[1, 0] > 1.0
    np.testing.assert_allclose(
        unnormalize_q01_q99(normalized, stats), values, atol=2e-6
    )


def test_18d_deployment_expansion_zeros_base_and_holds_measured_lift():
    policy = np.arange(18, dtype=np.float32)

    expanded = expand_policy_action_to_robot_action(policy, lift_height_mm=321.0)

    np.testing.assert_array_equal(expanded[:16], policy[:16])
    np.testing.assert_array_equal(expanded[16:19], 0.0)
    np.testing.assert_array_equal(expanded[19:21], policy[16:18])
    assert expanded[21] == 321.0
