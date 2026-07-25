import unittest

import numpy as np

from starVLA.action_representation import (
    Q01_Q99_UNCLIPPED,
    REALMAN_18D_ACTION_CONTRACT,
)

from deployment.realman.pipeline import (
    JOINT_DELTA_GRIPPER_ABSOLUTE,
    REALMAN_ACTION_DIM,
    REALMAN_POLICY_ACTION_DIM_NO_BASE,
    REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT,
    REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT,
    REALMAN_SOURCE_STATE_DIM,
    REALMAN_STATE_DIM,
    build_policy_payload,
    expand_policy_action_to_robot_action,
    realman_omitted_robot_action_indices,
    realman_absolute_actions_to_policy_representation,
    realman_policy_actions_to_absolute,
    realman_policy_action_names,
    split_action_vector,
    validate_realman_policy_payload,
    validate_realman_server_metadata,
)
from deployment.realman.vr_teleop_bridge import (
    model_compatible_observation,
    vector_from_action_payload,
)


class RealmanPipelineTest(unittest.TestCase):
    def test_split_action_vector(self):
        action = np.arange(REALMAN_ACTION_DIM, dtype=np.float32)
        split = split_action_vector(action)
        np.testing.assert_allclose(split["left_arm_joints"], np.arange(0, 7, dtype=np.float32))
        self.assertEqual(split["left_gripper"], 7.0)
        np.testing.assert_allclose(split["right_arm_joints"], np.arange(8, 15, dtype=np.float32))
        self.assertEqual(split["right_gripper"], 15.0)
        self.assertEqual(split["base_velocity"]["linear_x_mps"], 16.0)
        self.assertEqual(split["base_velocity"]["linear_y_mps"], 17.0)
        self.assertEqual(split["base_velocity"]["angular_z_radps"], 18.0)
        np.testing.assert_allclose(split["head_joints"], np.array([19.0, 20.0], dtype=np.float32))
        self.assertEqual(split["lift_height_mm"], 21.0)

    def test_build_policy_payload(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        observation = {
            "observation.images.head": image,
            "observation.images.wrist_left": image,
            "observation.images.wrist_right": image,
            "source.observation.state": np.zeros((REALMAN_SOURCE_STATE_DIM,), dtype=np.float32),
        }
        stats = {
            "q01": [-1.0] * REALMAN_STATE_DIM,
            "q99": [1.0] * REALMAN_STATE_DIM,
        }
        payload = build_policy_payload(observation, "test", image_size=8, state_stats=stats)
        self.assertEqual(np.asarray(payload["qwen_frames"]).shape, (1, 3, 8, 8, 3))
        self.assertEqual(np.asarray(payload["qwen_frames"]).dtype, np.uint8)
        self.assertEqual(payload["state"].shape, (1, 1, REALMAN_STATE_DIM))
        self.assertEqual(payload["state"].dtype, np.float32)

    def test_policy_payload_matches_training_tensor_contract(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        observation = {
            "observation.images.head": image,
            "observation.images.wrist_left": image,
            "observation.images.wrist_right": image,
            "source.observation.state": np.zeros((REALMAN_SOURCE_STATE_DIM,), dtype=np.float32),
        }
        metadata = {
            "video_resolution_size": 8,
            "realman_input_contract": {
                "payload_key": "qwen_frames",
                "frame_size": 8,
            },
        }

        payload = build_policy_payload(observation, "test", image_size=8)

        validate_realman_policy_payload(payload, metadata)

    def test_validate_metadata(self):
        statistics_sha256 = "a" * 64
        warnings = validate_realman_server_metadata(
            {
                "action_type": JOINT_DELTA_GRIPPER_ABSOLUTE,
                "action_dim": REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT,
                "state_dim": REALMAN_STATE_DIM,
                "default_state_norm_mode": Q01_Q99_UNCLIPPED,
                "default_action_norm_mode": Q01_Q99_UNCLIPPED,
                "normalization_statistics_sha256": statistics_sha256,
                "realman_action_contract": {
                    "action_type": JOINT_DELTA_GRIPPER_ABSOLUTE,
                    "delta_anchor": "chunk_start_state",
                    "policy_action_state_indices": list(
                        REALMAN_18D_ACTION_CONTRACT.action_to_state_indices
                    ),
                    "absolute_gripper_policy_action_indices": [7, 15],
                    "representation": REALMAN_18D_ACTION_CONTRACT.to_dict(),
                    "representation_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
                    "normalization_statistics_sha256": statistics_sha256,
                },
            }
        )
        self.assertEqual(warnings, [])

    def test_validate_metadata_rejects_ambiguous_delta_checkpoint(self):
        warnings = validate_realman_server_metadata(
            {
                "action_type": "delta_qpos",
                "action_dim": REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT,
                "state_dim": REALMAN_STATE_DIM,
            }
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn(JOINT_DELTA_GRIPPER_ABSOLUTE, warnings[0])

    def test_validate_metadata_rejects_legacy_19d_no_base_policy_action_dim(self):
        warnings = validate_realman_server_metadata(
            {
                "action_type": JOINT_DELTA_GRIPPER_ABSOLUTE,
                "action_dim": REALMAN_POLICY_ACTION_DIM_NO_BASE,
                "state_dim": REALMAN_STATE_DIM,
            }
        )
        self.assertTrue(any("action_dim" in warning for warning in warnings))

    def test_mixed_delta_round_trip_keeps_grippers_absolute(self):
        state = np.arange(REALMAN_STATE_DIM, dtype=np.float32) + 10.0
        absolute = state[:18].copy()
        absolute[:7] += 0.25
        absolute[8:15] -= 0.5
        absolute[16:18] += np.asarray([1.0, -1.0], dtype=np.float32)
        absolute[[7, 15]] = [0.2, 0.8]

        encoded = realman_absolute_actions_to_policy_representation(
            absolute,
            state,
            action_type=JOINT_DELTA_GRIPPER_ABSOLUTE,
        )

        np.testing.assert_allclose(encoded[:7], 0.25)
        np.testing.assert_allclose(encoded[8:15], -0.5)
        np.testing.assert_allclose(encoded[16:18], [1.0, -1.0])
        np.testing.assert_allclose(encoded[[7, 15]], [0.2, 0.8])
        np.testing.assert_allclose(
            realman_policy_actions_to_absolute(
                encoded,
                state,
                action_type=JOINT_DELTA_GRIPPER_ABSOLUTE,
            ),
            absolute,
        )

    def test_expand_no_base_policy_action_to_robot_action(self):
        policy_action = np.arange(REALMAN_POLICY_ACTION_DIM_NO_BASE, dtype=np.float32)

        expanded = expand_policy_action_to_robot_action(policy_action)

        self.assertEqual(expanded.shape, (REALMAN_ACTION_DIM,))
        np.testing.assert_allclose(expanded[:16], policy_action[:16])
        np.testing.assert_allclose(expanded[16:19], np.zeros((3,), dtype=np.float32))
        np.testing.assert_allclose(expanded[19:22], policy_action[16:19])

    def test_expand_no_base_no_lift_policy_action_preserves_measured_lift(self):
        policy_action = np.arange(REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT, dtype=np.float32)

        expanded = expand_policy_action_to_robot_action(policy_action, lift_height_mm=321.0)

        self.assertEqual(expanded.shape, (REALMAN_ACTION_DIM,))
        np.testing.assert_allclose(expanded[:16], policy_action[:16])
        np.testing.assert_allclose(expanded[16:19], np.zeros((3,), dtype=np.float32))
        np.testing.assert_allclose(expanded[19:21], policy_action[16:18])
        self.assertEqual(expanded[21], 321.0)

    def test_expand_no_base_no_lift_policy_action_requires_measured_lift(self):
        policy_action = np.zeros((REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT,), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "requires the current measured lift_height_mm"):
            expand_policy_action_to_robot_action(policy_action)

    def test_no_base_no_lift_contract_names_and_omitted_indices(self):
        self.assertEqual(
            realman_policy_action_names(REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT),
            REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT,
        )
        self.assertEqual(
            realman_omitted_robot_action_indices(REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT),
            (16, 17, 18, 21),
        )

    def test_vr_teleop_observation_adds_trained_source_state(self):
        state = np.arange(REALMAN_SOURCE_STATE_DIM + 2, dtype=np.float32)
        observation = {"observation.state": state}

        converted = model_compatible_observation(observation)

        np.testing.assert_allclose(
            converted["source.observation.state"],
            np.arange(REALMAN_SOURCE_STATE_DIM, dtype=np.float32),
        )
        np.testing.assert_allclose(converted["observation.state"], state)

    def test_vr_teleop_action_payload_rebuilds_vector(self):
        vector = np.arange(REALMAN_ACTION_DIM, dtype=np.float32)
        split = split_action_vector(vector)

        rebuilt = vector_from_action_payload(split)

        np.testing.assert_allclose(rebuilt, vector)

    def test_vr_teleop_action_payload_expands_no_base_vector(self):
        policy_action = np.arange(REALMAN_POLICY_ACTION_DIM_NO_BASE, dtype=np.float32)

        rebuilt = vector_from_action_payload(policy_action)

        np.testing.assert_allclose(rebuilt, expand_policy_action_to_robot_action(policy_action))

    def test_vr_teleop_action_payload_expands_no_lift_vector_from_measured_state(self):
        policy_action = np.arange(REALMAN_POLICY_ACTION_DIM_NO_BASE_NO_LIFT, dtype=np.float32)

        rebuilt = vector_from_action_payload(policy_action, lift_height_mm=275.0)

        np.testing.assert_allclose(
            rebuilt,
            expand_policy_action_to_robot_action(policy_action, lift_height_mm=275.0),
        )


if __name__ == "__main__":
    unittest.main()
