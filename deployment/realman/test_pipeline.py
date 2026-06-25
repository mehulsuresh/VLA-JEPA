import unittest

import numpy as np

from deployment.realman.pipeline import (
    REALMAN_ACTION_DIM,
    REALMAN_POLICY_ACTION_DIM_NO_BASE,
    REALMAN_STATE_DIM,
    build_policy_payload,
    expand_policy_action_to_robot_action,
    split_action_vector,
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
            "source.observation.state": np.zeros((REALMAN_STATE_DIM,), dtype=np.float32),
        }
        stats = {
            "q01": [-1.0] * REALMAN_STATE_DIM,
            "q99": [1.0] * REALMAN_STATE_DIM,
        }
        payload = build_policy_payload(observation, "test", image_size=8, state_stats=stats)
        self.assertEqual(len(payload["batch_images"][0]), 3)
        self.assertEqual(payload["batch_images"][0][0].shape, (8, 8, 3))
        self.assertEqual(payload["state"].shape, (1, 1, REALMAN_STATE_DIM))

    def test_validate_metadata(self):
        warnings = validate_realman_server_metadata(
            {
                "action_type": "absolute_qpos",
                "action_dim": REALMAN_ACTION_DIM,
                "state_dim": REALMAN_STATE_DIM,
            }
        )
        self.assertEqual(warnings, [])

    def test_validate_metadata_accepts_no_base_policy_action_dim(self):
        warnings = validate_realman_server_metadata(
            {
                "action_type": "absolute_qpos",
                "action_dim": REALMAN_POLICY_ACTION_DIM_NO_BASE,
                "state_dim": REALMAN_STATE_DIM,
            }
        )
        self.assertEqual(warnings, [])

    def test_expand_no_base_policy_action_to_robot_action(self):
        policy_action = np.arange(REALMAN_POLICY_ACTION_DIM_NO_BASE, dtype=np.float32)

        expanded = expand_policy_action_to_robot_action(policy_action)

        self.assertEqual(expanded.shape, (REALMAN_ACTION_DIM,))
        np.testing.assert_allclose(expanded[:16], policy_action[:16])
        np.testing.assert_allclose(expanded[16:19], np.zeros((3,), dtype=np.float32))
        np.testing.assert_allclose(expanded[19:22], policy_action[16:19])

    def test_vr_teleop_observation_adds_trained_source_state(self):
        state = np.arange(REALMAN_STATE_DIM + 2, dtype=np.float32)
        observation = {"observation.state": state}

        converted = model_compatible_observation(observation)

        np.testing.assert_allclose(
            converted["source.observation.state"],
            np.arange(REALMAN_STATE_DIM, dtype=np.float32),
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


if __name__ == "__main__":
    unittest.main()
