from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from deployment.model_server.checkpoint_utils import resolve_policy_checkpoint
from deployment.trossen.pipeline import (
    DEFAULT_CAMERA_ORDER,
    build_policy_payload,
    compute_action_state_stats_closeness,
    compute_absolute_goal,
    compute_absolute_goal_chunk,
    continuous_normalize,
    continuous_unnormalize,
    continuous_minmax_unnormalize,
    infer_action_mode_from_stats,
    resolve_action_stats,
    resolve_norm_mode,
    resolve_state_stats,
)


class CheckpointResolutionTest(unittest.TestCase):
    def test_resolve_final_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            ckpt_file = run_root / "final_model" / "pytorch_model.pt"
            ckpt_file.parent.mkdir(parents=True)
            ckpt_file.write_bytes(b"test")
            self.assertEqual(resolve_policy_checkpoint(run_root), ckpt_file.resolve())


class TrossenPipelineTest(unittest.TestCase):
    def test_build_policy_payload_shapes(self) -> None:
        observation = {
            "observation.state": np.arange(14, dtype=np.float32),
        }
        for idx, camera_name in enumerate(DEFAULT_CAMERA_ORDER):
            observation[f"observation.images.{camera_name}"] = np.full((480, 640, 3), idx * 32, dtype=np.uint8)

        payload = build_policy_payload(observation, "Pick the object.", image_size=224)
        self.assertEqual(payload["state"].shape, (1, 1, 14))
        self.assertEqual(len(payload["batch_images"]), 1)
        self.assertEqual(len(payload["batch_images"][0]), 3)
        for image in payload["batch_images"][0]:
            self.assertEqual(image.shape, (224, 224, 3))
            self.assertEqual(image.dtype, np.uint8)

    def test_continuous_unnormalize_keeps_gripper_continuous(self) -> None:
        normalized = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, 0.2]], dtype=np.float32)
        action_stats = {
            "q01": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -50.0, -20.0],
            "q99": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 50.0, 20.0],
            "mask": [True] * 8,
        }
        unnormalized = continuous_minmax_unnormalize(normalized, action_stats)
        self.assertAlmostEqual(float(unnormalized[0, 6]), -10.0)
        self.assertAlmostEqual(float(unnormalized[0, 7]), 4.0)

    def test_min_max_roundtrip_matches_training_transform(self) -> None:
        raw = np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32)
        stats = {
            "min": [0.0, 5.0],
            "max": [10.0, 15.0],
        }
        normalized = continuous_normalize(raw, stats, mode="min_max")
        np.testing.assert_allclose(normalized, np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32))
        reconstructed = continuous_unnormalize(normalized, stats, mode="min_max")
        np.testing.assert_allclose(reconstructed, raw)

    def test_absolute_goal_adds_delta(self) -> None:
        current = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        delta = np.array([0.5, -1.0, 2.0], dtype=np.float32)
        goal = compute_absolute_goal(current, delta, action_scale=2.0, delta_clip=1.5)
        np.testing.assert_allclose(goal, np.array([2.0, 0.0, 6.0], dtype=np.float32))

    def test_absolute_goal_chunk_accumulates_deltas(self) -> None:
        current = np.array([10.0, 20.0], dtype=np.float32)
        delta_chunk = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], dtype=np.float32)
        goals = compute_absolute_goal_chunk(current, delta_chunk, chunk_size=2, action_scale=1.0)
        np.testing.assert_allclose(
            goals,
            np.array([[11.0, 19.0], [13.0, 17.0]], dtype=np.float32),
        )

    def test_resolve_action_stats(self) -> None:
        metadata = {
            "default_unnorm_key": "new_embodiment",
            "action_stats_by_key": {
                "new_embodiment": {"q01": [0.0], "q99": [1.0], "mask": [True]},
            },
        }
        self.assertEqual(resolve_action_stats(metadata)["q99"], [1.0])

    def test_resolve_state_stats(self) -> None:
        metadata = {
            "default_unnorm_key": "new_embodiment",
            "state_stats_by_key": {
                "new_embodiment": {"q01": [0.0], "q99": [1.0]},
            },
        }
        self.assertEqual(resolve_state_stats(metadata)["q01"], [0.0])

    def test_infer_action_mode_absolute_when_stats_match(self) -> None:
        action_stats = {"q01": [0.0, 10.0], "q99": [1.0, 20.0]}
        state_stats = {"q01": [0.01, 10.02], "q99": [1.02, 19.98]}
        self.assertEqual(infer_action_mode_from_stats(action_stats, state_stats), "absolute_qpos")
        self.assertLess(compute_action_state_stats_closeness(action_stats, state_stats), 0.05)

    def test_infer_action_mode_delta_when_stats_differ(self) -> None:
        action_stats = {"q01": [0.0, 10.0], "q99": [1.0, 20.0]}
        state_stats = {"q01": [-5.0, 100.0], "q99": [5.0, 200.0]}
        self.assertEqual(infer_action_mode_from_stats(action_stats, state_stats), "delta_qpos")

    def test_resolve_norm_mode_prefers_metadata_default(self) -> None:
        metadata = {
            "default_action_norm_mode": "min_max",
            "default_state_norm_mode": "min_max",
        }
        self.assertEqual(resolve_norm_mode(metadata, "action"), "min_max")
        self.assertEqual(resolve_norm_mode(metadata, "state"), "min_max")


if __name__ == "__main__":
    unittest.main()
