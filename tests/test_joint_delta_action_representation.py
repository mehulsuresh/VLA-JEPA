from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.canonical_subset_dataset import (
    ACTION_DIM,
    JOINT_DELTA_GRIPPER_ABSOLUTE,
    SHARD_Q01_Q99_UNCLIPPED,
    STATE_DIM,
    CanonicalSubsetVLADataset,
    EpisodeSpec,
)
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    AnchorRelativeActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.data_config import (
    RealmanBimanualSourceNoBaseNoLiftDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lerobot_anchor_relative_transform_uses_chunk_state_and_keeps_grippers():
    transform = AnchorRelativeActionTransform(
        apply_to=["action.source_controls", "action.source_head"],
        mappings={
            "source_controls": {
                "state_key": "source",
                "state_indices": list(range(16)),
                "delta_mask": [True] * 7 + [False] + [True] * 7 + [False],
            },
            "source_head": {
                "state_key": "source",
                "state_indices": [16, 17],
                "delta_mask": [True, True],
            },
        },
    )
    state = np.arange(19, dtype=np.float32)[None, :]
    controls = np.stack([state[0, :16] + 1.0, state[0, :16] + 2.0])
    controls[:, [7, 15]] = [[0.2, 0.8], [0.3, 0.7]]
    head = np.stack([state[0, 16:18] + 3.0, state[0, 16:18] + 4.0])

    output = transform(
        {
            "state.source": state,
            "action.source_controls": controls,
            "action.source_head": head,
        }
    )

    np.testing.assert_allclose(output["action.source_controls"][0, :7], 1.0)
    np.testing.assert_allclose(output["action.source_controls"][1, 8:15], 2.0)
    np.testing.assert_allclose(
        output["action.source_controls"][:, [7, 15]],
        [[0.2, 0.8], [0.3, 0.7]],
    )
    np.testing.assert_allclose(output["action.source_head"], [[3.0, 3.0], [4.0, 4.0]])


def test_realman_data_config_places_delta_conversion_before_normalization():
    config = RealmanBimanualSourceNoBaseNoLiftDataConfig(
        observation_indices=[0],
        action_indices=[0, 1],
    )
    config.data_cfg = {
        "action_type": JOINT_DELTA_GRIPPER_ABSOLUTE,
        "action_delta_anchor": "chunk_start_state",
        "action_delta_mappings": {
            "source_controls": {
                "state_key": "source",
                "state_indices": list(range(16)),
                "delta_mask": [True] * 7 + [False] + [True] * 7 + [False],
            },
            "source_head": {
                "state_key": "source",
                "state_indices": [16, 17],
                "delta_mask": [True, True],
            },
        },
    }

    transforms = config.transform().transforms

    assert isinstance(transforms[0], AnchorRelativeActionTransform)


def test_canonical_adapter_mask_converts_joint_spans_but_not_hands_or_base():
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.action_type = JOINT_DELTA_GRIPPER_ABSOLUTE
    dataset.absolute_action_references = {"absolute", "mixed"}
    adapter = SimpleNamespace(
        metadata={"action_reference": "mixed"},
        action_mappings=(
            SimpleNamespace(target="left_arm_joint", target_indices=None),
            SimpleNamespace(target="left_hand", target_indices=(0,)),
            SimpleNamespace(target="neck", target_indices=None),
            SimpleNamespace(target="base_twist", target_indices=None),
        ),
    )

    mask = dataset._joint_delta_mask_for_adapter(adapter)

    assert mask[:7].all()
    assert mask[40:42].all()
    assert not mask[28:34].any()
    assert not mask[46:49].any()


def test_canonical_native_relative_actions_are_not_delta_encoded_twice():
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.action_type = JOINT_DELTA_GRIPPER_ABSOLUTE
    dataset.absolute_action_references = {"absolute", "mixed"}
    adapter = SimpleNamespace(
        metadata={"action_reference": "relative"},
        action_mappings=(
            SimpleNamespace(target="left_arm_joint", target_indices=None),
            SimpleNamespace(target="right_arm_joint", target_indices=None),
        ),
    )

    mapping = dataset._joint_delta_mapping_for_adapter(adapter)

    assert np.all(mapping == -1)


def test_canonical_delta_statistics_use_future_targets_against_same_anchor():
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.sample_stride = 1
    dataset._action_offsets = np.arange(2, dtype=np.int64)
    actions = np.zeros((3, ACTION_DIM), dtype=np.float32)
    states = np.zeros((3, STATE_DIM), dtype=np.float32)
    actions[:, 0] = [10.0, 12.0, 16.0]
    states[:, 0] = [9.0, 11.0, 15.0]
    action_mask = np.zeros_like(actions, dtype=bool)
    state_mask = np.zeros_like(states, dtype=bool)
    action_mask[:, 0] = True
    state_mask[:, 0] = True
    delta_mask = np.zeros((ACTION_DIM,), dtype=bool)
    delta_mask[0] = True
    action_to_state = np.full((ACTION_DIM,), -1, dtype=np.int64)
    action_to_state[0] = 0
    episode = EpisodeSpec(
        local_start=0,
        length=3,
        task="test",
        video_paths={},
        video_base_frames={},
    )

    low, high = dataset._action_robust_bounds(
        action_values=actions,
        action_mask=action_mask,
        state_values=states,
        state_mask=state_mask,
        episodes=[episode],
        action_delta_mask=delta_mask,
        action_to_state_indices=action_to_state,
    )

    # Anchor/future pairs are [1, 3], [1, 5], [1, 1] (last target pads).
    expected = np.asarray([1.0, 3.0, 1.0, 5.0, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(low[0], np.percentile(expected, 1))
    np.testing.assert_allclose(high[0], np.percentile(expected, 99))


def test_all_canonical_configs_declare_comparable_mixed_action_contract():
    for config_path in sorted((REPO_ROOT / "scripts/config").glob("*canonical*.yaml")):
        cfg = OmegaConf.load(config_path).datasets.vla_data
        assert cfg.action_type == JOINT_DELTA_GRIPPER_ABSOLUTE
        assert cfg.action_delta_anchor == "chunk_start_state"
        assert cfg.gripper_action_type == "absolute"
        assert cfg.sidecar_normalization == SHARD_Q01_Q99_UNCLIPPED
