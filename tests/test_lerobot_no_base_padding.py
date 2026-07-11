import pickle
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.data_config import (
    ROBOT_TYPE_CONFIG_MAP,
    RealmanBimanualSourceNoBaseDataConfig,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform


def test_retrieve_data_and_pad_first_last_returns_edge_values_and_mask():
    dataset = object.__new__(LeRobotSingleDataset)
    array = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    step_indices = np.array([-2, 0, 2, 3, 4, 9])

    output, padding_mask = dataset.retrieve_data_and_pad(
        array,
        step_indices,
        max_length=4,
        padding_strategy="first_last",
        return_padding_mask=True,
    )

    np.testing.assert_allclose(output, array[[0, 0, 2, 3, 3, 3]])
    np.testing.assert_array_equal(padding_mask, [True, False, False, False, True, True])


def test_retrieve_data_and_pad_zero_still_available_for_delta_modalities():
    dataset = object.__new__(LeRobotSingleDataset)
    array = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    step_indices = np.array([-1, 0, 3, 4])

    output, padding_mask = dataset.retrieve_data_and_pad(
        array,
        step_indices,
        max_length=4,
        padding_strategy="zero",
        return_padding_mask=True,
    )

    np.testing.assert_allclose(output, np.asarray([[0, 0], [0, 1], [6, 7], [0, 0]], dtype=np.float32))
    np.testing.assert_array_equal(padding_mask, [True, False, False, True])


def test_realman_source_no_base_config_slices_source_actions():
    assert ROBOT_TYPE_CONFIG_MAP["realman_bimanual_source_no_base"] is RealmanBimanualSourceNoBaseDataConfig
    assert DATASET_NAMED_MIXTURES["ogrealman_source_no_base_v3"] == [
        ("", 1.0, "realman_bimanual_source_no_base", "v3.0")
    ]

    data_config = RealmanBimanualSourceNoBaseDataConfig(
        observation_indices=[0],
        action_indices=[0, 1, 2],
    )

    assert data_config.action_keys == ["action.source_controls", "action.source_head_lift"]
    modality_config = data_config.modality_config()
    assert modality_config["action"].modality_keys == data_config.action_keys
    assert modality_config["action"].delta_indices == [0, 1, 2]

    action_norm_modes = {}
    for transform in data_config.transform().transforms:
        if isinstance(transform, StateActionTransform):
            action_norm_modes.update(
                {
                    key: mode
                    for key, mode in transform.normalization_modes.items()
                    if key.startswith("action.")
                }
            )
    assert action_norm_modes == {
        "action.source_controls": "min_max",
        "action.source_head_lift": "min_max",
    }


def test_lerobot_loads_v3_subtask_labels_from_metadata(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    pd.DataFrame(
        {
            "subtask_index": [0, 2],
            "local_subtask_text": ["__unlabeled__", "reach into the bin for the chain"],
            "description": ["fallback zero", "fallback reach"],
        }
    ).to_parquet(meta_dir / "subtasks.parquet")

    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = tmp_path
    dataset._lerobot_version = "v3.0"
    dataset.data_cfg = {"subtask_prompt_label_column": "local_subtask_text"}

    labels = dataset._get_subtask_labels()

    assert labels[2] == "reach into the bin for the chain"
    dataset._subtask_labels = labels
    assert dataset._subtask_label_for_index(np.asarray([2], dtype=np.int64)) == "reach into the bin for the chain"


def test_lerobot_synthesizes_missing_modality_metadata_from_info_and_overrides(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        """
{
  "features": {
    "action": {"dtype": "float32", "shape": [14]},
    "observation.state": {"dtype": "float32", "shape": [14]},
    "source.action": {"dtype": "float32", "shape": [22]},
    "source.observation.state": {"dtype": "float32", "shape": [19]},
    "observation.images.head": {"dtype": "video", "shape": [3, 480, 640], "names": ["channel", "height", "width"], "info": {"video.channels": 3, "video.fps": 20}},
    "observation.images.wrist_left": {"dtype": "video", "shape": [3, 480, 640], "names": ["channel", "height", "width"], "info": {"video.channels": 3, "video.fps": 20}},
    "task_index": {"dtype": "int64", "shape": [1]}
  }
}
""",
        encoding="utf-8",
    )

    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = tmp_path
    dataset.data_cfg = {
        "modality_metadata_overrides": {
            "state": {
                "source": {
                    "original_key": "source.observation.state",
                    "start": 0,
                    "end": 19,
                    "absolute": True,
                }
            },
            "action": {
                "source_controls": {
                    "original_key": "source.action",
                    "start": 0,
                    "end": 16,
                    "absolute": True,
                }
            },
        }
    }

    modality_meta = dataset._apply_modality_metadata_overrides(dataset._load_lerobot_modality_dict())

    assert modality_meta["video"]["base_view"]["original_key"] == "observation.images.head"
    assert modality_meta["video"]["left_wrist"]["original_key"] == "observation.images.wrist_left"
    assert modality_meta["annotation"]["human.action.task_description"]["original_key"] == "task_index"
    assert modality_meta["state"]["source"]["end"] == 19
    assert modality_meta["action"]["source_controls"]["original_key"] == "source.action"


def _minimal_lerobot_mixture_for_action_mask():
    mixture = object.__new__(LeRobotMixtureDataset)
    mixture.use_action_validity_prefix_mask = True
    mixture.action_validity_invalid_run_length = 3
    return mixture


def test_lerobot_action_validity_mask_uses_explicit_valid_state_column():
    mixture = _minimal_lerobot_mixture_for_action_mask()
    dataset = object.__new__(LeRobotSingleDataset)
    dataset.curr_traj_data = pd.DataFrame({"valid_state": [1, 1, 0, 0, 0, 1]})
    dataset._modality_keys = {"action": ["action.source_controls"]}
    dataset._delta_indices = {"action.source_controls": np.arange(6)}
    dataset.data_cfg = {
        "use_action_validity_prefix_mask": True,
        "action_validity_label_key": "valid_state",
        "action_validity_positive_is_valid": True,
        "action_validity_invalid_run_length": 3,
    }
    action = np.ones((6, 19), dtype=np.float32)

    mask = mixture._build_action_validity_mask(
        dataset,
        step=0,
        action=action,
        action_is_pad=None,
    )

    assert mask.shape == action.shape
    assert mask[:, 0].astype(int).tolist() == [1, 1, 0, 0, 0, 0]


def test_lerobot_action_validity_mask_missing_label_falls_back_to_all_ones():
    mixture = _minimal_lerobot_mixture_for_action_mask()
    dataset = object.__new__(LeRobotSingleDataset)
    dataset.curr_traj_data = pd.DataFrame({"frame_index": [0, 1, 2, 3]})
    dataset._modality_keys = {"action": ["action.source_controls"]}
    dataset._delta_indices = {"action.source_controls": np.arange(4)}
    dataset.data_cfg = {
        "use_action_validity_prefix_mask": True,
        "action_validity_label_key": "valid_state",
        "action_validity_invalid_run_length": 3,
    }
    action = np.ones((4, 19), dtype=np.float32)

    mask = mixture._build_action_validity_mask(
        dataset,
        step=0,
        action=action,
        action_is_pad=np.asarray([False, False, False, True]),
    )

    assert mask.shape == action.shape
    assert mask.all()


def _minimal_lerobot_dataset(tmp_path, *, delete_pause_frame=False):
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = tmp_path
    dataset._dataset_name = tmp_path.name
    dataset._lerobot_version = "v3.0"
    dataset.tag = "new_embodiment"
    dataset.delete_pause_frame = delete_pause_frame
    dataset.data_cfg = {
        "data_mix": "unit_mix",
        "action_type": "absolute_qpos",
        "modality_metadata_overrides": {
            "action": {
                "source_controls": {
                    "original_key": "source.action",
                    "start": 0,
                    "end": 16,
                    "absolute": True,
                }
            }
        },
    }
    dataset._trajectory_ids = np.asarray([0, 1], dtype=np.int64)
    dataset._trajectory_lengths = np.asarray([2, 3], dtype=np.int64)
    dataset._modality_keys = {
        "action": ["action.source_controls"],
        "language": ["annotation.human.action.task_description"],
        "state": ["state.source"],
        "video": ["video.head"],
    }
    return dataset


def test_lerobot_step_cache_uses_validated_config_key_not_legacy_magic_names(tmp_path):
    dataset = _minimal_lerobot_dataset(tmp_path)
    dataset.data_cfg["validate_language_for_step_index"] = True
    legacy_cache_path = tmp_path / "meta" / "steps_332420bad1ab.pkl"
    legacy_cache_path.parent.mkdir(parents=True)
    with legacy_cache_path.open("wb") as handle:
        pickle.dump({"steps": [(99, 99)]}, handle)

    calls = []

    def compute_steps(self):
        calls.append("computed")
        return [(0, 0), (0, 1), (1, 0)]

    dataset._get_all_steps_single_process = types.MethodType(compute_steps, dataset)

    steps = dataset._get_all_steps()
    assert steps == [(0, 0), (0, 1), (1, 0)]
    assert calls == ["computed"]

    config_key = dataset._get_steps_config_key()
    cache_path = tmp_path / "meta" / f"steps_{config_key}.pkl"
    assert cache_path.exists()
    with cache_path.open("rb") as handle:
        cached_data = pickle.load(handle)
    assert cached_data["cache_metadata"] == dataset._get_steps_cache_metadata()

    steps = dataset._get_all_steps()
    assert steps == [(0, 0), (0, 1), (1, 0)]
    assert calls == ["computed"]


def test_lerobot_dense_step_index_uses_trajectory_lengths_without_episode_scan(tmp_path):
    dataset = _minimal_lerobot_dataset(tmp_path)

    def compute_steps(self):
        raise AssertionError("dense step index should not call the per-trajectory scanner")

    dataset._get_all_steps_single_process = types.MethodType(compute_steps, dataset)

    steps = dataset._get_all_steps()
    assert steps == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]


def test_lerobot_step_cache_key_changes_when_pause_filter_changes(tmp_path):
    dataset = _minimal_lerobot_dataset(tmp_path, delete_pause_frame=False)
    key_without_pause_filter = dataset._get_steps_config_key()

    dataset.delete_pause_frame = True
    key_with_pause_filter = dataset._get_steps_config_key()

    assert key_with_pause_filter != key_without_pause_filter


def test_trajectory_data_cache_sets_id_and_reuses_loaded_parquet(tmp_path, monkeypatch):
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = tmp_path
    dataset._lerobot_version = "v2.0"
    dataset._data_path_pattern = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    dataset._chunk_size = 1000
    dataset.curr_traj_id = None
    dataset.curr_traj_data = None

    parquet_path = tmp_path / "data/chunk-000/episode_000003.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.touch()

    calls = []
    frame = pd.DataFrame({"episode_index": [3, 3], "value": [10, 11]})

    def fake_read_parquet(path):
        calls.append(path)
        return frame

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    first = dataset.get_trajectory_data(3)
    second = dataset.get_trajectory_data(3)

    assert first is frame
    assert second is frame
    assert dataset.curr_traj_id == 3
    assert dataset.curr_traj_data is frame
    assert calls == [parquet_path]


def test_lerobot_v3_uses_episode_offsets_and_bounded_arrow_shard_cache(tmp_path, monkeypatch):
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = tmp_path
    dataset._lerobot_version = "v3.0"
    dataset._data_path_pattern = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    dataset.data_cfg = {"lerobot_v3_parquet_cache_size": 1}
    dataset.curr_traj_id = None
    dataset.curr_traj_data = None
    dataset._v3_parquet_shard_cache = OrderedDict()
    dataset._v3_data_file_start_indices = {(0, 0): 100}
    dataset._trajectory_ids = np.asarray([7, 8], dtype=np.int64)
    dataset._trajectory_lengths = np.asarray([2, 3], dtype=np.int64)
    dataset.trajectory_ids_to_metadata = {
        7: {
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 100,
            "dataset_to_index": 102,
        },
        8: {
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": 102,
            "dataset_to_index": 105,
        },
    }

    parquet_path = tmp_path / "data/chunk-000/file-000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [7, 7, 8, 8, 8],
            "value": [10, 11, 20, 21, 22],
        }
    ).to_parquet(parquet_path)

    from starVLA.dataloader.gr00t_lerobot import datasets as datasets_module

    original_read_table = datasets_module.pq.read_table
    calls = []

    def counting_read_table(path, **kwargs):
        calls.append(Path(path))
        return original_read_table(path, **kwargs)

    monkeypatch.setattr(datasets_module.pq, "read_table", counting_read_table)

    first = dataset.get_trajectory_data(7)
    second = dataset.get_trajectory_data(8)

    assert first["value"].tolist() == [10, 11]
    assert second["value"].tolist() == [20, 21, 22]
    assert calls == [parquet_path]
    assert list(dataset._v3_parquet_shard_cache) == [parquet_path]

    state = dataset.__getstate__()
    assert not state["_v3_parquet_shard_cache"]
    assert state["curr_traj_data"] is None
