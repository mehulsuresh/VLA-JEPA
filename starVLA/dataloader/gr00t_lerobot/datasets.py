# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""

import copy
import hashlib
import json
import math
import numbers
import os
import sys
import time
from collections import OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import random
import torch
import cv2

from starVLA.action_representation import (
    JOINT_DELTA_GRIPPER_ABSOLUTE,
    Q01_Q99_UNCLIPPED,
    REALMAN_ACTION_HORIZON,
    REALMAN_18D_ACTION_CONTRACT,
    REALMAN_POLICY_DIM,
    load_openpi_realman_union_statistics,
)
from starVLA.dataloader.dataset_view import (
    STATISTICS_POPULATION_CANDIDATE_PURPOSE,
    load_frozen_view,
)
from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from starVLA.dataloader.action_validity_mask import (
    build_action_mask_from_valid_flags,
    cfg_get,
    valid_flags_from_label_values,
)
from starVLA.dataloader.prompt_labels import (
    append_resolved_label_to_language,
    append_subtask_label_to_language,
    append_task_id_label_to_language,
    subtask_label_is_ignored,
    subtask_prompt_append_probability,
    subtask_prompt_ignored_labels,
)
from starVLA.dataloader.gr00t_lerobot.episode_split import (
    build_episode_catalog_binding,
    load_episode_split_selection,
)

from functools import partial
from typing import Tuple, List
import pickle

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
LE_ROBOT_RAW_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_SUBTASKS_FILENAME = "meta/subtasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


class SubtaskPromptDataError(ValueError):
    """A sample violates the validated subtask-prompt data contract."""


EPSILON = 5e-4
# v2 binds cached indices to the per-camera LeRobot v3 video shard and its
# episode-local timestamp offset. v1 caches were built through get_video_path()
# when that method incorrectly used the parquet shard for every camera.
GPU_DECODE_FRAME_INDEX_CACHE_DIRNAME = "gpu_decode_frame_indices_v2"


def get_gpu_decode_frame_index_cache_path(dataset_path: Path, trajectory_id: int) -> Path:
    return (
        dataset_path
        / "meta"
        / GPU_DECODE_FRAME_INDEX_CACHE_DIRNAME
        / f"trajectory_{int(trajectory_id):06d}.npz"
    )


@lru_cache(maxsize=512)
def _load_gpu_decode_frame_index_cache(cache_path: str) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def detect_lerobot_version(dataset_path: Path) -> str | None:
    """Infer the LeRobot dataset format from version-specific metadata files."""
    if (dataset_path / LE_ROBOT3_TASKS_FILENAME).exists():
        return "v3.0"
    if (dataset_path / LE_ROBOT_EPISODE_FILENAME).exists():
        return "v2.0"
    return None

def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    # parquet_paths = parquet_paths[:3]
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        try:
            parquet_data = pd.read_parquet(parquet_path)
            parquet_data = parquet_data
            all_low_dim_data_list.append(parquet_data)
        except Exception as e:
            print(f"Failed to load parquet file {parquet_path}: {e}")
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns:
        if le_modality.startswith("annotation."):
            continue
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            # Keep the row count with the generated statistics.  LeRobot v3's
            # meta/info.json exposes total_frames, so this is the minimum
            # provenance needed to reject a table copied from an older dataset.
            "count": [int(np_data.shape[0])],
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg=None,
        lerobot_version: str | None = None,
        episode_split_role: str | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.data_cfg = data_cfg
        detected_version = detect_lerobot_version(Path(dataset_path))
        if lerobot_version is not None:
            self._lerobot_version = lerobot_version
        elif detected_version is not None:
            self._lerobot_version = detected_version
        elif data_cfg is not None and data_cfg.get("lerobot_version", None) is not None:
            self._lerobot_version = str(data_cfg.get("lerobot_version"))
        else:
            self._lerobot_version = "v2.0"

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        # Read the complete episode catalog before loading normalization
        # statistics.  When an immutable split is configured, this lets us
        # verify the manifest and select its train-only statistics artifact
        # before any transform metadata is constructed.
        self.trajectory_ids_to_metadata = {}
        self._v3_data_file_start_indices = {}
        self._v3_parquet_shard_cache = OrderedDict()
        self.curr_traj_data = None
        self.curr_traj_id = None
        full_trajectory_ids, full_trajectory_lengths = self._get_trajectories()
        self._full_trajectory_ids = np.asarray(full_trajectory_ids, dtype=np.int64)
        self._full_trajectory_lengths = np.asarray(
            full_trajectory_lengths, dtype=np.int64
        )
        self._episode_catalog_binding = build_episode_catalog_binding(
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            lerobot_version=self._lerobot_version,
            trajectory_ids=self._full_trajectory_ids,
            trajectory_lengths=self._full_trajectory_lengths,
        )
        self._episode_split_selection = self._resolve_episode_split_selection(
            episode_split_role=episode_split_role,
        )
        self._apply_episode_split_to_catalog()

        self._metadata = self._get_metadata(EmbodimentTag(self.tag))

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self._subtask_labels = self._get_subtask_labels()
        self._validate_subtask_prompt_schema()
        self._validate_selected_subtask_prompt_coverage()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")


        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        le_modality_meta = LeRobotModalityMetadata.model_validate(
            self._apply_modality_metadata_overrides(self._load_lerobot_modality_dict())
        )
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        if self._lerobot_version == "v3.0":
            self._validate_lerobot_v3_frame_catalog(le_info)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics.  For v3, only accept a table whose selected
        # state/action columns prove that they cover info.json.total_frames.
        # This prevents a hard-linked stats_gr00t.json from an older dataset
        # from silently winning merely because it has the expected keys.
        le_statistics = self._load_lerobot_statistics(le_modality_meta, le_info)

        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    if stat_name not in {"min", "max", "mean", "std", "q01", "q99"}:
                        continue
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        self._apply_action_representation_statistics(
            dataset_statistics=dataset_statistics,
            le_modality_meta=le_modality_meta,
        )

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _apply_action_representation_statistics(
        self,
        *,
        dataset_statistics: dict[str, dict[str, dict[str, Any]]],
        le_modality_meta: LeRobotModalityMetadata,
    ) -> None:
        """Load exact train-split statistics for the model's transformed values."""

        configured_union_path = self._get_data_cfg_value(
            "normalization_statistics_artifact", None
        )
        configured_union_sha256 = self._get_data_cfg_value(
            "normalization_statistics_artifact_sha256", None
        )
        has_union_path = bool(
            configured_union_path is not None
            and str(configured_union_path).strip()
        )
        has_union_sha256 = bool(
            configured_union_sha256 is not None
            and str(configured_union_sha256).strip()
        )
        if has_union_path != has_union_sha256:
            raise ValueError(
                "LeRobot shared normalization requires both "
                "normalization_statistics_artifact and "
                "normalization_statistics_artifact_sha256."
            )

        if str(self._get_data_cfg_value("action_type", "")) != JOINT_DELTA_GRIPPER_ABSOLUTE:
            if has_union_path:
                raise ValueError(
                    "normalization_statistics_artifact is only supported for "
                    "joint_delta_gripper_absolute LeRobot datasets."
                )
            return
        if str(self._get_data_cfg_value("action_delta_anchor", "chunk_start_state")) != "chunk_start_state":
            raise ValueError(
                "joint_delta_gripper_absolute requires action_delta_anchor=chunk_start_state"
            )
        if str(self._get_data_cfg_value("gripper_action_type", "absolute")) != "absolute":
            raise ValueError(
                "joint_delta_gripper_absolute requires gripper_action_type=absolute"
            )
        if str(
            self._get_data_cfg_value("state_action_normalization", Q01_Q99_UNCLIPPED)
        ) != Q01_Q99_UNCLIPPED:
            raise ValueError(
                "OpenPI-compatible RealMan deltas require "
                f"state_action_normalization={Q01_Q99_UNCLIPPED}."
            )
        split_selection = getattr(self, "_episode_split_selection", None)
        if split_selection is None:
            raise ValueError(
                "OpenPI-compatible delta actions require an immutable episode split."
            )
        if has_union_path:
            if split_selection.role == "train":
                view_path = self._get_data_cfg_value(
                    "frozen_train_view_manifest", None
                )
                view_sha256 = self._get_data_cfg_value(
                    "frozen_train_view_manifest_sha256", None
                )
                if not (
                    view_path is not None
                    and str(view_path).strip()
                    and view_sha256 is not None
                    and str(view_sha256).strip()
                ):
                    raise ValueError(
                        "Shared union normalization for LeRobot training "
                        "requires a byte-bound frozen_train_view_manifest and "
                        "frozen_train_view_manifest_sha256."
                    )
            payload = load_openpi_realman_union_statistics(
                str(configured_union_path),
                str(configured_union_sha256),
            )
            self._install_action_representation_statistics(
                dataset_statistics=dataset_statistics,
                le_modality_meta=le_modality_meta,
                modalities=payload["modalities"],
                source_label="Shared union statistics",
            )
            statistics_path = Path(str(configured_union_path)).expanduser().resolve()
            self._action_representation_statistics_path = statistics_path
            self._action_representation_statistics_sha256 = str(
                configured_union_sha256
            )
            self._action_representation_contract_sha256 = (
                REALMAN_18D_ACTION_CONTRACT.sha256()
            )
            self._action_representation_statistics_scope = (
                "immutable_union_train_only"
            )
            self._action_representation_statistics_schema = payload["schema"]
            self._action_representation_statistics_population = copy.deepcopy(
                payload["population"]
            )
            # The union population and this local stage intentionally have
            # different catalogs/split files.  Preserve both hashes rather
            # than requiring byte equality: the frozen local view below binds
            # this dataset's exact holdout exclusion, while the union artifact
            # independently binds the shared cross-stage holdout contract.
            self._action_representation_local_split_manifest_sha256 = (
                split_selection.manifest_sha256
            )
            return

        statistics_path = split_selection.action_representation_statistics_path
        expected_sha256 = split_selection.action_representation_statistics_sha256
        if statistics_path is None or expected_sha256 is None:
            raise ValueError(
                "The split manifest does not bind action_representation_statistics. "
                "Generate them before constructing a delta-action dataset."
            )
        actual_sha256 = hashlib.sha256(statistics_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Action-representation statistics changed after split validation."
            )
        try:
            payload = json.loads(statistics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Could not load action-representation statistics {statistics_path}: {exc}"
            ) from exc
        if payload.get("schema") != "openpi-realman-18d-statistics-v1":
            raise ValueError(
                "Unsupported action-representation statistics schema: "
                f"{payload.get('schema')!r}."
            )
        contract = payload.get("contract")
        if contract != REALMAN_18D_ACTION_CONTRACT.to_dict():
            raise ValueError(
                "Action-representation statistics contract does not match the "
                "required RealMan 18-D contract."
            )
        if payload.get("contract_sha256") != REALMAN_18D_ACTION_CONTRACT.sha256():
            raise ValueError("Action-representation contract SHA-256 is invalid.")
        provenance = payload.get("provenance", {})
        if provenance.get("full_catalog_sha256") != self._episode_catalog_binding[
            "episode_catalog_sha256"
        ]:
            raise ValueError(
                "Action-representation statistics do not match the dataset catalog."
            )
        if provenance.get("train_catalog_sha256") != split_selection.train_episode_set_sha256:
            raise ValueError(
                "Action-representation statistics do not match the selected train split."
            )
        if provenance.get("train_frame_count") != split_selection.train_frame_count:
            raise ValueError(
                "Action-representation statistics frame count does not match the train split."
            )
        if provenance.get(
            "split_manifest_sha256_without_statistics_binding"
        ) != split_selection.split_manifest_sha256_without_statistics_binding:
            raise ValueError(
                "Action-representation statistics do not match the immutable "
                "episode split manifest."
            )

        modalities = payload.get("modalities")
        if not isinstance(modalities, dict):
            raise ValueError("Action-representation statistics are missing modalities.")
        self._install_action_representation_statistics(
            dataset_statistics=dataset_statistics,
            le_modality_meta=le_modality_meta,
            modalities=modalities,
            source_label="Statistics",
        )

        self._action_representation_statistics_path = statistics_path
        self._action_representation_statistics_sha256 = actual_sha256
        self._action_representation_contract_sha256 = REALMAN_18D_ACTION_CONTRACT.sha256()
        self._action_representation_statistics_scope = "train_split_only"
        self._action_representation_statistics_schema = payload["schema"]
        self._action_representation_statistics_population = None
        self._action_representation_local_split_manifest_sha256 = (
            split_selection.manifest_sha256
        )

    @staticmethod
    def _install_action_representation_statistics(
        *,
        dataset_statistics: dict[str, dict[str, dict[str, Any]]],
        le_modality_meta: LeRobotModalityMetadata,
        modalities: Any,
        source_label: str,
    ) -> None:
        """Install one validated 18-D state/action statistics view."""

        if not isinstance(modalities, dict):
            raise ValueError(
                f"{source_label} are missing state/action modalities."
            )
        total_widths: dict[str, int] = {}
        for modality_name in ("state", "action"):
            source_stats = modalities.get(modality_name)
            if not isinstance(source_stats, dict):
                raise ValueError(
                    f"{source_label} are missing {modality_name}."
                )
            expected_keys = set(getattr(le_modality_meta, modality_name))
            if set(source_stats) != expected_keys:
                raise ValueError(
                    f"{source_label} {modality_name} keys "
                    f"{sorted(source_stats)} do not "
                    f"match configured keys {sorted(expected_keys)}."
                )
            total_widths[modality_name] = 0
            for subkey, statistics in source_stats.items():
                width = int(
                    getattr(le_modality_meta, modality_name)[subkey].end
                    - getattr(le_modality_meta, modality_name)[subkey].start
                )
                total_widths[modality_name] += width
                if not isinstance(statistics, dict):
                    raise ValueError(
                        f"{source_label} for {modality_name}.{subkey} must be "
                        "an object."
                    )
                for statistic_name in ("min", "max", "mean", "std", "q01", "q99"):
                    values = statistics.get(statistic_name)
                    if not isinstance(values, list) or len(values) != width:
                        raise ValueError(
                            f"{source_label} "
                            f"{modality_name}.{subkey}.{statistic_name} "
                            f"must have width {width}."
                        )
                    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                        raise ValueError(
                            f"{source_label} "
                            f"{modality_name}.{subkey}.{statistic_name} "
                            "contain non-finite values."
                        )
                dataset_statistics[modality_name][subkey] = {
                    key: value
                    for key, value in statistics.items()
                    if key in {"min", "max", "mean", "std", "q01", "q99"}
                }
        if total_widths != {
            "state": REALMAN_POLICY_DIM,
            "action": REALMAN_POLICY_DIM,
        }:
            raise ValueError(
                f"{source_label} must cover exact 18-D state/action metadata; "
                f"got {total_widths}."
            )

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        if self._lerobot_version == "v2.0":
            episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(episode_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)

        if self._lerobot_version == "v3.0":
            file_paths = sorted((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME))
            trajectory_ids = []
            trajectory_lengths = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                video_metadata_columns: dict[str, dict[str, str]] = defaultdict(dict)
                for column in episodes_data.columns:
                    column_name = str(column)
                    if not column_name.startswith("videos/"):
                        continue
                    remainder = column_name[len("videos/") :]
                    if "/" not in remainder:
                        continue
                    video_key, field = remainder.rsplit("/", 1)
                    if field in {"chunk_index", "file_index", "from_timestamp", "to_timestamp"}:
                        video_metadata_columns[video_key][field] = column_name
                for file_row_index, (_, episode) in enumerate(episodes_data.iterrows()):
                    trajectory_id = int(episode["episode_index"])
                    data_chunk_index = int(episode["data/chunk_index"])
                    data_file_index = int(episode["data/file_index"])
                    trajectory_ids.append(trajectory_id)
                    trajectory_lengths.append(int(episode["length"]))

                    dataset_from_index = episode.get("dataset_from_index")
                    dataset_to_index = episode.get("dataset_to_index")
                    if pd.notna(dataset_from_index) and pd.notna(dataset_to_index):
                        dataset_from_index = int(dataset_from_index)
                        dataset_to_index = int(dataset_to_index)
                        file_key = (data_chunk_index, data_file_index)
                        previous_start = self._v3_data_file_start_indices.get(file_key)
                        self._v3_data_file_start_indices[file_key] = (
                            dataset_from_index
                            if previous_start is None
                            else min(previous_start, dataset_from_index)
                        )
                    else:
                        dataset_from_index = None
                        dataset_to_index = None

                    videos: dict[str, dict[str, int | float]] = {}
                    from_timestamps: dict[str, float] = {}
                    for video_key, columns in video_metadata_columns.items():
                        values: dict[str, int | float] = {}
                        for field, column in columns.items():
                            value = episode[column]
                            if pd.isna(value):
                                continue
                            values[field] = (
                                int(value)
                                if field in {"chunk_index", "file_index"}
                                else float(value)
                            )
                        if values:
                            videos[video_key] = values
                        if "from_timestamp" in values:
                            from_timestamps[video_key] = float(values["from_timestamp"])

                    self.trajectory_ids_to_metadata[trajectory_id] = {
                        "data/chunk_index": data_chunk_index,
                        "data/file_index": data_file_index,
                        "data/file_from_index": int(file_row_index),
                        "dataset_from_index": dataset_from_index,
                        "dataset_to_index": dataset_to_index,
                        "videos": videos,
                        # Retain this compatibility view for the cache builder and
                        # older callers while ``videos`` remains authoritative.
                        "videos/from_timestamps": from_timestamps,
                    }
            return np.array(trajectory_ids), np.array(trajectory_lengths)

        raise ValueError(f"Unsupported LeRobot version: {self._lerobot_version}")

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
        """
        frozen_steps = self._get_frozen_train_view_steps()
        if frozen_steps is not None:
            return frozen_steps

        cache_metadata = self._get_steps_cache_metadata()
        config_key = self._get_steps_config_key()
        steps_filename = f"steps_{config_key}.pkl"
        steps_path = self.dataset_path / "meta" / steps_filename

        if steps_path.exists():
            try:
                with open(steps_path, "rb") as f:
                    cached_data = pickle.load(f)
                cached_steps = self._validate_steps_cache(
                    cached_data,
                    expected_config_key=config_key,
                    expected_metadata=cache_metadata,
                    cache_path=steps_path,
                )
                if cached_steps is not None:
                    return cached_steps
            except (pickle.PickleError, EOFError, OSError, KeyError, TypeError, ValueError) as e:
                print(f"Ignoring invalid LeRobot step cache at {steps_path}: {e}")
        else:
            print(f"No LeRobot step cache found at {steps_path}; computing steps from scratch...")

        if not self.delete_pause_frame and not bool(
            self._get_data_cfg_value("validate_language_for_step_index", False)
        ):
            all_steps = self._get_all_steps_from_trajectory_lengths()
        else:
            all_steps = self._get_all_steps_single_process()
        
        # Cache the computed steps with unique filename
        try:
            cache_data = {
                "config_key": config_key,
                "cache_metadata": cache_metadata,
                "steps": all_steps,
                "num_trajectories": len(self.trajectory_ids),
                "total_steps": len(all_steps),
                "computed_timestamp": pd.Timestamp.now().isoformat(),
                "delete_pause_frame": self.delete_pause_frame,
            }
            
            # Ensure the meta directory exists
            steps_path.parent.mkdir(parents=True, exist_ok=True)

            tmp_path = steps_path.with_name(f".{steps_path.name}.{os.getpid()}.tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(steps_path)
            print(f"Cached steps saved to {steps_path}")
        except Exception as e:
            print(f"Failed to cache steps: {e}")
        
        return all_steps

    def _get_frozen_train_view_steps(
        self,
    ) -> list[tuple[int, int]] | None:
        """Return the exact ordered LeRobot rows from a frozen train view."""

        configured_path = self._get_data_cfg_value(
            "frozen_train_view_manifest", None
        )
        configured_sha256 = self._get_data_cfg_value(
            "frozen_train_view_manifest_sha256", None
        )
        has_path = bool(
            configured_path is not None and str(configured_path).strip()
        )
        has_sha256 = bool(
            configured_sha256 is not None
            and str(configured_sha256).strip()
        )
        if has_path != has_sha256:
            raise ValueError(
                "LeRobot frozen training views require both "
                "frozen_train_view_manifest and "
                "frozen_train_view_manifest_sha256."
            )
        if not has_path:
            self._frozen_train_view_provenance = {
                "enabled": False,
            }
            return None

        split_selection = getattr(self, "_episode_split_selection", None)
        if split_selection is None:
            raise ValueError(
                "A frozen LeRobot training view requires an immutable episode "
                "split/holdout manifest."
            )
        if split_selection.role != "train":
            raise ValueError(
                "frozen_train_view_manifest is train-only and cannot be used "
                f"for split role {split_selection.role!r}."
            )
        if (
            str(self._get_data_cfg_value("action_type", ""))
            != JOINT_DELTA_GRIPPER_ABSOLUTE
        ):
            raise ValueError(
                "A frozen RealMan training view requires "
                "action_type=joint_delta_gripper_absolute."
            )

        expected_manifest_sha256 = str(configured_sha256)
        if (
            len(expected_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_manifest_sha256
            )
        ):
            raise ValueError(
                "frozen_train_view_manifest_sha256 must be a lowercase "
                "SHA-256 hex digest."
            )
        source_hashes = {
            self.dataset_name: {
                "catalog_sha256": self._episode_catalog_binding[
                    "episode_catalog_sha256"
                ],
                "info_sha256": self._episode_catalog_binding["info_sha256"],
            }
        }
        frozen = load_frozen_view(
            configured_path,
            expected_representation_contract_sha256=(
                REALMAN_18D_ACTION_CONTRACT.sha256()
            ),
            expected_source_hashes=source_hashes,
        )
        if frozen.manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "Frozen LeRobot train-view manifest SHA-256 mismatch: "
                f"expected {expected_manifest_sha256}, got "
                f"{frozen.manifest_sha256}."
            )

        descriptor = frozen.descriptor
        if (
            descriptor.get("purpose")
            == STATISTICS_POPULATION_CANDIDATE_PURPOSE
        ):
            raise ValueError(
                "LeRobot training cannot consume a "
                "statistics_population_candidate view. Use the separate "
                "holdout-free frozen training view."
            )
        sources = descriptor["sources"]
        if (
            len(sources) != 1
            or sources[0]["source_id"] != self.dataset_name
            or sources[0]["backend"] != "lerobot"
        ):
            raise ValueError(
                "A LeRobot single dataset requires a frozen view containing "
                "exactly its one lerobot source."
            )
        representation = descriptor["representation"]
        expected_representation = {
            "state_dim": REALMAN_POLICY_DIM,
            "action_dim": REALMAN_POLICY_DIM,
            "horizon": REALMAN_ACTION_HORIZON,
            "action_type": JOINT_DELTA_GRIPPER_ABSOLUTE,
            "normalization": Q01_Q99_UNCLIPPED,
        }
        representation_mismatches = {
            key: {
                "expected": expected,
                "actual": representation.get(key),
            }
            for key, expected in expected_representation.items()
            if representation.get(key) != expected
        }
        if representation_mismatches:
            raise ValueError(
                "Frozen LeRobot train-view representation is incompatible: "
                f"{representation_mismatches}."
            )

        action_keys = tuple(self.modality_keys.get("action", ()))
        if not action_keys:
            raise ValueError(
                "Frozen LeRobot train-view validation requires an action "
                "modality."
            )
        expected_action_offsets = np.arange(
            REALMAN_ACTION_HORIZON, dtype=np.int64
        )
        for action_key in action_keys:
            offsets = np.asarray(
                self.delta_indices[action_key], dtype=np.int64
            )
            if not np.array_equal(offsets, expected_action_offsets):
                raise ValueError(
                    "Frozen LeRobot train views require exact H=50 action "
                    f"offsets 0:50; {action_key!r} has {offsets.tolist()}."
                )

        dataset_fps = int(self.lerobot_info_meta.get("fps", 0))
        if dataset_fps <= 0 or representation.get("target_fps") != dataset_fps:
            raise ValueError(
                "Frozen LeRobot train-view target_fps does not match the "
                f"dataset: {representation.get('target_fps')} != "
                f"{dataset_fps}."
            )

        holdout = descriptor["holdout_exclusions"]
        if holdout["source_id"] != self.dataset_name:
            raise ValueError(
                "Frozen LeRobot train-view holdout source does not match the "
                "dataset."
            )
        frozen_holdout_ids = tuple(
            int(value) for value in holdout["episode_indices"]
        )
        if frozen_holdout_ids != split_selection.holdout_episode_ids:
            raise ValueError(
                "Frozen LeRobot train-view holdout episode IDs do not match "
                "the immutable episode split."
            )

        selection_episode_ids = descriptor["selection"].get(
            "selected_episode_indices"
        )
        if not isinstance(selection_episode_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in selection_episode_ids
        ):
            raise ValueError(
                "Frozen LeRobot train-view selection must bind integer "
                "selected_episode_indices."
            )
        if selection_episode_ids != sorted(set(selection_episode_ids)):
            raise ValueError(
                "Frozen LeRobot train-view selected episode IDs must be "
                "sorted and unique."
            )
        selected_split_ids = set(split_selection.selected_episode_ids)
        if not set(selection_episode_ids) <= selected_split_ids:
            raise ValueError(
                "Frozen LeRobot train-view selection includes an episode "
                "outside the immutable training split."
            )
        verified_data_shards = self._verify_frozen_train_view_data_shards(
            source=sources[0],
            selected_episode_ids=selection_episode_ids,
        )

        lengths_by_id = {
            int(episode_id): int(length)
            for episode_id, length in zip(
                self._full_trajectory_ids.tolist(),
                self._full_trajectory_lengths.tolist(),
            )
        }
        steps: list[tuple[int, int]] = []
        row_pairs: set[tuple[int, int]] = set()
        sample_ids: set[str] = set()
        row_episode_ids: set[int] = set()
        for row_number, row in enumerate(frozen.iter_rows()):
            if (
                row.get("backend") != "lerobot"
                or row.get("source_id") != self.dataset_name
            ):
                raise ValueError(
                    f"Frozen LeRobot ledger row {row_number} references the "
                    "wrong backend/source."
                )
            episode_id = int(row["episode_index"])
            base_index = int(row["base_index"])
            if episode_id not in selected_split_ids:
                raise ValueError(
                    f"Frozen LeRobot ledger row {row_number} references "
                    f"episode {episode_id} outside the train split."
                )
            episode_length = lengths_by_id.get(episode_id)
            if episode_length is None:
                raise ValueError(
                    f"Frozen LeRobot ledger row {row_number} references "
                    f"unknown episode {episode_id}."
                )
            if base_index < 0 or base_index >= episode_length:
                raise ValueError(
                    f"Frozen LeRobot ledger row {row_number} base index "
                    f"{base_index} is outside episode {episode_id} length "
                    f"{episode_length}."
                )
            if row.get("episode_length") != episode_length:
                raise ValueError(
                    f"Frozen LeRobot ledger row {row_number} episode length "
                    "does not match the catalog."
                )
            expected_end = min(
                base_index + REALMAN_ACTION_HORIZON - 1,
                episode_length - 1,
            )
            expected_clamped = (
                expected_end < base_index + REALMAN_ACTION_HORIZON - 1
            )
            if (
                row.get("end_index") != expected_end
                or row.get("end_clamped") is not expected_clamped
                or row.get("end_clamp_policy") != "repeat_last"
            ):
                raise ValueError(
                    f"Frozen LeRobot ledger row {row_number} has an invalid "
                    "H=50 end-clamp contract."
                )
            pair = (episode_id, base_index)
            if pair in row_pairs or row["sample_id"] in sample_ids:
                raise ValueError(
                    f"Frozen LeRobot ledger contains duplicate row/sample at "
                    f"ordinal {row_number}."
                )
            row_pairs.add(pair)
            sample_ids.add(str(row["sample_id"]))
            row_episode_ids.add(episode_id)
            steps.append(pair)

        if not steps:
            raise ValueError("Frozen LeRobot train-view ledger is empty.")
        if not row_episode_ids <= set(selection_episode_ids):
            raise ValueError(
                "Frozen LeRobot ledger contains an episode not declared by "
                "selection.selected_episode_indices."
            )

        self._frozen_train_view_provenance = {
            "enabled": True,
            "manifest_path": frozen.manifest_path.as_posix(),
            "manifest_sha256": frozen.manifest_sha256,
            "view_id": frozen.view_id,
            "ledger_path": frozen.ledger_path.as_posix(),
            "ledger_sha256": descriptor["rows"]["sha256"],
            "row_count": frozen.row_count,
            "unique_sample_count": frozen.unique_sample_count,
            "episode_count": frozen.episode_count,
            "source_id": self.dataset_name,
            "source_catalog_sha256": sources[0]["catalog_sha256"],
            "representation_contract_sha256": representation[
                "contract_sha256"
            ],
            "holdout_exclusions_sha256": holdout["sha256"],
            "holdout_manifest_sha256": split_selection.manifest_sha256,
            "selected_data_shards": verified_data_shards,
        }
        return steps

    def _verify_frozen_train_view_data_shards(
        self,
        *,
        source: dict[str, Any],
        selected_episode_ids: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Verify every selected LeRobot parquet shard against manifest bytes."""

        configured_required = bool(
            self._get_data_cfg_value(
                "frozen_train_view_require_data_shard_hashes",
                False,
            )
        )
        raw_bindings = source.get("selected_data_shards")
        if raw_bindings is None:
            if configured_required:
                raise ValueError(
                    "Frozen LeRobot train view is required to bind "
                    "selected_data_shards, but the manifest source has none."
                )
            return []
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise ValueError(
                "Frozen LeRobot train-view selected_data_shards must be a "
                "non-empty list."
            )

        expected_paths: set[str] = set()
        for episode_id in selected_episode_ids:
            episode_meta = self.trajectory_ids_to_metadata.get(int(episode_id))
            if episode_meta is None:
                raise ValueError(
                    "Frozen LeRobot shard verification cannot resolve selected "
                    f"episode {episode_id} in trajectory metadata."
                )
            if self._lerobot_version == "v3.0":
                relative_path = self.data_path_pattern.format(
                    chunk_index=episode_meta["data/chunk_index"],
                    file_index=episode_meta["data/file_index"],
                )
            elif self._lerobot_version == "v2.0":
                relative_path = self.data_path_pattern.format(
                    episode_chunk=self.get_episode_chunk(int(episode_id)),
                    episode_index=int(episode_id),
                )
            else:
                raise ValueError(
                    "Frozen LeRobot shard verification does not support "
                    f"version {self._lerobot_version!r}."
                )
            expected_paths.add(Path(relative_path).as_posix())

        bound_paths = {str(binding.get("path")) for binding in raw_bindings}
        if bound_paths != expected_paths:
            raise ValueError(
                "Frozen LeRobot train-view selected_data_shards do not exactly "
                "cover the selected episodes: "
                f"expected {sorted(expected_paths)}, got {sorted(bound_paths)}."
            )

        root = self.dataset_path.resolve()
        verified: list[dict[str, Any]] = []
        for binding in raw_bindings:
            relative_path = str(binding["path"])
            parsed = Path(relative_path)
            candidate = (root / parsed).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    "Frozen LeRobot data-shard binding escapes the dataset "
                    f"root: {relative_path!r}."
                ) from exc
            if not candidate.is_file():
                raise ValueError(
                    f"Frozen LeRobot data shard is missing: {candidate}"
                )
            expected_size = int(binding["size_bytes"])
            actual_size = int(candidate.stat().st_size)
            if actual_size != expected_size:
                raise ValueError(
                    "Frozen LeRobot data-shard size mismatch for "
                    f"{relative_path}: expected {expected_size}, got "
                    f"{actual_size}."
                )
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            actual_sha256 = digest.hexdigest()
            expected_sha256 = str(binding["sha256"])
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "Frozen LeRobot data-shard SHA-256 mismatch for "
                    f"{relative_path}: expected {expected_sha256}, got "
                    f"{actual_sha256}."
                )
            verified.append(
                {
                    "path": relative_path,
                    "sha256": actual_sha256,
                    "size_bytes": actual_size,
                }
            )
        return verified

    @staticmethod
    def _normalize_cache_value(value):
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict) or hasattr(value, "items"):
            return {
                str(key): LeRobotSingleDataset._normalize_cache_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [LeRobotSingleDataset._normalize_cache_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _get_steps_cache_metadata(self) -> dict:
        data_cfg_cache_keys = (
            "data_mix",
            "action_type",
            "modality_metadata_overrides",
        )
        data_cfg_values = {
            key: self._normalize_cache_value(value)
            for key in data_cfg_cache_keys
            if (value := self._get_data_cfg_value(key, None)) is not None
        }
        trajectory_signature_payload = {
            "trajectory_ids": self._normalize_cache_value(self.trajectory_ids),
            "trajectory_lengths": self._normalize_cache_value(self.trajectory_lengths),
        }
        trajectory_signature = hashlib.md5(
            json.dumps(trajectory_signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "cache_schema": 2,
            "dataset_name": self.dataset_name,
            "lerobot_version": self._lerobot_version,
            "embodiment_tag": self.tag,
            "delete_pause_frame": bool(self.delete_pause_frame),
            "modality_keys": self._normalize_cache_value(self.modality_keys),
            "data_cfg": data_cfg_values,
            "num_trajectories": int(len(self.trajectory_ids)),
            "total_frames": int(np.asarray(self.trajectory_lengths, dtype=np.int64).sum()),
            "trajectory_signature": trajectory_signature,
        }

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        config_str = json.dumps(
            self._get_steps_cache_metadata(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:12]

    def _validate_steps_cache(
        self,
        cached_data: dict,
        *,
        expected_config_key: str,
        expected_metadata: dict,
        cache_path: Path,
    ) -> list[tuple[int, int]] | None:
        if not isinstance(cached_data, dict):
            raise TypeError(f"Expected dict cache payload, got {type(cached_data).__name__}")
        cached_config_key = cached_data.get("config_key")
        cached_metadata = cached_data.get("cache_metadata")
        if cached_config_key != expected_config_key or cached_metadata != expected_metadata:
            print(
                "Ignoring stale LeRobot step cache at "
                f"{cache_path}: expected config_key={expected_config_key}, found {cached_config_key}"
            )
            return None
        steps = cached_data["steps"]
        if not isinstance(steps, list):
            raise TypeError(f"Cached steps must be a list, got {type(steps).__name__}")
        return steps


    def _get_all_steps_from_trajectory_lengths(self) -> list[tuple[int, int]]:
        """Build the dense step index without loading each episode parquet.

        This is valid when pause-frame deletion is disabled. Language is still
        retrieved when samples are read; this just avoids a slow first-run scan
        through every trajectory to build the list of candidate base indices.
        """
        all_steps = [
            (int(trajectory_id), int(base_index))
            for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths)
            for base_index in range(int(trajectory_length))
        ]
        print(
            "Built dense LeRobot step index from trajectory lengths: "
            f"{len(all_steps)} steps from {len(self.trajectory_ids)} trajectories"
        )
        return all_steps


    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        """Original single-process implementation as fallback."""
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0
        
        # Check if language modality is configured
        has_language_modality = 'language' in self.modality_keys and len(self.modality_keys['language']) > 0
        
        for trajectory_id, trajectory_length in tqdm(zip(self.trajectory_ids, self.trajectory_lengths), total=len(self.trajectory_ids), desc="Getting All Step"):
            try:
                data = self.get_trajectory_data(trajectory_id)
            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to data loading error: {e}")
                skipped_trajectories += 1
                continue
            trajectory_skipped = False
            
            # Check if trajectory has valid language instruction (if language modality is configured)
            if has_language_modality:
                self.curr_traj_data = data  # Set current trajectory data for get_language to work
                try:
                    language_instruction = self.get_language(trajectory_id, self.modality_keys['language'][0], 0)
                    if not language_instruction or language_instruction[0] == "":
                        language_instruction = "think and complete the task that a human might want you to accomplish"
                        #print(f"Skipping trajectory {trajectory_id} due to empty language instruction")
                        #skipped_trajectories += 1
                        #trajectory_skipped = True
                        #continue
                except Exception as e:
                    print(f"Skipping trajectory {trajectory_id} due to language retrieval error: {e}")
                    skipped_trajectories += 1
                    trajectory_skipped = True
                    continue
            
            if not trajectory_skipped:
                processed_trajectories += 1
            
            if self.delete_pause_frame:
                # Get position and gripper fields based on available columns
                delta_position_values, gripper_values = self._get_position_and_gripper_values(data)
                previous_gripper = gripper_values[0]
                for base_index in range(trajectory_length):
                    if base_index >= len(delta_position_values) or base_index >= len(gripper_values):
                        break
                        
                    # Check for translation change using the detected position fields
                    has_translation_change = np.any(np.abs(delta_position_values[base_index]) > EPSILON)
                    has_gripper_change = gripper_values[base_index] != (previous_gripper if base_index == 0 else gripper_values[base_index-1])
                    
                    if has_translation_change or has_gripper_change:
                        all_steps.append((trajectory_id, base_index))
            else:
                for base_index in range(trajectory_length):
                    all_steps.append((trajectory_id, base_index))
                    
        # Print summary statistics
        print(f"Single-process summary: Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} empty trajectories")
        print(f"Total steps: {len(all_steps)} from {len(self.trajectory_ids)} trajectories")
                   
        return all_steps

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get('action', [])
        
        # Extract position data
        delta_position_values = None
        position_candidates = ['delta_eef_position']
        coordinate_candidates = ['x', 'y', 'z']
        
        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue
        
        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == 'x':
                                    x_data = coord_data
                                elif coord == 'y':
                                    y_data = coord_data
                                elif coord == 'z':
                                    z_data = coord_data
                    except Exception:
                        continue
            
            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()
        
        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.delta_eef_position' in data.columns:
                delta_position_values = data['action.delta_eef_position'].to_numpy().tolist()
            elif 'action' in data.columns:
                # Generic vector-action fallback for datasets that store all action dims in one field.
                delta_position_values = np.stack(data['action']).tolist()
            elif all(col in data.columns for col in ['action.x', 'action.y', 'action.z']):
                x_vals = data['action.x'].to_numpy()
                y_vals = data['action.y'].to_numpy() 
                z_vals = data['action.z'].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")
        
        # Extract gripper data
        gripper_values = None
        gripper_candidates = ['gripper_close', 'gripper']
        
        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue
        
        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.gripper_close' in data.columns:
                gripper_values = data['action.gripper_close'].to_numpy().tolist()
            elif 'action.gripper' in data.columns:
                gripper_values = data['action.gripper'].to_numpy().tolist()
            elif 'action' in data.columns:
                gripper_values = [0.0] * len(data)
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")
        
        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta = LeRobotModalityMetadata.model_validate(
            self._apply_modality_metadata_overrides(self._load_lerobot_modality_dict())
        )
        return modality_meta

    def _load_lerobot_modality_dict(self) -> dict:
        """Load modality.json, or synthesize the common LeRobot v3 metadata if it is absent."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        if modality_meta_path.exists():
            with open(modality_meta_path, "r") as f:
                return json.load(f)

        info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        if not info_path.exists():
            raise FileNotFoundError(
                f"Please provide {LE_ROBOT_MODALITY_FILENAME} or {LE_ROBOT_INFO_FILENAME} in {self.dataset_path}"
            )
        with open(info_path, "r") as f:
            info = json.load(f)
        features = info.get("features", {})

        modality_meta: dict[str, dict] = {
            "state": {},
            "action": {},
            "video": {},
            "annotation": {
                "human.action.task_description": {
                    "original_key": "task_index",
                },
            },
        }

        if "observation.state" in features:
            state_shape = features["observation.state"].get("shape", [0])
            modality_meta["state"]["joints"] = {
                "start": 0,
                "end": int(state_shape[0]),
                "original_key": "observation.state",
                "absolute": True,
                "dtype": str(features["observation.state"].get("dtype", "float32")),
            }
        if "action" in features:
            action_shape = features["action"].get("shape", [0])
            modality_meta["action"]["delta_joints"] = {
                "start": 0,
                "end": int(action_shape[0]),
                "original_key": "action",
                "absolute": False,
                "dtype": str(features["action"].get("dtype", "float32")),
            }

        video_key_map = {
            "observation.images.head": "base_view",
            "observation.images.main": "base_view",
            "observation.images.top": "base_view",
            "observation.images.wrist_left": "left_wrist",
            "observation.images.left": "left_wrist",
            "observation.images.wrist_right": "right_wrist",
            "observation.images.right": "right_wrist",
            "observation.images.extra": "extra_view",
        }
        for original_key, subkey in video_key_map.items():
            if original_key in features:
                modality_meta["video"][subkey] = {"original_key": original_key}

        if not modality_meta["video"]:
            for original_key in sorted(k for k in features if k.startswith("observation.images.")):
                subkey = original_key.rsplit(".", 1)[-1]
                modality_meta["video"][subkey] = {"original_key": original_key}

        return modality_meta

    def _get_data_cfg_value(self, key: str, default=None):
        if self.data_cfg is None:
            return default
        getter = getattr(self.data_cfg, "get", None)
        if callable(getter):
            return getter(key, default)
        return getattr(self.data_cfg, key, default)

    def _resolve_episode_split_selection(self, *, episode_split_role: str | None):
        """Load the configured immutable split before statistics/steps exist."""

        manifest = self._get_data_cfg_value("episode_split_manifest", None)
        configured_role = self._get_data_cfg_value("episode_split_role", None)
        if configured_role is not None:
            configured_role = str(configured_role).strip().lower()
            if not configured_role:
                raise ValueError("episode_split_role cannot be empty")
        caller_role = (
            str(episode_split_role).strip().lower()
            if episode_split_role is not None
            else None
        )
        if configured_role is not None and caller_role is not None and configured_role != caller_role:
            raise ValueError(
                "Configured episode_split_role conflicts with the dataset mode: "
                f"{configured_role!r} != {caller_role!r}. Remove the override or "
                "construct an explicit matching role; refusing a silent split change."
            )
        requested_role = caller_role or configured_role or "train"

        if manifest is None or not str(manifest).strip():
            if configured_role is not None:
                raise ValueError(
                    "datasets.vla_data.episode_split_role was set without an "
                    "episode_split_manifest"
                )
            if str(self._get_data_cfg_value("lerobot_statistics_source", "auto")).lower() == "split_train":
                raise ValueError(
                    "lerobot_statistics_source=split_train requires an "
                    "episode_split_manifest"
                )
            return None

        if bool(self._get_data_cfg_value("load_all_data_for_training", False)):
            raise ValueError(
                "episode_split_manifest conflicts with "
                "load_all_data_for_training=true; refusing to leak holdout "
                "episodes into training"
            )
        statistics_source = str(
            self._get_data_cfg_value("lerobot_statistics_source", "")
        ).strip().lower()
        if statistics_source != "split_train":
            raise ValueError(
                "An episode_split_manifest requires "
                "lerobot_statistics_source=split_train so neither full-dataset "
                "nor holdout-derived normalization can be loaded"
            )

        return load_episode_split_selection(
            manifest_path=manifest,
            dataset_name=self.dataset_name,
            role=requested_role,
            catalog_binding=self._episode_catalog_binding,
            trajectory_ids=self._full_trajectory_ids,
            trajectory_lengths=self._full_trajectory_lengths,
        )

    def _apply_episode_split_to_catalog(self) -> None:
        """Materialize the selected episode view before step indexing/sampling."""

        split_selection = getattr(self, "_episode_split_selection", None)
        if split_selection is None:
            self._trajectory_ids = self._full_trajectory_ids.copy()
            self._trajectory_lengths = self._full_trajectory_lengths.copy()
            return

        selected_ids = set(split_selection.selected_episode_ids)
        selection_mask = np.asarray(
            [int(value) in selected_ids for value in self._full_trajectory_ids],
            dtype=bool,
        )
        self._trajectory_ids = self._full_trajectory_ids[selection_mask].copy()
        self._trajectory_lengths = self._full_trajectory_lengths[selection_mask].copy()
        if set(map(int, self._trajectory_ids.tolist())) != selected_ids:
            raise RuntimeError(
                "Validated episode split did not materialize exactly the selected IDs"
            )
        if self._lerobot_version == "v3.0":
            self.trajectory_ids_to_metadata = {
                int(trajectory_id): metadata
                for trajectory_id, metadata in self.trajectory_ids_to_metadata.items()
                if int(trajectory_id) in selected_ids
            }

    def _apply_modality_metadata_overrides(self, modality_meta: dict) -> dict:
        """Allow run configs to remap logical state/action keys to preserved dataset columns."""
        overrides = self._get_data_cfg_value("modality_metadata_overrides", None)
        if not overrides:
            return modality_meta

        modality_meta = copy.deepcopy(modality_meta)
        replace_modalities = bool(
            self._get_data_cfg_value("replace_modality_metadata_with_overrides", False)
        )
        for modality in ("state", "action"):
            modality_overrides = overrides.get(modality, None)
            if not modality_overrides:
                continue
            if replace_modalities:
                modality_meta[modality] = {}
            else:
                modality_meta.setdefault(modality, {})
            for subkey, raw_spec in modality_overrides.items():
                spec = dict(raw_spec.items()) if hasattr(raw_spec, "items") else dict(raw_spec)
                missing = {"original_key", "start", "end"} - set(spec.keys())
                if missing:
                    raise ValueError(
                        f"modality_metadata_overrides.{modality}.{subkey} missing {sorted(missing)}"
                    )

                entry = copy.deepcopy(spec)
                entry["original_key"] = str(entry["original_key"])
                entry["start"] = int(entry["start"])
                entry["end"] = int(entry["end"])
                entry["absolute"] = bool(entry.get("absolute", modality == "state"))
                entry["dtype"] = str(entry.get("dtype", "float32"))
                modality_meta[modality][str(subkey)] = entry
        return modality_meta

    def _missing_lerobot_stat_keys(
        self,
        le_modality_meta: LeRobotModalityMetadata,
        le_statistics: dict,
    ) -> list[str]:
        missing_keys = []
        for modality in ("state", "action"):
            for state_action_meta in getattr(le_modality_meta, modality).values():
                le_key = state_action_meta.original_key
                if le_key is not None and le_key not in le_statistics:
                    missing_keys.append(le_key)
        return sorted(set(missing_keys))

    @staticmethod
    def _lerobot_stat_count(stat: dict) -> int | None:
        """Return a scalar, non-negative statistics row count when present."""

        if not isinstance(stat, dict) or "count" not in stat:
            return None
        try:
            values = np.asarray(stat["count"]).reshape(-1)
            if values.size != 1:
                return None
            value = float(values[0])
            if not np.isfinite(value) or value < 0 or not value.is_integer():
                return None
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    def _validate_lerobot_statistics(
        self,
        le_modality_meta: LeRobotModalityMetadata,
        statistics: dict,
        *,
        total_frames: int | None,
        require_frame_count: bool,
    ) -> list[str]:
        """Return reasons a candidate statistics payload is unsafe to use."""

        if not isinstance(statistics, dict):
            return ["top-level payload is not an object"]

        missing_stat_keys = self._missing_lerobot_stat_keys(
            le_modality_meta,
            statistics,
        )
        if missing_stat_keys:
            return [f"missing required columns {missing_stat_keys}"]

        required_widths: dict[str, int] = {}
        for modality in ("state", "action"):
            for state_action_meta in getattr(le_modality_meta, modality).values():
                key = state_action_meta.original_key
                if key is not None:
                    required_widths[key] = max(
                        required_widths.get(key, 0),
                        int(state_action_meta.end),
                    )
        errors: list[str] = []
        for key, required_width in sorted(required_widths.items()):
            stat = statistics[key]
            try:
                DatasetStatisticalValues.model_validate(stat)
            except ValidationError as exc:
                errors.append(f"column {key!r} has invalid statistics: {exc}")
                continue

            arrays: dict[str, np.ndarray] = {}
            for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
                try:
                    value = np.asarray(stat[stat_name], dtype=np.float64)
                except (TypeError, ValueError, OverflowError) as exc:
                    errors.append(
                        f"column {key!r} {stat_name} is not numeric: {exc}"
                    )
                    continue
                if value.ndim != 1 or value.size < required_width:
                    errors.append(
                        f"column {key!r} {stat_name} shape is {value.shape}; "
                        f"required width is at least {required_width}"
                    )
                    continue
                selected = value[:required_width]
                if not np.all(np.isfinite(selected)):
                    errors.append(
                        f"column {key!r} {stat_name} contains NaN/Inf in used dimensions"
                    )
                    continue
                arrays[stat_name] = selected

            if {"min", "max"} <= arrays.keys() and np.any(
                arrays["min"] > arrays["max"]
            ):
                errors.append(f"column {key!r} has min greater than max")
            if "std" in arrays and np.any(arrays["std"] < 0):
                errors.append(f"column {key!r} has negative std")
            if {"q01", "q99"} <= arrays.keys() and np.any(
                arrays["q01"] > arrays["q99"]
            ):
                errors.append(f"column {key!r} has q01 greater than q99")

            count = self._lerobot_stat_count(stat)
            if total_frames is not None:
                if count is None and require_frame_count:
                    errors.append(
                        f"column {key!r} has no scalar count; expected {total_frames}"
                    )
                elif count is not None and count != total_frames:
                    errors.append(
                        f"column {key!r} count is {count}; expected {total_frames}"
                    )
        return errors

    def _statistics_require_quantiles(self) -> bool:
        """Whether any configured state/action transform consumes q01/q99."""

        for transform in getattr(getattr(self, "transforms", None), "transforms", []):
            modes = getattr(transform, "normalization_modes", {})
            if any(
                str(mode) in {"q99", Q01_Q99_UNCLIPPED}
                for mode in modes.values()
            ):
                return True
        return False

    def _complete_unused_lerobot_quantiles(
        self,
        le_modality_meta: LeRobotModalityMetadata,
        statistics: dict,
    ) -> tuple[dict, bool]:
        """Fill schema-only quantiles from extrema when transforms do not use them.

        Historical official stats.json files can contain count/min/max/mean/std
        but no q01/q99. DatasetMetadata still requires all six fields. For a
        min-max or mean-std pipeline, copying min/max into the unused quantile
        slots is deterministic and safer than borrowing uncounted values from
        another file. A q99 pipeline must provide real counted quantiles.
        """

        if self._statistics_require_quantiles():
            return statistics, False
        required_keys = {
            state_action_meta.original_key
            for modality in ("state", "action")
            for state_action_meta in getattr(le_modality_meta, modality).values()
            if state_action_meta.original_key is not None
        }
        completed = copy.deepcopy(statistics)
        synthesized = False
        for key in required_keys:
            stat = completed.get(key)
            if not isinstance(stat, dict):
                continue
            if "q01" not in stat and "min" in stat:
                stat["q01"] = copy.deepcopy(stat["min"])
                synthesized = True
            if "q99" not in stat and "max" in stat:
                stat["q99"] = copy.deepcopy(stat["max"])
                synthesized = True
        return completed, synthesized

    def _validate_lerobot_v3_frame_catalog(self, le_info: dict) -> None:
        """Cross-check v3 frame counts using independent metadata sources."""

        raw_total_frames = le_info.get("total_frames")
        if isinstance(raw_total_frames, bool):
            raise ValueError("LeRobot v3 info.json total_frames must be a positive integer")
        try:
            total_frames = int(raw_total_frames)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "LeRobot v3 info.json must contain a positive integer total_frames"
            ) from exc
        if total_frames <= 0:
            raise ValueError(
                f"LeRobot v3 info.json total_frames is invalid: {raw_total_frames!r}"
            )
        if isinstance(raw_total_frames, float) and not raw_total_frames.is_integer():
            raise ValueError(
                f"LeRobot v3 info.json total_frames is non-integral: {raw_total_frames!r}"
            )

        episode_paths = sorted(self.dataset_path.glob(LE_ROBOT3_EPISODE_FILENAME))
        data_paths = sorted(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))
        if not episode_paths or not data_paths:
            raise ValueError(
                "LeRobot v3 frame-catalog validation requires episode metadata and data parquet files"
            )

        episode_length_sum = 0
        episode_manifest = []
        for path in episode_paths:
            table = pq.read_table(path, columns=["length"])
            lengths = np.asarray(table.column("length").to_numpy(), dtype=np.int64)
            if lengths.size == 0 or np.any(lengths <= 0):
                raise ValueError(f"Invalid episode lengths in {path}")
            episode_length_sum += int(lengths.sum(dtype=np.int64))
            episode_manifest.append(
                (
                    str(path.relative_to(self.dataset_path)),
                    int(path.stat().st_size),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )

        parquet_row_sum = 0
        data_manifest = []
        for path in data_paths:
            row_count = int(pq.ParquetFile(path).metadata.num_rows)
            if row_count <= 0:
                raise ValueError(f"Data parquet has no rows: {path}")
            parquet_row_sum += row_count
            data_manifest.append(
                (
                    str(path.relative_to(self.dataset_path)),
                    int(path.stat().st_size),
                    row_count,
                )
            )

        if episode_length_sum != total_frames or parquet_row_sum != total_frames:
            raise ValueError(
                "LeRobot v3 frame catalog is inconsistent: "
                f"info.total_frames={total_frames}, episode length sum={episode_length_sum}, "
                f"parquet row sum={parquet_row_sum}"
            )
        fingerprint_payload = {
            "total_frames": total_frames,
            "episode_files": episode_manifest,
            "data_files": data_manifest,
        }
        self._dataset_catalog_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._dataset_catalog_counts = {
            "info_total_frames": total_frames,
            "episode_length_sum": episode_length_sum,
            "parquet_row_sum": parquet_row_sum,
        }

    def _load_lerobot_statistics(
        self,
        le_modality_meta: LeRobotModalityMetadata,
        le_info: dict,
    ) -> dict:
        """Select a current statistics table, rejecting stale v3 candidates.

        ``meta/stats.json`` is LeRobot's canonical table and normally includes
        a per-column count.  ``meta/stats_gr00t.json`` remains a compatibility
        fallback, but v3 tables without a matching count are intentionally not
        trusted.  Set ``lerobot_statistics_source`` to ``raw`` or ``gr00t`` to
        require one source explicitly; neither option bypasses validation.
        """

        split_selection = getattr(self, "_episode_split_selection", None)
        if split_selection is not None:
            source = str(
                self._get_data_cfg_value("lerobot_statistics_source", "")
            ).strip().lower()
            if source != "split_train":
                raise ValueError(
                    "Immutable episode splits require "
                    "lerobot_statistics_source=split_train"
                )
            path = split_selection.train_statistics_path
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha256 != split_selection.train_statistics_sha256:
                raise ValueError(
                    "Bound train-only statistics changed after split validation: "
                    f"{path}"
                )
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    statistics = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Could not load bound train-only statistics {path}: {exc}"
                ) from exc
            statistics, quantiles_synthesized = self._complete_unused_lerobot_quantiles(
                le_modality_meta,
                statistics,
            )
            expected_train_frames = int(split_selection.train_frame_count)
            errors = self._validate_lerobot_statistics(
                le_modality_meta,
                statistics,
                total_frames=expected_train_frames,
                require_frame_count=True,
            )
            if errors:
                raise ValueError(
                    "Bound train-only normalization statistics failed validation: "
                    + "; ".join(errors)
                )
            self._statistics_source_path = path
            self._statistics_total_frames = expected_train_frames
            self._statistics_source_sha256 = actual_sha256
            self._statistics_effective_sha256 = hashlib.sha256(
                json.dumps(
                    statistics,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            self._statistics_quantiles_synthesized = quantiles_synthesized
            self._statistics_scope = "train_split_only"
            print(
                "Using manifest-bound train-only LeRobot statistics "
                f"from {path} (train_frames={expected_train_frames}, "
                f"active_role={split_selection.role}, "
                f"quantiles_synthesized={quantiles_synthesized})"
            )
            return statistics

        source = str(
            self._get_data_cfg_value("lerobot_statistics_source", "auto")
        ).strip().lower()
        if source not in {"auto", "raw", "gr00t"}:
            raise ValueError(
                "datasets.vla_data.lerobot_statistics_source must be one of "
                f"auto, raw, or gr00t; got {source!r}"
            )

        total_frames_value = le_info.get("total_frames")
        try:
            total_frames = (
                int(total_frames_value) if total_frames_value is not None else None
            )
        except (TypeError, ValueError, OverflowError):
            total_frames = None
        configured_require_count = self._get_data_cfg_value(
            "require_statistics_frame_count",
            None,
        )
        if self._lerobot_version == "v3.0" and configured_require_count is False:
            raise ValueError(
                "LeRobot v3 cannot disable statistics frame-count validation"
            )
        require_frame_count = self._lerobot_version == "v3.0" or bool(
            configured_require_count
        )
        if require_frame_count and (total_frames is None or total_frames <= 0):
            raise ValueError(
                "Cannot validate LeRobot statistics because info.json total_frames "
                f"is missing or invalid: {total_frames_value!r}"
            )
        raw_path = self.dataset_path / LE_ROBOT_RAW_STATS_FILENAME
        gr00t_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        candidate_paths = {
            "raw": raw_path,
            "gr00t": gr00t_path,
        }
        if source == "auto":
            candidate_names = (
                ["raw", "gr00t"]
                if self._lerobot_version == "v3.0"
                else ["gr00t", "raw"]
            )
        else:
            candidate_names = [source]
        rejection_reasons: list[str] = []

        for candidate_name in candidate_names:
            path = candidate_paths[candidate_name]
            if not path.exists():
                rejection_reasons.append(f"{path}: file does not exist")
                continue
            try:
                with open(path, "r") as handle:
                    statistics = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                rejection_reasons.append(f"{path}: could not load JSON: {exc}")
                continue

            statistics, quantiles_synthesized = self._complete_unused_lerobot_quantiles(
                le_modality_meta,
                statistics,
            )
            errors = self._validate_lerobot_statistics(
                le_modality_meta,
                statistics,
                total_frames=total_frames,
                require_frame_count=require_frame_count,
            )
            if errors:
                rejection_reasons.append(f"{path}: " + "; ".join(errors))
                continue

            self._statistics_source_path = path
            self._statistics_total_frames = total_frames
            self._statistics_source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self._statistics_effective_sha256 = hashlib.sha256(
                json.dumps(
                    statistics,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            self._statistics_quantiles_synthesized = quantiles_synthesized
            self._statistics_scope = "full_dataset_catalog"
            print(
                "Using validated LeRobot statistics "
                f"from {self._statistics_source_path} (total_frames={total_frames}, "
                f"frame_count_required={require_frame_count}, "
                f"quantiles_synthesized={quantiles_synthesized})"
            )
            return statistics

        diagnostics = "\n  - ".join(rejection_reasons)
        raise ValueError(
            "No safe LeRobot statistics table is available for "
            f"{self.dataset_name}. Refusing to train with unverifiable "
            f"normalization.\n  - {diagnostics}"
        )

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            return self._apply_task_text_overrides(df.set_index("task_index"))

        if self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            if "task" not in df.columns and df.index.name == "task":
                df = df.reset_index()
            if "task_index" in df.columns:
                return self._apply_task_text_overrides(df.set_index("task_index"))
            if "task" in df.columns:
                df = df.reset_index().rename(columns={"index": "task_index"})
                return self._apply_task_text_overrides(df.set_index("task_index"))
            raise ValueError(f"Unexpected LeRobot v3 task schema in {tasks_path}: {list(df.columns)}")

        raise ValueError(f"Unsupported LeRobot version: {self._lerobot_version}")

    def _apply_task_text_overrides(self, tasks: pd.DataFrame) -> pd.DataFrame:
        """Override placeholder task strings from the run config without mutating dataset files."""
        if self.data_cfg is None:
            return tasks

        getter = getattr(self.data_cfg, "get", None)
        overrides = getter("task_text_overrides", None) if callable(getter) else None
        if not overrides:
            return tasks

        tasks = tasks.copy()
        items = overrides.items() if hasattr(overrides, "items") else []
        for task_index, task_text in items:
            task_text = str(task_text).strip()
            if not task_text:
                continue
            tasks.loc[int(task_index), "task"] = task_text
        return tasks

    @staticmethod
    def _coerce_label_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                item = value.reshape(-1)[0]
                return item.item() if isinstance(item, np.generic) else item
            return value
        if isinstance(value, (list, tuple)) and len(value) == 1:
            item = value[0]
            return item.item() if isinstance(item, np.generic) else item
        return value

    @classmethod
    def _coerce_subtask_index(cls, value) -> int:
        """Return an exact integer ID without truncating booleans or fractions."""
        value = cls._coerce_label_value(value)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("boolean values are not valid subtask IDs")
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"non-integral numeric subtask ID {value!r}")
            return int(numeric)
        if isinstance(value, str):
            text = value.strip()
            if not text or not text.lstrip("+-").isdigit():
                raise ValueError(f"non-integer string subtask ID {value!r}")
            return int(text)
        raise ValueError(f"unsupported subtask ID {value!r}")

    def _get_subtask_labels(self) -> dict[int, str]:
        """Load optional LeRobot v3 subtask_index -> text labels."""
        subtasks_path = self.dataset_path / LE_ROBOT3_SUBTASKS_FILENAME
        enabled = bool(self._get_data_cfg_value("append_subtask_to_prompt", False))
        self._subtask_metadata_path = subtasks_path if subtasks_path.exists() else None
        self._subtask_metadata_sha256 = None

        if self._lerobot_version != "v3.0":
            if enabled:
                raise ValueError("append_subtask_to_prompt requires a LeRobot v3 dataset")
            return {}
        if not subtasks_path.exists():
            if enabled:
                raise ValueError(
                    "append_subtask_to_prompt=true requires "
                    f"{subtasks_path}"
                )
            return {}

        label_column = str(
            self._get_data_cfg_value("subtask_prompt_label_column", "local_subtask_text")
        )

        try:
            subtasks = pd.read_parquet(subtasks_path)
            self._subtask_metadata_sha256 = hashlib.sha256(
                subtasks_path.read_bytes()
            ).hexdigest()
        except Exception as exc:
            if enabled:
                raise ValueError(
                    f"Failed to load required subtask labels from {subtasks_path}: {exc}"
                ) from exc
            print(f"Failed to load subtask labels from {subtasks_path}: {exc}")
            return {}

        if "subtask_index" not in subtasks.columns:
            if enabled:
                raise ValueError(f"{subtasks_path} is missing required column 'subtask_index'")
            return {}
        if label_column not in subtasks.columns:
            if enabled:
                raise ValueError(
                    f"{subtasks_path} is missing configured subtask label column "
                    f"{label_column!r}"
                )
            return {}

        labels: dict[int, str] = {}
        for _, row in subtasks.iterrows():
            raw_index = self._coerce_label_value(row["subtask_index"])
            raw_text = self._coerce_label_value(row[label_column])
            if raw_index is None or pd.isna(raw_index):
                continue
            if raw_text is None or pd.isna(raw_text):
                continue
            text = str(raw_text).strip()
            if not text:
                continue
            try:
                subtask_index = self._coerce_subtask_index(raw_index)
            except ValueError as exc:
                if enabled:
                    raise ValueError(
                        f"{subtasks_path} contains an invalid subtask_index {raw_index!r}: {exc}"
                    ) from exc
                continue
            previous_text = labels.get(subtask_index)
            if previous_text is not None and previous_text != text:
                if enabled:
                    raise ValueError(
                        f"{subtasks_path} maps subtask_index {subtask_index} to conflicting "
                        f"labels {previous_text!r} and {text!r}"
                    )
                continue
            labels[subtask_index] = text
        if enabled and not any(
            not subtask_label_is_ignored(label, self.data_cfg)
            for label in labels.values()
        ):
            raise ValueError(
                "append_subtask_to_prompt=true but the configured subtask metadata "
                f"contains no usable labels: {subtasks_path}"
            )
        return labels

    def _validate_subtask_prompt_schema(self) -> None:
        task_id_source = str(
            self._get_data_cfg_value("task_id_prompt_source_column", "task_id")
        )
        if bool(self._get_data_cfg_value("append_task_id_to_prompt", False)) and (
            task_id_source == "subtask_index"
        ):
            raise ValueError(
                "task_id_prompt_source_column=subtask_index is no longer supported; "
                "use append_subtask_to_prompt and subtask_prompt_source_column instead"
            )

        if not bool(self._get_data_cfg_value("append_subtask_to_prompt", False)):
            return
        source_column = str(
            self._get_data_cfg_value("subtask_prompt_source_column", "subtask_index")
        )
        features = self.lerobot_info_meta.get("features", {})
        if source_column not in features:
            raise ValueError(
                "append_subtask_to_prompt=true but the configured source column "
                f"{source_column!r} is absent from {self.dataset_path / LE_ROBOT_INFO_FILENAME}"
            )

    def _validate_selected_subtask_prompt_coverage(self) -> None:
        """Require the selected split itself to contain a usable mapped label."""
        self._selected_subtask_source_ids: list[int] = []
        self._selected_subtask_prompt_frame_count = 0
        self._selected_subtask_prompt_usable_frame_count = 0
        self._selected_subtask_assignment_sha256 = None
        if not bool(self._get_data_cfg_value("append_subtask_to_prompt", False)):
            return

        source_column = str(
            self._get_data_cfg_value("subtask_prompt_source_column", "subtask_index")
        )
        selected_files: dict[Path, set[int]] = defaultdict(set)
        for raw_trajectory_id in self.trajectory_ids:
            trajectory_id = int(raw_trajectory_id)
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=episode_meta["data/chunk_index"],
                file_index=episode_meta["data/file_index"],
            )
            selected_files[parquet_path].add(trajectory_id)

        source_counts: dict[int, int] = defaultdict(int)
        selected_frame_count = 0
        assignment_digest = hashlib.sha256()
        assignment_digest.update(b"starvla-selected-subtask-assignments-v1\0")
        for parquet_path in sorted(selected_files, key=lambda path: path.as_posix()):
            episode_ids = selected_files[parquet_path]
            if not parquet_path.exists():
                raise ValueError(
                    "Cannot validate selected subtask prompt coverage because the data "
                    f"shard is missing: {parquet_path}"
                )
            try:
                table = pq.read_table(
                    parquet_path,
                    columns=["episode_index", source_column],
                    memory_map=True,
                    pre_buffer=False,
                    use_threads=False,
                )
            except Exception as exc:
                raise ValueError(
                    "Failed to read selected subtask prompt columns from "
                    f"{parquet_path}: {exc}"
                ) from exc
            episode_values = table.column("episode_index").to_numpy(
                zero_copy_only=False
            )
            source_values = table.column(source_column).to_numpy(zero_copy_only=False)
            selected_mask = np.isin(
                episode_values,
                np.fromiter(sorted(episode_ids), dtype=np.int64),
            )
            selected_sources = source_values[selected_mask]
            selected_episodes = episode_values[selected_mask]
            selected_frame_count += int(selected_sources.size)
            normalized_sources: dict[object, int] = {}
            for raw_source, count in pd.Series(selected_sources).value_counts(
                dropna=False
            ).items():
                if raw_source is None or pd.isna(raw_source):
                    raise ValueError(
                        "Selected training rows contain a missing value in configured "
                        f"subtask source column {source_column!r}: {parquet_path}"
                    )
                try:
                    source_id = self._coerce_subtask_index(raw_source)
                except ValueError as exc:
                    raise ValueError(
                        "Selected training rows contain an invalid subtask source "
                        f"value {raw_source!r} in {source_column!r}: {parquet_path}"
                    ) from exc
                source_counts[source_id] += int(count)
                normalized_sources[raw_source] = source_id

            normalized_source_values = np.fromiter(
                (normalized_sources[value] for value in selected_sources),
                dtype="<i8",
                count=int(selected_sources.size),
            )
            assignment_pairs = np.column_stack(
                (
                    np.asarray(selected_episodes, dtype="<i8"),
                    normalized_source_values,
                )
            )
            assignment_digest.update(
                int(assignment_pairs.shape[0]).to_bytes(8, "little", signed=False)
            )
            assignment_digest.update(assignment_pairs.tobytes(order="C"))

        expected_frame_count = int(
            np.asarray(self.trajectory_lengths, dtype=np.int64).sum(dtype=np.int64)
        )
        if selected_frame_count != expected_frame_count:
            raise ValueError(
                "Selected subtask prompt coverage scan did not bind every selected frame: "
                f"expected {expected_frame_count}, found {selected_frame_count}"
            )
        missing_source_ids = sorted(set(source_counts) - set(self._subtask_labels))
        if missing_source_ids:
            raise ValueError(
                "Selected training rows contain subtask IDs missing from the configured "
                f"metadata mapping: {missing_source_ids[:20]}"
            )

        usable_frame_count = sum(
            count
            for source_id, count in source_counts.items()
            if not subtask_label_is_ignored(
                self._subtask_labels[source_id],
                self.data_cfg,
            )
        )
        if usable_frame_count <= 0:
            raise ValueError(
                "append_subtask_to_prompt=true, but the selected dataset split contains "
                "zero frames with a usable nonignored subtask label"
            )
        self._selected_subtask_source_ids = sorted(source_counts)
        self._selected_subtask_prompt_frame_count = selected_frame_count
        self._selected_subtask_prompt_usable_frame_count = int(usable_frame_count)
        self._selected_subtask_assignment_sha256 = assignment_digest.hexdigest()

    def _subtask_label_for_index(self, subtask_index) -> str | None:
        try:
            return self._subtask_labels.get(self._coerce_subtask_index(subtask_index))
        except ValueError:
            return None

    def _subtask_prompt_label_for_row(self, label_row: pd.Series) -> str | None:
        """Resolve the configured sample-local subtask ID through this dataset's metadata."""
        source_column = str(
            self._get_data_cfg_value("subtask_prompt_source_column", "subtask_index")
        )
        if source_column not in label_row.index:
            if bool(self._get_data_cfg_value("append_subtask_to_prompt", False)):
                raise SubtaskPromptDataError(
                    "append_subtask_to_prompt=true but sample is missing configured "
                    f"source column {source_column!r}"
                )
            return None
        source_value = self._coerce_label_value(label_row[source_column])
        label = self._subtask_label_for_index(source_value)
        if label is None and bool(
            self._get_data_cfg_value("append_subtask_to_prompt", False)
        ):
            raise SubtaskPromptDataError(
                f"No subtask label mapping for {source_column}={source_value!r} "
                f"in dataset {self.dataset_name}"
            )
        return label

    def _append_prompt_labels_for_row(
        self,
        language,
        label_row: pd.Series,
        *,
        deterministic_key=None,
    ) -> tuple[str, str | None, str | None]:
        """Apply independent subtask and task-ID prompt policies to one sample."""
        subtask_label = self._subtask_prompt_label_for_row(label_row)
        language, _ = append_subtask_label_to_language(
            language,
            subtask_label,
            self.data_cfg,
            deterministic_key=(
                None
                if deterministic_key is None
                else ("subtask", deterministic_key)
            ),
        )

        task_id_label = None
        task_id_source_column = str(
            self._get_data_cfg_value("task_id_prompt_source_column", "task_id")
        )
        if task_id_source_column in label_row.index:
            task_id = self._coerce_label_value(label_row[task_id_source_column])
            task_id_text_column = self._get_data_cfg_value("task_id_prompt_text_column", None)
            if task_id_text_column and str(task_id_text_column) in label_row.index:
                resolved_task_id_label = self._coerce_label_value(
                    label_row[str(task_id_text_column)]
                )
                language, task_id_label = append_resolved_label_to_language(
                    language,
                    resolved_task_id_label,
                    self.data_cfg,
                    deterministic_key=(
                        None
                        if deterministic_key is None
                        else ("task_id", deterministic_key)
                    ),
                )
            else:
                language, task_id_label = append_task_id_label_to_language(
                    language,
                    task_id,
                    self.data_cfg,
                    deterministic_key=(
                        None
                        if deterministic_key is None
                        else ("task_id", deterministic_key)
                    ),
                )

        return language, subtask_label, task_id_label

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"


    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        data = self.get_step_data(trajectory_id, base_index)
        
        # Process all video keys dynamically
        images = []
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]
            
            # Apply image cropping if enabled and the video key is base_view
            # Note: crop_obs_camera functionality has been removed
            
            image = Image.fromarray(image).resize((224, 224))
            images.append(image)
        
        # Get language and action data
        language = data[self.modality_keys["language"][0]][0]
        action = []
        action_pad_masks = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
            pad_mask = data.get(f"{action_key}_is_pad")
            if pad_mask is not None:
                action_pad_masks.append(np.asarray(pad_mask, dtype=bool))
        action = np.concatenate(action, axis=1)

        sample = dict(action=action, image=images, language=language)
        if action_pad_masks:
            sample["action_is_pad"] = np.logical_or.reduce(action_pad_masks)
        return sample

    def get_step_data(
        self,
        trajectory_id: int,
        base_index: int,
        modalities: Sequence[str] | None = None,
    ) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        selected_modalities = list(self.modality_keys.keys()) if modalities is None else list(modalities)
        # TODO @JinhuiYE The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in selected_modalities:
            if modality not in self.modality_keys:
                raise KeyError(f"Unknown modality `{modality}`. Available modalities: {list(self.modality_keys.keys())}")
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                if modality == "action":
                    value, padding_mask = self.get_state_or_action(
                        trajectory_id,
                        modality,
                        key,
                        base_index,
                        return_padding_mask=True,
                    )
                    data[key] = value
                    data[f"{key}_is_pad"] = padding_mask
                else:
                    data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        if self._lerobot_version == "v2.0":
            chunk_index = self.get_episode_chunk(trajectory_id)
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return self._set_current_trajectory_data(trajectory_id, pd.read_parquet(parquet_path))
        if self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)
        raise ValueError(f"Unsupported LeRobot version: {self._lerobot_version}")

    def _set_current_trajectory_data(self, trajectory_id: int, data: pd.DataFrame) -> pd.DataFrame:
        self.curr_traj_id = trajectory_id
        self.curr_traj_data = data
        return data

    def _get_v3_parquet_cache_size(self) -> int:
        configured = self._get_data_cfg_value("lerobot_v3_parquet_cache_size", 1)
        try:
            return max(0, int(configured or 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "lerobot_v3_parquet_cache_size must be a non-negative integer, "
                f"got {configured!r}"
            ) from exc

    def _get_v3_parquet_shard(self, parquet_path: Path):
        """Load a shared v3 shard once per worker and keep a bounded Arrow cache."""
        cache = getattr(self, "_v3_parquet_shard_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._v3_parquet_shard_cache = cache

        cached = cache.pop(parquet_path, None)
        if cached is not None:
            cache[parquet_path] = cached
            return cached

        table = pq.read_table(
            parquet_path,
            memory_map=True,
            pre_buffer=False,
            use_threads=False,
        )
        cache_size = self._get_v3_parquet_cache_size()
        if cache_size > 0:
            cache[parquet_path] = table
            while len(cache) > cache_size:
                cache.popitem(last=False)
        return table

    def close_parquet_cache(self) -> None:
        cache = getattr(self, "_v3_parquet_shard_cache", None)
        if cache is not None:
            cache.clear()
        self.curr_traj_id = None
        self.curr_traj_data = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Spawned workers must create their own bounded cache. Serializing an
        # already-warm Arrow table would copy the complete shard into each worker.
        state["_v3_parquet_shard_cache"] = OrderedDict()
        state["curr_traj_id"] = None
        state["curr_traj_data"] = None
        return state

    def __setstate__(self, state):
        state.setdefault("_v3_data_file_start_indices", {})
        state.setdefault("_v3_parquet_shard_cache", OrderedDict())
        state.setdefault("curr_traj_id", None)
        state.setdefault("curr_traj_data", None)
        self.__dict__.update(state)

    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Get a single trajectory from a shared LeRobot v3 parquet shard."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data

        episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
        parquet_path = self.dataset_path / self.data_path_pattern.format(
            chunk_index=episode_meta["data/chunk_index"],
            file_index=episode_meta["data/file_index"],
        )
        assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
        file_data = self._get_v3_parquet_shard(parquet_path)

        file_key = (
            int(episode_meta["data/chunk_index"]),
            int(episode_meta["data/file_index"]),
        )
        file_start = getattr(self, "_v3_data_file_start_indices", {}).get(file_key)
        dataset_from_index = episode_meta.get("dataset_from_index")
        dataset_to_index = episode_meta.get("dataset_to_index")
        trajectory_table = None
        if (
            file_start is not None
            and dataset_from_index is not None
            and dataset_to_index is not None
        ):
            local_start = int(dataset_from_index) - int(file_start)
            trajectory_length = int(dataset_to_index) - int(dataset_from_index)
            if local_start >= 0 and trajectory_length >= 0 and local_start + trajectory_length <= file_data.num_rows:
                candidate = file_data.slice(local_start, trajectory_length)
                if candidate.num_rows == trajectory_length:
                    candidate_episode_ids = candidate.column("episode_index").to_numpy(zero_copy_only=False)
                    if np.all(candidate_episode_ids == trajectory_id):
                        trajectory_table = candidate

        if trajectory_table is None:
            trajectory_table = file_data.filter(
                pc.equal(file_data.column("episode_index"), int(trajectory_id))
            )

        expected_length = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
        if trajectory_table.num_rows != expected_length:
            raise RuntimeError(
                f"LeRobot v3 episode {trajectory_id} resolved to {trajectory_table.num_rows} rows, "
                f"expected {expected_length} in {parquet_path}"
            )

        # Convert only the selected episode, not the complete shared shard.
        trajectory_data = trajectory_table.to_pandas().reset_index(drop=True)
        return self._set_current_trajectory_data(trajectory_id, trajectory_data)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the data file index for a LeRobot v3 episode."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
        return_padding_mask: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        step_indices = np.asarray(step_indices, dtype=np.int64)
        if max_length <= 0:
            raise ValueError(f"Cannot retrieve from an empty trajectory: {max_length=}")

        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)

        clamped_indices = np.clip(step_indices, 0, max_length - 1)
        output = np.asarray(array[clamped_indices]).copy()

        # If there exists some padding, apply the requested padding policy.
        if padding_positions.any():
            if padding_strategy == "first_last":
                # The clamped indices already repeat the nearest valid frame.
                pass
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        if return_padding_mask:
            return output, padding_positions.astype(bool, copy=False)
        return output

    def _get_lerobot_v3_video_metadata(
        self,
        trajectory_id: int,
        key: str,
    ) -> tuple[str, dict[str, int | float]]:
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
        video_meta = episode_meta.get("videos", {}).get(original_key)
        if not isinstance(video_meta, dict):
            raise KeyError(
                "LeRobot v3 episode metadata is missing camera binding "
                f"videos/{original_key} for trajectory {trajectory_id}."
            )
        missing = {"chunk_index", "file_index", "from_timestamp"} - set(video_meta)
        if missing:
            raise KeyError(
                "LeRobot v3 episode metadata has an incomplete camera binding for "
                f"trajectory {trajectory_id}, camera {original_key!r}: missing {sorted(missing)}."
            )
        return original_key, video_meta

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            original_key, video_meta = self._get_lerobot_v3_video_metadata(
                trajectory_id,
                key,
            )
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=int(video_meta["chunk_index"]),
                file_index=int(video_meta["file_index"]),
            )
        else:
            raise ValueError(f"Unsupported LeRobot version: {self._lerobot_version}")
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]
        if self._lerobot_version == "v3.0":
            _, video_meta = self._get_lerobot_v3_video_metadata(trajectory_id, key)
            video_timestamp = video_timestamp + float(video_meta["from_timestamp"])

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_video_by_step_indices(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        step_indices = np.asarray(step_indices, dtype=np.int64)
        trajectory_index = self.get_trajectory_index(trajectory_id)
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        video_timestamp = timestamp[step_indices]
        if self._lerobot_version == "v3.0":
            _, video_meta = self._get_lerobot_v3_video_metadata(trajectory_id, key)
            video_timestamp = video_timestamp + float(video_meta["from_timestamp"])

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
        return_padding_mask: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy=(
                "first_last"
                if modality == "action" or state_or_action_cfg.absolute
                else "zero"
            ),
            return_padding_mask=return_padding_mask,
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):
            task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the data structure to save
        statistics_data = {}
        
        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)
        
        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}
        
        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'action') and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action
            
            # Filter to only include used action keys and reorder: non-gripper first, gripper last
            non_gripper_keys = []
            gripper_keys = []
            
            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)
            
            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys
            
            filtered_action_stats = {}
            for key in reordered_keys:
                filtered_action_stats[key] = action_stats[key]
            
            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                combined_action_stats = combine_modality_stats(filtered_action_stats)
                
                # Add mask field based on whether it's gripper or not
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action, filtered_action_stats.keys()
                )
                combined_action_stats["mask"] = mask
                
                tag_stats["action"] = combined_action_stats
        
        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, 'state') and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state
            
            # Filter to only include used state keys, optionally reorder gripper to end
            non_gripper_keys = []
            gripper_keys = []
            
            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)
            
            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys
            
            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]
            
            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats
        
        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)
        
        statistics_data[tag] = tag_stats
        
        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(used_action_keys)}")
        print(f"Used state keys (reordered): {list(used_state_keys)}")

    def dataset_provenance(self) -> dict:
        """Return the exact input/statistics binding used by this dataset."""

        info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        split_selection = getattr(self, "_episode_split_selection", None)
        full_ids = np.asarray(
            getattr(self, "_full_trajectory_ids", self.trajectory_ids),
            dtype=np.int64,
        )
        full_lengths = np.asarray(
            getattr(self, "_full_trajectory_lengths", self.trajectory_lengths),
            dtype=np.int64,
        )
        raw_action_delta_mappings = self._get_data_cfg_value(
            "action_delta_mappings", {}
        ) or {}
        action_delta_mappings = {
            str(key): {
                "state_key": str(value.get("state_key")),
                "state_indices": [int(index) for index in value.get("state_indices", [])],
                "delta_mask": [bool(flag) for flag in value.get("delta_mask", [])],
            }
            for key, value in raw_action_delta_mappings.items()
        }
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": str(self.dataset_path.resolve()),
            "lerobot_version": self._lerobot_version,
            "info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
            "full_catalog_episode_count": int(full_ids.size),
            "full_catalog_frame_count": int(full_lengths.sum(dtype=np.int64)),
            "selected_episode_count": int(len(self.trajectory_ids)),
            "selected_frame_count": int(
                np.asarray(self.trajectory_lengths, dtype=np.int64).sum(dtype=np.int64)
            ),
            # Backward-compatible alias; the explicit scope/count fields below
            # remove the old ambiguity once a holdout split is active.
            "total_frames": getattr(self, "_statistics_total_frames", None),
            "statistics_frame_count": getattr(self, "_statistics_total_frames", None),
            "statistics_scope": getattr(
                self,
                "_statistics_scope",
                "unknown",
            ),
            "statistics_source_path": str(
                getattr(self, "_statistics_source_path", "")
            ),
            "statistics_source_sha256": getattr(
                self,
                "_statistics_source_sha256",
                None,
            ),
            "statistics_effective_sha256": getattr(
                self,
                "_statistics_effective_sha256",
                None,
            ),
            "statistics_quantiles_synthesized": bool(
                getattr(self, "_statistics_quantiles_synthesized", False)
            ),
            "frame_catalog_counts": getattr(
                self,
                "_dataset_catalog_counts",
                None,
            ),
            "frame_catalog_fingerprint": getattr(
                self,
                "_dataset_catalog_fingerprint",
                None,
            ),
            "episode_catalog_binding": getattr(
                self,
                "_episode_catalog_binding",
                None,
            ),
            "action_representation": {
                "schema_version": 1,
                "action_type": str(
                    self._get_data_cfg_value("action_type", "dataset_native")
                ),
                "delta_anchor": str(
                    self._get_data_cfg_value(
                        "action_delta_anchor", "chunk_start_state"
                    )
                ),
                "gripper_action_type": str(
                    self._get_data_cfg_value("gripper_action_type", "absolute")
                ),
                "mappings": action_delta_mappings,
                "normalization": str(
                    self._get_data_cfg_value(
                        "state_action_normalization", Q01_Q99_UNCLIPPED
                    )
                ),
                "contract_sha256": getattr(
                    self, "_action_representation_contract_sha256", None
                ),
                "statistics_path": str(
                    getattr(self, "_action_representation_statistics_path", "")
                ),
                "statistics_sha256": getattr(
                    self, "_action_representation_statistics_sha256", None
                ),
                "statistics_scope": getattr(
                    self,
                    "_action_representation_statistics_scope",
                    None,
                ),
                "statistics_schema": getattr(
                    self,
                    "_action_representation_statistics_schema",
                    None,
                ),
                "local_split_manifest_sha256": getattr(
                    self,
                    "_action_representation_local_split_manifest_sha256",
                    None,
                ),
                "statistics_population": copy.deepcopy(
                    getattr(
                        self,
                        "_action_representation_statistics_population",
                        None,
                    )
                ),
            },
            "frozen_train_view": copy.deepcopy(
                getattr(
                    self,
                    "_frozen_train_view_provenance",
                    {"enabled": False},
                )
            ),
            "subtask_prompt": {
                "enabled": bool(
                    self._get_data_cfg_value("append_subtask_to_prompt", False)
                ),
                "source_column": str(
                    self._get_data_cfg_value(
                        "subtask_prompt_source_column", "subtask_index"
                    )
                ),
                "label_column": str(
                    self._get_data_cfg_value(
                        "subtask_prompt_label_column", "local_subtask_text"
                    )
                ),
                "append_probability": subtask_prompt_append_probability(
                    self.data_cfg
                ),
                "separator": str(
                    self._get_data_cfg_value("subtask_prompt_separator", " | ")
                ),
                "ignored_labels": list(subtask_prompt_ignored_labels(self.data_cfg)),
                "metadata_path": (
                    str(self._subtask_metadata_path.resolve())
                    if getattr(self, "_subtask_metadata_path", None) is not None
                    else None
                ),
                "metadata_sha256": getattr(
                    self, "_subtask_metadata_sha256", None
                ),
                "label_count": len(getattr(self, "_subtask_labels", {})),
                "usable_label_count": sum(
                    not subtask_label_is_ignored(label, self.data_cfg)
                    for label in getattr(self, "_subtask_labels", {}).values()
                ),
                "selected_source_ids": list(
                    getattr(self, "_selected_subtask_source_ids", [])
                ),
                "selected_frame_count": int(
                    getattr(self, "_selected_subtask_prompt_frame_count", 0)
                ),
                "selected_usable_frame_count": int(
                    getattr(self, "_selected_subtask_prompt_usable_frame_count", 0)
                ),
                "selected_assignment_sha256": getattr(
                    self, "_selected_subtask_assignment_sha256", None
                ),
            },
            "episode_split": (
                split_selection.provenance()
                if split_selection is not None
                else {
                    "enabled": False,
                    "normalization_statistics_scope": "full_dataset_catalog",
                }
            ),
            "gpu_decode_frame_index_cache_version": GPU_DECODE_FRAME_INDEX_CACHE_DIRNAME,
        }


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                
                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed
                
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(
        self,
        trajectory_id: int,
        base_index: int,
        modalities: Sequence[str] | None = None,
    ) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        selected_modalities = list(self.modality_keys.keys()) if modalities is None else list(modalities)
        # Get the data for all modalities
        for modality in selected_modalities:
            if modality not in self.modality_keys:
                raise KeyError(f"Unknown modality `{modality}`. Available modalities: {list(self.modality_keys.keys())}")
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


def apply_transforms_for_present_keys(transforms, data: dict) -> dict:
    """Apply only the transforms whose declared keys are present in the sample.

    This lets callers skip redundant video decoding while still applying state/action
    normalization on partial samples.
    """
    if transforms is None:
        return data
    if isinstance(transforms, ComposedModalityTransform):
        for transform in transforms.transforms:
            apply_to = getattr(transform, "apply_to", None)
            if apply_to and not all(key in data for key in apply_to):
                continue
            data = transform(data)
        return data
    return transforms(data)


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics

def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.
    
    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.
        
    Returns:
        dict: Combined statistics
    """
    combined_stats = {
        "mean": [],
        "std": [],
        "max": [],
        "min": [],
        "q01": [],
        "q99": []
    }
    
    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object
        
        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, 'tolist'):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))
    
    return combined_stats

def generate_action_mask_for_used_keys(action_modalities: dict, used_action_keys_ordered) -> list[bool]:
    """
    Generate mask based on action modalities, but only for used keys.
    Gripper-related are False, others are True.
    
    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys in the correct order.
        
    Returns:
        list[bool]: List of mask values
    """
    mask = []
    
    # Generate mask in the same order as the statistics were combined
    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]
            
            # Get dimension count from shape
            if hasattr(subkey_config, 'shape') and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1
            
            # Check if it's gripper-related
            is_gripper = "gripper" in subkey.lower()
            
            # Generate mask value for each dimension
            for _ in range(dim_count):
                mask.append(not is_gripper)  # gripper is False, others are True
    
    return mask

def get_used_modality_keys(modality_keys: dict) -> tuple[list[str], list[str]]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys: list[str] = []
    used_state_keys: list[str] = []
    seen_action_keys: set[str] = set()
    seen_state_keys: set[str] = set()
    
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            if clean_key not in seen_action_keys:
                used_action_keys.append(clean_key)
                seen_action_keys.add(clean_key)
    
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            if clean_key not in seen_state_keys:
                used_state_keys.append(clean_key)
                seen_state_keys.add(clean_key)
    
    return used_action_keys, used_state_keys

class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    # Keep a class-level default so older pickled instances still have a sane
    # fallback after reloads or worker respawns.
    gpu_video_decode_on_rank = False
    cpu_video_decode_drop_worker_images = False

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        primary_dataset_flags: Sequence[bool] | None = None,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        with_state: bool = False,
        resolution_size: int = 224,
        video_resolution_size: int = 256,
        video_frame_stride: int = 1,
        video_target_shift_steps: int = 0,
        gpu_video_decode_on_rank: bool = False,
        cpu_video_decode_drop_worker_images: bool = False,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        metadata_config = metadata_config or {}
        requested_epoch_sampling_strategy = str(
            metadata_config.get("epoch_sampling_strategy", "with_replacement")
        ).strip().lower()
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        if (
            primary_dataset_flags is not None
            and len(primary_dataset_flags) != len(data_mixture)
        ):
            raise ValueError(
                "primary_dataset_flags must have one exact boolean per "
                "configured dataset"
            )
        retained_primary_flags: list[bool] = []
        for dataset_index, (dataset, weight) in enumerate(data_mixture):
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
            if primary_dataset_flags is not None:
                flag = primary_dataset_flags[dataset_index]
                if type(flag) is not bool:
                    raise ValueError(
                        "primary_dataset_flags must contain exact booleans"
                    )
                retained_primary_flags.append(flag)
        
        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")
        
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        self.with_state = with_state
        self.resolution_size = resolution_size
        self.video_resolution_size = video_resolution_size
        self.video_frame_stride = max(int(video_frame_stride), 1)
        self.video_target_shift_steps = max(int(video_target_shift_steps), 0)
        self.gpu_video_decode_on_rank = bool(gpu_video_decode_on_rank)
        self.cpu_video_decode_drop_worker_images = bool(cpu_video_decode_drop_worker_images)
        self.use_action_validity_prefix_mask = bool(
            metadata_config.get("use_action_validity_prefix_mask", False)
        )
        self.action_validity_fail_closed = bool(
            metadata_config.get("action_validity_fail_closed", False)
        )
        self.action_validity_invalid_run_length = max(
            1,
            int(metadata_config.get("action_validity_invalid_run_length", 3)),
        )
        timing_env = str(os.environ.get("STARVLA_DATASET_TIMING", "0")).lower()
        self.dataset_timing_enabled = timing_env in {"1", "true", "yes", "on"} or bool(
            metadata_config.get("dataset_timing_logging", False)
        )
        self.dataset_timing_every = max(
            1,
            int(
                os.environ.get(
                    "STARVLA_DATASET_TIMING_EVERY",
                    metadata_config.get("dataset_timing_every", 200),
                )
            ),
        )
        self.dataset_timing_slow_threshold = float(
            os.environ.get(
                "STARVLA_DATASET_TIMING_SLOW_SECONDS",
                metadata_config.get("dataset_timing_slow_seconds", 2.0),
            )
        )
        self._dataset_timing_counter = 0

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")

        # 2. Dataset sampling weights
        self._raw_dataset_sampling_weights = np.asarray(
            dataset_sampling_weights, dtype=np.float64
        )
        self._dataset_sampling_weights = self._raw_dataset_sampling_weights.copy()
        
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        
        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)
        
        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print(f"Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths
            
            # Check for zero or negative weights before normalization
            if np.any(trajectory_sampling_weights <= 0):
                print(f"Warning: Dataset {i} has zero or negative trajectory weights")
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)
            
            # Normalize weights
            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                print(f"Error: Dataset {i} has invalid trajectory weights sum: {weights_sum}")
                # Fallback to equal weights
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum
            
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        if primary_dataset_flags is not None:
            self._primary_dataset_indices = np.asarray(
                retained_primary_flags, dtype=np.bool_
            )
            if (
                not np.any(self._primary_dataset_indices)
                and requested_epoch_sampling_strategy
                != "all_sources_exhaustive"
            ):
                raise ValueError(
                    "At least one configured dataset must declare primary=true"
                )
        else:
            # Backward-compatible inference for the static named-mixture
            # catalog. Config-native sources must declare primary explicitly.
            self._primary_dataset_indices = (
                np.array(dataset_sampling_weights) == 1.0
            )
            if not np.any(self._primary_dataset_indices):
                print(
                    "Warning: No dataset with weight 1.0 found. "
                    f"Original weights: {dataset_sampling_weights}"
                )
                max_weight = max(dataset_sampling_weights)
                self._primary_dataset_indices = (
                    np.array(dataset_sampling_weights) == max_weight
                )
                print(
                    f"Using datasets with maximum weight {max_weight} as "
                    f"primary: {self._primary_dataset_indices}"
                )

        # ``with_replacement`` preserves the historical behavior.
        #
        # ``primary_exhaustive`` exhausts only the explicitly selected primary
        # sources and treats the remaining sources as weighted replay.
        #
        # ``all_sources_exhaustive`` is the strict production contract: every
        # eligible row from every retained configured source appears exactly
        # once per logical epoch.  Source weights and primary flags do not
        # change coverage in that mode.
        self.epoch_sampling_strategy = requested_epoch_sampling_strategy
        if self.epoch_sampling_strategy not in {
            "with_replacement",
            "primary_exhaustive",
            "all_sources_exhaustive",
        }:
            raise ValueError(
                "Unsupported epoch_sampling_strategy "
                f"{self.epoch_sampling_strategy!r}; expected "
                "'with_replacement', 'primary_exhaustive', or "
                "'all_sources_exhaustive'."
            )
        self.epoch_sampling_algorithm_version = {
            "with_replacement": "legacy_with_replacement_v1",
            "primary_exhaustive": "primary_exhaustive_affine_v1",
            "all_sources_exhaustive": "all_sources_exhaustive_affine_v1",
        }[self.epoch_sampling_strategy]
        if self._uses_exhaustive_epoch_sampling() and self.balance_dataset_weights:
            raise ValueError(
                f"epoch_sampling_strategy={self.epoch_sampling_strategy} requires "
                "balance_dataset_weights=false. Dataset sizes already define "
                "complete logical-epoch coverage."
            )
        self.fail_on_sample_error = bool(
            metadata_config.get(
                "fail_on_sample_error",
                self._uses_exhaustive_epoch_sampling(),
            )
        )
        self._epoch_dataset_counts = self._build_epoch_dataset_counts()
        self._epoch_dataset_offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(self._epoch_dataset_counts, dtype=np.int64),
            )
        )
        self._epoch_permutation_cache: dict[tuple[int, int, int], tuple[int, int]] = {}

        # A normal integer is copied into persistent DataLoader workers and
        # never changes there.  Keep the logical epoch in shared memory so a
        # main-process set_epoch() is visible to already-running spawn or
        # forkserver workers.
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

        # Set the epoch before sampling the first epoch.
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    def __setstate__(self, state):
        state.setdefault("gpu_video_decode_on_rank", False)
        state.setdefault("cpu_video_decode_drop_worker_images", False)
        state.setdefault("dataset_timing_enabled", False)
        state.setdefault("dataset_timing_every", 200)
        state.setdefault("dataset_timing_slow_threshold", 2.0)
        state.setdefault("_dataset_timing_counter", 0)
        state.setdefault("epoch_sampling_strategy", "with_replacement")
        state.setdefault(
            "epoch_sampling_algorithm_version",
            "legacy_with_replacement_v1",
        )
        state.setdefault("fail_on_sample_error", False)
        state.setdefault("action_validity_fail_closed", False)
        state.setdefault("_epoch_permutation_cache", {})
        self.__dict__.update(state)
        if not hasattr(self, "_shared_epoch"):
            self._shared_epoch = torch.tensor(
                int(getattr(self, "epoch", 0)), dtype=torch.int64
            ).share_memory_()

    def _build_epoch_dataset_counts(self) -> np.ndarray:
        """Return deterministic per-dataset slots for one logical epoch."""
        if self.epoch_sampling_strategy == "all_sources_exhaustive":
            return self.dataset_lengths.astype(np.int64, copy=True)
        if self.epoch_sampling_strategy != "primary_exhaustive":
            return np.zeros(len(self.datasets), dtype=np.int64)

        counts = np.zeros(len(self.datasets), dtype=np.int64)
        counts[self.primary_dataset_indices] = self.dataset_lengths[
            self.primary_dataset_indices
        ].astype(np.int64)
        primary_total = int(counts.sum(dtype=np.int64))
        if primary_total <= 0:
            raise ValueError(
                "primary_exhaustive sampling requires at least one non-empty "
                "primary dataset."
            )

        replay_mask = ~self.primary_dataset_indices
        if not np.any(replay_mask):
            return counts

        primary_weight = float(
            self._raw_dataset_sampling_weights[self.primary_dataset_indices].sum()
        )
        replay_weights = self._raw_dataset_sampling_weights[replay_mask]
        replay_weight = float(replay_weights.sum())
        if primary_weight <= 0.0 or replay_weight <= 0.0:
            return counts

        # Example: primary weight 1.0 and replay weight 0.25 means replay gets
        # 25 rows for every 100 exhaustive primary rows (20% of all slots).
        replay_total = int(round(primary_total * replay_weight / primary_weight))
        if replay_total <= 0:
            return counts

        exact = replay_total * replay_weights / replay_weight
        allocated = np.floor(exact).astype(np.int64)
        remainder = replay_total - int(allocated.sum(dtype=np.int64))
        if remainder > 0:
            fractional = exact - allocated
            # Stable tie-breaking by dataset index keeps the schedule
            # reproducible across Python and NumPy versions.
            replay_indices = np.flatnonzero(replay_mask)
            order = sorted(
                range(len(replay_indices)),
                key=lambda position: (-float(fractional[position]), int(replay_indices[position])),
            )
            for position in order[:remainder]:
                allocated[position] += 1
        counts[replay_mask] = allocated
        return counts

    def _uses_exhaustive_epoch_sampling(self) -> bool:
        """Whether indices form a deterministic, finite epoch schedule."""

        return getattr(
            self, "epoch_sampling_strategy", "with_replacement"
        ) in {
            "primary_exhaustive",
            "all_sources_exhaustive",
        }

    def _log_dataset_timing(
        self,
        *,
        original_index: int,
        final_index: int,
        attempts: int,
        dataset: LeRobotSingleDataset,
        trajectory_name: int,
        step: int,
        timings: dict[str, float],
    ) -> None:
        if not self.dataset_timing_enabled:
            return
        self._dataset_timing_counter += 1
        total_time = float(timings.get("total", 0.0))
        should_log = (
            self._dataset_timing_counter % self.dataset_timing_every == 0
            or total_time >= self.dataset_timing_slow_threshold
            or attempts > 1
        )
        if not should_log:
            return

        try:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else "main"
        except Exception:
            worker_id = "unknown"
        timing_parts = " ".join(
            f"{key}={float(value):.4f}s"
            for key, value in sorted(timings.items())
            if isinstance(value, (int, float, np.floating))
        )
        print(
            "[dataset_timing] "
            f"pid={os.getpid()} worker={worker_id} original_index={original_index} "
            f"final_index={final_index} attempts={attempts} dataset={dataset.dataset_name} "
            f"trajectory={trajectory_name} step={step} {timing_parts}",
            file=sys.stderr,
            flush=True,
        )

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    @property
    def epoch_dataset_counts(self) -> np.ndarray:
        """Number of slots contributed by each dataset to one logical epoch."""
        return self._epoch_dataset_counts.copy()

    @property
    def current_epoch(self) -> int:
        """Return the worker-visible logical epoch."""
        shared_epoch = getattr(self, "_shared_epoch", None)
        if shared_epoch is not None:
            return int(shared_epoch.item())
        return int(getattr(self, "epoch", 0))

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}")
        epoch = int(epoch)
        self.epoch = epoch
        shared_epoch = getattr(self, "_shared_epoch", None)
        if shared_epoch is not None:
            shared_epoch.fill_(epoch)
        for dataset in self.datasets:
            child_set_epoch = getattr(dataset, "set_epoch", None)
            if callable(child_set_epoch):
                child_set_epoch(epoch)

    @staticmethod
    def _affine_permutation_parameters(length: int, seed: int) -> tuple[int, int]:
        """Build a cheap stateless permutation ``(a*x+b) mod length``."""
        if length <= 1:
            return 1, 0
        multiplier = int(seed % length)
        if multiplier == 0:
            multiplier = 1
        while math.gcd(multiplier, length) != 1:
            multiplier += 1
            if multiplier >= length:
                multiplier = 1
        offset = int((seed >> 64) % length)
        return multiplier, offset

    def _permuted_dataset_step_index(
        self,
        *,
        dataset_index: int,
        local_slot: int,
    ) -> int:
        dataset_length = int(self.dataset_lengths[dataset_index])
        if dataset_length <= 0:
            raise RuntimeError(f"Dataset index {dataset_index} is empty.")
        cycle, position = divmod(int(local_slot), dataset_length)
        epoch = self.current_epoch if self.mode == "train" else 0
        cache_key = (epoch, int(dataset_index), int(cycle))
        parameters = self._epoch_permutation_cache.get(cache_key)
        if parameters is None:
            seed = safe_hash(
                (
                    self.epoch_sampling_strategy,
                    epoch,
                    int(dataset_index),
                    int(cycle),
                    int(self.seed),
                )
            )
            parameters = self._affine_permutation_parameters(dataset_length, seed)
            if len(self._epoch_permutation_cache) >= 128:
                self._epoch_permutation_cache.clear()
            self._epoch_permutation_cache[cache_key] = parameters
        multiplier, offset = parameters
        return int((multiplier * position + offset) % dataset_length)

    def _permuted_epoch_slot(self, index: int) -> int:
        """Map a loader index bijectively across the complete epoch schedule."""
        epoch_length = int(self._epoch_dataset_offsets[-1])
        if epoch_length <= 1:
            return int(index)
        epoch = self.current_epoch if self.mode == "train" else 0
        cache_key = (epoch, -1, 0)
        parameters = self._epoch_permutation_cache.get(cache_key)
        if parameters is None:
            seed = safe_hash(
                (
                    f"{self.epoch_sampling_strategy}_epoch",
                    epoch,
                    int(self.seed),
                )
            )
            parameters = self._affine_permutation_parameters(epoch_length, seed)
            if len(self._epoch_permutation_cache) >= 128:
                self._epoch_permutation_cache.clear()
            self._epoch_permutation_cache[cache_key] = parameters
        multiplier, offset = parameters
        return int((multiplier * int(index) + offset) % epoch_length)

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        if self._uses_exhaustive_epoch_sampling():
            index = int(index)
            epoch_length = int(self._epoch_dataset_offsets[-1])
            if index < 0 or index >= epoch_length:
                raise IndexError(
                    f"Mixture index {index} is outside exhaustive epoch [0, {epoch_length})."
                )
            index = self._permuted_epoch_slot(index)
            dataset_index = int(
                np.searchsorted(self._epoch_dataset_offsets, index, side="right") - 1
            )
            local_slot = index - int(self._epoch_dataset_offsets[dataset_index])
            dataset = self.datasets[dataset_index]
            single_step_index = self._permuted_dataset_step_index(
                dataset_index=dataset_index,
                local_slot=local_slot,
            )
            trajectory_id, base_index = dataset.all_steps[single_step_index]
            return dataset, trajectory_id, base_index

        # Set seed
        seed = (
            index
            if self.mode != "train"
            else safe_hash((self.current_epoch, index, self.seed))
        )
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        # trajectory_index = rng.choice(
        #     len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        # )
        # trajectory_id = dataset.trajectory_ids[trajectory_index]

        # # Sample step
        # base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        # return dataset, trajectory_id, base_index
        single_step_index = rng.choice(len(dataset.all_steps))
        trajectory_id, base_index = dataset.all_steps[single_step_index]
        return dataset, trajectory_id, base_index

    def _build_action_validity_mask(
        self,
        dataset: LeRobotSingleDataset,
        *,
        step: int,
        action: np.ndarray,
        action_is_pad: np.ndarray | None,
    ) -> np.ndarray | None:
        if not bool(cfg_get(dataset.data_cfg, "use_action_validity_prefix_mask", self.use_action_validity_prefix_mask)):
            return None
        fail_closed = bool(
            cfg_get(
                dataset.data_cfg,
                "action_validity_fail_closed",
                getattr(self, "action_validity_fail_closed", False),
            )
        )

        def _unsafe(message: str, *, return_none: bool = False):
            if fail_closed:
                raise ValueError(
                    "Fail-closed action-validity supervision rejected sample: "
                    + message
                )
            if return_none:
                return None
            return np.ones(action.shape, dtype=bool)

        if dataset.curr_traj_data is None:
            return _unsafe("trajectory metadata is unavailable.", return_none=True)
        action_keys = dataset.modality_keys.get("action")
        if not action_keys:
            return _unsafe("action modality metadata is unavailable.", return_none=True)
        if action.ndim != 2 or action.shape[0] <= 0 or action.shape[1] <= 0:
            return _unsafe(
                f"action tensor must be a non-empty rank-2 array, got {action.shape}."
            )

        action_key = action_keys[0]
        if action_key not in dataset.delta_indices:
            return _unsafe(
                f"action offsets are missing for configured key {action_key!r}."
            )
        action_step_indices = (
            np.asarray(dataset.delta_indices[action_key], dtype=np.int64)
            + int(step)
        )
        if action_step_indices.shape[0] != action.shape[0]:
            return _unsafe(
                "action-offset count does not match the action horizon: "
                f"{action_step_indices.shape[0]} != {action.shape[0]}."
            )

        max_index = len(dataset.curr_traj_data) - 1
        if max_index < 0:
            return _unsafe("trajectory metadata is empty.")
        clamped_indices = np.clip(action_step_indices, 0, max_index)
        label_rows = dataset.curr_traj_data.iloc[clamped_indices]

        valid_flags = None
        label_key = cfg_get(dataset.data_cfg, "action_validity_label_key", None)
        positive_is_valid = cfg_get(dataset.data_cfg, "action_validity_positive_is_valid", None)
        if fail_closed and (
            label_key is None or not str(label_key).strip()
        ):
            return _unsafe(
                "action_validity_label_key must be configured explicitly."
            )
        if label_key:
            label_key = str(label_key)
            if label_key in label_rows.columns:
                if positive_is_valid is None:
                    positive_is_valid = label_key not in {
                        "mistake",
                        "mistake_label",
                        "is_mistake",
                        "failure",
                        "error",
                    }
                if fail_closed:
                    try:
                        label_values = np.asarray(
                            label_rows[label_key].to_numpy(),
                            dtype=np.float64,
                        ).reshape(-1)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError(
                            "Fail-closed action-validity supervision rejected "
                            f"nonnumeric labels in {label_key!r}."
                        ) from exc
                    if not np.isfinite(label_values).all():
                        return _unsafe(
                            f"configured label {label_key!r} contains NaN/Inf."
                        )
                    if not np.isin(label_values, (0.0, 1.0)).all():
                        return _unsafe(
                            f"configured label {label_key!r} contains values "
                            "outside exact 0/1."
                        )
                valid_flags = valid_flags_from_label_values(
                    label_rows[label_key].to_numpy(),
                    positive_is_valid=bool(positive_is_valid),
                )
            elif fail_closed:
                return _unsafe(
                    f"configured label column {label_key!r} is missing."
                )
        else:
            for candidate_key, candidate_positive_is_valid in (
                ("valid_state", True),
                ("sub_task_id", True),
                ("mistake", False),
                ("mistake_label", False),
                ("is_mistake", False),
                ("failure", False),
                ("error", False),
            ):
                if candidate_key in label_rows.columns:
                    valid_flags = valid_flags_from_label_values(
                        label_rows[candidate_key].to_numpy(),
                        positive_is_valid=candidate_positive_is_valid,
                    )
                    break
        if valid_flags is None:
            return _unsafe("no usable action-validity labels were found.")
        if valid_flags.shape != (action.shape[0],):
            return _unsafe(
                "validity-label count does not match the action horizon: "
                f"{valid_flags.shape} != {(action.shape[0],)}."
            )

        invalid_run_length = int(
            cfg_get(
                dataset.data_cfg,
                "action_validity_invalid_run_length",
                self.action_validity_invalid_run_length,
            )
        )
        return build_action_mask_from_valid_flags(
            valid_flags,
            invalid_run_length=max(1, invalid_run_length),
            action_dim=int(action.shape[1]),
            action_is_pad=action_is_pad,
        ).astype(bool, copy=False)
    
    def resize_video_opencv(self, video: np.ndarray, N: int) -> np.ndarray:
        """
        使用OpenCV将视频调整为(N, N)大小
        
        参数:
            video: 形状为(T, H, W, C)的numpy数组
            N: 目标尺寸
            
        返回:
            形状为(T, N, N, C)的numpy数组
        """
        T, H, W, C = video.shape
        
        # 创建结果数组
        resized_video = np.zeros((T, N, N, C), dtype=video.dtype)
        
        # 逐帧调整大小
        for t in range(T):
            frame = video[t]
            resized_frame = cv2.resize(frame, (N, N), interpolation=cv2.INTER_LINEAR)
            resized_video[t] = resized_frame
        
        return resized_video

    def _build_shifted_video_views(
        self,
        dataset: LeRobotSingleDataset,
        trajectory_name: int,
        step: int,
        video_horizon: int,
        build_images: bool = True,
    ) -> tuple[list[np.ndarray], list[Image.Image]]:
        if self.video_target_shift_steps <= 0:
            raise ValueError("video_target_shift_steps must be positive to build shifted video targets")
        if video_horizon <= self.video_target_shift_steps:
            raise ValueError(
                f"video_horizon ({video_horizon}) must be greater than video_target_shift_steps ({self.video_target_shift_steps})"
            )

        context_horizon = video_horizon - self.video_target_shift_steps
        union_offsets = np.arange(
            -(context_horizon - 1),
            self.video_target_shift_steps + 1,
            dtype=np.int64,
        ) * self.video_frame_stride

        videos, images = [], []
        for video_key in dataset.modality_keys["video"]:
            merged_video = dataset.get_video_by_step_indices(
                trajectory_name,
                video_key,
                step + union_offsets,
            )
            merged_video = self.resize_video_opencv(merged_video, self.video_resolution_size)
            videos.append(merged_video)
            if build_images:
                images.append(
                    Image.fromarray(merged_video[context_horizon - 1]).resize((self.resolution_size, self.resolution_size))
                )

        return videos, images

    def _build_video_decode_specs(
        self,
        dataset: LeRobotSingleDataset,
        trajectory_name: int,
        step: int,
        step_offsets: np.ndarray | None = None,
    ) -> list[dict]:
        trajectory_index = dataset.get_trajectory_index(trajectory_name)
        assert self.video_frame_stride >= 1
        assert self.video_resolution_size > 0
        assert self.resolution_size > 0
        assert self.video_target_shift_steps >= 0
        assert dataset.curr_traj_data is not None, f"No data found for {trajectory_name=}"
        assert "timestamp" in dataset.curr_traj_data.columns, f"No timestamp found in {trajectory_name=}"
        timestamp: np.ndarray | None = None
        cache_path = get_gpu_decode_frame_index_cache_path(dataset.dataset_path, trajectory_name)
        cached_frame_indices = None
        if cache_path.exists():
            try:
                cached_frame_indices = _load_gpu_decode_frame_index_cache(cache_path.as_posix())
            except Exception:
                cached_frame_indices = None

        expected_length = int(dataset.trajectory_lengths[trajectory_index])
        specs = []
        for video_key in dataset.modality_keys["video"]:
            if step_offsets is None:
                indices = np.asarray(dataset.delta_indices[video_key], dtype=np.int64) + step
            else:
                indices = np.asarray(step_offsets, dtype=np.int64) + step
            indices = np.maximum(indices, 0)
            indices = np.minimum(indices, dataset.trajectory_lengths[trajectory_index] - 1)
            video_subkey = video_key.replace("video.", "")
            if cached_frame_indices is not None:
                cached_length = cached_frame_indices.get("__length__")
                cached_video_indices = cached_frame_indices.get(video_subkey)
                if (
                    cached_length is not None
                    and int(np.asarray(cached_length).reshape(-1)[0]) == expected_length
                    and cached_video_indices is not None
                    and cached_video_indices.shape[0] == expected_length
                ):
                    specs.append(
                        {
                            "video_path": dataset.get_video_path(
                                trajectory_name, video_subkey
                            ).as_posix(),
                            "frame_indices": np.asarray(cached_video_indices[indices], dtype=np.int64),
                        }
                    )
                    continue
            if timestamp is None:
                timestamp = dataset.curr_traj_data["timestamp"].to_numpy()
            video_timestamps = np.asarray(timestamp[indices], dtype=np.float32)
            if dataset._lerobot_version == "v3.0":
                _, video_meta = dataset._get_lerobot_v3_video_metadata(
                    trajectory_name,
                    video_subkey,
                )
                video_timestamps = video_timestamps + np.float32(
                    video_meta["from_timestamp"]
                )
            specs.append(
                {
                    "video_path": dataset.get_video_path(
                        trajectory_name, video_subkey
                    ).as_posix(),
                    "timestamps": video_timestamps,
                }
            )
        return specs

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        max_retries = 10
        last_exception = None
        original_index = index
        
        for attempt in range(max_retries):
            try:
                attempt_start = time.perf_counter()
                timings = {
                    "sample_step": 0.0,
                    "data_fetch_transform": 0.0,
                    "video_or_spec": 0.0,
                    "pack_action_state": 0.0,
                    "labels": 0.0,
                }
                sample_start = time.perf_counter()
                dataset, trajectory_name, step = self.sample_step(index)
                timings["sample_step"] = time.perf_counter() - sample_start
                compact_video_mode = self.video_target_shift_steps > 0
                build_worker_images = not (
                    (not self.gpu_video_decode_on_rank)
                    and self.cpu_video_decode_drop_worker_images
                )

                # Process all video keys dynamically.
                if self.gpu_video_decode_on_rank:
                    non_video_modalities = [
                        modality for modality in ("state", "action", "language")
                        if modality in dataset.modality_keys
                    ]
                    data_start = time.perf_counter()
                    data = apply_transforms_for_present_keys(
                        dataset.transforms,
                        dataset.get_step_data(trajectory_name, step, modalities=non_video_modalities),
                    )
                    timings["data_fetch_transform"] += time.perf_counter() - data_start
                    images = []
                    video_start = time.perf_counter()
                    if compact_video_mode:
                        video_horizon = len(dataset.delta_indices[dataset.modality_keys["video"][0]])
                        context_horizon = video_horizon - self.video_target_shift_steps
                        union_offsets = np.arange(
                            -(context_horizon - 1),
                            self.video_target_shift_steps + 1,
                            dtype=np.int64,
                        ) * self.video_frame_stride
                        video_specs = self._build_video_decode_specs(
                            dataset,
                            trajectory_name,
                            step,
                            step_offsets=union_offsets,
                        )
                    else:
                        video_specs = self._build_video_decode_specs(
                            dataset,
                            trajectory_name,
                            step,
                        )
                    videos = []
                    timings["video_or_spec"] += time.perf_counter() - video_start
                elif compact_video_mode:
                    non_video_modalities = [
                        modality for modality in ("state", "action", "language")
                        if modality in dataset.modality_keys
                    ]
                    data_start = time.perf_counter()
                    data = apply_transforms_for_present_keys(
                        dataset.transforms,
                        dataset.get_step_data(trajectory_name, step, modalities=non_video_modalities),
                    )
                    timings["data_fetch_transform"] += time.perf_counter() - data_start
                    video_horizon = len(dataset.delta_indices[dataset.modality_keys["video"][0]])
                    video_start = time.perf_counter()
                    videos, images = self._build_shifted_video_views(
                        dataset,
                        trajectory_name,
                        step,
                        video_horizon=video_horizon,
                        build_images=build_worker_images,
                    )
                    timings["video_or_spec"] += time.perf_counter() - video_start
                else:
                    data_start = time.perf_counter()
                    data = dataset.transforms(dataset.get_step_data(trajectory_name, step))    # video T = 1, action T = horizon
                    timings["data_fetch_transform"] += time.perf_counter() - data_start
                    videos, images = [], []
                    video_start = time.perf_counter()
                    for video_key in dataset.modality_keys["video"]:
                        video = data[video_key] # Shape: (T, H, W, C)
                        video = self.resize_video_opencv(video, self.video_resolution_size)
                        videos.append(video)
                        if build_worker_images:
                            primary_image = Image.fromarray(video[0]).resize((self.resolution_size, self.resolution_size))
                            images.append(primary_image)
                    timings["video_or_spec"] += time.perf_counter() - video_start

                pack_start = time.perf_counter()
                if not self.gpu_video_decode_on_rank:
                    if len(dataset.modality_keys["video"]) == 1:
                        videos = [videos[0], videos[0].copy()]  # Duplicate if only one video
                        if build_worker_images:
                            images = [images[0], images[0].copy()]
                    videos = np.stack(videos, axis=0)  # Shape: (V, T, H, W, C)
                    
                # Get language and action data
                language = data[dataset.modality_keys["language"][0]][0]
                action = []
                action_pad_masks = []
                for action_key in dataset.modality_keys["action"]:
                    action.append(data[action_key])
                    pad_mask = data.get(f"{action_key}_is_pad")
                    if pad_mask is not None:
                        action_pad_masks.append(np.asarray(pad_mask, dtype=bool))
                action = np.concatenate(action, axis=1).astype(np.float16)

                return_dict = dict(action=action, lang=language)
                action_is_pad = None
                if action_pad_masks:
                    action_is_pad = np.logical_or.reduce(action_pad_masks)
                    return_dict["action_is_pad"] = action_is_pad
                if build_worker_images:
                    return_dict["image"] = images
                if self.gpu_video_decode_on_rank:
                    return_dict.pop("image", None)
                    if compact_video_mode:
                        return_dict["video_compact_decode_specs"] = video_specs
                    else:
                        return_dict["video_decode_specs"] = video_specs
                elif compact_video_mode:
                    return_dict["video_compact"] = videos
                else:
                    return_dict["video"] = videos
                if self.with_state:
                    state = []
                    for state_key in dataset.modality_keys["state"]:
                        state.append(data[state_key])
                    state = np.concatenate(state, axis=1).astype(np.float16)
                    return_dict["state"] = state[0:1]
                timings["pack_action_state"] += time.perf_counter() - pack_start

                label_start = time.perf_counter()
                action_validity_mask = self._build_action_validity_mask(
                    dataset,
                    step=step,
                    action=action,
                    action_is_pad=action_is_pad,
                )
                if action_validity_mask is not None:
                    return_dict["action_mask"] = action_validity_mask
                if dataset.curr_traj_data is not None and step < len(dataset.curr_traj_data):
                    label_row = dataset.curr_traj_data.iloc[step]
                    for label_key in (
                        "index",
                        "frame_index",
                        "episode_index",
                        "task_index",
                        "task_id",
                        "subtask_index",
                        "sub_task_id",
                        "reward",
                        "valid_state",
                        "valid_state_source",
                        "global_complexity_to_go",
                        "local_complexity_to_go",
                    ):
                        if label_key in label_row.index:
                            return_dict[label_key] = dataset._coerce_label_value(label_row[label_key])

                    future_step = min(step + action.shape[0] - 1, len(dataset.curr_traj_data) - 1)
                    future_row = dataset.curr_traj_data.iloc[future_step]
                    for label_key in (
                        "reward",
                        "global_complexity_to_go",
                        "local_complexity_to_go",
                        "task_id",
                        "subtask_index",
                        "sub_task_id",
                        "valid_state",
                        "valid_state_source",
                    ):
                        if label_key in future_row.index:
                            return_dict[f"future_{label_key}"] = dataset._coerce_label_value(future_row[label_key])

                    prompt_language, subtask_label, task_id_label = (
                        dataset._append_prompt_labels_for_row(
                            return_dict["lang"],
                            label_row,
                            deterministic_key=(
                                "lerobot_prompt_gate_v1",
                                int(self.current_epoch),
                                int(index),
                                str(dataset.dataset_path.resolve()),
                                int(trajectory_name),
                                int(step),
                                int(self.seed),
                            ),
                        )
                    )
                    return_dict["lang"] = prompt_language
                    if subtask_label is not None:
                        return_dict["subtask_label"] = subtask_label
                    if task_id_label is not None:
                        return_dict["task_id_label"] = task_id_label

                    if "sub_task_id" in label_row.index:
                        current_ok_flag = float(label_row["sub_task_id"])
                        future_ok_flag = float(future_row.get("sub_task_id", current_ok_flag))
                        current_mistake = 1.0 - current_ok_flag
                        future_mistake = 1.0 - future_ok_flag
                        return_dict["mistake_label"] = current_mistake
                        return_dict["future_mistake_label"] = future_mistake

                    progress_candidates = (
                        "mistake",
                        "mistake_label",
                        "is_mistake",
                        "failure",
                        "error",
                    )
                    for mistake_key in progress_candidates:
                        if mistake_key in label_row.index:
                            current_value = label_row[mistake_key]
                            future_value = future_row[mistake_key]
                            if isinstance(current_value, np.generic):
                                current_value = current_value.item()
                            if isinstance(future_value, np.generic):
                                future_value = future_value.item()
                            return_dict[mistake_key] = current_value
                            return_dict[f"future_{mistake_key}"] = future_value
                            break

                    def _safe_progress(value, default=0.0):
                        if value is None:
                            return default
                        value = float(value)
                        if np.isnan(value):
                            return default
                        return float(np.clip(value, 0.0, 1.0))

                    if "global_complexity_to_go" in label_row.index:
                        current_global_progress = 1.0 - _safe_progress(label_row["global_complexity_to_go"], 1.0)
                        future_global_progress = 1.0 - _safe_progress(
                            future_row.get("global_complexity_to_go", label_row["global_complexity_to_go"]),
                            1.0,
                        )
                        return_dict["rabc_global_progress"] = current_global_progress
                        return_dict["rabc_future_global_progress"] = future_global_progress
                        return_dict["rabc_global_progress_delta"] = future_global_progress - current_global_progress

                    if "task_id" in label_row.index and "local_complexity_to_go" in label_row.index:
                        stage_series = dataset.curr_traj_data["task_id"]
                        stage_min = int(stage_series.min())
                        stage_max = int(stage_series.max())
                        num_stages = max(stage_max - stage_min + 1, 1)
                        current_stage_idx = int(label_row["task_id"]) - stage_min
                        future_stage_idx = int(future_row.get("task_id", label_row["task_id"])) - stage_min
                        current_local_progress = 1.0 - _safe_progress(label_row["local_complexity_to_go"], 1.0)
                        future_local_progress = 1.0 - _safe_progress(
                            future_row.get("local_complexity_to_go", label_row["local_complexity_to_go"]),
                            1.0,
                        )
                        current_stage_progress = (current_stage_idx + current_local_progress) / num_stages
                        future_stage_progress = (future_stage_idx + future_local_progress) / num_stages
                        return_dict["rabc_stage_progress"] = current_stage_progress
                        return_dict["rabc_future_stage_progress"] = future_stage_progress
                        return_dict["rabc_progress_delta"] = future_stage_progress - current_stage_progress
                timings["labels"] += time.perf_counter() - label_start
                #print(videos[0].shape) #[horizon, H, W, 3]
                #print(action.shape) #[horizon, action_dim]
                #print(images[0]) #PIL.Image
                #print(len(images))# len(dataset.modality_keys["video"])
                #print(language)
                #exit()
                timings["total"] = time.perf_counter() - attempt_start
                self._log_dataset_timing(
                    original_index=original_index,
                    final_index=index,
                    attempts=attempt + 1,
                    dataset=dataset,
                    trajectory_name=trajectory_name,
                    step=step,
                    timings=timings,
                )
                return return_dict
                
            except SubtaskPromptDataError:
                raise
            except Exception as e:
                if bool(getattr(self, "fail_on_sample_error", False)):
                    raise RuntimeError(
                        "Exhaustive epoch sampling cannot silently replace a failed "
                        f"sample (mixture_index={original_index}, attempt_index={index}). "
                        "Fix or explicitly exclude the corrupt row so epoch coverage "
                        "and data-quality accounting remain truthful."
                    ) from e
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print(f"Retrying with new sample...")
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    # index = (index + 1) % len(self)

                    index = random.randint(0, len(self) - 1)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception
                

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        if self._uses_exhaustive_epoch_sampling():
            epoch_length = int(self._epoch_dataset_offsets[-1])
            if epoch_length <= 0:
                raise RuntimeError(
                    f"{self.epoch_sampling_strategy} produced an empty logical epoch."
                )
            return epoch_length

        # Check for potential issues
        if len(self.datasets) == 0:
            return 0
            
        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)
        
        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0
        
        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}")
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0
        
        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]
        
        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()
        
        result = int(max_ratio)
        if result == 0:
            print(f"Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=cfg_get(metadata_config, "percentile_mixing_method", "min_max"),
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the data structure to save
        statistics_data = {}
        
        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []
        seen_action_keys = set()
        seen_state_keys = set()
        
        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for key in used_action_keys:
                if key not in seen_action_keys:
                    all_used_action_keys.append(key)
                    seen_action_keys.add(key)
            for key in used_state_keys:
                if key not in seen_state_keys:
                    all_used_state_keys.append(key)
                    seen_state_keys.add(key)
        
        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}
            
            # Process action statistics
            if hasattr(merged_metadata.statistics, 'action') and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action
                
                # Filter and reorder keys
                non_gripper_keys = []
                gripper_keys = []
                
                for key in action_stats.keys():
                    if key in all_used_action_keys:
                        if "gripper" in key.lower():
                            gripper_keys.append(key)
                        else:
                            non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]
                
                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)
                    
                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys()
                    )
                    combined_action_stats["mask"] = mask
                    
                    tag_stats["action"] = combined_action_stats
            
            # Process state statistics
            if hasattr(merged_metadata.statistics, 'state') and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state
                
                # Filter and reorder keys
                non_gripper_keys = []
                gripper_keys = []
                
                for key in state_stats.keys():
                    if key in all_used_state_keys:
                        if "gripper" in key.lower():
                            gripper_keys.append(key)
                        else:
                            non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]
                
                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats
            
            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))
            
            statistics_data[tag] = tag_stats
        
        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")

    def save_dataset_provenance(self, save_path: Path | str) -> None:
        """Persist immutable source hashes/counts beside checkpoint statistics."""

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "datasets": [dataset.dataset_provenance() for dataset in self.datasets],
        }
        if save_path.exists():
            try:
                existing = json.loads(save_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(
                    f"Existing dataset provenance is unreadable: {save_path}: {exc}"
                ) from exc
            if existing != payload:
                raise ValueError(
                    "Dataset provenance changed for an existing run directory; refusing "
                    f"to overwrite immutable resume binding: {save_path}"
                )
            print(f"Dataset provenance verified unchanged: {save_path}")
            return

        tmp_path = save_path.with_name(f".{save_path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        tmp_path.replace(save_path)
        print(f"Dataset provenance saved to: {save_path}")

    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.
        
        Args:
            tag (str): embodiment tag
            
        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0
        
        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)
        
        return {
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.
        
        Args:
            load_path (Path | str): Path to the statistics file
            
        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")
        
        if load_path.suffix.lower() == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif load_path.suffix.lower() == '.pkl':
            import pickle
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.
        
        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        # Validate that cached statistics match current datasets
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)
            
            if cached_dataset_names != current_dataset_names:
                print("Warning: Cached statistics dataset names don't match current datasets.")
                print(f"Cached: {cached_dataset_names}")
                print(f"Current: {current_dataset_names}")
                return
        
        # Apply cached statistics
        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field
                continue
                
            # Convert back to DatasetMetadata format
            metadata_dict = {
                "embodiment_tag": tag,
                "statistics": {
                    "action": {},
                    "state": {}
                },
                "modalities": {}
            }
            
            # Convert action statistics back
            if "action" in stats_data:
                action_data = stats_data["action"]
                # This is simplified - you may need to split back to sub-keys
                metadata_dict["statistics"]["action"] = action_data
            
            # Convert state statistics back
            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data
            
            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)
        
        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])
        
        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags.")
