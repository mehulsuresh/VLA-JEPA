from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from starVLA.action_representation import (
    OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
    OPENPI_REALMAN_UNION_STATISTICS_SCHEMA,
    Q01_Q99_UNCLIPPED,
    REALMAN_18D_ACTION_CONTRACT,
    serialize_openpi_realman_union_statistics,
)
from starVLA.dataloader import dataset_view
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _statistic(width: int, offset: float) -> dict:
    low = [offset + index for index in range(width)]
    high = [value + 2.0 for value in low]
    return {
        "count": [100] * width,
        "mean": [value + 1.0 for value in low],
        "std": [0.5] * width,
        "min": low,
        "max": high,
        "q01": [value + 0.25 for value in low],
        "q99": [value + 1.75 for value in low],
    }


def _slice_statistic(statistics: dict, start: int, end: int) -> dict:
    return {
        key: list(values[start:end])
        for key, values in statistics.items()
    }


def _write_union_artifact(
    path: Path,
    *,
    holdout_manifest_sha256: str,
) -> tuple[dict, str]:
    state = _statistic(18, -20.0)
    action = _statistic(18, 20.0)
    payload = {
        "schema": OPENPI_REALMAN_UNION_STATISTICS_SCHEMA,
        "contract": REALMAN_18D_ACTION_CONTRACT.to_dict(),
        "contract_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
        "normalization": Q01_Q99_UNCLIPPED,
        "algorithm": {
            "quantile_bins": 5000,
            "update_batch": "one_episode",
            "action_horizon": 50,
            "action_padding": "repeat_episode_final_frame",
            "deduplication": "episode-content-sha256-plus-base-frame-v1",
        },
        "population": {
            "schema": OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
            "manifest_sha256": "1" * 64,
            "ledger_sha256": "2" * 64,
            "holdout_manifest_sha256": holdout_manifest_sha256,
            "source_order": ["unrelated-union-source"],
            "sources": [
                {
                    "id": "unrelated-union-source",
                    "catalog_sha256": "3" * 64,
                    "selected_content_sha256": "4" * 64,
                }
            ],
            "candidate_base_frames": 100,
            "unique_base_frames": 100,
            "duplicate_base_frames": 0,
            "holdout_excluded_base_frames": 0,
        },
        "selected": {"state": state, "action": action},
        "modalities": {
            "state": {"source": state},
            "action": {
                "source_controls": _slice_statistic(action, 0, 16),
                "source_head": _slice_statistic(action, 16, 18),
            },
        },
    }
    path.write_bytes(serialize_openpi_realman_union_statistics(payload))
    return payload, _sha256(path)


def _modality_metadata() -> LeRobotModalityMetadata:
    return LeRobotModalityMetadata.model_validate(
        {
            "state": {
                "source": {
                    "original_key": "source.observation.state",
                    "start": 0,
                    "end": 18,
                    "dtype": "float32",
                }
            },
            "action": {
                "source_controls": {
                    "original_key": "source.action",
                    "start": 0,
                    "end": 16,
                    "dtype": "float32",
                },
                "source_head": {
                    "original_key": "source.action",
                    "start": 19,
                    "end": 21,
                    "dtype": "float32",
                },
            },
            "video": {},
        }
    )


def _empty_statistics() -> dict:
    return {
        "state": {"source": {}},
        "action": {"source_controls": {}, "source_head": {}},
    }


def test_union_statistics_override_lerobot_without_local_catalog_equality(
    tmp_path: Path,
):
    holdout_sha256 = "a" * 64
    artifact_path = tmp_path / "union.json"
    artifact, artifact_sha256 = _write_union_artifact(
        artifact_path,
        holdout_manifest_sha256=holdout_sha256,
    )
    dataset = object.__new__(LeRobotSingleDataset)
    dataset.data_cfg = {
        "action_type": "joint_delta_gripper_absolute",
        "action_delta_anchor": "chunk_start_state",
        "gripper_action_type": "absolute",
        "state_action_normalization": "q01_q99_unclipped",
        "normalization_statistics_artifact": str(artifact_path),
        "normalization_statistics_artifact_sha256": artifact_sha256,
        # Presence is required for train role; the view itself is verified
        # later when the exact step index is built.
        "frozen_train_view_manifest": str(tmp_path / "view.json"),
        "frozen_train_view_manifest_sha256": "b" * 64,
    }
    dataset._episode_split_selection = SimpleNamespace(
        role="train",
        # A local stage split is deliberately a different artifact from the
        # global cross-stage holdout bound by the union population.
        manifest_sha256="c" * 64,
    )
    dataset._episode_catalog_binding = {
        # Deliberately different from every union source catalog.
        "episode_catalog_sha256": "f" * 64,
    }
    statistics = _empty_statistics()

    dataset._apply_action_representation_statistics(
        dataset_statistics=statistics,
        le_modality_meta=_modality_metadata(),
    )

    assert statistics["state"]["source"]["q01"] == artifact["modalities"][
        "state"
    ]["source"]["q01"]
    assert statistics["action"]["source_head"]["q99"] == artifact[
        "modalities"
    ]["action"]["source_head"]["q99"]
    assert (
        dataset._action_representation_statistics_scope
        == "immutable_union_train_only"
    )
    assert dataset._action_representation_statistics_population[
        "holdout_manifest_sha256"
    ] == holdout_sha256
    assert (
        dataset._action_representation_local_split_manifest_sha256
        == "c" * 64
    )


def test_union_statistics_fail_closed_on_partial_configuration_or_sha_drift(
    tmp_path: Path,
):
    artifact_path = tmp_path / "union.json"
    _, artifact_sha256 = _write_union_artifact(
        artifact_path,
        holdout_manifest_sha256="a" * 64,
    )
    dataset = object.__new__(LeRobotSingleDataset)
    dataset.data_cfg = {
        "action_type": "joint_delta_gripper_absolute",
        "action_delta_anchor": "chunk_start_state",
        "gripper_action_type": "absolute",
        "state_action_normalization": "q01_q99_unclipped",
        "normalization_statistics_artifact": str(artifact_path),
    }
    dataset._episode_split_selection = SimpleNamespace(
        role="eval",
        manifest_sha256="a" * 64,
    )
    dataset._episode_catalog_binding = {"episode_catalog_sha256": "f" * 64}
    with pytest.raises(ValueError, match="requires both"):
        dataset._apply_action_representation_statistics(
            dataset_statistics=_empty_statistics(),
            le_modality_meta=_modality_metadata(),
        )

    dataset.data_cfg["normalization_statistics_artifact_sha256"] = "0" * 64
    dataset._episode_split_selection = SimpleNamespace(
        role="eval",
        manifest_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dataset._apply_action_representation_statistics(
            dataset_statistics=_empty_statistics(),
            le_modality_meta=_modality_metadata(),
        )

    dataset.data_cfg[
        "normalization_statistics_artifact_sha256"
    ] = artifact_sha256
    statistics = _empty_statistics()
    dataset._apply_action_representation_statistics(
        dataset_statistics=statistics,
        le_modality_meta=_modality_metadata(),
    )
    assert dataset._action_representation_local_split_manifest_sha256 == "c" * 64


def _write_frozen_view(
    tmp_path: Path,
    *,
    catalog_sha256: str,
    info_sha256: str,
    holdout_ids: tuple[int, ...] = (9,),
    base_indices: tuple[int, ...] = (1, 0),
    horizon: int = 50,
    selected_data_shards: list[dict] | None = None,
    purpose: str | None = None,
) -> tuple[Path, str]:
    source_id = "local-realman"
    holdout = dataset_view.make_holdout_exclusions(
        source_id=source_id,
        episode_indices=holdout_ids,
        lineage_ids=["9" * 64],
        content_ids=["8" * 64],
    )
    representation = {
        "contract_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
        "state_dim": 18,
        "action_dim": 18,
        "horizon": horizon,
        "target_fps": 20,
        "action_type": "joint_delta_gripper_absolute",
        "normalization": "q01_q99_unclipped",
    }
    episode_content_id = dataset_view.make_episode_content_id(
        frame_content_sha256="7" * 64,
        length=3,
        content_contract="unit-test-content-v1",
    )
    episode_lineage_id = dataset_view.make_episode_lineage_id(
        backend="lerobot",
        source_id=source_id,
        catalog_sha256=catalog_sha256,
        episode_index=0,
        length=3,
    )
    rows = []
    for base_index in base_indices:
        end_index = min(base_index + horizon - 1, 2)
        rows.append(
            {
                "backend": "lerobot",
                "source_id": source_id,
                "episode_index": 0,
                "episode_length": 3,
                "data_file": "data/chunk-000/file-000.parquet",
                "base_index": base_index,
                "end_index": end_index,
                "horizon": horizon,
                "target_fps": 20,
                "end_clamp_policy": "repeat_last",
                "end_clamped": end_index < base_index + horizon - 1,
                "episode_lineage_id": episode_lineage_id,
                "episode_content_id": episode_content_id,
                "sample_id": dataset_view.make_sample_id(
                    episode_content_id=episode_content_id,
                    base_index=base_index,
                    horizon=horizon,
                    target_fps=20,
                    representation_contract_sha256=(
                        REALMAN_18D_ACTION_CONTRACT.sha256()
                    ),
                    end_clamp=True,
                ),
            }
        )
    manifest = tmp_path / "view.json"
    source = {
        "source_id": source_id,
        "backend": "lerobot",
        "catalog_sha256": catalog_sha256,
        "info_sha256": info_sha256,
        "annotation_sha256": "5" * 64,
        "source_content_sha256": "6" * 64,
    }
    if selected_data_shards is not None:
        source["selected_data_shards"] = selected_data_shards
    result = dataset_view.write_frozen_view(
        manifest,
        descriptor={
            "view_name": "unit_lerobot_view",
            **({"purpose": purpose} if purpose is not None else {}),
            "sources": [source],
            "representation": representation,
            "selection": {
                "selected_episode_indices": [0],
            },
            "holdout_exclusions": holdout,
            "epoch_contract": {
                "mode": "all_exhaustive",
                "epoch_passes": 1,
                "replacement": False,
                "drop_last": False,
                "shuffle": "deterministic_bijection_per_epoch",
            },
        },
        rows=rows,
    )
    return manifest, result.manifest_sha256


def test_lerobot_training_rejects_statistics_population_candidate(
    tmp_path: Path,
) -> None:
    catalog_sha256 = "1" * 64
    info_sha256 = "2" * 64
    manifest, manifest_sha256 = _write_frozen_view(
        tmp_path,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
        purpose="statistics_population_candidate",
    )
    dataset = _frozen_dataset(
        manifest,
        manifest_sha256,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
    )
    with pytest.raises(
        ValueError, match="statistics_population_candidate"
    ):
        dataset._get_frozen_train_view_steps()


def _frozen_dataset(
    manifest: Path,
    manifest_sha256: str,
    *,
    catalog_sha256: str,
    info_sha256: str,
    role: str = "train",
    dataset_root: Path | None = None,
) -> LeRobotSingleDataset:
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_name = "local-realman"
    if dataset_root is not None:
        dataset._dataset_path = dataset_root
    dataset.data_cfg = {
        "action_type": "joint_delta_gripper_absolute",
        "frozen_train_view_manifest": str(manifest),
        "frozen_train_view_manifest_sha256": manifest_sha256,
    }
    dataset._episode_split_selection = SimpleNamespace(
        role=role,
        holdout_episode_ids=(9,),
        selected_episode_ids=(0,),
        manifest_sha256="a" * 64,
        provenance=lambda: {
            "enabled": True,
            "manifest_sha256": "a" * 64,
        },
    )
    dataset._episode_catalog_binding = {
        "episode_catalog_sha256": catalog_sha256,
        "info_sha256": info_sha256,
    }
    dataset._full_trajectory_ids = np.asarray([0, 9], dtype=np.int64)
    dataset._full_trajectory_lengths = np.asarray([3, 2], dtype=np.int64)
    dataset._lerobot_info_meta = {"fps": 20}
    dataset._modality_keys = {
        "action": ["action.source_controls", "action.source_head"]
    }
    dataset._delta_indices = {
        "action.source_controls": np.arange(50, dtype=np.int64),
        "action.source_head": np.arange(50, dtype=np.int64),
    }
    if dataset_root is not None:
        dataset._lerobot_version = "v3.0"
        dataset._data_path_pattern = (
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        )
        dataset.trajectory_ids_to_metadata = {
            0: {
                "data/chunk_index": 0,
                "data/file_index": 0,
            }
        }
    return dataset


def test_frozen_view_replaces_steps_in_exact_ledger_order_and_emits_hashes(
    tmp_path: Path,
):
    catalog_sha256 = "1" * 64
    info_sha256 = "2" * 64
    manifest, manifest_sha256 = _write_frozen_view(
        tmp_path,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
    )
    dataset = _frozen_dataset(
        manifest,
        manifest_sha256,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
    )

    assert dataset._get_frozen_train_view_steps() == [(0, 1), (0, 0)]
    provenance = dataset._frozen_train_view_provenance
    assert provenance["manifest_sha256"] == manifest_sha256
    assert provenance["view_id"]
    assert provenance["ledger_sha256"]
    assert provenance["representation_contract_sha256"] == (
        REALMAN_18D_ACTION_CONTRACT.sha256()
    )


def test_frozen_view_rejects_eval_role_wrong_catalog_and_duplicate_rows(
    tmp_path: Path,
):
    catalog_sha256 = "1" * 64
    info_sha256 = "2" * 64
    manifest, manifest_sha256 = _write_frozen_view(
        tmp_path,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
        base_indices=(0, 0),
    )
    dataset = _frozen_dataset(
        manifest,
        manifest_sha256,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
        role="eval",
    )
    with pytest.raises(ValueError, match="train-only"):
        dataset._get_frozen_train_view_steps()

    dataset._episode_split_selection.role = "train"
    dataset._episode_catalog_binding["episode_catalog_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="catalog_sha256 mismatch"):
        dataset._get_frozen_train_view_steps()

    dataset._episode_catalog_binding["episode_catalog_sha256"] = catalog_sha256
    with pytest.raises(ValueError, match="duplicate row/sample"):
        dataset._get_frozen_train_view_steps()


def test_frozen_view_rejects_non_h50_and_out_of_range_rows(tmp_path: Path):
    catalog_sha256 = "1" * 64
    info_sha256 = "2" * 64
    non_h50_dir = tmp_path / "non-h50"
    non_h50_dir.mkdir()
    manifest, manifest_sha256 = _write_frozen_view(
        non_h50_dir,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
        horizon=49,
    )
    dataset = _frozen_dataset(
        manifest,
        manifest_sha256,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
    )
    with pytest.raises(ValueError, match="representation is incompatible"):
        dataset._get_frozen_train_view_steps()


def test_frozen_view_verifies_selected_parquet_bytes_and_rejects_mutation(
    tmp_path: Path,
):
    catalog_sha256 = "1" * 64
    info_sha256 = "2" * 64
    dataset_root = tmp_path / "local-realman"
    shard_path = dataset_root / "data/chunk-000/file-000.parquet"
    shard_path.parent.mkdir(parents=True)
    shard_path.write_bytes(b"original-selected-shard")
    binding = {
        "path": "data/chunk-000/file-000.parquet",
        "sha256": _sha256(shard_path),
        "size_bytes": shard_path.stat().st_size,
    }
    view_dir = tmp_path / "view"
    view_dir.mkdir()
    manifest, manifest_sha256 = _write_frozen_view(
        view_dir,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
        selected_data_shards=[binding],
    )
    dataset = _frozen_dataset(
        manifest,
        manifest_sha256,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
        dataset_root=dataset_root,
    )
    dataset.data_cfg["frozen_train_view_require_data_shard_hashes"] = True

    assert dataset._get_frozen_train_view_steps() == [(0, 1), (0, 0)]
    assert dataset._frozen_train_view_provenance["selected_data_shards"] == [
        binding
    ]

    # Keep the byte length identical so this specifically exercises SHA-256,
    # not only the cheaper size guard.
    shard_path.write_bytes(b"mutated!-selected-shard")
    assert shard_path.stat().st_size == binding["size_bytes"]
    with pytest.raises(ValueError, match="data-shard SHA-256 mismatch"):
        dataset._get_frozen_train_view_steps()

    out_of_range_dir = tmp_path / "out-of-range"
    out_of_range_dir.mkdir()
    manifest, manifest_sha256 = _write_frozen_view(
        out_of_range_dir,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
        base_indices=(3,),
    )
    dataset = _frozen_dataset(
        manifest,
        manifest_sha256,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
    )
    with pytest.raises(ValueError, match="outside episode"):
        dataset._get_frozen_train_view_steps()


def test_dataset_provenance_emits_union_and_frozen_view_bindings(
    tmp_path: Path,
):
    catalog_sha256 = "1" * 64
    info_sha256 = "2" * 64
    view_dir = tmp_path / "artifacts"
    view_dir.mkdir()
    manifest, manifest_sha256 = _write_frozen_view(
        view_dir,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
    )
    dataset = _frozen_dataset(
        manifest,
        manifest_sha256,
        catalog_sha256=catalog_sha256,
        info_sha256=info_sha256,
    )
    assert dataset._get_frozen_train_view_steps()

    union_path = tmp_path / "union.json"
    artifact, artifact_sha256 = _write_union_artifact(
        union_path,
        holdout_manifest_sha256="d" * 64,
    )
    dataset.data_cfg.update(
        {
            "action_delta_anchor": "chunk_start_state",
            "gripper_action_type": "absolute",
            "state_action_normalization": "q01_q99_unclipped",
            "normalization_statistics_artifact": str(union_path),
            "normalization_statistics_artifact_sha256": artifact_sha256,
        }
    )
    dataset._apply_action_representation_statistics(
        dataset_statistics=_empty_statistics(),
        le_modality_meta=_modality_metadata(),
    )

    dataset_root = tmp_path / "local-realman"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text(
        json.dumps({"fps": 20}) + "\n",
        encoding="utf-8",
    )
    dataset._dataset_path = dataset_root
    dataset._lerobot_version = "v3.0"
    dataset._trajectory_ids = np.asarray([0], dtype=np.int64)
    dataset._trajectory_lengths = np.asarray([3], dtype=np.int64)

    provenance = dataset.dataset_provenance()
    assert provenance["frozen_train_view"]["manifest_sha256"] == (
        manifest_sha256
    )
    action_representation = provenance["action_representation"]
    assert action_representation["statistics_sha256"] == artifact_sha256
    assert (
        action_representation["statistics_population"][
            "holdout_manifest_sha256"
        ]
        == artifact["population"]["holdout_manifest_sha256"]
    )
    assert action_representation["local_split_manifest_sha256"] == "a" * 64
