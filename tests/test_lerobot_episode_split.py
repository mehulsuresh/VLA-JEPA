import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
)
from starVLA.dataloader.gr00t_lerobot.episode_split import (
    TRAIN_STATISTICS_PROVENANCE_KEY,
    TRAIN_STATISTICS_SCHEMA,
    build_episode_catalog_binding,
    episode_set_sha256,
    file_sha256,
    load_episode_split_selection,
)
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _catalog(tmp_path: Path):
    root = tmp_path / "robot_data"
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    lengths = np.asarray([2, 3, 5, 7], dtype=np.int64)
    pd.DataFrame({"episode_index": ids, "length": lengths}).to_parquet(
        episodes_path
    )
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "total_episodes": 4,
            "total_frames": 17,
        },
    )
    binding = build_episode_catalog_binding(
        dataset_path=root,
        dataset_name=root.name,
        lerobot_version="v3.0",
        trajectory_ids=ids,
        trajectory_lengths=lengths,
    )
    return root, ids, lengths, binding


def _statistics_payload(count: int) -> dict:
    def values(offset: float):
        return {
            "count": [count],
            "mean": [offset, offset + 1],
            "std": [1.0, 1.0],
            "min": [offset - 1, offset],
            "max": [offset + 1, offset + 2],
            "q01": [offset - 1, offset],
            "q99": [offset + 1, offset + 2],
        }

    return {
        "source.observation.state": values(0.0),
        "source.action": values(10.0),
    }


def _manifest(
    tmp_path: Path,
    *,
    root: Path,
    ids: np.ndarray,
    lengths: np.ndarray,
    binding: dict,
    holdout=(20, 40),
):
    lengths_by_id = {
        int(episode_id): int(length)
        for episode_id, length in zip(ids.tolist(), lengths.tolist())
    }
    holdout = tuple(sorted(holdout))
    train = tuple(sorted(set(lengths_by_id) - set(holdout)))
    train_frames = sum(lengths_by_id[value] for value in train)
    holdout_frames = sum(lengths_by_id[value] for value in holdout)
    train_catalog_sha256 = episode_set_sha256(
        train,
        lengths_by_id=lengths_by_id,
    )
    statistics_path = tmp_path / "train_statistics.json"
    statistics_payload = _statistics_payload(train_frames)
    statistics_payload[TRAIN_STATISTICS_PROVENANCE_KEY] = {
        "schema": TRAIN_STATISTICS_SCHEMA,
        "full_catalog_sha256": binding["episode_catalog_sha256"],
        "train_catalog_sha256": train_catalog_sha256,
        "train_episode_count": len(train),
        "train_frame_count": train_frames,
    }
    _write_json(statistics_path, statistics_payload)
    payload = {
        "schema_version": 1,
        "split_id": "unit-split-v1",
        "role_contract": {
            "train_episode_selection": "complement_of_holdout",
            "evaluation_episode_selection": "holdout_episode_indices",
            "normalization_statistics": "train_statistics_only",
        },
        "selection": {"algorithm": "unit-test"},
        "datasets": [
            {
                "dataset_name": root.name,
                "dataset_root": str(root),
                "lerobot_version": "v3.0",
                "info_sha256": binding["info_sha256"],
                "full_catalog_sha256": binding["episode_catalog_sha256"],
                "full_episode_count": len(ids),
                "full_frame_count": int(lengths.sum()),
                "holdout_episode_indices": list(holdout),
                "holdout_episode_count": len(holdout),
                "holdout_frame_count": holdout_frames,
                "holdout_catalog_sha256": episode_set_sha256(
                    holdout,
                    lengths_by_id=lengths_by_id,
                ),
                "train_episode_selection": {"kind": "complement_of_holdout"},
                "train_episode_count": len(train),
                "train_frame_count": train_frames,
                "train_catalog_sha256": train_catalog_sha256,
                "train_statistics": {
                    "path": statistics_path.name,
                    "sha256": file_sha256(statistics_path),
                    "frame_count": train_frames,
                    "catalog_sha256": train_catalog_sha256,
                },
            }
        ],
    }
    manifest_path = tmp_path / "split.json"
    _write_json(manifest_path, payload)
    return manifest_path, payload, statistics_path


def _selection(tmp_path: Path, *, role="train"):
    root, ids, lengths, binding = _catalog(tmp_path)
    manifest_path, payload, statistics_path = _manifest(
        tmp_path,
        root=root,
        ids=ids,
        lengths=lengths,
        binding=binding,
    )
    selection = load_episode_split_selection(
        manifest_path=manifest_path,
        dataset_name=root.name,
        role=role,
        catalog_binding=binding,
        trajectory_ids=ids,
        trajectory_lengths=lengths,
    )
    return selection, root, ids, lengths, binding, manifest_path, payload, statistics_path


def test_manifest_selects_complement_for_train_and_holdout_for_eval(tmp_path):
    train, root, ids, lengths, binding, manifest_path, _, _ = _selection(tmp_path)
    evaluation = load_episode_split_selection(
        manifest_path=manifest_path,
        dataset_name=root.name,
        role="eval",
        catalog_binding=binding,
        trajectory_ids=ids,
        trajectory_lengths=lengths,
    )

    assert train.selected_episode_ids == (10, 30)
    assert train.selected_frame_count == 7
    assert evaluation.selected_episode_ids == (20, 40)
    assert evaluation.selected_frame_count == 10
    assert evaluation.train_statistics_path == train.train_statistics_path
    assert evaluation.train_statistics_sha256 == train.train_statistics_sha256
    provenance = train.provenance()
    assert provenance["selected_episode_ids"] == [10, 30]
    assert provenance["excluded_episode_ids"] == [20, 40]
    assert provenance["holdout_episode_ids"] == [20, 40]
    assert provenance["manifest_sha256"] == file_sha256(manifest_path)
    assert provenance["normalization_statistics_scope"] == "train_split_only"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["datasets"][0].__setitem__(
                "full_catalog_sha256", "0" * 64
            ),
            "catalog binding",
        ),
        (
            lambda payload: payload["datasets"][0].__setitem__(
                "train_frame_count", 999
            ),
            "count or catalog bindings",
        ),
        (
            lambda payload: payload["datasets"][0]["train_statistics"].__setitem__(
                "catalog_sha256", "1" * 64
            ),
            "catalog_sha256",
        ),
        (
            lambda payload: payload["role_contract"].__setitem__(
                "normalization_statistics", "full_dataset"
            ),
            "role_contract",
        ),
    ],
)
def test_manifest_rejects_tampered_catalog_counts_and_contract(
    tmp_path, mutate, match
):
    _, root, ids, lengths, binding, manifest_path, payload, _ = _selection(tmp_path)
    mutate(payload)
    _write_json(manifest_path, payload)

    with pytest.raises(ValueError, match=match):
        load_episode_split_selection(
            manifest_path=manifest_path,
            dataset_name=root.name,
            role="train",
            catalog_binding=binding,
            trajectory_ids=ids,
            trajectory_lengths=lengths,
        )


def test_manifest_rejects_changed_train_statistics_bytes(tmp_path):
    _, root, ids, lengths, binding, manifest_path, _, statistics_path = _selection(
        tmp_path
    )
    statistics_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_episode_split_selection(
            manifest_path=manifest_path,
            dataset_name=root.name,
            role="train",
            catalog_binding=binding,
            trajectory_ids=ids,
            trajectory_lengths=lengths,
        )


def test_manifest_rejects_statistics_without_embedded_split_provenance(tmp_path):
    _, root, ids, lengths, binding, manifest_path, payload, statistics_path = _selection(
        tmp_path
    )
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics.pop(TRAIN_STATISTICS_PROVENANCE_KEY)
    _write_json(statistics_path, statistics)
    payload["datasets"][0]["train_statistics"]["sha256"] = file_sha256(
        statistics_path
    )
    _write_json(manifest_path, payload)

    with pytest.raises(ValueError, match="_split_provenance"):
        load_episode_split_selection(
            manifest_path=manifest_path,
            dataset_name=root.name,
            role="train",
            catalog_binding=binding,
            trajectory_ids=ids,
            trajectory_lengths=lengths,
        )


def test_filtering_precedes_dense_steps_and_prunes_v3_episode_metadata(tmp_path):
    selection, root, ids, lengths, binding, _, _, statistics_path = _selection(
        tmp_path,
        role="train",
    )
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = root
    dataset._dataset_name = root.name
    dataset._full_trajectory_ids = ids
    dataset._full_trajectory_lengths = lengths
    dataset._episode_catalog_binding = binding
    dataset._episode_split_selection = selection
    dataset._lerobot_version = "v3.0"
    dataset.tag = "new_embodiment"
    dataset.delete_pause_frame = False
    dataset.data_cfg = {"data_mix": "unit"}
    dataset._modality_keys = {
        "video": [],
        "state": [],
        "action": [],
        "language": [],
    }
    dataset.trajectory_ids_to_metadata = {
        int(episode_id): {"marker": int(episode_id)} for episode_id in ids
    }

    dataset._apply_episode_split_to_catalog()
    steps = dataset._get_all_steps_from_trajectory_lengths()

    np.testing.assert_array_equal(dataset.trajectory_ids, [10, 30])
    np.testing.assert_array_equal(dataset.trajectory_lengths, [2, 5])
    assert set(dataset.trajectory_ids_to_metadata) == {10, 30}
    assert steps == [
        (10, 0),
        (10, 1),
        (30, 0),
        (30, 1),
        (30, 2),
        (30, 3),
        (30, 4),
    ]
    cache_metadata = dataset._get_steps_cache_metadata()
    assert cache_metadata["num_trajectories"] == 2
    assert cache_metadata["total_frames"] == 7
    dataset._statistics_total_frames = 7
    dataset._statistics_scope = "train_split_only"
    dataset._statistics_source_path = statistics_path
    dataset._statistics_source_sha256 = file_sha256(statistics_path)
    dataset._statistics_effective_sha256 = "effective"
    dataset._statistics_quantiles_synthesized = False
    provenance = dataset.dataset_provenance()
    assert provenance["full_catalog_episode_count"] == 4
    assert provenance["full_catalog_frame_count"] == 17
    assert provenance["selected_episode_count"] == 2
    assert provenance["selected_frame_count"] == 7
    assert provenance["statistics_scope"] == "train_split_only"
    assert provenance["episode_split"]["excluded_episode_ids"] == [20, 40]


def _modality_metadata():
    return LeRobotModalityMetadata.model_validate(
        {
            "state": {
                "source": {
                    "original_key": "source.observation.state",
                    "start": 0,
                    "end": 2,
                    "dtype": "float32",
                }
            },
            "action": {
                "source": {
                    "original_key": "source.action",
                    "start": 0,
                    "end": 2,
                    "dtype": "float32",
                }
            },
            "video": {},
        }
    )


def test_split_role_loads_only_manifest_bound_train_statistics(tmp_path):
    selection, root, _, _, _, _, _, statistics_path = _selection(
        tmp_path,
        role="eval",
    )
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = root
    dataset._dataset_name = root.name
    dataset._lerobot_version = "v3.0"
    dataset._episode_split_selection = selection
    dataset.data_cfg = {"lerobot_statistics_source": "split_train"}
    dataset.transforms = SimpleNamespace(transforms=[])
    # A full-catalog table must not be considered while a split is active.
    _write_json(root / "meta/stats.json", _statistics_payload(17))

    selected = dataset._load_lerobot_statistics(
        _modality_metadata(),
        {"total_frames": 17},
    )

    assert selected["source.action"]["count"] == [7]
    assert dataset._statistics_source_path == statistics_path
    assert dataset._statistics_total_frames == 7
    assert dataset._statistics_scope == "train_split_only"


def test_manifest_conflicts_with_legacy_load_all_flag(tmp_path):
    manifest = tmp_path / "split.json"
    _write_json(manifest, {})
    dataset = object.__new__(LeRobotSingleDataset)
    dataset.data_cfg = {
        "episode_split_manifest": str(manifest),
        "load_all_data_for_training": True,
        "lerobot_statistics_source": "split_train",
    }

    with pytest.raises(ValueError, match="load_all_data_for_training=true"):
        dataset._resolve_episode_split_selection(episode_split_role="train")


def test_configured_role_cannot_silently_override_caller_mode(tmp_path):
    manifest = tmp_path / "split.json"
    _write_json(manifest, {})
    dataset = object.__new__(LeRobotSingleDataset)
    dataset.data_cfg = {
        "episode_split_manifest": str(manifest),
        "episode_split_role": "train",
        "lerobot_statistics_source": "split_train",
    }

    with pytest.raises(ValueError, match="conflicts with the dataset mode"):
        dataset._resolve_episode_split_selection(episode_split_role="eval")


def test_mixture_trajectory_weights_use_filtered_lengths(tmp_path, monkeypatch):
    selection, _, ids, lengths, _, _, _, _ = _selection(tmp_path, role="train")
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._full_trajectory_ids = ids
    dataset._full_trajectory_lengths = lengths
    dataset._episode_split_selection = selection
    dataset._lerobot_version = "v3.0"
    dataset.trajectory_ids_to_metadata = {
        int(episode_id): {} for episode_id in ids
    }
    dataset._dataset_name = "robot_data"
    dataset.tag = "new_embodiment"
    dataset._apply_episode_split_to_catalog()
    dataset._all_steps = dataset._get_all_steps_from_trajectory_lengths()
    monkeypatch.setattr(LeRobotMixtureDataset, "update_metadata", lambda *args, **kwargs: None)

    mixture = LeRobotMixtureDataset(
        [(dataset, 1.0)],
        mode="train",
        balance_dataset_weights=False,
        balance_trajectory_weights=True,
        metadata_config={},
    )

    np.testing.assert_allclose(mixture.trajectory_sampling_weights[0], [2 / 7, 5 / 7])
    np.testing.assert_array_equal(mixture.dataset_lengths, [7])
