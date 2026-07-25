from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from deployment.realman.holdout_identity import (
    SOURCE_ACTION_NAMES,
    SOURCE_STATE_NAMES,
    detect_selected_source_overlaps,
    enumerate_v3_source_identity_catalog,
    validate_holdout_proof,
)


def _episode(seed: int, length: int = 5) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    action = generator.normal(size=(length, 22)).astype(np.float32)
    state = generator.normal(size=(length, 19)).astype(np.float32)
    # Exercise canonical -0.0 handling without making the trajectories trivial.
    action[0, 0] = np.float32(-0.0)
    state[0, 0] = np.float32(0.0)
    return action, state


def _statistics(values: np.ndarray) -> dict[str, list[float]]:
    quantiles = {
        "q01": 0.01,
        "q10": 0.10,
        "q50": 0.50,
        "q90": 0.90,
        "q99": 0.99,
    }
    result = {
        "min": np.min(values, axis=0).astype(np.float64).tolist(),
        "max": np.max(values, axis=0).astype(np.float64).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
    }
    result.update(
        {
            name: np.quantile(values, quantile, axis=0).astype(np.float64).tolist()
            for name, quantile in quantiles.items()
        }
    )
    return result


def _arrow_vectors(values: np.ndarray, *, variable_list: bool) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    if variable_list:
        return pa.array(values.tolist(), type=pa.list_(pa.float32()))
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()),
        values.shape[1],
    )


def _write_v3_dataset(
    root: Path,
    episodes: Sequence[tuple[int, np.ndarray, np.ndarray]],
    *,
    variable_list: bool = False,
) -> Path:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    total_frames = sum(len(action) for _, action, _ in episodes)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "realman_bimanual",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "fps": 20,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "source.action": {
                "dtype": "float32",
                "shape": [22],
                "names": list(SOURCE_ACTION_NAMES),
            },
            "source.observation.state": {
                "dtype": "float32",
                "shape": [19],
                "names": list(SOURCE_STATE_NAMES),
            },
        },
    }
    (root / "meta/info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    metadata_columns: dict[str, list[Any]] = {
        "episode_index": [],
        "length": [],
        "data/chunk_index": [],
        "data/file_index": [],
        "dataset_from_index": [],
        "dataset_to_index": [],
    }
    for field in ("source.action", "source.observation.state"):
        for statistic in (
            "count",
            "min",
            "max",
            "mean",
            "std",
            "q01",
            "q10",
            "q50",
            "q90",
            "q99",
        ):
            metadata_columns[f"stats/{field}/{statistic}"] = []

    action_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    episode_rows: list[np.ndarray] = []
    frame_rows: list[np.ndarray] = []
    index_rows: list[np.ndarray] = []
    timestamp_rows: list[np.ndarray] = []
    dataset_from_index = 0
    for episode_index, action, state in episodes:
        action = np.asarray(action, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        assert action.shape == (len(action), 22)
        assert state.shape == (len(action), 19)
        length = len(action)
        dataset_to_index = dataset_from_index + length
        metadata_columns["episode_index"].append(episode_index)
        metadata_columns["length"].append(length)
        metadata_columns["data/chunk_index"].append(0)
        metadata_columns["data/file_index"].append(0)
        metadata_columns["dataset_from_index"].append(dataset_from_index)
        metadata_columns["dataset_to_index"].append(dataset_to_index)
        for field, values in (
            ("source.action", action),
            ("source.observation.state", state),
        ):
            metadata_columns[f"stats/{field}/count"].append([length])
            for statistic, statistic_values in _statistics(values).items():
                metadata_columns[f"stats/{field}/{statistic}"].append(
                    statistic_values
                )
        frames = np.arange(length, dtype=np.int64)
        action_rows.append(action)
        state_rows.append(state)
        episode_rows.append(np.full(length, episode_index, dtype=np.int64))
        frame_rows.append(frames)
        index_rows.append(dataset_from_index + frames)
        timestamp_rows.append((frames / 20.0).astype(np.float32))
        dataset_from_index = dataset_to_index

    pq.write_table(
        pa.table(metadata_columns),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    all_actions = np.concatenate(action_rows, axis=0)
    all_states = np.concatenate(state_rows, axis=0)
    data_table = pa.table(
        {
            "source.action": _arrow_vectors(
                all_actions, variable_list=variable_list
            ),
            "source.observation.state": _arrow_vectors(
                all_states, variable_list=variable_list
            ),
            "timestamp": pa.array(
                np.concatenate(timestamp_rows), type=pa.float32()
            ),
            "frame_index": pa.array(
                np.concatenate(frame_rows), type=pa.int64()
            ),
            "episode_index": pa.array(
                np.concatenate(episode_rows), type=pa.int64()
            ),
            "index": pa.array(np.concatenate(index_rows), type=pa.int64()),
        }
    )
    pq.write_table(data_table, root / "data/chunk-000/file-000.parquet")
    return root


def _replace_info(root: Path, **updates: Any) -> None:
    path = root / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info.update(updates)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def _replace_data_column(root: Path, name: str, values: pa.Array) -> None:
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    index = table.schema.get_field_index(name)
    pq.write_table(table.set_column(index, name, values), path)


def _replace_metadata_column(root: Path, name: str, values: pa.Array) -> None:
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    index = table.schema.get_field_index(name)
    pq.write_table(table.set_column(index, name, values), path)


def _proof_payload(training: Any, evaluation: Any, selected: Sequence[int]) -> dict[str, Any]:
    return {
        "proof_kind": "test_reconstruction",
        "episode_identity_algorithm": "realman-source-episode-content-v1",
        "training": {
            "episode_count": training.episode_count,
            "frame_count": training.frame_count,
            "source_content_catalog_sha256": training.content_catalog_sha256,
            "source_statistics_catalog_sha256": training.statistics_catalog_sha256,
        },
        "evaluation": {
            "episode_count": evaluation.episode_count,
            "frame_count": evaluation.frame_count,
            "source_content_catalog_sha256": evaluation.content_catalog_sha256,
            "source_statistics_catalog_sha256": evaluation.statistics_catalog_sha256,
        },
        "selected_evaluation_episodes": [
            evaluation.episode(episode_index).manifest_record()
            for episode_index in selected
        ],
    }


def test_identity_is_invariant_to_root_rename_episode_reindex_and_storage_order(
    tmp_path: Path,
) -> None:
    first_action, first_state = _episode(1, length=5)
    second_action, second_state = _episode(2, length=4)
    original = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(
            tmp_path / "original_name",
            [
                (0, first_action, first_state),
                (1, second_action, second_state),
            ],
        ),
        batch_size=2,
    )
    renamed_reindexed = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(
            tmp_path / "different_name",
            [
                (91, second_action, second_state),
                (7, first_action, first_state),
            ],
            # The production training root uses variable-width Arrow lists,
            # while the held-out root uses fixed-size lists.
            variable_list=True,
        ),
        batch_size=2,
    )

    assert original.content_catalog_sha256 == renamed_reindexed.content_catalog_sha256
    assert (
        original.statistics_catalog_sha256
        == renamed_reindexed.statistics_catalog_sha256
    )
    assert {
        episode.source_trajectory_sha256 for episode in original.episodes
    } == {
        episode.source_trajectory_sha256 for episode in renamed_reindexed.episodes
    }
    assert {
        episode.locator.episode_index for episode in original.episodes
    } == {0, 1}
    assert {
        episode.locator.episode_index for episode in renamed_reindexed.episodes
    } == {7, 91}

    overlaps = detect_selected_source_overlaps(
        original,
        renamed_reindexed,
        selected_episode_indices=[91, 7],
    )
    assert len(overlaps) == 2
    assert {overlap.match_kind for overlap in overlaps} == {"source_trajectory"}
    assert {
        overlap.evaluation.episode_index for overlap in overlaps
    } == {7, 91}


def test_changed_numeric_data_changes_content_hash(tmp_path: Path) -> None:
    action, state = _episode(3)
    changed_action = action.copy()
    changed_action[2, 6] += np.float32(0.125)
    original = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(tmp_path / "original", [(4, action, state)])
    )
    changed = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(tmp_path / "changed", [(99, changed_action, state)])
    )

    original_episode = original.episodes[0]
    changed_episode = changed.episodes[0]
    assert (
        original_episode.source_action_sha256
        != changed_episode.source_action_sha256
    )
    assert (
        original_episode.source_observation_state_sha256
        == changed_episode.source_observation_state_sha256
    )
    assert (
        original_episode.source_trajectory_sha256
        != changed_episode.source_trajectory_sha256
    )
    assert original.content_catalog_sha256 != changed.content_catalog_sha256
    assert not detect_selected_source_overlaps(
        original,
        changed,
        include_state_only_matches=False,
    )
    conservative = detect_selected_source_overlaps(original, changed)
    assert len(conservative) == 1
    assert conservative[0].match_kind == "source_observation_state"


def test_malformed_source_feature_schema_fails_closed(tmp_path: Path) -> None:
    action, state = _episode(4)
    root = _write_v3_dataset(tmp_path / "bad_schema", [(0, action, state)])
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["source.action"]["names"][7] = "ambiguous_gripper"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic names/order"):
        enumerate_v3_source_identity_catalog(root)


def test_nonfinite_source_data_fails_closed(tmp_path: Path) -> None:
    action, state = _episode(5)
    root = _write_v3_dataset(tmp_path / "nonfinite", [(0, action, state)])
    corrupt_action = action.copy()
    corrupt_action[1, 3] = np.nan
    _replace_data_column(
        root,
        "source.action",
        _arrow_vectors(corrupt_action, variable_list=False),
    )

    with pytest.raises(ValueError, match="non-finite"):
        enumerate_v3_source_identity_catalog(root, batch_size=2)


def test_out_of_order_frames_fail_closed(tmp_path: Path) -> None:
    action, state = _episode(6)
    root = _write_v3_dataset(tmp_path / "out_of_order", [(0, action, state)])
    _replace_data_column(
        root,
        "frame_index",
        pa.array([1, 0, 2, 3, 4], type=pa.int64()),
    )

    with pytest.raises(ValueError, match="contiguous order"):
        enumerate_v3_source_identity_catalog(root, batch_size=2)


def test_statistics_count_mismatch_fails_closed(tmp_path: Path) -> None:
    action, state = _episode(7)
    root = _write_v3_dataset(tmp_path / "bad_count", [(0, action, state)])
    _replace_metadata_column(
        root,
        "stats/source.action/count",
        pa.array([[len(action) - 1]], type=pa.list_(pa.int64())),
    )

    with pytest.raises(ValueError, match="statistics count.*does not match"):
        enumerate_v3_source_identity_catalog(root)


def test_wrong_fps_fails_closed(tmp_path: Path) -> None:
    action, state = _episode(8)
    root = _write_v3_dataset(tmp_path / "wrong_fps", [(0, action, state)])
    _replace_info(root, fps=30)

    with pytest.raises(ValueError, match="does not match required RealMan fps"):
        enumerate_v3_source_identity_catalog(root)


def test_duplicate_episode_content_retains_catalog_multiplicity(
    tmp_path: Path,
) -> None:
    action, state = _episode(9)
    one = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(tmp_path / "one", [(0, action, state)])
    )
    duplicate = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(
            tmp_path / "duplicate",
            [(10, action, state), (11, action, state)],
        )
    )

    assert duplicate.episode_count == 2
    assert len(
        {episode.source_trajectory_sha256 for episode in duplicate.episodes}
    ) == 1
    assert len(
        {episode.source_statistics_sha256 for episode in duplicate.episodes}
    ) == 1
    assert one.content_catalog_sha256 != duplicate.content_catalog_sha256
    assert one.statistics_catalog_sha256 != duplicate.statistics_catalog_sha256
    overlaps = detect_selected_source_overlaps(one, duplicate)
    assert len(overlaps) == 2
    assert all(len(overlap.training) == 1 for overlap in overlaps)


def test_validate_holdout_proof_exactly_binds_catalogs_and_selected_records(
    tmp_path: Path,
) -> None:
    training_action, training_state = _episode(10)
    evaluation_action, evaluation_state = _episode(11, length=4)
    training = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(
            tmp_path / "training",
            [(0, training_action, training_state)],
        )
    )
    evaluation = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(
            tmp_path / "evaluation",
            [(17, evaluation_action, evaluation_state)],
        )
    )
    proof = _proof_payload(training, evaluation, [17])

    assert validate_holdout_proof(proof, training, evaluation, [17]) is True

    bad_count = deepcopy(proof)
    bad_count["training"]["episode_count"] += 1
    with pytest.raises(ValueError, match="training.episode_count does not match"):
        validate_holdout_proof(bad_count, training, evaluation, [17])

    bad_catalog = deepcopy(proof)
    bad_catalog["evaluation"]["source_content_catalog_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source_content_catalog_sha256 does not match"):
        validate_holdout_proof(bad_catalog, training, evaluation, [17])

    bad_selected_record = deepcopy(proof)
    bad_selected_record["selected_evaluation_episodes"][0][
        "source_trajectory_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_holdout_proof(bad_selected_record, training, evaluation, [17])


def test_validate_holdout_proof_rejects_renamed_reindexed_overlap(
    tmp_path: Path,
) -> None:
    action, state = _episode(12)
    training = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(tmp_path / "training", [(0, action, state)])
    )
    copied_evaluation = enumerate_v3_source_identity_catalog(
        _write_v3_dataset(tmp_path / "renamed_eval", [(999, action, state)])
    )
    proof = _proof_payload(training, copied_evaluation, [999])

    with pytest.raises(ValueError, match="overlap training source trajectories"):
        validate_holdout_proof(
            proof,
            training,
            copied_evaluation,
            [999],
        )
