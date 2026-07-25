from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from starVLA.realman_union_holdout import (
    LEROBOT_SOURCE_KIND,
    derive_source_holdout,
    write_global_holdout_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_leaf(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


view = _load_leaf(
    "_test_dataset_view",
    REPO_ROOT / "starVLA/dataloader/dataset_view.py",
)
generator = _load_leaf(
    "_test_realman_dataset_view_generator",
    REPO_ROOT / "scripts/build_realman_dataset_views.py",
)
union_stats = _load_leaf(
    "_test_realman_union_stats",
    REPO_ROOT / "scripts/compute_openpi_realman_union_stats.py",
)


def _vector(seed: float, frame_index: int, width: int) -> list[float]:
    return [
        float(seed + frame_index * 0.01 + channel * 0.001)
        for channel in range(width)
    ]


def _write_lerobot_fixture(
    root: Path,
    episodes: list[dict],
) -> None:
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    data_path = root / "data/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    data_path.parent.mkdir(parents=True)
    episode_rows = []
    data_rows = []
    global_index = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        valid_state = list(episode["valid_state"])
        seed = float(episode.get("content_seed", episode_index))
        length = len(valid_state)
        episode_rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
            }
        )
        for frame_index, valid in enumerate(valid_state):
            data_row = {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "index": global_index + frame_index,
                "timestamp": frame_index / 20.0,
                "source.observation.state": _vector(
                    seed, frame_index, 19
                ),
                "source.action": _vector(seed + 1, frame_index, 22),
                "observation.state": _vector(seed + 2, frame_index, 19),
                "action": _vector(seed + 3, frame_index, 22),
                "valid_state": int(valid),
                "valid_state_source": 1,
                "subtask_index": int(seed) % 7,
                "task_id": int(seed) % 3,
                "task_index": int(seed) % 3,
            }
            if "valid_action" in episode:
                data_row["valid_action"] = int(
                    episode["valid_action"][frame_index]
                )
            if "action_owner" in episode:
                data_row["action_owner"] = episode["action_owner"][
                    frame_index
                ]
            if "action_source" in episode:
                data_row["action_source"] = episode["action_source"][
                    frame_index
                ]
            data_rows.append(data_row)
        global_index += length
    pd.DataFrame(episode_rows).to_parquet(episode_path, index=False)
    pd.DataFrame(data_rows).to_parquet(data_path, index=False)
    pd.DataFrame(
        [
            {
                "subtask_index": index,
                "local_subtask_text": f"fixture subtask {index}",
                "global_subtask_type": "fixture",
                "local_subtask_id": f"fixture_{index}",
                "is_mistake": False,
                "source": "test_fixture",
                "optional": False,
            }
            for index in range(7)
        ]
    ).to_parquet(root / "meta/subtasks.parquet", index=False)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 20,
                "total_episodes": len(episode_rows),
                "total_frames": len(data_rows),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_write(path: Path, payload: dict) -> None:
    path.write_bytes(view.canonical_json_bytes(payload) + b"\n")


def _write_local_split_manifest(
    path: Path,
    *,
    dataset_root: Path,
    holdout_episode_ids: tuple[int, ...],
) -> Path:
    catalog, binding = generator._load_episode_catalog(dataset_root)
    lengths = {
        episode_id: record.length
        for episode_id, record in catalog.items()
    }

    def set_sha256(episode_ids: list[int]) -> str:
        return view.canonical_json_sha256(
            {
                "schema": "lerobot-episode-set-v1",
                "episodes": [
                    {
                        "episode_id": episode_id,
                        "length": lengths[episode_id],
                    }
                    for episode_id in sorted(episode_ids)
                ],
            }
        )

    holdout = sorted(holdout_episode_ids)
    train = sorted(set(catalog) - set(holdout))
    payload = {
        "schema_version": 1,
        "split_id": "fixture-split-v1",
        "role_contract": {
            "train_episode_selection": "complement_of_holdout",
            "evaluation_episode_selection": "holdout_episode_indices",
            "normalization_statistics": "train_statistics_only",
        },
        "datasets": [
            {
                "dataset_name": dataset_root.name,
                "full_catalog_sha256": binding["catalog_sha256"],
                "full_episode_count": binding["episode_count"],
                "full_frame_count": binding["frame_count"],
                "info_sha256": binding["info_sha256"],
                "holdout_episode_indices": holdout,
                "holdout_episode_count": len(holdout),
                "holdout_frame_count": sum(
                    lengths[value] for value in holdout
                ),
                "holdout_catalog_sha256": set_sha256(holdout),
                "train_episode_selection": {
                    "kind": "complement_of_holdout"
                },
                "train_episode_count": len(train),
                "train_frame_count": sum(
                    lengths[value] for value in train
                ),
                "train_catalog_sha256": set_sha256(train),
            }
        ],
        "selection": {
            "selected_episode_ids_sorted": holdout,
        },
    }
    _canonical_write(path, payload)
    return path


def test_incremental_generator_is_exhaustive_deterministic_and_copy_safe(
    tmp_path: Path,
):
    dataset_root = tmp_path / "incremental_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {
                "episode_index": 10,
                "valid_state": [1, 0, 1, 1, 1],
                "content_seed": 10,
            },
            {
                "episode_index": 11,
                "valid_state": [1, 1, 1, 1],
                "content_seed": 20,
            },
            {
                # Same canonical content as held-out episode 11 even though
                # its dataset index and global indices differ.
                "episode_index": 12,
                "valid_state": [1, 1, 1, 1],
                "content_seed": 20,
            },
        ],
    )
    first_manifest = tmp_path / "first/view.json"
    second_manifest = tmp_path / "second/view.json"
    first = generator.build_intervention_incremental_view(
        dataset_root=dataset_root,
        output_manifest=first_manifest,
        first_episode=10,
        last_episode=12,
        holdout_episode_ids=(11,),
        horizon=3,
    )
    second = generator.build_intervention_incremental_view(
        dataset_root=dataset_root,
        output_manifest=second_manifest,
        first_episode=10,
        last_episode=12,
        holdout_episode_ids=(11,),
        horizon=3,
    )

    assert first.row_count == 5
    assert first.unique_sample_count == 5
    assert first.episode_count == 1
    assert first.view_id == second.view_id
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first.ledger_path.read_bytes() == second.ledger_path.read_bytes()

    loaded = view.load_frozen_view(
        first_manifest,
        expected_representation_contract_sha256=(
            generator.DEFAULT_REPRESENTATION_SHA256
        ),
    )
    descriptor = loaded.descriptor
    coverage = descriptor["selection"]["subtask_prompt_coverage"]
    assert coverage["selected_row_count"] == 5
    assert coverage["useful_prompt_row_count"] == 5
    action_audit = descriptor["action_supervision_audit"]
    assert action_audit["status"] == "unverified"
    assert action_audit["verification_mode"] is None
    assert action_audit[
        "valid_state_is_not_generic_action_ownership"
    ] is True
    assert any(
        "valid_state" in reason for reason in action_audit["reasons"]
    ) is False
    assert any(
        "no reviewed SHA-bound" in reason
        for reason in action_audit["reasons"]
    )
    assert descriptor["sources"][0]["subtasks_file_sha256"]
    assert descriptor["sources"][0]["subtask_catalog_sha256"]
    assert descriptor["sources"][0]["selected_data_shards"] == [
        {
            "path": "data/chunk-000/file-000.parquet",
            "sha256": view.file_sha256(
                dataset_root / "data/chunk-000/file-000.parquet"
            ),
            "size_bytes": (
                dataset_root / "data/chunk-000/file-000.parquet"
            ).stat().st_size,
        }
    ]
    assert descriptor["epoch_contract"] == {
        "ddp_tail": "duplicated_padding_reported_separately",
        "drop_last": False,
        "epoch_passes": 1,
        "mode": "all_exhaustive",
        "replacement": False,
        "shuffle": "deterministic_bijection_per_epoch",
    }
    assert descriptor["selection"]["selected_episode_indices"] == [10]
    assert descriptor["selection"]["content_copy_excluded_episode_indices"] == [
        12
    ]
    rows = list(loaded.iter_rows())
    assert [row["base_index"] for row in rows] == [0, 1, 2, 3, 4]
    assert rows[-1]["end_clamped"] is True
    assert all(row["episode_index"] == 10 for row in rows)
    assert not (
        {row["episode_content_id"] for row in rows}
        & set(descriptor["holdout_exclusions"]["content_ids"])
    )


def test_intervention_action_semantics_contract_is_hash_bound_and_preserves_recovery(
    tmp_path: Path,
):
    dataset_root = tmp_path / "contract_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {
                "episode_index": 0,
                "valid_state": [1, 0, 0, 1, 1, 1],
            },
            {
                "episode_index": 1,
                "valid_state": [1, 1, 1, 1],
            },
        ],
    )
    unverified_manifest = tmp_path / "unverified/view.json"
    generator.build_intervention_incremental_view(
        dataset_root=dataset_root,
        output_manifest=unverified_manifest,
        first_episode=0,
        last_episode=1,
        holdout_episode_ids=(1,),
        horizon=5,
    )
    unverified = view.load_frozen_view(unverified_manifest).descriptor[
        "action_supervision_audit"
    ]
    assert unverified["status"] == "unverified"
    binding = unverified["dataset_binding"]

    contract = {
        "schema": generator.ACTION_LABEL_SEMANTICS_SCHEMA,
        "status": "verified",
        "dataset": {
            key: binding[key]
            for key in (
                "source_id",
                "catalog_sha256",
                "annotation_sha256",
                "source_content_sha256",
            )
        },
        "semantics": {
            "validity_column": "valid_state",
            "mistake_value": 0,
            "mistake_meaning": "mistake_action_do_not_supervise",
            "supervised_value": 1,
            "supervised_meaning": (
                "expert_or_recovery_action_supervise"
            ),
            "recovery_anchor_definition": (
                "first_valid_frame_after_invalid_frame"
            ),
        },
        "masking": {
            "policy": (
                "chunk_prefix_until_first_sustained_invalid_or_padding"
            ),
            "invalid_run_length": 10,
            "recovery_windows": (
                "windows_anchored_at_valid_recovery_frames_are_supervised"
            ),
        },
        "review": {
            "reviewer": "fixture reviewer",
            "reviewed_at_utc": "2026-07-24T00:00:00Z",
            "evidence": "fixture labeling audit",
        },
    }
    contract_path = tmp_path / "action-label-semantics.json"
    contract_path.write_text(
        json.dumps(contract, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verified_manifest = tmp_path / "verified/view.json"
    generator.build_intervention_incremental_view(
        dataset_root=dataset_root,
        output_manifest=verified_manifest,
        first_episode=0,
        last_episode=1,
        holdout_episode_ids=(1,),
        horizon=5,
        action_label_semantics_contract=contract_path,
    )
    audit = view.load_frozen_view(verified_manifest).descriptor[
        "action_supervision_audit"
    ]
    assert audit["status"] == "verified"
    assert audit["verification_mode"] == (
        "reviewed_valid_state_action_semantics_contract"
    )
    assert audit["action_label_semantics_contract"]["sha256"]
    recovery = audit["recovery_from_invalid_state_supervision"]
    assert recovery["status"] == "verified"
    assert recovery["recovery_anchor_window_count"] == 1
    assert (
        recovery["recovery_anchor_with_nonzero_action_mask_count"]
        == 1
    )
    assert recovery["minimum_supervised_action_timesteps_per_anchor"] == 3
    assert recovery["supervised_action_element_count"] == 3 * 18


def test_explicit_action_ownership_can_verify_without_valid_state_semantics_contract(
    tmp_path: Path,
):
    dataset_root = tmp_path / "explicit_action_labels"
    _write_lerobot_fixture(
        dataset_root,
        [
            {
                "episode_index": 0,
                "valid_state": [1, 0, 1, 1],
                "valid_action": [1, 1, 1, 1],
                "action_owner": ["human"] * 4,
            },
            {
                "episode_index": 1,
                "valid_state": [1, 1, 1],
                "valid_action": [1, 1, 1],
                "action_owner": ["human"] * 3,
            },
        ],
    )
    manifest = tmp_path / "explicit/view.json"
    generator.build_intervention_incremental_view(
        dataset_root=dataset_root,
        output_manifest=manifest,
        first_episode=0,
        last_episode=1,
        holdout_episode_ids=(1,),
        horizon=3,
    )
    audit = view.load_frozen_view(manifest).descriptor[
        "action_supervision_audit"
    ]
    assert audit["status"] == "verified"
    assert audit["verification_mode"] == (
        "explicit_valid_action_and_expert_owner"
    )
    assert audit["explicit_action_labels"][
        "invalid_state_with_explicit_expert_action_row_count"
    ] == 1


def test_hq_generator_emits_every_and_only_clean_full_horizon_window(
    tmp_path: Path,
):
    dataset_root = tmp_path / "hq_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {
                "episode_index": 0,
                "valid_state": [1, 1, 1, 1, 0, 1],
            },
            {
                "episode_index": 1,
                "valid_state": [1, 1, 1, 1],
            },
            {
                "episode_index": 2,
                "valid_state": [1, 1, 1, 1, 1],
            },
            {
                "episode_index": 3,
                "valid_state": [1, 1],
            },
        ],
    )
    manifest = tmp_path / "hq/view.json"
    result = generator.build_hq_clean_h50_view(
        dataset_root=dataset_root,
        output_manifest=manifest,
        holdout_episode_ids=(1,),
        horizon=3,
    )
    loaded = view.load_frozen_view(manifest)
    rows = list(loaded.iter_rows())

    assert result.row_count == 5
    assert result.unique_sample_count == 5
    assert result.episode_count == 2
    assert [
        (row["episode_index"], row["base_index"])
        for row in rows
    ] == [(0, 0), (0, 1), (2, 0), (2, 1), (2, 2)]
    assert all(row["end_clamped"] is False for row in rows)
    assert loaded.descriptor["selection"]["selected_episode_indices"] == [
        0,
        2,
        3,
    ]


def test_dry_run_computes_exact_artifact_identity_without_writing(
    tmp_path: Path,
):
    dataset_root = tmp_path / "dry_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {"episode_index": 4, "valid_state": [1, 1, 1]},
            {"episode_index": 5, "valid_state": [1, 1, 1]},
        ],
    )
    manifest = tmp_path / "does-not-exist/view.json"
    result = generator.build_intervention_incremental_view(
        dataset_root=dataset_root,
        output_manifest=manifest,
        first_episode=4,
        last_episode=5,
        holdout_episode_ids=(5,),
        horizon=2,
        dry_run=True,
    )

    assert result.written is False
    assert result.row_count == 3
    assert not manifest.exists()
    assert not manifest.parent.exists()


def test_verifier_fails_closed_on_ledger_and_descriptor_mutation(
    tmp_path: Path,
):
    dataset_root = tmp_path / "mutation_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {"episode_index": 0, "valid_state": [1, 1, 1]},
            {"episode_index": 1, "valid_state": [1, 1, 1]},
        ],
    )
    manifest = tmp_path / "view.json"
    generator.build_hq_clean_h50_view(
        dataset_root=dataset_root,
        output_manifest=manifest,
        holdout_episode_ids=(1,),
        horizon=2,
    )
    loaded = view.load_frozen_view(manifest)
    original_ledger = loaded.ledger_path.read_bytes()
    loaded.ledger_path.write_bytes(original_ledger + b" ")
    with pytest.raises(ValueError, match="ledger SHA-256 mismatch"):
        view.load_frozen_view(manifest)
    loaded.ledger_path.write_bytes(original_ledger)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["rows"]["row_count"] += 1
    _canonical_write(manifest, payload)
    with pytest.raises(ValueError, match="view_id mismatch"):
        view.load_frozen_view(manifest)


def test_verifier_requires_external_source_and_representation_bindings(
    tmp_path: Path,
):
    dataset_root = tmp_path / "binding_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {"episode_index": 0, "valid_state": [1, 1, 1]},
            {"episode_index": 1, "valid_state": [1, 1, 1]},
        ],
    )
    manifest = tmp_path / "view.json"
    generator.build_hq_clean_h50_view(
        dataset_root=dataset_root,
        output_manifest=manifest,
        holdout_episode_ids=(1,),
        horizon=2,
    )
    loaded = view.load_frozen_view(manifest)
    source = loaded.descriptor["sources"][0]

    with pytest.raises(ValueError, match="representation contract mismatch"):
        view.load_frozen_view(
            manifest,
            expected_representation_contract_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="catalog_sha256 mismatch"):
        view.load_frozen_view(
            manifest,
            expected_source_hashes={
                source["source_id"]: {"catalog_sha256": "1" * 64}
            },
        )


def test_all_exhaustive_contract_rejects_weights_and_fractional_passes(
    tmp_path: Path,
):
    dataset_root = tmp_path / "policy_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {"episode_index": 0, "valid_state": [1, 1, 1]},
            {"episode_index": 1, "valid_state": [1, 1, 1]},
        ],
    )
    manifest = tmp_path / "view.json"
    generator.build_hq_clean_h50_view(
        dataset_root=dataset_root,
        output_manifest=manifest,
        holdout_episode_ids=(1,),
        horizon=2,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sources"][0]["weight"] = 0.5
    payload["view_id"] = view.descriptor_view_id(payload)
    _canonical_write(manifest, payload)
    with pytest.raises(ValueError, match="forbid implicit sampling"):
        view.load_frozen_view(manifest)

    payload.pop("view_id")
    payload["sources"][0].pop("weight")
    payload["epoch_contract"]["epoch_passes"] = 1.5
    payload["view_id"] = view.descriptor_view_id(payload)
    _canonical_write(manifest, payload)
    with pytest.raises(ValueError, match="epoch_passes"):
        view.load_frozen_view(manifest)


def test_labeled_view_fails_closed_without_consistent_subtask_catalog(
    tmp_path: Path,
):
    dataset_root = tmp_path / "subtask_binding_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {"episode_index": 0, "valid_state": [1, 1, 1]},
            {"episode_index": 1, "valid_state": [1, 1, 1]},
        ],
    )
    subtasks_path = dataset_root / "meta/subtasks.parquet"
    subtasks_path.unlink()
    with pytest.raises(FileNotFoundError, match="meta/subtasks.parquet"):
        generator.build_hq_clean_h50_view(
            dataset_root=dataset_root,
            output_manifest=tmp_path / "missing.json",
            holdout_episode_ids=(1,),
            horizon=2,
        )

    inconsistent_root = tmp_path / "subtask_inconsistent_data"
    _write_lerobot_fixture(
        inconsistent_root,
        [
            {"episode_index": 0, "valid_state": [1, 1, 1]},
            {"episode_index": 1, "valid_state": [1, 1, 1]},
        ],
    )
    data_path = inconsistent_root / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(data_path)
    data.loc[data["episode_index"] == 0, "subtask_index"] = 99
    data.to_parquet(data_path, index=False)
    with pytest.raises(ValueError, match="absent from meta/subtasks.parquet"):
        generator.build_hq_clean_h50_view(
            dataset_root=inconsistent_root,
            output_manifest=tmp_path / "unknown.json",
            holdout_episode_ids=(1,),
            horizon=2,
        )


def test_union_stats_reader_loads_local_frozen_lerobot_view(
    tmp_path: Path,
):
    dataset_root = tmp_path / "stats_reader_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {"episode_index": 0, "valid_state": [1, 1, 1]},
            {"episode_index": 1, "valid_state": [1, 1, 1]},
        ],
    )
    manifest = tmp_path / "view.json"
    result = generator.build_hq_clean_h50_view(
        dataset_root=dataset_root,
        output_manifest=manifest,
        holdout_episode_ids=(1,),
        horizon=2,
        statistics_population_candidate=True,
    )
    loaded = view.load_frozen_view(manifest)
    source = loaded.descriptor["sources"][0]
    reader = union_stats.FrozenParquetViewReader(
        {
            "id": "fixture",
            "catalog_sha256": source["catalog_sha256"],
            "reader": {
                "kind": "frozen_parquet_view",
                "view_manifest": str(manifest),
                "view_manifest_sha256": result.manifest_sha256,
                "dataset_root": str(dataset_root),
            },
        },
        tmp_path,
    )
    references = list(reader.references())
    assert len(references) == 2
    episode = references[0].load()
    assert episode.state.shape == (3, 18)
    assert episode.action.shape == (3, 18)
    assert tuple(episode.base_frame_indices) == (0, 1)


def test_statistics_candidate_includes_but_never_accumulates_holdout(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "statistics_candidate_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {
                "episode_index": 0,
                "valid_state": [1, 1, 1],
                "content_seed": 0,
            },
            {
                "episode_index": 1,
                "valid_state": [1, 1, 1],
                # Make accidental holdout accumulation obvious in max/q99.
                "content_seed": 100,
            },
        ],
    )
    split_manifest = _write_local_split_manifest(
        tmp_path / "episode-split.json",
        dataset_root=dataset_root,
        holdout_episode_ids=(1,),
    )
    train_manifest = tmp_path / "train/view.json"
    train_result = generator.build_intervention_incremental_view(
        dataset_root=dataset_root,
        output_manifest=train_manifest,
        first_episode=0,
        last_episode=1,
        eval_holdout_manifest=split_manifest,
        horizon=50,
    )
    train_view = view.load_frozen_view(train_manifest)
    assert train_view.descriptor["purpose"] == "intervention_incremental"
    assert {
        row["episode_index"] for row in train_view.iter_rows()
    } == {0}

    train_source = train_view.descriptor["sources"][0]
    with pytest.raises(
        ValueError, match="statistics_population_candidate"
    ):
        union_stats.FrozenParquetViewReader(
            {
                "id": "intervention",
                "catalog_sha256": train_source["catalog_sha256"],
                "reader": {
                    "kind": "frozen_parquet_view",
                    "view_manifest": str(train_manifest),
                    "view_manifest_sha256": (
                        train_result.manifest_sha256
                    ),
                    "dataset_root": str(dataset_root),
                },
            },
            tmp_path,
        )

    candidate_manifest = tmp_path / "candidate/view.json"
    candidate_result = generator.build_intervention_incremental_view(
        dataset_root=dataset_root,
        output_manifest=candidate_manifest,
        first_episode=0,
        last_episode=1,
        eval_holdout_manifest=split_manifest,
        horizon=50,
        statistics_population_candidate=True,
    )
    candidate_view = view.load_frozen_view(candidate_manifest)
    descriptor = candidate_view.descriptor
    assert descriptor["purpose"] == (
        view.STATISTICS_POPULATION_CANDIDATE_PURPOSE
    )
    assert descriptor["usage_contract"]["training_allowed"] is False
    assert descriptor["selection"]["evaluation_holdout"][
        "manifest_sha256"
    ] == view.file_sha256(split_manifest)
    assert {
        row["episode_index"] for row in candidate_view.iter_rows()
    } == {0, 1}
    assert descriptor["selection"][
        "authenticated_holdout_episode_indices_in_ledger"
    ] == [1]

    candidate_source = descriptor["sources"][0]
    source_id = "intervention"
    holdout_key = f"{source_id}/{dataset_root.name}:1"
    holdout_path = tmp_path / "holdout.json"
    derived_holdout_source = derive_source_holdout(
        source_id=source_id,
        kind=LEROBOT_SOURCE_KIND,
        evaluation_manifest=split_manifest,
        evaluation_manifest_sha256=view.file_sha256(split_manifest),
        candidate_view_manifest=candidate_manifest,
        candidate_view_manifest_sha256=candidate_result.manifest_sha256,
        manifest_dir=tmp_path,
    )
    assert derived_holdout_source["episode_keys"] == [holdout_key]
    _, holdout_sha256 = write_global_holdout_manifest(
        holdout_path,
        [derived_holdout_source],
    )
    population_manifest = tmp_path / "population.json"
    population_manifest.write_bytes(
        view.canonical_json_bytes(
            {
                "schema": (
                    union_stats.OPENPI_REALMAN_UNION_POPULATION_SCHEMA
                ),
                "contract_sha256": (
                    generator.DEFAULT_REPRESENTATION_SHA256
                ),
                "source_order": [source_id],
                "sources": [
                    {
                        "id": source_id,
                        "catalog_sha256": candidate_source[
                            "catalog_sha256"
                        ],
                        "reader": {
                            "kind": "frozen_parquet_view",
                            "view_manifest": str(candidate_manifest),
                            "view_manifest_sha256": (
                                candidate_result.manifest_sha256
                            ),
                            "dataset_root": str(dataset_root),
                        },
                        "provenance": {
                            "purpose": (
                                view.STATISTICS_POPULATION_CANDIDATE_PURPOSE
                            )
                        },
                    }
                ],
                "holdout": {
                    "manifest": str(holdout_path),
                    "manifest_sha256": holdout_sha256,
                    "episode_keys": [holdout_key],
                },
            }
        )
        + b"\n"
    )
    artifact, ledger = union_stats.build_union_statistics(
        population_manifest
    )
    population = artifact["population"]
    assert population["candidate_base_frames"] == 6
    assert population["unique_base_frames"] == 3
    assert population["holdout_excluded_base_frames"] == 3
    assert artifact["selected"]["state"]["count"] == [3] * 18
    assert artifact["selected"]["action"]["count"] == [150] * 18
    assert max(artifact["selected"]["state"]["max"]) < 1.0
    assert max(artifact["selected"]["action"]["max"]) < 5.0
    holdout_ledger = next(
        episode
        for episode in ledger["episodes"]
        if episode["episode_key"] == holdout_key
    )
    assert holdout_ledger["kept_base_frames"] == []
    assert holdout_ledger["holdout_base_frames"] == [0, 1, 2]

    stale_holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    stale_holdout["sources"][0]["evaluation_manifest_sha256"] = "0" * 64
    _canonical_write(holdout_path, stale_holdout)
    stale_population = json.loads(
        population_manifest.read_text(encoding="utf-8")
    )
    stale_population["holdout"]["manifest_sha256"] = view.file_sha256(
        holdout_path
    )
    _canonical_write(population_manifest, stale_population)
    with pytest.raises(
        ValueError, match="evaluation manifest SHA-256 mismatch"
    ):
        union_stats.build_union_statistics(population_manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dataset_name", "wrong-dataset", "exactly one dataset entry"),
        ("full_catalog_sha256", "0" * 64, "full_catalog_sha256"),
    ),
)
def test_local_view_rejects_unbound_eval_split(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    dataset_root = tmp_path / "split_binding_data"
    _write_lerobot_fixture(
        dataset_root,
        [
            {"episode_index": 0, "valid_state": [1, 1, 1]},
            {"episode_index": 1, "valid_state": [1, 1, 1]},
        ],
    )
    split_manifest = _write_local_split_manifest(
        tmp_path / "episode-split.json",
        dataset_root=dataset_root,
        holdout_episode_ids=(1,),
    )
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    payload["datasets"][0][field] = value
    _canonical_write(split_manifest, payload)
    with pytest.raises(ValueError, match=message):
        generator.build_hq_clean_h50_view(
            dataset_root=dataset_root,
            output_manifest=tmp_path / "view.json",
            eval_holdout_manifest=split_manifest,
            horizon=2,
        )
