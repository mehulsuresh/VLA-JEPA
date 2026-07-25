from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import starVLA.dataloader.canonical_subset_dataset as canonical_module
from starVLA.dataloader.canonical_subset_dataset import (
    CANONICAL_EVAL_SELECTION_ALGORITHM,
    CHECKPOINT_HANDOFF_SMOKE_PURPOSE,
    CHECKPOINT_HANDOFF_SMOKE_SCOPE,
    CHECKPOINT_HANDOFF_SMOKE_STATISTICS_SCHEMA,
    CHECKPOINT_HANDOFF_SMOKE_VIEW_SCHEMA,
    JOINT_DELTA_GRIPPER_ABSOLUTE,
    Q01_Q99_UNCLIPPED,
    CanonicalEvalManifest,
    CanonicalEvalSelection,
    CanonicalEvalWindow,
    CanonicalSubsetVLADataset,
    EpisodeSpec,
    ShardSpec,
    _stable_json_sha256,
)


class _FakeFrozenView(SimpleNamespace):
    def iter_rows(self):
        yield from self.identity_rows


def _evaluation() -> CanonicalEvalManifest:
    windows = tuple(
        CanonicalEvalWindow(
            dataset_id="RealSourceData/RealSource-World/fixture",
            sid="sid",
            revision="revision",
            data_file="data/chunk-000/file-000.parquet",
            episode_index=episode_index,
            base_index=0,
        )
        for episode_index in (10, 20)
    )
    selection = CanonicalEvalSelection(
        algorithm=CANONICAL_EVAL_SELECTION_ALGORITHM,
        seed=42,
        window_count=len(windows),
        candidate_count=32,
        action_horizon=50,
        action_dim=18,
        action_type=JOINT_DELTA_GRIPPER_ABSOLUTE,
        normalization=Q01_Q99_UNCLIPPED,
        adapter_contract_sha256="1" * 64,
        action_sidecar_variant="2" * 16,
        # This is intentionally the production catalog cardinality, not the
        # one-episode smoke population cardinality.
        configured_episode_count=12120,
        configured_episode_catalog_sha256="3" * 64,
        holdout_episode_count=len(windows),
    )
    return CanonicalEvalManifest(
        path=Path("/immutable/realsource-eval.json"),
        sha256="4" * 64,
        purpose="heldout",
        source_manifest_sha256="5" * 64,
        selection=selection,
        windows=windows,
    )


def _dataset() -> CanonicalSubsetVLADataset:
    evaluation = _evaluation()
    episode_identities = sorted(
        evaluation.heldout_episode_identities
    )
    binding = {
        "schema": "realsource-canonical-eval-holdout-binding-v1",
        "manifest_filename": "realsource-eval.json",
        "manifest_sha256": evaluation.sha256,
        "source_manifest_sha256": evaluation.source_manifest_sha256,
        "window_count": len(evaluation.windows),
        "episode_count": len(episode_identities),
        "episode_identity_fields": [
            "dataset_id",
            "sid",
            "revision",
            "data_file",
            "episode_index",
        ],
        "episode_identities_sha256": _stable_json_sha256(
            episode_identities
        ),
        "selected_population_overlap_episode_count": len(
            episode_identities
        ),
        "selected_population_overlap_target_row_count": 1000,
        "copy_detection": [
            "episode_identity",
            "episode_lineage_id",
            "episode_content_id",
        ],
    }
    binding["sha256"] = _stable_json_sha256(binding)
    descriptor = {
        "purpose": CHECKPOINT_HANDOFF_SMOKE_PURPOSE,
        "usage_contract": {
            "training_allowed": True,
            "scope": CHECKPOINT_HANDOFF_SMOKE_SCOPE,
            "model_quality_claim_allowed": False,
            "statistics_accumulation": "forbidden",
        },
        "selection": {
            "schema": CHECKPOINT_HANDOFF_SMOKE_VIEW_SCHEMA,
            "algorithm": (
                "ordered_logical_prefix_from_authenticated_parent_v1"
            ),
            "parent_manifest": "/immutable/production-view.json",
            "parent_manifest_sha256": "6" * 64,
            "parent_view_id": "7" * 64,
            "parent_ledger_sha256": "8" * 64,
            "requested_logical_row_count": 128,
            "evaluation_holdout": binding,
        },
        "holdout_exclusions": {
            "source_id": (
                f"canonical_eval_manifest:{evaluation.sha256}"
            ),
            "lineage_ids": ["9" * 64],
            "content_ids": ["a" * 64],
        },
        "rows": {"sha256": "b" * 64},
    }
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.frozen_train_view = SimpleNamespace(
        descriptor=descriptor,
        manifest_sha256="c" * 64,
        row_count=128,
        unique_sample_count=128,
    )
    dataset.canonical_eval_manifest = evaluation
    dataset.normalization_statistics = {
        "population": {
            "source_order": ["realsource", "intervention", "hq"],
            "unique_base_frames": 384,
            "sources": [
                {
                    "id": "realsource",
                    "provenance": {
                        "handoff_smoke": {
                            "schema": (
                                CHECKPOINT_HANDOFF_SMOKE_STATISTICS_SCHEMA
                            ),
                            "scope": CHECKPOINT_HANDOFF_SMOKE_SCOPE,
                            "model_quality_claim_allowed": False,
                            "exact_training_logical_rows": 128,
                            "parent_smoke_view_sha256": "c" * 64,
                            "parent_smoke_ledger_sha256": "b" * 64,
                        }
                    },
                },
                {"id": "intervention"},
                {"id": "hq"},
            ],
        }
    }
    return dataset


def test_handoff_smoke_accepts_production_eval_catalog_lineage() -> None:
    dataset = _dataset()

    assert (
        dataset._validate_checkpoint_handoff_smoke_eval_binding()
        is True
    )
    assert (
        dataset.canonical_eval_manifest.selection.configured_episode_count
        == 12120
    )


def _shard_with_episodes(
    tmp_path: Path,
    *episode_indices: int,
) -> ShardSpec:
    return ShardSpec(
        dataset_id="RealSourceData/RealSource-World/fixture",
        sid="sid",
        revision="revision",
        adapter_group_id="realman",
        adapter_path=tmp_path / "adapter.yaml",
        root=tmp_path,
        gcs_prefix="gs://fixture",
        data_relative_path="data/chunk-000/file-000.parquet",
        data_path=tmp_path / "data.parquet",
        sidecar_path=tmp_path / "sidecar.npz",
        fps=30.0,
        camera_source_keys={},
        qwen_camera_slots=(),
        vjepa_camera_slots=(),
        decode_camera_slots=(),
        task_map={0: "fixture"},
        episodes=[
            EpisodeSpec(
                local_start=index * 100,
                length=100,
                task="fixture",
                video_paths={},
                video_base_frames={},
                episode_index=episode_index,
            )
            for index, episode_index in enumerate(episode_indices)
        ],
    )


def test_authenticated_handoff_smoke_filters_partial_holdout_catalog(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    shard = _shard_with_episodes(tmp_path, 10, 30)

    authenticated_smoke = (
        dataset._validate_checkpoint_handoff_smoke_eval_binding()
    )
    filtered = dataset._without_heldout_episodes(
        [shard],
        allow_partial_catalog=authenticated_smoke,
    )

    # Heldout episode 10 is present in this tiny smoke source and must still
    # be removed. Heldout episode 20 belongs to the larger authenticated
    # production eval population and is intentionally absent from this smoke.
    assert authenticated_smoke is True
    assert [
        episode.episode_index
        for episode in filtered[0].episodes
    ] == [30]


def test_ordinary_training_rejects_partial_holdout_catalog(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    dataset.frozen_train_view.descriptor["purpose"] = "training"
    shard = _shard_with_episodes(tmp_path, 10, 30)

    authenticated_smoke = (
        dataset._validate_checkpoint_handoff_smoke_eval_binding()
    )
    assert authenticated_smoke is False
    with pytest.raises(
        ValueError,
        match="episodes outside the configured canonical stream",
    ):
        dataset._without_heldout_episodes(
            [shard],
            allow_partial_catalog=authenticated_smoke,
        )


def test_ordinary_production_view_does_not_enter_smoke_exception() -> None:
    dataset = _dataset()
    dataset.frozen_train_view.descriptor["purpose"] = "training"

    assert (
        dataset._validate_checkpoint_handoff_smoke_eval_binding()
        is False
    )


def test_handoff_smoke_rejects_eval_manifest_drift() -> None:
    dataset = _dataset()
    dataset.canonical_eval_manifest = deepcopy(
        dataset.canonical_eval_manifest
    )
    object.__setattr__(
        dataset.canonical_eval_manifest, "sha256", "d" * 64
    )

    with pytest.raises(
        ValueError, match="exact production evaluation population"
    ):
        dataset._validate_checkpoint_handoff_smoke_eval_binding()


def test_handoff_smoke_rejects_missing_eval_manifest() -> None:
    dataset = _dataset()
    dataset.canonical_eval_manifest = None

    with pytest.raises(
        ValueError, match="requires its exact production evaluation manifest"
    ):
        dataset._validate_checkpoint_handoff_smoke_eval_binding()


def test_handoff_smoke_rejects_unhashed_binding_drift() -> None:
    dataset = _dataset()
    dataset.frozen_train_view.descriptor["selection"][
        "evaluation_holdout"
    ]["window_count"] = 1

    with pytest.raises(
        ValueError, match="exact production evaluation population"
    ):
        dataset._validate_checkpoint_handoff_smoke_eval_binding()


def test_handoff_smoke_rejects_missing_parent_lineage() -> None:
    dataset = _dataset()
    dataset.frozen_train_view.descriptor["selection"][
        "parent_ledger_sha256"
    ] = "not-a-digest"

    with pytest.raises(
        ValueError, match="production-parent lineage"
    ):
        dataset._validate_checkpoint_handoff_smoke_eval_binding()


def test_handoff_smoke_rejects_full_or_other_statistics() -> None:
    dataset = _dataset()
    dataset.normalization_statistics["population"]["sources"][0][
        "provenance"
    ].pop("handoff_smoke")

    with pytest.raises(
        ValueError, match="smoke-only normalization population"
    ):
        dataset._validate_checkpoint_handoff_smoke_eval_binding()


def test_handoff_smoke_rejects_statistics_from_another_smoke_view() -> None:
    dataset = _dataset()
    smoke = dataset.normalization_statistics["population"]["sources"][0][
        "provenance"
    ]["handoff_smoke"]
    smoke["parent_smoke_view_sha256"] = "e" * 64

    with pytest.raises(
        ValueError, match="smoke-only normalization population"
    ):
        dataset._validate_checkpoint_handoff_smoke_eval_binding()


def _eval_catalog_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    through_smoke_parent: bool,
) -> tuple[
    CanonicalSubsetVLADataset,
    _FakeFrozenView,
    frozenset[tuple[str, str, str, str, int]],
    Path,
    str,
]:
    dataset = _dataset()
    original = dataset.canonical_eval_manifest
    train_identity = (
        "RealSourceData/RealSource-World/fixture",
        "sid",
        "revision",
        "data/chunk-000/file-000.parquet",
        30,
    )
    logical_catalog = frozenset(
        {*original.heldout_episode_identities, train_identity}
    )
    selection = replace(
        original.selection,
        configured_episode_count=len(logical_catalog),
        configured_episode_catalog_sha256=_stable_json_sha256(
            sorted(logical_catalog)
        ),
    )
    evaluation = replace(original, selection=selection)
    dataset.mode = "eval"
    dataset.canonical_eval_manifest = evaluation
    dataset.frozen_train_view = None
    source_manifest = tmp_path / "canonical.jsonl.gz"
    source_manifest.write_bytes(b"canonical source\n")
    dataset.manifest_path = source_manifest

    binding = deepcopy(
        _dataset().frozen_train_view.descriptor["selection"][
            "evaluation_holdout"
        ]
    )
    source = {
        "backend": "canonical",
        "source_id": "fixture",
        "dataset_id": train_identity[0],
        "sid": train_identity[1],
        "revision": train_identity[2],
        "manifest_sha256": evaluation.source_manifest_sha256,
    }
    representation = {
        "contract_sha256": canonical_module.REALMAN_18D_ACTION_CONTRACT.sha256(),
        "state_dim": 18,
        "action_dim": 18,
        "horizon": 50,
        "action_type": JOINT_DELTA_GRIPPER_ABSOLUTE,
        "normalization": Q01_Q99_UNCLIPPED,
    }
    parent_path = tmp_path / "production.json"
    parent_path.write_bytes(b"production view\n")
    parent_sha = "d" * 64
    parent_ledger_sha = "e" * 64
    parent_view_id = "f" * 64
    parent = _FakeFrozenView(
        manifest_path=parent_path.resolve(),
        manifest_sha256=parent_sha,
        view_id=parent_view_id,
        episode_count=1,
        identity_rows=[
            {
                "dataset_id": train_identity[0],
                "sid": train_identity[1],
                "revision": train_identity[2],
                "data_file": train_identity[3],
                "episode_index": train_identity[4],
            }
        ],
        descriptor={
            "purpose": "realsource_pretraining",
            "usage_contract": {
                "training_allowed": True,
                "eval_manifest_generation": False,
            },
            "representation": representation,
            "epoch_contract": {"epoch_passes": 1},
            "sources": [source],
            "selection": {"evaluation_holdout": binding},
            "holdout_exclusions": {
                "source_id": (
                    f"canonical_eval_manifest:{evaluation.sha256}"
                ),
                "lineage_ids": ["1" * 64],
                "content_ids": ["2" * 64],
            },
            "rows": {"sha256": parent_ledger_sha},
        },
    )

    configured_path = parent_path
    configured_sha = parent_sha
    views = {parent_path.resolve(): parent}
    hashes = {
        source_manifest.resolve(): evaluation.source_manifest_sha256,
        parent_path.resolve(): parent_sha,
    }
    if through_smoke_parent:
        smoke_path = tmp_path / "smoke.json"
        smoke_path.write_bytes(b"smoke view\n")
        smoke_sha = "a" * 64
        smoke_ledger_sha = "b" * 64
        smoke_descriptor = deepcopy(
            _dataset().frozen_train_view.descriptor
        )
        smoke_descriptor["selection"].update(
            {
                "parent_manifest": str(parent_path.resolve()),
                "parent_manifest_sha256": parent_sha,
                "parent_view_id": parent_view_id,
                "parent_ledger_sha256": parent_ledger_sha,
            }
        )
        smoke_descriptor["representation"] = representation
        smoke_descriptor["epoch_contract"] = {"epoch_passes": 1}
        smoke_descriptor["sources"] = [source]
        smoke_descriptor["rows"] = {"sha256": smoke_ledger_sha}
        smoke = _FakeFrozenView(
            manifest_path=smoke_path.resolve(),
            manifest_sha256=smoke_sha,
            view_id="c" * 64,
            episode_count=1,
            row_count=128,
            unique_sample_count=128,
            identity_rows=[
                {
                    "dataset_id": train_identity[0],
                    "sid": train_identity[1],
                    "revision": train_identity[2],
                    "data_file": train_identity[3],
                    "episode_index": train_identity[4],
                }
            ],
            descriptor=smoke_descriptor,
        )
        smoke_statistics = dataset.normalization_statistics["population"][
            "sources"
        ][0]["provenance"]["handoff_smoke"]
        smoke_statistics["parent_smoke_view_sha256"] = smoke_sha
        smoke_statistics["parent_smoke_ledger_sha256"] = smoke_ledger_sha
        configured_path = smoke_path
        configured_sha = smoke_sha
        views[smoke_path.resolve()] = smoke
        hashes[smoke_path.resolve()] = smoke_sha

    def fake_hash(path: Path) -> str:
        return hashes[Path(path).resolve()]

    def fake_load(path: Path, **kwargs):
        view = views[Path(path).resolve()]
        expected_view_id = kwargs.get("expected_view_id")
        if expected_view_id is not None:
            assert view.view_id == expected_view_id
        return view

    monkeypatch.setattr(canonical_module, "_hash_file", fake_hash)
    monkeypatch.setattr(canonical_module, "load_frozen_view", fake_load)
    return (
        dataset,
        parent,
        logical_catalog,
        configured_path,
        configured_sha,
    )


@pytest.mark.parametrize("through_smoke_parent", [False, True])
def test_eval_catalog_is_exact_production_train_union_holdout(
    tmp_path: Path,
    monkeypatch,
    through_smoke_parent: bool,
) -> None:
    (
        dataset,
        parent,
        logical_catalog,
        configured_path,
        configured_sha,
    ) = _eval_catalog_fixture(
        tmp_path,
        monkeypatch,
        through_smoke_parent=through_smoke_parent,
    )

    dataset._load_eval_catalog_view_descriptor(
        configured_path,
        expected_manifest_sha256=configured_sha,
    )

    assert dataset.eval_catalog_view is parent
    assert dataset._eval_logical_episode_identities == logical_catalog
    assert (
        dataset.eval_catalog_view_manifest_sha256
        == parent.manifest_sha256
    )


def test_eval_catalog_rejects_train_holdout_overlap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (
        dataset,
        parent,
        _,
        configured_path,
        configured_sha,
    ) = _eval_catalog_fixture(
        tmp_path,
        monkeypatch,
        through_smoke_parent=False,
    )
    heldout = next(
        iter(dataset.canonical_eval_manifest.heldout_episode_identities)
    )
    parent.identity_rows[0] = {
        "dataset_id": heldout[0],
        "sid": heldout[1],
        "revision": heldout[2],
        "data_file": heldout[3],
        "episode_index": heldout[4],
    }

    with pytest.raises(ValueError, match="contains heldout episodes"):
        dataset._load_eval_catalog_view_descriptor(
            configured_path,
            expected_manifest_sha256=configured_sha,
        )
