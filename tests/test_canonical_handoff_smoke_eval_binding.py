from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    _stable_json_sha256,
)


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
