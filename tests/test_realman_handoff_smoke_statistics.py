from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_realman_handoff_smoke_view
from scripts import materialize_openpi_realman_handoff_smoke_statistics as smoke_stats
from scripts import materialize_realman_handoff_smoke
from starVLA.action_representation import (
    OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
    REALMAN_18D_ACTION_CONTRACT,
    deterministic_json_bytes,
)
from starVLA.dataloader import dataset_view


def _write_parent(tmp_path: Path) -> dataset_view.FrozenDatasetViewBuild:
    catalog_sha256 = "a" * 64
    content = dataset_view.make_episode_content_id(
        frame_content_sha256="b" * 64,
        length=256,
        content_contract="handoff-smoke-fixture-v1",
    )
    lineage = dataset_view.make_episode_lineage_id(
        backend="lerobot",
        source_id="fixture",
        catalog_sha256=catalog_sha256,
        episode_index=7,
        length=256,
    )
    contract_sha256 = REALMAN_18D_ACTION_CONTRACT.sha256()
    descriptor = {
        "view_name": "handoff_smoke_parent_fixture",
        "description": "test-only production parent",
        "purpose": "hq_finetune",
        "sources": [
            {
                "source_id": "fixture",
                "backend": "lerobot",
                "dataset_name": "fixture",
                "catalog_sha256": catalog_sha256,
                "annotation_sha256": "c" * 64,
                "source_content_sha256": "d" * 64,
            }
        ],
        "representation": {
            "contract_sha256": contract_sha256,
            "state_dim": 18,
            "action_dim": 18,
            "horizon": 50,
            "target_fps": 20,
        },
        "selection": {"schema": "fixture-v1"},
        "holdout_exclusions": dataset_view.make_holdout_exclusions(
            source_id="fixture",
            episode_indices=(),
            lineage_ids=(),
            content_ids=(),
        ),
        "epoch_contract": {
            "mode": dataset_view.ALL_EXHAUSTIVE_MODE,
            "epoch_passes": 1,
            "replacement": False,
            "drop_last": False,
        },
        "usage_contract": {"training_allowed": True},
    }
    rows = [
        {
            "backend": "lerobot",
            "source_id": "fixture",
            "episode_index": 7,
            "episode_length": 256,
            "base_index": base_index,
            "horizon": 50,
            "target_fps": 20,
            "end_clamp_policy": "repeat_last",
            "episode_lineage_id": lineage,
            "episode_content_id": content,
            "sample_id": dataset_view.make_sample_id(
                episode_content_id=content,
                base_index=base_index,
                horizon=50,
                target_fps=20,
                representation_contract_sha256=contract_sha256,
                end_clamp=True,
            ),
        }
        for base_index in range(160)
    ]
    return dataset_view.write_frozen_view(
        tmp_path / "parent.json",
        descriptor=descriptor,
        rows=rows,
    )


def test_smoke_parent_is_rederived_not_just_hash_copied(tmp_path: Path):
    parent = _write_parent(tmp_path)
    smoke_path = tmp_path / "smoke.json"
    built = build_realman_handoff_smoke_view.build_handoff_smoke_view(
        parent_manifest=parent.manifest_path,
        output_manifest=smoke_path,
        logical_rows=128,
    )
    smoke = dataset_view.load_frozen_view(smoke_path)

    authenticated = smoke_stats._authenticate_smoke_parent(smoke)
    assert authenticated.manifest_sha256 == parent.manifest_sha256
    assert built.row_count == 128

    payload = json.loads(smoke_path.read_text(encoding="utf-8"))
    payload["selection"]["parent_manifest_sha256"] = "f" * 64
    payload["view_id"] = dataset_view.descriptor_view_id(payload)
    smoke_path.write_bytes(dataset_view.canonical_json_bytes(payload) + b"\n")
    mutated = dataset_view.load_frozen_view(smoke_path)
    with pytest.raises(ValueError, match="parent manifest SHA-256 mismatch"):
        smoke_stats._authenticate_smoke_parent(mutated)


def test_population_is_marked_handoff_only_for_exact_source_order(
    tmp_path: Path,
):
    parent = _write_parent(tmp_path)
    smoke_path = tmp_path / "smoke.json"
    build_realman_handoff_smoke_view.build_handoff_smoke_view(
        parent_manifest=parent.manifest_path,
        output_manifest=smoke_path,
        logical_rows=128,
    )
    smoke = dataset_view.load_frozen_view(smoke_path)
    population_path = tmp_path / "population.json"
    population = {
        "schema": OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
        "contract_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
        "source_order": list(smoke_stats.SOURCE_ORDER),
        "sources": [
            {
                "id": source_id,
                "catalog_sha256": "a" * 64,
                "reader": {"kind": "fixture"},
                "provenance": {},
            }
            for source_id in smoke_stats.SOURCE_ORDER
        ],
        "holdout": {
            "manifest": "holdout.json",
            "manifest_sha256": "e" * 64,
            "episode_keys": ["hq/fixture:1"],
        },
    }
    population_path.write_bytes(deterministic_json_bytes(population) + b"\n")

    digest = smoke_stats._mark_population_handoff_only(
        population_path,
        smoke_views={source_id: smoke for source_id in smoke_stats.SOURCE_ORDER},
    )
    assert digest == hashlib.sha256(population_path.read_bytes()).hexdigest()
    marked = json.loads(population_path.read_text(encoding="utf-8"))
    for source in marked["sources"]:
        provenance = source["provenance"]["handoff_smoke"]
        assert provenance == {
            "schema": smoke_stats.HANDOFF_STATISTICS_SCHEMA,
            "scope": smoke_stats.HANDOFF_SCOPE,
            "model_quality_claim_allowed": False,
            "exact_training_logical_rows": 128,
            "parent_smoke_view": str(smoke.manifest_path),
            "parent_smoke_view_sha256": smoke.manifest_sha256,
            "parent_smoke_ledger_sha256": (
                smoke.descriptor["rows"]["sha256"]
            ),
        }


def test_population_marker_rejects_source_order_drift(tmp_path: Path):
    population_path = tmp_path / "population.json"
    population_path.write_bytes(
        deterministic_json_bytes(
            {
                "schema": OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
                "source_order": ["hq", "intervention", "realsource"],
            }
        )
        + b"\n"
    )
    with pytest.raises(ValueError, match="schema/source order"):
        smoke_stats._mark_population_handoff_only(
            population_path, smoke_views={}
        )


def test_smoke_config_materializer_requires_exact_statistics_parents():
    view_hashes = ["1" * 64, "2" * 64, "3" * 64]
    payload = {
        "population": {
            "unique_base_frames": 384,
            "sources": [
                {
                    "id": source_id,
                    "provenance": {
                        "handoff_smoke": {
                            "schema": smoke_stats.HANDOFF_STATISTICS_SCHEMA,
                            "scope": smoke_stats.HANDOFF_SCOPE,
                            "model_quality_claim_allowed": False,
                            "exact_training_logical_rows": 128,
                            "parent_smoke_view_sha256": digest,
                        }
                    },
                }
                for source_id, digest in zip(
                    smoke_stats.SOURCE_ORDER, view_hashes, strict=True
                )
            ],
        }
    }
    materialize_realman_handoff_smoke._validate_smoke_statistics_population(
        payload,
        view_hashes=view_hashes,
    )

    payload["population"]["sources"][1]["provenance"]["handoff_smoke"][
        "parent_smoke_view_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="exact supplied view"):
        materialize_realman_handoff_smoke._validate_smoke_statistics_population(
            payload,
            view_hashes=view_hashes,
        )
