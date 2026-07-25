#!/usr/bin/env python3
"""Materialize isolated normalization for the three-stage handoff smoke.

This command is deliberately *not* a production statistics builder.  It
authenticates each 128-row checkpoint-handoff view against the production
training view from which it was derived, preserves those exact 128 training
rows, and adds the immutable evaluation episodes as reference-only rows.  The
normal production union holdout and OpenPI q01/q99 implementations then:

* re-derive the exact eval episode keys;
* hash the projected content of every referenced episode;
* exclude every eval episode from accumulation; and
* accumulate only the 384 smoke-training rows.

The resulting artifact exists only to let a one-step RealSource -> intervention
-> HQ run exercise checkpoint serialization and resume.  Its provenance
explicitly forbids model-quality or production-normalization claims.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_openpi_realman_union_contract as union_contract  # noqa: E402
from scripts import build_realman_dataset_views as view_builder  # noqa: E402
from scripts import build_realman_handoff_smoke_view as smoke_builder  # noqa: E402
from scripts import compute_openpi_realman_union_stats as union_stats  # noqa: E402
from scripts import (  # noqa: E402
    materialize_realman_handoff_smoke as smoke_materializer,
)
from starVLA.action_representation import (  # noqa: E402
    OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
    REALMAN_18D_ACTION_CONTRACT,
    deterministic_json_bytes,
)
from starVLA.dataloader import dataset_view  # noqa: E402


SOURCE_ORDER = ("realsource", "intervention", "hq")
HANDOFF_STATISTICS_SCHEMA = "realman-handoff-smoke-statistics-v1"
HANDOFF_SCOPE = "checkpoint_handoff_validation_only"
RESULT_PREFIX = "REALMAN_HANDOFF_SMOKE_STATISTICS="


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_generated(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("schema", None)
    result.pop("ordinal", None)
    return result


def _input_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _input_dir(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _smoke_rows(view: dataset_view.FrozenDatasetView) -> list[dict[str, Any]]:
    return [_strip_generated(row) for row in view.iter_rows()]


def _resolve_parent_path(
    smoke: dataset_view.FrozenDatasetView,
    override: str | Path | None,
) -> Path:
    selection = smoke.descriptor["selection"]
    raw = override if override is not None else selection["parent_manifest"]
    return _input_file(raw, label="authenticated smoke parent view")


def _authenticate_smoke_parent(
    smoke: dataset_view.FrozenDatasetView,
    *,
    parent_override: str | Path | None = None,
) -> dataset_view.FrozenDatasetView:
    """Re-derive the smoke prefix from the exact authenticated parent."""

    selection = smoke.descriptor["selection"]
    parent_path = _resolve_parent_path(smoke, parent_override)
    expected_manifest = str(selection["parent_manifest_sha256"])
    actual_manifest = _sha256(parent_path)
    if actual_manifest != expected_manifest:
        raise ValueError(
            "Smoke parent manifest SHA-256 mismatch: "
            f"expected {expected_manifest}, got {actual_manifest}."
        )
    parent = dataset_view.load_frozen_view(
        parent_path,
        expected_view_id=str(selection["parent_view_id"]),
        expected_representation_contract_sha256=(
            REALMAN_18D_ACTION_CONTRACT.sha256()
        ),
        verify_ledger=True,
    )
    if parent.descriptor["rows"]["sha256"] != selection[
        "parent_ledger_sha256"
    ]:
        raise ValueError("Smoke parent ledger SHA-256 does not match lineage.")
    requested = int(selection["requested_logical_row_count"])
    if parent.encoding == dataset_view.EXPANDED_ROWS_ENCODING:
        expected = smoke_builder._expanded_prefix(parent, requested)
    elif parent.encoding == dataset_view.EPISODE_RANGES_ENCODING:
        expected = smoke_builder._range_prefix(parent, requested)
    else:  # pragma: no cover - load_frozen_view rejects this first.
        raise ValueError(f"Unsupported parent encoding {parent.encoding!r}.")
    if expected != _smoke_rows(smoke):
        raise ValueError(
            "Smoke ledger is not the declared exact ordered parent prefix."
        )
    return parent


def _require_eval_binding(
    smoke: dataset_view.FrozenDatasetView,
    parent: dataset_view.FrozenDatasetView,
    evaluation_manifest: Path,
) -> Mapping[str, Any]:
    evaluation_sha256 = _sha256(evaluation_manifest)
    bindings: list[Mapping[str, Any]] = []
    for label, view in (("smoke", smoke), ("parent", parent)):
        raw = view.descriptor.get("selection", {}).get(
            "evaluation_holdout"
        )
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{label} view lacks an authenticated evaluation binding."
            )
        if raw.get("manifest_sha256") != evaluation_sha256:
            raise ValueError(
                f"{label} view binds a different evaluation manifest."
            )
        bindings.append(raw)
    if dict(bindings[0]) != dict(bindings[1]):
        raise ValueError("Smoke and parent evaluation bindings differ.")
    return bindings[0]


def _statistics_descriptor(
    *,
    source_id: str,
    smoke: dataset_view.FrozenDatasetView,
    parent: dataset_view.FrozenDatasetView,
    sources: Sequence[Mapping[str, Any]],
    evaluation_binding: Mapping[str, Any],
    holdout_exclusions: Mapping[str, Any],
    selected_episode_count: int,
    authenticated_holdout: Sequence[Any],
) -> dict[str, Any]:
    return {
        "view_name": f"{source_id}_handoff_smoke_statistics_population_v1",
        "description": (
            "Reference-complete statistics population for the one-step "
            "checkpoint-handoff smoke. Only the exact authenticated 128-row "
            "smoke prefix may contribute to q01/q99."
        ),
        "purpose": dataset_view.STATISTICS_POPULATION_CANDIDATE_PURPOSE,
        "sources": [deepcopy(dict(source)) for source in sources],
        "representation": deepcopy(dict(smoke.descriptor["representation"])),
        "selection": {
            "schema": HANDOFF_STATISTICS_SCHEMA,
            "scope": HANDOFF_SCOPE,
            "model_quality_claim_allowed": False,
            "statistics_use": "handoff_smoke_runtime_only",
            "source_id": source_id,
            "parent_smoke_view": str(smoke.manifest_path),
            "parent_smoke_view_sha256": smoke.manifest_sha256,
            "parent_smoke_ledger_sha256": (
                smoke.descriptor["rows"]["sha256"]
            ),
            "production_parent_view": str(parent.manifest_path),
            "production_parent_view_sha256": parent.manifest_sha256,
            "exact_training_logical_rows": 128,
            "selected_episode_count": int(selected_episode_count),
            "authenticated_holdout_episode_indices_in_ledger": list(
                authenticated_holdout
            ),
            "evaluation_holdout": deepcopy(dict(evaluation_binding)),
        },
        "holdout_exclusions": deepcopy(dict(holdout_exclusions)),
        "epoch_contract": {
            "mode": dataset_view.ALL_EXHAUSTIVE_MODE,
            "epoch_passes": 1,
            "replacement": False,
            "drop_last": False,
            "shuffle": "deterministic_bijection_per_epoch",
            "ddp_tail": "duplicated_padding_reported_separately",
        },
        "usage_contract": {
            "training_allowed": False,
            "statistics_accumulation": (
                "union_builder_must_exclude_authenticated_holdout_keys"
            ),
            "scope": HANDOFF_SCOPE,
            "model_quality_claim_allowed": False,
        },
        "generator": {
            "schema": HANDOFF_STATISTICS_SCHEMA,
            "script": (
                "scripts/"
                "materialize_openpi_realman_handoff_smoke_statistics.py"
            ),
        },
    }


def _lerobot_reference_row(
    snapshot: view_builder.EpisodeSnapshot,
    *,
    source_id: str,
    horizon: int,
    target_fps: int,
) -> dict[str, Any]:
    base_index = 0
    end_index = min(horizon - 1, snapshot.record.length - 1)
    return {
        "backend": "lerobot",
        "source_id": source_id,
        "episode_index": snapshot.record.episode_index,
        "episode_length": snapshot.record.length,
        "base_index": base_index,
        "end_index": end_index,
        "horizon": horizon,
        "target_fps": target_fps,
        "end_clamp_policy": "repeat_last",
        "end_clamped": end_index < horizon - 1,
        "data_file": snapshot.record.data_file,
        "selection_kind": "authenticated_holdout_reference_only",
        "episode_lineage_id": snapshot.lineage_id,
        "episode_content_id": snapshot.content_id,
        "sample_id": dataset_view.make_sample_id(
            episode_content_id=snapshot.content_id,
            base_index=base_index,
            horizon=horizon,
            target_fps=target_fps,
            representation_contract_sha256=(
                REALMAN_18D_ACTION_CONTRACT.sha256()
            ),
            end_clamp=True,
        ),
    }


def _build_lerobot_candidate(
    *,
    source_id: str,
    smoke_path: Path,
    parent_override: Path | None,
    evaluation_manifest: Path,
    dataset_root: Path,
    output_path: Path,
) -> dataset_view.FrozenDatasetViewBuild:
    smoke_materializer._validate_smoke_view(
        smoke_path, expected_source=source_id
    )
    smoke = dataset_view.load_frozen_view(
        smoke_path,
        expected_representation_contract_sha256=(
            REALMAN_18D_ACTION_CONTRACT.sha256()
        ),
        verify_ledger=True,
    )
    parent = _authenticate_smoke_parent(
        smoke, parent_override=parent_override
    )
    _require_eval_binding(smoke, parent, evaluation_manifest)
    catalog, catalog_binding = view_builder._load_episode_catalog(
        dataset_root
    )
    holdout, evaluation_binding = view_builder._bind_local_eval_holdout(
        manifest_path=evaluation_manifest,
        dataset_root=dataset_root,
        catalog=catalog,
        catalog_binding=catalog_binding,
    )
    source = smoke.descriptor["sources"][0]
    if (
        source.get("catalog_sha256")
        != catalog_binding["catalog_sha256"]
        or source.get("dataset_name") != dataset_root.name
    ):
        raise ValueError(
            f"{source_id} smoke view does not bind the supplied dataset root."
        )

    training_rows = _smoke_rows(smoke)
    candidate_episode_ids = sorted(
        {int(row["episode_index"]) for row in training_rows}
    )
    if set(candidate_episode_ids) & set(holdout):
        raise ValueError(
            f"{source_id} smoke training rows overlap eval holdout episodes."
        )
    scope = sorted(set(candidate_episode_ids) | set(holdout))
    snapshots, annotation_sha256, source_content_sha256 = (
        view_builder._load_episode_snapshots(
            dataset_root=dataset_root,
            source_id=dataset_root.name,
            catalog=catalog,
            catalog_sha256=catalog_binding["catalog_sha256"],
            episode_ids=scope,
        )
    )
    for row in training_rows:
        snapshot = snapshots[int(row["episode_index"])]
        expected = {
            "data_file": snapshot.record.data_file,
            "episode_lineage_id": snapshot.lineage_id,
            "episode_content_id": snapshot.content_id,
        }
        mismatches = {
            key: {"smoke": row.get(key), "dataset": value}
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{source_id} smoke row/dataset identity mismatch: "
                f"{mismatches}."
            )
        if int(row["base_index"]) >= snapshot.record.length:
            raise ValueError(f"{source_id} smoke base index is out of range.")

    holdout_snapshots = [snapshots[index] for index in holdout]
    rows = [
        *training_rows,
        *(
            _lerobot_reference_row(
                snapshot,
                source_id=dataset_root.name,
                horizon=int(smoke.descriptor["representation"]["horizon"]),
                target_fps=int(
                    smoke.descriptor["representation"]["target_fps"]
                ),
            )
            for snapshot in holdout_snapshots
        ),
    ]
    rows.sort(
        key=lambda row: (
            int(row["episode_index"]),
            int(row["base_index"]),
        )
    )
    source_descriptor = deepcopy(dict(source))
    source_descriptor.update(
        {
            "dataset_root_hint": str(dataset_root),
            "annotation_sha256": annotation_sha256,
            "source_content_sha256": source_content_sha256,
            "annotation_scope_episode_count": len(scope),
            "selected_data_shards": (
                view_builder._selected_data_shard_bindings(
                    dataset_root=dataset_root,
                    snapshots=snapshots,
                    selected_episode_ids=scope,
                )
            ),
            "episode_split_manifest_sha256": _sha256(
                evaluation_manifest
            ),
        }
    )
    exclusions = dataset_view.make_holdout_exclusions(
        source_id=dataset_root.name,
        episode_indices=holdout,
        lineage_ids=[
            snapshot.lineage_id for snapshot in holdout_snapshots
        ],
        content_ids=[
            snapshot.content_id for snapshot in holdout_snapshots
        ],
    )
    descriptor = _statistics_descriptor(
        source_id=source_id,
        smoke=smoke,
        parent=parent,
        sources=[source_descriptor],
        evaluation_binding=evaluation_binding,
        holdout_exclusions=exclusions,
        selected_episode_count=len(scope),
        authenticated_holdout=holdout,
    )
    return dataset_view.write_frozen_view(
        output_path,
        descriptor=descriptor,
        rows=rows,
    )


def _canonical_reference_range(
    *,
    catalog: view_builder.RealSourceTaskCatalog,
    episode: view_builder.RealSourceEpisode,
    horizon: int,
    adapter_sha256: str,
) -> dict[str, Any]:
    base_start = 0
    base_stop = 1
    return {
        "backend": "canonical",
        "source_id": catalog.dataset_id,
        "dataset_id": catalog.dataset_id,
        "sid": catalog.sid,
        "revision": catalog.revision,
        "episode_index": episode.episode_index,
        "source_episode_length": episode.length,
        "target_episode_length": episode.target_row_count,
        "base_start": base_start,
        "base_stop": base_stop,
        "base_step": 1,
        "sample_count": 1,
        "horizon": horizon,
        "target_fps": view_builder.DEFAULT_TARGET_FPS,
        "source_fps": catalog.fps,
        "end_clamp_policy": "repeat_last",
        "data_file": episode.data_file,
        "adapter_sha256": adapter_sha256,
        "selection_kind": "authenticated_holdout_reference_only",
        "annotation_ordinal": episode.annotation_ordinal,
        "annotation_episode_index": episode.annotation_episode_index,
        "annotation_sha256": episode.annotation_sha256,
        "episode_lineage_id": episode.lineage_id,
        "episode_content_id": episode.content_id,
        "range_id": dataset_view.make_range_id(
            episode_content_id=episode.content_id,
            base_start=base_start,
            base_stop=base_stop,
            base_step=1,
            horizon=horizon,
            target_fps=view_builder.DEFAULT_TARGET_FPS,
            representation_contract_sha256=(
                REALMAN_18D_ACTION_CONTRACT.sha256()
            ),
            end_clamp=True,
        ),
    }


def _canonical_source_descriptor(
    catalog: view_builder.RealSourceTaskCatalog,
    *,
    manifest_sha256: str,
    adapter_sha256: str,
) -> dict[str, Any]:
    return {
        "source_id": catalog.dataset_id,
        "backend": "canonical",
        "dataset_id": catalog.dataset_id,
        "sid": catalog.sid,
        "revision": catalog.revision,
        "fps": catalog.fps,
        "gcs_prefix": (
            "gs://robotics-datasets-yonduai/raw/"
            f"{catalog.sid}/{catalog.revision}/files"
        ),
        "catalog_sha256": manifest_sha256,
        "manifest_sha256": manifest_sha256,
        "adapter_sha256": adapter_sha256,
        "annotation_sha256": catalog.annotation_sha256,
        "subtask_segments_sha256": catalog.subtask_segments_sha256,
        "subtask_segments_schema": view_builder.SUBTASK_SEGMENTS_SCHEMA,
        "subtask_segments_summary": dict(
            catalog.subtask_segments_summary
        ),
        "source_content_sha256": (
            view_builder._realsource_source_content_sha256(catalog)
        ),
        "episode_metadata_sha256": catalog.metadata_sha256,
        "annotation_alignment": dict(catalog.alignment),
    }


def _build_realsource_candidate(
    *,
    smoke_path: Path,
    parent_override: Path | None,
    evaluation_manifest: Path,
    canonical_manifest: Path,
    adapter_path: Path,
    cache_dir: Path,
    output_path: Path,
) -> dataset_view.FrozenDatasetViewBuild:
    smoke_materializer._validate_smoke_view(
        smoke_path, expected_source="realsource"
    )
    smoke = dataset_view.load_frozen_view(
        smoke_path,
        expected_representation_contract_sha256=(
            REALMAN_18D_ACTION_CONTRACT.sha256()
        ),
        verify_ledger=True,
    )
    parent = _authenticate_smoke_parent(
        smoke, parent_override=parent_override
    )
    _require_eval_binding(smoke, parent, evaluation_manifest)
    catalogs, manifest_sha256, adapter_sha256 = (
        view_builder._load_realsource_catalog(
            canonical_manifest=canonical_manifest,
            adapter_path=adapter_path,
            cache_dir=cache_dir,
        )
    )
    if any(
        source.get("catalog_sha256") != manifest_sha256
        or source.get("adapter_sha256") != adapter_sha256
        for source in smoke.descriptor["sources"]
    ):
        raise ValueError(
            "RealSource smoke view does not bind the supplied canonical "
            "manifest/adapter."
        )
    evaluation_binding, holdout_identities, holdout_episodes = (
        view_builder._bind_realsource_eval_holdout(
            eval_manifest_path=evaluation_manifest,
            canonical_manifest_sha256=manifest_sha256,
            catalogs=catalogs,
        )
    )
    catalog_by_id = {catalog.dataset_id: catalog for catalog in catalogs}
    episode_by_identity = {
        (
            catalog.dataset_id,
            catalog.sid,
            catalog.revision,
            episode.data_file,
            episode.episode_index,
        ): (catalog, episode)
        for catalog in catalogs
        for episode in catalog.valid_episodes
    }
    training_ranges = _smoke_rows(smoke)
    training_identities = {
        (
            row["dataset_id"],
            row["sid"],
            row["revision"],
            row["data_file"],
            int(row["episode_index"]),
        )
        for row in training_ranges
    }
    overlap = training_identities & set(holdout_identities)
    if overlap:
        raise ValueError(
            "RealSource smoke training rows overlap eval holdout episodes: "
            f"{sorted(overlap)[:5]}."
        )
    for row in training_ranges:
        identity = (
            row["dataset_id"],
            row["sid"],
            row["revision"],
            row["data_file"],
            int(row["episode_index"]),
        )
        if identity not in episode_by_identity:
            raise ValueError(
                f"RealSource smoke row is outside the strict-valid catalog: "
                f"{identity}."
            )
        _catalog, episode = episode_by_identity[identity]
        expected = {
            "source_episode_length": episode.length,
            "episode_lineage_id": episode.lineage_id,
            "episode_content_id": episode.content_id,
            "annotation_sha256": episode.annotation_sha256,
        }
        mismatches = {
            key: {"smoke": row.get(key), "catalog": value}
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "RealSource smoke row/catalog identity mismatch: "
                f"{mismatches}."
            )

    horizon = int(smoke.descriptor["representation"]["horizon"])
    holdout_ranges = [
        _canonical_reference_range(
            catalog=catalog_by_id[episode.dataset_id],
            episode=episode,
            horizon=horizon,
            adapter_sha256=adapter_sha256,
        )
        for episode in holdout_episodes
    ]
    ranges = [*training_ranges, *holdout_ranges]
    ranges.sort(
        key=lambda row: (
            str(row["dataset_id"]),
            str(row["data_file"]),
            int(row["episode_index"]),
            int(row["base_start"]),
        )
    )
    referenced_source_ids = sorted(
        {str(row["source_id"]) for row in ranges}
    )
    sources = [
        _canonical_source_descriptor(
            catalog_by_id[source_id],
            manifest_sha256=manifest_sha256,
            adapter_sha256=adapter_sha256,
        )
        for source_id in referenced_source_ids
    ]
    exclusions = dataset_view.make_holdout_exclusions(
        source_id=f"canonical_eval_manifest:{_sha256(evaluation_manifest)}",
        episode_indices=[
            episode.episode_index for episode in holdout_episodes
        ],
        lineage_ids=[
            episode.lineage_id for episode in holdout_episodes
        ],
        content_ids=[
            episode.content_id for episode in holdout_episodes
        ],
    )
    descriptor = _statistics_descriptor(
        source_id="realsource",
        smoke=smoke,
        parent=parent,
        sources=sources,
        evaluation_binding=evaluation_binding,
        holdout_exclusions=exclusions,
        selected_episode_count=(
            len(training_identities) + len(holdout_identities)
        ),
        authenticated_holdout=[
            list(identity) for identity in sorted(holdout_identities)
        ],
    )
    return dataset_view.write_frozen_range_view(
        output_path,
        descriptor=descriptor,
        ranges=ranges,
    )


def _mark_population_handoff_only(
    population_path: Path,
    *,
    smoke_views: Mapping[str, dataset_view.FrozenDatasetView],
) -> str:
    payload = json.loads(population_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != OPENPI_REALMAN_UNION_POPULATION_SCHEMA
        or payload.get("source_order") != list(SOURCE_ORDER)
    ):
        raise ValueError("Unexpected union population schema/source order.")
    sources = payload.get("sources")
    if not isinstance(sources, list) or [
        source.get("id") for source in sources
    ] != list(SOURCE_ORDER):
        raise ValueError("Union population sources do not match source order.")
    for source in sources:
        source_id = str(source["id"])
        provenance = source.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"Population source {source_id} lacks provenance."
            )
        smoke = smoke_views[source_id]
        provenance["handoff_smoke"] = {
            "schema": HANDOFF_STATISTICS_SCHEMA,
            "scope": HANDOFF_SCOPE,
            "model_quality_claim_allowed": False,
            "exact_training_logical_rows": 128,
            "parent_smoke_view": str(smoke.manifest_path),
            "parent_smoke_view_sha256": smoke.manifest_sha256,
            "parent_smoke_ledger_sha256": (
                smoke.descriptor["rows"]["sha256"]
            ),
        }
    encoded = deterministic_json_bytes(payload) + b"\n"
    population_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _atomic_publish(staging: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if output.exists():
        backup = output.with_name(
            f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.backup"
        )
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def materialize(
    *,
    output_dir: str | Path,
    realsource_view: str | Path,
    intervention_view: str | Path,
    hq_view: str | Path,
    realsource_eval_manifest: str | Path,
    intervention_eval_manifest: str | Path,
    hq_eval_manifest: str | Path,
    realsource_canonical_manifest: str | Path,
    realsource_adapter: str | Path,
    realsource_cache_dir: str | Path,
    intervention_dataset_root: str | Path,
    hq_dataset_root: str | Path,
    realsource_parent_view: str | Path | None = None,
    intervention_parent_view: str | Path | None = None,
    hq_parent_view: str | Path | None = None,
    allow_gcs_download: bool = False,
    gcs_download_timeout_seconds: int = 900,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output}")
    staging = output.with_name(
        f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    )
    if staging.exists():  # pragma: no cover - UUID collision defense.
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    views = {
        "realsource": _input_file(
            realsource_view, label="RealSource smoke view"
        ),
        "intervention": _input_file(
            intervention_view, label="intervention smoke view"
        ),
        "hq": _input_file(hq_view, label="HQ smoke view"),
    }
    evals = {
        "realsource": _input_file(
            realsource_eval_manifest, label="RealSource eval manifest"
        ),
        "intervention": _input_file(
            intervention_eval_manifest,
            label="intervention eval manifest",
        ),
        "hq": _input_file(hq_eval_manifest, label="HQ eval manifest"),
    }
    parents = {
        "realsource": (
            None
            if realsource_parent_view is None
            else _input_file(
                realsource_parent_view, label="RealSource parent view"
            )
        ),
        "intervention": (
            None
            if intervention_parent_view is None
            else _input_file(
                intervention_parent_view,
                label="intervention parent view",
            )
        ),
        "hq": (
            None
            if hq_parent_view is None
            else _input_file(hq_parent_view, label="HQ parent view")
        ),
    }
    canonical_manifest = _input_file(
        realsource_canonical_manifest,
        label="RealSource canonical manifest",
    )
    adapter = _input_file(
        realsource_adapter, label="RealSource canonical adapter"
    )
    cache = _input_dir(
        realsource_cache_dir, label="RealSource canonical cache"
    )
    intervention_root = _input_dir(
        intervention_dataset_root, label="intervention dataset root"
    )
    hq_root = _input_dir(hq_dataset_root, label="HQ dataset root")

    try:
        candidate_paths = {
            source_id: staging / f"{source_id}_statistics_candidate.json"
            for source_id in SOURCE_ORDER
        }
        _build_realsource_candidate(
            smoke_path=views["realsource"],
            parent_override=parents["realsource"],
            evaluation_manifest=evals["realsource"],
            canonical_manifest=canonical_manifest,
            adapter_path=adapter,
            cache_dir=cache,
            output_path=candidate_paths["realsource"],
        )
        _build_lerobot_candidate(
            source_id="intervention",
            smoke_path=views["intervention"],
            parent_override=parents["intervention"],
            evaluation_manifest=evals["intervention"],
            dataset_root=intervention_root,
            output_path=candidate_paths["intervention"],
        )
        _build_lerobot_candidate(
            source_id="hq",
            smoke_path=views["hq"],
            parent_override=parents["hq"],
            evaluation_manifest=evals["hq"],
            dataset_root=hq_root,
            output_path=candidate_paths["hq"],
        )

        holdout_path = staging / "holdout.json"
        population_path = staging / "population.json"
        union_contract.build_union_contract(
            holdout_output=holdout_path,
            population_output=population_path,
            realsource_eval_manifest=evals["realsource"],
            realsource_candidate_view=candidate_paths["realsource"],
            realsource_cache_dir=cache,
            intervention_eval_manifest=evals["intervention"],
            intervention_candidate_view=candidate_paths["intervention"],
            intervention_dataset_root=intervention_root,
            hq_eval_manifest=evals["hq"],
            hq_candidate_view=candidate_paths["hq"],
            hq_dataset_root=hq_root,
            allow_gcs_download=bool(allow_gcs_download),
            gcs_download_timeout_seconds=int(
                gcs_download_timeout_seconds
            ),
        )
        loaded_smoke = {
            source_id: dataset_view.load_frozen_view(path)
            for source_id, path in views.items()
        }
        population_sha256 = _mark_population_handoff_only(
            population_path, smoke_views=loaded_smoke
        )
        statistics_path = staging / "openpi_q01q99_unclipped.json"
        statistics_ledger_path = staging / "statistics_ledger.json"
        statistics_sha256, ledger_sha256 = (
            union_stats.write_union_statistics(
                manifest_path=population_path,
                output_path=statistics_path,
                ledger_path=statistics_ledger_path,
            )
        )
        statistics = json.loads(
            statistics_path.read_text(encoding="utf-8")
        )
        if (
            statistics.get("population", {}).get("unique_base_frames")
            != 128 * len(SOURCE_ORDER)
        ):
            raise ValueError(
                "Smoke statistics must contain exactly 384 unique training "
                "base frames."
            )
        if statistics["population"]["source_order"] != list(SOURCE_ORDER):
            raise ValueError("Smoke statistics source order changed.")

        holdout_sha256 = _sha256(holdout_path)
        candidate_hashes = {
            source_id: _sha256(path)
            for source_id, path in candidate_paths.items()
        }
        result = {
            "schema": HANDOFF_STATISTICS_SCHEMA,
            "scope": HANDOFF_SCOPE,
            "model_quality_claim_allowed": False,
            "source_order": list(SOURCE_ORDER),
            "exact_training_logical_rows_per_source": 128,
            "output_dir": str(output),
            "statistics": {
                "path": str(output / statistics_path.name),
                "sha256": statistics_sha256,
                "ledger_path": str(
                    output / statistics_ledger_path.name
                ),
                "ledger_sha256": ledger_sha256,
            },
            "holdout": {
                "path": str(output / holdout_path.name),
                "sha256": holdout_sha256,
            },
            "population": {
                "path": str(output / population_path.name),
                "sha256": population_sha256,
            },
            "candidate_views": {
                source_id: {
                    "path": str(output / path.name),
                    "sha256": candidate_hashes[source_id],
                }
                for source_id, path in candidate_paths.items()
            },
            "parent_smoke_view_sha256": {
                source_id: loaded_smoke[source_id].manifest_sha256
                for source_id in SOURCE_ORDER
            },
        }
        result_path = staging / "materialization.json"
        result_path.write_bytes(deterministic_json_bytes(result) + b"\n")
        _atomic_publish(staging, output, overwrite=overwrite)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--realsource-view", required=True)
    parser.add_argument("--intervention-view", required=True)
    parser.add_argument("--hq-view", required=True)
    parser.add_argument("--realsource-eval-manifest", required=True)
    parser.add_argument("--intervention-eval-manifest", required=True)
    parser.add_argument("--hq-eval-manifest", required=True)
    parser.add_argument("--realsource-canonical-manifest", required=True)
    parser.add_argument("--realsource-adapter", required=True)
    parser.add_argument("--realsource-cache-dir", required=True)
    parser.add_argument("--intervention-dataset-root", required=True)
    parser.add_argument("--hq-dataset-root", required=True)
    parser.add_argument("--realsource-parent-view")
    parser.add_argument("--intervention-parent-view")
    parser.add_argument("--hq-parent-view")
    parser.add_argument(
        "--allow-gcs-download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--gcs-download-timeout-seconds", type=int, default=900
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize(
        output_dir=args.output_dir,
        realsource_view=args.realsource_view,
        intervention_view=args.intervention_view,
        hq_view=args.hq_view,
        realsource_eval_manifest=args.realsource_eval_manifest,
        intervention_eval_manifest=args.intervention_eval_manifest,
        hq_eval_manifest=args.hq_eval_manifest,
        realsource_canonical_manifest=args.realsource_canonical_manifest,
        realsource_adapter=args.realsource_adapter,
        realsource_cache_dir=args.realsource_cache_dir,
        intervention_dataset_root=args.intervention_dataset_root,
        hq_dataset_root=args.hq_dataset_root,
        realsource_parent_view=args.realsource_parent_view,
        intervention_parent_view=args.intervention_parent_view,
        hq_parent_view=args.hq_parent_view,
        allow_gcs_download=args.allow_gcs_download,
        gcs_download_timeout_seconds=(
            args.gcs_download_timeout_seconds
        ),
        overwrite=args.overwrite,
    )
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
