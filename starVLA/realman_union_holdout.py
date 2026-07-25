"""Authenticated global holdout contract for RealMan union statistics.

The shared OpenPI q01/q99 table is computed over three independently frozen
sources.  This module derives the exact population episode keys that must be
excluded from each source's evaluation manifest and statistics-candidate
view.  The resulting contract is both human-readable and mechanically
re-derivable; a copied digest cannot silently bind a different exclusion list.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import uuid

from starVLA.action_representation import REALMAN_18D_ACTION_CONTRACT


_DATASET_VIEW_PATH = Path(__file__).resolve().parent / "dataloader/dataset_view.py"
_DATASET_VIEW_SPEC = importlib.util.spec_from_file_location(
    "_realman_union_holdout_dataset_view",
    _DATASET_VIEW_PATH,
)
if _DATASET_VIEW_SPEC is None or _DATASET_VIEW_SPEC.loader is None:
    raise ImportError(
        f"Could not load dataset-view helpers from {_DATASET_VIEW_PATH}"
    )
dataset_view = importlib.util.module_from_spec(_DATASET_VIEW_SPEC)
sys.modules[_DATASET_VIEW_SPEC.name] = dataset_view
_DATASET_VIEW_SPEC.loader.exec_module(dataset_view)


REALMAN_UNION_HOLDOUT_SCHEMA = "openpi-realman-union-holdout-v1"
CANONICAL_SOURCE_KIND = "canonical"
LEROBOT_SOURCE_KIND = "lerobot"
SUPPORTED_SOURCE_KINDS = frozenset(
    {CANONICAL_SOURCE_KIND, LEROBOT_SOURCE_KIND}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _resolve_bound_file(
    value: Any,
    *,
    base_dir: Path,
    label: str,
) -> Path:
    if isinstance(value, Path):
        raw = str(value)
    else:
        raw = _require_nonempty(value, label=label)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _file_binding(
    path: Path,
    *,
    manifest_dir: Path,
) -> str:
    """Prefer a portable relative path when both files share a tree."""

    try:
        return str(path.relative_to(manifest_dir))
    except ValueError:
        return str(path)


def _candidate_view(
    path: Path,
    *,
    expected_sha256: str,
) -> dataset_view.FrozenDatasetView:
    actual_sha256 = dataset_view.file_sha256(path)
    if actual_sha256 != _require_sha256(
        expected_sha256, label="statistics candidate view SHA-256"
    ):
        raise ValueError(
            "Statistics candidate view SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    view = dataset_view.load_frozen_view(
        path,
        expected_representation_contract_sha256=(
            REALMAN_18D_ACTION_CONTRACT.sha256()
        ),
        verify_ledger=True,
    )
    if (
        view.descriptor.get("purpose")
        != dataset_view.STATISTICS_POPULATION_CANDIDATE_PURPOSE
    ):
        raise ValueError(
            "Global holdout sources must use a "
            "statistics_population_candidate view."
        )
    usage = view.descriptor.get("usage_contract")
    if (
        not isinstance(usage, Mapping)
        or usage.get("training_allowed") is not False
        or usage.get("statistics_accumulation")
        != "union_builder_must_exclude_authenticated_holdout_keys"
    ):
        raise ValueError(
            "Statistics candidate view has an invalid usage contract."
        )
    return view


def _canonical_eval_identities(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, str, str, str, int], ...]:
    if (
        payload.get("schema_version") != 1
        or payload.get("purpose") != "heldout"
    ):
        raise ValueError(
            "Canonical global holdout source requires a schema_version=1 "
            "manifest with purpose='heldout'."
        )
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Canonical evaluation manifest has no windows.")
    identities: set[tuple[str, str, str, str, int]] = set()
    window_identities: set[tuple[str, str, str, str, int, int]] = set()
    for ordinal, raw in enumerate(windows):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Canonical evaluation window {ordinal} must be an object."
            )
        strings = tuple(
            raw.get(field)
            for field in ("dataset_id", "sid", "revision", "data_file")
        )
        episode_index = raw.get("episode_index")
        base_index = raw.get("base_index")
        if (
            any(not isinstance(value, str) or not value for value in strings)
            or isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
            or isinstance(base_index, bool)
            or not isinstance(base_index, int)
            or base_index < 0
        ):
            raise ValueError(
                f"Canonical evaluation window {ordinal} has an invalid identity."
            )
        identity = (*strings, int(episode_index))
        window_identity = (*identity, int(base_index))
        if window_identity in window_identities:
            raise ValueError("Canonical evaluation manifest repeats a window.")
        window_identities.add(window_identity)
        identities.add(identity)
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Canonical evaluation manifest lacks selection.")
    if selection.get("window_count") != len(window_identities):
        raise ValueError(
            "Canonical evaluation window_count does not match its windows."
        )
    if selection.get("holdout_episode_count") != len(identities):
        raise ValueError(
            "Canonical evaluation holdout_episode_count does not match its "
            "episode identities."
        )
    return tuple(sorted(identities))


def _derive_canonical_source(
    *,
    source_id: str,
    evaluation_manifest: Path,
    evaluation_manifest_sha256: str,
    candidate_view_path: Path,
    candidate_view_sha256: str,
    manifest_dir: Path,
) -> dict[str, Any]:
    view = _candidate_view(
        candidate_view_path,
        expected_sha256=candidate_view_sha256,
    )
    sources = view.descriptor["sources"]
    if not sources or any(
        source.get("backend") != CANONICAL_SOURCE_KIND for source in sources
    ):
        raise ValueError(
            "Canonical global holdout source view must contain only canonical "
            "sources."
        )
    catalog_hashes = {source.get("catalog_sha256") for source in sources}
    source_manifest_hashes = {
        source.get("manifest_sha256") for source in sources
    }
    if len(catalog_hashes) != 1 or len(source_manifest_hashes) != 1:
        raise ValueError(
            "Canonical statistics view must bind one shared source catalog."
        )
    catalog_sha256 = _require_sha256(
        next(iter(catalog_hashes)), label=f"{source_id} catalog SHA-256"
    )
    source_manifest_sha256 = _require_sha256(
        next(iter(source_manifest_hashes)),
        label=f"{source_id} source manifest SHA-256",
    )
    actual_eval_sha256 = dataset_view.file_sha256(evaluation_manifest)
    if actual_eval_sha256 != _require_sha256(
        evaluation_manifest_sha256,
        label=f"{source_id} evaluation manifest SHA-256",
    ):
        raise ValueError(
            f"{source_id} evaluation manifest SHA-256 mismatch."
        )
    evaluation = _load_json(
        evaluation_manifest, label=f"{source_id} evaluation manifest"
    )
    if evaluation.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError(
            f"{source_id} evaluation manifest binds a different canonical "
            "source catalog."
        )
    identities = _canonical_eval_identities(evaluation)
    identity_digest = dataset_view.canonical_json_sha256(
        [list(identity) for identity in identities]
    )
    binding = view.descriptor.get("selection", {}).get(
        "evaluation_holdout"
    )
    if not isinstance(binding, Mapping):
        raise ValueError(
            f"{source_id} statistics view lacks its evaluation binding."
        )
    expected_binding = {
        "schema": "realsource-canonical-eval-holdout-binding-v1",
        "manifest_sha256": actual_eval_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "window_count": len(evaluation["windows"]),
        "episode_count": len(identities),
        "episode_identities_sha256": identity_digest,
    }
    mismatches = {
        field: {"view": binding.get(field), "expected": expected}
        for field, expected in expected_binding.items()
        if binding.get(field) != expected
    }
    if mismatches:
        raise ValueError(
            f"{source_id} canonical view/eval binding mismatch: {mismatches}."
        )
    candidate_identities = {
        (
            row.get("dataset_id"),
            row.get("sid"),
            row.get("revision"),
            row.get("data_file"),
            row.get("episode_index"),
        )
        for row in view.iter_rows()
    }
    missing = sorted(set(identities) - candidate_identities)
    if missing:
        raise ValueError(
            f"{source_id} statistics view omits heldout episodes: {missing[:5]}."
        )
    episode_keys = sorted(
        f"{source_id}/{dataset_id}@{revision}:{episode_index}"
        for dataset_id, _sid, revision, _data_file, episode_index in identities
    )
    return {
        "id": source_id,
        "kind": CANONICAL_SOURCE_KIND,
        "catalog_sha256": catalog_sha256,
        "evaluation_manifest": _file_binding(
            evaluation_manifest, manifest_dir=manifest_dir
        ),
        "evaluation_manifest_sha256": actual_eval_sha256,
        "statistics_population_candidate_view": _file_binding(
            candidate_view_path, manifest_dir=manifest_dir
        ),
        "statistics_population_candidate_view_sha256": view.manifest_sha256,
        "statistics_population_candidate_view_id": view.view_id,
        "episode_keys": episode_keys,
    }


def _derive_lerobot_source(
    *,
    source_id: str,
    evaluation_manifest: Path,
    evaluation_manifest_sha256: str,
    candidate_view_path: Path,
    candidate_view_sha256: str,
    manifest_dir: Path,
) -> dict[str, Any]:
    view = _candidate_view(
        candidate_view_path,
        expected_sha256=candidate_view_sha256,
    )
    view_sources = view.descriptor["sources"]
    if len(view_sources) != 1 or view_sources[0].get("backend") != LEROBOT_SOURCE_KIND:
        raise ValueError(
            f"{source_id} global holdout source requires one LeRobot view source."
        )
    view_source = view_sources[0]
    catalog_sha256 = _require_sha256(
        view_source.get("catalog_sha256"),
        label=f"{source_id} catalog SHA-256",
    )
    dataset_name = _require_nonempty(
        view_source.get("dataset_name", view_source.get("source_id")),
        label=f"{source_id} dataset name",
    )
    actual_eval_sha256 = dataset_view.file_sha256(evaluation_manifest)
    if actual_eval_sha256 != _require_sha256(
        evaluation_manifest_sha256,
        label=f"{source_id} evaluation manifest SHA-256",
    ):
        raise ValueError(
            f"{source_id} evaluation manifest SHA-256 mismatch."
        )
    evaluation = _load_json(
        evaluation_manifest, label=f"{source_id} evaluation manifest"
    )
    if evaluation.get("schema_version") != 1:
        raise ValueError(
            f"{source_id} LeRobot evaluation manifest schema is invalid."
        )
    datasets = evaluation.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError(
            f"{source_id} evaluation manifest must bind exactly one dataset."
        )
    entry = datasets[0]
    if not isinstance(entry, Mapping):
        raise ValueError(f"{source_id} evaluation dataset entry is invalid.")
    if (
        entry.get("dataset_name") != dataset_name
        or entry.get("full_catalog_sha256") != catalog_sha256
    ):
        raise ValueError(
            f"{source_id} evaluation manifest binds a different dataset/catalog."
        )
    raw_indices = entry.get("holdout_episode_indices")
    if (
        not isinstance(raw_indices, list)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in raw_indices
        )
        or raw_indices != sorted(set(raw_indices))
        or entry.get("holdout_episode_count") != len(raw_indices)
        or not raw_indices
    ):
        raise ValueError(
            f"{source_id} evaluation holdout episode indices are invalid."
        )
    binding = view.descriptor.get("selection", {}).get(
        "evaluation_holdout"
    )
    if not isinstance(binding, Mapping):
        raise ValueError(
            f"{source_id} statistics view lacks its evaluation binding."
        )
    expected_binding = {
        "schema": "lerobot-eval-holdout-binding-v1",
        "manifest_sha256": actual_eval_sha256,
        "dataset_name": dataset_name,
        "full_catalog_sha256": catalog_sha256,
        "holdout_episode_count": len(raw_indices),
        "split_id": evaluation.get("split_id"),
    }
    mismatches = {
        field: {"view": binding.get(field), "expected": expected}
        for field, expected in expected_binding.items()
        if binding.get(field) != expected
    }
    if mismatches:
        raise ValueError(
            f"{source_id} LeRobot view/eval binding mismatch: {mismatches}."
        )
    holdout_exclusions = view.descriptor.get("holdout_exclusions")
    if (
        not isinstance(holdout_exclusions, Mapping)
        or holdout_exclusions.get("episode_indices") != raw_indices
    ):
        raise ValueError(
            f"{source_id} statistics view holdout exclusions do not match "
            "the evaluation manifest."
        )
    authenticated = view.descriptor.get("selection", {}).get(
        "authenticated_holdout_episode_indices_in_ledger"
    )
    if authenticated != raw_indices:
        raise ValueError(
            f"{source_id} statistics view does not authenticate the exact "
            "holdout episodes in its ledger."
        )
    candidate_indices = {
        row.get("episode_index") for row in view.iter_rows()
    }
    missing = sorted(set(raw_indices) - candidate_indices)
    if missing:
        raise ValueError(
            f"{source_id} statistics view omits heldout episodes: {missing[:5]}."
        )
    revision = str(view_source.get("revision", ""))
    dataset_id = str(view_source.get("dataset_id", view_source["source_id"]))
    episode_keys = sorted(
        (
            f"{source_id}/{dataset_id}@{revision}:{index}"
            if revision
            else f"{source_id}/{dataset_id}:{index}"
        )
        for index in raw_indices
    )
    return {
        "id": source_id,
        "kind": LEROBOT_SOURCE_KIND,
        "catalog_sha256": catalog_sha256,
        "evaluation_manifest": _file_binding(
            evaluation_manifest, manifest_dir=manifest_dir
        ),
        "evaluation_manifest_sha256": actual_eval_sha256,
        "statistics_population_candidate_view": _file_binding(
            candidate_view_path, manifest_dir=manifest_dir
        ),
        "statistics_population_candidate_view_sha256": view.manifest_sha256,
        "statistics_population_candidate_view_id": view.view_id,
        "episode_keys": episode_keys,
    }


def derive_source_holdout(
    *,
    source_id: str,
    kind: str,
    evaluation_manifest: str | Path,
    evaluation_manifest_sha256: str,
    candidate_view_manifest: str | Path,
    candidate_view_manifest_sha256: str,
    manifest_dir: str | Path,
) -> dict[str, Any]:
    source_id = _require_nonempty(source_id, label="global holdout source id")
    if kind not in SUPPORTED_SOURCE_KINDS:
        raise ValueError(
            f"Unsupported global holdout source kind {kind!r}; expected "
            f"{sorted(SUPPORTED_SOURCE_KINDS)}."
        )
    base = Path(manifest_dir).expanduser().resolve()
    evaluation_path = _resolve_bound_file(
        evaluation_manifest,
        base_dir=base,
        label=f"{source_id} evaluation manifest",
    )
    candidate_path = _resolve_bound_file(
        candidate_view_manifest,
        base_dir=base,
        label=f"{source_id} statistics candidate view",
    )
    common = {
        "source_id": source_id,
        "evaluation_manifest": evaluation_path,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "candidate_view_path": candidate_path,
        "candidate_view_sha256": candidate_view_manifest_sha256,
        "manifest_dir": base,
    }
    if kind == CANONICAL_SOURCE_KIND:
        return _derive_canonical_source(**common)
    return _derive_lerobot_source(**common)


def build_global_holdout_payload(
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not sources:
        raise ValueError("Global holdout requires at least one source.")
    normalized = [dict(source) for source in sources]
    source_order = [source.get("id") for source in normalized]
    if (
        any(not isinstance(value, str) or not value for value in source_order)
        or source_order != list(dict.fromkeys(source_order))
    ):
        raise ValueError(
            "Global holdout source IDs must be ordered unique strings."
        )
    all_keys: list[str] = []
    for source in normalized:
        source_id = source["id"]
        source_keys = source.get("episode_keys")
        if (
            not isinstance(source_keys, list)
            or source_keys != sorted(set(source_keys))
            or any(
                not isinstance(key, str)
                or not key.startswith(f"{source_id}/")
                for key in source_keys
            )
        ):
            raise ValueError(
                f"Global holdout source {source_id!r} episode keys must be a "
                "sorted, unique list using that source prefix."
            )
        all_keys.extend(source_keys)
    all_keys.sort()
    if (
        not all_keys
        or len(all_keys) != len(set(all_keys))
        or any("/" not in key for key in all_keys)
    ):
        raise ValueError(
            "Global holdout episode keys must be non-empty and globally unique."
        )
    return {
        "schema": REALMAN_UNION_HOLDOUT_SCHEMA,
        "contract_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
        "source_order": source_order,
        "sources": normalized,
        "episode_keys": all_keys,
    }


def write_global_holdout_manifest(
    output_path: str | Path,
    sources: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> tuple[Path, str]:
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Global holdout manifest already exists: {output}"
        )
    payload = build_global_holdout_payload(sources)
    serialized = dataset_view.canonical_json_bytes(payload) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_bytes(serialized)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output, hashlib.sha256(serialized).hexdigest()


def validate_global_holdout_manifest(
    manifest_path: str | Path,
    *,
    expected_episode_keys: Sequence[str],
    population_sources: Sequence[Mapping[str, Any]],
    population_manifest_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Re-derive every source key and compare it with the population contract."""

    path = Path(manifest_path).expanduser().resolve()
    population_base = (
        path.parent
        if population_manifest_dir is None
        else Path(population_manifest_dir).expanduser().resolve()
    )
    raw = path.read_bytes()
    payload = _load_json(path, label="global holdout manifest")
    if raw != dataset_view.canonical_json_bytes(payload) + b"\n":
        raise ValueError(
            "Global holdout manifest must use canonical JSON encoding."
        )
    required = {
        "schema",
        "contract_sha256",
        "source_order",
        "sources",
        "episode_keys",
    }
    if set(payload) != required:
        raise ValueError(
            f"Global holdout manifest must contain exactly {sorted(required)}."
        )
    if payload["schema"] != REALMAN_UNION_HOLDOUT_SCHEMA:
        raise ValueError("Global holdout manifest schema is invalid.")
    if payload["contract_sha256"] != REALMAN_18D_ACTION_CONTRACT.sha256():
        raise ValueError(
            "Global holdout representation contract does not match RealMan 18-D."
        )
    source_order = payload["source_order"]
    raw_sources = payload["sources"]
    if (
        not isinstance(source_order, list)
        or any(not isinstance(value, str) or not value for value in source_order)
        or len(source_order) != len(set(source_order))
        or not isinstance(raw_sources, list)
        or len(raw_sources) != len(source_order)
    ):
        raise ValueError("Global holdout source ordering is invalid.")
    by_id = {
        source.get("id"): source
        for source in raw_sources
        if isinstance(source, Mapping)
    }
    if list(by_id) != source_order or len(by_id) != len(raw_sources):
        raise ValueError(
            "Global holdout sources do not exactly match source_order."
        )
    population_by_id = {
        source.get("id"): source
        for source in population_sources
        if isinstance(source, Mapping)
    }
    population_order = [
        source.get("id")
        for source in population_sources
        if isinstance(source, Mapping)
    ]
    if (
        population_order != source_order
        or len(population_by_id) != len(population_sources)
    ):
        raise ValueError(
            "Global holdout source order does not exactly match the union "
            "population source order."
        )
    derived_sources: list[dict[str, Any]] = []
    for source_id in source_order:
        bound = by_id[source_id]
        required_source = {
            "id",
            "kind",
            "catalog_sha256",
            "evaluation_manifest",
            "evaluation_manifest_sha256",
            "statistics_population_candidate_view",
            "statistics_population_candidate_view_sha256",
            "statistics_population_candidate_view_id",
            "episode_keys",
        }
        if set(bound) != required_source:
            raise ValueError(
                f"Global holdout source {source_id!r} must contain exactly "
                f"{sorted(required_source)}."
            )
        population_source = population_by_id[source_id]
        reader = population_source.get("reader")
        if (
            not isinstance(reader, Mapping)
            or reader.get("kind") != "frozen_parquet_view"
        ):
            raise ValueError(
                "Production global holdout requires frozen_parquet_view "
                f"population source {source_id!r}."
            )
        if population_source.get("catalog_sha256") != bound["catalog_sha256"]:
            raise ValueError(
                f"Global holdout/population catalog mismatch for {source_id}."
            )
        if (
            reader.get("view_manifest_sha256")
            != bound["statistics_population_candidate_view_sha256"]
        ):
            raise ValueError(
                f"Global holdout/population candidate SHA mismatch for {source_id}."
            )
        population_view = _resolve_bound_file(
            reader.get("view_manifest"),
            base_dir=population_base,
            label=f"{source_id} population candidate view",
        )
        bound_view = _resolve_bound_file(
            bound["statistics_population_candidate_view"],
            base_dir=path.parent,
            label=f"{source_id} holdout candidate view",
        )
        if population_view != bound_view:
            raise ValueError(
                f"Global holdout/population candidate path mismatch for {source_id}."
            )
        derived = derive_source_holdout(
            source_id=source_id,
            kind=bound["kind"],
            evaluation_manifest=bound["evaluation_manifest"],
            evaluation_manifest_sha256=bound[
                "evaluation_manifest_sha256"
            ],
            candidate_view_manifest=bound_view,
            candidate_view_manifest_sha256=bound[
                "statistics_population_candidate_view_sha256"
            ],
            manifest_dir=path.parent,
        )
        if derived != dict(bound):
            raise ValueError(
                f"Global holdout source {source_id!r} is stale or mismatched."
            )
        derived_sources.append(derived)
    derived_payload = build_global_holdout_payload(derived_sources)
    if derived_payload != dict(payload):
        raise ValueError(
            "Global holdout manifest is stale or does not match its bound eval "
            "manifests/views."
        )
    expected = list(expected_episode_keys)
    if expected != sorted(set(expected)):
        raise ValueError(
            "Union population holdout episode keys must be sorted and unique."
        )
    if payload["episode_keys"] != expected:
        raise ValueError(
            "Global holdout episode keys do not exactly match the union "
            "population exclusion list."
        )
    return dict(payload)
