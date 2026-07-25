"""Immutable, backend-neutral dataset views.

A dataset view is a small canonical JSON descriptor plus an ordered canonical
JSONL row ledger.  The descriptor commits to the source catalogs, annotations,
action representation, holdout identities, exhaustive-epoch policy, and the
exact ledger bytes.  It is intentionally dependency-light so launchers and
offline verification tools can validate a view without importing Torch.

The module does not load robot data.  Dataset-specific generators resolve raw
sources into the common ledger row schema and then call :func:`write_frozen_view`.
Training loaders can consume the same ledger through :func:`load_frozen_view`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence
import uuid


VIEW_SCHEMA = "vla-dataset-view-v1"
ROW_SCHEMA = "vla-dataset-view-row-v1"
RANGE_ROW_SCHEMA = "vla-dataset-view-range-row-v1"
EXPANDED_ROWS_ENCODING = "expanded_rows_v1"
EPISODE_RANGES_ENCODING = "episode_ranges_v1"
HOLDOUT_SCHEMA = "vla-dataset-view-holdout-exclusions-v1"
LINEAGE_ID_SCHEMA = "vla-episode-lineage-id-v1"
CONTENT_ID_SCHEMA = "vla-episode-content-id-v1"
SAMPLE_ID_SCHEMA = "vla-sample-id-v1"
RANGE_ID_SCHEMA = "vla-sample-range-id-v1"
ALL_EXHAUSTIVE_MODE = "all_exhaustive"
STATISTICS_POPULATION_CANDIDATE_PURPOSE = (
    "statistics_population_candidate"
)
EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE = (
    "eval_selection_population_candidate"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_SAMPLING_KEYS = frozenset(
    {
        "weight",
        "weights",
        "sampling_ratio",
        "sampling_weight",
        "with_replacement",
        "max_windows",
    }
)


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the one permitted JSON encoding for immutable view artifacts."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest.")
    return value


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _require_int(value: Any, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}.")
    return value


def _stable_id(schema: str, payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256({"schema": schema, **dict(payload)})


def make_episode_lineage_id(
    *,
    backend: str,
    source_id: str,
    catalog_sha256: str,
    episode_index: int,
    length: int,
    episode_metadata_sha256: str | None = None,
) -> str:
    """Build a provenance-bound episode identity.

    A lineage ID deliberately changes when the source catalog changes.  A
    content ID, below, remains useful for recognizing the same episode copied
    into a differently named or re-indexed dataset.
    """

    payload: dict[str, Any] = {
        "backend": _require_nonempty_string(backend, context="backend"),
        "source_id": _require_nonempty_string(source_id, context="source_id"),
        "catalog_sha256": _require_sha256(
            catalog_sha256, context="catalog_sha256"
        ),
        "episode_index": _require_int(
            episode_index, context="episode_index", minimum=0
        ),
        "length": _require_int(length, context="length", minimum=1),
    }
    if episode_metadata_sha256 is not None:
        payload["episode_metadata_sha256"] = _require_sha256(
            episode_metadata_sha256,
            context="episode_metadata_sha256",
        )
    return _stable_id(LINEAGE_ID_SCHEMA, payload)


def make_episode_content_id(
    *,
    frame_content_sha256: str,
    length: int,
    content_contract: str,
) -> str:
    """Build a location-independent identity for canonical episode content."""

    return _stable_id(
        CONTENT_ID_SCHEMA,
        {
            "frame_content_sha256": _require_sha256(
                frame_content_sha256, context="frame_content_sha256"
            ),
            "length": _require_int(length, context="length", minimum=1),
            "content_contract": _require_nonempty_string(
                content_contract, context="content_contract"
            ),
        },
    )


def make_sample_id(
    *,
    episode_content_id: str,
    base_index: int,
    horizon: int,
    target_fps: int,
    representation_contract_sha256: str,
    end_clamp: bool,
) -> str:
    """Build a cross-view sample identity.

    Selection-view names are intentionally absent.  The same H-step sample in
    an intervention view and a recovery view therefore receives the same ID,
    making intentional replay and accidental overlap measurable.
    """

    if not isinstance(end_clamp, bool):
        raise ValueError("end_clamp must be boolean.")
    return _stable_id(
        SAMPLE_ID_SCHEMA,
        {
            "episode_content_id": _require_sha256(
                episode_content_id, context="episode_content_id"
            ),
            "base_index": _require_int(
                base_index, context="base_index", minimum=0
            ),
            "horizon": _require_int(horizon, context="horizon", minimum=1),
            "target_fps": _require_int(
                target_fps, context="target_fps", minimum=1
            ),
            "representation_contract_sha256": _require_sha256(
                representation_contract_sha256,
                context="representation_contract_sha256",
            ),
            "end_clamp": end_clamp,
        },
    )


def make_range_id(
    *,
    episode_content_id: str,
    base_start: int,
    base_stop: int,
    base_step: int,
    horizon: int,
    target_fps: int,
    representation_contract_sha256: str,
    end_clamp: bool,
) -> str:
    """Build an immutable identity for a compact arithmetic sample range."""

    if not isinstance(end_clamp, bool):
        raise ValueError("end_clamp must be boolean.")
    start = _require_int(base_start, context="base_start", minimum=0)
    stop = _require_int(base_stop, context="base_stop", minimum=1)
    step = _require_int(base_step, context="base_step", minimum=1)
    if stop <= start:
        raise ValueError("base_stop must be greater than base_start.")
    return _stable_id(
        RANGE_ID_SCHEMA,
        {
            "episode_content_id": _require_sha256(
                episode_content_id, context="episode_content_id"
            ),
            "base_start": start,
            "base_stop": stop,
            "base_step": step,
            "horizon": _require_int(
                horizon, context="horizon", minimum=1
            ),
            "target_fps": _require_int(
                target_fps, context="target_fps", minimum=1
            ),
            "representation_contract_sha256": _require_sha256(
                representation_contract_sha256,
                context="representation_contract_sha256",
            ),
            "end_clamp": end_clamp,
        },
    )


def make_holdout_exclusions(
    *,
    episode_indices: Sequence[int],
    lineage_ids: Sequence[str],
    content_ids: Sequence[str],
    source_id: str,
) -> dict[str, Any]:
    """Create a self-hashed, dual-identity holdout exclusion contract."""

    indices = sorted(
        {
            _require_int(value, context="holdout episode index", minimum=0)
            for value in episode_indices
        }
    )
    lineages = sorted(
        {
            _require_sha256(value, context="holdout lineage ID")
            for value in lineage_ids
        }
    )
    contents = sorted(
        {
            _require_sha256(value, context="holdout content ID")
            for value in content_ids
        }
    )
    body = {
        "schema": HOLDOUT_SCHEMA,
        "source_id": _require_nonempty_string(source_id, context="source_id"),
        "episode_indices": indices,
        "lineage_ids": lineages,
        "content_ids": contents,
    }
    return {**body, "sha256": canonical_json_sha256(body)}


def descriptor_view_id(payload: Mapping[str, Any]) -> str:
    """Hash a descriptor without its self-referential ``view_id`` field."""

    stripped = deepcopy(dict(payload))
    stripped.pop("view_id", None)
    return canonical_json_sha256(stripped)


def _find_forbidden_sampling_keys(
    payload: Any,
    *,
    path: str = "$",
) -> list[str]:
    violations: list[str] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if key in _FORBIDDEN_SAMPLING_KEYS:
                violations.append(child_path)
            violations.extend(
                _find_forbidden_sampling_keys(value, path=child_path)
            )
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            violations.extend(
                _find_forbidden_sampling_keys(value, path=f"{path}[{index}]")
            )
    return violations


def _validate_holdout(payload: Any) -> tuple[set[str], set[str]]:
    if not isinstance(payload, Mapping):
        raise ValueError("holdout_exclusions must be an object.")
    if payload.get("schema") != HOLDOUT_SCHEMA:
        raise ValueError(
            f"holdout_exclusions.schema must be {HOLDOUT_SCHEMA!r}."
        )
    _require_nonempty_string(
        payload.get("source_id"), context="holdout_exclusions.source_id"
    )
    raw_indices = payload.get("episode_indices")
    raw_lineages = payload.get("lineage_ids")
    raw_contents = payload.get("content_ids")
    if not isinstance(raw_indices, list):
        raise ValueError("holdout_exclusions.episode_indices must be a list.")
    if not isinstance(raw_lineages, list):
        raise ValueError("holdout_exclusions.lineage_ids must be a list.")
    if not isinstance(raw_contents, list):
        raise ValueError("holdout_exclusions.content_ids must be a list.")
    indices = [
        _require_int(value, context="holdout episode index", minimum=0)
        for value in raw_indices
    ]
    lineages = [
        _require_sha256(value, context="holdout lineage ID")
        for value in raw_lineages
    ]
    contents = [
        _require_sha256(value, context="holdout content ID")
        for value in raw_contents
    ]
    if indices != sorted(set(indices)):
        raise ValueError(
            "holdout_exclusions.episode_indices must be sorted and unique."
        )
    if lineages != sorted(set(lineages)):
        raise ValueError(
            "holdout_exclusions.lineage_ids must be sorted and unique."
        )
    if contents != sorted(set(contents)):
        raise ValueError(
            "holdout_exclusions.content_ids must be sorted and unique."
        )
    expected_hash = _require_sha256(
        payload.get("sha256"), context="holdout_exclusions.sha256"
    )
    unhashed = dict(payload)
    unhashed.pop("sha256", None)
    if canonical_json_sha256(unhashed) != expected_hash:
        raise ValueError("holdout_exclusions SHA-256 mismatch.")
    return set(lineages), set(contents)


def _validate_ledger_holdout_membership(
    row: Mapping[str, Any],
    *,
    descriptor_purpose: Any,
    excluded_lineages: set[str],
    excluded_contents: set[str],
    context: str,
) -> None:
    """Enforce train-view exclusion or exact candidate holdout membership.

    Statistics population candidates are the sole view kind allowed to carry
    authenticated holdout references.  Even there, both the lineage and
    content identities must be members of the exclusion contract.  This
    prevents a non-heldout copy of heldout content from entering the
    population under a different lineage.
    """

    is_holdout_lineage = (
        row["episode_lineage_id"] in excluded_lineages
    )
    is_holdout_content = row["episode_content_id"] in excluded_contents
    if descriptor_purpose == STATISTICS_POPULATION_CANDIDATE_PURPOSE:
        if is_holdout_lineage != is_holdout_content:
            raise ValueError(
                f"{context} has inconsistent holdout lineage/content "
                "membership."
            )
        return
    if is_holdout_lineage:
        raise ValueError(f"{context} overlaps holdout lineage.")
    if is_holdout_content:
        raise ValueError(f"{context} overlaps holdout content.")


def _validate_source(source: Any, *, index: int) -> None:
    context = f"sources[{index}]"
    if not isinstance(source, Mapping):
        raise ValueError(f"{context} must be an object.")
    _require_nonempty_string(source.get("source_id"), context=f"{context}.source_id")
    backend = _require_nonempty_string(
        source.get("backend"), context=f"{context}.backend"
    )
    if backend not in {"lerobot", "canonical"}:
        raise ValueError(f"{context}.backend must be 'lerobot' or 'canonical'.")
    _require_sha256(
        source.get("catalog_sha256"), context=f"{context}.catalog_sha256"
    )
    _require_sha256(
        source.get("annotation_sha256"), context=f"{context}.annotation_sha256"
    )
    _require_sha256(
        source.get("source_content_sha256"),
        context=f"{context}.source_content_sha256",
    )
    for optional_hash in (
        "info_sha256",
        "manifest_sha256",
        "adapter_sha256",
    ):
        if optional_hash in source:
            _require_sha256(
                source[optional_hash], context=f"{context}.{optional_hash}"
            )
    selected_data_shards = source.get("selected_data_shards")
    if selected_data_shards is not None:
        if backend != "lerobot":
            raise ValueError(
                f"{context}.selected_data_shards is only valid for lerobot sources."
            )
        if not isinstance(selected_data_shards, list) or not selected_data_shards:
            raise ValueError(
                f"{context}.selected_data_shards must be a non-empty list."
            )
        normalized_paths: list[str] = []
        for shard_index, shard in enumerate(selected_data_shards):
            shard_context = (
                f"{context}.selected_data_shards[{shard_index}]"
            )
            if not isinstance(shard, Mapping):
                raise ValueError(f"{shard_context} must be an object.")
            relative_path = _require_nonempty_string(
                shard.get("path"), context=f"{shard_context}.path"
            )
            parsed_path = Path(relative_path)
            if (
                parsed_path.is_absolute()
                or parsed_path.as_posix() != relative_path
                or ".." in parsed_path.parts
                or "." in parsed_path.parts
            ):
                raise ValueError(
                    f"{shard_context}.path must be a normalized relative POSIX path."
                )
            _require_sha256(
                shard.get("sha256"), context=f"{shard_context}.sha256"
            )
            _require_int(
                shard.get("size_bytes"),
                context=f"{shard_context}.size_bytes",
                minimum=1,
            )
            normalized_paths.append(relative_path)
        if normalized_paths != sorted(set(normalized_paths)):
            raise ValueError(
                f"{context}.selected_data_shards paths must be sorted and unique."
            )


def _validate_descriptor(payload: Any, *, require_rows: bool) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Dataset-view descriptor must be a JSON object.")
    violations = _find_forbidden_sampling_keys(payload)
    if violations:
        raise ValueError(
            "Frozen views forbid implicit sampling controls: "
            + ", ".join(violations)
        )
    if payload.get("schema") != VIEW_SCHEMA:
        raise ValueError(f"schema must be {VIEW_SCHEMA!r}.")
    _require_nonempty_string(payload.get("view_name"), context="view_name")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list.")
    source_ids: list[str] = []
    for index, source in enumerate(sources):
        _validate_source(source, index=index)
        source_ids.append(str(source["source_id"]))
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("sources contain duplicate source_id values.")

    representation = payload.get("representation")
    if not isinstance(representation, Mapping):
        raise ValueError("representation must be an object.")
    _require_sha256(
        representation.get("contract_sha256"),
        context="representation.contract_sha256",
    )
    _require_int(
        representation.get("state_dim"),
        context="representation.state_dim",
        minimum=1,
    )
    _require_int(
        representation.get("action_dim"),
        context="representation.action_dim",
        minimum=1,
    )
    _require_int(
        representation.get("horizon"),
        context="representation.horizon",
        minimum=1,
    )
    _require_int(
        representation.get("target_fps"),
        context="representation.target_fps",
        minimum=1,
    )

    epoch = payload.get("epoch_contract")
    if not isinstance(epoch, Mapping):
        raise ValueError("epoch_contract must be an object.")
    if epoch.get("mode") != ALL_EXHAUSTIVE_MODE:
        raise ValueError(
            f"epoch_contract.mode must be {ALL_EXHAUSTIVE_MODE!r}."
        )
    _require_int(
        epoch.get("epoch_passes"),
        context="epoch_contract.epoch_passes",
        minimum=1,
    )
    if epoch.get("drop_last") is not False:
        raise ValueError("epoch_contract.drop_last must be false.")
    if epoch.get("replacement") is not False:
        raise ValueError("epoch_contract.replacement must be false.")

    _validate_holdout(payload.get("holdout_exclusions"))
    if not isinstance(payload.get("selection"), Mapping):
        raise ValueError("selection must be an object.")

    if not require_rows:
        return
    rows = payload.get("rows")
    if not isinstance(rows, Mapping):
        raise ValueError("rows must be an object.")
    # Authenticate the descriptor before reporting secondary cross-field
    # inconsistencies.  This preserves fail-closed mutation diagnostics for
    # legacy expanded manifests whose generated record_count equals row_count.
    view_id = _require_sha256(payload.get("view_id"), context="view_id")
    if descriptor_view_id(payload) != view_id:
        raise ValueError("Dataset-view descriptor view_id mismatch.")
    encoding = rows.get("encoding", EXPANDED_ROWS_ENCODING)
    if encoding not in {
        EXPANDED_ROWS_ENCODING,
        EPISODE_RANGES_ENCODING,
    }:
        raise ValueError(
            "rows.encoding must be 'expanded_rows_v1' or "
            "'episode_ranges_v1'."
        )
    ledger_path = _require_nonempty_string(
        rows.get("path"), context="rows.path"
    )
    if Path(ledger_path).is_absolute():
        raise ValueError("rows.path must be relative to the manifest.")
    _require_sha256(rows.get("sha256"), context="rows.sha256")
    row_count = _require_int(
        rows.get("row_count"), context="rows.row_count", minimum=1
    )
    record_count = rows.get("record_count", row_count)
    record_count = _require_int(
        record_count, context="rows.record_count", minimum=1
    )
    if encoding == EXPANDED_ROWS_ENCODING and record_count != row_count:
        raise ValueError(
            "Expanded dataset views require rows.record_count == "
            "rows.row_count."
        )
    if encoding == EPISODE_RANGES_ENCODING and record_count > row_count:
        raise ValueError(
            "Range-view rows.record_count cannot exceed logical row_count."
        )
    _require_int(
        rows.get("unique_sample_count"),
        context="rows.unique_sample_count",
        minimum=1,
    )
    _require_int(
        rows.get("episode_count"), context="rows.episode_count", minimum=1
    )


def _validate_row(
    row: Any,
    *,
    expected_ordinal: int,
    source_backends: Mapping[str, str],
    representation: Mapping[str, Any],
) -> None:
    context = f"ledger row {expected_ordinal}"
    if not isinstance(row, Mapping):
        raise ValueError(f"{context} must be a JSON object.")
    if row.get("schema") != ROW_SCHEMA:
        raise ValueError(f"{context}.schema must be {ROW_SCHEMA!r}.")
    if row.get("ordinal") != expected_ordinal:
        raise ValueError(
            f"{context} has ordinal {row.get('ordinal')!r}; "
            f"expected {expected_ordinal}."
        )
    backend = row.get("backend")
    if backend not in {"lerobot", "canonical"}:
        raise ValueError(f"{context}.backend is invalid.")
    source_id = _require_nonempty_string(
        row.get("source_id"), context=f"{context}.source_id"
    )
    if source_id not in source_backends:
        raise ValueError(f"{context} references unknown source {source_id!r}.")
    if backend != source_backends[source_id]:
        raise ValueError(
            f"{context}.backend={backend!r} does not match source "
            f"{source_id!r} backend={source_backends[source_id]!r}."
        )
    _require_int(
        row.get("episode_index"),
        context=f"{context}.episode_index",
        minimum=0,
    )
    _require_int(
        row.get("base_index"), context=f"{context}.base_index", minimum=0
    )
    horizon = _require_int(
        row.get("horizon"), context=f"{context}.horizon", minimum=1
    )
    if horizon != int(representation["horizon"]):
        raise ValueError(
            f"{context}.horizon={horizon} does not match representation "
            f"horizon={representation['horizon']}."
        )
    target_fps = _require_int(
        row.get("target_fps"), context=f"{context}.target_fps", minimum=1
    )
    if target_fps != int(representation["target_fps"]):
        raise ValueError(
            f"{context}.target_fps={target_fps} does not match representation "
            f"target_fps={representation['target_fps']}."
        )
    for key in ("episode_lineage_id", "episode_content_id", "sample_id"):
        _require_sha256(row.get(key), context=f"{context}.{key}")
    end_clamp_policy = row.get("end_clamp_policy")
    if end_clamp_policy not in {"repeat_last", "no_end_clamp"}:
        raise ValueError(
            f"{context}.end_clamp_policy must be 'repeat_last' or "
            "'no_end_clamp'."
        )
    expected_sample_id = make_sample_id(
        episode_content_id=str(row["episode_content_id"]),
        base_index=int(row["base_index"]),
        horizon=horizon,
        target_fps=target_fps,
        representation_contract_sha256=str(
            representation["contract_sha256"]
        ),
        end_clamp=end_clamp_policy == "repeat_last",
    )
    if row["sample_id"] != expected_sample_id:
        raise ValueError(f"{context}.sample_id does not match its row identity.")


def range_sample_count(
    *,
    base_start: int,
    base_stop: int,
    base_step: int,
) -> int:
    """Return the cardinality of ``range(start, stop, step)``."""

    start = _require_int(base_start, context="base_start", minimum=0)
    stop = _require_int(base_stop, context="base_stop", minimum=1)
    step = _require_int(base_step, context="base_step", minimum=1)
    if stop <= start:
        raise ValueError("base_stop must be greater than base_start.")
    return (stop - start + step - 1) // step


def _validate_range_row(
    row: Any,
    *,
    expected_ordinal: int,
    source_backends: Mapping[str, str],
    representation: Mapping[str, Any],
) -> None:
    context = f"range ledger row {expected_ordinal}"
    if not isinstance(row, Mapping):
        raise ValueError(f"{context} must be a JSON object.")
    if row.get("schema") != RANGE_ROW_SCHEMA:
        raise ValueError(
            f"{context}.schema must be {RANGE_ROW_SCHEMA!r}."
        )
    if row.get("ordinal") != expected_ordinal:
        raise ValueError(
            f"{context} has ordinal {row.get('ordinal')!r}; "
            f"expected {expected_ordinal}."
        )
    backend = row.get("backend")
    if backend not in {"lerobot", "canonical"}:
        raise ValueError(f"{context}.backend is invalid.")
    source_id = _require_nonempty_string(
        row.get("source_id"), context=f"{context}.source_id"
    )
    if source_id not in source_backends:
        raise ValueError(f"{context} references unknown source {source_id!r}.")
    if backend != source_backends[source_id]:
        raise ValueError(
            f"{context}.backend={backend!r} does not match source "
            f"{source_id!r} backend={source_backends[source_id]!r}."
        )
    _require_int(
        row.get("episode_index"),
        context=f"{context}.episode_index",
        minimum=0,
    )
    start = _require_int(
        row.get("base_start"),
        context=f"{context}.base_start",
        minimum=0,
    )
    stop = _require_int(
        row.get("base_stop"),
        context=f"{context}.base_stop",
        minimum=1,
    )
    step = _require_int(
        row.get("base_step"),
        context=f"{context}.base_step",
        minimum=1,
    )
    expected_count = range_sample_count(
        base_start=start,
        base_stop=stop,
        base_step=step,
    )
    count = _require_int(
        row.get("sample_count"),
        context=f"{context}.sample_count",
        minimum=1,
    )
    if count != expected_count:
        raise ValueError(
            f"{context}.sample_count={count} does not match "
            f"range({start}, {stop}, {step}) cardinality={expected_count}."
        )
    horizon = _require_int(
        row.get("horizon"), context=f"{context}.horizon", minimum=1
    )
    if horizon != int(representation["horizon"]):
        raise ValueError(
            f"{context}.horizon={horizon} does not match representation "
            f"horizon={representation['horizon']}."
        )
    target_fps = _require_int(
        row.get("target_fps"),
        context=f"{context}.target_fps",
        minimum=1,
    )
    if target_fps != int(representation["target_fps"]):
        raise ValueError(
            f"{context}.target_fps={target_fps} does not match "
            f"representation target_fps={representation['target_fps']}."
        )
    for key in (
        "episode_lineage_id",
        "episode_content_id",
        "range_id",
    ):
        _require_sha256(row.get(key), context=f"{context}.{key}")
    end_clamp_policy = row.get("end_clamp_policy")
    if end_clamp_policy not in {"repeat_last", "no_end_clamp"}:
        raise ValueError(
            f"{context}.end_clamp_policy must be 'repeat_last' or "
            "'no_end_clamp'."
        )
    expected_range_id = make_range_id(
        episode_content_id=str(row["episode_content_id"]),
        base_start=start,
        base_stop=stop,
        base_step=step,
        horizon=horizon,
        target_fps=target_fps,
        representation_contract_sha256=str(
            representation["contract_sha256"]
        ),
        end_clamp=end_clamp_policy == "repeat_last",
    )
    if row["range_id"] != expected_range_id:
        raise ValueError(
            f"{context}.range_id does not match its range identity."
        )


def _safe_relative_ledger_path(manifest_path: Path, raw_path: str) -> Path:
    manifest_parent = manifest_path.parent.resolve()
    resolved = (manifest_parent / raw_path).resolve()
    try:
        resolved.relative_to(manifest_parent)
    except ValueError as exc:
        raise ValueError(
            f"rows.path escapes the manifest directory: {raw_path!r}."
        ) from exc
    return resolved


@dataclass(frozen=True, slots=True)
class FrozenDatasetView:
    manifest_path: Path
    manifest_sha256: str
    ledger_path: Path
    descriptor: Mapping[str, Any]

    @property
    def view_id(self) -> str:
        return str(self.descriptor["view_id"])

    @property
    def row_count(self) -> int:
        return int(self.descriptor["rows"]["row_count"])

    @property
    def encoding(self) -> str:
        return str(
            self.descriptor["rows"].get(
                "encoding", EXPANDED_ROWS_ENCODING
            )
        )

    @property
    def record_count(self) -> int:
        return int(
            self.descriptor["rows"].get(
                "record_count", self.row_count
            )
        )

    @property
    def unique_sample_count(self) -> int:
        return int(self.descriptor["rows"]["unique_sample_count"])

    @property
    def episode_count(self) -> int:
        return int(self.descriptor["rows"]["episode_count"])

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        with self.ledger_path.open("rb") as handle:
            for raw_line in handle:
                yield json.loads(raw_line)


@dataclass(frozen=True, slots=True)
class FrozenDatasetViewBuild:
    manifest_path: Path
    ledger_path: Path
    manifest_sha256: str
    ledger_sha256: str
    view_id: str
    row_count: int
    unique_sample_count: int
    episode_count: int
    written: bool
    record_count: int | None = None
    encoding: str = EXPANDED_ROWS_ENCODING

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "ledger_path": str(self.ledger_path),
            "manifest_sha256": self.manifest_sha256,
            "ledger_sha256": self.ledger_sha256,
            "view_id": self.view_id,
            "row_count": self.row_count,
            "record_count": (
                self.row_count
                if self.record_count is None
                else self.record_count
            ),
            "encoding": self.encoding,
            "unique_sample_count": self.unique_sample_count,
            "episode_count": self.episode_count,
            "written": self.written,
        }


def load_frozen_view(
    manifest_path: Path | str,
    *,
    expected_view_id: str | None = None,
    expected_representation_contract_sha256: str | None = None,
    expected_source_hashes: Mapping[str, Mapping[str, str]] | None = None,
    verify_ledger: bool = True,
) -> FrozenDatasetView:
    """Load and fail-closed verify a frozen view."""

    manifest = Path(manifest_path).expanduser().resolve()
    raw_manifest = manifest.read_bytes()
    try:
        payload = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset-view JSON at {manifest}: {exc}") from exc
    if raw_manifest != canonical_json_bytes(payload) + b"\n":
        raise ValueError(
            f"Dataset-view descriptor is not canonical JSON: {manifest}."
        )
    _validate_descriptor(payload, require_rows=True)
    if expected_view_id is not None:
        expected = _require_sha256(expected_view_id, context="expected_view_id")
        if payload["view_id"] != expected:
            raise ValueError(
                f"Dataset-view ID mismatch: expected {expected}, "
                f"found {payload['view_id']}."
            )
    if expected_representation_contract_sha256 is not None:
        expected = _require_sha256(
            expected_representation_contract_sha256,
            context="expected_representation_contract_sha256",
        )
        found = payload["representation"]["contract_sha256"]
        if found != expected:
            raise ValueError(
                "Dataset-view representation contract mismatch: "
                f"expected {expected}, found {found}."
            )
    if expected_source_hashes is not None:
        found_sources = {
            str(source["source_id"]): source for source in payload["sources"]
        }
        for source_id, hashes in expected_source_hashes.items():
            if source_id not in found_sources:
                raise ValueError(
                    f"Dataset view is missing expected source {source_id!r}."
                )
            for key, expected_hash in hashes.items():
                expected = _require_sha256(
                    expected_hash,
                    context=f"expected_source_hashes[{source_id!r}][{key!r}]",
                )
                found = found_sources[source_id].get(key)
                if found != expected:
                    raise ValueError(
                        f"Dataset-view source {source_id!r} {key} mismatch: "
                        f"expected {expected}, found {found}."
                    )

    ledger_path = _safe_relative_ledger_path(
        manifest, str(payload["rows"]["path"])
    )
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Dataset-view ledger is missing: {ledger_path}")
    ledger_hash = file_sha256(ledger_path)
    if ledger_hash != payload["rows"]["sha256"]:
        raise ValueError(
            "Dataset-view ledger SHA-256 mismatch: "
            f"expected {payload['rows']['sha256']}, found {ledger_hash}."
        )

    if verify_ledger:
        source_backends = {
            str(source["source_id"]): str(source["backend"])
            for source in payload["sources"]
        }
        excluded_lineages, excluded_contents = _validate_holdout(
            payload["holdout_exclusions"]
        )
        encoding = payload["rows"].get(
            "encoding", EXPANDED_ROWS_ENCODING
        )
        sample_ids: set[str] = set()
        episodes: set[tuple[str, str]] = set()
        content_owners: dict[str, tuple[str, str]] = {}
        episode_last_bases: dict[tuple[str, str], int] = {}
        logical_row_count = 0
        record_count = 0
        with ledger_path.open("rb") as handle:
            for ordinal, raw_line in enumerate(handle):
                if not raw_line.endswith(b"\n"):
                    raise ValueError(
                        f"Dataset-view ledger row {ordinal} lacks a newline."
                    )
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid dataset-view ledger JSON at row {ordinal}: {exc}"
                    ) from exc
                if raw_line != canonical_json_bytes(row) + b"\n":
                    raise ValueError(
                        f"Dataset-view ledger row {ordinal} is not canonical JSON."
                    )
                if encoding == EXPANDED_ROWS_ENCODING:
                    _validate_row(
                        row,
                        expected_ordinal=ordinal,
                        source_backends=source_backends,
                        representation=payload["representation"],
                    )
                else:
                    _validate_range_row(
                        row,
                        expected_ordinal=ordinal,
                        source_backends=source_backends,
                        representation=payload["representation"],
                    )
                _validate_ledger_holdout_membership(
                    row,
                    descriptor_purpose=payload.get("purpose"),
                    excluded_lineages=excluded_lineages,
                    excluded_contents=excluded_contents,
                    context=f"Ledger row {ordinal}",
                )
                episode_identity = (
                    str(row["source_id"]),
                    str(row["episode_lineage_id"]),
                )
                episodes.add(episode_identity)
                if encoding == EXPANDED_ROWS_ENCODING:
                    sample_ids.add(str(row["sample_id"]))
                    logical_row_count += 1
                else:
                    content_id = str(row["episode_content_id"])
                    content_owner = content_owners.setdefault(
                        content_id, episode_identity
                    )
                    if content_owner != episode_identity:
                        raise ValueError(
                            "Range ledger contains the same episode content "
                            "under multiple episode identities; unique logical "
                            "sample cardinality would be ambiguous."
                        )
                    previous_last_base = episode_last_bases.get(
                        episode_identity
                    )
                    if (
                        previous_last_base is not None
                        and int(row["base_start"]) <= previous_last_base
                    ):
                        raise ValueError(
                            "Range ledger contains overlapping or reordered "
                            "logical base indices for one episode."
                        )
                    episode_last_bases[episode_identity] = int(
                        row["base_start"]
                    ) + (int(row["sample_count"]) - 1) * int(
                        row["base_step"]
                    )
                    logical_row_count += int(row["sample_count"])
                record_count += 1
        expected_rows = payload["rows"]
        observed = {
            "record_count": record_count,
            "row_count": logical_row_count,
            "unique_sample_count": (
                len(sample_ids)
                if encoding == EXPANDED_ROWS_ENCODING
                else logical_row_count
            ),
            "episode_count": len(episodes),
        }
        for key, value in observed.items():
            expected_value = expected_rows.get(
                key,
                expected_rows["row_count"]
                if key == "record_count"
                else None,
            )
            if value != int(expected_value):
                raise ValueError(
                    f"Dataset-view {key} mismatch: expected "
                    f"{expected_rows[key]}, found {value}."
                )

    return FrozenDatasetView(
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        ledger_path=ledger_path,
        descriptor=payload,
    )


def write_frozen_view(
    manifest_path: Path | str,
    *,
    descriptor: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    ledger_filename: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> FrozenDatasetViewBuild:
    """Materialize a canonical descriptor and ordered ledger deterministically."""

    manifest = Path(manifest_path).expanduser().resolve()
    ledger_name = ledger_filename or f"{manifest.stem}.rows.jsonl"
    if (
        not ledger_name
        or Path(ledger_name).name != ledger_name
        or Path(ledger_name).is_absolute()
    ):
        raise ValueError("ledger_filename must be a single relative filename.")
    ledger = manifest.parent / ledger_name
    if not dry_run and not overwrite:
        existing = [path for path in (manifest, ledger) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite frozen dataset-view artifact(s): "
                + ", ".join(str(path) for path in existing)
            )

    base = deepcopy(dict(descriptor))
    for generated_key in ("schema", "view_id", "rows"):
        if generated_key in base:
            raise ValueError(
                f"write_frozen_view generates {generated_key!r}; "
                "the descriptor input must omit it."
            )
    base["schema"] = VIEW_SCHEMA
    _validate_descriptor(base, require_rows=False)
    source_backends = {
        str(source["source_id"]): str(source["backend"])
        for source in base["sources"]
    }
    excluded_lineages, excluded_contents = _validate_holdout(
        base["holdout_exclusions"]
    )

    if not dry_run:
        manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_ledger = manifest.parent / (
        f".{ledger.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    handle = None
    if not dry_run:
        handle = temporary_ledger.open("xb")
    ledger_digest = hashlib.sha256()
    sample_ids: set[str] = set()
    episodes: set[tuple[str, str]] = set()
    row_count = 0
    try:
        for ordinal, raw_row in enumerate(rows):
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"Input row {ordinal} must be an object.")
            if "schema" in raw_row or "ordinal" in raw_row:
                raise ValueError(
                    "Input rows must omit generated schema and ordinal fields."
                )
            row = {
                **dict(raw_row),
                "ordinal": ordinal,
                "schema": ROW_SCHEMA,
            }
            _validate_row(
                row,
                expected_ordinal=ordinal,
                source_backends=source_backends,
                representation=base["representation"],
            )
            _validate_ledger_holdout_membership(
                row,
                descriptor_purpose=base.get("purpose"),
                excluded_lineages=excluded_lineages,
                excluded_contents=excluded_contents,
                context=f"Input row {ordinal}",
            )
            encoded = canonical_json_bytes(row) + b"\n"
            ledger_digest.update(encoded)
            if handle is not None:
                handle.write(encoded)
            sample_ids.add(str(row["sample_id"]))
            episodes.add(
                (str(row["source_id"]), str(row["episode_lineage_id"]))
            )
            row_count += 1
        if row_count == 0:
            raise ValueError("A frozen dataset view cannot have an empty ledger.")
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None

        payload = {
            **base,
            "rows": {
                "encoding": EXPANDED_ROWS_ENCODING,
                "path": ledger.name,
                "sha256": ledger_digest.hexdigest(),
                "record_count": row_count,
                "row_count": row_count,
                "unique_sample_count": len(sample_ids),
                "episode_count": len(episodes),
            },
        }
        payload["view_id"] = descriptor_view_id(payload)
        _validate_descriptor(payload, require_rows=True)
        manifest_bytes = canonical_json_bytes(payload) + b"\n"
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()

        if not dry_run:
            temporary_manifest = manifest.parent / (
                f".{manifest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary_manifest.open("xb") as manifest_handle:
                    manifest_handle.write(manifest_bytes)
                    manifest_handle.flush()
                    os.fsync(manifest_handle.fileno())
                os.replace(temporary_ledger, ledger)
                os.replace(temporary_manifest, manifest)
            finally:
                temporary_manifest.unlink(missing_ok=True)

        return FrozenDatasetViewBuild(
            manifest_path=manifest,
            ledger_path=ledger,
            manifest_sha256=manifest_digest,
            ledger_sha256=ledger_digest.hexdigest(),
            view_id=str(payload["view_id"]),
            row_count=row_count,
            unique_sample_count=len(sample_ids),
            episode_count=len(episodes),
            written=not dry_run,
            record_count=row_count,
            encoding=EXPANDED_ROWS_ENCODING,
        )
    finally:
        if handle is not None:
            handle.close()
        temporary_ledger.unlink(missing_ok=True)


def write_frozen_range_view(
    manifest_path: Path | str,
    *,
    descriptor: Mapping[str, Any],
    ranges: Iterable[Mapping[str, Any]],
    ledger_filename: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> FrozenDatasetViewBuild:
    """Materialize a compact arithmetic-range dataset view.

    ``rows.row_count`` remains the logical sample count.  ``record_count`` is
    the much smaller number of JSONL range records.  No per-sample IDs are
    materialized: disjoint ranges plus unique content ownership prove that
    every derived sample ID is unique.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    ledger_name = ledger_filename or f"{manifest.stem}.ranges.jsonl"
    if (
        not ledger_name
        or Path(ledger_name).name != ledger_name
        or Path(ledger_name).is_absolute()
    ):
        raise ValueError("ledger_filename must be a single relative filename.")
    ledger = manifest.parent / ledger_name
    if not dry_run and not overwrite:
        existing = [path for path in (manifest, ledger) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite frozen range-view artifact(s): "
                + ", ".join(str(path) for path in existing)
            )

    base = deepcopy(dict(descriptor))
    for generated_key in ("schema", "view_id", "rows"):
        if generated_key in base:
            raise ValueError(
                f"write_frozen_range_view generates {generated_key!r}; "
                "the descriptor input must omit it."
            )
    base["schema"] = VIEW_SCHEMA
    _validate_descriptor(base, require_rows=False)
    source_backends = {
        str(source["source_id"]): str(source["backend"])
        for source in base["sources"]
    }
    excluded_lineages, excluded_contents = _validate_holdout(
        base["holdout_exclusions"]
    )

    if not dry_run:
        manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_ledger = manifest.parent / (
        f".{ledger.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    handle = None
    if not dry_run:
        handle = temporary_ledger.open("xb")
    ledger_digest = hashlib.sha256()
    episodes: set[tuple[str, str]] = set()
    content_owners: dict[str, tuple[str, str]] = {}
    episode_last_bases: dict[tuple[str, str], int] = {}
    record_count = 0
    logical_row_count = 0
    try:
        for ordinal, raw_range in enumerate(ranges):
            if not isinstance(raw_range, Mapping):
                raise ValueError(
                    f"Input range {ordinal} must be an object."
                )
            if "schema" in raw_range or "ordinal" in raw_range:
                raise ValueError(
                    "Input ranges must omit generated schema and ordinal "
                    "fields."
                )
            row = {
                **dict(raw_range),
                "ordinal": ordinal,
                "schema": RANGE_ROW_SCHEMA,
            }
            _validate_range_row(
                row,
                expected_ordinal=ordinal,
                source_backends=source_backends,
                representation=base["representation"],
            )
            _validate_ledger_holdout_membership(
                row,
                descriptor_purpose=base.get("purpose"),
                excluded_lineages=excluded_lineages,
                excluded_contents=excluded_contents,
                context=f"Input range {ordinal}",
            )
            episode_identity = (
                str(row["source_id"]),
                str(row["episode_lineage_id"]),
            )
            content_id = str(row["episode_content_id"])
            owner = content_owners.setdefault(content_id, episode_identity)
            if owner != episode_identity:
                raise ValueError(
                    "The same episode content appears under multiple range "
                    "episode identities."
                )
            previous_last_base = episode_last_bases.get(episode_identity)
            if (
                previous_last_base is not None
                and int(row["base_start"]) <= previous_last_base
            ):
                raise ValueError(
                    f"Input range {ordinal} overlaps or reorders an earlier "
                    "range for the same episode."
                )
            episode_last_bases[episode_identity] = int(
                row["base_start"]
            ) + (int(row["sample_count"]) - 1) * int(row["base_step"])

            encoded = canonical_json_bytes(row) + b"\n"
            ledger_digest.update(encoded)
            if handle is not None:
                handle.write(encoded)
            episodes.add(episode_identity)
            record_count += 1
            logical_row_count += int(row["sample_count"])

        if record_count == 0:
            raise ValueError(
                "A frozen range dataset view cannot have an empty ledger."
            )
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None

        payload = {
            **base,
            "rows": {
                "encoding": EPISODE_RANGES_ENCODING,
                "path": ledger.name,
                "sha256": ledger_digest.hexdigest(),
                "record_count": record_count,
                "row_count": logical_row_count,
                "unique_sample_count": logical_row_count,
                "episode_count": len(episodes),
            },
        }
        payload["view_id"] = descriptor_view_id(payload)
        _validate_descriptor(payload, require_rows=True)
        manifest_bytes = canonical_json_bytes(payload) + b"\n"
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()

        if not dry_run:
            temporary_manifest = manifest.parent / (
                f".{manifest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary_manifest.open("xb") as manifest_handle:
                    manifest_handle.write(manifest_bytes)
                    manifest_handle.flush()
                    os.fsync(manifest_handle.fileno())
                os.replace(temporary_ledger, ledger)
                os.replace(temporary_manifest, manifest)
            finally:
                temporary_manifest.unlink(missing_ok=True)

        return FrozenDatasetViewBuild(
            manifest_path=manifest,
            ledger_path=ledger,
            manifest_sha256=manifest_digest,
            ledger_sha256=ledger_digest.hexdigest(),
            view_id=str(payload["view_id"]),
            row_count=logical_row_count,
            unique_sample_count=logical_row_count,
            episode_count=len(episodes),
            written=not dry_run,
            record_count=record_count,
            encoding=EPISODE_RANGES_ENCODING,
        )
    finally:
        if handle is not None:
            handle.close()
        temporary_ledger.unlink(missing_ok=True)
