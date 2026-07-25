#!/usr/bin/env python3
"""Build immutable, exhaustive RealMan dataset views.

The local generators intentionally produce row ledgers, not sampler weights:

* ``intervention-incremental`` includes every frame in episodes 1475--1637,
  except the frozen historical holdout and any content-identical copies.
* ``hq-clean-h50`` includes every base frame whose complete 50-action target
  window is valid, except the frozen 32-episode HQ holdout and content copies.

Both artifacts bind the complete LeRobot episode catalog, scoped annotation
content, 18-D representation contract, dual-identity holdout exclusions, and
an ``all_exhaustive`` one-pass epoch contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the dependency-light leaf directly. Importing starVLA.dataloader runs
# the training-time package initializer, which requires Accelerate and Torch.
_DATASET_VIEW_PATH = REPO_ROOT / "starVLA/dataloader/dataset_view.py"
_DATASET_VIEW_SPEC = importlib.util.spec_from_file_location(
    "_realman_dataset_view", _DATASET_VIEW_PATH
)
if _DATASET_VIEW_SPEC is None or _DATASET_VIEW_SPEC.loader is None:
    raise ImportError(f"Could not load dataset-view helpers from {_DATASET_VIEW_PATH}")
dataset_view = importlib.util.module_from_spec(_DATASET_VIEW_SPEC)
sys.modules[_DATASET_VIEW_SPEC.name] = dataset_view
_DATASET_VIEW_SPEC.loader.exec_module(dataset_view)

from starVLA.action_representation import (  # noqa: E402
    REALMAN_18D_ACTION_CONTRACT,
)


DEFAULT_INTERVENTION_ROOT = Path(
    "/home/mehul/work/reward_model_small/"
    "magna_training_data_with_interventions_final_subtask_labelled"
)
DEFAULT_HQ_ROOT = Path(
    "/home/mehul/work/reward_model_small/"
    "latest_high_quality_magna_data_final_subtask_labelled"
)
DEFAULT_REPRESENTATION_SHA256 = REALMAN_18D_ACTION_CONTRACT.sha256()
DEFAULT_HORIZON = 50
DEFAULT_TARGET_FPS = 20
DEFAULT_INCREMENTAL_FIRST_EPISODE = 1475
DEFAULT_INCREMENTAL_LAST_EPISODE = 1637
DEFAULT_CANONICALIZATION_ROOT = Path(
    "/home/mehul/work/dataset-canonicalization"
)
DEFAULT_CANONICAL_MANIFEST = (
    DEFAULT_CANONICALIZATION_ROOT
    / "configs/manifests/dataset_manifests.jsonl.gz"
)
DEFAULT_CANONICAL_ADAPTER = (
    DEFAULT_CANONICALIZATION_ROOT
    / "configs/dataset_adapters/"
    "RealSourceData_RealSource-World__607cbd4f6adf.json"
)
DEFAULT_CANONICAL_CACHE = (
    DEFAULT_CANONICALIZATION_ROOT / ".cache/gcs_lerobot"
)
REALSOURCE_SOURCE_GROUP = "RealSourceData/RealSource-World"
REALSOURCE_VIEW_SELECTION_ALGORITHM = (
    "task_waterfill_rows_sha256_episode_prefix_v1"
)
REALSOURCE_ANNOTATION_ALIGNMENT_SCHEMA = (
    "realsource-source-subtasks-alignment-v1"
)
REALSOURCE_PROVENANCE_CONTENT_CONTRACT = (
    "realsource-canonical-episode-provenance-v1"
)
REALSOURCE_RESAMPLING_ALGORITHM = "nearest_half_up_30_to_20_v1"
REALSOURCE_KNOWN_FULL_EPISODES = 25_927
REALSOURCE_KNOWN_FULL_RAW_FRAMES = 31_091_008
REALSOURCE_KNOWN_VALID_EPISODES = 25_850
REALSOURCE_KNOWN_VALID_RAW_FRAMES = 30_992_703
REALSOURCE_KNOWN_VALID_TARGET_ROWS = 20_661_808
REALSOURCE_EXPECTED_TASKS = 35
DEFAULT_INCREMENTAL_HOLDOUT = (
    1478,
    1480,
    1487,
    1492,
    1494,
    1498,
    1509,
    1511,
    1519,
    1528,
    1541,
    1543,
    1566,
    1622,
    1628,
)
DEFAULT_HQ_HOLDOUT = (
    38,
    41,
    48,
    69,
    101,
    106,
    132,
    163,
    177,
    189,
    254,
    257,
    262,
    267,
    297,
    309,
    318,
    319,
    338,
    342,
    378,
    383,
    390,
    392,
    393,
    405,
    412,
    427,
    451,
    465,
    471,
    498,
)

CONTENT_CONTRACT = "realman-lerobot-numeric-episode-content-v1"
ANNOTATION_CONTRACT = "realman-lerobot-frame-annotations-v1"
CATALOG_SCHEMA = "lerobot-episode-catalog-v1"
SOURCE_ANNOTATION_SCHEMA = "vla-source-annotation-catalog-v1"
SUBTASK_SEGMENTS_SCHEMA = "canonical-subtask-segments-parquet-v1"
SUBTASK_SEGMENTS_REQUIRED_COLUMNS = frozenset(
    {
        "episode_index",
        "segment_index",
        "subtask_index",
        "subtask",
        "start_frame",
        "end_frame_exclusive",
    }
)
SOURCE_CONTENT_SCHEMA = "vla-source-content-catalog-v1"
SUBTASK_CATALOG_SCHEMA = "realman-subtask-prompt-catalog-v1"
ACTION_SUPERVISION_AUDIT_SCHEMA = "realman-action-supervision-audit-v1"
ACTION_LABEL_SEMANTICS_SCHEMA = "realman-action-label-semantics-v1"
RESULT_PREFIX = "REALMAN_DATASET_VIEW_RESULT="

EXPLICIT_ACTION_VALIDITY_COLUMN = "valid_action"
EXPLICIT_ACTION_OWNER_COLUMNS = ("action_owner", "action_source")
EXPERT_ACTION_OWNER_VALUES = frozenset(
    {
        "correction",
        "expert",
        "expert_correction",
        "human",
        "human_correction",
        "human_intervention",
        "intervention",
        "operator",
        "recovery",
        "teleop",
        "teleoperator",
    }
)
DEFAULT_ACTION_VALIDITY_INVALID_RUN_LENGTH = 10

CONTENT_COLUMNS = (
    "frame_index",
    "timestamp",
    "source.observation.state",
    "source.action",
    "observation.state",
    "action",
    "valid_state",
    "valid_state_source",
    "subtask_index",
    "task_id",
    "task_index",
)
ANNOTATION_COLUMNS = (
    "frame_index",
    "valid_state",
    "valid_state_source",
    "subtask_index",
    "task_id",
    "task_index",
)


@dataclass(frozen=True, slots=True)
class EpisodeCatalogRecord:
    episode_index: int
    length: int
    data_file: str


@dataclass(frozen=True, slots=True)
class EpisodeSnapshot:
    record: EpisodeCatalogRecord
    lineage_id: str
    content_id: str
    frame_content_sha256: str
    annotation_sha256: str
    valid_state: tuple[bool, ...]
    subtask_index: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RealSourceEpisode:
    dataset_id: str
    sid: str
    revision: str
    episode_index: int
    length: int
    target_row_count: int
    data_file: str
    annotation_ordinal: int
    annotation_episode_index: int
    annotation_sha256: str
    lineage_id: str
    content_id: str


@dataclass(frozen=True, slots=True)
class RealSourceTaskCatalog:
    dataset_id: str
    sid: str
    revision: str
    fps: int
    metadata_path: Path
    metadata_sha256: str
    annotation_path: Path
    annotation_sha256: str
    subtask_segments_path: Path
    subtask_segments_sha256: str
    subtask_segments_summary: Mapping[str, Any]
    alignment: Mapping[str, Any]
    catalog_episode_count: int
    catalog_raw_frame_count: int
    quality_value_counts: Mapping[str, int]
    valid_episodes: tuple[RealSourceEpisode, ...]

    @property
    def valid_raw_frame_count(self) -> int:
        return sum(episode.length for episode in self.valid_episodes)

    @property
    def valid_target_row_count(self) -> int:
        return sum(episode.target_row_count for episode in self.valid_episodes)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or value is None:
        return {"special": "missing"}
    if isinstance(value, float):
        if math.isnan(value):
            return {"special_float": "nan"}
        if math.isinf(value):
            return {"special_float": "positive_infinity" if value > 0 else "negative_infinity"}
        return value
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _strict_valid_flag(value: Any, *, episode_index: int, frame_index: int) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"Episode {episode_index} frame {frame_index} has non-finite valid_state."
            )
        if value in (0, 1):
            return bool(value)
    raise ValueError(
        f"Episode {episode_index} frame {frame_index} has invalid valid_state={value!r}; "
        "expected exact 0/1."
    )


def _strict_subtask_index(value: Any, *, label: str) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a non-negative integer; got {value!r}."
        ) from exc
    if not math.isfinite(numeric) or numeric < 0 or numeric != int(numeric):
        raise ValueError(
            f"{label} must be a non-negative integer; got {value!r}."
        )
    return int(numeric)


def _load_subtask_catalog(
    dataset_root: Path,
) -> tuple[dict[int, dict[str, Any]], str, str]:
    path = dataset_root / "meta/subtasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            "Labeled RealMan dataset is missing meta/subtasks.parquet: "
            f"{path}"
        )
    required = {
        "subtask_index",
        "local_subtask_text",
        "global_subtask_type",
    }
    names = set(pq.read_schema(path).names)
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            f"Subtask catalog {path} is missing columns {missing}."
        )
    optional = [
        name
        for name in ("local_subtask_id", "is_mistake", "source", "optional")
        if name in names
    ]
    frame = pq.read_table(
        path, columns=sorted(required) + optional
    ).to_pandas()
    catalog: dict[int, dict[str, Any]] = {}
    for ordinal, row in frame.iterrows():
        index = _strict_subtask_index(
            row["subtask_index"],
            label=f"subtask catalog row {ordinal} index",
        )
        if index in catalog:
            raise ValueError(
                f"Subtask catalog {path} contains duplicate index {index}."
            )
        local_text = str(row["local_subtask_text"]).strip()
        global_type = str(row["global_subtask_type"]).strip()
        if not local_text or not global_type:
            raise ValueError(
                f"Subtask catalog index {index} has empty prompt metadata."
            )
        useful = (
            local_text.casefold() != "__unlabeled__"
            and global_type.casefold() != "unlabeled"
        )
        entry: dict[str, Any] = {
            "subtask_index": index,
            "local_subtask_text": local_text,
            "global_subtask_type": global_type,
            "useful_prompt": useful,
        }
        for name in optional:
            entry[name] = _json_safe(row[name])
        catalog[index] = entry
    if not catalog:
        raise ValueError(f"Subtask catalog {path} is empty.")
    catalog_sha256 = dataset_view.canonical_json_sha256(
        {
            "schema": SUBTASK_CATALOG_SCHEMA,
            "entries": [catalog[index] for index in sorted(catalog)],
        }
    )
    return catalog, dataset_view.file_sha256(path), catalog_sha256


def _canonical_manifest_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical dataset manifest is missing: {path}")
    raw = path.read_bytes()
    opener = gzip.open if path.suffix == ".gz" else path.open
    rows: list[dict[str, Any]] = []
    try:
        with opener(path, "rt", encoding="utf-8") if path.suffix == ".gz" else opener(
            "rt", encoding="utf-8"
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Canonical manifest row {line_number} is not an object."
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read canonical manifest {path}: {exc}") from exc
    return rows, hashlib.sha256(raw).hexdigest()


def _parse_fraction(value: str | float | Decimal) -> Decimal:
    try:
        fraction = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid RealSource view fraction {value!r}.") from exc
    if not fraction.is_finite() or fraction <= 0 or fraction > 1:
        raise ValueError("RealSource view fraction must be in (0, 1].")
    return fraction


def _target_20hz_row_count(source_length: int) -> int:
    """Number of valid half-up 20 Hz indices in a 30 Hz episode.

    ``source_index = floor(target_index * 30 / 20 + 0.5)`` is exactly
    ``(3 * target_index + 1) // 2``.  The final target index must map below
    ``source_length``.  This is *not* ``ceil(2 * length / 3)`` when
    ``length % 3 == 2``.
    """

    if isinstance(source_length, bool) or not isinstance(source_length, int):
        raise ValueError("source_length must be an integer.")
    if source_length <= 0:
        raise ValueError("source_length must be positive.")
    return (2 * source_length + 1) // 3


def _annotation_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"RealSource source annotation catalog is missing: {path}"
        )
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid RealSource annotation JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"RealSource annotation {path}:{line_number} is not an object."
            )
        records.append(payload)
    return records, hashlib.sha256(raw).hexdigest()


def _validate_realsource_subtask_segments(
    path: Path,
    *,
    dataset_id: str,
    episode_lengths: Mapping[int, int],
    source_episode_index_map: Mapping[int, int | None],
) -> tuple[str, Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"RealSource subtask segment sidecar is missing: {path}"
        )
    segments = pd.read_parquet(path)
    missing_columns = sorted(
        SUBTASK_SEGMENTS_REQUIRED_COLUMNS.difference(segments.columns)
    )
    if missing_columns:
        raise ValueError(
            f"{dataset_id} subtask segment sidecar lacks required columns "
            f"{missing_columns}."
        )
    if segments.duplicated(["episode_index", "segment_index"]).any():
        raise ValueError(
            f"{dataset_id} subtask segment sidecar has duplicate "
            "(episode_index, segment_index) rows."
        )
    zero_length_count = 0
    unaligned_source_row_count = 0
    usable_episode_ids: set[int] = set()
    for row_number, row in enumerate(segments.to_dict("records")):
        values: dict[str, int] = {}
        for column in (
            "episode_index",
            "segment_index",
            "subtask_index",
            "start_frame",
            "end_frame_exclusive",
        ):
            raw = row[column]
            if isinstance(raw, bool) or not isinstance(
                raw, (int, np.integer)
            ):
                raise ValueError(
                    f"{dataset_id} subtask segment row {row_number} "
                    f"column {column!r} must be an integer."
                )
            values[column] = int(raw)
        source_episode_index = values["episode_index"]
        episode_index = source_episode_index_map.get(
            source_episode_index
        )
        if not str(row["subtask"]).strip():
            raise ValueError(
                f"{dataset_id} subtask segment row {row_number} has an "
                "empty label."
            )
        if episode_index is None:
            unaligned_source_row_count += 1
            continue
        if episode_index not in episode_lengths:
            raise ValueError(
                f"{dataset_id} subtask segment row {row_number} references "
                f"unknown source episode {source_episode_index} mapped to "
                f"{episode_index}."
            )
        episode_index = int(episode_index)
        start_frame = values["start_frame"]
        end_frame = values["end_frame_exclusive"]
        episode_length = int(episode_lengths[episode_index])
        if (
            start_frame < 0
            or end_frame < start_frame
            or (
                end_frame > start_frame
                and end_frame > episode_length
            )
        ):
            raise ValueError(
                f"{dataset_id} subtask segment row {row_number} has invalid "
                f"bounds [{start_frame}, {end_frame}) for episode length "
                f"{episode_length}."
            )
        source_bounds = {
            "source_start_frame": start_frame,
            "source_end_frame": end_frame,
        }
        for source_column in tuple(source_bounds):
            if source_column in row and not pd.isna(row[source_column]):
                observed = row[source_column]
                if isinstance(observed, bool) or not isinstance(
                    observed, (int, np.integer)
                ):
                    raise ValueError(
                        f"{dataset_id} subtask segment row {row_number} "
                        f"column {source_column!r} must be an integer."
                    )
                source_bounds[source_column] = int(observed)
        source_start = source_bounds["source_start_frame"]
        source_end = source_bounds["source_end_frame"]
        if (
            source_start > start_frame
            or source_end < end_frame
            or source_end < source_start
        ):
            raise ValueError(
                f"{dataset_id} subtask segment row {row_number} canonical "
                "bounds are not a valid clipping of source bounds."
            )
        if start_frame == end_frame:
            zero_length_count += 1
        else:
            usable_episode_ids.add(episode_index)
    summary = {
        "schema": SUBTASK_SEGMENTS_SCHEMA,
        "row_count": int(len(segments)),
        "episode_count": int(segments["episode_index"].nunique()),
        "usable_episode_count": len(usable_episode_ids),
        "zero_length_span_count": zero_length_count,
        "unaligned_source_row_count": unaligned_source_row_count,
        "frame_coordinates": "raw_source_frames",
        "boundary_semantics": "start_inclusive_end_exclusive",
        "labels_synthesized": False,
    }
    return dataset_view.file_sha256(path), summary


def _align_realsource_annotations(
    *,
    dataset_id: str,
    episode_rows: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[Mapping[str, Any], int, Mapping[str, Any]]], dict[str, Any]]:
    """Return content-verified episode/annotation pairs.

    The uploaded ``Collect_the_mail`` annotation stream contains one extra
    record at annotation ordinal 122.  Its subsequent annotation IDs are one
    larger than the data episode IDs.  ``total_frames`` proves the unique
    alignment after dropping that record.  Every other RealSource task is an
    exact episode-index join and must also match ``total_frames``.
    """

    ordered = sorted(episode_rows, key=lambda row: int(row["episode_index"]))
    if dataset_id.endswith("/Collect_the_mail"):
        if len(annotations) != len(ordered) + 1:
            raise ValueError(
                "Collect_the_mail must contain exactly one extra source "
                "annotation record."
            )
        if len(ordered) <= 122:
            raise ValueError("Collect_the_mail catalog is unexpectedly short.")
        aligned = []
        for position, episode in enumerate(ordered):
            annotation_ordinal = position if position <= 121 else position + 1
            annotation = annotations[annotation_ordinal]
            if int(annotation.get("total_frames", -1)) != int(episode["length"]):
                raise ValueError(
                    "Collect_the_mail alignment failed total_frames check at "
                    f"data episode {episode['episode_index']} and annotation "
                    f"ordinal {annotation_ordinal}."
                )
            aligned.append((episode, annotation_ordinal, annotation))
        dropped = annotations[122]
        if int(dropped.get("episode_index", -1)) != 122:
            raise ValueError(
                "Collect_the_mail extra annotation is no longer ordinal/id 122."
            )
        alignment = {
            "schema": REALSOURCE_ANNOTATION_ALIGNMENT_SCHEMA,
            "algorithm": (
                "ordinals_0_121_to_data_0_121_drop_ordinal_122_"
                "ordinals_123_390_to_data_122_389"
            ),
            "content_verification": "annotation.total_frames == episode.length",
            "dropped_annotation_ordinal": 122,
            "dropped_annotation_episode_index": int(
                dropped["episode_index"]
            ),
            "dropped_annotation_sha256": dataset_view.canonical_json_sha256(
                dropped
            ),
        }
        return aligned, alignment

    if len(annotations) != len(ordered):
        raise ValueError(
            f"{dataset_id} annotation/catalog count mismatch: "
            f"{len(annotations)} vs {len(ordered)}."
        )
    by_episode: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for ordinal, annotation in enumerate(annotations):
        raw_episode_index = annotation.get("episode_index")
        if (
            isinstance(raw_episode_index, bool)
            or not isinstance(raw_episode_index, int)
            or raw_episode_index < 0
            or raw_episode_index in by_episode
        ):
            raise ValueError(
                f"{dataset_id} has invalid/duplicate annotation episode ID "
                f"{raw_episode_index!r}."
            )
        by_episode[raw_episode_index] = (ordinal, annotation)
    aligned = []
    for episode in ordered:
        episode_index = int(episode["episode_index"])
        if episode_index not in by_episode:
            raise ValueError(
                f"{dataset_id} lacks annotation for episode {episode_index}."
            )
        ordinal, annotation = by_episode[episode_index]
        if int(annotation.get("total_frames", -1)) != int(episode["length"]):
            raise ValueError(
                f"{dataset_id} episode {episode_index} annotation total_frames "
                "does not match episode length."
            )
        aligned.append((episode, ordinal, annotation))
    return aligned, {
        "schema": REALSOURCE_ANNOTATION_ALIGNMENT_SCHEMA,
        "algorithm": "exact_episode_index_join_v1",
        "content_verification": "annotation.total_frames == episode.length",
    }


def _load_realsource_catalog(
    *,
    canonical_manifest: Path,
    adapter_path: Path,
    cache_dir: Path,
    expected_inventory: Mapping[str, int] | None = None,
) -> tuple[
    tuple[RealSourceTaskCatalog, ...],
    str,
    str,
]:
    manifest_rows, manifest_sha256 = _canonical_manifest_rows(
        canonical_manifest
    )
    if not adapter_path.is_file():
        raise FileNotFoundError(f"RealSource adapter is missing: {adapter_path}")
    adapter_sha256 = dataset_view.file_sha256(adapter_path)
    rows = sorted(
        (
            row
            for row in manifest_rows
            if row.get("source_group") == REALSOURCE_SOURCE_GROUP
        ),
        key=lambda row: (
            str(row.get("dataset_id", "")),
            str(row.get("sid", "")),
            str(row.get("revision", "")),
        ),
    )
    if len(rows) != REALSOURCE_EXPECTED_TASKS:
        raise ValueError(
            f"Expected {REALSOURCE_EXPECTED_TASKS} RealSource tasks, "
            f"found {len(rows)}."
        )
    catalogs: list[RealSourceTaskCatalog] = []
    for source in rows:
        dataset_id = str(source.get("dataset_id") or "")
        sid = str(source.get("sid") or "")
        revision = str(source.get("revision") or "")
        if not dataset_id or not sid or not revision:
            raise ValueError("RealSource manifest row lacks dataset identity.")
        if int(source.get("fps", -1)) != 30:
            raise ValueError(f"{dataset_id} is not exact 30 Hz.")
        root = cache_dir / sid / revision / "meta"
        metadata_path = root / "episodes/chunk-000/file-000.parquet"
        annotation_path = root / "source_sub_tasks.jsonl"
        subtask_segments_path = root / "subtask_segments.parquet"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"RealSource episode metadata is missing: {metadata_path}"
            )
        metadata_columns = (
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
        )
        metadata = pd.read_parquet(
            metadata_path, columns=list(metadata_columns)
        ).sort_values("episode_index")
        if metadata["episode_index"].duplicated().any():
            raise ValueError(f"{dataset_id} has duplicate episode indices.")
        metadata_rows = metadata.to_dict("records")
        annotations, annotation_file_sha256 = _annotation_records(
            annotation_path
        )
        aligned, alignment = _align_realsource_annotations(
            dataset_id=dataset_id,
            episode_rows=metadata_rows,
            annotations=annotations,
        )
        source_episode_index_map = {
            int(annotation["episode_index"]): int(
                episode["episode_index"]
            )
            for episode, _, annotation in aligned
        }
        (
            subtask_segments_sha256,
            subtask_segments_summary,
        ) = _validate_realsource_subtask_segments(
            subtask_segments_path,
            dataset_id=dataset_id,
            episode_lengths={
                int(row["episode_index"]): int(row["length"])
                for row in metadata_rows
            },
            source_episode_index_map=source_episode_index_map,
        )
        catalog_episode_count = len(metadata_rows)
        catalog_raw_frame_count = sum(
            int(row["length"]) for row in metadata_rows
        )
        if (
            int(source.get("total_episodes", -1)) != catalog_episode_count
            or int(source.get("total_frames", -1))
            != catalog_raw_frame_count
        ):
            raise ValueError(
                f"{dataset_id} manifest/metadata inventory mismatch."
            )
        metadata_sha256 = dataset_view.file_sha256(metadata_path)
        valid_episodes: list[RealSourceEpisode] = []
        quality_value_counts: dict[str, int] = defaultdict(int)
        for episode, annotation_ordinal, annotation in aligned:
            quality = annotation.get("quality_assessments")
            if not isinstance(quality, Mapping):
                raise ValueError(
                    f"{dataset_id} annotation {annotation_ordinal} lacks "
                    "quality_assessments."
                )
            overall_valid = quality.get("overall_valid")
            quality_value_counts[
                overall_valid if isinstance(overall_valid, str) else "<MISSING>"
            ] += 1
            if overall_valid != "VALID":
                continue
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            chunk_index = int(episode["data/chunk_index"])
            file_index = int(episode["data/file_index"])
            annotation_digest = dataset_view.canonical_json_sha256(annotation)
            episode_metadata_digest = dataset_view.canonical_json_sha256(
                {
                    "dataset_id": dataset_id,
                    "sid": sid,
                    "revision": revision,
                    "episode_index": episode_index,
                    "length": length,
                    "data/chunk_index": chunk_index,
                    "data/file_index": file_index,
                    "annotation_ordinal": annotation_ordinal,
                    "annotation_sha256": annotation_digest,
                }
            )
            content_id = dataset_view.make_episode_content_id(
                frame_content_sha256=episode_metadata_digest,
                length=length,
                content_contract=REALSOURCE_PROVENANCE_CONTENT_CONTRACT,
            )
            lineage_id = dataset_view.make_episode_lineage_id(
                backend="canonical",
                source_id=dataset_id,
                catalog_sha256=manifest_sha256,
                episode_index=episode_index,
                length=length,
                episode_metadata_sha256=episode_metadata_digest,
            )
            valid_episodes.append(
                RealSourceEpisode(
                    dataset_id=dataset_id,
                    sid=sid,
                    revision=revision,
                    episode_index=episode_index,
                    length=length,
                    target_row_count=_target_20hz_row_count(length),
                    data_file=(
                        f"data/chunk-{chunk_index:03d}/"
                        f"file-{file_index:03d}.parquet"
                    ),
                    annotation_ordinal=annotation_ordinal,
                    annotation_episode_index=int(
                        annotation["episode_index"]
                    ),
                    annotation_sha256=annotation_digest,
                    lineage_id=lineage_id,
                    content_id=content_id,
                )
            )
        catalogs.append(
            RealSourceTaskCatalog(
                dataset_id=dataset_id,
                sid=sid,
                revision=revision,
                fps=30,
                metadata_path=metadata_path,
                metadata_sha256=metadata_sha256,
                annotation_path=annotation_path,
                annotation_sha256=annotation_file_sha256,
                subtask_segments_path=subtask_segments_path,
                subtask_segments_sha256=(
                    subtask_segments_sha256
                ),
                subtask_segments_summary=(
                    subtask_segments_summary
                ),
                alignment=alignment,
                catalog_episode_count=catalog_episode_count,
                catalog_raw_frame_count=catalog_raw_frame_count,
                quality_value_counts=dict(
                    sorted(quality_value_counts.items())
                ),
                valid_episodes=tuple(valid_episodes),
            )
        )

    totals = {
        "catalog_episodes": sum(item.catalog_episode_count for item in catalogs),
        "catalog_raw_frames": sum(
            item.catalog_raw_frame_count for item in catalogs
        ),
        "valid_episodes": sum(len(item.valid_episodes) for item in catalogs),
        "valid_raw_frames": sum(
            item.valid_raw_frame_count for item in catalogs
        ),
        "valid_target_rows": sum(
            item.valid_target_row_count for item in catalogs
        ),
    }
    expected = (
        {
            "catalog_episodes": REALSOURCE_KNOWN_FULL_EPISODES,
            "catalog_raw_frames": REALSOURCE_KNOWN_FULL_RAW_FRAMES,
            "valid_episodes": REALSOURCE_KNOWN_VALID_EPISODES,
            "valid_raw_frames": REALSOURCE_KNOWN_VALID_RAW_FRAMES,
            "valid_target_rows": REALSOURCE_KNOWN_VALID_TARGET_ROWS,
        }
        if expected_inventory is None
        else dict(expected_inventory)
    )
    if totals != expected:
        raise ValueError(
            "RealSource frozen inventory changed; refuse to silently reuse the "
            f"known production contract: observed={totals}, expected={expected}."
        )
    return tuple(catalogs), manifest_sha256, adapter_sha256


def _load_info(dataset_root: Path) -> tuple[dict[str, Any], str]:
    info_path = dataset_root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot info.json: {info_path}")
    raw = info_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LeRobot info.json at {info_path}: {exc}") from exc
    if payload.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot v3.0 at {dataset_root}, "
            f"found {payload.get('codebase_version')!r}."
        )
    fps = payload.get("fps")
    if fps != DEFAULT_TARGET_FPS:
        raise ValueError(
            f"Expected {DEFAULT_TARGET_FPS} Hz RealMan data, found fps={fps!r}."
        )
    return payload, hashlib.sha256(raw).hexdigest()


def _load_episode_catalog(
    dataset_root: Path,
) -> tuple[
    dict[int, EpisodeCatalogRecord],
    dict[str, Any],
]:
    info, info_sha256 = _load_info(dataset_root)
    episode_files = sorted((dataset_root / "meta/episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(
            f"No LeRobot episode metadata files under {dataset_root / 'meta/episodes'}."
        )
    frames = [pd.read_parquet(path) for path in episode_files]
    table = pd.concat(frames, ignore_index=True)
    required = {
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Episode catalog is missing columns: {missing}.")
    if table["episode_index"].duplicated().any():
        duplicates = sorted(
            int(value)
            for value in table.loc[
                table["episode_index"].duplicated(keep=False), "episode_index"
            ].unique()
        )
        raise ValueError(f"Episode catalog contains duplicate IDs: {duplicates[:20]}.")

    records: dict[int, EpisodeCatalogRecord] = {}
    canonical_records: list[dict[str, int]] = []
    for _, row in table.sort_values("episode_index").iterrows():
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        chunk_index = int(row["data/chunk_index"])
        file_index = int(row["data/file_index"])
        if episode_index < 0 or length <= 0 or chunk_index < 0 or file_index < 0:
            raise ValueError(
                "Invalid episode catalog row: "
                f"episode={episode_index}, length={length}, "
                f"chunk={chunk_index}, file={file_index}."
            )
        data_file = f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        records[episode_index] = EpisodeCatalogRecord(
            episode_index=episode_index,
            length=length,
            data_file=data_file,
        )
        canonical_records.append(
            {"episode_id": episode_index, "length": length}
        )

    declared_episode_count = info.get("total_episodes")
    declared_frame_count = info.get("total_frames")
    observed_frames = sum(record.length for record in records.values())
    if declared_episode_count != len(records) or declared_frame_count != observed_frames:
        raise ValueError(
            "LeRobot info/catalog count mismatch: "
            f"declared episodes/frames={declared_episode_count}/{declared_frame_count}, "
            f"observed={len(records)}/{observed_frames}."
        )
    file_bindings = [
        {
            "path": path.relative_to(dataset_root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": dataset_view.file_sha256(path),
        }
        for path in episode_files
    ]
    digest_payload = {
        "schema": CATALOG_SCHEMA,
        "dataset_name": dataset_root.name,
        "lerobot_version": "v3.0",
        "info_sha256": info_sha256,
        "episode_files": file_bindings,
        "episodes": canonical_records,
    }
    binding = {
        "catalog_sha256": dataset_view.canonical_json_sha256(digest_payload),
        "info_sha256": info_sha256,
        "episode_count": len(records),
        "frame_count": observed_frames,
        "episode_metadata_files": file_bindings,
    }
    return records, binding


def _bind_local_eval_holdout(
    *,
    manifest_path: Path | str,
    dataset_root: Path,
    catalog: Mapping[int, EpisodeCatalogRecord],
    catalog_binding: Mapping[str, Any],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Authenticate the local LeRobot split and return its holdout IDs.

    The view builder intentionally validates only catalog/split membership,
    not the split's legacy train-statistics attachment. The shared union
    statistics do not exist yet during bootstrap; the training loader later
    validates the complete split manifest including all statistics bindings.
    """

    resolved = Path(manifest_path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(
            "Local evaluation holdout manifest must be a regular file: "
            f"{resolved}"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Local evaluation holdout manifest is invalid: {exc}"
        ) from exc
    expected_role_contract = {
        "train_episode_selection": "complement_of_holdout",
        "evaluation_episode_selection": "holdout_episode_indices",
        "normalization_statistics": "train_statistics_only",
    }
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("role_contract") != expected_role_contract
    ):
        raise ValueError(
            "Local evaluation holdout must be a schema_version=1 episode "
            "split with the exact complement-of-holdout role contract."
        )
    split_id = payload.get("split_id")
    if not isinstance(split_id, str) or not split_id.strip():
        raise ValueError(
            "Local evaluation holdout split_id must be non-empty."
        )
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(
            "Local evaluation holdout datasets must be a list."
        )
    matches = [
        item
        for item in datasets
        if isinstance(item, Mapping)
        and item.get("dataset_name") == dataset_root.name
    ]
    if len(matches) != 1:
        raise ValueError(
            "Local evaluation holdout must contain exactly one dataset entry "
            f"for {dataset_root.name!r}; found {len(matches)}."
        )
    entry = matches[0]
    expected_catalog_sha256 = str(catalog_binding["catalog_sha256"])
    if entry.get("full_catalog_sha256") != expected_catalog_sha256:
        raise ValueError(
            "Local evaluation holdout full_catalog_sha256 does not match "
            "the selected dataset."
        )
    expected_counts = {
        "full_episode_count": int(catalog_binding["episode_count"]),
        "full_frame_count": int(catalog_binding["frame_count"]),
        "info_sha256": str(catalog_binding["info_sha256"]),
    }
    mismatches = {
        key: {
            "expected": value,
            "found": entry.get(key),
        }
        for key, value in expected_counts.items()
        if entry.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Local evaluation holdout catalog binding mismatch: "
            f"{mismatches}."
        )
    raw_holdout = entry.get("holdout_episode_indices")
    if (
        not isinstance(raw_holdout, list)
        or not raw_holdout
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in raw_holdout
        )
    ):
        raise ValueError(
            "Local evaluation holdout episode IDs must be a non-empty "
            "integer list."
        )
    holdout = tuple(sorted(raw_holdout))
    if len(holdout) != len(set(holdout)):
        raise ValueError(
            "Local evaluation holdout episode IDs contain duplicates."
        )
    unknown = sorted(set(holdout) - set(catalog))
    if unknown:
        raise ValueError(
            "Local evaluation holdout references unknown episodes: "
            f"{unknown}."
        )
    lengths_by_id = {
        int(episode_id): int(record.length)
        for episode_id, record in catalog.items()
    }

    def episode_set_sha256(episode_ids: Sequence[int]) -> str:
        return dataset_view.canonical_json_sha256(
            {
                "schema": "lerobot-episode-set-v1",
                "episodes": [
                    {
                        "episode_id": int(episode_id),
                        "length": lengths_by_id[int(episode_id)],
                    }
                    for episode_id in sorted(episode_ids)
                ],
            }
        )

    train = tuple(sorted(set(catalog) - set(holdout)))
    if not train:
        raise ValueError(
            "Local evaluation holdout leaves no training episodes."
        )
    expected_split_bindings = {
        "holdout_episode_count": len(holdout),
        "holdout_frame_count": sum(
            lengths_by_id[value] for value in holdout
        ),
        "holdout_catalog_sha256": episode_set_sha256(holdout),
        "train_episode_count": len(train),
        "train_frame_count": sum(lengths_by_id[value] for value in train),
        "train_catalog_sha256": episode_set_sha256(train),
        "train_episode_selection": {"kind": "complement_of_holdout"},
    }
    split_mismatches = {
        key: {
            "expected": value,
            "found": entry.get(key),
        }
        for key, value in expected_split_bindings.items()
        if entry.get(key) != value
    }
    if split_mismatches:
        raise ValueError(
            "Local evaluation holdout train/holdout binding mismatch: "
            f"{split_mismatches}."
        )
    selection = payload.get("selection")
    if isinstance(selection, Mapping):
        selected_sorted = selection.get("selected_episode_ids_sorted")
        if (
            selected_sorted is not None
            and selected_sorted != list(holdout)
        ):
            raise ValueError(
                "Local evaluation holdout selection ranking does not match "
                "datasets[].holdout_episode_indices."
            )
    binding = {
        "schema": "lerobot-eval-holdout-binding-v1",
        "manifest_filename": resolved.name,
        "manifest_sha256": dataset_view.file_sha256(resolved),
        "split_id": split_id,
        "dataset_name": dataset_root.name,
        "full_catalog_sha256": expected_catalog_sha256,
        "holdout_episode_count": len(holdout),
        "holdout_catalog_sha256": expected_split_bindings[
            "holdout_catalog_sha256"
        ],
    }
    binding["sha256"] = dataset_view.canonical_json_sha256(binding)
    return holdout, binding


def _catalog_data_file_indices(
    dataset_root: Path,
) -> dict[int, tuple[int, int]]:
    frames = [
        pd.read_parquet(
            path,
            columns=("episode_index", "data/chunk_index", "data/file_index"),
        )
        for path in sorted((dataset_root / "meta/episodes").glob("**/*.parquet"))
    ]
    table = pd.concat(frames, ignore_index=True)
    return {
        int(row["episode_index"]): (
            int(row["data/chunk_index"]),
            int(row["data/file_index"]),
        )
        for _, row in table.iterrows()
    }


def _hash_frame_rows(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    contract: str,
    episode_length: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        dataset_view.canonical_json_bytes(
            {
                "schema": contract,
                "columns": list(columns),
                "episode_length": episode_length,
            }
        )
        + b"\n"
    )
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        digest.update(
            dataset_view.canonical_json_bytes(
                [_json_safe(value) for value in row]
            )
            + b"\n"
        )
    return digest.hexdigest()


def _load_episode_snapshots(
    *,
    dataset_root: Path,
    source_id: str,
    catalog: Mapping[int, EpisodeCatalogRecord],
    catalog_sha256: str,
    episode_ids: Sequence[int],
) -> tuple[dict[int, EpisodeSnapshot], str, str]:
    missing_ids = sorted(set(int(value) for value in episode_ids) - set(catalog))
    if missing_ids:
        raise ValueError(
            f"Requested episode IDs are absent from the catalog: {missing_ids[:20]}."
        )
    file_indices = _catalog_data_file_indices(dataset_root)
    by_file: dict[str, list[int]] = defaultdict(list)
    for episode_index in sorted(set(int(value) for value in episode_ids)):
        if episode_index not in file_indices:
            raise ValueError(
                f"Episode {episode_index} has no data-file catalog binding."
            )
        chunk_index, file_index = file_indices[episode_index]
        expected = f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        if catalog[episode_index].data_file != expected:
            raise ValueError(
                f"Episode {episode_index} has inconsistent data-file binding."
            )
        by_file[expected].append(episode_index)

    snapshots: dict[int, EpisodeSnapshot] = {}
    for relative_path in sorted(by_file):
        data_path = dataset_root / relative_path
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot data shard: {data_path}")
        schema_names = set(pq.read_schema(data_path).names)
        required = {"episode_index", *CONTENT_COLUMNS}
        missing_columns = sorted(required - schema_names)
        if missing_columns:
            raise ValueError(
                f"Data shard {data_path} is missing columns {missing_columns}."
            )
        table = pq.read_table(
            data_path,
            columns=["episode_index", *CONTENT_COLUMNS],
            filters=[("episode_index", "in", by_file[relative_path])],
        )
        data = table.to_pandas()
        for episode_index in sorted(by_file[relative_path]):
            frame = data.loc[data["episode_index"] == episode_index].copy()
            frame.sort_values("frame_index", inplace=True)
            frame.reset_index(drop=True, inplace=True)
            record = catalog[episode_index]
            if len(frame) != record.length:
                raise ValueError(
                    f"Episode {episode_index} length mismatch: catalog "
                    f"{record.length}, data {len(frame)}."
                )
            observed_indices = frame["frame_index"].to_numpy(dtype=np.int64)
            expected_indices = np.arange(record.length, dtype=np.int64)
            if not np.array_equal(observed_indices, expected_indices):
                raise ValueError(
                    f"Episode {episode_index} frame_index is not exact 0.."
                    f"{record.length - 1}."
                )
            frame_content_sha256 = _hash_frame_rows(
                frame,
                columns=CONTENT_COLUMNS,
                contract=CONTENT_CONTRACT,
                episode_length=record.length,
            )
            annotation_sha256 = _hash_frame_rows(
                frame,
                columns=ANNOTATION_COLUMNS,
                contract=ANNOTATION_CONTRACT,
                episode_length=record.length,
            )
            content_id = dataset_view.make_episode_content_id(
                frame_content_sha256=frame_content_sha256,
                length=record.length,
                content_contract=CONTENT_CONTRACT,
            )
            lineage_id = dataset_view.make_episode_lineage_id(
                backend="lerobot",
                source_id=source_id,
                catalog_sha256=catalog_sha256,
                episode_index=episode_index,
                length=record.length,
            )
            valid_state = tuple(
                _strict_valid_flag(
                    value,
                    episode_index=episode_index,
                    frame_index=frame_index,
                )
                for frame_index, value in enumerate(frame["valid_state"].tolist())
            )
            subtask_index = tuple(
                _strict_subtask_index(
                    value,
                    label=(
                        f"episode {episode_index} frame {frame_index} "
                        "subtask_index"
                    ),
                )
                for frame_index, value in enumerate(
                    frame["subtask_index"].tolist()
                )
            )
            snapshots[episode_index] = EpisodeSnapshot(
                record=record,
                lineage_id=lineage_id,
                content_id=content_id,
                frame_content_sha256=frame_content_sha256,
                annotation_sha256=annotation_sha256,
                valid_state=valid_state,
                subtask_index=subtask_index,
            )

    ordered_snapshots = [snapshots[value] for value in sorted(snapshots)]
    annotation_catalog_sha256 = dataset_view.canonical_json_sha256(
        {
            "schema": SOURCE_ANNOTATION_SCHEMA,
            "contract": ANNOTATION_CONTRACT,
            "episodes": [
                {
                    "episode_index": snapshot.record.episode_index,
                    "annotation_sha256": snapshot.annotation_sha256,
                }
                for snapshot in ordered_snapshots
            ],
        }
    )
    source_content_sha256 = dataset_view.canonical_json_sha256(
        {
            "schema": SOURCE_CONTENT_SCHEMA,
            "contract": CONTENT_CONTRACT,
            "episodes": [
                {
                    "episode_index": snapshot.record.episode_index,
                    "episode_content_id": snapshot.content_id,
                }
                for snapshot in ordered_snapshots
            ],
        }
    )
    return snapshots, annotation_catalog_sha256, source_content_sha256


def _selected_data_shard_bindings(
    *,
    dataset_root: Path,
    snapshots: Mapping[int, EpisodeSnapshot],
    selected_episode_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Bind a local LeRobot view to the complete bytes of every selected shard.

    Episode/catalog identities intentionally cover only interpreted columns.
    These bindings additionally make any byte-level replacement of a selected
    parquet shard visible before a training loader starts.
    """

    relative_paths = sorted(
        {
            snapshots[int(episode_index)].record.data_file
            for episode_index in selected_episode_ids
        }
    )
    if not relative_paths:
        raise ValueError("Cannot bind an empty selected LeRobot shard set.")
    bindings: list[dict[str, Any]] = []
    for relative_path in relative_paths:
        parsed = Path(relative_path)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != relative_path
            or ".." in parsed.parts
            or "." in parsed.parts
        ):
            raise ValueError(
                "Selected LeRobot data shard path must be a normalized "
                f"relative POSIX path: {relative_path!r}"
            )
        path = (dataset_root / parsed).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(
                f"Selected LeRobot shard escapes the dataset root: {relative_path}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(f"Selected LeRobot data shard is missing: {path}")
        size_bytes = int(path.stat().st_size)
        if size_bytes <= 0:
            raise ValueError(f"Selected LeRobot data shard is empty: {path}")
        bindings.append(
            {
                "path": relative_path,
                "sha256": dataset_view.file_sha256(path),
                "size_bytes": size_bytes,
            }
        )
    return bindings


def _clean_horizon_base_indices(
    valid_state: Sequence[bool],
    *,
    horizon: int,
) -> tuple[int, ...]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if len(valid_state) < horizon:
        return ()
    invalid_prefix = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(np.logical_not(valid_state), dtype=np.int64),
        )
    )
    return tuple(
        base_index
        for base_index in range(len(valid_state) - horizon + 1)
        if invalid_prefix[base_index + horizon] == invalid_prefix[base_index]
    )


def _waterfill_task_row_budgets(
    task_capacities: Mapping[str, int],
    *,
    requested_total: int,
    seed: int,
) -> dict[str, int]:
    if not task_capacities:
        raise ValueError("Task capacities cannot be empty.")
    if requested_total <= 0 or requested_total > sum(task_capacities.values()):
        raise ValueError("Requested RealSource row total is out of range.")
    budgets: dict[str, int] = {}
    active = set(task_capacities)
    remaining = requested_total
    while active:
        share, remainder = divmod(remaining, len(active))
        capped = sorted(
            task
            for task in active
            if int(task_capacities[task]) <= share
        )
        if capped:
            for task in capped:
                budget = int(task_capacities[task])
                budgets[task] = budget
                remaining -= budget
                active.remove(task)
            continue
        ordered = sorted(
            active,
            key=lambda task: hashlib.sha256(
                (
                    f"{REALSOURCE_VIEW_SELECTION_ALGORITHM}\0"
                    f"{seed}\0{task}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        for index, task in enumerate(ordered):
            budgets[task] = share + (1 if index < remainder else 0)
        remaining = 0
        break
    if sum(budgets.values()) != requested_total:
        raise AssertionError("Task water-fill row accounting bug.")
    if any(
        budgets[task] <= 0 or budgets[task] > int(task_capacities[task])
        for task in task_capacities
    ):
        raise AssertionError("Task water-fill produced an invalid budget.")
    return budgets


def _select_task_episode_prefix(
    catalog: RealSourceTaskCatalog,
    *,
    target_rows: int,
    seed: int,
) -> tuple[RealSourceEpisode, ...]:
    ordered = sorted(
        catalog.valid_episodes,
        key=lambda episode: (
            hashlib.sha256(
                (
                    f"{REALSOURCE_VIEW_SELECTION_ALGORITHM}\0{seed}\0"
                    f"{catalog.dataset_id}\0{episode.episode_index}"
                ).encode("utf-8")
            ).hexdigest(),
            episode.episode_index,
        ),
    )
    if target_rows >= catalog.valid_target_row_count:
        return tuple(sorted(ordered, key=lambda episode: episode.episode_index))
    selected: list[RealSourceEpisode] = []
    selected_rows = 0
    for episode in ordered:
        candidate_rows = selected_rows + episode.target_row_count
        if not selected or abs(candidate_rows - target_rows) <= abs(
            selected_rows - target_rows
        ):
            selected.append(episode)
            selected_rows = candidate_rows
        else:
            break
    if not selected:
        raise AssertionError("Every positive task budget must select an episode.")
    return tuple(sorted(selected, key=lambda episode: episode.episode_index))


def _select_realsource_episodes(
    catalogs: Sequence[RealSourceTaskCatalog],
    *,
    fraction: Decimal,
    seed: int,
) -> tuple[dict[str, tuple[RealSourceEpisode, ...]], dict[str, int]]:
    capacities = {
        catalog.dataset_id: catalog.valid_target_row_count
        for catalog in catalogs
    }
    full_rows = sum(capacities.values())
    if fraction == Decimal(1):
        budgets = dict(capacities)
    else:
        requested_total = max(
            len(capacities),
            int(
                (Decimal(full_rows) * fraction).to_integral_value(
                    rounding="ROUND_HALF_UP"
                )
            ),
        )
        budgets = _waterfill_task_row_budgets(
            capacities,
            requested_total=requested_total,
            seed=seed,
        )
    selected = {
        catalog.dataset_id: _select_task_episode_prefix(
            catalog,
            target_rows=budgets[catalog.dataset_id],
            seed=seed,
        )
        for catalog in catalogs
    }
    return selected, budgets


def _realsource_source_content_sha256(
    catalog: RealSourceTaskCatalog,
) -> str:
    return dataset_view.canonical_json_sha256(
        {
            "schema": "realsource-canonical-source-binding-v2",
            "dataset_id": catalog.dataset_id,
            "sid": catalog.sid,
            "revision": catalog.revision,
            "metadata_sha256": catalog.metadata_sha256,
            "annotation_sha256": catalog.annotation_sha256,
            "subtask_segments_sha256": (
                catalog.subtask_segments_sha256
            ),
            "subtask_segments_summary": dict(
                catalog.subtask_segments_summary
            ),
            "alignment": catalog.alignment,
            "quality_value_counts": dict(catalog.quality_value_counts),
            "valid_episodes": [
                {
                    "episode_index": episode.episode_index,
                    "length": episode.length,
                    "target_row_count": episode.target_row_count,
                    "annotation_ordinal": episode.annotation_ordinal,
                    "annotation_episode_index": (
                        episode.annotation_episode_index
                    ),
                    "annotation_sha256": episode.annotation_sha256,
                    "episode_lineage_id": episode.lineage_id,
                    "episode_content_id": episode.content_id,
                }
                for episode in catalog.valid_episodes
            ],
        }
    )


def _realsource_ranges(
    *,
    catalogs: Sequence[RealSourceTaskCatalog],
    selected: Mapping[str, Sequence[RealSourceEpisode]],
    representation_contract_sha256: str,
    adapter_sha256: str,
    horizon: int,
) -> Iterator[dict[str, Any]]:
    for catalog in sorted(catalogs, key=lambda item: item.dataset_id):
        for episode in selected[catalog.dataset_id]:
            base_start = 0
            base_stop = episode.target_row_count
            base_step = 1
            yield {
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
                "base_step": base_step,
                "sample_count": episode.target_row_count,
                "horizon": horizon,
                "target_fps": DEFAULT_TARGET_FPS,
                "source_fps": catalog.fps,
                "end_clamp_policy": "repeat_last",
                "data_file": episode.data_file,
                "adapter_sha256": adapter_sha256,
                "selection_kind": "strict_valid_resampled_dense",
                "annotation_ordinal": episode.annotation_ordinal,
                "annotation_episode_index": (
                    episode.annotation_episode_index
                ),
                "annotation_sha256": episode.annotation_sha256,
                "episode_lineage_id": episode.lineage_id,
                "episode_content_id": episode.content_id,
                "range_id": dataset_view.make_range_id(
                    episode_content_id=episode.content_id,
                    base_start=base_start,
                    base_stop=base_stop,
                    base_step=base_step,
                    horizon=horizon,
                    target_fps=DEFAULT_TARGET_FPS,
                    representation_contract_sha256=(
                        representation_contract_sha256
                    ),
                    end_clamp=True,
                ),
            }


def _bind_realsource_eval_holdout(
    *,
    eval_manifest_path: Path,
    canonical_manifest_sha256: str,
    catalogs: Sequence[RealSourceTaskCatalog],
) -> tuple[
    dict[str, Any],
    frozenset[tuple[str, str, str, str, int]],
    tuple[RealSourceEpisode, ...],
]:
    """Authenticate an eval manifest and resolve its episodes to view IDs."""

    resolved = eval_manifest_path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(
            "RealSource evaluation holdout manifest must be a regular file: "
            f"{resolved}"
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"RealSource evaluation holdout manifest is invalid: {exc}"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("purpose") != "heldout"
    ):
        raise ValueError(
            "RealSource evaluation holdout must be a schema_version=1 "
            "canonical manifest with purpose='heldout'."
        )
    if payload.get("source_manifest_sha256") != canonical_manifest_sha256:
        raise ValueError(
            "RealSource evaluation holdout is not bound to the configured "
            "canonical source manifest."
        )
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError(
            "RealSource evaluation holdout must contain at least one window."
        )
    episode_lookup = {
        (
            catalog.dataset_id,
            catalog.sid,
            catalog.revision,
            episode.data_file,
            episode.episode_index,
        ): episode
        for catalog in catalogs
        for episode in catalog.valid_episodes
    }
    identities: set[tuple[str, str, str, str, int]] = set()
    window_identities: set[tuple[str, str, str, str, int, int]] = set()
    for index, raw_window in enumerate(windows):
        if not isinstance(raw_window, Mapping):
            raise ValueError(
                f"RealSource evaluation window {index} must be an object."
            )
        string_values = tuple(
            raw_window.get(field)
            for field in ("dataset_id", "sid", "revision", "data_file")
        )
        episode_index = raw_window.get("episode_index")
        base_index = raw_window.get("base_index")
        if (
            any(
                not isinstance(value, str) or not value
                for value in string_values
            )
            or isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
            or isinstance(base_index, bool)
            or not isinstance(base_index, int)
            or base_index < 0
        ):
            raise ValueError(
                f"RealSource evaluation window {index} has an invalid identity."
            )
        identity = (*string_values, episode_index)
        window_identity = (*identity, base_index)
        if window_identity in window_identities:
            raise ValueError(
                "RealSource evaluation holdout contains duplicate windows."
            )
        window_identities.add(window_identity)
        identities.add(identity)
    missing = sorted(identities - set(episode_lookup))
    if missing:
        raise ValueError(
            "RealSource evaluation holdout contains episodes outside the "
            f"strict-valid 35-task catalog: {missing[:5]}"
        )
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError(
            "RealSource evaluation holdout lacks its selection contract."
        )
    declared_windows = selection.get("window_count")
    declared_episodes = selection.get("holdout_episode_count")
    if (
        declared_windows != len(window_identities)
        or declared_episodes != len(identities)
    ):
        raise ValueError(
            "RealSource evaluation holdout selection counts do not match its "
            "windows."
        )
    episodes = tuple(
        episode_lookup[identity] for identity in sorted(identities)
    )
    binding = {
        "schema": "realsource-canonical-eval-holdout-binding-v1",
        # Bind content, not a machine-local absolute path, so the frozen view
        # remains portable between the workstation and H100 scratch disk.
        "manifest_filename": resolved.name,
        "manifest_sha256": dataset_view.file_sha256(resolved),
        "source_manifest_sha256": canonical_manifest_sha256,
        "window_count": len(window_identities),
        "episode_count": len(identities),
        "episode_identity_fields": [
            "dataset_id",
            "sid",
            "revision",
            "data_file",
            "episode_index",
        ],
        "episode_identities_sha256": dataset_view.canonical_json_sha256(
            [list(identity) for identity in sorted(identities)]
        ),
    }
    binding["sha256"] = dataset_view.canonical_json_sha256(binding)
    return binding, frozenset(identities), episodes


def build_realsource_canonical_view(
    *,
    canonical_manifest: Path | str = DEFAULT_CANONICAL_MANIFEST,
    adapter_path: Path | str = DEFAULT_CANONICAL_ADAPTER,
    cache_dir: Path | str = DEFAULT_CANONICAL_CACHE,
    eval_holdout_manifest: Path | str | None = None,
    output_manifest: Path | str,
    fraction: str | float | Decimal = "1",
    seed: int = 0,
    representation_contract_sha256: str = DEFAULT_REPRESENTATION_SHA256,
    horizon: int = DEFAULT_HORIZON,
    statistics_population_candidate: bool = False,
    eval_selection_population_candidate: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    summary_only: bool = False,
) -> Any:
    manifest_path = Path(canonical_manifest).expanduser().resolve()
    adapter = Path(adapter_path).expanduser().resolve()
    cache = Path(cache_dir).expanduser().resolve()
    parsed_fraction = _parse_fraction(fraction)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("RealSource selection seed must be an integer.")
    if horizon != DEFAULT_HORIZON:
        raise ValueError("RealSource curriculum contract requires H=50.")
    if statistics_population_candidate and eval_selection_population_candidate:
        raise ValueError(
            "statistics_population_candidate and "
            "eval_selection_population_candidate are mutually exclusive."
        )
    catalogs, manifest_sha256, adapter_sha256 = _load_realsource_catalog(
        canonical_manifest=manifest_path,
        adapter_path=adapter,
        cache_dir=cache,
    )
    selected, budgets = _select_realsource_episodes(
        catalogs, fraction=parsed_fraction, seed=seed
    )
    if (
        eval_holdout_manifest is None
        and not eval_selection_population_candidate
    ):
        raise ValueError(
            "A production RealSource training view requires "
            "eval_holdout_manifest so its frozen population and planned "
            "optimizer steps are holdout-free."
        )
    if (
        eval_holdout_manifest is not None
        and eval_selection_population_candidate
    ):
        raise ValueError(
            "eval_selection_population_candidate is the pre-holdout "
            "bootstrap view and must not receive eval_holdout_manifest."
        )
    if eval_selection_population_candidate:
        holdout_binding = {
            "schema": (
                "realsource-canonical-eval-selection-candidate-binding-v1"
            ),
            "status": "pending_deterministic_eval_selection",
            "source_manifest_sha256": manifest_sha256,
            "episode_identity_fields": [
                "dataset_id",
                "sid",
                "revision",
                "data_file",
                "episode_index",
            ],
        }
        holdout_binding["sha256"] = (
            dataset_view.canonical_json_sha256(holdout_binding)
        )
        heldout_identities = frozenset()
        heldout_episodes = ()
    else:
        assert eval_holdout_manifest is not None
        holdout_binding, heldout_identities, heldout_episodes = (
            _bind_realsource_eval_holdout(
                eval_manifest_path=Path(eval_holdout_manifest),
                canonical_manifest_sha256=manifest_sha256,
                catalogs=catalogs,
            )
        )
    heldout_lineage_ids = frozenset(
        episode.lineage_id for episode in heldout_episodes
    )
    heldout_content_ids = frozenset(
        episode.content_id for episode in heldout_episodes
    )

    def is_heldout_or_copy(
        catalog: RealSourceTaskCatalog,
        episode: RealSourceEpisode,
    ) -> bool:
        return (
            (
                catalog.dataset_id,
                catalog.sid,
                catalog.revision,
                episode.data_file,
                episode.episode_index,
            )
            in heldout_identities
            or episode.lineage_id in heldout_lineage_ids
            or episode.content_id in heldout_content_ids
        )

    selected_before_holdout = selected
    training_selected = {
        catalog.dataset_id: tuple(
            episode
            for episode in selected_before_holdout[catalog.dataset_id]
            if not is_heldout_or_copy(catalog, episode)
        )
        for catalog in catalogs
    }
    if statistics_population_candidate:
        # The statistics view carries the exact heldout episodes in addition
        # to the selected train candidates. This lets the union builder hash
        # projected 18-D content and exclude authenticated holdout keys. A
        # non-heldout copy of holdout content remains excluded.
        heldout_by_dataset: dict[str, list[RealSourceEpisode]] = defaultdict(
            list
        )
        for episode in heldout_episodes:
            heldout_by_dataset[episode.dataset_id].append(episode)
        selected = {}
        for catalog in catalogs:
            by_identity = {
                (
                    episode.data_file,
                    episode.episode_index,
                ): episode
                for episode in selected_before_holdout[catalog.dataset_id]
                if (
                    episode.content_id not in heldout_content_ids
                    and episode.lineage_id not in heldout_lineage_ids
                )
            }
            for episode in heldout_by_dataset[catalog.dataset_id]:
                by_identity[
                    (episode.data_file, episode.episode_index)
                ] = episode
            selected[catalog.dataset_id] = tuple(
                sorted(
                    by_identity.values(),
                    key=lambda item: (
                        item.data_file,
                        item.episode_index,
                    ),
                )
            )
    else:
        selected = training_selected
    selected_holdout_overlap = tuple(
        episode
        for catalog in catalogs
        for episode in selected_before_holdout[catalog.dataset_id]
        if is_heldout_or_copy(catalog, episode)
    )
    if any(
        not training_selected[catalog.dataset_id]
        for catalog in catalogs
    ):
        emptied = [
            catalog.dataset_id
            for catalog in catalogs
            if not training_selected[catalog.dataset_id]
        ]
        raise ValueError(
            "Evaluation holdout removed every selected episode from one or "
            f"more RealSource tasks: {emptied}"
        )
    task_summaries = []
    for catalog in catalogs:
        task_selected = selected[catalog.dataset_id]
        task_summaries.append(
            {
                "dataset_id": catalog.dataset_id,
                "sid": catalog.sid,
                "revision": catalog.revision,
                "catalog_episode_count": catalog.catalog_episode_count,
                "catalog_raw_frame_count": catalog.catalog_raw_frame_count,
                "strict_valid_episode_count": len(catalog.valid_episodes),
                "strict_valid_raw_frame_count": (
                    catalog.valid_raw_frame_count
                ),
                "strict_valid_target_row_count": (
                    catalog.valid_target_row_count
                ),
                "quality_value_counts": dict(
                    catalog.quality_value_counts
                ),
                "row_budget": budgets[catalog.dataset_id],
                "selected_episode_count": len(task_selected),
                "selected_target_row_count": sum(
                    episode.target_row_count for episode in task_selected
                ),
                "selected_episode_indices": [
                    episode.episode_index for episode in task_selected
                ],
                "metadata_sha256": catalog.metadata_sha256,
                "annotation_sha256": catalog.annotation_sha256,
                "subtask_segments_sha256": (
                    catalog.subtask_segments_sha256
                ),
                "subtask_segments_summary": dict(
                    catalog.subtask_segments_summary
                ),
                "alignment": dict(catalog.alignment),
            }
        )
    selected_episode_count = sum(
        len(episodes) for episodes in selected.values()
    )
    selected_row_count = sum(
        episode.target_row_count
        for episodes in selected.values()
        for episode in episodes
    )
    evaluation_holdout = {
        key: value
        for key, value in holdout_binding.items()
        if key != "sha256"
    }
    evaluation_holdout.update(
        {
            "selected_population_overlap_episode_count": len(
                selected_holdout_overlap
            ),
            "selected_population_overlap_target_row_count": sum(
                episode.target_row_count
                for episode in selected_holdout_overlap
            ),
            "copy_detection": [
                "episode_identity",
                "episode_lineage_id",
                "episode_content_id",
            ],
        }
    )
    evaluation_holdout["sha256"] = dataset_view.canonical_json_sha256(
        evaluation_holdout
    )
    summary = {
        "schema": "realsource-frozen-view-selection-summary-v1",
        "fraction": format(parsed_fraction, "f"),
        "seed": seed,
        "algorithm": REALSOURCE_VIEW_SELECTION_ALGORITHM,
        "canonical_manifest": str(manifest_path),
        "canonical_manifest_sha256": manifest_sha256,
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_sha256,
        "full_catalog_episode_count": REALSOURCE_KNOWN_FULL_EPISODES,
        "full_catalog_raw_frame_count": (
            REALSOURCE_KNOWN_FULL_RAW_FRAMES
        ),
        "strict_valid_episode_count": REALSOURCE_KNOWN_VALID_EPISODES,
        "strict_valid_raw_frame_count": (
            REALSOURCE_KNOWN_VALID_RAW_FRAMES
        ),
        "strict_valid_target_row_count": (
            REALSOURCE_KNOWN_VALID_TARGET_ROWS
        ),
        "selected_episode_count": selected_episode_count,
        "selected_target_row_count": selected_row_count,
        "task_count": len(catalogs),
        "tasks": task_summaries,
        "evaluation_holdout": evaluation_holdout,
    }
    summary["selection_sha256"] = dataset_view.canonical_json_sha256(summary)
    if summary_only:
        return summary

    sources = [
        {
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
            "subtask_segments_sha256": (
                catalog.subtask_segments_sha256
            ),
            "subtask_segments_schema": SUBTASK_SEGMENTS_SCHEMA,
            "subtask_segments_summary": dict(
                catalog.subtask_segments_summary
            ),
            "source_content_sha256": (
                _realsource_source_content_sha256(catalog)
            ),
            "episode_metadata_sha256": catalog.metadata_sha256,
            "annotation_alignment": dict(catalog.alignment),
        }
        for catalog in catalogs
    ]
    descriptor = {
        "view_name": (
            (
                "realsource_strict_valid_full_"
                "statistics_population_candidate_v1"
            )
            if (
                statistics_population_candidate
                and parsed_fraction == Decimal(1)
            )
            else (
                "realsource_strict_valid_breadth_fraction_"
                "statistics_population_candidate_v1"
            )
            if statistics_population_candidate
            else (
                "realsource_strict_valid_full_"
                "eval_selection_population_candidate_v1"
            )
            if (
                eval_selection_population_candidate
                and parsed_fraction == Decimal(1)
            )
            else (
                "realsource_strict_valid_breadth_fraction_"
                "eval_selection_population_candidate_v1"
            )
            if eval_selection_population_candidate
            else "realsource_strict_valid_full_exhaustive_v1"
            if parsed_fraction == Decimal(1)
            else "realsource_strict_valid_breadth_fraction_exhaustive_v1"
        ),
        "description": (
            (
                "Non-trainable strict-valid RealSource statistics population "
                "candidate. It includes authenticated heldout episode "
                "references so the union builder can exclude them."
            )
            if statistics_population_candidate
            else (
                "Non-trainable strict-valid RealSource evaluation-selection "
                "candidate. It freezes the deterministic pre-holdout "
                "population used only to choose the eval episodes."
            )
            if eval_selection_population_candidate
            else (
                "Immutable strict-valid RealSource view. Each logical epoch "
                "globally exhausts every selected row exactly once; the "
                "configured fraction defines a frozen whole-episode "
                "population, never a partial epoch."
            )
        ),
        "generator": {
            "schema_version": 1,
            "name": "build_realman_dataset_views.py",
            "source_sha256": dataset_view.file_sha256(Path(__file__)),
        },
        "purpose": (
            dataset_view.STATISTICS_POPULATION_CANDIDATE_PURPOSE
            if statistics_population_candidate
            else dataset_view.EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE
            if eval_selection_population_candidate
            else "realsource_pretraining"
        ),
        "sources": sources,
        "representation": {
            "contract_sha256": representation_contract_sha256,
            "state_dim": 18,
            "action_dim": 18,
            "horizon": horizon,
            "target_fps": DEFAULT_TARGET_FPS,
            "action_type": "joint_delta_gripper_absolute",
            "normalization": "q01_q99_unclipped",
        },
        "selection": {
            **summary,
            "quality_predicate": {
                "column": "quality_assessments.overall_valid",
                "comparison": "exact_string_equality",
                "value": "VALID",
            },
            "resampling": {
                "algorithm": REALSOURCE_RESAMPLING_ALGORITHM,
                "source_fps": 30,
                "target_fps": DEFAULT_TARGET_FPS,
                "source_index_formula": "(3 * target_index + 1) // 2",
                "target_row_count_formula": "(2 * source_length + 1) // 3",
            },
            "episode_selection": "whole_episodes",
            "task_coverage": "all_35_tasks",
            "statistics_population_candidate": bool(
                statistics_population_candidate
            ),
            "eval_selection_population_candidate": bool(
                eval_selection_population_candidate
            ),
            "authenticated_holdout_episode_indices_in_ledger": (
                sorted(
                    {
                        episode.episode_index
                        for episode in heldout_episodes
                    }
                )
                if statistics_population_candidate
                else []
            ),
        },
        "holdout_exclusions": dataset_view.make_holdout_exclusions(
            source_id=(
                "pending_canonical_eval_manifest"
                if eval_selection_population_candidate
                else (
                    "canonical_eval_manifest:"
                    f"{holdout_binding['manifest_sha256']}"
                )
            ),
            episode_indices=tuple(
                episode.episode_index for episode in heldout_episodes
            ),
            lineage_ids=tuple(
                episode.lineage_id for episode in heldout_episodes
            ),
            content_ids=tuple(
                episode.content_id for episode in heldout_episodes
            ),
        ),
        "epoch_contract": {
            "mode": dataset_view.ALL_EXHAUSTIVE_MODE,
            "epoch_passes": 1,
            "replacement": False,
            "drop_last": False,
            "shuffle": "deterministic_bijection_per_epoch",
            "global_membership": (
                "every_selected_ledger_row_exactly_once_before_ddp_padding"
            ),
            "ddp_tail": "duplicated_padding_reported_separately",
            "fraction_semantics": (
                "immutable_population_fraction_not_fractional_epoch"
            ),
        },
        "usage_contract": {
            "training_allowed": not (
                statistics_population_candidate
                or eval_selection_population_candidate
            ),
            "eval_manifest_generation": bool(
                eval_selection_population_candidate
            ),
            "statistics_accumulation": (
                "union_builder_must_exclude_authenticated_holdout_keys"
                if statistics_population_candidate
                else "forbidden_eval_selection_bootstrap_only"
                if eval_selection_population_candidate
                else "not_a_statistics_population_source"
            ),
        },
        "content_identity": {
            "contract": REALSOURCE_PROVENANCE_CONTENT_CONTRACT,
            "strength": "provenance_bound",
            "note": (
                "The frozen view identity binds metadata, aligned annotations, "
                "and the authenticated source-frame subtask segment sidecar. "
                "The union-statistics reader separately hashes projected numeric "
                "episode payloads for cross-source duplicate detection."
            ),
        },
    }
    content_ids = [
        episode.content_id
        for episodes in selected.values()
        for episode in episodes
    ]
    if len(content_ids) != len(set(content_ids)):
        raise ValueError("RealSource provenance episode identities collide.")
    return dataset_view.write_frozen_range_view(
        output_manifest,
        descriptor=descriptor,
        ranges=_realsource_ranges(
            catalogs=catalogs,
            selected=selected,
            representation_contract_sha256=(
                representation_contract_sha256
            ),
            adapter_sha256=adapter_sha256,
            horizon=horizon,
        ),
        overwrite=overwrite,
        dry_run=dry_run,
    )


def _selected_base_indices(
    snapshot: EpisodeSnapshot,
    *,
    selection_kind: str,
    horizon: int,
) -> Iterable[int]:
    if selection_kind == "dense_all_frames":
        return range(snapshot.record.length)
    if selection_kind == "clean_full_horizon":
        return _clean_horizon_base_indices(
            snapshot.valid_state, horizon=horizon
        )
    raise ValueError(f"Unsupported selection_kind {selection_kind!r}.")


def _subtask_prompt_coverage(
    *,
    snapshots: Mapping[int, EpisodeSnapshot],
    selected_episode_ids: Sequence[int],
    catalog: Mapping[int, Mapping[str, Any]],
    selection_kind: str,
    horizon: int,
    catalog_sha256: str,
) -> dict[str, Any]:
    row_counts: dict[int, int] = defaultdict(int)
    episode_sets: dict[int, set[int]] = defaultdict(set)
    total = 0
    for episode_index in selected_episode_ids:
        snapshot = snapshots[int(episode_index)]
        for base_index in _selected_base_indices(
            snapshot,
            selection_kind=selection_kind,
            horizon=horizon,
        ):
            subtask_index = snapshot.subtask_index[base_index]
            if subtask_index not in catalog:
                raise ValueError(
                    f"Episode {episode_index} frame {base_index} references "
                    f"unknown subtask_index {subtask_index}."
                )
            row_counts[subtask_index] += 1
            episode_sets[subtask_index].add(int(episode_index))
            total += 1
    useful = sum(
        count
        for index, count in row_counts.items()
        if bool(catalog[index]["useful_prompt"])
    )
    if total <= 0:
        raise ValueError("Frozen local view contains no selected rows.")
    if useful <= 0:
        raise ValueError(
            "Frozen local view has no selected rows with a useful subtask "
            "prompt; refusing to bind an effectively unlabeled source."
        )
    return {
        "schema": "realman-subtask-prompt-coverage-v1",
        "subtask_catalog_sha256": catalog_sha256,
        "selected_row_count": total,
        "useful_prompt_row_count": useful,
        "unlabeled_or_nonuseful_row_count": total - useful,
        "useful_prompt_fraction": useful / total,
        "rows_by_subtask": [
            {
                **dict(catalog[index]),
                "selected_row_count": row_counts.get(index, 0),
                "selected_episode_count": len(episode_sets.get(index, set())),
            }
            for index in sorted(catalog)
        ],
    }


def _schema_binding(path: Path, *, dataset_root: Path) -> dict[str, Any]:
    schema = pq.read_schema(path)
    fields = [
        {"name": field.name, "type": str(field.type)}
        for field in schema
    ]
    return {
        "path": path.relative_to(dataset_root).as_posix(),
        "schema_sha256": dataset_view.canonical_json_sha256(
            {"fields": fields}
        ),
        "fields": fields,
    }


def _normalize_action_owner(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return (
        str(value)
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _strict_action_valid_flag(
    value: Any,
    *,
    episode_index: int,
    frame_index: int,
) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"Episode {episode_index} frame {frame_index} has a "
                "non-finite valid_action label."
            )
        if value in (0, 1):
            return bool(value)
    raise ValueError(
        f"Episode {episode_index} frame {frame_index} has invalid "
        f"valid_action={value!r}; expected exact 0/1."
    )


def _recovery_anchor_mask_coverage(
    *,
    snapshots: Mapping[int, EpisodeSnapshot],
    selected_episode_ids: Sequence[int],
    selection_kind: str,
    horizon: int,
    invalid_run_length: int,
) -> dict[str, Any]:
    """Report exact action-mask coverage for 0→1 recovery anchors.

    The computation mirrors ``action_validity_prefix_mask`` without importing
    the training package.  Episode-end padding is always masked.  A window
    whose anchor is the first valid action after a mistake therefore retains
    at least its first recovery action, while an earlier pre-mistake window
    may mask a causally unobservable recovery suffix.
    """

    per_anchor: list[dict[str, int]] = []
    for episode_index in sorted(int(value) for value in selected_episode_ids):
        snapshot = snapshots[episode_index]
        labels = np.asarray(snapshot.valid_state, dtype=bool)
        eligible = set(
            int(value)
            for value in _selected_base_indices(
                snapshot,
                selection_kind=selection_kind,
                horizon=horizon,
            )
        )
        for base_index in range(1, len(labels)):
            if (
                base_index not in eligible
                or not bool(labels[base_index])
                or bool(labels[base_index - 1])
            ):
                continue
            available = min(horizon, len(labels) - base_index)
            valid = np.zeros(horizon, dtype=bool)
            valid[:available] = labels[
                base_index : base_index + available
            ]
            invalid = np.logical_not(valid)
            if invalid_run_length <= invalid.size:
                for start in range(
                    invalid.size - invalid_run_length + 1
                ):
                    if bool(
                        invalid[start : start + invalid_run_length].all()
                    ):
                        valid[start:] = False
                        break
            supervised = int(valid.sum())
            per_anchor.append(
                {
                    "episode_index": episode_index,
                    "base_index": base_index,
                    "supervised_action_timesteps": supervised,
                    "supervised_action_elements": supervised * 18,
                }
            )
    timestep_counts = [
        record["supervised_action_timesteps"] for record in per_anchor
    ]
    return {
        "definition": "first_valid_state_1_frame_after_valid_state_0",
        "mask_policy": (
            "valid_flags_with_padding_masked_then_suffix_masked_at_first_"
            "sustained_invalid_run"
        ),
        "invalid_run_length": invalid_run_length,
        "action_dim": 18,
        "recovery_anchor_window_count": len(per_anchor),
        "recovery_anchor_with_nonzero_action_mask_count": sum(
            value > 0 for value in timestep_counts
        ),
        "supervised_action_timestep_count": sum(timestep_counts),
        "supervised_action_element_count": sum(timestep_counts) * 18,
        "minimum_supervised_action_timesteps_per_anchor": (
            min(timestep_counts) if timestep_counts else 0
        ),
        "maximum_supervised_action_timesteps_per_anchor": (
            max(timestep_counts) if timestep_counts else 0
        ),
        "anchors": per_anchor,
    }


def _load_action_label_semantics_contract(
    path: Path,
    *,
    source_id: str,
    catalog_sha256: str,
    annotation_sha256: str,
    source_content_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "parse_error": str(exc),
            },
            ["the action-label semantics contract is not valid JSON"],
        )
    if not isinstance(payload, Mapping):
        return (
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            ["the action-label semantics contract root is not an object"],
        )

    expected_dataset = {
        "source_id": source_id,
        "catalog_sha256": catalog_sha256,
        "annotation_sha256": annotation_sha256,
        "source_content_sha256": source_content_sha256,
    }
    expected_semantics = {
        "validity_column": "valid_state",
        "mistake_value": 0,
        "mistake_meaning": "mistake_action_do_not_supervise",
        "supervised_value": 1,
        "supervised_meaning": "expert_or_recovery_action_supervise",
        "recovery_anchor_definition": (
            "first_valid_frame_after_invalid_frame"
        ),
    }
    expected_masking = {
        "policy": (
            "chunk_prefix_until_first_sustained_invalid_or_padding"
        ),
        "invalid_run_length": DEFAULT_ACTION_VALIDITY_INVALID_RUN_LENGTH,
        "recovery_windows": (
            "windows_anchored_at_valid_recovery_frames_are_supervised"
        ),
    }
    if payload.get("schema") != ACTION_LABEL_SEMANTICS_SCHEMA:
        errors.append(
            f"schema must be {ACTION_LABEL_SEMANTICS_SCHEMA!r}"
        )
    if payload.get("status") != "verified":
        errors.append("status must be 'verified'")
    if payload.get("dataset") != expected_dataset:
        errors.append(
            "dataset identity does not match the audited view scope"
        )
    if payload.get("semantics") != expected_semantics:
        errors.append(
            "semantics must exactly bind valid_state 0 to mistake actions "
            "and valid_state 1 to expert/recovery supervision"
        )
    if payload.get("masking") != expected_masking:
        errors.append(
            "masking must bind the production sustained-invalid/padding "
            "policy and recovery-anchor behavior"
        )
    review = payload.get("review")
    if not isinstance(review, Mapping):
        errors.append("review must be an object")
    else:
        for key in ("reviewer", "reviewed_at_utc", "evidence"):
            value = review.get(key)
            if value is None or value == "" or value == []:
                errors.append(f"review.{key} must be non-empty")
    return (
        {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "payload_sha256": dataset_view.canonical_json_sha256(payload),
        },
        errors,
    )


def _action_supervision_audit(
    *,
    dataset_root: Path,
    catalog: Mapping[int, EpisodeCatalogRecord],
    catalog_sha256: str,
    annotation_sha256: str,
    source_content_sha256: str,
    snapshots: Mapping[int, EpisodeSnapshot],
    selected_episode_ids: Sequence[int],
    selection_kind: str,
    horizon: int,
    action_label_semantics_contract: Path | str | None,
) -> dict[str, Any]:
    """Audit action-label ownership without inventing semantics.

    ``valid_state`` is ordinarily state-quality metadata and is never treated
    as action validity merely because the column exists.  A dataset-specific,
    SHA-bound reviewed contract may explicitly establish the Magna convention
    that 0 is a mistake action and 1 is an expert/recovery action.
    """

    source_id = dataset_root.name
    selected = tuple(sorted(int(value) for value in selected_episode_ids))
    relative_files = sorted(
        {catalog[episode_index].data_file for episode_index in selected}
    )
    schema_bindings = [
        _schema_binding(
            dataset_root / relative_path,
            dataset_root=dataset_root,
        )
        for relative_path in relative_files
    ]
    fields_by_file = {
        item["path"]: {field["name"] for field in item["fields"]}
        for item in schema_bindings
    }
    valid_action_everywhere = all(
        EXPLICIT_ACTION_VALIDITY_COLUMN in fields
        for fields in fields_by_file.values()
    )
    owner_column = next(
        (
            candidate
            for candidate in EXPLICIT_ACTION_OWNER_COLUMNS
            if all(
                candidate in fields for fields in fields_by_file.values()
            )
        ),
        None,
    )
    missing_columns = {
        relative_path: sorted(
            {
                *(
                    ()
                    if EXPLICIT_ACTION_VALIDITY_COLUMN in fields
                    else (EXPLICIT_ACTION_VALIDITY_COLUMN,)
                ),
                *(
                    ()
                    if any(
                        candidate in fields
                        for candidate in EXPLICIT_ACTION_OWNER_COLUMNS
                    )
                    else ("action_owner_or_action_source",)
                ),
            }
        )
        for relative_path, fields in fields_by_file.items()
    }
    missing_columns = {
        path: values for path, values in missing_columns.items() if values
    }
    recovery_coverage = _recovery_anchor_mask_coverage(
        snapshots=snapshots,
        selected_episode_ids=selected,
        selection_kind=selection_kind,
        horizon=horizon,
        invalid_run_length=DEFAULT_ACTION_VALIDITY_INVALID_RUN_LENGTH,
    )
    base = {
        "schema": ACTION_SUPERVISION_AUDIT_SCHEMA,
        "status": "unverified",
        "verification_mode": None,
        "valid_state_is_not_generic_action_ownership": True,
        "note": (
            "valid_state alone is insufficient action-ownership provenance; "
            "it is accepted only through a reviewed dataset-specific "
            "semantics contract bound below."
        ),
        "dataset_binding": {
            "source_id": source_id,
            "catalog_sha256": catalog_sha256,
            "annotation_sha256": annotation_sha256,
            "source_content_sha256": source_content_sha256,
            "selected_episode_indices_sha256": (
                dataset_view.canonical_json_sha256(list(selected))
            ),
        },
        "column_provenance": {
            "schema_bindings": schema_bindings,
            "explicit_valid_action_column": (
                EXPLICIT_ACTION_VALIDITY_COLUMN
                if valid_action_everywhere
                else None
            ),
            "explicit_action_owner_or_source_column": owner_column,
            "missing_explicit_columns_by_shard": missing_columns,
        },
        "recovery_from_invalid_state_supervision": {
            "status": "unverified",
            **recovery_coverage,
        },
        "reasons": [],
    }

    if valid_action_everywhere and owner_column is not None:
        by_file: dict[str, list[int]] = defaultdict(list)
        for episode_index in selected:
            by_file[catalog[episode_index].data_file].append(episode_index)
        supervised_rows = 0
        expert_supervised_rows = 0
        unknown_supervised_rows = 0
        explicit_recovery_rows = 0
        digest = hashlib.sha256()
        for relative_path in sorted(by_file):
            data = pq.read_table(
                dataset_root / relative_path,
                columns=[
                    "episode_index",
                    "frame_index",
                    EXPLICIT_ACTION_VALIDITY_COLUMN,
                    owner_column,
                ],
                filters=[
                    ("episode_index", "in", by_file[relative_path])
                ],
            ).to_pandas()
            for episode_index in sorted(by_file[relative_path]):
                frame = data.loc[
                    data["episode_index"] == episode_index
                ].sort_values("frame_index")
                by_index = frame.set_index("frame_index")
                snapshot = snapshots[episode_index]
                for base_index in _selected_base_indices(
                    snapshot,
                    selection_kind=selection_kind,
                    horizon=horizon,
                ):
                    row = by_index.loc[int(base_index)]
                    is_valid = _strict_action_valid_flag(
                        row[EXPLICIT_ACTION_VALIDITY_COLUMN],
                        episode_index=episode_index,
                        frame_index=int(base_index),
                    )
                    owner = _normalize_action_owner(row[owner_column])
                    digest.update(
                        dataset_view.canonical_json_bytes(
                            [
                                episode_index,
                                int(base_index),
                                is_valid,
                                owner,
                            ]
                        )
                        + b"\n"
                    )
                    if not is_valid:
                        continue
                    supervised_rows += 1
                    if owner in EXPERT_ACTION_OWNER_VALUES:
                        expert_supervised_rows += 1
                        if not snapshot.valid_state[int(base_index)]:
                            explicit_recovery_rows += 1
                    else:
                        unknown_supervised_rows += 1
        base["explicit_action_labels"] = {
            "validity_column": EXPLICIT_ACTION_VALIDITY_COLUMN,
            "owner_or_source_column": owner_column,
            "expert_owner_vocabulary": sorted(
                EXPERT_ACTION_OWNER_VALUES
            ),
            "audited_values_sha256": digest.hexdigest(),
            "supervised_row_count": supervised_rows,
            "expert_supervised_row_count": expert_supervised_rows,
            "unknown_owner_supervised_row_count": (
                unknown_supervised_rows
            ),
            "invalid_state_with_explicit_expert_action_row_count": (
                explicit_recovery_rows
            ),
        }
        if (
            supervised_rows > 0
            and unknown_supervised_rows == 0
            and explicit_recovery_rows > 0
        ):
            base["status"] = "verified"
            base["verification_mode"] = (
                "explicit_valid_action_and_expert_owner"
            )
            base["recovery_from_invalid_state_supervision"][
                "status"
            ] = "verified"
            base["recovery_from_invalid_state_supervision"][
                "explicit_invalid_state_expert_action_row_count"
            ] = explicit_recovery_rows
            return base
        base["reasons"].append(
            "explicit action labels do not prove at least one "
            "invalid-state expert recovery action with no unknown owner"
        )

    if action_label_semantics_contract is not None:
        contract_path = (
            Path(action_label_semantics_contract).expanduser().resolve()
        )
        if not contract_path.is_file() or contract_path.is_symlink():
            raise FileNotFoundError(
                "Action-label semantics contract must be a regular file: "
                f"{contract_path}"
            )
        contract_binding, errors = _load_action_label_semantics_contract(
            contract_path,
            source_id=source_id,
            catalog_sha256=catalog_sha256,
            annotation_sha256=annotation_sha256,
            source_content_sha256=source_content_sha256,
        )
        base["action_label_semantics_contract"] = contract_binding
        base["reasons"].extend(errors)
        positive_rows = sum(
            bool(snapshots[episode_index].valid_state[base_index])
            for episode_index in selected
            for base_index in _selected_base_indices(
                snapshots[episode_index],
                selection_kind=selection_kind,
                horizon=horizon,
            )
        )
        selected_rows = sum(
            1
            for episode_index in selected
            for _ in _selected_base_indices(
                snapshots[episode_index],
                selection_kind=selection_kind,
                horizon=horizon,
            )
        )
        base["dataset_specific_valid_state_action_semantics"] = {
            "validity_column": "valid_state",
            "mistake_action_row_count": selected_rows - positive_rows,
            "expert_or_recovery_action_row_count": positive_rows,
            "audited_values_sha256": annotation_sha256,
        }
        recovery_count = recovery_coverage[
            "recovery_anchor_window_count"
        ]
        recovery_nonzero = recovery_coverage[
            "recovery_anchor_with_nonzero_action_mask_count"
        ]
        if (
            not errors
            and positive_rows > 0
            and selected_rows - positive_rows > 0
            and recovery_count > 0
            and recovery_nonzero == recovery_count
        ):
            base["status"] = "verified"
            base["verification_mode"] = (
                "reviewed_valid_state_action_semantics_contract"
            )
            base["recovery_from_invalid_state_supervision"][
                "status"
            ] = "verified"
            base["reasons"] = []
            return base
        if not errors:
            base["reasons"].append(
                "the selected view does not contain both mistake actions "
                "and nonzero-masked recovery-anchor windows"
            )

    if action_label_semantics_contract is None:
        base["reasons"].append(
            "no reviewed SHA-bound action-label semantics contract was "
            "provided"
        )
    if not valid_action_everywhere or owner_column is None:
        base["reasons"].append(
            "explicit valid_action plus action_owner/action_source columns "
            "are not present in every selected shard"
        )
    return base


def _rows_for_snapshots(
    *,
    snapshots: Mapping[int, EpisodeSnapshot],
    selected_episode_ids: Sequence[int],
    source_id: str,
    representation_contract_sha256: str,
    horizon: int,
    target_fps: int,
    selection_kind: str,
) -> Iterator[dict[str, Any]]:
    for episode_index in sorted(int(value) for value in selected_episode_ids):
        snapshot = snapshots[episode_index]
        base_indices = _selected_base_indices(
            snapshot,
            selection_kind=selection_kind,
            horizon=horizon,
        )
        for base_index in base_indices:
            end_index = min(
                base_index + horizon - 1, snapshot.record.length - 1
            )
            yield {
                "backend": "lerobot",
                "source_id": source_id,
                "episode_index": episode_index,
                "episode_length": snapshot.record.length,
                "base_index": base_index,
                "end_index": end_index,
                "horizon": horizon,
                "target_fps": target_fps,
                "end_clamp_policy": "repeat_last",
                "end_clamped": end_index < base_index + horizon - 1,
                "data_file": snapshot.record.data_file,
                "selection_kind": selection_kind,
                "episode_lineage_id": snapshot.lineage_id,
                "episode_content_id": snapshot.content_id,
                "sample_id": dataset_view.make_sample_id(
                    episode_content_id=snapshot.content_id,
                    base_index=base_index,
                    horizon=horizon,
                    target_fps=target_fps,
                    representation_contract_sha256=(
                        representation_contract_sha256
                    ),
                    end_clamp=True,
                ),
            }


def _build_local_view(
    *,
    dataset_root: Path | str,
    output_manifest: Path | str,
    view_name: str,
    purpose: str,
    candidate_episode_ids: Sequence[int] | None,
    holdout_episode_ids: Sequence[int] | None,
    eval_holdout_manifest: Path | str | None,
    selection_kind: str,
    selection_details: Mapping[str, Any],
    representation_contract_sha256: str = DEFAULT_REPRESENTATION_SHA256,
    horizon: int = DEFAULT_HORIZON,
    target_fps: int = DEFAULT_TARGET_FPS,
    include_action_supervision_audit: bool = False,
    action_label_semantics_contract: Path | str | None = None,
    statistics_population_candidate: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Any:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if target_fps != DEFAULT_TARGET_FPS:
        raise ValueError(
            "The local RealMan generators currently require exact 20 Hz data."
        )
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    catalog, binding = _load_episode_catalog(root)
    (
        subtask_catalog,
        subtasks_file_sha256,
        subtask_catalog_sha256,
    ) = _load_subtask_catalog(root)
    source_id = root.name
    candidates = (
        tuple(sorted(catalog))
        if candidate_episode_ids is None
        else tuple(sorted(set(int(value) for value in candidate_episode_ids)))
    )
    eval_holdout_binding: dict[str, Any] | None = None
    if eval_holdout_manifest is not None:
        derived_holdout, eval_holdout_binding = _bind_local_eval_holdout(
            manifest_path=eval_holdout_manifest,
            dataset_root=root,
            catalog=catalog,
            catalog_binding=binding,
        )
        if holdout_episode_ids is not None:
            configured_holdout = tuple(
                sorted(set(int(value) for value in holdout_episode_ids))
            )
            if configured_holdout != derived_holdout:
                raise ValueError(
                    "Explicit holdout_episode_ids do not match the "
                    "authenticated eval_holdout_manifest."
                )
        holdout = derived_holdout
    else:
        if holdout_episode_ids is None:
            raise ValueError(
                "A local frozen view requires eval_holdout_manifest or "
                "explicit holdout_episode_ids."
            )
        holdout = tuple(
            sorted(set(int(value) for value in holdout_episode_ids))
        )
    if statistics_population_candidate and not holdout:
        raise ValueError(
            "A statistics population candidate requires at least one "
            "authenticated holdout episode."
        )
    missing_holdout = sorted(set(holdout) - set(catalog))
    if missing_holdout:
        raise ValueError(
            f"Holdout episode IDs are absent from the dataset: {missing_holdout}."
        )
    scope = tuple(sorted(set(candidates) | set(holdout)))
    snapshots, annotation_sha256, source_content_sha256 = (
        _load_episode_snapshots(
            dataset_root=root,
            source_id=source_id,
            catalog=catalog,
            catalog_sha256=binding["catalog_sha256"],
            episode_ids=scope,
        )
    )
    holdout_snapshots = [snapshots[value] for value in holdout]
    holdout_contents = {snapshot.content_id for snapshot in holdout_snapshots}
    training_selected = [
        episode_index
        for episode_index in candidates
        if episode_index not in set(holdout)
        and snapshots[episode_index].content_id not in holdout_contents
    ]
    copied_holdout_exclusions = sorted(
        episode_index
        for episode_index in candidates
        if episode_index not in set(holdout)
        and snapshots[episode_index].content_id in holdout_contents
    )
    if statistics_population_candidate:
        # A statistics-candidate view is intentionally broader than a
        # training view: it carries the exact held-out episode references so
        # the union-statistics builder can authenticate their projected
        # content and exclude their frames.  Non-heldout copies of holdout
        # content remain absent, matching the train-view leakage contract.
        selected = [
            episode_index
            for episode_index in scope
            if episode_index in set(holdout)
            or snapshots[episode_index].content_id not in holdout_contents
        ]
    else:
        selected = training_selected
    if not training_selected:
        raise ValueError(
            "View selection contains no holdout-free training candidates."
        )
    unknown_subtasks = sorted(
        {
            index
            for episode_index in scope
            for index in snapshots[episode_index].subtask_index
            if index not in subtask_catalog
        }
    )
    if unknown_subtasks:
        raise ValueError(
            "Frame annotations reference subtask indices absent from "
            f"meta/subtasks.parquet: {unknown_subtasks}."
        )
    subtask_coverage = _subtask_prompt_coverage(
        snapshots=snapshots,
        selected_episode_ids=selected,
        catalog=subtask_catalog,
        selection_kind=selection_kind,
        horizon=horizon,
        catalog_sha256=subtask_catalog_sha256,
    )
    action_supervision_audit = (
        _action_supervision_audit(
            dataset_root=root,
            catalog=catalog,
            catalog_sha256=binding["catalog_sha256"],
            annotation_sha256=annotation_sha256,
            source_content_sha256=source_content_sha256,
            snapshots=snapshots,
            selected_episode_ids=selected,
            selection_kind=selection_kind,
            horizon=horizon,
            action_label_semantics_contract=(
                action_label_semantics_contract
            ),
        )
        if include_action_supervision_audit
        else None
    )
    selected_data_shards = _selected_data_shard_bindings(
        dataset_root=root,
        snapshots=snapshots,
        selected_episode_ids=selected,
    )
    holdout_exclusions = dataset_view.make_holdout_exclusions(
        source_id=source_id,
        episode_indices=holdout,
        lineage_ids=[
            snapshot.lineage_id for snapshot in holdout_snapshots
        ],
        content_ids=[
            snapshot.content_id for snapshot in holdout_snapshots
        ],
    )
    descriptor_purpose = (
        dataset_view.STATISTICS_POPULATION_CANDIDATE_PURPOSE
        if statistics_population_candidate
        else purpose
    )
    descriptor = {
        "view_name": (
            f"{view_name}_statistics_population_candidate"
            if statistics_population_candidate
            else view_name
        ),
        "description": (
            (
                f"Non-trainable statistics population candidate for "
                f"{source_id}; includes authenticated holdout episode "
                "references so the union builder can exclude them."
            )
            if statistics_population_candidate
            else (
                f"Frozen exhaustive {purpose} view for {source_id}; "
                "no sampling weights or replacement."
            )
        ),
        "generator": {
            "schema_version": 1,
            "name": "build_realman_dataset_views.py",
            "source_sha256": dataset_view.file_sha256(Path(__file__)),
        },
        "purpose": descriptor_purpose,
        "sources": [
            {
                "source_id": source_id,
                "backend": "lerobot",
                "dataset_name": source_id,
                "dataset_root_hint": str(root),
                "lerobot_version": "v3.0",
                "fps": DEFAULT_TARGET_FPS,
                "info_sha256": binding["info_sha256"],
                "catalog_sha256": binding["catalog_sha256"],
                "annotation_sha256": annotation_sha256,
                "source_content_sha256": source_content_sha256,
                "subtasks_file_sha256": subtasks_file_sha256,
                "subtask_catalog_sha256": subtask_catalog_sha256,
                **(
                    {
                        "episode_split_manifest_sha256": (
                            eval_holdout_binding["manifest_sha256"]
                        )
                    }
                    if eval_holdout_binding is not None
                    else {}
                ),
                "full_episode_count": binding["episode_count"],
                "full_frame_count": binding["frame_count"],
                "annotation_scope_episode_count": len(scope),
                "content_columns": list(CONTENT_COLUMNS),
                "annotation_columns": list(ANNOTATION_COLUMNS),
                "selected_data_shards": selected_data_shards,
            }
        ],
        "representation": {
            "contract_sha256": representation_contract_sha256,
            "state_dim": 18,
            "action_dim": 18,
            "horizon": horizon,
            "target_fps": target_fps,
            "action_type": "joint_delta_gripper_absolute",
            "normalization": "q01_q99_unclipped",
        },
        "selection": {
            **dict(selection_details),
            "selection_kind": selection_kind,
            "candidate_episode_count": len(candidates),
            "selected_episode_count": len(selected),
            "selected_episode_indices": selected,
            "statistics_population_candidate": bool(
                statistics_population_candidate
            ),
            "authenticated_holdout_episode_indices_in_ledger": (
                list(holdout) if statistics_population_candidate else []
            ),
            "content_copy_excluded_episode_indices": (
                copied_holdout_exclusions
            ),
            "subtask_prompt_coverage": subtask_coverage,
            **(
                {"evaluation_holdout": eval_holdout_binding}
                if eval_holdout_binding is not None
                else {}
            ),
        },
        **(
            {"action_supervision_audit": action_supervision_audit}
            if action_supervision_audit is not None
            else {}
        ),
        "holdout_exclusions": holdout_exclusions,
        "usage_contract": {
            "training_allowed": not statistics_population_candidate,
            "statistics_accumulation": (
                "union_builder_must_exclude_authenticated_holdout_keys"
                if statistics_population_candidate
                else "not_a_statistics_population_source"
            ),
        },
        "epoch_contract": {
            "mode": dataset_view.ALL_EXHAUSTIVE_MODE,
            "epoch_passes": 1,
            "replacement": False,
            "drop_last": False,
            "shuffle": "deterministic_bijection_per_epoch",
            "ddp_tail": "duplicated_padding_reported_separately",
        },
    }
    rows = _rows_for_snapshots(
        snapshots=snapshots,
        selected_episode_ids=selected,
        source_id=source_id,
        representation_contract_sha256=representation_contract_sha256,
        horizon=horizon,
        target_fps=target_fps,
        selection_kind=selection_kind,
    )
    return dataset_view.write_frozen_view(
        output_manifest,
        descriptor=descriptor,
        rows=rows,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def build_intervention_incremental_view(
    *,
    dataset_root: Path | str = DEFAULT_INTERVENTION_ROOT,
    output_manifest: Path | str,
    first_episode: int = DEFAULT_INCREMENTAL_FIRST_EPISODE,
    last_episode: int = DEFAULT_INCREMENTAL_LAST_EPISODE,
    holdout_episode_ids: Sequence[int] | None = None,
    eval_holdout_manifest: Path | str | None = None,
    representation_contract_sha256: str = DEFAULT_REPRESENTATION_SHA256,
    horizon: int = DEFAULT_HORIZON,
    action_label_semantics_contract: Path | str | None = None,
    statistics_population_candidate: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Any:
    if first_episode < 0 or last_episode < first_episode:
        raise ValueError("Invalid incremental episode range.")
    if holdout_episode_ids is None and eval_holdout_manifest is None:
        # Python-fixture/backward compatibility only. The production CLI
        # requires an authenticated split manifest.
        holdout_episode_ids = DEFAULT_INCREMENTAL_HOLDOUT
    return _build_local_view(
        dataset_root=dataset_root,
        output_manifest=output_manifest,
        view_name="magna_intervention_incremental_exhaustive_v1",
        purpose="intervention_incremental",
        candidate_episode_ids=tuple(range(first_episode, last_episode + 1)),
        holdout_episode_ids=holdout_episode_ids,
        eval_holdout_manifest=eval_holdout_manifest,
        selection_kind="dense_all_frames",
        selection_details={
            "algorithm": "inclusive_episode_range_minus_dual_identity_holdout_v1",
            "first_episode": first_episode,
            "last_episode": last_episode,
            "embedded_realsource_episode_range": [0, first_episode - 1],
        },
        representation_contract_sha256=representation_contract_sha256,
        horizon=horizon,
        include_action_supervision_audit=True,
        action_label_semantics_contract=action_label_semantics_contract,
        statistics_population_candidate=statistics_population_candidate,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def build_hq_clean_h50_view(
    *,
    dataset_root: Path | str = DEFAULT_HQ_ROOT,
    output_manifest: Path | str,
    holdout_episode_ids: Sequence[int] | None = None,
    eval_holdout_manifest: Path | str | None = None,
    representation_contract_sha256: str = DEFAULT_REPRESENTATION_SHA256,
    horizon: int = DEFAULT_HORIZON,
    statistics_population_candidate: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Any:
    if holdout_episode_ids is None and eval_holdout_manifest is None:
        # Python-fixture/backward compatibility only. The production CLI
        # requires an authenticated split manifest.
        holdout_episode_ids = DEFAULT_HQ_HOLDOUT
    return _build_local_view(
        dataset_root=dataset_root,
        output_manifest=output_manifest,
        view_name="magna_hq_clean_h50_exhaustive_v1",
        purpose="hq_clean_h50",
        candidate_episode_ids=None,
        holdout_episode_ids=holdout_episode_ids,
        eval_holdout_manifest=eval_holdout_manifest,
        selection_kind="clean_full_horizon",
        selection_details={
            "algorithm": "all_full_horizon_valid_windows_minus_dual_identity_holdout_v1",
            "validity_column": "valid_state",
            "validity_value": 1,
            "end_clamping_allowed": False,
        },
        representation_contract_sha256=representation_contract_sha256,
        horizon=horizon,
        statistics_population_candidate=statistics_population_candidate,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def _add_common_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eval-holdout-manifest",
        type=Path,
        required=True,
        help=(
            "Immutable LeRobot episode-split manifest. Holdout IDs are "
            "derived from its exact dataset/catalog binding; production "
            "commands cannot use hard-coded episode lists."
        ),
    )
    parser.add_argument(
        "--representation-contract-sha256",
        default=DEFAULT_REPRESENTATION_SHA256,
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--statistics-population-candidate",
        action="store_true",
        help=(
            "Build a non-trainable population-candidate view that includes "
            "authenticated holdout episode references. The union-statistics "
            "builder excludes those keys; training loaders reject this view."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    intervention = subparsers.add_parser(
        "intervention-incremental",
        help="Build every row from incremental intervention episodes.",
    )
    _add_common_build_arguments(intervention)
    intervention.add_argument(
        "--first-episode",
        type=int,
        default=DEFAULT_INCREMENTAL_FIRST_EPISODE,
    )
    intervention.add_argument(
        "--last-episode",
        type=int,
        default=DEFAULT_INCREMENTAL_LAST_EPISODE,
    )
    intervention.add_argument(
        "--action-label-semantics-contract",
        type=Path,
        default=None,
        help=(
            "Reviewed JSON contract binding this dataset's action-label "
            "semantics. Without it (or explicit valid_action plus expert "
            "owner/source columns), the view is still written for audit but "
            "is marked unverified and cannot be used for Stage B."
        ),
    )

    hq = subparsers.add_parser(
        "hq-clean-h50",
        help="Build every fully valid HQ action-horizon window.",
    )
    _add_common_build_arguments(hq)

    realsource = subparsers.add_parser(
        "realsource-canonical",
        help=(
            "Build a strict-valid full or breadth-balanced frozen RealSource "
            "canonical view."
        ),
    )
    realsource.add_argument(
        "--canonical-manifest",
        type=Path,
        default=DEFAULT_CANONICAL_MANIFEST,
    )
    realsource.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_CANONICAL_ADAPTER,
    )
    realsource.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CANONICAL_CACHE,
    )
    realsource.add_argument(
        "--eval-holdout-manifest",
        type=Path,
        help=(
            "Immutable canonical heldout-eval manifest. Every referenced "
            "episode is removed before the frozen ledger and row count are "
            "written, and the manifest/hash are bound into the view. Required "
            "unless --eval-selection-population-candidate is used."
        ),
    )
    realsource.add_argument("--output", type=Path, required=True)
    realsource.add_argument(
        "--fraction",
        default="1",
        help=(
            "Frozen whole-episode population fraction, e.g. 0.10, 0.50, or "
            "1. This never means a fractional epoch."
        ),
    )
    realsource.add_argument("--seed", type=int, default=0)
    realsource.add_argument(
        "--representation-contract-sha256",
        default=DEFAULT_REPRESENTATION_SHA256,
    )
    realsource.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    candidate_mode = realsource.add_mutually_exclusive_group()
    candidate_mode.add_argument(
        "--statistics-population-candidate",
        action="store_true",
        help=(
            "Build a non-trainable population-candidate view containing the "
            "authenticated eval-holdout episodes. The union-statistics "
            "builder excludes those keys; training loaders reject this view."
        ),
    )
    candidate_mode.add_argument(
        "--eval-selection-population-candidate",
        action="store_true",
        help=(
            "Build the non-trainable, pre-holdout deterministic population "
            "used only by generate_canonical_eval_manifest.py. It cannot be "
            "used for training or union statistics."
        ),
    )
    realsource.add_argument("--overwrite", action="store_true")
    realsource.add_argument("--dry-run", action="store_true")
    realsource.add_argument(
        "--summary-only",
        action="store_true",
        help="Validate catalogs and print selection counts without a ledger.",
    )

    verify = subparsers.add_parser(
        "verify", help="Fail-closed verify a descriptor and its row ledger."
    )
    verify.add_argument("manifest", type=Path)
    verify.add_argument(
        "--expected-representation-contract-sha256",
        default=DEFAULT_REPRESENTATION_SHA256,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "intervention-incremental":
        result = build_intervention_incremental_view(
            dataset_root=args.dataset_root,
            output_manifest=args.output,
            eval_holdout_manifest=args.eval_holdout_manifest,
            first_episode=args.first_episode,
            last_episode=args.last_episode,
            representation_contract_sha256=(
                args.representation_contract_sha256
            ),
            horizon=args.horizon,
            action_label_semantics_contract=(
                args.action_label_semantics_contract
            ),
            statistics_population_candidate=(
                args.statistics_population_candidate
            ),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        payload = result.to_dict()
    elif args.command == "hq-clean-h50":
        result = build_hq_clean_h50_view(
            dataset_root=args.dataset_root,
            output_manifest=args.output,
            eval_holdout_manifest=args.eval_holdout_manifest,
            representation_contract_sha256=(
                args.representation_contract_sha256
            ),
            horizon=args.horizon,
            statistics_population_candidate=(
                args.statistics_population_candidate
            ),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        payload = result.to_dict()
    elif args.command == "realsource-canonical":
        result = build_realsource_canonical_view(
            canonical_manifest=args.canonical_manifest,
            adapter_path=args.adapter_path,
            cache_dir=args.cache_dir,
            eval_holdout_manifest=args.eval_holdout_manifest,
            output_manifest=args.output,
            fraction=args.fraction,
            seed=args.seed,
            representation_contract_sha256=(
                args.representation_contract_sha256
            ),
            horizon=args.horizon,
            statistics_population_candidate=(
                args.statistics_population_candidate
            ),
            eval_selection_population_candidate=(
                args.eval_selection_population_candidate
            ),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            summary_only=args.summary_only,
        )
        payload = result if isinstance(result, dict) else result.to_dict()
    else:
        view = dataset_view.load_frozen_view(
            args.manifest,
            expected_representation_contract_sha256=(
                args.expected_representation_contract_sha256
            ),
        )
        payload = {
            "manifest_path": str(view.manifest_path),
            "manifest_sha256": view.manifest_sha256,
            "view_id": view.view_id,
            "row_count": view.row_count,
            "unique_sample_count": view.unique_sample_count,
            "episode_count": view.episode_count,
            "verified": True,
        }
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
