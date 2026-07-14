"""Fail-closed, content-bound episode splits for LeRobot datasets.

The split is intentionally enforced by :class:`LeRobotSingleDataset`, before
its dense step index is built.  A manifest is not just a convenient list of
episode numbers: it commits to the dataset metadata catalog and to one
train-only statistics artifact.  Evaluation roles consequently use the same
normalization table as training and can never derive statistics from held-out
episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SPLIT_MANIFEST_SCHEMA_VERSION = 1
CATALOG_BINDING_SCHEMA = "lerobot-episode-catalog-v1"
EPISODE_SET_SCHEMA = "lerobot-episode-set-v1"
TRAIN_STATISTICS_SCHEMA = "lerobot-train-statistics-v1"
TRAIN_STATISTICS_PROVENANCE_KEY = "_split_provenance"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _as_episode_records(
    trajectory_ids: Sequence[int] | np.ndarray,
    trajectory_lengths: Sequence[int] | np.ndarray,
) -> list[dict[str, int]]:
    ids = np.asarray(trajectory_ids).reshape(-1)
    lengths = np.asarray(trajectory_lengths).reshape(-1)
    if ids.size != lengths.size:
        raise ValueError(
            "LeRobot episode catalog has different numbers of IDs and lengths: "
            f"{ids.size} != {lengths.size}."
        )
    records: list[dict[str, int]] = []
    seen: set[int] = set()
    for raw_id, raw_length in zip(ids.tolist(), lengths.tolist()):
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, np.integer)):
            raise ValueError(f"Episode ID must be an integer, got {raw_id!r}.")
        if isinstance(raw_length, bool) or not isinstance(
            raw_length, (int, np.integer)
        ):
            raise ValueError(
                f"Episode {raw_id!r} length must be an integer, got {raw_length!r}."
            )
        episode_id = int(raw_id)
        length = int(raw_length)
        if episode_id < 0 or length <= 0:
            raise ValueError(
                f"Invalid LeRobot episode record id={episode_id}, length={length}."
            )
        if episode_id in seen:
            raise ValueError(f"Duplicate LeRobot episode ID {episode_id}.")
        seen.add(episode_id)
        records.append({"episode_id": episode_id, "length": length})
    if not records:
        raise ValueError("LeRobot episode catalog is empty.")
    return sorted(records, key=lambda record: record["episode_id"])


def episode_set_sha256(
    episode_ids: Sequence[int],
    *,
    lengths_by_id: Mapping[int, int],
) -> str:
    records = []
    for episode_id in sorted(int(value) for value in episode_ids):
        if episode_id not in lengths_by_id:
            raise ValueError(
                f"Episode {episode_id} is absent from the bound dataset catalog."
            )
        records.append(
            {"episode_id": episode_id, "length": int(lengths_by_id[episode_id])}
        )
    return canonical_json_sha256(
        {"schema": EPISODE_SET_SCHEMA, "episodes": records}
    )


def build_episode_catalog_binding(
    *,
    dataset_path: Path | str,
    dataset_name: str,
    lerobot_version: str,
    trajectory_ids: Sequence[int] | np.ndarray,
    trajectory_lengths: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Build the immutable catalog fields required in a split manifest.

    This hashes ``meta/info.json`` plus the complete episode metadata files.
    Episode selection is therefore invalidated by reindexing, length changes,
    metadata edits, or a different dataset version/name.  Large video/data
    shards are deliberately not re-hashed at every process startup; their
    episode-to-shard bindings live in the hashed episode metadata catalog.
    """

    root = Path(dataset_path).expanduser().resolve()
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot catalog metadata: {info_path}")

    if lerobot_version == "v3.0":
        episode_paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    elif lerobot_version == "v2.0":
        episode_paths = [root / "meta/episodes.jsonl"]
    else:
        raise ValueError(f"Unsupported LeRobot version {lerobot_version!r}.")
    if not episode_paths or any(not path.is_file() for path in episode_paths):
        raise FileNotFoundError(
            f"Missing LeRobot {lerobot_version} episode catalog under {root}."
        )

    records = _as_episode_records(trajectory_ids, trajectory_lengths)
    episode_files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in episode_paths
    ]
    digest_payload = {
        "schema": CATALOG_BINDING_SCHEMA,
        "dataset_name": str(dataset_name),
        "lerobot_version": str(lerobot_version),
        "info_sha256": file_sha256(info_path),
        "episode_files": episode_files,
        "episodes": records,
    }
    return {
        "schema": CATALOG_BINDING_SCHEMA,
        "dataset_name": str(dataset_name),
        "lerobot_version": str(lerobot_version),
        "info_sha256": digest_payload["info_sha256"],
        "episode_catalog_sha256": canonical_json_sha256(digest_payload),
        "total_episodes": len(records),
        "total_frames": sum(record["length"] for record in records),
    }


def _validated_episode_ids(raw_ids: Any, *, context: str) -> tuple[int, ...]:
    if not isinstance(raw_ids, list):
        raise ValueError(f"{context} must be a JSON list of episode IDs.")
    ids: list[int] = []
    for raw_id in raw_ids:
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise ValueError(f"{context} contains non-integer ID {raw_id!r}.")
        if raw_id < 0:
            raise ValueError(f"{context} contains negative ID {raw_id}.")
        ids.append(raw_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context} contains duplicate episode IDs.")
    return tuple(sorted(ids))


@dataclass(frozen=True, slots=True)
class EpisodeSplitSelection:
    manifest_path: Path
    manifest_sha256: str
    role: str
    selected_episode_ids: tuple[int, ...]
    selected_frame_count: int
    train_episode_ids: tuple[int, ...]
    holdout_episode_ids: tuple[int, ...]
    train_frame_count: int
    train_episode_set_sha256: str
    train_statistics_path: Path
    train_statistics_sha256: str
    catalog_binding: Mapping[str, Any]
    role_counts: Mapping[str, Mapping[str, int]]

    def provenance(self) -> dict[str, Any]:
        full_episode_ids = tuple(
            int(record["episode_id"])
            for record in self.catalog_binding.get("episodes", ())
        )
        selected_ids = set(self.selected_episode_ids)
        return {
            "enabled": True,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "role": self.role,
            "selected_episode_count": len(self.selected_episode_ids),
            "selected_episode_ids": list(self.selected_episode_ids),
            "excluded_episode_ids": [
                episode_id
                for episode_id in full_episode_ids
                if episode_id not in selected_ids
            ],
            "selected_frame_count": self.selected_frame_count,
            "selected_episode_set_sha256": episode_set_sha256(
                self.selected_episode_ids,
                lengths_by_id={
                    int(record["episode_id"]): int(record["length"])
                    for record in self.catalog_binding["episodes"]
                },
            )
            if "episodes" in self.catalog_binding
            else None,
            "role_counts": dict(self.role_counts),
            "train_episode_count": len(self.train_episode_ids),
            "train_episode_ids": list(self.train_episode_ids),
            "holdout_episode_count": len(self.holdout_episode_ids),
            "holdout_episode_ids": list(self.holdout_episode_ids),
            "holdout_episode_set_sha256": episode_set_sha256(
                self.holdout_episode_ids,
                lengths_by_id={
                    int(record["episode_id"]): int(record["length"])
                    for record in self.catalog_binding["episodes"]
                },
            ),
            "train_frame_count": self.train_frame_count,
            "train_episode_set_sha256": self.train_episode_set_sha256,
            "normalization_statistics_scope": "train_split_only",
            "full_catalog_sha256": self.catalog_binding[
                "episode_catalog_sha256"
            ],
            "train_statistics_path": str(self.train_statistics_path),
            "train_statistics_sha256": self.train_statistics_sha256,
        }


def load_episode_split_selection(
    *,
    manifest_path: Path | str,
    dataset_name: str,
    role: str,
    catalog_binding: Mapping[str, Any],
    trajectory_ids: Sequence[int] | np.ndarray,
    trajectory_lengths: Sequence[int] | np.ndarray,
) -> EpisodeSplitSelection:
    """Validate a manifest and return one role's immutable episode selection."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Episode split manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Episode split manifest is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Episode split manifest root must be a JSON object.")
    if payload.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Episode split manifest schema_version must be "
            f"{SPLIT_MANIFEST_SCHEMA_VERSION}, got {payload.get('schema_version')!r}."
        )
    split_id = payload.get("split_id")
    if not isinstance(split_id, str) or not split_id.strip():
        raise ValueError("Episode split manifest split_id must be a non-empty string.")
    expected_role_contract = {
        "train_episode_selection": "complement_of_holdout",
        "evaluation_episode_selection": "holdout_episode_indices",
        "normalization_statistics": "train_statistics_only",
    }
    if payload.get("role_contract") != expected_role_contract:
        raise ValueError(
            "Episode split manifest role_contract must exactly encode the "
            f"fail-closed contract {expected_role_contract!r}."
        )
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Episode split manifest datasets must be a JSON list.")
    matches = [
        entry
        for entry in datasets
        if isinstance(entry, dict) and entry.get("dataset_name") == dataset_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Episode split manifest must have exactly one entry for dataset "
            f"{dataset_name!r}; found {len(matches)}."
        )
    entry = matches[0]

    expected_binding = dict(catalog_binding)
    if entry.get("full_catalog_sha256") != expected_binding["episode_catalog_sha256"]:
        raise ValueError(
            f"Episode split catalog binding does not match dataset {dataset_name!r}. "
            f"Expected full_catalog_sha256 "
            f"{expected_binding['episode_catalog_sha256']!r}, got "
            f"{entry.get('full_catalog_sha256')!r}."
        )
    if entry.get("full_episode_count") != expected_binding["total_episodes"]:
        raise ValueError(
            "Episode split full_episode_count does not match the dataset catalog."
        )
    if entry.get("full_frame_count") != expected_binding["total_frames"]:
        raise ValueError(
            "Episode split full_frame_count does not match the dataset catalog."
        )
    optional_info_sha = entry.get("info_sha256")
    if optional_info_sha is not None and optional_info_sha != expected_binding["info_sha256"]:
        raise ValueError("Episode split info_sha256 does not match meta/info.json.")
    optional_version = entry.get("lerobot_version")
    if optional_version is not None and optional_version != expected_binding["lerobot_version"]:
        raise ValueError("Episode split lerobot_version does not match the dataset.")

    records = _as_episode_records(trajectory_ids, trajectory_lengths)
    lengths_by_id = {
        int(record["episode_id"]): int(record["length"]) for record in records
    }
    catalog_ids = set(lengths_by_id)

    holdout_ids = _validated_episode_ids(
        entry.get("holdout_episode_indices"),
        context=f"datasets[{dataset_name}].holdout_episode_indices",
    )
    if not holdout_ids:
        raise ValueError(
            f"Holdout split is empty for dataset {dataset_name!r}."
        )
    unknown_holdout = sorted(set(holdout_ids) - catalog_ids)
    if unknown_holdout:
        raise ValueError(
            f"Holdout split references unknown episodes {unknown_holdout}."
        )
    raw_train_selection = entry.get("train_episode_selection")
    if raw_train_selection != {"kind": "complement_of_holdout"}:
        raise ValueError(
            "train_episode_selection must be exactly "
            "{'kind': 'complement_of_holdout'}."
        )
    train_ids = tuple(sorted(catalog_ids - set(holdout_ids)))
    if not train_ids:
        raise ValueError(f"Train split is empty for dataset {dataset_name!r}.")

    role_aliases = {
        "train": "train",
        "eval": "holdout",
        "evaluation": "holdout",
        "val": "holdout",
        "validation": "holdout",
        "test": "holdout",
        "holdout": "holdout",
    }
    if role not in role_aliases:
        raise ValueError(
            f"Unsupported episode split role {role!r}; expected one of "
            f"{sorted(role_aliases)}."
        )
    selected_ids = train_ids if role_aliases[role] == "train" else holdout_ids

    train_set_sha = episode_set_sha256(train_ids, lengths_by_id=lengths_by_id)
    holdout_set_sha = episode_set_sha256(holdout_ids, lengths_by_id=lengths_by_id)
    train_frame_count = sum(lengths_by_id[episode_id] for episode_id in train_ids)
    holdout_frame_count = sum(
        lengths_by_id[episode_id] for episode_id in holdout_ids
    )
    count_bindings = {
        "holdout_episode_count": len(holdout_ids),
        "holdout_frame_count": holdout_frame_count,
        "holdout_catalog_sha256": holdout_set_sha,
        "train_episode_count": len(train_ids),
        "train_frame_count": train_frame_count,
        "train_catalog_sha256": train_set_sha,
    }
    mismatched_bindings = {
        key: (entry.get(key), expected)
        for key, expected in count_bindings.items()
        if entry.get(key) != expected
    }
    if mismatched_bindings:
        raise ValueError(
            "Episode split train/holdout count or catalog bindings do not match: "
            f"{mismatched_bindings!r}."
        )

    raw_statistics = entry.get("train_statistics")
    if not isinstance(raw_statistics, dict):
        raise ValueError(
            f"Split entry for {dataset_name!r} must bind train_statistics."
        )
    required_statistics_fields = {
        "path",
        "sha256",
        "frame_count",
        "catalog_sha256",
    }
    missing_statistics = sorted(required_statistics_fields - set(raw_statistics))
    if missing_statistics:
        raise ValueError(
            f"train_statistics is missing required fields {missing_statistics}."
        )
    raw_statistics_path = raw_statistics["path"]
    if not isinstance(raw_statistics_path, str) or not raw_statistics_path.strip():
        raise ValueError("train_statistics.path must be a non-empty string.")
    statistics_path = Path(raw_statistics_path).expanduser()
    if not statistics_path.is_absolute():
        statistics_path = path.parent / statistics_path
    statistics_path = statistics_path.resolve()
    if not statistics_path.is_file():
        raise FileNotFoundError(
            f"Bound train-only statistics file does not exist: {statistics_path}"
        )
    expected_statistics_sha = raw_statistics["sha256"]
    actual_statistics_sha = file_sha256(statistics_path)
    if expected_statistics_sha != actual_statistics_sha:
        raise ValueError(
            "Train-only statistics SHA-256 mismatch: expected "
            f"{expected_statistics_sha!r}, got {actual_statistics_sha!r}."
        )
    if raw_statistics["frame_count"] != train_frame_count:
        raise ValueError(
            "train_statistics.frame_count does not match the train split: "
            f"{raw_statistics['frame_count']!r} != {train_frame_count}."
        )
    if raw_statistics["catalog_sha256"] != train_set_sha:
        raise ValueError(
            "train_statistics.catalog_sha256 does not match the train split."
        )
    try:
        statistics_payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Bound train-only statistics is invalid JSON: {statistics_path}: {exc}"
        ) from exc
    expected_statistics_provenance = {
        "schema": TRAIN_STATISTICS_SCHEMA,
        "full_catalog_sha256": expected_binding["episode_catalog_sha256"],
        "train_catalog_sha256": train_set_sha,
        "train_episode_count": len(train_ids),
        "train_frame_count": train_frame_count,
    }
    if not isinstance(statistics_payload, dict) or statistics_payload.get(
        TRAIN_STATISTICS_PROVENANCE_KEY
    ) != expected_statistics_provenance:
        raise ValueError(
            "Bound train-only statistics does not contain the exact "
            f"{TRAIN_STATISTICS_PROVENANCE_KEY} record for this split: "
            f"{expected_statistics_provenance!r}."
        )

    role_counts = {
        "train": {"episodes": len(train_ids), "frames": train_frame_count},
        "holdout": {
            "episodes": len(holdout_ids),
            "frames": holdout_frame_count,
        },
    }
    # Store records in the in-memory binding only for rich provenance.  These
    # are derived from the already validated catalog and are not expected in
    # the JSON manifest's compact catalog_binding object.
    rich_binding = dict(expected_binding)
    rich_binding["episodes"] = records
    return EpisodeSplitSelection(
        manifest_path=path,
        manifest_sha256=file_sha256(path),
        role=role,
        selected_episode_ids=selected_ids,
        selected_frame_count=sum(lengths_by_id[value] for value in selected_ids),
        train_episode_ids=train_ids,
        holdout_episode_ids=holdout_ids,
        train_frame_count=train_frame_count,
        train_episode_set_sha256=train_set_sha,
        train_statistics_path=statistics_path,
        train_statistics_sha256=actual_statistics_sha,
        catalog_binding=rich_binding,
        role_counts=role_counts,
    )
