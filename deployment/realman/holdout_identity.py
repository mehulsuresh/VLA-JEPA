"""Content identities for fail-closed RealMan held-out evaluation.

LeRobot episode indices and dataset directory names are storage locators, not
episode identities.  This module derives stable identities from the exact
22-D source action and 19-D source observation-state trajectories instead.
The resulting hashes are invariant to dataset renaming, episode reindexing,
Parquet row-group changes, and video repackaging.

The statistics digest is deliberately secondary.  Per-episode summary
statistics are useful provenance and a cheap diagnostic, but different
trajectories can have identical summaries and statistics implementations can
change between LeRobot versions.  Exact source-trajectory digests are the
authoritative overlap key.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SOURCE_ACTION_KEY = "source.action"
SOURCE_STATE_KEY = "source.observation.state"
SOURCE_ACTION_DIM = 22
SOURCE_STATE_DIM = 19
REALMAN_FPS = 20

SOURCE_ACTION_NAMES = (
    *(f"left_joint_{index}" for index in range(7)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(7)),
    "right_gripper",
    "base_linear_x_mps",
    "base_linear_y_mps",
    "base_angular_z_radps",
    "head_joint_1_rad",
    "head_joint_2_rad",
    "lift_height_mm",
)
SOURCE_STATE_NAMES = (
    *(f"left_joint_{index}" for index in range(7)),
    "left_gripper_open",
    *(f"right_joint_{index}" for index in range(7)),
    "right_gripper_open",
    "head_joint_1_rad",
    "head_joint_2_rad",
    "lift_height_mm",
)

_SOURCE_FIELDS = (
    (SOURCE_ACTION_KEY, SOURCE_ACTION_DIM, SOURCE_ACTION_NAMES),
    (SOURCE_STATE_KEY, SOURCE_STATE_DIM, SOURCE_STATE_NAMES),
)
_STATISTIC_NAMES = (
    "min",
    "max",
    "mean",
    "std",
    "q01",
    "q10",
    "q50",
    "q90",
    "q99",
)
_ARRAY_HASH_DOMAIN = b"realman-source-array-v1\0"
_STATISTICS_HASH_DOMAIN = b"realman-source-episode-statistics-v1\0"
_EPISODE_CONTENT_SCHEMA = "realman-source-episode-content-v1"
_CONTENT_CATALOG_SCHEMA = "realman-source-episode-content-catalog-v1"
_STATISTICS_CATALOG_SCHEMA = "realman-source-episode-statistics-catalog-v1"


@dataclass(frozen=True, slots=True)
class EpisodeLocator:
    """Storage coordinates excluded from all content hashes."""

    dataset_root: Path
    episode_index: int
    data_relative_path: str
    dataset_from_index: int
    dataset_to_index: int


@dataclass(frozen=True, slots=True)
class SourceEpisodeIdentity:
    """Exact source-trajectory identity plus its local storage locator."""

    locator: EpisodeLocator
    fps: int
    length: int
    source_action_sha256: str
    source_observation_state_sha256: str
    source_trajectory_sha256: str
    source_statistics_sha256: str

    def content_record(self) -> dict[str, int | str]:
        """Return the locator-independent record committed by a catalog."""

        return {
            "fps": self.fps,
            "length": self.length,
            "source_action_sha256": self.source_action_sha256,
            "source_observation_state_sha256": (
                self.source_observation_state_sha256
            ),
            "source_trajectory_sha256": self.source_trajectory_sha256,
        }

    def manifest_record(self) -> dict[str, int | str]:
        """Return a manifest record that binds a locator to its content."""

        return {
            "episode_index": self.locator.episode_index,
            "length": self.length,
            "fps": self.fps,
            "source_action_sha256": self.source_action_sha256,
            "source_observation_state_sha256": (
                self.source_observation_state_sha256
            ),
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "source_statistics_sha256": self.source_statistics_sha256,
        }


@dataclass(frozen=True, slots=True)
class SourceIdentityCatalog:
    """Validated content catalog for one LeRobot v3 dataset root."""

    dataset_root: Path
    fps: int
    episode_count: int
    frame_count: int
    episodes: tuple[SourceEpisodeIdentity, ...]
    content_catalog_sha256: str
    statistics_catalog_sha256: str

    def episode(self, episode_index: int) -> SourceEpisodeIdentity:
        matches = [
            identity
            for identity in self.episodes
            if identity.locator.episode_index == int(episode_index)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Episode index {episode_index} is not present exactly once in "
                f"{self.dataset_root}."
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class SourceEpisodeOverlap:
    """A selected evaluation episode matching training source content."""

    match_kind: str
    digest: str
    evaluation: EpisodeLocator
    training: tuple[EpisodeLocator, ...]


@dataclass(frozen=True, slots=True)
class _EpisodeMetadata:
    episode_index: int
    length: int
    data_relative_path: str
    dataset_from_index: int
    dataset_to_index: int
    statistics_sha256: str


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_float_bytes(
    values: Any,
    *,
    dtype: np.dtype[Any],
    context: str,
) -> bytes:
    array = np.asarray(values, dtype=dtype).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains non-finite values.")
    # -0.0 and +0.0 are numerically identical and can differ after otherwise
    # lossless dataset conversions.  Canonicalize both to the same bit pattern.
    array[array == 0] = 0.0
    little_endian = array.astype(dtype.newbyteorder("<"), copy=False)
    return little_endian.tobytes(order="C")


def _new_array_hasher(field: str, *, length: int, dimension: int) -> Any:
    hasher = hashlib.sha256()
    hasher.update(_ARRAY_HASH_DOMAIN)
    hasher.update(field.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<QQ", length, dimension))
    return hasher


def _validate_info(root: Path, *, expected_fps: int) -> tuple[dict[str, Any], int, int]:
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(
            f"Holdout content identity requires LeRobot v3, got "
            f"{info.get('codebase_version')!r} in {info_path}."
        )

    raw_fps = info.get("fps")
    if isinstance(raw_fps, bool) or not isinstance(raw_fps, (int, float)):
        raise ValueError(f"Dataset fps must be numeric, got {raw_fps!r}.")
    if not math.isfinite(float(raw_fps)) or float(raw_fps) != float(expected_fps):
        raise ValueError(
            f"Dataset fps {raw_fps!r} does not match required RealMan fps "
            f"{expected_fps}."
        )

    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("meta/info.json must contain a feature mapping.")
    for field, dimension, expected_names in _SOURCE_FIELDS:
        feature = features.get(field)
        if not isinstance(feature, Mapping):
            raise ValueError(f"Dataset is missing required source feature {field!r}.")
        if feature.get("dtype") != "float32":
            raise ValueError(
                f"Source feature {field!r} must have dtype float32, got "
                f"{feature.get('dtype')!r}."
            )
        if list(feature.get("shape") or ()) != [dimension]:
            raise ValueError(
                f"Source feature {field!r} must have shape [{dimension}], got "
                f"{feature.get('shape')!r}."
            )
        if tuple(feature.get("names") or ()) != expected_names:
            raise ValueError(
                f"Source feature {field!r} semantic names/order do not match the "
                "RealMan source contract."
            )

    try:
        raw_total_episodes = info["total_episodes"]
        raw_total_frames = info["total_frames"]
    except KeyError as exc:
        raise ValueError(
            "meta/info.json must contain integral total_episodes and total_frames."
        ) from exc
    if (
        isinstance(raw_total_episodes, bool)
        or not isinstance(raw_total_episodes, int)
        or isinstance(raw_total_frames, bool)
        or not isinstance(raw_total_frames, int)
    ):
        raise ValueError(
            "meta/info.json total_episodes and total_frames must be integers."
        )
    total_episodes = raw_total_episodes
    total_frames = raw_total_frames
    if total_episodes <= 0 or total_frames <= 0:
        raise ValueError(
            f"Dataset totals must be positive, got episodes={total_episodes}, "
            f"frames={total_frames}."
        )
    return dict(info), total_episodes, total_frames


def _statistics_sha256(row: Mapping[str, Any], *, episode_index: int, length: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(_STATISTICS_HASH_DOMAIN)
    hasher.update(struct.pack("<Q", length))
    for field, dimension, _ in _SOURCE_FIELDS:
        hasher.update(field.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(struct.pack("<Q", dimension))
        raw_count = row[f"stats/{field}/count"]
        if not isinstance(raw_count, Sequence) or isinstance(raw_count, (str, bytes)):
            raise ValueError(
                f"Episode {episode_index} {field} statistics count must be a list."
            )
        if len(raw_count) != 1 or int(raw_count[0]) != length:
            raise ValueError(
                f"Episode {episode_index} {field} statistics count {raw_count!r} "
                f"does not match length {length}."
            )
        hasher.update(struct.pack("<Q", length))
        for statistic in _STATISTIC_NAMES:
            values = row[f"stats/{field}/{statistic}"]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(
                    f"Episode {episode_index} {field} {statistic} statistics must "
                    "be a list."
                )
            if len(values) != dimension:
                raise ValueError(
                    f"Episode {episode_index} {field} {statistic} statistics width "
                    f"{len(values)} does not match {dimension}."
                )
            hasher.update(statistic.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(
                _canonical_float_bytes(
                    values,
                    dtype=np.dtype(np.float64),
                    context=(
                        f"Episode {episode_index} {field} {statistic} statistics"
                    ),
                )
            )
    return hasher.hexdigest()


def _safe_data_relative_path(
    root: Path,
    template: str,
    *,
    chunk_index: int,
    file_index: int,
) -> str:
    try:
        rendered = template.format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"Invalid LeRobot data_path template {template!r}.") from exc
    relative = Path(rendered)
    if relative.is_absolute():
        raise ValueError(f"LeRobot data_path must be relative, got {rendered!r}.")
    absolute = (root / relative).resolve()
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"LeRobot data_path escapes the dataset root: {rendered!r}."
        ) from exc
    return relative.as_posix()


def _read_episode_metadata(
    root: Path,
    *,
    info: Mapping[str, Any],
    expected_episode_count: int,
    expected_frame_count: int,
) -> tuple[_EpisodeMetadata, ...]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot v3 episode metadata found under {root}.")
    location_columns = (
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    )
    statistic_columns = tuple(
        f"stats/{field}/{statistic}"
        for field, _, _ in _SOURCE_FIELDS
        for statistic in ("count", *_STATISTIC_NAMES)
    )
    required_columns = (*location_columns, *statistic_columns)
    data_path_template = str(info.get("data_path") or "")
    if not data_path_template:
        raise ValueError("meta/info.json is missing data_path.")

    metadata: list[_EpisodeMetadata] = []
    seen_episode_indices: set[int] = set()
    for path in paths:
        schema = pq.read_schema(path)
        schema_names = set(schema.names)
        missing = sorted(set(required_columns) - schema_names)
        if missing:
            raise ValueError(
                f"Episode metadata {path} is missing required columns: {missing}."
            )
        for column in location_columns:
            if schema.field(column).type != pa.int64():
                raise ValueError(
                    f"Episode metadata {path} column {column!r} must have type "
                    f"int64, got {schema.field(column).type}."
                )
        for field, _, _ in _SOURCE_FIELDS:
            count_type = schema.field(f"stats/{field}/count").type
            if not pa.types.is_list(count_type) or count_type.value_type != pa.int64():
                raise ValueError(
                    f"Episode metadata {path} {field} statistics count must have "
                    f"type list<int64>, got {count_type}."
                )
            for statistic in _STATISTIC_NAMES:
                statistic_type = schema.field(f"stats/{field}/{statistic}").type
                if (
                    not pa.types.is_list(statistic_type)
                    or statistic_type.value_type != pa.float64()
                ):
                    raise ValueError(
                        f"Episode metadata {path} {field} {statistic} must have "
                        f"type list<float64>, got {statistic_type}."
                    )
        for row in pq.read_table(path, columns=list(required_columns)).to_pylist():
            episode_index = int(row["episode_index"])
            if episode_index < 0:
                raise ValueError(f"Episode index must be non-negative, got {episode_index}.")
            if episode_index in seen_episode_indices:
                raise ValueError(f"Duplicate episode_index {episode_index} in {root}.")
            seen_episode_indices.add(episode_index)
            length = int(row["length"])
            dataset_from_index = int(row["dataset_from_index"])
            dataset_to_index = int(row["dataset_to_index"])
            if length <= 0:
                raise ValueError(
                    f"Episode {episode_index} has non-positive length {length}."
                )
            if dataset_from_index < 0 or dataset_to_index - dataset_from_index != length:
                raise ValueError(
                    f"Episode {episode_index} metadata range "
                    f"[{dataset_from_index}, {dataset_to_index}) does not match "
                    f"length {length}."
                )
            chunk_index = int(row["data/chunk_index"])
            file_index = int(row["data/file_index"])
            if chunk_index < 0 or file_index < 0:
                raise ValueError(
                    f"Episode {episode_index} has negative data shard indices: "
                    f"chunk={chunk_index}, file={file_index}."
                )
            data_relative_path = _safe_data_relative_path(
                root,
                data_path_template,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            metadata.append(
                _EpisodeMetadata(
                    episode_index=episode_index,
                    length=length,
                    data_relative_path=data_relative_path,
                    dataset_from_index=dataset_from_index,
                    dataset_to_index=dataset_to_index,
                    statistics_sha256=_statistics_sha256(
                        row,
                        episode_index=episode_index,
                        length=length,
                    ),
                )
            )

    if len(metadata) != expected_episode_count:
        raise ValueError(
            f"Episode metadata count {len(metadata)} does not match info.total_episodes "
            f"{expected_episode_count}."
        )
    ordered = sorted(metadata, key=lambda item: item.dataset_from_index)
    expected_from = 0
    for item in ordered:
        if item.dataset_from_index != expected_from:
            raise ValueError(
                "Episode dataset ranges are not contiguous: expected next range at "
                f"{expected_from}, found episode {item.episode_index} at "
                f"{item.dataset_from_index}."
            )
        expected_from = item.dataset_to_index
    if expected_from != expected_frame_count:
        raise ValueError(
            f"Episode metadata covers {expected_from} frames, but info.total_frames "
            f"is {expected_frame_count}."
        )
    return tuple(ordered)


def _validate_physical_data_schema(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    scalar_types = {
        "episode_index": pa.int64(),
        "frame_index": pa.int64(),
        "index": pa.int64(),
        "timestamp": pa.float32(),
    }
    for field, expected_type in scalar_types.items():
        if field not in schema.names or schema.field(field).type != expected_type:
            actual = schema.field(field).type if field in schema.names else None
            raise ValueError(
                f"Data file {path} field {field!r} must have type {expected_type}, "
                f"got {actual}."
            )
    for field, dimension, _ in _SOURCE_FIELDS:
        if field not in schema.names:
            raise ValueError(f"Data file {path} is missing source field {field!r}.")
        actual = schema.field(field).type
        is_fixed_width = (
            pa.types.is_fixed_size_list(actual) and actual.list_size == dimension
        )
        is_variable_width = pa.types.is_list(actual)
        if (
            not (is_fixed_width or is_variable_width)
            or actual.value_type != pa.float32()
        ):
            raise ValueError(
                f"Data file {path} source field {field!r} must be "
                f"list<float32> with every row width {dimension}, got {actual}."
            )


def _fixed_size_list_numpy(column: Any, *, dimension: int, context: str) -> np.ndarray:
    if column.null_count:
        raise ValueError(f"{context} contains null source vectors.")
    import pyarrow as pa

    if pa.types.is_fixed_size_list(column.type):
        start = int(column.offset) * dimension
        value_count = len(column) * dimension
    elif pa.types.is_list(column.type):
        offsets = np.asarray(
            column.offsets.to_numpy(zero_copy_only=False), dtype=np.int64
        )
        widths = np.diff(offsets)
        if not np.all(widths == dimension):
            bad_row = int(np.nonzero(widths != dimension)[0][0])
            raise ValueError(
                f"{context} row {bad_row} has width {int(widths[bad_row])}, "
                f"expected {dimension}."
            )
        start = int(offsets[0])
        value_count = int(offsets[-1] - offsets[0])
    else:  # Protected by physical schema validation; keep this fail closed.
        raise ValueError(f"{context} is not an Arrow list array.")
    flat = column.values.slice(start, value_count)
    if flat.null_count:
        raise ValueError(f"{context} contains null source values.")
    return np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(
        len(column), dimension
    )


def _trajectory_hashes(
    root: Path,
    metadata: Sequence[_EpisodeMetadata],
    *,
    fps: int,
    batch_size: int,
) -> dict[int, tuple[str, str, str]]:
    import pyarrow.parquet as pq

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    by_episode = {item.episode_index: item for item in metadata}
    by_path: dict[str, list[_EpisodeMetadata]] = defaultdict(list)
    for item in metadata:
        by_path[item.data_relative_path].append(item)
    for items in by_path.values():
        items.sort(key=lambda item: item.dataset_from_index)

    expected_paths = {root / relative for relative in by_path}
    actual_paths = set((root / "data").glob("**/*.parquet"))
    if actual_paths != expected_paths:
        missing = sorted(str(path.relative_to(root)) for path in expected_paths - actual_paths)
        unexpected = sorted(str(path.relative_to(root)) for path in actual_paths - expected_paths)
        raise ValueError(
            f"Data shard catalog does not match episode metadata: missing={missing}, "
            f"unexpected={unexpected}."
        )

    hashers: dict[int, dict[str, Any]] = {}
    next_frame = {item.episode_index: 0 for item in metadata}
    for item in metadata:
        hashers[item.episode_index] = {
            field: _new_array_hasher(
                field,
                length=item.length,
                dimension=dimension,
            )
            for field, dimension, _ in _SOURCE_FIELDS
        }

    columns = (
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        SOURCE_ACTION_KEY,
        SOURCE_STATE_KEY,
    )
    for relative, expected_episodes in sorted(
        by_path.items(), key=lambda item: item[1][0].dataset_from_index
    ):
        path = root / relative
        _validate_physical_data_schema(path)
        parquet_file = pq.ParquetFile(path)
        expected_rows = sum(item.length for item in expected_episodes)
        if parquet_file.metadata.num_rows != expected_rows:
            raise ValueError(
                f"Data file {path} has {parquet_file.metadata.num_rows} rows, but "
                f"episode metadata assigns {expected_rows}."
            )
        expected_episode_order = [item.episode_index for item in expected_episodes]
        active_order_index = 0
        active_episode: int | None = None
        rows_seen = 0
        for batch in parquet_file.iter_batches(
            columns=list(columns), batch_size=batch_size
        ):
            for column_index, field in enumerate(columns[:4]):
                if batch.column(column_index).null_count:
                    raise ValueError(f"Data file {path} field {field!r} contains nulls.")
            episode_indices = np.asarray(
                batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64
            )
            frame_indices = np.asarray(
                batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int64
            )
            dataset_indices = np.asarray(
                batch.column(2).to_numpy(zero_copy_only=False), dtype=np.int64
            )
            timestamps = np.asarray(
                batch.column(3).to_numpy(zero_copy_only=False), dtype=np.float32
            )
            if not np.all(np.isfinite(timestamps)):
                raise ValueError(f"Data file {path} contains non-finite timestamps.")
            source_values = {
                SOURCE_ACTION_KEY: _fixed_size_list_numpy(
                    batch.column(4),
                    dimension=SOURCE_ACTION_DIM,
                    context=f"Data file {path} {SOURCE_ACTION_KEY}",
                ),
                SOURCE_STATE_KEY: _fixed_size_list_numpy(
                    batch.column(5),
                    dimension=SOURCE_STATE_DIM,
                    context=f"Data file {path} {SOURCE_STATE_KEY}",
                ),
            }

            if len(batch) == 0:
                continue
            group_starts = np.r_[
                0,
                np.nonzero(episode_indices[1:] != episode_indices[:-1])[0] + 1,
            ]
            group_ends = np.r_[group_starts[1:], len(batch)]
            for start, end in zip(group_starts, group_ends, strict=True):
                episode_index = int(episode_indices[start])
                if episode_index not in by_episode:
                    raise ValueError(
                        f"Data file {path} contains undeclared episode {episode_index}."
                    )
                item = by_episode[episode_index]
                if item.data_relative_path != relative:
                    raise ValueError(
                        f"Episode {episode_index} rows are in {relative}, but metadata "
                        f"declares {item.data_relative_path}."
                    )
                if active_episode != episode_index:
                    if active_episode is not None:
                        active_order_index += 1
                    if (
                        active_order_index >= len(expected_episode_order)
                        or expected_episode_order[active_order_index] != episode_index
                    ):
                        raise ValueError(
                            f"Data file {path} episodes are not in contiguous metadata "
                            f"order; encountered episode {episode_index}."
                        )
                    active_episode = episode_index

                count = int(end - start)
                first_frame = next_frame[episode_index]
                expected_frames = np.arange(
                    first_frame,
                    first_frame + count,
                    dtype=np.int64,
                )
                if not np.array_equal(frame_indices[start:end], expected_frames):
                    raise ValueError(
                        f"Episode {episode_index} frame_index values are not in "
                        f"contiguous order starting at {first_frame}."
                    )
                expected_indices = item.dataset_from_index + expected_frames
                if not np.array_equal(dataset_indices[start:end], expected_indices):
                    raise ValueError(
                        f"Episode {episode_index} global index values do not match its "
                        "metadata range and frame order."
                    )
                expected_timestamps = expected_frames.astype(np.float64) / float(fps)
                if not np.allclose(
                    timestamps[start:end].astype(np.float64),
                    expected_timestamps,
                    rtol=0.0,
                    atol=1e-4,
                ):
                    raise ValueError(
                        f"Episode {episode_index} timestamps do not match frame_index / "
                        f"fps within 1e-4 seconds."
                    )
                for field, _, _ in _SOURCE_FIELDS:
                    hashers[episode_index][field].update(
                        _canonical_float_bytes(
                            source_values[field][start:end],
                            dtype=np.dtype(np.float32),
                            context=f"Episode {episode_index} {field}",
                        )
                    )
                next_frame[episode_index] += count
                if next_frame[episode_index] > item.length:
                    raise ValueError(
                        f"Episode {episode_index} contains more rows than length "
                        f"{item.length}."
                    )
            rows_seen += len(batch)
        if rows_seen != expected_rows:
            raise ValueError(
                f"Data file {path} yielded {rows_seen} rows, expected {expected_rows}."
            )
        if active_episode is not None:
            active_order_index += 1
        if active_order_index != len(expected_episode_order):
            raise ValueError(
                f"Data file {path} did not contain every declared episode in order."
            )

    result: dict[int, tuple[str, str, str]] = {}
    for item in metadata:
        episode_index = item.episode_index
        if next_frame[episode_index] != item.length:
            raise ValueError(
                f"Episode {episode_index} contains {next_frame[episode_index]} rows, "
                f"expected {item.length}."
            )
        action_sha256 = hashers[episode_index][SOURCE_ACTION_KEY].hexdigest()
        state_sha256 = hashers[episode_index][SOURCE_STATE_KEY].hexdigest()
        trajectory_record = {
            "schema": _EPISODE_CONTENT_SCHEMA,
            "fps": fps,
            "length": item.length,
            "source_action_sha256": action_sha256,
            "source_observation_state_sha256": state_sha256,
        }
        result[episode_index] = (
            action_sha256,
            state_sha256,
            _canonical_json_sha256(trajectory_record),
        )
    return result


def _content_catalog_sha256(episodes: Sequence[SourceEpisodeIdentity]) -> str:
    records = sorted(
        (episode.content_record() for episode in episodes),
        key=lambda record: (
            record["source_trajectory_sha256"],
            record["length"],
            record["source_action_sha256"],
            record["source_observation_state_sha256"],
        ),
    )
    return _canonical_json_sha256(
        {
            "schema": _CONTENT_CATALOG_SCHEMA,
            "episodes": records,
        }
    )


def _statistics_catalog_sha256(episodes: Sequence[SourceEpisodeIdentity]) -> str:
    records = sorted(
        (
            {
                "length": episode.length,
                "source_statistics_sha256": episode.source_statistics_sha256,
            }
            for episode in episodes
        ),
        key=lambda record: (
            record["source_statistics_sha256"],
            record["length"],
        ),
    )
    return _canonical_json_sha256(
        {
            "schema": _STATISTICS_CATALOG_SCHEMA,
            "episodes": records,
        }
    )


def enumerate_v3_source_identity_catalog(
    dataset_root: str | Path,
    *,
    expected_fps: int = REALMAN_FPS,
    batch_size: int = 65_536,
) -> SourceIdentityCatalog:
    """Validate and content-hash a RealMan LeRobot v3 dataset.

    Only the two source numeric modalities are read from data shards.  Video
    bytes, derived 14-D modalities, tasks, and mutable dataset names do not
    affect identity.  Data are processed in bounded Arrow batches.
    """

    if isinstance(expected_fps, bool) or not isinstance(expected_fps, int):
        raise ValueError(f"expected_fps must be an integer, got {expected_fps!r}.")
    if expected_fps <= 0:
        raise ValueError(f"expected_fps must be positive, got {expected_fps}.")
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset root does not exist: {root}")
    info, expected_episode_count, expected_frame_count = _validate_info(
        root,
        expected_fps=expected_fps,
    )
    metadata = _read_episode_metadata(
        root,
        info=info,
        expected_episode_count=expected_episode_count,
        expected_frame_count=expected_frame_count,
    )
    trajectory_hashes = _trajectory_hashes(
        root,
        metadata,
        fps=expected_fps,
        batch_size=batch_size,
    )
    episodes: list[SourceEpisodeIdentity] = []
    for item in metadata:
        action_sha256, state_sha256, trajectory_sha256 = trajectory_hashes[
            item.episode_index
        ]
        episodes.append(
            SourceEpisodeIdentity(
                locator=EpisodeLocator(
                    dataset_root=root,
                    episode_index=item.episode_index,
                    data_relative_path=item.data_relative_path,
                    dataset_from_index=item.dataset_from_index,
                    dataset_to_index=item.dataset_to_index,
                ),
                fps=expected_fps,
                length=item.length,
                source_action_sha256=action_sha256,
                source_observation_state_sha256=state_sha256,
                source_trajectory_sha256=trajectory_sha256,
                source_statistics_sha256=item.statistics_sha256,
            )
        )
    episode_tuple = tuple(episodes)
    return SourceIdentityCatalog(
        dataset_root=root,
        fps=expected_fps,
        episode_count=len(episode_tuple),
        frame_count=sum(episode.length for episode in episode_tuple),
        episodes=episode_tuple,
        content_catalog_sha256=_content_catalog_sha256(episode_tuple),
        statistics_catalog_sha256=_statistics_catalog_sha256(episode_tuple),
    )


def detect_selected_source_overlaps(
    training: SourceIdentityCatalog,
    evaluation: SourceIdentityCatalog,
    selected_episode_indices: Iterable[int] | None = None,
    *,
    include_state_only_matches: bool = True,
) -> tuple[SourceEpisodeOverlap, ...]:
    """Find selected evaluation trajectories already present in training.

    Exact action+state trajectory matches are authoritative duplicates.  By
    default an exact full state-sequence match is also reported conservatively
    when actions differ, because it strongly indicates a copied recording.
    Statistics-only and action-only matches are never treated as identity.
    """

    if training.fps != evaluation.fps:
        raise ValueError(
            f"Training/evaluation fps differ: {training.fps} versus {evaluation.fps}."
        )
    evaluation_by_index = {
        episode.locator.episode_index: episode for episode in evaluation.episodes
    }
    if selected_episode_indices is None:
        selected = list(evaluation.episodes)
    else:
        requested: list[int] = []
        for raw_episode_index in selected_episode_indices:
            if isinstance(raw_episode_index, bool) or not isinstance(
                raw_episode_index, Integral
            ):
                raise ValueError("Selected episode indices must be integers, not bools.")
            episode_index = int(raw_episode_index)
            if episode_index not in evaluation_by_index:
                raise ValueError(
                    f"Selected evaluation episode {episode_index} is not in "
                    f"{evaluation.dataset_root}."
                )
            requested.append(episode_index)
        if len(set(requested)) != len(requested):
            raise ValueError("Selected evaluation episode indices contain duplicates.")
        selected = [evaluation_by_index[index] for index in requested]

    by_trajectory: dict[str, list[EpisodeLocator]] = defaultdict(list)
    by_state: dict[str, list[EpisodeLocator]] = defaultdict(list)
    for episode in training.episodes:
        by_trajectory[episode.source_trajectory_sha256].append(episode.locator)
        by_state[episode.source_observation_state_sha256].append(episode.locator)

    overlaps: list[SourceEpisodeOverlap] = []
    for episode in selected:
        exact = by_trajectory.get(episode.source_trajectory_sha256, ())
        if exact:
            overlaps.append(
                SourceEpisodeOverlap(
                    match_kind="source_trajectory",
                    digest=episode.source_trajectory_sha256,
                    evaluation=episode.locator,
                    training=tuple(exact),
                )
            )
            continue
        if include_state_only_matches:
            state_matches = by_state.get(
                episode.source_observation_state_sha256,
                (),
            )
            if state_matches:
                overlaps.append(
                    SourceEpisodeOverlap(
                        match_kind="source_observation_state",
                        digest=episode.source_observation_state_sha256,
                        evaluation=episode.locator,
                        training=tuple(state_matches),
                    )
                )
    return tuple(overlaps)


def _validate_catalog_proof_binding(
    payload: Any,
    *,
    label: str,
    catalog: SourceIdentityCatalog,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"holdout_proof.{label} must be an object.")
    expected: dict[str, int | str] = {
        "episode_count": catalog.episode_count,
        "frame_count": catalog.frame_count,
        "source_content_catalog_sha256": catalog.content_catalog_sha256,
        "source_statistics_catalog_sha256": catalog.statistics_catalog_sha256,
    }
    for field, expected_value in expected.items():
        actual = payload.get(field)
        if isinstance(expected_value, int):
            if isinstance(actual, bool) or not isinstance(actual, int):
                raise ValueError(
                    f"holdout_proof.{label}.{field} must be an integer, got "
                    f"{actual!r}."
                )
        elif not isinstance(actual, str):
            raise ValueError(
                f"holdout_proof.{label}.{field} must be a string, got "
                f"{actual!r}."
            )
        if actual != expected_value:
            raise ValueError(
                f"holdout_proof.{label}.{field} does not match the independently "
                f"enumerated dataset: manifest={actual!r}, "
                f"computed={expected_value!r}."
            )


def validate_holdout_proof(
    manifest_proof: Mapping[str, Any],
    training_catalog: SourceIdentityCatalog,
    evaluation_catalog: SourceIdentityCatalog,
    selected_episode_indices: Iterable[int] | None,
) -> bool:
    """Validate all manifest bindings and prove selected source disjointness.

    The caller supplies catalogs freshly enumerated from the training and
    evaluation roots.  This function does not trust directory names, episode
    indices as identity, or manifest-provided hashes.  It raises on the first
    mismatch and returns ``True`` only after exact selected-record matching and
    zero source trajectory/state overlaps.
    """

    if not isinstance(manifest_proof, Mapping):
        raise ValueError("holdout_proof must be an object.")
    algorithm = manifest_proof.get("episode_identity_algorithm")
    if algorithm != _EPISODE_CONTENT_SCHEMA:
        raise ValueError(
            "holdout_proof.episode_identity_algorithm does not match this "
            f"verifier: manifest={algorithm!r}, expected={_EPISODE_CONTENT_SCHEMA!r}."
        )
    _validate_catalog_proof_binding(
        manifest_proof.get("training"),
        label="training",
        catalog=training_catalog,
    )
    _validate_catalog_proof_binding(
        manifest_proof.get("evaluation"),
        label="evaluation",
        catalog=evaluation_catalog,
    )

    evaluation_by_index = {
        episode.locator.episode_index: episode
        for episode in evaluation_catalog.episodes
    }
    if selected_episode_indices is None:
        selected_indices = sorted(evaluation_by_index)
    else:
        selected_indices = []
        for raw_episode_index in selected_episode_indices:
            if isinstance(raw_episode_index, bool) or not isinstance(
                raw_episode_index, Integral
            ):
                raise ValueError(
                    "Selected evaluation episode indices must be integers."
                )
            episode_index = int(raw_episode_index)
            if episode_index not in evaluation_by_index:
                raise ValueError(
                    f"Selected evaluation episode {episode_index} is not in "
                    f"{evaluation_catalog.dataset_root}."
                )
            selected_indices.append(episode_index)
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("Selected evaluation episode indices contain duplicates.")
        selected_indices.sort()
    if not selected_indices:
        raise ValueError("At least one evaluation episode must be selected.")

    raw_manifest_records = manifest_proof.get("selected_evaluation_episodes")
    if not isinstance(raw_manifest_records, list):
        raise ValueError(
            "holdout_proof.selected_evaluation_episodes must be a list."
        )
    if not all(isinstance(record, Mapping) for record in raw_manifest_records):
        raise ValueError(
            "Every holdout_proof.selected_evaluation_episodes entry must be an object."
        )
    expected_records = sorted(
        (
            evaluation_by_index[episode_index].manifest_record()
            for episode_index in selected_indices
        ),
        key=lambda record: int(record["episode_index"]),
    )
    actual_records = sorted(
        (dict(record) for record in raw_manifest_records),
        key=lambda record: (
            record.get("episode_index")
            if isinstance(record.get("episode_index"), int)
            and not isinstance(record.get("episode_index"), bool)
            else -1
        ),
    )
    if actual_records != expected_records:
        raise ValueError(
            "holdout_proof.selected_evaluation_episodes does not exactly match "
            f"the selected computed identities: manifest={actual_records!r}, "
            f"computed={expected_records!r}."
        )

    overlaps = detect_selected_source_overlaps(
        training_catalog,
        evaluation_catalog,
        selected_indices,
        include_state_only_matches=True,
    )
    if overlaps:
        summaries = [
            {
                "match_kind": overlap.match_kind,
                "evaluation_episode_index": overlap.evaluation.episode_index,
                "training_episode_indices": [
                    locator.episode_index for locator in overlap.training
                ],
                "digest": overlap.digest,
            }
            for overlap in overlaps
        ]
        raise ValueError(
            "Selected evaluation episodes overlap training source trajectories: "
            f"{summaries!r}."
        )
    return True


__all__ = [
    "EpisodeLocator",
    "REALMAN_FPS",
    "SOURCE_ACTION_DIM",
    "SOURCE_ACTION_KEY",
    "SOURCE_ACTION_NAMES",
    "SOURCE_STATE_DIM",
    "SOURCE_STATE_KEY",
    "SOURCE_STATE_NAMES",
    "SourceEpisodeIdentity",
    "SourceEpisodeOverlap",
    "SourceIdentityCatalog",
    "detect_selected_source_overlaps",
    "enumerate_v3_source_identity_catalog",
    "validate_holdout_proof",
]
