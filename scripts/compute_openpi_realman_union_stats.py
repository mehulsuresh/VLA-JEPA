#!/usr/bin/env python3
"""Build one immutable, deduplicated OpenPI-compatible RealMan 18-D table.

The command consumes a frozen population manifest.  A reader backend turns
each manifest episode into selected absolute 18-D state/action arrays plus
availability masks.  Statistics are then computed directly from the union of
unique base frames; per-dataset quantiles are never averaged.

The built-in ``npz_episodes`` backend is deliberately small and deterministic.
It is suitable for materialized production views and parity fixtures.  GCS or
other streaming readers can register the same EpisodeReference interface
without changing the union, deduplication, or statistics implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
import uuid

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DATASET_VIEW_PATH = REPO_ROOT / "starVLA/dataloader/dataset_view.py"
_DATASET_VIEW_SPEC = importlib.util.spec_from_file_location(
    "_realman_union_dataset_view", _DATASET_VIEW_PATH
)
if _DATASET_VIEW_SPEC is None or _DATASET_VIEW_SPEC.loader is None:
    raise ImportError(f"Could not load dataset-view helpers from {_DATASET_VIEW_PATH}")
dataset_view = importlib.util.module_from_spec(_DATASET_VIEW_SPEC)
sys.modules[_DATASET_VIEW_SPEC.name] = dataset_view
_DATASET_VIEW_SPEC.loader.exec_module(dataset_view)

from starVLA.action_representation import (  # noqa: E402
    OPENPI_REALMAN_EPISODE_CONTENT_SCHEMA,
    OPENPI_REALMAN_UNION_DEDUP_ALGORITHM,
    OPENPI_REALMAN_UNION_LEDGER_SCHEMA,
    OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
    OPENPI_REALMAN_UNION_STATISTIC_NAMES,
    OPENPI_REALMAN_UNION_STATISTICS_SCHEMA,
    PiCompatibleMaskedRunningStats,
    Q01_Q99_UNCLIPPED,
    REALMAN_18D_ACTION_CONTRACT,
    REALMAN_ACTION_HORIZON,
    REALMAN_ACTION_SOURCE_INDICES,
    REALMAN_POLICY_DIM,
    REALMAN_STATE_SOURCE_INDICES,
    deterministic_json_bytes,
    encode_actions,
    select_canonical_realman_policy_action_mask,
    select_canonical_realman_policy_actions,
    select_canonical_realman_policy_state,
    select_canonical_realman_policy_state_mask,
    select_realman_policy_actions,
    select_realman_policy_state,
    serialize_openpi_realman_union_statistics,
)
from starVLA.realman_union_holdout import (  # noqa: E402
    REALMAN_UNION_HOLDOUT_SCHEMA,
    validate_global_holdout_manifest,
)


NPZ_READER_KIND = "npz_episodes"
FROZEN_PARQUET_READER_KIND = "frozen_parquet_view"
SUPPORTED_NPZ_REPRESENTATIONS = {
    "policy18",
    "lerobot_realman",
    "canonical_realman",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest.")
    return value


def _as_mask(mask: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError(f"{label} shape {array.shape} does not match {shape}.")
    if array.dtype != np.bool_:
        if not np.isin(array, (0, 1)).all():
            raise ValueError(f"{label} must contain only booleans.")
        array = array.astype(bool, copy=False)
    return np.ascontiguousarray(array, dtype=bool)


def _select_lerobot_mask(mask: Any, *, modality: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 0:
        raise ValueError(f"LeRobot RealMan {modality} mask has no feature axis.")
    if modality == "state":
        if array.shape[-1] == REALMAN_POLICY_DIM:
            indices = tuple(range(REALMAN_POLICY_DIM))
        elif array.shape[-1] in (19, 21):
            indices = REALMAN_STATE_SOURCE_INDICES
        else:
            raise ValueError(
                "LeRobot RealMan state mask must be 18-D, 19-D, or 21-D; "
                f"got {array.shape}."
            )
    elif modality == "action":
        if array.shape[-1] == REALMAN_POLICY_DIM:
            indices = tuple(range(REALMAN_POLICY_DIM))
        elif array.shape[-1] == 22:
            indices = REALMAN_ACTION_SOURCE_INDICES
        else:
            raise ValueError(
                "LeRobot RealMan action mask must be 18-D or 22-D; "
                f"got {array.shape}."
            )
    else:  # pragma: no cover - internal invariant
        raise AssertionError(modality)
    return np.ascontiguousarray(
        array[..., np.asarray(indices, dtype=np.int64)],
        dtype=bool,
    )


@dataclass(frozen=True, slots=True)
class PopulationEpisode:
    """One fully projected absolute episode used by the union builder."""

    key: str
    source_id: str
    episode_id: str
    state: np.ndarray
    action: np.ndarray
    state_mask: np.ndarray
    action_mask: np.ndarray
    base_frame_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EpisodeReference:
    """Frozen reference returned by a population reader backend."""

    key: str
    source_id: str
    episode_id: str
    path_text: str
    path: Path
    expected_sha256: str
    representation: str
    raw_base_frame_indices: Any
    duplicate_of: str | None

    def load(self) -> PopulationEpisode:
        if not self.path.is_file():
            raise FileNotFoundError(f"Population episode is missing: {self.path}")
        actual_sha256 = _file_sha256(self.path)
        if actual_sha256 != self.expected_sha256:
            raise ValueError(
                f"Population episode {self.key} SHA-256 mismatch: expected "
                f"{self.expected_sha256}, got {actual_sha256}."
            )
        try:
            with np.load(self.path, allow_pickle=False) as payload:
                raw_state = np.asarray(payload["state"])
                raw_action = np.asarray(payload["action"])
                raw_state_mask = (
                    np.asarray(payload["state_mask"])
                    if "state_mask" in payload
                    else np.ones(raw_state.shape, dtype=bool)
                )
                raw_action_mask = (
                    np.asarray(payload["action_mask"])
                    if "action_mask" in payload
                    else np.ones(raw_action.shape, dtype=bool)
                )
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"Could not load frozen population episode {self.key}: {self.path}: {exc}"
            ) from exc

        if raw_state.ndim != 2 or raw_action.ndim != 2:
            raise ValueError(
                f"Population episode {self.key} state/action must be rank two."
            )
        if raw_state.shape[0] <= 0 or raw_state.shape[0] != raw_action.shape[0]:
            raise ValueError(
                f"Population episode {self.key} has inconsistent lengths: "
                f"{raw_state.shape} vs {raw_action.shape}."
            )
        raw_state_mask = _as_mask(
            raw_state_mask, shape=raw_state.shape, label=f"{self.key} state_mask"
        )
        raw_action_mask = _as_mask(
            raw_action_mask, shape=raw_action.shape, label=f"{self.key} action_mask"
        )

        if self.representation == "policy18":
            if (
                raw_state.shape[-1] != REALMAN_POLICY_DIM
                or raw_action.shape[-1] != REALMAN_POLICY_DIM
            ):
                raise ValueError(
                    f"policy18 episode {self.key} must have 18-D state/action."
                )
            state = np.asarray(raw_state, dtype=np.float32)
            action = np.asarray(raw_action, dtype=np.float32)
            state_mask = raw_state_mask
            action_mask = raw_action_mask
        elif self.representation == "lerobot_realman":
            state = select_realman_policy_state(raw_state)
            action = select_realman_policy_actions(raw_action)
            state_mask = _select_lerobot_mask(raw_state_mask, modality="state")
            action_mask = _select_lerobot_mask(raw_action_mask, modality="action")
        elif self.representation == "canonical_realman":
            state = select_canonical_realman_policy_state(raw_state)
            action = select_canonical_realman_policy_actions(raw_action)
            state_mask = select_canonical_realman_policy_state_mask(raw_state_mask)
            action_mask = select_canonical_realman_policy_action_mask(raw_action_mask)
        else:  # pragma: no cover - validated before construction
            raise AssertionError(self.representation)

        if not np.isfinite(state[state_mask]).all():
            raise ValueError(f"Population episode {self.key} has non-finite state.")
        if not np.isfinite(action[action_mask]).all():
            raise ValueError(f"Population episode {self.key} has non-finite action.")
        # Masked padding is semantically absent. Canonicalize it to zero so
        # content hashes do not depend on arbitrary source padding bytes.
        state = np.ascontiguousarray(
            np.where(state_mask, state, np.float32(0.0)), dtype=np.float32
        )
        action = np.ascontiguousarray(
            np.where(action_mask, action, np.float32(0.0)), dtype=np.float32
        )
        state_mask = np.ascontiguousarray(state_mask, dtype=bool)
        action_mask = np.ascontiguousarray(action_mask, dtype=bool)

        if self.raw_base_frame_indices in (None, "all"):
            base_frame_indices = tuple(range(state.shape[0]))
        else:
            if not isinstance(self.raw_base_frame_indices, list):
                raise ValueError(
                    f"{self.key}.base_frame_indices must be 'all' or a list."
                )
            values: list[int] = []
            for raw_index in self.raw_base_frame_indices:
                if (
                    isinstance(raw_index, bool)
                    or not isinstance(raw_index, int)
                    or raw_index < 0
                    or raw_index >= state.shape[0]
                ):
                    raise ValueError(
                        f"{self.key} has invalid base frame {raw_index!r}."
                    )
                values.append(raw_index)
            if values != sorted(set(values)):
                raise ValueError(
                    f"{self.key}.base_frame_indices must be sorted and unique."
                )
            base_frame_indices = tuple(values)
        if not base_frame_indices:
            raise ValueError(f"Population episode {self.key} selects no base frames.")

        return PopulationEpisode(
            key=self.key,
            source_id=self.source_id,
            episode_id=self.episode_id,
            state=state,
            action=action,
            state_mask=state_mask,
            action_mask=action_mask,
            base_frame_indices=base_frame_indices,
        )


class PopulationReader(Protocol):
    def references(self) -> Iterator["PopulationReference"]: ...


class PopulationReference(Protocol):
    key: str
    source_id: str
    episode_id: str
    path_text: str
    expected_sha256: str
    duplicate_of: str | None

    def load(self) -> PopulationEpisode: ...


PopulationReaderFactory = Callable[[Mapping[str, Any], Path], PopulationReader]
_POPULATION_READER_FACTORIES: dict[str, PopulationReaderFactory] = {}


def register_population_reader(
    kind: str,
    factory: PopulationReaderFactory,
) -> None:
    """Register an explicit population reader backend."""

    if not isinstance(kind, str) or not kind:
        raise ValueError("Population reader kind must be a non-empty string.")
    if kind in _POPULATION_READER_FACTORIES:
        raise ValueError(f"Population reader {kind!r} is already registered.")
    _POPULATION_READER_FACTORIES[kind] = factory


class NpzEpisodeReader:
    def __init__(self, source: Mapping[str, Any], manifest_dir: Path) -> None:
        self.source = source
        self.manifest_dir = manifest_dir

    def references(self) -> Iterator[EpisodeReference]:
        source_id = str(self.source["id"])
        reader = self.source["reader"]
        representation = reader.get("representation")
        if representation not in SUPPORTED_NPZ_REPRESENTATIONS:
            raise ValueError(
                f"Source {source_id!r} representation must be one of "
                f"{sorted(SUPPORTED_NPZ_REPRESENTATIONS)}."
            )
        episodes = reader.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"Source {source_id!r} reader.episodes must be non-empty.")
        raw_ids = [
            episode.get("id") if isinstance(episode, Mapping) else None
            for episode in episodes
        ]
        if (
            any(not isinstance(value, str) or not value for value in raw_ids)
            or raw_ids != sorted(set(raw_ids))
        ):
            raise ValueError(
                f"Source {source_id!r} episode IDs must be sorted, unique strings."
            )
        for index, episode in enumerate(episodes):
            if not isinstance(episode, Mapping):
                raise ValueError(
                    f"Source {source_id!r} episode {index} must be an object."
                )
            required = {"id", "path", "sha256"}
            allowed = required | {"base_frame_indices", "duplicate_of"}
            if set(episode) - allowed or not required.issubset(episode):
                raise ValueError(
                    f"Source {source_id!r} episode {index} fields are invalid."
                )
            path_text = episode["path"]
            if not isinstance(path_text, str) or not path_text:
                raise ValueError(f"Source {source_id!r} episode path is invalid.")
            path = Path(path_text).expanduser()
            if not path.is_absolute():
                path = self.manifest_dir / path
            duplicate_of = episode.get("duplicate_of")
            if duplicate_of is not None and (
                not isinstance(duplicate_of, str) or "/" not in duplicate_of
            ):
                raise ValueError(
                    f"Source {source_id!r} episode duplicate_of is invalid."
                )
            yield EpisodeReference(
                key=f"{source_id}/{episode['id']}",
                source_id=source_id,
                episode_id=episode["id"],
                path_text=path_text,
                path=path.resolve(),
                expected_sha256=_require_sha256(
                    episode["sha256"],
                    label=f"{source_id}/{episode['id']}.sha256",
                ),
                representation=representation,
                raw_base_frame_indices=episode.get("base_frame_indices", "all"),
                duplicate_of=duplicate_of,
            )


def _npz_reader_factory(
    source: Mapping[str, Any],
    manifest_dir: Path,
) -> PopulationReader:
    return NpzEpisodeReader(source, manifest_dir)


register_population_reader(NPZ_READER_KIND, _npz_reader_factory)


def _stack_vector_column(series: pd.Series, *, label: str) -> np.ndarray:
    values = series.tolist()
    if not values:
        raise ValueError(f"{label} is empty.")
    try:
        array = np.stack(
            [np.asarray(value, dtype=np.float32) for value in values],
            axis=0,
        )
    except ValueError as exc:
        raise ValueError(f"{label} contains ragged vectors.") from exc
    if array.ndim != 2:
        raise ValueError(f"{label} must be a rank-two vector column.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _target_source_indices(
    *,
    source_length: int,
    source_fps: float,
    target_fps: int,
) -> np.ndarray:
    if source_length <= 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("Episode length/FPS must be positive.")
    # One extra estimate is harmless; filter by the exact half-up mapping.
    estimated = int(np.ceil(source_length * target_fps / source_fps)) + 2
    target = np.arange(estimated, dtype=np.float64)
    source = np.floor(target * source_fps / target_fps + 0.5).astype(
        np.int64
    )
    source = source[source < source_length]
    if not len(source) or np.any(source[1:] <= source[:-1]):
        raise ValueError(
            "Source/target FPS mapping is empty or not strictly increasing."
        )
    return source


def _ensure_gcs_parquet(
    *,
    local_path: Path,
    gcs_path: str,
    allow_download: bool,
    timeout_seconds: int,
) -> Path:
    if local_path.is_file():
        return local_path
    if not allow_download:
        raise FileNotFoundError(
            f"Frozen population shard is not cached: {local_path}; "
            "reader.allow_gcs_download=false."
        )
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid frozen population GCS path: {gcs_path!r}.")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_name(
        f".{local_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        completed = subprocess.run(
            ["gcloud", "storage", "cp", gcs_path, str(temporary)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to copy {gcs_path}: {completed.stderr.strip()}"
            )
        os.replace(temporary, local_path)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out copying frozen population shard {gcs_path}."
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return local_path


@dataclass(frozen=True, slots=True)
class FrozenParquetEpisodeReference:
    key: str
    source_id: str
    episode_id: str
    path_text: str
    expected_sha256: str
    duplicate_of: str | None
    reader: "FrozenParquetViewReader"
    backend: str
    view_source_id: str
    dataset_id: str
    sid: str
    revision: str
    episode_index: int
    source_episode_length: int
    source_fps: float
    target_fps: int
    data_file: str
    base_frame_indices: range | tuple[int, ...]

    def load(self) -> PopulationEpisode:
        return self.reader.load_episode(self)


class FrozenParquetViewReader:
    """Read numeric episode payloads referenced by a frozen range view.

    Canonical RealSource files are loaded from the canonical cache or copied
    from their descriptor-bound GCS prefix.  Local LeRobot files are loaded
    from a configured root.  Only numeric state/action columns are read; video
    payloads are intentionally outside normalization statistics.
    """

    def __init__(self, source: Mapping[str, Any], manifest_dir: Path) -> None:
        self.source = source
        self.manifest_dir = manifest_dir
        self.reader_config = dict(source["reader"])
        allowed = {
            "kind",
            "view_manifest",
            "view_manifest_sha256",
            "cache_dir",
            "dataset_root",
            "allow_gcs_download",
            "gcs_download_timeout_seconds",
            "duplicate_of",
        }
        unknown = sorted(set(self.reader_config) - allowed)
        if unknown:
            raise ValueError(
                f"Frozen parquet reader has unknown fields: {unknown}."
            )
        raw_view = self.reader_config.get("view_manifest")
        if not isinstance(raw_view, str) or not raw_view:
            raise ValueError("frozen_parquet_view requires view_manifest.")
        view_path = Path(raw_view).expanduser()
        if not view_path.is_absolute():
            view_path = manifest_dir / view_path
        self.view_path = view_path.resolve()
        expected_view_sha256 = _require_sha256(
            self.reader_config.get("view_manifest_sha256"),
            label="frozen_parquet_view.view_manifest_sha256",
        )
        actual_view_sha256 = _file_sha256(self.view_path)
        if actual_view_sha256 != expected_view_sha256:
            raise ValueError(
                "Frozen parquet view manifest SHA-256 mismatch: expected "
                f"{expected_view_sha256}, got {actual_view_sha256}."
            )
        self.view = dataset_view.load_frozen_view(
            self.view_path,
            expected_representation_contract_sha256=(
                REALMAN_18D_ACTION_CONTRACT.sha256()
            ),
            verify_ledger=True,
        )
        if (
            self.view.descriptor.get("purpose")
            != dataset_view.STATISTICS_POPULATION_CANDIDATE_PURPOSE
        ):
            raise ValueError(
                "frozen_parquet_view union sources must use a separately "
                "materialized statistics_population_candidate view; "
                "holdout-free training views are not valid statistics "
                "population sources."
            )
        self.view_sources = {
            str(item["source_id"]): item
            for item in self.view.descriptor["sources"]
        }
        source_catalog_hashes = {
            str(item["catalog_sha256"])
            for item in self.view.descriptor["sources"]
        }
        if len(source_catalog_hashes) != 1:
            raise ValueError(
                "Frozen parquet view must have one shared source catalog hash."
            )
        if next(iter(source_catalog_hashes)) != source["catalog_sha256"]:
            raise ValueError(
                "Union source catalog SHA does not match frozen view."
            )
        self.cache_dir = Path(
            self.reader_config.get(
                "cache_dir",
                "/home/mehul/work/dataset-canonicalization/"
                ".cache/gcs_lerobot",
            )
        ).expanduser().resolve()
        raw_dataset_root = self.reader_config.get("dataset_root")
        self.dataset_root = (
            None
            if raw_dataset_root is None
            else Path(str(raw_dataset_root)).expanduser().resolve()
        )
        self.allow_gcs_download = bool(
            self.reader_config.get("allow_gcs_download", False)
        )
        self.timeout_seconds = int(
            self.reader_config.get("gcs_download_timeout_seconds", 900)
        )
        if self.timeout_seconds <= 0:
            raise ValueError("gcs_download_timeout_seconds must be positive.")
        duplicate_map = self.reader_config.get("duplicate_of", {})
        if not isinstance(duplicate_map, Mapping):
            raise ValueError("frozen_parquet_view.duplicate_of must be an object.")
        self.duplicate_map = {
            str(key): str(value) for key, value in duplicate_map.items()
        }
        self._file_hashes: dict[Path, str] = {}
        self._file_stats: dict[Path, tuple[int, int, int]] = {}
        self._loaded_path: Path | None = None
        self._loaded_frame: pd.DataFrame | None = None

    def _iter_range_rows(self) -> Iterator[dict[str, Any]]:
        with self.view.ledger_path.open("rb") as handle:
            for ordinal, raw_line in enumerate(handle):
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Frozen range ledger row {ordinal} is invalid JSON."
                    ) from exc
                schema = row.get("schema")
                if schema == getattr(
                    dataset_view,
                    "RANGE_ROW_SCHEMA",
                    "vla-dataset-view-range-row-v1",
                ):
                    yield row
                    continue
                if schema == dataset_view.ROW_SCHEMA:
                    # Backward-compatible grouping is handled by references().
                    yield row
                    continue
                raise ValueError(
                    f"Unsupported frozen parquet ledger schema {schema!r}."
                )

    def _resolve_file(
        self,
        *,
        backend: str,
        source_descriptor: Mapping[str, Any],
        sid: str,
        revision: str,
        data_file: str,
    ) -> tuple[Path, str]:
        if backend == "canonical":
            local_path = self.cache_dir / sid / revision / data_file
            gcs_prefix = source_descriptor.get("gcs_prefix")
            if not isinstance(gcs_prefix, str) or not gcs_prefix.startswith(
                "gs://"
            ):
                raise ValueError(
                    f"Canonical source {sid!r} lacks a valid gcs_prefix."
                )
            gcs_path = f"{gcs_prefix.rstrip('/')}/{data_file}"
            path = _ensure_gcs_parquet(
                local_path=local_path,
                gcs_path=gcs_path,
                allow_download=self.allow_gcs_download,
                timeout_seconds=self.timeout_seconds,
            )
            path_text = gcs_path
        elif backend == "lerobot":
            root = self.dataset_root
            if root is None:
                hint = source_descriptor.get("dataset_root_hint")
                if not isinstance(hint, str) or not hint:
                    raise ValueError(
                        "LeRobot frozen source requires reader.dataset_root or "
                        "descriptor dataset_root_hint."
                    )
                root = Path(hint).expanduser().resolve()
            path = root / data_file
            if not path.is_file():
                raise FileNotFoundError(
                    f"LeRobot frozen population shard is missing: {path}"
                )
            path_text = str(path)
        else:
            raise ValueError(f"Unsupported frozen parquet backend {backend!r}.")
        return path, path_text

    def _reference_from_range(
        self, row: Mapping[str, Any]
    ) -> FrozenParquetEpisodeReference:
        view_source_id = str(row["source_id"])
        descriptor = self.view_sources[view_source_id]
        backend = str(row["backend"])
        if backend != str(descriptor["backend"]):
            raise ValueError(
                f"Frozen row backend {backend!r} disagrees with source "
                f"descriptor backend {descriptor['backend']!r}."
            )
        dataset_id = str(
            row.get(
                "dataset_id",
                descriptor.get("dataset_id", view_source_id),
            )
        )
        sid = str(row.get("sid", descriptor.get("sid", "")))
        revision = str(row.get("revision", descriptor.get("revision", "")))
        if not dataset_id:
            raise ValueError("Frozen parquet range lacks dataset identity.")
        if backend == "canonical" and (not sid or not revision):
            raise ValueError("Frozen canonical range lacks dataset identity.")
        if backend == "canonical":
            expected_adapter = str(descriptor.get("adapter_sha256", ""))
            observed_adapter = str(row.get("adapter_sha256", ""))
            if (
                not expected_adapter
                or observed_adapter != expected_adapter
            ):
                raise ValueError(
                    "Frozen canonical range adapter SHA does not match its "
                    "source descriptor."
                )
        episode_index = int(row["episode_index"])
        data_file = str(row.get("data_file") or "")
        if not data_file:
            raise ValueError("Frozen parquet range lacks data_file.")
        source_episode_length = int(
            row.get("source_episode_length", row.get("episode_length", 0))
        )
        if source_episode_length <= 0:
            raise ValueError("Frozen parquet range lacks source episode length.")
        source_fps = float(row.get("source_fps", descriptor.get("fps", 0)))
        if (
            not math.isfinite(source_fps)
            or source_fps <= 0
            or source_fps != float(descriptor.get("fps", source_fps))
        ):
            raise ValueError(
                "Frozen parquet range source_fps does not match its source "
                "descriptor."
            )
        target_fps = int(row["target_fps"])
        if "base_start" in row:
            base_start = int(row["base_start"])
            base_stop = int(row["base_stop"])
            base_step = int(row["base_step"])
            base_indices: range | tuple[int, ...] = range(
                base_start, base_stop, base_step
            )
        else:
            base_indices = (int(row["base_index"]),)
        local_path, path_text = self._resolve_file(
            backend=backend,
            source_descriptor=descriptor,
            sid=sid,
            revision=revision,
            data_file=data_file,
        )
        expected_sha256 = self._file_hashes.get(local_path)
        if expected_sha256 is None:
            expected_sha256 = _file_sha256(local_path)
            self._file_hashes[local_path] = expected_sha256
            stat = local_path.stat()
            self._file_stats[local_path] = (
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            )
        episode_id = (
            f"{dataset_id}@{revision}:{episode_index}"
            if revision
            else f"{dataset_id}:{episode_index}"
        )
        key = f"{self.source['id']}/{episode_id}"
        return FrozenParquetEpisodeReference(
            key=key,
            source_id=str(self.source["id"]),
            episode_id=episode_id,
            path_text=path_text,
            expected_sha256=expected_sha256,
            duplicate_of=self.duplicate_map.get(episode_id),
            reader=self,
            backend=backend,
            view_source_id=view_source_id,
            dataset_id=dataset_id,
            sid=sid,
            revision=revision,
            episode_index=episode_index,
            source_episode_length=source_episode_length,
            source_fps=source_fps,
            target_fps=target_fps,
            data_file=data_file,
            base_frame_indices=base_indices,
        )

    def references(self) -> Iterator[PopulationReference]:
        pending: dict[str, Any] | None = None
        for row in self._iter_range_rows():
            if "base_start" in row:
                if pending is not None:
                    yield self._reference_from_range(pending)
                    pending = None
                yield self._reference_from_range(row)
                continue
            # Legacy verbose rows are contiguous by episode. Collapse them
            # without retaining the full view.
            identity = (
                row.get("source_id"),
                row.get("dataset_id"),
                row.get("sid"),
                row.get("revision"),
                row.get("episode_index"),
            )
            if pending is None:
                pending = dict(row)
                pending["_identity"] = identity
                pending["_indices"] = [int(row["base_index"])]
            elif pending["_identity"] == identity:
                pending["_indices"].append(int(row["base_index"]))
            else:
                converted = dict(pending)
                indices = converted.pop("_indices")
                converted.pop("_identity")
                converted["base_start"] = indices[0]
                converted["base_stop"] = indices[-1] + 1
                converted["base_step"] = 1
                yield self._reference_from_range(converted)
                pending = dict(row)
                pending["_identity"] = identity
                pending["_indices"] = [int(row["base_index"])]
        if pending is not None:
            converted = dict(pending)
            indices = converted.pop("_indices")
            converted.pop("_identity")
            converted["base_start"] = indices[0]
            converted["base_stop"] = indices[-1] + 1
            converted["base_step"] = 1
            yield self._reference_from_range(converted)

    def _load_shard(self, path: Path) -> pd.DataFrame:
        if self._loaded_path == path and self._loaded_frame is not None:
            return self._loaded_frame
        names = set(pq.read_schema(path).names)
        state_column = (
            "source.observation.state"
            if "source.observation.state" in names
            else "observation.state"
        )
        action_column = "source.action" if "source.action" in names else "action"
        required = {"episode_index", "frame_index", state_column, action_column}
        missing = sorted(required - names)
        if missing:
            raise ValueError(
                f"Frozen population shard {path} lacks columns {missing}."
            )
        frame = pq.read_table(
            path,
            columns=["episode_index", "frame_index", state_column, action_column],
        ).to_pandas()
        frame.attrs["state_column"] = state_column
        frame.attrs["action_column"] = action_column
        self._loaded_path = path
        self._loaded_frame = frame
        return frame

    def load_episode(
        self, reference: FrozenParquetEpisodeReference
    ) -> PopulationEpisode:
        local_path, _ = self._resolve_file(
            backend=reference.backend,
            source_descriptor=self.view_sources[reference.view_source_id],
            sid=reference.sid,
            revision=reference.revision,
            data_file=reference.data_file,
        )
        stat = local_path.stat()
        observed_stat = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        if observed_stat != self._file_stats[local_path]:
            raise ValueError(
                f"Frozen population shard changed after hashing: {local_path}."
            )
        shard = self._load_shard(local_path)
        frame = shard.loc[
            shard["episode_index"] == reference.episode_index
        ].sort_values("frame_index")
        if len(frame) != reference.source_episode_length:
            raise ValueError(
                f"Frozen episode {reference.key} length mismatch: "
                f"{len(frame)} vs {reference.source_episode_length}."
            )
        observed_frames = frame["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            observed_frames,
            np.arange(reference.source_episode_length, dtype=np.int64),
        ):
            raise ValueError(
                f"Frozen episode {reference.key} frame indices are not exact."
            )
        raw_state = _stack_vector_column(
            frame[shard.attrs["state_column"]],
            label=f"{reference.key} state",
        )
        raw_action = _stack_vector_column(
            frame[shard.attrs["action_column"]],
            label=f"{reference.key} action",
        )
        if reference.backend == "canonical":
            if raw_state.shape[1] == 71 and raw_action.shape[1] == 17:
                state = np.zeros(
                    (len(raw_state), REALMAN_POLICY_DIM), dtype=np.float32
                )
                action = np.zeros(
                    (len(raw_action), REALMAN_POLICY_DIM), dtype=np.float32
                )
                state[:, :16] = raw_state[:, :16]
                action[:, :16] = raw_action[:, :16]
                state_mask = np.zeros_like(state, dtype=bool)
                action_mask = np.zeros_like(action, dtype=bool)
                state_mask[:, :16] = True
                action_mask[:, :16] = True
            elif raw_state.shape[1] == 53 and raw_action.shape[1] == 49:
                state = select_canonical_realman_policy_state(raw_state)
                action = select_canonical_realman_policy_actions(raw_action)
                state_mask = select_canonical_realman_policy_state_mask(
                    np.ones_like(raw_state, dtype=bool)
                )
                action_mask = select_canonical_realman_policy_action_mask(
                    np.ones_like(raw_action, dtype=bool)
                )
            else:
                raise ValueError(
                    f"Unsupported canonical RealSource widths for {reference.key}: "
                    f"{raw_state.shape[1]}/{raw_action.shape[1]}."
                )
        else:
            state = select_realman_policy_state(raw_state)
            action = select_realman_policy_actions(raw_action)
            state_mask = _select_lerobot_mask(
                np.ones_like(raw_state, dtype=bool), modality="state"
            )
            action_mask = _select_lerobot_mask(
                np.ones_like(raw_action, dtype=bool), modality="action"
            )
        source_indices = _target_source_indices(
            source_length=reference.source_episode_length,
            source_fps=reference.source_fps,
            target_fps=reference.target_fps,
        )
        target_length = len(source_indices)
        if reference.base_frame_indices:
            final_base = reference.base_frame_indices[-1]
            if final_base >= target_length:
                raise ValueError(
                    f"Frozen range for {reference.key} extends beyond target "
                    f"episode length {target_length}."
                )
        return PopulationEpisode(
            key=reference.key,
            source_id=reference.source_id,
            episode_id=reference.episode_id,
            state=np.ascontiguousarray(state[source_indices], dtype=np.float32),
            action=np.ascontiguousarray(
                action[source_indices], dtype=np.float32
            ),
            state_mask=np.ascontiguousarray(
                state_mask[source_indices], dtype=bool
            ),
            action_mask=np.ascontiguousarray(
                action_mask[source_indices], dtype=bool
            ),
            base_frame_indices=reference.base_frame_indices,
        )


def _frozen_parquet_reader_factory(
    source: Mapping[str, Any],
    manifest_dir: Path,
) -> PopulationReader:
    return FrozenParquetViewReader(source, manifest_dir)


register_population_reader(
    FROZEN_PARQUET_READER_KIND, _frozen_parquet_reader_factory
)


def _episode_content_sha256(episode: PopulationEpisode) -> str:
    digest = hashlib.sha256()
    digest.update(
        deterministic_json_bytes(
            {
                "schema": OPENPI_REALMAN_EPISODE_CONTENT_SCHEMA,
                "frame_count": int(episode.state.shape[0]),
                "state_dim": REALMAN_POLICY_DIM,
                "action_dim": REALMAN_POLICY_DIM,
            }
        )
    )
    for array in (
        np.asarray(episode.state, dtype="<f4"),
        np.asarray(episode.action, dtype="<f4"),
        np.asarray(episode.state_mask, dtype=np.uint8),
        np.asarray(episode.action_mask, dtype=np.uint8),
    ):
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _validate_population_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Union population manifest root must be a JSON object.")
    required = {
        "schema",
        "contract_sha256",
        "source_order",
        "sources",
        "holdout",
    }
    if set(payload) != required:
        raise ValueError(
            f"Union population manifest must contain exactly {sorted(required)}."
        )
    if payload["schema"] != OPENPI_REALMAN_UNION_POPULATION_SCHEMA:
        raise ValueError("Union population manifest schema is invalid.")
    if payload["contract_sha256"] != REALMAN_18D_ACTION_CONTRACT.sha256():
        raise ValueError(
            "Union population manifest contract does not match RealMan 18-D."
        )
    source_order = payload["source_order"]
    sources = payload["sources"]
    if (
        not isinstance(source_order, list)
        or not source_order
        or any(not isinstance(value, str) or not value for value in source_order)
        or len(source_order) != len(set(source_order))
    ):
        raise ValueError("source_order must contain unique non-empty strings.")
    if not isinstance(sources, list):
        raise ValueError("sources must be a JSON list.")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"sources[{index}] must be an object.")
        required_source = {"id", "catalog_sha256", "reader", "provenance"}
        if set(source) != required_source:
            raise ValueError(
                f"sources[{index}] must contain exactly {sorted(required_source)}."
            )
        source_id = source["id"]
        if not isinstance(source_id, str) or not source_id or source_id in by_id:
            raise ValueError(f"sources[{index}].id is invalid or duplicated.")
        _require_sha256(
            source["catalog_sha256"],
            label=f"sources[{index}].catalog_sha256",
        )
        if not isinstance(source["provenance"], Mapping):
            raise ValueError(f"sources[{index}].provenance must be an object.")
        reader = source["reader"]
        if not isinstance(reader, Mapping):
            raise ValueError(f"sources[{index}].reader must be an object.")
        kind = reader.get("kind")
        if kind not in _POPULATION_READER_FACTORIES:
            raise ValueError(
                f"Population reader {kind!r} is not registered; registered "
                f"readers are {sorted(_POPULATION_READER_FACTORIES)}."
            )
        by_id[source_id] = source
    if set(by_id) != set(source_order):
        raise ValueError("source_order does not exactly enumerate sources.")

    holdout = payload["holdout"]
    if not isinstance(holdout, Mapping) or set(holdout) != {
        "manifest",
        "manifest_sha256",
        "episode_keys",
    }:
        raise ValueError(
            "holdout must contain exactly manifest, manifest_sha256, and "
            "episode_keys."
        )
    if (
        not isinstance(holdout["manifest"], str)
        or not holdout["manifest"].strip()
    ):
        raise ValueError("holdout.manifest must be a non-empty path.")
    _require_sha256(
        holdout["manifest_sha256"], label="holdout.manifest_sha256"
    )
    holdout_keys = holdout["episode_keys"]
    if (
        not isinstance(holdout_keys, list)
        or not holdout_keys
        or any(not isinstance(value, str) or "/" not in value for value in holdout_keys)
        or holdout_keys != sorted(set(holdout_keys))
    ):
        raise ValueError(
            "holdout.episode_keys must be a sorted, unique, non-empty list."
        )

    validated = json.loads(deterministic_json_bytes(dict(payload)))
    validated["sources"] = [dict(by_id[source_id]) for source_id in source_order]
    return validated


def _load_references(
    manifest: Mapping[str, Any],
    *,
    manifest_dir: Path,
) -> list[PopulationReference]:
    references: list[PopulationReference] = []
    keys: set[str] = set()
    for source in manifest["sources"]:
        factory = _POPULATION_READER_FACTORIES[source["reader"]["kind"]]
        for reference in factory(source, manifest_dir).references():
            if reference.key in keys:
                raise ValueError(f"Duplicate population episode key {reference.key!r}.")
            keys.add(reference.key)
            references.append(reference)
    unknown_holdout = sorted(set(manifest["holdout"]["episode_keys"]) - keys)
    if unknown_holdout:
        raise ValueError(
            f"Holdout references unknown population episodes: {unknown_holdout}."
        )
    return references


def _validate_authenticated_holdout(
    *,
    holdout_path: Path,
    manifest: Mapping[str, Any],
    population_manifest_dir: Path,
) -> None:
    """Require the bound holdout bytes to authenticate the exclusion keys.

    Production frozen-view populations must use the re-derivable global
    holdout contract.  Small all-NPZ unit fixtures retain their deliberately
    minimal schema, but even those fixtures must put the exact same sorted
    episode-key list in the authenticated file and the population manifest.
    """

    source_reader_kinds = {
        source["reader"]["kind"] for source in manifest["sources"]
    }
    if FROZEN_PARQUET_READER_KIND in source_reader_kinds:
        if source_reader_kinds != {FROZEN_PARQUET_READER_KIND}:
            raise ValueError(
                "Production frozen_parquet_view union populations cannot mix "
                "reader backends."
            )
        validate_global_holdout_manifest(
            holdout_path,
            expected_episode_keys=manifest["holdout"]["episode_keys"],
            population_sources=manifest["sources"],
            population_manifest_dir=population_manifest_dir,
        )
        return

    if source_reader_kinds != {NPZ_READER_KIND}:
        raise ValueError(
            "Only all-NPZ test fixtures may use a non-production global "
            f"holdout schema; production requires {REALMAN_UNION_HOLDOUT_SCHEMA!r}."
        )
    try:
        payload = json.loads(holdout_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Holdout manifest is invalid JSON: {holdout_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Holdout manifest root must be a JSON object.")
    authenticated_keys = payload.get("episode_keys")
    if authenticated_keys != manifest["holdout"]["episode_keys"]:
        raise ValueError(
            "Authenticated holdout episode keys do not exactly match the "
            "union population exclusion list."
        )


def _split_statistics(
    statistics: Mapping[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    return {
        name: list(statistics[name][start:end])
        for name in OPENPI_REALMAN_UNION_STATISTIC_NAMES
    }


def build_union_statistics(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build union statistics and its complete deterministic episode ledger."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Population manifest does not exist: {path}")
    raw_manifest = path.read_bytes()
    try:
        manifest = _validate_population_manifest(json.loads(raw_manifest))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Population manifest is invalid JSON: {path}: {exc}") from exc
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    holdout_path = Path(manifest["holdout"]["manifest"]).expanduser()
    if not holdout_path.is_absolute():
        holdout_path = path.parent / holdout_path
    if holdout_path.is_symlink():
        raise ValueError(
            f"Holdout manifest must not be a symlink: {holdout_path}"
        )
    try:
        holdout_path = holdout_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Holdout manifest does not exist: {holdout_path}"
        ) from exc
    if not holdout_path.is_file():
        raise ValueError(
            f"Holdout manifest must be a regular file: {holdout_path}"
        )
    actual_holdout_sha256 = _file_sha256(holdout_path)
    expected_holdout_sha256 = manifest["holdout"]["manifest_sha256"]
    if actual_holdout_sha256 != expected_holdout_sha256:
        raise ValueError(
            "Holdout manifest SHA-256 mismatch: expected "
            f"{expected_holdout_sha256}, got {actual_holdout_sha256}."
        )
    _validate_authenticated_holdout(
        holdout_path=holdout_path,
        manifest=manifest,
        population_manifest_dir=path.parent,
    )
    references = _load_references(manifest, manifest_dir=path.parent)
    holdout_keys = set(manifest["holdout"]["episode_keys"])

    # First pass: bind every logical episode to projected 18-D content before
    # deciding what is allowed into the training population.
    episode_metadata: dict[str, dict[str, Any]] = {}
    content_owner: dict[str, str] = {}
    for reference in references:
        episode = reference.load()
        content_sha256 = _episode_content_sha256(episode)
        previous_owner = content_owner.get(content_sha256)
        if reference.duplicate_of is None:
            if previous_owner is not None:
                raise ValueError(
                    f"Unmarked duplicate episode content: {reference.key} is "
                    f"identical to {previous_owner}. Declare duplicate_of or "
                    "remove the overlapping population."
                )
            content_owner[content_sha256] = reference.key
        else:
            if reference.duplicate_of not in episode_metadata:
                raise ValueError(
                    f"{reference.key}.duplicate_of must reference an earlier episode."
                )
            expected_digest = episode_metadata[reference.duplicate_of][
                "content_sha256"
            ]
            if content_sha256 != expected_digest:
                raise ValueError(
                    f"{reference.key}.duplicate_of content does not match "
                    f"{reference.duplicate_of}."
                )
            if previous_owner != reference.duplicate_of:
                raise ValueError(
                    f"{reference.key}.duplicate_of does not name the canonical "
                    f"content owner {previous_owner!r}."
                )
        episode_metadata[reference.key] = {
            "content_sha256": content_sha256,
            "frame_count": int(episode.state.shape[0]),
            "base_frame_indices": episode.base_frame_indices,
        }

    holdout_content: dict[str, str] = {}
    for holdout_key in sorted(holdout_keys):
        digest = episode_metadata[holdout_key]["content_sha256"]
        other = holdout_content.get(digest)
        if other is not None:
            raise ValueError(
                f"Global holdout contains duplicate content: {holdout_key} and {other}."
            )
        holdout_content[digest] = holdout_key
    for reference in references:
        if reference.key in holdout_keys:
            continue
        digest = episode_metadata[reference.key]["content_sha256"]
        if digest in holdout_content:
            raise ValueError(
                f"Holdout content leak: training episode {reference.key} is "
                f"identical to held-out episode {holdout_content[digest]}."
            )

    state_running = PiCompatibleMaskedRunningStats(REALMAN_POLICY_DIM)
    action_running = PiCompatibleMaskedRunningStats(REALMAN_POLICY_DIM)
    ledger_episodes: list[dict[str, Any]] = []
    source_summaries: dict[str, dict[str, Any]] = {
        source_id: {
            "id": source_id,
            "catalog_sha256": next(
                source["catalog_sha256"]
                for source in manifest["sources"]
                if source["id"] == source_id
            ),
            "provenance": next(
                source["provenance"]
                for source in manifest["sources"]
                if source["id"] == source_id
            ),
            "episode_count": 0,
            "candidate_base_frames": 0,
            "unique_base_frames": 0,
            "duplicate_base_frames": 0,
            "holdout_excluded_base_frames": 0,
        }
        for source_id in manifest["source_order"]
    }
    source_selected_digests = {
        source_id: hashlib.sha256()
        for source_id in manifest["source_order"]
    }
    candidate_count = 0
    unique_count = 0
    duplicate_count = 0
    holdout_count = 0
    mapping = np.asarray(
        REALMAN_18D_ACTION_CONTRACT.action_to_state_indices,
        dtype=np.int64,
    )
    delta_indices = np.flatnonzero(mapping >= 0)

    for reference in references:
        metadata = episode_metadata[reference.key]
        base_indices = metadata["base_frame_indices"]
        source_summary = source_summaries[reference.source_id]
        source_summary["episode_count"] += 1
        source_summary["candidate_base_frames"] += len(base_indices)
        candidate_count += len(base_indices)
        if reference.key in holdout_keys:
            accepted_indices: list[int] = []
            duplicate_indices: list[int] = []
            holdout_indices = list(base_indices)
            holdout_count += len(base_indices)
            source_summary["holdout_excluded_base_frames"] += len(base_indices)
        else:
            accepted_indices = []
            duplicate_indices = []
            holdout_indices = []
            if reference.duplicate_of is None:
                accepted_indices = list(base_indices)
            else:
                owner_indices = episode_metadata[reference.duplicate_of][
                    "base_frame_indices"
                ]
                if isinstance(owner_indices, range):
                    owner_contains = owner_indices.__contains__
                else:
                    owner_set = set(owner_indices)
                    owner_contains = owner_set.__contains__
                for base_index in base_indices:
                    if owner_contains(base_index):
                        duplicate_indices.append(base_index)
                    else:
                        accepted_indices.append(base_index)
            unique_count += len(accepted_indices)
            duplicate_count += len(duplicate_indices)
            source_summary["unique_base_frames"] += len(accepted_indices)
            source_summary["duplicate_base_frames"] += len(duplicate_indices)
            content_sha256 = metadata["content_sha256"]
            selected_digest = source_selected_digests[reference.source_id]
            for base_index in accepted_indices:
                selected_digest.update(
                    deterministic_json_bytes(
                        f"{content_sha256}:{base_index}"
                    )
                    + b"\n"
                )

        if accepted_indices:
            episode = reference.load()
            bases = np.asarray(accepted_indices, dtype=np.int64)
            state_running.update(
                episode.state[bases],
                episode.state_mask[bases],
            )
            gather = np.minimum(
                bases[:, None]
                + np.arange(REALMAN_ACTION_HORIZON, dtype=np.int64)[None, :],
                episode.action.shape[0] - 1,
            )
            action_chunks = encode_actions(
                episode.action[gather],
                episode.state[bases],
            )
            action_masks = episode.action_mask[gather].copy()
            anchor_valid = episode.state_mask[bases[:, None], mapping[delta_indices]]
            action_masks[:, :, delta_indices] &= anchor_valid[:, None, :]
            action_running.update(action_chunks, action_masks)

        ledger_episodes.append(
            {
                "source_id": reference.source_id,
                "episode_id": reference.episode_id,
                "episode_key": reference.key,
                "path": reference.path_text,
                "file_sha256": reference.expected_sha256,
                "content_sha256": metadata["content_sha256"],
                "frame_count": metadata["frame_count"],
                "candidate_base_frames": len(base_indices),
                "kept_base_frames": accepted_indices,
                "duplicate_base_frames": duplicate_indices,
                "holdout_base_frames": holdout_indices,
                "duplicate_of": reference.duplicate_of,
            }
        )

    if unique_count + duplicate_count + holdout_count != candidate_count:
        raise AssertionError("Union base-frame accounting bug.")
    state_statistics = state_running.get_statistics()
    action_statistics = action_running.get_statistics()

    for source_id, source_summary in source_summaries.items():
        source_summary["selected_content_sha256"] = (
            source_selected_digests[source_id].hexdigest()
        )
    ledger = {
        "schema": OPENPI_REALMAN_UNION_LEDGER_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "episodes": ledger_episodes,
    }
    ledger_sha256 = hashlib.sha256(deterministic_json_bytes(ledger)).hexdigest()
    artifact = {
        "schema": OPENPI_REALMAN_UNION_STATISTICS_SCHEMA,
        "contract": REALMAN_18D_ACTION_CONTRACT.to_dict(),
        "contract_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
        "normalization": Q01_Q99_UNCLIPPED,
        "algorithm": {
            "quantile_bins": 5000,
            "update_batch": "one_episode",
            "action_horizon": REALMAN_ACTION_HORIZON,
            "action_padding": "repeat_episode_final_frame",
            "deduplication": OPENPI_REALMAN_UNION_DEDUP_ALGORITHM,
        },
        "population": {
            "schema": OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
            "manifest_sha256": manifest_sha256,
            "ledger_sha256": ledger_sha256,
            # This digest was recomputed from the bound holdout file above;
            # it is not merely copied from the population manifest.
            "holdout_manifest_sha256": actual_holdout_sha256,
            "source_order": list(manifest["source_order"]),
            "sources": [
                source_summaries[source_id]
                for source_id in manifest["source_order"]
            ],
            "candidate_base_frames": candidate_count,
            "unique_base_frames": unique_count,
            "duplicate_base_frames": duplicate_count,
            "holdout_excluded_base_frames": holdout_count,
        },
        "selected": {
            "state": state_statistics,
            "action": action_statistics,
        },
        "modalities": {
            "state": {"source": state_statistics},
            "action": {
                "source_controls": _split_statistics(
                    action_statistics, 0, 16
                ),
                "source_head": _split_statistics(
                    action_statistics, 16, 18
                ),
            },
        },
    }
    # Validate before returning so direct Python callers get the same
    # fail-closed behavior as CLI users.
    serialize_openpi_realman_union_statistics(artifact)
    return artifact, ledger


def write_union_statistics(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    ledger_path: str | Path | None = None,
) -> tuple[str, str]:
    artifact, ledger = build_union_statistics(manifest_path)
    output = Path(output_path).expanduser().resolve()
    ledger_output = (
        Path(ledger_path).expanduser().resolve()
        if ledger_path is not None
        else output.with_suffix(output.suffix + ".ledger.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger_output.parent.mkdir(parents=True, exist_ok=True)
    artifact_bytes = serialize_openpi_realman_union_statistics(artifact)
    ledger_bytes = deterministic_json_bytes(ledger)
    if hashlib.sha256(ledger_bytes).hexdigest() != artifact["population"]["ledger_sha256"]:
        raise AssertionError("Union ledger changed after artifact construction.")

    for destination, payload in (
        (output, artifact_bytes),
        (ledger_output, ledger_bytes),
    ):
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    return hashlib.sha256(artifact_bytes).hexdigest(), hashlib.sha256(
        ledger_bytes
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path)
    args = parser.parse_args()

    artifact_sha256, ledger_sha256 = write_union_statistics(
        manifest_path=args.manifest,
        output_path=args.output,
        ledger_path=args.ledger_output,
    )
    print(f"output={args.output.expanduser().resolve()}")
    print(f"sha256={artifact_sha256}")
    print(f"ledger_sha256={ledger_sha256}")
    print(f"contract_sha256={REALMAN_18D_ACTION_CONTRACT.sha256()}")


if __name__ == "__main__":
    main()
