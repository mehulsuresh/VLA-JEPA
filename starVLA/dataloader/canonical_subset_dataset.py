from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from contextlib import contextmanager
import copy
import fcntl
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch

from ..action_representation import (
    Q01_Q99_UNCLIPPED,
    REALMAN_18D_ACTION_CONTRACT,
    load_openpi_realman_union_statistics,
    normalize_q01_q99_unclipped,
    select_canonical_realman_policy_action_mask,
    select_canonical_realman_policy_actions,
    select_canonical_realman_policy_state,
    select_canonical_realman_policy_state_mask,
)
from ..canonical_contract import (
    CANONICAL_EVAL_SELECTION_ALGORITHM,
    DEFAULT_ABSOLUTE_ACTION_REFERENCES,
    SHARD_Q01_Q99_UNCLIPPED,
    canonical_action_sidecar_variant,
    canonical_adapter_contract_sha256,
)
from ..eval_sampling_policy import (
    derive_episode_holdout_sampling_plan,
    validate_holdout_sampling_policy,
)
from .dataset_view import (
    EPISODE_RANGES_ENCODING,
    EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE,
    EXPANDED_ROWS_ENCODING,
    RANGE_ROW_SCHEMA as FROZEN_VIEW_RANGE_ROW_SCHEMA,
    ROW_SCHEMA as FROZEN_VIEW_ROW_SCHEMA,
    STATISTICS_POPULATION_CANDIDATE_PURPOSE,
    FrozenDatasetView,
    canonical_json_bytes as frozen_view_canonical_json_bytes,
    file_sha256 as frozen_view_file_sha256,
    load_frozen_view,
    make_range_id as make_frozen_view_range_id,
    make_sample_id as make_frozen_view_sample_id,
    range_sample_count as frozen_view_range_sample_count,
)
from .prompt_labels import (
    append_subtask_label_to_language,
    subtask_label_is_ignored,
    subtask_prompt_append_probability,
    subtask_prompt_ignored_labels,
)

try:
    import av
except ImportError:  # pragma: no cover
    av = None

if av is not None:
    try:
        av.logging.set_level(av.logging.PANIC)
    except Exception:  # pragma: no cover
        pass

try:
    import decord
except ImportError:  # pragma: no cover
    decord = None

try:
    import imageio.v3 as imageio_v3
except ImportError:  # pragma: no cover
    imageio_v3 = None


STATE_DIM = 53
ACTION_DIM = 49
DEFAULT_BUCKET_ROOT = "gs://robotics-datasets-yonduai/raw"
CANONICAL_INDEX_CACHE_VERSION = 6
DEFAULT_QWEN_CAMERA_SLOTS = ("main", "left", "right", "extra")
DEFAULT_VJEPA_CAMERA_SLOTS = ("left", "right", "main")
JOINT_DELTA_GRIPPER_ABSOLUTE = "joint_delta_gripper_absolute"
SHARD_Q01_Q99 = "shard_q01_q99"
CHECKPOINT_HANDOFF_SMOKE_PURPOSE = "checkpoint_handoff_smoke"
CHECKPOINT_HANDOFF_SMOKE_VIEW_SCHEMA = (
    "realman-checkpoint-handoff-smoke-view-v1"
)
CHECKPOINT_HANDOFF_SMOKE_STATISTICS_SCHEMA = (
    "realman-handoff-smoke-statistics-v1"
)
CHECKPOINT_HANDOFF_SMOKE_SCOPE = "checkpoint_handoff_validation_only"
REALSOURCE_DATASET_PREFIX = "RealSourceData/RealSource-World/"
REALSOURCE_COLLECT_MAIL_DATASET_ID = (
    "RealSourceData/RealSource-World/Collect_the_mail"
)
REALSOURCE_COLLECT_MAIL_ALIGNMENT_ALGORITHM = (
    "ordinals_0_121_to_data_0_121_drop_ordinal_122_"
    "ordinals_123_390_to_data_122_389"
)
SUBTASK_SEGMENTS_RELATIVE_PATH = "meta/subtask_segments.parquet"
SUBTASK_SEGMENTS_SCHEMA = "canonical-subtask-segments-parquet-v1"
SUBTASK_OVERLAP_RESOLUTION = (
    "latest_start_then_shortest_end_then_lowest_segment_index_v1"
)
_FROZEN_CANONICAL_INDEX_SCHEMA = "canonical-frozen-view-offset-index-v1"
_FROZEN_CANONICAL_RANGE_INDEX_SCHEMA = (
    "canonical-frozen-view-range-index-v1"
)
_FROZEN_CANONICAL_ROW_BINDING_SCHEMA = "canonical-frozen-view-row-binding-v1"
CANONICAL_JOINT_ACTION_SPANS = {
    "left_arm_joint": (0, 7),
    "right_arm_joint": (7, 14),
    "neck": (40, 42),
    "torso": (42, 46),
}
CANONICAL_JOINT_STATE_SPANS = {
    "left_arm_joint": (0, 7),
    "right_arm_joint": (7, 14),
    "neck": (40, 42),
    "torso": (42, 46),
}
class _RecoverableSampleError(RuntimeError):
    """Sample failure that can be handled by sampling a different window."""

    def __init__(self, message: str, path_key: str | None = None):
        super().__init__(message)
        self.path_key = path_key


class _RecoverableVideoDecodeError(_RecoverableSampleError):
    """Video decode failure that can be handled by sampling a different window."""


def collate_fn(batch):
    return batch


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _canonical_subtask_prompt_settings(data_cfg: Any) -> tuple[bool, str, str]:
    enabled = bool(_cfg_get(data_cfg, "append_subtask_to_prompt", False))
    source_column = str(
        _cfg_get(data_cfg, "subtask_prompt_source_column", "subtask_index")
    )
    label_column = str(
        _cfg_get(data_cfg, "subtask_prompt_label_column", "local_subtask_text")
    )
    if enabled and source_column != "subtask_index":
        raise ValueError(
            "Canonical append_subtask_to_prompt=true requires "
            "subtask_prompt_source_column='subtask_index'; "
            f"got {source_column!r}. Canonical resolves this logical ID through native episode spans."
        )
    if enabled and label_column != "local_subtask_text":
        raise ValueError(
            "Canonical append_subtask_to_prompt=true requires "
            "subtask_prompt_label_column='local_subtask_text'; "
            f"got {label_column!r}. Canonical resolves this logical label through native episode spans."
        )
    return enabled, source_column, label_column


def _as_list(value: Any, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    return list(value)


def _unique_preserve_order(values: list[str] | tuple[str, ...]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _select_qwen_camera_slots(
    image_mapping: dict[str, str],
    qwen_camera_slots: list[str] | tuple[str, ...] = DEFAULT_QWEN_CAMERA_SLOTS,
) -> list[str]:
    return [slot for slot in qwen_camera_slots if slot in image_mapping]


def _select_vjepa_camera_slots(
    image_mapping: dict[str, str],
    vjepa_camera_slots: list[str] | tuple[str, ...] = DEFAULT_VJEPA_CAMERA_SLOTS,
) -> list[str]:
    available = [slot for slot in DEFAULT_QWEN_CAMERA_SLOTS if slot in image_mapping]
    if not available:
        return []

    selected = []
    for target_slot in vjepa_camera_slots:
        if target_slot in image_mapping:
            selected.append(target_slot)
            continue
        fallback_order = []
        if target_slot == "left":
            fallback_order = ["right", "main", "extra"]
        elif target_slot == "right":
            fallback_order = ["left", "main", "extra"]
        elif target_slot == "main":
            fallback_order = ["extra", "left", "right"]
        else:
            fallback_order = list(DEFAULT_QWEN_CAMERA_SLOTS)
        selected.append(next((slot for slot in fallback_order if slot in image_mapping), available[0]))
    return selected


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_metadata_sequence(value: Any) -> list[Any]:
    if _is_missing_value(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_optional_int(value: Any) -> int | None:
    if _is_missing_value(value):
        return None
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if _is_missing_value(value):
        return None
    return float(value)


def _clean_label_text(value: Any) -> str:
    if _is_missing_value(value):
        return ""
    return str(value).strip()


def _as_float_filter_set(value: Any) -> set[float]:
    disabled_tokens = {"", "*", "all", "any", "none", "off"}
    values = set()
    for item in _as_list(value):
        if isinstance(item, str):
            token = item.strip().lower()
            if token in disabled_tokens:
                return set()
            item = token
        values.add(float(item))
    return values


def _parse_pyav_thread_count(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "auto", "default"}:
            return 1
        value = token
    return max(int(value), 0)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_line_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Configured canonical dataset list does not exist: {path}")
    values = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            values.append(line)
    return values


def _hash_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _fingerprint_directory(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        stat = child.stat()
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _read_parquet_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        schema_columns = set(pq.read_schema(path).names)
        selected_columns = [column for column in columns if column in schema_columns]
        if selected_columns:
            return pd.read_parquet(path, columns=selected_columns)
    except Exception:
        pass
    return pd.read_parquet(path)


def _is_realsource_dataset(dataset_id: str) -> bool:
    return str(dataset_id).startswith(REALSOURCE_DATASET_PREFIX)


def _strict_segment_integer(
    value: Any,
    *,
    column: str,
    row_number: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{SUBTASK_SEGMENTS_RELATIVE_PATH} row {row_number} column "
            f"{column!r} must be an integer; found {value!r}."
        )
    return int(value)


def _load_subtask_segment_spans(
    path: Path,
    *,
    episode_lengths: dict[int, int],
    source_episode_index_map: dict[int, int | None] | None = None,
) -> tuple[dict[int, tuple[SubtaskSpan, ...]], dict[str, int | str]]:
    """Load authenticated source-frame subtask spans.

    ``start_frame`` and ``end_frame_exclusive`` are raw source-frame
    coordinates. Empty source annotations are retained in the authenticated
    row count but contribute no span; labels are never synthesized.
    """

    required_columns = {
        "episode_index",
        "segment_index",
        "subtask_index",
        "subtask",
        "start_frame",
        "end_frame_exclusive",
    }
    frame = pd.read_parquet(path)
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns {missing_columns}."
        )
    if frame.duplicated(["episode_index", "segment_index"]).any():
        duplicate = frame.loc[
            frame.duplicated(
                ["episode_index", "segment_index"], keep=False
            ),
            ["episode_index", "segment_index"],
        ].iloc[0]
        raise ValueError(
            f"{path} contains duplicate segment identity "
            f"(episode_index={int(duplicate['episode_index'])}, "
            f"segment_index={int(duplicate['segment_index'])})."
        )

    spans_by_episode: dict[int, list[SubtaskSpan]] = {}
    zero_length_count = 0
    unaligned_row_count = 0
    for row_number, row in enumerate(frame.to_dict("records")):
        source_episode_index = _strict_segment_integer(
            row["episode_index"],
            column="episode_index",
            row_number=row_number,
        )
        episode_index = (
            source_episode_index
            if source_episode_index_map is None
            else source_episode_index_map.get(source_episode_index)
        )
        segment_index = _strict_segment_integer(
            row["segment_index"],
            column="segment_index",
            row_number=row_number,
        )
        subtask_index = _strict_segment_integer(
            row["subtask_index"],
            column="subtask_index",
            row_number=row_number,
        )
        start_frame = _strict_segment_integer(
            row["start_frame"],
            column="start_frame",
            row_number=row_number,
        )
        end_frame = _strict_segment_integer(
            row["end_frame_exclusive"],
            column="end_frame_exclusive",
            row_number=row_number,
        )
        label = _clean_label_text(row["subtask"])
        if not label:
            raise ValueError(
                f"{path} row {row_number} has an empty subtask label."
            )
        if episode_index is None:
            unaligned_row_count += 1
            continue
        if episode_index not in episode_lengths:
            raise ValueError(
                f"{path} row {row_number} references unknown episode "
                f"{source_episode_index} (mapped to {episode_index})."
            )
        episode_index = int(episode_index)
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
                f"{path} row {row_number} has invalid source-frame bounds "
                f"[{start_frame}, {end_frame}) for episode {episode_index} "
                f"with length {episode_length}."
            )
        source_start = start_frame
        source_end = end_frame
        if "source_start_frame" in row and not _is_missing_value(
            row["source_start_frame"]
        ):
            source_start = _strict_segment_integer(
                row["source_start_frame"],
                column="source_start_frame",
                row_number=row_number,
            )
        if "source_end_frame" in row and not _is_missing_value(
            row["source_end_frame"]
        ):
            source_end = _strict_segment_integer(
                row["source_end_frame"],
                column="source_end_frame",
                row_number=row_number,
            )
        if (
            source_start > start_frame
            or source_end < end_frame
            or source_end < source_start
        ):
            raise ValueError(
                f"{path} row {row_number} canonical bounds "
                f"[{start_frame}, {end_frame}) are not a valid clipping of "
                f"source bounds [{source_start}, {source_end})."
            )
        if start_frame == end_frame:
            zero_length_count += 1
            continue
        spans_by_episode.setdefault(episode_index, []).append(
            SubtaskSpan(
                label=label,
                start_frame=start_frame,
                end_frame=end_frame,
                subtask_index=subtask_index,
                segment_index=segment_index,
                boundary_semantics="source_frame_half_open",
            )
        )

    resolved = {
        episode_index: tuple(
            sorted(
                spans,
                key=lambda span: (
                    int(span.start_frame or 0),
                    int(span.end_frame or 0),
                    int(span.segment_index or 0),
                    int(span.subtask_index or 0),
                    span.label,
                ),
            )
        )
        for episode_index, spans in sorted(spans_by_episode.items())
    }
    return resolved, {
        "schema": SUBTASK_SEGMENTS_SCHEMA,
        "row_count": int(len(frame)),
        "usable_span_count": int(sum(map(len, resolved.values()))),
        "zero_length_span_count": int(zero_length_count),
        "unaligned_source_row_count": int(unaligned_row_count),
        "episode_count": int(frame["episode_index"].nunique()),
    }


def _realsource_subtask_episode_index_map(
    *,
    dataset_id: str,
    episode_lengths: Mapping[int, int],
    annotation_alignment: Mapping[str, Any] | None,
) -> dict[int, int | None]:
    if dataset_id != REALSOURCE_COLLECT_MAIL_DATASET_ID:
        return {
            int(episode_index): int(episode_index)
            for episode_index in episode_lengths
        }
    if annotation_alignment is not None:
        algorithm = annotation_alignment.get("algorithm")
        if algorithm != REALSOURCE_COLLECT_MAIL_ALIGNMENT_ALGORITHM:
            raise ValueError(
                "Collect_the_mail frozen source has an unexpected annotation "
                f"alignment algorithm: {algorithm!r}."
            )
    mapping: dict[int, int | None] = {
        source_episode_index: source_episode_index
        for source_episode_index in range(122)
    }
    mapping[122] = None
    mapping.update(
        {
            source_episode_index: source_episode_index - 1
            for source_episode_index in range(123, 391)
        }
    )
    if set(episode_lengths) != set(range(390)):
        raise ValueError(
            "Collect_the_mail episode catalog no longer matches the "
            "authenticated 390-episode alignment contract."
        )
    return mapping


def _derive_default_task(dataset_id: str) -> str:
    return dataset_id.rsplit("/", 1)[-1].replace("_", " ")


def _gcs_join(*parts: str) -> str:
    head = parts[0].rstrip("/")
    tail = "/".join(part.strip("/") for part in parts[1:])
    return f"{head}/{tail}" if tail else head


def _cleanup_gcloud_temp_path(path: Path, *, recursive: bool) -> None:
    if recursive:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _run_gcloud_cp(
    source: str,
    destination: Path,
    timeout_seconds: int = 900,
    recursive: bool = False,
    retries: int = 3,
    retry_backoff_seconds: float = 5.0,
) -> None:
    if recursive:
        destination.mkdir(parents=True, exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    attempts = max(1, int(retries))
    timeout_seconds = max(1, int(timeout_seconds))
    retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        if not recursive:
            copy_destination = destination.with_name(
                f".{destination.name}.{os.getpid()}.{time.time_ns()}.attempt{attempt}.tmp"
            )
        else:
            copy_destination = destination

        cmd = ["gcloud", "storage", "cp"]
        if recursive:
            cmd.append("--recursive")
        cmd.extend([source, str(copy_destination)])

        try:
            subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
            if not recursive:
                copy_destination.replace(destination)
            return
        except subprocess.CalledProcessError as exc:
            _cleanup_gcloud_temp_path(copy_destination, recursive=recursive)
            stderr = (exc.stderr or "").strip()
            last_error = RuntimeError(
                "Failed to copy canonical dataset shard from GCS. "
                "Refresh gcloud auth with `gcloud auth login` if credentials expired. "
                f"Attempt {attempt}/{attempts}. Command: {' '.join(cmd)}\n{stderr}"
            )
        except subprocess.TimeoutExpired as exc:
            _cleanup_gcloud_temp_path(copy_destination, recursive=recursive)
            last_error = RuntimeError(
                "Timed out copying canonical dataset shard from GCS. "
                f"Attempt {attempt}/{attempts}, timeout_seconds={timeout_seconds}. "
                f"Command: {' '.join(cmd)}"
            )
        except Exception as exc:
            _cleanup_gcloud_temp_path(copy_destination, recursive=recursive)
            last_error = exc

        if attempt < attempts:
            print(
                "Canonical GCS copy failed; retrying "
                f"attempt={attempt}/{attempts} timeout_seconds={timeout_seconds} "
                f"source={source}",
                file=sys.stderr,
                flush=True,
            )
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * attempt)

    assert last_error is not None
    raise last_error


@contextmanager
def _exclusive_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _shared_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _try_exclusive_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _relative_copy_lock_path(root: Path, relative_path: str) -> Path:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
    readable_stem = relative_path.replace("/", "__")[-96:]
    return root / ".locks" / f"{readable_stem}.{digest}.lock"


def _cache_file_copy_lock_path(cache_dir: Path, path: Path) -> Path | None:
    try:
        relative = path.relative_to(cache_dir)
    except ValueError:
        return None
    if len(relative.parts) < 4 or relative.parts[2] != "videos":
        return None
    root = cache_dir / relative.parts[0] / relative.parts[1]
    inner_relative = Path(*relative.parts[2:]).as_posix()
    return _relative_copy_lock_path(root, inner_relative)


def _ensure_metadata_root(
    *,
    root: Path,
    gcs_prefix: str,
    allow_gcs_download: bool,
    gcs_timeout_seconds: int = 900,
    gcs_retries: int = 3,
    gcs_retry_backoff_seconds: float = 5.0,
) -> Path | None:
    info_path = root / "meta/info.json"
    if info_path.exists():
        return root
    if not allow_gcs_download:
        return None

    lock_path = root / ".locks/meta.lock"
    with _exclusive_file_lock(lock_path):
        if info_path.exists():
            return root
        _run_gcloud_cp(
            _gcs_join(gcs_prefix, "files/meta"),
            root,
            timeout_seconds=gcs_timeout_seconds,
            recursive=True,
            retries=gcs_retries,
            retry_backoff_seconds=gcs_retry_backoff_seconds,
        )
    return root if info_path.exists() else None


def _ensure_relative_path(
    *,
    root: Path,
    gcs_prefix: str,
    relative_path: str,
    allow_gcs_download: bool,
    force_download: bool = False,
    gcs_timeout_seconds: int = 900,
    gcs_retries: int = 3,
    gcs_retry_backoff_seconds: float = 5.0,
) -> Path | None:
    local_path = root / relative_path
    if local_path.exists() and not force_download:
        return local_path
    if not allow_gcs_download:
        return None

    with _exclusive_file_lock(_relative_copy_lock_path(root, relative_path)):
        if local_path.exists() and not force_download:
            return local_path
        _run_gcloud_cp(
            _gcs_join(gcs_prefix, "files", relative_path),
            local_path,
            timeout_seconds=gcs_timeout_seconds,
            retries=gcs_retries,
            retry_backoff_seconds=gcs_retry_backoff_seconds,
        )
    return local_path


def _load_task_map(tasks_path: Path, default_task: str) -> dict[int, str]:
    if not tasks_path.exists():
        return {0: default_task}
    tasks = pd.read_parquet(tasks_path)
    task_map: dict[int, str] = {}
    if "task" in tasks.columns:
        for _, row in tasks.iterrows():
            task_index = int(row.get("task_index", len(task_map)))
            task_map[task_index] = str(row["task"])
    elif "task_index" in tasks.columns:
        for task_text, row in tasks.iterrows():
            task_map[int(row["task_index"])] = str(task_text)
    return task_map or {0: default_task}


def _load_canonical_modules(dataset_canonicalization_root: Path):
    src_root = dataset_canonicalization_root / "src"
    if src_root.as_posix() not in sys.path:
        sys.path.insert(0, src_root.as_posix())
    from model_v0.data.adapters import apply_unified_adapter, load_adapter_config

    return apply_unified_adapter, load_adapter_config


@dataclass(frozen=True)
class SubtaskSpan:
    label: str
    start_frame: int | None = None
    end_frame: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    subtask_index: int | None = None
    segment_index: int | None = None
    boundary_semantics: str = "legacy"


@dataclass(frozen=True)
class EpisodeSpec:
    local_start: int
    length: int
    task: str
    video_paths: dict[str, Path]
    video_base_frames: dict[str, int]
    subtask_spans: tuple[SubtaskSpan, ...] = ()
    episode_index: int | None = None
    dataset_from_index: int | None = None


@dataclass(frozen=True)
class WindowSpec:
    shard_index: int
    episode_index: int
    base_index: int


@dataclass(frozen=True)
class CanonicalEvalWindow:
    dataset_id: str
    sid: str
    revision: str
    data_file: str
    episode_index: int
    base_index: int

    @property
    def episode_identity(self) -> tuple[str, str, str, str, int]:
        return (
            self.dataset_id,
            self.sid,
            self.revision,
            self.data_file,
            self.episode_index,
        )


@dataclass(frozen=True)
class CanonicalEvalManifest:
    path: Path
    sha256: str
    purpose: str
    source_manifest_sha256: str
    selection: "CanonicalEvalSelection"
    windows: tuple[CanonicalEvalWindow, ...]

    @property
    def heldout_episode_identities(
        self,
    ) -> frozenset[tuple[str, str, str, str, int]]:
        return frozenset(window.episode_identity for window in self.windows)


@dataclass(frozen=True)
class CanonicalEvalSelection:
    algorithm: str
    seed: int
    window_count: int
    candidate_count: int
    action_horizon: int
    action_dim: int
    action_type: str
    normalization: str
    adapter_contract_sha256: str
    action_sidecar_variant: str
    configured_episode_count: int
    configured_episode_catalog_sha256: str
    holdout_episode_count: int | None = None
    # Deprecated compatibility alias. This is the common/base count for new
    # balanced manifests and the exact uniform count for legacy manifests.
    frames_per_episode: int | None = 1
    base_frames_per_episode: int = 1
    extra_window_episode_count: int = 0
    maximum_frames_per_episode: int = 1
    window_allocation_algorithm: str = "uniform_per_episode_v1"
    extra_window_episode_identities: tuple[
        tuple[str, str, str, str, int], ...
    ] = ()
    holdout_sampling_policy: dict[str, Any] | None = None
    holdout_sampling_plan: dict[str, Any] | None = None


def load_canonical_eval_manifest(
    path: str | Path,
    *,
    source_manifest_path: str | Path,
    expected_selection: dict[str, Any] | None = None,
) -> CanonicalEvalManifest:
    """Load exact canonical-stream eval windows and bind them to the catalog.

    The manifest is intentionally small and human-readable.  It contains only
    immutable source identities; GCS/local cache paths never become part of the
    split.  Every listed episode is excluded from the training dataset and from
    train-derived sidecar quantiles.
    """

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Canonical evaluation manifest does not exist: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Canonical evaluation manifest is invalid JSON: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(
            "Canonical evaluation manifest must be a schema_version=1 object."
        )
    purpose = str(payload.get("purpose", "")).lower()
    if purpose != "heldout":
        raise ValueError(
            "Canonical evaluation manifest purpose must be 'heldout', "
            f"got {purpose!r}."
        )
    source_path = Path(source_manifest_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Configured canonical source manifest does not exist: {source_path}"
        )
    source_sha256 = _hash_file(source_path)
    if payload.get("source_manifest_sha256") != source_sha256:
        raise ValueError(
            "Canonical evaluation manifest is not bound to the configured "
            "dataset-canonicalization manifest."
        )
    raw_selection = payload.get("selection")
    if not isinstance(raw_selection, dict):
        raise ValueError(
            "Canonical evaluation manifest requires a selection contract."
        )
    raw_selection = dict(raw_selection)
    if raw_selection.get("holdout_episode_count") is None:
        raw_selection["holdout_episode_count"] = raw_selection.get(
            "window_count"
        )
    if (
        "frames_per_episode" not in raw_selection
        and "base_frames_per_episode" not in raw_selection
    ):
        # Original schema-v1 manifests implicitly selected one window per
        # episode.
        raw_selection["frames_per_episode"] = 1
    else:
        raw_selection.setdefault("frames_per_episode", None)
    raw_selection.setdefault(
        "base_frames_per_episode",
        (
            raw_selection["frames_per_episode"]
            if raw_selection["frames_per_episode"] is not None
            else 1
        ),
    )
    raw_selection.setdefault("extra_window_episode_count", 0)
    raw_extra_window_episode_count = raw_selection[
        "extra_window_episode_count"
    ]
    raw_base_frames_per_episode = raw_selection[
        "base_frames_per_episode"
    ]
    raw_selection.setdefault(
        "maximum_frames_per_episode",
        (
            raw_base_frames_per_episode
            if (
                isinstance(raw_base_frames_per_episode, int)
                and not isinstance(raw_base_frames_per_episode, bool)
            )
            else 1
        )
        + int(
            isinstance(raw_extra_window_episode_count, int)
            and not isinstance(raw_extra_window_episode_count, bool)
            and raw_extra_window_episode_count > 0
        ),
    )
    raw_selection.setdefault(
        "window_allocation_algorithm",
        "uniform_per_episode_v1",
    )
    raw_selection.setdefault("extra_window_episode_identities", [])
    raw_selection.setdefault("holdout_sampling_policy", None)
    raw_selection.setdefault("holdout_sampling_plan", None)
    integer_fields = (
        "seed",
        "window_count",
        "holdout_episode_count",
        "base_frames_per_episode",
        "extra_window_episode_count",
        "maximum_frames_per_episode",
        "candidate_count",
        "action_horizon",
        "action_dim",
        "configured_episode_count",
    )
    string_fields = (
        "algorithm",
        "action_type",
        "normalization",
        "adapter_contract_sha256",
        "action_sidecar_variant",
        "configured_episode_catalog_sha256",
        "window_allocation_algorithm",
    )
    for field in integer_fields:
        value = raw_selection.get(field)
        minimum = (
            0
            if field in {"seed", "extra_window_episode_count"}
            else 1
        )
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                "Canonical evaluation selection requires valid integer "
                f"{field!r}, got {value!r}."
            )
    for field in string_fields:
        value = raw_selection.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                "Canonical evaluation selection requires non-empty string "
                f"{field!r}."
            )
    if raw_selection["algorithm"] != CANONICAL_EVAL_SELECTION_ALGORITHM:
        raise ValueError(
            "Unsupported canonical evaluation selection algorithm: "
            f"{raw_selection['algorithm']!r}."
        )
    if raw_selection["candidate_count"] < 3:
        raise ValueError(
            "Canonical evaluation selection candidate_count must be at least 3."
        )
    legacy_frames_per_episode = raw_selection["frames_per_episode"]
    if legacy_frames_per_episode is not None:
        if (
            isinstance(legacy_frames_per_episode, bool)
            or not isinstance(legacy_frames_per_episode, int)
            or legacy_frames_per_episode <= 0
        ):
            raise ValueError(
                "Canonical evaluation selection frames_per_episode must be a "
                "positive integer when present."
            )
        if (
            raw_selection["extra_window_episode_count"] != 0
            or legacy_frames_per_episode
            != raw_selection["base_frames_per_episode"]
        ):
            raise ValueError(
                "Canonical evaluation selection frames_per_episode is only a "
                "valid compatibility alias for a uniform allocation."
            )
    if (
        raw_selection["holdout_episode_count"]
        * raw_selection["base_frames_per_episode"]
        + raw_selection["extra_window_episode_count"]
        != raw_selection["window_count"]
    ):
        raise ValueError(
            "Canonical evaluation selection requires "
            "holdout_episode_count * base_frames_per_episode + "
            "extra_window_episode_count == window_count."
        )
    if raw_selection["extra_window_episode_count"] >= raw_selection[
        "holdout_episode_count"
    ] and raw_selection["extra_window_episode_count"] != 0:
        raise ValueError(
            "Canonical evaluation selection cannot assign an extra window to "
            "every heldout episode; fold that window into "
            "base_frames_per_episode instead."
        )
    expected_maximum_frames = raw_selection["base_frames_per_episode"] + int(
        raw_selection["extra_window_episode_count"] > 0
    )
    if (
        raw_selection["maximum_frames_per_episode"]
        != expected_maximum_frames
    ):
        raise ValueError(
            "Canonical evaluation selection maximum_frames_per_episode does "
            "not match its balanced allocation."
        )
    raw_extra_identities = raw_selection[
        "extra_window_episode_identities"
    ]
    if not isinstance(raw_extra_identities, list):
        raise ValueError(
            "Canonical evaluation selection "
            "extra_window_episode_identities must be a list."
        )
    extra_window_episode_identities: list[
        tuple[str, str, str, str, int]
    ] = []
    for index, raw_identity in enumerate(raw_extra_identities):
        if (
            not isinstance(raw_identity, (list, tuple))
            or len(raw_identity) != 5
            or any(
                not isinstance(value, str) or not value
                for value in raw_identity[:4]
            )
            or isinstance(raw_identity[4], bool)
            or not isinstance(raw_identity[4], int)
            or raw_identity[4] < 0
        ):
            raise ValueError(
                "Canonical evaluation selection extra episode identity "
                f"{index} is invalid: {raw_identity!r}."
            )
        extra_window_episode_identities.append(
            (
                str(raw_identity[0]),
                str(raw_identity[1]),
                str(raw_identity[2]),
                str(raw_identity[3]),
                int(raw_identity[4]),
            )
        )
    if (
        len(extra_window_episode_identities)
        != raw_selection["extra_window_episode_count"]
        or len(set(extra_window_episode_identities))
        != len(extra_window_episode_identities)
    ):
        raise ValueError(
            "Canonical evaluation selection must bind one distinct explicit "
            "episode identity per extra window."
        )
    if raw_selection["holdout_sampling_policy"] is not None:
        try:
            expected_sampling_plan = derive_episode_holdout_sampling_plan(
                total_episode_count=raw_selection["configured_episode_count"],
                evaluation_observation_count=raw_selection["window_count"],
                policy=raw_selection["holdout_sampling_policy"],
            )
        except ValueError as exc:
            raise ValueError(
                f"Canonical holdout sampling policy is invalid: {exc}"
            ) from exc
        if raw_selection["holdout_sampling_plan"] != expected_sampling_plan:
            raise ValueError(
                "Canonical holdout sampling plan does not match its policy and "
                "configured episode catalog."
            )
        for field in (
            "holdout_episode_count",
            "base_frames_per_episode",
            "extra_window_episode_count",
            "maximum_frames_per_episode",
            "window_allocation_algorithm",
        ):
            if raw_selection[field] != expected_sampling_plan[field]:
                raise ValueError(
                    "Canonical holdout allocation does not match the derived "
                    f"sampling plan field {field!r}."
                )
    elif raw_selection["holdout_sampling_plan"] is not None:
        raise ValueError(
            "Canonical holdout_sampling_plan requires holdout_sampling_policy."
        )
    if len(raw_selection["action_sidecar_variant"]) != 16 or any(
        character not in "0123456789abcdef"
        for character in raw_selection["action_sidecar_variant"]
    ):
        raise ValueError(
            "Canonical evaluation selection action_sidecar_variant must be "
            "a 16-character lowercase SHA-256 prefix."
        )
    for field in (
        "adapter_contract_sha256",
        "configured_episode_catalog_sha256",
    ):
        sha256 = raw_selection[field]
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise ValueError(
                f"Canonical evaluation selection {field} must be lowercase "
                "SHA-256."
            )
    if expected_selection is not None:
        mismatches = {
            key: {
                "manifest": raw_selection.get(key),
                "expected": expected,
            }
            for key, expected in expected_selection.items()
            if raw_selection.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "Canonical evaluation manifest selection contract does not "
                f"match the current dataset config: {mismatches}."
            )
    raw_windows = payload.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError(
            "Canonical evaluation manifest must contain at least one window."
        )
    windows: list[CanonicalEvalWindow] = []
    for index, raw in enumerate(raw_windows):
        if not isinstance(raw, dict):
            raise ValueError(
                f"Canonical evaluation window {index} must be an object."
            )
        required_strings = ("dataset_id", "sid", "revision", "data_file")
        missing = [
            key
            for key in required_strings
            if not isinstance(raw.get(key), str) or not raw[key]
        ]
        if missing:
            raise ValueError(
                f"Canonical evaluation window {index} has invalid fields: {missing}."
            )
        episode_index = raw.get("episode_index")
        base_index = raw.get("base_index")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
            or isinstance(base_index, bool)
            or not isinstance(base_index, int)
            or base_index < 0
        ):
            raise ValueError(
                f"Canonical evaluation window {index} requires non-negative "
                "integer episode_index and base_index."
            )
        windows.append(
            CanonicalEvalWindow(
                dataset_id=raw["dataset_id"],
                sid=raw["sid"],
                revision=raw["revision"],
                data_file=raw["data_file"],
                episode_index=episode_index,
                base_index=base_index,
            )
        )
    identities = [
        (*window.episode_identity, window.base_index) for window in windows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Canonical evaluation manifest contains duplicate windows.")
    if raw_selection["window_count"] != len(windows):
        legacy_detail = (
            " Legacy contract requires exactly one deterministic window per "
            "heldout episode."
            if (
                raw_selection["base_frames_per_episode"] == 1
                and raw_selection["extra_window_episode_count"] == 0
            )
            else ""
        )
        raise ValueError(
            "Canonical evaluation selection window_count does not match windows: "
            f"{raw_selection['window_count']} != {len(windows)}."
            f"{legacy_detail}"
        )
    episode_counts: dict[tuple[str, str, str, str, int], int] = {}
    for window in windows:
        episode_counts[window.episode_identity] = (
            episode_counts.get(window.episode_identity, 0) + 1
        )
    extra_identity_set = set(extra_window_episode_identities)
    if not extra_identity_set.issubset(episode_counts):
        raise ValueError(
            "Canonical evaluation selection binds extra windows to episodes "
            "that are absent from the manifest."
        )
    if len(episode_counts) != raw_selection["holdout_episode_count"] or any(
        count
        != (
            raw_selection["base_frames_per_episode"]
            + int(identity in extra_identity_set)
        )
        for identity, count in episode_counts.items()
    ):
        if (
            raw_selection["base_frames_per_episode"] == 1
            and raw_selection["extra_window_episode_count"] == 0
        ):
            raise ValueError(
                "Canonical evaluation manifest must select exactly one "
                "deterministic window per heldout episode."
            )
        raise ValueError(
            "Canonical evaluation manifest windows do not match the explicit "
            "balanced per-episode allocation."
        )
    selection = CanonicalEvalSelection(
        **{
            field: raw_selection[field]
            for field in (*string_fields, *integer_fields)
        },
        extra_window_episode_identities=tuple(
            extra_window_episode_identities
        ),
        frames_per_episode=legacy_frames_per_episode,
        holdout_sampling_policy=raw_selection["holdout_sampling_policy"],
        holdout_sampling_plan=raw_selection["holdout_sampling_plan"],
    )
    return CanonicalEvalManifest(
        path=manifest_path,
        sha256=_hash_file(manifest_path),
        purpose=purpose,
        source_manifest_sha256=source_sha256,
        selection=selection,
        windows=tuple(windows),
    )


@dataclass(frozen=True)
class EpisodeWindowRange:
    shard_index: int
    episode_index: int
    cumulative_end: int


@dataclass
class ShardSpec:
    dataset_id: str
    sid: str
    revision: str
    adapter_group_id: str
    adapter_path: Path
    root: Path
    gcs_prefix: str
    data_relative_path: str
    data_path: Path
    sidecar_path: Path
    fps: float
    camera_source_keys: dict[str, str]
    qwen_camera_slots: tuple[str, ...]
    vjepa_camera_slots: tuple[str, ...]
    decode_camera_slots: tuple[str, ...]
    task_map: dict[int, str]
    episodes: list[EpisodeSpec]
    episode_metadata_path: Path | None = None
    episode_metadata_sha256: str | None = None
    episode_metadata_size: int | None = None
    episode_metadata_mtime_ns: int | None = None
    episode_metadata_ctime_ns: int | None = None
    subtask_segments_path: Path | None = None
    subtask_segments_sha256: str | None = None
    subtask_segments_size: int | None = None
    subtask_segments_mtime_ns: int | None = None
    subtask_segments_ctime_ns: int | None = None
    subtask_segments_row_count: int | None = None
    subtask_segments_zero_length_count: int | None = None
    subtask_segments_unaligned_source_row_count: int | None = None
    adapter_sha256: str | None = None


@dataclass
class _PyAVReader:
    container: Any
    stream: Any
    fps: float
    time_base: float
    start_time: int
    frame_count: int


class _ShardData:
    def __init__(self, sidecar_path: Path):
        payload = np.load(sidecar_path)
        self.state = payload["state_values"]
        self.state_mask = payload["state_mask"]
        self.action = payload["action_values"]
        self.action_mask = payload["action_mask"]
        self.action_low = payload["action_low"]
        self.action_high = payload["action_high"]
        self.state_low = payload["state_low"]
        self.state_high = payload["state_high"]
        self.action_delta_mask = payload.get(
            "action_delta_mask", np.zeros((ACTION_DIM,), dtype=bool)
        ).astype(bool)
        self.action_to_state_indices = payload.get(
            "action_to_state_indices", np.full((ACTION_DIM,), -1, dtype=np.int64)
        ).astype(np.int64)
        self.timestamp = payload["timestamp"]
        self.frame_index = payload["frame_index"]
        self.episode_index = payload["episode_index"]
        self.task_index = payload["task_index"]


class CanonicalSubsetVLADataset(torch.utils.data.Dataset):
    """Canonical LeRobot v3 subset reader for the existing CPU-worker VLA-JEPA path.

    The dataset consumes dataset-canonicalization manifests/adapters and cached GCS
    LeRobot v3 shards. It returns the same list-of-dicts sample shape as the
    current LeRobot path: ``video_compact``, ``state``, ``action`` and ``lang``.
    """

    @staticmethod
    def _validate_exact_epoch_settings(
        *,
        mode: str,
        epoch_sampling_strategy: str,
        max_shards: int,
        max_shards_per_dataset: int,
        max_windows: int,
        max_windows_per_dataset: int,
        shuffle_shards: bool,
        sample_stride: int,
    ) -> None:
        if not (
            mode == "train"
            and epoch_sampling_strategy == "all_sources_exhaustive"
        ):
            return
        incompatible = []
        if max_shards:
            incompatible.append("max_shards")
        if max_shards_per_dataset:
            incompatible.append("max_shards_per_dataset")
        if max_windows:
            incompatible.append("max_windows")
        if max_windows_per_dataset:
            incompatible.append("max_windows_per_dataset")
        if shuffle_shards:
            incompatible.append("shuffle_shards")
        if sample_stride != 1:
            incompatible.append("sample_stride")
        if incompatible:
            raise ValueError(
                "Canonical all_sources_exhaustive must enumerate every "
                "selected training window exactly once; incompatible "
                f"settings: {', '.join(incompatible)}."
            )

    def _load_frozen_train_view_descriptor(
        self,
        manifest_path: Path,
        *,
        expected_manifest_sha256: str,
    ) -> None:
        if self.epoch_sampling_strategy != "all_sources_exhaustive":
            raise ValueError(
                "Canonical frozen_train_view_manifest requires "
                "epoch_sampling_strategy=all_sources_exhaustive."
            )
        resolved = manifest_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Canonical frozen training-view manifest is missing: {resolved}"
            )
        actual_manifest_sha256 = _hash_file(resolved)
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "Canonical frozen training-view manifest SHA-256 mismatch: "
                f"expected {expected_manifest_sha256}, "
                f"found {actual_manifest_sha256}."
            )
        view = load_frozen_view(
            resolved,
            expected_representation_contract_sha256=(
                REALMAN_18D_ACTION_CONTRACT.sha256()
            ),
            verify_ledger=False,
        )
        purpose = view.descriptor.get("purpose")
        if purpose == STATISTICS_POPULATION_CANDIDATE_PURPOSE:
            raise ValueError(
                "Canonical training cannot consume a "
                "statistics_population_candidate view. Use the separate "
                "holdout-free frozen training view."
            )
        if purpose == EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE:
            if not self.allow_eval_selection_population_candidate:
                raise ValueError(
                    "Canonical training cannot consume an "
                    "eval_selection_population_candidate view. This "
                    "non-trainable view is accepted only by the canonical "
                    "eval-manifest generator through its explicit code-only "
                    "bootstrap gate."
                )
            usage_contract = view.descriptor.get("usage_contract")
            if (
                not isinstance(usage_contract, dict)
                or usage_contract.get("training_allowed") is not False
                or usage_contract.get("eval_manifest_generation") is not True
            ):
                raise ValueError(
                    "Canonical eval-selection candidate has an invalid "
                    "usage_contract."
                )
        elif self.normalization_statistics is None:
            raise ValueError(
                "Canonical frozen training views require the shared immutable "
                "18-D normalization statistics artifact."
            )
        representation = view.descriptor["representation"]
        expected_representation = {
            "state_dim": REALMAN_18D_ACTION_CONTRACT.state_dim,
            "action_dim": REALMAN_18D_ACTION_CONTRACT.action_dim,
            "horizon": REALMAN_18D_ACTION_CONTRACT.action_horizon,
            "action_type": JOINT_DELTA_GRIPPER_ABSOLUTE,
            "normalization": Q01_Q99_UNCLIPPED,
        }
        mismatches = {
            key: {
                "expected": expected,
                "found": representation.get(key),
            }
            for key, expected in expected_representation.items()
            if representation.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "Canonical frozen training-view representation mismatch: "
                f"{mismatches}."
            )
        if int(view.descriptor["epoch_contract"]["epoch_passes"]) != 1:
            raise ValueError(
                "Canonical frozen training view must define exactly one "
                "exhaustive pass per logical epoch."
            )
        if view.unique_sample_count != view.row_count:
            raise ValueError(
                "Canonical frozen training view must contain one unique "
                "sample per ledger row."
            )
        source_manifest_sha256 = _hash_file(self.manifest_path)
        canonical_sources = [
            source
            for source in view.descriptor["sources"]
            if source.get("backend") == "canonical"
        ]
        if len(canonical_sources) != len(view.descriptor["sources"]):
            raise ValueError(
                "canonical_subset_vla cannot consume non-canonical rows from "
                "a frozen training view."
            )
        for source in canonical_sources:
            if source.get("manifest_sha256") != source_manifest_sha256:
                raise ValueError(
                    "Canonical frozen training-view source manifest mismatch "
                    f"for source_id={source.get('source_id')!r}: expected "
                    f"{source_manifest_sha256}, found "
                    f"{source.get('manifest_sha256')!r}."
                )

        self.frozen_train_view_manifest_path = resolved
        self.frozen_train_view_manifest_sha256 = actual_manifest_sha256
        self.frozen_train_view = view

    def _validate_checkpoint_handoff_smoke_eval_binding(self) -> bool:
        """Authenticate the one-batch smoke against its production eval split.

        A handoff smoke is intentionally a tiny prefix of a production frozen
        view, so ``smoke train episodes ∪ production holdout episodes`` cannot
        reproduce the production eval manifest's full catalog count/hash.  We
        accept that mismatch only when the frozen view, exact eval manifest,
        and isolated smoke statistics mutually authenticate one another.

        Returning ``False`` means this is an ordinary production view and the
        caller must retain the full catalog count/hash validation.
        """

        view = self.frozen_train_view
        if (
            view is None
            or view.descriptor.get("purpose")
            != CHECKPOINT_HANDOFF_SMOKE_PURPOSE
        ):
            return False
        evaluation = self.canonical_eval_manifest
        if evaluation is None:
            raise ValueError(
                "Canonical checkpoint-handoff smoke requires its exact "
                "production evaluation manifest."
            )
        if view.row_count != 128 or view.unique_sample_count != 128:
            raise ValueError(
                "Canonical checkpoint-handoff smoke must contain exactly "
                "128 unique logical rows."
            )

        usage = view.descriptor.get("usage_contract")
        expected_usage = {
            "training_allowed": True,
            "scope": CHECKPOINT_HANDOFF_SMOKE_SCOPE,
            "model_quality_claim_allowed": False,
            "statistics_accumulation": "forbidden",
        }
        if usage != expected_usage:
            raise ValueError(
                "Canonical checkpoint-handoff smoke has an unsafe "
                "usage_contract."
            )

        selection = view.descriptor.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError(
                "Canonical checkpoint-handoff smoke lacks selection lineage."
            )
        parent_manifest = selection.get("parent_manifest")
        parent_digests = {
            "parent_manifest_sha256": selection.get(
                "parent_manifest_sha256"
            ),
            "parent_view_id": selection.get("parent_view_id"),
            "parent_ledger_sha256": selection.get(
                "parent_ledger_sha256"
            ),
        }
        if (
            selection.get("schema")
            != CHECKPOINT_HANDOFF_SMOKE_VIEW_SCHEMA
            or selection.get("algorithm")
            != "ordered_logical_prefix_from_authenticated_parent_v1"
            or selection.get("requested_logical_row_count") != 128
            or not isinstance(parent_manifest, str)
            or not parent_manifest
            or any(
                not self._valid_sha256(value)
                for value in parent_digests.values()
            )
        ):
            raise ValueError(
                "Canonical checkpoint-handoff smoke lacks authenticated "
                "production-parent lineage."
            )

        binding = selection.get("evaluation_holdout")
        if not isinstance(binding, Mapping):
            raise ValueError(
                "Canonical checkpoint-handoff smoke lacks its production "
                "evaluation-holdout binding."
            )
        episode_identities = sorted(
            evaluation.heldout_episode_identities
        )
        unsigned_binding = dict(binding)
        binding_sha256 = unsigned_binding.pop("sha256", None)
        expected_binding = {
            "schema": "realsource-canonical-eval-holdout-binding-v1",
            "manifest_sha256": evaluation.sha256,
            "source_manifest_sha256": (
                evaluation.source_manifest_sha256
            ),
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
            "copy_detection": [
                "episode_identity",
                "episode_lineage_id",
                "episode_content_id",
            ],
        }
        mismatches = {
            key: {"expected": expected, "found": binding.get(key)}
            for key, expected in expected_binding.items()
            if binding.get(key) != expected
        }
        if (
            mismatches
            or not self._valid_sha256(binding_sha256)
            or binding_sha256
            != _stable_json_sha256(unsigned_binding)
        ):
            raise ValueError(
                "Canonical checkpoint-handoff smoke is not bound to the "
                f"exact production evaluation population: {mismatches}."
            )

        exclusions = view.descriptor.get("holdout_exclusions")
        if (
            not isinstance(exclusions, Mapping)
            or exclusions.get("source_id")
            != f"canonical_eval_manifest:{evaluation.sha256}"
            or not exclusions.get("lineage_ids")
            or not exclusions.get("content_ids")
        ):
            raise ValueError(
                "Canonical checkpoint-handoff smoke lacks the production "
                "dual-identity holdout exclusions."
            )

        statistics = self.normalization_statistics
        population = (
            statistics.get("population")
            if isinstance(statistics, Mapping)
            else None
        )
        sources = (
            population.get("sources")
            if isinstance(population, Mapping)
            else None
        )
        if (
            not isinstance(population, Mapping)
            or population.get("source_order")
            != ["realsource", "intervention", "hq"]
            or population.get("unique_base_frames") != 384
            or not isinstance(sources, list)
            or [
                candidate.get("id")
                if isinstance(candidate, Mapping)
                else None
                for candidate in sources
            ]
            != ["realsource", "intervention", "hq"]
        ):
            raise ValueError(
                "Canonical checkpoint-handoff smoke requires the exact "
                "384-row RealSource/intervention/HQ normalization population."
            )
        source = next(
            (
                candidate
                for candidate in sources
                if isinstance(candidate, Mapping)
                and candidate.get("id") == "realsource"
            ),
            None,
        )
        provenance = (
            source.get("provenance")
            if isinstance(source, Mapping)
            else None
        )
        smoke_statistics = (
            provenance.get("handoff_smoke")
            if isinstance(provenance, Mapping)
            else None
        )
        expected_statistics = {
            "schema": CHECKPOINT_HANDOFF_SMOKE_STATISTICS_SCHEMA,
            "scope": CHECKPOINT_HANDOFF_SMOKE_SCOPE,
            "model_quality_claim_allowed": False,
            "exact_training_logical_rows": 128,
            "parent_smoke_view_sha256": view.manifest_sha256,
            "parent_smoke_ledger_sha256": (
                view.descriptor["rows"]["sha256"]
            ),
        }
        statistics_mismatches = {
            key: {
                "expected": expected,
                "found": (
                    smoke_statistics.get(key)
                    if isinstance(smoke_statistics, Mapping)
                    else None
                ),
            }
            for key, expected in expected_statistics.items()
            if (
                not isinstance(smoke_statistics, Mapping)
                or smoke_statistics.get(key) != expected
            )
        }
        if statistics_mismatches:
            raise ValueError(
                "Canonical checkpoint-handoff smoke is not bound to its "
                "isolated smoke-only normalization population: "
                f"{statistics_mismatches}."
            )
        return True

    def __init__(
        self,
        data_cfg: Any,
        *,
        mode: str = "train",
        action_horizon: int,
        video_horizon: int,
        video_frame_stride: int,
        allow_eval_selection_population_candidate: bool = False,
    ) -> None:
        self.data_cfg = data_cfg
        self.mode = str(mode).lower()
        # Deliberately code-only: this cannot be enabled from a training YAML.
        # The eval-manifest generator passes it explicitly while bootstrapping
        # a frozen pre-holdout population.
        self.allow_eval_selection_population_candidate = bool(
            allow_eval_selection_population_candidate
        )
        if self.mode not in {"train", "eval"}:
            raise ValueError(
                f"Canonical dataset mode must be 'train' or 'eval', got {mode!r}."
            )
        configured_frozen_view = _cfg_get(
            data_cfg, "frozen_train_view_manifest", None
        )
        configured_frozen_view_sha256 = _cfg_get(
            data_cfg, "frozen_train_view_manifest_sha256", None
        )
        if bool(configured_frozen_view) != bool(
            configured_frozen_view_sha256
        ):
            raise ValueError(
                "Canonical frozen training views require both "
                "frozen_train_view_manifest and "
                "frozen_train_view_manifest_sha256."
            )
        self.frozen_train_view_manifest_path: Path | None = None
        self.frozen_train_view_manifest_sha256: str | None = None
        self.frozen_train_view: FrozenDatasetView | None = None
        self.frozen_train_view_index_path: Path | None = None
        self.frozen_train_view_index_metadata_path: Path | None = None
        self.frozen_train_view_index_sha256: str | None = None
        self.frozen_train_view_identity_catalog_sha256: str | None = None
        self._frozen_view_offsets: np.memmap | None = None
        self._frozen_view_ledger_handle = None
        self._frozen_view_episode_lookup: dict[
            tuple[str, str, str, int], tuple[int, int]
        ] = {}
        configured_union_statistics = _cfg_get(
            data_cfg, "normalization_statistics_artifact", None
        )
        configured_union_statistics_sha256 = _cfg_get(
            data_cfg, "normalization_statistics_artifact_sha256", None
        )
        if bool(configured_union_statistics) != bool(
            configured_union_statistics_sha256
        ):
            raise ValueError(
                "Canonical shared normalization requires both "
                "normalization_statistics_artifact and "
                "normalization_statistics_artifact_sha256."
            )
        self.normalization_statistics_artifact_path: Path | None = None
        self.normalization_statistics_artifact_sha256: str | None = None
        self.normalization_statistics: dict[str, Any] | None = None
        self.policy_state_dim = (
            REALMAN_18D_ACTION_CONTRACT.state_dim
            if self.allow_eval_selection_population_candidate
            else STATE_DIM
        )
        self.policy_action_dim = (
            REALMAN_18D_ACTION_CONTRACT.action_dim
            if self.allow_eval_selection_population_candidate
            else ACTION_DIM
        )
        if configured_union_statistics:
            statistics_path = Path(
                str(configured_union_statistics)
            ).expanduser().resolve()
            expected_sha256 = str(configured_union_statistics_sha256)
            self.normalization_statistics = (
                load_openpi_realman_union_statistics(
                    statistics_path,
                    expected_sha256,
                )
            )
            self.normalization_statistics_artifact_path = statistics_path
            self.normalization_statistics_artifact_sha256 = expected_sha256
            self.policy_state_dim = REALMAN_18D_ACTION_CONTRACT.state_dim
            self.policy_action_dim = REALMAN_18D_ACTION_CONTRACT.action_dim
        self.dataset_canonicalization_root = Path(
            _cfg_get(data_cfg, "dataset_canonicalization_root", "/home/mehul/work/dataset-canonicalization")
        )
        self.manifest_path = Path(
            _cfg_get(
                data_cfg,
                "manifest_path",
                self.dataset_canonicalization_root / "configs/manifests/dataset_manifests.jsonl.gz",
            )
        )
        configured_eval_manifest = _cfg_get(
            data_cfg, "canonical_eval_manifest", None
        )
        self.adapter_contract_sha256 = canonical_adapter_contract_sha256(
            data_cfg
        )
        expected_eval_selection = {
            "algorithm": CANONICAL_EVAL_SELECTION_ALGORITHM,
            "seed": int(
                _cfg_get(data_cfg, "canonical_eval_selection_seed", 0)
            ),
            "candidate_count": int(
                _cfg_get(data_cfg, "canonical_eval_candidate_count", 32)
            ),
            "action_horizon": int(action_horizon),
            "action_dim": self.policy_action_dim,
            "action_type": str(
                _cfg_get(data_cfg, "action_type", "dataset_native")
            ).lower(),
            "normalization": (
                Q01_Q99_UNCLIPPED
                if self.normalization_statistics is not None
                else str(
                    _cfg_get(
                        data_cfg,
                        "sidecar_normalization",
                        SHARD_Q01_Q99_UNCLIPPED,
                    )
                ).lower()
            ),
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "action_sidecar_variant": canonical_action_sidecar_variant(
                data_cfg,
                action_horizon=int(action_horizon),
                canonical_eval_manifest_sha256=None,
                exclude_eval_episodes_from_training=False,
                adapter_contract_sha256=self.adapter_contract_sha256,
            ),
        }
        configured_holdout_sampling = _cfg_get(
            data_cfg,
            "holdout_sampling",
            None,
        )
        if configured_holdout_sampling is not None:
            expected_eval_selection["holdout_sampling_policy"] = (
                validate_holdout_sampling_policy(
                    configured_holdout_sampling
                )
            )
        self.canonical_eval_manifest = (
            None
            if not configured_eval_manifest
            else load_canonical_eval_manifest(
                configured_eval_manifest,
                source_manifest_path=self.manifest_path,
                expected_selection=expected_eval_selection,
            )
        )
        self.exclude_eval_episodes_from_training = bool(
            _cfg_get(
                data_cfg,
                "canonical_exclude_eval_episodes_from_training",
                False,
            )
        )
        if self.canonical_eval_manifest is not None and not (
            self.exclude_eval_episodes_from_training
        ):
            raise ValueError(
                "canonical_eval_manifest requires "
                "canonical_exclude_eval_episodes_from_training=true so heldout "
                "episodes cannot leak into training or normalization statistics."
            )
        self.adapter_dir = Path(
            _cfg_get(
                data_cfg,
                "adapter_dir",
                self.dataset_canonicalization_root / "configs/dataset_adapters",
            )
        )
        self.cache_dir = Path(
            _cfg_get(
                data_cfg,
                "cache_dir",
                self.dataset_canonicalization_root / ".cache/gcs_lerobot",
            )
        )
        self.bucket_root = str(_cfg_get(data_cfg, "bucket_root", DEFAULT_BUCKET_ROOT)).rstrip("/")
        self.allow_gcs_download = bool(_cfg_get(data_cfg, "allow_gcs_download", False))
        self.gcs_download_timeout_seconds = max(
            1, int(_cfg_get(data_cfg, "gcs_download_timeout_seconds", 900))
        )
        self.gcs_download_retries = max(1, int(_cfg_get(data_cfg, "gcs_download_retries", 3)))
        self.gcs_download_retry_backoff_seconds = max(
            0.0, float(_cfg_get(data_cfg, "gcs_download_retry_backoff_seconds", 5.0) or 0.0)
        )
        self.vjepa_camera_slots = _as_list(
            _cfg_get(
                data_cfg,
                "vjepa_camera_slots",
                _cfg_get(data_cfg, "camera_slots", list(DEFAULT_VJEPA_CAMERA_SLOTS)),
            )
        )
        self.qwen_camera_slots = _as_list(_cfg_get(data_cfg, "qwen_camera_slots", list(DEFAULT_QWEN_CAMERA_SLOTS)))
        self.camera_slots = self.vjepa_camera_slots
        (
            self.append_subtask_to_prompt,
            self.subtask_prompt_source_column,
            self.subtask_prompt_label_column,
        ) = _canonical_subtask_prompt_settings(data_cfg)
        self.dataset_id_list = [str(value) for value in _as_list(_cfg_get(data_cfg, "dataset_ids", []))]
        self.dataset_ids = set(self.dataset_id_list)
        self.dataset_order = {dataset_id: idx for idx, dataset_id in enumerate(self.dataset_id_list)}
        self.exclude_dataset_ids_path_list = [
            Path(value) for value in _as_list(_cfg_get(data_cfg, "exclude_dataset_ids_path", []))
        ]
        self.exclude_sid_path_list = [
            Path(value) for value in _as_list(_cfg_get(data_cfg, "exclude_sids_path", []))
        ]
        exclude_dataset_ids = [str(value) for value in _as_list(_cfg_get(data_cfg, "exclude_dataset_ids", []))]
        for path in self.exclude_dataset_ids_path_list:
            exclude_dataset_ids.extend(_read_line_list(path))
        exclude_sids = [str(value) for value in _as_list(_cfg_get(data_cfg, "exclude_sids", []))]
        for path in self.exclude_sid_path_list:
            exclude_sids.extend(_read_line_list(path))
        self.exclude_dataset_id_list = list(dict.fromkeys(exclude_dataset_ids))
        self.exclude_dataset_ids = set(self.exclude_dataset_id_list)
        self.exclude_sid_list = list(dict.fromkeys(exclude_sids))
        self.exclude_sids = set(self.exclude_sid_list)
        self.adapter_group_ids = set(_as_list(_cfg_get(data_cfg, "adapter_group_ids", [])))
        self.preferred_fps = _as_float_filter_set(_cfg_get(data_cfg, "preferred_fps", []))
        self.epoch_sampling_strategy = str(
            _cfg_get(data_cfg, "epoch_sampling_strategy", "with_replacement")
        ).strip().lower()
        if self.epoch_sampling_strategy not in {
            "with_replacement",
            "all_sources_exhaustive",
        }:
            raise ValueError(
                "Canonical epoch_sampling_strategy must be "
                "'with_replacement' or 'all_sources_exhaustive'; got "
                f"{self.epoch_sampling_strategy!r}."
            )
        # ``max_shards=1`` is retained only as the legacy smoke-test default
        # for with-replacement sampling.  Once the user selects an exhaustive
        # epoch, an omitted cap must mean *all* shards; silently inheriting the
        # legacy one-shard default would contradict the requested contract.
        default_max_shards = (
            0
            if self.epoch_sampling_strategy == "all_sources_exhaustive"
            else 1
        )
        self.max_shards = int(
            _cfg_get(data_cfg, "max_shards", default_max_shards)
        )
        self.max_shards_per_dataset = int(
            _cfg_get(data_cfg, "max_shards_per_dataset", 0) or 0
        )
        self.max_windows = int(
            _cfg_get(data_cfg, "max_windows", 0) or 0
        )
        self.max_windows_per_dataset = int(
            _cfg_get(data_cfg, "max_windows_per_dataset", 0) or 0
        )
        self.epoch_sampling_algorithm_version = {
            "with_replacement": "legacy_canonical_index_v1",
            "all_sources_exhaustive": "all_sources_exhaustive_affine_v1",
        }[self.epoch_sampling_strategy]
        self.seed = int(
            _cfg_get(
                data_cfg,
                "seed",
                _cfg_get(data_cfg, "window_sample_seed", 0),
            )
        )
        raw_min_episodes = _cfg_get(
            data_cfg,
            "canonical_eval_min_episodes_per_shard",
            1,
        )
        if isinstance(raw_min_episodes, bool) or not isinstance(
            raw_min_episodes, int
        ):
            raise ValueError(
                "canonical_eval_min_episodes_per_shard must be an integer."
            )
        self.canonical_eval_min_episodes_per_shard = int(raw_min_episodes)
        if self.canonical_eval_min_episodes_per_shard < 1:
            raise ValueError(
                "canonical_eval_min_episodes_per_shard must be at least 1."
            )
        if (
            self.canonical_eval_manifest is not None
            and self.canonical_eval_min_episodes_per_shard < 2
        ):
            raise ValueError(
                "canonical_eval_manifest requires "
                "canonical_eval_min_episodes_per_shard>=2 so every selected "
                "heldout shard can retain a distinct training episode."
            )
        configured_sample_stride = int(_cfg_get(data_cfg, "sample_stride", 1))
        self.sample_stride = max(1, configured_sample_stride)
        self.video_horizon = int(video_horizon)
        self.action_horizon = int(action_horizon)
        self.video_frame_stride = max(1, int(video_frame_stride))
        self.video_target_shift_steps = max(0, int(_cfg_get(data_cfg, "video_target_shift_steps", 0)))
        self._action_offsets = np.arange(self.action_horizon, dtype=np.int64)
        self._compact_offsets_cache: np.ndarray | None = None
        self.video_resolution_size = int(_cfg_get(data_cfg, "video_resolution_size", 384))
        self.video_decode_backend = str(_cfg_get(data_cfg, "video_decode_backend", "auto")).lower()
        self.sidecar_normalization = str(
            _cfg_get(
                data_cfg,
                "sidecar_normalization",
                SHARD_Q01_Q99_UNCLIPPED,
            )
        ).lower()
        if self.sidecar_normalization not in {
            "none",
            SHARD_Q01_Q99,
            SHARD_Q01_Q99_UNCLIPPED,
            Q01_Q99_UNCLIPPED,
        }:
            raise ValueError(
                "Unsupported canonical sidecar_normalization "
                f"{self.sidecar_normalization!r}; expected 'none', "
                f"{SHARD_Q01_Q99!r}, {SHARD_Q01_Q99_UNCLIPPED!r}, "
                f"or shared-statistics mode {Q01_Q99_UNCLIPPED!r}."
            )
        if (
            self.sidecar_normalization == Q01_Q99_UNCLIPPED
            and self.normalization_statistics is None
            and not self.allow_eval_selection_population_candidate
        ):
            raise ValueError(
                "Canonical sidecar_normalization='q01_q99_unclipped' is "
                "reserved for exact 18-D shared union statistics and requires "
                "normalization_statistics_artifact."
            )
        self.sidecar_dtype = np.float16 if str(_cfg_get(data_cfg, "sidecar_dtype", "float16")) == "float16" else np.float32
        self.action_type = str(_cfg_get(data_cfg, "action_type", "dataset_native")).lower()
        self.action_delta_anchor = str(
            _cfg_get(data_cfg, "action_delta_anchor", "chunk_start_state")
        ).lower()
        self.gripper_action_type = str(
            _cfg_get(data_cfg, "gripper_action_type", "absolute")
        ).lower()
        self.absolute_action_references = {
            str(value).lower()
            for value in _as_list(
                _cfg_get(
                    data_cfg,
                    "absolute_action_references",
                    list(DEFAULT_ABSOLUTE_ACTION_REFERENCES),
                )
            )
        }
        if self.action_type not in {
            "dataset_native",
            "absolute_qpos",
            JOINT_DELTA_GRIPPER_ABSOLUTE,
        }:
            raise ValueError(f"Unsupported canonical action_type: {self.action_type!r}")
        if (
            self.action_type == JOINT_DELTA_GRIPPER_ABSOLUTE
            and self.action_delta_anchor != "chunk_start_state"
        ):
            raise ValueError(
                "Canonical joint deltas require action_delta_anchor=chunk_start_state."
            )
        if (
            self.action_type == JOINT_DELTA_GRIPPER_ABSOLUTE
            and self.gripper_action_type != "absolute"
        ):
            raise ValueError(
                "joint_delta_gripper_absolute requires gripper_action_type=absolute."
            )
        if self.normalization_statistics is not None:
            if self.action_type != JOINT_DELTA_GRIPPER_ABSOLUTE:
                raise ValueError(
                    "Canonical 18-D shared normalization requires "
                    "action_type=joint_delta_gripper_absolute."
                )
            if self.action_horizon != REALMAN_18D_ACTION_CONTRACT.action_horizon:
                raise ValueError(
                    "Canonical 18-D shared normalization requires "
                    f"action_horizon={REALMAN_18D_ACTION_CONTRACT.action_horizon}, "
                    f"got {self.action_horizon}."
                )
            if self.sidecar_normalization not in {
                SHARD_Q01_Q99_UNCLIPPED,
                Q01_Q99_UNCLIPPED,
            }:
                raise ValueError(
                    "Canonical 18-D shared normalization requires raw-valued "
                    "joint-delta sidecars and either the legacy cache variant "
                    f"{SHARD_Q01_Q99_UNCLIPPED!r} or the explicit shared mode "
                    f"{Q01_Q99_UNCLIPPED!r}."
                )
            selected_statistics = self.normalization_statistics.get("selected")
            if (
                not isinstance(selected_statistics, dict)
                or not isinstance(selected_statistics.get("state"), dict)
                or not isinstance(selected_statistics.get("action"), dict)
            ):
                raise ValueError(
                    "Canonical shared normalization artifact must contain "
                    "selected.state and selected.action statistics."
                )
        self.action_sidecar_variant = canonical_action_sidecar_variant(
            data_cfg,
            action_horizon=self.action_horizon,
            canonical_eval_manifest_sha256=(
                None
                if self.canonical_eval_manifest is None
                else self.canonical_eval_manifest.sha256
            ),
            exclude_eval_episodes_from_training=(
                self.exclude_eval_episodes_from_training
            ),
            adapter_contract_sha256=self.adapter_contract_sha256,
        )
        if configured_frozen_view and self.mode == "train":
            self._load_frozen_train_view_descriptor(
                Path(str(configured_frozen_view)),
                expected_manifest_sha256=str(
                    configured_frozen_view_sha256
                ),
            )
        self.lazy_cache_shards = bool(_cfg_get(data_cfg, "lazy_cache_shards", False))
        self.index_windows_lazily = bool(_cfg_get(data_cfg, "index_windows_lazily", False))
        self.prefetch_metadata_across_ranks = bool(_cfg_get(data_cfg, "prefetch_metadata_across_ranks", False))
        self.metadata_index_cache = bool(_cfg_get(data_cfg, "metadata_index_cache", True))
        self.metadata_prefetch_workers = max(1, int(_cfg_get(data_cfg, "metadata_prefetch_workers", 1)))
        self.data_file_prefetch_shards = max(0, int(_cfg_get(data_cfg, "data_file_prefetch_shards", 0)))
        self.metadata_index_cache_dir = Path(
            _cfg_get(data_cfg, "metadata_index_cache_dir", self.cache_dir / ".canonical_index_cache")
        )
        configured_index_cache_path = _cfg_get(data_cfg, "metadata_index_cache_path", None)
        self.metadata_index_cache_path = Path(configured_index_cache_path) if configured_index_cache_path else None
        self.shuffle_shards = bool(_cfg_get(data_cfg, "shuffle_shards", False))
        self.shuffle_seed = int(_cfg_get(data_cfg, "shuffle_seed", _cfg_get(data_cfg, "window_sample_seed", 0)))
        self._validate_exact_epoch_settings(
            mode=self.mode,
            epoch_sampling_strategy=self.epoch_sampling_strategy,
            max_shards=self.max_shards,
            max_shards_per_dataset=self.max_shards_per_dataset,
            max_windows=self.max_windows,
            max_windows_per_dataset=self.max_windows_per_dataset,
            shuffle_shards=self.shuffle_shards,
            sample_stride=configured_sample_stride,
        )
        self.reader_cache_size = max(int(_cfg_get(data_cfg, "reader_cache_size", 64)), 0)
        self.sidecar_cache_size = max(int(_cfg_get(data_cfg, "sidecar_cache_size", 16)), 0)
        self.slow_sample_log_seconds = float(_cfg_get(data_cfg, "slow_sample_log_seconds", 0.0) or 0.0)
        self.pyav_corrupt_warning_limit = max(int(_cfg_get(data_cfg, "pyav_corrupt_warning_limit", 20)), 0)
        self.pyav_decode_retry_extra_frames = max(int(_cfg_get(data_cfg, "pyav_decode_retry_extra_frames", 120)), 0)
        self.skip_corrupt_videos = bool(_cfg_get(data_cfg, "skip_corrupt_videos", True))
        self.max_sample_decode_retries = max(int(_cfg_get(data_cfg, "max_sample_decode_retries", 128)), 0)
        self.fail_on_sample_error = bool(
            _cfg_get(
                data_cfg,
                "fail_on_sample_error",
                self.epoch_sampling_strategy == "all_sources_exhaustive",
            )
        )
        if (
            self.mode == "train"
            and self.epoch_sampling_strategy == "all_sources_exhaustive"
            and not self.fail_on_sample_error
        ):
            raise ValueError(
                "Canonical all_sources_exhaustive requires "
                "fail_on_sample_error=true."
            )
        self.pyav_max_missing_frames_for_fill = max(int(_cfg_get(data_cfg, "pyav_max_missing_frames_for_fill", 0)), 0)
        self.pyav_fail_on_decode_error_recovery = bool(_cfg_get(data_cfg, "pyav_fail_on_decode_error_recovery", True))
        self.pyav_max_nearest_fill_distance = max(
            int(_cfg_get(data_cfg, "pyav_max_nearest_fill_distance", 2)),
            0,
        )
        self.pyav_reader_cache_size = max(int(_cfg_get(data_cfg, "pyav_reader_cache_size", self.reader_cache_size)), 0)
        self.pyav_thread_count = _parse_pyav_thread_count(_cfg_get(data_cfg, "pyav_thread_count", 1))
        self.pyav_thread_type = str(_cfg_get(data_cfg, "pyav_thread_type", "SLICE")).upper()
        self.video_cache_max_bytes = int(
            float(_cfg_get(data_cfg, "video_cache_max_gb", 0) or 0) * 1024 * 1024 * 1024
        )
        self.video_cache_prune_interval_downloads = max(
            1, int(_cfg_get(data_cfg, "video_cache_prune_interval_downloads", 16))
        )
        self.video_cache_prune_target_fraction = min(
            1.0, max(0.1, float(_cfg_get(data_cfg, "video_cache_prune_target_fraction", 0.9)))
        )
        self._apply_unified_adapter, self._load_adapter_config = _load_canonical_modules(
            self.dataset_canonicalization_root
        )
        self._adapter_manifest = self._load_adapter_manifest()
        self._decord_readers: OrderedDict[str, Any] = OrderedDict()
        self._pyav_readers: OrderedDict[str, _PyAVReader] = OrderedDict()
        self._pyav_corrupt_warning_count = 0
        self._bad_video_paths: set[str] = set()
        self._bad_video_warning_count = 0
        self._loaded_shards: OrderedDict[int, _ShardData] = OrderedDict()
        self._known_local_relative_paths: set[tuple[str, str]] = set()
        self._redownloaded_relative_paths: set[tuple[str, str]] = set()
        self._video_cache_prune_download_count = 0
        self._shard_prefetch_executor: ThreadPoolExecutor | None = None
        self._shard_prefetch_futures: OrderedDict[int, Any] = OrderedDict()
        self._shard_prefetch_seen: set[int] = set()
        self._metadata_index_cache_key = (
            self._build_metadata_index_cache_key() if self.metadata_index_cache else ""
        )

        self.shards = self._resolve_shards_with_index_cache()
        self._full_episode_identities = self._episode_identity_set(self.shards)
        handoff_smoke = (
            self._validate_checkpoint_handoff_smoke_eval_binding()
        )
        if self.canonical_eval_manifest is not None:
            if not handoff_smoke:
                selection = self.canonical_eval_manifest.selection
                configured_episode_identities = self._full_episode_identities
                if self.frozen_train_view is not None:
                    # The frozen view defines this experiment's population.
                    # Its production ledger is already holdout-free, so
                    # reconstruct the exact pre-holdout selection as train ∪
                    # holdout.
                    configured_episode_identities = frozenset(
                        {
                            *self._frozen_view_episode_identity_set(),
                            *self.canonical_eval_manifest.heldout_episode_identities,
                        }
                    )
                catalog_count = len(configured_episode_identities)
                catalog_sha256 = _stable_json_sha256(
                    sorted(configured_episode_identities)
                )
                if selection.configured_episode_count != catalog_count:
                    raise ValueError(
                        "Canonical evaluation manifest "
                        "configured_episode_count does not match the current "
                        f"filtered stream: {selection.configured_episode_count} "
                        f"!= {catalog_count}."
                    )
                if (
                    selection.configured_episode_catalog_sha256
                    != catalog_sha256
                ):
                    raise ValueError(
                        "Canonical evaluation manifest episode catalog hash "
                        "does not match the current filtered stream."
                    )
        if self.mode == "train" and self.canonical_eval_manifest is not None:
            self.shards = self._without_heldout_episodes(self.shards)
            if not self.shards:
                raise RuntimeError(
                    "Canonical heldout split removed every selected training episode."
                )
        self._validate_subtask_prompt_coverage()
        if self.shuffle_shards:
            random.Random(self.shuffle_seed).shuffle(self.shards)
        if self.frozen_train_view is not None:
            self._initialize_frozen_train_view_index()
            self.index_windows_lazily = True
            self._window_ranges = []
            self._window_range_ends = []
            self.total_windows = int(self.frozen_train_view.row_count)
            self.windows = []
            self.epoch_sampling_algorithm_version = (
                "all_sources_exhaustive_frozen_view_affine_v1"
            )
        else:
            self._window_ranges = self._build_window_ranges()
            self._window_range_ends = [
                window_range.cumulative_end
                for window_range in self._window_ranges
            ]
            self.total_windows = (
                self._window_range_ends[-1]
                if self._window_range_ends
                else 0
            )
            self.windows = (
                [] if self.index_windows_lazily else self._build_windows()
            )
        if (self.index_windows_lazily and self.total_windows <= 0) or (
            not self.index_windows_lazily and not self.windows
        ):
            raise RuntimeError(
                "No canonical VLA training windows were found. Check cached shards, camera slots, "
                "adapter filters and gcloud auth if allow_gcs_download=true."
            )
        self._initialize_epoch_schedule()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # File descriptors and memmaps are worker-local runtime state.  A
        # spawned/forkserver worker reopens the immutable ledger and offset
        # index lazily; serializing either object can retain a stale cursor or
        # duplicate an open descriptor across processes.
        state["_frozen_view_offsets"] = None
        state["_frozen_view_ledger_handle"] = None
        state["_decord_readers"] = OrderedDict()
        state["_pyav_readers"] = OrderedDict()
        state["_pyav_corrupt_warning_count"] = 0
        state["_bad_video_paths"] = set()
        state["_bad_video_warning_count"] = 0
        state["_loaded_shards"] = OrderedDict()
        state["_known_local_relative_paths"] = set()
        state["_redownloaded_relative_paths"] = set()
        state["_video_cache_prune_download_count"] = 0
        state["_shard_prefetch_executor"] = None
        state["_shard_prefetch_futures"] = OrderedDict()
        state["_shard_prefetch_seen"] = set()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._frozen_view_offsets = None
        self._frozen_view_ledger_handle = None
        self.__dict__.setdefault(
            "epoch_sampling_strategy", "with_replacement"
        )
        self.__dict__.setdefault(
            "epoch_sampling_algorithm_version",
            "legacy_canonical_index_v1",
        )
        self.__dict__.setdefault("seed", 0)
        self.__dict__.setdefault("_epoch_permutation_cache", {})
        if not hasattr(self, "_shared_epoch"):
            self._shared_epoch = torch.tensor(
                int(getattr(self, "epoch", 0)), dtype=torch.int64
            ).share_memory_()

    def _initialize_epoch_schedule(self) -> None:
        """Bind one exact logical epoch to every selected canonical window."""

        window_count = int(self.total_windows)
        if not self.index_windows_lazily:
            window_count = int(len(self.windows))
            if window_count != int(self.total_windows):
                raise RuntimeError(
                    "Canonical eager and lazy window indices disagree: "
                    f"{window_count} != {self.total_windows}."
                )
        if window_count <= 0:
            raise RuntimeError("Canonical epoch schedule cannot be empty.")
        self._dataset_lengths = np.asarray([window_count], dtype=np.int64)
        self._raw_dataset_sampling_weights = np.asarray([1.0], dtype=np.float64)
        self._primary_dataset_indices = np.asarray([True], dtype=np.bool_)
        self._epoch_dataset_counts = self._dataset_lengths.copy()
        self._epoch_permutation_cache: dict[tuple[int, int], tuple[int, int]] = {}
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        # The trainer fingerprints child provenance for exact cursor
        # authentication. Canonical is a single logical source, so expose
        # itself as that source without introducing a sampling wrapper.
        self.datasets = (self,)
        self.set_epoch(0)

    @staticmethod
    def _valid_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def _build_frozen_episode_lookup(
        self,
    ) -> tuple[
        dict[tuple[str, str, str, int], tuple[int, int]],
        str,
    ]:
        lookup: dict[tuple[str, str, str, int], tuple[int, int]] = {}
        catalog_rows: list[dict[str, Any]] = []
        for shard_index, shard in enumerate(self.shards):
            for local_episode_index, episode in enumerate(shard.episodes):
                if episode.episode_index is None:
                    raise ValueError(
                        "Frozen canonical views require source episode_index "
                        "metadata for every selected episode."
                    )
                key = (
                    str(shard.dataset_id),
                    str(shard.sid),
                    str(shard.revision),
                    int(episode.episode_index),
                )
                if key in lookup:
                    other_shard_index, _ = lookup[key]
                    other = self.shards[other_shard_index]
                    raise ValueError(
                        "Frozen canonical row identity is ambiguous across "
                        "multiple data files: "
                        f"{key!r} maps to {other.data_relative_path!r} and "
                        f"{shard.data_relative_path!r}. Extend the ledger "
                        "identity with data_file before using this catalog."
                    )
                lookup[key] = (shard_index, local_episode_index)
                catalog_rows.append(
                    {
                        "dataset_id": key[0],
                        "sid": key[1],
                        "revision": key[2],
                        "episode_index": key[3],
                        "data_file": shard.data_relative_path,
                        "length": int(episode.length),
                        "adapter_sha256": shard.adapter_sha256,
                    }
                )
        if not lookup:
            raise RuntimeError(
                "Frozen canonical view cannot bind to an empty episode catalog."
            )
        catalog_sha256 = _stable_json_sha256(
            {
                "schema": _FROZEN_CANONICAL_ROW_BINDING_SCHEMA,
                "source_manifest_sha256": _hash_file(self.manifest_path),
                "adapter_contract_sha256": self.adapter_contract_sha256,
                "episodes": sorted(
                    catalog_rows,
                    key=lambda row: (
                        row["dataset_id"],
                        row["sid"],
                        row["revision"],
                        row["episode_index"],
                    ),
                ),
            }
        )
        return lookup, catalog_sha256

    def _frozen_index_cache_paths(
        self, ledger_sha256: str
    ) -> tuple[Path, Path, Path]:
        configured = _cfg_get(
            self.data_cfg, "frozen_train_view_index_cache_dir", None
        )
        cache_dir = (
            Path(str(configured)).expanduser().resolve()
            if configured
            else (
                self.metadata_index_cache_dir
                / "frozen_train_views"
            ).resolve()
        )
        encoding = self.frozen_train_view.encoding
        stem = (
            f"{self.frozen_train_view.view_id[:16]}."
            f"{ledger_sha256[:16]}."
            f"{encoding}"
        )
        return (
            cache_dir / f"{stem}.offsets.u64",
            cache_dir / f"{stem}.offsets.json",
            cache_dir / f"{stem}.offsets.lock",
        )

    def _frozen_index_expected_metadata(
        self,
        *,
        ledger_sha256: str,
        ledger_size: int,
        identity_catalog_sha256: str,
    ) -> dict[str, Any]:
        assert self.frozen_train_view is not None
        is_range_view = (
            self.frozen_train_view.encoding
            == EPISODE_RANGES_ENCODING
        )
        return {
            "schema": (
                _FROZEN_CANONICAL_RANGE_INDEX_SCHEMA
                if is_range_view
                else _FROZEN_CANONICAL_INDEX_SCHEMA
            ),
            "ledger_encoding": self.frozen_train_view.encoding,
            "view_id": self.frozen_train_view.view_id,
            "manifest_sha256": (
                self.frozen_train_view_manifest_sha256
            ),
            "ledger_sha256": ledger_sha256,
            "ledger_size_bytes": int(ledger_size),
            "row_count": int(self.frozen_train_view.row_count),
            "record_count": int(self.frozen_train_view.record_count),
            "offset_count": (
                int(self.frozen_train_view.record_count) + 1
            ),
            "offset_dtype": "<u8",
            "index_columns": (
                ["ledger_byte_offset", "logical_cumulative_end"]
                if is_range_view
                else ["ledger_byte_offset"]
            ),
            "identity_catalog_sha256": identity_catalog_sha256,
            "representation_contract_sha256": (
                REALMAN_18D_ACTION_CONTRACT.sha256()
            ),
            "row_binding_schema": _FROZEN_CANONICAL_ROW_BINDING_SCHEMA,
        }

    @staticmethod
    def _validate_frozen_offset_array(
        index_path: Path,
        *,
        row_count: int,
        record_count: int,
        ledger_size: int,
        ledger_encoding: str,
    ) -> None:
        column_count = (
            2
            if ledger_encoding == EPISODE_RANGES_ENCODING
            else 1
        )
        expected_size = (
            (int(record_count) + 1)
            * column_count
            * np.dtype("<u8").itemsize
        )
        actual_size = index_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                "Frozen canonical offset index has the wrong byte size: "
                f"expected {expected_size}, found {actual_size}."
            )
        offsets = np.memmap(
            index_path,
            mode="r",
            dtype="<u8",
            shape=(
                (int(record_count) + 1, column_count)
                if column_count > 1
                else (int(record_count) + 1,)
            ),
        )
        try:
            byte_offsets = (
                offsets[:, 0] if column_count > 1 else offsets
            )
            if (
                int(byte_offsets[0]) != 0
                or int(byte_offsets[-1]) != int(ledger_size)
            ):
                raise RuntimeError(
                    "Frozen canonical offset index boundary mismatch."
                )
            chunk_size = 1_000_000
            for start in range(0, int(record_count), chunk_size):
                stop = min(
                    int(record_count) + 1,
                    start + chunk_size + 1,
                )
                block = np.asarray(
                    byte_offsets[start:stop], dtype=np.uint64
                )
                if np.any(block[1:] <= block[:-1]):
                    raise RuntimeError(
                        "Frozen canonical offset index is not strictly "
                        f"increasing near row {start}."
                    )
            if column_count > 1:
                cumulative = offsets[:, 1]
                if (
                    int(cumulative[0]) != 0
                    or int(cumulative[-1]) != int(row_count)
                ):
                    raise RuntimeError(
                        "Frozen canonical range index logical-count "
                        "boundary mismatch."
                    )
                for start in range(
                    0, int(record_count), chunk_size
                ):
                    stop = min(
                        int(record_count) + 1,
                        start + chunk_size + 1,
                    )
                    block = np.asarray(
                        cumulative[start:stop], dtype=np.uint64
                    )
                    if np.any(block[1:] <= block[:-1]):
                        raise RuntimeError(
                            "Frozen canonical range cumulative index is not "
                            f"strictly increasing near record {start}."
                        )
        finally:
            del offsets

    def _validate_cached_frozen_index(
        self,
        *,
        index_path: Path,
        metadata_path: Path,
        expected_metadata: dict[str, Any],
    ) -> str:
        if not index_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(
                "Frozen canonical offset index is incomplete; both the "
                f"binary index and metadata are required: {index_path}, "
                f"{metadata_path}."
            )
        raw_metadata = metadata_path.read_bytes()
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Frozen canonical offset metadata is invalid: {metadata_path}"
            ) from exc
        if raw_metadata != frozen_view_canonical_json_bytes(metadata) + b"\n":
            raise RuntimeError(
                "Frozen canonical offset metadata is not canonical JSON."
            )
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    "Frozen canonical offset metadata binding mismatch for "
                    f"{key}: expected {expected!r}, "
                    f"found {metadata.get(key)!r}."
                )
        index_sha256 = metadata.get("index_sha256")
        if not self._valid_sha256(index_sha256):
            raise RuntimeError(
                "Frozen canonical offset metadata lacks index_sha256."
            )
        actual_sha256 = frozen_view_file_sha256(index_path)
        if actual_sha256 != index_sha256:
            raise RuntimeError(
                "Frozen canonical offset index SHA-256 mismatch: "
                f"expected {index_sha256}, found {actual_sha256}."
            )
        self._validate_frozen_offset_array(
            index_path,
            row_count=int(expected_metadata["row_count"]),
            record_count=int(expected_metadata["record_count"]),
            ledger_size=int(expected_metadata["ledger_size_bytes"]),
            ledger_encoding=str(
                expected_metadata["ledger_encoding"]
            ),
        )
        return actual_sha256

    def _canonical_frozen_row_identity(
        self, row: dict[str, Any], *, ordinal: int
    ) -> tuple[tuple[str, str, str, int], int, int]:
        assert self.frozen_train_view is not None
        context = f"frozen canonical ledger row {ordinal}"
        if row.get("schema") != FROZEN_VIEW_ROW_SCHEMA:
            raise ValueError(f"{context} has an invalid schema.")
        if row.get("ordinal") != ordinal:
            raise ValueError(
                f"{context} ordinal mismatch: {row.get('ordinal')!r}."
            )
        if row.get("backend") != "canonical":
            raise ValueError(f"{context} is not a canonical row.")
        sources = getattr(self, "_frozen_view_sources_by_id", None)
        if sources is None:
            sources = {
                str(source["source_id"]): source
                for source in self.frozen_train_view.descriptor["sources"]
            }
            self._frozen_view_sources_by_id = sources
        source_id = row.get("source_id")
        if source_id not in sources:
            raise ValueError(
                f"{context} references unknown source_id={source_id!r}."
            )
        identity_values = []
        for field in ("dataset_id", "sid", "revision"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{context}.{field} must be a non-empty string."
                )
            expected = sources[str(source_id)].get(field)
            if expected is not None and expected != value:
                raise ValueError(
                    f"{context}.{field}={value!r} does not match its source "
                    f"descriptor value {expected!r}."
                )
            identity_values.append(value)
        episode_index = row.get("episode_index")
        base_index = row.get("base_index")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ValueError(f"{context}.episode_index is invalid.")
        if (
            isinstance(base_index, bool)
            or not isinstance(base_index, int)
            or base_index < 0
        ):
            raise ValueError(f"{context}.base_index is invalid.")
        representation = self.frozen_train_view.descriptor["representation"]
        horizon = row.get("horizon")
        target_fps = row.get("target_fps")
        if horizon != int(representation["horizon"]):
            raise ValueError(f"{context}.horizon is inconsistent.")
        if target_fps != int(representation["target_fps"]):
            raise ValueError(f"{context}.target_fps is inconsistent.")
        if row.get("end_clamp_policy") not in {
            "repeat_last",
            "no_end_clamp",
        }:
            raise ValueError(f"{context}.end_clamp_policy is invalid.")
        for field in (
            "episode_lineage_id",
            "episode_content_id",
            "sample_id",
        ):
            if not self._valid_sha256(row.get(field)):
                raise ValueError(f"{context}.{field} is invalid.")
        expected_sample_id = make_frozen_view_sample_id(
            episode_content_id=str(row["episode_content_id"]),
            base_index=int(base_index),
            horizon=int(horizon),
            target_fps=int(target_fps),
            representation_contract_sha256=str(
                representation["contract_sha256"]
            ),
            end_clamp=row["end_clamp_policy"] == "repeat_last",
        )
        if row["sample_id"] != expected_sample_id:
            raise ValueError(f"{context}.sample_id is inconsistent.")
        key = (
            identity_values[0],
            identity_values[1],
            identity_values[2],
            int(episode_index),
        )
        location = self._frozen_view_episode_lookup.get(key)
        if location is None:
            raise ValueError(
                f"{context} does not resolve in the selected canonical "
                f"catalog: identity={key!r}."
            )
        shard_index, local_episode_index = location
        shard = self.shards[shard_index]
        episode = shard.episodes[local_episode_index]
        if (
            row.get("data_file") is not None
            and row["data_file"] != shard.data_relative_path
        ):
            raise ValueError(
                f"{context}.data_file does not match the resolved shard."
            )
        if (
            row.get("adapter_sha256") is not None
            and row["adapter_sha256"] != shard.adapter_sha256
        ):
            raise ValueError(
                f"{context}.adapter_sha256 does not match the resolved shard."
            )
        source_base_index = self._target_to_source_episode_indices(
            np.asarray([base_index], dtype=np.int64),
            source_fps=float(shard.fps),
            target_fps=int(target_fps),
        )[0]
        if source_base_index < 0 or source_base_index >= int(episode.length):
            raise ValueError(
                f"{context}.base_index maps outside the source episode."
            )
        if row["end_clamp_policy"] == "no_end_clamp":
            final_source_index = self._target_to_source_episode_indices(
                np.asarray(
                    [int(base_index) + int(horizon) - 1],
                    dtype=np.int64,
                ),
                source_fps=float(shard.fps),
                target_fps=int(target_fps),
            )[0]
            if final_source_index >= int(episode.length):
                raise ValueError(
                    f"{context} declares no_end_clamp but its target horizon "
                    "extends beyond the source episode."
                )
        return key, int(base_index), int(source_base_index)

    def _canonical_frozen_range_identity(
        self,
        row: dict[str, Any],
        *,
        ordinal: int,
    ) -> tuple[tuple[str, str, str, int], int, int]:
        """Validate and bind one compact canonical range record."""

        assert self.frozen_train_view is not None
        context = f"frozen canonical range ledger row {ordinal}"
        if row.get("schema") != FROZEN_VIEW_RANGE_ROW_SCHEMA:
            raise ValueError(f"{context} has an invalid schema.")
        if row.get("ordinal") != ordinal:
            raise ValueError(
                f"{context} ordinal mismatch: {row.get('ordinal')!r}."
            )
        start = row.get("base_start")
        stop = row.get("base_stop")
        step = row.get("base_step")
        sample_count = row.get("sample_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (start, stop, step, sample_count)
        ):
            raise ValueError(
                f"{context} range bounds/counts must be integers."
            )
        expected_count = frozen_view_range_sample_count(
            base_start=int(start),
            base_stop=int(stop),
            base_step=int(step),
        )
        if int(sample_count) != expected_count:
            raise ValueError(
                f"{context}.sample_count={sample_count} does not match "
                f"range cardinality {expected_count}."
            )
        representation = self.frozen_train_view.descriptor[
            "representation"
        ]
        if row.get("horizon") != int(representation["horizon"]):
            raise ValueError(f"{context}.horizon is inconsistent.")
        if row.get("target_fps") != int(
            representation["target_fps"]
        ):
            raise ValueError(f"{context}.target_fps is inconsistent.")
        if row.get("end_clamp_policy") not in {
            "repeat_last",
            "no_end_clamp",
        }:
            raise ValueError(
                f"{context}.end_clamp_policy is invalid."
            )
        expected_range_id = make_frozen_view_range_id(
            episode_content_id=str(row.get("episode_content_id", "")),
            base_start=int(start),
            base_stop=int(stop),
            base_step=int(step),
            horizon=int(row["horizon"]),
            target_fps=int(row["target_fps"]),
            representation_contract_sha256=str(
                representation["contract_sha256"]
            ),
            end_clamp=row["end_clamp_policy"] == "repeat_last",
        )
        if row.get("range_id") != expected_range_id:
            raise ValueError(f"{context}.range_id is inconsistent.")

        # Resolve the first logical sample through the already strict expanded
        # row binder, including dataset/sid/revision/source selection.
        first_sample = {
            **row,
            "schema": FROZEN_VIEW_ROW_SCHEMA,
            "base_index": int(start),
            "sample_id": make_frozen_view_sample_id(
                episode_content_id=str(row["episode_content_id"]),
                base_index=int(start),
                horizon=int(row["horizon"]),
                target_fps=int(row["target_fps"]),
                representation_contract_sha256=str(
                    representation["contract_sha256"]
                ),
                end_clamp=(
                    row["end_clamp_policy"] == "repeat_last"
                ),
            ),
        }
        identity, _, source_base_index = (
            self._canonical_frozen_row_identity(
                first_sample, ordinal=ordinal
            )
        )
        shard_index, local_episode_index = (
            self._frozen_view_episode_lookup[identity]
        )
        shard = self.shards[shard_index]
        episode = shard.episodes[local_episode_index]

        data_file = row.get("data_file")
        if not isinstance(data_file, str) or not data_file:
            raise ValueError(
                f"{context}.data_file must be a non-empty string."
            )
        adapter_sha256 = row.get("adapter_sha256")
        if not self._valid_sha256(adapter_sha256):
            raise ValueError(
                f"{context}.adapter_sha256 must be a SHA-256 digest."
            )
        if data_file != shard.data_relative_path:
            raise ValueError(
                f"{context}.data_file does not match the resolved shard."
            )
        if adapter_sha256 != shard.adapter_sha256:
            raise ValueError(
                f"{context}.adapter_sha256 does not match the resolved shard."
            )
        if row.get("source_fps") is not None:
            source_fps = row["source_fps"]
            if (
                isinstance(source_fps, bool)
                or not isinstance(source_fps, (int, float))
                or not np.isfinite(source_fps)
                or not math.isclose(
                    float(source_fps),
                    float(shard.fps),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    f"{context}.source_fps does not match the source shard."
                )
        if row.get("source_episode_length") is not None:
            source_length = row["source_episode_length"]
            if (
                isinstance(source_length, bool)
                or not isinstance(source_length, int)
                or source_length != int(episode.length)
            ):
                raise ValueError(
                    f"{context}.source_episode_length does not match the "
                    "resolved source episode."
                )
        if row.get("annotation_sha256") is not None and not (
            self._valid_sha256(row["annotation_sha256"])
        ):
            raise ValueError(
                f"{context}.annotation_sha256 is invalid."
            )
        for field in (
            "annotation_ordinal",
            "annotation_episode_index",
        ):
            value = row.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{context}.{field} is invalid.")
        if row.get("selection_kind") is not None and (
            not isinstance(row["selection_kind"], str)
            or not row["selection_kind"]
        ):
            raise ValueError(
                f"{context}.selection_kind is invalid."
            )

        last_base = int(start) + (int(sample_count) - 1) * int(step)
        last_source_index = self._target_to_source_episode_indices(
            np.asarray([last_base], dtype=np.int64),
            source_fps=float(shard.fps),
            target_fps=int(row["target_fps"]),
        )[0]
        if last_source_index >= int(episode.length):
            raise ValueError(
                f"{context} extends beyond the source episode."
            )
        if row["end_clamp_policy"] == "no_end_clamp":
            final_horizon_source_index = (
                self._target_to_source_episode_indices(
                    np.asarray(
                        [
                            last_base
                            + int(row["horizon"])
                            - 1
                        ],
                        dtype=np.int64,
                    ),
                    source_fps=float(shard.fps),
                    target_fps=int(row["target_fps"]),
                )[0]
            )
            if final_horizon_source_index >= int(episode.length):
                raise ValueError(
                    f"{context} declares no_end_clamp but its final horizon "
                    "extends beyond the source episode."
                )
        return identity, int(start), int(sample_count)

    @staticmethod
    def _target_to_source_episode_indices(
        indices: np.ndarray,
        *,
        source_fps: float,
        target_fps: int,
    ) -> np.ndarray:
        if (
            not np.isfinite(source_fps)
            or source_fps <= 0
            or target_fps <= 0
        ):
            raise ValueError(
                "Frozen canonical source/target FPS must be positive."
            )
        values = np.asarray(indices, dtype=np.float64)
        # Nearest source frame with half ties toward +infinity. For the
        # production 30->20 Hz contract this is exactly (3*i + 1) // 2.
        return np.floor(
            values * float(source_fps) / float(target_fps) + 0.5
        ).astype(np.int64)

    def _build_frozen_offset_index(
        self,
        *,
        index_path: Path,
        metadata_path: Path,
        expected_metadata: dict[str, Any],
    ) -> str:
        assert self.frozen_train_view is not None
        if (
            self.frozen_train_view.encoding
            == EPISODE_RANGES_ENCODING
        ):
            return self._build_frozen_range_offset_index(
                index_path=index_path,
                metadata_path=metadata_path,
                expected_metadata=expected_metadata,
            )
        row_count = int(self.frozen_train_view.row_count)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_index = index_path.with_name(
            f".{index_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        temporary_metadata = metadata_path.with_name(
            f".{metadata_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        excluded = self.frozen_train_view.descriptor["holdout_exclusions"]
        excluded_lineages = set(excluded["lineage_ids"])
        excluded_contents = set(excluded["content_ids"])
        offsets = np.memmap(
            temporary_index,
            mode="w+",
            dtype="<u8",
            shape=(row_count + 1,),
        )
        position = 0
        observed_rows = 0
        observed_episodes = 0
        previous_order_key: tuple[str, str, str, int, int] | None = None
        previous_episode_key: tuple[str, str, str, int] | None = None
        try:
            with self.frozen_train_view.ledger_path.open("rb") as handle:
                for ordinal, raw_line in enumerate(handle):
                    if ordinal >= row_count:
                        raise ValueError(
                            "Frozen canonical ledger contains more rows than "
                            "its descriptor."
                        )
                    if not raw_line.endswith(b"\n"):
                        raise ValueError(
                            f"Frozen canonical ledger row {ordinal} lacks a newline."
                        )
                    offsets[ordinal] = np.uint64(position)
                    try:
                        row = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Frozen canonical ledger row {ordinal} is invalid JSON."
                        ) from exc
                    if (
                        raw_line
                        != frozen_view_canonical_json_bytes(row) + b"\n"
                    ):
                        raise ValueError(
                            f"Frozen canonical ledger row {ordinal} is not "
                            "canonical JSON."
                        )
                    identity, base_index, _ = (
                        self._canonical_frozen_row_identity(
                            row, ordinal=ordinal
                        )
                    )
                    if row["episode_lineage_id"] in excluded_lineages:
                        raise ValueError(
                            f"Frozen canonical ledger row {ordinal} overlaps "
                            "the heldout lineage set."
                        )
                    if row["episode_content_id"] in excluded_contents:
                        raise ValueError(
                            f"Frozen canonical ledger row {ordinal} overlaps "
                            "the heldout content set."
                        )
                    order_key = (*identity, base_index)
                    if (
                        previous_order_key is not None
                        and order_key <= previous_order_key
                    ):
                        raise ValueError(
                            "Frozen canonical ledger must be strictly ordered "
                            "by dataset_id/sid/revision/episode_index/base_index; "
                            f"row {ordinal} is out of order or duplicated."
                        )
                    if identity != previous_episode_key:
                        observed_episodes += 1
                        previous_episode_key = identity
                    previous_order_key = order_key
                    position += len(raw_line)
                    observed_rows += 1
            if observed_rows != row_count:
                raise ValueError(
                    "Frozen canonical ledger row-count mismatch: "
                    f"expected {row_count}, found {observed_rows}."
                )
            if observed_episodes != self.frozen_train_view.episode_count:
                raise ValueError(
                    "Frozen canonical ledger episode-count mismatch: "
                    f"expected {self.frozen_train_view.episode_count}, "
                    f"found {observed_episodes}."
                )
            if position != int(expected_metadata["ledger_size_bytes"]):
                raise ValueError(
                    "Frozen canonical ledger size changed while indexing."
                )
            offsets[row_count] = np.uint64(position)
            offsets.flush()
            del offsets
            index_sha256 = frozen_view_file_sha256(temporary_index)
            metadata = {
                **expected_metadata,
                "index_sha256": index_sha256,
            }
            temporary_metadata.write_bytes(
                frozen_view_canonical_json_bytes(metadata) + b"\n"
            )
            os.replace(temporary_index, index_path)
            os.replace(temporary_metadata, metadata_path)
            return index_sha256
        except Exception:
            try:
                del offsets
            except UnboundLocalError:
                pass
            for path in (temporary_index, temporary_metadata):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def _build_frozen_range_offset_index(
        self,
        *,
        index_path: Path,
        metadata_path: Path,
        expected_metadata: dict[str, Any],
    ) -> str:
        """Build byte offsets plus cumulative logical sample counts."""

        assert self.frozen_train_view is not None
        record_count = int(self.frozen_train_view.record_count)
        logical_row_count = int(self.frozen_train_view.row_count)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_index = index_path.with_name(
            f".{index_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        temporary_metadata = metadata_path.with_name(
            f".{metadata_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        excluded = self.frozen_train_view.descriptor[
            "holdout_exclusions"
        ]
        excluded_lineages = set(excluded["lineage_ids"])
        excluded_contents = set(excluded["content_ids"])
        index = np.memmap(
            temporary_index,
            mode="w+",
            dtype="<u8",
            shape=(record_count + 1, 2),
        )
        position = 0
        cumulative = 0
        observed_records = 0
        observed_episodes = 0
        previous_order_key: tuple[
            str, str, str, int, int
        ] | None = None
        previous_episode_key: tuple[str, str, str, int] | None = None
        previous_episode_last_base = -1
        content_owners: dict[
            str, tuple[str, str, str, int]
        ] = {}
        try:
            with self.frozen_train_view.ledger_path.open("rb") as handle:
                for ordinal, raw_line in enumerate(handle):
                    if ordinal >= record_count:
                        raise ValueError(
                            "Frozen canonical range ledger contains more "
                            "records than its descriptor."
                        )
                    if not raw_line.endswith(b"\n"):
                        raise ValueError(
                            f"Frozen canonical range record {ordinal} lacks "
                            "a newline."
                        )
                    index[ordinal, 0] = np.uint64(position)
                    index[ordinal, 1] = np.uint64(cumulative)
                    try:
                        row = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "Frozen canonical range ledger record "
                            f"{ordinal} is invalid JSON."
                        ) from exc
                    if (
                        raw_line
                        != frozen_view_canonical_json_bytes(row) + b"\n"
                    ):
                        raise ValueError(
                            "Frozen canonical range ledger record "
                            f"{ordinal} is not canonical JSON."
                        )
                    identity, base_start, sample_count = (
                        self._canonical_frozen_range_identity(
                            row, ordinal=ordinal
                        )
                    )
                    if row["episode_lineage_id"] in excluded_lineages:
                        raise ValueError(
                            "Frozen canonical range record "
                            f"{ordinal} overlaps the heldout lineage set."
                        )
                    if row["episode_content_id"] in excluded_contents:
                        raise ValueError(
                            "Frozen canonical range record "
                            f"{ordinal} overlaps the heldout content set."
                        )
                    content_id = str(row["episode_content_id"])
                    owner = content_owners.setdefault(content_id, identity)
                    if owner != identity:
                        raise ValueError(
                            "Frozen canonical range ledger reuses episode "
                            "content under multiple source identities."
                        )
                    order_key = (*identity, base_start)
                    if (
                        previous_order_key is not None
                        and order_key <= previous_order_key
                    ):
                        raise ValueError(
                            "Frozen canonical range ledger must be strictly "
                            "ordered by dataset_id/sid/revision/"
                            "episode_index/base_start."
                        )
                    last_base = base_start + (
                        sample_count - 1
                    ) * int(row["base_step"])
                    if identity == previous_episode_key:
                        if base_start <= previous_episode_last_base:
                            raise ValueError(
                                "Frozen canonical range ledger contains "
                                "overlapping base indices for one episode."
                            )
                    else:
                        observed_episodes += 1
                        previous_episode_key = identity
                    previous_episode_last_base = last_base
                    previous_order_key = order_key
                    position += len(raw_line)
                    cumulative += sample_count
                    observed_records += 1

            if observed_records != record_count:
                raise ValueError(
                    "Frozen canonical range record-count mismatch: "
                    f"expected {record_count}, found {observed_records}."
                )
            if cumulative != logical_row_count:
                raise ValueError(
                    "Frozen canonical range logical row-count mismatch: "
                    f"expected {logical_row_count}, found {cumulative}."
                )
            if observed_episodes != self.frozen_train_view.episode_count:
                raise ValueError(
                    "Frozen canonical range episode-count mismatch: "
                    f"expected {self.frozen_train_view.episode_count}, "
                    f"found {observed_episodes}."
                )
            if position != int(expected_metadata["ledger_size_bytes"]):
                raise ValueError(
                    "Frozen canonical range ledger size changed while "
                    "indexing."
                )
            index[record_count, 0] = np.uint64(position)
            index[record_count, 1] = np.uint64(cumulative)
            index.flush()
            del index
            index_sha256 = frozen_view_file_sha256(temporary_index)
            metadata = {
                **expected_metadata,
                "index_sha256": index_sha256,
            }
            temporary_metadata.write_bytes(
                frozen_view_canonical_json_bytes(metadata) + b"\n"
            )
            os.replace(temporary_index, index_path)
            os.replace(temporary_metadata, metadata_path)
            return index_sha256
        except Exception:
            try:
                del index
            except UnboundLocalError:
                pass
            for path in (temporary_index, temporary_metadata):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def _initialize_frozen_train_view_index(self) -> None:
        assert self.frozen_train_view is not None
        self._frozen_view_sources_by_id = {
            str(source["source_id"]): source
            for source in self.frozen_train_view.descriptor["sources"]
        }
        (
            self._frozen_view_episode_lookup,
            identity_catalog_sha256,
        ) = self._build_frozen_episode_lookup()
        self.frozen_train_view_identity_catalog_sha256 = (
            identity_catalog_sha256
        )
        ledger_path = self.frozen_train_view.ledger_path
        ledger_sha256 = str(
            self.frozen_train_view.descriptor["rows"]["sha256"]
        )
        ledger_size = int(ledger_path.stat().st_size)
        index_path, metadata_path, lock_path = (
            self._frozen_index_cache_paths(ledger_sha256)
        )
        expected_metadata = self._frozen_index_expected_metadata(
            ledger_sha256=ledger_sha256,
            ledger_size=ledger_size,
            identity_catalog_sha256=identity_catalog_sha256,
        )
        with _exclusive_file_lock(lock_path):
            index_exists = index_path.exists()
            metadata_exists = metadata_path.exists()
            if index_exists or metadata_exists:
                index_sha256 = self._validate_cached_frozen_index(
                    index_path=index_path,
                    metadata_path=metadata_path,
                    expected_metadata=expected_metadata,
                )
            else:
                index_sha256 = self._build_frozen_offset_index(
                    index_path=index_path,
                    metadata_path=metadata_path,
                    expected_metadata=expected_metadata,
                )
                self._validate_cached_frozen_index(
                    index_path=index_path,
                    metadata_path=metadata_path,
                    expected_metadata=expected_metadata,
                )
        self.frozen_train_view_index_path = index_path
        self.frozen_train_view_index_metadata_path = metadata_path
        self.frozen_train_view_index_sha256 = index_sha256

    def _open_frozen_view_readers(self) -> tuple[np.memmap, Any]:
        if (
            self.frozen_train_view is None
            or self.frozen_train_view_index_path is None
        ):
            raise RuntimeError("Frozen canonical view is not initialized.")
        offsets = self._frozen_view_offsets
        if offsets is None:
            is_range_view = (
                self.frozen_train_view.encoding
                == EPISODE_RANGES_ENCODING
            )
            offsets = np.memmap(
                self.frozen_train_view_index_path,
                mode="r",
                dtype="<u8",
                shape=(
                    (self.frozen_train_view.record_count + 1, 2)
                    if is_range_view
                    else (self.frozen_train_view.record_count + 1,)
                ),
            )
            self._frozen_view_offsets = offsets
        handle = self._frozen_view_ledger_handle
        if handle is None or handle.closed:
            handle = self.frozen_train_view.ledger_path.open("rb")
            self._frozen_view_ledger_handle = handle
        return offsets, handle

    def _frozen_view_row(self, index: int) -> dict[str, Any]:
        assert self.frozen_train_view is not None
        index = int(index)
        if index < 0 or index >= self.frozen_train_view.row_count:
            raise IndexError(
                f"Frozen canonical row {index} is outside the ledger."
            )
        offsets, handle = self._open_frozen_view_readers()
        is_range_view = (
            self.frozen_train_view.encoding
            == EPISODE_RANGES_ENCODING
        )
        if is_range_view:
            cumulative = offsets[:, 1]
            record_index = int(
                np.searchsorted(
                    cumulative,
                    np.uint64(index),
                    side="right",
                )
                - 1
            )
            if (
                record_index < 0
                or record_index
                >= self.frozen_train_view.record_count
            ):
                raise RuntimeError(
                    "Frozen canonical range index failed to resolve logical "
                    f"row {index}."
                )
            start = int(offsets[record_index, 0])
            stop = int(offsets[record_index + 1, 0])
        else:
            record_index = index
            start = int(offsets[index])
            stop = int(offsets[index + 1])
        handle.seek(start)
        raw_line = handle.read(stop - start)
        if len(raw_line) != stop - start:
            raise RuntimeError(
                f"Frozen canonical ledger short read at row {index}."
            )
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Frozen canonical ledger row {index} became invalid."
            ) from exc
        if is_range_view:
            self._canonical_frozen_range_identity(
                row, ordinal=record_index
            )
            range_start = int(row["base_start"])
            local_index = index - int(offsets[record_index, 1])
            if local_index < 0 or local_index >= int(
                row["sample_count"]
            ):
                raise RuntimeError(
                    "Frozen canonical range index produced an invalid local "
                    f"offset {local_index} for record {record_index}."
                )
            base_index = range_start + local_index * int(
                row["base_step"]
            )
            representation = self.frozen_train_view.descriptor[
                "representation"
            ]
            expanded = {
                **row,
                "schema": FROZEN_VIEW_ROW_SCHEMA,
                "ordinal": index,
                "base_index": base_index,
                "sample_id": make_frozen_view_sample_id(
                    episode_content_id=str(
                        row["episode_content_id"]
                    ),
                    base_index=base_index,
                    horizon=int(row["horizon"]),
                    target_fps=int(row["target_fps"]),
                    representation_contract_sha256=str(
                        representation["contract_sha256"]
                    ),
                    end_clamp=(
                        row["end_clamp_policy"] == "repeat_last"
                    ),
                ),
                "range_ordinal": record_index,
                "range_id": str(row["range_id"]),
            }
            self._canonical_frozen_row_identity(
                expanded, ordinal=index
            )
            return expanded
        self._canonical_frozen_row_identity(row, ordinal=index)
        return row

    @property
    def dataset_lengths(self) -> np.ndarray:
        return self._dataset_lengths.copy()

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        return self._primary_dataset_indices.copy()

    @property
    def epoch_dataset_counts(self) -> np.ndarray:
        return self._epoch_dataset_counts.copy()

    @property
    def current_epoch(self) -> int:
        shared_epoch = getattr(self, "_shared_epoch", None)
        if shared_epoch is not None:
            return int(shared_epoch.item())
        return int(getattr(self, "epoch", 0))

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError(
                f"epoch must be a non-negative integer, got {epoch!r}"
            )
        epoch = int(epoch)
        self.epoch = epoch
        shared_epoch = getattr(self, "_shared_epoch", None)
        if shared_epoch is not None:
            shared_epoch.fill_(epoch)

    @staticmethod
    def _affine_permutation_parameters(
        length: int,
        seed: int,
    ) -> tuple[int, int]:
        if length <= 1:
            return 1, 0
        multiplier = int(seed % length) or 1
        while math.gcd(multiplier, length) != 1:
            multiplier += 1
            if multiplier >= length:
                multiplier = 1
        offset = int((seed >> 64) % length)
        return multiplier, offset

    def _epoch_window_index(self, index: int) -> int:
        """Map a loader slot bijectively onto the selected canonical windows."""

        index = int(index)
        # Keep the legacy direct-index behavior for compatibility with older
        # serialized datasets and lightweight test fixtures. Production
        # instances always initialize the authenticated schedule below.
        dataset_lengths = getattr(self, "_dataset_lengths", None)
        if dataset_lengths is None:
            return index
        length = int(dataset_lengths[0])
        if index < 0 or index >= length:
            raise IndexError(
                f"Canonical index {index} is outside [0, {length})."
            )
        if not (
            getattr(self, "mode", None) == "train"
            and getattr(self, "epoch_sampling_strategy", "with_replacement")
            == "all_sources_exhaustive"
        ):
            return index
        epoch = self.current_epoch
        cache_key = (epoch, length)
        parameters = self._epoch_permutation_cache.get(cache_key)
        if parameters is None:
            seed_payload = (
                f"{self.epoch_sampling_strategy}|{epoch}|{self.seed}|{length}"
            ).encode("utf-8")
            permutation_seed = int.from_bytes(
                hashlib.sha256(seed_payload).digest()[:16],
                byteorder="big",
                signed=False,
            )
            parameters = self._affine_permutation_parameters(
                length, permutation_seed
            )
            if len(self._epoch_permutation_cache) >= 128:
                self._epoch_permutation_cache.clear()
            self._epoch_permutation_cache[cache_key] = parameters
        multiplier, offset = parameters
        return int((multiplier * index + offset) % length)

    @staticmethod
    def _episode_identity(
        shard: ShardSpec,
        episode: EpisodeSpec,
    ) -> tuple[str, str, str, str, int]:
        if episode.episode_index is None:
            raise ValueError(
                "Canonical split binding requires source episode_index metadata."
            )
        return (
            str(shard.dataset_id),
            str(shard.sid),
            str(shard.revision),
            str(shard.data_relative_path),
            int(episode.episode_index),
        )

    @classmethod
    def _episode_identity_set(
        cls,
        shards: Sequence[ShardSpec],
    ) -> frozenset[tuple[str, str, str, str, int]]:
        return frozenset(
            cls._episode_identity(shard, episode)
            for shard in shards
            for episode in shard.episodes
        )

    def _frozen_view_episode_identity_set(
        self,
    ) -> frozenset[tuple[str, str, str, str, int]]:
        view = self.frozen_train_view
        if view is None:
            return frozenset()
        identities: set[tuple[str, str, str, str, int]] = set()
        for ordinal, row in enumerate(view.iter_rows()):
            strings = tuple(
                row.get(field)
                for field in ("dataset_id", "sid", "revision", "data_file")
            )
            episode_index = row.get("episode_index")
            if (
                any(
                    not isinstance(value, str) or not value
                    for value in strings
                )
                or isinstance(episode_index, bool)
                or not isinstance(episode_index, int)
                or episode_index < 0
            ):
                raise ValueError(
                    "Frozen canonical view contains an invalid exact episode "
                    f"identity at ledger record {ordinal}."
                )
            identities.add((*strings, int(episode_index)))
        if len(identities) != int(view.episode_count):
            raise ValueError(
                "Frozen canonical view episode identity cardinality does not "
                "match its authenticated descriptor."
            )
        return frozenset(identities)

    def _without_heldout_episodes(
        self,
        shards: Sequence[ShardSpec],
    ) -> list[ShardSpec]:
        assert self.canonical_eval_manifest is not None
        heldout = self.canonical_eval_manifest.heldout_episode_identities
        missing = heldout - self._episode_identity_set(shards)
        if missing:
            preview = sorted(missing)[:5]
            raise ValueError(
                "Canonical evaluation manifest references episodes outside the "
                f"configured canonical stream: {preview}."
            )
        filtered: list[ShardSpec] = []
        for shard in shards:
            kept = [
                episode
                for episode in shard.episodes
                if self._episode_identity(shard, episode) not in heldout
            ]
            if not kept:
                continue
            clone = copy.copy(shard)
            clone.episodes = kept
            filtered.append(clone)
        return filtered

    def _statistics_scope_for_shard(
        self,
        shard: ShardSpec,
        *,
        row_count: int,
    ) -> tuple[list[EpisodeSpec], np.ndarray]:
        episodes = list(shard.episodes)
        selected_rows = np.ones((int(row_count),), dtype=bool)
        if self.canonical_eval_manifest is None:
            return episodes, selected_rows

        heldout = self.canonical_eval_manifest.heldout_episode_identities
        episodes = [
            episode
            for episode in shard.episodes
            if self._episode_identity(shard, episode) not in heldout
        ]
        if not episodes:
            raise ValueError(
                "Canonical shard contains only heldout episodes, so train-only "
                "normalization statistics cannot be computed: "
                f"{shard.dataset_id}/{shard.sid}/{shard.data_relative_path}."
            )
        selected_rows[:] = False
        for episode in episodes:
            start = max(0, int(episode.local_start))
            stop = min(int(row_count), start + int(episode.length))
            selected_rows[start:stop] = True
        return episodes, selected_rows

    def close_video_readers(self) -> None:
        frozen_handle = getattr(
            self, "_frozen_view_ledger_handle", None
        )
        if frozen_handle is not None:
            try:
                frozen_handle.close()
            except Exception:
                pass
            self._frozen_view_ledger_handle = None
        frozen_offsets = getattr(self, "_frozen_view_offsets", None)
        if frozen_offsets is not None:
            mmap = getattr(frozen_offsets, "_mmap", None)
            if mmap is not None:
                try:
                    mmap.close()
                except Exception:
                    pass
            self._frozen_view_offsets = None
        executor = getattr(self, "_shard_prefetch_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            self._shard_prefetch_executor = None
            self._shard_prefetch_futures = OrderedDict()
        decord_readers = getattr(self, "_decord_readers", None)
        if decord_readers is not None:
            decord_readers.clear()
        pyav_readers = getattr(self, "_pyav_readers", None)
        if pyav_readers is None:
            return
        for reader in pyav_readers.values():
            try:
                reader.container.close()
            except Exception:
                pass
        pyav_readers.clear()

    def _make_pyav_reader(self, path_key: str) -> _PyAVReader:
        if av is None:
            raise RuntimeError("PyAV is required for canonical PyAV video decoding.")
        container = av.open(path_key, mode="r")
        try:
            stream = container.streams.video[0]
            if self.pyav_thread_count > 0:
                try:
                    stream.codec_context.thread_count = self.pyav_thread_count
                except Exception:
                    pass
            if self.pyav_thread_type and self.pyav_thread_type != "DEFAULT":
                try:
                    stream.thread_type = self.pyav_thread_type
                except Exception:
                    pass
            fps = float(stream.average_rate or stream.base_rate or 30.0)
            time_base = float(stream.time_base or 0.0)
            start_time = int(stream.start_time or 0)
            frame_count = int(stream.frames or 0)
            if frame_count <= 0 and stream.duration and time_base > 0:
                frame_count = int(round(float(stream.duration) * time_base * fps))
            return _PyAVReader(
                container=container,
                stream=stream,
                fps=fps,
                time_base=time_base,
                start_time=start_time,
                frame_count=frame_count,
            )
        except Exception:
            container.close()
            raise

    def _get_pyav_reader(self, path_key: str) -> _PyAVReader:
        if self.pyav_reader_cache_size <= 0:
            return self._make_pyav_reader(path_key)
        reader = self._pyav_readers.get(path_key)
        if reader is None:
            reader = self._make_pyav_reader(path_key)
            self._pyav_readers[path_key] = reader
        else:
            self._pyav_readers.move_to_end(path_key)
        while len(self._pyav_readers) > self.pyav_reader_cache_size:
            _, evicted = self._pyav_readers.popitem(last=False)
            try:
                evicted.container.close()
            except Exception:
                pass
        return reader

    def _drop_pyav_reader(self, path_key: str) -> None:
        reader = self._pyav_readers.pop(path_key, None)
        if reader is not None:
            try:
                reader.container.close()
            except Exception:
                pass

    def _load_adapter_manifest(self) -> dict[str, dict[str, Any]]:
        payload = json.loads((self.adapter_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        return {item["space_detail_fingerprint"]: item for item in payload["adapters"]}

    def _adapter_for_manifest_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        fingerprint = str(row.get("space_detail_fingerprint", ""))
        for adapter_fingerprint, adapter in self._adapter_manifest.items():
            if fingerprint.startswith(adapter_fingerprint[:12]) or adapter_fingerprint.startswith(fingerprint[:12]):
                if self.adapter_group_ids and adapter["adapter_group_id"] not in self.adapter_group_ids:
                    return None
                return adapter
        return None

    def _candidate_rows(self) -> list[tuple[dict[str, Any], dict[str, Any], Any]]:
        rows = _read_jsonl_gz(self.manifest_path)
        frozen_sources: frozenset[tuple[str, str, str]] | None = None
        if self.frozen_train_view is not None:
            frozen_sources = frozenset(
                (
                    str(source["dataset_id"]),
                    str(source["sid"]),
                    str(source["revision"]),
                )
                for source in self.frozen_train_view.descriptor["sources"]
            )
        candidates = []
        for row in rows:
            dataset_id = str(row.get("dataset_id", ""))
            sid = str(row.get("sid", ""))
            revision = str(row.get("revision", ""))
            if (
                frozen_sources is not None
                and (dataset_id, sid, revision) not in frozen_sources
            ):
                continue
            if self.dataset_ids and dataset_id not in self.dataset_ids:
                continue
            if self.exclude_dataset_ids and dataset_id in self.exclude_dataset_ids:
                continue
            if self.exclude_sids and sid in self.exclude_sids:
                continue
            if self.preferred_fps and float(row.get("fps") or 0.0) not in self.preferred_fps:
                continue
            adapter_meta = self._adapter_for_manifest_row(row)
            if adapter_meta is None:
                continue
            adapter_path = self.dataset_canonicalization_root / adapter_meta["path"]
            adapter = self._load_adapter_config(adapter_path)
            if not _select_qwen_camera_slots(adapter.image_mapping, self.qwen_camera_slots):
                continue
            candidates.append((row, adapter_meta, adapter))

        def sort_key(item: tuple[dict[str, Any], dict[str, Any], Any]) -> tuple[int, int, int]:
            row, _, _ = item
            root = self.cache_dir / row["sid"] / row["revision"]
            cached = int((root / "meta/info.json").exists())
            if self.dataset_order:
                dataset_rank = -self.dataset_order.get(str(row.get("dataset_id")), len(self.dataset_order))
            else:
                dataset_rank = 0
            return dataset_rank, cached, int(row.get("total_frames") or 0)

        candidates.sort(key=sort_key, reverse=True)
        return candidates

    @staticmethod
    def _distributed_rank_world() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _prefetch_metadata_roots(self, candidates: list[tuple[dict[str, Any], dict[str, Any], Any]]) -> None:
        if not self.allow_gcs_download:
            return
        rank, world_size = self._distributed_rank_world()
        if world_size <= 1 and self.metadata_prefetch_workers <= 1:
            return

        prefetch_candidates = candidates
        if self.max_shards:
            prefetch_candidates = candidates[: self.max_shards]

        pending: list[tuple[int, dict[str, Any]]] = []
        for index, (row, _, _) in enumerate(prefetch_candidates):
            root = self.cache_dir / row["sid"] / row["revision"]
            if not (root / "meta/info.json").exists():
                pending.append((index, row))

        if world_size <= 1:
            if pending:
                print(
                    "Canonical metadata local prefetch: "
                    f"{len(pending)} uncached metadata roots with {self.metadata_prefetch_workers} workers.",
                    file=sys.stderr,
                    flush=True,
                )

            def fetch_metadata(row: dict[str, Any]) -> bool:
                root = self.cache_dir / row["sid"] / row["revision"]
                gcs_prefix = _gcs_join(self.bucket_root, row["sid"], row["revision"])
                return (
                    _ensure_metadata_root(
                        root=root,
                        gcs_prefix=gcs_prefix,
                        allow_gcs_download=self.allow_gcs_download,
                        gcs_timeout_seconds=self.gcs_download_timeout_seconds,
                        gcs_retries=self.gcs_download_retries,
                        gcs_retry_backoff_seconds=self.gcs_download_retry_backoff_seconds,
                    )
                    is not None
                )

            local_count = 0
            with ThreadPoolExecutor(max_workers=self.metadata_prefetch_workers) as executor:
                futures = [executor.submit(fetch_metadata, row) for _, row in pending]
                for future in as_completed(futures):
                    if future.result():
                        local_count += 1
            if pending:
                print(
                    "Canonical metadata local prefetch complete; "
                    f"fetched {local_count} metadata roots.",
                    file=sys.stderr,
                    flush=True,
                )
            return

        if rank == 0 and pending:
            print(
                "Canonical metadata rank-sharded prefetch: "
                f"{len(pending)} uncached metadata roots across {world_size} ranks.",
                file=sys.stderr,
                flush=True,
            )

        local_count = 0
        for index, row in pending:
            if index % world_size != rank:
                continue
            root = self.cache_dir / row["sid"] / row["revision"]
            gcs_prefix = _gcs_join(self.bucket_root, row["sid"], row["revision"])
            if _ensure_metadata_root(
                root=root,
                gcs_prefix=gcs_prefix,
                allow_gcs_download=self.allow_gcs_download,
                gcs_timeout_seconds=self.gcs_download_timeout_seconds,
                gcs_retries=self.gcs_download_retries,
                gcs_retry_backoff_seconds=self.gcs_download_retry_backoff_seconds,
            ):
                local_count += 1

        if pending:
            torch.distributed.barrier()
            if rank == 0:
                print(
                    "Canonical metadata rank-sharded prefetch complete; "
                    f"rank 0 fetched {local_count} metadata roots.",
                    file=sys.stderr,
                    flush=True,
                )

    def _build_metadata_index_cache_key(self) -> str:
        payload = {
            "version": CANONICAL_INDEX_CACHE_VERSION,
            "manifest_path": self.manifest_path.resolve().as_posix(),
            "manifest_sha256": _hash_file(self.manifest_path),
            "adapter_dir": self.adapter_dir.resolve().as_posix(),
            "adapter_manifest_sha256": _hash_file(self.adapter_dir / "MANIFEST.json"),
            "adapter_dir_fingerprint": _fingerprint_directory(self.adapter_dir),
            "cache_dir": self.cache_dir.resolve().as_posix(),
            "bucket_root": self.bucket_root,
            "dataset_ids": self.dataset_id_list,
            "exclude_dataset_ids": self.exclude_dataset_id_list,
            "exclude_dataset_ids_path": [path.resolve().as_posix() for path in self.exclude_dataset_ids_path_list],
            "exclude_dataset_ids_path_sha256": {
                path.resolve().as_posix(): _hash_file(path) for path in self.exclude_dataset_ids_path_list
            },
            "exclude_sids": self.exclude_sid_list,
            "exclude_sids_path": [path.resolve().as_posix() for path in self.exclude_sid_path_list],
            "exclude_sids_path_sha256": {
                path.resolve().as_posix(): _hash_file(path) for path in self.exclude_sid_path_list
            },
            "adapter_group_ids": sorted(self.adapter_group_ids),
            "camera_slots": self.camera_slots,
            "qwen_camera_slots": self.qwen_camera_slots,
            "vjepa_camera_slots": self.vjepa_camera_slots,
            "append_subtask_to_prompt": self.append_subtask_to_prompt,
            "preferred_fps": sorted(self.preferred_fps),
            "allow_gcs_download": self.allow_gcs_download,
            "max_shards": self.max_shards,
            "max_shards_per_dataset": self.max_shards_per_dataset,
            "max_windows": self.max_windows,
            "max_windows_per_dataset": self.max_windows_per_dataset,
            "canonical_eval_min_episodes_per_shard": (
                self.canonical_eval_min_episodes_per_shard
            ),
            "sample_stride": self.sample_stride,
            "video_horizon": self.video_horizon,
            "action_horizon": self.action_horizon,
            "action_type": self.action_type,
            "action_delta_anchor": self.action_delta_anchor,
            "absolute_action_references": sorted(self.absolute_action_references),
            "action_sidecar_variant": self.action_sidecar_variant,
            "video_frame_stride": self.video_frame_stride,
            "video_target_shift_steps": self.video_target_shift_steps,
            "lazy_cache_shards": self.lazy_cache_shards,
            "index_windows_lazily": self.index_windows_lazily,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _metadata_index_cache_file(self) -> Path | None:
        if not self.metadata_index_cache:
            return None
        if self.metadata_index_cache_path is not None:
            return self.metadata_index_cache_path
        return self.metadata_index_cache_dir / f"{self._metadata_index_cache_key}.pkl"

    def _read_metadata_index_cache(self, cache_path: Path) -> list[ShardSpec] | None:
        if not cache_path.exists():
            return None
        rank, _ = self._distributed_rank_world()
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("version") != CANONICAL_INDEX_CACHE_VERSION:
                return None
            if payload.get("cache_key") != self._metadata_index_cache_key:
                return None
            shards = payload.get("shards")
            if not isinstance(shards, list) or not shards:
                return None
            checked_paths: set[Path] = set()
            for shard in shards:
                metadata_path = getattr(shard, "episode_metadata_path", None)
                expected_sha256 = getattr(shard, "episode_metadata_sha256", None)
                expected_size = getattr(shard, "episode_metadata_size", None)
                expected_mtime_ns = getattr(shard, "episode_metadata_mtime_ns", None)
                expected_ctime_ns = getattr(shard, "episode_metadata_ctime_ns", None)
                if (
                    metadata_path is None
                    or not expected_sha256
                    or expected_size is None
                    or expected_mtime_ns is None
                    or expected_ctime_ns is None
                ):
                    return None
                metadata_path = Path(metadata_path)
                if metadata_path in checked_paths:
                    continue
                checked_paths.add(metadata_path)
                try:
                    metadata_stat = metadata_path.stat()
                except OSError:
                    return None
                if (
                    metadata_stat.st_size != expected_size
                    or metadata_stat.st_mtime_ns != expected_mtime_ns
                    or metadata_stat.st_ctime_ns != expected_ctime_ns
                    or _hash_file(metadata_path) != expected_sha256
                ):
                    if rank == 0:
                        print(
                            "Canonical shard index cache invalidated by changed episode "
                            f"metadata: {metadata_path}.",
                            file=sys.stderr,
                            flush=True,
                        )
                    return None
                subtask_segments_path = getattr(
                    shard, "subtask_segments_path", None
                )
                expected_segments_sha256 = getattr(
                    shard, "subtask_segments_sha256", None
                )
                expected_segments_size = getattr(
                    shard, "subtask_segments_size", None
                )
                expected_segments_mtime_ns = getattr(
                    shard, "subtask_segments_mtime_ns", None
                )
                expected_segments_ctime_ns = getattr(
                    shard, "subtask_segments_ctime_ns", None
                )
                requires_segments = bool(
                    getattr(self, "append_subtask_to_prompt", False)
                    and _is_realsource_dataset(
                        getattr(shard, "dataset_id", "")
                    )
                )
                has_segments_binding = any(
                    value is not None
                    for value in (
                        subtask_segments_path,
                        expected_segments_sha256,
                        expected_segments_size,
                        expected_segments_mtime_ns,
                        expected_segments_ctime_ns,
                    )
                )
                if requires_segments or has_segments_binding:
                    if (
                        subtask_segments_path is None
                        or not expected_segments_sha256
                        or expected_segments_size is None
                        or expected_segments_mtime_ns is None
                        or expected_segments_ctime_ns is None
                    ):
                        return None
                    subtask_segments_path = Path(
                        subtask_segments_path
                    )
                    try:
                        segment_stat = subtask_segments_path.stat()
                    except OSError:
                        return None
                    if (
                        segment_stat.st_size
                        != expected_segments_size
                        or segment_stat.st_mtime_ns
                        != expected_segments_mtime_ns
                        or segment_stat.st_ctime_ns
                        != expected_segments_ctime_ns
                        or _hash_file(subtask_segments_path)
                        != expected_segments_sha256
                    ):
                        if rank == 0:
                            print(
                                "Canonical shard index cache invalidated "
                                "by changed subtask segment sidecar: "
                                f"{subtask_segments_path}.",
                                file=sys.stderr,
                                flush=True,
                            )
                        return None
            if rank == 0:
                print(
                    f"Canonical shard index cache hit: {cache_path} ({len(shards)} shards).",
                    file=sys.stderr,
                    flush=True,
                )
            return shards
        except Exception as exc:
            if rank == 0:
                print(
                    f"Canonical shard index cache ignored after load failure: {cache_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            return None

    def _write_metadata_index_cache(self, cache_path: Path, shards: list[ShardSpec]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        payload = {
            "version": CANONICAL_INDEX_CACHE_VERSION,
            "cache_key": self._metadata_index_cache_key,
            "shards": shards,
        }
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)

    def _resolve_shards_with_index_cache(self) -> list[ShardSpec]:
        cache_path = self._metadata_index_cache_file()
        if cache_path is None:
            return self._resolve_shards()

        cached_shards = self._read_metadata_index_cache(cache_path)
        if cached_shards is not None:
            return cached_shards

        candidates = None
        if self.prefetch_metadata_across_ranks:
            candidates = self._candidate_rows()
            self._prefetch_metadata_roots(candidates)

        lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
        with _exclusive_file_lock(lock_path):
            cached_shards = self._read_metadata_index_cache(cache_path)
            if cached_shards is not None:
                return cached_shards

            rank, _ = self._distributed_rank_world()
            print(
                f"Canonical shard index cache miss on rank {rank}; building {cache_path}.",
                file=sys.stderr,
                flush=True,
            )
            shards = self._resolve_shards(
                candidates=candidates,
                prefetch_metadata=not self.prefetch_metadata_across_ranks,
            )
            self._write_metadata_index_cache(cache_path, shards)
            print(
                f"Canonical shard index cache wrote {len(shards)} shards: {cache_path}.",
                file=sys.stderr,
                flush=True,
            )
            return shards

    def _episode_metadata_columns(self, camera_source_keys: dict[str, str]) -> list[str]:
        columns = [
            "data/chunk_index",
            "data/file_index",
            "episode_index",
            "dataset_from_index",
            "length",
            "task_index",
            "tasks",
        ]
        if self.append_subtask_to_prompt:
            columns.extend(
                [
                    "subtask_names",
                    "subtask_start_frames",
                    "subtask_end_frames",
                    "subtask_start_times",
                    "subtask_end_times",
                ]
            )
        for source_key in camera_source_keys.values():
            columns.extend(
                [
                    f"videos/{source_key}/chunk_index",
                    f"videos/{source_key}/file_index",
                    f"videos/{source_key}/from_timestamp",
                ]
            )
        return list(dict.fromkeys(columns))

    def _resolve_shards(
        self,
        candidates: list[tuple[dict[str, Any], dict[str, Any], Any]] | None = None,
        *,
        prefetch_metadata: bool = True,
    ) -> list[ShardSpec]:
        shards: list[ShardSpec] = []
        shards_per_dataset: dict[str, int] = {}
        if candidates is None:
            candidates = self._candidate_rows()
        if prefetch_metadata and self.prefetch_metadata_across_ranks:
            self._prefetch_metadata_roots(candidates)
        for row, adapter_meta, adapter in candidates:
            if self.max_shards and len(shards) >= self.max_shards:
                break
            if self.max_windows and self._preview_total_window_count_from_shards(shards) >= self.max_windows:
                break
            dataset_id = str(row["dataset_id"])
            if (
                self.max_shards_per_dataset
                and shards_per_dataset.get(dataset_id, 0) >= self.max_shards_per_dataset
            ):
                continue
            if (
                self.max_windows_per_dataset
                and self._preview_window_count_from_shards(shards, dataset_id=dataset_id)
                >= self.max_windows_per_dataset
            ):
                continue
            sid = row["sid"]
            revision = row["revision"]
            root = self.cache_dir / sid / revision
            gcs_prefix = _gcs_join(self.bucket_root, sid, revision)
            meta_root = root / "meta"
            if not (meta_root / "info.json").exists():
                if (
                    _ensure_metadata_root(
                        root=root,
                        gcs_prefix=gcs_prefix,
                        allow_gcs_download=self.allow_gcs_download,
                        gcs_timeout_seconds=self.gcs_download_timeout_seconds,
                        gcs_retries=self.gcs_download_retries,
                        gcs_retry_backoff_seconds=self.gcs_download_retry_backoff_seconds,
                    )
                    is None
                ):
                    continue

            episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
            tasks_path = root / "meta/tasks.parquet"
            if not episodes_path.exists():
                continue
            episode_metadata_stat = episodes_path.stat()
            episode_metadata_sha256 = _hash_file(episodes_path)
            default_task = _derive_default_task(str(row.get("dataset_id") or sid))
            task_map = _load_task_map(tasks_path, default_task)
            qwen_camera_slots = tuple(_select_qwen_camera_slots(adapter.image_mapping, self.qwen_camera_slots))
            vjepa_camera_slots = tuple(_select_vjepa_camera_slots(adapter.image_mapping, self.vjepa_camera_slots))
            if not qwen_camera_slots or not vjepa_camera_slots:
                continue
            decode_camera_slots = tuple(_unique_preserve_order([*qwen_camera_slots, *vjepa_camera_slots]))
            camera_source_keys = {slot: adapter.image_mapping[slot] for slot in decode_camera_slots}
            episodes = _read_parquet_selected(
                episodes_path,
                self._episode_metadata_columns(camera_source_keys),
            )
            subtask_segments_path: Path | None = None
            subtask_segments_sha256: str | None = None
            subtask_segments_stat = None
            subtask_segments_summary: dict[str, int | str] = {}
            subtask_spans_by_episode: (
                dict[int, tuple[SubtaskSpan, ...]] | None
            ) = None
            if (
                self.append_subtask_to_prompt
                and _is_realsource_dataset(dataset_id)
            ):
                subtask_segments_path = _ensure_relative_path(
                    root=root,
                    gcs_prefix=gcs_prefix,
                    relative_path=SUBTASK_SEGMENTS_RELATIVE_PATH,
                    allow_gcs_download=self.allow_gcs_download,
                    gcs_timeout_seconds=self.gcs_download_timeout_seconds,
                    gcs_retries=self.gcs_download_retries,
                    gcs_retry_backoff_seconds=(
                        self.gcs_download_retry_backoff_seconds
                    ),
                )
                if subtask_segments_path is None:
                    raise RuntimeError(
                        "Canonical RealSource subtask prompting requires "
                        f"{SUBTASK_SEGMENTS_RELATIVE_PATH}; it is missing "
                        f"for dataset_id={dataset_id!r}, and downloads are "
                        "disabled."
                    )
                subtask_segments_sha256 = _hash_file(
                    subtask_segments_path
                )
                subtask_segments_stat = subtask_segments_path.stat()
                source_descriptor = None
                if self.frozen_train_view is not None:
                    source_descriptor = next(
                        (
                            source
                            for source in self.frozen_train_view.descriptor[
                                "sources"
                            ]
                            if str(source.get("dataset_id")) == dataset_id
                            and str(source.get("sid")) == str(sid)
                            and str(source.get("revision"))
                            == str(revision)
                        ),
                        None,
                    )
                    expected_segments_sha256 = (
                        None
                        if source_descriptor is None
                        else source_descriptor.get(
                            "subtask_segments_sha256"
                        )
                    )
                    if (
                        not isinstance(expected_segments_sha256, str)
                        or len(expected_segments_sha256) != 64
                    ):
                        raise ValueError(
                            "Canonical RealSource frozen-view source "
                            "descriptor does not bind "
                            "subtask_segments_sha256. Rebuild the view with "
                            "the current dataset-view generator."
                        )
                    if (
                        subtask_segments_sha256
                        != expected_segments_sha256
                    ):
                        raise ValueError(
                            "Canonical RealSource subtask segment sidecar "
                            "SHA-256 mismatch for "
                            f"dataset_id={dataset_id!r}: expected "
                            f"{expected_segments_sha256}, found "
                            f"{subtask_segments_sha256}."
                        )
                episode_lengths = {
                    int(record["episode_index"]): int(record["length"])
                    for record in episodes[
                        ["episode_index", "length"]
                    ].to_dict("records")
                }
                source_episode_index_map = (
                    _realsource_subtask_episode_index_map(
                        dataset_id=dataset_id,
                        episode_lengths=episode_lengths,
                        annotation_alignment=(
                            None
                            if source_descriptor is None
                            else source_descriptor.get(
                                "annotation_alignment"
                            )
                        ),
                    )
                )
                (
                    subtask_spans_by_episode,
                    subtask_segments_summary,
                ) = _load_subtask_segment_spans(
                    subtask_segments_path,
                    episode_lengths=episode_lengths,
                    source_episode_index_map=(
                        source_episode_index_map
                    ),
                )

            unique_data_files = (
                episodes[["data/chunk_index", "data/file_index"]]
                .drop_duplicates()
                .sort_values(["data/chunk_index", "data/file_index"])
            )
            for chunk_index_raw, file_index_raw in unique_data_files.itertuples(index=False, name=None):
                if self.max_shards and len(shards) >= self.max_shards:
                    break
                remaining_window_limit = self._remaining_window_limit_for_candidate(shards, dataset_id)
                if remaining_window_limit == 0:
                    break
                chunk_index = int(chunk_index_raw)
                file_index = int(file_index_raw)
                data_relative = f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
                data_path = root / data_relative
                if not self.lazy_cache_shards:
                    data_path = _ensure_relative_path(
                        root=root,
                        gcs_prefix=gcs_prefix,
                        relative_path=data_relative,
                        allow_gcs_download=self.allow_gcs_download,
                        gcs_timeout_seconds=self.gcs_download_timeout_seconds,
                        gcs_retries=self.gcs_download_retries,
                        gcs_retry_backoff_seconds=self.gcs_download_retry_backoff_seconds,
                    )
                    if data_path is None:
                        continue
                shard_episodes_df = episodes[
                    (episodes["data/chunk_index"] == chunk_index)
                    & (episodes["data/file_index"] == file_index)
                ]
                shard_episodes = self._build_episode_specs(
                    root=root,
                    gcs_prefix=gcs_prefix,
                    episodes=shard_episodes_df,
                    camera_source_keys=camera_source_keys,
                    task_map=task_map,
                    fps=float(row.get("fps") or 30.0),
                    max_windows_remaining=remaining_window_limit,
                    lazy_cache=self.lazy_cache_shards,
                    subtask_spans_by_episode=subtask_spans_by_episode,
                )
                if not shard_episodes:
                    continue
                adapter_path = (
                    self.dataset_canonicalization_root
                    / adapter_meta["path"]
                )
                adapter_sha256 = _hash_file(adapter_path)
                if len(adapter_sha256) != 64:
                    raise RuntimeError(
                        f"Canonical adapter is missing: {adapter_path}"
                    )
                sidecar_path = (
                    root
                    / "canonical_sidecars"
                    / self.action_type
                    / self.action_sidecar_variant
                    / adapter_sha256[:16]
                    / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.npz"
                )
                shard = ShardSpec(
                    dataset_id=row["dataset_id"],
                    sid=sid,
                    revision=revision,
                    adapter_group_id=adapter_meta["adapter_group_id"],
                    adapter_path=adapter_path,
                    root=root,
                    gcs_prefix=gcs_prefix,
                    data_relative_path=data_relative,
                    data_path=data_path,
                    sidecar_path=sidecar_path,
                    fps=float(row.get("fps") or 30.0),
                    camera_source_keys=camera_source_keys,
                    qwen_camera_slots=qwen_camera_slots,
                    vjepa_camera_slots=vjepa_camera_slots,
                    decode_camera_slots=decode_camera_slots,
                    task_map=task_map,
                    episodes=shard_episodes,
                    episode_metadata_path=episodes_path,
                    episode_metadata_sha256=episode_metadata_sha256,
                    episode_metadata_size=episode_metadata_stat.st_size,
                    episode_metadata_mtime_ns=episode_metadata_stat.st_mtime_ns,
                    episode_metadata_ctime_ns=episode_metadata_stat.st_ctime_ns,
                    subtask_segments_path=subtask_segments_path,
                    subtask_segments_sha256=subtask_segments_sha256,
                    subtask_segments_size=(
                        None
                        if subtask_segments_stat is None
                        else subtask_segments_stat.st_size
                    ),
                    subtask_segments_mtime_ns=(
                        None
                        if subtask_segments_stat is None
                        else subtask_segments_stat.st_mtime_ns
                    ),
                    subtask_segments_ctime_ns=(
                        None
                        if subtask_segments_stat is None
                        else subtask_segments_stat.st_ctime_ns
                    ),
                    subtask_segments_row_count=(
                        None
                        if not subtask_segments_summary
                        else int(
                            subtask_segments_summary["row_count"]
                        )
                    ),
                    subtask_segments_zero_length_count=(
                        None
                        if not subtask_segments_summary
                        else int(
                            subtask_segments_summary[
                                "zero_length_span_count"
                            ]
                        )
                    ),
                    subtask_segments_unaligned_source_row_count=(
                        None
                        if not subtask_segments_summary
                        else int(
                            subtask_segments_summary[
                                "unaligned_source_row_count"
                            ]
                        )
                    ),
                    adapter_sha256=adapter_sha256,
                )
                if not self.lazy_cache_shards:
                    self._ensure_sidecar(shard)
                shards.append(shard)
                shards_per_dataset[dataset_id] = shards_per_dataset.get(dataset_id, 0) + 1
                if self.max_windows and self._preview_total_window_count_from_shards(shards) >= self.max_windows:
                    break
        if not shards:
            raise RuntimeError(
                "No canonical shards selected. Existing cached shards are required when "
                "allow_gcs_download=false; otherwise refresh gcloud auth and enable downloads."
        )
        return shards

    def _remaining_window_limit_for_candidate(self, shards: list[ShardSpec], dataset_id: str) -> int | None:
        if self.max_windows_per_dataset:
            used = self._preview_window_count_from_shards(shards, dataset_id=dataset_id)
            remaining = self.max_windows_per_dataset - used
            return max(remaining, 1) if remaining > 0 else 0
        if self.max_windows:
            used = self._preview_total_window_count_from_shards(shards)
            remaining = self.max_windows - used
            return max(remaining, 1) if remaining > 0 else 0
        return None

    def _episode_subtask_spans(self, episode: dict[str, Any]) -> tuple[SubtaskSpan, ...]:
        if not self.append_subtask_to_prompt:
            return ()
        names = _as_metadata_sequence(episode.get("subtask_names"))
        if not names:
            return ()
        start_frames = _as_metadata_sequence(episode.get("subtask_start_frames"))
        end_frames = _as_metadata_sequence(episode.get("subtask_end_frames"))
        start_times = _as_metadata_sequence(episode.get("subtask_start_times"))
        end_times = _as_metadata_sequence(episode.get("subtask_end_times"))
        spans = []
        for index, raw_label in enumerate(names):
            label = _clean_label_text(raw_label)
            if not label:
                continue
            spans.append(
                SubtaskSpan(
                    label=label,
                    start_frame=_as_optional_int(start_frames[index]) if index < len(start_frames) else None,
                    end_frame=_as_optional_int(end_frames[index]) if index < len(end_frames) else None,
                    start_time=_as_optional_float(start_times[index]) if index < len(start_times) else None,
                    end_time=_as_optional_float(end_times[index]) if index < len(end_times) else None,
                )
            )
        return tuple(spans)

    def _validate_subtask_prompt_coverage(self) -> None:
        if not self.append_subtask_to_prompt:
            return
        for shard in self.shards:
            for episode in shard.episodes:
                for span in episode.subtask_spans:
                    if not subtask_label_is_ignored(span.label, self.data_cfg):
                        return
        raise RuntimeError(
            "Canonical append_subtask_to_prompt=true, but the selected shards contain zero "
            "usable nonignored subtask spans. Ensure the selected canonical episode metadata "
            "contains subtask_names with frame/time spans and that "
            "subtask_prompt_ignored_labels does not exclude every label."
        )

    def _subtask_label_for_window(
        self,
        episode: EpisodeSpec,
        base_index: int,
        timestamp: float | None,
    ) -> str | None:
        if not episode.subtask_spans:
            return None
        frame = int(base_index)
        fallback_label = None
        matches: list[SubtaskSpan] = []
        for span in episode.subtask_spans:
            if fallback_label is None:
                fallback_label = span.label
            if span.start_frame is not None and span.end_frame is not None:
                if span.start_frame <= frame < span.end_frame:
                    matches.append(span)
                continue
            if span.start_time is not None and span.end_time is not None and timestamp is not None:
                if span.start_time <= timestamp < span.end_time:
                    matches.append(span)
                continue
            if span.start_frame is None and span.end_frame is None and span.start_time is None and span.end_time is None:
                matches.append(span)
        if matches:
            resolved = min(
                matches,
                key=lambda span: (
                    -(
                        int(span.start_frame)
                        if span.start_frame is not None
                        else -1
                    ),
                    (
                        int(span.end_frame)
                        if span.end_frame is not None
                        else sys.maxsize
                    ),
                    (
                        int(span.segment_index)
                        if span.segment_index is not None
                        else sys.maxsize
                    ),
                    (
                        int(span.subtask_index)
                        if span.subtask_index is not None
                        else sys.maxsize
                    ),
                    span.label,
                ),
            )
            return resolved.label
        last_span = episode.subtask_spans[-1]
        if (
            last_span.boundary_semantics != "source_frame_half_open"
            and last_span.end_frame is not None
            and frame == last_span.end_frame
        ):
            return last_span.label
        if (
            last_span.boundary_semantics != "source_frame_half_open"
            and timestamp is not None
            and last_span.end_time is not None
            and timestamp == last_span.end_time
        ):
            return last_span.label
        if any(
            span.boundary_semantics == "source_frame_half_open"
            for span in episode.subtask_spans
        ):
            return None
        return None if len(episode.subtask_spans) > 1 else fallback_label

    def _subtask_prompt_deterministic_key(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind the prompt gate to logical epoch and immutable source identity.

        Loader slots are intentionally absent: the exhaustive epoch
        permutation changes them every epoch and DDP may pad them differently.
        The raw source coordinates remain stable across exact resume and are
        visible to persistent workers through ``_shared_epoch``.
        """

        shard = context["shard"]
        episode = context["episode"]
        window = context.get("window")
        source_base_index = int(
            context.get(
                "source_base_index",
                0 if window is None else window.base_index,
            )
        )
        target_base_index = int(
            source_base_index
            if window is None
            else window.base_index
        )
        return {
            "schema": "canonical-subtask-prompt-gate-v1",
            "seed": int(getattr(self, "seed", 0)),
            "epoch": int(self.current_epoch),
            "dataset_id": str(getattr(shard, "dataset_id", "")),
            "sid": str(getattr(shard, "sid", "")),
            "revision": str(getattr(shard, "revision", "")),
            "data_file": str(
                getattr(shard, "data_relative_path", "")
            ),
            "episode_index": (
                None
                if getattr(episode, "episode_index", None) is None
                else int(episode.episode_index)
            ),
            "episode_dataset_from_index": (
                None
                if getattr(episode, "dataset_from_index", None) is None
                else int(episode.dataset_from_index)
            ),
            "source_base_index": source_base_index,
            "target_base_index": target_base_index,
            "frozen_view_sample_id": context.get(
                "frozen_view_sample_id"
            ),
        }

    def _language_with_subtask(
        self,
        task: str,
        subtask_label: str | None,
        *,
        deterministic_key: Any | None = None,
    ) -> str:
        language, _ = append_subtask_label_to_language(
            task,
            subtask_label,
            self.data_cfg,
            deterministic_key=deterministic_key,
        )
        return language

    def _subtask_prompt_provenance(self) -> dict[str, Any]:
        spans = [
            span
            for shard in self.shards
            for episode in shard.episodes
            for span in episode.subtask_spans
        ]
        return {
            "enabled": self.append_subtask_to_prompt,
            "source_column": self.subtask_prompt_source_column,
            "label_column": self.subtask_prompt_label_column,
            "segment_schema": SUBTASK_SEGMENTS_SCHEMA,
            "segment_frame_coordinates": "raw_source_frames",
            "segment_boundary_semantics": "start_inclusive_end_exclusive",
            "overlap_resolution": SUBTASK_OVERLAP_RESOLUTION,
            "labels_synthesized": False,
            "gate_algorithm": "sha256_epoch_source_identity_v1",
            "append_probability": subtask_prompt_append_probability(self.data_cfg),
            "separator": str(_cfg_get(self.data_cfg, "subtask_prompt_separator", " | ")),
            "ignored_labels": list(subtask_prompt_ignored_labels(self.data_cfg)),
            "selected_span_count": len(spans),
            "selected_usable_span_count": sum(
                not subtask_label_is_ignored(span.label, self.data_cfg)
                for span in spans
            ),
        }

    def _build_episode_specs(
        self,
        *,
        root: Path,
        gcs_prefix: str,
        episodes: pd.DataFrame,
        camera_source_keys: dict[str, str],
        task_map: dict[int, str],
        fps: float,
        max_windows_remaining: int | None = None,
        lazy_cache: bool = False,
        subtask_spans_by_episode: (
            dict[int, tuple[SubtaskSpan, ...]] | None
        ) = None,
    ) -> list[EpisodeSpec]:
        specs: list[EpisodeSpec] = []
        projected_windows = 0
        dataset_start_min = int(episodes["dataset_from_index"].min()) if len(episodes) else 0
        for episode in episodes.to_dict("records"):
            video_paths: dict[str, Path] = {}
            video_base_frames: dict[str, int] = {}
            missing_video = False
            for slot, source_key in camera_source_keys.items():
                chunk_col = f"videos/{source_key}/chunk_index"
                file_col = f"videos/{source_key}/file_index"
                from_col = f"videos/{source_key}/from_timestamp"
                if chunk_col not in episode or file_col not in episode:
                    missing_video = True
                    break
                video_relative = (
                    f"videos/{source_key}/chunk-{int(episode[chunk_col]):03d}/"
                    f"file-{int(episode[file_col]):03d}.mp4"
                )
                video_path = root / video_relative
                if not lazy_cache:
                    video_path = _ensure_relative_path(
                        root=root,
                        gcs_prefix=gcs_prefix,
                        relative_path=video_relative,
                        allow_gcs_download=self.allow_gcs_download,
                        gcs_timeout_seconds=self.gcs_download_timeout_seconds,
                        gcs_retries=self.gcs_download_retries,
                        gcs_retry_backoff_seconds=self.gcs_download_retry_backoff_seconds,
                    )
                    if video_path is None:
                        missing_video = True
                        break
                video_paths[slot] = video_path
                video_base_frames[slot] = int(round(float(episode.get(from_col, 0.0)) * fps))
            if missing_video:
                continue
            task_index = int(episode.get("task_index", 0)) if "task_index" in episode else 0
            if "tasks" in episode and isinstance(episode["tasks"], (list, tuple)) and episode["tasks"]:
                task = str(episode["tasks"][0])
            else:
                task = task_map.get(task_index, next(iter(task_map.values())))
            episode_index = (
                int(episode["episode_index"])
                if "episode_index" in episode
                else len(specs)
            )
            specs.append(
                EpisodeSpec(
                    local_start=int(episode["dataset_from_index"]) - dataset_start_min,
                    length=int(episode["length"]),
                    task=task,
                    video_paths=video_paths,
                    video_base_frames=video_base_frames,
                    subtask_spans=(
                        self._episode_subtask_spans(episode)
                        if subtask_spans_by_episode is None
                        else subtask_spans_by_episode.get(
                            episode_index, ()
                        )
                    ),
                    episode_index=episode_index,
                    dataset_from_index=int(episode["dataset_from_index"]),
                )
            )
            projected_windows += max(1, (int(episode["length"]) + self.sample_stride - 1) // self.sample_stride)
            if (
                max_windows_remaining is not None
                and projected_windows >= max_windows_remaining
                and len(specs) >= self.canonical_eval_min_episodes_per_shard
            ):
                break
        return specs

    def _preview_windows_from_shards(self, shards: list[ShardSpec]) -> list[WindowSpec]:
        preview: list[WindowSpec] = []
        for shard_index, shard in enumerate(shards):
            for episode_index, episode in enumerate(shard.episodes):
                for base_index in range(0, episode.length, self.sample_stride):
                    preview.append(WindowSpec(shard_index, episode_index, base_index))
                    if self.max_windows and len(preview) >= self.max_windows:
                        return preview
        return preview

    def _preview_window_count_from_shards(self, shards: list[ShardSpec], *, dataset_id: str) -> int:
        count = 0
        for shard in shards:
            if shard.dataset_id != dataset_id:
                continue
            for episode in shard.episodes:
                count += max(1, (episode.length + self.sample_stride - 1) // self.sample_stride)
        return count

    def _preview_total_window_count_from_shards(self, shards: list[ShardSpec]) -> int:
        count = 0
        for shard in shards:
            for episode in shard.episodes:
                count += max(1, (episode.length + self.sample_stride - 1) // self.sample_stride)
        return count

    def _ensure_sidecar(self, shard: ShardSpec) -> None:
        adapter_sha256 = _hash_file(shard.adapter_path)
        if len(adapter_sha256) != 64:
            raise RuntimeError(
                f"Canonical adapter is missing: {shard.adapter_path}"
            )
        if (
            shard.adapter_sha256 is not None
            and shard.adapter_sha256 != adapter_sha256
        ):
            raise RuntimeError(
                "Canonical adapter changed after shard indexing: "
                f"{shard.adapter_path}."
            )
        shard.adapter_sha256 = adapter_sha256
        if shard.sidecar_path.exists():
            self._validate_sidecar_contract(shard)
            return
        lock_path = _relative_copy_lock_path(
            shard.root,
            f"canonical_sidecars/{shard.data_relative_path}",
        )
        with _exclusive_file_lock(lock_path):
            if shard.sidecar_path.exists():
                self._validate_sidecar_contract(shard)
                return

            shard.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            data_path = _ensure_relative_path(
                root=shard.root,
                gcs_prefix=shard.gcs_prefix,
                relative_path=shard.data_relative_path,
                allow_gcs_download=self.allow_gcs_download,
                gcs_timeout_seconds=self.gcs_download_timeout_seconds,
                gcs_retries=self.gcs_download_retries,
                gcs_retry_backoff_seconds=self.gcs_download_retry_backoff_seconds,
            )
            if data_path is None:
                raise RuntimeError(
                    f"Canonical shard data file is missing and downloads are disabled: {shard.data_relative_path}"
                )
            shard.data_path = data_path
            adapter = self._load_adapter_config(shard.adapter_path)
            frame_df = pd.read_parquet(shard.data_path)
            records = frame_df.to_dict("records")
            state_values = np.zeros((len(records), STATE_DIM), dtype=np.float32)
            state_mask = np.zeros((len(records), STATE_DIM), dtype=bool)
            action_values = np.zeros((len(records), ACTION_DIM), dtype=np.float32)
            action_mask = np.zeros((len(records), ACTION_DIM), dtype=bool)
            for idx, raw_sample in enumerate(records):
                projected = self._apply_unified_adapter(raw_sample, adapter)
                state_values[idx] = np.asarray(projected["observation"]["state"]["values"], dtype=np.float32)
                state_mask[idx] = np.asarray(projected["observation"]["state"]["mask"], dtype=bool)
                action_values[idx] = np.asarray(projected["action"]["values"], dtype=np.float32)
                action_mask[idx] = np.asarray(projected["action"]["mask"], dtype=bool)

            action_to_state_indices = self._joint_delta_mapping_for_adapter(adapter)
            action_delta_mask = action_to_state_indices >= 0
            statistics_episodes, statistics_rows = (
                self._statistics_scope_for_shard(
                    shard,
                    row_count=len(records),
                )
            )
            statistics_action_mask = action_mask & statistics_rows[:, None]
            statistics_state_mask = state_mask & statistics_rows[:, None]
            action_low, action_high = self._action_robust_bounds(
                action_values=action_values,
                action_mask=statistics_action_mask,
                state_values=state_values,
                state_mask=statistics_state_mask,
                episodes=statistics_episodes,
                action_delta_mask=action_delta_mask,
                action_to_state_indices=action_to_state_indices,
            )
            state_low, state_high = self._robust_bounds(
                state_values,
                statistics_state_mask,
            )
            store_raw_values = self.action_type == JOINT_DELTA_GRIPPER_ABSOLUTE
            quantile_normalization = self.sidecar_normalization in {
                SHARD_Q01_Q99,
                SHARD_Q01_Q99_UNCLIPPED,
            }
            clip_quantiles = self.sidecar_normalization == SHARD_Q01_Q99
            if quantile_normalization and not store_raw_values:
                action_values = self._normalize(
                    action_values,
                    action_mask,
                    action_low,
                    action_high,
                    clip=clip_quantiles,
                )
                state_values = self._normalize(
                    state_values,
                    state_mask,
                    state_low,
                    state_high,
                    clip=clip_quantiles,
                )
            elif self.sidecar_normalization != "none":
                if not (
                    store_raw_values
                    and (
                        quantile_normalization
                        or (
                            self.sidecar_normalization
                            == Q01_Q99_UNCLIPPED
                            and self.allow_eval_selection_population_candidate
                        )
                    )
                ):
                    raise ValueError(f"Unsupported sidecar_normalization: {self.sidecar_normalization}")

            storage_dtype = np.float32 if store_raw_values else self.sidecar_dtype

            tmp_path = shard.sidecar_path.with_name(
                f".{shard.sidecar_path.name}.{os.getpid()}.tmp"
            )
            with tmp_path.open("wb") as handle:
                np.savez(
                    handle,
                    state_values=state_values.astype(storage_dtype),
                    state_mask=state_mask,
                    action_values=action_values.astype(storage_dtype),
                    action_mask=action_mask,
                    timestamp=frame_df.get("timestamp", pd.Series(np.zeros(len(frame_df)))).to_numpy(dtype=np.float32),
                    frame_index=frame_df.get("frame_index", pd.Series(np.arange(len(frame_df)))).to_numpy(dtype=np.int64),
                    episode_index=frame_df.get("episode_index", pd.Series(np.zeros(len(frame_df)))).to_numpy(dtype=np.int64),
                    task_index=frame_df.get("task_index", pd.Series(np.zeros(len(frame_df)))).to_numpy(dtype=np.int64),
                    action_low=action_low,
                    action_high=action_high,
                    state_low=state_low,
                    state_high=state_high,
                    action_delta_mask=action_delta_mask,
                    action_to_state_indices=action_to_state_indices,
                    action_sidecar_variant=np.asarray(
                        self.action_sidecar_variant
                    ),
                    adapter_contract_sha256=np.asarray(
                        self.adapter_contract_sha256
                    ),
                    adapter_sha256=np.asarray(adapter_sha256),
                )
            tmp_path.replace(shard.sidecar_path)

    def _validate_sidecar_contract(self, shard: ShardSpec) -> None:
        try:
            with np.load(shard.sidecar_path) as payload:
                actual = {
                    "action_sidecar_variant": str(
                        payload["action_sidecar_variant"].item()
                    ),
                    "adapter_contract_sha256": str(
                        payload["adapter_contract_sha256"].item()
                    ),
                    "adapter_sha256": str(payload["adapter_sha256"].item()),
                }
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeError(
                "Canonical sidecar is missing its adapter/action provenance: "
                f"{shard.sidecar_path}"
            ) from exc
        expected = {
            "action_sidecar_variant": self.action_sidecar_variant,
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "adapter_sha256": shard.adapter_sha256,
        }
        if actual != expected:
            raise RuntimeError(
                "Canonical sidecar provenance does not match current adapter "
                f"semantics: path={shard.sidecar_path}, actual={actual}, "
                f"expected={expected}."
            )

    def _joint_delta_mapping_for_adapter(self, adapter: Any) -> np.ndarray:
        """Map each absolute joint action channel to its canonical state channel.

        Action and state have different total widths (49 versus 53), so applying
        an action-width boolean mask directly to a state tensor is forbidden.
        Native relative/velocity/end-effector actions retain -1 and are not
        converted.
        """

        mapping = np.full((ACTION_DIM,), -1, dtype=np.int64)
        if self.action_type != JOINT_DELTA_GRIPPER_ABSOLUTE:
            return mapping
        action_reference = str(adapter.metadata.get("action_reference", "")).lower()
        if action_reference not in self.absolute_action_references:
            return mapping
        for rule in adapter.action_mappings:
            target = str(rule.target)
            action_span = CANONICAL_JOINT_ACTION_SPANS.get(target)
            state_span = CANONICAL_JOINT_STATE_SPANS.get(target)
            if action_span is None or state_span is None:
                continue
            action_start, action_end = action_span
            state_start, state_end = state_span
            if (action_end - action_start) != (state_end - state_start):
                raise ValueError(
                    f"Canonical action/state span widths differ for {target!r}."
                )
            indices = (
                np.arange(action_end - action_start, dtype=np.int64)
                if rule.target_indices is None
                else np.asarray(rule.target_indices, dtype=np.int64)
            )
            mapping[action_start + indices] = state_start + indices
        return mapping

    def _joint_delta_mask_for_adapter(self, adapter: Any) -> np.ndarray:
        return self._joint_delta_mapping_for_adapter(adapter) >= 0

    def _action_robust_bounds(
        self,
        *,
        action_values: np.ndarray,
        action_mask: np.ndarray,
        state_values: np.ndarray,
        state_mask: np.ndarray,
        episodes: list[EpisodeSpec],
        action_delta_mask: np.ndarray,
        action_to_state_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        low, high = self._robust_bounds(action_values, action_mask)
        if not np.any(action_delta_mask):
            return low, high

        for dim in np.flatnonzero(action_delta_mask):
            state_dim = int(action_to_state_indices[dim])
            if state_dim < 0 or state_dim >= state_values.shape[1]:
                raise ValueError(
                    f"Canonical action channel {dim} has invalid state mapping {state_dim}."
                )
            pieces: list[np.ndarray] = []
            for episode in episodes:
                base_rows = episode.local_start + np.arange(
                    0, episode.length, self.sample_stride, dtype=np.int64
                )
                if base_rows.size == 0:
                    continue
                target_local = np.minimum(
                    np.arange(0, episode.length, self.sample_stride, dtype=np.int64)[:, None]
                    + self._action_offsets[None, :],
                    episode.length - 1,
                )
                target_rows = episode.local_start + target_local
                valid = (
                    action_mask[target_rows, dim]
                    & state_mask[base_rows, state_dim][:, None]
                )
                delta = (
                    action_values[target_rows, dim]
                    - state_values[base_rows, state_dim, None]
                )
                if np.any(valid):
                    pieces.append(delta[valid].astype(np.float32, copy=False))
            if not pieces:
                low[dim], high[dim] = 0.0, 1.0
                continue
            valid_values = np.concatenate(pieces)
            low[dim] = np.percentile(valid_values, 1)
            high[dim] = np.percentile(valid_values, 99)
            if abs(float(high[dim] - low[dim])) < 1e-6:
                midpoint = float(valid_values.mean())
                low[dim], high[dim] = midpoint - 1.0, midpoint + 1.0
        return low, high

    @staticmethod
    def _robust_bounds(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = np.zeros(values.shape[1], dtype=np.float32)
        high = np.ones(values.shape[1], dtype=np.float32)
        for dim in range(values.shape[1]):
            valid = values[mask[:, dim], dim]
            if valid.size == 0:
                continue
            low[dim] = np.percentile(valid, 1)
            high[dim] = np.percentile(valid, 99)
            if abs(float(high[dim] - low[dim])) < 1e-6:
                low[dim] = float(valid.mean()) - 1.0
                high[dim] = float(valid.mean()) + 1.0
        return low, high

    @staticmethod
    def _normalize(
        values: np.ndarray,
        mask: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        *,
        clip: bool = False,
    ) -> np.ndarray:
        denom = np.maximum(high - low, 1e-6)
        normalized = (2.0 * (values - low[None, :]) / denom[None, :]) - 1.0
        if clip:
            normalized = np.clip(normalized, -1.0, 1.0)
        normalized[~mask] = 0.0
        return normalized

    def _build_windows(self) -> list[WindowSpec]:
        windows: list[WindowSpec] = []
        windows_per_dataset: dict[str, int] = {}
        for shard_index, shard in enumerate(self.shards):
            for episode_index, episode in enumerate(shard.episodes):
                for base_index in range(0, episode.length, self.sample_stride):
                    if self.max_windows_per_dataset:
                        current = windows_per_dataset.get(shard.dataset_id, 0)
                        if current >= self.max_windows_per_dataset:
                            break
                    windows.append(WindowSpec(shard_index, episode_index, base_index))
                    windows_per_dataset[shard.dataset_id] = windows_per_dataset.get(shard.dataset_id, 0) + 1
                    if self.max_windows and len(windows) >= self.max_windows:
                        return windows
        return windows

    def _build_window_ranges(self) -> list[EpisodeWindowRange]:
        ranges: list[EpisodeWindowRange] = []
        windows_per_dataset: dict[str, int] = {}
        total_windows = 0
        for shard_index, shard in enumerate(self.shards):
            for episode_index, episode in enumerate(shard.episodes):
                window_count = max(1, (episode.length + self.sample_stride - 1) // self.sample_stride)
                if self.max_windows_per_dataset:
                    current = windows_per_dataset.get(shard.dataset_id, 0)
                    remaining = self.max_windows_per_dataset - current
                    if remaining <= 0:
                        break
                    window_count = min(window_count, remaining)
                if self.max_windows:
                    remaining = self.max_windows - total_windows
                    if remaining <= 0:
                        return ranges
                    window_count = min(window_count, remaining)
                if window_count <= 0:
                    continue
                total_windows += window_count
                windows_per_dataset[shard.dataset_id] = (
                    windows_per_dataset.get(shard.dataset_id, 0) + window_count
                )
                ranges.append(
                    EpisodeWindowRange(
                        shard_index=shard_index,
                        episode_index=episode_index,
                        cumulative_end=total_windows,
                    )
                )
        return ranges

    def __len__(self) -> int:
        if self.index_windows_lazily:
            return self.total_windows
        return len(self.windows)

    def _window_from_index(self, index: int) -> WindowSpec:
        if self.total_windows <= 0:
            raise IndexError("canonical dataset has no windows")
        window_index = int(index) % self.total_windows
        range_index = bisect_right(self._window_range_ends, window_index)
        window_range = self._window_ranges[range_index]
        previous_end = self._window_ranges[range_index - 1].cumulative_end if range_index > 0 else 0
        offset = window_index - previous_end
        episode = self.shards[window_range.shard_index].episodes[window_range.episode_index]
        base_index = min(int(offset) * self.sample_stride, max(episode.length - 1, 0))
        return WindowSpec(
            shard_index=window_range.shard_index,
            episode_index=window_range.episode_index,
            base_index=base_index,
        )

    def _get_shard_data(self, shard_index: int) -> _ShardData:
        future = self._shard_prefetch_futures.pop(shard_index, None)
        if future is not None:
            try:
                future.result()
            except Exception as exc:
                print(
                    "Canonical shard data prefetch failed; falling back to synchronous fetch "
                    f"shard_index={shard_index} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if shard_index not in self._loaded_shards:
            self._ensure_sidecar(self.shards[shard_index])
            self._loaded_shards[shard_index] = _ShardData(self.shards[shard_index].sidecar_path)
        self._loaded_shards.move_to_end(shard_index)
        if self.sidecar_cache_size > 0:
            while len(self._loaded_shards) > self.sidecar_cache_size:
                self._loaded_shards.popitem(last=False)
        return self._loaded_shards[shard_index]

    def _cleanup_shard_prefetch_futures(self) -> None:
        futures = getattr(self, "_shard_prefetch_futures", None)
        if not futures:
            return
        for shard_index, future in list(futures.items()):
            if not future.done():
                continue
            futures.pop(shard_index, None)
            try:
                future.result()
            except Exception as exc:
                print(
                    "Canonical shard data prefetch failed "
                    f"shard_index={shard_index} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    def _get_shard_prefetch_executor(self) -> ThreadPoolExecutor:
        executor = getattr(self, "_shard_prefetch_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="canonical-shard-prefetch",
            )
            self._shard_prefetch_executor = executor
        return executor

    def _prefetch_shard_data_file(self, shard_index: int) -> bool:
        shard = self.shards[shard_index]
        if shard.sidecar_path.exists() or shard.data_path.exists():
            return True
        local_path = _ensure_relative_path(
            root=shard.root,
            gcs_prefix=shard.gcs_prefix,
            relative_path=shard.data_relative_path,
            allow_gcs_download=self.allow_gcs_download,
            gcs_timeout_seconds=self.gcs_download_timeout_seconds,
            gcs_retries=self.gcs_download_retries,
            gcs_retry_backoff_seconds=self.gcs_download_retry_backoff_seconds,
        )
        return local_path is not None

    def _schedule_shard_data_prefetch(self, shard_index: int) -> None:
        if self.data_file_prefetch_shards <= 0 or not self.allow_gcs_download:
            return
        if not self.shards:
            return
        self._cleanup_shard_prefetch_futures()
        for offset in range(1, self.data_file_prefetch_shards + 1):
            target_index = shard_index + offset
            if target_index >= len(self.shards):
                break
            if target_index in self._shard_prefetch_seen:
                continue
            shard = self.shards[target_index]
            if shard.sidecar_path.exists() or shard.data_path.exists():
                self._shard_prefetch_seen.add(target_index)
                continue
            executor = self._get_shard_prefetch_executor()
            self._shard_prefetch_futures[target_index] = executor.submit(
                self._prefetch_shard_data_file,
                target_index,
            )
            self._shard_prefetch_seen.add(target_index)

    def _ensure_episode_video(self, shard: ShardSpec, video_path: Path) -> Path:
        try:
            relative_path = video_path.relative_to(shard.root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Canonical video path is outside shard root: {video_path}") from exc
        cache_key = (shard.root.as_posix(), relative_path)
        if cache_key in self._known_local_relative_paths:
            return video_path
        was_missing = not video_path.exists()
        local_path = _ensure_relative_path(
            root=shard.root,
            gcs_prefix=shard.gcs_prefix,
            relative_path=relative_path,
            allow_gcs_download=self.allow_gcs_download,
            gcs_timeout_seconds=self.gcs_download_timeout_seconds,
            gcs_retries=self.gcs_download_retries,
            gcs_retry_backoff_seconds=self.gcs_download_retry_backoff_seconds,
        )
        if local_path is None:
            raise RuntimeError(
                f"Canonical video file is missing and downloads are disabled: {relative_path}"
            )
        self._known_local_relative_paths.add(cache_key)
        if was_missing:
            self._maybe_prune_video_cache({local_path})
        return local_path

    def _redownload_episode_video(
        self,
        shard: ShardSpec,
        video_path: Path,
        *,
        allow_repeat: bool = False,
    ) -> Path | None:
        if not self.allow_gcs_download:
            return None
        try:
            relative_path = video_path.relative_to(shard.root).as_posix()
        except ValueError:
            return None
        cache_key = (shard.root.as_posix(), relative_path)
        if cache_key in self._redownloaded_relative_paths and not allow_repeat:
            return None
        self._redownloaded_relative_paths.add(cache_key)
        self._known_local_relative_paths.discard(cache_key)
        self._drop_pyav_reader(video_path.as_posix())
        local_path = _ensure_relative_path(
            root=shard.root,
            gcs_prefix=shard.gcs_prefix,
            relative_path=relative_path,
            allow_gcs_download=True,
            force_download=True,
            gcs_timeout_seconds=self.gcs_download_timeout_seconds,
            gcs_retries=self.gcs_download_retries,
            gcs_retry_backoff_seconds=self.gcs_download_retry_backoff_seconds,
        )
        if local_path is not None:
            self._known_local_relative_paths.add(cache_key)
            self._maybe_prune_video_cache({local_path})
        return local_path

    def _episode_video_lock_path(self, shard: ShardSpec, video_path: Path) -> Path:
        try:
            relative_path = video_path.relative_to(shard.root).as_posix()
        except ValueError:
            relative_path = video_path.as_posix()
        return _relative_copy_lock_path(shard.root, relative_path)

    def _decode_episode_video(
        self,
        shard: ShardSpec,
        video_path: Path,
        frame_indices: np.ndarray,
        lock_path: Path,
    ) -> np.ndarray:
        try:
            with _shared_file_lock(lock_path):
                return self._decode_video(video_path, frame_indices)
        except _RecoverableVideoDecodeError:
            raise
        except Exception:
            redownloaded_path = self._redownload_episode_video(
                shard,
                video_path,
                allow_repeat=not video_path.exists(),
            )
            if redownloaded_path is None:
                raise
            try:
                with _shared_file_lock(lock_path):
                    return self._decode_video(redownloaded_path, frame_indices)
            except _RecoverableVideoDecodeError:
                raise
            except Exception as retry_exc:
                raise RuntimeError(
                    "Canonical video decode failed after forced GCS redownload: "
                    f"{video_path}"
                ) from retry_exc

    def _maybe_prune_video_cache(self, protect_paths: set[Path]) -> None:
        if self.video_cache_max_bytes <= 0:
            return
        self._video_cache_prune_download_count += 1
        if self._video_cache_prune_download_count % self.video_cache_prune_interval_downloads != 0:
            return
        self._prune_video_cache(protect_paths)

    def _prune_video_cache(self, protect_paths: set[Path]) -> None:
        prune_lock_path = self.cache_dir / ".locks/video-cache-prune.lock"
        with _try_exclusive_file_lock(prune_lock_path) as acquired:
            if not acquired:
                return
            protected = {path.resolve() for path in protect_paths}
            video_files: list[tuple[float, int, Path]] = []
            total_bytes = 0
            for path in self.cache_dir.rglob("*.mp4"):
                if not path.is_file() or path.name.startswith("."):
                    continue
                try:
                    resolved = path.resolve()
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                total_bytes += int(stat.st_size)
                if resolved in protected:
                    continue
                video_files.append((float(stat.st_mtime), int(stat.st_size), path))

            if total_bytes <= self.video_cache_max_bytes:
                return

            target_bytes = int(self.video_cache_max_bytes * self.video_cache_prune_target_fraction)
            deleted_count = 0
            deleted_bytes = 0
            for _, size, path in sorted(video_files):
                if total_bytes <= target_bytes:
                    break
                lock_path = _cache_file_copy_lock_path(self.cache_dir, path)
                if lock_path is None:
                    continue
                with _try_exclusive_file_lock(lock_path) as acquired_file:
                    if not acquired_file:
                        continue
                    try:
                        current_size = path.stat().st_size
                        path.unlink()
                    except FileNotFoundError:
                        continue
                    total_bytes -= int(current_size)
                    deleted_count += 1
                    deleted_bytes += int(current_size)
            if deleted_count:
                print(
                    "Canonical video cache pruned: "
                    f"deleted_files={deleted_count} deleted_gib={deleted_bytes / 1024**3:.2f} "
                    f"remaining_gib={total_bytes / 1024**3:.2f} cap_gib={self.video_cache_max_bytes / 1024**3:.2f}",
                    file=sys.stderr,
                    flush=True,
                )

    def _compact_offsets(self) -> np.ndarray:
        if self._compact_offsets_cache is not None:
            return self._compact_offsets_cache
        if self.video_target_shift_steps <= 0:
            self._compact_offsets_cache = np.arange(self.video_horizon, dtype=np.int64) * self.video_frame_stride
            return self._compact_offsets_cache
        if self.video_horizon <= self.video_target_shift_steps:
            raise ValueError(
                f"video_horizon ({self.video_horizon}) must be greater than video_target_shift_steps "
                f"({self.video_target_shift_steps})"
            )
        context_horizon = self.video_horizon - self.video_target_shift_steps
        self._compact_offsets_cache = (
            np.arange(-(context_horizon - 1), self.video_target_shift_steps + 1, dtype=np.int64)
            * self.video_frame_stride
        )
        return self._compact_offsets_cache

    def _qwen_frame_offset(self) -> int:
        context_horizon = len(self._compact_offsets()) - self.video_target_shift_steps
        return max(context_horizon - 1, 0)

    def _decode_video_decord(self, path_key: str, frame_indices: np.ndarray) -> np.ndarray:
        if decord is None:
            raise RuntimeError("decord is required for canonical Decord video decoding.")
        if self.reader_cache_size > 0:
            reader = self._decord_readers.get(path_key)
            if reader is None:
                reader = decord.VideoReader(path_key, ctx=decord.cpu(0), num_threads=1)
                self._decord_readers[path_key] = reader
            else:
                self._decord_readers.move_to_end(path_key)
            while len(self._decord_readers) > self.reader_cache_size:
                self._decord_readers.popitem(last=False)
        else:
            reader = decord.VideoReader(path_key, ctx=decord.cpu(0), num_threads=1)
        indices = np.clip(frame_indices, 0, len(reader) - 1).astype(np.int64)
        frames = reader.get_batch(indices).asnumpy()
        return self._resize_video(frames)

    def _decode_video_pyav(self, path_key: str, frame_indices: np.ndarray) -> np.ndarray:
        if av is None:
            raise RuntimeError("PyAV is required for canonical PyAV video decoding.")
        if self.skip_corrupt_videos and path_key in getattr(self, "_bad_video_paths", set()):
            raise _RecoverableVideoDecodeError(
                f"Canonical video was previously marked corrupt in this worker: {path_key}",
                path_key=path_key,
            )

        base_reader = self._get_pyav_reader(path_key)
        base_reader_is_cached = self.pyav_reader_cache_size > 0
        fps = base_reader.fps
        time_base = base_reader.time_base
        start_time = base_reader.start_time
        frame_count = base_reader.frame_count

        indices = np.asarray(frame_indices, dtype=np.int64)
        if frame_count > 0:
            indices = np.clip(indices, 0, frame_count - 1)
        else:
            indices = np.maximum(indices, 0)
        unique_targets = sorted({int(value) for value in indices.tolist()})
        if not unique_targets:
            if not base_reader_is_cached:
                base_reader.container.close()
            raise RuntimeError(f"No frame indices requested for {path_key}")

        min_target = unique_targets[0]
        max_target = unique_targets[-1]
        target_set = set(unique_targets)
        stream_start_seconds = float(start_time) * time_base if time_base > 0 else 0.0
        max_decode_slop = max(120, (max_target - min_target) + 30)

        found: dict[int, np.ndarray] = {}
        fill_candidates: dict[int, np.ndarray] = {}
        retry_extra_frames = max(0, self.pyav_decode_retry_extra_frames)
        positive_offsets = [0, 10, 20, 30, 35, 40, 45, 50, 60, 75, 90, 120, 180, 240, 360, 540, 720, 900]
        negative_offsets = [-30, -60, -120, -240, -480, -720, -900]
        attempt_frames: list[int] = []
        for offset in positive_offsets + negative_offsets:
            if abs(offset) > retry_extra_frames:
                continue
            seek_frame = max(0, min_target + offset)
            if seek_frame not in attempt_frames:
                attempt_frames.append(seek_frame)

        last_error: Exception | None = None
        attempted_after_error = False
        retry_recovered_seek_frame: int | None = None

        def _seek(container: Any, stream: Any, seek_frame: int) -> None:
            if time_base > 0 and fps > 0:
                seek_pts = start_time + int((seek_frame / fps) / time_base)
                try:
                    container.seek(max(seek_pts - 2, 0), stream=stream, backward=True, any_frame=False)
                except Exception:
                    container.seek(0, stream=stream, backward=True, any_frame=False)

        def _frame_index(frame: Any, last_decoded_index: int | None) -> int:
            if frame.time is not None:
                return int(round((float(frame.time) - stream_start_seconds) * fps))
            if frame.pts is not None and time_base > 0:
                return int(round((int(frame.pts) - start_time) * time_base * fps))
            return 0 if last_decoded_index is None else last_decoded_index + 1

        for attempt_idx, seek_frame in enumerate(attempt_frames):
            if attempt_idx > 0 and len(found) == len(target_set):
                break
            reader = base_reader if attempt_idx == 0 else self._make_pyav_reader(path_key)
            should_close_reader = attempt_idx > 0 or not base_reader_is_cached
            attempt_error: Exception | None = None
            first_candidate: tuple[int, np.ndarray] | None = None
            last_candidate: tuple[int, np.ndarray] | None = None
            last_decoded_index: int | None = None
            capture_candidates = attempt_idx > 0
            found_before_attempt = len(found)
            try:
                _seek(reader.container, reader.stream, seek_frame)
                try:
                    for frame in reader.container.decode(reader.stream):
                        frame_index = _frame_index(frame, last_decoded_index)
                        last_decoded_index = frame_index

                        if (
                            capture_candidates
                            and min_target - retry_extra_frames <= frame_index <= max_target + retry_extra_frames
                        ):
                            candidate_array = frame.to_ndarray(format="rgb24")
                            if first_candidate is None:
                                first_candidate = (frame_index, candidate_array)
                            last_candidate = (frame_index, candidate_array)

                        if frame_index in target_set and frame_index not in found:
                            found[frame_index] = frame.to_ndarray(format="rgb24")
                            if len(found) == len(target_set):
                                break
                        if frame_index > max_target + max_decode_slop:
                            break
                except Exception as exc:
                    attempt_error = exc
                    last_error = exc

                if attempt_error is not None or len(found) != len(target_set):
                    for candidate in (first_candidate, last_candidate):
                        if candidate is None:
                            continue
                        candidate_index, candidate_array = candidate
                        if candidate_index not in fill_candidates:
                            fill_candidates[candidate_index] = candidate_array
            except Exception as exc:
                attempt_error = exc
                last_error = exc
            finally:
                if should_close_reader:
                    try:
                        reader.container.close()
                    except Exception:
                        pass

            if attempt_error is not None:
                attempted_after_error = True
                if attempt_idx == 0 and base_reader_is_cached:
                    self._drop_pyav_reader(path_key)
                continue
            if attempt_idx > 0 and len(found) > found_before_attempt:
                retry_recovered_seek_frame = seek_frame
            if found:
                break
            if fill_candidates and attempt_idx > 0:
                retry_recovered_seek_frame = seek_frame
                break

        if len(found) != len(unique_targets):
            missing = [target for target in unique_targets if target not in found]
            available_frames = {**fill_candidates, **found}
            if available_frames:
                available = sorted(available_frames)
                should_fail_recovery = (
                    self.skip_corrupt_videos
                    and (
                        len(missing) > self.pyav_max_missing_frames_for_fill
                        or (attempted_after_error and self.pyav_fail_on_decode_error_recovery)
                    )
                )
                if should_fail_recovery:
                    raise _RecoverableVideoDecodeError(
                        "Canonical PyAV corrupt recovery exceeded training skip threshold: "
                        f"path={path_key} requested={unique_targets[:8]} missing={missing[:8]} "
                        f"total_missing={len(missing)} available={available[:8]} "
                        f"retried_after_error={attempted_after_error} "
                        f"last_error={self._format_pyav_error(last_error)}",
                        path_key=path_key,
                    )
                distant_fills: list[tuple[int, int, int]] = []
                for target in missing:
                    nearest = min(available, key=lambda value: abs(value - target))
                    distance = abs(nearest - target)
                    if distance > self.pyav_max_nearest_fill_distance:
                        distant_fills.append((target, nearest, distance))
                        continue
                    found[target] = available_frames[nearest]
                if distant_fills:
                    message = (
                        "PyAV nearest-frame fill exceeded safety distance: "
                        f"path={path_key} requested={unique_targets[:8]} missing={missing[:8]} "
                        f"max_distance={self.pyav_max_nearest_fill_distance} "
                        f"distant_fills={distant_fills[:8]} available={available[:8]} "
                        f"last_error={self._format_pyav_error(last_error)}"
                    )
                    if self.skip_corrupt_videos:
                        raise _RecoverableVideoDecodeError(message, path_key=path_key)
                    raise RuntimeError(message)
                self._warn_pyav_recovery(path_key, unique_targets, missing, available, last_error, attempted_after_error)
            else:
                message = f"PyAV decoded no usable frames from {path_key}; requested={unique_targets[:8]}"
                if self.skip_corrupt_videos:
                    raise _RecoverableVideoDecodeError(message, path_key=path_key)
                raise RuntimeError(message)
        elif attempted_after_error and retry_recovered_seek_frame is not None:
            if self.skip_corrupt_videos and self.pyav_fail_on_decode_error_recovery:
                raise _RecoverableVideoDecodeError(
                    "Canonical PyAV recovered only after decode-error retry; skipping for production training: "
                    f"path={path_key} requested={unique_targets[:8]} retry_seek_frame={retry_recovered_seek_frame} "
                    f"last_error={self._format_pyav_error(last_error)}",
                    path_key=path_key,
                )
            self._warn_pyav_retry_recovery(path_key, unique_targets, retry_recovered_seek_frame, last_error)

        frames = np.stack([found[int(index)] for index in indices], axis=0)
        return self._resize_video(frames)

    def _format_pyav_error(self, error: Exception | None) -> str:
        if error is None:
            return "none"
        error_summary = f"{type(error).__name__}: {error}"
        if len(error_summary) > 220:
            error_summary = error_summary[:217] + "..."
        return error_summary

    def _claim_pyav_warning_slot(self) -> tuple[bool, bool]:
        warning_count = int(getattr(self, "_pyav_corrupt_warning_count", 0))
        self._pyav_corrupt_warning_count = warning_count + 1
        if warning_count >= self.pyav_corrupt_warning_limit:
            return False, False
        return True, warning_count + 1 == self.pyav_corrupt_warning_limit

    def _warn_pyav_retry_recovery(
        self,
        path_key: str,
        requested: list[int],
        retry_seek_frame: int,
        error: Exception | None,
    ) -> None:
        should_warn, suppress_after = self._claim_pyav_warning_slot()
        if not should_warn:
            return
        suffix = " Further PyAV recovery warnings are suppressed in this worker." if suppress_after else ""
        print(
            "Canonical PyAV recovered decode error by lookahead retry: "
            f"path={path_key} requested={requested[:8]} retry_seek_frame={retry_seek_frame} "
            f"last_error={self._format_pyav_error(error)}.{suffix}",
            file=sys.stderr,
            flush=True,
        )

    def _warn_pyav_recovery(
        self,
        path_key: str,
        requested: list[int],
        missing: list[int],
        available: list[int],
        error: Exception | None,
        retried: bool,
    ) -> None:
        should_warn, suppress_after = self._claim_pyav_warning_slot()
        if not should_warn:
            return
        suffix = ""
        if suppress_after:
            suffix = " Further PyAV recovery warnings are suppressed in this worker."
        print(
            "Canonical PyAV recovered corrupt video frames by nearest-frame fill: "
            f"path={path_key} requested={requested[:8]} missing={missing[:8]} total_missing={len(missing)} "
            f"available={available[:8]} retried_after_error={retried} "
            f"last_error={self._format_pyav_error(error)}.{suffix}",
            file=sys.stderr,
            flush=True,
        )

    def _decode_video_imageio(self, path_key: str, frame_indices: np.ndarray) -> np.ndarray:
        if imageio_v3 is None:
            raise RuntimeError("imageio is required for canonical imageio video fallback decoding.")
        frames = []
        for frame_index in frame_indices:
            frame = imageio_v3.imread(path_key, index=int(max(frame_index, 0)))
            frames.append(frame)
        return self._resize_video(np.stack(frames, axis=0))

    def _decode_video(self, video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
        path_key = video_path.as_posix()
        backend = self.video_decode_backend
        if backend == "pyav":
            return self._decode_video_pyav(path_key, frame_indices)
        if backend == "imageio":
            return self._decode_video_imageio(path_key, frame_indices)
        if backend in {"auto", "decord"} and decord is not None:
            try:
                return self._decode_video_decord(path_key, frame_indices)
            except Exception:
                if backend == "decord":
                    raise
        if av is not None:
            return self._decode_video_pyav(path_key, frame_indices)
        return self._decode_video_imageio(path_key, frame_indices)

    def _decode_video_frame_map(
        self,
        shard: ShardSpec,
        video_path: Path,
        frame_indices: np.ndarray,
        lock_path: Path,
    ) -> dict[int, np.ndarray]:
        unique_indices = np.asarray(sorted({int(index) for index in frame_indices.tolist()}), dtype=np.int64)
        decoded = self._decode_episode_video(shard, video_path, unique_indices, lock_path)
        return {int(index): decoded[offset] for offset, index in enumerate(unique_indices.tolist())}

    def _mark_bad_video(self, path_key: str | None, reason: str) -> None:
        if not path_key:
            return
        self._bad_video_paths.add(path_key)
        warning_count = int(getattr(self, "_bad_video_warning_count", 0))
        self._bad_video_warning_count = warning_count + 1
        if warning_count >= self.pyav_corrupt_warning_limit:
            return
        suffix = ""
        if warning_count + 1 == self.pyav_corrupt_warning_limit:
            suffix = " Further corrupt-video skip warnings are suppressed in this worker."
        print(
            "Canonical corrupt video marked for skip: "
            f"path={path_key} reason={reason}.{suffix}",
            file=sys.stderr,
            flush=True,
        )

    def _context_bad_video_path(self, context: dict[str, Any]) -> str | None:
        bad_paths = getattr(self, "_bad_video_paths", set())
        if not bad_paths:
            return None
        for video_path, _, _ in context["video_frames"].values():
            path_key = video_path.as_posix()
            if path_key in bad_paths:
                return path_key
        return None

    def _retry_index(self, index: int, attempt: int) -> int:
        if self.total_windows <= 0:
            return int(index)
        # Use a large odd stride so repeated retries escape local corrupt spans
        # without needing shared mutable state across dataloader workers.
        stride = 100_003
        return int(index + attempt * stride) % self.total_windows

    def _decode_failure_to_recoverable(self, exc: Exception) -> _RecoverableSampleError | None:
        if isinstance(exc, _RecoverableSampleError):
            return exc
        return None

    def _resize_video(self, video: np.ndarray) -> np.ndarray:
        if video.shape[1] == self.video_resolution_size and video.shape[2] == self.video_resolution_size:
            return video
        resized = np.empty(
            (video.shape[0], self.video_resolution_size, self.video_resolution_size, video.shape[3]),
            dtype=video.dtype,
        )
        for idx, frame in enumerate(video):
            resized[idx] = cv2.resize(
                frame,
                (self.video_resolution_size, self.video_resolution_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return resized

    def _sample_context(self, index: int) -> dict[str, Any]:
        if getattr(self, "frozen_train_view", None) is not None:
            return self._sample_context_for_frozen_row(
                self._frozen_view_row(index)
            )
        window = self._window_from_index(index) if self.index_windows_lazily else self.windows[index]
        return self._sample_context_for_window(window)

    def _sample_context_for_frozen_row(
        self,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve one immutable target-FPS ledger row without materializing it.

        Ledger ``base_index`` values are expressed at the view's target FPS.
        Canonical shards may have a different native FPS, so action and video
        positions are mapped independently before the normal sample path reads
        the sidecar.  The complete H=50 target chunk remains anchored to the
        single mapped chunk-start state.
        """

        ordinal = int(row["ordinal"])
        identity, target_base_index, source_base_index = (
            self._canonical_frozen_row_identity(row, ordinal=ordinal)
        )
        shard_index, local_episode_index = (
            self._frozen_view_episode_lookup[identity]
        )
        shard = self.shards[shard_index]
        episode = shard.episodes[local_episode_index]
        target_fps = int(
            self.frozen_train_view.descriptor["representation"]["target_fps"]
        )
        target_action_indices = (
            target_base_index + self._action_offsets
        )
        action_episode_indices = self._target_to_source_episode_indices(
            target_action_indices,
            source_fps=float(shard.fps),
            target_fps=target_fps,
        )
        compact_target_indices = (
            target_base_index + self._compact_offsets()
        )
        compact_episode_indices = self._target_to_source_episode_indices(
            compact_target_indices,
            source_fps=float(shard.fps),
            target_fps=target_fps,
        )
        context = self._sample_context_for_window(
            WindowSpec(
                shard_index=shard_index,
                episode_index=local_episode_index,
                # Keep the immutable ledger coordinate in the public window.
                base_index=target_base_index,
            ),
            source_base_index=source_base_index,
            action_episode_indices=action_episode_indices,
            compact_episode_indices=compact_episode_indices,
        )
        context["frozen_view_ordinal"] = ordinal
        context["frozen_view_sample_id"] = str(row["sample_id"])
        context["frozen_view_end_clamp_policy"] = str(
            row["end_clamp_policy"]
        )
        return context

    def _sample_context_for_window(
        self,
        window: WindowSpec,
        *,
        source_base_index: int | None = None,
        action_episode_indices: np.ndarray | None = None,
        compact_episode_indices: np.ndarray | None = None,
    ) -> dict[str, Any]:
        shard = self.shards[window.shard_index]
        try:
            self._schedule_shard_data_prefetch(window.shard_index)
            shard_data = self._get_shard_data(window.shard_index)
        except Exception as exc:
            raise _RecoverableSampleError(
                "Canonical shard data fetch failed: "
                f"sid={shard.sid} data_file={shard.data_relative_path} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
        episode = shard.episodes[window.episode_index]
        effective_base_index = (
            int(window.base_index)
            if source_base_index is None
            else int(source_base_index)
        )
        row_base = episode.local_start + effective_base_index
        available_rows = min(
            len(shard_data.state),
            len(shard_data.action),
            len(shard_data.action_mask),
            len(shard_data.timestamp),
            len(shard_data.frame_index),
            len(shard_data.episode_index),
            len(shard_data.task_index),
        )
        if available_rows <= 0:
            raise _RecoverableSampleError(
                "Canonical shard sidecar has no usable rows: "
                f"sid={shard.sid} data_file={shard.data_relative_path}"
            )
        if row_base >= available_rows:
            raise _RecoverableSampleError(
                "Canonical window starts beyond available sidecar rows: "
                f"sid={shard.sid} data_file={shard.data_relative_path} "
                f"episode_index={window.episode_index} "
                f"base_index={effective_base_index} "
                f"row_base={row_base} available_rows={available_rows}"
            )
        if action_episode_indices is None:
            action_episode_indices = (
                effective_base_index + self._action_offsets
            )
        else:
            action_episode_indices = np.asarray(
                action_episode_indices, dtype=np.int64
            )
            if action_episode_indices.shape != self._action_offsets.shape:
                raise ValueError(
                    "Canonical frozen action index mapping must preserve the "
                    f"H={len(self._action_offsets)} chunk."
                )
        action_is_pad = np.logical_or(
            action_episode_indices < 0,
            action_episode_indices >= episode.length,
        )
        action_rows = episode.local_start + np.clip(
            action_episode_indices,
            0,
            episode.length - 1,
        )
        action_is_pad = np.logical_or(action_is_pad, action_rows >= available_rows)
        action_rows = np.clip(action_rows, 0, available_rows - 1)
        compact_offsets = self._compact_offsets()
        if compact_episode_indices is None:
            compact_episode_indices = (
                effective_base_index + compact_offsets
            )
        else:
            compact_episode_indices = np.asarray(
                compact_episode_indices, dtype=np.int64
            )
            if compact_episode_indices.shape != compact_offsets.shape:
                raise ValueError(
                    "Canonical frozen video index mapping must preserve the "
                    "configured compact video horizon."
                )
        qwen_frame_offset = self._qwen_frame_offset()
        vjepa_decode_slots = set(shard.vjepa_camera_slots)
        video_frames: dict[str, tuple[Path, np.ndarray, Path]] = {}
        qwen_frame_positions: dict[str, int] = {}
        for slot in shard.decode_camera_slots:
            episode_positions = (
                compact_episode_indices
                if slot in vjepa_decode_slots
                else np.asarray(
                    [compact_episode_indices[qwen_frame_offset]],
                    dtype=np.int64,
                )
            )
            frame_indices = episode.video_base_frames[slot] + np.clip(
                episode_positions,
                0,
                episode.length - 1,
            )
            try:
                video_path = self._ensure_episode_video(shard, episode.video_paths[slot])
            except Exception as exc:
                video_path_key = episode.video_paths[slot].as_posix()
                raise _RecoverableSampleError(
                    "Canonical video fetch failed: "
                    f"path={video_path_key} error={type(exc).__name__}: {exc}",
                    path_key=video_path_key,
                ) from exc
            lock_path = self._episode_video_lock_path(shard, video_path)
            video_frames[slot] = (video_path, frame_indices.astype(np.int64, copy=False), lock_path)
            qwen_frame_positions[slot] = qwen_frame_offset if slot in vjepa_decode_slots else 0
        return {
            "window": window,
            "shard": shard,
            "shard_data": shard_data,
            "episode": episode,
            "row_base": row_base,
            "source_base_index": effective_base_index,
            "action_rows": action_rows,
            "action_is_pad": action_is_pad.astype(bool, copy=False),
            "video_frames": video_frames,
            "qwen_frame_positions": qwen_frame_positions,
        }

    def _sample_from_context(
        self,
        context: dict[str, Any],
        decoded_frames: dict[tuple[str, str], dict[int, np.ndarray]] | None = None,
    ) -> dict[str, Any]:
        shard_data = context["shard_data"]
        shard = context["shard"]
        episode = context["episode"]
        row_base = context["row_base"]
        action_rows = context["action_rows"]
        action_is_pad = np.asarray(
            context.get("action_is_pad", np.zeros(len(action_rows), dtype=bool)),
            dtype=bool,
        )
        window = context.get("window")
        base_index = int(
            context.get(
                "source_base_index",
                (
                    window.base_index
                    if window is not None
                    else row_base - episode.local_start
                ),
            )
        )
        video_cache: dict[str, np.ndarray] = {}

        def _video_for_slot(slot: str) -> np.ndarray:
            cached = video_cache.get(slot)
            if cached is not None:
                return cached
            video_path, frame_indices, lock_path = context["video_frames"][slot]
            if decoded_frames is None:
                video = self._decode_episode_video(shard, video_path, frame_indices, lock_path)
            else:
                frame_map = decoded_frames[(slot, video_path.as_posix())]
                video = np.stack([frame_map[int(index)] for index in frame_indices], axis=0)
            video_cache[slot] = video
            return video

        videos = [_video_for_slot(slot) for slot in shard.vjepa_camera_slots]
        qwen_frame_positions = context.get("qwen_frame_positions", {})
        default_qwen_frame_position = self._qwen_frame_offset()
        qwen_frames = np.stack(
            [
                _video_for_slot(slot)[int(qwen_frame_positions.get(slot, default_qwen_frame_position))]
                for slot in shard.qwen_camera_slots
            ],
            axis=0,
        )
        qwen_slot_to_index = {slot: index for index, slot in enumerate(shard.qwen_camera_slots)}
        qwen_vjepa_view_indices = np.asarray(
            [qwen_slot_to_index[slot] for slot in shard.vjepa_camera_slots],
            dtype=np.int64,
        )
        timestamp = float(shard_data.timestamp[row_base]) if hasattr(shard_data, "timestamp") else None
        subtask_label = self._subtask_label_for_window(episode, base_index, timestamp)
        state = shard_data.state[row_base : row_base + 1].astype(np.float32)
        if hasattr(shard_data, "state_mask"):
            state_mask = shard_data.state_mask[
                row_base : row_base + 1
            ].astype(bool)
        elif getattr(self, "normalization_statistics", None) is not None:
            raise ValueError(
                "Canonical shared 18-D normalization requires an explicit "
                "state_mask; missing channels must not be treated as valid."
            )
        else:
            # Legacy canonical fixtures predate state-mask plumbing. Their
            # state tensors are dense, so the equivalent mask is all-valid.
            state_mask = np.ones_like(state, dtype=bool)
        action = shard_data.action[action_rows].astype(np.float32)
        action_mask = shard_data.action_mask[action_rows].astype(bool)
        if getattr(self, "action_type", "dataset_native") == JOINT_DELTA_GRIPPER_ABSOLUTE:
            if getattr(self, "normalization_statistics", None) is not None:
                state = select_canonical_realman_policy_state(state)
                state_mask = select_canonical_realman_policy_state_mask(
                    state_mask
                )
                action = select_canonical_realman_policy_actions(action)
                action_mask = select_canonical_realman_policy_action_mask(
                    action_mask
                )
                semantic_delta_mask = (
                    select_canonical_realman_policy_action_mask(
                        np.asarray(shard_data.action_delta_mask, dtype=bool)
                    )
                )
                contract_mapping = np.asarray(
                    REALMAN_18D_ACTION_CONTRACT.action_to_state_indices,
                    dtype=np.int64,
                )
                contract_delta_mask = contract_mapping >= 0
                incompatible_native_values = (
                    action_mask
                    & contract_delta_mask[None, :]
                    & ~semantic_delta_mask[None, :]
                )
                if np.any(incompatible_native_values):
                    channels = np.flatnonzero(
                        incompatible_native_values.any(axis=0)
                    ).tolist()
                    raise ValueError(
                        "Canonical shared 18-D statistics cannot normalize "
                        "native/relative action channels as chunk-start joint "
                        f"deltas; incompatible policy channels={channels}."
                    )
                delta_dimensions = np.flatnonzero(
                    contract_delta_mask & semantic_delta_mask
                )
                mapped_state = contract_mapping[delta_dimensions]
                anchor_valid = state_mask[0, mapped_state]
                valid_actions = delta_dimensions[anchor_valid]
                invalid_actions = delta_dimensions[~anchor_valid]
                action[:, valid_actions] -= state[
                    0, contract_mapping[valid_actions]
                ]
                action_mask[:, invalid_actions] = False

                selected_statistics = self.normalization_statistics["selected"]
                state = normalize_q01_q99_unclipped(
                    state, selected_statistics["state"]
                )
                action = normalize_q01_q99_unclipped(
                    action, selected_statistics["action"]
                )
                state[~state_mask] = 0.0
                action[~action_mask] = 0.0
            else:
                delta_mask = np.asarray(shard_data.action_delta_mask, dtype=bool)
                action_to_state = np.asarray(
                    shard_data.action_to_state_indices, dtype=np.int64
                )
                if action_to_state.shape != (ACTION_DIM,):
                    raise ValueError(
                        "Canonical action-to-state mapping must have action width "
                        f"{ACTION_DIM}, got {action_to_state.shape}."
                    )
                mapped_action_indices = np.flatnonzero(delta_mask)
                mapped_state_indices = action_to_state[mapped_action_indices]
                anchor_valid = state_mask[0, mapped_state_indices]
                valid_actions = mapped_action_indices[anchor_valid]
                invalid_actions = mapped_action_indices[~anchor_valid]
                action[:, valid_actions] -= state[
                    0, action_to_state[valid_actions]
                ]
                action_mask[:, invalid_actions] = False
                if self.sidecar_normalization in {
                    SHARD_Q01_Q99,
                    SHARD_Q01_Q99_UNCLIPPED,
                }:
                    clip_quantiles = (
                        self.sidecar_normalization == SHARD_Q01_Q99
                    )
                    state = self._normalize(
                        state,
                        state_mask,
                        shard_data.state_low,
                        shard_data.state_high,
                        clip=clip_quantiles,
                    )
                    action = self._normalize(
                        action,
                        action_mask,
                        shard_data.action_low,
                        shard_data.action_high,
                        clip=clip_quantiles,
                    )
        return {
            "video_compact": np.stack(videos, axis=0),
            "qwen_frames": qwen_frames,
            "qwen_view_slots": tuple(shard.qwen_camera_slots),
            "qwen_view_count": len(shard.qwen_camera_slots),
            "vjepa_view_slots": tuple(shard.vjepa_camera_slots),
            "qwen_vjepa_view_indices": qwen_vjepa_view_indices,
            "state": state,
            "state_mask": state_mask,
            "action": action,
            "action_mask": action_mask,
            "action_is_pad": action_is_pad,
            "lang": self._language_with_subtask(
                episode.task,
                subtask_label,
                deterministic_key=(
                    self._subtask_prompt_deterministic_key(context)
                ),
            ),
            "dataset_id": shard.dataset_id,
            "episode_index": int(shard_data.episode_index[row_base]),
            "frame_index": int(shard_data.frame_index[row_base]),
            "task_index": int(shard_data.task_index[row_base]),
            "subtask_label": subtask_label or "",
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        start_time = time.monotonic()
        loader_index = int(index)
        original_index = self._epoch_window_index(loader_index)
        exact_epoch = (
            getattr(self, "mode", None) == "train"
            and getattr(self, "epoch_sampling_strategy", "with_replacement")
            == "all_sources_exhaustive"
        )
        window: WindowSpec | None = None
        shard: ShardSpec | None = None
        episode: EpisodeSpec | None = None
        touched_videos: list[str] = []
        try:
            max_attempts = (
                1
                if exact_epoch
                else (
                    self.max_sample_decode_retries + 1
                    if self.skip_corrupt_videos
                    else 1
                )
            )
            last_error: _RecoverableSampleError | None = None
            for attempt in range(max_attempts):
                sample_index = original_index if attempt == 0 else self._retry_index(original_index, attempt)
                try:
                    context = self._sample_context(sample_index)
                except Exception as exc:
                    recoverable = self._decode_failure_to_recoverable(exc)
                    if recoverable is None or not self.skip_corrupt_videos:
                        raise
                    last_error = recoverable
                    self._mark_bad_video(recoverable.path_key, type(recoverable).__name__)
                    continue
                window = context["window"]
                shard = context["shard"]
                episode = context["episode"]
                touched_videos = []
                for slot in context["video_frames"]:
                    video_path, _, _ = context["video_frames"][slot]
                    try:
                        touched_videos.append(f"{slot}:{video_path.relative_to(shard.root).as_posix()}")
                    except ValueError:
                        touched_videos.append(f"{slot}:{video_path.as_posix()}")
                bad_path = self._context_bad_video_path(context)
                if bad_path is not None:
                    last_error = _RecoverableVideoDecodeError(
                        f"Canonical sample touches known corrupt video: {bad_path}",
                        path_key=bad_path,
                    )
                    continue
                try:
                    return self._sample_from_context(context)
                except Exception as exc:
                    recoverable = self._decode_failure_to_recoverable(exc)
                    if recoverable is None or not self.skip_corrupt_videos:
                        raise
                    last_error = recoverable
                    self._mark_bad_video(recoverable.path_key, type(recoverable).__name__)
                    continue
            raise RuntimeError(
                (
                    "Canonical exhaustive sample failed; retry substitution is "
                    "forbidden"
                    if exact_epoch
                    else "Canonical sample decode failed after corrupt-video retries"
                )
                + f": loader_index={loader_index} window_index={original_index} "
                f"attempts={max_attempts}"
            ) from last_error
        finally:
            if self.slow_sample_log_seconds > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.slow_sample_log_seconds:
                    shard_context = ""
                    if shard is not None:
                        shard_context = (
                            f" dataset_id={shard.dataset_id} sid={shard.sid} "
                            f"data_file={shard.data_relative_path}"
                        )
                    window_context = ""
                    if window is not None:
                        window_context = (
                            f" shard_index={window.shard_index} episode_index={window.episode_index} "
                            f"base_index={window.base_index}"
                        )
                    episode_context = f" episode_length={episode.length}" if episode is not None else ""
                    print(
                        "Canonical slow sample: "
                        f"elapsed={elapsed:.3f}s index={int(index)} pid={os.getpid()}"
                        f"{shard_context}{window_context}{episode_context} "
                        f"videos={','.join(touched_videos)}",
                        file=sys.stderr,
                        flush=True,
                    )

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        loader_indices = [int(index) for index in indices]
        original_indices = [
            self._epoch_window_index(index) for index in loader_indices
        ]
        exact_epoch = (
            getattr(self, "mode", None) == "train"
            and getattr(self, "epoch_sampling_strategy", "with_replacement")
            == "all_sources_exhaustive"
        )
        max_attempts = (
            1
            if exact_epoch
            else (
                self.max_sample_decode_retries + 1
                if self.skip_corrupt_videos
                else 1
            )
        )
        last_error: _RecoverableSampleError | None = None
        for attempt in range(max_attempts):
            batch_indices = (
                original_indices
                if attempt == 0
                else [self._retry_index(index, attempt) for index in original_indices]
            )
            try:
                contexts = [self._sample_context(index) for index in batch_indices]
            except Exception as exc:
                recoverable = self._decode_failure_to_recoverable(exc)
                if recoverable is None or not self.skip_corrupt_videos:
                    raise
                last_error = recoverable
                self._mark_bad_video(recoverable.path_key, type(recoverable).__name__)
                continue
            bad_path = next((path for context in contexts if (path := self._context_bad_video_path(context))), None)
            if bad_path is not None:
                last_error = _RecoverableVideoDecodeError(
                    f"Canonical batch touches known corrupt video: {bad_path}",
                    path_key=bad_path,
                )
                continue
            frame_requests: dict[tuple[str, str], tuple[ShardSpec, Path, Path, list[np.ndarray]]] = {}
            for context in contexts:
                for slot in context["video_frames"]:
                    video_path, frame_indices, lock_path = context["video_frames"][slot]
                    key = (slot, video_path.as_posix())
                    if key not in frame_requests:
                        frame_requests[key] = (context["shard"], video_path, lock_path, [])
                    frame_requests[key][3].append(frame_indices)

            decoded_frames: dict[tuple[str, str], dict[int, np.ndarray]] = {}
            try:
                for key, (shard, video_path, lock_path, request_chunks) in frame_requests.items():
                    decoded_frames[key] = self._decode_video_frame_map(
                        shard,
                        video_path,
                        np.concatenate(request_chunks),
                        lock_path,
                    )
                return [self._sample_from_context(context, decoded_frames) for context in contexts]
            except Exception as exc:
                recoverable = self._decode_failure_to_recoverable(exc)
                if recoverable is None or not self.skip_corrupt_videos:
                    raise
                last_error = recoverable
                self._mark_bad_video(recoverable.path_key, type(recoverable).__name__)
                continue
        raise RuntimeError(
            (
                "Canonical exhaustive batch failed; retry substitution is forbidden"
                if exact_epoch
                else "Canonical batch decode failed after corrupt-video retries"
            )
            + f": loader_indices={loader_indices[:8]} "
            f"window_indices={original_indices[:8]} "
            f"batch_size={len(original_indices)} attempts={max_attempts}"
        ) from last_error

    def save_dataset_statistics(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        window_counts: dict[str, int] = {}
        if self.index_windows_lazily:
            previous_end = 0
            for window_range in self._window_ranges:
                dataset_id = self.shards[window_range.shard_index].dataset_id
                window_count = window_range.cumulative_end - previous_end
                window_counts[dataset_id] = window_counts.get(dataset_id, 0) + window_count
                previous_end = window_range.cumulative_end
        else:
            for window in self.windows:
                dataset_id = self.shards[window.shard_index].dataset_id
                window_counts[dataset_id] = window_counts.get(dataset_id, 0) + 1
        for shard in self.shards:
            video_files: dict[str, list[str]] = {}
            for slot in shard.decode_camera_slots:
                video_files[slot] = sorted(
                    {
                        episode.video_paths[slot].as_posix()
                        for episode in shard.episodes
                        if slot in episode.video_paths
                    }
                )
            manifest_rows.append(
                {
                    "dataset_id": shard.dataset_id,
                    "sid": shard.sid,
                    "revision": shard.revision,
                    "adapter_group_id": getattr(
                        shard, "adapter_group_id", None
                    ),
                    "adapter_sha256": getattr(shard, "adapter_sha256", None),
                    "gcs_prefix": shard.gcs_prefix,
                    "data_file": shard.data_relative_path,
                    "local_data_file": shard.data_path.as_posix(),
                    "sidecar_file": shard.sidecar_path.as_posix(),
                    "episode_metadata_file": (
                        shard.episode_metadata_path.as_posix()
                        if shard.episode_metadata_path is not None
                        else None
                    ),
                    "episode_metadata_sha256": shard.episode_metadata_sha256,
                    "episode_metadata_size": shard.episode_metadata_size,
                    "episode_metadata_mtime_ns": shard.episode_metadata_mtime_ns,
                    "episode_metadata_ctime_ns": shard.episode_metadata_ctime_ns,
                    "subtask_segments_file": (
                        shard.subtask_segments_path.as_posix()
                        if shard.subtask_segments_path is not None
                        else None
                    ),
                    "subtask_segments_sha256": (
                        shard.subtask_segments_sha256
                    ),
                    "subtask_segments_size": shard.subtask_segments_size,
                    "subtask_segments_mtime_ns": (
                        shard.subtask_segments_mtime_ns
                    ),
                    "subtask_segments_ctime_ns": (
                        shard.subtask_segments_ctime_ns
                    ),
                    "subtask_segments_row_count": (
                        shard.subtask_segments_row_count
                    ),
                    "subtask_segments_zero_length_count": (
                        shard.subtask_segments_zero_length_count
                    ),
                    "subtask_segments_unaligned_source_row_count": (
                        shard.subtask_segments_unaligned_source_row_count
                    ),
                    "video_files": video_files,
                    "qwen_camera_slots": list(shard.qwen_camera_slots),
                    "vjepa_camera_slots": list(shard.vjepa_camera_slots),
                    "decode_camera_slots": list(shard.decode_camera_slots),
                    "fps": shard.fps,
                    "episodes": len(shard.episodes),
                }
            )
        stats = {
            "canonical_subset": {
                "num_shards": len(self.shards),
                "num_windows": len(self),
                "total_indexed_windows": self.total_windows,
                "index_windows_lazily": self.index_windows_lazily,
                "state_dim": self.policy_state_dim,
                "action_dim": self.policy_action_dim,
                "source_state_dim": STATE_DIM,
                "source_action_dim": ACTION_DIM,
                "normalization": (
                    Q01_Q99_UNCLIPPED
                    if self.normalization_statistics is not None
                    else self.sidecar_normalization
                ),
                "normalization_statistics_scope": (
                    "immutable_union_train_only"
                    if self.normalization_statistics is not None
                    else (
                        "train_episodes_only"
                        if self.canonical_eval_manifest is not None
                        else "configured_stream"
                    )
                ),
                "normalization_statistics_artifact": (
                    None
                    if self.normalization_statistics_artifact_path is None
                    else self.normalization_statistics_artifact_path.as_posix()
                ),
                "normalization_statistics_artifact_sha256": (
                    self.normalization_statistics_artifact_sha256
                ),
                "epoch_sampling_strategy": self.epoch_sampling_strategy,
                "epoch_sampling_algorithm_version": (
                    self.epoch_sampling_algorithm_version
                ),
                "frozen_train_view": (
                    self._frozen_train_view_provenance()
                ),
                "action_type": self.action_type,
                "action_delta_anchor": self.action_delta_anchor,
                "gripper_action_type": self.gripper_action_type,
                "action_sidecar_variant": self.action_sidecar_variant,
                "adapter_contract_sha256": self.adapter_contract_sha256,
                "qwen_camera_slots": self.qwen_camera_slots,
                "vjepa_camera_slots": self.vjepa_camera_slots,
                "exclude_dataset_ids": self.exclude_dataset_id_list,
                "exclude_sids": self.exclude_sid_list,
                "windows_per_dataset": window_counts,
                "subtask_prompt": self._subtask_prompt_provenance(),
                "shards": [
                    {
                        "dataset_id": shard.dataset_id,
                        "sid": shard.sid,
                        "revision": shard.revision,
                        "adapter_group_id": shard.adapter_group_id,
                        "adapter_sha256": shard.adapter_sha256,
                        "data_file": shard.data_relative_path,
                        "episode_metadata_file": (
                            shard.episode_metadata_path.as_posix()
                            if shard.episode_metadata_path is not None
                            else None
                        ),
                        "episode_metadata_sha256": shard.episode_metadata_sha256,
                        "episode_metadata_size": shard.episode_metadata_size,
                        "episode_metadata_mtime_ns": shard.episode_metadata_mtime_ns,
                        "episode_metadata_ctime_ns": shard.episode_metadata_ctime_ns,
                        "subtask_segments_file": (
                            shard.subtask_segments_path.as_posix()
                            if shard.subtask_segments_path is not None
                            else None
                        ),
                        "subtask_segments_sha256": (
                            shard.subtask_segments_sha256
                        ),
                        "subtask_segments_size": (
                            shard.subtask_segments_size
                        ),
                        "subtask_segments_mtime_ns": (
                            shard.subtask_segments_mtime_ns
                        ),
                        "subtask_segments_ctime_ns": (
                            shard.subtask_segments_ctime_ns
                        ),
                        "subtask_segments_row_count": (
                            shard.subtask_segments_row_count
                        ),
                        "subtask_segments_zero_length_count": (
                            shard.subtask_segments_zero_length_count
                        ),
                        "subtask_segments_unaligned_source_row_count": (
                            shard.subtask_segments_unaligned_source_row_count
                        ),
                        "qwen_camera_slots": list(shard.qwen_camera_slots),
                        "vjepa_camera_slots": list(shard.vjepa_camera_slots),
                        "episodes": len(shard.episodes),
                    }
                    for shard in self.shards
                ],
            }
        }
        save_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        manifest_path = save_path.parent / "canonical_subset_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary_path = save_path.parent / "canonical_subset_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "num_shards": len(self.shards),
                    "num_windows": len(self),
                    "total_indexed_windows": self.total_windows,
                    "index_windows_lazily": self.index_windows_lazily,
                    "state_dim": self.policy_state_dim,
                    "action_dim": self.policy_action_dim,
                    "source_state_dim": STATE_DIM,
                    "source_action_dim": ACTION_DIM,
                    "normalization": (
                        Q01_Q99_UNCLIPPED
                        if self.normalization_statistics is not None
                        else self.sidecar_normalization
                    ),
                    "normalization_statistics_scope": (
                        "immutable_union_train_only"
                        if self.normalization_statistics is not None
                        else (
                            "train_episodes_only"
                            if self.canonical_eval_manifest is not None
                            else "configured_stream"
                        )
                    ),
                    "normalization_statistics_artifact": (
                        None
                        if self.normalization_statistics_artifact_path is None
                        else self.normalization_statistics_artifact_path.as_posix()
                    ),
                    "normalization_statistics_artifact_sha256": (
                        self.normalization_statistics_artifact_sha256
                    ),
                    "epoch_sampling_strategy": self.epoch_sampling_strategy,
                    "epoch_sampling_algorithm_version": (
                        self.epoch_sampling_algorithm_version
                    ),
                    "frozen_train_view": (
                        self._frozen_train_view_provenance()
                    ),
                    "action_type": self.action_type,
                    "action_delta_anchor": self.action_delta_anchor,
                    "gripper_action_type": self.gripper_action_type,
                    "action_sidecar_variant": self.action_sidecar_variant,
                    "adapter_contract_sha256": self.adapter_contract_sha256,
                    "qwen_camera_slots": self.qwen_camera_slots,
                    "vjepa_camera_slots": self.vjepa_camera_slots,
                    "exclude_dataset_ids": self.exclude_dataset_id_list,
                    "exclude_sids": self.exclude_sid_list,
                    "windows_per_dataset": window_counts,
                    "subtask_prompt": self._subtask_prompt_provenance(),
                    "manifest": manifest_path.name,
                    "shards": manifest_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _frozen_train_view_provenance(
        self,
    ) -> dict[str, Any] | None:
        view = getattr(self, "frozen_train_view", None)
        if view is None:
            return None
        index_path = getattr(
            self, "frozen_train_view_index_path", None
        )
        metadata_path = getattr(
            self, "frozen_train_view_index_metadata_path", None
        )
        return {
            "manifest_path": (
                self.frozen_train_view_manifest_path.as_posix()
            ),
            "manifest_sha256": (
                self.frozen_train_view_manifest_sha256
            ),
            "view_id": view.view_id,
            "ledger_encoding": view.encoding,
            "ledger_path": view.ledger_path.as_posix(),
            "ledger_sha256": str(
                view.descriptor["rows"]["sha256"]
            ),
            "row_count": int(view.row_count),
            "record_count": int(view.record_count),
            "unique_sample_count": int(view.unique_sample_count),
            "episode_count": int(view.episode_count),
            "offset_index_path": (
                None if index_path is None else index_path.as_posix()
            ),
            "offset_index_metadata_path": (
                None
                if metadata_path is None
                else metadata_path.as_posix()
            ),
            "offset_index_sha256": getattr(
                self, "frozen_train_view_index_sha256", None
            ),
            "identity_catalog_sha256": getattr(
                self,
                "frozen_train_view_identity_catalog_sha256",
                None,
            ),
        }

    def dataset_provenance(self) -> dict[str, Any]:
        metadata_sources: dict[str, str | None] = {}
        subtask_segment_sources: dict[str, str | None] = {}
        for shard in self.shards:
            if shard.episode_metadata_path is not None:
                metadata_sources[shard.episode_metadata_path.as_posix()] = (
                    shard.episode_metadata_sha256
                )
            subtask_segments_path = getattr(
                shard, "subtask_segments_path", None
            )
            if subtask_segments_path is not None:
                subtask_segment_sources[
                    Path(subtask_segments_path).as_posix()
                ] = getattr(shard, "subtask_segments_sha256", None)
        return {
            "dataset_type": "canonical_subset_vla",
            "manifest_path": self.manifest_path.resolve().as_posix(),
            "manifest_sha256": _hash_file(self.manifest_path),
            "metadata_index_cache_key": self._metadata_index_cache_key,
            "action_type": getattr(self, "action_type", "dataset_native"),
            "action_delta_anchor": getattr(
                self, "action_delta_anchor", "chunk_start_state"
            ),
            "gripper_action_type": getattr(self, "gripper_action_type", "absolute"),
            "action_sidecar_variant": getattr(self, "action_sidecar_variant", None),
            "adapter_contract_sha256": getattr(
                self, "adapter_contract_sha256", None
            ),
            "state_dim": int(getattr(self, "policy_state_dim", STATE_DIM)),
            "action_dim": int(getattr(self, "policy_action_dim", ACTION_DIM)),
            "source_state_dim": STATE_DIM,
            "source_action_dim": ACTION_DIM,
            "normalization": (
                Q01_Q99_UNCLIPPED
                if getattr(self, "normalization_statistics", None) is not None
                else getattr(
                    self,
                    "sidecar_normalization",
                    SHARD_Q01_Q99_UNCLIPPED,
                )
            ),
            "normalization_statistics_scope": (
                "immutable_union_train_only"
                if getattr(self, "normalization_statistics", None) is not None
                else (
                    "train_episodes_only"
                    if getattr(self, "canonical_eval_manifest", None) is not None
                    else "configured_stream"
                )
            ),
            "normalization_statistics_artifact": (
                None
                if getattr(
                    self, "normalization_statistics_artifact_path", None
                )
                is None
                else self.normalization_statistics_artifact_path.as_posix()
            ),
            "normalization_statistics_artifact_sha256": getattr(
                self, "normalization_statistics_artifact_sha256", None
            ),
            "epoch_sampling_strategy": getattr(
                self, "epoch_sampling_strategy", "with_replacement"
            ),
            "epoch_sampling_algorithm_version": getattr(
                self,
                "epoch_sampling_algorithm_version",
                "legacy_canonical_index_v1",
            ),
            "frozen_train_view": self._frozen_train_view_provenance(),
            "canonical_eval_manifest": (
                None
                if getattr(self, "canonical_eval_manifest", None) is None
                else {
                    "path": self.canonical_eval_manifest.path.as_posix(),
                    "sha256": self.canonical_eval_manifest.sha256,
                    "source_manifest_sha256": (
                        self.canonical_eval_manifest.source_manifest_sha256
                    ),
                    "heldout_episode_count": len(
                        self.canonical_eval_manifest.heldout_episode_identities
                    ),
                }
            ),
            "canonical_exclude_eval_episodes_from_training": (
                getattr(self, "exclude_eval_episodes_from_training", False)
            ),
            "subtask_prompt": self._subtask_prompt_provenance(),
            "episode_metadata_sources": [
                {"path": path, "sha256": sha256}
                for path, sha256 in sorted(metadata_sources.items())
            ],
            "subtask_segment_sources": [
                {
                    "path": path,
                    "sha256": sha256,
                    "schema": SUBTASK_SEGMENTS_SCHEMA,
                    "frame_coordinates": "raw_source_frames",
                    "boundary_semantics": (
                        "start_inclusive_end_exclusive"
                    ),
                    "overlap_resolution": (
                        SUBTASK_OVERLAP_RESOLUTION
                    ),
                }
                for path, sha256 in sorted(
                    subtask_segment_sources.items()
                )
            ],
            "selected_shards": [
                {
                    "dataset_id": shard.dataset_id,
                    "sid": shard.sid,
                    "revision": shard.revision,
                    "data_file": shard.data_relative_path,
                    "adapter_group_id": getattr(
                        shard, "adapter_group_id", None
                    ),
                    "adapter_sha256": getattr(shard, "adapter_sha256", None),
                }
                for shard in self.shards
            ],
        }

    def save_dataset_provenance(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "canonical_subset": self.dataset_provenance(),
        }
        if save_path.exists():
            try:
                existing = json.loads(save_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(
                    f"Existing dataset provenance is unreadable: {save_path}: {exc}"
                ) from exc
            if existing != payload:
                raise ValueError(
                    "Dataset provenance changed for an existing run directory; refusing "
                    f"to overwrite immutable resume binding: {save_path}"
                )
            print(f"Dataset provenance verified unchanged: {save_path}")
            return

        tmp_path = save_path.with_name(f".{save_path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        tmp_path.replace(save_path)
        print(f"Dataset provenance saved to: {save_path}")


class DeterministicCanonicalEvalDataset(torch.utils.data.Dataset):
    """Small, exact heldout view for canonical/streaming training.

    This deliberately avoids task-success heuristics.  It answers the three
    questions needed during training: are immutable heldout examples really
    excluded from train/statistics, are their targets finite and supervised,
    and is prediction error improving on those exact examples?
    """

    def __init__(self, source: CanonicalSubsetVLADataset) -> None:
        if source.mode != "eval":
            raise ValueError(
                "DeterministicCanonicalEvalDataset requires a canonical source "
                "constructed with mode='eval'."
            )
        manifest = source.canonical_eval_manifest
        if manifest is None:
            raise ValueError(
                "Canonical checkpoint evaluation requires canonical_eval_manifest."
            )
        self.source = source
        self.manifest = manifest
        self.windows = tuple(
            self._resolve_window(window) for window in manifest.windows
        )
        self.heldout_window_digest = _stable_json_sha256(
            [
                {
                    "dataset_id": window.dataset_id,
                    "sid": window.sid,
                    "revision": window.revision,
                    "data_file": window.data_file,
                    "episode_index": window.episode_index,
                    "base_index": window.base_index,
                }
                for window in manifest.windows
            ]
        )
        self._sampling_report = self._build_sampling_report()

    def _resolve_window(self, requested: CanonicalEvalWindow) -> WindowSpec:
        candidates: list[WindowSpec] = []
        for shard_index, shard in enumerate(self.source.shards):
            if (
                shard.dataset_id != requested.dataset_id
                or shard.sid != requested.sid
                or shard.revision != requested.revision
                or shard.data_relative_path != requested.data_file
            ):
                continue
            for episode_index, episode in enumerate(shard.episodes):
                if int(episode.episode_index) == requested.episode_index:
                    candidates.append(
                        WindowSpec(
                            shard_index=shard_index,
                            episode_index=episode_index,
                            base_index=requested.base_index,
                        )
                    )
        if len(candidates) != 1:
            raise ValueError(
                "Canonical evaluation window must resolve exactly once in the "
                "configured stream: "
                f"{requested}; matches={len(candidates)}."
            )
        resolved = candidates[0]
        episode = self.source.shards[resolved.shard_index].episodes[
            resolved.episode_index
        ]
        if resolved.base_index >= episode.length:
            raise ValueError(
                "Canonical evaluation base_index is outside its source episode: "
                f"{requested}; episode_length={episode.length}."
            )
        return resolved

    def __len__(self) -> int:
        return len(self.windows)

    def _window_masks(
        self,
        window: WindowSpec,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        shard = self.source.shards[window.shard_index]
        episode = shard.episodes[window.episode_index]
        shard_data = self.source._get_shard_data(window.shard_index)
        available_rows = min(
            len(shard_data.state),
            len(shard_data.action),
            len(shard_data.action_mask),
        )
        row_base = episode.local_start + window.base_index
        if row_base >= available_rows:
            raise ValueError(
                "Canonical heldout window starts beyond its sidecar rows: "
                f"{shard.dataset_id}/{shard.sid}/{episode.episode_index}."
            )
        local_indices = window.base_index + self.source._action_offsets
        action_is_pad = np.logical_or(
            local_indices < 0,
            local_indices >= episode.length,
        )
        action_rows = episode.local_start + np.clip(
            local_indices,
            0,
            episode.length - 1,
        )
        action_is_pad |= action_rows >= available_rows
        action_rows = np.clip(action_rows, 0, available_rows - 1)
        action_mask = shard_data.action_mask[action_rows].astype(bool, copy=True)
        if self.source.action_type == JOINT_DELTA_GRIPPER_ABSOLUTE:
            delta_dimensions = np.flatnonzero(shard_data.action_delta_mask)
            mapped_state = shard_data.action_to_state_indices[delta_dimensions]
            anchor_valid = shard_data.state_mask[row_base, mapped_state]
            action_mask[:, delta_dimensions[~anchor_valid]] = False
        if getattr(self.source, "normalization_statistics", None) is not None:
            action_mask = select_canonical_realman_policy_action_mask(
                action_mask
            )
        action_mask &= ~action_is_pad[:, None]
        return action_mask, action_is_pad.astype(bool), row_base

    def _normalization_sha256(self) -> str:
        if getattr(self.source, "normalization_statistics", None) is not None:
            sha256 = self.source.normalization_statistics_artifact_sha256
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise RuntimeError(
                    "Canonical shared normalization artifact lacks its "
                    "validated SHA-256."
                )
            return sha256
        digest = hashlib.sha256()
        digest.update(self.source.action_sidecar_variant.encode("utf-8"))
        digest.update(self.source.sidecar_normalization.encode("utf-8"))
        eval_shard_indices = sorted(
            {window.shard_index for window in self.windows}
        )
        for shard_index in eval_shard_indices:
            shard = self.source.shards[shard_index]
            shard_data = self.source._get_shard_data(shard_index)
            digest.update(
                _stable_json_sha256(
                    {
                        "dataset_id": shard.dataset_id,
                        "sid": shard.sid,
                        "revision": shard.revision,
                        "data_file": shard.data_relative_path,
                    }
                ).encode("ascii")
            )
            for value in (
                shard_data.state_low,
                shard_data.state_high,
                shard_data.action_low,
                shard_data.action_high,
                shard_data.action_delta_mask,
                shard_data.action_to_state_indices,
            ):
                array = np.ascontiguousarray(value)
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _identity_payload(
        identities: Sequence[tuple[str, str, str, str, int]],
    ) -> list[list[Any]]:
        return [list(identity) for identity in sorted(identities)]

    def _build_sampling_report(self) -> dict[str, Any]:
        valid_elements = 0
        valid_observations = 0
        action_dim = int(getattr(self.source, "policy_action_dim", ACTION_DIM))
        state_dim = int(getattr(self.source, "policy_state_dim", STATE_DIM))
        channel_counts = np.zeros((action_dim,), dtype=np.int64)
        zero_valid: list[str] = []
        for requested, window in zip(self.manifest.windows, self.windows):
            action_mask, _, _ = self._window_masks(window)
            count = int(action_mask.sum())
            valid_elements += count
            channel_counts += action_mask.sum(axis=0, dtype=np.int64)
            if count > 0:
                valid_observations += 1
            else:
                zero_valid.append(
                    f"{requested.dataset_id}/{requested.sid}/"
                    f"episode={requested.episode_index}"
                )

        holdout = self.manifest.heldout_episode_identities
        full = self.source._full_episode_identities
        train = full - holdout
        if not holdout or not train or not holdout.isdisjoint(train):
            raise ValueError(
                "Canonical heldout/train episode sets are empty or overlap."
            )
        if not holdout.issubset(full):
            raise ValueError(
                "Canonical heldout manifest contains episodes outside the full "
                "configured catalog."
            )
        normalization_sha256 = self._normalization_sha256()
        report = {
            "schema_version": 1,
            "purpose": "canonical_manifest_heldout_training_health",
            "view": "unbiased",
            "algorithm": "exact_manifest_window_v1",
            "observation_mode": (
                "one_window_per_heldout_episode"
                if (
                    self.manifest.selection.base_frames_per_episode == 1
                    and self.manifest.selection.extra_window_episode_count == 0
                )
                else "balanced_deterministic_windows_per_heldout_episode"
            ),
            "holdout_episode_count": int(
                self.manifest.selection.holdout_episode_count
                or len(holdout)
            ),
            "base_frames_per_episode": int(
                self.manifest.selection.base_frames_per_episode
            ),
            "extra_window_episode_count": int(
                self.manifest.selection.extra_window_episode_count
            ),
            "maximum_frames_per_episode": int(
                self.manifest.selection.maximum_frames_per_episode
            ),
            "window_allocation_algorithm": str(
                self.manifest.selection.window_allocation_algorithm
            ),
            "extra_window_episode_identities": [
                list(identity)
                for identity in (
                    self.manifest.selection.extra_window_episode_identities
                )
            ],
            "observation_count": len(self.windows),
            "action_evaluable_observation_count": valid_observations,
            "valid_action_element_count": valid_elements,
            "possible_action_element_count": (
                len(self.windows) * self.source.action_horizon * action_dim
            ),
            "valid_action_fraction": (
                valid_elements
                / max(
                    len(self.windows) * self.source.action_horizon * action_dim,
                    1,
                )
            ),
            "valid_action_channel_counts": channel_counts.tolist(),
            "zero_valid_action_episodes": zero_valid,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "action_horizon": self.source.action_horizon,
            "normalization": (
                Q01_Q99_UNCLIPPED
                if getattr(self.source, "normalization_statistics", None)
                is not None
                else self.source.sidecar_normalization
            ),
            "action_type": self.source.action_type,
            "normalization_statistics_scope": "train_episodes_only",
            "window_selection_sha256": self.heldout_window_digest,
            "canonical_eval_manifest_sha256": self.manifest.sha256,
            "source_manifest_sha256": self.manifest.source_manifest_sha256,
            "train_holdout_disjoint": True,
            "normalization_excludes_holdout": True,
            "control_metadata_required": False,
            "subtask_labels_required": False,
            "metric_horizons": [10, 50],
            "metric_groups": ["all_action", "arm", "hand"],
            "production_valid": True,
            "checkpoint_selection_eligible": True,
            "subtask_observation_counts": {},
            "subtask_evaluable_observation_counts": {},
            "subtask_action_timestep_counts_by_horizon": {},
            "subtask_valid_action_element_counts_by_horizon": {},
            "episode_split_provenance": [
                {
                    "dataset_name": "canonical_stream",
                    "role": "holdout",
                    "manifest_sha256": self.manifest.sha256,
                    "selected_episode_set_sha256": _stable_json_sha256(
                        self._identity_payload(holdout)
                    ),
                    "train_episode_set_sha256": _stable_json_sha256(
                        self._identity_payload(train)
                    ),
                    "holdout_episode_set_sha256": _stable_json_sha256(
                        self._identity_payload(holdout)
                    ),
                    "full_catalog_sha256": _stable_json_sha256(
                        self._identity_payload(full)
                    ),
                    "train_statistics_sha256": normalization_sha256,
                    "selected_episode_count": len(holdout),
                }
            ],
        }
        if self.manifest.selection.frames_per_episode is not None:
            report["frames_per_episode"] = int(
                self.manifest.selection.frames_per_episode
            )
        if zero_valid:
            raise ValueError(
                "Canonical heldout manifest contains windows without supervised "
                f"action elements: {zero_valid}."
            )
        return report

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[int(index)]
        context = self.source._sample_context_for_window(window)
        sample = self.source._sample_from_context(context)
        shard = context["shard"]
        episode = context["episode"]
        sample["_heldout_eval_index"] = int(index)
        sample["_heldout_eval_dataset_name"] = shard.dataset_id
        sample["_heldout_eval_episode_id"] = int(episode.episode_index)
        sample["_heldout_eval_base_index"] = int(window.base_index)
        sample["_heldout_eval_view"] = "unbiased"
        return sample

    def make_torch_generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(
            int(_cfg_get(self.source.data_cfg, "eval_window_seed", 0))
        )
        return generator

    def sampling_report(self) -> dict[str, Any]:
        return copy.deepcopy(self._sampling_report)

    def save_sampling_report(self, path: Path | str) -> None:
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                self.sampling_report(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def close_video_readers(self) -> None:
        self.source.close_video_readers()


def get_vla_dataset(
    data_cfg: Any,
    mode: str = "train",
    action_horizon: int = 50,
    video_horizon: int = 8,
    video_frame_stride: int = 1,
    allow_eval_selection_population_candidate: bool = False,
    **_: Any,
) -> CanonicalSubsetVLADataset:
    return CanonicalSubsetVLADataset(
        data_cfg,
        mode=mode,
        action_horizon=action_horizon,
        video_horizon=video_horizon,
        video_frame_stride=video_frame_stride,
        allow_eval_selection_population_candidate=(
            allow_eval_selection_population_candidate
        ),
    )
