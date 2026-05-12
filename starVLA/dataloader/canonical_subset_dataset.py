from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

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
CANONICAL_INDEX_CACHE_VERSION = 2
DEFAULT_QWEN_CAMERA_SLOTS = ("main", "left", "right", "extra")
DEFAULT_VJEPA_CAMERA_SLOTS = ("left", "right", "main")


def collate_fn(batch):
    return batch


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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
class EpisodeSpec:
    local_start: int
    length: int
    task: str
    video_paths: dict[str, Path]
    video_base_frames: dict[str, int]


@dataclass(frozen=True)
class WindowSpec:
    shard_index: int
    episode_index: int
    base_index: int


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

    def __init__(
        self,
        data_cfg: Any,
        *,
        action_horizon: int,
        video_horizon: int,
        video_frame_stride: int,
    ) -> None:
        self.data_cfg = data_cfg
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
        self.max_shards = int(_cfg_get(data_cfg, "max_shards", 1))
        self.max_shards_per_dataset = int(_cfg_get(data_cfg, "max_shards_per_dataset", 0) or 0)
        self.max_windows = int(_cfg_get(data_cfg, "max_windows", 0) or 0)
        self.max_windows_per_dataset = int(_cfg_get(data_cfg, "max_windows_per_dataset", 0) or 0)
        self.sample_stride = max(1, int(_cfg_get(data_cfg, "sample_stride", 1)))
        self.video_horizon = int(video_horizon)
        self.action_horizon = int(action_horizon)
        self.video_frame_stride = max(1, int(video_frame_stride))
        self.video_target_shift_steps = max(0, int(_cfg_get(data_cfg, "video_target_shift_steps", 0)))
        self._action_offsets = np.arange(self.action_horizon, dtype=np.int64)
        self._compact_offsets_cache: np.ndarray | None = None
        self.video_resolution_size = int(_cfg_get(data_cfg, "video_resolution_size", 384))
        self.video_decode_backend = str(_cfg_get(data_cfg, "video_decode_backend", "auto")).lower()
        self.sidecar_normalization = str(_cfg_get(data_cfg, "sidecar_normalization", "shard_q01_q99")).lower()
        self.sidecar_dtype = np.float16 if str(_cfg_get(data_cfg, "sidecar_dtype", "float16")) == "float16" else np.float32
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
        self.reader_cache_size = max(int(_cfg_get(data_cfg, "reader_cache_size", 64)), 0)
        self.sidecar_cache_size = max(int(_cfg_get(data_cfg, "sidecar_cache_size", 16)), 0)
        self.slow_sample_log_seconds = float(_cfg_get(data_cfg, "slow_sample_log_seconds", 0.0) or 0.0)
        self.pyav_corrupt_warning_limit = max(int(_cfg_get(data_cfg, "pyav_corrupt_warning_limit", 20)), 0)
        self.pyav_decode_retry_extra_frames = max(int(_cfg_get(data_cfg, "pyav_decode_retry_extra_frames", 900)), 0)
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
        if self.shuffle_shards:
            random.Random(self.shuffle_seed).shuffle(self.shards)
        self._window_ranges = self._build_window_ranges()
        self._window_range_ends = [window_range.cumulative_end for window_range in self._window_ranges]
        self.total_windows = self._window_range_ends[-1] if self._window_range_ends else 0
        self.windows = [] if self.index_windows_lazily else self._build_windows()
        if (self.index_windows_lazily and self.total_windows <= 0) or (
            not self.index_windows_lazily and not self.windows
        ):
            raise RuntimeError(
                "No canonical VLA training windows were found. Check cached shards, camera slots, "
                "adapter filters and gcloud auth if allow_gcs_download=true."
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_decord_readers"] = OrderedDict()
        state["_pyav_readers"] = OrderedDict()
        state["_pyav_corrupt_warning_count"] = 0
        state["_loaded_shards"] = OrderedDict()
        state["_known_local_relative_paths"] = set()
        state["_redownloaded_relative_paths"] = set()
        state["_video_cache_prune_download_count"] = 0
        state["_shard_prefetch_executor"] = None
        state["_shard_prefetch_futures"] = OrderedDict()
        state["_shard_prefetch_seen"] = set()
        return state

    def close_video_readers(self) -> None:
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
        candidates = []
        for row in rows:
            dataset_id = str(row.get("dataset_id", ""))
            sid = str(row.get("sid", ""))
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
            "preferred_fps": sorted(self.preferred_fps),
            "allow_gcs_download": self.allow_gcs_download,
            "max_shards": self.max_shards,
            "max_shards_per_dataset": self.max_shards_per_dataset,
            "max_windows": self.max_windows,
            "max_windows_per_dataset": self.max_windows_per_dataset,
            "sample_stride": self.sample_stride,
            "video_horizon": self.video_horizon,
            "action_horizon": self.action_horizon,
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

    @staticmethod
    def _episode_metadata_columns(camera_source_keys: dict[str, str]) -> list[str]:
        columns = [
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "length",
            "task_index",
            "tasks",
        ]
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
                )
                if not shard_episodes:
                    continue
                sidecar_path = (
                    root
                    / "canonical_sidecars"
                    / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.npz"
                )
                shard = ShardSpec(
                    dataset_id=row["dataset_id"],
                    sid=sid,
                    revision=revision,
                    adapter_group_id=adapter_meta["adapter_group_id"],
                    adapter_path=self.dataset_canonicalization_root / adapter_meta["path"],
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
            specs.append(
                EpisodeSpec(
                    local_start=int(episode["dataset_from_index"]) - dataset_start_min,
                    length=int(episode["length"]),
                    task=task,
                    video_paths=video_paths,
                    video_base_frames=video_base_frames,
                )
            )
            projected_windows += max(1, (int(episode["length"]) + self.sample_stride - 1) // self.sample_stride)
            if max_windows_remaining is not None and projected_windows >= max_windows_remaining:
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
        if shard.sidecar_path.exists():
            return
        lock_path = _relative_copy_lock_path(
            shard.root,
            f"canonical_sidecars/{shard.data_relative_path}",
        )
        with _exclusive_file_lock(lock_path):
            if shard.sidecar_path.exists():
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

            action_low, action_high = self._robust_bounds(action_values, action_mask)
            state_low, state_high = self._robust_bounds(state_values, state_mask)
            if self.sidecar_normalization == "shard_q01_q99":
                action_values = self._normalize(action_values, action_mask, action_low, action_high)
                state_values = self._normalize(state_values, state_mask, state_low, state_high)
            elif self.sidecar_normalization != "none":
                raise ValueError(f"Unsupported sidecar_normalization: {self.sidecar_normalization}")

            tmp_path = shard.sidecar_path.with_name(
                f".{shard.sidecar_path.name}.{os.getpid()}.tmp"
            )
            with tmp_path.open("wb") as handle:
                np.savez(
                    handle,
                    state_values=state_values.astype(self.sidecar_dtype),
                    state_mask=state_mask,
                    action_values=action_values.astype(self.sidecar_dtype),
                    action_mask=action_mask,
                    timestamp=frame_df.get("timestamp", pd.Series(np.zeros(len(frame_df)))).to_numpy(dtype=np.float32),
                    frame_index=frame_df.get("frame_index", pd.Series(np.arange(len(frame_df)))).to_numpy(dtype=np.int64),
                    episode_index=frame_df.get("episode_index", pd.Series(np.zeros(len(frame_df)))).to_numpy(dtype=np.int64),
                    task_index=frame_df.get("task_index", pd.Series(np.zeros(len(frame_df)))).to_numpy(dtype=np.int64),
                    action_low=action_low,
                    action_high=action_high,
                    state_low=state_low,
                    state_high=state_high,
                )
            tmp_path.replace(shard.sidecar_path)

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
    def _normalize(values: np.ndarray, mask: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        denom = np.maximum(high - low, 1e-6)
        normalized = (2.0 * (values - low[None, :]) / denom[None, :]) - 1.0
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
        retry_extra_frames = max(self.pyav_decode_retry_extra_frames, max_decode_slop)
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
                for target in missing:
                    nearest = min(available, key=lambda value: abs(value - target))
                    found[target] = available_frames[nearest]
                self._warn_pyav_recovery(path_key, unique_targets, missing, available, last_error, attempted_after_error)
            else:
                raise RuntimeError(
                    f"PyAV decoded no usable frames from {path_key}; requested={unique_targets[:8]}"
                )
        elif attempted_after_error and retry_recovered_seek_frame is not None:
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
        window = self._window_from_index(index) if self.index_windows_lazily else self.windows[index]
        shard = self.shards[window.shard_index]
        self._schedule_shard_data_prefetch(window.shard_index)
        shard_data = self._get_shard_data(window.shard_index)
        episode = shard.episodes[window.episode_index]
        row_base = episode.local_start + window.base_index
        action_rows = episode.local_start + np.clip(
            window.base_index + self._action_offsets,
            0,
            episode.length - 1,
        )
        compact_offsets = self._compact_offsets()
        qwen_frame_offset = self._qwen_frame_offset()
        qwen_compact_offset = compact_offsets[qwen_frame_offset]
        vjepa_decode_slots = set(shard.vjepa_camera_slots)
        video_frames: dict[str, tuple[Path, np.ndarray, Path]] = {}
        qwen_frame_positions: dict[str, int] = {}
        for slot in shard.decode_camera_slots:
            offsets = (
                compact_offsets
                if slot in vjepa_decode_slots
                else np.asarray([qwen_compact_offset], dtype=np.int64)
            )
            frame_indices = episode.video_base_frames[slot] + np.clip(
                window.base_index + offsets,
                0,
                episode.length - 1,
            )
            video_path = self._ensure_episode_video(shard, episode.video_paths[slot])
            lock_path = self._episode_video_lock_path(shard, video_path)
            video_frames[slot] = (video_path, frame_indices.astype(np.int64, copy=False), lock_path)
            qwen_frame_positions[slot] = qwen_frame_offset if slot in vjepa_decode_slots else 0
        return {
            "window": window,
            "shard": shard,
            "shard_data": shard_data,
            "episode": episode,
            "row_base": row_base,
            "action_rows": action_rows,
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
        return {
            "video_compact": np.stack(videos, axis=0),
            "qwen_frames": qwen_frames,
            "qwen_view_slots": tuple(shard.qwen_camera_slots),
            "qwen_view_count": len(shard.qwen_camera_slots),
            "vjepa_view_slots": tuple(shard.vjepa_camera_slots),
            "qwen_vjepa_view_indices": qwen_vjepa_view_indices,
            "state": shard_data.state[row_base : row_base + 1].astype(np.float32),
            "action": shard_data.action[action_rows].astype(np.float32),
            "action_mask": shard_data.action_mask[action_rows].astype(bool),
            "lang": episode.task,
            "dataset_id": shard.dataset_id,
            "episode_index": int(shard_data.episode_index[row_base]),
            "frame_index": int(shard_data.frame_index[row_base]),
            "task_index": int(shard_data.task_index[row_base]),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        start_time = time.monotonic()
        window: WindowSpec | None = None
        shard: ShardSpec | None = None
        episode: EpisodeSpec | None = None
        touched_videos: list[str] = []
        try:
            context = self._sample_context(int(index))
            window = context["window"]
            shard = context["shard"]
            episode = context["episode"]
            for slot in context["video_frames"]:
                video_path, _, _ = context["video_frames"][slot]
                try:
                    touched_videos.append(f"{slot}:{video_path.relative_to(shard.root).as_posix()}")
                except ValueError:
                    touched_videos.append(f"{slot}:{video_path.as_posix()}")
            return self._sample_from_context(context)
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
        contexts = [self._sample_context(int(index)) for index in indices]
        frame_requests: dict[tuple[str, str], tuple[ShardSpec, Path, Path, list[np.ndarray]]] = {}
        for context in contexts:
            for slot in context["video_frames"]:
                video_path, frame_indices, lock_path = context["video_frames"][slot]
                key = (slot, video_path.as_posix())
                if key not in frame_requests:
                    frame_requests[key] = (context["shard"], video_path, lock_path, [])
                frame_requests[key][3].append(frame_indices)

        decoded_frames: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        for key, (shard, video_path, lock_path, request_chunks) in frame_requests.items():
            decoded_frames[key] = self._decode_video_frame_map(
                shard,
                video_path,
                np.concatenate(request_chunks),
                lock_path,
            )

        return [self._sample_from_context(context, decoded_frames) for context in contexts]

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
                    "adapter_group_id": shard.adapter_group_id,
                    "gcs_prefix": shard.gcs_prefix,
                    "data_file": shard.data_relative_path,
                    "local_data_file": shard.data_path.as_posix(),
                    "sidecar_file": shard.sidecar_path.as_posix(),
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
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "normalization": self.sidecar_normalization,
                "qwen_camera_slots": self.qwen_camera_slots,
                "vjepa_camera_slots": self.vjepa_camera_slots,
                "exclude_dataset_ids": self.exclude_dataset_id_list,
                "exclude_sids": self.exclude_sid_list,
                "windows_per_dataset": window_counts,
                "shards": [
                    {
                        "dataset_id": shard.dataset_id,
                        "sid": shard.sid,
                        "revision": shard.revision,
                        "adapter_group_id": shard.adapter_group_id,
                        "data_file": shard.data_relative_path,
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
                    "state_dim": STATE_DIM,
                    "action_dim": ACTION_DIM,
                    "normalization": self.sidecar_normalization,
                    "qwen_camera_slots": self.qwen_camera_slots,
                    "vjepa_camera_slots": self.vjepa_camera_slots,
                    "exclude_dataset_ids": self.exclude_dataset_id_list,
                    "exclude_sids": self.exclude_sid_list,
                    "windows_per_dataset": window_counts,
                    "manifest": manifest_path.name,
                    "shards": manifest_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def get_vla_dataset(
    data_cfg: Any,
    mode: str = "train",
    action_horizon: int = 50,
    video_horizon: int = 8,
    video_frame_stride: int = 1,
    **_: Any,
) -> CanonicalSubsetVLADataset:
    if mode != "train":
        raise ValueError("canonical_subset_vla currently supports train mode only")
    return CanonicalSubsetVLADataset(
        data_cfg,
        action_horizon=action_horizon,
        video_horizon=video_horizon,
        video_frame_stride=video_frame_stride,
    )
