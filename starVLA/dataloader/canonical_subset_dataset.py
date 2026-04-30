from __future__ import annotations

import gzip
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

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


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _derive_default_task(dataset_id: str) -> str:
    return dataset_id.rsplit("/", 1)[-1].replace("_", " ")


def _gcs_join(*parts: str) -> str:
    head = parts[0].rstrip("/")
    tail = "/".join(part.strip("/") for part in parts[1:])
    return f"{head}/{tail}" if tail else head


def _run_gcloud_cp(source: str, destination: Path, timeout_seconds: int = 300, recursive: bool = False) -> None:
    if recursive:
        destination.mkdir(parents=True, exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gcloud", "storage", "cp"]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([source, str(destination)])
    try:
        subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            "Failed to copy canonical dataset shard from GCS. "
            "Refresh gcloud auth with `gcloud auth login` if credentials expired. "
            f"Command: {' '.join(cmd)}\n{stderr}"
        ) from exc


def _ensure_relative_path(
    *,
    root: Path,
    gcs_prefix: str,
    relative_path: str,
    allow_gcs_download: bool,
) -> Path | None:
    local_path = root / relative_path
    if local_path.exists():
        return local_path
    if not allow_gcs_download:
        return None
    _run_gcloud_cp(_gcs_join(gcs_prefix, "files", relative_path), local_path)
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
    task_map: dict[int, str]
    episodes: list[EpisodeSpec]


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
        self.camera_slots = _as_list(_cfg_get(data_cfg, "camera_slots", ["left", "right", "main"]))
        self.dataset_id_list = [str(value) for value in _as_list(_cfg_get(data_cfg, "dataset_ids", []))]
        self.dataset_ids = set(self.dataset_id_list)
        self.dataset_order = {dataset_id: idx for idx, dataset_id in enumerate(self.dataset_id_list)}
        self.adapter_group_ids = set(_as_list(_cfg_get(data_cfg, "adapter_group_ids", [])))
        self.preferred_fps = set(float(value) for value in _as_list(_cfg_get(data_cfg, "preferred_fps", [])))
        self.max_shards = int(_cfg_get(data_cfg, "max_shards", 1))
        self.max_shards_per_dataset = int(_cfg_get(data_cfg, "max_shards_per_dataset", 0) or 0)
        self.max_windows = int(_cfg_get(data_cfg, "max_windows", 0) or 0)
        self.max_windows_per_dataset = int(_cfg_get(data_cfg, "max_windows_per_dataset", 0) or 0)
        self.sample_stride = max(1, int(_cfg_get(data_cfg, "sample_stride", 1)))
        self.video_horizon = int(video_horizon)
        self.action_horizon = int(action_horizon)
        self.video_frame_stride = max(1, int(video_frame_stride))
        self.video_target_shift_steps = max(0, int(_cfg_get(data_cfg, "video_target_shift_steps", 0)))
        self.video_resolution_size = int(_cfg_get(data_cfg, "video_resolution_size", 384))
        self.video_decode_backend = str(_cfg_get(data_cfg, "video_decode_backend", "auto")).lower()
        self.sidecar_normalization = str(_cfg_get(data_cfg, "sidecar_normalization", "shard_q01_q99")).lower()
        self.sidecar_dtype = np.float16 if str(_cfg_get(data_cfg, "sidecar_dtype", "float16")) == "float16" else np.float32
        self._apply_unified_adapter, self._load_adapter_config = _load_canonical_modules(
            self.dataset_canonicalization_root
        )
        self._adapter_manifest = self._load_adapter_manifest()
        self._decord_readers: dict[str, Any] = {}
        self._loaded_shards: dict[int, _ShardData] = {}

        self.shards = self._resolve_shards()
        self.windows = self._build_windows()
        if not self.windows:
            raise RuntimeError(
                "No canonical VLA training windows were found. Check cached shards, camera slots, "
                "adapter filters and gcloud auth if allow_gcs_download=true."
            )

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
            if self.dataset_ids and row.get("dataset_id") not in self.dataset_ids:
                continue
            if self.preferred_fps and float(row.get("fps") or 0.0) not in self.preferred_fps:
                continue
            adapter_meta = self._adapter_for_manifest_row(row)
            if adapter_meta is None:
                continue
            adapter_path = self.dataset_canonicalization_root / adapter_meta["path"]
            adapter = self._load_adapter_config(adapter_path)
            if any(slot not in adapter.image_mapping for slot in self.camera_slots):
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

    def _resolve_shards(self) -> list[ShardSpec]:
        shards: list[ShardSpec] = []
        shards_per_dataset: dict[str, int] = {}
        for row, adapter_meta, adapter in self._candidate_rows():
            if len(shards) >= self.max_shards:
                break
            dataset_id = str(row["dataset_id"])
            if (
                self.max_shards_per_dataset
                and shards_per_dataset.get(dataset_id, 0) >= self.max_shards_per_dataset
            ):
                continue
            sid = row["sid"]
            revision = row["revision"]
            root = self.cache_dir / sid / revision
            gcs_prefix = _gcs_join(self.bucket_root, sid, revision)
            meta_root = root / "meta"
            if not (meta_root / "info.json").exists():
                if not self.allow_gcs_download:
                    continue
                _run_gcloud_cp(_gcs_join(gcs_prefix, "files/meta"), root, recursive=True)

            episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
            tasks_path = root / "meta/tasks.parquet"
            if not episodes_path.exists():
                continue
            episodes = pd.read_parquet(episodes_path)
            default_task = _derive_default_task(str(row.get("dataset_id") or sid))
            task_map = _load_task_map(tasks_path, default_task)
            camera_source_keys = {slot: adapter.image_mapping[slot] for slot in self.camera_slots}

            unique_data_files = (
                episodes[["data/chunk_index", "data/file_index"]]
                .drop_duplicates()
                .sort_values(["data/chunk_index", "data/file_index"])
            )
            for _, data_file_row in unique_data_files.iterrows():
                if len(shards) >= self.max_shards:
                    break
                remaining_window_limit = self._remaining_window_limit_for_candidate(shards, dataset_id)
                if remaining_window_limit == 0:
                    continue
                chunk_index = int(data_file_row["data/chunk_index"])
                file_index = int(data_file_row["data/file_index"])
                data_relative = f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
                data_path = _ensure_relative_path(
                    root=root,
                    gcs_prefix=gcs_prefix,
                    relative_path=data_relative,
                    allow_gcs_download=self.allow_gcs_download,
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
                    task_map=task_map,
                    episodes=shard_episodes,
                )
                self._ensure_sidecar(shard)
                shards.append(shard)
                shards_per_dataset[dataset_id] = shards_per_dataset.get(dataset_id, 0) + 1
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
            used = len(self._preview_windows_from_shards(shards))
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
    ) -> list[EpisodeSpec]:
        specs: list[EpisodeSpec] = []
        projected_windows = 0
        for _, episode in episodes.iterrows():
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
                video_path = _ensure_relative_path(
                    root=root,
                    gcs_prefix=gcs_prefix,
                    relative_path=video_relative,
                    allow_gcs_download=self.allow_gcs_download,
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
                    local_start=int(episode["dataset_from_index"]) - int(episodes["dataset_from_index"].min()),
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

    def _ensure_sidecar(self, shard: ShardSpec) -> None:
        if shard.sidecar_path.exists():
            return
        shard.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
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

        np.savez(
            shard.sidecar_path,
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

    def __len__(self) -> int:
        return len(self.windows)

    def _get_shard_data(self, shard_index: int) -> _ShardData:
        if shard_index not in self._loaded_shards:
            self._loaded_shards[shard_index] = _ShardData(self.shards[shard_index].sidecar_path)
        return self._loaded_shards[shard_index]

    def _compact_offsets(self) -> np.ndarray:
        if self.video_target_shift_steps <= 0:
            return np.arange(self.video_horizon, dtype=np.int64) * self.video_frame_stride
        if self.video_horizon <= self.video_target_shift_steps:
            raise ValueError(
                f"video_horizon ({self.video_horizon}) must be greater than video_target_shift_steps "
                f"({self.video_target_shift_steps})"
            )
        context_horizon = self.video_horizon - self.video_target_shift_steps
        return (
            np.arange(-(context_horizon - 1), self.video_target_shift_steps + 1, dtype=np.int64)
            * self.video_frame_stride
        )

    def _decode_video(self, video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
        path_key = video_path.as_posix()
        backend = self.video_decode_backend
        if backend in {"auto", "decord"} and decord is not None:
            try:
                reader = self._decord_readers.get(path_key)
                if reader is None:
                    reader = decord.VideoReader(path_key, ctx=decord.cpu(0), num_threads=1)
                    self._decord_readers[path_key] = reader
                indices = np.clip(frame_indices, 0, len(reader) - 1).astype(np.int64)
                frames = reader.get_batch(indices).asnumpy()
                return self._resize_video(frames)
            except Exception:
                if backend == "decord":
                    raise
        if imageio_v3 is None:
            raise RuntimeError("imageio is required for canonical AV1 video fallback decoding.")
        frames = []
        for frame_index in frame_indices:
            frame = imageio_v3.imread(path_key, index=int(max(frame_index, 0)))
            frames.append(frame)
        return self._resize_video(np.stack(frames, axis=0))

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        shard = self.shards[window.shard_index]
        shard_data = self._get_shard_data(window.shard_index)
        episode = shard.episodes[window.episode_index]

        row_base = episode.local_start + window.base_index
        action_offsets = np.arange(self.action_horizon, dtype=np.int64)
        action_rows = episode.local_start + np.clip(
            window.base_index + action_offsets,
            0,
            episode.length - 1,
        )
        state = shard_data.state[row_base : row_base + 1].astype(np.float32)
        action = shard_data.action[action_rows].astype(np.float32)
        action_mask = shard_data.action_mask[action_rows].astype(bool)

        compact_offsets = self._compact_offsets()
        videos = []
        for slot in self.camera_slots:
            video_frame_indices = episode.video_base_frames[slot] + np.clip(
                window.base_index + compact_offsets,
                0,
                episode.length - 1,
            )
            videos.append(self._decode_video(episode.video_paths[slot], video_frame_indices))
        video_compact = np.stack(videos, axis=0)

        return {
            "video_compact": video_compact,
            "state": state,
            "action": action,
            "action_mask": action_mask,
            "lang": episode.task,
            "dataset_id": shard.dataset_id,
            "episode_index": int(shard_data.episode_index[row_base]),
            "frame_index": int(shard_data.frame_index[row_base]),
            "task_index": int(shard_data.task_index[row_base]),
        }

    def save_dataset_statistics(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        window_counts: dict[str, int] = {}
        for window in self.windows:
            dataset_id = self.shards[window.shard_index].dataset_id
            window_counts[dataset_id] = window_counts.get(dataset_id, 0) + 1
        for shard in self.shards:
            video_files: dict[str, list[str]] = {}
            for slot in self.camera_slots:
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
                    "fps": shard.fps,
                    "episodes": len(shard.episodes),
                }
            )
        stats = {
            "canonical_subset": {
                "num_shards": len(self.shards),
                "num_windows": len(self.windows),
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "normalization": self.sidecar_normalization,
                "windows_per_dataset": window_counts,
                "shards": [
                    {
                        "dataset_id": shard.dataset_id,
                        "sid": shard.sid,
                        "revision": shard.revision,
                        "adapter_group_id": shard.adapter_group_id,
                        "data_file": shard.data_relative_path,
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
                    "num_windows": len(self.windows),
                    "state_dim": STATE_DIM,
                    "action_dim": ACTION_DIM,
                    "normalization": self.sidecar_normalization,
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
