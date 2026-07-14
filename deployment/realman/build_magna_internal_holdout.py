#!/usr/bin/env python3
"""Build the immutable, batch-sized Magna train/holdout split.

The holdout is a deterministic SHA-256 ranking prefix over complete episodes.
Its size is derived from the effective global batch of the supplied training
configuration; it is never typed independently into the manifest.  The script
also computes normalization statistics from the train complement only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import this leaf module directly. Importing ``starVLA.dataloader`` executes
# its training-time package initializer (and therefore requires Accelerate),
# while split generation itself only needs the dependency-light hash helpers.
_EPISODE_SPLIT_PATH = (
    REPO_ROOT / "starVLA/dataloader/gr00t_lerobot/episode_split.py"
)
_EPISODE_SPLIT_SPEC = importlib.util.spec_from_file_location(
    "_magna_episode_split_helpers", _EPISODE_SPLIT_PATH
)
if _EPISODE_SPLIT_SPEC is None or _EPISODE_SPLIT_SPEC.loader is None:
    raise ImportError(f"Could not load episode split helpers from {_EPISODE_SPLIT_PATH}")
_EPISODE_SPLIT_MODULE = importlib.util.module_from_spec(_EPISODE_SPLIT_SPEC)
sys.modules[_EPISODE_SPLIT_SPEC.name] = _EPISODE_SPLIT_MODULE
_EPISODE_SPLIT_SPEC.loader.exec_module(_EPISODE_SPLIT_MODULE)
build_episode_catalog_binding = _EPISODE_SPLIT_MODULE.build_episode_catalog_binding
canonical_json_sha256 = _EPISODE_SPLIT_MODULE.canonical_json_sha256
episode_set_sha256 = _EPISODE_SPLIT_MODULE.episode_set_sha256
file_sha256 = _EPISODE_SPLIT_MODULE.file_sha256


DEFAULT_DATASET_ROOT = Path(
    "/home/mehul/work/reward_model_small/magna_training_data_with_interventions"
)
DEFAULT_CONFIG = Path(
    "scripts/config/"
    "vlajepa_robot_ft_lerobot_magna_interventions_a100x8_"
    "qwen35_2b_full_moge_vitb_vjepa_large.yaml"
)
DEFAULT_LAUNCHER = Path(
    "scripts/vlajepa_robot_ft_lerobot_magna_clean_rtc0_pilot_a100x8.sh"
)
DEFAULT_MANIFEST_TEMPLATE = (
    "deployment/realman/eval_manifests/"
    "magna_internal_holdout_global_batch{effective_global_batch_size}_v1.json"
)
DEFAULT_SEED_TEXT = "magna-internal-holdout-global-batch-v1"
RANKING_SCHEMA = "sha256-episode-ranking-v1"
RESULT_PREFIX = "MAGNA_HOLDOUT_BUILD_RESULT="


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")


def _default_manifest_argument(effective_global_batch_size: int) -> Path:
    if effective_global_batch_size <= 0:
        raise ValueError("effective_global_batch_size must be positive")
    return Path(
        DEFAULT_MANIFEST_TEMPLATE.format(
            effective_global_batch_size=effective_global_batch_size
        )
    )


def _load_episode_table(dataset_root: Path) -> pd.DataFrame:
    paths = sorted((dataset_root / "meta/episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(dataset_root / "meta/episodes")
    episodes = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "episode_index",
                    "length",
                    "data/chunk_index",
                    "data/file_index",
                    "dataset_from_index",
                    "dataset_to_index",
                    "videos/observation.images.head/chunk_index",
                    "videos/observation.images.head/file_index",
                    "videos/observation.images.head/from_timestamp",
                    "videos/observation.images.head/to_timestamp",
                    "videos/observation.images.wrist_left/chunk_index",
                    "videos/observation.images.wrist_left/file_index",
                    "videos/observation.images.wrist_left/from_timestamp",
                    "videos/observation.images.wrist_left/to_timestamp",
                    "videos/observation.images.wrist_right/chunk_index",
                    "videos/observation.images.wrist_right/file_index",
                    "videos/observation.images.wrist_right/from_timestamp",
                    "videos/observation.images.wrist_right/to_timestamp",
                ],
            )
            for path in paths
        ],
        ignore_index=True,
    ).sort_values("episode_index", kind="stable")
    ids = episodes["episode_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, np.arange(len(episodes), dtype=np.int64)):
        raise ValueError("Episode IDs are not unique and contiguous from zero")
    if np.any(episodes["length"].to_numpy(dtype=np.int64) <= 0):
        raise ValueError("Episode catalog contains a non-positive length")
    return episodes.reset_index(drop=True)


def _data_path(dataset_root: Path, chunk_index: int, file_index: int) -> Path:
    return (
        dataset_root
        / "data"
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.parquet"
    )


def _verify_complete_catalog(
    dataset_root: Path,
    episodes: pd.DataFrame,
    *,
    fps: float,
) -> dict[str, Any]:
    """Verify every episode has complete numeric rows and three video bindings."""

    verified_ids: set[int] = set()
    data_paths: list[Path] = []
    grouped = episodes.groupby(
        ["data/chunk_index", "data/file_index"], sort=True, dropna=False
    )
    for (raw_chunk, raw_file), bound in grouped:
        chunk_index = int(raw_chunk)
        file_index = int(raw_file)
        path = _data_path(dataset_root, chunk_index, file_index)
        if not path.is_file():
            raise FileNotFoundError(path)
        data_paths.append(path)
        table = pq.read_table(path, columns=["episode_index", "frame_index"])
        data_episode_ids = np.asarray(
            table.column("episode_index").to_numpy(), dtype=np.int64
        )
        data_frame_ids = np.asarray(
            table.column("frame_index").to_numpy(), dtype=np.int64
        )
        expected_rows = int(bound["length"].sum())
        if len(data_episode_ids) != expected_rows:
            raise ValueError(
                f"{path} has {len(data_episode_ids)} rows, expected {expected_rows}"
            )
        for row in bound.itertuples(index=False):
            episode_id = int(row.episode_index)
            length = int(row.length)
            mask = data_episode_ids == episode_id
            observed = data_frame_ids[mask]
            if not np.array_equal(observed, np.arange(length, dtype=np.int64)):
                raise ValueError(
                    f"Episode {episode_id} does not contain exactly frame_index "
                    f"0:{length} in {path}"
                )
            verified_ids.add(episode_id)

    camera_keys = (
        "observation.images.head",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    )
    video_paths: set[Path] = set()
    for row_dict in episodes.to_dict(orient="records"):
        episode_id = int(row_dict["episode_index"])
        length = int(row_dict["length"])
        expected_duration = length / fps
        for camera_key in camera_keys:
            prefix = f"videos/{camera_key}"
            chunk_index = int(row_dict[f"{prefix}/chunk_index"])
            file_index = int(row_dict[f"{prefix}/file_index"])
            from_timestamp = float(row_dict[f"{prefix}/from_timestamp"])
            to_timestamp = float(row_dict[f"{prefix}/to_timestamp"])
            if not np.isfinite([from_timestamp, to_timestamp]).all():
                raise ValueError(
                    f"Episode {episode_id} camera {camera_key} has NaN/Inf timestamps"
                )
            observed_duration = to_timestamp - from_timestamp
            if not np.isclose(observed_duration, expected_duration, atol=2e-4):
                raise ValueError(
                    f"Episode {episode_id} camera {camera_key} duration "
                    f"{observed_duration} != {expected_duration}"
                )
            path = (
                dataset_root
                / "videos"
                / camera_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            video_paths.add(path)

    if verified_ids != set(episodes["episode_index"].astype(int).tolist()):
        raise ValueError("Not every catalog episode was verified in a data shard")
    return {
        "complete_episode_count": len(verified_ids),
        "complete_frame_count": int(episodes["length"].sum()),
        "data_file_count": len(data_paths),
        "video_file_count": len(video_paths),
        "camera_count": len(camera_keys),
        "checks": [
            "episode IDs contiguous and unique",
            "positive episode lengths",
            "per-shard row counts equal bound episode lengths",
            "each episode has exact contiguous frame_index 0:length",
            "all three per-camera video files exist",
            "per-camera to_timestamp-from_timestamp equals length/fps",
        ],
    }


def _derive_effective_global_batch(
    config_path: Path,
    launcher_path: Path,
    *,
    world_size: int,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    per_device = int(config["datasets"]["vla_data"]["per_device_batch_size"])
    grad_accum = int(config["trainer"].get("gradient_accumulation_steps", 1))
    if per_device <= 0 or grad_accum <= 0 or world_size <= 0:
        raise ValueError("Batch factors must all be positive")
    launcher_text = launcher_path.read_text(encoding="utf-8")
    pinned_world_sizes = {
        int(value)
        for value in re.findall(r"(?:export\s+)?NUM_PROCESSES(?:=|\}\s*!=\s*\")([0-9]+)", launcher_text)
    }
    if pinned_world_sizes and pinned_world_sizes != {world_size}:
        raise ValueError(
            f"Launcher NUM_PROCESSES pins {sorted(pinned_world_sizes)}, "
            f"but --world-size is {world_size}"
        )
    effective = per_device * world_size * grad_accum
    data_cfg = config["datasets"]["vla_data"]
    framework_cfg = config.get("framework", {})
    action_model_cfg = framework_cfg.get("action_model", {})
    vj2_cfg = framework_cfg.get("vj2_model", {})
    action_horizon = int(action_model_cfg.get("action_horizon", 1))
    video_horizon = int(vj2_cfg.get("num_frames", 1))
    video_stride = max(int(data_cfg.get("video_frame_stride", 1)), 1)
    target_shift = max(int(data_cfg.get("video_target_shift_steps", 0)), 0)
    qwen_frame_index = str(
        data_cfg.get("qwen_observation_frame_index", "current")
    ).strip().lower()
    if action_horizon <= 0 or video_horizon <= 0:
        raise ValueError("Action and video horizons must be positive")
    if qwen_frame_index != "current":
        raise ValueError(
            "Immutable deployment-action evaluation requires "
            "qwen_observation_frame_index=current"
        )
    # The action policy and deployed server consume the current Qwen RGB frame
    # plus current state. The V-JEPA temporal clip is an auxiliary training
    # target and is deliberately not part of action-only heldout evaluation.
    required_min_offset = 0
    required_max_offset = action_horizon - 1
    minimum_episode_length = required_max_offset - required_min_offset + 1
    return {
        "per_device_batch_size": per_device,
        "world_size": world_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_global_batch_size": effective,
        "formula": (
            "per_device_batch_size * world_size * gradient_accumulation_steps"
        ),
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "launcher_path": str(launcher_path.resolve()),
        "launcher_sha256": file_sha256(launcher_path),
        "evaluation_structural_window": {
            "action_horizon": action_horizon,
            "observation_mode": "deployment_action_current_qwen_rgb_v1",
            "qwen_observation_frame_index": qwen_frame_index,
            "evaluation_video_offsets": [0],
            "state_offsets": [0],
            "action_min_offset": 0,
            "action_max_offset": action_horizon - 1,
            "required_min_offset": required_min_offset,
            "required_max_offset": required_max_offset,
            "minimum_episode_length": minimum_episode_length,
            "excluded_auxiliary_training_context": {
                "vj_video_horizon": video_horizon,
                "vj_video_frame_stride": video_stride,
                "vj_video_target_shift_steps": target_shift,
                "reason": (
                    "V-JEPA clip is an auxiliary training target and is not "
                    "consumed by predict_action or the deployment server"
                ),
            },
        },
    }


def _rank_episodes(
    episodes: pd.DataFrame,
    *,
    seed_sha256: str,
    full_catalog_sha256: str,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in episodes.itertuples(index=False):
        episode_id = int(row.episode_index)
        length = int(row.length)
        ranking_payload = {
            "schema": RANKING_SCHEMA,
            "seed_sha256": seed_sha256,
            "full_catalog_sha256": full_catalog_sha256,
            "episode_id": episode_id,
            "length": length,
        }
        ranked.append(
            {
                "episode_id": episode_id,
                "length": length,
                "ranking_sha256": canonical_json_sha256(ranking_payload),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (item["ranking_sha256"], item["episode_id"]),
    )


def _arrow_column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        lengths = np.asarray(
            pc.list_value_length(array).to_numpy(), dtype=np.int64
        )
        if len(lengths) == 0 or np.any(lengths != lengths[0]):
            raise ValueError(f"Ragged list column cannot be summarized: {array.type}")
        width = int(lengths[0])
        values = np.asarray(array.values.to_numpy(zero_copy_only=False))
        return values.reshape(len(array), width)
    if pa.types.is_integer(array.type) or pa.types.is_floating(array.type):
        return np.asarray(array.to_numpy(zero_copy_only=False)).reshape(-1, 1)
    raise TypeError(f"Unsupported statistics column type {array.type}")


def _statistics_for_array(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Invalid statistics array shape {values.shape}")
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "count": [int(values.shape[0])],
        "max": np.max(values, axis=0).astype(np.float64).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "min": np.min(values, axis=0).astype(np.float64).tolist(),
        "q01": quantiles[0].astype(np.float64).tolist(),
        "q10": quantiles[1].astype(np.float64).tolist(),
        "q50": quantiles[2].astype(np.float64).tolist(),
        "q90": quantiles[3].astype(np.float64).tolist(),
        "q99": quantiles[4].astype(np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
    }


def _compute_train_statistics(
    data_paths: list[Path],
    *,
    holdout_ids: set[int],
    expected_train_frames: int,
) -> tuple[dict[str, Any], dict[Path, np.ndarray]]:
    keep_masks: dict[Path, np.ndarray] = {}
    observed_train_frames = 0
    for path in data_paths:
        episode_ids = np.asarray(
            pq.read_table(path, columns=["episode_index"])
            .column("episode_index")
            .to_numpy(),
            dtype=np.int64,
        )
        keep = ~np.isin(episode_ids, np.fromiter(holdout_ids, dtype=np.int64))
        keep_masks[path] = keep
        observed_train_frames += int(keep.sum())
    if observed_train_frames != expected_train_frames:
        raise ValueError(
            f"Train masks contain {observed_train_frames} frames, "
            f"expected {expected_train_frames}"
        )

    schema = pq.ParquetFile(data_paths[0]).schema_arrow
    numeric_columns = []
    for field in schema:
        value_type = field.type.value_type if pa.types.is_list(field.type) else field.type
        if pa.types.is_integer(value_type) or pa.types.is_floating(value_type):
            numeric_columns.append(field.name)

    statistics: dict[str, Any] = {}
    for column_name in numeric_columns:
        pieces: list[np.ndarray] = []
        for path in data_paths:
            column = pq.read_table(path, columns=[column_name]).column(column_name)
            values = _arrow_column_to_numpy(column)
            pieces.append(values[keep_masks[path]])
        combined = np.concatenate(pieces, axis=0)
        statistics[column_name] = _statistics_for_array(combined)
        del combined, pieces
    return statistics, keep_masks


def _counter(values: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {
        str(int(key)): int(value)
        for key, value in sorted(Counter(values[mask].astype(int).tolist()).items())
    }


def _length_summary(values: np.ndarray) -> dict[str, Any]:
    quantiles = np.quantile(values, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    return {
        "count": int(len(values)),
        "frames": int(values.sum()),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q01": float(quantiles[0]),
        "q10": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "q99": float(quantiles[6]),
    }


def _source_cohort(episode_id: int) -> str:
    if episode_id < 1442:
        return "base_cleaned_v5"
    if episode_id < 1475:
        return "v6_latest_append"
    if episode_id < 1600:
        return "intervention_append_20260701_to_20260709"
    return "intervention_append_20260709_to_20260710"


def _recover_selected_source_sessions(
    dataset_root: Path, selected_episode_ids: list[int]
) -> dict[str, Any]:
    """Recover surviving collection-session bindings without affecting selection."""

    selected = set(selected_episode_ids)
    workspace = dataset_root.parent
    bindings: dict[int, dict[str, Any]] = {}
    source_files: list[str] = []

    main_source_path = (
        workspace
        / "ogrealman_lerobot_v3_dataset/meta/source_manifest.json"
    )
    v6_source_path = (
        workspace
        / "ogrealman_lerobot_v3_dataset_cleaned_v6_latest/meta/source_manifest.json"
    )
    if main_source_path.is_file() and v6_source_path.is_file():
        source_files.extend([str(main_source_path), str(v6_source_path)])
        raw_manifest = json.loads(main_source_path.read_text(encoding="utf-8"))
        raw_bindings = {
            int(row["episode_index"]): row
            for row in raw_manifest.get("included", [])
        }
        v6_manifest = json.loads(v6_source_path.read_text(encoding="utf-8"))
        for row in v6_manifest.get("cleanup", {}).get("appended", []):
            final_episode_id = int(row["new_episode_index"])
            if final_episode_id not in selected:
                continue
            raw_row = raw_bindings.get(int(row["old_episode_index"]))
            if raw_row is None:
                continue
            bindings[final_episode_id] = {
                "episode_index": final_episode_id,
                "source_dataset": str(raw_row["source_dataset"]),
                "source_episode_index": int(raw_row["source_episode_index"]),
                "provenance": "main source_manifest + v6 cleanup appended mapping",
            }

    june_cleanup_path = (
        workspace
        / "ogrealman_lerobot_v3_dataset_new_since_20260630_142609_cleaned"
        / "meta/cleanup_manifest.json"
    )
    if june_cleanup_path.is_file():
        source_files.append(str(june_cleanup_path))
        june_manifest = json.loads(june_cleanup_path.read_text(encoding="utf-8"))
        for row in june_manifest.get("kept", []):
            final_episode_id = 1475 + int(row["new_episode_index"])
            if final_episode_id in selected:
                bindings[final_episode_id] = {
                    "episode_index": final_episode_id,
                    "source_dataset": str(row["source_dataset"]),
                    "source_episode_index": int(row["source_dataset_episode_index"]),
                    "provenance": "June-30 append cleanup kept mapping",
                }

    july_source_path = (
        workspace
        / "ogrealman_lerobot_v3_dataset_new_since_20260709_172403"
        / "meta/source_manifest.json"
    )
    if july_source_path.is_file():
        source_files.append(str(july_source_path))
        july_manifest = json.loads(july_source_path.read_text(encoding="utf-8"))
        for row in july_manifest.get("included", []):
            final_episode_id = 1600 + int(row["episode_index"])
            if final_episode_id in selected:
                bindings[final_episode_id] = {
                    "episode_index": final_episode_id,
                    "source_dataset": str(row["source_dataset"]),
                    "source_episode_index": int(row["source_episode_index"]),
                    "provenance": "July-9 append source_manifest",
                }

    unknown = sorted(selected - set(bindings))
    session_counts = Counter(
        row["source_dataset"] for row in bindings.values()
    )
    return {
        "status": "complete" if not unknown else "partial",
        "known_selected_episode_count": len(bindings),
        "unknown_selected_episode_count": len(unknown),
        "unknown_selected_episode_ids": unknown,
        "selected_bindings": [bindings[key] for key in sorted(bindings)],
        "known_source_session_episode_counts": dict(sorted(session_counts.items())),
        "source_manifest_paths": source_files,
        "note": (
            "Source-session metadata was audit-only and never affected SHA rank "
            "selection. The base-v5 cleanup manifest is no longer present, so "
            "final episodes 0:1442 cannot be bound directly to original sessions."
        ),
    }


def _reuse_existing_outputs(
    *,
    manifest_path: Path,
    stats_path: Path,
    report_path: Path,
    catalog: dict[str, Any],
    batch: dict[str, Any],
    seed_sha256: str,
    holdout_ids: list[int],
    holdout_frames: int,
    holdout_catalog_sha256: str,
    train_ids: list[int],
    train_frames: int,
    train_catalog_sha256: str,
) -> bool:
    """Validate and reuse a previously generated immutable artifact set."""

    paths = (manifest_path, stats_path, report_path)
    existing = [path.is_file() for path in paths]
    if not any(existing):
        return False
    if not all(existing):
        missing = [str(path) for path, present in zip(paths, existing) if not present]
        raise FileNotFoundError(
            "Internal-holdout artifact set is incomplete; missing " + ", ".join(missing)
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_contract = {
        "train_episode_selection": "complement_of_holdout",
        "evaluation_episode_selection": "holdout_episode_indices",
        "normalization_statistics": "train_statistics_only",
    }
    if manifest.get("schema_version") != 1:
        raise ValueError("Existing split manifest has the wrong schema_version")
    if manifest.get("role_contract") != expected_contract:
        raise ValueError("Existing split manifest has the wrong role_contract")
    selection = manifest.get("selection", {})
    if selection.get("algorithm") != RANKING_SCHEMA:
        raise ValueError("Existing split manifest has the wrong ranking algorithm")
    if selection.get("seed_sha256") != seed_sha256:
        raise ValueError("Existing split manifest has a different selection seed")
    recorded_batch = selection.get("holdout_count_derivation", {})
    for key in (
        "per_device_batch_size",
        "world_size",
        "gradient_accumulation_steps",
        "effective_global_batch_size",
        "config_sha256",
        "launcher_sha256",
    ):
        if recorded_batch.get(key) != batch[key]:
            raise ValueError(
                f"Existing split manifest batch/source binding {key} changed: "
                f"{recorded_batch.get(key)!r} != {batch[key]!r}"
            )
    if recorded_batch.get("evaluation_structural_window") != batch[
        "evaluation_structural_window"
    ]:
        raise ValueError(
            "Existing split manifest evaluation structural window changed"
        )
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("Existing split manifest must bind exactly one dataset")
    entry = datasets[0]
    expected_entry_values = {
        "full_catalog_sha256": catalog["episode_catalog_sha256"],
        "full_episode_count": catalog["total_episodes"],
        "full_frame_count": catalog["total_frames"],
        "holdout_episode_indices": holdout_ids,
        "holdout_episode_count": len(holdout_ids),
        "holdout_frame_count": holdout_frames,
        "holdout_catalog_sha256": holdout_catalog_sha256,
        "train_episode_count": len(train_ids),
        "train_frame_count": train_frames,
        "train_catalog_sha256": train_catalog_sha256,
    }
    mismatches = {
        key: (entry.get(key), expected)
        for key, expected in expected_entry_values.items()
        if entry.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Existing split manifest no longer matches: {mismatches}")

    statistics = json.loads(stats_path.read_text(encoding="utf-8"))
    expected_provenance = {
        "schema": "lerobot-train-statistics-v1",
        "full_catalog_sha256": catalog["episode_catalog_sha256"],
        "train_catalog_sha256": train_catalog_sha256,
        "train_episode_count": len(train_ids),
        "train_frame_count": train_frames,
    }
    if statistics.get("_split_provenance") != expected_provenance:
        raise ValueError("Existing train statistics has stale split provenance")
    numeric_columns = sorted(
        key for key in statistics if not str(key).startswith("_")
    )
    if not numeric_columns:
        raise ValueError("Existing train statistics has no numeric columns")
    for column_name in numeric_columns:
        stat = statistics[column_name]
        if not isinstance(stat, dict) or stat.get("count") != [train_frames]:
            raise ValueError(
                f"Existing train statistics count mismatch for {column_name!r}"
            )
        for field in ("min", "max", "mean", "std", "q01", "q99"):
            if field not in stat:
                raise ValueError(
                    f"Existing train statistics {column_name!r} lacks {field!r}"
                )
    statistics_sha256 = file_sha256(stats_path)
    bound_statistics = entry.get("train_statistics", {})
    if bound_statistics.get("sha256") != statistics_sha256:
        raise ValueError("Existing manifest does not bind the train statistics bytes")
    if bound_statistics.get("frame_count") != train_frames:
        raise ValueError("Existing manifest has the wrong train statistics count")
    if bound_statistics.get("catalog_sha256") != train_catalog_sha256:
        raise ValueError("Existing manifest has the wrong train statistics catalog")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Existing split report does not bind the manifest bytes")
    if report.get("statistics_sha256") != statistics_sha256:
        raise ValueError("Existing split report does not bind the statistics bytes")
    print(f"reused_manifest={manifest_path}")
    print(f"manifest_sha256={file_sha256(manifest_path)}")
    print(f"reused_train_statistics={stats_path}")
    print(f"train_statistics_sha256={statistics_sha256}")
    print(f"reused_report={report_path}")
    return True


def _distribution_report(
    episodes: pd.DataFrame,
    data_paths: list[Path],
    *,
    holdout_ids: set[int],
) -> dict[str, Any]:
    episode_ids = episodes["episode_index"].to_numpy(dtype=np.int64)
    lengths = episodes["length"].to_numpy(dtype=np.int64)
    holdout_episode_mask = np.isin(
        episode_ids, np.fromiter(holdout_ids, dtype=np.int64)
    )
    splits = {
        "full": np.ones(len(episodes), dtype=bool),
        "train": ~holdout_episode_mask,
        "holdout": holdout_episode_mask,
    }

    scalar_columns = [
        "episode_index",
        "valid_state",
        "valid_state_source",
        "subtask_index",
        "task_id",
    ]
    frame_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in scalar_columns
    }
    for path in data_paths:
        table = pq.read_table(path, columns=scalar_columns)
        for key in scalar_columns:
            frame_parts[key].append(
                np.asarray(table.column(key).to_numpy(), dtype=np.int64)
            )
    frame_values = {
        key: np.concatenate(parts) for key, parts in frame_parts.items()
    }
    frame_is_holdout = np.isin(
        frame_values["episode_index"], np.fromiter(holdout_ids, dtype=np.int64)
    )
    frame_splits = {
        "full": np.ones(len(frame_is_holdout), dtype=bool),
        "train": ~frame_is_holdout,
        "holdout": frame_is_holdout,
    }

    report: dict[str, Any] = {}
    for split_name, episode_mask in splits.items():
        frame_mask = frame_splits[split_name]
        split_episode_ids = episode_ids[episode_mask]
        cohort_episode_counts = Counter(
            _source_cohort(int(value)) for value in split_episode_ids
        )
        cohort_frame_counts: Counter[str] = Counter()
        for episode_id, length in zip(
            episode_ids[episode_mask], lengths[episode_mask]
        ):
            cohort_frame_counts[_source_cohort(int(episode_id))] += int(length)

        valid_counts = np.bincount(
            frame_values["episode_index"][frame_mask],
            weights=(frame_values["valid_state"][frame_mask] == 1).astype(np.int64),
            minlength=len(episodes),
        ).astype(np.int64)
        selected_valid_counts = valid_counts[split_episode_ids]
        selected_lengths = lengths[episode_mask]
        valid_fractions = selected_valid_counts / selected_lengths
        all_invalid_ids = split_episode_ids[selected_valid_counts == 0]
        all_valid_ids = split_episode_ids[selected_valid_counts == selected_lengths]
        report[split_name] = {
            "episode_count": int(episode_mask.sum()),
            "frame_count": int(frame_mask.sum()),
            "length": _length_summary(selected_lengths),
            "valid_state_frame_counts": _counter(
                frame_values["valid_state"], frame_mask
            ),
            "valid_state_source_frame_counts": _counter(
                frame_values["valid_state_source"], frame_mask
            ),
            "subtask_index_frame_counts": _counter(
                frame_values["subtask_index"], frame_mask
            ),
            "task_id_frame_counts": _counter(
                frame_values["task_id"], frame_mask
            ),
            "episode_valid_fraction": {
                "mean": float(valid_fractions.mean()),
                "min": float(valid_fractions.min()),
                "max": float(valid_fractions.max()),
                "all_valid_episode_count": int(
                    np.count_nonzero(selected_valid_counts == selected_lengths)
                ),
                "all_valid_episode_ids": all_valid_ids.astype(int).tolist(),
                "mixed_episode_count": int(
                    np.count_nonzero(
                        (selected_valid_counts > 0)
                        & (selected_valid_counts < selected_lengths)
                    )
                ),
                "all_invalid_episode_count": int(
                    np.count_nonzero(selected_valid_counts == 0)
                ),
                "all_invalid_episode_ids": all_invalid_ids.astype(int).tolist(),
            },
            "source_cohort_episode_counts": dict(sorted(cohort_episode_counts.items())),
            "source_cohort_frame_counts": dict(sorted(cohort_frame_counts.items())),
        }

    full = report["full"]
    full_valid_fraction = (
        full["valid_state_frame_counts"].get("1", 0) / full["frame_count"]
    )
    for split_name in ("train", "holdout"):
        split = report[split_name]
        valid_fraction = (
            split["valid_state_frame_counts"].get("1", 0) / split["frame_count"]
        )
        cohort_delta = {}
        cohort_names = sorted(
            set(full["source_cohort_episode_counts"])
            | set(split["source_cohort_episode_counts"])
        )
        for cohort_name in cohort_names:
            full_fraction = (
                full["source_cohort_episode_counts"].get(cohort_name, 0)
                / full["episode_count"]
            )
            split_fraction = (
                split["source_cohort_episode_counts"].get(cohort_name, 0)
                / split["episode_count"]
            )
            cohort_delta[cohort_name] = {
                "full_fraction": full_fraction,
                "split_fraction": split_fraction,
                "delta_percentage_points": 100.0 * (split_fraction - full_fraction),
            }
        report[split_name]["deltas_vs_full"] = {
            "valid_frame_fraction": valid_fraction,
            "full_valid_frame_fraction": full_valid_fraction,
            "valid_frame_fraction_delta_percentage_points": 100.0
            * (valid_fraction - full_valid_fraction),
            "mean_episode_length_delta_frames": (
                split["length"]["mean"] - full["length"]["mean"]
            ),
            "mean_episode_length_delta_percent": 100.0
            * (split["length"]["mean"] / full["length"]["mean"] - 1.0),
            "source_cohort_episode_fraction": cohort_delta,
        }
    return report


def build(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = args.dataset_root.expanduser().resolve()
    config_path = (
        args.config if args.config.is_absolute() else repo_root / args.config
    ).resolve()
    launcher_path = (
        args.launcher if args.launcher.is_absolute() else repo_root / args.launcher
    ).resolve()
    batch = _derive_effective_global_batch(
        config_path,
        launcher_path,
        world_size=args.world_size,
    )
    if args.manifest is None:
        manifest_argument = _default_manifest_argument(
            int(batch["effective_global_batch_size"])
        )
    else:
        manifest_argument = args.manifest
    manifest_path = (
        manifest_argument
        if manifest_argument.is_absolute()
        else repo_root / manifest_argument
    ).resolve()
    stats_path = manifest_path.parent / "artifacts" / (
        manifest_path.stem + "_train_stats.json"
    )
    report_path = manifest_path.with_name(manifest_path.stem + "_report.json")

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    episodes = _load_episode_table(dataset_root)
    completeness = _verify_complete_catalog(dataset_root, episodes, fps=fps)
    if int(info["total_episodes"]) != len(episodes):
        raise ValueError("info.total_episodes does not match episode catalog")
    if int(info["total_frames"]) != int(episodes["length"].sum()):
        raise ValueError("info.total_frames does not match episode length sum")

    catalog = build_episode_catalog_binding(
        dataset_path=dataset_root,
        dataset_name=dataset_root.name,
        lerobot_version="v3.0",
        trajectory_ids=episodes["episode_index"].to_numpy(dtype=np.int64),
        trajectory_lengths=episodes["length"].to_numpy(dtype=np.int64),
    )
    holdout_count = int(batch["effective_global_batch_size"])
    if holdout_count >= len(episodes):
        raise ValueError("Effective global batch leaves no training episodes")

    structural_window = batch["evaluation_structural_window"]
    minimum_episode_length = int(structural_window["minimum_episode_length"])
    eligible_mask = episodes["length"].to_numpy(dtype=np.int64) >= minimum_episode_length
    eligible_episodes = episodes[eligible_mask].reset_index(drop=True)
    ineligible_episodes = episodes[~eligible_mask]
    if holdout_count > len(eligible_episodes):
        raise ValueError(
            f"Effective global batch requests {holdout_count} held-out episodes, "
            f"but only {len(eligible_episodes)} episodes can provide an unpadded "
            f"window spanning offsets [{structural_window['required_min_offset']}, "
            f"{structural_window['required_max_offset']}]."
        )

    seed_sha256 = hashlib.sha256(args.seed_text.encode("utf-8")).hexdigest()
    ranked = _rank_episodes(
        eligible_episodes,
        seed_sha256=seed_sha256,
        full_catalog_sha256=catalog["episode_catalog_sha256"],
    )
    selected_ranked = ranked[:holdout_count]
    holdout_ids = sorted(int(row["episode_id"]) for row in selected_ranked)
    holdout_set = set(holdout_ids)
    all_ids = episodes["episode_index"].astype(int).tolist()
    train_ids = sorted(set(all_ids) - holdout_set)
    lengths_by_id = {
        int(row.episode_index): int(row.length)
        for row in episodes.itertuples(index=False)
    }
    holdout_frames = sum(lengths_by_id[value] for value in holdout_ids)
    train_frames = sum(lengths_by_id[value] for value in train_ids)
    holdout_catalog_sha256 = episode_set_sha256(
        holdout_ids, lengths_by_id=lengths_by_id
    )
    train_catalog_sha256 = episode_set_sha256(
        train_ids, lengths_by_id=lengths_by_id
    )

    if not args.overwrite and _reuse_existing_outputs(
        manifest_path=manifest_path,
        stats_path=stats_path,
        report_path=report_path,
        catalog=catalog,
        batch=batch,
        seed_sha256=seed_sha256,
        holdout_ids=holdout_ids,
        holdout_frames=holdout_frames,
        holdout_catalog_sha256=holdout_catalog_sha256,
        train_ids=train_ids,
        train_frames=train_frames,
        train_catalog_sha256=train_catalog_sha256,
    ):
        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "report_path": str(report_path),
                    "reused": True,
                    "statistics_path": str(stats_path),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return

    data_paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    statistics, _ = _compute_train_statistics(
        data_paths,
        holdout_ids=holdout_set,
        expected_train_frames=train_frames,
    )
    numeric_statistic_columns = sorted(statistics)
    statistics["_split_provenance"] = {
        "schema": "lerobot-train-statistics-v1",
        "full_catalog_sha256": catalog["episode_catalog_sha256"],
        "train_catalog_sha256": train_catalog_sha256,
        "train_episode_count": len(train_ids),
        "train_frame_count": train_frames,
    }
    _write_json(stats_path, statistics)
    stats_sha256 = file_sha256(stats_path)

    evaluation_seed_text = (
        args.seed_text + "|nonzero_valid_unpadded_uniform_v1"
    )
    evaluation_seed_sha256 = hashlib.sha256(
        evaluation_seed_text.encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "split_id": manifest_path.stem,
        "description": (
            "One effective-global-batch deterministic random episode holdout "
            "for the corrected Magna training run."
        ),
        "role_contract": {
            "train_episode_selection": "complement_of_holdout",
            "evaluation_episode_selection": "holdout_episode_indices",
            "normalization_statistics": "train_statistics_only",
        },
        "selection": {
            "algorithm": RANKING_SCHEMA,
            "kind": "deterministic_sha256_rank_prefix",
            "seed_text_utf8": args.seed_text,
            "seed_sha256": seed_sha256,
            "ranking_payload_fields": [
                "schema",
                "seed_sha256",
                "full_catalog_sha256",
                "episode_id",
                "length",
            ],
            "tie_breaker": "episode_id_ascending",
            "nested_prefix": True,
            "eligible_episode_policy": (
                "complete episodes with at least one structurally unpadded "
                "deployment-observation/action window"
            ),
            "eligibility": {
                **structural_window,
                "full_complete_episode_count": len(episodes),
                "eligible_episode_count": len(eligible_episodes),
                "ineligible_episode_count": len(ineligible_episodes),
                "ineligible_episodes": [
                    {
                        "episode_id": int(row.episode_index),
                        "length": int(row.length),
                        "reason": (
                            f"length<{minimum_episode_length} cannot span offsets "
                            f"[{structural_window['required_min_offset']},"
                            f"{structural_window['required_max_offset']}]"
                        ),
                    }
                    for row in ineligible_episodes.itertuples(index=False)
                ],
            },
            "selected_episode_ids_in_rank_order": [
                int(row["episode_id"]) for row in selected_ranked
            ],
            "selected_ranking_sha256": [
                str(row["ranking_sha256"]) for row in selected_ranked
            ],
            "holdout_count_derivation": batch,
        },
        "evaluation_sampling": {
            "algorithm": "nonzero_valid_unpadded_uniform_v1",
            "frames_per_episode": 1,
            "observation_mode": structural_window["observation_mode"],
            "evaluation_video_offsets": structural_window[
                "evaluation_video_offsets"
            ],
            "action_offset_range_inclusive": [
                structural_window["action_min_offset"],
                structural_window["action_max_offset"],
            ],
            "seed_text_utf8": evaluation_seed_text,
            "seed_sha256": evaluation_seed_sha256,
            "candidate_policy": (
                "structurally unpadded frames with at least one valid action-mask element"
            ),
            "all_invalid_episode_fallback": (
                "uniform over all structurally unpadded frames; report zero valid elements"
            ),
        },
        "datasets": [
            {
                "dataset_name": dataset_root.name,
                "dataset_root": str(dataset_root),
                "lerobot_version": "v3.0",
                "info_sha256": catalog["info_sha256"],
                "full_catalog_sha256": catalog["episode_catalog_sha256"],
                "full_episode_count": int(catalog["total_episodes"]),
                "full_frame_count": int(catalog["total_frames"]),
                "holdout_episode_indices": holdout_ids,
                "holdout_episode_count": len(holdout_ids),
                "holdout_frame_count": holdout_frames,
                "holdout_catalog_sha256": holdout_catalog_sha256,
                "train_episode_selection": {"kind": "complement_of_holdout"},
                "train_episode_count": len(train_ids),
                "train_frame_count": train_frames,
                "train_catalog_sha256": train_catalog_sha256,
                "train_statistics": {
                    "path": stats_path.relative_to(manifest_path.parent).as_posix(),
                    "sha256": stats_sha256,
                    "frame_count": train_frames,
                    "catalog_sha256": train_catalog_sha256,
                    "numeric_columns": numeric_statistic_columns,
                    "image_statistics_included": False,
                },
            }
        ],
    }
    _write_json(manifest_path, manifest)

    selected_lengths = np.asarray(
        [lengths_by_id[value] for value in holdout_ids], dtype=np.int64
    )
    report = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "statistics_path": str(stats_path),
        "statistics_sha256": stats_sha256,
        "catalog_binding": catalog,
        "completeness_verification": completeness,
        "selection": {
            "holdout_episode_count": len(holdout_ids),
            "holdout_frame_count": holdout_frames,
            "train_episode_count": len(train_ids),
            "train_frame_count": train_frames,
            "minimum_selected_episode_length": int(selected_lengths.min()),
            "eligibility": manifest["selection"]["eligibility"],
            "selected_episode_ids_sorted": holdout_ids,
            "selected_episode_ids_in_rank_order": [
                int(row["episode_id"]) for row in selected_ranked
            ],
            "source_session_provenance": _recover_selected_source_sessions(
                dataset_root, holdout_ids
            ),
        },
        "distributions": _distribution_report(
            episodes, data_paths, holdout_ids=holdout_set
        ),
        "statistics": {
            "scope": "train_split_only",
            "frame_count": train_frames,
            "numeric_columns": numeric_statistic_columns,
            "image_statistics_included": False,
            "image_statistics_note": (
                "Image tensors use fixed model preprocessing; no dataset image "
                "statistics are consumed by the state/action normalizer."
            ),
        },
    }
    _write_json(report_path, report)
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={file_sha256(manifest_path)}")
    print(f"train_statistics={stats_path}")
    print(f"train_statistics_sha256={stats_sha256}")
    print(f"report={report_path}")
    print(f"holdout={len(holdout_ids)} episodes / {holdout_frames} frames")
    print(f"train={len(train_ids)} episodes / {train_frames} frames")
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "manifest_path": str(manifest_path),
                "report_path": str(report_path),
                "reused": False,
                "statistics_path": str(stats_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seed-text", default=DEFAULT_SEED_TEXT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Output manifest. By default the filename is derived after reading "
            "the effective global batch: " + DEFAULT_MANIFEST_TEMPLATE
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
