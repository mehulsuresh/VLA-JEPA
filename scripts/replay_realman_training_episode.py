#!/usr/bin/env python3
"""Replay one RealMan training episode through a policy server and plot predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.realman.pipeline import (
    MAGNA_DEFAULT_INSTRUCTION,
    REALMAN_CAMERA_ORDER,
    build_policy_payload,
    realman_continuous_unnormalize,
    resolve_qwen_frame_size,
    validate_realman_policy_payload,
    validate_realman_server_metadata,
)
from deployment.trossen.pipeline import (
    continuous_normalize,
    resolve_action_stats,
    resolve_norm_mode,
    resolve_state_stats,
)


MODEL_ACTION_SOURCE_INDICES = tuple(range(16)) + (19, 20)
MODEL_STATE_SOURCE_INDICES = tuple(range(18))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument(
        "--compare-window",
        type=int,
        default=25,
        help="Number of leading actions from every prediction used for the stitched comparison.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--unnorm-key", default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_episode_metadata(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    episodes = pads.dataset(str(dataset_root / "meta" / "episodes"), format="parquet")
    table = episodes.to_table(filter=pads.field("episode_index") == int(episode_index))
    if len(table) != 1:
        raise RuntimeError(
            f"Expected exactly one metadata row for episode {episode_index}, found {len(table)}."
        )
    return table.to_pylist()[0]


def _load_episode_rows(dataset_root: Path, meta: dict[str, Any]) -> dict[str, np.ndarray]:
    data_path = (
        dataset_root
        / "data"
        / f"chunk-{int(meta['data/chunk_index']):03d}"
        / f"file-{int(meta['data/file_index']):03d}.parquet"
    )
    columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "source.action",
        "source.observation.state",
        "valid_state",
    ]
    table = pq.read_table(
        data_path,
        columns=columns,
        filters=[("episode_index", "=", int(meta["episode_index"]))],
    )
    order = np.argsort(np.asarray(table["frame_index"].to_numpy(), dtype=np.int64))
    result = {
        "frame_index": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)[order],
        "timestamp": np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)[order],
        "source_action": np.asarray(table["source.action"].to_pylist(), dtype=np.float32)[order],
        "source_state": np.asarray(
            table["source.observation.state"].to_pylist(), dtype=np.float32
        )[order],
        "valid_state": np.asarray(table["valid_state"].to_numpy(), dtype=bool)[order],
    }
    expected_length = int(meta["length"])
    if len(result["frame_index"]) != expected_length:
        raise RuntimeError(
            f"Episode row count {len(result['frame_index'])} does not match metadata {expected_length}."
        )
    expected_frames = np.arange(expected_length, dtype=np.int64)
    if not np.array_equal(result["frame_index"], expected_frames):
        raise RuntimeError("Episode frame_index is not contiguous from zero.")
    return result


def _video_path(dataset_root: Path, meta: dict[str, Any], camera: str) -> Path:
    prefix = f"videos/observation.images.{camera}"
    return (
        dataset_root
        / "videos"
        / f"observation.images.{camera}"
        / f"chunk-{int(meta[f'{prefix}/chunk_index']):03d}"
        / f"file-{int(meta[f'{prefix}/file_index']):03d}.mp4"
    )


def _decode_anchor_images(
    dataset_root: Path,
    meta: dict[str, Any],
    anchors: np.ndarray,
    expected_fps: float,
) -> dict[int, dict[str, np.ndarray]]:
    anchor_set = {int(index) for index in anchors}
    max_anchor = int(anchors.max())
    images: dict[int, dict[str, np.ndarray]] = {int(index): {} for index in anchors}

    for camera in REALMAN_CAMERA_ORDER:
        path = _video_path(dataset_root, meta, camera)
        if not path.exists():
            raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open {path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f"Video {path} reports {fps} fps, expected {expected_fps}.")
            prefix = f"videos/observation.images.{camera}"
            start_timestamp = float(meta[f"{prefix}/from_timestamp"])
            start_frame = int(round(start_timestamp * fps))
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
                raise RuntimeError(f"Failed to seek {path} to frame {start_frame}.")
            for local_index in range(max_anchor + 1):
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise RuntimeError(
                        f"Video decode failed for {camera} at episode frame {local_index}."
                    )
                if local_index in anchor_set:
                    images[local_index][camera] = np.ascontiguousarray(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
                    )
        finally:
            capture.release()

    for anchor, camera_images in images.items():
        missing = set(REALMAN_CAMERA_ORDER) - set(camera_images)
        if missing:
            raise RuntimeError(f"Anchor {anchor} is missing decoded cameras {sorted(missing)}.")
    return images


def _masked_mae(values: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    expanded = np.broadcast_to(mask[..., None], values.shape)
    errors = np.abs(values - targets)[expanded]
    return float(errors.mean()) if errors.size else math.nan


def _per_dim_mae(values: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = np.broadcast_to(mask[..., None], values.shape)
    errors = np.abs(values - targets)
    numerator = (errors * expanded).sum(axis=tuple(range(errors.ndim - 1)))
    denominator = expanded.sum(axis=tuple(range(expanded.ndim - 1)))
    return np.divide(
        numerator,
        denominator,
        out=np.full(values.shape[-1], np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _plot_trajectories(
    output_path: Path,
    episode_index: int,
    action_names: list[str],
    actual_episode: np.ndarray,
    anchors: np.ndarray,
    predictions: np.ndarray,
    compare_window: int,
    raw_mae: float,
    normalized_mae: float,
    checkpoint_path: str,
) -> None:
    rows, cols = 6, 3
    figure, axes = plt.subplots(rows, cols, figsize=(24, 29), sharex=True)
    x_full = np.arange(len(actual_episode), dtype=np.int64)
    for dim, axis in enumerate(axes.flat):
        axis.plot(x_full, actual_episode[:, dim], color="black", linewidth=1.45, zorder=3)
        for anchor_index, anchor in enumerate(anchors):
            horizon_x = int(anchor) + np.arange(predictions.shape[1])
            axis.plot(
                horizon_x,
                predictions[anchor_index, :, dim],
                color="#4C9AD4",
                alpha=0.22,
                linewidth=0.9,
                zorder=1,
            )
            used = min(compare_window, predictions.shape[1])
            axis.plot(
                horizon_x[:used],
                predictions[anchor_index, :used, dim],
                color="#E24A33",
                alpha=0.82,
                linewidth=1.05,
                zorder=2,
            )
        axis.set_title(action_names[dim], fontsize=11)
        axis.grid(alpha=0.2)
        if "gripper" in action_names[dim]:
            axis.set_ylim(-0.08, 1.08)
        if dim % cols == 0:
            axis.set_ylabel("absolute target")
        if dim // cols == rows - 1:
            axis.set_xlabel("episode action index")

    legend = [
        Line2D([0], [0], color="black", linewidth=2, label="recorded training action"),
        Line2D([0], [0], color="#E24A33", linewidth=2, label=f"prediction used (first {compare_window})"),
        Line2D([0], [0], color="#4C9AD4", linewidth=2, alpha=0.45, label="full 50-step forecast"),
    ]
    checkpoint_name = Path(checkpoint_path).parent.name if checkpoint_path else "unknown"
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        f"Training episode {episode_index}: actual vs policy prediction every {int(anchors[1] - anchors[0]) if len(anchors) > 1 else 0} actions\n"
        f"checkpoint={checkpoint_name} | stitched raw MAE={raw_mae:.4f} | normalized MAE={normalized_mae:.4f}",
        fontsize=15,
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_error_bars(
    output_path: Path,
    action_names: list[str],
    model_mae: np.ndarray,
    hold_mae: np.ndarray,
) -> None:
    x = np.arange(len(action_names))
    width = 0.39
    figure, axis = plt.subplots(figsize=(18, 7))
    axis.bar(x - width / 2, model_mae, width, label="model", color="#4C9AD4")
    axis.bar(x + width / 2, hold_mae, width, label="hold-current-state baseline", color="#A6A6A6")
    axis.set_ylabel("normalized MAE (lower is better)")
    axis.set_title("Every-25-action replay error by output dimension")
    axis.set_xticks(x)
    axis.set_xticklabels(action_names, rotation=58, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_horizon_mae(
    output_path: Path,
    model_mae: np.ndarray,
    hold_mae: np.ndarray,
) -> None:
    horizons = np.arange(1, len(model_mae) + 1)
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(horizons, model_mae, color="#4C9AD4", linewidth=2.2, label="model")
    axis.plot(
        horizons,
        hold_mae,
        color="#777777",
        linewidth=2.2,
        label="hold-current-state baseline",
    )
    axis.set_xlabel("forecast horizon (actions)")
    axis.set_ylabel("cumulative normalized MAE")
    axis.set_title("Prediction error as the forecast extends into the future")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    holdout_manifest_path = args.holdout_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with holdout_manifest_path.open("r", encoding="utf-8") as handle:
        holdout_manifest = json.load(handle)
    dataset_entry = holdout_manifest["datasets"][0]
    holdout_indices = {int(index) for index in dataset_entry["holdout_episode_indices"]}
    if args.episode_index in holdout_indices:
        raise RuntimeError(
            f"Episode {args.episode_index} is in the held-out split; choose a training episode."
        )

    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        dataset_info = json.load(handle)
    fps = float(dataset_info["fps"])
    episode_meta = _load_episode_metadata(dataset_root, args.episode_index)
    episode = _load_episode_rows(dataset_root, episode_meta)
    instruction = args.instruction or (episode_meta.get("tasks") or [MAGNA_DEFAULT_INSTRUCTION])[0]

    client = WebsocketClientPolicy(host=args.host, port=args.port, timeout=15)
    try:
        metadata = client.get_server_metadata()
        warnings = validate_realman_server_metadata(metadata, require_input_contract=True)
        if warnings:
            raise RuntimeError("Policy metadata mismatch: " + "; ".join(warnings))
        action_horizon = int(metadata["action_horizon"])
        action_dim = int(metadata["action_dim"])
        if action_dim != len(MODEL_ACTION_SOURCE_INDICES):
            raise RuntimeError(f"Expected 18 policy actions, server reports {action_dim}.")
        if args.stride <= 0 or args.compare_window <= 0:
            raise ValueError("stride and compare-window must be positive.")
        if args.compare_window > action_horizon:
            raise ValueError("compare-window cannot exceed the server action horizon.")

        anchors = np.arange(
            0,
            len(episode["frame_index"]) - action_horizon + 1,
            args.stride,
            dtype=np.int64,
        )
        if anchors.size == 0:
            raise RuntimeError("Episode is shorter than the policy action horizon.")
        anchor_images = _decode_anchor_images(dataset_root, episode_meta, anchors, fps)

        action_stats = resolve_action_stats(metadata, args.unnorm_key)
        state_stats = resolve_state_stats(metadata, args.unnorm_key)
        action_norm_mode = resolve_norm_mode(metadata, "action", "auto")
        state_norm_mode = resolve_norm_mode(metadata, "state", "auto")
        image_size = resolve_qwen_frame_size(metadata)
        action_names = [str(name) for name in metadata["policy_action_names"]]

        actual_actions = episode["source_action"][:, MODEL_ACTION_SOURCE_INDICES]
        input_states = episode["source_state"][:, MODEL_STATE_SOURCE_INDICES]
        predictions: list[np.ndarray] = []
        normalized_predictions: list[np.ndarray] = []
        latencies_ms: list[float] = []

        for request_index, anchor in enumerate(anchors, start=1):
            observation = {
                "source.observation.state": episode["source_state"][anchor],
                **{
                    f"observation.images.{camera}": anchor_images[int(anchor)][camera]
                    for camera in REALMAN_CAMERA_ORDER
                },
            }
            payload = build_policy_payload(
                observation,
                instruction=instruction,
                image_size=image_size,
                state_stats=state_stats,
                state_norm_mode=state_norm_mode,
            )
            validate_realman_policy_payload(payload, metadata)
            started = time.perf_counter()
            response = client.infer(payload)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if not response.get("ok", False):
                raise RuntimeError(f"Inference failed at anchor {anchor}: {response}")
            normalized = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)
            if normalized.shape != (1, action_horizon, action_dim):
                raise RuntimeError(
                    f"Unexpected prediction shape {normalized.shape}; expected "
                    f"{(1, action_horizon, action_dim)}."
                )
            normalized = normalized[0]
            prediction = realman_continuous_unnormalize(
                normalized,
                action_stats,
                mode=action_norm_mode,
            )
            normalized_predictions.append(normalized)
            predictions.append(prediction)
            latencies_ms.append(latency_ms)
            print(
                f"[{request_index:02d}/{len(anchors):02d}] anchor={int(anchor):4d} "
                f"latency={latency_ms:7.1f} ms",
                flush=True,
            )
    finally:
        client.close()

    predictions_array = np.stack(predictions)
    normalized_predictions_array = np.stack(normalized_predictions)
    ground_truth_chunks = np.stack(
        [actual_actions[anchor : anchor + predictions_array.shape[1]] for anchor in anchors]
    )
    valid_chunks = np.stack(
        [episode["valid_state"][anchor : anchor + predictions_array.shape[1]] for anchor in anchors]
    )
    normalized_ground_truth = continuous_normalize(
        ground_truth_chunks,
        action_stats,
        mode=action_norm_mode,
    )

    compare_window = min(args.compare_window, predictions_array.shape[1])
    used_predictions = predictions_array[:, :compare_window]
    used_ground_truth = ground_truth_chunks[:, :compare_window]
    used_valid = valid_chunks[:, :compare_window]
    used_normalized_predictions = normalized_predictions_array[:, :compare_window]
    used_normalized_ground_truth = normalized_ground_truth[:, :compare_window]
    hold_predictions = np.repeat(
        input_states[anchors, None, :], compare_window, axis=1
    ).astype(np.float32)
    hold_normalized = continuous_normalize(
        hold_predictions,
        action_stats,
        mode=action_norm_mode,
    )
    full_hold_predictions = np.repeat(
        input_states[anchors, None, :], predictions_array.shape[1], axis=1
    ).astype(np.float32)
    full_hold_normalized = continuous_normalize(
        full_hold_predictions,
        action_stats,
        mode=action_norm_mode,
    )

    raw_mae = _masked_mae(used_predictions, used_ground_truth, used_valid)
    normalized_mae = _masked_mae(
        used_normalized_predictions, used_normalized_ground_truth, used_valid
    )
    hold_raw_mae = _masked_mae(hold_predictions, used_ground_truth, used_valid)
    hold_normalized_mae = _masked_mae(
        hold_normalized, used_normalized_ground_truth, used_valid
    )
    full_raw_mae = _masked_mae(predictions_array, ground_truth_chunks, valid_chunks)
    full_normalized_mae = _masked_mae(
        normalized_predictions_array, normalized_ground_truth, valid_chunks
    )
    full_hold_raw_mae = _masked_mae(
        full_hold_predictions, ground_truth_chunks, valid_chunks
    )
    full_hold_normalized_mae = _masked_mae(
        full_hold_normalized, normalized_ground_truth, valid_chunks
    )
    horizon_model_normalized_mae = np.asarray(
        [
            _masked_mae(
                normalized_predictions_array[:, :horizon],
                normalized_ground_truth[:, :horizon],
                valid_chunks[:, :horizon],
            )
            for horizon in range(1, predictions_array.shape[1] + 1)
        ],
        dtype=np.float64,
    )
    horizon_hold_normalized_mae = np.asarray(
        [
            _masked_mae(
                full_hold_normalized[:, :horizon],
                normalized_ground_truth[:, :horizon],
                valid_chunks[:, :horizon],
            )
            for horizon in range(1, predictions_array.shape[1] + 1)
        ],
        dtype=np.float64,
    )
    per_dim_raw = _per_dim_mae(used_predictions, used_ground_truth, used_valid)
    per_dim_hold_raw = _per_dim_mae(hold_predictions, used_ground_truth, used_valid)
    per_dim_normalized = _per_dim_mae(
        used_normalized_predictions, used_normalized_ground_truth, used_valid
    )
    per_dim_hold_normalized = _per_dim_mae(
        hold_normalized, used_normalized_ground_truth, used_valid
    )

    gripper_metrics: dict[str, Any] = {}
    for name in ("left_gripper", "right_gripper"):
        dim = action_names.index(name)
        mask = used_valid.reshape(-1)
        predicted_binary = used_predictions[..., dim].reshape(-1)[mask] >= 0.5
        actual_binary = used_ground_truth[..., dim].reshape(-1)[mask] >= 0.5
        gripper_metrics[name] = {
            "binary_accuracy": float(np.mean(predicted_binary == actual_binary)),
            "actual_open_fraction": float(np.mean(actual_binary)),
            "predicted_open_fraction": float(np.mean(predicted_binary)),
        }

    metrics = {
        "provenance": {
            "dataset_root": str(dataset_root),
            "dataset_info_sha256": _sha256(info_path),
            "manifest": str(holdout_manifest_path),
            "manifest_sha256": _sha256(holdout_manifest_path),
            "split_id": holdout_manifest.get("split_id"),
            "episode_index": args.episode_index,
            "episode_is_training": args.episode_index not in holdout_indices,
            "episode_length": len(episode["frame_index"]),
            "instruction": instruction,
            "valid_state_fraction": float(episode["valid_state"].mean()),
            "server_checkpoint_path": metadata.get("checkpoint_path"),
            "server_run_id": metadata.get("run_id"),
        },
        "replay": {
            "fps": fps,
            "stride": args.stride,
            "compare_window": compare_window,
            "model_horizon": predictions_array.shape[1],
            "anchor_count": len(anchors),
            "anchors": anchors,
            "latency_ms_mean": float(np.mean(latencies_ms)),
            "latency_ms_p95": float(np.quantile(latencies_ms, 0.95)),
        },
        "metrics": {
            "stitched_model_raw_mae": raw_mae,
            "stitched_hold_raw_mae": hold_raw_mae,
            "stitched_model_normalized_mae": normalized_mae,
            "stitched_hold_normalized_mae": hold_normalized_mae,
            "model_vs_hold_normalized_mae_ratio": (
                normalized_mae / hold_normalized_mae if hold_normalized_mae else math.nan
            ),
            "full_horizon_model_raw_mae": full_raw_mae,
            "full_horizon_model_normalized_mae": full_normalized_mae,
            "full_horizon_hold_raw_mae": full_hold_raw_mae,
            "full_horizon_hold_normalized_mae": full_hold_normalized_mae,
            "full_horizon_model_vs_hold_normalized_mae_ratio": (
                full_normalized_mae / full_hold_normalized_mae
                if full_hold_normalized_mae
                else math.nan
            ),
            "cumulative_normalized_mae_by_horizon": horizon_model_normalized_mae,
            "cumulative_hold_normalized_mae_by_horizon": horizon_hold_normalized_mae,
            "per_dimension_raw_mae": dict(zip(action_names, per_dim_raw, strict=True)),
            "per_dimension_hold_raw_mae": dict(
                zip(action_names, per_dim_hold_raw, strict=True)
            ),
            "per_dimension_normalized_mae": dict(
                zip(action_names, per_dim_normalized, strict=True)
            ),
            "per_dimension_hold_normalized_mae": dict(
                zip(action_names, per_dim_hold_normalized, strict=True)
            ),
            "grippers": gripper_metrics,
        },
    }

    arrays_path = output_dir / f"episode_{args.episode_index:06d}_replay_arrays.npz"
    np.savez_compressed(
        arrays_path,
        anchors=anchors,
        predictions=predictions_array,
        normalized_predictions=normalized_predictions_array,
        ground_truth_chunks=ground_truth_chunks,
        normalized_ground_truth=normalized_ground_truth,
        actual_episode=actual_actions,
        input_state_episode=input_states,
        valid_state_episode=episode["valid_state"],
        action_names=np.asarray(action_names),
    )

    metrics_path = output_dir / f"episode_{args.episode_index:06d}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(metrics), handle, indent=2)

    csv_path = output_dir / f"episode_{args.episode_index:06d}_per_joint_errors.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "action_name",
                "model_raw_mae",
                "hold_raw_mae",
                "model_normalized_mae",
                "hold_normalized_mae",
            ]
        )
        for index, name in enumerate(action_names):
            writer.writerow(
                [
                    name,
                    float(per_dim_raw[index]),
                    float(per_dim_hold_raw[index]),
                    float(per_dim_normalized[index]),
                    float(per_dim_hold_normalized[index]),
                ]
            )

    trajectory_plot = output_dir / f"episode_{args.episode_index:06d}_all_joints.png"
    _plot_trajectories(
        trajectory_plot,
        args.episode_index,
        action_names,
        actual_actions,
        anchors,
        predictions_array,
        compare_window,
        raw_mae,
        normalized_mae,
        str(metadata.get("checkpoint_path") or ""),
    )
    error_plot = output_dir / f"episode_{args.episode_index:06d}_error_by_joint.png"
    _plot_error_bars(error_plot, action_names, per_dim_normalized, per_dim_hold_normalized)
    horizon_plot = output_dir / f"episode_{args.episode_index:06d}_mae_by_horizon.png"
    _plot_horizon_mae(
        horizon_plot,
        horizon_model_normalized_mae,
        horizon_hold_normalized_mae,
    )

    print(json.dumps(_json_value(metrics), indent=2), flush=True)
    print(f"trajectory_plot={trajectory_plot}", flush=True)
    print(f"error_plot={error_plot}", flush=True)
    print(f"horizon_plot={horizon_plot}", flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"arrays={arrays_path}", flush=True)


if __name__ == "__main__":
    main()
