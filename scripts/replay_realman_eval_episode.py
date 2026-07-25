#!/usr/bin/env python3
"""Replay one held-out RealMan episode and plot decoded policy predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    realman_policy_actions_to_absolute,
    resolve_qwen_frame_size,
    validate_realman_policy_payload,
    validate_realman_server_metadata,
)
from deployment.trossen.pipeline import (
    resolve_action_stats,
    resolve_norm_mode,
    resolve_state_stats,
)
from scripts.replay_realman_training_episode import (
    _decode_anchor_images,
    _load_episode_metadata,
    _load_episode_rows,
)
from starVLA.action_representation import (
    REALMAN_18D_ACTION_CONTRACT,
    encode_actions,
    normalize_q01_q99_unclipped,
    select_realman_policy_actions,
    select_realman_policy_state,
)


PICKUP_SUBTASKS = {
    1: "left pickup",
    5: "right pickup",
}
SUBTASK_COLORS = {
    0: "#EEEEEE",
    1: "#B9DDF5",
    2: "#F7D9A6",
    3: "#D8C4ED",
    4: "#BFE4C5",
    5: "#F4B9B7",
    6: "#B8DED8",
    7: "#D8D8D8",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--compare-window", type=int, default=25)
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


def _load_subtask_text(dataset_root: Path) -> dict[int, str]:
    table = pq.read_table(
        dataset_root / "meta" / "subtasks.parquet",
        columns=["subtask_index", "local_subtask_text"],
    )
    return {
        int(row["subtask_index"]): str(row["local_subtask_text"])
        for row in table.to_pylist()
    }


def _load_episode_subtasks(
    dataset_root: Path,
    episode_meta: dict[str, Any],
) -> np.ndarray:
    data_path = (
        dataset_root
        / "data"
        / f"chunk-{int(episode_meta['data/chunk_index']):03d}"
        / f"file-{int(episode_meta['data/file_index']):03d}.parquet"
    )
    table = pq.read_table(
        data_path,
        columns=["frame_index", "subtask_index"],
        filters=[("episode_index", "=", int(episode_meta["episode_index"]))],
    )
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    order = np.argsort(frame_index)
    frame_index = frame_index[order]
    expected = np.arange(int(episode_meta["length"]), dtype=np.int64)
    if not np.array_equal(frame_index, expected):
        raise RuntimeError("Subtask frame indices are not contiguous from zero.")
    return np.asarray(table["subtask_index"].to_numpy(), dtype=np.int64)[order]


def _subtask_spans(values: np.ndarray) -> list[tuple[int, int, int]]:
    if values.size == 0:
        return []
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(values)]))
    return [
        (int(start), int(end), int(values[start]))
        for start, end in zip(starts, ends, strict=True)
    ]


def _masked_mae(values: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    expanded = np.broadcast_to(mask[..., None], values.shape)
    errors = np.abs(values - targets)[expanded]
    return float(errors.mean()) if errors.size else math.nan


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


def _plot(
    output_path: Path,
    *,
    episode_index: int,
    action_names: list[str],
    actual_episode: np.ndarray,
    anchors: np.ndarray,
    predictions: np.ndarray,
    compare_window: int,
    subtask_indices: np.ndarray,
    subtask_text: dict[int, str],
    raw_mae: float,
    normalized_mae: float,
    checkpoint_path: str,
) -> None:
    rows, cols = 6, 3
    figure, axes = plt.subplots(rows, cols, figsize=(24, 29), sharex=True)
    x_full = np.arange(len(actual_episode), dtype=np.int64)
    spans = _subtask_spans(subtask_indices)

    for dim, axis in enumerate(axes.flat):
        for start, end, subtask_index in spans:
            axis.axvspan(
                start,
                end,
                color=SUBTASK_COLORS.get(subtask_index, "#EEEEEE"),
                alpha=0.10,
                linewidth=0,
                zorder=0,
            )
        axis.plot(
            x_full,
            actual_episode[:, dim],
            color="black",
            linewidth=1.45,
            zorder=3,
        )
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
            axis.set_xlabel("held-out episode action index")

    pickup_indices = [
        index for index in np.unique(subtask_indices).tolist() if index in PICKUP_SUBTASKS
    ]
    pickup_label = " / ".join(PICKUP_SUBTASKS[index] for index in pickup_indices) or "no pickup"
    checkpoint_name = Path(checkpoint_path).parent.name if checkpoint_path else "unknown"
    legend = [
        Line2D([0], [0], color="black", linewidth=2, label="recorded held-out action"),
        Line2D(
            [0],
            [0],
            color="#E24A33",
            linewidth=2,
            label=f"decoded prediction (first {compare_window})",
        ),
        Line2D(
            [0],
            [0],
            color="#4C9AD4",
            linewidth=2,
            alpha=0.45,
            label="decoded full 50-step forecast",
        ),
    ]
    for index in sorted(set(int(value) for value in subtask_indices)):
        if index in subtask_text:
            legend.append(
                Line2D(
                    [0],
                    [0],
                    color=SUBTASK_COLORS.get(index, "#EEEEEE"),
                    linewidth=8,
                    alpha=0.45,
                    label=f"stage {index}: {subtask_text[index]}",
                )
            )
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.973),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        f"Held-out episode {episode_index} ({pickup_label}): actual vs decoded policy prediction\n"
        f"checkpoint={checkpoint_name} | stitched absolute MAE={raw_mae:.4f} | "
        f"mixed-normalized MAE={normalized_mae:.4f}",
        fontsize=15,
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    if args.stride <= 0 or args.compare_window <= 0:
        raise ValueError("stride and compare-window must be positive.")

    dataset_root = args.dataset_root.resolve()
    holdout_manifest_path = args.holdout_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with holdout_manifest_path.open("r", encoding="utf-8") as handle:
        holdout_manifest = json.load(handle)
    dataset_entry = holdout_manifest["datasets"][0]
    holdout_indices = {int(index) for index in dataset_entry["holdout_episode_indices"]}
    if args.episode_index not in holdout_indices:
        raise RuntimeError(
            f"Episode {args.episode_index} is not in the frozen held-out split."
        )

    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        dataset_info = json.load(handle)
    fps = float(dataset_info["fps"])
    episode_meta = _load_episode_metadata(dataset_root, args.episode_index)
    episode = _load_episode_rows(dataset_root, episode_meta)
    episode_subtasks = _load_episode_subtasks(dataset_root, episode_meta)
    subtask_text = _load_subtask_text(dataset_root)
    base_instruction = (
        args.instruction
        or (episode_meta.get("tasks") or [MAGNA_DEFAULT_INSTRUCTION])[0]
    )

    client = WebsocketClientPolicy(host=args.host, port=args.port, timeout=30)
    try:
        metadata = client.get_server_metadata()
        warnings = validate_realman_server_metadata(metadata, require_input_contract=True)
        if warnings:
            raise RuntimeError("Policy metadata mismatch: " + "; ".join(warnings))
        action_horizon = int(metadata["action_horizon"])
        action_dim = int(metadata["action_dim"])
        if action_horizon != REALMAN_18D_ACTION_CONTRACT.action_horizon:
            raise RuntimeError(
                f"Expected H={REALMAN_18D_ACTION_CONTRACT.action_horizon}, "
                f"server reports {action_horizon}."
            )
        if action_dim != REALMAN_18D_ACTION_CONTRACT.action_dim:
            raise RuntimeError(
                f"Expected {REALMAN_18D_ACTION_CONTRACT.action_dim} actions, "
                f"server reports {action_dim}."
            )
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
        if action_norm_mode != "q01_q99_unclipped":
            raise RuntimeError(
                f"Expected q01_q99_unclipped action normalization, got {action_norm_mode!r}."
            )

        actual_actions = select_realman_policy_actions(episode["source_action"])
        input_states = select_realman_policy_state(episode["source_state"])
        predictions: list[np.ndarray] = []
        mixed_predictions: list[np.ndarray] = []
        normalized_predictions: list[np.ndarray] = []
        instructions: list[str] = []
        latencies_ms: list[float] = []

        for request_index, anchor in enumerate(anchors, start=1):
            subtask_index = int(episode_subtasks[anchor])
            local_text = subtask_text.get(subtask_index, "")
            instruction = base_instruction
            if local_text and subtask_index not in {0, 7}:
                instruction = f"{base_instruction} | {local_text}"
            instructions.append(instruction)
            observation = {
                # Deployment payloads use the 19-D control state. Some captures
                # append two diagnostic torque channels that are intentionally
                # excluded before the shared 18-D policy-state selection.
                "source.observation.state": episode["source_state"][anchor, :19],
                **{
                    f"observation.images.{camera}": anchor_images[int(anchor)][camera]
                    for camera in REALMAN_CAMERA_ORDER
                },
            }
            payload = build_policy_payload(
                observation,
                instruction=instruction,
                image_size=image_size,
            )
            if state_norm_mode != "q01_q99_unclipped":
                raise RuntimeError(
                    "Expected q01_q99_unclipped state normalization, got "
                    f"{state_norm_mode!r}."
                )
            payload["state"] = np.ascontiguousarray(
                normalize_q01_q99_unclipped(
                    input_states[anchor],
                    state_stats,
                )[None, None, :],
                dtype=np.float32,
            )
            validate_realman_policy_payload(payload, metadata)
            started = time.perf_counter()
            response = client.infer(payload)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            if not response.get("ok", False):
                raise RuntimeError(f"Inference failed at anchor {anchor}: {response}")
            normalized = np.asarray(
                response["data"]["normalized_actions"],
                dtype=np.float32,
            )
            if normalized.shape != (1, action_horizon, action_dim):
                raise RuntimeError(
                    f"Unexpected prediction shape {normalized.shape}; expected "
                    f"{(1, action_horizon, action_dim)}."
                )
            normalized = normalized[0]
            mixed = realman_continuous_unnormalize(
                normalized,
                action_stats,
                mode=action_norm_mode,
            )
            absolute = realman_policy_actions_to_absolute(
                mixed,
                input_states[anchor],
                action_type=str(metadata["action_type"]),
            )
            normalized_predictions.append(normalized)
            mixed_predictions.append(mixed)
            predictions.append(absolute)
            print(
                f"[{request_index:02d}/{len(anchors):02d}] episode={args.episode_index} "
                f"anchor={int(anchor):4d} subtask={subtask_index} "
                f"latency={latencies_ms[-1]:7.1f} ms",
                flush=True,
            )
    finally:
        client.close()

    predictions_array = np.stack(predictions)
    mixed_predictions_array = np.stack(mixed_predictions)
    normalized_predictions_array = np.stack(normalized_predictions)
    ground_truth_chunks = np.stack(
        [actual_actions[anchor : anchor + action_horizon] for anchor in anchors]
    )
    mixed_ground_truth = np.stack(
        [
            encode_actions(
                ground_truth_chunks[index],
                input_states[anchor],
                REALMAN_18D_ACTION_CONTRACT,
            )
            for index, anchor in enumerate(anchors)
        ]
    )
    normalized_ground_truth = normalize_q01_q99_unclipped(
        mixed_ground_truth,
        action_stats,
    )
    valid_chunks = np.stack(
        [episode["valid_state"][anchor : anchor + action_horizon] for anchor in anchors]
    )

    compare_window = min(args.compare_window, action_horizon)
    used_valid = valid_chunks[:, :compare_window]
    raw_mae = _masked_mae(
        predictions_array[:, :compare_window],
        ground_truth_chunks[:, :compare_window],
        used_valid,
    )
    normalized_mae = _masked_mae(
        normalized_predictions_array[:, :compare_window],
        normalized_ground_truth[:, :compare_window],
        used_valid,
    )
    arm_dims = np.asarray(tuple(range(7)) + tuple(range(8, 15)), dtype=np.int64)
    arm_raw_mae = _masked_mae(
        predictions_array[:, :compare_window, arm_dims],
        ground_truth_chunks[:, :compare_window, arm_dims],
        used_valid,
    )
    gripper_metrics: dict[str, Any] = {}
    flat_valid = used_valid.reshape(-1)
    for name in ("left_gripper", "right_gripper"):
        dim = action_names.index(name)
        predicted = (
            predictions_array[:, :compare_window, dim].reshape(-1)[flat_valid] >= 0.5
        )
        target = (
            ground_truth_chunks[:, :compare_window, dim].reshape(-1)[flat_valid] >= 0.5
        )
        gripper_metrics[name] = {
            "binary_accuracy": float(np.mean(predicted == target)),
            "predicted_open_fraction": float(np.mean(predicted)),
            "target_open_fraction": float(np.mean(target)),
        }

    metrics = {
        "provenance": {
            "dataset_root": str(dataset_root),
            "dataset_info_sha256": _sha256(info_path),
            "manifest": str(holdout_manifest_path),
            "manifest_sha256": _sha256(holdout_manifest_path),
            "episode_index": args.episode_index,
            "episode_is_holdout": True,
            "episode_length": len(episode["frame_index"]),
            "base_instruction": base_instruction,
            "server_checkpoint_path": metadata.get("checkpoint_path"),
            "server_run_id": metadata.get("run_id"),
            "representation_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
        },
        "replay": {
            "fps": fps,
            "stride": args.stride,
            "compare_window": compare_window,
            "model_horizon": action_horizon,
            "anchor_count": len(anchors),
            "anchors": anchors,
            "anchor_subtask_indices": episode_subtasks[anchors],
            "instructions": instructions,
            "latency_ms_mean": float(np.mean(latencies_ms)),
            "latency_ms_p95": float(np.quantile(latencies_ms, 0.95)),
        },
        "metrics": {
            "stitched_absolute_mae": raw_mae,
            "stitched_arm_absolute_mae_rad": arm_raw_mae,
            "stitched_mixed_normalized_mae": normalized_mae,
            "grippers": gripper_metrics,
        },
    }

    stem = f"heldout_episode_{args.episode_index:06d}"
    arrays_path = output_dir / f"{stem}_replay_arrays.npz"
    np.savez_compressed(
        arrays_path,
        anchors=anchors,
        predictions_absolute=predictions_array,
        predictions_mixed=mixed_predictions_array,
        predictions_normalized=normalized_predictions_array,
        ground_truth_absolute=ground_truth_chunks,
        ground_truth_mixed=mixed_ground_truth,
        ground_truth_normalized=normalized_ground_truth,
        actual_episode=actual_actions,
        input_state_episode=input_states,
        valid_state_episode=episode["valid_state"],
        subtask_index_episode=episode_subtasks,
        action_names=np.asarray(action_names),
    )
    metrics_path = output_dir / f"{stem}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(metrics), handle, indent=2)
    plot_path = output_dir / f"{stem}_all_joints.png"
    _plot(
        plot_path,
        episode_index=args.episode_index,
        action_names=action_names,
        actual_episode=actual_actions,
        anchors=anchors,
        predictions=predictions_array,
        compare_window=compare_window,
        subtask_indices=episode_subtasks,
        subtask_text=subtask_text,
        raw_mae=raw_mae,
        normalized_mae=normalized_mae,
        checkpoint_path=str(metadata.get("checkpoint_path") or ""),
    )

    print(json.dumps(_json_value(metrics), indent=2), flush=True)
    print(f"plot={plot_path}", flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"arrays={arrays_path}", flush=True)


if __name__ == "__main__":
    main()
