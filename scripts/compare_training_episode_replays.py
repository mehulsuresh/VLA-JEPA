#!/usr/bin/env python3
"""Compare two matched RealMan training-episode replay artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SOURCE_ACTION_INDICES = np.asarray(tuple(range(16)) + (19, 20), dtype=np.int64)
GROUPS = {
    "all": np.arange(18, dtype=np.int64),
    "arms": np.asarray(tuple(range(7)) + tuple(range(8, 15)), dtype=np.int64),
    "grippers": np.asarray((7, 15), dtype=np.int64),
    "head": np.asarray((16, 17), dtype=np.int64),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-arrays", type=Path, required=True)
    parser.add_argument("--pi-arrays", type=Path, required=True)
    parser.add_argument("--train-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=120)
    parser.add_argument("--compare-window", type=int, default=25)
    parser.add_argument("--vla-label", default="VLA-JEPA step 52,500")
    parser.add_argument("--pi-label", default="pi0.5 Magna")
    return parser.parse_args()


def _normalize(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(values, dtype=np.float32)
    varying = minimum != maximum
    result[..., varying] = (
        np.float32(2.0)
        * (values[..., varying] - minimum[varying])
        / (maximum[varying] - minimum[varying])
        - np.float32(1.0)
    )
    return result


def _masked_mae(
    values: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    dimensions: np.ndarray | None = None,
) -> float:
    if dimensions is not None:
        values = values[..., dimensions]
        targets = targets[..., dimensions]
    expanded = np.broadcast_to(mask[..., None], values.shape)
    errors = np.abs(values - targets)[expanded]
    return float(errors.mean()) if errors.size else math.nan


def _per_dim_mae(values: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = np.broadcast_to(mask[..., None], values.shape)
    numerator = (np.abs(values - targets) * expanded).sum(axis=(0, 1))
    denominator = expanded.sum(axis=(0, 1))
    return np.divide(
        numerator,
        denominator,
        out=np.full(values.shape[-1], np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _require_matched(vla: dict[str, np.ndarray], pi: dict[str, np.ndarray]) -> None:
    for key in (
        "anchors",
        "ground_truth_chunks",
        "actual_episode",
        "input_state_episode",
        "valid_state_episode",
        "action_names",
    ):
        if vla[key].shape != pi[key].shape:
            raise RuntimeError(
                f"matched replay contract failed for {key}: {vla[key].shape} != {pi[key].shape}"
            )
        if np.issubdtype(vla[key].dtype, np.number):
            if not np.allclose(vla[key], pi[key], rtol=0.0, atol=1e-6):
                difference = float(np.max(np.abs(vla[key] - pi[key])))
                raise RuntimeError(
                    f"matched replay contract failed for {key}: max diff {difference}"
                )
        elif not np.array_equal(vla[key], pi[key]):
            raise RuntimeError(f"matched replay contract failed for {key}")


def _model_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    hold: np.ndarray,
    mask: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    compare_window: int,
    action_names: list[str],
) -> dict[str, Any]:
    prediction_norm = _normalize(prediction, minimum, maximum)
    target_norm = _normalize(target, minimum, maximum)
    hold_norm = _normalize(hold, minimum, maximum)
    metrics: dict[str, Any] = {"groups": {}}
    for group_name, dimensions in GROUPS.items():
        group: dict[str, float] = {}
        for horizon_name, horizon in (("stitched", compare_window), ("full", 50)):
            selected_mask = mask[:, :horizon]
            raw_mae = _masked_mae(
                prediction[:, :horizon], target[:, :horizon], selected_mask, dimensions
            )
            normalized_mae = _masked_mae(
                prediction_norm[:, :horizon],
                target_norm[:, :horizon],
                selected_mask,
                dimensions,
            )
            hold_raw_mae = _masked_mae(
                hold[:, :horizon], target[:, :horizon], selected_mask, dimensions
            )
            hold_normalized_mae = _masked_mae(
                hold_norm[:, :horizon],
                target_norm[:, :horizon],
                selected_mask,
                dimensions,
            )
            group[f"{horizon_name}_raw_mae"] = raw_mae
            group[f"{horizon_name}_common_normalized_mae"] = normalized_mae
            group[f"{horizon_name}_hold_raw_mae"] = hold_raw_mae
            group[f"{horizon_name}_hold_common_normalized_mae"] = hold_normalized_mae
            group[f"{horizon_name}_vs_hold_ratio"] = (
                normalized_mae / hold_normalized_mae
            )
        metrics["groups"][group_name] = group

    metrics["stitched_per_dimension_raw_mae"] = dict(
        zip(
            action_names,
            _per_dim_mae(
                prediction[:, :compare_window],
                target[:, :compare_window],
                mask[:, :compare_window],
            ),
            strict=True,
        )
    )
    metrics["stitched_per_dimension_common_normalized_mae"] = dict(
        zip(
            action_names,
            _per_dim_mae(
                prediction_norm[:, :compare_window],
                target_norm[:, :compare_window],
                mask[:, :compare_window],
            ),
            strict=True,
        )
    )
    metrics["cumulative_common_normalized_mae_by_horizon"] = [
        _masked_mae(
            prediction_norm[:, :horizon],
            target_norm[:, :horizon],
            mask[:, :horizon],
        )
        for horizon in range(1, 51)
    ]
    metrics["cumulative_hold_common_normalized_mae_by_horizon"] = [
        _masked_mae(
            hold_norm[:, :horizon],
            target_norm[:, :horizon],
            mask[:, :horizon],
        )
        for horizon in range(1, 51)
    ]
    metrics["grippers"] = {}
    flat_mask = mask[:, :compare_window].reshape(-1)
    for name in ("left_gripper", "right_gripper"):
        dim = action_names.index(name)
        predicted_open = prediction[:, :compare_window, dim].reshape(-1)[flat_mask] >= 0.5
        target_open = target[:, :compare_window, dim].reshape(-1)[flat_mask] >= 0.5
        metrics["grippers"][name] = {
            "binary_accuracy": float(np.mean(predicted_open == target_open)),
            "predicted_open_fraction": float(predicted_open.mean()),
            "target_open_fraction": float(target_open.mean()),
        }
    return metrics


def _plot_overlay(
    output_path: Path,
    episode_index: int,
    action_names: list[str],
    actual: np.ndarray,
    anchors: np.ndarray,
    vla_prediction: np.ndarray,
    pi_prediction: np.ndarray,
    compare_window: int,
    vla_label: str,
    pi_label: str,
    vla_mae: float,
    pi_mae: float,
) -> None:
    rows, cols = 6, 3
    figure, axes = plt.subplots(rows, cols, figsize=(24, 29), sharex=True)
    x_full = np.arange(actual.shape[0])
    colors = {"vla": "#E24A33", "pi": "#3A78C2"}
    for dim, axis in enumerate(axes.flat):
        axis.plot(x_full, actual[:, dim], color="black", linewidth=1.5, zorder=4)
        for anchor_index, anchor in enumerate(anchors):
            horizon_x = int(anchor) + np.arange(50)
            axis.plot(
                horizon_x,
                vla_prediction[anchor_index, :, dim],
                color=colors["vla"],
                alpha=0.07,
                linewidth=0.65,
                zorder=1,
            )
            axis.plot(
                horizon_x,
                pi_prediction[anchor_index, :, dim],
                color=colors["pi"],
                alpha=0.07,
                linewidth=0.65,
                zorder=1,
            )
            axis.plot(
                horizon_x[:compare_window],
                vla_prediction[anchor_index, :compare_window, dim],
                color=colors["vla"],
                alpha=0.78,
                linewidth=1.0,
                zorder=2,
            )
            axis.plot(
                horizon_x[:compare_window],
                pi_prediction[anchor_index, :compare_window, dim],
                color=colors["pi"],
                alpha=0.78,
                linewidth=1.0,
                zorder=3,
            )
        axis.set_title(action_names[dim], fontsize=11)
        axis.grid(alpha=0.2)
        if "gripper" in action_names[dim]:
            axis.set_ylim(-0.08, 1.08)
        if dim % cols == 0:
            axis.set_ylabel("absolute target")
        if dim // cols == rows - 1:
            axis.set_xlabel("episode action index")

    figure.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=2, label="recorded training action"),
            Line2D([0], [0], color=colors["vla"], linewidth=2, label=vla_label),
            Line2D([0], [0], color=colors["pi"], linewidth=2, label=pi_label),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        f"Training episode {episode_index}: matched replay every 25 actions "
        f"(solid first {compare_window}, faint full 50)\n"
        f"common-normalized stitched MAE: {vla_label}={vla_mae:.4f} | "
        f"{pi_label}={pi_mae:.4f}",
        fontsize=15,
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_error_bars(
    output_path: Path,
    action_names: list[str],
    vla: np.ndarray,
    pi: np.ndarray,
    hold: np.ndarray,
    vla_label: str,
    pi_label: str,
) -> None:
    x = np.arange(len(action_names))
    width = 0.27
    figure, axis = plt.subplots(figsize=(19, 7))
    axis.bar(x - width, vla, width, label=vla_label, color="#E24A33")
    axis.bar(x, pi, width, label=pi_label, color="#3A78C2")
    axis.bar(x + width, hold, width, label="hold current state", color="#A6A6A6")
    axis.set_ylabel("common-normalized MAE (lower is better)")
    axis.set_title("Training episode 120: stitched first-25 error by action dimension")
    axis.set_xticks(x)
    axis.set_xticklabels(action_names, rotation=58, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_horizon(
    output_path: Path,
    vla: np.ndarray,
    pi: np.ndarray,
    hold: np.ndarray,
    vla_label: str,
    pi_label: str,
) -> None:
    horizons = np.arange(1, 51)
    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.plot(horizons, vla, color="#E24A33", linewidth=2.3, label=vla_label)
    axis.plot(horizons, pi, color="#3A78C2", linewidth=2.3, label=pi_label)
    axis.plot(horizons, hold, color="#777777", linewidth=2.0, label="hold current state")
    axis.axvline(25, color="#AAAAAA", linestyle="--", linewidth=1)
    axis.set_xlabel("forecast horizon (actions)")
    axis.set_ylabel("cumulative common-normalized MAE")
    axis.set_title("Training episode 120: prediction error by forecast horizon")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vla = _load(args.vla_arrays.expanduser().resolve())
    pi = _load(args.pi_arrays.expanduser().resolve())
    _require_matched(vla, pi)
    action_names = [str(name) for name in vla["action_names"].tolist()]
    stats = json.loads(args.train_stats.expanduser().read_text())["source.action"]
    minimum = np.asarray(stats["min"], dtype=np.float32)[SOURCE_ACTION_INDICES]
    maximum = np.asarray(stats["max"], dtype=np.float32)[SOURCE_ACTION_INDICES]
    target = np.asarray(vla["ground_truth_chunks"], dtype=np.float32)
    mask = np.stack(
        [vla["valid_state_episode"][anchor : anchor + 50] for anchor in vla["anchors"]]
    ).astype(bool)
    hold = np.repeat(vla["input_state_episode"][vla["anchors"], None, :], 50, axis=1)
    vla_metrics = _model_metrics(
        np.asarray(vla["predictions"], dtype=np.float32),
        target,
        hold,
        mask,
        minimum,
        maximum,
        args.compare_window,
        action_names,
    )
    pi_metrics = _model_metrics(
        np.asarray(pi["predictions"], dtype=np.float32),
        target,
        hold,
        mask,
        minimum,
        maximum,
        args.compare_window,
        action_names,
    )
    vla_all = vla_metrics["groups"]["all"]
    pi_all = pi_metrics["groups"]["all"]
    comparison = {
        "contract": {
            "episode_index": args.episode_index,
            "episode_length": int(vla["actual_episode"].shape[0]),
            "anchors": vla["anchors"],
            "stride": int(vla["anchors"][1] - vla["anchors"][0]),
            "compare_window": args.compare_window,
            "full_horizon": 50,
            "same_observations_and_targets_verified": True,
            "normalization": "shared VLA training-split source.action min/max",
            "training_episode_for_both_models": True,
            "vla_arrays": str(args.vla_arrays.expanduser().resolve()),
            "pi_arrays": str(args.pi_arrays.expanduser().resolve()),
            "train_stats": str(args.train_stats.expanduser().resolve()),
        },
        "models": {args.vla_label: vla_metrics, args.pi_label: pi_metrics},
        "comparison": {
            "pi_stitched_raw_mae_reduction_fraction_vs_vla": (
                1.0 - pi_all["stitched_raw_mae"] / vla_all["stitched_raw_mae"]
            ),
            "pi_stitched_common_normalized_mae_reduction_fraction_vs_vla": (
                1.0
                - pi_all["stitched_common_normalized_mae"]
                / vla_all["stitched_common_normalized_mae"]
            ),
            "vla_to_pi_stitched_raw_mae_ratio": (
                vla_all["stitched_raw_mae"] / pi_all["stitched_raw_mae"]
            ),
            "vla_to_pi_stitched_common_normalized_mae_ratio": (
                vla_all["stitched_common_normalized_mae"]
                / pi_all["stitched_common_normalized_mae"]
            ),
            "pi_full_raw_mae_reduction_fraction_vs_vla": (
                1.0 - pi_all["full_raw_mae"] / vla_all["full_raw_mae"]
            ),
            "pi_full_common_normalized_mae_reduction_fraction_vs_vla": (
                1.0
                - pi_all["full_common_normalized_mae"]
                / vla_all["full_common_normalized_mae"]
            ),
            "vla_to_pi_full_raw_mae_ratio": (
                vla_all["full_raw_mae"] / pi_all["full_raw_mae"]
            ),
            "vla_to_pi_full_common_normalized_mae_ratio": (
                vla_all["full_common_normalized_mae"]
                / pi_all["full_common_normalized_mae"]
            ),
        },
    }
    report_path = output_dir / "episode_000120_comparison.json"
    report_path.write_text(json.dumps(_json_value(comparison), indent=2))

    target_norm = _normalize(target, minimum, maximum)
    hold_norm = _normalize(hold, minimum, maximum)
    used_mask = mask[:, : args.compare_window]
    vla_per_dim = _per_dim_mae(
        _normalize(vla["predictions"], minimum, maximum)[:, : args.compare_window],
        target_norm[:, : args.compare_window],
        used_mask,
    )
    pi_per_dim = _per_dim_mae(
        _normalize(pi["predictions"], minimum, maximum)[:, : args.compare_window],
        target_norm[:, : args.compare_window],
        used_mask,
    )
    hold_per_dim = _per_dim_mae(
        hold_norm[:, : args.compare_window],
        target_norm[:, : args.compare_window],
        used_mask,
    )

    overlay_path = output_dir / "episode_000120_vla_vs_pi_all_joints.png"
    _plot_overlay(
        overlay_path,
        args.episode_index,
        action_names,
        vla["actual_episode"],
        vla["anchors"],
        vla["predictions"],
        pi["predictions"],
        args.compare_window,
        args.vla_label,
        args.pi_label,
        vla_all["stitched_common_normalized_mae"],
        pi_all["stitched_common_normalized_mae"],
    )
    bars_path = output_dir / "episode_000120_vla_vs_pi_error_by_joint.png"
    _plot_error_bars(
        bars_path,
        action_names,
        vla_per_dim,
        pi_per_dim,
        hold_per_dim,
        args.vla_label,
        args.pi_label,
    )
    horizon_path = output_dir / "episode_000120_vla_vs_pi_mae_by_horizon.png"
    _plot_horizon(
        horizon_path,
        np.asarray(vla_metrics["cumulative_common_normalized_mae_by_horizon"]),
        np.asarray(pi_metrics["cumulative_common_normalized_mae_by_horizon"]),
        np.asarray(vla_metrics["cumulative_hold_common_normalized_mae_by_horizon"]),
        args.vla_label,
        args.pi_label,
    )
    print(json.dumps(_json_value(comparison["comparison"]), indent=2))
    print(f"overlay={overlay_path}")
    print(f"error_by_joint={bars_path}")
    print(f"mae_by_horizon={horizon_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
