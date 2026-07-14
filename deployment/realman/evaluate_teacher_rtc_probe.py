"""Read-only paired probe of expert-prefix RTC on recorded Realman data.

This measures the condition used by training: rows ``[0, prefix_len)`` come
from the normalized demonstration action window and are fixed by RTC prefix
conditioning, while MAE is computed only on the predicted suffix.  It is
deliberately kept separate from realistic RTC replay, where prefixes come from
the policy's own previous plan.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from deployment.realman.audit_checkpoint_offline import (
    LocalPredictor,
    _json_safe,
    _metadata_stats,
    _resolve_run_file,
    _stable_seed,
    assert_checkpoint_dataset_stats_match,
    enumerate_episodes,
    merged_dataset_modality_stats,
    realman_action_groups,
    validate_dataset_camera_order,
)
from deployment.realman.evaluate_rtc_replay import (
    _episode_ref_for_request,
    _load_replay_observation,
    _predict_local_plan,
    fresh_plan_inference_seed,
    validate_rtc_inference_contract,
    validate_unit_step_action_offsets,
)
from deployment.realman.pipeline import realman_continuous_unnormalize


class RegionMetrics:
    def __init__(self, *, rows: Sequence[int], arm_dimensions: Sequence[int]) -> None:
        self.rows = np.asarray(tuple(rows), dtype=np.int64)
        self.arm_dimensions = np.asarray(tuple(arm_dimensions), dtype=np.int64)
        self.raw_arm_abs_sum = 0.0
        self.raw_arm_count = 0
        self.normalized_arm_abs_sum = 0.0
        self.normalized_arm_count = 0
        self.normalized_all_abs_sum = 0.0
        self.normalized_all_count = 0

    def update(
        self,
        *,
        prediction_normalized: np.ndarray,
        target_normalized: np.ndarray,
        prediction_raw: np.ndarray,
        target_raw: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        rows = self.rows
        arms = self.arm_dimensions
        region_valid = valid_mask[rows]
        arm_valid = region_valid[:, arms]

        raw_arm_error = np.abs(
            prediction_raw[np.ix_(rows, arms)] - target_raw[np.ix_(rows, arms)]
        )
        normalized_arm_error = np.abs(
            prediction_normalized[np.ix_(rows, arms)]
            - target_normalized[np.ix_(rows, arms)]
        )
        normalized_all_error = np.abs(
            prediction_normalized[rows] - target_normalized[rows]
        )

        self.raw_arm_abs_sum += float(raw_arm_error[arm_valid].sum())
        self.raw_arm_count += int(arm_valid.sum())
        self.normalized_arm_abs_sum += float(normalized_arm_error[arm_valid].sum())
        self.normalized_arm_count += int(arm_valid.sum())
        self.normalized_all_abs_sum += float(
            normalized_all_error[region_valid].sum()
        )
        self.normalized_all_count += int(region_valid.sum())

    def finalize(self) -> dict[str, Any]:
        return {
            "rows": [int(value) for value in self.rows],
            "raw_arm_mae_rad": (
                self.raw_arm_abs_sum / self.raw_arm_count
                if self.raw_arm_count
                else None
            ),
            "raw_arm_elements": self.raw_arm_count,
            "normalized_arm_mae": (
                self.normalized_arm_abs_sum / self.normalized_arm_count
                if self.normalized_arm_count
                else None
            ),
            "normalized_all_action_mae": (
                self.normalized_all_abs_sum / self.normalized_all_count
                if self.normalized_all_count
                else None
            ),
            "normalized_all_action_elements": self.normalized_all_count,
        }


def _new_regions(*, prefix_len: int, horizon: int, arm_dimensions: Sequence[int]):
    return {
        "h0_reference": RegionMetrics(rows=(0,), arm_dimensions=arm_dimensions),
        "first_predicted_row": RegionMetrics(
            rows=(prefix_len,), arm_dimensions=arm_dimensions
        ),
        "early_suffix": RegionMetrics(
            rows=range(prefix_len, min(10, horizon)),
            arm_dimensions=arm_dimensions,
        ),
        "full_supervised_suffix": RegionMetrics(
            rows=range(prefix_len, horizon), arm_dimensions=arm_dimensions
        ),
    }


def _finalize_regions(regions: Mapping[str, RegionMetrics]) -> dict[str, Any]:
    return {name: metric.finalize() for name, metric in regions.items()}


def _update_regions_from_normalized_plan(
    regions: Mapping[str, RegionMetrics],
    *,
    prediction_normalized: np.ndarray,
    target_normalized: np.ndarray,
    action_stats: Mapping[str, Any],
    action_mode: str,
    target_raw: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Score one plan using the live RealMan training-affine inverse."""

    prediction_raw = realman_continuous_unnormalize(
        prediction_normalized,
        action_stats,
        mode=action_mode,
    )
    for metric in regions.values():
        metric.update(
            prediction_normalized=prediction_normalized,
            target_normalized=target_normalized,
            prediction_raw=prediction_raw,
            target_raw=target_raw,
            valid_mask=valid_mask,
        )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    instruction = str(args.instruction).strip()
    if not instruction:
        raise ValueError("--instruction must be non-empty.")
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else _resolve_run_file(checkpoint_path, "config.yaml")
    )
    cfg = OmegaConf.load(config_path)
    if args.dataset_root:
        cfg.datasets.vla_data.data_root_dir = str(
            Path(args.dataset_root).expanduser().resolve()
        )

    horizon = int(cfg.framework.action_model.action_horizon)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="val",
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=int(args.seed),
        action_horizon=horizon,
        video_horizon=int(cfg.framework.vj2_model.num_frames),
        video_frame_stride=int(cfg.datasets.vla_data.get("video_frame_stride", 1)),
    )
    episode_ref = _episode_ref_for_request(
        enumerate_episodes(dataset.datasets, seed=int(args.seed)),
        dataset_name=args.dataset_name,
        episode_id=int(args.episode_id),
    )
    stop_frame = (
        int(args.stop_frame)
        if args.stop_frame is not None
        else int(episode_ref.length)
    )
    if not (0 <= int(args.start_frame) < stop_frame <= int(episode_ref.length)):
        raise ValueError(
            f"Invalid frame interval [{args.start_frame}, {stop_frame}) for episode "
            f"length {episode_ref.length}."
        )
    if int(args.frame_stride) <= 0:
        raise ValueError("--frame-stride must be positive.")
    frame_indices = tuple(
        range(int(args.start_frame), stop_frame, int(args.frame_stride))
    )
    if not frame_indices:
        raise ValueError("Frame selection is empty.")

    single_dataset = dataset.datasets[episode_ref.dataset_index]
    validate_unit_step_action_offsets(single_dataset, horizon=horizon)
    predictor = LocalPredictor(
        checkpoint_path=checkpoint_path, device_name=str(args.device)
    )
    try:
        metadata = predictor.metadata
        action_stats, state_stats, action_mode, state_mode, _ = _metadata_stats(
            metadata
        )
        action_dim = int(metadata["action_dim"])
        state_dim = int(metadata["state_dim"])
        if int(metadata["action_horizon"]) != horizon:
            raise ValueError("Checkpoint and config action horizons differ.")
        prefix_len = int(args.prefix_len)
        validate_rtc_inference_contract(
            metadata, requested_prefix_lengths=(prefix_len,)
        )
        if not (0 < prefix_len < horizon):
            raise ValueError(f"prefix_len must be in [1,{horizon - 1}].")
        validate_dataset_camera_order(single_dataset, metadata)
        assert_checkpoint_dataset_stats_match(
            action_stats,
            merged_dataset_modality_stats(dataset, single_dataset, modality="action"),
            modality=f"{episode_ref.dataset_name} action",
        )
        assert_checkpoint_dataset_stats_match(
            state_stats,
            merged_dataset_modality_stats(dataset, single_dataset, modality="state"),
            modality=f"{episode_ref.dataset_name} state",
        )

        groups = realman_action_groups(action_dim)
        arm_dimensions = tuple(groups["arm"])
        required_dimensions = tuple(sorted((*groups["arm"], *groups["gripper"])))
        clip_state = state_mode == "q99"
        video_target_shift_steps = int(
            cfg.datasets.vla_data.get("video_target_shift_steps", 0)
        )
        fresh_regions = _new_regions(
            prefix_len=prefix_len,
            horizon=horizon,
            arm_dimensions=arm_dimensions,
        )
        teacher_regions = _new_regions(
            prefix_len=prefix_len,
            horizon=horizon,
            arm_dimensions=arm_dimensions,
        )
        prefix_abs_sum = 0.0
        prefix_elements = 0
        prefix_max_abs = 0.0
        fresh_latencies: list[float] = []
        teacher_latencies: list[float] = []

        for frame_index in frame_indices:
            sample_seed = _stable_seed(
                int(args.seed),
                episode_ref.dataset_name,
                episode_ref.episode_id,
                frame_index,
            )
            observation = _load_replay_observation(
                mixture_dataset=dataset,
                single_dataset=single_dataset,
                episode_id=episode_ref.episode_id,
                frame_index=frame_index,
                prompt_seed=sample_seed,
                video_target_shift_steps=video_target_shift_steps,
                state_dim=state_dim,
                horizon=horizon,
                action_dim=action_dim,
                state_stats=state_stats,
                state_mode=state_mode,
                clip_state=clip_state,
                metadata=metadata,
                required_action_dimensions=required_dimensions,
            )
            if not bool(
                np.all(
                    observation.target_action_valid_mask[
                        :prefix_len, np.asarray(required_dimensions, dtype=np.int64)
                    ]
                )
            ):
                raise ValueError(
                    f"Expert prefix is invalid at frame {frame_index}; choose an earlier stop."
                )
            inference_seed = fresh_plan_inference_seed(
                base_seed=int(args.seed),
                dataset_name=episode_ref.dataset_name,
                episode_id=episode_ref.episode_id,
                frame_index=frame_index,
            )
            fresh, fresh_latency = _predict_local_plan(
                predictor,
                qwen_frames=observation.qwen_frames,
                instruction=instruction,
                state_normalized=observation.state_normalized,
                seed=inference_seed,
                horizon=horizon,
                action_dim=action_dim,
            )
            teacher, teacher_latency = _predict_local_plan(
                predictor,
                qwen_frames=observation.qwen_frames,
                instruction=instruction,
                state_normalized=observation.state_normalized,
                seed=inference_seed,
                horizon=horizon,
                action_dim=action_dim,
                prev_actions=observation.target_action_normalized,
                prefix_len=prefix_len,
            )
            fresh_latencies.append(fresh_latency)
            teacher_latencies.append(teacher_latency)

            prefix_valid = observation.target_action_valid_mask[:prefix_len]
            prefix_error = np.abs(
                teacher[:prefix_len] - observation.target_action_normalized[:prefix_len]
            )
            prefix_abs_sum += float(prefix_error[prefix_valid].sum())
            prefix_elements += int(prefix_valid.sum())
            if prefix_elements:
                prefix_max_abs = max(
                    prefix_max_abs, float(prefix_error[prefix_valid].max())
                )

            for regions, prediction_normalized in (
                (fresh_regions, fresh),
                (teacher_regions, teacher),
            ):
                _update_regions_from_normalized_plan(
                    regions,
                    prediction_normalized=prediction_normalized,
                    target_normalized=observation.target_action_normalized,
                    action_stats=action_stats,
                    action_mode=action_mode,
                    target_raw=observation.target_action_raw_window,
                    valid_mask=observation.target_action_valid_mask,
                )

        fresh_final = _finalize_regions(fresh_regions)
        teacher_final = _finalize_regions(teacher_regions)
        ratios: dict[str, Any] = {}
        for region_name in fresh_final:
            fresh_mae = fresh_final[region_name]["raw_arm_mae_rad"]
            teacher_mae = teacher_final[region_name]["raw_arm_mae_rad"]
            ratios[region_name] = {
                "teacher_div_fresh_raw_arm_mae": (
                    teacher_mae / fresh_mae if fresh_mae else None
                ),
                "teacher_percent_change_vs_fresh": (
                    100.0 * (teacher_mae / fresh_mae - 1.0)
                    if fresh_mae
                    else None
                ),
            }

        report = {
            "schema_version": 2,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "evaluation_kind": "read_only_expert_prefix_rtc_probe",
            "checkpoint": str(predictor.resolved_checkpoint),
            "safety": {
                "robot_commands_sent": False,
                "policy_server_contacted": False,
                "training_process_contacted": False,
                "dataset_writes": False,
                "checkpoint_writes": False,
            },
            "dataset": {
                "root": str(Path(cfg.datasets.vla_data.data_root_dir).resolve()),
                "name": episode_ref.dataset_name,
                "episode_id": episode_ref.episode_id,
                "episode_length": episode_ref.length,
                "frames": list(frame_indices),
                "frame_count": len(frame_indices),
                "frame_stride": int(args.frame_stride),
            },
            "instruction": instruction,
            "prefix_len": prefix_len,
            "metric_semantics": {
                "expert_prefix_rows": [0, prefix_len],
                "prefix_rows_scored": False,
                "first_predicted_row": prefix_len,
                "same_noise_seed_for_fresh_and_teacher": True,
                "raw_arm_units": "radians",
                "normalized_policy_outputs_clipped_before_metrics": False,
                "raw_action_inverse": (
                    "unclipped_training_affine_matches_live_realman_rollout"
                ),
            },
            "expert_prefix_copy": {
                "normalized_mae": prefix_abs_sum / prefix_elements,
                "normalized_max_abs": prefix_max_abs,
                "elements": prefix_elements,
            },
            "fresh_no_rtc": fresh_final,
            "expert_prefix_rtc": teacher_final,
            "paired_comparison": ratios,
            "latency_ms": {
                "fresh_mean": float(np.mean(fresh_latencies)),
                "teacher_mean": float(np.mean(teacher_latencies)),
            },
        }
        return _json_safe(report)
    finally:
        predictor.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--episode-id", required=True, type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--prefix-len", type=int, default=3)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_probe(args)
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
