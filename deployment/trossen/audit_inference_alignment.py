from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from deployment.model_server.checkpoint_utils import resolve_policy_checkpoint
from deployment.trossen.pipeline import continuous_normalize, continuous_unnormalize, resize_for_training
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import baseframework


DEFAULT_CKPT = Path(
    "/home/mehul/work/vjepa/checkpoints/"
    "robot_ft_trossen_vjepa21_small_a100x4_weekend_20260404_5ep/final_model"
)
DEFAULT_CONFIG = Path(
    "/home/mehul/work/vjepa/checkpoints/"
    "robot_ft_trossen_vjepa21_small_a100x4_weekend_20260404_5ep/config.yaml"
)
DEFAULT_DATASET_ROOT = Path("/home/mehul/work/reward_model_small/Subtask_dataset/subtask_labelled_combined")
DEFAULT_REPORT = Path("/home/mehul/work/vjepa/checkpoints/trossen_alignment_audit_20260408.json")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_mae(a: Any, b: Any) -> float:
    arr_a = _to_numpy(a).astype(np.float32)
    arr_b = _to_numpy(b).astype(np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)))


def _scalar_max_abs(a: Any, b: Any) -> float:
    arr_a = _to_numpy(a).astype(np.float32)
    arr_b = _to_numpy(b).astype(np.float32)
    return float(np.max(np.abs(arr_a - arr_b)))


def _tensor_compare(a: Any, b: Any) -> dict[str, Any]:
    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)
    same_shape = tuple(arr_a.shape) == tuple(arr_b.shape)
    result = {
        "shape_a": list(arr_a.shape),
        "shape_b": list(arr_b.shape),
        "same_shape": same_shape,
        "dtype_a": str(arr_a.dtype),
        "dtype_b": str(arr_b.dtype),
    }
    if not same_shape:
        return result

    if np.issubdtype(arr_a.dtype, np.floating) or np.issubdtype(arr_b.dtype, np.floating):
        diff = np.abs(arr_a.astype(np.float32) - arr_b.astype(np.float32))
        result["mae"] = float(diff.mean())
        result["max_abs"] = float(diff.max())
    else:
        equal = arr_a == arr_b
        result["exact_match"] = bool(np.all(equal))
        result["match_ratio"] = float(equal.mean())
    return result


def _serialize_stats(stats: Any) -> dict[str, Any]:
    if hasattr(stats, "model_dump"):
        return stats.model_dump()
    if hasattr(stats, "dict"):
        return stats.dict()
    raise TypeError(f"Unsupported statistics object: {type(stats)}")


def _get_state_action_stats(mixture_dataset, single_dataset) -> tuple[dict[str, Any], dict[str, Any]]:
    merged_metadata = mixture_dataset.merged_metadata[single_dataset.tag]
    state_stats = _serialize_stats(merged_metadata.statistics.state["joints"])
    action_stats = _serialize_stats(merged_metadata.statistics.action["delta_joints"])
    return state_stats, action_stats


def _compact_current_frame_index(video_compact: np.ndarray, *, video_target_shift_steps: int) -> int:
    context_horizon = int(video_compact.shape[1]) - int(video_target_shift_steps)
    if context_horizon <= 0:
        raise ValueError(
            f"Invalid compact video length {video_compact.shape[1]} for shift {video_target_shift_steps}"
        )
    return context_horizon - 1


def _build_compact_frame_images(
    example: dict[str, Any],
    *,
    frame_index: int,
) -> list[Image.Image]:
    video_compact = np.asarray(example["video_compact"], dtype=np.uint8)
    return [
        Image.fromarray(np.asarray(video_compact[view_index, frame_index], dtype=np.uint8), mode="RGB")
        for view_index in range(video_compact.shape[0])
    ]


def _raw_semantics(raw_state_seq: np.ndarray, raw_action_seq: np.ndarray) -> dict[str, Any]:
    state_seq = np.asarray(raw_state_seq, dtype=np.float32)
    action_seq = np.asarray(raw_action_seq, dtype=np.float32)
    max_horizon = min(len(action_seq), len(state_seq) - 1)
    result: dict[str, Any] = {
        "state_seq_shape": list(state_seq.shape),
        "action_seq_shape": list(action_seq.shape),
    }
    if max_horizon <= 0:
        return result

    next_state_seq = state_seq[1 : max_horizon + 1]
    delta_state_seq = next_state_seq - state_seq[:max_horizon]
    action_head = action_seq[:max_horizon]
    result["action_vs_next_state_mae"] = _scalar_mae(action_head, next_state_seq)
    result["action_vs_state_delta_mae"] = _scalar_mae(action_head, delta_state_seq)
    result["action_vs_current_state_mae"] = _scalar_mae(action_head, state_seq[:max_horizon])
    return result


def _resize_match_report(
    raw_data: dict[str, Any],
    example: dict[str, Any],
    *,
    video_target_shift_steps: int,
) -> dict[str, Any]:
    report = {}
    current_index = _compact_current_frame_index(
        np.asarray(example["video_compact"]),
        video_target_shift_steps=video_target_shift_steps,
    )
    example_images = _build_compact_frame_images(example, frame_index=current_index)
    raw_video_keys = ("video.base_view", "video.left_wrist", "video.right_wrist")
    image_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    for idx, (raw_key, image_name) in enumerate(zip(raw_video_keys, image_names, strict=True)):
        raw_first = np.asarray(raw_data[raw_key][0], dtype=np.uint8)
        expected = resize_for_training(raw_first, image_size=224)
        example_img = resize_for_training(np.asarray(example_images[idx], dtype=np.uint8), image_size=224)
        report[image_name] = {
            "train_image_vs_pipeline_resize_mae": _scalar_mae(expected, example_img),
            "train_image_vs_pipeline_resize_max_abs": _scalar_max_abs(expected, example_img),
        }
    return report


def _qwen_compare_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(a) | set(b))
    out = {}
    for key in keys:
        if key not in a or key not in b:
            out[key] = {"present_a": key in a, "present_b": key in b}
            continue
        out[key] = _tensor_compare(a[key], b[key])
    return out


def _aggregate_numeric(sample_reports: list[dict[str, Any]], path: list[str]) -> float | None:
    values = []
    for report in sample_reports:
        current: Any = report
        missing = False
        for part in path:
            if part not in current:
                missing = True
                break
            current = current[part]
        if not missing and isinstance(current, (int, float)) and math.isfinite(float(current)):
            values.append(float(current))
    if not values:
        return None
    return float(np.mean(values))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--use-bf16", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    cfg.datasets.vla_data.data_root_dir = str(args.dataset_root)

    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="val",
        action_horizon=int(cfg.framework.action_model.action_horizon),
        video_horizon=int(cfg.framework.vj2_model.num_frames),
        video_frame_stride=int(cfg.datasets.vla_data.get("video_frame_stride", 1)),
    )
    if len(dataset.datasets) != 1:
        raise ValueError(f"Expected a single dataset in the mixture, got {len(dataset.datasets)}")

    single_dataset = dataset.datasets[0]
    state_stats, action_stats = _get_state_action_stats(dataset, single_dataset)

    resolved_ckpt = resolve_policy_checkpoint(args.ckpt_path)
    model = baseframework.from_pretrained(str(resolved_ckpt))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.cuda)}")
    else:
        device = torch.device("cpu")
    if args.use_bf16 and device.type == "cuda":
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()
    print(f"Loaded model on {device} from {resolved_ckpt}", flush=True)

    report: dict[str, Any] = {
        "checkpoint": str(resolved_ckpt),
        "dataset_root": str(args.dataset_root),
        "num_samples": int(args.num_samples),
        "action_horizon": int(cfg.framework.action_model.action_horizon),
        "video_horizon": int(cfg.framework.vj2_model.num_frames),
        "sample_reports": [],
    }

    prompt_replace = {
        "{actions}": model.replace_prompt,
        "{e_actions}": model.embodied_replace_prompt,
    }
    prompt_template = cfg.datasets.vla_data.get("CoT_prompt", "")

    for sample_index in range(int(args.num_samples)):
        print(f"[audit] sample {sample_index + 1}/{int(args.num_samples)}", flush=True)
        example = dataset[sample_index]
        sampled_dataset, trajectory_id, base_index = dataset.sample_step(sample_index)
        if sampled_dataset is not single_dataset:
            raise AssertionError("Mixture unexpectedly sampled a different dataset")

        raw_data = single_dataset.get_step_data(trajectory_id, base_index)
        raw_state_seq = np.asarray(raw_data["state.joints"], dtype=np.float32)
        raw_action_seq = np.asarray(raw_data["action.delta_joints"], dtype=np.float32)
        normalized_state = np.asarray(example["state"], dtype=np.float32)
        normalized_action = np.asarray(example["action"], dtype=np.float32)
        compact_video = np.asarray(example["video_compact"], dtype=np.uint8)
        current_frame_index = _compact_current_frame_index(
            compact_video,
            video_target_shift_steps=int(cfg.datasets.vla_data.get("video_target_shift_steps", 0)),
        )
        current_frame_images = _build_compact_frame_images(example, frame_index=current_frame_index)
        earliest_frame_images = _build_compact_frame_images(example, frame_index=0)
        future_last_images = _build_compact_frame_images(example, frame_index=compact_video.shape[1] - 1)

        deploy_normalized_state = continuous_normalize(raw_state_seq[0], state_stats, mode="min_max")[None, :]
        deploy_unnormalized_action = continuous_unnormalize(normalized_action, action_stats, mode="min_max")

        with torch.inference_mode():
            print("[audit]   predict train-path", flush=True)
            pred_train = np.asarray(model.predict_action(batch=[example])["normalized_actions"][0], dtype=np.float32)
            print("[audit]   predict deploy-current", flush=True)
            pred_deploy_first = np.asarray(
                model.predict_action(
                    batch_images=[current_frame_images],
                    instructions=[example["lang"]],
                    state=normalized_state[None, ...],
                )["normalized_actions"][0],
                dtype=np.float32,
            )
            print("[audit]   predict deploy-earliest", flush=True)
            pred_deploy_last = np.asarray(
                model.predict_action(
                    batch_images=[earliest_frame_images],
                    instructions=[example["lang"]],
                    state=normalized_state[None, ...],
                )["normalized_actions"][0],
                dtype=np.float32,
            )
            print("[audit]   predict deploy-future", flush=True)
            pred_deploy_future = np.asarray(
                model.predict_action(
                    batch_images=[future_last_images],
                    instructions=[example["lang"]],
                    state=normalized_state[None, ...],
                )["normalized_actions"][0],
                dtype=np.float32,
            )

            print("[audit]   build qwen inputs", flush=True)
            batch_videos, _ = model._extract_training_videos([example])
            qwen_train = model._build_qwen_inputs_from_video_tensor(
                batch_videos=batch_videos,
                instructions=[example["lang"]],
                has_actions=True,
                prompt_replace_dict=prompt_replace,
                prompt_template=prompt_template,
            )
            qwen_deploy_first = model.qwen_vl_interface.build_qwenvl_inputs(
                images=[current_frame_images],
                instructions=[example["lang"]],
                prompt_replace_dict=prompt_replace,
            )
            qwen_deploy_last = model.qwen_vl_interface.build_qwenvl_inputs(
                images=[earliest_frame_images],
                instructions=[example["lang"]],
                prompt_replace_dict=prompt_replace,
            )
            qwen_deploy_future = model.qwen_vl_interface.build_qwenvl_inputs(
                images=[future_last_images],
                instructions=[example["lang"]],
                prompt_replace_dict=prompt_replace,
            )

        sample_report = {
            "sample_index": int(sample_index),
            "trajectory_id": int(trajectory_id),
            "base_index": int(base_index),
            "instruction": str(example["lang"]),
            "raw_shapes": {
                "state_seq": list(raw_state_seq.shape),
                "action_seq": list(raw_action_seq.shape),
                "video_compact": list(np.asarray(example["video_compact"]).shape),
                "normalized_state": list(normalized_state.shape),
                "normalized_action": list(normalized_action.shape),
            },
            "normalization_checks": {
                "state_pipeline_vs_training_mae": _scalar_mae(deploy_normalized_state, normalized_state),
                "state_pipeline_vs_training_max_abs": _scalar_max_abs(deploy_normalized_state, normalized_state),
                "action_unnorm_vs_raw_mae": _scalar_mae(deploy_unnormalized_action, raw_action_seq),
                "action_unnorm_vs_raw_max_abs": _scalar_max_abs(deploy_unnormalized_action, raw_action_seq),
            },
            "raw_action_semantics": _raw_semantics(raw_state_seq, raw_action_seq),
            "image_resize_checks": _resize_match_report(
                raw_data,
                example,
                video_target_shift_steps=int(cfg.datasets.vla_data.get("video_target_shift_steps", 0)),
            ),
            "frame_gap_checks": {
                "earliest_vs_current_frame_mae": {
                    camera_name: _scalar_mae(
                        np.asarray(earliest_frame_images[idx], dtype=np.uint8),
                        np.asarray(current_frame_images[idx], dtype=np.uint8),
                    )
                    for idx, camera_name in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist"))
                },
                "current_vs_future_frame_mae": {
                    camera_name: _scalar_mae(
                        np.asarray(current_frame_images[idx], dtype=np.uint8),
                        np.asarray(future_last_images[idx], dtype=np.uint8),
                    )
                    for idx, camera_name in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist"))
                },
            },
            "model_alignment": {
                "pred_train_vs_target_norm_mae": _scalar_mae(pred_train, normalized_action),
                "pred_deploy_current_vs_target_norm_mae": _scalar_mae(pred_deploy_first, normalized_action),
                "pred_deploy_earliest_vs_target_norm_mae": _scalar_mae(pred_deploy_last, normalized_action),
                "pred_deploy_future_vs_target_norm_mae": _scalar_mae(pred_deploy_future, normalized_action),
                "pred_train_vs_deploy_current_norm_mae": _scalar_mae(pred_train, pred_deploy_first),
                "pred_train_vs_deploy_earliest_norm_mae": _scalar_mae(pred_train, pred_deploy_last),
                "pred_train_vs_deploy_future_norm_mae": _scalar_mae(pred_train, pred_deploy_future),
            },
            "qwen_input_alignment": {
                "train_vs_deploy_current": _qwen_compare_dict(qwen_train, qwen_deploy_first),
                "train_vs_deploy_earliest": _qwen_compare_dict(qwen_train, qwen_deploy_last),
                "train_vs_deploy_future": _qwen_compare_dict(qwen_train, qwen_deploy_future),
            },
        }
        report["sample_reports"].append(sample_report)
        print(
            "[audit] sample_done",
            sample_index,
            "train_vs_target",
            f"{sample_report['model_alignment']['pred_train_vs_target_norm_mae']:.4f}",
            "deploy_current_vs_target",
            f"{sample_report['model_alignment']['pred_deploy_current_vs_target_norm_mae']:.4f}",
            "deploy_earliest_vs_target",
            f"{sample_report['model_alignment']['pred_deploy_earliest_vs_target_norm_mae']:.4f}",
            "deploy_future_vs_target",
            f"{sample_report['model_alignment']['pred_deploy_future_vs_target_norm_mae']:.4f}",
            flush=True,
        )

    report["summary"] = {
        "mean_state_pipeline_vs_training_mae": _aggregate_numeric(
            report["sample_reports"], ["normalization_checks", "state_pipeline_vs_training_mae"]
        ),
        "mean_action_unnorm_vs_raw_mae": _aggregate_numeric(
            report["sample_reports"], ["normalization_checks", "action_unnorm_vs_raw_mae"]
        ),
        "mean_raw_action_vs_next_state_mae": _aggregate_numeric(
            report["sample_reports"], ["raw_action_semantics", "action_vs_next_state_mae"]
        ),
        "mean_raw_action_vs_state_delta_mae": _aggregate_numeric(
            report["sample_reports"], ["raw_action_semantics", "action_vs_state_delta_mae"]
        ),
        "mean_pred_train_vs_target_norm_mae": _aggregate_numeric(
            report["sample_reports"], ["model_alignment", "pred_train_vs_target_norm_mae"]
        ),
        "mean_pred_deploy_current_vs_target_norm_mae": _aggregate_numeric(
            report["sample_reports"], ["model_alignment", "pred_deploy_current_vs_target_norm_mae"]
        ),
        "mean_pred_deploy_earliest_vs_target_norm_mae": _aggregate_numeric(
            report["sample_reports"], ["model_alignment", "pred_deploy_earliest_vs_target_norm_mae"]
        ),
        "mean_pred_deploy_future_vs_target_norm_mae": _aggregate_numeric(
            report["sample_reports"], ["model_alignment", "pred_deploy_future_vs_target_norm_mae"]
        ),
        "mean_pred_train_vs_deploy_current_norm_mae": _aggregate_numeric(
            report["sample_reports"], ["model_alignment", "pred_train_vs_deploy_current_norm_mae"]
        ),
        "mean_pred_train_vs_deploy_earliest_norm_mae": _aggregate_numeric(
            report["sample_reports"], ["model_alignment", "pred_train_vs_deploy_earliest_norm_mae"]
        ),
        "mean_pred_train_vs_deploy_future_norm_mae": _aggregate_numeric(
            report["sample_reports"], ["model_alignment", "pred_train_vs_deploy_future_norm_mae"]
        ),
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Saved report to {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
