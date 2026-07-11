# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deployment.model_server.checkpoint_utils import build_policy_metadata, resolve_policy_checkpoint
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


_REALMAN_ACTION_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
    "base_linear_x_mps",
    "base_linear_y_mps",
    "base_angular_z_radps",
    "head_joint_1_rad",
    "head_joint_2_rad",
    "lift_height_mm",
]

_REALMAN_STATE_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper_open",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper_open",
    "head_joint_1_rad",
    "head_joint_2_rad",
    "lift_height_mm",
]

_REALMAN_POLICY_ACTION_NAMES_NO_BASE = _REALMAN_ACTION_NAMES[:16] + _REALMAN_ACTION_NAMES[19:22]
_REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT = _REALMAN_ACTION_NAMES[:16] + _REALMAN_ACTION_NAMES[19:21]
_REALMAN_POLICY_ACTION_DIMS = (
    len(_REALMAN_ACTION_NAMES),
    len(_REALMAN_POLICY_ACTION_NAMES_NO_BASE),
    len(_REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT),
)
_REALMAN_DATA_MIXES = {
    "ogrealman_source_v3",
    "ogrealman_source_no_base_v3",
    "ogrealman_source_no_base_human_labelled_cloud_v3",
    "magna_source_no_base_interventions_v3",
    "magna_source_no_base_no_lift_interventions_v3",
    "ogrealman_canonical_v3",
}


def _is_realman_data_mix(metadata: dict[str, Any]) -> bool:
    robot_type = str(metadata.get("robot_type") or "")
    return metadata.get("data_mix") in _REALMAN_DATA_MIXES or robot_type.startswith("realman_bimanual")


def _expand_realman_policy_actions(actions: np.ndarray) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.shape[-1] == len(_REALMAN_ACTION_NAMES):
        return array
    if array.shape[-1] not in {
        len(_REALMAN_POLICY_ACTION_NAMES_NO_BASE),
        len(_REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT),
    }:
        raise ValueError(
            f"Expected Realman policy action dim in {_REALMAN_POLICY_ACTION_DIMS}, got {array.shape[-1]}."
        )
    expanded = np.zeros((*array.shape[:-1], len(_REALMAN_ACTION_NAMES)), dtype=np.float32)
    expanded[..., :16] = array[..., :16]
    if array.shape[-1] == len(_REALMAN_POLICY_ACTION_NAMES_NO_BASE):
        expanded[..., 19:22] = array[..., 16:19]
    else:
        expanded[..., 19:21] = array[..., 16:18]
    return expanded


def default_policy_output_log_path() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"/tmp/realman_server_policy_io_{timestamp}.jsonl"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _named_vector(values: np.ndarray, names: list[str] | None) -> dict[str, float] | list[float]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if names is None or len(names) != vector.shape[0]:
        return vector.tolist()
    return {name: float(value) for name, value in zip(names, vector, strict=True)}


def _continuous_unnormalize(values: np.ndarray, stats: dict[str, Any], *, mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if mode == "min_max":
        min_value = np.asarray(stats["min"], dtype=np.float32)
        max_value = np.asarray(stats["max"], dtype=np.float32)
        clipped = np.clip(array, -1.0, 1.0)
        return ((clipped + 1.0) / 2.0 * (max_value - min_value) + min_value).astype(np.float32, copy=False)
    if mode == "q99":
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        clipped = np.clip(array, -1.0, 1.0)
        mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)
        return np.where(mask, 0.5 * (clipped + 1.0) * (q99 - q01) + q01, clipped).astype(np.float32, copy=False)
    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return (array * std + mean).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported action normalization mode `{mode}`")


class PolicyOutputJsonlLogger:
    def __init__(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any],
        full_arrays: bool = True,
        every_n: int = 1,
        input_image_dir: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        latest_path = Path("/tmp/realman_server_policy_io_latest.jsonl")
        try:
            if latest_path.is_symlink() or not latest_path.exists():
                latest_path.unlink(missing_ok=True)
                latest_path.symlink_to(self.path)
            else:
                logging.warning("Not updating `%s` because it already exists and is not a symlink", latest_path)
        except OSError:
            logging.exception("Failed to update latest policy output log symlink")
        self.input_image_dir = Path(input_image_dir).expanduser().resolve() if input_image_dir else None
        if self.input_image_dir is not None:
            self.input_image_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.full_arrays = bool(full_arrays)
        self.every_n = max(1, int(every_n))
        self._counter = 0
        self._lock = threading.Lock()
        if _is_realman_data_mix(metadata) and int(metadata.get("action_dim") or 0) == len(_REALMAN_POLICY_ACTION_NAMES_NO_BASE):
            self.action_names = _REALMAN_POLICY_ACTION_NAMES_NO_BASE
        elif _is_realman_data_mix(metadata) and int(metadata.get("action_dim") or 0) == len(_REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT):
            self.action_names = _REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT
        elif _is_realman_data_mix(metadata) and int(metadata.get("action_dim") or 0) == len(_REALMAN_ACTION_NAMES):
            self.action_names = _REALMAN_ACTION_NAMES
        else:
            self.action_names = None
        self.state_names = (
            _REALMAN_STATE_NAMES
            if _is_realman_data_mix(metadata)
            else None
        )

        default_key = metadata.get("default_unnorm_key")
        action_stats_by_key = metadata.get("action_stats_by_key") or {}
        self.action_stats = action_stats_by_key.get(default_key) if default_key else None
        self.action_norm_mode = metadata.get("default_action_norm_mode") or "q99"

    def __call__(self, *, request_id: str, remote_address: Any, payload: dict[str, Any], output: dict[str, Any]) -> None:
        with self._lock:
            self._counter += 1
            sequence_id = self._counter
        if sequence_id % self.every_n != 0:
            return

        normalized = output.get("normalized_actions")
        normalized_actions = np.asarray(normalized, dtype=np.float32) if normalized is not None else None
        unnormalized_actions = None
        if normalized_actions is not None and self.action_stats is not None:
            try:
                unnormalized_actions = _continuous_unnormalize(
                    normalized_actions,
                    self.action_stats,
                    mode=self.action_norm_mode,
                )
            except Exception:
                logging.exception("Failed to unnormalize policy output for logging")

        record = {
            "event": "policy_server_output",
            "sequence_id": sequence_id,
            "timestamp_unix_s": time.time(),
            "request_id": request_id,
            "remote_address": str(remote_address),
            "metadata": {
                "run_id": self.metadata.get("run_id"),
                "checkpoint_path": self.metadata.get("checkpoint_path"),
                "data_mix": self.metadata.get("data_mix"),
                "action_type": self.metadata.get("action_type"),
                "action_horizon": self.metadata.get("action_horizon"),
                "action_dim": self.metadata.get("action_dim"),
                "state_dim": self.metadata.get("state_dim"),
                "num_inference_timesteps": self.metadata.get("num_inference_timesteps"),
                "default_unnorm_key": self.metadata.get("default_unnorm_key"),
                "default_action_norm_mode": self.metadata.get("default_action_norm_mode"),
                "camera_order_hint": self.metadata.get("camera_order_hint"),
            },
            "payload_summary": self._payload_summary(payload, sequence_id=sequence_id),
            "output_keys": sorted(output.keys()),
        }
        if normalized_actions is not None:
            record["normalized_actions_shape"] = list(normalized_actions.shape)
            record["normalized_actions_summary"] = self._action_summary(normalized_actions)
            if self.full_arrays:
                record["normalized_actions"] = normalized_actions
        if unnormalized_actions is not None:
            record["unnormalized_actions_shape"] = list(unnormalized_actions.shape)
            record["unnormalized_actions_summary"] = self._action_summary(unnormalized_actions)
            if self.full_arrays:
                record["unnormalized_actions"] = unnormalized_actions
        if "embodied_action_tokens" in output:
            tokens = np.asarray(output["embodied_action_tokens"], dtype=np.float32)
            record["embodied_action_tokens_summary"] = {
                "shape": list(tokens.shape),
                "mean": float(tokens.mean()),
                "std": float(tokens.std()),
                "min": float(tokens.min()),
                "max": float(tokens.max()),
            }
        if "action_guard" in output:
            record["action_guard"] = output["action_guard"]

        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(record)) + "\n")

    def _payload_summary(self, payload: dict[str, Any], *, sequence_id: int) -> dict[str, Any]:
        summary: dict[str, Any] = {"payload_keys": sorted(payload.keys())}
        instructions = payload.get("instructions")
        if instructions is not None:
            summary["instructions"] = instructions
        state = payload.get("state")
        if state is not None:
            state_arr = np.asarray(state, dtype=np.float32)
            summary["state"] = {
                "shape": list(state_arr.shape),
                "mean": float(state_arr.mean()),
                "std": float(state_arr.std()),
                "min": float(state_arr.min()),
                "max": float(state_arr.max()),
                "values": state_arr,
            }
            if state_arr.shape[-1] == len(_REALMAN_STATE_NAMES):
                flat = state_arr.reshape(-1, state_arr.shape[-1])
                if flat.shape[0] == 1:
                    summary["state"]["named_values"] = _named_vector(flat[0], self.state_names)
                else:
                    summary["state"]["named_values"] = [
                        _named_vector(row, self.state_names) for row in flat
                    ]
        batch_images = payload.get("batch_images")
        if batch_images is not None:
            summary["batch_images"] = self._image_summary(batch_images, sequence_id=sequence_id)
        return summary

    def _image_summary(self, batch_images: Any, *, sequence_id: int) -> dict[str, Any]:
        batch_count = len(batch_images) if hasattr(batch_images, "__len__") else None
        view_counts = []
        samples = []
        try:
            for sample_index, sample in enumerate(batch_images):
                view_counts.append(len(sample))
                sample_summary = []
                for view_index, image in enumerate(sample):
                    arr = np.asarray(image)
                    contiguous = np.ascontiguousarray(arr)
                    digest = hashlib.sha256(contiguous.tobytes()).hexdigest()
                    entry = {
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "mode": getattr(image, "mode", None),
                        "mean": float(arr.mean()),
                        "std": float(arr.std()),
                        "min": int(arr.min()),
                        "max": int(arr.max()),
                        "sha256": digest,
                    }
                    if self.input_image_dir is not None:
                        filename = (
                            f"request_{sequence_id:06d}_sample_{sample_index:02d}_"
                            f"view_{view_index:02d}_{digest[:12]}.png"
                        )
                        path = self.input_image_dir / filename
                        save = getattr(image, "save", None)
                        if callable(save):
                            save(path)
                        else:
                            from PIL import Image

                            Image.fromarray(contiguous).save(path)
                        entry["saved_path"] = str(path)
                    sample_summary.append(entry)
                samples.append(sample_summary)
        except Exception:
            logging.exception("Failed to summarize policy input images")
        return {
            "batch_count": batch_count,
            "view_counts": view_counts,
            "samples": samples,
        }

    def _action_summary(self, actions: np.ndarray) -> dict[str, Any]:
        array = np.asarray(actions, dtype=np.float32)
        flat_dim = array.shape[-1]
        flat = array.reshape(-1, flat_dim)
        summary: dict[str, Any] = {
            "shape": list(array.shape),
            "per_dim_min": _named_vector(flat.min(axis=0), self.action_names),
            "per_dim_max": _named_vector(flat.max(axis=0), self.action_names),
            "per_dim_mean": _named_vector(flat.mean(axis=0), self.action_names),
            "per_dim_abs_mean": _named_vector(np.abs(flat).mean(axis=0), self.action_names),
        }
        if flat_dim in _REALMAN_POLICY_ACTION_DIMS and self.action_names is not None:
            robot_actions = _expand_realman_policy_actions(array)
            lift_is_policy_controlled = flat_dim != len(_REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT)
            left_l2 = np.linalg.norm(robot_actions[..., 0:7], axis=-1)
            right_l2 = np.linalg.norm(robot_actions[..., 8:15], axis=-1)
            base_l2 = np.linalg.norm(robot_actions[..., 16:19], axis=-1)
            head_l2 = np.linalg.norm(robot_actions[..., 19:21], axis=-1)
            summary["realman_groups"] = {
                "left_arm_l2": left_l2.tolist(),
                "right_arm_l2": right_l2.tolist(),
                "base_l2": base_l2.tolist(),
                "head_l2": head_l2.tolist(),
                "lift_is_policy_controlled": lift_is_policy_controlled,
                "lift_height_mm": robot_actions[..., 21].tolist() if lift_is_policy_controlled else None,
                "near_zero_fraction": {
                    "arms_l2_lt_0_05": float(((left_l2 + right_l2) < 0.05).mean()),
                    "arms_l2_lt_0_25": float(((left_l2 + right_l2) < 0.25).mean()),
                    "base_l2_lt_0_001": float((base_l2 < 0.001).mean()),
                },
            }
        return summary


class ActionGuardRetryPolicy:
    """Retry stochastic action samples and reject chunks outside Realman bounds."""

    def __init__(
        self,
        policy: Any,
        *,
        metadata: dict[str, Any],
        max_attempts: int,
        first_n: int,
        tail_start: int,
        late_start: int,
        last_n: int,
        min_first_arms_mean: float,
        min_tail_arms_mean: float,
        min_late_arms_mean: float,
        min_last_arms_mean: float,
        min_tail_lift_mean: float,
    ) -> None:
        self.policy = policy
        self.metadata = metadata
        self.max_attempts = max(1, int(max_attempts))
        self.first_n = max(1, int(first_n))
        self.tail_start = max(0, int(tail_start))
        self.late_start = max(0, int(late_start))
        self.last_n = max(1, int(last_n))
        self.min_first_arms_mean = float(min_first_arms_mean)
        self.min_tail_arms_mean = float(min_tail_arms_mean)
        self.min_late_arms_mean = float(min_late_arms_mean)
        self.min_last_arms_mean = float(min_last_arms_mean)
        self.min_tail_lift_mean = float(min_tail_lift_mean)

        default_key = metadata.get("default_unnorm_key")
        action_stats_by_key = metadata.get("action_stats_by_key") or {}
        self.action_stats = action_stats_by_key.get(default_key) if default_key else None
        self.action_norm_mode = metadata.get("default_action_norm_mode") or "q99"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def predict_action(self, *args, **kwargs) -> dict[str, Any]:
        rejected: list[dict[str, Any]] = []
        for attempt_index in range(self.max_attempts):
            output = self.policy.predict_action(*args, **kwargs)
            valid, metrics, reasons = self._validate_output(output)
            guard_record = {
                "enabled": True,
                "max_attempts": self.max_attempts,
                "accepted": valid,
                "attempt": attempt_index + 1,
                "metrics": metrics,
                "reasons": reasons,
                "rejected_attempts": rejected,
            }
            if valid:
                output["action_guard"] = guard_record
                if attempt_index > 0:
                    logging.warning(
                        "Action guard accepted attempt %d/%d after rejecting %d samples",
                        attempt_index + 1,
                        self.max_attempts,
                        attempt_index,
                    )
                return output
            rejected.append(
                {
                    "attempt": attempt_index + 1,
                    "metrics": metrics,
                    "reasons": reasons,
                }
            )
            logging.warning(
                "Action guard rejected attempt %d/%d: %s",
                attempt_index + 1,
                self.max_attempts,
                ", ".join(reasons) if reasons else "unknown reason",
            )

        raise RuntimeError(
            "Action guard rejected all sampled chunks; refusing to send an unsafe action. "
            f"attempts={json.dumps(_json_safe(rejected))}"
        )

    def _validate_output(self, output: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
        reasons: list[str] = []
        normalized = output.get("normalized_actions")
        metrics: dict[str, Any] = {}
        if normalized is None:
            return False, metrics, ["missing normalized_actions"]
        if self.action_stats is None:
            return False, metrics, ["missing action normalization stats"]

        normalized_actions = np.asarray(normalized, dtype=np.float32)
        if normalized_actions.ndim != 3:
            return False, metrics, [f"expected normalized_actions [B,T,D], got {normalized_actions.shape}"]
        if not np.isfinite(normalized_actions).all():
            return False, metrics, ["normalized_actions contains non-finite values"]

        actions = _continuous_unnormalize(
            normalized_actions,
            self.action_stats,
            mode=self.action_norm_mode,
        )
        if actions.shape[-1] not in _REALMAN_POLICY_ACTION_DIMS:
            return True, {"shape": list(actions.shape), "skipped": "unsupported_realman_action_dim"}, []
        policy_controls_lift = actions.shape[-1] != len(_REALMAN_POLICY_ACTION_NAMES_NO_BASE_NO_LIFT)
        actions = _expand_realman_policy_actions(actions)
        if not np.isfinite(actions).all():
            return False, {"shape": list(actions.shape)}, ["unnormalized actions contain non-finite values"]

        horizon = actions.shape[1]
        first_end = min(self.first_n, horizon)
        tail_start = min(self.tail_start, max(horizon - 1, 0))
        late_start = min(self.late_start, max(horizon - 1, 0))
        last_n = min(self.last_n, horizon)

        left_l2 = np.linalg.norm(actions[..., 0:7], axis=-1)
        right_l2 = np.linalg.norm(actions[..., 8:15], axis=-1)
        arms_l2 = left_l2 + right_l2
        lift = actions[..., 21]

        first_mean = arms_l2[:, :first_end].mean(axis=1)
        tail_mean = arms_l2[:, tail_start:].mean(axis=1)
        late_mean = arms_l2[:, late_start:].mean(axis=1)
        last_mean = arms_l2[:, -last_n:].mean(axis=1)
        tail_lift_mean = lift[:, tail_start:].mean(axis=1) if policy_controls_lift else None

        metrics = {
            "shape": list(actions.shape),
            "first_n": first_end,
            "tail_start": tail_start,
            "late_start": late_start,
            "last_n": last_n,
            "first_arms_mean_min": float(first_mean.min()),
            "tail_arms_mean_min": float(tail_mean.min()),
            "late_arms_mean_min": float(late_mean.min()),
            "last_arms_mean_min": float(last_mean.min()),
            "lift_is_policy_controlled": policy_controls_lift,
            "tail_lift_mean_min": float(tail_lift_mean.min()) if tail_lift_mean is not None else None,
            "normalized_abs_max": float(np.abs(normalized_actions).max()),
        }

        if metrics["first_arms_mean_min"] < self.min_first_arms_mean:
            reasons.append(
                f"first arms mean {metrics['first_arms_mean_min']:.3f} < {self.min_first_arms_mean:.3f}"
            )
        if metrics["tail_arms_mean_min"] < self.min_tail_arms_mean:
            reasons.append(
                f"tail arms mean {metrics['tail_arms_mean_min']:.3f} < {self.min_tail_arms_mean:.3f}"
            )
        if metrics["late_arms_mean_min"] < self.min_late_arms_mean:
            reasons.append(
                f"late arms mean {metrics['late_arms_mean_min']:.3f} < {self.min_late_arms_mean:.3f}"
            )
        if metrics["last_arms_mean_min"] < self.min_last_arms_mean:
            reasons.append(
                f"last arms mean {metrics['last_arms_mean_min']:.3f} < {self.min_last_arms_mean:.3f}"
            )
        if (
            policy_controls_lift
            and self.min_tail_lift_mean >= 0
            and metrics["tail_lift_mean_min"] < self.min_tail_lift_mean
        ):
            reasons.append(
                f"tail lift mean {metrics['tail_lift_mean_min']:.3f} < {self.min_tail_lift_mean:.3f}"
            )

        return len(reasons) == 0, metrics, reasons


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    resolved_ckpt_path = resolve_policy_checkpoint(args.ckpt_path)
    logging.info("Loading policy checkpoint from `%s`", resolved_ckpt_path)

    if torch.cuda.is_available():
        device_index = int(args.cuda)
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
        if args.use_bf16:
            logging.warning("Ignoring --use_bf16 because CUDA is not available")

    vla = baseframework.from_pretrained(
        str(resolved_ckpt_path),
        inference_only=not args.load_training_backbones,
        skip_training_backbones=not args.load_training_backbones,
    )  # TODO should auto detect framework from model path

    if args.use_bf16 and device.type == "cuda":
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    metadata = build_policy_metadata(vla, resolved_ckpt_path)
    output_logger = None
    if args.policy_output_log_path:
        output_logger = PolicyOutputJsonlLogger(
            args.policy_output_log_path,
            metadata=metadata,
            full_arrays=not args.policy_output_log_summary_only,
            every_n=args.policy_output_log_every,
            input_image_dir=args.policy_input_image_dir,
        )
        logging.info("Policy server output logging enabled at `%s`", output_logger.path)

    served_policy = vla
    guard_is_realman = _is_realman_data_mix(metadata)
    guard_enabled = (
        not args.disable_action_guard
        and guard_is_realman
        and int(metadata.get("action_dim") or 0) in _REALMAN_POLICY_ACTION_DIMS
        and int(args.action_guard_max_attempts) > 1
    )
    if guard_enabled:
        served_policy = ActionGuardRetryPolicy(
            vla,
            metadata=metadata,
            max_attempts=args.action_guard_max_attempts,
            first_n=args.action_guard_first_n,
            tail_start=args.action_guard_tail_start,
            late_start=args.action_guard_late_start,
            last_n=args.action_guard_last_n,
            min_first_arms_mean=args.action_guard_min_first_arms_mean,
            min_tail_arms_mean=args.action_guard_min_tail_arms_mean,
            min_late_arms_mean=args.action_guard_min_late_arms_mean,
            min_last_arms_mean=args.action_guard_min_last_arms_mean,
            min_tail_lift_mean=args.action_guard_min_tail_lift_mean,
        )
        metadata["action_guard"] = {
            "enabled": True,
            "max_attempts": int(args.action_guard_max_attempts),
            "first_n": int(args.action_guard_first_n),
            "tail_start": int(args.action_guard_tail_start),
            "late_start": int(args.action_guard_late_start),
            "last_n": int(args.action_guard_last_n),
            "min_first_arms_mean": float(args.action_guard_min_first_arms_mean),
            "min_tail_arms_mean": float(args.action_guard_min_tail_arms_mean),
            "min_late_arms_mean": float(args.action_guard_min_late_arms_mean),
            "min_last_arms_mean": float(args.action_guard_min_last_arms_mean),
            "min_tail_lift_mean": float(args.action_guard_min_tail_lift_mean),
        }
        logging.info("Action guard enabled: %s", metadata["action_guard"])
    elif args.disable_action_guard:
        logging.warning("Action guard disabled by CLI flag")
    elif guard_is_realman:
        logging.warning("Action guard inactive because max_attempts <= 1")

    # start websocket server
    server = WebsocketPolicyServer(
        policy=served_policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
        output_logger=output_logger,
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--load_training_backbones",
        action="store_true",
        default=env_flag_enabled("POLICY_LOAD_TRAINING_BACKBONES"),
        help="Load train-only V-JEPA/MoGe backbones. Disabled by default for policy serving/eval.",
    )
    parser.add_argument(
        "--policy_output_log_path",
        type=str,
        default=os.getenv("POLICY_OUTPUT_LOG_PATH") or default_policy_output_log_path(),
        help=(
            "JSONL path for logging every policy request/output emitted by the websocket server. "
            "Defaults to /tmp/realman_server_policy_io_<timestamp>.jsonl."
        ),
    )
    parser.add_argument(
        "--policy_output_log_every",
        type=int,
        default=int(os.getenv("POLICY_OUTPUT_LOG_EVERY", "1")),
        help="Log every Nth inference response.",
    )
    parser.add_argument(
        "--policy_output_log_summary_only",
        action="store_true",
        default=env_flag_enabled("POLICY_OUTPUT_LOG_SUMMARY_ONLY"),
        help="Only log shapes/summaries instead of full action arrays.",
    )
    parser.add_argument(
        "--policy_input_image_dir",
        type=str,
        default=os.getenv("POLICY_INPUT_IMAGE_DIR"),
        help=(
            "Optional directory for saving the input camera frames received by the policy server. "
            "The JSONL log stores image stats and SHA256 hashes either way."
        ),
    )
    parser.add_argument(
        "--disable_action_guard",
        action="store_true",
        default=env_flag_enabled("POLICY_DISABLE_ACTION_GUARD"),
        help="Disable Realman action safety retry checks.",
    )
    parser.add_argument(
        "--action_guard_max_attempts",
        type=int,
        default=int(os.getenv("POLICY_ACTION_GUARD_MAX_ATTEMPTS", "8")),
        help="Maximum stochastic samples before refusing to return an unsafe Realman action chunk.",
    )
    parser.add_argument("--action_guard_first_n", type=int, default=int(os.getenv("POLICY_ACTION_GUARD_FIRST_N", "10")))
    parser.add_argument("--action_guard_tail_start", type=int, default=int(os.getenv("POLICY_ACTION_GUARD_TAIL_START", "20")))
    parser.add_argument("--action_guard_late_start", type=int, default=int(os.getenv("POLICY_ACTION_GUARD_LATE_START", "30")))
    parser.add_argument("--action_guard_last_n", type=int, default=int(os.getenv("POLICY_ACTION_GUARD_LAST_N", "5")))
    parser.add_argument(
        "--action_guard_min_first_arms_mean",
        type=float,
        default=float(os.getenv("POLICY_ACTION_GUARD_MIN_FIRST_ARMS_MEAN", "3.0")),
    )
    parser.add_argument(
        "--action_guard_min_tail_arms_mean",
        type=float,
        default=float(os.getenv("POLICY_ACTION_GUARD_MIN_TAIL_ARMS_MEAN", "4.0")),
    )
    parser.add_argument(
        "--action_guard_min_late_arms_mean",
        type=float,
        default=float(os.getenv("POLICY_ACTION_GUARD_MIN_LATE_ARMS_MEAN", "3.0")),
    )
    parser.add_argument(
        "--action_guard_min_last_arms_mean",
        type=float,
        default=float(os.getenv("POLICY_ACTION_GUARD_MIN_LAST_ARMS_MEAN", "3.0")),
    )
    parser.add_argument(
        "--action_guard_min_tail_lift_mean",
        type=float,
        default=float(os.getenv("POLICY_ACTION_GUARD_MIN_TAIL_LIFT_MEAN", "250.0")),
        help="Minimum mean unnormalized lift over tail_start..end. Set negative to disable.",
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10091))
    logging.info("Waiting for VSCode attach on 0.0.0.0:10091")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


def env_flag_enabled(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if env_flag_enabled("DEBUG"):
        logging.info("DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
