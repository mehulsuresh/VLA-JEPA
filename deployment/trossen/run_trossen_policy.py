from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.trossen.pipeline import (
    DEFAULT_CAMERA_ORDER,
    DEFAULT_IMAGE_SIZE,
    add_to_syspath,
    build_policy_payload,
    build_rollout_record,
    build_stationary_robot_config,
    compute_action_state_stats_closeness,
    compute_absolute_goal_chunk,
    continuous_unnormalize,
    extract_ordered_images,
    extract_state_vector,
    infer_action_mode_from_stats,
    resolve_action_stats,
    resolve_norm_mode,
    resolve_state_stats,
    resolve_yondu_root,
    save_rollout_record,
    validate_server_metadata,
)


def _parse_optional_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    lowered = raw_value.strip().lower()
    if lowered in {"none", "null", ""}:
        return None
    return float(raw_value)


def _save_images(images: list[np.ndarray], output_dir: Path, step_index: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_name, image in zip(DEFAULT_CAMERA_ORDER, images, strict=True):
        Image.fromarray(image, mode="RGB").save(output_dir / f"step_{step_index:06d}_{camera_name}.jpg", quality=95)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--unnorm-key", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--action-mode", choices=("auto", "delta_qpos", "absolute_qpos"), default="auto")
    parser.add_argument("--action-norm-mode", choices=("auto", "min_max", "q99", "mean_std"), default="auto")
    parser.add_argument("--state-norm-mode", choices=("auto", "min_max", "q99", "mean_std"), default="auto")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--delta-clip", type=float, default=None)
    parser.add_argument("--max-relative-target", type=str, default="5")
    parser.add_argument("--follower-gripper-force-limit", type=float, default=None)
    parser.add_argument("--yondu-lerobot-root", type=str, default=os.getenv("YONDU_TROSSEN_LEROBOT_ROOT"))
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--save-images-every", type=int, default=0)
    parser.add_argument("--image-log-dir", type=str, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--connect-leaders", action="store_true")
    parser.add_argument(
        "--confirm-each-replan",
        dest="confirm_each_replan",
        action="store_true",
        help="Wait for Enter before each live replan. This is the default behavior.",
    )
    parser.add_argument(
        "--no-confirm-each-replan",
        dest="confirm_each_replan",
        action="store_false",
        help="Run continuously without waiting for Enter before each live replan.",
    )
    parser.set_defaults(confirm_each_replan=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--print-metadata", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--left-leader-ip", type=str, default=None)
    parser.add_argument("--right-leader-ip", type=str, default=None)
    parser.add_argument("--left-follower-ip", type=str, default=None)
    parser.add_argument("--right-follower-ip", type=str, default=None)
    parser.add_argument("--cam-high-serial", type=int, default=None)
    parser.add_argument("--cam-left-wrist-serial", type=int, default=None)
    parser.add_argument("--cam-right-wrist-serial", type=int, default=None)
    return parser


def main(args) -> None:
    logging.info("Connecting to policy server at %s:%s", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()

    if args.print_metadata:
        print(json.dumps(metadata, indent=2))

    warnings = validate_server_metadata(metadata)
    for warning in warnings:
        logging.warning(warning)

    action_stats = resolve_action_stats(metadata, args.unnorm_key)
    try:
        state_stats = resolve_state_stats(metadata, args.unnorm_key)
    except (KeyError, ValueError):
        state_stats = None
    action_norm_mode = resolve_norm_mode(metadata, "action", args.action_norm_mode)
    state_norm_mode = resolve_norm_mode(metadata, "state", args.state_norm_mode)

    stats_closeness = compute_action_state_stats_closeness(action_stats, state_stats)
    if args.action_mode == "auto":
        action_mode = infer_action_mode_from_stats(action_stats, state_stats)
    else:
        action_mode = args.action_mode

    logging.info(
        "Using action_mode=%s (requested=%s server_action_type=%s stats_closeness=%s)",
        action_mode,
        args.action_mode,
        metadata.get("action_type"),
        "n/a" if stats_closeness is None else f"{stats_closeness:.4f}",
    )
    logging.info(
        "Using norm modes: action=%s (requested=%s) state=%s (requested=%s)",
        action_norm_mode,
        args.action_norm_mode,
        state_norm_mode,
        args.state_norm_mode,
    )
    if action_mode == "absolute_qpos" and metadata.get("action_type") == "delta_qpos":
        logging.warning(
            "Checkpoint config reports delta_qpos, but action/state stats are near-identical. "
            "Using absolute_qpos rollout mode."
        )
    if action_mode == "absolute_qpos" and args.chunk_size > 1:
        logging.warning(
            "chunk_size=%s executes multiple absolute joint-goal commands open-loop. "
            "The recorded Trossen dataset stores one commanded goal per control step, "
            "so chunk_size=1 is the closest match to training.",
            args.chunk_size,
        )
    if action_mode == "absolute_qpos":
        if args.action_scale != 1.0:
            logging.warning("--action-scale is ignored for absolute_qpos mode")
        if args.delta_clip is not None:
            logging.warning("--delta-clip is ignored for absolute_qpos mode")

    if args.check_only:
        logging.info("Server metadata validated. Exiting without touching the robot.")
        client.close()
        return

    yondu_root = resolve_yondu_root(args.yondu_lerobot_root)
    add_to_syspath(yondu_root)
    from lerobot.common.robot_devices.robots.utils import make_robot_from_config

    robot_cfg = build_stationary_robot_config(
        yondu_lerobot_root=yondu_root,
        max_relative_target=_parse_optional_float(args.max_relative_target),
        mock=args.mock,
        connect_leaders=args.connect_leaders,
        camera_order=DEFAULT_CAMERA_ORDER,
        left_leader_ip=args.left_leader_ip,
        right_leader_ip=args.right_leader_ip,
        left_follower_ip=args.left_follower_ip,
        right_follower_ip=args.right_follower_ip,
        cam_high_serial=args.cam_high_serial,
        cam_left_wrist_serial=args.cam_left_wrist_serial,
        cam_right_wrist_serial=args.cam_right_wrist_serial,
    )
    robot = make_robot_from_config(robot_cfg)
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else None
    image_log_dir = Path(args.image_log_dir).expanduser().resolve() if args.image_log_dir else None

    try:
        robot.connect()
        for follower_arm in robot.follower_arms.values():
            follower_arm.fps = float(args.fps)
            follower_arm.is_policy_in_radians = False
            if args.follower_gripper_force_limit is not None:
                follower_arm.write("Gripper_Force_Limit", args.follower_gripper_force_limit)

        action_horizon = int(metadata["action_horizon"])
        if args.action_index < 0 or args.action_index >= action_horizon:
            raise ValueError(
                f"--action-index {args.action_index} is out of range for action_horizon={action_horizon}"
            )
        if args.chunk_size <= 0 or args.chunk_size > action_horizon:
            raise ValueError(f"--chunk-size must be in [1, {action_horizon}], got {args.chunk_size}")

        period_s = 1.0 / float(args.fps)
        step_index = 0
        replan_index = 0

        while step_index < int(args.num_steps):
            remaining_steps = int(args.num_steps) - step_index
            max_actions_this_replan = min(
                args.chunk_size,
                remaining_steps,
                action_horizon - args.action_index,
            )
            live_actions_in_chunk = max(
                0,
                min(step_index + max_actions_this_replan, int(args.num_steps)) - max(step_index, args.warmup_steps),
            )
            if args.confirm_each_replan and args.live and live_actions_in_chunk > 0:
                input(
                    "Press Enter to capture a fresh observation and execute the next "
                    f"{live_actions_in_chunk} live action(s) from replan {replan_index} "
                    f"(steps {step_index}..{min(step_index + max_actions_this_replan - 1, int(args.num_steps) - 1)}), "
                    "or Ctrl-C to abort..."
                )

            observation = robot.capture_observation()
            payload = build_policy_payload(
                observation,
                instruction=args.instruction,
                camera_order=DEFAULT_CAMERA_ORDER,
                image_size=args.image_size,
                state_stats=state_stats,
                state_norm_mode=state_norm_mode,
            )
            current_state = extract_state_vector(observation)
            inference_start = time.perf_counter()
            response = client.infer(payload)
            latency_ms = (time.perf_counter() - inference_start) * 1000.0

            if not response.get("ok", False):
                raise RuntimeError(f"Inference failed: {response}")

            normalized_chunk = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)[0]
            policy_action_chunk = continuous_unnormalize(
                normalized_chunk,
                action_stats,
                mode=action_norm_mode,
            )
            execution_start = min(args.action_index, policy_action_chunk.shape[0] - 1)
            execution_chunk = policy_action_chunk[execution_start:]
            chunk_len = min(args.chunk_size, remaining_steps)
            planned_chunk = np.asarray(execution_chunk[:chunk_len], dtype=np.float32)

            if action_mode == "delta_qpos":
                goal_chunk = compute_absolute_goal_chunk(
                    current_state,
                    planned_chunk,
                    action_scale=args.action_scale,
                    delta_clip=args.delta_clip,
                )
            else:
                goal_chunk = np.asarray(planned_chunk, dtype=np.float32)
            next_deadline = time.perf_counter()

            if image_log_dir is not None and args.save_images_every > 0 and step_index % args.save_images_every == 0:
                images = extract_ordered_images(
                    observation,
                    camera_order=DEFAULT_CAMERA_ORDER,
                    image_size=args.image_size,
                )
                _save_images(images, image_log_dir, step_index)

            for chunk_offset, goal_action in enumerate(goal_chunk):
                if step_index >= int(args.num_steps):
                    break
                policy_action = planned_chunk[chunk_offset]
                sent_action = None
                if step_index >= args.warmup_steps and args.live:
                    sent_tensor = robot.send_action(torch.from_numpy(goal_action.astype(np.float32)))
                    sent_action = np.asarray(sent_tensor.detach().cpu().numpy(), dtype=np.float32)

                logging.info(
                    "step=%s replan=%s chunk=%s latency_ms=%.1f live=%s warm=%s action_mode=%s policy_l2=%.3f",
                    step_index,
                    replan_index,
                    chunk_offset,
                    latency_ms,
                    args.live,
                    step_index < args.warmup_steps,
                    action_mode,
                    float(np.linalg.norm(policy_action)),
                )

                if log_path is not None:
                    record = build_rollout_record(
                        step_index=step_index,
                        replan_index=replan_index,
                        chunk_offset=chunk_offset,
                        action_mode=action_mode,
                        instruction=args.instruction,
                        latency_ms=latency_ms,
                        current_state=current_state,
                        policy_action=policy_action,
                        goal_action=goal_action,
                        sent_action=sent_action,
                        normalized_chunk=normalized_chunk,
                    )
                    save_rollout_record(log_path, record)

                step_index += 1
                if chunk_offset == len(goal_chunk) - 1:
                    break

                next_deadline += period_s
                sleep_s = next_deadline - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    lag_ms = abs(sleep_s) * 1000.0
                    next_deadline = time.perf_counter()
                    logging.warning(
                        "Actuation loop is behind schedule by %.1f ms on step %s",
                        lag_ms,
                        step_index,
                    )

            replan_index += 1

    finally:
        client.close()
        if "robot" in locals() and getattr(robot, "is_connected", False):
            robot.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    main(parser.parse_args())
