from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.trossen.pipeline import (
    continuous_unnormalize,
    resolve_action_stats,
    resolve_norm_mode,
    resolve_state_stats,
)
from deployment.realman.pipeline import (
    DEFAULT_IMAGE_SIZE,
    REALMAN_ACTION_DIM,
    REALMAN_ACTION_NAMES,
    REALMAN_CAMERA_ORDER,
    REALMAN_POLICY_ACTION_DIMS,
    REALMAN_STATE_DIM,
    action_summary,
    build_policy_payload,
    expand_policy_action_to_robot_action,
    json_safe,
    load_observation_npz,
    split_action_chunk,
    split_action_vector,
    validate_realman_server_metadata,
    write_jsonl,
)


DEFAULT_INSTRUCTION = (
    "reach into the bin, pickup the metal chain with both hands, place it on the jig "
    "and align the corners to fit into the recessed channel."
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VLA-JEPA policy inference for the Realman source-action setup.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--unnorm-key", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--action-norm-mode", choices=("auto", "min_max", "q99", "mean_std"), default="auto")
    parser.add_argument("--state-norm-mode", choices=("auto", "min_max", "q99", "mean_std"), default="auto")
    parser.add_argument("--observation-npz", type=str, default=None)
    parser.add_argument(
        "--robot-module",
        type=str,
        default=None,
        help=(
            "Optional hardware adapter factory as `module:function`. The factory is called with this argparse "
            "namespace when it accepts arguments, otherwise with no arguments. The returned object must provide "
            "`capture_observation()` and may provide `connect()`, `disconnect()`, and `send_action(...)`."
        ),
    )
    parser.add_argument("--send-format", choices=("vector", "split"), default="split")
    parser.add_argument("--live", action="store_true", help="Actually call robot.send_action for planned actions.")
    parser.add_argument(
        "--confirm-each-replan",
        dest="confirm_each_replan",
        action="store_true",
        help="Wait for Enter before each live replan. Enabled by default when --live is set.",
    )
    parser.add_argument(
        "--no-confirm-each-replan",
        dest="confirm_each_replan",
        action="store_false",
        help="Run without waiting for Enter before each live replan.",
    )
    parser.set_defaults(confirm_each_replan=True)
    parser.add_argument("--check-only", action="store_true", help="Validate server metadata and exit.")
    parser.add_argument("--print-metadata", action="store_true")
    parser.add_argument("--allow-metadata-mismatch", action="store_true")
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def _load_robot(spec: str | None, args: argparse.Namespace):
    if spec is None:
        return None
    module_name, sep, attr_name = spec.partition(":")
    if not sep:
        attr_name = "create_robot"
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    try:
        signature = inspect.signature(factory)
        if len(signature.parameters) == 0:
            return factory()
    except (TypeError, ValueError):
        pass
    return factory(args)


def _send_action(robot: Any, action: np.ndarray, send_format: str):
    if not hasattr(robot, "send_action"):
        raise AttributeError("Robot adapter does not provide `send_action(...)`.")
    if send_format == "vector":
        return robot.send_action(np.asarray(action, dtype=np.float32))
    return robot.send_action(split_action_vector(action))


def _capture_observation(robot: Any, observation_npz: str | None) -> dict[str, Any]:
    if observation_npz is not None:
        return load_observation_npz(observation_npz)
    if robot is None:
        raise ValueError("Pass --observation-npz, --robot-module, or use --check-only.")
    if not hasattr(robot, "capture_observation"):
        raise AttributeError("Robot adapter does not provide `capture_observation()`.")
    observation = robot.capture_observation()
    if not isinstance(observation, dict):
        raise TypeError(f"capture_observation() must return a dict, got {type(observation).__name__}.")
    return observation


def _infer_once(
    *,
    client: Any,
    observation: dict[str, Any],
    instruction: str,
    action_stats: dict[str, Any],
    state_stats: dict[str, Any] | None,
    action_norm_mode: str,
    state_norm_mode: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    payload = build_policy_payload(
        observation,
        instruction=instruction,
        image_size=image_size,
        state_stats=state_stats,
        state_norm_mode=state_norm_mode,
    )
    started = time.perf_counter()
    response = client.infer(payload)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if not response.get("ok", False):
        raise RuntimeError(f"Inference failed: {response}")

    normalized_chunk = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)[0]
    policy_action_chunk = continuous_unnormalize(normalized_chunk, action_stats, mode=action_norm_mode)
    if policy_action_chunk.shape[-1] not in REALMAN_POLICY_ACTION_DIMS:
        raise RuntimeError(
            f"Policy returned action dim {policy_action_chunk.shape[-1]}, "
            f"expected one of {REALMAN_POLICY_ACTION_DIMS}."
        )
    action_chunk = expand_policy_action_to_robot_action(policy_action_chunk)
    return normalized_chunk, action_chunk, latency_ms, response


def _named_action_vector(action: np.ndarray) -> dict[str, float]:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape[0] != REALMAN_ACTION_DIM:
        raise ValueError(f"Expected Realman action dim {REALMAN_ACTION_DIM}, got {vector.shape[0]}")
    return {name: float(value) for name, value in zip(REALMAN_ACTION_NAMES, vector, strict=True)}


def _chunk_summary(chunk: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(chunk, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != REALMAN_ACTION_DIM:
        raise ValueError(f"Expected action chunk [T, {REALMAN_ACTION_DIM}], got {actions.shape}")

    left_l2 = np.linalg.norm(actions[:, 0:7], axis=1)
    right_l2 = np.linalg.norm(actions[:, 8:15], axis=1)
    base_l2 = np.linalg.norm(actions[:, 16:19], axis=1)
    head_l2 = np.linalg.norm(actions[:, 19:21], axis=1)
    per_dim_mean = actions.mean(axis=0)
    per_dim_min = actions.min(axis=0)
    per_dim_max = actions.max(axis=0)

    return {
        "shape": list(actions.shape),
        "left_arm_l2": left_l2.tolist(),
        "right_arm_l2": right_l2.tolist(),
        "base_l2": base_l2.tolist(),
        "head_l2": head_l2.tolist(),
        "lift_height_mm": actions[:, 21].tolist(),
        "per_dim_mean": _named_action_vector(per_dim_mean),
        "per_dim_min": _named_action_vector(per_dim_min),
        "per_dim_max": _named_action_vector(per_dim_max),
        "near_zero_fraction": {
            "arms_l2_lt_0_05": float(((left_l2 + right_l2) < 0.05).mean()),
            "arms_l2_lt_0_25": float(((left_l2 + right_l2) < 0.25).mean()),
            "base_l2_lt_0_001": float((base_l2 < 0.001).mean()),
        },
    }


def _log_policy_chunk(
    *,
    log_path: str | None,
    step_index: int,
    replan_index: int,
    instruction: str,
    latency_ms: float,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    execution_start: int,
    normalized_chunk: np.ndarray,
    action_chunk: np.ndarray,
    planned_chunk: np.ndarray,
) -> None:
    if log_path is None:
        return
    write_jsonl(
        log_path,
        {
            "event": "policy_chunk",
            "step_index": step_index,
            "replan_index": replan_index,
            "instruction": instruction,
            "latency_ms": latency_ms,
            "fps": args.fps,
            "chunk_size": args.chunk_size,
            "action_index": args.action_index,
            "execution_start": execution_start,
            "planned_chunk_len": int(planned_chunk.shape[0]),
            "live": bool(args.live),
            "send_format": args.send_format,
            "action_names": list(REALMAN_ACTION_NAMES),
            "metadata": {
                "run_id": metadata.get("run_id"),
                "checkpoint_path": metadata.get("checkpoint_path"),
                "action_horizon": metadata.get("action_horizon"),
                "action_dim": metadata.get("action_dim"),
                "state_dim": metadata.get("state_dim"),
                "action_type": metadata.get("action_type"),
                "num_inference_timesteps": metadata.get("num_inference_timesteps"),
                "default_unnorm_key": metadata.get("default_unnorm_key"),
                "default_action_norm_mode": metadata.get("default_action_norm_mode"),
                "default_state_norm_mode": metadata.get("default_state_norm_mode"),
                "camera_order_hint": metadata.get("camera_order_hint"),
            },
            "policy_action_dim": int(normalized_chunk.shape[-1]),
            "robot_action_dim": int(action_chunk.shape[-1]),
            "normalized_policy_chunk": normalized_chunk,
            "unnormalized_policy_chunk": action_chunk,
            "planned_chunk": planned_chunk,
            "policy_chunk_split": split_action_chunk(action_chunk),
            "planned_chunk_split": split_action_chunk(planned_chunk),
            "policy_chunk_summary": _chunk_summary(action_chunk),
            "planned_chunk_summary": _chunk_summary(planned_chunk),
        },
    )


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    logging.info("Connecting to policy server at %s:%s", args.host, args.port)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()

    if args.print_metadata:
        print(json.dumps(json_safe(metadata), indent=2))

    warnings = validate_realman_server_metadata(metadata)
    for warning in warnings:
        logging.warning(warning)
    if warnings and not args.allow_metadata_mismatch:
        raise RuntimeError(
            "Server metadata does not match Realman source-action setup. "
            "Use --allow-metadata-mismatch only for debugging."
        )

    action_stats = resolve_action_stats(metadata, args.unnorm_key)
    try:
        state_stats = resolve_state_stats(metadata, args.unnorm_key)
    except (KeyError, ValueError):
        state_stats = None
    action_norm_mode = resolve_norm_mode(metadata, "action", args.action_norm_mode)
    state_norm_mode = resolve_norm_mode(metadata, "state", args.state_norm_mode)

    action_horizon = int(metadata["action_horizon"])
    if int(metadata["action_dim"]) not in REALMAN_POLICY_ACTION_DIMS or int(metadata["state_dim"]) != REALMAN_STATE_DIM:
        raise RuntimeError(
            f"Expected Realman action/state dims {REALMAN_POLICY_ACTION_DIMS}/{REALMAN_STATE_DIM}, "
            f"got {metadata.get('action_dim')}/{metadata.get('state_dim')}."
        )
    if args.action_index < 0 or args.action_index >= action_horizon:
        raise ValueError(f"--action-index must be in [0, {action_horizon - 1}], got {args.action_index}.")
    if args.chunk_size <= 0 or args.chunk_size > action_horizon:
        raise ValueError(f"--chunk-size must be in [1, {action_horizon}], got {args.chunk_size}.")

    logging.info(
        "Connected. action_dim=%s state_dim=%s horizon=%s norm(action=%s,state=%s) cameras=%s actions=%s",
        metadata.get("action_dim"),
        metadata.get("state_dim"),
        action_horizon,
        action_norm_mode,
        state_norm_mode,
        list(REALMAN_CAMERA_ORDER),
        list(REALMAN_ACTION_NAMES),
    )

    if args.check_only:
        logging.info("Realman server metadata validated. Exiting.")
        client.close()
        return

    robot = _load_robot(args.robot_module, args)
    if robot is not None and hasattr(robot, "connect"):
        robot.connect()

    try:
        period_s = 1.0 / float(args.fps)
        step_index = 0
        replan_index = 0

        while step_index < int(args.num_steps):
            if args.live and args.confirm_each_replan:
                input(
                    "Press Enter to capture a fresh Realman observation and execute the next "
                    f"{min(args.chunk_size, int(args.num_steps) - step_index)} action(s), or Ctrl-C to abort..."
                )

            observation = _capture_observation(robot, args.observation_npz)
            normalized_chunk, action_chunk, latency_ms, _ = _infer_once(
                client=client,
                observation=observation,
                instruction=args.instruction,
                action_stats=action_stats,
                state_stats=state_stats,
                action_norm_mode=action_norm_mode,
                state_norm_mode=state_norm_mode,
                image_size=args.image_size,
            )

            execution_start = min(args.action_index, action_chunk.shape[0] - 1)
            execution_chunk = action_chunk[execution_start:]
            chunk_len = min(args.chunk_size, int(args.num_steps) - step_index)
            planned_chunk = np.asarray(execution_chunk[:chunk_len], dtype=np.float32)
            _log_policy_chunk(
                log_path=args.log_path,
                step_index=step_index,
                replan_index=replan_index,
                instruction=args.instruction,
                latency_ms=latency_ms,
                metadata=metadata,
                args=args,
                execution_start=execution_start,
                normalized_chunk=normalized_chunk,
                action_chunk=action_chunk,
                planned_chunk=planned_chunk,
            )

            next_deadline = time.perf_counter()
            for chunk_offset, action in enumerate(planned_chunk):
                sent_action = None
                if args.live and step_index >= int(args.warmup_steps):
                    sent_action = _send_action(robot, action, args.send_format)

                summary = action_summary(action)
                logging.info(
                    "step=%s replan=%s chunk=%s latency_ms=%.1f live=%s action_summary=%s",
                    step_index,
                    replan_index,
                    chunk_offset,
                    latency_ms,
                    args.live and step_index >= int(args.warmup_steps),
                    summary,
                )

                if args.log_path is not None:
                    write_jsonl(
                        args.log_path,
                        {
                            "event": "sent_action",
                            "step_index": step_index,
                            "replan_index": replan_index,
                            "chunk_offset": chunk_offset,
                            "policy_horizon_index": execution_start + chunk_offset,
                            "instruction": args.instruction,
                            "latency_ms": latency_ms,
                            "fps": args.fps,
                            "chunk_size": args.chunk_size,
                            "live": args.live and step_index >= int(args.warmup_steps),
                            "send_format": args.send_format,
                            "action_names": list(REALMAN_ACTION_NAMES),
                            "action": action,
                            "action_named": _named_action_vector(action),
                            "action_split": split_action_vector(action),
                            "action_summary": summary,
                            "sent_action": sent_action,
                        },
                    )

                step_index += 1
                if step_index >= int(args.num_steps) or chunk_offset == len(planned_chunk) - 1:
                    break

                next_deadline += period_s
                sleep_s = next_deadline - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    logging.warning("Realman rollout loop is behind by %.1f ms", abs(sleep_s) * 1000.0)
                    next_deadline = time.perf_counter()

            replan_index += 1
            if args.observation_npz is not None:
                break

    finally:
        client.close()
        if robot is not None and hasattr(robot, "disconnect"):
            robot.disconnect()


if __name__ == "__main__":
    main(build_argparser().parse_args())
