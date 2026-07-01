from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.realman.pipeline import (
    DEFAULT_IMAGE_SIZE,
    REALMAN_ACTION_DIM,
    REALMAN_ACTION_NAMES,
    action_summary,
    expand_policy_action_to_robot_action,
    extract_ordered_images,
    extract_state_vector,
    split_action_vector,
    write_jsonl,
)
from deployment.realman.run_realman_policy import (
    _capture_observation,
    _load_robot,
    _named_action_vector,
    _send_action,
)


DEFAULT_SERVER_ADDRESS = "tcp://192.168.10.29:5556"
DEFAULT_INSTRUCTION = "Pick the chain from bin and place it in the rig"
OPENPI_REALMAN_ACTION_DIM = 18


class OpenPiZmqPolicyClient:
    def __init__(self, address: str, timeout_ms: int | None = None) -> None:
        self.address = address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        if timeout_ms is not None:
            self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
            self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.connect(address)

    def request_action(self, observation: dict[str, Any]) -> np.ndarray:
        self.socket.send(pickle.dumps(observation, protocol=pickle.HIGHEST_PROTOCOL))
        response = pickle.loads(self.socket.recv())
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI ZMQ server returned error: {response}")
        return _to_numpy(response).astype(np.float32, copy=False).reshape(-1)

    def reset_policy(self) -> np.ndarray:
        payload = {
            "rest_policy": np.asarray([1], dtype=np.int32),
            "observation.state": np.zeros((OPENPI_REALMAN_ACTION_DIM,), dtype=np.float32),
        }
        return self.request_action(payload)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Realman VR teleop stack against an OpenPI ZMQ policy server."
    )
    parser.add_argument("--server-address", default=DEFAULT_SERVER_ADDRESS)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--num-steps", type=int, default=0, help="Number of actions to send; 0 means run forever.")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--observation-npz", type=str, default=None)
    parser.add_argument("--robot-module", type=str, default="deployment.realman.vr_teleop_bridge:create_robot")
    parser.add_argument("--send-format", choices=("vector", "split"), default="vector")
    parser.add_argument("--live", dest="live", action="store_true")
    parser.add_argument("--no-live", dest="live", action="store_false")
    parser.set_defaults(live=True)
    parser.add_argument("--confirm-before-start", dest="confirm_before_start", action="store_true")
    parser.add_argument("--no-confirm-before-start", dest="confirm_before_start", action="store_false")
    parser.set_defaults(confirm_before_start=True)
    parser.add_argument("--reset-policy-first", action="store_true")
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")

    teleop = parser.add_argument_group("Yondu VR teleop bridge")
    teleop.add_argument(
        "--teleop-root",
        default=None,
        help="Path to YonduAI/yondu-vr-teleop checkout. Defaults to $YONDU_VR_TELEOP_ROOT or common local paths.",
    )
    teleop.add_argument("--teleop-camera-labels", default="head,left,right")
    teleop.add_argument("--teleop-rgb-downsample-stride", type=int, default=1)
    teleop.add_argument("--teleop-local-rgb-max-age-s", type=float, default=1.0)
    teleop.add_argument("--teleop-max-camera-age-s", type=float, default=0.1)
    teleop.add_argument("--teleop-state-reader-hz", type=float, default=30.0)
    teleop.add_argument("--teleop-state-cache-max-age-s", type=float, default=0.10)
    teleop.add_argument("--teleop-wait-for-state-s", type=float, default=3.0)
    state_reader = teleop.add_mutually_exclusive_group()
    state_reader.add_argument(
        "--teleop-background-state-reader",
        dest="teleop_background_state_reader",
        action="store_true",
    )
    state_reader.add_argument(
        "--no-teleop-background-state-reader",
        dest="teleop_background_state_reader",
        action="store_false",
    )
    parser.set_defaults(teleop_background_state_reader=True)

    teleop.add_argument("--teleop-simulation", action="store_true")
    teleop.add_argument("--teleop-view", action="store_true")
    teleop.add_argument("--teleop-disable-base", action="store_true")
    teleop.add_argument("--teleop-disable-lift", action="store_true")
    teleop.add_argument("--teleop-disable-head", action="store_true")
    teleop.add_argument("--teleop-arm-follow-mode", choices=("low", "high"), default="low")
    teleop.add_argument("--teleop-head-port", default="/dev/ttyUSB0")
    teleop.add_argument("--teleop-head-baud", type=int, default=9600)
    teleop.add_argument("--teleop-sim-hz", type=float, default=100.0)
    teleop.add_argument("--teleop-control-hz", type=float, default=100.0)
    teleop.add_argument("--teleop-deadman-grip-threshold", type=float, default=0.7)
    teleop.add_argument("--teleop-log-dir", default="logs")
    return parser


def build_openpi_observation(
    observation: dict[str, Any],
    *,
    instruction: str,
    image_size: int,
) -> dict[str, Any]:
    state = extract_state_vector(observation)
    head, wrist_left, wrist_right = extract_ordered_images(observation, image_size=image_size)
    return {
        "observation.state": np.ascontiguousarray(state.astype(np.float32, copy=False)),
        "observation.images.head": np.ascontiguousarray(head),
        "observation.images.wrist_left": np.ascontiguousarray(wrist_left),
        "observation.images.wrist_right": np.ascontiguousarray(wrist_right),
        "prompt": instruction,
    }


def openpi_action_to_robot_action(action: Any, observation: dict[str, Any]) -> np.ndarray:
    action_array = _to_numpy(action).astype(np.float32, copy=False).reshape(-1)
    if action_array.size == REALMAN_ACTION_DIM:
        return np.ascontiguousarray(action_array)
    if action_array.size == REALMAN_ACTION_DIM - 3:
        return expand_policy_action_to_robot_action(action_array)
    if action_array.size != OPENPI_REALMAN_ACTION_DIM:
        raise ValueError(
            f"OpenPI Realman action dim is {action_array.size}, expected {OPENPI_REALMAN_ACTION_DIM}."
        )

    state = extract_state_vector(observation)
    robot_action = np.zeros((REALMAN_ACTION_DIM,), dtype=np.float32)
    robot_action[:16] = action_array[:16]
    robot_action[19:21] = action_array[16:18]
    robot_action[21] = float(state[18])
    return np.ascontiguousarray(robot_action)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _should_continue(step_index: int, num_steps: int) -> bool:
    return int(num_steps) <= 0 or step_index < int(num_steps)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    logging.info("Connecting to OpenPI ZMQ policy server at %s", args.server_address)
    client = OpenPiZmqPolicyClient(args.server_address, timeout_ms=args.timeout_ms)

    robot = _load_robot(args.robot_module, args)
    if robot is not None and hasattr(robot, "connect"):
        robot.connect()

    try:
        if args.reset_policy_first:
            client.reset_policy()
            logging.info("Reset OpenPI server action queue")

        if args.live and args.confirm_before_start:
            input(
                "Press Enter to start sending OpenPI ZMQ policy actions to the Realman robot, "
                "or Ctrl-C to abort..."
            )

        period_s = 1.0 / max(float(args.fps), 1e-6)
        next_deadline = time.perf_counter()
        step_index = 0

        while _should_continue(step_index, args.num_steps):
            observation = _capture_observation(robot, args.observation_npz)
            request = build_openpi_observation(
                observation,
                instruction=args.instruction,
                image_size=args.image_size,
            )

            started = time.perf_counter()
            openpi_action = client.request_action(request)
            latency_ms = (time.perf_counter() - started) * 1000.0
            robot_action = openpi_action_to_robot_action(openpi_action, observation)

            sent_action = None
            is_live_step = bool(args.live) and step_index >= int(args.warmup_steps)
            if is_live_step:
                sent_action = _send_action(robot, robot_action, args.send_format)

            summary = action_summary(robot_action)
            logging.info(
                "step=%s latency_ms=%.1f live=%s openpi_dim=%s robot_action_summary=%s",
                step_index,
                latency_ms,
                is_live_step,
                int(openpi_action.size),
                summary,
            )

            if args.log_path is not None:
                write_jsonl(
                    args.log_path,
                    {
                        "event": "openpi_zmq_action",
                        "step_index": step_index,
                        "server_address": args.server_address,
                        "instruction": args.instruction,
                        "latency_ms": latency_ms,
                        "fps": args.fps,
                        "live": is_live_step,
                        "send_format": args.send_format,
                        "action_names": list(REALMAN_ACTION_NAMES),
                        "openpi_action": openpi_action,
                        "robot_action": robot_action,
                        "robot_action_named": _named_action_vector(robot_action),
                        "robot_action_split": split_action_vector(robot_action),
                        "robot_action_summary": summary,
                        "sent_action": sent_action,
                    },
                )

            step_index += 1
            if args.observation_npz is not None:
                break

            next_deadline += period_s
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                logging.warning("OpenPI ZMQ Realman loop is behind by %.1f ms", abs(sleep_s) * 1000.0)
                next_deadline = time.perf_counter()

    finally:
        client.close()
        if robot is not None and hasattr(robot, "disconnect"):
            robot.disconnect()


if __name__ == "__main__":
    main(build_argparser().parse_args())
