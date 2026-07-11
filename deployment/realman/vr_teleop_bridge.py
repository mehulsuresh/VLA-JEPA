from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from deployment.realman.pipeline import (
    REALMAN_ACTION_DIM,
    REALMAN_STATE_DIM,
    expand_policy_action_to_robot_action,
)


_DEFAULT_CAMERA_LABELS = ("head", "left", "right")
_TELEOP_ROOT_ENV = "YONDU_VR_TELEOP_ROOT"


def create_robot(args: Any | None = None) -> "RealmanVrTeleopPolicyRobot":
    """Factory used by run_realman_policy.py --robot-module."""
    return RealmanVrTeleopPolicyRobot(args)


def resolve_teleop_root(root: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if root:
        candidates.append(Path(root).expanduser())
    env_root = os.environ.get(_TELEOP_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path("/home/mehul/work/yondu-vr-teleop"),
            Path("/home/mehul/work/vr_teleop"),
            Path("/tmp/yondu-vr-teleop-data-collection-2"),
        ]
    )

    for candidate in candidates:
        if (candidate / "robot_unified_teleop.py").is_file():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find Yondu VR teleop repo. Pass --teleop-root or set "
        f"{_TELEOP_ROOT_ENV}. Searched: {searched}"
    )


def install_teleop_import_paths(root: str | Path) -> Path:
    repo_root = Path(root).expanduser().resolve()
    wrapper_root = repo_root / "yondu-realman-lerobot"
    for path in (repo_root, wrapper_root):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


def model_compatible_observation(
    observation: dict[str, Any],
    *,
    model_state_dim: int = REALMAN_STATE_DIM,
) -> dict[str, Any]:
    """Keep collected observation keys and add the trained source-state key.

    The current data-collection branch may append two read-only arm-current
    telemetry values to ``observation.state``. The deployed Realman checkpoint
    was trained on the first 19 state values, so the policy payload consumes
    ``source.observation.state`` with that trained schema.
    """
    if "observation.state" not in observation and "source.observation.state" in observation:
        return observation

    raw_state = np.asarray(observation["observation.state"], dtype=np.float32).reshape(-1)
    if raw_state.size < int(model_state_dim):
        raise ValueError(
            f"Teleop observation.state has dim {raw_state.size}, expected at least {model_state_dim}."
        )

    output = dict(observation)
    output["source.observation.state"] = np.ascontiguousarray(raw_state[: int(model_state_dim)])
    return output


def vector_from_action_payload(
    action: Any,
    *,
    lift_height_mm: float | None = None,
) -> np.ndarray:
    if isinstance(action, dict):
        if "vector" in action:
            vector = action["vector"]
        else:
            vector = np.empty((REALMAN_ACTION_DIM,), dtype=np.float32)
            vector[0:7] = np.asarray(action["left_arm_joints"], dtype=np.float32).reshape(7)
            vector[7] = float(action["left_gripper"])
            vector[8:15] = np.asarray(action["right_arm_joints"], dtype=np.float32).reshape(7)
            vector[15] = float(action["right_gripper"])
            base = action["base_velocity"]
            vector[16] = float(base["linear_x_mps"])
            vector[17] = float(base["linear_y_mps"])
            vector[18] = float(base["angular_z_radps"])
            vector[19:21] = np.asarray(action["head_joints"], dtype=np.float32).reshape(2)
            vector[21] = float(action["lift_height_mm"])
            return vector
    else:
        vector = action

    arr = expand_policy_action_to_robot_action(
        np.asarray(vector, dtype=np.float32).reshape(-1),
        lift_height_mm=lift_height_mm,
    ).reshape(-1)
    if arr.size != REALMAN_ACTION_DIM:
        raise ValueError(f"Realman policy action has dim {arr.size}, expected {REALMAN_ACTION_DIM}.")
    return np.ascontiguousarray(arr)


class RealmanVrTeleopPolicyRobot:
    """Policy adapter backed by the same teleop collection/send code.

    Observation path:
      RealmanRobot.capture_observation() -> collection-style images/state
      -> source.observation.state trimmed to the trained 19D schema.

    Action path:
      policy 22D vector -> RealmanRobot.send_action()
      -> teleop session arm/base/head/lift hardware methods.
    """

    def __init__(self, args: Any | None = None) -> None:
        self.args = args or SimpleNamespace()
        self.log = logging.getLogger("realman-vr-teleop-policy")
        self.teleop_root: Path | None = None
        self._last_lift_height_mm: float | None = None
        self.teleop_args: SimpleNamespace | None = None
        self.session: Any | None = None
        self.robot: Any | None = None
        self.frame_source: Any | None = None
        self.base_drive: Any | None = None
        self.lift_drive: Any | None = None
        self.head: Any | None = None
        self.head_servo_controller: Any | None = None
        self.is_connected = False

    def connect(self) -> None:
        if self.is_connected:
            return

        self.teleop_root = install_teleop_import_paths(
            resolve_teleop_root(_arg(self.args, "teleop_root", None))
        )
        self.log.info("Using Yondu VR teleop repo at %s", self.teleop_root)

        from realman_lerobot.cameras import DownsampledFrameSource, LocalRgbFrameSource
        from realman_lerobot.realman_robot import RealmanRobot
        from robot_control import RobotControlSession
        from robot_control.base import BaseDriveController
        from robot_control.head import HeadController, HeadServoController
        from robot_control.lift import LiftDriveController
        from robot_unified_teleop import _configure_realman_gripper_env, _runtime_defaults

        teleop_args = self._build_teleop_args(_runtime_defaults())
        self.teleop_args = teleop_args
        _configure_realman_gripper_env(teleop_args, self.log)
        self._configure_ik_model_env()

        if not bool(getattr(teleop_args, "disable_base", False)):
            self.base_drive = BaseDriveController(teleop_args, self.log.getChild("base"))
            self.base_drive.start()

        if not bool(getattr(teleop_args, "disable_lift", False)):
            self.lift_drive = LiftDriveController(teleop_args, self.log.getChild("lift"))
            self.lift_drive.start()

        if not bool(getattr(teleop_args, "disable_head", False)):
            try:
                self.head = HeadController(port=teleop_args.port, baudrate=teleop_args.baud)
                self.head.connect()
                self.head_servo_controller = HeadServoController(
                    get_args=lambda: teleop_args,
                    get_head=lambda: self.head,
                    log=self.log.getChild("head"),
                )
            except Exception:
                self.log.exception("Head controller failed to connect; continuing without head servo control")

        self.session = RobotControlSession(
            simulation=bool(teleop_args.simulation),
            enable_drag_targets=bool(getattr(teleop_args, "view", False)),
            viewer=bool(getattr(teleop_args, "view", False)),
            dt=1.0 / max(float(teleop_args.sim_hz), 1.0),
            deadman_grip_threshold=float(teleop_args.deadman_grip_threshold),
            arm_process_owns_hardware=False,
        )
        self.session.initialize()
        self.session.start()
        setattr(self.session, "head_servo_controller", self.head_servo_controller)
        self.session.configure_control_pipeline(
            teleop_args,
            log=self.log,
            base_drive_controller=self.base_drive,
            lift_drive_controller=self.lift_drive,
        )
        self._configure_arm_stream()

        labels = _parse_csv(_arg(self.args, "teleop_camera_labels", ",".join(_DEFAULT_CAMERA_LABELS)))
        self.frame_source = LocalRgbFrameSource(
            labels=tuple(labels),
            max_frame_age_s=float(_arg(self.args, "teleop_local_rgb_max_age_s", 1.0)),
        )
        stride = int(_arg(self.args, "teleop_rgb_downsample_stride", 1))
        if stride > 1:
            self.frame_source = DownsampledFrameSource(self.frame_source, stride=stride)

        self.robot = RealmanRobot(
            session=self.session,
            frame_source=self.frame_source,
            include_grippers=True,
            require_cameras=True,
            max_camera_sample_age_s=float(_arg(self.args, "teleop_max_camera_age_s", 0.1)),
        )
        self.robot.connect()

        if bool(_arg(self.args, "teleop_background_state_reader", True)):
            self.robot.start_background_state_reader(
                target_hz=float(_arg(self.args, "teleop_state_reader_hz", 30.0)),
                max_age_s=float(_arg(self.args, "teleop_state_cache_max_age_s", 0.10)),
            )

        wait_s = float(_arg(self.args, "teleop_wait_for_state_s", 3.0))
        if wait_s > 0.0 and not self.robot.wait_for_measured_state(timeout_s=wait_s):
            raise RuntimeError(f"Timed out waiting {wait_s:.1f}s for measured Realman arm state")

        self.is_connected = True

    def capture_observation(self) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("RealmanVrTeleopPolicyRobot is not connected")
        observation = model_compatible_observation(self.robot.capture_observation())
        self._last_lift_height_mm = float(
            np.asarray(observation["source.observation.state"], dtype=np.float32).reshape(-1)[-1]
        )
        return observation

    def send_action(self, action: Any) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("RealmanVrTeleopPolicyRobot is not connected")
        vector = vector_from_action_payload(
            action,
            lift_height_mm=self._last_lift_height_mm,
        )
        returned = self.robot.send_action(vector)
        return np.asarray(returned, dtype=np.float32)

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:
                self.log.exception("Failed to disconnect RealmanRobot adapter")
        for obj, name in (
            (self.frame_source, "frame source"),
            (self.base_drive, "base drive"),
            (self.lift_drive, "lift drive"),
            (self.head, "head controller"),
            (self.session, "arm session"),
        ):
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    self.log.exception("Failed to close %s", name)
        self.is_connected = False

    def _build_teleop_args(self, defaults: dict[str, Any]) -> SimpleNamespace:
        values = dict(defaults)
        values.update(
            {
                "simulation": bool(_arg(self.args, "teleop_simulation", False)),
                "view": bool(_arg(self.args, "teleop_view", False)),
                "headless": not bool(_arg(self.args, "teleop_view", False)),
                "disable_arms": False,
                "disable_base": bool(_arg(self.args, "teleop_disable_base", False)),
                "disable_lift": bool(_arg(self.args, "teleop_disable_lift", False)),
                "disable_head": bool(_arg(self.args, "teleop_disable_head", False)),
                "disable_stream_control": True,
                "disable_camera_pose_telemetry": True,
                "dashboard": False,
                "dashboard_direct_hardware": False,
                "dashboard_datachannel": False,
                "datachannel_control": False,
                "arm_ik_process": False,
                "arm_process_owns_hardware": False,
                "arm_follow_mode": str(_arg(self.args, "teleop_arm_follow_mode", defaults.get("arm_follow_mode", "low"))),
                "port": str(_arg(self.args, "teleop_head_port", defaults.get("port", "/dev/ttyUSB0"))),
                "baud": int(_arg(self.args, "teleop_head_baud", defaults.get("baud", 9600))),
                "sim_hz": float(_arg(self.args, "teleop_sim_hz", defaults.get("sim_hz", 100.0))),
                "control_hz": float(_arg(self.args, "teleop_control_hz", defaults.get("control_hz", 100.0))),
                "deadman_grip_threshold": float(
                    _arg(self.args, "teleop_deadman_grip_threshold", defaults.get("deadman_grip_threshold", 0.7))
                ),
                "log_dir": str(_arg(self.args, "teleop_log_dir", defaults.get("log_dir", "logs"))),
            }
        )
        return SimpleNamespace(**values)

    def _configure_ik_model_env(self) -> None:
        if os.environ.get("ROBOT_IK_MJCF") or self.teleop_root is None:
            return
        override = self.teleop_root / "robot_description" / "robot_mjcf_full_box_collision.xml"
        if override.is_file():
            os.environ["ROBOT_IK_MJCF"] = str(override)

    def _configure_arm_stream(self) -> None:
        if self.session is None or self.teleop_args is None:
            return
        configure_stream = getattr(getattr(self.session, "arm_hardware", None), "configure_stream", None)
        if not callable(configure_stream):
            return
        follow_mode = str(getattr(self.teleop_args, "arm_follow_mode", "low")).strip().lower()
        high_follow = follow_mode == "high"
        configure_stream(
            robot_left=getattr(self.session, "robot_left", None),
            robot_right=getattr(self.session, "robot_right", None),
            follow=high_follow,
            trajectory_mode=int(getattr(self.teleop_args, "arm_high_trajectory_mode", 1)) if high_follow else 0,
            radio=int(getattr(self.teleop_args, "arm_high_radio", 50)) if high_follow else 0,
        )


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _parse_csv(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)
    labels = [str(value).strip() for value in values if str(value).strip()]
    if not labels:
        raise ValueError("At least one camera label is required")
    return labels
