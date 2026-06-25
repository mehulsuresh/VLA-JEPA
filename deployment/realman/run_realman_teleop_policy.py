from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.realman.run_realman_policy import build_argparser, main


def parse_args():
    parser = build_argparser()
    parser.description = "Run a Realman VLA-JEPA policy through the Yondu VR teleop robot stack."
    parser.set_defaults(
        robot_module="deployment.realman.vr_teleop_bridge:create_robot",
        send_format="vector",
        live=True,
    )

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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
