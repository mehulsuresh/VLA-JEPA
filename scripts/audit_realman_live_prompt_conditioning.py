#!/usr/bin/env python3
"""Run a matched-noise left/right prompt intervention on a saved live request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.realman.pipeline import validate_realman_policy_payload
from deployment.trossen.pipeline import resolve_action_stats, resolve_norm_mode
from deployment.realman.pipeline import realman_continuous_unnormalize


LEFT_SUFFIX = "Reach into the left bin and raise the chain out of the bin"
RIGHT_SUFFIX = "Reach into the right bin and raise the chain out of the bin"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--compare-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fixture = np.load(args.fixture, allow_pickle=False)
    frames = np.ascontiguousarray(fixture["qwen_frames"], dtype=np.uint8)
    state = np.ascontiguousarray(fixture["state"], dtype=np.float32)
    base_instruction = str(fixture["base_instruction"].item())

    client = WebsocketClientPolicy(host=args.host, port=args.port, timeout=60)
    try:
        metadata = client.get_server_metadata()
        action_stats = resolve_action_stats(metadata, None)
        action_norm_mode = resolve_norm_mode(metadata, "action", "auto")
        conditions = {
            "left": f"{base_instruction} | {LEFT_SUFFIX}",
            "right": f"{base_instruction} | {RIGHT_SUFFIX}",
            "base_only": base_instruction,
        }
        outputs: dict[str, np.ndarray] = {}
        for name, instruction in conditions.items():
            payload = {
                "qwen_frames": frames,
                "state": state,
                "instructions": [instruction],
                "inference_seed": int(args.seed),
            }
            validate_realman_policy_payload(payload, metadata)
            response = client.infer(payload)
            if not response.get("ok", False):
                raise RuntimeError(f"{name} inference failed: {response}")
            normalized = np.asarray(
                response["data"]["normalized_actions"], dtype=np.float32
            )[0]
            outputs[name] = realman_continuous_unnormalize(
                normalized,
                action_stats,
                mode=action_norm_mode,
            )
    finally:
        client.close()

    horizon = min(args.compare_window, outputs["left"].shape[0])
    arm_dims = np.asarray(tuple(range(7)) + tuple(range(8, 15)))
    left_dims = np.arange(0, 7)
    right_dims = np.arange(8, 15)

    def effect(first: str, second: str, dims: np.ndarray) -> float:
        return float(
            np.abs(
                outputs[first][:horizon, dims]
                - outputs[second][:horizon, dims]
            ).mean()
        )

    def magnitude(name: str, dims: np.ndarray) -> float:
        return float(np.abs(outputs[name][:horizon, dims]).mean())

    result = {
        "schema_version": 1,
        "fixture": str(args.fixture.resolve()),
        "checkpoint_path": metadata.get("checkpoint_path"),
        "compare_window": horizon,
        "seed": args.seed,
        "base_instruction": base_instruction,
        "left_vs_right_arm_delta_mae_rad": effect(
            "left", "right", arm_dims
        ),
        "left_vs_right_left_arm_delta_mae_rad": effect(
            "left", "right", left_dims
        ),
        "left_vs_right_right_arm_delta_mae_rad": effect(
            "left", "right", right_dims
        ),
        "left_prompt_left_arm_abs_delta_mean_rad": magnitude(
            "left", left_dims
        ),
        "left_prompt_right_arm_abs_delta_mean_rad": magnitude(
            "left", right_dims
        ),
        "right_prompt_left_arm_abs_delta_mean_rad": magnitude(
            "right", left_dims
        ),
        "right_prompt_right_arm_abs_delta_mean_rad": magnitude(
            "right", right_dims
        ),
        "left_vs_base_arm_delta_mae_rad": effect(
            "left", "base_only", arm_dims
        ),
        "right_vs_base_arm_delta_mae_rad": effect(
            "right", "base_only", arm_dims
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
