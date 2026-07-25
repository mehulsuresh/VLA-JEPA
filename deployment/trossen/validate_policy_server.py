from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from deployment.model_server.checkpoint_utils import resolve_policy_checkpoint
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.trossen.pipeline import DEFAULT_CAMERA_ORDER


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10096)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser


def main(args) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = resolve_policy_checkpoint(args.ckpt_path)

    with tempfile.NamedTemporaryFile(prefix="trossen_policy_server_", suffix=".log", delete=False) as tmp_log:
        log_path = Path(tmp_log.name)

    cmd = [
        args.python,
        "deployment/model_server/server_policy.py",
        "--ckpt_path",
        str(ckpt_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--cuda",
        str(args.cuda),
    ]
    if args.use_bf16:
        cmd.append("--use_bf16")

    logging.info("Launching policy server: %s", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log_file:
        env = dict(os.environ)
        env.pop("DEBUG", None)
        server = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

    try:
        client = WebsocketClientPolicy(host=args.host, port=args.port, timeout=args.timeout_s)
        metadata = client.get_server_metadata()
        logging.info("Connected to server. action_dim=%s state_dim=%s", metadata.get("action_dim"), metadata.get("state_dim"))

        image_size = int(metadata.get("resolution_size") or 224)
        state_dim = int(metadata["state_dim"])
        action_dim = int(metadata["action_dim"])
        action_horizon = int(metadata["action_horizon"])

        rng = np.random.default_rng(42)
        payload = {
            "batch_images": [[
                rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
                for _ in DEFAULT_CAMERA_ORDER
            ]],
            "instructions": ["Smoke-test the policy server."],
            "state": np.zeros((1, 1, state_dim), dtype=np.float32),
        }
        started = time.perf_counter()
        warmup_response = client.infer(payload)
        warmup_elapsed_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        response = client.infer(payload)
        steady_elapsed_ms = (time.perf_counter() - started) * 1000.0
        client.close()

        if not warmup_response.get("ok", False):
            raise RuntimeError(f"Warmup inference failed: {warmup_response}")
        if not response.get("ok", False):
            raise RuntimeError(f"Server returned an error payload: {response}")

        normalized_actions = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)
        if normalized_actions.shape != (1, action_horizon, action_dim):
            raise RuntimeError(
                "Unexpected action tensor shape "
                f"{normalized_actions.shape}; expected {(1, action_horizon, action_dim)}"
            )
        if not np.isfinite(normalized_actions).all():
            raise RuntimeError("Policy server returned non-finite actions.")

        logging.info(
            "Smoke test passed. warmup_ms=%.1f steady_ms=%.1f",
            warmup_elapsed_ms,
            steady_elapsed_ms,
        )
        print(
            f"ok action_shape={normalized_actions.shape} warmup_ms={warmup_elapsed_ms:.1f} "
            f"steady_ms={steady_elapsed_ms:.1f} "
            f"log={log_path}"
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=max(5.0, args.timeout_s / 10.0))
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    main(parser.parse_args())
