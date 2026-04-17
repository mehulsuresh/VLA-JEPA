from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from deployment.model_server.checkpoint_utils import resolve_policy_checkpoint
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.trossen.pipeline import (
    DEFAULT_CAMERA_ORDER,
    build_policy_payload,
    compute_action_state_stats_closeness,
    continuous_unnormalize,
    infer_action_mode_from_stats,
    resolve_action_stats,
    resolve_norm_mode,
    resolve_state_stats,
    validate_server_metadata,
)


@dataclass
class SampleMetric:
    episode: str
    frame_index: int
    latency_ms: float
    first_step_action_mae: float
    full_chunk_action_mae: float
    first_step_current_state_mae: float
    full_chunk_current_state_mae: float
    first_step_next_state_mae: float
    full_chunk_next_state_mae: float
    first_step_state_delta_mae: float
    full_chunk_state_delta_mae: float


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10103)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--frame-stride", type=int, default=40)
    parser.add_argument("--max-samples-per-episode", type=int, default=64)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def load_episode_metadata(episode_dir: Path) -> dict:
    with (episode_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_task_text(input_root: Path) -> str:
    tasks_path = input_root / "meta" / "tasks.jsonl"
    with tasks_path.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))
    return str(first["task"])


def load_frame_observation(episode_dir: Path, frame_index: int, state: np.ndarray) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {
        "observation.state": np.asarray(state, dtype=np.float32),
    }
    for camera_name in DEFAULT_CAMERA_ORDER:
        image_path = episode_dir / f"observation_images_{camera_name}_f{frame_index:06d}.jpg"
        with Image.open(image_path) as img:
            observation[f"observation.images.{camera_name}"] = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return observation


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main(args) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = resolve_policy_checkpoint(args.ckpt_path)
    input_root = Path(args.input_root).expanduser().resolve()
    instruction = load_task_text(input_root)

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

    env = dict(os.environ)
    env.pop("DEBUG", None)

    logging.info("Launching policy server for offline smoke test")
    with log_path.open("w", encoding="utf-8") as log_file:
        server = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

    sample_metrics: list[SampleMetric] = []
    episode_summaries: list[dict] = []
    try:
        client = WebsocketClientPolicy(host=args.host, port=args.port, timeout=args.timeout_s)
        metadata = client.get_server_metadata()
        action_stats = resolve_action_stats(metadata)
        try:
            state_stats = resolve_state_stats(metadata)
        except (KeyError, ValueError):
            state_stats = None
        action_norm_mode = resolve_norm_mode(metadata, "action")
        state_norm_mode = resolve_norm_mode(metadata, "state")
        inferred_action_mode = infer_action_mode_from_stats(action_stats, state_stats)
        stats_closeness = compute_action_state_stats_closeness(action_stats, state_stats)
        warnings = validate_server_metadata(metadata)
        for warning in warnings:
            logging.warning(warning)

        action_horizon = int(metadata["action_horizon"])
        episode_dirs = sorted(p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("episode_"))[: args.max_episodes]
        logging.info(
            "Running offline smoke test on %s episodes with stride=%s horizon=%s",
            len(episode_dirs),
            args.frame_stride,
            action_horizon,
        )

        for episode_dir in episode_dirs:
            metadata_json = load_episode_metadata(episode_dir)
            actions = np.asarray(metadata_json["frame_features"]["action"], dtype=np.float32)
            states = np.asarray(metadata_json["frame_features"]["observation.state"], dtype=np.float32)
            max_start = len(actions) - action_horizon
            candidate_indices = list(range(0, max_start, args.frame_stride))[: args.max_samples_per_episode]

            if not candidate_indices:
                logging.warning("Skipping %s because it is shorter than action horizon", episode_dir.name)
                continue

            episode_action_vs_next_state = []
            episode_action_vs_state_delta = []
            for idx in candidate_indices:
                gt_action_chunk = actions[idx : idx + action_horizon]
                gt_current_state_chunk = np.repeat(states[idx][None, :], action_horizon, axis=0)
                gt_next_state_chunk = states[idx + 1 : idx + 1 + action_horizon]
                gt_state_delta_chunk = states[idx + 1 : idx + 1 + action_horizon] - states[idx : idx + action_horizon]

                observation = load_frame_observation(
                    episode_dir=episode_dir,
                    frame_index=int(metadata_json["frame_indices"][idx]),
                    state=states[idx],
                )
                payload = build_policy_payload(
                    observation,
                    instruction=instruction,
                    state_stats=state_stats,
                    state_norm_mode=state_norm_mode,
                )
                started = time.perf_counter()
                response = client.infer(payload)
                latency_ms = (time.perf_counter() - started) * 1000.0

                if not response.get("ok", False):
                    raise RuntimeError(f"Inference failed for {episode_dir.name} frame {idx}: {response}")

                normalized_chunk = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)[0]
                pred_chunk = continuous_unnormalize(normalized_chunk, action_stats, mode=action_norm_mode)

                action_mae = np.abs(pred_chunk - gt_action_chunk)
                current_state_mae = np.abs(pred_chunk - gt_current_state_chunk)
                next_state_mae = np.abs(pred_chunk - gt_next_state_chunk)
                state_delta_mae = np.abs(pred_chunk - gt_state_delta_chunk)

                metric = SampleMetric(
                    episode=episode_dir.name,
                    frame_index=int(metadata_json["frame_indices"][idx]),
                    latency_ms=float(latency_ms),
                    first_step_action_mae=float(action_mae[0].mean()),
                    full_chunk_action_mae=float(action_mae.mean()),
                    first_step_current_state_mae=float(current_state_mae[0].mean()),
                    full_chunk_current_state_mae=float(current_state_mae.mean()),
                    first_step_next_state_mae=float(next_state_mae[0].mean()),
                    full_chunk_next_state_mae=float(next_state_mae.mean()),
                    first_step_state_delta_mae=float(state_delta_mae[0].mean()),
                    full_chunk_state_delta_mae=float(state_delta_mae.mean()),
                )
                sample_metrics.append(metric)
                episode_action_vs_next_state.append(float(np.abs(gt_action_chunk - gt_next_state_chunk).mean()))
                episode_action_vs_state_delta.append(float(np.abs(gt_action_chunk - gt_state_delta_chunk).mean()))

            episode_summaries.append(
                {
                    "episode": episode_dir.name,
                    "num_samples": len(candidate_indices),
                    "action_vs_next_state_mae": summarize(episode_action_vs_next_state),
                    "action_vs_state_delta_mae": summarize(episode_action_vs_state_delta),
                }
            )

        client.close()

    finally:
        server.terminate()
        try:
            server.wait(timeout=max(5.0, args.timeout_s / 10.0))
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)

    summary = {
        "checkpoint": str(ckpt_path),
        "input_root": str(input_root),
        "instruction": instruction,
        "server_log": str(log_path),
        "server_action_type": metadata.get("action_type"),
        "action_norm_mode": action_norm_mode,
        "state_norm_mode": state_norm_mode,
        "inferred_action_mode": inferred_action_mode,
        "action_state_stats_closeness": stats_closeness,
        "num_samples": len(sample_metrics),
        "latency_ms": summarize([m.latency_ms for m in sample_metrics]),
        "first_step_action_mae": summarize([m.first_step_action_mae for m in sample_metrics]),
        "full_chunk_action_mae": summarize([m.full_chunk_action_mae for m in sample_metrics]),
        "first_step_current_state_mae": summarize([m.first_step_current_state_mae for m in sample_metrics]),
        "full_chunk_current_state_mae": summarize([m.full_chunk_current_state_mae for m in sample_metrics]),
        "first_step_next_state_mae": summarize([m.first_step_next_state_mae for m in sample_metrics]),
        "full_chunk_next_state_mae": summarize([m.full_chunk_next_state_mae for m in sample_metrics]),
        "first_step_state_delta_mae": summarize([m.first_step_state_delta_mae for m in sample_metrics]),
        "full_chunk_state_delta_mae": summarize([m.full_chunk_state_delta_mae for m in sample_metrics]),
        "episode_summaries": episode_summaries,
        "sample_metrics_head": [asdict(m) for m in sample_metrics[:10]],
    }

    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    main(parser.parse_args())
