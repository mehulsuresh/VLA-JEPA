#!/usr/bin/env python3
"""Watch a TensorBoard-backed training run until a target step is reached."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


SCALAR_TAGS = (
    "total_loss",
    "action_loss",
    "wm_loss",
    "depth_teacher_loss",
    "rtc_training_probability",
    "samples_per_sec",
    "avg_samples_per_sec",
    "wall_step_time",
    "data_time",
    "cuda_memory_allocated_gb",
    "cuda_max_memory_allocated_gb",
)

RED_FLAG_RE = re.compile(
    r"Traceback|RuntimeError|CUDA out of memory|NCCL|Watchdog|timeout|\\bnan\\b|\\binf\\b",
    re.IGNORECASE,
)


def _fetch_scalars(tensorboard_url: str, run: str, tag: str) -> list[list[float]]:
    query = urllib.parse.urlencode({"run": run, "tag": tag})
    url = f"{tensorboard_url.rstrip('/')}/data/plugin/scalars/scalars?{query}"
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read().decode("utf-8")
    return json.loads(body or "[]")


def latest_scalars(tensorboard_url: str, run: str) -> tuple[int, dict[str, float]]:
    latest_step = 0
    metrics: dict[str, float] = {}
    for tag in SCALAR_TAGS:
        try:
            values = _fetch_scalars(tensorboard_url, run, tag)
        except Exception:
            continue
        if not values:
            continue
        _, step, value = values[-1]
        step_i = int(step)
        latest_step = max(latest_step, step_i)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        metrics[tag] = value_f
    return latest_step, metrics


def process_count(run_id: str) -> int:
    try:
        output = subprocess.check_output(["pgrep", "-af", run_id], text=True)
    except subprocess.CalledProcessError:
        return 0
    return sum(
        1
        for line in output.splitlines()
        if "train_starvla.py" in line or "accelerate launch" in line
    )


def gpu_snapshot() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return f"gpu_snapshot_error={exc}"
    compact = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            compact.append(f"{parts[0]}:{parts[1]}MiB/{parts[2]}%")
    return "gpu=" + " ".join(compact)


def checkpoint_status(run_dir: Path, target_step: int) -> tuple[int, bool]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return 0, False
    checkpoints = [path for path in checkpoint_dir.iterdir() if path.is_dir()]
    target = checkpoint_dir / f"steps_{target_step}"
    return len(checkpoints), target.is_dir() and (target / "trainer_state.json").exists()


def recent_red_flags(train_log: Path, bytes_to_read: int = 256_000) -> list[str]:
    if not train_log.exists():
        return []
    with train_log.open("rb") as handle:
        try:
            handle.seek(-bytes_to_read, os.SEEK_END)
        except OSError:
            handle.seek(0)
        text = handle.read().decode("utf-8", errors="replace")
    hits = []
    for line in text.splitlines():
        if not RED_FLAG_RE.search(line):
            continue
        # PyAV recovery messages include an underlying decoder error, but they
        # are the intended recovery path for this dataset.
        if "Canonical PyAV recovered corrupt video frames" in line:
            continue
        if "Further PyAV recovery warnings are suppressed" in line:
            continue
        if "Initializing TorchBackend in DeepSpeed with backend nccl" in line:
            continue
        if "Distributed environment: DistributedType.DEEPSPEED  Backend: nccl" in line:
            continue
        hits.append(line[-500:])
    return hits[-5:]


def format_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for tag in SCALAR_TAGS:
        if tag not in metrics:
            continue
        value = metrics[tag]
        if math.isfinite(value):
            parts.append(f"{tag}={value:.6g}")
        else:
            parts.append(f"{tag}={value}")
    return " ".join(parts) if parts else "metrics=unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--train-log", required=True, type=Path)
    parser.add_argument("--target-step", type=int, default=1000)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--checkpoint-grace-minutes", type=float, default=45.0)
    parser.add_argument("--tensorboard-url", default="http://127.0.0.1:6006")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = f"{args.run_id}/starvla"
    reached_target_at: float | None = None
    print(
        f"watcher start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
        f"run={args.run_id} target_step={args.target_step}",
        flush=True,
    )
    while True:
        step, metrics = latest_scalars(args.tensorboard_url, run)
        procs = process_count(args.run_id)
        ckpt_count, target_ckpt_ready = checkpoint_status(args.run_dir, args.target_step)
        flags = recent_red_flags(args.train_log)
        status = (
            f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] "
            f"step={step} procs={procs} ckpts={ckpt_count} "
            f"target_ckpt_ready={target_ckpt_ready} "
            f"{format_metrics(metrics)} {gpu_snapshot()}"
        )
        print(status, flush=True)
        if flags:
            print("recent_red_flags=" + " || ".join(flags), flush=True)

        if procs < 2:
            print("watcher stop: training process count dropped", flush=True)
            return 2

        if step >= args.target_step:
            if reached_target_at is None:
                reached_target_at = time.time()
                print("target step reached; waiting for checkpoint/eval boundary to settle", flush=True)
            if target_ckpt_ready:
                print(f"watcher stop: checkpoint steps_{args.target_step} is ready", flush=True)
                return 0
            if time.time() - reached_target_at > args.checkpoint_grace_minutes * 60:
                print("watcher stop: target reached but checkpoint did not appear before grace timeout", flush=True)
                return 3

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
