#!/usr/bin/env python3
"""Build live TensorBoard comparison logs on a physical-samples-seen axis."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import signal
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


@dataclass
class RunState:
    name: str
    source: Path
    output: Path
    samples_per_step: float | None = None
    accumulator: EventAccumulator | None = None
    writer: EventFileWriter | None = None
    cursors: dict[str, int] = field(default_factory=dict)
    points_written: int = 0

    def open(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self.accumulator = EventAccumulator(
            str(self.source), size_guidance={"scalars": 0}
        )
        self.writer = EventFileWriter(str(self.output))

    def reload(self) -> None:
        assert self.accumulator is not None
        self.accumulator.Reload()
        if self.samples_per_step is None:
            self.samples_per_step = infer_samples_per_step(self.accumulator, self.name)

    def append_through(self, max_samples: int, tag_pattern: re.Pattern[str] | None) -> int:
        assert self.accumulator is not None
        assert self.writer is not None
        assert self.samples_per_step is not None

        pending: dict[tuple[int, float], list[Summary.Value]] = defaultdict(list)
        tags = self.accumulator.Tags().get("scalars", [])
        for tag in tags:
            if tag_pattern is not None and tag_pattern.search(tag) is None:
                continue
            points = sorted(
                self.accumulator.Scalars(tag), key=lambda point: (point.step, point.wall_time)
            )
            cursor = self.cursors.get(tag, 0)
            while cursor < len(points):
                point = points[cursor]
                sample_step = round(point.step * self.samples_per_step)
                if sample_step > max_samples:
                    break
                pending[(sample_step, point.wall_time)].append(
                    Summary.Value(tag=tag, simple_value=point.value)
                )
                cursor += 1
            self.cursors[tag] = cursor

        added = 0
        for (sample_step, wall_time), values in sorted(pending.items()):
            self.writer.add_event(
                Event(
                    wall_time=wall_time,
                    step=sample_step,
                    summary=Summary(value=values),
                )
            )
            added += len(values)
        if added:
            self.writer.flush()
            self.points_written += added
        return added

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def infer_samples_per_step(accumulator: EventAccumulator, run_name: str) -> float:
    try:
        sample_points = accumulator.Scalars("samples_seen")
    except KeyError as exc:
        raise ValueError(f"{run_name!r} has no samples_seen scalar") from exc

    ratios = [
        point.value / point.step
        for point in sample_points
        if point.step > 0 and point.value > 0 and math.isfinite(point.value)
    ]
    if not ratios:
        raise ValueError(f"{run_name!r} has no usable samples_seen points")

    samples_per_step = statistics.median(ratios)
    max_relative_error = max(
        abs(ratio - samples_per_step) / samples_per_step for ratio in ratios
    )
    if max_relative_error > 1e-4:
        raise ValueError(
            f"{run_name!r} has non-linear samples_seen values "
            f"(max relative error {max_relative_error:.3%})"
        )
    return samples_per_step


def parse_named_value(value: str) -> tuple[str, str]:
    try:
        name, raw_value = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=VALUE") from exc
    if not name or not raw_value:
        raise argparse.ArgumentTypeError("expected non-empty NAME=VALUE")
    return name, raw_value


def write_metadata(output: Path, reference: RunState, states: list[RunState]) -> None:
    assert reference.accumulator is not None
    sample_points = reference.accumulator.Scalars("samples_seen")
    metadata = {
        "axis": "physical_samples_seen",
        "reference_run": reference.name,
        "latest_reference_samples": round(max(point.value for point in sample_points)),
        "updated_at": time.time(),
        "runs": {
            state.name: {
                "source": str(state.source),
                "samples_per_optimizer_step": state.samples_per_step,
                "scalar_points_written": state.points_written,
            }
            for state in states
        },
    }
    temporary = output / "comparison.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(output / "comparison.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_named_value,
        metavar="NAME=LOGDIR",
    )
    parser.add_argument("--reference-run", required=True)
    parser.add_argument(
        "--samples-per-step",
        action="append",
        default=[],
        type=parse_named_value,
        metavar="NAME=COUNT",
    )
    parser.add_argument("--tag-regex")
    parser.add_argument("--refresh-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset and args.output.exists():
        for child in args.output.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    args.output.mkdir(parents=True, exist_ok=True)

    explicit_scales = {name: float(value) for name, value in args.samples_per_step}
    states = [
        RunState(
            name=name,
            source=Path(source).resolve(),
            output=args.output / name,
            samples_per_step=explicit_scales.get(name),
        )
        for name, source in args.run
    ]
    state_by_name = {state.name: state for state in states}
    if len(state_by_name) != len(states):
        parser.error("run names must be unique")
    if args.reference_run not in state_by_name:
        parser.error("--reference-run must match a --run name")
    for state in states:
        if not state.source.is_dir():
            parser.error(f"source log directory does not exist: {state.source}")

    tag_pattern = re.compile(args.tag_regex) if args.tag_regex else None
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        for state in states:
            state.open()
        reference = state_by_name[args.reference_run]

        while not stop:
            for state in states:
                state.reload()
            assert reference.accumulator is not None
            reference_samples = reference.accumulator.Scalars("samples_seen")
            max_samples = round(max(point.value for point in reference_samples))

            additions = {
                state.name: state.append_through(max_samples, tag_pattern)
                for state in states
            }
            write_metadata(args.output, reference, states)
            print(f"samples={max_samples} additions={additions}", flush=True)
            if args.once:
                break
            time.sleep(args.refresh_seconds)
    finally:
        for state in states:
            state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
