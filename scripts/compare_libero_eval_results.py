#!/usr/bin/env python3
"""Summarize and compare LIBERO / LIBERO-Plus eval logs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


SUITE_RE = re.compile(r"Task suite:\s*([^\s]+)")
RATE_RE = re.compile(r"Total success rate:\s*([0-9.]+)")
EPISODES_RE = re.compile(r"Total episodes:\s*([0-9]+)")


@dataclass(frozen=True)
class EvalResult:
    run: str
    suite_or_category: str
    success_rate: float
    episodes: int | None
    log_path: Path


def _infer_label(path: Path, text: str) -> str:
    suite_match = SUITE_RE.search(text)
    suite = suite_match.group(1) if suite_match else "unknown"

    parts = path.parts
    if "plus_libero_mix" in parts:
        idx = parts.index("plus_libero_mix")
        if idx + 1 < len(parts):
            return f"{suite}/{parts[idx + 1]}"
    return suite


def _parse_log(path: Path, run: str) -> EvalResult | None:
    text = path.read_text(errors="replace")
    rate_matches = RATE_RE.findall(text)
    if not rate_matches:
        return None
    episode_matches = EPISODES_RE.findall(text)
    return EvalResult(
        run=run,
        suite_or_category=_infer_label(path, text),
        success_rate=float(rate_matches[-1]),
        episodes=int(episode_matches[-1]) if episode_matches else None,
        log_path=path,
    )


def _collect(root: Path) -> list[EvalResult]:
    logs = [root] if root.is_file() else sorted(root.rglob("eval.log"))
    results: list[EvalResult] = []
    run = root.name
    for log in logs:
        parsed = _parse_log(log, run)
        if parsed is not None:
            results.append(parsed)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Eval log file or result root directory.")
    parser.add_argument("--tsv", action="store_true", help="Print machine-readable TSV.")
    args = parser.parse_args()

    all_results: list[EvalResult] = []
    for path in args.paths:
        all_results.extend(_collect(path))

    if not all_results:
        raise SystemExit("No eval.log files with `Total success rate:` found.")

    baseline_by_label: dict[str, float] = {}
    for result in all_results:
        baseline_by_label.setdefault(result.suite_or_category, result.success_rate)

    header = ["run", "suite_or_category", "success_rate", "delta_vs_first", "episodes", "log_path"]
    rows = []
    for result in sorted(all_results, key=lambda item: (item.suite_or_category, item.run, str(item.log_path))):
        baseline = baseline_by_label[result.suite_or_category]
        rows.append(
            [
                result.run,
                result.suite_or_category,
                f"{result.success_rate:.6f}",
                f"{result.success_rate - baseline:+.6f}",
                "" if result.episodes is None else str(result.episodes),
                str(result.log_path),
            ]
        )

    if args.tsv:
        print("\t".join(header))
        for row in rows:
            print("\t".join(row))
        return 0

    widths = [len(col) for col in header]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    print("  ".join(col.ljust(width) for col, width in zip(header, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
