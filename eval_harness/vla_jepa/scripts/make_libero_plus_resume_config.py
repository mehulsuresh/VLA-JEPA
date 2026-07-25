#!/usr/bin/env python3
"""Build a LIBERO-Plus resume config from a partial harness SQLite DB."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from typing import Any

import yaml


DEFAULT_CLASSIFICATION = (
    Path("/home/mehul/work/vjepa/libero_plus_work/src/LIBERO-plus")
    / "libero/libero/benchmark/task_classification.json"
)


def _completed_keys(db_path: Path) -> set[tuple[str, str, int, int]]:
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "select status, context from episode_results where status in ('success', 'fail')"
    ).fetchall()
    completed: set[tuple[str, str, int, int]] = set()
    for _, context_json in rows:
        context = json.loads(context_json or "{}")
        suite = context.get("suite")
        category = context.get("category")
        task_id = context.get("task_id")
        episode_idx = context.get("episode_idx", 0)
        if suite is None or category is None or task_id is None:
            continue
        completed.add((str(suite), str(category), int(task_id), int(episode_idx)))
    return completed


def _category_counts(classification_path: Path) -> dict[tuple[str, str], int]:
    raw = json.loads(classification_path.read_text())
    counts: dict[tuple[str, str], int] = {}
    for suite, entries in raw.items():
        by_category: dict[str, int] = {}
        for entry in entries:
            category = str(entry["category"])
            by_category[category] = by_category.get(category, 0) + 1
        for category, count in by_category.items():
            counts[(suite, category)] = count
    return counts


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    args = parser.parse_args()

    base = _load_yaml(args.base_config)
    completed = _completed_keys(args.db)
    counts = _category_counts(args.classification)

    resume = deepcopy(base)
    resume["benchmarks"] = []
    scheduled = 0

    for bench in base.get("benchmarks", []):
        params = dict(bench.get("params") or {})
        suite = params.get("suite")
        category = params.get("category")
        if suite is None or category is None:
            continue
        total = counts.get((str(suite), str(category)))
        if total is None:
            raise KeyError(f"No task_classification count for suite={suite!r} category={category!r}")
        episodes = int(bench.get("episodes_per_task", 1))
        missing_ids: list[int] = []
        for task_id in range(total):
            if any((str(suite), str(category), task_id, ep) not in completed for ep in range(episodes)):
                missing_ids.append(task_id)
        if not missing_ids:
            continue
        new_bench = deepcopy(bench)
        new_bench["subname"] = f"{bench.get('subname', suite)}_resume"
        new_params = dict(new_bench.get("params") or {})
        new_params["task_ids"] = missing_ids
        new_bench["params"] = new_params
        resume["benchmarks"].append(new_bench)
        scheduled += len(missing_ids) * episodes

    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    with args.out_config.open("w") as f:
        yaml.safe_dump(resume, f, sort_keys=False, width=120)

    print(f"completed={len(completed)} scheduled={scheduled} benchmarks={len(resume['benchmarks'])}")
    print(args.out_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
