"""VLA-JEPA LIBERO-Plus benchmark wrappers.

The upstream harness filters LIBERO-Plus categories after constructing the
LIBERO suite. This local LIBERO-Plus install also filters tasks while building
the suite and defaults that inner filter to "Camera Viewpoints", so category
evals must pass the requested category into the LIBERO suite constructor.
"""

from __future__ import annotations

import functools
import json
import random
from typing import Any

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.benchmarks.libero_plus.benchmark import LIBEROPlusBenchmark
from vla_eval.benchmarks.libero_plus.benchmark import _registry_name
from vla_eval.types import Task


class VLAJEPALIBEROPlusBenchmark(LIBEROPlusBenchmark):
    """LIBERO-Plus benchmark that applies category filtering at source."""

    def __init__(self, *, task_ids: list[int] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.task_ids = {int(task_id) for task_id in task_ids} if task_ids is not None else None

    @staticmethod
    def _build_zero_based_task_orders(libero_benchmark_module):
        """Return a LIBERO-Plus task-order helper with corrected 0-based ids.

        The local LIBERO-Plus fork stores task IDs in task_classification.json
        as 1-based positions, but its Benchmark._make_benchmark indexes
        directly into Python lists. Categories whose IDs do not reach the end
        of a suite silently shift by one; categories that do reach the end
        crash. Patch the helper at source so suite construction remains fast
        and category-scoped.
        """

        def _get_ids_by_category(category_value: str) -> dict[str, list[list[int]]]:
            with libero_benchmark_module._resolve_task_classification_path().open(
                "r", encoding="utf-8"
            ) as f:
                data = json.load(f)

            result: dict[str, list[list[int]]] = {}
            for suite_name, tasks in data.items():
                matching_ids = [
                    int(task["id"]) - 1
                    for task in tasks
                    if task.get("category") == category_value
                ]
                if not matching_ids:
                    continue
                orders = [list(matching_ids)]
                rng = random.Random(0)
                for _ in range(19):
                    shuffled = list(matching_ids)
                    rng.shuffle(shuffled)
                    orders.append(shuffled)
                result[suite_name] = orders
            return result

        return _get_ids_by_category

    def _init_libero(self) -> None:
        if self._task_suite is not None:
            return

        import torch

        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _patched_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_load

        from libero.libero import benchmark

        benchmark.get_ids_by_category = self._build_zero_based_task_orders(benchmark)

        benchmark_dict = benchmark.get_benchmark_dict()
        suite_cls = benchmark_dict[self.suite]
        suite_kwargs: dict[str, Any] = {}
        if self.category is not None:
            suite_kwargs["category_value"] = self.category
        self._task_suite = suite_cls(**suite_kwargs)

    def get_tasks(self) -> list[Task]:
        tasks = LIBEROBenchmark.get_tasks(self)
        if self.task_ids is not None:
            tasks = [task for task in tasks if int(task["task_id"]) in self.task_ids]
        classification = self._load_classification()
        for task in tasks:
            entry = classification.get(_registry_name(task) or "")
            if self.category is not None:
                task["category"] = self.category
            elif entry is not None:
                task["category"] = entry.get("category")
            if entry is not None:
                task["difficulty_level"] = entry.get("difficulty_level")
        return tasks
