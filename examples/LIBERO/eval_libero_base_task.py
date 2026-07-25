import argparse
from pathlib import Path

import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import Task

from examples.LIBERO.eval_libero import Args, eval_libero


def _load_init_states(problem_folder: str, init_states_file: str):
    init_root = Path(get_libero_path("init_states"))
    candidates = [
        init_root / problem_folder / init_states_file,
        init_root / "libero_mix" / init_states_file,
    ]

    for path in candidates:
        if path.exists():
            states = torch.load(path, weights_only=False)
            if "_add_" in init_states_file or "_level" in init_states_file:
                states = states.reshape(1, -1)
            return states

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find init states file. Searched:\n{searched}")


class _SingleTaskSuite:
    n_tasks = 1

    def __init__(self, task: Task):
        self._task = task

    def get_task(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self._task

    def get_task_init_states(self, index: int):
        if index != 0:
            raise IndexError(index)
        return _load_init_states(self._task.problem_folder, self._task.init_states_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one explicit base LIBERO task against an already-running policy server."
    )
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--suite-name", default="libero_goal")
    parser.add_argument("--task-name", default="turn_on_the_stove")
    parser.add_argument("--task-language", default="turn on the stove")
    parser.add_argument("--bddl-file", default=None)
    parser.add_argument("--init-states-file", default=None)
    parser.add_argument("--video-out-path", required=True)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-ddim-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--action-execution-mode", choices=["chunk", "receding"], default="receding")
    parser.add_argument("--action-ensemble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-ensemble-horizon", type=int, default=None)
    parser.add_argument("--adaptive-ensemble-alpha", type=float, default=0.1)
    parser.add_argument("--with-state", default="true")
    parser.add_argument("--seed", type=int, default=7)
    parsed = parser.parse_args()

    bddl_file = parsed.bddl_file or f"{parsed.task_name}.bddl"
    init_states_file = parsed.init_states_file or f"{parsed.task_name}.pruned_init"

    task = Task(
        name=parsed.task_name,
        language=parsed.task_language,
        problem="Libero",
        problem_folder=parsed.suite_name,
        bddl_file=bddl_file,
        init_states_file=init_states_file,
    )

    original_get_benchmark_dict = benchmark.get_benchmark_dict
    benchmark.get_benchmark_dict = lambda *args, **kwargs: {
        parsed.suite_name: lambda **suite_kwargs: _SingleTaskSuite(task)
    }
    try:
        eval_libero(
            Args(
                pretrained_path=parsed.pretrained_path,
                host=parsed.host,
                port=parsed.port,
                task_suite_name=parsed.suite_name,
                category_value="BASE_NO_PERTURBATION",
                num_trials_per_task=parsed.num_trials_per_task,
                video_out_path=parsed.video_out_path,
                with_state=parsed.with_state,
                max_steps_override=parsed.max_steps,
                num_ddim_steps=parsed.num_ddim_steps,
                num_steps_wait=parsed.num_steps_wait,
                action_execution_mode=parsed.action_execution_mode,
                action_ensemble=parsed.action_ensemble,
                action_ensemble_horizon=parsed.action_ensemble_horizon,
                adaptive_ensemble_alpha=parsed.adaptive_ensemble_alpha,
                seed=parsed.seed,
            )
        )
    finally:
        benchmark.get_benchmark_dict = original_get_benchmark_dict


if __name__ == "__main__":
    main()
