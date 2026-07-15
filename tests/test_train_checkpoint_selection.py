import hashlib
import json
import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from starVLA.training import train_starvla
from starVLA.training.train_starvla import VLATrainer


METRIC = "heldout_focused_eval_task_failure_score_h10"


@pytest.fixture(autouse=True)
def _use_plain_test_logger(monkeypatch):
    monkeypatch.setattr(
        train_starvla, "logger", logging.getLogger("checkpoint-selection-test")
    )


class _Accelerator:
    is_main_process = True
    num_processes = 2

    def __init__(self):
        self.loaded_path = None
        self.wait_count = 0

    def load_state(self, checkpoint_path):
        self.loaded_path = str(checkpoint_path)

    def save_state(self, checkpoint_path):
        checkpoint = Path(checkpoint_path)
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "model.safetensors").write_bytes(b"saved-model")
        (checkpoint / "optimizer.bin").write_bytes(b"saved-optimizer")
        (checkpoint / "scheduler.bin").write_bytes(b"saved-scheduler")
        for rank in range(self.num_processes):
            (checkpoint / f"random_states_{rank}.pkl").write_bytes(
                f"saved-rng-{rank}".encode("utf-8")
            )

    def wait_for_everyone(self):
        self.wait_count += 1

    def print(self, *_args, **_kwargs):
        return None


def _trainer(tmp_path: Path, *, checkpoint_max_to_keep=3) -> VLATrainer:
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "run_id": "checkpoint-selection-test",
            "seed": 17,
            "trainer": {
                "best_metric_name": METRIC,
                "best_metric_mode": "min",
                "checkpoint_max_to_keep": checkpoint_max_to_keep,
                "resume_load_optimizer_state": True,
                "save_best_only": False,
                "_accelerate_num_processes": 2,
            },
        }
    )
    trainer.accelerator = _Accelerator()
    trainer.checkpoint_dir = str(tmp_path / "checkpoints")
    Path(trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    trainer.completed_steps = 0
    trainer.best_metric_name = METRIC
    trainer.best_metric_mode = "min"
    trainer.best_metric_value = None
    trainer.best_metric_step = None
    trainer.legacy_underfilled_eval = False
    trainer._warned_missing_best_metric = False
    trainer._warned_ineligible_best_metric = False
    trainer.vla_eval_dataloader = object()
    trainer.vla_focused_eval_dataloader = object()
    eligible_report = {
        "window_selection_sha256": "a" * 64,
        "observation_mode": "deterministic_fixture_window",
        "production_valid": True,
        "checkpoint_selection_eligible": True,
        "episode_split_provenance": {"manifest_sha256": "b" * 64},
    }
    trainer.heldout_eval_sampling_report = dict(eligible_report)
    trainer.heldout_focused_eval_sampling_report = {
        **eligible_report,
        "window_selection_sha256": "c" * 64,
    }
    trainer.lr_scheduler = None
    config_bytes = b"run_id: checkpoint-selection-test\n"
    (tmp_path / "config.yaml").write_bytes(config_bytes)
    (tmp_path / "resolved_training_schedule.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "resolved": {
                    "effective_global_batch_size": 2,
                    "eval_interval": 5,
                    "max_train_steps": 100,
                    "num_warmup_steps": 0,
                    "save_interval": 5,
                },
                "source_config": {
                    "path": "config.yaml",
                    "sha256": hashlib.sha256(config_bytes).hexdigest(),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return trainer


def _write_checkpoint_selection_metadata(
    trainer: VLATrainer,
    path: Path,
    step: int,
    *,
    best_step: int | None = None,
    best_value: float | None = None,
) -> None:
    trainer_state = {
        "completed_steps": step,
        "selection_state_schema_version": 1,
        "best_metric_name": METRIC,
        "best_metric_mode": "min",
        "best_metric_value": best_value,
        "best_metric_step": best_step,
    }
    (path / "trainer_state.json").write_text(
        json.dumps(trainer_state, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selection_state = {
        "schema_version": 1,
        "best_metric_name": METRIC,
        "best_metric_mode": "min",
        "best_metric_value": best_value,
        "best_metric_step": best_step,
        "checkpoint_relative_path": (
            None if best_step is None else f"checkpoints/steps_{best_step}"
        ),
    }
    (path / "selection_state.json").write_text(
        json.dumps(selection_state, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checkpoint(
    trainer: VLATrainer,
    step: int,
    *,
    best_step: int | None = None,
    best_value: float | None = None,
) -> Path:
    path = Path(trainer.checkpoint_dir) / f"steps_{step}"
    path.mkdir(parents=True, exist_ok=True)
    if not (path / "trainer_state.json").exists():
        _write_checkpoint_selection_metadata(
            trainer,
            path,
            step,
            best_step=best_step,
            best_value=best_value,
        )
    model_path = path / "model.safetensors"
    if not model_path.exists():
        model_path.write_bytes(f"fixture-model-step-{step}".encode("utf-8"))
    for filename in ("optimizer.bin", "scheduler.bin"):
        state_path = path / filename
        if not state_path.exists():
            state_path.write_bytes(f"fixture-{filename}-{step}".encode("utf-8"))
    for rank in range(trainer.accelerator.num_processes):
        rng_path = path / f"random_states_{rank}.pkl"
        if not rng_path.exists():
            rng_path.write_bytes(f"fixture-rng-{rank}-{step}".encode("utf-8"))
    return path


def _write_heldout_artifact(
    trainer: VLATrainer,
    checkpoint: Path,
    *,
    metric_value: float | None,
    production_valid: bool = True,
    eligible: bool = True,
    checkpoint_bound: bool = True,
    **overrides,
) -> Path:
    trainer_state_sha256 = hashlib.sha256(
        (checkpoint / "trainer_state.json").read_bytes()
    ).hexdigest()
    model_path = checkpoint / "model.safetensors"
    step = int(checkpoint.name.removeprefix("steps_"))
    payload = {
        "schema_version": 1,
        "checkpoint_step": step,
        "checkpoint_relative_path": (
            f"checkpoints/steps_{step}" if checkpoint_bound else None
        ),
        "checkpoint": {
            "step": step,
            "source_path": str(checkpoint.resolve()) if checkpoint_bound else None,
            "source_kind": (
                "checkpoint" if checkpoint_bound else "live_in_memory_model"
            ),
        },
        "run": {
            "run_id": str(trainer.config.run_id),
            "output_dir": str(Path(trainer.config.output_dir).resolve()),
            "seed": int(trainer.config.seed),
            "config_path": "config.yaml",
            "config_sha256": hashlib.sha256(
                (Path(trainer.config.output_dir) / "config.yaml").read_bytes()
            ).hexdigest(),
            "resolved_training_schedule": {
                "path": "resolved_training_schedule.json",
                "sha256": hashlib.sha256(
                    (
                        Path(trainer.config.output_dir)
                        / "resolved_training_schedule.json"
                    ).read_bytes()
                ).hexdigest(),
            },
            "source_training_config": None,
        },
        "sampling_reports": {},
        "production_valid": production_valid,
        "checkpoint_selection_eligible": eligible,
        "selection_metric": {
            "name": METRIC,
            "mode": "min",
            "eligible": eligible,
            "value": metric_value,
        },
        "metrics": {
            "unbiased": {"heldout_eval_normalized_action_mae": 0.5},
            "focused": ({} if metric_value is None else {METRIC: metric_value}),
        },
    }
    if checkpoint_bound:
        payload["checkpoint"].update(
            {
                "trainer_state_sha256": trainer_state_sha256,
                "model_file": model_path.name,
                "model_file_size_bytes": model_path.stat().st_size,
                "model_file_sha256": hashlib.sha256(
                    model_path.read_bytes()
                ).hexdigest(),
            }
        )
    unbiased_report = dict(trainer.heldout_eval_sampling_report)
    focused_report = dict(trainer.heldout_focused_eval_sampling_report)
    for report in (unbiased_report, focused_report):
        report["production_valid"] = production_valid
        report["checkpoint_selection_eligible"] = eligible
    trainer.heldout_eval_sampling_report = dict(unbiased_report)
    trainer.heldout_focused_eval_sampling_report = dict(focused_report)
    payload["sampling_reports"] = {
        "unbiased": trainer._heldout_report_evidence(
            unbiased_report, label="fixture unbiased"
        ),
        "focused": trainer._heldout_report_evidence(
            focused_report, label="fixture focused"
        ),
    }
    payload.update(overrides)
    artifact_path = (
        Path(trainer.config.output_dir)
        / "heldout_eval_metrics"
        / f"step_{step:08d}.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path


@pytest.mark.parametrize("metric", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_heldout_metric_never_selects(tmp_path, metric):
    trainer = _trainer(tmp_path)
    trainer.completed_steps = 10
    metrics = {} if metric is None else {METRIC: metric}

    assert trainer._consider_best_checkpoint_metric(
        metrics, require_heldout_eligibility=True
    ) is False
    assert trainer.best_metric_value is None
    assert trainer.best_metric_step is None


@pytest.mark.parametrize("invalid_field", ["production_valid", "checkpoint_selection_eligible"])
def test_ineligible_heldout_report_never_selects(tmp_path, invalid_field):
    trainer = _trainer(tmp_path)
    trainer.completed_steps = 10
    trainer.heldout_focused_eval_sampling_report[invalid_field] = False

    assert trainer._consider_best_checkpoint_metric(
        {METRIC: 0.25}, require_heldout_eligibility=True
    ) is False
    assert trainer.best_metric_value is None
    assert trainer.best_metric_step is None


def test_deferred_selection_keeps_trainer_state_hash_valid_and_prunes_best_plus_newest(
    tmp_path,
):
    trainer = _trainer(tmp_path)
    for step in (10, 20, 30, 40):
        _checkpoint(trainer, step, best_step=10, best_value=0.30)
    trainer.completed_steps = 40
    trainer.best_metric_value = 0.30
    trainer.best_metric_step = 10
    trainer_state_path = _checkpoint(trainer, 40) / "trainer_state.json"
    trainer_state_bytes = (
        json.dumps(
            {
                "completed_steps": 40,
                "selection_state_schema_version": 1,
                "best_metric_name": METRIC,
                "best_metric_mode": "min",
                "best_metric_value": 0.3,
                "best_metric_step": 10,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    trainer_state_path.write_bytes(trainer_state_bytes)
    trainer_state_sha256 = hashlib.sha256(trainer_state_bytes).hexdigest()

    selected = trainer._finalize_deferred_checkpoint(
        {METRIC: 0.35}, track_heldout_best=True
    )

    assert selected is False
    assert trainer_state_path.read_bytes() == trainer_state_bytes
    assert hashlib.sha256(trainer_state_path.read_bytes()).hexdigest() == (
        trainer_state_sha256
    )
    assert {
        path.name for path in Path(trainer.checkpoint_dir).glob("steps_*")
    } == {"steps_10", "steps_30", "steps_40"}
    pointer = json.loads(
        (tmp_path / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    assert pointer["best_metric_value"] == pytest.approx(0.30)
    assert pointer["best_metric_step"] == 10
    assert pointer["checkpoint_relative_path"] == "checkpoints/steps_10"
    current_selection = json.loads(
        (_checkpoint(trainer, 40) / "selection_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert current_selection == pointer
    assert list(tmp_path.rglob("*.tmp-*")) == []


def test_new_best_replaces_old_best_before_best_aware_pruning(tmp_path):
    trainer = _trainer(tmp_path)
    for step in (10, 20, 30, 40):
        _checkpoint(trainer, step, best_step=10, best_value=0.30)
    trainer.completed_steps = 40
    trainer.best_metric_value = 0.30
    trainer.best_metric_step = 10

    selected = trainer._finalize_deferred_checkpoint(
        {METRIC: 0.20}, track_heldout_best=True
    )

    assert selected is True
    assert trainer.best_metric_value == pytest.approx(0.20)
    assert trainer.best_metric_step == 40
    assert {
        path.name for path in Path(trainer.checkpoint_dir).glob("steps_*")
    } == {"steps_10", "steps_30", "steps_40"}
    pointer = json.loads(
        (tmp_path / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    assert pointer["best_metric_step"] == 40


def test_resume_accepts_exact_artifact_derived_selection_over_pre_eval_trainer_state(
    tmp_path,
):
    trainer = _trainer(tmp_path)
    _checkpoint(trainer, 10)
    checkpoint = _checkpoint(trainer, 40)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "completed_steps": 40,
                "selection_state_schema_version": 1,
                "best_metric_name": METRIC,
                "best_metric_mode": "min",
                "best_metric_value": 0.30,
                "best_metric_step": 10,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "selection_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "best_metric_name": METRIC,
                "best_metric_mode": "min",
                "best_metric_value": 0.20,
                "best_metric_step": 40,
                "checkpoint_relative_path": "checkpoints/steps_40",
            }
        ),
        encoding="utf-8",
    )
    _write_heldout_artifact(trainer, checkpoint, metric_value=0.20)

    trainer._load_checkpoint(checkpoint)

    assert trainer.completed_steps == 40
    assert trainer.best_metric_value == pytest.approx(0.20)
    assert trainer.best_metric_step == 40
    pointer = json.loads(
        (tmp_path / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    assert pointer["best_metric_step"] == 40
    assert trainer.accelerator.loaded_path == str(checkpoint.resolve())


def test_save_best_only_uses_eligible_metric_but_legacy_nonheldout_still_works(
    tmp_path,
):
    trainer = _trainer(tmp_path)
    trainer.config.trainer.save_best_only = True
    trainer.completed_steps = 10

    assert trainer._should_save_checkpoint({METRIC: 0.40}) is True
    assert trainer.best_metric_step == 10
    trainer.completed_steps = 20
    assert trainer._should_save_checkpoint({METRIC: 0.50}) is False
    assert trainer.best_metric_step == 10

    legacy = _trainer(tmp_path / "legacy")
    legacy.config.trainer.save_best_only = True
    legacy.vla_eval_dataloader = None
    legacy.vla_focused_eval_dataloader = None
    legacy.completed_steps = 7
    assert legacy._should_save_checkpoint({METRIC: 0.45}) is True
    assert legacy.best_metric_step == 7


def test_checkpoint_cap_one_fails_closed_when_best_and_newest_differ(tmp_path):
    trainer = _trainer(tmp_path, checkpoint_max_to_keep=1)
    _checkpoint(trainer, 10, best_step=10, best_value=0.25)
    _checkpoint(trainer, 20, best_step=10, best_value=0.25)
    trainer.completed_steps = 20
    trainer.best_metric_value = 0.25
    trainer.best_metric_step = 10

    with pytest.raises(RuntimeError, match="cannot preserve.*dependencies"):
        trainer._prune_old_checkpoints()

    assert {
        path.name for path in Path(trainer.checkpoint_dir).glob("steps_*")
    } == {"steps_10", "steps_20"}


def test_malformed_selection_state_fails_closed_on_resume(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 40)
    (checkpoint / "selection_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "best_metric_name": METRIC,
                "best_metric_mode": "min",
                "best_metric_value": float("nan"),
                "best_metric_step": 40,
                "checkpoint_relative_path": "checkpoints/steps_40",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="must be finite"):
        trainer._load_checkpoint(checkpoint)


def test_selection_state_metric_mode_identity_is_exact(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 40)
    selection_path = checkpoint / "selection_state.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["best_metric_mode"] = "MIN"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metric mode does not match"):
        trainer._load_checkpoint(checkpoint)


def test_new_checkpoint_cannot_silently_resume_without_selection_state(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 40)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "completed_steps": 40,
                "selection_state_schema_version": 1,
                "best_metric_name": METRIC,
                "best_metric_mode": "min",
                "best_metric_value": 0.20,
                "best_metric_step": 40,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "selection_state.json").unlink()

    with pytest.raises(RuntimeError, match="selection state.*missing"):
        trainer._load_checkpoint(checkpoint)


def test_dependency_closed_retention_preserves_resumable_history_across_new_best(
    tmp_path,
):
    trainer = _trainer(tmp_path, checkpoint_max_to_keep=3)
    _checkpoint(trainer, 5, best_step=5, best_value=0.30)
    _checkpoint(trainer, 10, best_step=5, best_value=0.30)
    _checkpoint(trainer, 15, best_step=5, best_value=0.30)
    _checkpoint(trainer, 20, best_step=20, best_value=0.20)
    trainer.completed_steps = 20
    trainer.best_metric_value = 0.20
    trainer.best_metric_step = 20

    trainer._prune_old_checkpoints()

    assert {
        path.name for path in Path(trainer.checkpoint_dir).glob("steps_*")
    } == {"steps_5", "steps_15", "steps_20"}
    checkpoints, dependencies = trainer._checkpoint_selection_inventory()
    retained_steps = {step for step, _ in checkpoints}
    assert trainer._checkpoint_dependency_closure(
        retained_steps, dependencies
    ) == retained_steps

    _checkpoint(trainer, 25, best_step=20, best_value=0.20)
    trainer.completed_steps = 25
    trainer._prune_old_checkpoints()

    assert {
        path.name for path in Path(trainer.checkpoint_dir).glob("steps_*")
    } == {"steps_5", "steps_20", "steps_25"}
    for step in (5, 20, 25):
        resume_probe = _trainer(tmp_path, checkpoint_max_to_keep=3)
        resume_probe._load_checkpoint(
            Path(resume_probe.checkpoint_dir) / f"steps_{step}"
        )
        assert resume_probe.completed_steps == step


def test_missing_historical_best_dependency_fails_before_any_pruning(tmp_path):
    trainer = _trainer(tmp_path, checkpoint_max_to_keep=2)
    _checkpoint(trainer, 10, best_step=5, best_value=0.30)
    _checkpoint(trainer, 20, best_step=10, best_value=0.25)
    _checkpoint(trainer, 30, best_step=10, best_value=0.25)
    trainer.completed_steps = 30
    trainer.best_metric_value = 0.25
    trainer.best_metric_step = 10

    with pytest.raises(RuntimeError, match="dependency is missing.*steps_5"):
        trainer._prune_old_checkpoints()

    assert {
        path.name for path in Path(trainer.checkpoint_dir).glob("steps_*")
    } == {"steps_10", "steps_20", "steps_30"}


def test_legacy_dependency_is_preserved_and_half_populated_legacy_state_fails_closed(
    tmp_path,
):
    trainer = _trainer(tmp_path, checkpoint_max_to_keep=2)
    _checkpoint(trainer, 5, best_step=5, best_value=0.30)
    legacy = _checkpoint(trainer, 15)
    (legacy / "selection_state.json").unlink()
    (legacy / "trainer_state.json").write_text(
        json.dumps(
            {
                "completed_steps": 15,
                "best_metric_value": 0.30,
                "best_metric_step": 5,
            }
        ),
        encoding="utf-8",
    )
    trainer.completed_steps = 15
    trainer.best_metric_value = 0.30
    trainer.best_metric_step = 5

    trainer._prune_old_checkpoints()
    assert {
        path.name for path in Path(trainer.checkpoint_dir).glob("steps_*")
    } == {"steps_5", "steps_15"}

    malformed = _checkpoint(trainer, 20)
    (malformed / "selection_state.json").unlink()
    (malformed / "trainer_state.json").write_text(
        json.dumps({"completed_steps": 20, "best_metric_step": 5}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="best step without a metric value"):
        trainer._prune_old_checkpoints()
    assert malformed.is_dir()


def test_resume_reconciles_kill_after_artifact_before_finalize_and_keeps_artifact_immutable(
    tmp_path,
):
    trainer = _trainer(tmp_path, checkpoint_max_to_keep=3)
    _checkpoint(trainer, 5, best_step=5, best_value=0.30)
    _checkpoint(trainer, 10, best_step=5, best_value=0.30)
    _checkpoint(trainer, 15, best_step=5, best_value=0.30)
    checkpoint = _checkpoint(trainer, 20, best_step=5, best_value=0.30)
    trainer_state_bytes = (checkpoint / "trainer_state.json").read_bytes()
    artifact_path = _write_heldout_artifact(
        trainer, checkpoint, metric_value=0.20
    )
    artifact_bytes = artifact_path.read_bytes()

    trainer._load_checkpoint(checkpoint)

    assert trainer.best_metric_value == pytest.approx(0.20)
    assert trainer.best_metric_step == 20
    assert (checkpoint / "trainer_state.json").read_bytes() == trainer_state_bytes
    assert artifact_path.read_bytes() == artifact_bytes
    selection = json.loads(
        (checkpoint / "selection_state.json").read_text(encoding="utf-8")
    )
    assert selection["best_metric_value"] == pytest.approx(0.20)
    assert selection["best_metric_step"] == 20
    pointer = json.loads(
        (tmp_path / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    assert pointer == selection
    assert {
        path.name for path in Path(trainer.checkpoint_dir).glob("steps_*")
    } == {"steps_5", "steps_15", "steps_20"}
    assert list(tmp_path.rglob("*.tmp-*")) == []


def test_resume_ignores_authenticated_ineligible_same_step_artifact(tmp_path):
    trainer = _trainer(tmp_path, checkpoint_max_to_keep=3)
    _checkpoint(trainer, 5, best_step=5, best_value=0.30)
    checkpoint = _checkpoint(trainer, 10, best_step=5, best_value=0.30)
    artifact_path = _write_heldout_artifact(
        trainer,
        checkpoint,
        metric_value=0.10,
        production_valid=False,
        eligible=False,
    )
    artifact_bytes = artifact_path.read_bytes()

    trainer._load_checkpoint(checkpoint)

    assert trainer.best_metric_value == pytest.approx(0.30)
    assert trainer.best_metric_step == 5
    assert artifact_path.read_bytes() == artifact_bytes
    selection = json.loads(
        (checkpoint / "selection_state.json").read_text(encoding="utf-8")
    )
    assert selection["best_metric_step"] == 5


@pytest.mark.parametrize(
    "corruption",
    [
        "schema_bool",
        "schema_float",
        "step",
        "relative_path",
        "source_path",
        "trainer_hash",
        "model_name",
        "model_size",
        "model_hash",
        "model_same_size_content",
        "model_missing",
        "model_symlink",
        "run_id",
        "run_output_dir",
        "run_config_sha",
        "run_schedule_sha",
        "sampling_window_digest",
        "sampling_observation_mode",
        "sampling_provenance",
        "sampling_count",
        "metric_name",
        "metric_mode",
        "nonfinite_metric",
        "metric_group_disagreement",
    ],
)
def test_resume_reconciliation_fails_closed_on_malformed_or_inconsistent_artifact(
    tmp_path,
    corruption,
):
    trainer = _trainer(tmp_path)
    _checkpoint(trainer, 5, best_step=5, best_value=0.30)
    checkpoint = _checkpoint(trainer, 10, best_step=5, best_value=0.30)
    artifact_path = _write_heldout_artifact(
        trainer, checkpoint, metric_value=0.20
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if corruption == "schema_bool":
        payload["schema_version"] = True
    elif corruption == "schema_float":
        payload["schema_version"] = 1.0
    elif corruption == "step":
        payload["checkpoint_step"] = 9
    elif corruption == "relative_path":
        payload["checkpoint_relative_path"] = "checkpoints/steps_9"
    elif corruption == "source_path":
        payload["checkpoint"]["source_path"] = str(tmp_path / "wrong-checkpoint")
    elif corruption == "trainer_hash":
        payload["checkpoint"]["trainer_state_sha256"] = "0" * 64
    elif corruption == "model_name":
        payload["checkpoint"]["model_file"] = "optimizer.bin"
    elif corruption == "model_size":
        payload["checkpoint"]["model_file_size_bytes"] += 1
    elif corruption == "model_hash":
        payload["checkpoint"]["model_file_sha256"] = "0" * 64
    elif corruption == "model_same_size_content":
        model_path = checkpoint / "model.safetensors"
        model_path.write_bytes(b"x" * model_path.stat().st_size)
    elif corruption == "model_missing":
        (checkpoint / "model.safetensors").unlink()
    elif corruption == "model_symlink":
        model_path = checkpoint / "model.safetensors"
        model_path.unlink()
        target = tmp_path / "outside-model.safetensors"
        target.write_bytes(b"fixture-model-step-10")
        model_path.symlink_to(target)
    elif corruption == "run_id":
        payload["run"]["run_id"] = "wrong-run"
    elif corruption == "run_output_dir":
        payload["run"]["output_dir"] = str(tmp_path / "wrong-run")
    elif corruption == "run_config_sha":
        payload["run"]["config_sha256"] = "0" * 64
    elif corruption == "run_schedule_sha":
        payload["run"]["resolved_training_schedule"]["sha256"] = "0" * 64
    elif corruption == "sampling_window_digest":
        payload["sampling_reports"]["unbiased"]["window_selection_sha256"] = (
            "0" * 64
        )
    elif corruption == "sampling_observation_mode":
        payload["sampling_reports"]["unbiased"]["observation_mode"] = (
            "different_window_contract"
        )
    elif corruption == "sampling_provenance":
        payload["sampling_reports"]["unbiased"]["episode_split_provenance"] = {
            "manifest_sha256": "0" * 64
        }
    elif corruption == "sampling_count":
        payload["sampling_reports"]["unbiased"]["observation_count"] = 999
    elif corruption == "metric_name":
        payload["selection_metric"]["name"] = "wrong_metric"
    elif corruption == "metric_mode":
        payload["selection_metric"]["mode"] = "max"
    elif corruption == "nonfinite_metric":
        payload["selection_metric"]["value"] = float("nan")
    elif corruption == "metric_group_disagreement":
        payload["metrics"]["focused"][METRIC] = 0.21
    artifact_path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    artifact_bytes = artifact_path.read_bytes()
    selection_bytes = (checkpoint / "selection_state.json").read_bytes()

    with pytest.raises(RuntimeError, match="same-step heldout eval|Same-step heldout eval"):
        trainer._load_checkpoint(checkpoint)

    assert artifact_path.read_bytes() == artifact_bytes
    assert (checkpoint / "selection_state.json").read_bytes() == selection_bytes
    assert not (tmp_path / "best_checkpoint.json").exists()


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_selection_state_schema_rejects_bool_and_float_one(tmp_path, schema_version):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 10)
    selection_path = checkpoint / "selection_state.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["schema_version"] = schema_version
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsupported or malformed"):
        trainer._load_checkpoint(checkpoint)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_trainer_selection_schema_rejects_bool_and_float_one(tmp_path, schema_version):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 10)
    trainer_state_path = checkpoint / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    trainer_state["selection_state_schema_version"] = schema_version
    trainer_state_path.write_text(json.dumps(trainer_state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsupported checkpoint selection-state"):
        trainer._load_checkpoint(checkpoint)


def test_rank_zero_operation_broadcasts_failure_without_entering_a_barrier(
    tmp_path,
    monkeypatch,
):
    trainer = _trainer(tmp_path)
    broadcasts = []
    monkeypatch.setattr(train_starvla.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_starvla.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(train_starvla.dist, "get_backend", lambda: "gloo")

    def broadcast_object_list(container, *, src, **kwargs):
        broadcasts.append((dict(container[0]), src, kwargs))

    monkeypatch.setattr(
        train_starvla.dist,
        "broadcast_object_list",
        broadcast_object_list,
    )

    def fail():
        raise ValueError("fixture failure")

    with pytest.raises(RuntimeError, match="Rank-0 fixture operation failed.*fixture failure"):
        trainer._run_rank_zero_operation(fail, label="fixture operation")

    assert len(broadcasts) == 1
    assert broadcasts[0][0]["ok"] is False
    assert broadcasts[0][1] == 0


def test_nonzero_rank_consumes_broadcast_result_without_running_operation(
    tmp_path,
    monkeypatch,
):
    trainer = _trainer(tmp_path)
    operation_called = False
    monkeypatch.setattr(train_starvla.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_starvla.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(train_starvla.dist, "get_backend", lambda: "gloo")

    def broadcast_object_list(container, *, src, **_kwargs):
        assert src == 0
        assert container == [None]
        container[0] = {"ok": True, "result": {"selected": True}}

    monkeypatch.setattr(
        train_starvla.dist,
        "broadcast_object_list",
        broadcast_object_list,
    )

    def operation():
        nonlocal operation_called
        operation_called = True

    result = trainer._run_rank_zero_operation(
        operation,
        label="fixture operation",
    )

    assert result == {"selected": True}
    assert operation_called is False


@pytest.mark.parametrize("completed_steps", ["9", True, 9.9])
def test_schema_v1_trainer_completed_steps_rejects_coercible_nonintegers(
    tmp_path,
    completed_steps,
):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 10)
    trainer_state_path = checkpoint / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    trainer_state["completed_steps"] = completed_steps
    trainer_state_path.write_text(json.dumps(trainer_state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="completed_steps must be"):
        trainer._load_checkpoint(checkpoint)


def test_schema_v1_trainer_step_must_match_canonical_checkpoint_path(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 10)
    trainer_state_path = checkpoint / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    trainer_state["completed_steps"] = 9
    trainer_state_path.write_text(json.dumps(trainer_state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match its canonical path"):
        trainer._load_checkpoint(checkpoint)


@pytest.mark.parametrize("trainer_state_payload", ["{", "[]", "true"])
def test_present_malformed_or_nonobject_trainer_state_fails_closed(
    tmp_path,
    trainer_state_payload,
):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 10)
    (checkpoint / "trainer_state.json").write_text(
        trainer_state_payload, encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="trainer state"):
        trainer._load_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("best_metric_value", "0.2", "best metric must be finite"),
        ("best_metric_value", True, "best metric must be finite"),
        ("best_metric_value", float("nan"), "best metric must be finite"),
        ("best_metric_step", "5", "best step is malformed"),
        ("best_metric_step", True, "best step is malformed"),
        ("best_metric_step", 5.5, "best step is malformed"),
        ("best_metric_step", 11, "best step is malformed"),
    ],
)
def test_schema_v1_trainer_best_state_has_strict_types(
    tmp_path,
    field,
    value,
    message,
):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 10, best_step=5, best_value=0.30)
    trainer_state_path = checkpoint / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    trainer_state[field] = value
    trainer_state_path.write_text(
        json.dumps(trainer_state, allow_nan=True), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match=message):
        trainer._load_checkpoint(checkpoint)


@pytest.mark.parametrize("selection_value", [0.10, 0.30, 0.20])
def test_authenticated_artifact_accepts_only_exact_pre_or_derived_post_state(
    tmp_path,
    selection_value,
):
    trainer = _trainer(tmp_path)
    _checkpoint(trainer, 5, best_step=5, best_value=0.40)
    checkpoint = _checkpoint(trainer, 10, best_step=5, best_value=0.40)
    _write_checkpoint_selection_metadata(
        trainer,
        checkpoint,
        10,
        best_step=10,
        best_value=selection_value,
    )
    # Restore the immutable pre-eval trainer state after changing only the
    # resume-authoritative selection state.
    trainer_state = {
        "completed_steps": 10,
        "selection_state_schema_version": 1,
        "best_metric_name": METRIC,
        "best_metric_mode": "min",
        "best_metric_value": 0.40,
        "best_metric_step": 5,
    }
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(trainer_state, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_heldout_artifact(trainer, checkpoint, metric_value=0.20)

    if selection_value == 0.20:
        trainer._load_checkpoint(checkpoint)
        assert trainer.best_metric_value == pytest.approx(0.20)
        assert trainer.best_metric_step == 10
    else:
        with pytest.raises(RuntimeError, match="neither the authenticated pre-eval"):
            trainer._load_checkpoint(checkpoint)


def test_selection_trainer_mismatch_without_artifact_is_corruption(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 10, best_step=5, best_value=0.40)
    selection = json.loads(
        (checkpoint / "selection_state.json").read_text(encoding="utf-8")
    )
    selection.update(
        {
            "best_metric_value": 0.20,
            "best_metric_step": 10,
            "checkpoint_relative_path": "checkpoints/steps_10",
        }
    )
    (checkpoint / "selection_state.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="without an authenticated"):
        trainer._load_checkpoint(checkpoint)


def test_save_best_only_new_best_live_artifact_trusts_post_eval_checkpoint_state(
    tmp_path,
):
    trainer = _trainer(tmp_path)
    trainer.config.trainer.save_best_only = True
    checkpoint = _checkpoint(trainer, 10, best_step=10, best_value=0.20)
    selection_bytes = (checkpoint / "selection_state.json").read_bytes()
    artifact = _write_heldout_artifact(
        trainer,
        checkpoint,
        metric_value=0.20,
        checkpoint_bound=False,
    )
    artifact_bytes = artifact.read_bytes()

    trainer._load_checkpoint(checkpoint)

    assert trainer.best_metric_value == pytest.approx(0.20)
    assert trainer.best_metric_step == 10
    assert (checkpoint / "selection_state.json").read_bytes() == selection_bytes
    assert artifact.read_bytes() == artifact_bytes


def test_save_best_only_forced_worse_live_artifact_retains_historical_best(
    tmp_path,
):
    trainer = _trainer(tmp_path)
    trainer.config.trainer.save_best_only = True
    _checkpoint(trainer, 5, best_step=5, best_value=0.20)
    checkpoint = _checkpoint(trainer, 10, best_step=5, best_value=0.20)
    selection_bytes = (checkpoint / "selection_state.json").read_bytes()
    _write_heldout_artifact(
        trainer,
        checkpoint,
        metric_value=0.30,
        checkpoint_bound=False,
    )

    trainer._load_checkpoint(checkpoint)

    assert trainer.best_metric_value == pytest.approx(0.20)
    assert trainer.best_metric_step == 5
    assert (checkpoint / "selection_state.json").read_bytes() == selection_bytes


def test_non_checkpoint_bound_artifact_is_rejected_without_save_best_only(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 10, best_step=10, best_value=0.20)
    _write_heldout_artifact(
        trainer,
        checkpoint,
        metric_value=0.20,
        checkpoint_bound=False,
    )

    with pytest.raises(RuntimeError, match="only valid for save_best_only"):
        trainer._load_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_optimizer",
        "empty_scheduler",
        "symlink_model",
        "missing_rng_rank",
        "extra_rng_rank",
    ],
)
def test_incomplete_full_state_candidate_aborts_before_any_pruning(
    tmp_path,
    damage,
):
    trainer = _trainer(tmp_path, checkpoint_max_to_keep=2)
    _checkpoint(trainer, 5, best_step=5, best_value=0.30)
    _checkpoint(trainer, 10, best_step=5, best_value=0.30)
    damaged = _checkpoint(trainer, 15, best_step=5, best_value=0.30)
    trainer.completed_steps = 15
    trainer.best_metric_value = 0.30
    trainer.best_metric_step = 5
    if damage == "missing_optimizer":
        (damaged / "optimizer.bin").unlink()
    elif damage == "empty_scheduler":
        (damaged / "scheduler.bin").write_bytes(b"")
    elif damage == "symlink_model":
        model_path = damaged / "model.safetensors"
        model_path.unlink()
        target = tmp_path / "outside-model"
        target.write_bytes(b"model")
        model_path.symlink_to(target)
    elif damage == "missing_rng_rank":
        (damaged / "random_states_1.pkl").unlink()
    elif damage == "extra_rng_rank":
        (damaged / "random_states_2.pkl").write_bytes(b"extra")

    with pytest.raises(RuntimeError, match="incomplete or inconsistent"):
        trainer._prune_old_checkpoints()

    assert {
        path.name for path in Path(trainer.checkpoint_dir).glob("steps_*")
    } == {"steps_5", "steps_10", "steps_15"}


def test_rmtree_failure_cannot_report_successful_retention(
    tmp_path,
    monkeypatch,
):
    trainer = _trainer(tmp_path, checkpoint_max_to_keep=2)
    _checkpoint(trainer, 5, best_step=5, best_value=0.30)
    _checkpoint(trainer, 10, best_step=5, best_value=0.30)
    _checkpoint(trainer, 15, best_step=5, best_value=0.30)
    trainer.completed_steps = 15
    trainer.best_metric_value = 0.30
    trainer.best_metric_step = 5

    monkeypatch.setattr(
        train_starvla.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("fixture delete failure")),
    )

    with pytest.raises(RuntimeError, match="failed to establish retention invariants"):
        trainer._prune_old_checkpoints()
    assert len(list(Path(trainer.checkpoint_dir).glob("steps_*"))) == 3


def test_eval_only_separate_output_loads_source_without_pointer_or_reconciliation(
    tmp_path,
):
    source_trainer = _trainer(tmp_path / "source-run")
    source_checkpoint = _checkpoint(
        source_trainer, 10, best_step=10, best_value=0.20
    )
    eval_trainer = _trainer(tmp_path / "eval-run")
    eval_trainer.config.trainer.eval_only = True
    artifact_path = (
        Path(eval_trainer.config.output_dir)
        / "heldout_eval_metrics"
        / "step_00000010.json"
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_bytes = b'{"preexisting":"eval-only-evidence"}\n'
    artifact_path.write_bytes(artifact_bytes)

    eval_trainer._load_checkpoint(source_checkpoint)

    assert eval_trainer.loaded_checkpoint_path == str(source_checkpoint.resolve())
    assert eval_trainer.completed_steps == 10
    assert eval_trainer.best_metric_step == 10
    assert not (Path(eval_trainer.config.output_dir) / "best_checkpoint.json").exists()
    assert not any(Path(eval_trainer.checkpoint_dir).glob("steps_*"))
    assert artifact_path.read_bytes() == artifact_bytes


def test_normal_save_broadcasts_rank_zero_retention_failure(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    trainer.completed_steps = 1
    trainer.model = object()
    broadcasts = []
    monkeypatch.setattr(train_starvla.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_starvla.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(train_starvla.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(train_starvla.dist, "barrier", lambda **_kwargs: None)

    def broadcast_object_list(container, *, src, **kwargs):
        broadcasts.append((dict(container[0]), src, kwargs))

    monkeypatch.setattr(
        train_starvla.dist, "broadcast_object_list", broadcast_object_list
    )
    monkeypatch.setattr(
        trainer,
        "_prune_old_checkpoints",
        lambda: (_ for _ in ()).throw(RuntimeError("fixture retention failure")),
    )

    with pytest.raises(RuntimeError, match="fixture retention failure"):
        trainer._save_checkpoint()

    assert broadcasts[-1][0]["ok"] is False


@pytest.mark.parametrize(
    "corruption",
    [
        "artifact_dir_symlink",
        "artifact_dir_file",
        "artifact_path_symlink",
        "artifact_path_directory",
        "config_symlink",
        "schedule_symlink",
        "trainer_state_symlink",
    ],
)
def test_heldout_artifact_writer_rejects_symlink_and_nonregular_inputs(
    tmp_path,
    corruption,
):
    trainer = _trainer(tmp_path)
    trainer.completed_steps = 0
    artifact_dir = tmp_path / "heldout_eval_metrics"
    artifact_path = artifact_dir / "step_00000000.json"
    protected_path = None

    if corruption == "artifact_dir_symlink":
        protected_path = tmp_path / "external-artifact-directory"
        protected_path.mkdir()
        artifact_dir.symlink_to(protected_path, target_is_directory=True)
    elif corruption == "artifact_dir_file":
        artifact_dir.write_bytes(b"not-a-directory")
    elif corruption == "artifact_path_symlink":
        artifact_dir.mkdir()
        protected_path = tmp_path / "external-artifact.json"
        protected_path.write_bytes(b"must-not-be-read-or-replaced")
        artifact_path.symlink_to(protected_path)
    elif corruption == "artifact_path_directory":
        artifact_path.mkdir(parents=True)
    elif corruption == "config_symlink":
        config_path = tmp_path / "config.yaml"
        protected_path = tmp_path / "external-config.yaml"
        protected_path.write_bytes(config_path.read_bytes())
        config_path.unlink()
        config_path.symlink_to(protected_path)
    elif corruption == "schedule_symlink":
        schedule_path = tmp_path / "resolved_training_schedule.json"
        protected_path = tmp_path / "external-schedule.json"
        protected_path.write_bytes(schedule_path.read_bytes())
        schedule_path.unlink()
        schedule_path.symlink_to(protected_path)
    elif corruption == "trainer_state_symlink":
        trainer.completed_steps = 10
        checkpoint = _checkpoint(trainer, 10)
        trainer_state_path = checkpoint / "trainer_state.json"
        protected_path = tmp_path / "external-trainer-state.json"
        protected_path.write_bytes(trainer_state_path.read_bytes())
        trainer_state_path.unlink()
        trainer_state_path.symlink_to(protected_path)

    protected_bytes = (
        protected_path.read_bytes()
        if protected_path is not None and protected_path.is_file()
        else None
    )
    with pytest.raises(RuntimeError, match="not a regular"):
        trainer._persist_heldout_eval_artifact(
            {
                "heldout_eval_normalized_action_mae": 0.50,
                METRIC: 0.20,
            }
        )

    if protected_bytes is not None:
        assert protected_path.read_bytes() == protected_bytes
    if corruption == "artifact_dir_symlink":
        assert list(protected_path.iterdir()) == []


def test_resume_rejects_dangling_same_step_artifact_symlink(tmp_path):
    trainer = _trainer(tmp_path)
    checkpoint = _checkpoint(trainer, 10)
    artifact_path = (
        tmp_path / "heldout_eval_metrics" / "step_00000010.json"
    )
    artifact_path.parent.mkdir()
    artifact_path.symlink_to(tmp_path / "missing-artifact-target.json")

    with pytest.raises(RuntimeError, match="artifact is not a regular file"):
        trainer._load_checkpoint(checkpoint)


def test_distributed_immutable_artifact_conflict_is_broadcast(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    trainer.completed_steps = 10
    _checkpoint(trainer, 10)
    metrics = {
        "heldout_eval_normalized_action_mae": 0.50,
        METRIC: 0.20,
    }
    artifact = trainer._persist_heldout_eval_artifact(metrics)
    original_bytes = artifact.read_bytes()
    broadcasts = []
    monkeypatch.setattr(train_starvla.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_starvla.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(train_starvla.dist, "get_backend", lambda: "gloo")

    def broadcast_object_list(container, *, src, **kwargs):
        broadcasts.append((dict(container[0]), src, kwargs))

    monkeypatch.setattr(
        train_starvla.dist, "broadcast_object_list", broadcast_object_list
    )

    with pytest.raises(RuntimeError, match="heldout eval artifact persistence failed"):
        trainer._persist_heldout_eval_artifact(
            {**metrics, METRIC: 0.21}
        )

    assert broadcasts[-1][0]["ok"] is False
    assert artifact.read_bytes() == original_bytes
