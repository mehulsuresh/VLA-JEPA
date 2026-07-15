import hashlib
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from starVLA.training import train_starvla


@pytest.fixture(autouse=True)
def _plain_logger(monkeypatch):
    monkeypatch.setattr(
        train_starvla,
        "logger",
        SimpleNamespace(
            info=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
        ),
    )


class _SizedLoader:
    drop_last = False
    sampler = None

    def __init__(self, length):
        self.length = int(length)

    def __len__(self):
        return self.length


def _cfg(tmp_path, *, resume=False, max_train_steps=15):
    return OmegaConf.create(
        {
            "run_root_dir": str(tmp_path),
            "run_id": "lifecycle",
            "seed": 42,
            "framework": {
                "action_model": {
                    "rtc_training": {
                        "enabled": False,
                        "rtc_prob": 0.0,
                        "warmup_steps": 0,
                        "ramp_steps": 0,
                    }
                },
                "depth_teacher_aux": {
                    "enabled": True,
                    "detach_vlm_steps": 5,
                    "detach_vlm_fraction": 0.01,
                },
            },
            "datasets": {
                "vla_data": {
                    "per_device_batch_size": 16,
                    "episode_split_manifest": "holdout-global128.json",
                    "task_text_overrides": {0: "zero-key fixture"},
                }
            },
            "trainer": {
                "epochs": 3,
                "max_train_steps": max_train_steps,
                "num_warmup_steps": 2250,
                "save_interval": 5,
                "eval_interval": 5,
                "checkpoint_max_to_keep": 3,
                "save_best_only": False,
                "best_metric_name": "heldout_eval_score",
                "best_metric_mode": "min",
                "eval_before_train": True,
                "logging_frequency": 1,
                "gradient_accumulation_steps": 1,
                "step_scheduler_with_optimizer": False,
                "lr_scheduler_type": "cosine_with_min_lr",
                "scheduler_specific_kwargs": {"min_lr": 1.0e-6},
                "learning_rate": {"base": 2.0e-5, "action_model": 1.0e-4},
                "loss_scale": {
                    "wm": 0.1,
                    "wm_initial": 0.3,
                    "wm_warmup_steps": 1500,
                },
                "is_resume": resume,
                "resume_from_checkpoint": (
                    str(tmp_path / "lifecycle/checkpoints/steps_10")
                    if resume
                    else None
                ),
                "resume_epoch": None,
                "resume_step": None,
                "resume_load_optimizer_state": True,
            },
        }
    )


def test_resume_preserves_source_config_and_records_invocation(tmp_path):
    fresh = _cfg(tmp_path)
    output_dir = train_starvla.setup_directories(fresh)
    source_yaml = (output_dir / "config.yaml").read_bytes()
    source_json = (output_dir / "config.json").read_bytes()
    with pytest.raises(RuntimeError, match="Fresh training output directory is not empty"):
        train_starvla.setup_directories(_cfg(tmp_path))
    assert (output_dir / "config.yaml").read_bytes() == source_yaml
    assert (output_dir / "config.json").read_bytes() == source_json

    resumed = _cfg(tmp_path, resume=True)
    train_starvla.setup_directories(resumed)

    assert (output_dir / "config.yaml").read_bytes() == source_yaml
    assert (output_dir / "config.json").read_bytes() == source_json
    assert not (output_dir / "resume_invocations").exists()
    invocation_payload = OmegaConf.to_yaml(resumed, resolve=True).encode("utf-8")
    resumed.trainer.max_train_steps = 999
    resumed.trainer.micro_batches_per_epoch = 123
    train_starvla._persist_pending_resume_invocation_snapshot(
        output_dir,
        resumed,
    )
    snapshots = list((output_dir / "resume_invocations").glob("steps_10-*.yaml"))
    assert len(snapshots) == 1
    snapshot = OmegaConf.load(snapshots[0])
    assert snapshot.trainer.is_resume is True
    assert snapshots[0].read_bytes() == invocation_payload
    assert snapshot.trainer.max_train_steps == 15
    assert snapshot.trainer.get("micro_batches_per_epoch", None) is None
    assert snapshot.trainer.resume_from_checkpoint.endswith(
        "/checkpoints/steps_10"
    )
    snapshots[0].write_bytes(b"corrupt snapshot")
    with pytest.raises(RuntimeError, match="snapshot is immutable"):
        train_starvla._persist_resume_invocation_snapshot(
            output_dir,
            resumed,
            invocation_payload,
        )


def test_resume_semantic_drift_fails_without_mutating_source(tmp_path):
    fresh = _cfg(tmp_path)
    output_dir = train_starvla.setup_directories(fresh)
    source_yaml = (output_dir / "config.yaml").read_bytes()
    source_json = (output_dir / "config.json").read_bytes()

    resumed = _cfg(tmp_path, resume=True)
    resumed.trainer.learning_rate.base = 3.0e-5
    with pytest.raises(
        RuntimeError,
        match=r"Resume configuration drift.*trainer\.learning_rate\.base",
    ):
        train_starvla.setup_directories(resumed)

    assert (output_dir / "config.yaml").read_bytes() == source_yaml
    assert (output_dir / "config.json").read_bytes() == source_json
    assert not list((output_dir / "resume_invocations").glob("steps_10-*.yaml"))

    model_only = _cfg(tmp_path, resume=True)
    model_only.trainer.resume_load_optimizer_state = False
    with pytest.raises(RuntimeError, match="resume_load_optimizer_state"):
        train_starvla.setup_directories(model_only)
    assert (output_dir / "config.yaml").read_bytes() == source_yaml
    assert not list((output_dir / "resume_invocations").glob("steps_10-*.yaml"))

    stale_step = _cfg(tmp_path, resume=True)
    stale_step.trainer.resume_step = 10
    with pytest.raises(RuntimeError, match="trainer.resume_step"):
        train_starvla.setup_directories(stale_step)
    assert (output_dir / "config.yaml").read_bytes() == source_yaml


def test_existing_resume_output_without_source_config_fails_closed(tmp_path):
    checkpoint = tmp_path / "lifecycle/checkpoints/steps_10"
    checkpoint.mkdir(parents=True)
    resumed = _cfg(tmp_path, resume=True)

    with pytest.raises(RuntimeError, match="missing its immutable source config"):
        train_starvla.setup_directories(resumed)

    assert not (tmp_path / "lifecycle/config.yaml").exists()
    assert not (tmp_path / "lifecycle/resume_invocations").exists()


def test_resolved_schedule_is_atomic_immutable_and_captures_auto_resolution(
    tmp_path,
):
    fresh = _cfg(tmp_path, max_train_steps="auto")
    output_dir = train_starvla.setup_directories(fresh)
    schedule = train_starvla.resolve_training_schedule(
        fresh,
        _SizedLoader(256),
        num_processes=8,
    )
    identity = train_starvla.persist_resolved_training_schedule(fresh, schedule)
    schedule_path = output_dir / "resolved_training_schedule.json"
    original_bytes = schedule_path.read_bytes()
    payload = json.loads(original_bytes)

    assert schedule["configured"]["max_train_steps"] == "auto"
    assert schedule["resolved"] == {
        "epochs": 3,
        "micro_batches_per_epoch": 32,
        "steps_per_epoch": 32,
        "max_train_steps": 96,
        "num_warmup_steps": 2250,
        "save_interval": 5,
        "eval_interval": 5,
        "gradient_accumulation_steps": 1,
        "per_device_batch_size": 16,
        "num_processes": 8,
        "effective_global_batch_size": 128,
        "step_scheduler_with_optimizer": False,
        "wm_warmup_steps": 1500,
        "depth_teacher_detach_steps": 5,
        "depth_teacher_detach_steps_floor": 5,
        "depth_teacher_detach_fraction": 0.01,
        "rtc_enabled": False,
        "rtc_warmup_steps": 0,
        "rtc_ramp_steps": 0,
    }
    assert payload["source_config"] == {
        "path": "config.yaml",
        "sha256": hashlib.sha256(
            (output_dir / "config.yaml").read_bytes()
        ).hexdigest(),
    }
    assert identity == {
        "path": "resolved_training_schedule.json",
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    assert list(output_dir.glob(".resolved_training_schedule.json.tmp-*")) == []

    resumed = _cfg(tmp_path, resume=True, max_train_steps="auto")
    train_starvla.setup_directories(resumed)
    same_schedule = train_starvla.resolve_training_schedule(
        resumed,
        _SizedLoader(256),
        num_processes=8,
    )
    same_identity = train_starvla.persist_resolved_training_schedule(
        resumed,
        same_schedule,
    )
    assert same_identity == identity
    assert schedule_path.read_bytes() == original_bytes
    train_starvla._persist_pending_resume_invocation_snapshot(
        output_dir,
        resumed,
    )
    assert len(list((output_dir / "resume_invocations").glob("steps_10-*.yaml"))) == 1

    drifted = _cfg(tmp_path, resume=True, max_train_steps="auto")
    drifted.trainer.resume_from_checkpoint = str(
        output_dir / "checkpoints/steps_5"
    )
    train_starvla.setup_directories(drifted)
    drifted_schedule = train_starvla.resolve_training_schedule(
        drifted,
        _SizedLoader(264),
        num_processes=8,
    )
    with pytest.raises(
        RuntimeError,
        match=r"Resolved training schedule drift.*max_train_steps",
    ):
        train_starvla.persist_resolved_training_schedule(
            drifted,
            drifted_schedule,
        )
    assert schedule_path.read_bytes() == original_bytes
    assert not list((output_dir / "resume_invocations").glob("steps_5-*.yaml"))


def test_explicit_lifecycle_schedule_keeps_identical_scheduler_definition(
    tmp_path,
):
    cfg = _cfg(tmp_path, max_train_steps=15)
    train_starvla.setup_directories(cfg)

    schedule = train_starvla.resolve_training_schedule(
        cfg,
        _SizedLoader(256),
        num_processes=8,
    )

    assert schedule["configured"] == {
        "epochs": 3,
        "max_train_steps": 15,
        "num_warmup_steps": 2250,
        "save_interval": 5,
        "eval_interval": 5,
    }
    assert schedule["resolved"]["max_train_steps"] == 15
    assert schedule["resolved"]["num_warmup_steps"] == 2250
    assert schedule["resolved"]["save_interval"] == 5
    assert schedule["resolved"]["eval_interval"] == 5
    assert schedule["resolved"]["step_scheduler_with_optimizer"] is False
    assert schedule["resolved"]["wm_warmup_steps"] == 1500
    assert schedule["resolved"]["rtc_enabled"] is False


def test_schedule_evidence_allows_configs_without_optional_framework_sections(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    del cfg.framework
    train_starvla.setup_directories(cfg)

    schedule = train_starvla.resolve_training_schedule(
        cfg,
        _SizedLoader(256),
        num_processes=8,
    )

    assert schedule["resolved"]["depth_teacher_detach_steps"] == 0
    assert schedule["resolved"]["rtc_enabled"] is False
