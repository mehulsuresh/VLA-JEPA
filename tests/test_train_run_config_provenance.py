import hashlib
import json
from pathlib import Path
import tempfile
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
    monkeypatch.setattr(
        train_starvla,
        "_current_repository_commit",
        lambda: "1" * 40,
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
            "human_launch": {
                "container_image": "test-image:latest",
                "container_image_id": "sha256:" + "d" * 64,
            },
            "framework": {
                "action_model": {
                    "action_dim": 18,
                    "state_dim": 18,
                    "action_horizon": 50,
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
                    "num_workers": 1,
                    "multiprocessing_context": "spawn",
                    "episode_split_manifest": "holdout-global128.json",
                    "action_type": "joint_delta_gripper_absolute",
                    "state_action_normalization": "q01_q99_unclipped",
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

    incompatible_representation = _cfg(tmp_path, resume=True)
    incompatible_representation.framework.action_model.state_dim = 19
    incompatible_representation.datasets.vla_data.action_type = "absolute_qpos"
    with pytest.raises(
        RuntimeError,
        match=r"Resume configuration drift.*(state_dim|action_type)",
    ):
        train_starvla.setup_directories(incompatible_representation)
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


def _repo_resume_runtime_override(num_workers: int) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".resume-runtime-test-",
        suffix=".yaml",
        dir=Path(train_starvla.__file__).resolve().parents[2] / "tests",
        delete=False,
    )
    try:
        handle.write(
            "schema_version: 1\n"
            "datasets:\n"
            "  vla_data:\n"
            f"    num_workers: {num_workers}\n"
            "    multiprocessing_context: forkserver\n"
        )
    finally:
        handle.close()
    return Path(handle.name)


def _resume_runtime_metadata(
    source: Path,
    *,
    previous: int = 1,
    resumed: int = 4,
    previous_context: str = "spawn",
    resumed_context: str = "forkserver",
) -> dict:
    helper = (
        Path(train_starvla.__file__).resolve().parents[2]
        / "scripts/h100_resume_runtime.py"
    )
    return {
        "schema": "starvla-resume-runtime-override-v1",
        "resume_helper_path": str(helper.resolve()),
        "resume_helper_sha256": hashlib.sha256(
            helper.read_bytes()
        ).hexdigest(),
        "source_commit": "1" * 40,
        "generated_utc": "2026-07-24T04:00:00+00:00",
        "container_image": "test-image:latest",
        "container_image_digest": "sha256:" + "d" * 64,
        "runtime_config_path": str(source.resolve()),
        "runtime_config_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "changes": {
            "datasets.vla_data.num_workers": {
                "previous": previous,
                "resumed": resumed,
            },
            "datasets.vla_data.multiprocessing_context": {
                "previous": previous_context,
                "resumed": resumed_context,
            },
        },
    }


def test_resume_worker_override_is_validated_and_snapshotted(tmp_path):
    output_dir = train_starvla.setup_directories(_cfg(tmp_path))
    source_yaml = (output_dir / "config.yaml").read_bytes()
    source_json = (output_dir / "config.json").read_bytes()
    override = _repo_resume_runtime_override(4)
    try:
        resumed = _cfg(tmp_path, resume=True)
        resumed.datasets.vla_data.num_workers = 4
        resumed.datasets.vla_data.multiprocessing_context = "forkserver"
        resumed.resume_runtime_override = _resume_runtime_metadata(override)

        train_starvla.setup_directories(resumed)
        snapshot = train_starvla._persist_pending_resume_invocation_snapshot(
            output_dir,
            resumed,
        )

        assert (output_dir / "config.yaml").read_bytes() == source_yaml
        assert (output_dir / "config.json").read_bytes() == source_json
        snapshot_cfg = OmegaConf.load(snapshot)
        assert snapshot_cfg.datasets.vla_data.num_workers == 4
        assert (
            snapshot_cfg.resume_runtime_override.runtime_config_sha256
            == hashlib.sha256(override.read_bytes()).hexdigest()
        )
        active = train_starvla._ACTIVE_RESUME_INVOCATION[
            str(output_dir.resolve())
        ]
        assert active["snapshot_path"] == snapshot
        assert active["sha256"] == hashlib.sha256(
            snapshot.read_bytes()
        ).hexdigest()
        assert active["resumed_from_step"] == 10
    finally:
        override.unlink()


def test_resume_worker_override_fails_closed_without_valid_metadata(tmp_path):
    output_dir = train_starvla.setup_directories(_cfg(tmp_path))
    source_yaml = (output_dir / "config.yaml").read_bytes()

    raw_drift = _cfg(tmp_path, resume=True)
    raw_drift.datasets.vla_data.num_workers = 4
    raw_drift.datasets.vla_data.multiprocessing_context = "forkserver"
    with pytest.raises(RuntimeError, match="requires validated"):
        train_starvla.setup_directories(raw_drift)

    override = _repo_resume_runtime_override(4)
    try:
        tampered_metadata = _cfg(tmp_path, resume=True)
        tampered_metadata.datasets.vla_data.num_workers = 4
        tampered_metadata.datasets.vla_data.multiprocessing_context = (
            "forkserver"
        )
        tampered_metadata.resume_runtime_override = _resume_runtime_metadata(
            override,
            previous=2,
        )
        with pytest.raises(RuntimeError, match="previous/resumed"):
            train_starvla.setup_directories(tampered_metadata)

        no_change = _cfg(tmp_path, resume=True)
        no_change.resume_runtime_override = _resume_runtime_metadata(
            override,
            resumed=1,
            resumed_context="spawn",
        )
        with pytest.raises(RuntimeError, match="forbidden without"):
            train_starvla.setup_directories(no_change)
    finally:
        override.unlink()

    assert (output_dir / "config.yaml").read_bytes() == source_yaml
    assert not (output_dir / "resume_invocations").exists()


def test_resume_worker_override_rejects_tampered_source_file(tmp_path):
    output_dir = train_starvla.setup_directories(_cfg(tmp_path))
    override = _repo_resume_runtime_override(4)
    try:
        resumed = _cfg(tmp_path, resume=True)
        resumed.datasets.vla_data.num_workers = 4
        resumed.datasets.vla_data.multiprocessing_context = "forkserver"
        resumed.resume_runtime_override = _resume_runtime_metadata(override)
        override.write_text(
            "schema_version: 1\n"
            "datasets:\n"
            "  vla_data:\n"
            "    num_workers: 5\n"
            "    multiprocessing_context: forkserver\n",
            encoding="utf-8",
        )

        with pytest.raises(
            RuntimeError,
            match="runtime config SHA-256 does not match",
        ):
            train_starvla.setup_directories(resumed)
    finally:
        override.unlink()

    assert not (output_dir / "resume_invocations").exists()


def test_checkpoint_resume_invocation_sidecar_is_self_validating(tmp_path):
    checkpoint = tmp_path / "checkpoints/steps_20"
    checkpoint.mkdir(parents=True)
    snapshot = OmegaConf.create(
        {
            "trainer": {
                "resume_from_checkpoint": "/run/checkpoints/steps_10",
            },
            "datasets": {
                "vla_data": {
                    "num_workers": 4,
                    "multiprocessing_context": "forkserver",
                }
            },
            "resume_runtime_override": {"schema": "fixture"},
        }
    )
    sidecar = checkpoint / "resume_invocation.yaml"
    sidecar.write_text(
        OmegaConf.to_yaml(snapshot, resolve=True),
        encoding="utf-8",
    )
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    trainer_state = {
        "completed_steps": 20,
        "resume_invocation": {
            "schema_version": 1,
            "file": "resume_invocation.yaml",
            "sha256": digest,
            "resumed_from_step": 10,
            "runtime_override_required": True,
        },
    }
    current = OmegaConf.create(
        {
            "datasets": {
                "vla_data": {
                    "num_workers": 4,
                    "multiprocessing_context": "forkserver",
                }
            },
            "resume_runtime_override": {"schema": "fixture"},
        }
    )

    train_starvla._validate_checkpoint_resume_invocation(
        checkpoint,
        trainer_state,
        current,
    )

    without_override = OmegaConf.create(
        {
            "datasets": {
                "vla_data": {
                    "num_workers": 1,
                    "multiprocessing_context": "spawn",
                }
            }
        }
    )
    with pytest.raises(RuntimeError, match="runtime-corrected resume"):
        train_starvla._validate_checkpoint_resume_invocation(
            checkpoint,
            trainer_state,
            without_override,
        )

    sidecar.write_text(sidecar.read_text(encoding="utf-8") + "# tampered\n")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        train_starvla._validate_checkpoint_resume_invocation(
            checkpoint,
            trainer_state,
            current,
        )


def test_legacy_checkpoint_without_resume_lineage_remains_valid(tmp_path):
    train_starvla._validate_checkpoint_resume_invocation(
        tmp_path,
        {"completed_steps": 10},
        OmegaConf.create({}),
    )


def test_post_resume_checkpoint_embeds_exact_invocation(tmp_path):
    output_dir = tmp_path / "run"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    snapshot = output_dir / "resume_invocations/steps_10-fixture.yaml"
    snapshot.parent.mkdir()
    snapshot.write_text(
        "trainer:\n"
        "  resume_from_checkpoint: /run/checkpoints/steps_10\n"
        "datasets:\n"
        "  vla_data:\n"
        "    num_workers: 4\n"
        "    multiprocessing_context: forkserver\n"
        "resume_runtime_override:\n"
        "  schema: fixture\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    key = str(output_dir.resolve())
    train_starvla._ACTIVE_RESUME_INVOCATION[key] = {
        "snapshot_path": snapshot,
        "sha256": digest,
        "resumed_from_step": 10,
    }

    class _Accelerator:
        def save_state(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

        def wait_for_everyone(self):
            return None

        def print(self, *_args, **_kwargs):
            return None

    trainer = object.__new__(train_starvla.VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "resume_runtime_override": {"schema": "fixture"},
            "trainer": {
                "save_plain_weights_in_checkpoints": False,
                "drop_checkpoint_page_cache": False,
                "trim_process_memory_after_checkpoint": False,
            },
        }
    )
    trainer.checkpoint_dir = str(checkpoint_dir)
    trainer.completed_steps = 20
    trainer.best_metric_name = "heldout_eval_score"
    trainer.best_metric_mode = "min"
    trainer.best_metric_value = None
    trainer.best_metric_step = None
    trainer.accelerator = _Accelerator()

    try:
        trainer._save_checkpoint(prune=False)
        checkpoint = checkpoint_dir / "steps_20"
        embedded = checkpoint / "resume_invocation.yaml"
        assert embedded.read_bytes() == snapshot.read_bytes()
        state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        assert state["resume_invocation"] == {
            "schema_version": 1,
            "file": "resume_invocation.yaml",
            "sha256": digest,
            "resumed_from_step": 10,
            "runtime_override_required": True,
        }
    finally:
        train_starvla._ACTIVE_RESUME_INVOCATION.pop(key, None)


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
        "evaluation_observation_count": 128,
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


def test_resolved_schedule_validates_and_records_checkpoint_eval_milestones(
    tmp_path,
):
    cfg = _cfg(tmp_path, max_train_steps=15)
    cfg.trainer.checkpoint_eval_milestone_steps = [3, 10, 15]
    cfg.trainer.checkpoint_eval_milestones_only = True

    schedule = train_starvla.resolve_training_schedule(
        cfg,
        _SizedLoader(256),
        num_processes=8,
    )

    assert schedule["configured"]["checkpoint_eval_milestone_steps"] == [
        3,
        10,
        15,
    ]
    assert schedule["resolved"]["checkpoint_eval_milestone_steps"] == [
        3,
        10,
        15,
    ]
    assert (
        schedule["configured"]["checkpoint_eval_milestones_only"] is True
    )
    assert schedule["resolved"]["checkpoint_eval_milestones_only"] is True


def test_config_owned_fraction_policy_resolves_first_epoch_and_all_epoch_boundaries(
    tmp_path,
):
    cfg = _cfg(tmp_path, max_train_steps="auto")
    cfg.trainer.checkpoint_eval_milestone_fractions = [0.25, 0.5, 1.0]
    cfg.trainer.checkpoint_eval_milestone_steps = "auto"
    cfg.trainer.checkpoint_eval_include_full_epoch_boundaries = True
    cfg.trainer.checkpoint_eval_milestones_only = True
    cfg.trainer.checkpoint_max_to_keep = 0

    schedule = train_starvla.resolve_training_schedule(
        cfg,
        _SizedLoader(32),
        num_processes=8,
    )

    assert schedule["configured"][
        "checkpoint_eval_milestone_fractions"
    ] == [0.25, 0.5, 1.0]
    assert (
        schedule["configured"]["checkpoint_eval_milestone_steps"] == "auto"
    )
    assert (
        schedule["configured"][
            "checkpoint_eval_include_full_epoch_boundaries"
        ]
        is True
    )
    assert schedule["resolved"]["steps_per_epoch"] == 4
    assert schedule["resolved"]["max_train_steps"] == 12
    assert schedule["resolved"]["checkpoint_eval_milestone_steps"] == [
        1,
        2,
        4,
        8,
        12,
    ]


def test_milestone_only_schedule_rejects_retention_that_would_prune_evidence(
    tmp_path,
):
    cfg = _cfg(tmp_path, max_train_steps=15)
    cfg.trainer.checkpoint_eval_milestone_steps = [3, 10, 15]
    cfg.trainer.checkpoint_eval_milestones_only = True
    cfg.trainer.checkpoint_max_to_keep = 2

    with pytest.raises(ValueError, match="cannot retain every intended"):
        train_starvla.resolve_training_schedule(
            cfg,
            _SizedLoader(256),
            num_processes=8,
        )


@pytest.mark.parametrize(
    "milestones",
    ([3, 3], [4, 2], [0, 2], [2, 16], [True, 2], "2,3"),
)
def test_resolved_schedule_rejects_invalid_checkpoint_eval_milestones(
    tmp_path,
    milestones,
):
    cfg = _cfg(tmp_path, max_train_steps=15)
    cfg.trainer.checkpoint_eval_milestone_steps = milestones

    with pytest.raises(
        ValueError,
        match="checkpoint_eval_milestone_steps",
    ):
        train_starvla.resolve_training_schedule(
            cfg,
            _SizedLoader(256),
            num_processes=8,
        )


def test_resolved_schedule_records_eval_cardinality_independent_of_train_batch(
    tmp_path,
):
    cfg = _cfg(tmp_path, max_train_steps=15)
    cfg.datasets.vla_data.per_device_batch_size = 13
    cfg.datasets.vla_data.holdout_sampling = {
        "evaluation_observation_count": 128,
    }

    schedule = train_starvla.resolve_training_schedule(
        cfg,
        _SizedLoader(256),
        num_processes=8,
    )

    assert schedule["resolved"]["effective_global_batch_size"] == 104
    assert schedule["resolved"]["evaluation_observation_count"] == 128


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
