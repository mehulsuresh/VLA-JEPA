from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import pytest

from scripts import h100_curriculum, h100_training
import starVLA.dataloader as dataloader_pkg


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE = "fixture:h100"
TEST_IMAGE_ID = "sha256:" + "d" * 64
SMOKE_STAGE_CONFIGS = (
    REPO_ROOT
    / "scripts/config/h100/realman_curriculum/handoff_smoke/"
    "realsource_one_step_v1.yaml",
    REPO_ROOT
    / "scripts/config/h100/realman_curriculum/handoff_smoke/"
    "intervention_one_step_v1.yaml",
    REPO_ROOT
    / "scripts/config/h100/realman_curriculum/handoff_smoke/"
    "hq_one_step_v1.yaml",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _changed_paths(
    before: Any,
    after: Any,
    *,
    prefix: str = "",
) -> set[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        changed: set[str] = set()
        for key in before.keys() | after.keys():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.add(path)
            else:
                changed.update(
                    _changed_paths(before[key], after[key], prefix=path)
                )
        return changed
    return set() if before == after else {prefix}


@pytest.mark.parametrize("config_path", SMOKE_STAGE_CONFIGS)
def test_handoff_smoke_source_yaml_owns_exact_one_step_schedule(
    config_path: Path,
):
    _, payload = h100_training._load_config(config_path)

    assert payload["trainer"]["epochs"] == 1
    assert payload["trainer"]["max_train_steps"] == 1
    assert payload["trainer"]["num_warmup_steps"] == 0
    assert payload["trainer"]["warmup_ratio"] == 0.0
    assert payload["trainer"]["checkpoint_eval_milestone_steps"] == [1]
    assert payload["trainer"]["checkpoint_eval_milestones_only"] is True


def test_stage_materialization_changes_only_runtime_identity_and_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("STARVLA_CONTAINER_IMAGE", TEST_IMAGE)
    monkeypatch.setenv("STARVLA_CONTAINER_IMAGE_ID", TEST_IMAGE_ID)
    monkeypatch.setenv("STARVLA_CONTAINER_IMAGE_DIGEST", TEST_IMAGE_ID)
    source_config = tmp_path / "source.yaml"
    source_config.write_text(
        """
run_id: reviewed_source
curriculum_stage:
  initialization: previous_stage_final
  optimizer_scheduler_rng_reset: true
  resume_policy: newest_complete_full_state_same_stage
trainer:
  is_resume: false
  resume_from_checkpoint: null
  pretrained_checkpoint: null
  pretrained_checkpoint_sha256: null
  reload_modules: null
  max_train_steps: 1
  checkpoint_eval_milestone_steps: [1]
""".lstrip(),
        encoding="utf-8",
    )
    source = OmegaConf.to_container(
        OmegaConf.load(source_config),
        resolve=True,
    )
    stage = {
        "id": "stage_b",
        "role": "adapt",
        "initialization": "previous_stage_final",
        "optimizer_scheduler_rng_reset": True,
        "resume_policy": "newest_complete_full_state_same_stage",
        "world_model_predictor_attention_backend": "torch_sdpa",
        "handoff_checkpoint_policy": "natural_final",
        "optimizer_scheduler_rng_reset": True,
        "resume_policy": "newest_complete_full_state_same_stage",
        "world_model_predictor_attention_backend": "torch_sdpa",
        "config_path": str(source_config),
        "config_sha256": _sha256(source_config),
        "model_architecture_sha256": "a" * 64,
        "frozen_train_view_manifest": "/views/stage_b.json",
        "frozen_train_view_manifest_sha256": "b" * 64,
        "local_evaluation_manifest": "/eval/stage_b.json",
        "local_evaluation_manifest_sha256": "c" * 64,
        "eligible_window_count": 128,
        "steps_per_epoch": 1,
        "expected_full_dataset_epochs": 1,
        "first_epoch_exposure_fractions": [1.0],
        "first_epoch_checkpoint_steps": [1],
        "full_epoch_boundary_steps": [1],
        "checkpoint_eval_milestone_steps": [1],
        "plan": {"runtime": {"container_image": TEST_IMAGE}},
    }
    curriculum_plan = {
        "curriculum_id": "fixture",
        "curriculum_path": "/curriculum.yaml",
        "curriculum_sha256": "d" * 64,
        "shared_contract": {
            "normalization_statistics_artifact": "/stats.json",
            "normalization_statistics_artifact_sha256": "e" * 64,
            "statistics_holdout_manifest": "/holdout.json",
            "statistics_holdout_manifest_sha256": "f" * 64,
        },
    }
    previous_handoff = {
        "handoff_checkpoint_policy": "natural_final",
        "run_dir": "/runs/stage_a",
        "run_config_sha256": "1" * 64,
        "model_architecture_sha256": "a" * 64,
        "checkpoint_step": 1,
        "checkpoint_relative_path": "checkpoints/steps_1",
        "model_path": "/runs/stage_a/checkpoints/steps_1/model.safetensors",
        "model_sha256": "2" * 64,
        "heldout_eval_sha256": "3" * 64,
        "best_metric_name": "heldout_failure",
        "best_metric_mode": "min",
        "best_metric_value": 0.9,
        "handoff_metric_name": "heldout_failure",
        "handoff_metric_mode": "min",
        "handoff_metric_value": 0.9,
    }
    output_config = tmp_path / "materialized.yaml"

    h100_curriculum._materialize_stage_config(
        curriculum_plan=curriculum_plan,
        stage=stage,
        run_id="stage_b_runtime",
        output_path=output_config,
        previous_handoff=previous_handoff,
    )
    materialized = OmegaConf.to_container(
        OmegaConf.load(output_config),
        resolve=True,
    )

    assert _changed_paths(source, materialized) == {
        "run_id",
        "trainer.pretrained_checkpoint",
        "trainer.pretrained_checkpoint_sha256",
        "curriculum_handoff",
    }


def test_external_materialized_and_resume_configs_use_tracked_source_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_config = SMOKE_STAGE_CONFIGS[1].resolve()
    source_cfg, source_payload = h100_training._load_config(source_config)
    source_sha = h100_training._config_contract_sha256(
        source_config,
        source_cfg,
    )
    base_payload = dict(source_payload)
    base_payload["run_id"] = (
        f"{source_payload['run_id']}_external_gate_test"
    )
    base_payload["curriculum_handoff"] = {
        "schema_version": 1,
        "source_stage_config_path": str(source_config),
        "source_stage_config_sha256": source_sha,
    }
    base_config = tmp_path / "outside-repo-materialized.yaml"
    base_config.write_text(
        OmegaConf.to_yaml(OmegaConf.create(base_payload), resolve=True),
        encoding="utf-8",
    )
    assert not base_config.resolve().is_relative_to(REPO_ROOT)

    checked_git_paths: list[Path] = []
    monkeypatch.setattr(
        h100_training,
        "_check_git",
        lambda path, required, **kwargs: (
            checked_git_paths.append(path.resolve())
            or {"commit": "a" * 40, "status": "clean"}
        ),
    )
    monkeypatch.setattr(h100_training, "_check_files", lambda plan: None)
    # This test isolates the authenticated tracked-source/materialization
    # gate. Dataset artifact parity is covered by the dedicated holdout,
    # statistics, and curriculum config suites.
    monkeypatch.setattr(
        h100_training,
        "_validate_dataset_artifacts",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        h100_training,
        "_check_canonical_gcs_access",
        lambda plan: None,
    )
    monkeypatch.setattr(h100_training, "_check_hardware", lambda plan: None)
    monkeypatch.setattr(h100_training, "_check_port", lambda port: None)
    monkeypatch.setattr(h100_training, "_print_plan", lambda plan: None)

    fresh_plan = h100_training.check_curriculum_materialized_plan(
        base_config,
        source_config_path=source_config,
        expected_source_config_sha256=source_sha,
        expected_materialized_config_sha256=_sha256(base_config),
        expected_run_id=base_payload["run_id"],
        expected_pretrained_checkpoint=base_payload["trainer"][
            "pretrained_checkpoint"
        ],
        expected_pretrained_checkpoint_sha256=base_payload["trainer"][
            "pretrained_checkpoint_sha256"
        ],
        expected_curriculum_handoff=base_payload["curriculum_handoff"],
        deep=False,
    )
    assert Path(fresh_plan["config_path"]) == base_config.resolve()
    assert checked_git_paths == [source_config]

    redirected_payload = copy.deepcopy(base_payload)
    redirected_payload["trainer"]["pretrained_checkpoint"] = (
        "/tmp/attacker/checkpoints/steps_999/model.safetensors"
    )
    redirected_payload["trainer"]["pretrained_checkpoint_sha256"] = (
        "f" * 64
    )
    redirected_config = tmp_path / "outside-repo-redirected.yaml"
    redirected_config.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(redirected_payload), resolve=True
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        h100_training.PlanError,
        match="independently authenticated predecessor",
    ):
        h100_training.check_curriculum_materialized_plan(
            redirected_config,
            source_config_path=source_config,
            expected_source_config_sha256=source_sha,
            expected_materialized_config_sha256=_sha256(
                redirected_config
            ),
            expected_run_id=base_payload["run_id"],
            expected_pretrained_checkpoint=None,
            expected_pretrained_checkpoint_sha256=None,
            expected_curriculum_handoff=base_payload[
                "curriculum_handoff"
            ],
            deep=False,
        )

    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "steps_1"
    checkpoint.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "trainer_state.json",
        *(f"random_states_{rank}.pkl" for rank in range(8)),
    ):
        (checkpoint / name).touch()
    (run_dir / "config.yaml").write_bytes(base_config.read_bytes())
    resume_payload = dict(base_payload)
    resume_payload["trainer"] = dict(base_payload["trainer"])
    resume_payload["trainer"]["is_resume"] = True
    resume_payload["trainer"]["resume_from_checkpoint"] = str(checkpoint)
    resume_config = tmp_path / "outside-repo-resume.yaml"
    resume_config.write_text(
        OmegaConf.to_yaml(OmegaConf.create(resume_payload), resolve=True),
        encoding="utf-8",
    )

    resume_plan = h100_training.check_curriculum_materialized_plan(
        resume_config,
        source_config_path=source_config,
        expected_source_config_sha256=source_sha,
        expected_materialized_config_sha256=_sha256(resume_config),
        base_materialized_config_path=base_config,
        expected_base_materialized_config_sha256=_sha256(base_config),
        expected_run_id=base_payload["run_id"],
        expected_pretrained_checkpoint=base_payload["trainer"][
            "pretrained_checkpoint"
        ],
        expected_pretrained_checkpoint_sha256=base_payload["trainer"][
            "pretrained_checkpoint_sha256"
        ],
        expected_curriculum_handoff=base_payload["curriculum_handoff"],
        expected_resume_checkpoint=checkpoint,
        deep=False,
    )
    assert Path(resume_plan["config_path"]) == resume_config.resolve()
    assert checked_git_paths == [source_config, source_config]

    tampered = OmegaConf.load(resume_config)
    tampered.trainer.epochs = int(tampered.trainer.epochs) + 1
    tampered_path = tmp_path / "outside-repo-resume-tampered.yaml"
    tampered_path.write_text(
        OmegaConf.to_yaml(tampered, resolve=True),
        encoding="utf-8",
    )
    with pytest.raises(
        h100_training.PlanError,
        match="outside the exact authenticated same-stage full-state",
    ):
        h100_training.check_curriculum_materialized_plan(
            tampered_path,
            source_config_path=source_config,
            expected_source_config_sha256=source_sha,
            expected_materialized_config_sha256=_sha256(tampered_path),
            base_materialized_config_path=base_config,
            expected_base_materialized_config_sha256=_sha256(base_config),
            expected_run_id=base_payload["run_id"],
            expected_pretrained_checkpoint=base_payload["trainer"][
                "pretrained_checkpoint"
            ],
            expected_pretrained_checkpoint_sha256=base_payload["trainer"][
                "pretrained_checkpoint_sha256"
            ],
            expected_curriculum_handoff=base_payload[
                "curriculum_handoff"
            ],
            expected_resume_checkpoint=checkpoint,
            deep=False,
        )


def test_materialized_curriculum_config_symlink_is_rejected_before_resolve(
    tmp_path: Path,
):
    source_config = SMOKE_STAGE_CONFIGS[1]
    source_cfg, _ = h100_training._load_config(source_config)
    source_sha = h100_training._config_contract_sha256(
        source_config, source_cfg
    )
    target = tmp_path / "materialized-target.yaml"
    target.write_text("run_id: hidden_target\n", encoding="utf-8")
    link = tmp_path / "materialized.yaml"
    link.symlink_to(target)

    with pytest.raises(h100_training.PlanError, match="non-symlink"):
        h100_training.check_curriculum_materialized_plan(
            link,
            source_config_path=source_config,
            expected_source_config_sha256=source_sha,
            expected_materialized_config_sha256=_sha256(target),
            expected_run_id="hidden_target",
            expected_pretrained_checkpoint=None,
            expected_pretrained_checkpoint_sha256=None,
            expected_curriculum_handoff={},
            deep=False,
        )


def test_base_materialized_curriculum_config_symlink_is_rejected_before_resolve(
    tmp_path: Path,
):
    source_config = SMOKE_STAGE_CONFIGS[1]
    source_cfg, _ = h100_training._load_config(source_config)
    source_sha = h100_training._config_contract_sha256(
        source_config, source_cfg
    )
    current = tmp_path / "resume.yaml"
    current.write_text("run_id: current\n", encoding="utf-8")
    base_target = tmp_path / "base-target.yaml"
    base_target.write_text("run_id: base\n", encoding="utf-8")
    base_link = tmp_path / "base.yaml"
    base_link.symlink_to(base_target)

    with pytest.raises(h100_training.PlanError, match="base.*non-symlink"):
        h100_training.check_curriculum_materialized_plan(
            current,
            source_config_path=source_config,
            expected_source_config_sha256=source_sha,
            expected_materialized_config_sha256=_sha256(current),
            expected_run_id="current",
            expected_pretrained_checkpoint=None,
            expected_pretrained_checkpoint_sha256=None,
            expected_curriculum_handoff={},
            base_materialized_config_path=base_link,
            expected_base_materialized_config_sha256=_sha256(base_target),
            expected_resume_checkpoint=tmp_path / "steps_1",
            deep=False,
        )


def test_stage_handoff_rejects_symlinked_materialized_config_before_resolve(
    tmp_path: Path,
):
    run_config = tmp_path / "config.yaml"
    run_config.write_text("run_id: stage_a\n", encoding="utf-8")
    target = tmp_path / "materialized-target.yaml"
    target.write_bytes(run_config.read_bytes())
    link = tmp_path / "materialized.yaml"
    link.symlink_to(target)

    with pytest.raises(h100_curriculum.CurriculumError, match="non-symlink"):
        h100_curriculum._validate_materialized_run_config(
            run_config=run_config,
            materialized_config=link,
            expected_materialized_config_sha256=_sha256(target),
        )


def test_immutable_run_config_yaml_symlink_is_rejected_before_resolve(
    tmp_path: Path,
):
    target = tmp_path / "target.yaml"
    target.write_text("run_id: immutable\n", encoding="utf-8")
    (tmp_path / "target.json").write_text(
        '{"run_id": "immutable"}\n', encoding="utf-8"
    )
    link = tmp_path / "config.yaml"
    link.symlink_to(target)

    with pytest.raises(h100_training.PlanError, match="non-symlink"):
        h100_training._load_immutable_run_config(link)


def test_resume_checkpoint_directory_symlink_is_rejected_before_resolve(
    tmp_path: Path,
):
    target = tmp_path / "external" / "checkpoints" / "steps_1"
    target.mkdir(parents=True)
    link_parent = tmp_path / "run" / "checkpoints"
    link_parent.mkdir(parents=True)
    link = link_parent / "steps_1"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(h100_training.PlanError, match="non-symlink"):
        h100_training._validate_checkpoint(link, expected_ranks=1)


def test_resume_checkpoint_rejects_symlinked_required_artifact(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "steps_1"
    checkpoint.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "scheduler.bin",
        "trainer_state.json",
        "random_states_0.pkl",
    ):
        (checkpoint / name).touch()
    optimizer_target = tmp_path / "optimizer.bin"
    optimizer_target.touch()
    (checkpoint / "optimizer.bin").symlink_to(optimizer_target)
    (run_dir / "config.yaml").write_text("run_id: fixture\n", encoding="utf-8")

    with pytest.raises(
        h100_training.PlanError,
        match="symlinked required artifacts",
    ):
        h100_training._validate_checkpoint(checkpoint, expected_ranks=1)


def test_curriculum_resume_materialization_strips_trainer_runtime_fields(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    immutable = OmegaConf.create(
        {
            "run_id": "stage_a",
            "run_root_dir": str(tmp_path),
            "output_dir": str(run_dir),
            "curriculum_stage": {
                "initialization": "upstream",
                "optimizer_scheduler_rng_reset": True,
                "resume_policy": (
                    h100_curriculum.CURRICULUM_RESUME_POLICY
                ),
            },
            "trainer": {
                "is_resume": False,
                "resume_from_checkpoint": None,
                "_accelerate_distributed_type": "MULTI_GPU",
                "_accelerate_gradient_accumulation_steps": 1,
                "_accelerate_num_processes": 8,
                "_accelerate_step_scheduler_with_optimizer": False,
            },
        }
    )
    OmegaConf.save(immutable, run_dir / "config.yaml")
    (run_dir / "config.json").write_text(
        json.dumps(
            OmegaConf.to_container(immutable, resolve=True),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = run_dir / "checkpoints" / "steps_1"
    checkpoint.mkdir(parents=True)
    output = tmp_path / "resume.yaml"

    digest = h100_curriculum._materialize_stage_resume_config(
        run_dir=run_dir,
        checkpoint=checkpoint,
        output_path=output,
    )
    payload = OmegaConf.to_container(OmegaConf.load(output), resolve=True)

    assert digest == _sha256(output)
    assert "output_dir" not in payload
    assert not any(
        key.startswith("_accelerate_")
        for key in payload["trainer"]
    )
    assert payload["trainer"]["is_resume"] is True
    assert payload["trainer"]["resume_from_checkpoint"] == str(checkpoint)


def test_canonical_worker_memory_safety_fails_instead_of_silent_clamp(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = {
        "enforce_worker_memory_budget": True,
        "estimated_worker_memory_gb": 5.0,
        "worker_memory_budget_fraction": 0.65,
    }
    monkeypatch.setattr(dataloader_pkg, "_host_memory_gib", lambda: 62.0)
    monkeypatch.setattr(
        dataloader_pkg,
        "_distributed_world_size",
        lambda: 1,
    )

    with pytest.raises((RuntimeError, ValueError), match="num_workers"):
        dataloader_pkg._maybe_clamp_canonical_workers_for_memory(cfg, 12)

    cfg["enforce_worker_memory_budget"] = False
    assert dataloader_pkg._maybe_clamp_canonical_workers_for_memory(cfg, 12) == 12


def test_curriculum_training_environment_scrubs_ambient_semantic_overrides(
    monkeypatch: pytest.MonkeyPatch,
):
    ambient_only = {
        "ACCELERATE_DISTRIBUTED_TYPE": "DEEPSPEED",
        "ACCELERATE_USE_DEEPSPEED": "true",
        "STARVLA_DEEPSPEED_STAGE": "3",
        "STARVLA_ALLOW_COMPILE_WITH_DEEPSPEED": "1",
        "STARVLA_DISABLE_TORCH_COMPILE": "0",
        "PER_DEVICE_BATCH_SIZE": "999",
        "DATALOADER_NUM_WORKERS": "999",
        "EPOCHS": "999",
        "MAX_TRAIN_STEPS": "999",
        "NUM_WARMUP_STEPS": "999",
        "SAVE_INTERVAL": "999",
        "EVAL_INTERVAL": "999",
    }
    for key, value in ambient_only.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STARVLA_USE_DEEPSPEED", "1")
    monkeypatch.setenv("STARVLA_ALLOW_TORCH_COMPILE", "1")
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "0")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ambient0")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "ambient0")

    env = h100_curriculum._training_environment(
        {
            "runtime": {
                "use_deepspeed": False,
                "torch_compile_environment": "disabled",
                "main_torch_threads": 1,
                "main_torch_interop_threads": 1,
                "disable_autograd_multithreading": True,
                "pytorch_cuda_alloc_conf": "expandable_segments:True",
                "tokenizers_parallelism": False,
                "network_interface": "lo",
            }
        }
    )

    assert not (set(ambient_only) & set(env))
    assert env["STARVLA_USE_DEEPSPEED"] == "0"
    assert env["STARVLA_ALLOW_TORCH_COMPILE"] == "0"
    assert env["TORCH_COMPILE_DISABLE"] == "1"
    assert env["TORCHDYNAMO_DISABLE"] == "1"
    assert env["VLA_JEPA_MAIN_TORCH_THREADS"] == "1"
    assert env["VLA_JEPA_MAIN_TORCH_INTEROP_THREADS"] == "1"
    assert env["VLA_JEPA_DISABLE_AUTOGRAD_MULTITHREADING"] == "1"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["NCCL_SOCKET_IFNAME"] == "lo"
    assert env["GLOO_SOCKET_IFNAME"] == "lo"
