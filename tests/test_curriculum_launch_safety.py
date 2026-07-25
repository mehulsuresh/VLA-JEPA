from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

from omegaconf import OmegaConf
import pytest
import torch
from safetensors.torch import save_file as save_safetensors

from scripts import h100_curriculum
from starVLA.training import train_starvla
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


TEST_IMAGE = "fixture:h100"
TEST_IMAGE_ID = "sha256:" + "d" * 64


def _set_test_image_identity(monkeypatch) -> None:
    monkeypatch.setenv("STARVLA_CONTAINER_IMAGE", TEST_IMAGE)
    monkeypatch.setenv("STARVLA_CONTAINER_IMAGE_ID", TEST_IMAGE_ID)
    monkeypatch.setenv("STARVLA_CONTAINER_IMAGE_DIGEST", TEST_IMAGE_ID)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_human_wrapper_exposes_only_explicit_authenticated_resume():
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "h100_curriculum.sh"
    )
    subprocess.run(["bash", "-n", str(wrapper)], check=True)

    missing_id = subprocess.run(
        [
            "bash",
            str(wrapper),
            "resume",
            "--config",
            "/does/not/need/to/exist/for/this/error.yaml",
        ],
        text=True,
        capture_output=True,
    )
    assert missing_id.returncode == 2
    assert "resume requires the original --run-id" in missing_id.stderr

    override = subprocess.run(
        [
            "bash",
            str(wrapper),
            "resume",
            "--run-id",
            "fixture",
            "--learning-rate",
            "1e-3",
        ],
        text=True,
        capture_output=True,
    )
    assert override.returncode == 2
    assert "unknown option --learning-rate" in override.stderr

    source = wrapper.read_text(encoding="utf-8")
    assert "h100_curriculum.py setup" in source
    assert "h100_curriculum.py run" in source
    assert '--run-id "${RUN_ID}"' in source
    assert "--resume" in source
    assert (
        'run_curriculum_container "starvla-curriculum-check-$$" gpus'
        in source
    )
    assert 'run_curriculum_container "${name}" gpus' in source
    assert "resolve_container_image_id" in source
    assert 'IMAGE="${image_reference}"' in source
    assert 'STARVLA_CONTAINER_IMAGE_ID="${CONTAINER_IMAGE_ID}"' in source


@pytest.mark.parametrize(
    "command",
    ("setup", "plan", "check", "start", "resume"),
)
def test_human_curriculum_commands_require_explicit_config(command: str):
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "h100_curriculum.sh"
    )
    result = subprocess.run(
        ["bash", str(wrapper), command],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert (
        f"{command} requires explicit --config YAML" in result.stderr
    )


def test_human_curriculum_wrapper_rejects_config_symlink(
    tmp_path: Path,
):
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "h100_curriculum.sh"
    )
    target = tmp_path / "curriculum-target.yaml"
    target.write_text("schema_version: 2\n", encoding="utf-8")
    link = tmp_path / "curriculum.yaml"
    link.symlink_to(target)

    result = subprocess.run(
        ["bash", str(wrapper), "plan", "--config", str(link)],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "regular non-symlink file" in result.stderr


@pytest.mark.parametrize("command", ("plan", "setup", "check", "run"))
def test_python_curriculum_commands_require_explicit_config(command: str):
    with pytest.raises(SystemExit):
        h100_curriculum._parser().parse_args([command])


def test_curriculum_requires_explicit_workflow_kind(monkeypatch):
    monkeypatch.setattr(
        h100_curriculum,
        "_load_curriculum",
        lambda _path: (
            OmegaConf.create({}),
            {
                "schema_version": 2,
                "curriculum_id": "fixture_curriculum",
            },
        ),
    )
    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="workflow_kind must be explicitly declared",
    ):
        h100_curriculum.resolve_curriculum(Path("/unused.yaml"))


def test_curriculum_config_symlink_is_rejected_before_resolve(
    tmp_path: Path,
):
    target = tmp_path / "curriculum-target.yaml"
    target.write_text("schema_version: 2\n", encoding="utf-8")
    link = tmp_path / "curriculum.yaml"
    link.symlink_to(target)

    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="non-symlink",
    ):
        h100_curriculum._load_curriculum(link)


def test_curriculum_state_file_symlink_is_rejected(tmp_path: Path):
    target = tmp_path / "state-target.json"
    h100_curriculum._write_curriculum_state(
        target,
        {
            "schema_version": 3,
            "curriculum_id": "fixture_curriculum",
        },
    )
    link = tmp_path / "curriculum_state.json"
    link.symlink_to(target)

    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="regular file",
    ):
        h100_curriculum._load_curriculum_state(link)


def test_curriculum_resume_state_directory_symlink_is_rejected(
    tmp_path: Path,
):
    state_root = tmp_path / "curricula"
    state_root.mkdir()
    target = tmp_path / "external-state"
    target.mkdir()
    run_id = "fixture_curriculum_symlink_resume"
    (state_root / run_id).symlink_to(target, target_is_directory=True)
    plan = {
        "curriculum_id": "fixture_curriculum",
        "state_root_dir": str(state_root),
    }

    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="non-symlink directory",
    ):
        h100_curriculum.run_curriculum(
            plan,
            run_id=run_id,
            resume=True,
        )


def test_curriculum_handoff_checkpoint_symlink_is_rejected_before_resolve(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    target = tmp_path / "external" / "steps_1"
    target.mkdir(parents=True)
    (checkpoint_root / "steps_1").symlink_to(
        target, target_is_directory=True
    )
    (run_dir / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "best_metric_step": 1,
                "checkpoint_relative_path": "checkpoints/steps_1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="checkpoint.*symlink",
    ):
        h100_curriculum._validate_selection_handoff(run_dir)


def _dependency_stage(
    *,
    stage_id: str,
    config: Path,
    profile: str,
    helpers: dict,
    requires_gcloud: bool,
    dataset_py: str,
    video_backend: str | None,
    multiprocessing_context: str = "forkserver",
    persistent_workers: bool = True,
    gpu_video_decode_on_rank: bool = False,
) -> dict:
    training = {
        "dataset_profile": profile,
        "requires_gcloud": requires_gcloud,
        "canonical_bucket_root": (
            "gs://fixture" if requires_gcloud else None
        ),
        "canonical_gcs_probe_object": (
            "gs://fixture/probe" if requires_gcloud else None
        ),
    }
    data = {
        "dataset_py": dataset_py,
        "multiprocessing_context": multiprocessing_context,
        "persistent_workers": persistent_workers,
        "gpu_video_decode_on_rank": gpu_video_decode_on_rank,
    }
    if video_backend is not None:
        data["video_backend"] = video_backend
    return {
        "id": stage_id,
        "config_path": str(config),
        "plan": {
            "runtime": {
                "container_image": "fixture:h100",
                "helper_repositories": helpers,
            },
            "training": training,
        },
        "payload": {"datasets": {"vla_data": data}},
    }


def test_curriculum_setup_covers_every_unique_stage_dependency(
    tmp_path: Path,
    monkeypatch,
):
    canonical_config = tmp_path / "canonical.yaml"
    intervention_config = tmp_path / "intervention.yaml"
    hq_config = tmp_path / "hq.yaml"
    common_helpers = {
        "moge": {
            "path": "/scratch/src/moge",
            "url": "https://example.invalid/moge.git",
            "commit": "a" * 40,
        }
    }
    canonical_helpers = common_helpers | {
        "canonical": {
            "path": "/scratch/src/canonical",
            "url": "https://example.invalid/canonical.git",
            "commit": "b" * 40,
        }
    }
    stages = [
        _dependency_stage(
            stage_id="canonical",
            config=canonical_config,
            profile="canonical_gcs",
            helpers=canonical_helpers,
            requires_gcloud=True,
            dataset_py="canonical_subset_vla",
            video_backend="pyav",
        ),
        _dependency_stage(
            stage_id="intervention",
            config=intervention_config,
            profile="realman_lerobot",
            helpers=common_helpers,
            requires_gcloud=False,
            dataset_py="lerobot_datasets",
            video_backend="pyav",
        ),
        _dependency_stage(
            stage_id="hq",
            config=hq_config,
            profile="realman_lerobot",
            helpers=common_helpers,
            requires_gcloud=False,
            dataset_py="lerobot_datasets",
            video_backend="pyav",
        ),
    ]
    calls: list[Path] = []
    monkeypatch.setattr(
        h100_curriculum.h100_training,
        "setup_dependencies",
        lambda config: calls.append(config),
    )

    h100_curriculum.setup({"stages": stages})

    assert calls == [canonical_config, intervention_config]


def test_curriculum_check_deep_preflights_each_distinct_stage_backend(
    tmp_path: Path,
    monkeypatch,
):
    stages = [
        _dependency_stage(
            stage_id="canonical",
            config=tmp_path / "canonical.yaml",
            profile="canonical_gcs",
            helpers={},
            requires_gcloud=True,
            dataset_py="canonical_subset_vla",
            video_backend="pyav",
        ),
        _dependency_stage(
            stage_id="intervention",
            config=tmp_path / "intervention.yaml",
            profile="realman_lerobot",
            helpers={},
            requires_gcloud=False,
            dataset_py="lerobot_datasets",
            video_backend="pyav",
        ),
        _dependency_stage(
            stage_id="hq",
            config=tmp_path / "hq.yaml",
            profile="realman_lerobot",
            helpers={},
            requires_gcloud=False,
            dataset_py="lerobot_datasets",
            video_backend="pyav",
            multiprocessing_context="spawn",
        ),
    ]
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        h100_curriculum.h100_training,
        "check_plan",
        lambda config, *, deep: calls.append((config, deep)),
    )

    h100_curriculum.check({"stages": stages})

    assert calls == [
        (tmp_path / "canonical.yaml", True),
        (tmp_path / "intervention.yaml", True),
        (tmp_path / "hq.yaml", True),
    ]


def test_curriculum_run_preflights_before_creating_run_state(
    tmp_path: Path,
    monkeypatch,
):
    plan = {"curriculum_id": "fixture"}
    events: list[str] = []
    monkeypatch.setattr(
        h100_curriculum,
        "resolve_curriculum",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        h100_curriculum,
        "check",
        lambda actual: events.append("check")
        if actual is plan
        else None,
    )
    monkeypatch.setattr(
        h100_curriculum,
        "run_curriculum",
        lambda actual, **_kwargs: events.append("run")
        if actual is plan
        else None,
    )

    assert (
        h100_curriculum.main(
            ["run", "--config", str(tmp_path / "curriculum.yaml")]
        )
        == 0
    )
    assert events == ["check", "run"]


@pytest.mark.parametrize(
    ("run_args", "expected_error"),
    (
        (
            ["--run-id", "different_curriculum_unit"],
            "curriculum run ID must start with fixture_",
        ),
        (
            ["--run-id", "INVALID RUN ID"],
            "invalid curriculum run ID",
        ),
        (
            ["--resume"],
            "curriculum resume requires the original --run-id",
        ),
    ),
)
def test_curriculum_run_rejects_invalid_request_before_preflight(
    tmp_path: Path,
    monkeypatch,
    capsys,
    run_args: list[str],
    expected_error: str,
):
    plan = {"curriculum_id": "fixture"}
    events: list[str] = []
    monkeypatch.setattr(
        h100_curriculum,
        "resolve_curriculum",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        h100_curriculum,
        "check",
        lambda _plan: events.append("check"),
    )
    monkeypatch.setattr(
        h100_curriculum,
        "run_curriculum",
        lambda _plan, **_kwargs: events.append("run"),
    )

    assert (
        h100_curriculum.main(
            [
                "run",
                "--config",
                str(tmp_path / "curriculum.yaml"),
                *run_args,
            ]
        )
        == 2
    )
    assert events == []
    assert expected_error in capsys.readouterr().err


def test_trainer_authenticates_config_owned_pretrained_model_before_load(
    tmp_path: Path,
):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"selected-stage-model")
    digest = _sha256(model)
    cfg = OmegaConf.create(
        {
            "trainer": {
                "pretrained_checkpoint": str(model),
                "pretrained_checkpoint_sha256": digest,
            },
            "curriculum_handoff": {
                "previous_stage_handoff": {
                    "model_path": str(model.resolve()),
                    "model_sha256": digest,
                }
            },
        }
    )

    assert (
        train_starvla._authenticated_pretrained_checkpoint(cfg)
        == str(model.resolve())
    )

    model.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch before model load"):
        train_starvla._authenticated_pretrained_checkpoint(cfg)


def test_tiny_a_to_b_to_c_handoff_loads_weights_with_fresh_optimizers(
    tmp_path: Path,
):
    def fresh_training_state(
        model: torch.nn.Module,
    ) -> tuple[
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LambdaLR,
        dict,
    ]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _step: 1.0
        )
        return optimizer, scheduler, copy.deepcopy(scheduler.state_dict())

    def one_real_optimizer_step(
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        fresh_scheduler_state: dict,
    ) -> None:
        before = [
            parameter.detach().clone() for parameter in model.parameters()
        ]
        inputs = torch.tensor([[0.25, -0.75]], dtype=torch.float32)
        target = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        assert optimizer.state
        assert scheduler.state_dict() != fresh_scheduler_state
        assert any(
            not torch.equal(previous, current)
            for previous, current in zip(
                before, model.parameters(), strict=True
            )
        )

    def handoff(
        *,
        source: torch.nn.Module,
        stage_name: str,
    ) -> tuple[
        torch.nn.Module,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LambdaLR,
        dict,
        Path,
    ]:
        checkpoint = tmp_path / stage_name / "model.safetensors"
        checkpoint.parent.mkdir(parents=True)
        save_safetensors(source.state_dict(), str(checkpoint))
        digest = _sha256(checkpoint)
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "pretrained_checkpoint": str(checkpoint),
                    "pretrained_checkpoint_sha256": digest,
                },
                "curriculum_handoff": {
                    "previous_stage_handoff": {
                        "model_path": str(checkpoint.resolve()),
                        "model_sha256": digest,
                    }
                },
            }
        )
        destination = torch.nn.Linear(2, 2)
        authenticated = (
            train_starvla._authenticated_pretrained_checkpoint(cfg)
        )
        TrainerUtils.load_pretrained_backbones(
            destination, authenticated
        )
        optimizer, scheduler, fresh_scheduler_state = fresh_training_state(
            destination
        )
        assert optimizer.state == {}
        assert scheduler.state_dict() == fresh_scheduler_state
        for source_value, destination_value in zip(
            source.parameters(), destination.parameters(), strict=True
        ):
            torch.testing.assert_close(
                source_value, destination_value, rtol=0, atol=0
            )
        return (
            destination,
            optimizer,
            scheduler,
            fresh_scheduler_state,
            checkpoint,
        )

    stage_a = torch.nn.Linear(2, 2)
    with torch.no_grad():
        stage_a.weight.fill_(1.25)
        stage_a.bias.fill_(-0.5)
    optimizer_a, scheduler_a, fresh_a = fresh_training_state(stage_a)
    one_real_optimizer_step(
        model=stage_a,
        optimizer=optimizer_a,
        scheduler=scheduler_a,
        fresh_scheduler_state=fresh_a,
    )
    (
        stage_b,
        optimizer_b,
        scheduler_b,
        fresh_b,
        _,
    ) = handoff(source=stage_a, stage_name="a_natural_final")
    one_real_optimizer_step(
        model=stage_b,
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        fresh_scheduler_state=fresh_b,
    )
    (
        stage_c,
        optimizer_c,
        scheduler_c,
        fresh_c,
        _,
    ) = handoff(source=stage_b, stage_name="b_natural_final")

    assert optimizer_b is not optimizer_c
    assert optimizer_b.state
    assert optimizer_c.state == {}
    assert scheduler_b.state_dict() != fresh_b
    assert scheduler_c.state_dict() == fresh_c
    for stage_b_value, stage_c_value in zip(
        stage_b.parameters(), stage_c.parameters(), strict=True
    ):
        torch.testing.assert_close(
            stage_b_value, stage_c_value, rtol=0, atol=0
        )
    one_real_optimizer_step(
        model=stage_c,
        optimizer=optimizer_c,
        scheduler=scheduler_c,
        fresh_scheduler_state=fresh_c,
    )
    stage_c_final = tmp_path / "c_natural_final" / "model.safetensors"
    stage_c_final.parent.mkdir(parents=True)
    save_safetensors(stage_c.state_dict(), str(stage_c_final))
    assert optimizer_c.state
    assert scheduler_c.state_dict() != fresh_c
    assert _sha256(stage_c_final)


def test_materialized_config_identity_is_bound_to_immutable_run_config(
    tmp_path: Path,
):
    materialized = tmp_path / "materialized.yaml"
    run_config = tmp_path / "config.yaml"
    payload = {
        "run_id": "fixture",
        "trainer": {
            "is_resume": False,
            "pretrained_checkpoint": None,
            "pretrained_checkpoint_sha256": None,
        },
    }
    materialized.write_text(
        OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True),
        encoding="utf-8",
    )
    run_payload = dict(payload)
    run_payload["output_dir"] = str(tmp_path / "run")
    run_payload["trainer"] = dict(payload["trainer"])
    run_payload["trainer"]["_accelerate_num_processes"] = 8
    run_config.write_text(
        OmegaConf.to_yaml(OmegaConf.create(run_payload), resolve=True),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    h100_curriculum._validate_materialized_run_config(
        run_config=run_config,
        materialized_config=materialized,
        expected_materialized_config_sha256=_sha256(materialized),
    )

    run_payload["trainer"]["pretrained_checkpoint_sha256"] = "f" * 64
    run_config.write_text(
        OmegaConf.to_yaml(OmegaConf.create(run_payload), resolve=True),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="does not match the expected materialized",
    ):
        h100_curriculum._validate_materialized_run_config(
            run_config=run_config,
            materialized_config=materialized,
            expected_materialized_config_sha256=_sha256(materialized),
        )


def test_completed_stage_requires_immutable_yaml_json_config_pair(
    tmp_path: Path,
):
    materialized = tmp_path / "materialized.yaml"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_config = run_dir / "config.yaml"
    payload = {
        "run_id": "fixture",
        "trainer": {
            "is_resume": False,
            "pretrained_checkpoint": None,
            "pretrained_checkpoint_sha256": None,
        },
    }
    text = OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True)
    materialized.write_text(text, encoding="utf-8")
    run_config.write_text(text, encoding="utf-8")

    with pytest.raises(
        h100_curriculum.CurriculumError,
        match=r"config YAML\+JSON pair",
    ):
        h100_curriculum._validate_materialized_run_config(
            run_config=run_config,
            materialized_config=materialized,
            expected_materialized_config_sha256=_sha256(materialized),
        )


def test_natural_final_handoff_never_rewinds_to_offline_best(
    tmp_path: Path,
    monkeypatch,
):
    _set_test_image_identity(monkeypatch)
    run_dir = tmp_path / "stage_a_run"
    run_dir.mkdir()
    config_payload = {
        "run_id": "stage_a",
        "framework": {
            "action_model": {
                "state_dim": 18,
                "action_dim": 18,
                "action_horizon": 50,
            }
        },
        "trainer": {
            "is_resume": False,
            "resume_from_checkpoint": None,
            "pretrained_checkpoint": None,
            "pretrained_checkpoint_sha256": None,
            "reload_modules": None,
            "checkpoint_eval_milestone_steps": [1, 2],
        },
    }
    config_text = OmegaConf.to_yaml(
        OmegaConf.create(config_payload), resolve=True
    )
    run_config = run_dir / "config.yaml"
    materialized_config = tmp_path / "stage_a_materialized.yaml"
    run_config.write_text(config_text, encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    materialized_config.write_text(config_text, encoding="utf-8")
    config_sha = _sha256(run_config)

    def write_checkpoint(step: int, metric: float) -> tuple[Path, str]:
        checkpoint = run_dir / "checkpoints" / f"steps_{step}"
        checkpoint.mkdir(parents=True)
        model = checkpoint / "model.safetensors"
        model.write_bytes(f"natural-step-{step}".encode())
        trainer_state = checkpoint / "trainer_state.json"
        trainer_state.write_text(
            json.dumps(
                {
                    "completed_steps": step,
                    "selection_state_schema_version": 1,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        evaluation = {
            "schema_version": 1,
            "production_valid": True,
            "checkpoint_selection_eligible": True,
            "checkpoint_step": step,
            "checkpoint_relative_path": f"checkpoints/steps_{step}",
            "selection_metric": {
                "name": "heldout_loss",
                "mode": "min",
                "value": metric,
            },
            "checkpoint": {
                "model_file": "model.safetensors",
                "model_file_sha256": _sha256(model),
                "trainer_state_sha256": _sha256(trainer_state),
            },
            "run": {"config_sha256": config_sha},
        }
        eval_path = (
            run_dir
            / "heldout_eval_metrics"
            / f"step_{step:08d}.json"
        )
        eval_path.parent.mkdir(exist_ok=True)
        eval_path.write_text(
            json.dumps(evaluation, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return model, _sha256(model)

    best_model, best_sha = write_checkpoint(1, 0.1)
    final_model, final_sha = write_checkpoint(2, 0.2)
    pointer = {
        "schema_version": 1,
        "best_metric_step": 1,
        "checkpoint_relative_path": "checkpoints/steps_1",
        "best_metric_name": "heldout_loss",
        "best_metric_mode": "min",
        "best_metric_value": 0.1,
    }
    pointer_path = run_dir / "best_checkpoint.json"
    pointer_path.write_text(
        json.dumps(pointer, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (best_model.parent / "selection_state.json").write_text(
        json.dumps(pointer, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_export = run_dir / "final_model" / "pytorch_model.pt"
    final_export.parent.mkdir()
    final_export.write_bytes(b"final-export")

    handoff = h100_curriculum._validate_natural_final_handoff(
        run_dir,
        final_step=2,
        materialized_config=materialized_config,
        expected_materialized_config_sha256=_sha256(
            materialized_config
        ),
    )
    assert handoff["checkpoint_step"] == 2
    assert handoff["model_path"] == str(final_model)
    assert handoff["model_sha256"] == final_sha
    assert handoff["model_sha256"] != best_sha
    assert (
        handoff["selection_diagnostics"]["best_checkpoint_step"] == 1
    )

    source_config = tmp_path / "stage_b_source.yaml"
    _source_stage_config(source_config)
    output_config = tmp_path / "stage_b_materialized.yaml"
    stage = {
        "id": "stage_b",
        "role": "adapt",
        "initialization": "previous_stage_final",
        "optimizer_scheduler_rng_reset": True,
        "resume_policy": "newest_complete_full_state_same_stage",
        "world_model_predictor_attention_backend": "torch_sdpa",
        "handoff_checkpoint_policy": "natural_final",
        "config_path": str(source_config),
        "config_sha256": _sha256(source_config),
        "model_architecture_sha256": handoff[
            "model_architecture_sha256"
        ],
        "frozen_train_view_manifest": "/views/stage_b.json",
        "frozen_train_view_manifest_sha256": "a" * 64,
        "local_evaluation_manifest": "/eval/stage_b.json",
        "local_evaluation_manifest_sha256": "b" * 64,
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
        "curriculum_sha256": "c" * 64,
        "shared_contract": {
            "normalization_statistics_artifact": "/stats.json",
            "normalization_statistics_artifact_sha256": "d" * 64,
            "statistics_holdout_manifest": "/holdout.json",
            "statistics_holdout_manifest_sha256": "e" * 64,
        },
    }
    h100_curriculum._materialize_stage_config(
        curriculum_plan=curriculum_plan,
        stage=stage,
        run_id="stage_b_run",
        output_path=output_config,
        previous_handoff=handoff,
    )
    downstream = OmegaConf.load(output_config)
    assert downstream.trainer.pretrained_checkpoint == str(final_model)
    assert downstream.trainer.pretrained_checkpoint_sha256 == final_sha
    assert downstream.trainer.reload_modules is None
    assert (
        downstream.curriculum_handoff.previous_stage_handoff[
            "checkpoint_step"
        ]
        == 2
    )


def _source_stage_config(path: Path) -> None:
    path.write_text(
        """
run_id: source
curriculum_stage:
  initialization: previous_stage_final
  optimizer_scheduler_rng_reset: true
  resume_policy: newest_complete_full_state_same_stage
framework:
  vj2_model:
    predictor_attention_backend: torch_sdpa
trainer:
  is_resume: false
  resume_from_checkpoint: null
  pretrained_checkpoint: null
  pretrained_checkpoint_sha256: null
  reload_modules: null
  checkpoint_eval_milestone_steps: null
  optimizer:
    name: AdamW
  lr_scheduler_type: cosine
""".lstrip(),
        encoding="utf-8",
    )


def _fake_handoff(run_dir: Path, state_stage: dict) -> dict:
    model = run_dir / "selected" / "model.safetensors"
    model.parent.mkdir(parents=True, exist_ok=True)
    if not model.exists():
        model.write_bytes(f"model:{run_dir.name}".encode())
    return {
        "schema_version": 2,
        "handoff_checkpoint_policy": "natural_final",
        "run_dir": str(run_dir),
        "run_config": str(run_dir / "config.yaml"),
        "run_config_sha256": "1" * 64,
        "model_architecture_sha256": "a" * 64,
        "materialized_config_path": state_stage["resolved_config_path"],
        "materialized_config_sha256": state_stage[
            "resolved_config_sha256"
        ],
        "selection_pointer": str(run_dir / "best_checkpoint.json"),
        "selection_pointer_sha256": "2" * 64,
        "checkpoint_step": 1,
        "checkpoint_relative_path": "checkpoints/steps_1",
        "checkpoint_path": str(run_dir / "checkpoints" / "steps_1"),
        "model_path": str(model),
        "model_sha256": _sha256(model),
        "trainer_state_sha256": "3" * 64,
        "heldout_eval_path": str(run_dir / "eval.json"),
        "heldout_eval_sha256": "4" * 64,
        "best_metric_name": "loss",
        "best_metric_mode": "min",
        "best_metric_value": 0.1,
        "handoff_metric_name": "loss",
        "handoff_metric_mode": "min",
        "handoff_metric_value": 0.2,
        "selection_diagnostics": {
            "best_checkpoint_step": 0,
        },
        "final_model_path": str(model),
        "final_model_sha256": _sha256(model),
    }


def test_three_stage_handoff_and_interrupted_resume_are_authenticated(
    tmp_path: Path,
    monkeypatch,
):
    _set_test_image_identity(monkeypatch)
    configs = []
    stages = []
    run_root = tmp_path / "runs"
    for index, stage_id in enumerate(("a", "b", "c")):
        config = tmp_path / f"{stage_id}.yaml"
        _source_stage_config(config)
        configs.append(config)
        stages.append(
            {
                "id": stage_id,
                "role": ("pretrain", "adapt", "finetune")[index],
                    "initialization": (
                        "upstream" if index == 0 else "previous_stage_final"
                    ),
                    "optimizer_scheduler_rng_reset": True,
                    "resume_policy": (
                        "newest_complete_full_state_same_stage"
                    ),
                    "world_model_predictor_attention_backend": "torch_sdpa",
                    "handoff_checkpoint_policy": "natural_final",
                "config_path": str(config),
                "config_sha256": _sha256(config),
                "model_architecture_sha256": "a" * 64,
                "checkpoint_eval_milestone_steps": [1],
                "planned_optimizer_steps": 1,
                "plan": {
                    "training": {
                        "run_id_prefix": f"stage-{stage_id}",
                        "run_root_dir": str(run_root),
                    },
                    "runtime": {
                        "container_image": TEST_IMAGE,
                        "num_processes": 8,
                        "use_deepspeed": False,
                        "torch_compile_environment": "disabled",
                        "network_interface": "fixture0",
                    },
                },
                "frozen_train_view_manifest": f"/view/{stage_id}.json",
                "frozen_train_view_manifest_sha256": "5" * 64,
                "local_evaluation_manifest": f"/eval/{stage_id}.json",
                "local_evaluation_manifest_sha256": "6" * 64,
                "eligible_window_count": 8,
                "steps_per_epoch": 1,
                "expected_full_dataset_epochs": 1,
                "first_epoch_exposure_fractions": [1.0],
                "first_epoch_checkpoint_steps": [1],
                "full_epoch_boundary_steps": [1],
            }
        )
    plan = {
        "curriculum_id": "fixture_curriculum",
        "curriculum_path": str(tmp_path / "curriculum.yaml"),
        "curriculum_sha256": "7" * 64,
        "state_root_dir": str(tmp_path / "state"),
        "shared_contract": {
            "normalization_statistics_artifact": "/stats.json",
            "normalization_statistics_artifact_sha256": "8" * 64,
            "statistics_holdout_manifest": "/holdout.json",
            "statistics_holdout_manifest_sha256": "9" * 64,
        },
        "stages": stages,
    }
    run_id = "fixture_curriculum_resume_test"
    calls: list[Path] = []

    monkeypatch.setattr(
        h100_curriculum.h100_training,
        "_accelerate_command",
        lambda _plan, config: ["fixture-train", str(config)],
    )
    monkeypatch.setattr(
        h100_curriculum.h100_training,
        "check_curriculum_materialized_plan",
        lambda _config, **_kwargs: {},
    )
    monkeypatch.setattr(
        h100_curriculum,
        "_training_environment",
        lambda _plan: {},
    )
    monkeypatch.setattr(
        h100_curriculum,
        "_validate_completed_stage",
        lambda *, stage, state_stage: _fake_handoff(
            Path(state_stage["run_dir"]), state_stage
        ),
    )

    class FakeProcess:
        next_pid = 800_000

        def __init__(self, command, **_kwargs):
            self.command = command
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            calls.append(Path(command[-1]))

        def poll(self):
            return None

        def send_signal(self, _signum):
            return None

        def wait(self):
            cfg = OmegaConf.load(self.command[-1])
            run_dir = run_root / str(cfg.run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            immutable = run_dir / "config.yaml"
            if not immutable.exists():
                frozen = OmegaConf.create(
                    OmegaConf.to_container(cfg, resolve=True)
                )
                frozen.output_dir = str(run_dir)
                frozen_payload = OmegaConf.to_container(
                    frozen, resolve=True
                )
                immutable.write_text(
                    OmegaConf.to_yaml(frozen, resolve=True),
                    encoding="utf-8",
                )
                (run_dir / "config.json").write_text(
                    json.dumps(
                        frozen_payload,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            # A succeeds; B's first launch fails; resumed B and then C succeed.
            return 1 if len(calls) == 2 else 0

    monkeypatch.setattr(h100_curriculum.subprocess, "Popen", FakeProcess)

    with pytest.raises(h100_curriculum.CurriculumError, match="stage b exited"):
        h100_curriculum.run_curriculum(
            plan, run_id=run_id, resume=False
        )

    state_path = (
        Path(plan["state_root_dir"])
        / run_id
        / "curriculum_state.json"
    )
    interrupted = h100_curriculum._load_curriculum_state(state_path)
    checkpoint = (
        Path(interrupted["stages"][1]["run_dir"])
        / "checkpoints"
        / "steps_1"
    )
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(
        h100_curriculum,
        "_latest_complete_resume_checkpoint",
        lambda **_kwargs: checkpoint,
    )

    completed = h100_curriculum.run_curriculum(
        plan, run_id=run_id, resume=True
    )
    state = h100_curriculum._load_curriculum_state(completed)
    assert state["status"] == "complete"
    assert [entry["launch_attempt"] for entry in state["stages"]] == [
        1,
        2,
        1,
    ]

    materialized = [
        OmegaConf.load(entry["resolved_config_path"])
        for entry in state["stages"]
    ]
    assert materialized[0].trainer.pretrained_checkpoint is None
    for index in (1, 2):
        previous = state["stages"][index - 1]
        previous_handoff = json.loads(
            Path(previous["handoff_path"]).read_text(encoding="utf-8")
        )
        assert (
            materialized[index].trainer.pretrained_checkpoint
            == previous_handoff["model_path"]
        )
        assert (
            materialized[index].trainer.pretrained_checkpoint_sha256
            == previous["selected_model_sha256"]
        )
        assert materialized[index].trainer.is_resume is False
        assert (
            materialized[index].curriculum_handoff[
                "optimizer_scheduler_rng_reset"
            ]
            is True
        )
    resume_cfg = OmegaConf.load(state["stages"][1]["resume_config_path"])
    assert resume_cfg.trainer.is_resume is True
    assert resume_cfg.trainer.resume_from_checkpoint == str(checkpoint)
