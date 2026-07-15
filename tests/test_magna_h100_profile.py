from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
import yaml

from deployment.realman import build_magna_internal_holdout as holdout


REPO_ROOT = Path(__file__).resolve().parents[1]
A100_CONFIG = REPO_ROOT / (
    "scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_"
    "a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml"
)
H100_CONFIG = REPO_ROOT / (
    "scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_"
    "h100x8_b16_qwen35_2b_full_moge_vitb_vjepa_large.yaml"
)
H100_LAUNCHER = REPO_ROOT / (
    "scripts/vlajepa_robot_ft_lerobot_magna_interventions_"
    "h100x8_b16_qwen35_2b_full_moge_vitb_vjepa_large.sh"
)
DOCKER_RUN = REPO_ROOT / "scripts/docker_run_training.sh"
A100_LAUNCHER_NAME = (
    "vlajepa_robot_ft_lerobot_magna_interventions_"
    "a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh"
)
DATA_ROOT = "/mnt/vla-jepa/datasets/magna_training_data_with_interventions"
TEST_RUN_ID = "robot_ft_lerobot_magna_interventions_h100x8_b16_test"
TEST_RUN_ROOT_NAME = "h100-run-root"


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _last_value(args: list[str], option: str) -> str:
    index = max(i for i, value in enumerate(args) if value == option)
    return args[index + 1]


def test_h100_profile_is_only_sample_scaled_strict_attention_variant():
    a100 = _load(A100_CONFIG)
    h100 = _load(H100_CONFIG)

    a100_data = a100["datasets"]["vla_data"]
    h100_data = h100["datasets"]["vla_data"]
    a100_trainer = a100["trainer"]
    h100_trainer = h100["trainer"]
    h100_qwen = h100["framework"]["qwenvl"]

    assert a100_data["per_device_batch_size"] == 12
    assert 12 * 8 * a100_trainer["gradient_accumulation_steps"] == 96
    assert "global_batch96" in a100_data["episode_split_manifest"]
    assert h100_data["per_device_batch_size"] == 16
    assert 16 * 8 * h100_trainer["gradient_accumulation_steps"] == 128
    assert "global_batch128" in h100_data["episode_split_manifest"]

    assert h100_qwen["attn_implementation"] == "flash_attention_2"
    assert h100_qwen["strict_attn_implementation"] is True
    assert h100_qwen["enable_fast_linear_attention"] is True
    assert h100_qwen["strict_fast_linear_attention"] is True
    assert h100_data["qwen_observation_frame_index"] == "current"
    assert h100_data["lerobot_statistics_source"] == "split_train"
    assert h100_data["load_all_data_for_training"] is False
    assert h100["framework"]["action_model"]["rtc_training"]["enabled"] is False
    assert h100["framework"]["action_model"]["rtc_training"]["rtc_prob"] == 0.0
    assert h100["framework"]["action_model"]["past_action_window_size"] == 0
    assert h100_trainer["use_rabc"] is False
    for key in (
        "compile_qwen_model",
        "compile_action_model",
        "compile_vj_predictor",
        "compile_vj_encoder",
        "compile_full_model",
    ):
        assert h100_trainer[key] is False
    assert h100["framework"]["depth_teacher_aux"]["detach_vlm_fraction"] == 0.01

    assert (a100_trainer["num_warmup_steps"], h100_trainer["num_warmup_steps"]) == (3000, 2250)
    assert (a100_trainer["loss_scale"]["wm_warmup_steps"], h100_trainer["loss_scale"]["wm_warmup_steps"]) == (2000, 1500)
    assert (a100_trainer["save_interval"], h100_trainer["save_interval"]) == (2500, 1875)
    assert (a100_trainer["eval_interval"], h100_trainer["eval_interval"]) == (2500, 1875)
    assert 3000 * 96 == 2250 * 128
    assert 2000 * 96 == 1500 * 128
    assert 2500 * 96 == 1875 * 128

    normalized = copy.deepcopy(h100)
    normalized["run_id"] = a100["run_id"]
    normalized_qwen = normalized["framework"]["qwenvl"]
    for key in (
        "strict_attn_implementation",
        "enable_fast_linear_attention",
        "strict_fast_linear_attention",
    ):
        normalized_qwen[key] = a100["framework"]["qwenvl"][key]
    normalized_data = normalized["datasets"]["vla_data"]
    normalized_data["per_device_batch_size"] = a100_data["per_device_batch_size"]
    normalized_data["episode_split_manifest"] = a100_data["episode_split_manifest"]
    normalized_trainer = normalized["trainer"]
    for key in ("num_warmup_steps", "save_interval", "eval_interval"):
        normalized_trainer[key] = a100_trainer[key]
    normalized_trainer["loss_scale"]["wm_warmup_steps"] = a100_trainer["loss_scale"]["wm_warmup_steps"]
    assert normalized == a100

    derived = holdout._derive_effective_global_batch(
        H100_CONFIG, H100_LAUNCHER, world_size=8
    )
    assert derived["effective_global_batch_size"] == 128


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "ACCELERATE_BIN",
        "ACCELERATE_CONFIG",
        "CONFIG_YAML",
        "CUDA_VISIBLE_DEVICES",
        "DATA_ROOT_DIR",
        "DATALOADER_NUM_WORKERS",
        "DATALOADER_PERSISTENT_WORKERS",
        "DATALOADER_PREFETCH_FACTOR",
        "DATALOADER_TIMEOUT_SECONDS",
        "DDP_BUCKET_CAP_MB",
        "DDP_GRADIENT_AS_BUCKET_VIEW",
        "DDP_STATIC_GRAPH",
        "EVAL_INTERVAL",
        "EPOCHS",
        "FIND_UNUSED_PARAMETERS",
        "LIBERO_DATA_ROOT",
        "LOGGING_FREQUENCY",
        "MAX_TRAIN_STEPS",
        "NUM_WARMUP_STEPS",
        "NUM_MACHINES",
        "NUM_PROCESSES",
        "PER_DEVICE_BATCH_SIZE",
        "REALMAN_DATA_ROOT",
        "RUN_ID",
        "SAVE_INTERVAL",
        "STARVLA_ALLOW_TORCH_COMPILE",
        "STARVLA_DEEPSPEED_STAGE",
        "STARVLA_DISABLE_TORCH_COMPILE",
        "STARVLA_H100_LIFECYCLE_TEST",
        "STARVLA_USE_DEEPSPEED",
        "TORCH_COMPILE_DISABLE",
        "TORCHDYNAMO_DISABLE",
        "VIDEO_BACKEND",
        "VIDEO_BACKEND_NUM_THREADS",
    ):
        env.pop(name, None)
    return env


def _fake_h100_repo(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    config_dir = scripts / "config"
    manifest_dir = repo / "deployment/realman/eval_manifests"
    fake_bin = tmp_path / "bin"
    config_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    fake_bin.mkdir()

    launcher = scripts / H100_LAUNCHER.name
    shutil.copy2(H100_LAUNCHER, launcher)
    launcher.write_text(
        launcher.read_text(encoding="utf-8").replace(
            'H100_RUN_ROOT="/mnt/vla-jepa/checkpoints"',
            f'H100_RUN_ROOT="{tmp_path / TEST_RUN_ROOT_NAME}"',
        ),
        encoding="utf-8",
    )
    (config_dir / H100_CONFIG.name).touch()
    manifest = manifest_dir / "magna_internal_holdout_global_batch128_v1.json"
    manifest.touch()
    generator_args = tmp_path / "generator_args.txt"

    python = fake_bin / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == -c ]]; then\n"
        f"  if [[ \"${{2:-}}\" == *completed_steps* ]]; then exec {shlex.quote(sys.executable)} \"$@\"; fi\n"
        "  printf '%s\\n' \"${H100_TEST_MANIFEST}\"\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$@\" > \"${H100_TEST_GENERATOR_ARGS}\"\n"
        "printf 'MAGNA_HOLDOUT_BUILD_RESULT={\"manifest_path\":\"%s\",\"reused\":true}\\n' \"${H100_TEST_MANIFEST}\"\n",
        encoding="utf-8",
    )
    accelerate = fake_bin / "accelerate"
    accelerate.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    capture = scripts / A100_LAUNCHER_NAME
    capture.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CAPTURED'\n"
        "printf ' %q' \"$@\"\n"
        "printf '\\n'\n",
        encoding="utf-8",
    )
    for path in (python, accelerate, capture):
        path.chmod(0o755)

    env = _clean_env()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "H100_TEST_GENERATOR_ARGS": str(generator_args),
            "H100_TEST_MANIFEST": str(manifest),
            "RUN_ID": TEST_RUN_ID,
        }
    )
    return launcher, env, generator_args


def _run_fake_h100(tmp_path: Path, *args: str, extra_env=None):
    launcher, env, generator_args = _fake_h100_repo(tmp_path)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", str(launcher), *args],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    return result, generator_args


def _full_state_checkpoint(
    tmp_path: Path,
    name: str = "steps_5",
    *,
    run_id: str = TEST_RUN_ID,
) -> Path:
    step = int(name.removeprefix("steps_"))
    checkpoint = tmp_path / TEST_RUN_ROOT_NAME / run_id / "checkpoints" / name
    checkpoint.mkdir(parents=True)
    for filename in ("model.safetensors", "optimizer.bin", "scheduler.bin"):
        (checkpoint / filename).write_bytes(b"checkpoint-state")
    (checkpoint / "trainer_state.json").write_text(
        f'{{"completed_steps": {step}}}\n', encoding="utf-8"
    )
    for rank in range(8):
        (checkpoint / f"random_states_{rank}.pkl").write_bytes(b"rng-state")
    return checkpoint


def _write_selection_state(
    checkpoint: Path,
    *,
    best_step: int | None,
    best_value: float | None,
) -> None:
    metric_name = "heldout_focused_eval_task_failure_score_h10"
    trainer_state = {
        "completed_steps": int(checkpoint.name.removeprefix("steps_")),
        "selection_state_schema_version": 1,
        "best_metric_name": metric_name,
        "best_metric_mode": "min",
        "best_metric_value": best_value,
        "best_metric_step": best_step,
    }
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(trainer_state) + "\n", encoding="utf-8"
    )
    selection_state = {
        "schema_version": 1,
        "best_metric_name": metric_name,
        "best_metric_mode": "min",
        "best_metric_value": best_value,
        "best_metric_step": best_step,
        "checkpoint_relative_path": (
            None if best_step is None else f"checkpoints/steps_{best_step}"
        ),
    }
    (checkpoint / "selection_state.json").write_text(
        json.dumps(selection_state) + "\n", encoding="utf-8"
    )


def test_h100_production_launcher_preserves_resume_controls_and_has_no_step_cap(tmp_path):
    checkpoint = _full_state_checkpoint(tmp_path)
    result, generator_args_path = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 0, result.stderr
    generator_args = generator_args_path.read_text(encoding="utf-8").splitlines()
    assert generator_args[generator_args.index("--world-size") + 1] == "8"
    assert generator_args[generator_args.index("--config") + 1].endswith(H100_CONFIG.name)
    assert generator_args[generator_args.index("--launcher") + 1].endswith(H100_LAUNCHER.name)
    assert generator_args[generator_args.index("--dataset-root") + 1] == DATA_ROOT

    captured_line = next(line for line in result.stdout.splitlines() if line.startswith("CAPTURED"))
    command = shlex.split(captured_line)[1:]
    assert "--trainer.max_train_steps" not in command
    assert _last_value(command, "--trainer.is_resume") == "true"
    assert _last_value(command, "--trainer.resume_from_checkpoint") == str(checkpoint)
    assert _last_value(command, "--datasets.vla_data.per_device_batch_size") == "16"
    assert _last_value(command, "--datasets.vla_data.data_root_dir") == DATA_ROOT
    assert _last_value(command, "--framework.action_model.rtc_training.enabled") == "false"
    assert _last_value(command, "--framework.qwenvl.enable_fast_linear_attention") == "true"
    assert _last_value(command, "--trainer.save_interval") == "1875"
    assert _last_value(command, "--trainer.eval_interval") == "1875"
    assert _last_value(command, "--trainer.logging_frequency") == "10"
    assert _last_value(command, "--trainer.num_warmup_steps") == "2250"
    assert _last_value(command, "--trainer.loss_scale.wm_warmup_steps") == "1500"
    assert _last_value(command, "--trainer.eval_before_train") == "true"
    assert _last_value(command, "--trainer.allow_training_stream_eval") == "false"
    assert _last_value(command, "--datasets.vla_data.video_backend") == "pyav"
    assert _last_value(command, "--trainer.use_rabc") == "false"


def test_h100_launcher_accepts_root_resume_checkpoint_alias(tmp_path):
    checkpoint = _full_state_checkpoint(tmp_path)
    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume=true",
        "--resume_from_checkpoint",
        str(checkpoint),
    )
    assert result.returncode == 0, result.stderr
    captured_line = next(line for line in result.stdout.splitlines() if line.startswith("CAPTURED"))
    command = shlex.split(captured_line)[1:]
    assert _last_value(command, "--trainer.is_resume") == "true"
    assert "--resume_from_checkpoint" not in command
    assert _last_value(command, "--trainer.resume_from_checkpoint") == str(
        checkpoint.resolve()
    )


def test_h100_production_launcher_accepts_fresh_without_resume_args(tmp_path):
    result, _ = _run_fake_h100(tmp_path)
    assert result.returncode == 0, result.stderr
    captured_line = next(line for line in result.stdout.splitlines() if line.startswith("CAPTURED"))
    command = shlex.split(captured_line)[1:]
    assert "--trainer.is_resume" not in command
    assert "--resume_from_checkpoint" not in command
    assert "--trainer.resume_from_checkpoint" not in command
    assert "--trainer.max_train_steps" not in command


@pytest.mark.parametrize(
    "resume",
    [False, True],
)
def test_h100_lifecycle_gate_pins_exact_budget_and_cadence_without_rescaling_warmup(
    tmp_path, resume
):
    if resume:
        checkpoint = _full_state_checkpoint(tmp_path)
        phase_args = (
            "--trainer.is_resume",
            "true",
            "--trainer.resume_from_checkpoint",
            str(checkpoint),
        )
    else:
        phase_args = ("--trainer.is_resume", "false")
    result, _ = _run_fake_h100(
        tmp_path,
        *phase_args,
        extra_env={"STARVLA_H100_LIFECYCLE_TEST": "1"},
    )
    assert result.returncode == 0, result.stderr
    captured_line = next(line for line in result.stdout.splitlines() if line.startswith("CAPTURED"))
    command = shlex.split(captured_line)[1:]
    assert _last_value(command, "--trainer.max_train_steps") == "15"
    assert _last_value(command, "--trainer.save_interval") == "5"
    assert _last_value(command, "--trainer.eval_interval") == "5"
    assert _last_value(command, "--trainer.logging_frequency") == "1"
    assert _last_value(command, "--trainer.num_warmup_steps") == "2250"
    assert _last_value(command, "--trainer.loss_scale.wm_warmup_steps") == "1500"


@pytest.mark.parametrize(
    "hostile_env",
    [
        {"MAX_TRAIN_STEPS": "14"},
        {"MAX_TRAIN_STEPS": "16"},
        {"SAVE_INTERVAL": "4"},
        {"SAVE_INTERVAL": "6"},
        {"EVAL_INTERVAL": "4"},
        {"EVAL_INTERVAL": "6"},
        {"LOGGING_FREQUENCY": "2"},
    ],
)
def test_h100_lifecycle_gate_rejects_non_exact_lifecycle_environment(tmp_path, hostile_env):
    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "false",
        extra_env={"STARVLA_H100_LIFECYCLE_TEST": "1", **hostile_env},
    )
    assert result.returncode == 2
    assert result.stdout == ""


@pytest.mark.parametrize(
    "hostile_args",
    [
        ("--datasets.vla_data.CoT_prompt", "PRIVILEGED-STAGE"),
        ("--datasets.vla_data.video_frame_stride", "1"),
        ("--framework.qwenvl.base_vlm", "stale/model"),
        ("--trainer.epochs", "99"),
        ("--framework.action_model.rtc_training.enabled", "true"),
        ("--datasets.vla_data.per_device_batch_size", "1"),
        ("--trainer.compile_qwen_model", "true"),
        ("--trainer.max_train_steps", "15"),
        ("--trainer.resume_epoch", "1"),
        ("--trainer.resume_step", "5"),
        ("--trainer.epochs=99",),
    ],
)
def test_h100_launcher_rejects_protected_cli_overrides(tmp_path, hostile_args):
    result, _ = _run_fake_h100(tmp_path, *hostile_args)
    assert result.returncode == 2
    assert result.stdout == ""


@pytest.mark.parametrize(
    "case",
    [
        "resume_true_without_checkpoint",
        "checkpoint_without_resume_true",
        "resume_false_with_checkpoint",
        "duplicate_checkpoint_spellings",
        "duplicate_is_resume",
    ],
)
def test_h100_launcher_rejects_ambiguous_or_incomplete_resume_controls(tmp_path, case):
    checkpoint = _full_state_checkpoint(tmp_path)
    cases = {
        "resume_true_without_checkpoint": ("--trainer.is_resume", "true"),
        "checkpoint_without_resume_true": (
            "--trainer.resume_from_checkpoint",
            str(checkpoint),
        ),
        "resume_false_with_checkpoint": (
            "--trainer.is_resume",
            "false",
            "--trainer.resume_from_checkpoint",
            str(checkpoint),
        ),
        "duplicate_checkpoint_spellings": (
            "--trainer.is_resume",
            "true",
            "--resume_from_checkpoint",
            str(checkpoint),
            "--trainer.resume_from_checkpoint",
            str(checkpoint),
        ),
        "duplicate_is_resume": (
            "--trainer.is_resume",
            "true",
            "--trainer.is_resume",
            "true",
            "--trainer.resume_from_checkpoint",
            str(checkpoint),
        ),
    }
    result, _ = _run_fake_h100(tmp_path, *cases[case])
    assert result.returncode == 2
    assert result.stdout == ""


@pytest.mark.parametrize("missing_file", [None, "optimizer.bin", "random_states_7.pkl"])
def test_h100_launcher_rejects_missing_or_incomplete_resume_checkpoint(tmp_path, missing_file):
    checkpoint = (
        tmp_path
        / TEST_RUN_ROOT_NAME
        / TEST_RUN_ID
        / "checkpoints"
        / "steps_9"
    )
    if missing_file is not None:
        checkpoint = _full_state_checkpoint(tmp_path, name="steps_9")
        (checkpoint / missing_file).unlink()
    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )
    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_rejects_resume_checkpoint_from_another_run(tmp_path):
    checkpoint = _full_state_checkpoint(
        tmp_path,
        run_id="robot_ft_lerobot_magna_interventions_h100x8_b16_other",
    )
    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )
    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_rejects_trainer_state_step_mismatch(tmp_path):
    checkpoint = _full_state_checkpoint(tmp_path, name="steps_5")
    (checkpoint / "trainer_state.json").write_text(
        '{"completed_steps": 4}\n', encoding="utf-8"
    )
    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )
    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_requires_selection_state_for_new_checkpoint_schema(tmp_path):
    checkpoint = _full_state_checkpoint(tmp_path)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "completed_steps": 5,
                "selection_state_schema_version": 1,
                "best_metric_name": "heldout_focused_eval_task_failure_score_h10",
                "best_metric_mode": "min",
                "best_metric_value": None,
                "best_metric_step": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_accepts_complete_new_selection_state(tmp_path):
    checkpoint = _full_state_checkpoint(tmp_path)
    _write_selection_state(checkpoint, best_step=5, best_value=0.25)

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", 1.0),
        ("schema_version", True),
        ("best_metric_name", "wrong_metric"),
        ("best_metric_value", float("nan")),
        ("best_metric_step", 6),
        ("checkpoint_relative_path", "checkpoints/steps_4"),
    ],
)
def test_h100_launcher_rejects_malformed_new_selection_state(
    tmp_path, field, value
):
    checkpoint = _full_state_checkpoint(tmp_path)
    _write_selection_state(checkpoint, best_step=5, best_value=0.25)
    selection_path = checkpoint / "selection_state.json"
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    payload[field] = value
    selection_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_rejects_missing_selected_best_checkpoint(tmp_path):
    checkpoint = _full_state_checkpoint(tmp_path, name="steps_10")
    _write_selection_state(checkpoint, best_step=5, best_value=0.25)

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_rejects_incomplete_selected_best_checkpoint(tmp_path):
    best_checkpoint = _full_state_checkpoint(tmp_path, name="steps_5")
    checkpoint = _full_state_checkpoint(tmp_path, name="steps_10")
    _write_selection_state(checkpoint, best_step=5, best_value=0.25)
    (best_checkpoint / "optimizer.bin").unlink()

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_accepts_empty_new_selection_state(tmp_path):
    checkpoint = _full_state_checkpoint(tmp_path)
    _write_selection_state(checkpoint, best_step=None, best_value=None)

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 0, result.stderr


def test_h100_launcher_accepts_complete_older_best_dependency(tmp_path):
    best_checkpoint = _full_state_checkpoint(tmp_path, name="steps_5")
    _write_selection_state(best_checkpoint, best_step=5, best_value=0.25)
    checkpoint = _full_state_checkpoint(tmp_path, name="steps_10")
    _write_selection_state(checkpoint, best_step=5, best_value=0.25)

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 0, result.stderr


def test_h100_launcher_accepts_transitive_selection_dependency_closure(tmp_path):
    checkpoint_5 = _full_state_checkpoint(tmp_path, name="steps_5")
    _write_selection_state(checkpoint_5, best_step=5, best_value=0.30)
    checkpoint_10 = _full_state_checkpoint(tmp_path, name="steps_10")
    _write_selection_state(checkpoint_10, best_step=5, best_value=0.30)
    checkpoint_15 = _full_state_checkpoint(tmp_path, name="steps_15")
    _write_selection_state(checkpoint_15, best_step=10, best_value=0.20)

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint_15),
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("schema", [2, 1.0, True])
def test_h100_launcher_rejects_nonexact_trainer_selection_schema(tmp_path, schema):
    checkpoint = _full_state_checkpoint(tmp_path)
    _write_selection_state(checkpoint, best_step=5, best_value=0.25)
    trainer_state_path = checkpoint / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    trainer_state["selection_state_schema_version"] = schema
    trainer_state_path.write_text(
        json.dumps(trainer_state) + "\n", encoding="utf-8"
    )

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_rejects_coordinated_wrong_metric_identity(tmp_path):
    checkpoint = _full_state_checkpoint(tmp_path)
    _write_selection_state(checkpoint, best_step=5, best_value=0.25)
    for filename in ("trainer_state.json", "selection_state.json"):
        path = checkpoint / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["best_metric_name"] = "coordinated_but_wrong_metric"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_rejects_malformed_legacy_sidecar(tmp_path):
    checkpoint = _full_state_checkpoint(tmp_path)
    (checkpoint / "selection_state.json").write_text("{", encoding="utf-8")

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_rejects_malformed_selected_best_sidecar(tmp_path):
    best_checkpoint = _full_state_checkpoint(tmp_path, name="steps_5")
    _write_selection_state(best_checkpoint, best_step=5, best_value=0.25)
    best_payload_path = best_checkpoint / "selection_state.json"
    best_payload = json.loads(best_payload_path.read_text(encoding="utf-8"))
    best_payload["schema_version"] = 2
    best_payload_path.write_text(
        json.dumps(best_payload) + "\n", encoding="utf-8"
    )
    checkpoint = _full_state_checkpoint(tmp_path, name="steps_10")
    _write_selection_state(checkpoint, best_step=5, best_value=0.25)

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_rejects_external_symlinked_best_checkpoint(tmp_path):
    external_best = _full_state_checkpoint(
        tmp_path,
        name="steps_5",
        run_id="robot_ft_lerobot_magna_interventions_h100x8_b16_external",
    )
    _write_selection_state(external_best, best_step=5, best_value=0.25)
    checkpoint = _full_state_checkpoint(tmp_path, name="steps_10")
    _write_selection_state(checkpoint, best_step=5, best_value=0.25)
    (checkpoint.parent / "steps_5").symlink_to(
        external_best, target_is_directory=True
    )

    result, _ = _run_fake_h100(
        tmp_path,
        "--trainer.is_resume",
        "true",
        "--trainer.resume_from_checkpoint",
        str(checkpoint),
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_rejects_fresh_start_into_existing_run_directory(tmp_path):
    _full_state_checkpoint(tmp_path)
    result, _ = _run_fake_h100(tmp_path)
    assert result.returncode == 2
    assert result.stdout == ""


@pytest.mark.parametrize(
    "hostile_env",
    [
        {"CONFIG_YAML": "/stale.yaml"},
        {"RUN_ID": "robot_ft_lerobot_magna_interventions_h100x8_b16_bad/path"},
        {"DATA_ROOT_DIR": "/stale"},
        {"LIBERO_DATA_ROOT": "/stale"},
        {"REALMAN_DATA_ROOT": "/stale"},
        {"NUM_PROCESSES": "7"},
        {"NUM_MACHINES": "2"},
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"PER_DEVICE_BATCH_SIZE": "12"},
        {"STARVLA_USE_DEEPSPEED": "1"},
        {"ACCELERATE_CONFIG": "/stale.yaml"},
        {"STARVLA_DEEPSPEED_STAGE": "2"},
        {"STARVLA_ALLOW_TORCH_COMPILE": "1"},
        {"STARVLA_DISABLE_TORCH_COMPILE": "0"},
        {"TORCH_COMPILE_DISABLE": "0"},
        {"TORCHDYNAMO_DISABLE": "0"},
        {"ACCELERATE_BIN": "/bin/true"},
        {"VIDEO_BACKEND": "decord"},
        {"EPOCHS": "99"},
        {"NUM_WARMUP_STEPS": "1"},
        {"LOGGING_FREQUENCY": "1"},
        {"FIND_UNUSED_PARAMETERS": "true"},
        {"DDP_GRADIENT_AS_BUCKET_VIEW": "false"},
        {"DDP_STATIC_GRAPH": "true"},
        {"DDP_BUCKET_CAP_MB": "25"},
        {"DATALOADER_NUM_WORKERS": "8"},
        {"DATALOADER_PREFETCH_FACTOR": "4"},
        {"DATALOADER_TIMEOUT_SECONDS": "30"},
        {"DATALOADER_PERSISTENT_WORKERS": "false"},
        {"VIDEO_BACKEND_NUM_THREADS": "2"},
        {"SAVE_INTERVAL": "5"},
        {"EVAL_INTERVAL": "5"},
        {"MAX_TRAIN_STEPS": "2550"},
    ],
)
def test_h100_production_launcher_rejects_conflicting_environment(hostile_env):
    env = _clean_env()
    env.update(hostile_env)
    result = subprocess.run(
        ["bash", str(H100_LAUNCHER)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert result.stdout == ""


def test_h100_launcher_is_executable_and_shell_valid():
    assert os.access(H100_LAUNCHER, os.X_OK)
    subprocess.run(["bash", "-n", str(H100_LAUNCHER)], check=True)
    assert "\n  STARVLA_H100_LIFECYCLE_TEST\n" in DOCKER_RUN.read_text(encoding="utf-8")


def test_docker_runner_forwards_h100_lifecycle_gate(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'DOCKER'\n"
        "printf ' %q' \"$@\"\n"
        "printf '\\n'\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    env = _clean_env()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "DOCKER_GPU_MODE": "none",
            "DOCKER_TTY": "0",
            "IMAGE": "h100-profile-test",
            "STARVLA_H100_LIFECYCLE_TEST": "1",
            "VLA_JEPA_SCRATCH": str(tmp_path / "scratch"),
        }
    )
    result = subprocess.run(
        ["bash", str(DOCKER_RUN), "bash", H100_LAUNCHER.name],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)[1:]
    docker_env = [command[index + 1] for index, arg in enumerate(command[:-1]) if arg == "-e"]
    assert "STARVLA_H100_LIFECYCLE_TEST=1" in docker_env
    assert command[-2:] == ["bash", H100_LAUNCHER.name]
