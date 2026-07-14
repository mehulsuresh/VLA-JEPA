import fcntl
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml"
)
LAUNCHER_PATH = (
    REPO_ROOT
    / "scripts/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh"
)
CLEAN_PILOT_PATH = (
    REPO_ROOT
    / "scripts/vlajepa_robot_ft_lerobot_magna_clean_rtc0_pilot_a100x8.sh"
)
DOCKER_RUN_PATH = REPO_ROOT / "scripts/docker_run_training.sh"
CLOUD_SERVICE_PATH = REPO_ROOT / "scripts/run_cloud_training_service.sh"
IMAGE_BUILD_PATH = REPO_ROOT / "scripts/build_magna_a100_image.sh"
TRAINING_ENV_PATH = REPO_ROOT / "scripts/lib/training_env.sh"


def test_magna_production_config_contract():
    cfg = OmegaConf.load(CONFIG_PATH)

    assert cfg.framework.qwenvl.base_vlm == "Qwen/Qwen3.5-2B"
    assert cfg.framework.qwenvl.lora.enabled is False
    assert cfg.framework.qwenvl.blockwise_attention.enabled is False
    assert cfg.framework.qwenvl.strict_full_trainable is True

    action_cfg = cfg.framework.action_model
    assert action_cfg.action_dim == 18
    assert action_cfg.state_dim == 19
    assert action_cfg.action_horizon == 50
    assert action_cfg.future_action_window_size == 49
    assert action_cfg.rtc_training.enabled is False
    assert action_cfg.rtc_training.max_delay == 0
    assert action_cfg.rtc_training.rtc_prob == 0.0

    data_cfg = cfg.datasets.vla_data
    assert data_cfg.data_mix == "magna_source_no_base_no_lift_interventions_v3"
    assert data_cfg.action_type == "absolute_qpos"
    assert data_cfg.modality_metadata_overrides.state.source.original_key == (
        "source.observation.state"
    )
    assert (data_cfg.modality_metadata_overrides.state.source.start, data_cfg.modality_metadata_overrides.state.source.end) == (0, 19)
    assert (data_cfg.modality_metadata_overrides.action.source_controls.start, data_cfg.modality_metadata_overrides.action.source_controls.end) == (0, 16)
    assert (data_cfg.modality_metadata_overrides.action.source_head.start, data_cfg.modality_metadata_overrides.action.source_head.end) == (19, 21)
    assert data_cfg.task_id_prompt_source_column == "subtask_index"
    assert data_cfg.append_task_id_to_prompt is False
    assert data_cfg.task_id_prompt_append_probability == 0.0
    assert data_cfg.lerobot_statistics_source == "split_train"
    assert data_cfg.load_all_data_for_training is False
    assert data_cfg.eval_num_workers == 0
    assert data_cfg.get("episode_split_role", None) is None
    split_manifest_path = REPO_ROOT / data_cfg.episode_split_manifest
    assert split_manifest_path.is_file()
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    assert split_manifest["evaluation_sampling"]["algorithm"] == (
        "nonzero_valid_unpadded_uniform_v1"
    )
    assert split_manifest["evaluation_sampling"]["frames_per_episode"] == 1
    assert sum(
        entry["holdout_episode_count"] for entry in split_manifest["datasets"]
    ) == data_cfg.per_device_batch_size * 8 * cfg.trainer.gradient_accumulation_steps
    assert data_cfg.require_statistics_frame_count is True
    assert "__unlabeled__" in data_cfg.subtask_prompt_ignored_labels
    assert data_cfg.use_action_validity_prefix_mask is True
    assert data_cfg.action_validity_label_key == "valid_state"
    assert data_cfg.action_validity_invalid_run_length == 10
    assert data_cfg.video_backend == "pyav"
    assert data_cfg.video_backend_num_threads == 1
    assert data_cfg.lerobot_v3_parquet_cache_size == 5
    assert data_cfg.per_device_batch_size == 12

    assert cfg.framework.vj2_model.num_video_views == 3
    assert cfg.framework.vj2_model.num_frames == 8
    assert cfg.framework.depth_teacher_aux.enabled is True
    assert cfg.trainer.use_rabc is False
    assert cfg.trainer.repeated_diffusion_steps == 8
    assert cfg.trainer.compile_qwen_model is False
    assert cfg.trainer.compile_action_model is False
    assert cfg.trainer.compile_vj_predictor is False
    assert cfg.trainer.compile_vj_encoder is False
    assert cfg.trainer.compile_full_model is False
    assert cfg.trainer.epochs == 3
    assert cfg.trainer.step_scheduler_with_optimizer is False
    assert cfg.trainer.allow_training_stream_eval is False
    assert cfg.trainer.eval_interval == cfg.trainer.save_interval == 2500
    assert cfg.trainer.best_metric_name == "heldout_eval_normalized_arm_mae_h20"


def test_magna_mixture_and_launcher_route_to_production_config():
    assert DATASET_NAMED_MIXTURES["magna_source_no_base_no_lift_interventions_v3"] == [
        ("", 1.0, "realman_bimanual_source_no_base_no_lift", "v3.0")
    ]

    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert CONFIG_PATH.name in launcher
    assert "vlajepa_robot_ft_libero_plus_a100x8_qwen3_full_moge_vitb_vjepa_large.sh" in launcher


def _clean_pilot_dry_run(tmp_path, *args, extra_env=None):
    env = os.environ.copy()
    for inherited_name in (
        "ACCELERATE_BIN",
        "STARVLA_ALLOW_TORCH_COMPILE",
        "STARVLA_DISABLE_TORCH_COMPILE",
        "TORCH_COMPILE_DISABLE",
        "TORCHDYNAMO_DISABLE",
    ):
        env.pop(inherited_name, None)
    env.update(
        {
            "VLA_JEPA_SCRATCH": str(tmp_path / "scratch"),
            "NUM_PROCESSES": "8",
            "NUM_MACHINES": "1",
            "MAIN_PROCESS_PORT": "29999",
            "STARVLA_USE_DEEPSPEED": "0",
            "STARVLA_CLEAN_PILOT_TEST_DRY_RUN": "1",
            "STARVLA_HOLDOUT_PREFLIGHT_DRY_RUN": "1",
            "RUN_ID": "magna_clean_v3_rtc0_pilot_a100x8_test",
            # These hostile inherited values must be pinned by the pilot.
            "CONFIG_YAML": "/tmp/stale-training-config.yaml",
            "MAX_TRAIN_STEPS": "67500",
            "EVAL_INTERVAL": "1",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(CLEAN_PILOT_PATH), *args],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )


def test_clean_pilot_pins_config_budget_and_clean_invariants(tmp_path):
    result = _clean_pilot_dry_run(
        tmp_path,
        "--datasets.vla_data.num_workers",
        "1",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[0] == "launch"
    assert command[command.index("--num_processes") + 1] == "8"
    assert command[command.index("--num_machines") + 1] == "1"
    assert "--datasets.vla_data.per_device_batch_size" not in command
    config_index = command.index("--config_yaml")
    assert command[config_index + 1] == str(CONFIG_PATH)

    expected_pairs = {
        "--trainer.max_train_steps": "2550",
        "--trainer.save_interval": "2500",
        "--trainer.checkpoint_max_to_keep": "1",
        "--trainer.eval_interval": "2500",
        "--trainer.pretrained_checkpoint": "null",
        "--trainer.resume_from_checkpoint": "null",
        "--trainer.is_resume": "false",
        "--trainer.eval_before_train": "true",
        "--trainer.allow_training_stream_eval": "false",
        "--trainer.save_final_model": "false",
        "--trainer.use_rabc": "false",
        "--framework.action_model.rtc_training.enabled": "false",
        "--framework.action_model.rtc_training.rtc_prob": "0.0",
        "--framework.action_model.past_action_window_size": "0",
        "--datasets.vla_data.dataset_py": "lerobot_datasets",
        "--datasets.vla_data.data_root_dir": (
            "/mnt/vla-jepa/datasets/magna_training_data_with_interventions"
        ),
        "--datasets.vla_data.lerobot_version": "v3.0",
        "--datasets.vla_data.episode_split_manifest": str(
            REPO_ROOT
            / "deployment/realman/eval_manifests/magna_internal_holdout_global_batch96_v1.json"
        ),
        "--datasets.vla_data.load_all_data_for_training": "false",
        "--datasets.vla_data.lerobot_statistics_source": "split_train",
        "--datasets.vla_data.require_statistics_frame_count": "true",
        "--datasets.vla_data.append_task_id_to_prompt": "false",
        "--datasets.vla_data.qwen_observation_frame_index": "current",
    }
    for option, value in expected_pairs.items():
        option_index = len(command) - 1 - command[::-1].index(option)
        assert command[option_index + 1] == value


def test_clean_pilot_parses_runtime_override_pairs_and_inline_values(tmp_path):
    result = _clean_pilot_dry_run(
        tmp_path,
        "--datasets.vla_data.num_workers",
        "2",
        "--datasets.vla_data.prefetch_factor=2",
        "--trainer.ddp_static_graph",
        "false",
        "--trainer.logging_frequency=5",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert "--datasets.vla_data.num_workers" in command
    assert "--datasets.vla_data.prefetch_factor=2" in command
    assert "--trainer.ddp_static_graph" in command
    assert "--trainer.logging_frequency=5" in command


@pytest.mark.parametrize(
    ("runtime_args", "expected_constraint"),
    [
        (("--trainer.logging_frequency", "0"), "a positive integer"),
        (("--trainer.logging_frequency=not-an-int",), "a positive integer"),
        (("--datasets.vla_data.num_workers", "-1"), "a non-negative integer"),
        (("--datasets.vla_data.prefetch_factor=0",), "a positive integer"),
        (("--trainer.ddp_bucket_cap_mb", "0"), "a positive integer"),
        (("--trainer.ddp_static_graph", "maybe"), "true or false"),
    ],
)
def test_clean_pilot_rejects_invalid_runtime_values(
    tmp_path,
    runtime_args,
    expected_constraint,
):
    result = _clean_pilot_dry_run(tmp_path, *runtime_args)

    assert result.returncode == 2
    assert "Invalid clean-pilot runtime override" in result.stderr
    assert expected_constraint in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("hostile_env", "expected_constraint"),
    [
        ({"LOGGING_FREQUENCY": "0"}, "a positive integer"),
        ({"DATALOADER_NUM_WORKERS": "-1"}, "a non-negative integer"),
        ({"DATALOADER_PREFETCH_FACTOR": "0"}, "a positive integer"),
        ({"DDP_STATIC_GRAPH": "maybe"}, "true or false"),
    ],
)
def test_clean_pilot_rejects_invalid_runtime_environment(
    tmp_path,
    hostile_env,
    expected_constraint,
):
    result = _clean_pilot_dry_run(tmp_path, extra_env=hostile_env)

    assert result.returncode == 2
    assert "Invalid clean-pilot runtime override" in result.stderr
    assert expected_constraint in result.stderr
    assert result.stdout == ""


def test_clean_pilot_refuses_inherited_accelerate_executable(tmp_path):
    result = _clean_pilot_dry_run(
        tmp_path,
        extra_env={"ACCELERATE_BIN": "/bin/true"},
    )

    assert result.returncode == 2
    assert "refuses inherited ACCELERATE_BIN=/bin/true" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("hostile_env", "expected_message"),
    [
        (
            {"STARVLA_ALLOW_TORCH_COMPILE": "1"},
            "STARVLA_ALLOW_TORCH_COMPILE must be 0",
        ),
        (
            {"STARVLA_DISABLE_TORCH_COMPILE": "0"},
            "STARVLA_DISABLE_TORCH_COMPILE must be 1",
        ),
    ],
)
def test_clean_pilot_rejects_stale_compile_environment(
    tmp_path,
    hostile_env,
    expected_message,
):
    result = _clean_pilot_dry_run(tmp_path, extra_env=hostile_env)

    assert result.returncode == 2
    assert expected_message in result.stderr
    assert result.stdout == ""


def test_clean_pilot_accepts_only_expected_compile_environment(tmp_path):
    result = _clean_pilot_dry_run(
        tmp_path,
        extra_env={
            "STARVLA_ALLOW_TORCH_COMPILE": "0",
            "STARVLA_DISABLE_TORCH_COMPILE": "1",
            "TORCH_COMPILE_DISABLE": "0",
            "TORCHDYNAMO_DISABLE": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert shlex.split(result.stdout)[0] == "launch"
    pilot = CLEAN_PILOT_PATH.read_text(encoding="utf-8")
    assert "export STARVLA_ALLOW_TORCH_COMPILE=0" in pilot
    assert "export STARVLA_DISABLE_TORCH_COMPILE=1" in pilot
    assert "export TORCH_COMPILE_DISABLE=1" in pilot
    assert "export TORCHDYNAMO_DISABLE=1" in pilot


@pytest.mark.parametrize(
    "extra_env",
    [
        {"STARVLA_CLEAN_PILOT_TEST_DRY_RUN": "1", "STARVLA_HOLDOUT_PREFLIGHT_DRY_RUN": "0"},
        {"STARVLA_CLEAN_PILOT_TEST_DRY_RUN": "0", "STARVLA_HOLDOUT_PREFLIGHT_DRY_RUN": "1"},
    ],
)
def test_clean_pilot_requires_paired_test_dry_run_flags(tmp_path, extra_env):
    result = _clean_pilot_dry_run(tmp_path, extra_env=extra_env)

    assert result.returncode == 2
    assert "DRY_RUN" in result.stderr
    assert result.stdout == ""


def test_clean_pilot_rejects_runtime_override_without_value(tmp_path):
    result = _clean_pilot_dry_run(
        tmp_path,
        "--datasets.vla_data.num_workers",
    )

    assert result.returncode == 2
    assert "runtime override requires a value" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "protected_args",
    [
        ("--trainer.is_resume", "true"),
        ("--trainer.resume_from_checkpoint", "/old/checkpoint"),
        ("--trainer.max_train_steps=67500",),
        ("--framework.action_model.rtc_training.enabled", "true"),
        ("--datasets.vla_data.append_task_id_to_prompt", "true"),
        ("--datasets.vla_data.lerobot_statistics_source", "gr00t"),
        ("--trainer.use_rabc", "true"),
        ("--config_yaml", "/tmp/stale.yaml"),
        ("--run_id", "failed-production-run"),
        (
            "--datasets.vla_data.data_root_dir",
            "/mnt/vla-jepa/datasets/stale-dataset",
        ),
        ("--framework.qwenvl.base_vlm", "/old/qwen-checkpoint"),
        ("--framework.action_model.action_horizon", "3"),
        (
            "--datasets.vla_data.modality_metadata_overrides.action.source_controls.start",
            "1",
        ),
        ("--datasets.vla_data.video_target_shift_steps", "0"),
        ("--trainer.freeze_modules", "qwen_vl_interface"),
        ("--trainer.learning_rate.action_model", "0.0"),
        ("--datasets.vla_data.per_device_batch_size", "1"),
    ],
)
def test_clean_pilot_rejects_protected_overrides(tmp_path, protected_args):
    result = _clean_pilot_dry_run(tmp_path, *protected_args)

    assert result.returncode == 2
    assert "Refusing protected clean-pilot override" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    ("hostile_env", "expected_message"),
    [
        (
            {"RUN_ID": "magna_interventions_failed_production_run"},
            "Clean pilot RUN_ID must start with",
        ),
        (
            {"DATA_ROOT_DIR": "/mnt/vla-jepa/datasets/stale-dataset"},
            "Clean pilot DATA_ROOT_DIR must be",
        ),
        ({"NUM_PROCESSES": "1"}, "Clean pilot NUM_PROCESSES must be 8"),
        ({"NUM_MACHINES": "2"}, "Clean pilot NUM_MACHINES must be 1"),
        (
            {"STARVLA_USE_DEEPSPEED": "1"},
            "Clean pilot STARVLA_USE_DEEPSPEED must be 0",
        ),
        (
            {"PER_DEVICE_BATCH_SIZE": "1"},
            "Clean pilot per-device batch size is fixed by config at 12",
        ),
        ({"VIDEO_BACKEND": "decord"}, "Clean pilot VIDEO_BACKEND must be pyav"),
    ],
)
def test_clean_pilot_rejects_stale_service_identity_or_data_root(
    tmp_path,
    hostile_env,
    expected_message,
):
    result = _clean_pilot_dry_run(tmp_path, extra_env=hostile_env)

    assert result.returncode == 2
    assert expected_message in result.stderr
    assert result.stdout == ""


def test_docker_launcher_forwards_generic_and_realman_data_roots():
    docker_run = DOCKER_RUN_PATH.read_text(encoding="utf-8")

    assert "  DATA_ROOT_DIR\n" in docker_run
    assert "  REALMAN_DATA_ROOT\n" in docker_run
    assert 'DOCKER_ARGS+=(--name "${DOCKER_NAME}")' in docker_run
    assert "STARVLA_CLEAN_PILOT_TEST_DRY_RUN" not in docker_run
    assert "STARVLA_HOLDOUT_PREFLIGHT_DRY_RUN" not in docker_run


def test_cloud_service_runner_has_reproducible_run_guards():
    for script_path in (CLOUD_SERVICE_PATH, IMAGE_BUILD_PATH, TRAINING_ENV_PATH):
        subprocess.run(["bash", "-n", str(script_path)], check=True)

    service = CLOUD_SERVICE_PATH.read_text(encoding="utf-8")
    assert "RUN_ENV_FILE" in service
    assert 'TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-}"' in service
    assert "require_value TRAIN_LAUNCHER" in service
    assert "RESUME_CHECKPOINT" in service
    assert "EXPECTED_SOURCE_COMMIT" in service
    assert "Runtime source mismatch" in service
    assert "Runtime source path mismatch" in service
    assert 'acquire_service_lock "${SERVICE_NAME}"' in service
    assert "SERVICE_NAME=tensorboard" in service
    assert "production_preflight_manifest.txt" in service
    assert "HANDOFF_METADATA_DIR" in service
    assert 'EXTRA_METADATA_DIR="${HANDOFF_METADATA_DIR}"' in service
    assert 'MOUNT_GCLOUD=0' in service
    assert 'DOCKER_GPU_MODE=none' in service
    assert 'DOCKER_NAME="${TRAIN_CONTAINER_NAME}"' in service

    image_build = IMAGE_BUILD_PATH.read_text(encoding="utf-8")
    assert 'INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-0}"' in image_build
    assert 'FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn==2.8.3.post1}"' in image_build
    assert 'FLASH_ATTN_CUDA_ARCH_LIST="${FLASH_ATTN_CUDA_ARCH_LIST:-8.0}"' in image_build
    assert './scripts/docker_build_training.sh "$@"' in image_build
    assert "Refusing to build the production image from a dirty worktree" in image_build

    training_env = TRAINING_ENV_PATH.read_text(encoding="utf-8")
    assert "/proc/net/route" in training_env
    assert 'detected_ifname="$(awk' in training_env


def test_cloud_service_runner_accepts_symlinked_run_source(tmp_path):
    linked_repo = tmp_path / "repo-link"
    linked_repo.symlink_to(REPO_ROOT, target_is_directory=True)
    scratch = tmp_path / "scratch"
    launch_env = tmp_path / "launch.env"
    launch_env.write_text("RUN_ID=symlink-source-smoke\n", encoding="utf-8")
    launch_env.chmod(0o644)
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": "symlink-source-smoke",
            "RUN_SOURCE": str(linked_repo),
            "VLA_JEPA_SCRATCH": str(scratch),
            "CHECKPOINT_ROOT": str(scratch / "checkpoints"),
            "LOG_ROOT": str(scratch / "logs"),
            "RUN_ENV_FILE": str(launch_env),
        }
    )

    result = subprocess.run(
        ["bash", str(linked_repo / "scripts/run_cloud_training_service.sh"), "status"],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Runtime source path mismatch" not in result.stderr


def test_cloud_service_runner_launches_named_container_and_records_exit(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_calls = tmp_path / "docker_calls.txt"
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(docker_calls))}\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    scratch = tmp_path / "scratch"
    checkpoint_root = scratch / "checkpoints"
    log_root = scratch / "logs"
    launch_env = tmp_path / "launch.env"
    launch_env.write_text("RUN_ID=service-smoke\n", encoding="utf-8")
    launch_env.chmod(0o644)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "RUN_ID": "service-smoke",
            "VLA_JEPA_SCRATCH": str(scratch),
            "CHECKPOINT_ROOT": str(checkpoint_root),
            "LOG_ROOT": str(log_root),
            "TRAIN_LAUNCHER": "/bin/true",
            "RUN_ENV_FILE": str(launch_env),
        }
    )

    subprocess.run(
        ["bash", str(CLOUD_SERVICE_PATH), "train"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    calls = docker_calls.read_text(encoding="utf-8")
    assert "--name service-smoke-train" in calls
    assert "-e RUN_ID=service-smoke" in calls
    assert (log_root / "service-smoke.exit").read_text(encoding="utf-8") == "0\n"


def test_cloud_service_runner_rejects_nonempty_run_without_resume(tmp_path):
    scratch = tmp_path / "scratch"
    run_dir = scratch / "checkpoints/service-smoke"
    run_dir.mkdir(parents=True)
    (run_dir / "partial-artifact").write_text("incomplete\n", encoding="utf-8")
    launch_env = tmp_path / "launch.env"
    launch_env.write_text("RUN_ID=service-smoke\n", encoding="utf-8")
    launch_env.chmod(0o644)
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": "service-smoke",
            "VLA_JEPA_SCRATCH": str(scratch),
            "CHECKPOINT_ROOT": str(scratch / "checkpoints"),
            "LOG_ROOT": str(scratch / "logs"),
            "TRAIN_LAUNCHER": "/bin/true",
            "RUN_ENV_FILE": str(launch_env),
        }
    )

    result = subprocess.run(
        ["bash", str(CLOUD_SERVICE_PATH), "train"],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Run directory is not empty" in result.stderr
    assert (scratch / "logs/service-smoke.exit").read_text(encoding="utf-8") == "2\n"


def test_cloud_service_runner_rejects_duplicate_without_replacing_exit_marker(
    tmp_path,
):
    scratch = tmp_path / "scratch"
    log_root = scratch / "logs"
    log_root.mkdir(parents=True)
    lock_path = log_root / "service-smoke.train.lock"
    launch_env = tmp_path / "launch.env"
    launch_env.write_text("RUN_ID=service-smoke\n", encoding="utf-8")
    launch_env.chmod(0o644)
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": "service-smoke",
            "VLA_JEPA_SCRATCH": str(scratch),
            "CHECKPOINT_ROOT": str(scratch / "checkpoints"),
            "LOG_ROOT": str(log_root),
            "RUN_ENV_FILE": str(launch_env),
        }
    )

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            ["bash", str(CLOUD_SERVICE_PATH), "train"],
            check=False,
            env=env,
            text=True,
            capture_output=True,
        )

    assert result.returncode == 2
    assert "Another train service already owns" in result.stderr
    assert not (log_root / "service-smoke.exit").exists()


def test_cloud_service_runner_rejects_unsafe_launch_environment(tmp_path):
    injected_path = tmp_path / "injected"
    cases = (
        (
            f"RUN_ID=$(touch {injected_path})\n",
            "only comments, blanks, and literal KEY=value assignments",
        ),
        (
            "RUN_ID=service-smoke\nWANDB_API_KEY=not-a-real-key\n",
            "secret-like variable",
        ),
    )

    for index, (contents, expected_error) in enumerate(cases):
        launch_env = tmp_path / f"unsafe-{index}.env"
        launch_env.write_text(contents, encoding="utf-8")
        launch_env.chmod(0o644)
        env = os.environ.copy()
        env["RUN_ENV_FILE"] = str(launch_env)
        result = subprocess.run(
            ["bash", str(CLOUD_SERVICE_PATH), "train"],
            check=False,
            env=env,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 2
        assert expected_error in result.stderr

    assert not injected_path.exists()
