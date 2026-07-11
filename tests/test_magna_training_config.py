import fcntl
import os
import shlex
import subprocess
from pathlib import Path

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
DOCKER_RUN_PATH = REPO_ROOT / "scripts/docker_run_training.sh"
CLOUD_SERVICE_PATH = REPO_ROOT / "scripts/run_cloud_training_service.sh"
IMAGE_BUILD_PATH = REPO_ROOT / "scripts/build_magna_a100_image.sh"


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
    assert action_cfg.rtc_training.enabled is True

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
    assert cfg.trainer.epochs == 3
    assert cfg.trainer.step_scheduler_with_optimizer is False


def test_magna_mixture_and_launcher_route_to_production_config():
    assert DATASET_NAMED_MIXTURES["magna_source_no_base_no_lift_interventions_v3"] == [
        ("", 1.0, "realman_bimanual_source_no_base_no_lift", "v3.0")
    ]

    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert CONFIG_PATH.name in launcher
    assert "vlajepa_robot_ft_libero_plus_a100x8_qwen3_full_moge_vitb_vjepa_large.sh" in launcher


def test_docker_launcher_forwards_generic_and_realman_data_roots():
    docker_run = DOCKER_RUN_PATH.read_text(encoding="utf-8")

    assert "  DATA_ROOT_DIR\n" in docker_run
    assert "  REALMAN_DATA_ROOT\n" in docker_run
    assert 'DOCKER_ARGS+=(--name "${DOCKER_NAME}")' in docker_run


def test_cloud_service_runner_has_reproducible_run_guards():
    for script_path in (CLOUD_SERVICE_PATH, IMAGE_BUILD_PATH):
        subprocess.run(["bash", "-n", str(script_path)], check=True)

    service = CLOUD_SERVICE_PATH.read_text(encoding="utf-8")
    assert "RUN_ENV_FILE" in service
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
    assert 'FLASH_ATTN_CUDA_ARCH_LIST="${FLASH_ATTN_CUDA_ARCH_LIST:-8.0}"' in image_build
    assert "Refusing to build the production image from a dirty worktree" in image_build


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
