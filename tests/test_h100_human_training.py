from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from scripts import h100_resume_runtime, h100_training


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / (
    "scripts/config/"
    "vlajepa_robot_ft_lerobot_magna_interventions_"
    "h100x8_b16_qwen35_2b_full_moge_vitb_vjepa_large.yaml"
)
SHELL = REPO_ROOT / "scripts/h100_training.sh"
DOCKER_RUN = REPO_ROOT / "scripts/docker_run_training.sh"
LIBERO_H100_CONFIG = REPO_ROOT / (
    "scripts/config/h100/"
    "vlajepa_robot_ft_libero_plus_h100x8_"
    "qwen35_2b_full_moge_vitb_vjepa_large.yaml"
)
CANONICAL_H100_CONFIG = REPO_ROOT / (
    "scripts/config/h100/"
    "vlajepa_robot_ft_canonical_full_h100x8_"
    "qwen_full_rawddp_moge_vits.yaml"
)
REALMAN_COMPOSED_H100_CONFIG = REPO_ROOT / (
    "scripts/config/h100/realman_curriculum/"
    "intervention_adapt_v1.yaml"
)
MAGNA_HQ_RESUME_WORKERS4_CONFIG = REPO_ROOT / (
    "scripts/config/h100/resume_runtime/"
    "magna_hq_delta_workers4.yaml"
)
CANONICAL_HOLDOUT_POLICY = {
    "algorithm": "dataset_fraction_divisor_v1",
    "minimum_episode_fraction": 0.05,
    "maximum_episode_fraction": 0.08,
    "episode_count_multiple": 8,
    "max_episode_count": 128,
    "evaluation_observation_count": 128,
}


def _payload() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _temporary_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _canonical_data(tmp_path: Path) -> dict:
    canonical_root = tmp_path / "dataset-canonicalization"
    adapter_dir = canonical_root / "configs/dataset_adapters"
    semantic_dir = canonical_root / "src/model_v0/data"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "MANIFEST.json").write_text(
        '{"adapters":[{"path":"adapter.json"}]}',
        encoding="utf-8",
    )
    (adapter_dir / "adapter.json").write_text(
        '{"adapter_group_id":"fixture"}',
        encoding="utf-8",
    )
    (semantic_dir / "adapters.py").write_text(
        "def apply_unified_adapter(): pass\n",
        encoding="utf-8",
    )
    (semantic_dir / "unified_schema.py").write_text(
        "STATE_DIM = 53\nACTION_DIM = 49\n",
        encoding="utf-8",
    )
    return {
        "dataset_ids": [],
        "dataset_canonicalization_root": str(canonical_root),
        "adapter_dir": str(adapter_dir),
        "action_type": "joint_delta_gripper_absolute",
        "action_delta_anchor": "chunk_start_state",
        "gripper_action_type": "absolute",
        "sidecar_normalization": "shard_q01_q99_unclipped",
        "sidecar_dtype": "float16",
        "sample_stride": 1,
    }


def _canonical_episode_identity(index: int) -> list:
    return [
        f"org/dataset-{index}",
        f"sid-{index}",
        "main",
        f"data-{index}.parquet",
        index,
    ]


def _canonical_selection(
    data: dict,
    *,
    configured_episode_count: int = 1000,
    **overrides,
) -> dict:
    adapter_contract_sha256 = (
        h100_training.canonical_adapter_contract_sha256(
            data,
            local_repo_root=REPO_ROOT,
        )
    )
    action_sidecar_variant = h100_training.canonical_action_sidecar_variant(
        data,
        action_horizon=50,
        canonical_eval_manifest_sha256=None,
        exclude_eval_episodes_from_training=False,
        adapter_contract_sha256=adapter_contract_sha256,
        local_repo_root=REPO_ROOT,
    )
    sampling_plan = h100_training.derive_episode_holdout_sampling_plan(
        total_episode_count=configured_episode_count,
        evaluation_observation_count=128,
        policy=CANONICAL_HOLDOUT_POLICY,
    )
    extra_identities = [
        _canonical_episode_identity(index)
        for index in range(
            int(sampling_plan["extra_window_episode_count"])
        )
    ]
    selection = {
        "algorithm": h100_training.CANONICAL_EVAL_SELECTION_ALGORITHM,
        "seed": 42,
        "window_count": 128,
        "holdout_episode_count": int(
            sampling_plan["holdout_episode_count"]
        ),
        "base_frames_per_episode": int(
            sampling_plan["base_frames_per_episode"]
        ),
        "extra_window_episode_count": int(
            sampling_plan["extra_window_episode_count"]
        ),
        "maximum_frames_per_episode": int(
            sampling_plan["maximum_frames_per_episode"]
        ),
        "window_allocation_algorithm": str(
            sampling_plan["window_allocation_algorithm"]
        ),
        "extra_window_episode_identities": extra_identities,
        "holdout_sampling_policy": dict(CANONICAL_HOLDOUT_POLICY),
        "holdout_sampling_plan": sampling_plan,
        "candidate_count": 32,
        "action_horizon": 50,
        "action_dim": 49,
        "action_type": "joint_delta_gripper_absolute",
        "normalization": "shard_q01_q99_unclipped",
        "adapter_contract_sha256": adapter_contract_sha256,
        "action_sidecar_variant": action_sidecar_variant,
        "configured_episode_count": configured_episode_count,
        "configured_episode_catalog_sha256": "b" * 64,
    }
    if int(sampling_plan["extra_window_episode_count"]) == 0:
        selection["frames_per_episode"] = int(
            sampling_plan["base_frames_per_episode"]
        )
    selection.update(overrides)
    return selection


def _canonical_windows(selection: dict) -> list[dict]:
    extras = {
        tuple(identity)
        for identity in selection["extra_window_episode_identities"]
    }
    windows: list[dict] = []
    for episode_index in range(selection["holdout_episode_count"]):
        identity = _canonical_episode_identity(episode_index)
        count = selection["base_frames_per_episode"] + int(
            tuple(identity) in extras
        )
        for window_index in range(count):
            windows.append(
                {
                    "dataset_id": identity[0],
                    "sid": identity[1],
                    "revision": identity[2],
                    "data_file": identity[3],
                    "episode_index": identity[4],
                    "base_index": 10 + window_index,
                }
            )
    assert len(windows) == selection["window_count"]
    return windows


def _canonical_contract(manifest: Path, source: Path) -> dict:
    return {
        "canonical_eval_manifest": str(manifest),
        "canonical_source_manifest": str(source),
        "canonical_eval_selection_seed": 42,
        "canonical_eval_candidate_count": 32,
        "canonical_action_type": "joint_delta_gripper_absolute",
        "canonical_sidecar_normalization": "shard_q01_q99_unclipped",
        "action_horizon": 50,
        "action_dim": 49,
        "global_batch_size": 208,
        "evaluation_observation_count": 128,
        "holdout_sampling_policy": dict(CANONICAL_HOLDOUT_POLICY),
    }


def test_h100_plan_resolves_complete_config_owned_contract():
    plan = h100_training.resolve_plan(CONFIG, validate_artifacts=False)

    assert plan["runtime"]["container_image"] == "vla-jepa:py313-cu130-h100"
    assert plan["runtime"]["container_build"] == {
        "DOCKERFILE": "docker/Dockerfile.py313",
        "BASE_IMAGE": "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",
        "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu130",
        "PYTHON_VERSION": "3.13",
        "INSTALL_DEEPSPEED": "0",
        "INSTALL_MOGE": "1",
        "INSTALL_FLASH_ATTN": "1",
        "FLASH_ATTN_SPEC": "flash-attn==2.8.3.post1",
        "FLASH_ATTN_CUDA_ARCH_LIST": "9.0",
        "FLASH_ATTN_MAX_JOBS": "32",
        "FLASH_ATTN_NVCC_THREADS": "2",
        "INSTALL_FAST_LINEAR_ATTN": "1",
        "FAST_LINEAR_ATTN_SPEC": "flash-linear-attention[cuda]==0.5.1",
        "CAUSAL_CONV1D_SPEC": "causal-conv1d==1.6.2.post1",
        "FAST_LINEAR_ATTN_TRANSFORMERS_SPEC": "transformers==5.13.1",
        "FAST_LINEAR_ATTN_TILELANG_SPEC": "tilelang==0.1.9",
        "FAST_LINEAR_ATTN_TVM_FFI_SPEC": "apache-tvm-ffi==0.1.10",
        "FAST_LINEAR_ATTN_CUDA_ARCH_LIST": "9.0",
        "FAST_LINEAR_ATTN_MAX_JOBS": "32",
    }
    assert plan["runtime"]["num_processes"] == 8
    assert plan["runtime"]["mixed_precision"] == "bf16"
    assert plan["runtime"]["use_deepspeed"] is False
    assert plan["runtime"]["torch_compile_environment"] == "disabled"
    assert set(plan["runtime"]["helper_repositories"]) == {"moge", "vjepa2"}
    assert all(
        len(entry["commit"]) == 40
        for entry in plan["runtime"]["helper_repositories"].values()
    )
    assert plan["training"]["state_dim"] == 18
    assert plan["training"]["action_dim"] == 18
    assert plan["training"]["action_horizon"] == 50
    assert plan["training"]["subtask_prompt_probability"] == 0.7
    assert plan["training"]["global_batch_size"] == 128
    assert plan["training"]["save_final_model"] is True
    assert plan["training"]["enable_force_checkpoint_file"] is True


def test_h100_plan_accepts_source_and_materialized_checkpoint_eval_milestones(
    tmp_path,
):
    source_payload = copy.deepcopy(_payload())
    source_payload["trainer"]["checkpoint_eval_milestone_fractions"] = [
        0.25,
        0.5,
        1.0,
    ]
    source_payload["trainer"]["checkpoint_eval_milestone_steps"] = None
    source_plan = h100_training.resolve_plan(
        _temporary_config(tmp_path, source_payload),
        validate_artifacts=False,
    )
    assert source_plan["training"]["checkpoint_eval_milestone_fractions"] == [
        0.25,
        0.5,
        1.0,
    ]
    assert source_plan["training"]["checkpoint_eval_milestone_steps"] is None

    materialized_payload = copy.deepcopy(source_payload)
    materialized_payload["trainer"]["checkpoint_eval_milestone_steps"] = [
        100,
        200,
    ]
    materialized_plan = h100_training.resolve_plan(
        _temporary_config(tmp_path, materialized_payload),
        validate_artifacts=False,
    )
    assert materialized_plan["training"]["checkpoint_eval_milestone_steps"] == [
        100,
        200,
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_eval_milestone_fractions", [0.5, 0.25]),
        ("checkpoint_eval_milestone_fractions", [0.25, 0.25]),
        ("checkpoint_eval_milestone_fractions", [0.0, 1.0]),
        ("checkpoint_eval_milestone_fractions", [0.25, 1.1]),
        ("checkpoint_eval_milestone_steps", [2, 2]),
        ("checkpoint_eval_milestone_steps", [3, 2]),
        ("checkpoint_eval_milestone_steps", [0, 2]),
        ("checkpoint_eval_milestone_steps", [True, 2]),
    ],
)
def test_h100_plan_rejects_invalid_checkpoint_eval_milestones(
    tmp_path,
    field,
    value,
):
    payload = copy.deepcopy(_payload())
    payload["trainer"][field] = value
    config = _temporary_config(tmp_path, payload)

    with pytest.raises(
        h100_training.PlanError,
        match=field,
    ):
        h100_training.resolve_plan(config, validate_artifacts=False)


def test_h100_plan_rejects_milestone_after_explicit_training_end(tmp_path):
    payload = copy.deepcopy(_payload())
    payload["trainer"]["max_train_steps"] = 10
    payload["trainer"]["checkpoint_eval_milestone_steps"] = [5, 11]
    config = _temporary_config(tmp_path, payload)

    with pytest.raises(
        h100_training.PlanError,
        match="cannot exceed trainer.max_train_steps",
    ):
        h100_training.resolve_plan(config, validate_artifacts=False)


@pytest.mark.parametrize(
    ("checkpoint", "digest", "message"),
    (
        ("/tmp/model.safetensors", None, "configured together"),
        (None, "a" * 64, "configured together"),
        (
            "/tmp/model.safetensors",
            "not-a-sha",
            "pretrained_checkpoint_sha256",
        ),
    ),
)
def test_h100_plan_rejects_unauthenticated_pretrained_checkpoint(
    tmp_path,
    checkpoint,
    digest,
    message,
):
    payload = copy.deepcopy(_payload())
    payload["trainer"]["pretrained_checkpoint"] = checkpoint
    payload["trainer"]["pretrained_checkpoint_sha256"] = digest

    with pytest.raises(h100_training.PlanError, match=message):
        h100_training.resolve_plan(
            _temporary_config(tmp_path, payload),
            validate_artifacts=False,
        )


def test_realman_train_statistics_contract_rejects_missing_loader_columns():
    data_cfg = {
        "modality_metadata_overrides": {
            "state": {
                "source": {
                    "original_key": "source.observation.state",
                    "start": 0,
                    "end": 18,
                }
            },
            "action": {
                "source_controls": {
                    "original_key": "source.action",
                    "start": 0,
                    "end": 16,
                },
                "source_head": {
                    "original_key": "source.action",
                    "start": 19,
                    "end": 21,
                },
            },
        }
    }
    with pytest.raises(
        h100_training.PlanError,
        match="omit required RealMan normalization columns",
    ):
        h100_training._validate_realman_train_statistics_columns(
            {"episode_index": {"count": [100]}},
            data_cfg=data_cfg,
            expected_train_frames=100,
            declared_numeric_columns=["episode_index"],
        )


def test_realman_explicit_null_holdout_sampling_uses_legacy_batch_contract(
    tmp_path,
):
    _, payload = h100_training._load_config(CONFIG)
    data = payload["datasets"]["vla_data"]
    data["holdout_sampling"] = None
    data.pop("holdout_episode_count", None)
    config = _temporary_config(tmp_path, payload)

    plan = h100_training.resolve_plan(config, validate_artifacts=False)

    assert plan["training"]["holdout_sampling_policy"] is None
    assert plan["training"]["holdout_episode_count"] == 128
    assert plan["training"]["evaluation_observation_count"] == 128


def _legacy_realman_evaluation_sampling() -> dict:
    return {
        "observation_count": 128,
        "frames_per_episode": 16,
        "window_allocation": {
            "algorithm": "uniform_per_episode_v1",
            "holdout_episode_count": 8,
            "base_frames_per_episode": 16,
            "extra_window_episode_count": 0,
            "maximum_frames_per_episode": 16,
            "extra_window_episode_identities": [],
        },
    }


def _validate_legacy_realman_evaluation_sampling(sampling: dict) -> dict:
    return h100_training._validate_realman_evaluation_sampling_contract(
        sampling,
        manifest_holdout_episode_count=8,
        expected_observation_count=128,
        expected_allocation={
            "algorithm": "uniform_per_episode_v1",
            "holdout_episode_count": 8,
            "base_frames_per_episode": 16,
            "extra_window_episode_count": 0,
            "maximum_frames_per_episode": 16,
            "extra_window_episode_identities": [],
        },
        require_explicit_allocation=False,
        dataset_name="fixture",
        selected_episode_ids_in_rank_order=list(range(8)),
    )


def test_legacy_realman_explicit_window_allocation_is_semantically_bound():
    validated = _validate_legacy_realman_evaluation_sampling(
        _legacy_realman_evaluation_sampling()
    )

    assert validated["window_allocation_algorithm"] == "uniform_per_episode_v1"
    assert validated["base_frames_per_episode"] == 16

    wrong_algorithm = _legacy_realman_evaluation_sampling()
    wrong_algorithm["window_allocation"]["algorithm"] = (
        "balanced_digest_rank_v1"
    )
    with pytest.raises(
        h100_training.PlanError,
        match="does not match the configured contract",
    ):
        _validate_legacy_realman_evaluation_sampling(wrong_algorithm)

    wrong_episode_count = _legacy_realman_evaluation_sampling()
    wrong_episode_count["window_allocation"].update(
        holdout_episode_count=16,
        base_frames_per_episode=8,
        maximum_frames_per_episode=8,
    )
    wrong_episode_count["frames_per_episode"] = 8
    with pytest.raises(
        h100_training.PlanError,
        match="invalid balanced q/r window allocation",
    ):
        _validate_legacy_realman_evaluation_sampling(wrong_episode_count)


@pytest.mark.parametrize("frames_alias", (None, 8, True))
def test_legacy_realman_explicit_uniform_allocation_requires_exact_alias(
    frames_alias,
):
    sampling = _legacy_realman_evaluation_sampling()
    if frames_alias is None:
        sampling.pop("frames_per_episode")
    else:
        sampling["frames_per_episode"] = frames_alias

    with pytest.raises(
        h100_training.PlanError,
        match="uniform allocation requires frames_per_episode",
    ):
        _validate_legacy_realman_evaluation_sampling(sampling)


def test_legacy_realman_alias_only_manifest_remains_compatible_and_bound():
    validated = _validate_legacy_realman_evaluation_sampling(
        {
            "observation_count": 128,
            "frames_per_episode": 16,
        }
    )
    assert validated["window_allocation_algorithm"] == (
        "legacy_uniform_per_episode_v1"
    )

    with pytest.raises(
        h100_training.PlanError,
        match="observation_count must equal",
    ):
        _validate_legacy_realman_evaluation_sampling(
            {
                "observation_count": 127,
                "frames_per_episode": 16,
            }
        )


@pytest.mark.parametrize(
    ("config", "profile", "state_dim", "action_dim", "horizon"),
    [
        (
            CONFIG,
            h100_training.REALMAN_LEROBOT_PROFILE,
            18,
            18,
            50,
        ),
        (
            LIBERO_H100_CONFIG,
            h100_training.LIBERO_LEROBOT_PROFILE,
            8,
            7,
            7,
        ),
        (
            CANONICAL_H100_CONFIG,
            h100_training.CANONICAL_GCS_PROFILE,
            53,
            49,
            50,
        ),
    ],
)
def test_h100_profiles_resolve_real_dataset_configs(
    config, profile, state_dim, action_dim, horizon
):
    plan = h100_training.resolve_plan(config, validate_artifacts=False)

    assert plan["training"]["dataset_profile"] == profile
    assert plan["training"]["state_dim"] == state_dim
    assert plan["training"]["action_dim"] == action_dim
    assert plan["training"]["action_horizon"] == horizon
    assert plan["training"]["subtask_prompt_probability"] == 0.7
    assert Path(plan["training"]["data_root_dir"]).is_relative_to(
        Path(plan["runtime"]["scratch_root"])
    )
    assert Path(plan["training"]["run_root_dir"]).is_relative_to(
        Path(plan["runtime"]["scratch_root"])
    )


def test_canonical_profile_pins_and_matches_canonicalization_checkout():
    plan = h100_training.resolve_plan(
        CANONICAL_H100_CONFIG,
        validate_artifacts=False,
    )

    helper = plan["runtime"]["helper_repositories"]["dataset-canonicalization"]
    assert helper["path"] == "/mnt/vla-jepa/src/dataset-canonicalization"
    assert len(helper["commit"]) == 40
    assert plan["training"]["representation"].startswith(
        "53-D semantic state / 49-D"
    )
    assert plan["training"]["requires_gcloud"] is True
    assert (
        plan["training"]["canonical_bucket_root"]
        == "gs://robotics-datasets-yonduai/raw"
    )
    assert plan["training"]["canonical_gcs_probe_object"].startswith(
        "gs://robotics-datasets-yonduai/raw/"
    )
    assert not plan["training"]["canonical_gcs_probe_object"].endswith("/")
    assert plan["training"]["canonical_eval_min_episodes_per_shard"] == 2
    assert plan["training"]["global_batch_size"] == 208
    assert plan["training"]["evaluation_observation_count"] == 128
    assert plan["training"]["eval_per_device_batch_size"] == 16
    canonical_manifest = Path(
        plan["training"]["canonical_eval_manifest"]
    )
    assert canonical_manifest.name == (
        "canonical_full_gcs_fractional_eval128_v2.json"
    )
    assert "canonical_full_gcs_heldout_v1" not in str(canonical_manifest)


def test_gcloud_is_required_only_for_canonical_download_profile(monkeypatch):
    canonical = h100_training.resolve_plan(
        CANONICAL_H100_CONFIG,
        validate_artifacts=False,
    )
    realman = h100_training.resolve_plan(CONFIG, validate_artifacts=False)
    libero = h100_training.resolve_plan(
        LIBERO_H100_CONFIG,
        validate_artifacts=False,
    )
    monkeypatch.setattr(h100_training.shutil, "which", lambda name: None)

    with pytest.raises(h100_training.PlanError, match="require the gcloud CLI"):
        h100_training._check_canonical_gcs_access(canonical)
    h100_training._check_canonical_gcs_access(realman)
    h100_training._check_canonical_gcs_access(libero)


def test_canonical_gcloud_check_proves_auth_and_bucket_access(
    monkeypatch, capsys
):
    canonical = h100_training.resolve_plan(
        CANONICAL_H100_CONFIG,
        validate_artifacts=False,
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        if command[1:3] == ["auth", "list"]:
            return SimpleNamespace(
                returncode=0,
                stdout="robot@example.com\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="gs://object\n", stderr="")

    monkeypatch.setattr(
        h100_training.shutil,
        "which",
        lambda name: "/usr/lib/google-cloud-sdk/bin/gcloud",
    )
    monkeypatch.setattr(h100_training.subprocess, "run", fake_run)

    h100_training._check_canonical_gcs_access(canonical)

    assert commands[0][1:3] == ["auth", "list"]
    assert commands[1][1:3] == ["storage", "ls"]
    assert commands[1][3] == canonical["training"]["canonical_gcs_probe_object"]
    assert len(commands[1]) == 4
    assert "--limit=1" not in commands[1]
    assert "Canonical GCS access" in capsys.readouterr().out


@pytest.mark.parametrize(
    "probe",
    [
        "gs://another-bucket/object.parquet",
        "gs://robotics-datasets-yonduai/raw/",
        "gs://robotics-datasets-yonduai/raw/a-prefix/",
    ],
)
def test_canonical_gcloud_probe_must_be_one_exact_object(tmp_path, probe):
    _, payload = h100_training._load_config(CANONICAL_H100_CONFIG)
    payload["datasets"]["vla_data"]["gcs_access_probe_object"] = probe
    config = _temporary_config(tmp_path, payload)

    with pytest.raises(
        h100_training.PlanError,
        match="gcs_access_probe_object must name one exact object",
    ):
        h100_training.resolve_plan(config, validate_artifacts=False)


def test_plan_can_inspect_fresh_canonical_profile_before_prepare(
    tmp_path, capsys
):
    _, payload = h100_training._load_config(CANONICAL_H100_CONFIG)
    missing = Path(
        "/mnt/vla-jepa/checkpoints/"
        "pytest-fresh-canonical-not-prepared-yet.json"
    )
    payload["datasets"]["vla_data"]["canonical_eval_manifest"] = str(missing)
    config = _temporary_config(tmp_path, payload)

    assert h100_training.main(["plan", "--config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "Data-contract artifacts" in output
    assert "not_checked" in output

    with pytest.raises(
        h100_training.PlanError,
        match="canonical eval manifest is missing",
    ):
        h100_training.resolve_plan(config, validate_artifacts=True)


def test_canonical_gcloud_check_rejects_missing_active_account(monkeypatch):
    canonical = h100_training.resolve_plan(
        CANONICAL_H100_CONFIG,
        validate_artifacts=False,
    )
    monkeypatch.setattr(
        h100_training.shutil,
        "which",
        lambda name: "/usr/bin/gcloud",
    )
    monkeypatch.setattr(
        h100_training.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    with pytest.raises(h100_training.PlanError, match="active gcloud account"):
        h100_training._check_canonical_gcs_access(canonical)


def test_canonical_setup_and_check_both_run_gcloud_gate(monkeypatch):
    plan = h100_training.resolve_plan(
        CANONICAL_H100_CONFIG,
        validate_artifacts=False,
    )
    setup_plan = copy.deepcopy(plan)
    setup_plan["runtime"]["helper_repositories"] = {}
    observed = []
    monkeypatch.setattr(h100_training, "_print_plan", lambda plan: None)
    monkeypatch.setattr(
        h100_training,
        "_check_canonical_gcs_access",
        lambda plan: observed.append(plan["training"]["dataset_profile"]),
    )
    monkeypatch.setattr(
        h100_training,
        "resolve_plan",
        lambda *args, **kwargs: setup_plan,
    )

    h100_training.setup_dependencies(CANONICAL_H100_CONFIG)

    monkeypatch.setattr(
        h100_training,
        "resolve_plan",
        lambda *args, **kwargs: plan,
    )
    monkeypatch.setattr(
        h100_training,
        "_check_git",
        lambda *args, **kwargs: {"commit": "a" * 40, "status": "clean"},
    )
    monkeypatch.setattr(h100_training, "_check_files", lambda plan: None)
    monkeypatch.setattr(h100_training, "_check_hardware", lambda plan: None)
    monkeypatch.setattr(h100_training, "_check_port", lambda port: None)

    h100_training.check_plan(CANONICAL_H100_CONFIG, deep=False)

    assert observed == [
        h100_training.CANONICAL_GCS_PROFILE,
        h100_training.CANONICAL_GCS_PROFILE,
    ]


def test_helper_git_remote_identity_accepts_only_dot_git_spelling():
    configured = (
        "https://github.com/YonduAI/dataset-canonicalization.git"
    )
    assert h100_training._git_remote_identity(configured) == (
        h100_training._git_remote_identity(
            "https://github.com/YonduAI/dataset-canonicalization"
        )
    )
    assert h100_training._git_remote_identity(configured) != (
        h100_training._git_remote_identity(
            "https://github.com/YonduAI/another-repository.git"
        )
    )


def test_only_exact_non_quality_handoff_contract_gets_smoke_behavior():
    payload = {
        "checkpoint_handoff_smoke": copy.deepcopy(
            h100_training.CHECKPOINT_HANDOFF_SMOKE_CONTRACT
        )
    }
    assert h100_training._is_checkpoint_handoff_smoke(payload)

    payload["checkpoint_handoff_smoke"]["model_quality_claim_allowed"] = True
    assert not h100_training._is_checkpoint_handoff_smoke(payload)


def test_setup_reuses_clean_helper_at_exact_commit_without_network(
    monkeypatch, tmp_path
):
    helper = tmp_path / "helper"
    subprocess.run(["git", "init", str(helper)], check=True)
    subprocess.run(
        ["git", "-C", str(helper), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(helper), "config", "user.name", "Test"],
        check=True,
    )
    (helper / "README.md").write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(helper), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(helper), "commit", "-m", "pinned"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(helper), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    origin = "https://github.com/YonduAI/does-not-exist.git"
    subprocess.run(
        ["git", "-C", str(helper), "remote", "add", "origin", origin],
        check=True,
    )
    plan = {
        "runtime": {
            "helper_repositories": {
                "dataset-canonicalization": {
                    "path": str(helper),
                    "url": origin,
                    "commit": commit,
                }
            }
        }
    }
    monkeypatch.setattr(
        h100_training,
        "resolve_plan",
        lambda *args, **kwargs: plan,
    )
    monkeypatch.setattr(h100_training, "_print_plan", lambda plan: None)
    monkeypatch.setattr(
        h100_training, "_check_canonical_gcs_access", lambda plan: None
    )

    h100_training.setup_dependencies(CANONICAL_H100_CONFIG)

    assert (
        subprocess.run(
            ["git", "-C", str(helper), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        == commit
    )


def test_canonical_manifest_accepts_empty_dataset_selector_as_all(tmp_path):
    source = tmp_path / "dataset_manifests.jsonl.gz"
    source.write_bytes(b"canonical-source")
    data = _canonical_data(tmp_path)
    selection = _canonical_selection(data)
    assert selection["extra_window_episode_count"] == 16
    assert "frames_per_episode" not in selection
    windows = _canonical_windows(selection)
    manifest = tmp_path / "heldout.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "heldout",
                "source_manifest_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "selection": selection,
                "windows": windows,
            }
        ),
        encoding="utf-8",
    )

    validated = h100_training._validate_canonical_manifest(
        {"datasets": {"vla_data": data}},
        _canonical_contract(manifest, source),
    )

    assert validated["window_count"] == 128
    assert validated["holdout_episode_count"] == 56
    assert validated["source_manifest_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_canonical_manifest_accepts_uniform_alias_only_without_remainder(
    tmp_path,
):
    source = tmp_path / "dataset_manifests.jsonl.gz"
    source.write_bytes(b"canonical-source")
    data = _canonical_data(tmp_path)
    selection = _canonical_selection(
        data,
        configured_episode_count=20,
    )
    assert selection["extra_window_episode_count"] == 0
    assert selection["frames_per_episode"] == 16
    manifest = tmp_path / "uniform-heldout.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "heldout",
                "source_manifest_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "selection": selection,
                "windows": _canonical_windows(selection),
            }
        ),
        encoding="utf-8",
    )

    validated = h100_training._validate_canonical_manifest(
        {"datasets": {"vla_data": data}},
        _canonical_contract(manifest, source),
    )

    assert validated["window_count"] == 128
    assert validated["holdout_episode_count"] == 8


def test_canonical_manifest_remainder_rejects_uniform_alias(tmp_path):
    source = tmp_path / "dataset_manifests.jsonl.gz"
    source.write_bytes(b"canonical-source")
    data = _canonical_data(tmp_path)
    selection = _canonical_selection(data)
    selection["frames_per_episode"] = selection[
        "base_frames_per_episode"
    ]
    manifest = tmp_path / "invalid-remainder-alias.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "heldout",
                "source_manifest_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "selection": selection,
                "windows": _canonical_windows(selection),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        h100_training.PlanError,
        match="remainder allocation must omit frames_per_episode",
    ):
        h100_training._validate_canonical_manifest(
            {"datasets": {"vla_data": data}},
            _canonical_contract(manifest, source),
        )


def test_canonical_manifest_uniform_allocation_requires_alias(tmp_path):
    source = tmp_path / "dataset_manifests.jsonl.gz"
    source.write_bytes(b"canonical-source")
    data = _canonical_data(tmp_path)
    selection = _canonical_selection(
        data,
        configured_episode_count=20,
    )
    selection.pop("frames_per_episode")
    manifest = tmp_path / "missing-uniform-alias.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "heldout",
                "source_manifest_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "selection": selection,
                "windows": _canonical_windows(selection),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        h100_training.PlanError,
        match="uniform allocation requires integer frames_per_episode",
    ):
        h100_training._validate_canonical_manifest(
            {"datasets": {"vla_data": data}},
            _canonical_contract(manifest, source),
        )


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        ("seed", 41),
        ("candidate_count", 31),
        ("window_count", 1),
        ("action_horizon", 49),
        ("action_dim", 48),
        ("action_type", "absolute"),
        ("normalization", "shard_q01_q99"),
        ("adapter_contract_sha256", "c" * 64),
        ("action_sidecar_variant", "c" * 16),
    ],
)
def test_canonical_manifest_rejects_stale_selection_contract(
    tmp_path,
    field,
    stale_value,
):
    source = tmp_path / "dataset_manifests.jsonl.gz"
    source.write_bytes(b"canonical-source")
    data = _canonical_data(tmp_path)
    selection = _canonical_selection(
        data,
        **{field: stale_value},
    )
    manifest = tmp_path / "heldout.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "heldout",
                "source_manifest_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "selection": selection,
                "windows": _canonical_windows(
                    _canonical_selection(data)
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        h100_training.PlanError,
        match="selection contract",
    ):
        h100_training._validate_canonical_manifest(
            {"datasets": {"vla_data": data}},
            _canonical_contract(manifest, source),
        )


def test_dataset_contract_does_not_fall_through_to_realman(tmp_path):
    payload = copy.deepcopy(_payload())
    payload["datasets"]["vla_data"]["data_mix"] = "other_robot_dataset"
    payload["datasets"]["vla_data"]["action_type"] = "delta_qpos"
    payload["framework"]["action_model"]["state_dim"] = 8
    payload["framework"]["action_model"]["action_dim"] = 7
    payload["framework"]["action_model"]["action_horizon"] = 7
    payload["framework"]["action_model"]["future_action_window_size"] = 6
    config = _temporary_config(tmp_path, payload)

    with pytest.raises(h100_training.PlanError, match="unsupported or ambiguous"):
        h100_training.resolve_plan(config, validate_artifacts=False)


@pytest.mark.parametrize(
    ("source_config", "field", "value", "profile_name"),
    (
        (
            LIBERO_H100_CONFIG,
            "holdout_sampling",
            CANONICAL_HOLDOUT_POLICY,
            "LIBERO LeRobot",
        ),
        (
            LIBERO_H100_CONFIG,
            "episode_split_manifest",
            "deployment/realman/eval_manifests/not-libero.json",
            "LIBERO LeRobot",
        ),
        (
            LIBERO_H100_CONFIG,
            "canonical_eval_manifest",
            "/mnt/vla-jepa/checkpoints/eval_manifests/not-libero.json",
            "LIBERO LeRobot",
        ),
        (
            CONFIG,
            "canonical_eval_manifest",
            "/mnt/vla-jepa/checkpoints/eval_manifests/not-realman.json",
            "RealMan LeRobot",
        ),
        (
            CANONICAL_H100_CONFIG,
            "episode_split_manifest",
            "deployment/realman/eval_manifests/not-canonical.json",
            "canonical GCS",
        ),
        (
            CANONICAL_H100_CONFIG,
            "holdout_episode_count",
            32,
            "canonical GCS",
        ),
    ),
)
def test_dataset_profiles_reject_cross_profile_eval_keys(
    tmp_path,
    source_config,
    field,
    value,
    profile_name,
):
    _, payload = h100_training._load_config(source_config)
    payload["datasets"]["vla_data"][field] = value
    config = _temporary_config(tmp_path, payload)

    with pytest.raises(
        h100_training.PlanError,
        match=rf"{profile_name} does not support",
    ):
        h100_training.resolve_plan(config, validate_artifacts=False)


def test_canonical_profile_rejects_non_delta_action_contract(tmp_path):
    _, payload = h100_training._load_config(CANONICAL_H100_CONFIG)
    payload["datasets"]["vla_data"]["action_type"] = "absolute"
    config = _temporary_config(tmp_path, payload)

    with pytest.raises(
        h100_training.PlanError,
        match="action_type must be 'joint_delta_gripper_absolute'",
    ):
        h100_training.resolve_plan(config, validate_artifacts=False)


@pytest.mark.parametrize("value", [1, 3, True])
def test_canonical_profile_requires_two_eval_episodes_per_shard(
    tmp_path, value
):
    _, payload = h100_training._load_config(CANONICAL_H100_CONFIG)
    payload["datasets"]["vla_data"][
        "canonical_eval_min_episodes_per_shard"
    ] = value
    config = _temporary_config(tmp_path, payload)

    with pytest.raises(
        h100_training.PlanError,
        match="canonical_eval_min_episodes_per_shard",
    ):
        h100_training.resolve_plan(config, validate_artifacts=False)


def test_h100_rejects_dataset_path_outside_mounted_scratch(tmp_path):
    payload = copy.deepcopy(_payload())
    payload["datasets"]["vla_data"]["data_root_dir"] = "/unmounted/dataset"
    config = _temporary_config(tmp_path, payload)

    with pytest.raises(h100_training.PlanError, match="inside runtime.scratch_root"):
        h100_training.resolve_plan(config, validate_artifacts=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cfg: cfg["runtime"].pop("mixed_precision"),
            "runtime.mixed_precision",
        ),
        (
            lambda cfg: cfg["datasets"]["vla_data"].pop(
                "subtask_prompt_append_probability"
            ),
            "subtask_prompt_append_probability",
        ),
        (
            lambda cfg: cfg["trainer"].pop("save_interval"),
            "trainer.save_interval",
        ),
        (
            lambda cfg: cfg["runtime"]["container_build"]["arguments"].__setitem__(
                "INSTALL_FLASH_ATTN", "0"
            ),
            "runtime.container_build.arguments",
        ),
    ],
)
def test_h100_plan_fails_closed_when_critical_setting_is_missing_or_wrong(
    tmp_path, mutation, message
):
    payload = copy.deepcopy(_payload())
    mutation(payload)
    config = _temporary_config(tmp_path, payload)

    with pytest.raises(h100_training.PlanError, match=message):
        h100_training.resolve_plan(config, validate_artifacts=False)


def test_accelerate_receives_only_one_resolved_trainer_config():
    plan = h100_training.resolve_plan(CONFIG, validate_artifacts=False)
    command = h100_training._accelerate_command(plan, Path("/tmp/resolved.yaml"))
    trainer_index = command.index("./starVLA/training/train_starvla.py")

    assert command[trainer_index + 1 :] == [
        "--config_yaml",
        "/tmp/resolved.yaml",
    ]
    assert not any(argument.startswith("--trainer.") for argument in command)
    assert not any(argument.startswith("--datasets.") for argument in command)
    assert "--config_file" not in command


def _resume_checkpoint_fixture(
    tmp_path: Path,
    *,
    source_config_sha256: str,
    launcher_sha256: str | None = None,
) -> tuple[Path, Path]:
    run_dir = tmp_path / "checkpoints" / "resume-fixture"
    checkpoint = run_dir / "checkpoints" / "steps_10"
    checkpoint.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "trainer_state.json",
        *(f"random_states_{rank}.pkl" for rank in range(8)),
    ):
        (checkpoint / name).write_bytes(b"fixture")
    immutable_config = OmegaConf.create(
        {
            "run_id": "resume-fixture",
            "datasets": {
                "vla_data": {
                    "num_workers": 1,
                    "multiprocessing_context": "spawn",
                }
            },
            "trainer": {
                "is_resume": False,
                "resume_from_checkpoint": None,
            },
            "human_launch": {
                "source_config_sha256": source_config_sha256,
                "launcher_sha256": (
                    launcher_sha256
                    if launcher_sha256 is not None
                    else h100_training._sha256(
                        Path(h100_training.__file__).resolve()
                    )
                ),
                "container_image": "test-image:latest",
            },
        }
    )
    immutable_path = run_dir / "config.yaml"
    immutable_path.write_text(
        OmegaConf.to_yaml(immutable_config, resolve=True),
        encoding="utf-8",
    )
    return checkpoint, immutable_path


def test_resume_runtime_config_is_applied_after_immutable_config_load(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(h100_resume_runtime, "_git_commit", lambda: "b" * 40)
    source_sha = "a" * 64
    checkpoint, immutable_path = _resume_checkpoint_fixture(
        tmp_path,
        source_config_sha256=source_sha,
    )
    original_immutable_bytes = immutable_path.read_bytes()
    plan = {
        "config_sha256": source_sha,
        "runtime": {
            "num_processes": 8,
            "container_image": "test-image:latest",
        },
    }

    resolved_path, run_id = h100_resume_runtime._resolved_resume_config(
        CONFIG,
        plan,
        checkpoint=checkpoint,
        resume_runtime_config=MAGNA_HQ_RESUME_WORKERS4_CONFIG,
    )
    try:
        resolved = OmegaConf.load(resolved_path)
        assert run_id == "resume-fixture"
        assert resolved.datasets.vla_data.num_workers == 4
        assert (
            resolved.datasets.vla_data.multiprocessing_context
            == "forkserver"
        )
        assert resolved.trainer.is_resume is True
        assert resolved.trainer.resume_from_checkpoint == str(
            checkpoint.resolve()
        )
        metadata = resolved.resume_runtime_override
        assert metadata.schema == "starvla-resume-runtime-override-v1"
        assert metadata.resume_helper_path == str(
            Path(h100_resume_runtime.__file__).resolve()
        )
        assert metadata.resume_helper_sha256 == h100_training._sha256(
            Path(h100_resume_runtime.__file__).resolve()
        )
        assert metadata.source_commit == "b" * 40
        assert metadata.generated_utc
        assert metadata.container_image == "test-image:latest"
        assert metadata.container_image_digest is None
        assert metadata.runtime_config_path == str(
            MAGNA_HQ_RESUME_WORKERS4_CONFIG.resolve()
        )
        assert metadata.runtime_config_sha256 == h100_training._sha256(
            MAGNA_HQ_RESUME_WORKERS4_CONFIG
        )
        change = metadata.changes["datasets.vla_data.num_workers"]
        assert change.previous == 1
        assert change.resumed == 4
        context_change = metadata.changes[
            "datasets.vla_data.multiprocessing_context"
        ]
        assert context_change.previous == "spawn"
        assert context_change.resumed == "forkserver"
    finally:
        resolved_path.unlink()

    assert immutable_path.read_bytes() == original_immutable_bytes


def test_checked_in_resume_runtime_config_validates():
    resolved, workers, context, digest = (
        h100_resume_runtime._validate_resume_runtime_config(
            MAGNA_HQ_RESUME_WORKERS4_CONFIG
        )
    )

    assert resolved == MAGNA_HQ_RESUME_WORKERS4_CONFIG.resolve()
    assert workers == 4
    assert context == "forkserver"
    assert digest == h100_training._sha256(
        MAGNA_HQ_RESUME_WORKERS4_CONFIG
    )


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        (
            "schema_version: 2\ndatasets:\n  vla_data:\n"
            "    num_workers: 4\n"
            "    multiprocessing_context: forkserver\n",
            "schema_version",
        ),
        (
            "schema_version: 1\nextra: true\ndatasets:\n"
            "  vla_data:\n"
            "    num_workers: 4\n"
            "    multiprocessing_context: forkserver\n",
            "may contain only schema_version and datasets",
        ),
        (
            "schema_version: 1\ndatasets:\n  vla_data:\n"
            "    num_workers: 2\n"
            "    multiprocessing_context: forkserver\n",
            "exactly 4",
        ),
        (
            "schema_version: 1\ndatasets:\n  vla_data:\n"
            "    num_workers: 4\n"
            "    multiprocessing_context: spawn\n",
            "exactly forkserver",
        ),
        (
            "schema_version: 1\ndatasets:\n  vla_data:\n"
            "    num_workers: 4\n"
            "    multiprocessing_context: forkserver\n"
            "    prefetch_factor: 8\n",
            "must contain exactly",
        ),
    ),
)
def test_resume_runtime_config_schema_fails_closed(contents, message):
    path = _repo_local_temporary_config(contents)
    try:
        with pytest.raises(h100_training.PlanError, match=message):
            h100_resume_runtime._validate_resume_runtime_config(path)
    finally:
        path.unlink()


def test_resume_runtime_config_must_be_repo_owned_and_resume_only(tmp_path):
    external = tmp_path / "resume-runtime.yaml"
    external.write_text(
        "schema_version: 1\n"
        "datasets:\n"
        "  vla_data:\n"
        "    num_workers: 4\n"
        "    multiprocessing_context: forkserver\n",
        encoding="utf-8",
    )
    with pytest.raises(h100_training.PlanError, match="inside the repository"):
        h100_resume_runtime._validate_resume_runtime_config(external)


def test_resume_runtime_helper_rejects_frozen_launcher_sha_mismatch(tmp_path):
    checkpoint, _ = _resume_checkpoint_fixture(
        tmp_path,
        source_config_sha256="a" * 64,
        launcher_sha256="c" * 64,
    )
    plan = {
        "config_sha256": "a" * 64,
        "runtime": {
            "num_processes": 8,
            "container_image": "test-image:latest",
        },
    }

    with pytest.raises(h100_training.PlanError, match="launcher SHA"):
        h100_resume_runtime._resolved_resume_config(
            CONFIG,
            plan,
            checkpoint=checkpoint,
            resume_runtime_config=MAGNA_HQ_RESUME_WORKERS4_CONFIG,
        )


def test_human_shell_has_no_duplicate_image_scratch_or_build_defaults():
    text = SHELL.read_text(encoding="utf-8")

    assert "vla-jepa:py313-cu130-h100" not in text
    assert 'DEFAULT_SCRATCH="/mnt/vla-jepa"' not in text
    assert "INSTALL_FLASH_ATTN=1" not in text
    assert "INSTALL_FAST_LINEAR_ATTN=1" not in text
    assert "FLASH_ATTN_CUDA_ARCH_LIST=9.0" not in text
    assert "runtime.container_build" in text


def test_human_shell_rejects_hyperparameter_overrides_before_docker():
    result = subprocess.run(
        ["bash", str(SHELL), "start", "--trainer.epochs", "2"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "training overrides are intentionally unsupported" in result.stderr
    assert "docker" not in result.stderr.lower()


def test_human_shell_passes_repo_owned_resume_runtime_config(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "docker.txt"
    (bin_dir / "mkdir").write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
    )
    (bin_dir / "docker").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%q ' \"$@\" > {shlex.quote(str(capture))}\n",
        encoding="utf-8",
    )
    (bin_dir / "python3").write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    for executable in ("mkdir", "docker", "python3"):
        (bin_dir / executable).chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SHELL),
            "resume",
            "--config",
            str(CONFIG),
            "--checkpoint",
            "/data/mehul-vla-jepa/checkpoints/run/checkpoints/steps_10",
            "--resume-runtime-config",
            str(MAGNA_HQ_RESUME_WORKERS4_CONFIG),
            "--detach",
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    arguments = shlex.split(capture.read_text(encoding="utf-8"))
    assert "--resume-runtime-config" in arguments
    override_index = arguments.index("--resume-runtime-config")
    assert arguments[override_index + 1] == (
        "/workspace/VLA-JEPA/scripts/config/h100/resume_runtime/"
        "magna_hq_delta_workers4.yaml"
    )
    assert "scripts/h100_resume_runtime.py" in arguments
    assert "--checkpoint" in arguments
    assert "--resume" not in arguments


def test_human_shell_rejects_resume_runtime_config_for_fresh_start():
    result = subprocess.run(
        [
            "bash",
            str(SHELL),
            "start",
            "--resume-runtime-config",
            str(MAGNA_HQ_RESUME_WORKERS4_CONFIG),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "accepted only by resume" in result.stderr


def _repo_local_temporary_config(contents: str):
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".h100-bootstrap-test-",
        suffix=".yaml",
        dir=REPO_ROOT / "tests",
        delete=False,
    )
    try:
        handle.write(contents)
    finally:
        handle.close()
    return Path(handle.name)


def test_human_shell_bootstrap_rejects_external_leaf_before_build(tmp_path):
    external = tmp_path / "external.yaml"
    external.write_text("runtime: {}\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SHELL), "plan", "--config", str(external)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "config must live inside the repository" in result.stderr


def test_human_shell_bootstrap_rejects_external_extends_before_build(tmp_path):
    external = tmp_path / "external-base.yaml"
    external.write_text("runtime: {}\n", encoding="utf-8")
    leaf = _repo_local_temporary_config(f"extends: {external}\n")
    try:
        result = subprocess.run(
            ["bash", str(SHELL), "plan", "--config", str(leaf)],
            check=False,
            text=True,
            capture_output=True,
        )
    finally:
        leaf.unlink()

    assert result.returncode == 2
    assert "extended config must live inside the repository" in result.stderr


def test_human_shell_bootstrap_rejects_dockerfile_traversal(tmp_path):
    external_dockerfile = tmp_path / "Dockerfile"
    external_dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    escaping_path = os.path.relpath(external_dockerfile, REPO_ROOT)
    leaf = _repo_local_temporary_config(
        "extends: ../scripts/config/"
        + CONFIG.name
        + "\nruntime:\n"
        + "  container_build:\n"
        + "    arguments:\n"
        + f"      DOCKERFILE: {escaping_path}\n"
    )
    try:
        result = subprocess.run(
            ["bash", str(SHELL), "plan", "--config", str(leaf)],
            check=False,
            text=True,
            capture_output=True,
        )
    finally:
        leaf.unlink()

    assert result.returncode == 2
    assert "DOCKERFILE escapes the repository" in result.stderr


def test_human_shell_bootstrap_accepts_repository_symlink(tmp_path):
    repo_link = tmp_path / "VLA-JEPA"
    repo_link.symlink_to(REPO_ROOT, target_is_directory=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "mkdir").write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
    )
    (bin_dir / "docker").write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
    )
    (bin_dir / "mkdir").chmod(0o755)
    (bin_dir / "docker").chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(repo_link / "scripts/h100_training.sh"),
            "plan",
            "--config",
            str(repo_link / CONFIG.relative_to(REPO_ROOT)),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "escapes the repository" not in result.stderr


@pytest.mark.parametrize("config", [LIBERO_H100_CONFIG, CANONICAL_H100_CONFIG])
def test_human_shell_bootstrap_resolves_composed_h100_profiles(
    tmp_path, config
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "docker.txt"
    (bin_dir / "mkdir").write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
    )
    (bin_dir / "docker").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%q ' \"$@\" > {shlex.quote(str(capture))}\n",
        encoding="utf-8",
    )
    (bin_dir / "python3").write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    (bin_dir / "mkdir").chmod(0o755)
    (bin_dir / "docker").chmod(0o755)
    (bin_dir / "python3").chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"

    result = subprocess.run(
        ["bash", str(SHELL), "plan", "--config", str(config)],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    arguments = shlex.split(capture.read_text(encoding="utf-8"))
    assert "vla-jepa:py313-cu130-h100" in arguments
    assert "scripts/h100_training.py" in arguments
    assert "/workspace/VLA-JEPA/scripts/config/h100/" in " ".join(arguments)


def test_human_shell_bootstrap_accepts_unindented_extends_sequence(
    tmp_path,
):
    leaf = _repo_local_temporary_config(
        "extends:\n"
        f"- ../scripts/config/{CONFIG.name}\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "docker.txt"
    (bin_dir / "docker").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%q ' \"$@\" > {shlex.quote(str(capture))}\n",
        encoding="utf-8",
    )
    (bin_dir / "python3").write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    (bin_dir / "docker").chmod(0o755)
    (bin_dir / "python3").chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"

    try:
        result = subprocess.run(
            ["bash", str(SHELL), "plan", "--config", str(leaf)],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
    finally:
        leaf.unlink()

    assert result.returncode == 0, result.stderr
    arguments = shlex.split(capture.read_text(encoding="utf-8"))
    assert "vla-jepa:py313-cu130-h100" in arguments
    assert "scripts/h100_training.py" in arguments


def test_deep_preflight_receives_flattened_composed_config(monkeypatch):
    plan = h100_training.resolve_plan(
        CANONICAL_H100_CONFIG,
        validate_artifacts=False,
    )
    observed = []

    def fake_run(command, **kwargs):
        config_path = Path(command[command.index("--config-yaml") + 1])
        cfg = OmegaConf.load(config_path)
        observed.append(
            (
                str(cfg.datasets.vla_data.dataset_py),
                int(cfg.framework.action_model.action_dim),
                str(cfg.runtime.platform),
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(h100_training.subprocess, "run", fake_run)

    h100_training._run_deep_preflight(CANONICAL_H100_CONFIG, plan)

    assert observed == [("canonical_subset_vla", 49, "h100x8")]


def test_canonical_prepare_dispatches_flattened_config(monkeypatch):
    plan = h100_training.resolve_plan(
        CANONICAL_H100_CONFIG,
        validate_artifacts=False,
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        config_path = Path(command[command.index("--config") + 1])
        cfg = OmegaConf.load(config_path)
        assert cfg.datasets.vla_data.dataset_py == "canonical_subset_vla"
        assert cfg.framework.action_model.action_dim == 49
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        h100_training,
        "resolve_plan",
        lambda path, *, validate_artifacts=True: plan,
    )
    monkeypatch.setattr(h100_training, "_print_plan", lambda plan: None)
    monkeypatch.setattr(h100_training.subprocess, "run", fake_run)

    h100_training.prepare(CANONICAL_H100_CONFIG, confirmed=True)

    assert len(commands) == 1
    assert commands[0][1].endswith("generate_canonical_eval_manifest.py")
    assert commands[0][commands[0].index("--world-size") + 1] == "8"


def test_realman_prepare_dispatches_hash_identical_flattened_config(
    monkeypatch,
):
    plan = h100_training.resolve_plan(
        REALMAN_COMPOSED_H100_CONFIG,
        validate_artifacts=False,
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        if "build_magna_internal_holdout.py" in command[1]:
            config_path = Path(command[command.index("--config") + 1])
            cfg = OmegaConf.load(config_path)
            assert cfg.get("extends") is None
            assert cfg.datasets.vla_data.dataset_py == "lerobot_datasets"
            assert cfg.framework.action_model.action_dim == 18
            assert h100_training._sha256(config_path) == plan["config_sha256"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        h100_training,
        "resolve_plan",
        lambda path, *, validate_artifacts=True: plan,
    )
    monkeypatch.setattr(h100_training, "_print_plan", lambda plan: None)
    monkeypatch.setattr(h100_training.subprocess, "run", fake_run)

    h100_training.prepare(REALMAN_COMPOSED_H100_CONFIG, confirmed=True)

    assert len(commands) == 2
    assert commands[0][1].endswith("build_magna_internal_holdout.py")
    assert commands[1][1].endswith("compute_openpi_realman_stats.py")


def test_libero_prepare_fails_clearly_as_unnecessary(monkeypatch):
    plan = h100_training.resolve_plan(
        LIBERO_H100_CONFIG,
        validate_artifacts=False,
    )
    monkeypatch.setattr(
        h100_training,
        "resolve_plan",
        lambda path, *, validate_artifacts=True: plan,
    )
    monkeypatch.setattr(h100_training, "_print_plan", lambda plan: None)

    with pytest.raises(h100_training.PlanError, match="no launcher-managed"):
        h100_training.prepare(LIBERO_H100_CONFIG, confirmed=True)


def test_unique_run_id_does_not_move_prepared_canonical_manifest(
    monkeypatch,
):
    plan = h100_training.resolve_plan(
        CANONICAL_H100_CONFIG,
        validate_artifacts=False,
    )
    prepared_manifest = plan["training"]["canonical_eval_manifest"]
    runtime_run_id = f"{plan['training']['run_id_prefix']}_unit_test"
    monkeypatch.setattr(
        h100_training.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="a" * 40 + "\n",
        ),
    )

    resolved_path, _ = h100_training._resolved_launch_config(
        CANONICAL_H100_CONFIG,
        plan,
        run_id=runtime_run_id,
        resume_checkpoint=None,
    )
    try:
        resolved = OmegaConf.load(resolved_path)
        assert str(resolved.run_id) == runtime_run_id
        assert (
            str(resolved.datasets.vla_data.canonical_eval_manifest)
            == prepared_manifest
        )
        assert runtime_run_id not in prepared_manifest
    finally:
        resolved_path.unlink()


def _capture_docker_run(tmp_path: Path, **extra_env: str) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "docker-args.txt"
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%q ' \"$@\" > {shlex.quote(str(capture))}\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    scratch = tmp_path / "scratch"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "IMAGE": "test-image:latest",
            "DOCKER_GPU_MODE": "none",
            "DOCKER_TTY": "0",
            "VLA_JEPA_SCRATCH": str(scratch),
            **extra_env,
        }
    )
    subprocess.run(
        ["bash", str(DOCKER_RUN), "python", "-V"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    return shlex.split(capture.read_text(encoding="utf-8"))


def test_docker_runner_remains_foreground_and_ephemeral_by_default(tmp_path):
    arguments = _capture_docker_run(tmp_path)

    assert arguments[0] == "run"
    assert "--rm" in arguments
    assert "-d" not in arguments
    assert arguments[-3:] == ["test-image:latest", "python", "-V"]


def test_docker_runner_supports_named_detached_human_run(tmp_path):
    arguments = _capture_docker_run(
        tmp_path,
        DOCKER_AUTO_REMOVE="0",
        DOCKER_DETACH="1",
        DOCKER_NAME="human-h100-test",
        DOCKER_USER="1000:1000",
        DOCKER_HOME="/tmp/human-home",
    )

    assert "--rm" not in arguments
    assert "-d" in arguments
    assert arguments[arguments.index("--name") + 1] == "human-h100-test"
    assert arguments[arguments.index("--user") + 1] == "1000:1000"
    assert "HOME=/tmp/human-home" in arguments


def test_docker_runner_mounts_gcloud_auth_under_nonroot_home(tmp_path):
    sdk = tmp_path / "google-cloud-sdk"
    config = tmp_path / "gcloud-config"
    sdk.mkdir()
    config.mkdir()

    arguments = _capture_docker_run(
        tmp_path,
        DOCKER_USER="1000:1000",
        DOCKER_HOME="/tmp/human-home",
        GCLOUD_SDK_ROOT=str(sdk),
        GCLOUD_CONFIG_DIR=str(config),
        MOUNT_GCLOUD="1",
    )

    assert "CLOUDSDK_CONFIG=/tmp/human-home/.config/gcloud" in arguments
    assert (
        f"{config}:/tmp/human-home/.config/gcloud"
        in arguments
    )
    assert not any("/root/.config/gcloud" in value for value in arguments)


def test_human_entrypoints_are_executable_and_shell_valid():
    assert os.access(SHELL, os.X_OK)
    assert os.access(REPO_ROOT / "scripts/h100_training.py", os.X_OK)
    assert os.access(REPO_ROOT / "scripts/h100_resume_runtime.py", os.X_OK)
    subprocess.run(["bash", "-n", str(SHELL)], check=True)
    result = subprocess.run(
        ["bash", str(SHELL), "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "setup" in result.stdout
    assert "plan" in result.stdout
    assert "check" in result.stdout
    assert "start" in result.stdout
