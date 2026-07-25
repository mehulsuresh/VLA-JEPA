from __future__ import annotations

from pathlib import Path

import yaml

from scripts import h100_training


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / (
    "scripts/config/h100/"
    "vlajepa_robot_ft_lerobot_magna_hq_subtasks_delta_h100x8_b16_"
    "qwen35_2b_full_moge_vitb_vjepa_large.yaml"
)


def test_magna_hq_subtasks_delta_h100_config_contract():
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    runtime = payload["runtime"]
    model = payload["framework"]["action_model"]
    data = payload["datasets"]["vla_data"]
    trainer = payload["trainer"]

    assert payload["run_id"] == (
        "robot_ft_lerobot_magna_hq_subtasks_delta_h100x8_b16"
    )
    assert payload["run_root_dir"] == "/data/mehul-vla-jepa/checkpoints"
    assert runtime["scratch_root"] == "/data/mehul-vla-jepa"
    assert runtime["helper_repositories"]["moge"]["path"].startswith(
        "/data/mehul-vla-jepa/"
    )
    assert runtime["helper_repositories"]["vjepa2"]["path"].startswith(
        "/data/mehul-vla-jepa/"
    )
    assert data["data_root_dir"] == (
        "/data/mehul-vla-jepa/datasets/"
        "latest_high_quality_magna_data_final_subtask_labelled"
    )
    assert data["episode_split_manifest"] == (
        "deployment/realman/eval_manifests/"
        "magna_hq_subtasks_delta_fractional_holdout_global_batch128_v1.json"
    )
    assert data["holdout_sampling"] == {
        "algorithm": "dataset_fraction_divisor_v1",
        "minimum_episode_fraction": 0.05,
        "maximum_episode_fraction": 0.08,
        "episode_count_multiple": 8,
        "max_episode_count": 128,
        "evaluation_observation_count": 128,
    }
    assert data["data_mix"] == (
        "magna_source_no_base_no_lift_interventions_v3"
    )
    assert model["state_dim"] == model["action_dim"] == 18
    assert model["action_horizon"] == 50
    assert data["action_type"] == "joint_delta_gripper_absolute"
    assert data["action_delta_anchor"] == "chunk_start_state"
    assert data["gripper_action_type"] == "absolute"
    assert data["state_action_normalization"] == "q01_q99_unclipped"
    assert data["use_action_validity_prefix_mask"] is True
    assert data["action_validity_label_key"] == "valid_state"
    assert data["append_subtask_to_prompt"] is True
    assert data["subtask_prompt_append_probability"] == 0.7
    assert data["subtask_prompt_ignored_labels"] == [
        "__unlabeled__",
        "Unclear transition while the chain or contact state is occluded",
    ]
    assert trainer["heldout_focused_eval_required_subtasks"] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert trainer["epochs"] == 3
    assert trainer["max_train_steps"] == "auto"
    assert trainer["learning_rate"] == {
        "base": 2.0e-05,
        "qwen_vl_interface": 1.0e-05,
        "qwen_state_projector": 1.0e-04,
        "action_model": 1.0e-04,
        "vj_predictor": 1.0e-04,
        "depth_teacher_aux_head": 1.0e-04,
    }
    assert trainer["save_interval"] == trainer["eval_interval"] == 1875
    assert data["dataset_py"] == "lerobot_datasets"
    assert data["num_workers"] == 4
    assert data["persistent_workers"] is True
    assert data["multiprocessing_context"] == "forkserver"
    assert data["epoch_sampling_strategy"] == "all_sources_exhaustive"
    assert data["fail_on_sample_error"] is True
    assert data["drop_last"] is False
    assert data["shuffle"] is False
    assert (
        data["per_device_batch_size"]
        * runtime["num_processes"]
        * trainer["gradient_accumulation_steps"]
        == 128
    )
    scratch_root = Path(runtime["scratch_root"])
    assert Path(payload["run_root_dir"]).is_relative_to(scratch_root)
    assert Path(data["data_root_dir"]).is_relative_to(scratch_root)


def test_magna_hq_subtasks_delta_artifacts_satisfy_launcher_contract():
    plan = h100_training.resolve_plan(CONFIG, validate_artifacts=True)
    assert plan["artifact_validation"]["status"] == "passed"
