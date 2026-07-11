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


def test_magna_production_config_contract():
    cfg = OmegaConf.load(CONFIG_PATH)

    assert cfg.framework.qwenvl.base_vlm == "Qwen/Qwen3.5-2B"
    assert cfg.framework.qwenvl.lora.enabled is False
    assert cfg.framework.qwenvl.blockwise_attention.enabled is False
    assert cfg.framework.qwenvl.strict_full_trainable is True

    action_cfg = cfg.framework.action_model
    assert action_cfg.action_dim == 19
    assert action_cfg.state_dim == 19
    assert action_cfg.action_horizon == 50
    assert action_cfg.future_action_window_size == 49
    assert action_cfg.rtc_training.enabled is True

    data_cfg = cfg.datasets.vla_data
    assert data_cfg.data_mix == "magna_source_no_base_interventions_v3"
    assert data_cfg.action_type == "absolute_qpos"
    assert data_cfg.modality_metadata_overrides.state.source.original_key == (
        "source.observation.state"
    )
    assert (data_cfg.modality_metadata_overrides.state.source.start, data_cfg.modality_metadata_overrides.state.source.end) == (0, 19)
    assert (data_cfg.modality_metadata_overrides.action.source_controls.start, data_cfg.modality_metadata_overrides.action.source_controls.end) == (0, 16)
    assert (data_cfg.modality_metadata_overrides.action.source_head_lift.start, data_cfg.modality_metadata_overrides.action.source_head_lift.end) == (19, 22)
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
    assert DATASET_NAMED_MIXTURES["magna_source_no_base_interventions_v3"] == [
        ("", 1.0, "realman_bimanual_source_no_base", "v3.0")
    ]

    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert CONFIG_PATH.name in launcher
    assert "vlajepa_robot_ft_libero_plus_a100x8_qwen3_full_moge_vitb_vjepa_large.sh" in launcher
