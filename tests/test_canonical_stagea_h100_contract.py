from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts import h100_curriculum, h100_training
from starVLA.action_representation import (
    Q01_Q99_UNCLIPPED,
    REALMAN_18D_ACTION_CONTRACT,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT_CONFIG = REPO_ROOT / (
    "scripts/config/h100/realman_curriculum/"
    "realsource_production_50_v1.yaml"
)
LEGACY_CONFIG = REPO_ROOT / (
    "scripts/config/h100/"
    "vlajepa_robot_ft_canonical_full_h100x8_"
    "qwen_full_rawddp_moge_vits.yaml"
)
STAGE_B_CONFIG = REPO_ROOT / (
    "scripts/config/h100/realman_curriculum/"
    "intervention_adapt_v1.yaml"
)
STAGE_C_CONFIG = REPO_ROOT / (
    "scripts/config/h100/realman_curriculum/"
    "hq_finetune_v1.yaml"
)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_exact_realman_canonical_stage_resolves_shared_contract():
    plan = h100_training.resolve_plan(
        EXACT_CONFIG,
        validate_artifacts=False,
    )
    training = plan["training"]

    assert training["dataset_profile"] == (
        h100_training.CANONICAL_GCS_PROFILE
    )
    assert training["canonical_exact_realman_contract"] is True
    assert (training["state_dim"], training["action_dim"]) == (18, 18)
    assert training["action_horizon"] == 50
    assert training["canonical_action_type"] == (
        "joint_delta_gripper_absolute"
    )
    assert training["canonical_eval_normalization"] == (
        Q01_Q99_UNCLIPPED
    )
    assert training["canonical_sidecar_normalization"] == (
        Q01_Q99_UNCLIPPED
    )
    assert training["action_representation_contract_sha256"] == (
        REALMAN_18D_ACTION_CONTRACT.sha256()
    )
    assert training["normalization_statistics_artifact"]
    assert training["normalization_statistics_artifact_sha256"]
    assert training["frozen_train_view_manifest"]
    assert training["frozen_train_view_manifest_sha256"]


def test_legacy_canonical_profile_keeps_53d_49d_shard_contract():
    plan = h100_training.resolve_plan(
        LEGACY_CONFIG,
        validate_artifacts=False,
    )
    training = plan["training"]

    assert training["canonical_exact_realman_contract"] is False
    assert (training["state_dim"], training["action_dim"]) == (53, 49)
    assert training["canonical_sidecar_normalization"] == (
        "shard_q01_q99_unclipped"
    )
    assert training["canonical_eval_normalization"] == (
        "shard_q01_q99_unclipped"
    )
    assert training["normalization_statistics_artifact"] is None
    assert training["frozen_train_view_manifest"] is None


def test_stage_a_checkpoint_contract_is_stage_b_input_contract():
    _, stage_a = h100_training._load_config(EXACT_CONFIG)
    _, stage_b = h100_training._load_config(STAGE_B_CONFIG)
    _, stage_c = h100_training._load_config(STAGE_C_CONFIG)

    stage_a_contract = h100_curriculum._stage_shared_contract(stage_a)
    stage_b_contract = h100_curriculum._stage_shared_contract(stage_b)

    assert stage_a_contract == stage_b_contract
    assert stage_b_contract == h100_curriculum._stage_shared_contract(
        stage_c
    )
    assert stage_a["framework"] == stage_b["framework"]
    assert stage_b["framework"] == stage_c["framework"]
    architecture_sha256 = h100_curriculum._model_architecture_sha256(
        stage_a
    )
    assert architecture_sha256 == h100_curriculum._model_architecture_sha256(
        stage_b
    )
    assert architecture_sha256 == h100_curriculum._model_architecture_sha256(
        stage_c
    )
    assert stage_a_contract == {
        "state_dim": 18,
        "action_dim": 18,
        "action_horizon": 50,
        "action_type": "joint_delta_gripper_absolute",
        "action_delta_anchor": "chunk_start_state",
        "gripper_action_type": "absolute",
        "state_action_normalization": Q01_Q99_UNCLIPPED,
        "normalization_statistics_artifact": (
            "/data/mehul-vla-jepa/data_contracts/"
            "realman_union_openpi_q01q99_v1.json"
        ),
        "normalization_statistics_artifact_sha256": (
            "79212d802c009c00ebb33bd3945b00401ee5304873d1a3b26e1ada5ee3cdc0be"
        ),
        "action_representation_contract_sha256": (
            REALMAN_18D_ACTION_CONTRACT.sha256()
        ),
    }


def test_curriculum_architecture_gate_rejects_recursive_framework_drift():
    _, stage_a = h100_training._load_config(EXACT_CONFIG)
    _, stage_b = h100_training._load_config(STAGE_B_CONFIG)
    _, stage_c = h100_training._load_config(STAGE_C_CONFIG)
    reviewed = [
        {
            "id": name,
            "model_architecture_sha256": (
                h100_curriculum._model_architecture_sha256(payload)
            ),
        }
        for name, payload in (
            ("pretrain", stage_a),
            ("adapt", stage_b),
            ("finetune", stage_c),
        )
    ]
    expected = reviewed[0]["model_architecture_sha256"]
    assert (
        h100_curriculum._require_identical_model_architectures(reviewed)
        == expected
    )

    drifted = copy.deepcopy(stage_b)
    drifted["framework"]["qwenvl"]["base_vlm"] = "Qwen/incompatible"
    reviewed[1]["model_architecture_sha256"] = (
        h100_curriculum._model_architecture_sha256(drifted)
    )
    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="same fully resolved framework/model architecture",
    ):
        h100_curriculum._require_identical_model_architectures(reviewed)


def test_stage_a_safetensors_strict_loads_into_stage_b_architecture(
    tmp_path: Path,
):
    import torch
    from safetensors.torch import load_file, save_file

    class ArchitectureProbe(torch.nn.Module):
        """Small shape-faithful checkpoint probe for the reviewed config."""

        def __init__(self, payload: dict):
            super().__init__()
            framework = payload["framework"]
            action = framework["action_model"]
            hidden = int(action["hidden_size"])
            state_dim = int(action["state_dim"])
            action_dim = int(action["action_dim"])
            horizon = int(action["action_horizon"])
            state_tokens = int(framework["qwen_state"]["num_tokens"])
            self.state_projector = torch.nn.Linear(state_dim, hidden)
            self.state_token_embeddings = torch.nn.Parameter(
                torch.empty(state_tokens, hidden)
            )
            self.action_head = torch.nn.Linear(
                hidden,
                action_dim * horizon,
            )

    _, stage_a = h100_training._load_config(EXACT_CONFIG)
    _, stage_b = h100_training._load_config(STAGE_B_CONFIG)
    stage_a_probe = ArchitectureProbe(stage_a)
    path = tmp_path / "stage_a_model.safetensors"
    save_file(stage_a_probe.state_dict(), path)

    stage_b_probe = ArchitectureProbe(stage_b)
    incompatible = stage_b_probe.load_state_dict(
        load_file(path),
        strict=True,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    drifted = copy.deepcopy(stage_b)
    drifted["framework"]["action_model"]["action_dim"] = 19
    with pytest.raises(RuntimeError, match="size mismatch"):
        ArchitectureProbe(drifted).load_state_dict(
            load_file(path),
            strict=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "action_representation_contract_sha256",
            "0" * 64,
            "action_representation_contract_sha256",
        ),
        (
            "state_action_normalization",
            "mean_std",
            "state_action_normalization",
        ),
        (
            "sidecar_normalization",
            "shard_q01_q99_unclipped",
            "sidecar_normalization",
        ),
        (
            "epoch_sampling_strategy",
            "with_replacement",
            "epoch_sampling_strategy",
        ),
    ),
)
def test_exact_realman_canonical_contract_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
):
    _, payload = h100_training._load_config(EXACT_CONFIG)
    payload["datasets"]["vla_data"][field] = value
    path = _write_config(tmp_path, payload)

    with pytest.raises(h100_training.PlanError, match=message):
        h100_training.resolve_plan(path, validate_artifacts=False)
