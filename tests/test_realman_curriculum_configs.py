from __future__ import annotations

import hashlib
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from scripts import h100_curriculum
from scripts import h100_training
from scripts import materialize_realman_handoff_smoke
from starVLA.dataloader import dataset_view


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "scripts/config/h100"
STAGE_ROOT = CONFIG_ROOT / "realman_curriculum"
SMOKE_STAGE_ROOT = STAGE_ROOT / "handoff_smoke"
CONTRACT_SHA256 = (
    "9e1df348fb137c206b42a183892e29ec6247595fffc232bb2d73d94b5f5f91ac"
)
PRODUCTION_CONFIG = (
    CONFIG_ROOT / "realman_realsource_intervention_hq_curriculum_v1.yaml"
)


def _load_yaml(path: Path) -> dict:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert isinstance(payload, dict)
    return payload


def test_curriculum_declares_bootstrap_and_complete_view_epochs():
    payload = _load_yaml(PRODUCTION_CONFIG)
    assert payload["schema_version"] == 2
    assert payload["workflow_kind"] == "production_curriculum"
    assert set(payload["bootstrap"]) == {
        "stage_config",
        "container_image",
        "scratch_root",
    }
    stages = payload["stages"]
    assert [stage["role"] for stage in stages] == [
        "pretrain",
        "adapt",
        "finetune",
    ]
    assert [stage["expected_full_dataset_epochs"] for stage in stages] == [
        1,
        2,
        4,
    ]
    assert [stage["initialization"] for stage in stages] == [
        "upstream",
        "previous_stage_final",
        "previous_stage_final",
    ]
    assert {
        stage["handoff_checkpoint"] for stage in stages
    } == {"natural_final"}
    for stage in stages:
        monitoring = stage["monitoring"]
        assert monitoring["full_epoch_required"] is True
        assert monitoring["partial_pass_is_epoch"] is False
        assert monitoring["first_epoch_exposure_fractions"][-1] == 1.0
    assert stages[1]["require_useful_subtask_coverage"] is True
    assert stages[1]["require_verified_action_supervision"] is True
    assert [
        stage["local_evaluation_manifest_sha256"] for stage in stages
    ] == [
        "592400d73c2e7bcea0969c1b150744bea44d0ae7738e807bf97e8b9157f2c794",
        "c0f64465eff57ff7e253cbf15bb3c612eb6befa5ce3e81bd12a32f9baa24e72b",
        "9730807c0ba8688c6ae525126fdf306cfbadb3d67366392c5f6eefe3399292a2",
    ]
    assert stages[0]["monitoring"]["first_epoch_exposure_fractions"] == [
        0.25,
        0.5,
        1.0,
    ]
    assert (
        payload["shared_contract"][
            "action_representation_contract_sha256"
        ]
        == CONTRACT_SHA256
    )


@pytest.mark.parametrize(
    ("name", "epochs", "base_lr", "interface_lr", "head_lr"),
    (
        ("realsource_production_50_v1.yaml", 1, 2e-5, 1e-5, 1e-4),
        ("intervention_adapt_v1.yaml", 2, 1e-5, 5e-6, 5e-5),
        ("hq_finetune_v1.yaml", 4, 4e-6, 2e-6, 3e-5),
    ),
)
def test_stage_configs_are_exhaustive_and_lr_contract_is_config_owned(
    name: str,
    epochs: int,
    base_lr: float,
    interface_lr: float,
    head_lr: float,
):
    _, payload = h100_training._load_config(STAGE_ROOT / name)
    data = payload["datasets"]["vla_data"]
    trainer = payload["trainer"]
    action = payload["framework"]["action_model"]

    assert (action["state_dim"], action["action_dim"]) == (18, 18)
    assert action["action_horizon"] == 50
    assert data["epoch_sampling_strategy"] == "all_sources_exhaustive"
    assert data["fail_on_sample_error"] is True
    assert data["drop_last"] is False
    assert data["shuffle"] is False
    assert data.get("max_shards", 0) == 0
    assert data.get("max_shards_per_dataset", 0) == 0
    assert data.get("max_windows", 0) == 0
    assert data.get("max_windows_per_dataset", 0) == 0
    assert data.get("sample_stride", 1) == 1
    assert data["subtask_prompt_append_probability"] == pytest.approx(0.7)

    assert trainer["epochs"] == epochs
    assert trainer["max_train_steps"] == "auto"
    expected_fractions = (
        [0.25, 0.5, 1.0] if name.startswith("realsource_") else [1.0]
    )
    assert trainer["checkpoint_eval_milestone_fractions"] == expected_fractions
    assert trainer["checkpoint_eval_milestone_steps"] == "auto"
    assert (
        trainer["checkpoint_eval_include_full_epoch_boundaries"] is True
    )
    assert trainer["checkpoint_eval_milestones_only"] is True
    assert trainer["checkpoint_max_to_keep"] == 0
    assert trainer["warmup_ratio"] == pytest.approx(0.05)
    assert trainer["num_warmup_steps"] == "auto"
    assert trainer["scheduler_specific_kwargs"]["min_lr_rate"] == pytest.approx(
        0.05
    )
    assert trainer["eval_before_train"] is True
    assert trainer["allow_training_stream_eval"] is False
    assert trainer["save_interval"] == trainer["eval_interval"]
    assert trainer["save_final_model"] is True
    assert trainer["pretrained_checkpoint"] is None
    assert trainer["pretrained_checkpoint_sha256"] is None
    assert trainer["reload_modules"] is None
    rates = trainer["learning_rate"]
    assert rates["base"] == pytest.approx(base_lr)
    assert rates["qwen_vl_interface"] == pytest.approx(interface_lr)
    for key in (
        "qwen_state_projector",
        "action_model",
        "vj_predictor",
        "depth_teacher_aux_head",
    ):
        assert rates[key] == pytest.approx(head_lr)
    if name == "intervention_adapt_v1.yaml":
        assert data["data_root_dir"].endswith(
            "magna_training_data_with_interventions_final_subtask_labelled"
        )
        assert data["use_action_validity_prefix_mask"] is True
        assert data["action_validity_label_key"] == "valid_state"
        assert data["action_validity_positive_is_valid"] is True
        assert data["action_validity_invalid_run_length"] == 10
        assert data["action_validity_fail_closed"] is True
        assert data["frozen_train_view_require_data_shard_hashes"] is True
        assert "magna_training_data_with_interventions" != Path(
            data["data_root_dir"]
        ).name


def test_only_reviewed_production_curriculum_and_stage_configs_are_present():
    assert {
        path.name
        for path in CONFIG_ROOT.glob("realman_*curriculum_v1.yaml")
    } == {PRODUCTION_CONFIG.name}
    assert {
        path.name
        for path in STAGE_ROOT.glob("*.yaml")
    } == {
        "realsource_common_v1.yaml",
        "realsource_production_50_v1.yaml",
        "intervention_adapt_v1.yaml",
        "hq_finetune_v1.yaml",
    }


def test_production_stages_have_identical_model_seed_batch_and_prompts():
    resolved = {}
    for name in (
        "realsource_production_50_v1.yaml",
        "intervention_adapt_v1.yaml",
        "hq_finetune_v1.yaml",
    ):
        _, payload = h100_training._load_config(STAGE_ROOT / name)
        data = payload["datasets"]["vla_data"]
        runtime = payload["runtime"]
        trainer = payload["trainer"]
        resolved[name] = {
            "architecture": h100_curriculum._model_architecture_sha256(
                payload
            ),
            "seed": payload["seed"],
            "global_batch": (
                int(runtime["num_processes"])
                * int(data["per_device_batch_size"])
                * int(trainer["gradient_accumulation_steps"])
            ),
            "prompt_probability": data[
                "subtask_prompt_append_probability"
            ],
            "statistics": data["normalization_statistics_artifact"],
            "statistics_sha256": data[
                "normalization_statistics_artifact_sha256"
            ],
        }
    assert len({entry["architecture"] for entry in resolved.values()}) == 1
    assert {entry["seed"] for entry in resolved.values()} == {42}
    assert {entry["global_batch"] for entry in resolved.values()} == {128}
    assert all(
        entry["prompt_probability"] == pytest.approx(0.7)
        for entry in resolved.values()
    )
    assert {entry["statistics"] for entry in resolved.values()} == {
        "/data/mehul-vla-jepa/data_contracts/"
        "realman_union_openpi_q01q99_v1.json"
    }
    assert len(
        {entry["statistics_sha256"] for entry in resolved.values()}
    ) == 1


def test_role_sequence_gate_accepts_only_the_three_stage_production_sequence():
    assert h100_curriculum.APPROVED_ROLE_SEQUENCES == {
        ("pretrain", "adapt", "finetune"),
    }
    for roles in h100_curriculum.APPROVED_ROLE_SEQUENCES:
        stages = [{"role": role} for role in roles]
        assert h100_curriculum._require_approved_role_sequence(stages) == roles
    for roles in (
        ("pretrain",),
        ("adapt",),
        ("finetune",),
        ("adapt", "finetune"),
        ("pretrain", "finetune"),
        ("finetune", "adapt"),
    ):
        with pytest.raises(
            h100_curriculum.CurriculumError,
            match="approved production sequence",
        ):
            h100_curriculum._require_approved_role_sequence(
                [{"role": role} for role in roles]
            )


def test_production_contract_gate_rejects_seed_batch_or_architecture_drift():
    base = {
        "id": "a",
        "model_architecture_sha256": "a" * 64,
        "seed": 42,
        "global_batch_size": 128,
    }
    matching = {
        "id": "b",
        "model_architecture_sha256": "a" * 64,
        "seed": 42,
        "global_batch_size": 128,
    }
    assert h100_curriculum._require_comparable_training_contract(
        [base, matching]
    ) == ("a" * 64, 42, 128)

    for drift, message in (
        ({"seed": 43}, "seed differs"),
        ({"global_batch_size": 64}, "global batch differs"),
        ({"model_architecture_sha256": "b" * 64}, "architecture"),
    ):
        changed = matching | drift
        with pytest.raises(h100_curriculum.CurriculumError, match=message):
            h100_curriculum._require_comparable_training_contract(
                [base, changed]
            )


def test_all_stages_share_the_same_persistent_h100_namespace():
    resolved = []
    for name in (
        "realsource_production_50_v1.yaml",
        "intervention_adapt_v1.yaml",
        "hq_finetune_v1.yaml",
    ):
        _, payload = h100_training._load_config(STAGE_ROOT / name)
        resolved.append(payload)
    assert {
        payload["runtime"]["scratch_root"] for payload in resolved
    } == {"/data/mehul-vla-jepa"}
    assert {
        payload["run_root_dir"] for payload in resolved
    } == {"/data/mehul-vla-jepa/checkpoints"}
    assert {
        payload["framework"]["depth_teacher_aux"]["moge_repo_path"]
        for payload in resolved
    } == {"/data/mehul-vla-jepa/src/MoGe"}
    assert {
        payload["framework"]["vj2_model"]["hub_repo_or_dir"]
        for payload in resolved
    } == {"/data/mehul-vla-jepa/src/vjepa2"}


def test_stage_shared_contract_excludes_curriculum_only_holdout_binding():
    _, payload = h100_training._load_config(
        STAGE_ROOT / "intervention_adapt_v1.yaml"
    )
    stage_contract = h100_curriculum._stage_shared_contract(payload)

    assert "statistics_holdout_manifest" not in stage_contract
    assert "statistics_holdout_manifest_sha256" not in stage_contract
    assert set(stage_contract) == {
        "state_dim",
        "action_dim",
        "action_horizon",
        "action_type",
        "action_delta_anchor",
        "gripper_action_type",
        "state_action_normalization",
        "normalization_statistics_artifact",
        "normalization_statistics_artifact_sha256",
        "action_representation_contract_sha256",
    }


def test_canonical_stage_uses_canonical_eval_when_episode_split_is_null():
    _, payload = h100_training._load_config(
        STAGE_ROOT / "realsource_production_50_v1.yaml"
    )
    data = payload["datasets"]["vla_data"]
    assert data["episode_split_manifest"] is None
    assert h100_curriculum._stage_local_evaluation_manifest(payload) == (
        data["canonical_eval_manifest"]
    )


def test_checkpoint_handoff_smoke_is_yaml_owned_and_exactly_one_step():
    template = _load_yaml(
        CONFIG_ROOT
        / "realman_checkpoint_handoff_smoke_v1.template.yaml"
    )
    assert template["workflow_kind"] == "checkpoint_handoff_smoke"
    assert [stage["role"] for stage in template["stages"]] == [
        "pretrain",
        "adapt",
        "finetune",
    ]
    assert [stage["initialization"] for stage in template["stages"]] == [
        "upstream",
        "previous_stage_final",
        "previous_stage_final",
    ]
    assert {
        stage["handoff_checkpoint"] for stage in template["stages"]
    } == {"natural_final"}
    assert {
        stage["expected_full_dataset_epochs"] for stage in template["stages"]
    } == {1}

    for name in (
        "realsource_one_step_v1.yaml",
        "intervention_one_step_v1.yaml",
        "hq_one_step_v1.yaml",
    ):
        _, payload = h100_training._load_config(SMOKE_STAGE_ROOT / name)
        smoke = payload["checkpoint_handoff_smoke"]
        assert smoke == {
            "schema": "realman-checkpoint-handoff-smoke-stage-v1",
            "scope": "checkpoint_handoff_only",
            "model_quality_claim_allowed": False,
            "expected_global_batch_rows": 128,
            "expected_optimizer_steps": 1,
        }
        assert payload["trainer"]["epochs"] == 1
        assert payload["trainer"]["eval_before_train"] is False
        assert payload["trainer"][
            "checkpoint_eval_milestone_fractions"
        ] == [1.0]
        assert payload["trainer"]["reload_modules"] is None
        assert (
            int(payload["runtime"]["num_processes"])
            * int(payload["datasets"]["vla_data"]["per_device_batch_size"])
            * int(payload["trainer"]["gradient_accumulation_steps"])
            == 128
        )

    _, production_intervention = h100_training._load_config(
        STAGE_ROOT / "intervention_adapt_v1.yaml"
    )
    _, smoke_intervention = h100_training._load_config(
        SMOKE_STAGE_ROOT / "intervention_one_step_v1.yaml"
    )
    for field in (
        "use_action_validity_prefix_mask",
        "action_validity_label_key",
        "action_validity_positive_is_valid",
        "action_validity_invalid_run_length",
        "action_validity_fail_closed",
    ):
        assert smoke_intervention["datasets"]["vla_data"][field] == (
            production_intervention["datasets"]["vla_data"][field]
        )


def test_smoke_materializer_uses_mount_stable_repo_relative_paths(
    tmp_path: Path,
):
    view = (
        REPO_ROOT
        / "deployment/realman/curriculum_manifests/handoff_smoke/"
        "intervention_handoff_smoke_128_v1.json"
    )
    evaluation = (
        REPO_ROOT
        / "deployment/realman/eval_manifests/"
        "magna_intervention_labelled_holdout_global_batch128_v1.json"
    )
    external_statistics = tmp_path / "statistics.json"
    override = materialize_realman_handoff_smoke._stage_override(
        template=SMOKE_STAGE_ROOT / "intervention_one_step_v1.yaml",
        view_path=view,
        view_sha256="a" * 64,
        statistics_path=external_statistics,
        statistics_sha256="b" * 64,
        evaluation_path=evaluation,
        evaluation_field="episode_split_manifest",
    )
    data = override["datasets"]["vla_data"]
    assert data["frozen_train_view_manifest"] == str(
        view.relative_to(REPO_ROOT)
    )
    assert data["episode_split_manifest"] == str(
        evaluation.relative_to(REPO_ROOT)
    )
    assert data["normalization_statistics_artifact"] == str(
        external_statistics.resolve()
    )


@pytest.mark.parametrize("source_id", ("intervention", "hq"))
def test_materialized_real_data_smoke_views_are_non_quality_exact_batches(
    source_id: str,
):
    path = (
        REPO_ROOT
        / "deployment/realman/curriculum_manifests/handoff_smoke"
        / f"{source_id}_handoff_smoke_128_v1.json"
    )
    digest = materialize_realman_handoff_smoke._validate_smoke_view(
        path, expected_source=source_id
    )
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_shared_statistics_artifact_must_bind_the_same_holdout(
    monkeypatch,
    tmp_path: Path,
):
    statistics_path = tmp_path / "statistics.json"
    statistics_path.write_text("{}\n", encoding="utf-8")
    expected_statistics_sha256 = "a" * 64
    expected_holdout_sha256 = "b" * 64
    expected_contract_sha256 = CONTRACT_SHA256

    def fake_load(path, digest):
        assert path == statistics_path
        assert digest == expected_statistics_sha256
        return {
            "contract_sha256": expected_contract_sha256,
            "normalization": "q01_q99_unclipped",
            "population": {"holdout_manifest_sha256": expected_holdout_sha256},
        }

    monkeypatch.setattr(
        h100_curriculum,
        "load_openpi_realman_union_statistics",
        fake_load,
    )
    h100_curriculum._validate_shared_statistics_artifact(
        statistics_path=statistics_path,
        expected_statistics_sha256=expected_statistics_sha256,
        expected_holdout_sha256=expected_holdout_sha256,
        expected_contract_sha256=expected_contract_sha256,
        expected_normalization="q01_q99_unclipped",
    )

    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="different statistics holdout",
    ):
        h100_curriculum._validate_shared_statistics_artifact(
            statistics_path=statistics_path,
            expected_statistics_sha256=expected_statistics_sha256,
            expected_holdout_sha256="c" * 64,
            expected_contract_sha256=expected_contract_sha256,
            expected_normalization="q01_q99_unclipped",
        )


def test_exhaustive_view_validator_rejects_non_unique_or_partial_epoch(
    tmp_path: Path,
):
    contract = "a" * 64
    content = dataset_view.make_episode_content_id(
        frame_content_sha256="b" * 64,
        length=2,
        content_contract="fixture",
    )
    lineage = dataset_view.make_episode_lineage_id(
        backend="lerobot",
        source_id="fixture",
        catalog_sha256="c" * 64,
        episode_index=0,
        length=2,
    )
    descriptor = {
        "view_name": "fixture",
        "sources": [
            {
                "source_id": "fixture",
                "backend": "lerobot",
                "catalog_sha256": "c" * 64,
                "annotation_sha256": "d" * 64,
                "source_content_sha256": "e" * 64,
            }
        ],
        "representation": {
            "contract_sha256": contract,
            "state_dim": 18,
            "action_dim": 18,
            "horizon": 50,
            "target_fps": 20,
        },
        "selection": {"kind": "fixture"},
        "holdout_exclusions": dataset_view.make_holdout_exclusions(
            episode_indices=(),
            lineage_ids=(),
            content_ids=(),
            source_id="fixture",
        ),
        "epoch_contract": {
            "mode": "all_exhaustive",
            "epoch_passes": 1,
            "replacement": False,
            "drop_last": False,
            "shuffle": "deterministic_bijection_per_epoch",
            "ddp_tail": "duplicated_padding_reported_separately",
        },
    }
    rows = [
        {
            "backend": "lerobot",
            "source_id": "fixture",
            "episode_index": 0,
            "base_index": index,
            "horizon": 50,
            "target_fps": 20,
            "end_clamp_policy": "repeat_last",
            "episode_lineage_id": lineage,
            "episode_content_id": content,
            "sample_id": dataset_view.make_sample_id(
                episode_content_id=content,
                base_index=index,
                horizon=50,
                target_fps=20,
                representation_contract_sha256=contract,
                end_clamp=True,
            ),
        }
        for index in range(2)
    ]
    manifest = tmp_path / "view.json"
    built = dataset_view.write_frozen_view(
        manifest,
        descriptor=descriptor,
        rows=rows,
    )
    _, count = h100_curriculum._validate_exhaustive_view(
        view_path=manifest,
        expected_manifest_sha256=built.manifest_sha256,
        stage_id="fixture",
    )
    assert count == 2

    payload = _load_yaml(manifest)
    payload["epoch_contract"]["epoch_passes"] = 2
    manifest.write_text(
        __import__("json").dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="not a one-pass exhaustive epoch",
    ):
        h100_curriculum._validate_exhaustive_view(
            view_path=manifest,
            expected_manifest_sha256=hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            stage_id="fixture",
        )


def _verified_action_audit_fixture() -> dict:
    return {
        "schema": "realman-action-supervision-audit-v1",
        "status": "verified",
        "verification_mode": (
            "reviewed_valid_state_action_semantics_contract"
        ),
        "action_label_semantics_contract": {
            "sha256": "f" * 64,
            "payload_sha256": "e" * 64,
        },
        "recovery_from_invalid_state_supervision": {
            "status": "verified",
            "invalid_run_length": 10,
            "recovery_anchor_window_count": 3,
            "recovery_anchor_with_nonzero_action_mask_count": 3,
            "minimum_supervised_action_timesteps_per_anchor": 1,
        },
    }


def test_intervention_action_supervision_gate_rejects_unverified_state_labels():
    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="valid_state.*alone is insufficient",
    ):
        h100_curriculum._validate_action_supervision_audit(
            view={
                "action_supervision_audit": {
                    "schema": "realman-action-supervision-audit-v1",
                    "status": "unverified",
                    "reasons": [
                        "no reviewed SHA-bound action-label semantics contract"
                    ],
                }
            },
            data_cfg={
                "use_action_validity_prefix_mask": True,
                "action_validity_label_key": "valid_state",
                "action_validity_positive_is_valid": True,
                "action_validity_invalid_run_length": 9,
            },
            stage_id="intervention_adapt",
        )


def test_intervention_action_supervision_gate_accepts_bound_recovery_fixture():
    audit = _verified_action_audit_fixture()
    accepted = h100_curriculum._validate_action_supervision_audit(
        view={
            "action_supervision_audit": audit,
            "sources": [
                {
                    "selected_data_shards": [
                        {
                            "path": "data/chunk-000/file-000.parquet",
                            "sha256": "d" * 64,
                            "size_bytes": 1,
                        }
                    ]
                }
            ],
        },
        data_cfg={
            "use_action_validity_prefix_mask": True,
            "action_validity_label_key": "valid_state",
            "action_validity_positive_is_valid": True,
            "action_validity_invalid_run_length": 10,
            "action_validity_fail_closed": True,
            "frozen_train_view_require_data_shard_hashes": True,
        },
        stage_id="intervention_adapt",
    )
    assert accepted is audit

    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="invalid-run length",
    ):
        h100_curriculum._validate_action_supervision_audit(
            view={
                "action_supervision_audit": audit,
                "sources": [
                    {
                        "selected_data_shards": [
                            {
                                "path": "data/chunk-000/file-000.parquet",
                                "sha256": "d" * 64,
                                "size_bytes": 1,
                            }
                        ]
                    }
                ],
            },
            data_cfg={
                "use_action_validity_prefix_mask": True,
                "action_validity_label_key": "valid_state",
                "action_validity_positive_is_valid": True,
                "action_validity_invalid_run_length": 9,
                "action_validity_fail_closed": True,
                "frozen_train_view_require_data_shard_hashes": True,
            },
            stage_id="intervention_adapt",
        )


def test_intervention_action_supervision_gate_requires_fail_closed_loader():
    audit = _verified_action_audit_fixture()
    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="action_validity_fail_closed=true",
    ):
        h100_curriculum._validate_action_supervision_audit(
            view={"action_supervision_audit": audit},
            data_cfg={
                "use_action_validity_prefix_mask": True,
                "action_validity_label_key": "valid_state",
                "action_validity_positive_is_valid": True,
                "action_validity_invalid_run_length": 10,
            },
            stage_id="intervention_adapt",
        )
