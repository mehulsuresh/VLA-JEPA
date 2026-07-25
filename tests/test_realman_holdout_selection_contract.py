from __future__ import annotations

import copy
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from scripts import h100_training
from starVLA.holdout_selection_contract import (
    REALMAN_HOLDOUT_SELECTION_CONTRACT_SCHEMA,
    build_realman_holdout_selection_contract,
    holdout_selection_contract_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSED_CONFIG = REPO_ROOT / (
    "scripts/config/h100/realman_curriculum/hq_finetune_v1.yaml"
)


def _contract(payload: dict) -> dict:
    data = payload["datasets"]["vla_data"]
    dataset_name = Path(data["data_root_dir"].rstrip("/")).name
    return build_realman_holdout_selection_contract(
        payload,
        world_size=8,
        dataset_name=dataset_name,
    )


def test_composed_and_flattened_configs_have_same_selection_contract(
    tmp_path: Path,
):
    cfg, composed_payload = h100_training._load_config(COMPOSED_CONFIG)
    flattened_path = tmp_path / "flattened.yaml"
    flattened_path.write_text(
        OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True),
        encoding="utf-8",
    )
    _, flattened_payload = h100_training._load_config(flattened_path)

    composed = _contract(composed_payload)
    flattened = _contract(flattened_payload)
    assert composed["schema"] == REALMAN_HOLDOUT_SELECTION_CONTRACT_SCHEMA
    assert flattened == composed
    assert holdout_selection_contract_sha256(flattened) == (
        holdout_selection_contract_sha256(composed)
    )


def test_downstream_artifacts_do_not_change_selection_contract():
    _, payload = h100_training._load_config(COMPOSED_CONFIG)
    baseline = _contract(payload)
    changed = copy.deepcopy(payload)
    data = changed["datasets"]["vla_data"]
    data.update(
        {
            "episode_split_manifest": "replacement/split.json",
            "frozen_train_view_manifest": "/different/view.json",
            "frozen_train_view_manifest_sha256": "1" * 64,
            "normalization_statistics_artifact": "/different/stats.json",
            "normalization_statistics_artifact_sha256": "2" * 64,
        }
    )
    changed["run_id"] = "different-run"
    changed["trainer"]["epochs"] = 99
    changed["trainer"]["learning_rate"]["action_model"] = 9.9e-4

    assert _contract(changed) == baseline
    assert holdout_selection_contract_sha256(_contract(changed)) == (
        holdout_selection_contract_sha256(baseline)
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["datasets"]["vla_data"][
            "holdout_sampling"
        ].update({"minimum_episode_fraction": 0.06}),
        lambda payload: payload["datasets"]["vla_data"].update(
            {"action_validity_invalid_run_length": 11}
        ),
        lambda payload: payload["datasets"]["vla_data"].update(
            {"data_mix": "different-realman-population"}
        ),
        lambda payload: payload["datasets"]["vla_data"].update(
            {"holdout_selection_seed_text": "different-seed"}
        ),
        lambda payload: payload["framework"]["action_model"].update(
            {"action_horizon": 49}
        ),
        lambda payload: payload["datasets"]["vla_data"].update(
            {"action_delta_anchor": "different-anchor"}
        ),
        lambda payload: payload["datasets"]["vla_data"].update(
            {"per_device_batch_size": 8}
        ),
    ],
)
def test_every_selection_sensitive_change_changes_digest(mutate):
    _, payload = h100_training._load_config(COMPOSED_CONFIG)
    baseline = holdout_selection_contract_sha256(_contract(payload))
    changed = copy.deepcopy(payload)
    mutate(changed)
    assert holdout_selection_contract_sha256(_contract(changed)) != baseline


def test_seed_override_cannot_disagree_with_config():
    _, payload = h100_training._load_config(COMPOSED_CONFIG)
    data = payload["datasets"]["vla_data"]
    with pytest.raises(ValueError, match="must match"):
        build_realman_holdout_selection_contract(
            payload,
            world_size=8,
            dataset_name=Path(data["data_root_dir"]).name,
            seed_text="unreviewed-command-line-seed",
        )

