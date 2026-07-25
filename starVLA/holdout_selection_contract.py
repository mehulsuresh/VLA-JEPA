"""Stable RealMan holdout-selection provenance.

The immutable split manifest must authenticate every input that can change
episode selection or evaluation-window sampling.  It must not authenticate
downstream artifacts that are built *from* that manifest (for example a
frozen training view or union-normalization sidecar), otherwise the config and
manifest form an impossible hash fixed-point.

This module is deliberately dependency-light so the offline split builder and
the human H100 launcher use the exact same canonical contract implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REALMAN_HOLDOUT_SELECTION_CONTRACT_SCHEMA = (
    "realman-holdout-selection-contract-v1"
)
DEFAULT_REALMAN_HOLDOUT_SELECTION_SEED_TEXT = (
    "magna-internal-holdout-global-batch-v1"
)
REALMAN_HOLDOUT_RANKING_ALGORITHM = "sha256-episode-ranking-v1"
REALMAN_EVALUATION_SAMPLING_ALGORITHM = (
    "nonzero_valid_unpadded_uniform_v1"
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_value(value: Any) -> Any:
    """Return a JSON-only, deterministically ordered representation."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "holdout selection contract values must be JSON-compatible, got "
        f"{type(value).__name__}"
    )


def canonical_holdout_selection_contract_bytes(
    contract: Mapping[str, Any],
) -> bytes:
    return json.dumps(
        _json_value(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def holdout_selection_contract_sha256(
    contract: Mapping[str, Any],
) -> str:
    return hashlib.sha256(
        canonical_holdout_selection_contract_bytes(contract)
    ).hexdigest()


def configured_realman_holdout_seed_text(
    config: Mapping[str, Any],
) -> str:
    data = _mapping(_mapping(config.get("datasets")).get("vla_data"))
    value = data.get(
        "holdout_selection_seed_text",
        DEFAULT_REALMAN_HOLDOUT_SELECTION_SEED_TEXT,
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "datasets.vla_data.holdout_selection_seed_text must be a "
            "non-empty string"
        )
    return value


def build_realman_holdout_selection_contract(
    config: Mapping[str, Any],
    *,
    world_size: int,
    dataset_name: str,
    seed_text: str | None = None,
) -> dict[str, Any]:
    """Build the config subset that completely owns RealMan holdout sampling.

    Dataset catalog hashes and launcher bytes remain separate manifest
    bindings.  The configured dataset identity is included here, while the
    catalog binding proves which concrete episode population was observed.
    """

    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise ValueError("world_size must be an integer")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset_name must be a non-empty string")

    datasets = _mapping(config.get("datasets"))
    data = _mapping(datasets.get("vla_data"))
    trainer = _mapping(config.get("trainer"))
    framework = _mapping(config.get("framework"))
    action_model = _mapping(framework.get("action_model"))
    vj2_model = _mapping(framework.get("vj2_model"))

    configured_seed = configured_realman_holdout_seed_text(config)
    effective_seed = configured_seed if seed_text is None else seed_text
    if not isinstance(effective_seed, str) or not effective_seed.strip():
        raise ValueError("holdout selection seed text must be non-empty")
    if effective_seed != configured_seed:
        raise ValueError(
            "the --seed-text value must match "
            "datasets.vla_data.holdout_selection_seed_text: "
            f"{effective_seed!r} != {configured_seed!r}"
        )

    per_device_batch_size = int(data.get("per_device_batch_size", 0))
    gradient_accumulation_steps = int(
        trainer.get("gradient_accumulation_steps", 1)
    )
    if per_device_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError(
            "per-device batch size and gradient accumulation must be positive"
        )

    configured_root = data.get("data_root_dir")
    configured_root_name = (
        None
        if not isinstance(configured_root, str) or not configured_root
        else Path(configured_root.rstrip("/")).name
    )
    if (
        configured_root_name is not None
        and configured_root_name != dataset_name
    ):
        raise ValueError(
            "configured RealMan data_root_dir names a different dataset: "
            f"{configured_root_name!r} != {dataset_name!r}"
        )
    configured_mix = data.get("data_mix")
    holdout_sampling = data.get("holdout_sampling")
    holdout_episode_count = data.get("holdout_episode_count")

    action_delta_mappings = _mapping(data.get("action_delta_mappings"))
    modality_overrides = _mapping(
        data.get("modality_metadata_overrides")
    )
    contract = {
        "schema": REALMAN_HOLDOUT_SELECTION_CONTRACT_SCHEMA,
        "dataset_identity": {
            "dataset_name": dataset_name,
            "configured_data_root_dir": configured_root,
            "configured_data_root_name": configured_root_name,
            "configured_data_mix": configured_mix,
            "dataset_loader": data.get("dataset_py"),
            "lerobot_version": data.get("lerobot_version"),
        },
        "ranking": {
            "algorithm": REALMAN_HOLDOUT_RANKING_ALGORITHM,
            "seed_text_utf8": effective_seed,
            "seed_sha256": hashlib.sha256(
                effective_seed.encode("utf-8")
            ).hexdigest(),
        },
        "batch": {
            "per_device_batch_size": per_device_batch_size,
            "world_size": world_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_global_batch_size": (
                per_device_batch_size
                * world_size
                * gradient_accumulation_steps
            ),
            "eval_per_device_batch_size": data.get(
                "eval_per_device_batch_size",
                per_device_batch_size,
            ),
        },
        "holdout_sampling": {
            "holdout_episode_count": holdout_episode_count,
            "policy": holdout_sampling,
        },
        "evaluation_window": {
            "sampling_algorithm": (
                REALMAN_EVALUATION_SAMPLING_ALGORITHM
            ),
            "action_horizon": action_model.get("action_horizon", 1),
            "qwen_observation_frame_index": data.get(
                "qwen_observation_frame_index",
                "current",
            ),
            "video_horizon": vj2_model.get("num_frames", 1),
            "video_frame_stride": data.get("video_frame_stride", 1),
            "video_target_shift_steps": data.get(
                "video_target_shift_steps",
                0,
            ),
        },
        "supervision_mask": {
            "use_action_validity_prefix_mask": data.get(
                "use_action_validity_prefix_mask",
                False,
            ),
            "action_validity_label_key": data.get(
                "action_validity_label_key",
                "valid_state",
            ),
            "action_validity_positive_is_valid": data.get(
                "action_validity_positive_is_valid",
                True,
            ),
            "action_validity_invalid_run_length": data.get(
                "action_validity_invalid_run_length",
                10,
            ),
            "action_validity_fail_closed": data.get(
                "action_validity_fail_closed",
                False,
            ),
            "delete_pause_frame": data.get("delete_pause_frame", False),
        },
        "representation": {
            "state_dim": action_model.get("state_dim"),
            "action_dim": action_model.get("action_dim"),
            "action_horizon": action_model.get("action_horizon", 1),
            "future_action_window_size": action_model.get(
                "future_action_window_size"
            ),
            "action_type": data.get("action_type"),
            "action_delta_anchor": data.get("action_delta_anchor"),
            "gripper_action_type": data.get("gripper_action_type"),
            "state_action_normalization": data.get(
                "state_action_normalization"
            ),
            "action_representation_contract_sha256": data.get(
                "action_representation_contract_sha256"
            ),
            "action_delta_mappings": action_delta_mappings,
            "state_modality_mapping": _mapping(
                modality_overrides.get("state")
            ),
            "action_modality_mapping": _mapping(
                modality_overrides.get("action")
            ),
        },
    }
    return _json_value(contract)
