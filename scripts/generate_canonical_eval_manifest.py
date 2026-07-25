#!/usr/bin/env python3
"""Prepare an immutable heldout manifest for canonical GCS training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.canonical_eval_manifest import (  # noqa: E402
    build_canonical_eval_manifest_payload,
    write_canonical_eval_manifest,
)
from starVLA.dataloader.canonical_subset_dataset import (  # noqa: E402
    get_vla_dataset,
    load_canonical_eval_manifest,
)
from starVLA.dataloader.dataset_view import (  # noqa: E402
    EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE,
)
from starVLA.eval_sampling_policy import (  # noqa: E402
    validate_holdout_sampling_policy,
)


def _load_training_config(config_path: Path):
    cfg = OmegaConf.load(config_path)
    if cfg.get("extends", None):
        # Human H100 profiles are intentionally small composed leaves. Reuse
        # the launcher's strict composition rules so prepare sees the same
        # resolved data contract that check/start will consume.
        from scripts.h100_training import _load_config

        cfg, _ = _load_config(config_path)
    return cfg


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Manifest destination. Defaults to "
            "datasets.vla_data.canonical_eval_manifest in the config."
        ),
    )
    parser.add_argument(
        "--world-size",
        required=True,
        type=int,
        help="Number of distributed training ranks that will use this config.",
    )
    parser.add_argument(
        "--window-count",
        type=int,
        help=(
            "Explicit heldout observation count. With holdout_sampling this "
            "must match evaluation_observation_count; legacy configs default "
            "to one effective global training batch."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Selection seed; defaults to the training config seed.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        help=(
            "Maximum candidate anchors inspected per episode. Defaults to "
            "datasets.vla_data.canonical_eval_candidate_count."
        ),
    )
    parser.add_argument(
        "--eval-selection-view-manifest",
        type=Path,
        help=(
            "Non-trainable eval_selection_population_candidate view used "
            "only to break the holdout/statistics bootstrap cycle. When "
            "provided, its SHA-256 must also be provided."
        ),
    )
    parser.add_argument(
        "--eval-selection-view-manifest-sha256",
        help="Expected SHA-256 of --eval-selection-view-manifest.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = _load_training_config(config_path)
    data_cfg = cfg.datasets.vla_data
    if str(data_cfg.get("dataset_py", "")) != "canonical_subset_vla":
        raise ValueError(
            "generate_canonical_eval_manifest.py requires "
            "datasets.vla_data.dataset_py=canonical_subset_vla."
        )
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive.")
    if bool(args.eval_selection_view_manifest) != bool(
        args.eval_selection_view_manifest_sha256
    ):
        raise ValueError(
            "--eval-selection-view-manifest and "
            "--eval-selection-view-manifest-sha256 must be provided together."
        )
    configured_output = data_cfg.get("canonical_eval_manifest", None)
    output_path = args.output or (
        None if not configured_output else Path(str(configured_output))
    )
    if output_path is None:
        raise ValueError(
            "Set datasets.vla_data.canonical_eval_manifest or pass --output."
        )
    if not bool(
        data_cfg.get("canonical_exclude_eval_episodes_from_training", False)
    ):
        raise ValueError(
            "Canonical config must set "
            "canonical_exclude_eval_episodes_from_training=true."
        )
    min_episodes_per_shard = int(
        data_cfg.get("canonical_eval_min_episodes_per_shard", 0)
    )
    if min_episodes_per_shard < 2:
        raise ValueError(
            "Canonical eval generation requires "
            "datasets.vla_data.canonical_eval_min_episodes_per_shard>=2 so "
            "a selected heldout shard retains train/statistics data."
        )
    configured_seed = int(
        data_cfg.get("canonical_eval_selection_seed", cfg.get("seed", 0))
    )
    if args.seed is not None and int(args.seed) != configured_seed:
        raise ValueError(
            "--seed must match datasets.vla_data.canonical_eval_selection_seed "
            f"so the loader can validate it: {args.seed} != {configured_seed}."
        )
    configured_candidate_count = int(
        data_cfg.get("canonical_eval_candidate_count", 32)
    )
    if (
        args.candidate_count is not None
        and int(args.candidate_count) != configured_candidate_count
    ):
        raise ValueError(
            "--candidate-count must match "
            "datasets.vla_data.canonical_eval_candidate_count so the loader "
            f"can validate it: {args.candidate_count} != "
            f"{configured_candidate_count}."
        )

    holdout_sampling_value = data_cfg.get("holdout_sampling", None)
    holdout_sampling_policy = (
        None
        if holdout_sampling_value is None
        else validate_holdout_sampling_policy(
            OmegaConf.to_container(
                holdout_sampling_value,
                resolve=True,
            )
        )
    )
    if holdout_sampling_policy is None:
        per_device_batch_size = int(data_cfg.per_device_batch_size)
        gradient_accumulation_steps = max(
            1,
            int(cfg.trainer.get("gradient_accumulation_steps", 1)),
        )
        expected_count = (
            per_device_batch_size
            * int(args.world_size)
            * gradient_accumulation_steps
        )
        expected_count_description = (
            "one effective global training batch: "
            f"{per_device_batch_size} per device * {args.world_size} ranks * "
            f"{gradient_accumulation_steps} accumulation"
        )
    else:
        expected_count = int(
            holdout_sampling_policy["evaluation_observation_count"]
        )
        expected_count_description = (
            "datasets.vla_data.holdout_sampling."
            f"evaluation_observation_count={expected_count}"
        )
    window_count = (
        expected_count if args.window_count is None else int(args.window_count)
    )
    if window_count != expected_count:
        raise ValueError(
            "--window-count must match the configured canonical evaluation "
            f"observation contract: requested={window_count}, "
            f"expected={expected_count} ({expected_count_description})."
        )

    source_cfg = copy.deepcopy(data_cfg)
    source_cfg["canonical_eval_manifest"] = None
    source_cfg["canonical_exclude_eval_episodes_from_training"] = False
    # Union statistics cannot exist until the holdout has been frozen.
    # Candidate selection inspects raw masks only and therefore uses the
    # explicit code-only 18-D bootstrap path below.
    source_cfg["normalization_statistics_artifact"] = None
    source_cfg["normalization_statistics_artifact_sha256"] = None
    if args.eval_selection_view_manifest is not None:
        source_cfg["frozen_train_view_manifest"] = str(
            args.eval_selection_view_manifest.expanduser().resolve()
        )
        source_cfg["frozen_train_view_manifest_sha256"] = str(
            args.eval_selection_view_manifest_sha256
        )
    source = get_vla_dataset(
        data_cfg=source_cfg,
        mode="train",
        action_horizon=int(cfg.framework.action_model.action_horizon),
        video_horizon=int(cfg.framework.vj2_model.num_frames),
        video_frame_stride=int(data_cfg.get("video_frame_stride", 1)),
        allow_eval_selection_population_candidate=True,
    )
    try:
        if (
            source.frozen_train_view is None
            or source.frozen_train_view.descriptor.get("purpose")
            != EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE
        ):
            raise ValueError(
                "Canonical eval generation requires "
                "frozen_train_view_manifest to reference a non-trainable "
                "eval_selection_population_candidate view."
            )
        payload = build_canonical_eval_manifest_payload(
            source,
            window_count=window_count,
            seed=configured_seed,
            candidate_count=configured_candidate_count,
            holdout_sampling_policy=holdout_sampling_policy,
        )
    finally:
        source.close_video_readers()

    path, created = write_canonical_eval_manifest(output_path, payload)
    validated = load_canonical_eval_manifest(
        path,
        source_manifest_path=source.manifest_path,
        expected_selection=payload["selection"],
    )
    result = {
        "status": "created" if created else "verified_unchanged",
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "window_count": len(validated.windows),
        "holdout_episode_count": int(
            validated.selection.holdout_episode_count
            or len(validated.heldout_episode_identities)
        ),
        "source_manifest_sha256": validated.source_manifest_sha256,
        "selection": payload["selection"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
