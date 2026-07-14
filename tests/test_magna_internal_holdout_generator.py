from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from deployment.realman import build_magna_internal_holdout as holdout


CAMERAS = (
    "observation.images.head",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


def _write_fixture(root: Path, lengths: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    for camera in CAMERAS:
        path = root / "videos" / camera / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True)
        path.touch()

    episode_rows = []
    data_rows = []
    global_index = 0
    video_timestamp = 0.0
    for episode_id, length in enumerate(lengths):
        row = {
            "episode_index": episode_id,
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + length,
        }
        for camera in CAMERAS:
            prefix = f"videos/{camera}"
            row[f"{prefix}/chunk_index"] = 0
            row[f"{prefix}/file_index"] = 0
            row[f"{prefix}/from_timestamp"] = video_timestamp
            row[f"{prefix}/to_timestamp"] = video_timestamp + length / 20.0
        episode_rows.append(row)
        for frame_id in range(length):
            value = float(episode_id * 100 + frame_id)
            data_rows.append(
                {
                    "action": [value],
                    "observation.state": [value + 0.25],
                    "source.action": [value, value + 0.5],
                    "source.observation.state": [value + 1.0, value + 1.5],
                    "timestamp": frame_id / 20.0,
                    "frame_index": frame_id,
                    "episode_index": episode_id,
                    "index": global_index + frame_id,
                    "task_index": 0,
                    "subtask_index": episode_id % 3,
                    "valid_state": int(frame_id % 4 != 0),
                    "valid_state_source": episode_id % 2,
                    "task_id": episode_id % 3,
                }
            )
        global_index += length
        video_timestamp += length / 20.0

    episodes = pd.DataFrame(episode_rows)
    data = pd.DataFrame(data_rows)
    episodes.to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    data.to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 20,
                "total_episodes": len(lengths),
                "total_frames": sum(lengths),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return episodes, data


def _write_config_and_launcher(
    tmp_path: Path,
    *,
    per_device_batch: int,
    gradient_accumulation: int,
    world_size: int,
) -> tuple[Path, Path]:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "framework": {
                    "action_model": {"action_horizon": 3},
                    "vj2_model": {"num_frames": 2},
                },
                "datasets": {
                    "vla_data": {
                        "per_device_batch_size": per_device_batch,
                        "video_frame_stride": 1,
                        "video_target_shift_steps": 1,
                    }
                },
                "trainer": {
                    "gradient_accumulation_steps": gradient_accumulation
                },
            }
        ),
        encoding="utf-8",
    )
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(
        f"#!/usr/bin/env bash\nexport NUM_PROCESSES={world_size}\n",
        encoding="utf-8",
    )
    return config, launcher


def _args(
    dataset_root: Path,
    config: Path,
    launcher: Path,
    manifest: Path,
    *,
    world_size: int,
    overwrite: bool = False,
) -> Namespace:
    return Namespace(
        dataset_root=dataset_root,
        config=config,
        launcher=launcher,
        world_size=world_size,
        seed_text="synthetic-magna-holdout-v1",
        manifest=manifest,
        overwrite=overwrite,
    )


def test_effective_batch_and_default_filename_change_together(tmp_path: Path):
    config, launcher = _write_config_and_launcher(
        tmp_path,
        per_device_batch=2,
        gradient_accumulation=3,
        world_size=4,
    )
    first = holdout._derive_effective_global_batch(
        config, launcher, world_size=4
    )
    assert first["effective_global_batch_size"] == 24
    assert "global_batch24" in str(holdout._default_manifest_argument(24))

    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["trainer"]["gradient_accumulation_steps"] = 4
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    second = holdout._derive_effective_global_batch(
        config, launcher, world_size=4
    )
    assert second["effective_global_batch_size"] == 32
    assert "global_batch32" in str(holdout._default_manifest_argument(32))


def test_sha_ranking_is_a_nested_prefix(tmp_path: Path):
    root = tmp_path / "dataset"
    episodes, _ = _write_fixture(root, [5, 6, 7, 8, 9, 10])
    ranked = holdout._rank_episodes(
        episodes,
        seed_sha256="a" * 64,
        full_catalog_sha256="b" * 64,
    )
    first_two = [row["episode_id"] for row in ranked[:2]]
    first_four = [row["episode_id"] for row in ranked[:4]]
    assert first_four[:2] == first_two
    assert len(set(first_four)) == 4


def test_generator_excludes_holdout_rows_from_statistics(tmp_path: Path):
    root = tmp_path / "dataset"
    _, data = _write_fixture(root, [5, 6, 7, 8, 9])
    config, launcher = _write_config_and_launcher(
        tmp_path,
        per_device_batch=1,
        gradient_accumulation=1,
        world_size=2,
    )
    manifest_path = tmp_path / "split.json"
    holdout.build(
        _args(root, config, launcher, manifest_path, world_size=2)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["datasets"][0]
    holdout_ids = set(entry["holdout_episode_indices"])
    train_data = data[~data["episode_index"].isin(holdout_ids)]
    stats_path = manifest_path.parent / entry["train_statistics"]["path"]
    statistics = json.loads(stats_path.read_text(encoding="utf-8"))

    expected = np.vstack(train_data["source.action"].to_numpy())
    assert statistics["source.action"]["count"] == [len(train_data)]
    assert np.allclose(statistics["source.action"]["mean"], expected.mean(axis=0))
    assert statistics["_split_provenance"] == {
        "schema": "lerobot-train-statistics-v1",
        "full_catalog_sha256": entry["full_catalog_sha256"],
        "train_catalog_sha256": entry["train_catalog_sha256"],
        "train_episode_count": entry["train_episode_count"],
        "train_frame_count": entry["train_frame_count"],
    }


def test_generator_ranks_only_structurally_evaluable_episodes(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_fixture(root, [2, 5, 6, 7, 8])
    config, launcher = _write_config_and_launcher(
        tmp_path,
        per_device_batch=1,
        gradient_accumulation=1,
        world_size=2,
    )
    manifest_path = tmp_path / "split.json"
    holdout.build(
        _args(root, config, launcher, manifest_path, world_size=2)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = manifest["selection"]
    assert selection["eligibility"]["minimum_episode_length"] == 3
    assert selection["eligibility"]["eligible_episode_count"] == 4
    assert selection["eligibility"]["ineligible_episodes"] == [
        {
            "episode_id": 0,
            "length": 2,
            "reason": "length<3 cannot span offsets [0,2]",
        }
    ]
    assert 0 not in manifest["datasets"][0]["holdout_episode_indices"]


def test_existing_artifacts_reuse_and_tamper_or_partial_fail(tmp_path: Path):
    root = tmp_path / "dataset"
    _write_fixture(root, [5, 6, 7, 8, 9])
    config, launcher = _write_config_and_launcher(
        tmp_path,
        per_device_batch=1,
        gradient_accumulation=1,
        world_size=2,
    )
    manifest_path = tmp_path / "split.json"
    args = _args(root, config, launcher, manifest_path, world_size=2)
    holdout.build(args)
    holdout.build(args)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats_path = manifest_path.parent / manifest["datasets"][0]["train_statistics"]["path"]
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["source.action"]["mean"][0] += 1.0
    holdout._write_json(stats_path, stats)
    with pytest.raises(ValueError, match="does not bind"):
        holdout.build(args)

    args.overwrite = True
    holdout.build(args)
    args.overwrite = False
    report_path = manifest_path.with_name(manifest_path.stem + "_report.json")
    report_path.unlink()
    with pytest.raises(FileNotFoundError, match="artifact set is incomplete"):
        holdout.build(args)


def test_catalog_completeness_rejects_noncontiguous_frames(tmp_path: Path):
    root = tmp_path / "dataset"
    _, data = _write_fixture(root, [5, 6, 7])
    data.loc[data.index[2], "frame_index"] = 1
    data.to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    episodes = holdout._load_episode_table(root)
    with pytest.raises(ValueError, match="exactly frame_index"):
        holdout._verify_complete_catalog(root, episodes, fps=20.0)


def test_real_magna_deployment_action_window_keeps_short_episode_1619():
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(
        "/home/mehul/work/reward_model_small/magna_training_data_with_interventions"
    )
    if not dataset_root.is_dir():
        pytest.skip("Local Magna dataset is unavailable")
    config = (
        repo_root
        / "scripts/config/"
        "vlajepa_robot_ft_lerobot_magna_interventions_a100x8_"
        "qwen35_2b_full_moge_vitb_vjepa_large.yaml"
    )
    launcher = (
        repo_root
        / "scripts/vlajepa_robot_ft_lerobot_magna_clean_rtc0_pilot_a100x8.sh"
    )
    batch = holdout._derive_effective_global_batch(
        config, launcher, world_size=8
    )
    assert batch["evaluation_structural_window"]["observation_mode"] == (
        "deployment_action_current_qwen_rgb_v1"
    )
    assert batch["evaluation_structural_window"]["evaluation_video_offsets"] == [0]
    assert batch["evaluation_structural_window"]["required_min_offset"] == 0
    assert batch["evaluation_structural_window"]["required_max_offset"] == 49
    assert batch["evaluation_structural_window"]["minimum_episode_length"] == 50

    episodes = holdout._load_episode_table(dataset_root)
    ineligible = episodes[episodes["length"] < 50][
        ["episode_index", "length"]
    ].to_dict(orient="records")
    assert ineligible == []
    eligible = episodes[episodes["length"] >= 50]
    ranked = holdout._rank_episodes(
        eligible,
        seed_sha256="a" * 64,
        full_catalog_sha256="b" * 64,
    )
    assert 1619 in [row["episode_id"] for row in ranked]
