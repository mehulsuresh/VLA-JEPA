from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import starVLA.dataloader.lerobot_datasets as lerobot_datasets


ROBOT_TYPE = "realman_bimanual_source_no_base_no_lift"


class _CapturedMixture:
    def __init__(self, dataset_mixture, **kwargs):
        self.dataset_mixture = dataset_mixture
        self.kwargs = kwargs


def _patch_dataset_constructors(monkeypatch):
    calls = []

    def fake_single(data_root_dir, data_name, robot_type, **kwargs):
        dataset = object()
        calls.append(
            {
                "dataset": dataset,
                "data_root_dir": Path(data_root_dir),
                "data_name": data_name,
                "robot_type": robot_type,
                **kwargs,
            }
        )
        return dataset

    monkeypatch.setattr(
        lerobot_datasets,
        "make_LeRobotSingleDataset",
        fake_single,
    )
    monkeypatch.setattr(
        lerobot_datasets,
        "LeRobotMixtureDataset",
        _CapturedMixture,
    )
    return calls


def _source(source_id, path, weight=1.0, primary=True, **extra):
    return {
        "id": source_id,
        "path": str(path),
        "weight": weight,
        "primary": primary,
        "robot_type": ROBOT_TYPE,
        "lerobot_version": "v3.0",
        **extra,
    }


def _config(sources):
    return OmegaConf.create(
        {
            "sources": sources,
            "delete_pause_frame": True,
            "append_subtask_to_prompt": True,
            "video_backend": "decord",
            "video_backend_num_threads": 2,
        }
    )


def test_explicit_sources_keep_same_path_as_independent_logical_views(
    tmp_path,
    monkeypatch,
):
    calls = _patch_dataset_constructors(monkeypatch)
    shared_path = tmp_path / "shared"
    cfg = _config(
        [
            _source(
                "recovery",
                shared_path,
                0.25,
                config_overrides={
                    "append_subtask_to_prompt": False,
                    "video_backend_num_threads": 3,
                },
            ),
            _source(
                "expert",
                shared_path,
                0.75,
                config_overrides={
                    "append_subtask_to_prompt": True,
                    "video_backend": "pyav",
                },
            ),
        ]
    )

    mixture = lerobot_datasets.get_vla_dataset(cfg, mode="train")

    assert len(calls) == 2
    assert [weight for _, weight in mixture.dataset_mixture] == [0.25, 0.75]
    assert mixture.kwargs["primary_dataset_flags"] == [True, True]
    assert all(call["data_root_dir"] == shared_path.parent for call in calls)
    assert all(call["data_name"] == shared_path.name for call in calls)
    assert calls[0]["data_cfg"] is not calls[1]["data_cfg"]
    assert calls[0]["data_cfg"].dataset_source_id == "recovery"
    assert calls[1]["data_cfg"].dataset_source_id == "expert"
    assert calls[0]["data_cfg"].append_subtask_to_prompt is False
    assert calls[1]["data_cfg"].append_subtask_to_prompt is True
    assert calls[0]["video_backend_kwargs"] == {"num_threads": 3}
    assert calls[1]["video_backend"] == "pyav"
    assert cfg.append_subtask_to_prompt is True
    assert "dataset_source_id" not in cfg


def test_explicit_source_fields_are_bound_into_each_merged_config(
    tmp_path,
    monkeypatch,
):
    calls = _patch_dataset_constructors(monkeypatch)
    source_path = tmp_path / "source"
    cfg = _config([_source("source", source_path)])

    lerobot_datasets.get_vla_dataset(cfg, mode="eval")

    source_cfg = calls[0]["data_cfg"]
    assert source_cfg.dataset_source_id == "source"
    assert source_cfg.dataset_path == str(source_path)
    assert source_cfg.data_root_dir == str(source_path.parent)
    assert source_cfg.data_name == source_path.name
    assert source_cfg.robot_type == ROBOT_TYPE
    assert source_cfg.lerobot_version == "v3.0"
    assert calls[0]["episode_split_role"] == "eval"


def test_static_data_mix_remains_supported(tmp_path, monkeypatch):
    calls = _patch_dataset_constructors(monkeypatch)
    cfg = OmegaConf.create(
        {
            "data_root_dir": str(tmp_path),
            "data_mix": "magna_source_no_base_no_lift_interventions_v3",
        }
    )

    mixture = lerobot_datasets.get_vla_dataset(cfg)

    assert len(calls) == 1
    assert calls[0]["data_root_dir"] == tmp_path
    assert calls[0]["data_name"] == ""
    assert calls[0]["data_cfg"] is cfg
    assert mixture.dataset_mixture[0][1] == 1.0


@pytest.mark.parametrize(
    ("sources", "error"),
    [
        ({}, "must be a non-empty list"),
        ([], "must not be empty"),
        (
            [{"id": "source"}],
            "missing required keys",
        ),
        (
            [
                {
                    **_source("source", "/datasets/source"),
                    "weigth": 1.0,
                }
            ],
            "unsupported keys",
        ),
        (
            [_source("bad/id", "/datasets/source")],
            "must match",
        ),
        (
            [
                _source("source", "/datasets/one"),
                _source("source", "/datasets/two"),
            ],
            "Duplicate",
        ),
        (
            [_source("source", "relative/source")],
            "absolute path",
        ),
        (
            [_source("source", "/datasets/source", weight=0.0)],
            "greater than zero",
        ),
        (
            [_source("source", "/datasets/source", weight=float("nan"))],
            "greater than zero",
        ),
        (
            [_source("source", "/datasets/source", weight=True)],
            "greater than zero",
        ),
        (
            [_source("source", "/datasets/source", primary=1)],
            "exact boolean",
        ),
        (
            [
                {
                    **_source("source", "/datasets/source"),
                    "robot_type": "not_registered",
                }
            ],
            "registered robot type",
        ),
        (
            [
                {
                    **_source("source", "/datasets/source"),
                    "lerobot_version": "latest",
                }
            ],
            "must be one of",
        ),
        (
            [
                _source(
                    "source",
                    "/datasets/source",
                    config_overrides=["not", "a", "mapping"],
                )
            ],
            "config_overrides must be a mapping",
        ),
    ],
)
def test_explicit_sources_fail_closed_on_malformed_config(sources, error):
    with pytest.raises(ValueError, match=error):
        lerobot_datasets.get_vla_dataset(_config(sources))


def test_source_specific_split_rejects_weight_balancing(tmp_path, monkeypatch):
    _patch_dataset_constructors(monkeypatch)
    cfg = _config(
        [
            _source(
                "source",
                tmp_path / "source",
                config_overrides={"episode_split_manifest": "/splits/source.json"},
            )
        ]
    )

    with pytest.raises(ValueError, match="balance_dataset_weights=true"):
        lerobot_datasets.get_vla_dataset(
            cfg,
            balance_dataset_weights=True,
        )
