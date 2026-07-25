import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.schema import LeRobotModalityMetadata


def _dataset(tmp_path: Path, *, version: str = "v3.0") -> LeRobotSingleDataset:
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._dataset_path = tmp_path
    dataset._dataset_name = "test_dataset"
    dataset._lerobot_version = version
    dataset.data_cfg = {}
    dataset.transforms = SimpleNamespace(transforms=[])
    return dataset


def _modality_metadata() -> LeRobotModalityMetadata:
    return LeRobotModalityMetadata.model_validate(
        {
            "state": {
                "source": {
                    "original_key": "source.observation.state",
                    "start": 0,
                    "end": 2,
                    "dtype": "float32",
                }
            },
            "action": {
                "source": {
                    "original_key": "source.action",
                    "start": 0,
                    "end": 2,
                    "dtype": "float32",
                }
            },
            "video": {},
        }
    )


def _stat(values, *, count=None, include_quantiles=True):
    payload = {
        "mean": values,
        "std": [1.0 for _ in values],
        "min": values,
        "max": values,
    }
    if include_quantiles:
        payload["q01"] = values
        payload["q99"] = values
    if count is not None:
        payload["count"] = [count]
    return payload


def _payload(marker: float, *, count=None, include_quantiles=True):
    values = [marker, marker + 1.0]
    return {
        "source.observation.state": _stat(
            values,
            count=count,
            include_quantiles=include_quantiles,
        ),
        "source.action": _stat(
            values,
            count=count,
            include_quantiles=include_quantiles,
        ),
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v3_auto_prefers_raw_statistics_with_matching_frame_count(tmp_path):
    dataset = _dataset(tmp_path)
    raw = _payload(10.0, count=123)
    stale_gr00t = _payload(-10.0)
    _write(tmp_path / "meta/stats.json", raw)
    _write(tmp_path / "meta/stats_gr00t.json", stale_gr00t)

    selected = dataset._load_lerobot_statistics(
        _modality_metadata(),
        {"total_frames": 123},
    )

    assert selected == raw
    assert dataset._statistics_source_path == tmp_path / "meta/stats.json"


def test_v3_rejects_wrong_count_and_unproven_gr00t_fallback(tmp_path):
    dataset = _dataset(tmp_path)
    _write(tmp_path / "meta/stats.json", _payload(10.0, count=122))
    _write(tmp_path / "meta/stats_gr00t.json", _payload(-10.0))

    with pytest.raises(ValueError) as exc_info:
        dataset._load_lerobot_statistics(
            _modality_metadata(),
            {"total_frames": 123},
        )

    message = str(exc_info.value)
    assert "count is 122; expected 123" in message
    assert "has no scalar count; expected 123" in message
    assert "Refusing to train with unverifiable normalization" in message


def test_explicit_gr00t_source_still_requires_matching_provenance(tmp_path):
    dataset = _dataset(tmp_path)
    dataset.data_cfg = {"lerobot_statistics_source": "gr00t"}
    _write(tmp_path / "meta/stats.json", _payload(10.0, count=123))
    _write(tmp_path / "meta/stats_gr00t.json", _payload(-10.0))

    with pytest.raises(ValueError, match="has no scalar count"):
        dataset._load_lerobot_statistics(
            _modality_metadata(),
            {"total_frames": 123},
        )


def test_v2_legacy_statistics_without_count_remain_compatible(tmp_path):
    dataset = _dataset(tmp_path, version="v2.0")
    legacy = _payload(3.0)
    _write(tmp_path / "meta/stats_gr00t.json", legacy)

    selected = dataset._load_lerobot_statistics(
        _modality_metadata(),
        {"total_frames": 123},
    )

    assert selected == legacy
    assert dataset._statistics_source_path == tmp_path / "meta/stats_gr00t.json"


def test_v3_synthesizes_schema_only_quantiles_for_minmax_pipeline(tmp_path):
    dataset = _dataset(tmp_path)
    raw = _payload(4.0, count=123, include_quantiles=False)
    raw_path = tmp_path / "meta/stats.json"
    _write(raw_path, raw)

    selected = dataset._load_lerobot_statistics(
        _modality_metadata(),
        {"total_frames": 123},
    )

    assert selected["source.action"]["count"] == [123]
    assert selected["source.action"]["q01"] == [4.0, 5.0]
    assert selected["source.action"]["q99"] == [4.0, 5.0]
    assert dataset._statistics_source_path == raw_path
    assert dataset._statistics_quantiles_synthesized is True


def test_v3_q99_pipeline_refuses_uncounted_quantile_companion(tmp_path):
    dataset = _dataset(tmp_path)
    dataset.transforms = SimpleNamespace(
        transforms=[SimpleNamespace(normalization_modes={"action.source": "q99"})]
    )
    raw_path = tmp_path / "meta/stats.json"
    gr00t_path = tmp_path / "meta/stats_gr00t.json"
    _write(raw_path, _payload(4.0, count=123, include_quantiles=False))
    _write(gr00t_path, _payload(4.0))

    with pytest.raises(ValueError) as exc_info:
        dataset._load_lerobot_statistics(
            _modality_metadata(),
            {"total_frames": 123},
        )
    message = str(exc_info.value)
    assert "Field required" in message
    assert "has no scalar count" in message


def test_v3_cannot_disable_frame_count_validation(tmp_path):
    dataset = _dataset(tmp_path)
    dataset.data_cfg = {"require_statistics_frame_count": False}
    _write(tmp_path / "meta/stats.json", _payload(4.0, count=123))

    with pytest.raises(ValueError, match="cannot disable"):
        dataset._load_lerobot_statistics(
            _modality_metadata(),
            {"total_frames": 123},
        )


@pytest.mark.parametrize("bad_total", [None, 0, -1, "not-an-integer"])
def test_v3_rejects_missing_or_invalid_info_frame_count(tmp_path, bad_total):
    dataset = _dataset(tmp_path)
    _write(tmp_path / "meta/stats.json", _payload(4.0, count=123))

    with pytest.raises(ValueError, match="total_frames"):
        dataset._load_lerobot_statistics(
            _modality_metadata(),
            {"total_frames": bad_total},
        )


def test_v3_rejects_short_or_nonfinite_used_statistics(tmp_path):
    dataset = _dataset(tmp_path)
    payload = _payload(4.0, count=123)
    payload["source.action"]["mean"] = [np.nan]
    _write(tmp_path / "meta/stats.json", payload)

    with pytest.raises(ValueError) as exc_info:
        dataset._load_lerobot_statistics(
            _modality_metadata(),
            {"total_frames": 123},
        )
    message = str(exc_info.value)
    assert "required width is at least 2" in message


def _write_v3_catalog(tmp_path: Path, *, episode_lengths, data_lengths) -> None:
    episodes_path = tmp_path / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": list(range(len(episode_lengths))),
            "length": episode_lengths,
        }
    ).to_parquet(episodes_path)
    data_dir = tmp_path / "data/chunk-000"
    data_dir.mkdir(parents=True)
    for index, length in enumerate(data_lengths):
        pd.DataFrame({"index": np.arange(length)}).to_parquet(
            data_dir / f"file-{index:03d}.parquet"
        )


def test_v3_frame_catalog_cross_checks_three_independent_counts(tmp_path):
    dataset = _dataset(tmp_path)
    _write_v3_catalog(tmp_path, episode_lengths=[2, 3], data_lengths=[4, 1])

    dataset._validate_lerobot_v3_frame_catalog({"total_frames": 5})

    assert dataset._dataset_catalog_counts == {
        "info_total_frames": 5,
        "episode_length_sum": 5,
        "parquet_row_sum": 5,
    }
    assert len(dataset._dataset_catalog_fingerprint) == 64


def test_v3_frame_catalog_rejects_episode_or_parquet_mismatch(tmp_path):
    dataset = _dataset(tmp_path)
    _write_v3_catalog(tmp_path, episode_lengths=[2, 3], data_lengths=[4])

    with pytest.raises(ValueError, match="frame catalog is inconsistent"):
        dataset._validate_lerobot_v3_frame_catalog({"total_frames": 5})
