from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader import (
    _resolve_epoch_loader_contract,
    _validate_exhaustive_dataloader,
    _validate_exhaustive_dataset_schedule,
    build_dataloader,
)


@dataclass
class _ExactDataset(Dataset):
    size: int = 11
    epoch_sampling_strategy: str = "all_sources_exhaustive"
    epoch_sampling_algorithm_version: str = (
        "all_sources_exhaustive_affine_v1"
    )
    fail_on_sample_error: bool = True

    def __post_init__(self) -> None:
        self.dataset_lengths = np.asarray([self.size], dtype=np.int64)
        self.epoch_dataset_counts = np.asarray(
            [self.size], dtype=np.int64
        )
        self.primary_dataset_indices = np.asarray([True], dtype=np.bool_)
        self.epoch = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return int(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


@pytest.mark.parametrize(
    "dataset_py", ("lerobot_datasets", "canonical_subset_vla")
)
def test_exhaustive_contract_owns_shuffle_and_partial_batch(
    dataset_py: str,
) -> None:
    exhaustive, shuffle, drop_last = _resolve_epoch_loader_contract(
        {
            "epoch_sampling_strategy": "all_sources_exhaustive",
            # Stale legacy defaults must not weaken the exact schedule.
            "shuffle": True,
            "drop_last": True,
        },
        dataset_py=dataset_py,
        is_eval=False,
    )

    assert exhaustive is True
    assert shuffle is False
    assert drop_last is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_shards", 1),
        ("max_shards_per_dataset", 1),
        ("max_windows", 8),
        ("max_windows_per_dataset", 8),
        ("shuffle_shards", True),
        ("sample_stride", 2),
    ),
)
def test_exhaustive_contract_rejects_population_shortcuts(
    field: str, value
) -> None:
    config = {
        "epoch_sampling_strategy": "all_sources_exhaustive",
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        _resolve_epoch_loader_contract(
            config,
            dataset_py="canonical_subset_vla",
            is_eval=False,
        )


def test_exhaustive_schedule_and_loader_surface_are_consistent() -> None:
    dataset = _ExactDataset()
    _validate_exhaustive_dataset_schedule(dataset)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )
    _validate_exhaustive_dataloader(dataset, loader)
    assert [value for batch in loader for value in batch.tolist()] == list(
        range(len(dataset))
    )

    shuffled = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=False,
    )
    with pytest.raises(RuntimeError, match="shuffle=false"):
        _validate_exhaustive_dataloader(dataset, shuffled)


def _config(dataset_py: str):
    return OmegaConf.create(
        {
            "datasets": {
                "vla_data": {
                    "dataset_py": dataset_py,
                    "per_device_batch_size": 4,
                    "num_workers": 0,
                    "pin_memory": False,
                    "epoch_sampling_strategy": "all_sources_exhaustive",
                    "fail_on_sample_error": True,
                    "shuffle": True,
                    "drop_last": True,
                }
            },
            "framework": {
                "action_model": {"action_horizon": 50},
                "vj2_model": {"num_frames": 8},
            },
        }
    )


@pytest.mark.parametrize(
    ("dataset_py", "module_name"),
    (
        ("lerobot_datasets", "starVLA.dataloader.lerobot_datasets"),
        (
            "canonical_subset_vla",
            "starVLA.dataloader.canonical_subset_dataset",
        ),
    ),
)
def test_build_dataloader_enforces_exact_contract_for_both_paths(
    monkeypatch,
    dataset_py: str,
    module_name: str,
) -> None:
    module = __import__(module_name, fromlist=["get_vla_dataset"])
    dataset = _ExactDataset()
    monkeypatch.setattr(
        module,
        "get_vla_dataset",
        lambda **_kwargs: dataset,
    )

    loader = build_dataloader(_config(dataset_py), dataset_py=dataset_py)

    assert loader.dataset is dataset
    assert loader.drop_last is False
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    assert len(loader) == 3
