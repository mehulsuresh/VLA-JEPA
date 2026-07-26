from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader import (
    _CANONICAL_VIDEO_LOCAL_SAMPLER_VERSION,
    _CanonicalVideoLocalExhaustiveSampler,
    _resolve_canonical_exhaustive_sampler,
    _resolve_epoch_loader_contract,
    _validate_exhaustive_dataloader,
    _validate_exhaustive_dataset_schedule,
    build_dataloader,
)
from starVLA.dataloader.canonical_subset_dataset import (
    CanonicalSubsetVLADataset,
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


class _CanonicalRangeDataset(_ExactDataset):
    def __init__(
        self,
        *,
        block_lengths: tuple[int, ...] = (256, 256, 256, 256),
        seed: int = 23,
        encoding: str = "episode_ranges_v1",
    ) -> None:
        self.block_lengths = tuple(int(value) for value in block_lengths)
        super().__init__(size=sum(self.block_lengths))
        self.seed = int(seed)
        self.epoch_sampling_algorithm_version = (
            "all_sources_exhaustive_frozen_view_affine_v1"
        )
        self.frozen_train_view = SimpleNamespace(encoding=encoding)
        cumulative = np.concatenate(
            (
                np.zeros((1,), dtype=np.uint64),
                np.cumsum(self.block_lengths, dtype=np.uint64),
            )
        )
        self._offsets = np.zeros(
            (cumulative.size, 2), dtype=np.uint64
        )
        self._offsets[:, 1] = cumulative

    @property
    def current_epoch(self) -> int:
        return int(self.epoch)

    def _open_frozen_view_readers(self):
        return self._offsets, None

    def __getitem__(self, index: int) -> int:
        length = len(self)
        seed_payload = (
            f"{self.epoch_sampling_strategy}|{self.current_epoch}|"
            f"{self.seed}|{length}"
        ).encode("utf-8")
        permutation_seed = int.from_bytes(
            hashlib.sha256(seed_payload).digest()[:16],
            byteorder="big",
            signed=False,
        )
        multiplier, offset = (
            CanonicalSubsetVLADataset._affine_permutation_parameters(
                length, permutation_seed
            )
        )
        return int((multiplier * int(index) + offset) % length)


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


def test_canonical_video_local_sampler_is_exact_and_episode_contiguous() -> None:
    dataset = _CanonicalRangeDataset()
    sampler = _resolve_canonical_exhaustive_sampler(
        {"exhaustive_window_order": "video_local_blocks"},
        dataset,
        exhaustive_training=True,
        is_eval=False,
    )
    assert isinstance(sampler, _CanonicalVideoLocalExhaustiveSampler)
    loader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
    )
    _validate_exhaustive_dataloader(dataset, loader)

    epoch_zero = [
        int(value)
        for batch in loader
        for value in batch.tolist()
    ]
    sampler.set_epoch(1)
    epoch_one = [
        int(value)
        for batch in loader
        for value in batch.tolist()
    ]

    assert dataset.epoch_sampling_algorithm_version == (
        _CANONICAL_VIDEO_LOCAL_SAMPLER_VERSION
    )
    assert sorted(epoch_zero) == list(range(len(dataset)))
    assert sorted(epoch_one) == list(range(len(dataset)))
    assert epoch_zero != epoch_one
    for values in (epoch_zero, epoch_one):
        for start in range(0, len(values), 16):
            batch = values[start : start + 16]
            assert len({value // 256 for value in batch}) == 1


def test_canonical_video_local_sampler_requires_compact_range_view() -> None:
    dataset = _CanonicalRangeDataset(encoding="expanded_rows_v1")
    with pytest.raises(ValueError, match="episode_ranges_v1"):
        _resolve_canonical_exhaustive_sampler(
            {"exhaustive_window_order": "video_local_blocks"},
            dataset,
            exhaustive_training=True,
            is_eval=False,
        )


def test_canonical_video_local_sampler_requires_exhaustive_strategy() -> None:
    dataset = _CanonicalRangeDataset()
    with pytest.raises(ValueError, match="all_sources_exhaustive"):
        _resolve_canonical_exhaustive_sampler(
            {"exhaustive_window_order": "video_local_blocks"},
            dataset,
            exhaustive_training=False,
            is_eval=False,
        )


def test_canonical_video_local_sampler_handles_single_row_epoch() -> None:
    dataset = _CanonicalRangeDataset(block_lengths=(1,))
    sampler = _resolve_canonical_exhaustive_sampler(
        {"exhaustive_window_order": "video_local_blocks"},
        dataset,
        exhaustive_training=True,
        is_eval=False,
    )

    assert list(iter(sampler)) == [0]


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


def test_build_canonical_dataloader_installs_video_local_sampler(
    monkeypatch,
) -> None:
    module = __import__(
        "starVLA.dataloader.canonical_subset_dataset",
        fromlist=["get_vla_dataset"],
    )
    dataset = _CanonicalRangeDataset(
        block_lengths=(8, 8, 8, 8)
    )
    monkeypatch.setattr(
        module,
        "get_vla_dataset",
        lambda **_kwargs: dataset,
    )
    config = _config("canonical_subset_vla")
    config.datasets.vla_data.exhaustive_window_order = (
        "video_local_blocks"
    )

    loader = build_dataloader(
        config, dataset_py="canonical_subset_vla"
    )

    assert isinstance(
        loader.sampler, _CanonicalVideoLocalExhaustiveSampler
    )
    values = [
        int(value)
        for batch in loader
        for value in batch
    ]
    assert sorted(values) == list(range(len(dataset)))
    assert all(
        len({value // 8 for value in values[start : start + 4]}) == 1
        for start in range(0, len(values), 4)
    )
