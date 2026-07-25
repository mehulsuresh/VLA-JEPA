from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


@dataclass
class _IndexDataset:
    dataset_name: str
    size: int
    epoch: int = 0

    def __post_init__(self) -> None:
        self.all_steps = [(0, index) for index in range(self.size)]
        self.trajectory_lengths = np.asarray([self.size], dtype=np.int64)

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __str__(self) -> str:
        return self.dataset_name


class _SamplingOnlyMixture(LeRobotMixtureDataset):
    def update_metadata(self, metadata_config) -> None:
        return None

    def __getitem__(self, index: int):
        dataset, _, base_index = self.sample_step(index)
        return dataset.dataset_name, int(base_index), self.current_epoch


def _mixture(*entries: tuple[_IndexDataset, float]) -> _SamplingOnlyMixture:
    return _SamplingOnlyMixture(
        entries,
        mode="train",
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=917,
        metadata_config={
            "epoch_sampling_strategy": "primary_exhaustive",
            "fail_on_sample_error": True,
        },
    )


def _all_sources_mixture(
    *entries: tuple[_IndexDataset, float],
    primary_dataset_flags: list[bool] | None = None,
) -> _SamplingOnlyMixture:
    return _SamplingOnlyMixture(
        entries,
        mode="train",
        primary_dataset_flags=primary_dataset_flags,
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        seed=917,
        metadata_config={
            "epoch_sampling_strategy": "all_sources_exhaustive",
            "fail_on_sample_error": True,
        },
    )


def _read_loader(loader: DataLoader) -> tuple[list[str], list[int], list[int]]:
    names: list[str] = []
    indices: list[int] = []
    epochs: list[int] = []
    for batch_names, batch_indices, batch_epochs in loader:
        names.extend(batch_names)
        indices.extend(int(value) for value in batch_indices.tolist())
        epochs.extend(int(value) for value in batch_epochs.tolist())
    return names, indices, epochs


def test_single_primary_dataset_is_exhaustive_without_replacement() -> None:
    mixture = _mixture((_IndexDataset("primary", 101), 1.0))

    epoch_zero = [mixture.sample_step(index)[2] for index in range(len(mixture))]
    mixture.set_epoch(1)
    epoch_one = [mixture.sample_step(index)[2] for index in range(len(mixture))]

    assert len(mixture) == 101
    assert sorted(epoch_zero) == list(range(101))
    assert sorted(epoch_one) == list(range(101))
    assert epoch_zero != epoch_one


def test_replay_ratio_does_not_sacrifice_primary_coverage() -> None:
    mixture = _mixture(
        (_IndexDataset("new_data", 100), 1.0),
        (_IndexDataset("replay", 1000), 0.25),
    )

    sampled = [
        (dataset.dataset_name, base_index)
        for dataset, _, base_index in (
            mixture.sample_step(index) for index in range(len(mixture))
        )
    ]
    primary = [index for name, index in sampled if name == "new_data"]
    replay = [index for name, index in sampled if name == "replay"]

    np.testing.assert_array_equal(mixture.epoch_dataset_counts, [100, 25])
    assert len(mixture) == 125
    assert sorted(primary) == list(range(100))
    assert len(replay) == 25
    assert len(set(replay)) == 25


def test_all_sources_exhaustive_ignores_weights_and_primary_flags() -> None:
    mixture = _all_sources_mixture(
        (_IndexDataset("large_weight", 11), 8.0),
        (_IndexDataset("tiny_weight", 7), 0.001),
        (_IndexDataset("nonprimary", 5), 0.25),
        primary_dataset_flags=[True, True, False],
    )

    sampled = [
        (dataset.dataset_name, base_index)
        for dataset, _, base_index in (
            mixture.sample_step(index) for index in range(len(mixture))
        )
    ]

    np.testing.assert_array_equal(mixture.epoch_dataset_counts, [11, 7, 5])
    assert len(mixture) == 23
    assert mixture.epoch_sampling_algorithm_version == (
        "all_sources_exhaustive_affine_v1"
    )
    for name, expected_size in (
        ("large_weight", 11),
        ("tiny_weight", 7),
        ("nonprimary", 5),
    ):
        indices = [index for sampled_name, index in sampled if sampled_name == name]
        assert sorted(indices) == list(range(expected_size))


def test_all_sources_exhaustive_repermutes_but_preserves_every_row() -> None:
    mixture = _all_sources_mixture(
        (_IndexDataset("first", 13), 1.0),
        (_IndexDataset("second", 17), 0.1),
        primary_dataset_flags=[True, False],
    )
    epoch_zero = [
        (dataset.dataset_name, base_index)
        for dataset, _, base_index in (
            mixture.sample_step(index) for index in range(len(mixture))
        )
    ]
    mixture.set_epoch(1)
    epoch_one = [
        (dataset.dataset_name, base_index)
        for dataset, _, base_index in (
            mixture.sample_step(index) for index in range(len(mixture))
        )
    ]

    assert sorted(epoch_zero) == sorted(epoch_one)
    assert len(set(epoch_zero)) == len(mixture) == 30
    assert epoch_zero != epoch_one


def test_persistent_spawn_worker_observes_new_epoch_and_keeps_full_coverage() -> None:
    mixture = _mixture((_IndexDataset("primary", 101), 1.0))
    loader = DataLoader(
        mixture,
        batch_size=11,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="spawn",
        drop_last=False,
    )
    try:
        _, epoch_zero_indices, epoch_zero_values = _read_loader(loader)
        mixture.set_epoch(1)
        _, epoch_one_indices, epoch_one_values = _read_loader(loader)
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()

    assert set(epoch_zero_values) == {0}
    assert set(epoch_one_values) == {1}
    assert sorted(epoch_zero_indices) == list(range(101))
    assert sorted(epoch_one_indices) == list(range(101))
    assert epoch_zero_indices != epoch_one_indices


def test_trainer_epoch_propagation_reaches_wrapped_dataset() -> None:
    mixture = _mixture((_IndexDataset("primary", 7), 1.0))

    class _Sampler:
        def __init__(self) -> None:
            self.epoch = None

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

    class _LoaderWrapper:
        def __init__(self) -> None:
            self.dataset = mixture
            self.sampler = _Sampler()

    wrapper = _LoaderWrapper()
    TrainerUtils._set_dataloader_epoch(wrapper, 3)

    assert wrapper.sampler.epoch == 3
    assert mixture.current_epoch == 3
    assert mixture.datasets[0].epoch == 3


def test_exhaustive_strategy_rejects_size_balancing() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        "requires balance_dataset_weights=false",
    ):
        _SamplingOnlyMixture(
            [(_IndexDataset("primary", 5), 1.0)],
            mode="train",
            balance_dataset_weights=True,
            balance_trajectory_weights=False,
            metadata_config={"epoch_sampling_strategy": "primary_exhaustive"},
        )


def test_all_sources_exhaustive_rejects_size_balancing() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        "all_sources_exhaustive requires balance_dataset_weights=false",
    ):
        _SamplingOnlyMixture(
            [(_IndexDataset("source", 5), 0.01)],
            mode="train",
            balance_dataset_weights=True,
            balance_trajectory_weights=False,
            metadata_config={
                "epoch_sampling_strategy": "all_sources_exhaustive"
            },
        )
