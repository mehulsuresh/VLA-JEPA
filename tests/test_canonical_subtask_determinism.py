from __future__ import annotations

from types import SimpleNamespace

import torch

from starVLA.dataloader.canonical_subset_dataset import (
    CanonicalSubsetVLADataset,
)


def _dataset(*, epoch: int = 0) -> CanonicalSubsetVLADataset:
    dataset = CanonicalSubsetVLADataset.__new__(
        CanonicalSubsetVLADataset
    )
    dataset.mode = "train"
    dataset.seed = 42
    dataset.epoch = epoch
    dataset._shared_epoch = torch.tensor(
        epoch, dtype=torch.int64
    ).share_memory_()
    dataset.data_cfg = {
        "append_subtask_to_prompt": True,
        "subtask_prompt_append_probability": 0.7,
        "subtask_prompt_separator": " | ",
        "subtask_prompt_ignored_labels": ["__unlabeled__"],
    }
    dataset.append_subtask_to_prompt = True
    dataset.subtask_prompt_source_column = "subtask_index"
    dataset.subtask_prompt_label_column = "local_subtask_text"
    return dataset


def _context(*, base_index: int = 17) -> dict:
    return {
        "shard": SimpleNamespace(
            dataset_id="org/realsource",
            sid="source-shard",
            revision="main",
            data_relative_path="data/chunk-000/file-000.parquet",
        ),
        "episode": SimpleNamespace(episode_index=11),
        "window": SimpleNamespace(base_index=base_index),
        "source_base_index": base_index,
        "frozen_view_sample_id": f"sample-{base_index}",
    }


def test_canonical_subtask_gate_is_repeatable_for_epoch_and_source():
    dataset = _dataset(epoch=3)
    context = _context()

    key = dataset._subtask_prompt_deterministic_key(context)
    first = dataset._language_with_subtask(
        "move the chain",
        "pick from left bin",
        deterministic_key=key,
    )
    for _ in range(20):
        assert dataset._subtask_prompt_deterministic_key(context) == key
        assert dataset._language_with_subtask(
            "move the chain",
            "pick from left bin",
            deterministic_key=key,
        ) == first


def test_canonical_subtask_gate_is_exact_resume_reproducible():
    context = _context(base_index=29)
    original = _dataset(epoch=7)
    resumed = _dataset(epoch=7)

    original_key = original._subtask_prompt_deterministic_key(context)
    resumed_key = resumed._subtask_prompt_deterministic_key(context)

    assert resumed_key == original_key
    assert resumed._language_with_subtask(
        "move the chain",
        "place in jig",
        deterministic_key=resumed_key,
    ) == original._language_with_subtask(
        "move the chain",
        "place in jig",
        deterministic_key=original_key,
    )


def test_shared_epoch_updates_persistent_worker_prompt_identity():
    dataset = _dataset(epoch=0)
    context = _context()

    epoch_zero_key = dataset._subtask_prompt_deterministic_key(context)
    dataset.set_epoch(1)
    epoch_one_key = dataset._subtask_prompt_deterministic_key(context)

    assert epoch_zero_key["epoch"] == 0
    assert epoch_one_key["epoch"] == 1
    assert epoch_zero_key != epoch_one_key


def test_gate_uses_source_identity_not_loader_slot_or_epoch_permutation():
    dataset = _dataset(epoch=2)
    left = dataset._subtask_prompt_deterministic_key(
        _context(base_index=10)
    )
    right = dataset._subtask_prompt_deterministic_key(
        _context(base_index=11)
    )

    assert left != right
    assert "loader_index" not in left
    assert "original_index" not in left


def test_seventy_percent_gate_is_statistically_well_formed_and_repeatable():
    dataset = _dataset(epoch=5)
    outcomes = []
    for base_index in range(2000):
        key = dataset._subtask_prompt_deterministic_key(
            _context(base_index=base_index)
        )
        outcomes.append(
            dataset._language_with_subtask(
                "move the chain",
                "pick from left bin",
                deterministic_key=key,
            ).endswith("pick from left bin")
        )

    assert outcomes == [
        dataset._language_with_subtask(
            "move the chain",
            "pick from left bin",
            deterministic_key=dataset._subtask_prompt_deterministic_key(
                _context(base_index=base_index)
            ),
        ).endswith("pick from left bin")
        for base_index in range(2000)
    ]
    observed = sum(outcomes) / len(outcomes)
    assert 0.67 <= observed <= 0.73
