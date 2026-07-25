from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderShard
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from starVLA.training.train_starvla import VLATrainer


class _RecordingExhaustiveDataset(Dataset):
    epoch_sampling_strategy = "primary_exhaustive"
    epoch_sampling_algorithm_version = "test_primary_exhaustive_v1"

    def __init__(self, length: int = 8):
        self.length = int(length)
        self.seed = 17
        self.epoch = 0
        self.dataset_lengths = [self.length]
        self._raw_dataset_sampling_weights = [1.0]
        self.primary_dataset_indices = [True]
        self.epoch_dataset_counts = [self.length]
        self.calls: list[tuple[int, int]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        self.calls.append((self.epoch, int(index)))
        return torch.tensor([self.epoch, int(index)], dtype=torch.int64)


def _trainer(
    *,
    gradient_accumulation_steps: int = 1,
    length: int = 8,
    epoch_sampling_strategy: str = "primary_exhaustive",
):
    dataset = _RecordingExhaustiveDataset(length=length)
    dataset.epoch_sampling_strategy = epoch_sampling_strategy
    dataset.epoch_sampling_algorithm_version = (
        f"test_{epoch_sampling_strategy}_v1"
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "seed": 17,
            "datasets": {
                "vla_data": {
                    "per_device_batch_size": 2,
                    "epoch_sampling_strategy": epoch_sampling_strategy,
                    "gpu_video_decode_on_rank": False,
                    "gpu_video_decode_async_prefetch": False,
                }
            },
            "trainer": {
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
        }
    )
    trainer.accelerator = SimpleNamespace(
        num_processes=1,
        is_main_process=False,
    )
    trainer.vla_train_dataloader = loader
    trainer.completed_steps = 0
    trainer.vla_epoch_count = 0
    trainer.vla_batches_consumed_in_epoch = 0
    trainer.vla_total_batches_consumed = 0
    trainer._restored_data_stream_state = None
    trainer._resume_vla_dataloader = None
    trainer._data_iterators_initialized = False
    trainer._rank_video_prefetcher = None
    trainer._last_prefetch_timing = None
    trainer._prefetch_model = None
    return trainer, dataset


def _resume_state(trainer, *, epoch: int, offset: int, total: int) -> dict:
    return {
        "schema_version": 1,
        "logical_epoch": epoch,
        "local_batch_offset_in_epoch": offset,
        "total_local_batches_consumed": total,
        "local_batches_per_epoch": len(trainer.vla_train_dataloader),
        "optimizer_steps_completed": trainer.completed_steps,
        "gradient_accumulation_steps": int(
            trainer.config.trainer.gradient_accumulation_steps
        ),
        "epoch_schedule_sha256": trainer._data_stream_schedule_sha256(),
        "checkpoint_boundary": "completed_optimizer_step",
    }


def test_mid_epoch_resume_skips_batches_without_decoding_them():
    trainer, dataset = _trainer()
    trainer.completed_steps = 6
    trainer._restored_data_stream_state = _resume_state(
        trainer,
        epoch=1,
        offset=2,
        total=6,
    )

    trainer._create_data_iterators()
    first = trainer._get_next_batch()
    second = trainer._get_next_batch()

    assert first.tolist() == [[1, 4], [1, 5]]
    assert second.tolist() == [[1, 6], [1, 7]]
    assert dataset.calls == [(1, 4), (1, 5), (1, 6), (1, 7)]
    assert trainer.vla_epoch_count == 1
    assert trainer.vla_batches_consumed_in_epoch == 4
    assert trainer.vla_total_batches_consumed == 8

    next_epoch = trainer._get_next_batch()
    assert next_epoch.tolist() == [[2, 0], [2, 1]]
    assert dataset.calls[-2:] == [(2, 0), (2, 1)]
    assert trainer.vla_epoch_count == 2
    assert trainer.vla_batches_consumed_in_epoch == 1
    assert trainer.vla_total_batches_consumed == 9


def test_all_sources_exhaustive_uses_exact_resume_cursor_path():
    trainer, dataset = _trainer(epoch_sampling_strategy="all_sources_exhaustive")
    trainer.completed_steps = 6
    trainer._restored_data_stream_state = _resume_state(
        trainer,
        epoch=1,
        offset=2,
        total=6,
    )

    assert trainer._exact_exhaustive_data_stream_enabled()
    trainer._create_data_iterators()
    assert trainer._get_next_batch().tolist() == [[1, 4], [1, 5]]
    assert dataset.calls == [(1, 4), (1, 5)]


def test_end_of_epoch_resume_canonicalizes_to_next_epoch():
    trainer, dataset = _trainer()
    trainer.completed_steps = 4
    trainer._restored_data_stream_state = _resume_state(
        trainer,
        epoch=0,
        offset=4,
        total=4,
    )

    trainer._create_data_iterators()
    batch = trainer._get_next_batch()

    assert batch.tolist() == [[1, 0], [1, 1]]
    assert dataset.calls == [(1, 0), (1, 1)]
    assert trainer.vla_epoch_count == 1
    assert trainer.vla_batches_consumed_in_epoch == 1
    assert trainer.vla_total_batches_consumed == 5


def test_mid_epoch_resume_through_real_accelerate_dataloader_shard():
    trainer, dataset = _trainer(length=10)
    accelerator = Accelerator(cpu=True)
    trainer.accelerator = accelerator
    trainer.vla_train_dataloader = accelerator.prepare(
        trainer.vla_train_dataloader
    )
    assert isinstance(trainer.vla_train_dataloader, DataLoaderShard)
    trainer.completed_steps = 12
    trainer._restored_data_stream_state = _resume_state(
        trainer,
        epoch=2,
        offset=2,
        total=12,
    )

    trainer._create_data_iterators()
    resumed_tail = [
        trainer._get_next_batch().tolist()
        for _ in range(3)
    ]

    assert resumed_tail == [
        [[2, 4], [2, 5]],
        [[2, 6], [2, 7]],
        [[2, 8], [2, 9]],
    ]
    assert dataset.calls == [
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (2, 9),
    ]


def test_checkpoint_cursor_tracks_microbatches_not_only_optimizer_steps():
    trainer, _ = _trainer(gradient_accumulation_steps=2)
    trainer.completed_steps = 3
    trainer.vla_epoch_count = 1
    trainer.vla_batches_consumed_in_epoch = 2
    trainer.vla_total_batches_consumed = 6
    trainer._data_iterators_initialized = True

    payload = trainer._data_stream_state_payload()

    assert payload["optimizer_steps_completed"] == 3
    assert payload["gradient_accumulation_steps"] == 2
    assert payload["total_local_batches_consumed"] == 6
    assert payload["logical_epoch"] == 1
    assert payload["local_batch_offset_in_epoch"] == 2


def test_nondivisible_accumulation_accepts_epoch_end_boundary():
    trainer, _ = _trainer(gradient_accumulation_steps=3, length=10)
    assert len(trainer.vla_train_dataloader) == 5
    trainer.completed_steps = 2
    trainer.vla_epoch_count = 0
    trainer.vla_batches_consumed_in_epoch = 5
    trainer.vla_total_batches_consumed = 5
    trainer._data_iterators_initialized = True

    payload = trainer._data_stream_state_payload()

    assert payload["local_batch_offset_in_epoch"] == 5
    assert payload["optimizer_steps_completed"] == 2


def test_impossible_partial_accumulation_cursor_is_rejected():
    trainer, _ = _trainer(gradient_accumulation_steps=3, length=10)
    trainer.completed_steps = 1
    payload = _resume_state(trainer, epoch=0, offset=1, total=1)

    with pytest.raises(RuntimeError, match="optimizer boundary"):
        trainer._validate_restored_data_stream_state(
            payload,
            source=SimpleNamespace(),
        )


def test_checkpoint_rejects_cursor_while_gradients_are_unsynchronized():
    trainer, _ = _trainer()
    trainer.completed_steps = 1
    trainer.vla_epoch_count = 0
    trainer.vla_batches_consumed_in_epoch = 1
    trainer.vla_total_batches_consumed = 1
    trainer._data_iterators_initialized = True
    trainer.accelerator.sync_gradients = False

    with pytest.raises(RuntimeError, match="middle of gradient accumulation"):
        trainer._data_stream_state_payload()


def test_resume_rejects_schedule_drift():
    trainer, _ = _trainer()
    trainer.completed_steps = 2
    payload = _resume_state(trainer, epoch=0, offset=2, total=2)
    payload["epoch_schedule_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="schedule changed"):
        trainer._validate_restored_data_stream_state(
            payload,
            source=SimpleNamespace(__str__=lambda self: "trainer_state.json"),
        )


def test_exact_mode_rejects_async_producer_that_can_run_ahead():
    trainer, _ = _trainer()
    trainer.config.datasets.vla_data.gpu_video_decode_on_rank = True
    trainer.config.datasets.vla_data.gpu_video_decode_async_prefetch = True

    with pytest.raises(RuntimeError, match="producer can run ahead"):
        trainer._create_data_iterators()
