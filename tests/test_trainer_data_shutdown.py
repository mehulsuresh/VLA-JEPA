from types import SimpleNamespace

from starVLA.training.train_starvla import (
    VLATrainer,
    _shutdown_dataloader_workers,
)


class _Iterator:
    def __init__(self):
        self.shutdown_calls = 0

    def _shutdown_workers(self):
        self.shutdown_calls += 1


def test_shutdown_dataloader_workers_releases_iterator_reference():
    iterator = _Iterator()
    dataloader = SimpleNamespace(_iterator=iterator)

    _shutdown_dataloader_workers(dataloader)

    assert iterator.shutdown_calls == 1
    assert dataloader._iterator is None


def test_trainer_data_runtime_shutdown_is_idempotent():
    trainer = object.__new__(VLATrainer)
    trainer._data_runtime_shutdown = False
    trainer._rank_video_prefetcher = None
    trainer.vla_iter = _Iterator()
    trainer.vla_eval_iter = None
    train_iterator = _Iterator()
    trainer.vla_train_dataloader = SimpleNamespace(_iterator=train_iterator)

    trainer._shutdown_data_runtime()
    trainer._shutdown_data_runtime()

    assert trainer.vla_iter is None
    assert trainer.vla_eval_iter is None
    assert train_iterator.shutdown_calls == 1
    assert trainer.vla_train_dataloader._iterator is None
