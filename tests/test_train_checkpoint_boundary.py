from collections import deque

import pytest
from omegaconf import OmegaConf

from starVLA.training import train_starvla
from starVLA.training.train_starvla import VLATrainer


class _ProgressBar:
    def __init__(self, *args, **kwargs):
        self.updates = 0

    def update(self, value):
        self.updates += int(value)

    def set_postfix(self, *args, **kwargs):
        return None


class _Optimizer:
    def zero_grad(self, **kwargs):
        return None


class _Accelerator:
    is_main_process = False
    is_local_main_process = False
    trackers = []

    def __init__(self):
        self.sync_gradients = False


def _loop_trainer(
    *,
    sync_sequence,
    max_train_steps=1,
    eval_interval=1,
    save_interval=1,
    save_best_only=False,
    eval_before_train=False,
    eval_error=None,
    force_requests=(),
):
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "max_train_steps": max_train_steps,
                "eval_interval": eval_interval,
                "save_interval": save_interval,
                "save_best_only": save_best_only,
                "eval_before_train": eval_before_train,
                "allow_training_stream_eval": False,
                "logging_frequency": 1000,
            },
            "datasets": {"vla_data": {"runtime_timing_logging": False}},
        }
    )
    trainer.accelerator = _Accelerator()
    trainer.optimizer = _Optimizer()
    trainer.completed_steps = 0
    trainer.vla_eval_dataloader = object()
    trainer.total_batch_size = 1
    trainer.progress_eta_warmup_steps = 10_000
    trainer._recent_wall_step_times = deque(maxlen=4)
    trainer._warned_training_stream_eval_disabled = False
    trainer.force_checkpoint_path = "/tmp/unused-force-checkpoint"

    events = []
    remaining_sync = deque(bool(value) for value in sync_sequence)
    remaining_force = deque(bool(value) for value in force_requests)

    trainer._log_training_config = lambda: None
    trainer._create_data_iterators = lambda: None
    trainer._get_next_batch = lambda: object()

    def train_step(_batch):
        if not remaining_sync:
            raise AssertionError("test exhausted its sync_gradients sequence")
        trainer.accelerator.sync_gradients = remaining_sync.popleft()
        events.append(
            ("train", trainer.completed_steps, trainer.accelerator.sync_gradients)
        )
        return {}

    def evaluate(metrics):
        events.append(("eval", trainer.completed_steps))
        if eval_error is not None:
            raise eval_error
        metrics["heldout_eval_normalized_arm_mae_h20"] = 0.25
        return metrics

    def save_checkpoint():
        events.append(("save", trainer.completed_steps))

    def should_save(_metrics):
        events.append(("best_decision", trainer.completed_steps))
        return True

    def force_requested():
        events.append(("force_check", trainer.completed_steps))
        return remaining_force.popleft() if remaining_force else False

    def clear_force():
        events.append(("force_clear", trainer.completed_steps))

    trainer._train_step = train_step
    trainer.eval_heldout_action_model = evaluate
    trainer._save_checkpoint = save_checkpoint
    trainer._should_save_checkpoint = should_save
    trainer._force_checkpoint_requested = force_requested
    trainer._clear_force_checkpoint_request = clear_force
    trainer._detailed_timing_frequency = lambda: 10_000
    trainer._log_metrics = lambda _metrics: None
    trainer._finalize_training = lambda: events.append(
        ("finalize", trainer.completed_steps)
    )
    return trainer, events


@pytest.fixture(autouse=True)
def _disable_real_progress_bar(monkeypatch):
    monkeypatch.setattr(train_starvla, "tqdm", _ProgressBar)


def test_unconditional_terminal_checkpoint_survives_fail_closed_eval():
    trainer, events = _loop_trainer(
        sync_sequence=[True],
        eval_error=RuntimeError("synthetic heldout failure"),
    )

    with pytest.raises(RuntimeError, match="synthetic heldout failure"):
        trainer.train()

    assert trainer.completed_steps == 1
    assert events == [
        ("train", 0, True),
        ("save", 1),
        ("eval", 1),
    ]


def test_unconditional_coincident_boundary_saves_once_and_satisfies_force():
    trainer, events = _loop_trainer(
        sync_sequence=[True],
        force_requests=[True],
    )

    trainer.train()

    assert [event for event in events if event[0] == "save"] == [("save", 1)]
    assert events.index(("save", 1)) < events.index(("eval", 1))
    assert ("force_clear", 1) in events
    assert events[-1] == ("finalize", 1)


def test_save_best_only_keeps_eval_and_metric_decision_before_save():
    trainer, events = _loop_trainer(
        sync_sequence=[True],
        save_best_only=True,
    )

    trainer.train()

    assert events.index(("eval", 1)) < events.index(("best_decision", 1))
    assert events.index(("best_decision", 1)) < events.index(("save", 1))
    assert [event for event in events if event[0] == "save"] == [("save", 1)]


def test_save_best_only_eval_failure_does_not_create_unselected_checkpoint():
    trainer, events = _loop_trainer(
        sync_sequence=[True],
        save_best_only=True,
        eval_error=RuntimeError("best metric unavailable"),
    )

    with pytest.raises(RuntimeError, match="best metric unavailable"):
        trainer.train()

    assert events == [
        ("train", 0, True),
        ("eval", 1),
    ]


def test_periodic_boundaries_do_not_repeat_on_non_sync_microbatch():
    trainer, events = _loop_trainer(
        sync_sequence=[False, True, False, True],
        max_train_steps=2,
    )

    trainer.train()

    assert [event for event in events if event[0] == "save"] == [
        ("save", 1),
        ("save", 2),
    ]
    assert [event for event in events if event[0] == "eval"] == [
        ("eval", 1),
        ("eval", 2),
    ]
    assert [event for event in events if event[0] == "force_check"] == [
        ("force_check", 1),
        ("force_check", 2),
    ]


def test_force_checkpoint_waits_for_completed_optimizer_step_then_saves_once():
    trainer, events = _loop_trainer(
        sync_sequence=[False, True],
        eval_interval=99,
        save_interval=99,
        force_requests=[True],
    )

    trainer.train()

    assert [event for event in events if event[0] == "force_check"] == [
        ("force_check", 1)
    ]
    assert [event for event in events if event[0] == "save"] == [("save", 1)]
    assert [event for event in events if event[0] == "force_clear"] == [
        ("force_clear", 1)
    ]


def test_baseline_eval_failure_happens_before_training_or_checkpointing():
    trainer, events = _loop_trainer(
        sync_sequence=[True],
        eval_before_train=True,
        eval_error=RuntimeError("baseline failed"),
    )

    with pytest.raises(RuntimeError, match="baseline failed"):
        trainer.train()

    assert trainer.completed_steps == 0
    assert events == [("eval", 0)]
