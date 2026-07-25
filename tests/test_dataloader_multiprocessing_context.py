from __future__ import annotations

import pytest

from starVLA.dataloader import _resolve_worker_multiprocessing_context


def test_worker_context_is_not_required_without_workers():
    assert (
        _resolve_worker_multiprocessing_context(
            {"multiprocessing_context": object()},
            is_eval=False,
            num_workers=0,
        )
        is None
    )


def test_worker_context_uses_eval_override(monkeypatch):
    monkeypatch.setattr(
        "starVLA.dataloader.mp.get_all_start_methods",
        lambda: ["spawn", "forkserver", "fork"],
    )
    assert (
        _resolve_worker_multiprocessing_context(
            {
                "multiprocessing_context": "spawn",
                "eval_multiprocessing_context": "forkserver",
            },
            is_eval=True,
            num_workers=4,
        )
        == "forkserver"
    )


def test_worker_context_rejects_fork_after_model_initialization(monkeypatch):
    monkeypatch.setattr(
        "starVLA.dataloader.mp.get_all_start_methods",
        lambda: ["spawn", "forkserver", "fork"],
    )
    with pytest.raises(ValueError, match="fork.*forbidden"):
        _resolve_worker_multiprocessing_context(
            {"multiprocessing_context": "fork"},
            is_eval=False,
            num_workers=4,
        )


def test_worker_context_rejects_unavailable_start_method(monkeypatch):
    monkeypatch.setattr(
        "starVLA.dataloader.mp.get_all_start_methods",
        lambda: ["spawn"],
    )
    with pytest.raises(ValueError, match="unavailable"):
        _resolve_worker_multiprocessing_context(
            {"multiprocessing_context": "forkserver"},
            is_eval=False,
            num_workers=4,
        )
