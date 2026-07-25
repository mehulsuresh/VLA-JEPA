from __future__ import annotations

import pytest

from starVLA.training.train_starvla import resolve_trackers


def test_resolve_trackers_accepts_and_normalizes_supported_names():
    assert resolve_trackers({"trackers": [" TensorBoard "]}) == ["tensorboard"]


@pytest.mark.parametrize(
    "configured",
    [
        "unsupported_tracker",
        ["tensorboard", "unsupported_tracker"],
        ["unsupported_tracker", "ANOTHER_UNSUPPORTED_TRACKER"],
    ],
)
def test_resolve_trackers_fails_closed_on_unsupported_names(configured):
    with pytest.raises(ValueError, match="Unsupported configured tracker"):
        resolve_trackers({"trackers": configured})


@pytest.mark.parametrize("configured", [None, []])
def test_resolve_trackers_allows_explicitly_disabled_tracking(configured):
    assert resolve_trackers({"trackers": configured}) == []


@pytest.mark.parametrize("configured", [42, {"tensorboard": {}}])
def test_resolve_trackers_rejects_invalid_container_types(configured):
    with pytest.raises(TypeError, match="trackers must be"):
        resolve_trackers({"trackers": configured})
