from __future__ import annotations

from typing import Any

import numpy as np


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            value = getter(key)
            return default if value is None else value
    return getattr(config, key, default)


def action_validity_prefix_mask(
    valid_flags: Any,
    *,
    invalid_run_length: int,
    action_is_pad: Any | None = None,
) -> np.ndarray:
    """Keep valid actions until the first sustained invalid run, then mask the suffix."""
    valid = np.asarray(valid_flags, dtype=bool).reshape(-1).copy()
    if invalid_run_length <= 0:
        raise ValueError(f"invalid_run_length must be positive, got {invalid_run_length}.")

    if action_is_pad is not None:
        is_pad = np.asarray(action_is_pad, dtype=bool).reshape(-1)
        if is_pad.shape != valid.shape:
            raise ValueError(
                f"action_is_pad shape {is_pad.shape} does not match valid_flags shape {valid.shape}."
            )
        valid &= ~is_pad

    invalid = ~valid
    run_length = int(invalid_run_length)
    if run_length <= invalid.size:
        for start in range(invalid.size - run_length + 1):
            if bool(invalid[start : start + run_length].all()):
                valid[start:] = False
                break
    return valid


def valid_flags_from_label_values(values: Any, *, positive_is_valid: bool) -> np.ndarray:
    """Convert label values to valid-action flags, treating missing/NaN as keep."""
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(array)
    if positive_is_valid:
        return np.where(finite, array > 0.5, True)
    return np.where(finite, array <= 0.5, True)


def expand_timestep_mask(mask: Any, action_dim: int) -> np.ndarray:
    timestep_mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    return np.repeat(timestep_mask[:, None], int(action_dim), axis=1)


def build_action_mask_from_valid_flags(
    valid_flags: Any,
    *,
    invalid_run_length: int,
    action_dim: int,
    action_is_pad: Any | None = None,
) -> np.ndarray:
    timestep_mask = action_validity_prefix_mask(
        valid_flags,
        invalid_run_length=invalid_run_length,
        action_is_pad=action_is_pad,
    )
    return expand_timestep_mask(timestep_mask, action_dim)
