"""Training-time RTC helpers for flow-matching action heads.

Implements the action-prefix conditioning described in Algorithm 1 of
"Training-Time Action Conditioning for Efficient Real-Time Chunking". The
delay sampling mirrors the Physical Intelligence Kinetix reference and the
PyTorch adaptation in FluxVLA.
"""

import math
from typing import Any

import torch


def plain_config(config: Any) -> Any:
    if config is None or isinstance(config, dict):
        return config
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except Exception:
        pass
    return config


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


def get_rtc_training_config(action_config: Any) -> Any:
    return plain_config(
        cfg_get(action_config, "rtc_training", None)
        or cfg_get(action_config, "rtc_training_config", None)
    )


def rtc_training_enabled(config: Any) -> bool:
    return bool(cfg_get(config, "enabled", False))


def get_total_training_steps(config: Any) -> int | None:
    trainer_cfg = cfg_get(config, "trainer", None)
    max_train_steps = cfg_get(trainer_cfg, "max_train_steps", None)
    try:
        max_train_steps = int(max_train_steps)
    except (TypeError, ValueError):
        return None
    return max_train_steps if max_train_steps > 0 else None


def rtc_training_probability(
    config: Any,
    train_step: int | None = None,
    total_steps: int | None = None,
) -> float:
    """Return the active RTC batch probability after warmup/ramp scheduling."""
    target_prob = max(0.0, min(1.0, float(cfg_get(config, "rtc_prob", 1.0))))
    if not rtc_training_enabled(config) or target_prob <= 0.0:
        return 0.0
    if train_step is None:
        return target_prob

    start_step = int(cfg_get(config, "start_step", cfg_get(config, "warmup_steps", 0)) or 0)
    start_step_ratio = cfg_get(config, "start_step_ratio", None)
    if start_step_ratio is not None and total_steps:
        start_step = max(start_step, int(math.ceil(float(start_step_ratio) * total_steps)))
    if train_step < start_step:
        return 0.0

    ramp_steps = int(cfg_get(config, "ramp_steps", 0) or 0)
    if ramp_steps <= 0:
        return target_prob
    ramp_progress = min(1.0, max(0.0, (train_step - start_step + 1) / float(ramp_steps)))
    return target_prob * ramp_progress


def sample_training_delay(
    batch_size: int,
    max_delay: int,
    distribution: str = "exponential",
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample integer delays in [0, max_delay)."""
    max_delay = int(max_delay)
    if max_delay <= 0:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    if temperature <= 0:
        raise ValueError(f"Invalid RTC temperature {temperature}; expected > 0.")

    if distribution == "exponential":
        # Kinetix uses exp(arange(max_delay)[::-1]), favoring small delays.
        weights = torch.exp(
            torch.arange(max_delay, dtype=torch.float32, device=device).flip(0)
            / float(temperature)
        )
        weights = weights / weights.sum()
        return torch.multinomial(weights.expand(batch_size, -1), num_samples=1).squeeze(-1)
    if distribution == "uniform":
        return torch.randint(0, max_delay, (batch_size,), device=device)
    raise ValueError(
        f"Unknown RTC delay distribution {distribution!r}; expected 'exponential' or 'uniform'."
    )


def sample_rtc_training_delays(
    config: Any,
    batch_size: int,
    n_action_steps: int,
    device: torch.device | str,
    train_step: int | None = None,
    total_steps: int | None = None,
) -> torch.Tensor:
    """Sample per-example RTC delays, with optional no-RTC mixing via rtc_prob."""
    max_delay = cfg_get(config, "max_delay", None)
    if max_delay is None:
        max_delay = cfg_get(config, "simulated_delay", 0)
    if max_delay is None:
        max_delay = 0
    max_delay = min(int(max_delay), max(int(n_action_steps), 0))
    delays = sample_training_delay(
        batch_size=batch_size,
        max_delay=max_delay,
        distribution=cfg_get(config, "distribution", "exponential"),
        temperature=float(cfg_get(config, "temperature", 1.0)),
        device=device,
    )

    rtc_prob = rtc_training_probability(config, train_step=train_step, total_steps=total_steps)
    if rtc_prob < 1.0:
        apply_rtc = torch.rand(batch_size, device=device) < max(0.0, rtc_prob)
        delays = torch.where(apply_rtc, delays, torch.zeros_like(delays))
    return delays


def apply_rtc_time_conditioning(
    time: torch.Tensor,
    delays: torch.Tensor,
    n_action_steps: int,
    clean_time: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set prefix positions to clean_time and return the prefix mask."""
    batch_size = time.shape[0]
    per_position_time = time[:, None].expand(batch_size, n_action_steps).clone()
    positions = torch.arange(n_action_steps, device=time.device).unsqueeze(0)
    prefix_mask = positions < delays.unsqueeze(1)
    per_position_time[prefix_mask] = float(clean_time)
    return per_position_time, prefix_mask


def build_sequence_timesteps(
    global_timesteps: torch.Tensor,
    action_timesteps: torch.Tensor,
    n_context_tokens: int,
) -> torch.Tensor:
    """Create per-token DiT timesteps for context tokens plus action tokens."""
    if action_timesteps.dim() == 1:
        return global_timesteps
    context_timesteps = global_timesteps[:, None].expand(-1, n_context_tokens)
    return torch.cat((context_timesteps, action_timesteps), dim=1)


def reduce_masked_loss(
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    reduction: str = "mean",
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reduce flow-matching loss with valid elements weighted equally.

    ``return_components`` returns per-sample numerator/denominator pairs so
    callers with sample weights can still form a global valid-token average.
    """
    if loss_mask is not None:
        loss_mask = loss_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
        loss_sum = (per_token_loss * loss_mask).sum(dim=(1, 2))
        loss_count = loss_mask.sum(dim=(1, 2))
    else:
        loss_sum = per_token_loss.sum(dim=(1, 2))
        count = per_token_loss.shape[1] * per_token_loss.shape[2]
        loss_count = torch.full_like(loss_sum, float(count))

    if return_components:
        if reduction != "none":
            raise ValueError("return_components=True requires reduction='none'.")
        return loss_sum, loss_count

    per_sample_loss = loss_sum / loss_count.clamp_min(1.0)
    if reduction == "none":
        return per_sample_loss
    if reduction == "mean":
        return loss_sum.sum() / loss_count.sum().clamp_min(1.0)
    raise ValueError(f"Unsupported reduction: {reduction}")
