from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

import starVLA.model.modules.action_model.GR00T_ActionHeader as gr00t_action_header
from starVLA.model.modules.action_model.GR00T_ActionHeader import ActionEncoder, FlowmatchingActionHead
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.action_model.rtc_training import (
    apply_rtc_time_conditioning,
    reduce_masked_loss,
    rtc_training_probability,
    sample_rtc_training_delays,
)


class _FixedBeta:
    def __init__(self, value):
        self.value = float(value)

    def sample(self, shape):
        return torch.full(tuple(shape), self.value)


def test_rtc_delay_sampling_and_time_conditioning():
    cfg = SimpleNamespace(enabled=True, max_delay=4, distribution="uniform", rtc_prob=1.0)
    delays = sample_rtc_training_delays(cfg, batch_size=16, n_action_steps=5, device="cpu")

    assert delays.shape == (16,)
    assert int(delays.min()) >= 0
    assert int(delays.max()) <= 3

    time, prefix_mask = apply_rtc_time_conditioning(
        torch.tensor([0.2, 0.5]),
        torch.tensor([2, 0]),
        n_action_steps=5,
    )

    assert time.shape == (2, 5)
    assert prefix_mask.shape == (2, 5)
    assert torch.allclose(time[0, :2], torch.ones(2))
    assert torch.allclose(time[0, 2:], torch.full((3,), 0.2))
    assert not prefix_mask[1].any()


def test_rtc_training_probability_warmup_and_ramp():
    cfg = SimpleNamespace(
        enabled=True,
        rtc_prob=0.8,
        warmup_steps=10,
        ramp_steps=10,
    )

    assert rtc_training_probability(cfg, train_step=9) == 0.0
    assert 0.0 < rtc_training_probability(cfg, train_step=10) < 0.8
    assert rtc_training_probability(cfg, train_step=19) == 0.8

    ratio_cfg = SimpleNamespace(enabled=True, rtc_prob=1.0, start_step_ratio=0.25)
    assert rtc_training_probability(ratio_cfg, train_step=24, total_steps=100) == 0.0
    assert rtc_training_probability(ratio_cfg, train_step=25, total_steps=100) == 1.0


def test_action_encoder_accepts_per_position_timesteps():
    encoder = ActionEncoder(action_dim=3, hidden_size=8)
    actions = torch.randn(2, 5, 3)

    per_sample = encoder(actions, torch.ones(2, dtype=torch.long))
    per_position = encoder(actions, torch.ones(2, 5, dtype=torch.long))

    assert per_sample.shape == (2, 5, 8)
    assert per_position.shape == (2, 5, 8)


def test_dit_accepts_per_token_timesteps():
    model = DiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=1,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        cross_attention_dim=6,
    )
    hidden = torch.randn(2, 7, 8)
    context = torch.randn(2, 3, 6)

    per_sample = model(hidden, context, timestep=torch.ones(2, dtype=torch.long))
    per_token = model(hidden, context, timestep=torch.ones(2, 7, dtype=torch.long))

    assert per_sample.shape == (2, 7, 8)
    assert per_token.shape == (2, 7, 8)


def test_dit_applies_encoder_attention_mask_to_cross_attention():
    torch.manual_seed(0)
    model = DiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=1,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        cross_attention_dim=6,
    )
    model.eval()
    hidden = torch.randn(2, 7, 8)
    context = torch.randn(2, 4, 6)
    changed_context = context.clone()
    changed_context[:, -1] += 1000.0
    encoder_attention_mask = torch.zeros(2, 1, 4)
    encoder_attention_mask[:, :, -1] = -10000.0
    timestep = torch.ones(2, dtype=torch.long)

    masked_reference = model(
        hidden,
        context,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
    )
    masked_changed = model(
        hidden,
        changed_context,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
    )
    unmasked_changed = model(hidden, changed_context, timestep=timestep)

    assert torch.allclose(masked_reference, masked_changed, atol=1e-5)
    assert not torch.allclose(masked_reference, unmasked_changed, atol=1e-3)


def test_dit_accepts_query_specific_encoder_attention_mask():
    torch.manual_seed(0)
    model = DiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=1,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        cross_attention_dim=6,
    )
    model.eval()
    hidden = torch.randn(1, 7, 8)
    context = torch.randn(1, 4, 6)
    changed_context = context.clone()
    changed_context[:, -1] += 1000.0
    encoder_attention_mask = torch.zeros(1, hidden.shape[1], context.shape[1])
    encoder_attention_mask[:, :3, -1] = -10000.0
    timestep = torch.ones(1, dtype=torch.long)

    masked_reference = model(
        hidden,
        context,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
    )
    masked_changed = model(
        hidden,
        changed_context,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
    )

    assert torch.allclose(masked_reference[:, :3], masked_changed[:, :3], atol=1e-5)
    assert not torch.allclose(masked_reference[:, 3:], masked_changed[:, 3:], atol=1e-3)


def _make_gr00t_full_cfg(rtc_training=None):
    full_cfg = SimpleNamespace(
        framework=SimpleNamespace(
            action_model=SimpleNamespace(
                hidden_size=768,
                action_model_type="DiT-B",
                diffusion_model_cfg={
                    "cross_attention_dim": 16,
                    "dropout": 0.0,
                    "final_dropout": False,
                    "interleave_self_attention": False,
                    "norm_type": "ada_norm",
                    "num_layers": 1,
                    "output_dim": 768,
                    "positional_embeddings": None,
                },
                action_dim=3,
                state_dim=3,
                future_action_window_size=4,
                action_horizon=5,
                num_inference_timesteps=2,
                num_target_vision_tokens=2,
                add_pos_embed=True,
                max_seq_len=32,
                noise_beta_alpha=1.5,
                noise_beta_beta=1.0,
                noise_s=0.999,
                num_timestep_buckets=1000,
                rtc_training=rtc_training,
            )
        )
    )
    return full_cfg


def test_gr00t_head_rtc_forward_and_prefix_inference():
    full_cfg = _make_gr00t_full_cfg(
        {
            "enabled": True,
            "max_delay": 3,
            "distribution": "uniform",
            "condition_dit_tokens": True,
        }
    )
    head = FlowmatchingActionHead(full_cfg)
    vl_embs = torch.randn(2, 4, 16)
    actions = torch.randn(2, 5, 3)
    state = torch.randn(2, 1, 3)

    loss = head(vl_embs, actions, state)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    prev_actions = torch.randn(2, 5, 3)
    pred = head.predict_action(
        vl_embs,
        state,
        prev_actions=prev_actions,
        prefix_len=2,
        rtc_config={"enabled": True, "method": "prefix"},
    )

    assert pred.shape == (2, 5, 3)
    assert torch.allclose(pred[:, :2], prev_actions[:, :2], atol=1e-4)


def test_gr00t_head_rtc_scalar_dit_forward_and_prefix_inference():
    full_cfg = _make_gr00t_full_cfg(
        {
            "enabled": True,
            "max_delay": 3,
            "distribution": "uniform",
        }
    )
    head = FlowmatchingActionHead(full_cfg)
    vl_embs = torch.randn(2, 4, 16)
    actions = torch.randn(2, 5, 3)
    state = torch.randn(2, 1, 3)

    loss = head(vl_embs, actions, state)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    prev_actions = torch.randn(2, 5, 3)
    pred = head.predict_action(
        vl_embs,
        state,
        prev_actions=prev_actions,
        prefix_len=2,
        rtc_config={"enabled": True, "method": "prefix"},
    )

    assert pred.shape == (2, 5, 3)
    assert torch.allclose(pred[:, :2], prev_actions[:, :2], atol=1e-4)


def test_gr00t_head_uses_reference_scalar_dit_timestep_by_default():
    full_cfg = _make_gr00t_full_cfg(
        {
            "enabled": True,
            "max_delay": 3,
            "distribution": "uniform",
        }
    )
    head = FlowmatchingActionHead(full_cfg)
    assert not head._rtc_condition_dit_tokens()

    t_global = torch.ones(2, dtype=torch.long)
    action_timesteps = torch.ones(2, 5, dtype=torch.long)
    model_timestep = head._build_model_timestep(t_global, action_timesteps, n_context_tokens=3)

    assert model_timestep.shape == (2,)


def test_gr00t_sample_time_clamps_beta_sample_to_noise_s():
    head = object.__new__(FlowmatchingActionHead)
    head.beta_dist = _FixedBeta(1.0)
    head.noise_s = 0.999

    t = head.sample_time(batch_size=4, device="cpu", dtype=torch.float32)

    assert torch.all(t >= 0)
    assert torch.allclose(t, torch.zeros(4))


def test_layerwise_sample_time_clamps_beta_sample_to_noise_s():
    head = object.__new__(LayerwiseFlowmatchingActionHead)
    head.beta_dist = _FixedBeta(1.0)
    head.config = SimpleNamespace(noise_s=0.999)

    t = head.sample_time(batch_size=4, device="cpu", dtype=torch.float32)

    assert torch.all(t >= 0)
    assert torch.allclose(t, torch.zeros(4))


def test_rtc_masked_loss_uses_global_valid_token_mean():
    per_token_loss = torch.tensor(
        [
            [[100.0, 100.0], [1.0, 3.0], [2.0, 4.0]],
            [[5.0, 7.0], [11.0, 13.0], [17.0, 19.0]],
        ]
    )
    prefix_mask = torch.tensor([[True, False, False], [False, False, False]])
    loss_mask = (~prefix_mask).unsqueeze(-1).expand_as(per_token_loss)

    expected = torch.cat((per_token_loss[0, 1:].flatten(), per_token_loss[1].flatten())).mean()
    actual = reduce_masked_loss(per_token_loss, loss_mask=loss_mask, reduction="mean")

    assert torch.allclose(actual, expected)

    per_sample = reduce_masked_loss(per_token_loss, loss_mask=loss_mask, reduction="none")
    assert torch.allclose(
        per_sample,
        torch.tensor([per_token_loss[0, 1:].mean(), per_token_loss[1].mean()]),
    )

    loss_sum, loss_count = reduce_masked_loss(
        per_token_loss,
        loss_mask=loss_mask,
        reduction="none",
        return_components=True,
    )
    assert torch.allclose(loss_sum, torch.tensor([10.0, 72.0]))
    assert torch.allclose(loss_count, torch.tensor([4.0, 6.0]))

    zero_mask = torch.zeros_like(loss_mask)
    zero_sum, zero_count = reduce_masked_loss(
        per_token_loss,
        loss_mask=zero_mask,
        reduction="none",
        return_components=True,
    )
    assert torch.allclose(zero_sum, torch.zeros(2))
    assert torch.allclose(zero_count, torch.zeros(2))
    assert torch.allclose(
        reduce_masked_loss(per_token_loss, loss_mask=zero_mask, reduction="mean"),
        torch.tensor(0.0),
    )
    with pytest.raises(ValueError, match="return_components=True requires reduction='none'"):
        reduce_masked_loss(per_token_loss, loss_mask=loss_mask, return_components=True)


class _ZeroModel(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, timestep, return_all_hidden_states=False):
        return torch.zeros_like(hidden_states)


class _CaptureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_mask = None
        self.encoder_attention_mask = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        attention_mask=None,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
    ):
        self.attention_mask = attention_mask
        self.encoder_attention_mask = encoder_attention_mask
        return torch.zeros_like(hidden_states)


class _ZeroActionDecoder(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, hidden_states):
        return torch.zeros(
            *hidden_states.shape[:-1],
            self.action_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )


def test_gr00t_head_passes_encoder_attention_mask_to_dit():
    head = FlowmatchingActionHead(_make_gr00t_full_cfg(None))
    capture_model = _CaptureModel()
    head.model = capture_model
    head.action_decoder = _ZeroActionDecoder(action_dim=3)

    vl_embs = torch.randn(2, 4, 16)
    actions = torch.randn(2, 5, 3)
    state = torch.randn(2, 1, 3)
    encoder_attention_mask = torch.zeros(2, 1, 4)
    encoder_attention_mask[:, :, -1] = -10000.0

    loss = head(
        vl_embs,
        actions,
        state,
        encoder_attention_mask=encoder_attention_mask,
    )

    assert loss.ndim == 0
    assert capture_model.encoder_attention_mask is encoder_attention_mask


def test_gr00t_head_passes_self_attention_mask_to_dit():
    head = FlowmatchingActionHead(_make_gr00t_full_cfg(None))
    capture_model = _CaptureModel()
    head.model = capture_model
    head.action_decoder = _ZeroActionDecoder(action_dim=3)

    vl_embs = torch.randn(2, 4, 16)
    actions = torch.randn(2, 5, 3)
    state = torch.randn(2, 1, 3)
    attention_mask = torch.zeros(2, 8, 8)
    attention_mask[:, :3, 3:] = -10000.0

    loss = head(
        vl_embs,
        actions,
        state,
        attention_mask=attention_mask,
    )

    assert loss.ndim == 0
    assert capture_model.attention_mask is attention_mask


def test_gr00t_predict_action_passes_encoder_attention_mask_to_dit():
    head = FlowmatchingActionHead(_make_gr00t_full_cfg(None))
    capture_model = _CaptureModel()
    head.model = capture_model
    head.action_decoder = _ZeroActionDecoder(action_dim=3)

    vl_embs = torch.randn(2, 4, 16)
    state = torch.randn(2, 1, 3)
    encoder_attention_mask = torch.zeros(2, 1, 4)
    encoder_attention_mask[:, :, -1] = -10000.0

    pred = head.predict_action(
        vl_embs,
        state,
        encoder_attention_mask=encoder_attention_mask,
    )

    assert pred.shape == (2, 5, 3)
    assert capture_model.encoder_attention_mask is encoder_attention_mask


def test_gr00t_predict_action_passes_self_attention_mask_to_dit():
    head = FlowmatchingActionHead(_make_gr00t_full_cfg(None))
    capture_model = _CaptureModel()
    head.model = capture_model
    head.action_decoder = _ZeroActionDecoder(action_dim=3)

    vl_embs = torch.randn(2, 4, 16)
    state = torch.randn(2, 1, 3)
    attention_mask = torch.zeros(2, 8, 8)
    attention_mask[:, :3, 3:] = -10000.0

    pred = head.predict_action(
        vl_embs,
        state,
        attention_mask=attention_mask,
    )

    assert pred.shape == (2, 5, 3)
    assert capture_model.attention_mask is attention_mask


def test_gr00t_head_rtc_fixed_delay_loss_matches_manual_mask(monkeypatch):
    rtc_cfg = {
        "enabled": True,
        "max_delay": 3,
        "distribution": "uniform",
    }
    rtc_head = FlowmatchingActionHead(_make_gr00t_full_cfg(rtc_cfg))
    baseline_head = FlowmatchingActionHead(_make_gr00t_full_cfg(None))
    for head in (rtc_head, baseline_head):
        head.model = _ZeroModel()
        head.action_decoder = _ZeroActionDecoder(action_dim=3)
        head.sample_time = lambda batch_size, device, dtype: torch.full(
            (batch_size,),
            0.25,
            device=device,
            dtype=dtype,
        )

    fixed_delays = torch.tensor([2, 0], dtype=torch.long)

    def _fixed_delays(*args, **kwargs):
        return fixed_delays.to(kwargs["device"])

    monkeypatch.setattr(gr00t_action_header, "sample_rtc_training_delays", _fixed_delays)

    vl_embs = torch.randn(2, 4, 16)
    actions = torch.randn(2, 5, 3)
    state = torch.randn(2, 1, 3)
    action_mask = torch.ones_like(actions)
    action_mask[0, :2] = 0.0

    torch.manual_seed(123)
    rtc_loss = rtc_head(vl_embs, actions, state)
    torch.manual_seed(123)
    manual_loss = baseline_head(vl_embs, actions, state, action_mask=action_mask)

    assert torch.allclose(rtc_loss, manual_loss)


def test_predict_action_warns_when_rtc_requested_without_prefix():
    full_cfg = _make_gr00t_full_cfg(None)
    head = FlowmatchingActionHead(full_cfg)
    vl_embs = torch.randn(2, 4, 16)
    state = torch.randn(2, 1, 3)

    with pytest.warns(RuntimeWarning, match="RTC inference was requested"):
        pred = head.predict_action(
            vl_embs,
            state,
            rtc_config={"enabled": True, "method": "prefix"},
        )

    assert pred.shape == (2, 5, 3)


def test_robot_ft_configs_keep_per_token_dit_disabled_except_full_qwen():
    explicit_rtc_configs = {
        Path("scripts/config/vlajepa_robot_ft_lerobot_trossen_qwen35_08b_lora_moge_vits_5090.yaml"),
        Path(
            "scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml"
        ),
        Path(
            "scripts/config/vlajepa_robot_ft_lerobot_ogrealman_human_labelled_cloud_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml"
        ),
        Path("scripts/config/vlajepa_robot_ft_lerobot_ogrealman_qwen35_08b_lora_moge_vits_5090.yaml"),
        Path(
            "scripts/config/vlajepa_robot_ft_lerobot_ogrealman_source_qwen35_08b_lora_moge_vits_5090.yaml"
        ),
        Path("scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml"),
    }
    for path in sorted(Path("scripts/config").glob("vlajepa_robot_ft*.yaml")):
        if path in explicit_rtc_configs:
            continue
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        rtc_config = cfg["framework"]["action_model"]["rtc_training"]
        assert rtc_config["enabled"] is False, path
        assert rtc_config["condition_dit_tokens"] is False, path


def test_lerobot_qwen35_lora_config_enables_rtc_without_per_token_dit():
    path = Path("scripts/config/vlajepa_robot_ft_lerobot_trossen_qwen35_08b_lora_moge_vits_5090.yaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rtc_config = cfg["framework"]["action_model"]["rtc_training"]
    assert rtc_config["enabled"] is True
    assert rtc_config["condition_dit_tokens"] is False
    assert rtc_config["max_delay"] == 11
    assert rtc_config["distribution"] == "uniform"
    assert rtc_config["warmup_steps"] == 10000
    assert rtc_config["ramp_steps"] == 50000


def test_full_qwen_zero3_config_enables_rtc_warmup():
    path = Path("scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rtc_config = cfg["framework"]["action_model"]["rtc_training"]
    assert rtc_config["enabled"] is True
    assert rtc_config["condition_dit_tokens"] is True
    assert rtc_config["max_delay"] == 7
    assert rtc_config["distribution"] == "exponential"
    assert rtc_config["warmup_steps"] > 0
    assert rtc_config["ramp_steps"] > 0
