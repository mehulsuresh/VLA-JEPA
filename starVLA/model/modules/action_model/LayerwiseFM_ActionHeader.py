# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].



from dataclasses import dataclass, field
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, SelfAttentionTransformer
from starVLA.model.modules.action_model.rtc_training import (
    apply_rtc_time_conditioning,
    build_sequence_timesteps,
    cfg_get,
    get_rtc_training_config,
    get_total_training_steps,
    reduce_masked_loss,
    rtc_training_enabled,
    sample_rtc_training_delays,
)

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) or (B, T)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        elif timesteps.dim() == 2 and timesteps.shape == (B, T):
            pass
        else:
            raise ValueError(
                f"Expected `timesteps` to have shape (B,) or (B, T), got {tuple(timesteps.shape)}."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) or (B, T)
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        elif timesteps.dim() == 2 and timesteps.shape == (B, T):
            pass
        else:
            raise ValueError(
                f"Expected `timesteps` to have shape (B,) or (B, T), got {tuple(timesteps.shape)}."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )
    rtc_training: dict = field(default=None)
    rtc_training_config: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)





class LayerwiseFlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
        **kwargs,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config
        self.hidden_size = config.hidden_size

        action_model_cfg = full_config.framework.action_model.DiTConfig
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg) # TODO better way is copy LLM from VLM
        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=config.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        ) if config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.rtc_training_config = get_rtc_training_config(config)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _rtc_condition_dit_tokens(self) -> bool:
        return rtc_training_enabled(self.rtc_training_config) and bool(
            cfg_get(self.rtc_training_config, "condition_dit_tokens", False)
        )

    def _build_model_timestep(
        self,
        global_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
        n_context_tokens: int,
    ) -> torch.Tensor:
        if not self._rtc_condition_dit_tokens():
            return global_timesteps
        return build_sequence_timesteps(global_timesteps, action_timesteps, n_context_tokens)

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        reduction: str = "mean",
        train_step: int = None,
        return_loss_components: bool = False,
    ):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = actions.device
        num_layers = len(vl_embs_list)
        B, L, D = vl_embs_list[0].shape
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t_scalar = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        action_horizon = actions.shape[1]
        prefix_mask = None

        if rtc_training_enabled(self.rtc_training_config):
            delays = sample_rtc_training_delays(
                self.rtc_training_config,
                batch_size=actions.shape[0],
                n_action_steps=action_horizon,
                device=actions.device,
                train_step=train_step,
                total_steps=get_total_training_steps(self.full_config),
            )
            t, prefix_mask = apply_rtc_time_conditioning(
                t_scalar,
                delays,
                n_action_steps=action_horizon,
                clean_time=float(cfg_get(self.rtc_training_config, "clean_time", 1.0)),
            )
        else:
            t = t_scalar[:, None].expand(-1, action_horizon)

        noisy_trajectory = (1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t * self.num_timestep_buckets).long()
        t_global = (t_scalar * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        n_context_tokens = sa_embs.shape[1] - action_horizon
        model_timestep = self._build_model_timestep(t_global, t_discretized, n_context_tokens)
        
        # Encode timesteps
        temb = self.model.timestep_encoder(model_timestep)

        # Layerwise cross-attention with vl_embs
        model_output = sa_embs
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_embs_list[layer_idx],  # Use layer-specific vl_embs
                temb=temb,
            )

            # hidden_states = block(
            #         hidden_states,
            #         attention_mask=None,
            #         encoder_hidden_states=encoder_hidden_states,
            #         encoder_attention_mask=None,
            #         temb=temb,
            #     )

        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        per_token_loss = (pred_actions - velocity) ** 2
        loss_mask = None
        if action_mask is not None:
            loss_mask = action_mask.to(device=per_token_loss.device, dtype=per_token_loss.dtype)
            if loss_mask.shape != per_token_loss.shape:
                raise ValueError(
                    f"action_mask shape {tuple(action_mask.shape)} does not match "
                    f"action loss shape {tuple(per_token_loss.shape)}"
                )
        if prefix_mask is not None:
            prefix_loss_mask = (~prefix_mask).unsqueeze(-1).to(dtype=per_token_loss.dtype)
            prefix_loss_mask = prefix_loss_mask.expand_as(per_token_loss)
            loss_mask = prefix_loss_mask if loss_mask is None else loss_mask * prefix_loss_mask
        return reduce_masked_loss(
            per_token_loss,
            loss_mask=loss_mask,
            reduction=reduction,
            return_components=return_loss_components,
        )

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        prev_actions: torch.Tensor = None,
        prefix_len: int = 0,
        rtc_config: dict = None,
        num_inference_timesteps: int | None = None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )

        num_steps = int(num_inference_timesteps or self.num_inference_timesteps)
        if num_steps <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {num_steps}.")
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None
        rtc_requested = bool(cfg_get(rtc_config, "enabled", False))
        rtc_enabled = (
            prev_actions is not None
            and prefix_len > 0
            and (rtc_config is None or bool(cfg_get(rtc_config, "enabled", True)))
        )
        if rtc_requested and not rtc_enabled:
            warnings.warn(
                "RTC inference was requested but disabled because prev_actions is missing "
                "or prefix_len <= 0.",
                RuntimeWarning,
                stacklevel=2,
            )
        rtc_method = cfg_get(rtc_config, "method", "prefix") if rtc_enabled else None
        if rtc_method not in (None, "prefix"):
            raise ValueError(f"Unsupported RTC inference method {rtc_method!r}; only 'prefix' is implemented.")
        if rtc_enabled:
            prefix_len = int(prefix_len)
            if prefix_len > self.action_horizon:
                raise ValueError(f"prefix_len={prefix_len} exceeds action_horizon={self.action_horizon}.")
            if prev_actions.shape[0] != batch_size or prev_actions.shape[1] < prefix_len:
                raise ValueError(
                    f"prev_actions shape {tuple(prev_actions.shape)} is incompatible with "
                    f"batch_size={batch_size}, prefix_len={prefix_len}."
                )
            if prev_actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"prev_actions last dimension {prev_actions.shape[-1]} must match action_dim={self.action_dim}."
                )
            prev_actions = prev_actions.to(device=device, dtype=actions.dtype)

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            if rtc_enabled:
                actions[:, :prefix_len] = prev_actions[:, :prefix_len]
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_timesteps = timesteps_tensor
            if rtc_enabled:
                action_timesteps = torch.full(
                    (batch_size, self.action_horizon),
                    fill_value=t_discretized_int,
                    device=device,
                    dtype=torch.long,
                )
                action_timesteps[:, :prefix_len] = self.num_timestep_buckets
            action_features = self.action_encoder(actions, action_timesteps)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )
            n_context_tokens = sa_embs.shape[1] - self.action_horizon
            model_timestep = self._build_model_timestep(
                timesteps_tensor,
                action_timesteps,
                n_context_tokens,
            )

            # Encode timestep
            temb = self.model.timestep_encoder(model_timestep)

            # Layerwise cross-attention with vl_embs_list
            model_output = sa_embs
            for layer_idx, layer in enumerate(self.model.transformer_blocks):
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=vl_embs_list[layer_idx],
                    temb=temb,
                )

            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity
            if rtc_enabled:
                actions[:, :prefix_len] = prev_actions[:, :prefix_len]
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return LayerwiseFlowmatchingActionHead(
        full_config=config
    )



if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass
