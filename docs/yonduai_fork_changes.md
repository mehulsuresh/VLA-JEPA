# YonduAI Fork Change Log

This document records the current fork delta between:

- `upstream/main` → `https://github.com/ginwind/VLA-JEPA`
- `main` → `https://github.com/YonduAI/VLA-JEPA`

Comparison basis used for this document:

```text
git diff upstream/main..main
git log --no-merges upstream/main..main
```

Current fork-only commits:

- `73651fa` `Harden training pipeline and add docs`
- `431338e` `Document YonduAI architecture and training changes`

## Scope Summary

Compared to the base repository, the YonduAI fork currently changes:

- `29` tracked files
- roughly `2456` insertions
- roughly `578` deletions

The delta is concentrated in:

- the active `VLA_JEPA` model path
- the single-GPU / distributed training path in `train_starvla.py`
- the preprocessed subtask dataset path
- Qwen 3.5 integration
- deployment hardening
- documentation and operational defaults

This is a functional fork, not just a config fork.

## High-Level Themes

### 1. New active training path centered on Qwen 3.5 + V-JEPA + GR00T

The fork adds and uses:

- `Qwen 3.5` multimodal model support
- a productionized `VLA_JEPA` path with:
  - frozen `Qwen`
  - frozen `V-JEPA encoder`
  - trainable `vj_predictor`
  - trainable `GR00T` action head

### 2. Silent training bugs were fixed

The fork fixes several correctness bugs present or latent in the base path:

- `vj_predictor` world-model branch was prevented from receiving gradients when Qwen was frozen
- checkpoint save / resume semantics were inconsistent with `Accelerate`
- eval previously stole batches from training
- distributed save / eval logic had collective / iterator hazards
- gradient accumulation semantics were unsafe / misleading
- train / inference `state` dtype behavior diverged
- per-step `.item()` calls forced unnecessary GPU synchronization

### 3. Dataloader and dataset handling were extended

The fork adds:

- a new `preprocessed_subtask_dataset` path
- a worker collator for Qwen input assembly
- safe worker clamping
- bounded frame caching
- richer per-sample metadata for RA-BC and subtask supervision

### 4. Configuration and runtime behavior were cleaned up

The fork removes dead config, resolves duplicate config sources, and tightens runtime defaults around:

- logging
- checkpointing
- periodic save/eval
- optimizer config interpretation
- tracker initialization

### 5. Deployment code was hardened

The websocket / server path now returns structured errors, validates requests more carefully, and handles CPU fallback more cleanly.

---

## Detailed File-by-File Change Inventory

This section lists every changed file relative to `upstream/main` and summarizes the effective functional delta.

### Repository and Documentation

#### [`.gitignore`](/home/mehul/work/vjepa/VLA-JEPA/.gitignore)

New in the fork.

Adds ignores for:

- Python build/cache artifacts
- training artifacts such as `checkpoints/`, `results/`, `wandb/`, `runs/`
- model files such as `*.pt`, `*.pth`, `*.ckpt`
- debug media outputs
- IDE and OS junk
- local benchmark directory `tmp_bench/`

#### [`README.md`](/home/mehul/work/vjepa/VLA-JEPA/README.md)

Expanded from the upstream paper-oriented README to document the YonduAI fork.

Adds sections for:

- fork summary
- current architecture
- what changed from base VLA-JEPA
- operational defaults
- train / TensorBoard / checkpoint runbook

#### [`assets/vla_jepa_architecture_research.svg`](/home/mehul/work/vjepa/VLA-JEPA/assets/vla_jepa_architecture_research.svg)

New architecture figure for the fork.

Documents:

- frozen perception backbones
- trainable world-model branch
- trainable GR00T policy branch
- dual-loss training objective

#### [`docs/vla_jepa_architecture.md`](/home/mehul/work/vjepa/VLA-JEPA/docs/vla_jepa_architecture.md)

New architecture explainer.

Includes:

- training pipeline diagram
- inference pipeline diagram
- freeze/train status
- main tensor / sequence shape notes

---

### Deployment

#### [`deployment/model_server/server_policy.py`](/home/mehul/work/vjepa/VLA-JEPA/deployment/model_server/server_policy.py)

Deployment server cleanup and hardening.

Changes:

- cleans up imports
- treats `--cuda` as an `int`
- falls back to CPU cleanly when CUDA is unavailable
- only enables bf16 if CUDA is actually present
- replaces raw `print()` debug messages with logging calls

#### [`deployment/model_server/tools/websocket_policy_server.py`](/home/mehul/work/vjepa/VLA-JEPA/deployment/model_server/tools/websocket_policy_server.py)

Structured websocket server hardening.

Changes:

- replaces f-string logging with structured logging in connect/disconnect events
- wraps unhandled server errors into packed structured error payloads instead of sending raw tracebacks
- validates that incoming websocket messages are dicts
- validates that inference payloads include `batch_images`
- avoids mutating the original payload dict during PIL conversion
- fixes output variable handling so the packed response uses `output_dict`

---

### Requirements

#### [`requirements.txt`](/home/mehul/work/vjepa/VLA-JEPA/requirements.txt)

Adds missing runtime dependencies used by the forked training / deployment paths:

- `mpi4py`
- `websockets`
- `msgpack`
- `opencv-python-headless`

---

### Active Configs

#### [`scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_5090.yaml`](/home/mehul/work/vjepa/VLA-JEPA/scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_5090.yaml)

New active YonduAI single-GPU training config.

Key properties:

- `framework.name: VLA_JEPA`
- `framework.qwenvl.base_vlm: Qwen/Qwen3.5-0.8B`
- `framework.qwenvl.attn_implementation: sdpa`
- `framework.vj2_model.source: torchhub`
- `framework.vj2_model.use_legacy_rope_bug: false`
- `datasets.vla_data.dataset_py: preprocessed_subtask_dataset`
- `datasets.vla_data.per_device_batch_size: 80`
- `datasets.vla_data.num_workers: 2`
- `datasets.vla_data.safe_num_workers_cap: 2`
- `datasets.vla_data.frame_cache_size: 256`
- `trainer.max_train_steps: auto`
- `trainer.num_warmup_steps: auto`
- `trainer.save_interval: 1000`
- `trainer.eval_interval: 1000`
- `trainer.save_best_only: true`
- `trainer.best_metric_name: mae_score`
- `trainer.gradient_accumulation_steps: 1`
- `trainer.repeated_diffusion_steps: 4`
- `trainer.optimizer.name: AdamW8bit`
- `trainer.optimizer.weight_decay: 1e-4`
- `trainer.enable_gradient_checkpointing: true`
- `trainer.channels_last: true`
- `trainer.compile_qwen_model: false`
- `trainer.compile_action_model: false`
- `trainer.compile_vj_predictor: false`
- `trainer.use_rabc: true`
- `trainer.rabc_mistake_weight: 0.25`

This file is the main expression of the fork’s operational defaults.

#### [`starVLA/config/training/starvla_cotrain_libero.yaml`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/config/training/starvla_cotrain_libero.yaml)

Cleanup only.

Removes dead config field:

- `trainer.max_grad_norm`

#### [`starVLA/config/training/starvla_cotrain_oxe.yaml`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/config/training/starvla_cotrain_oxe.yaml)

Cleanup only.

Removes dead config field:

- `trainer.max_grad_norm`

---

### Dataloaders and Datasets

#### [`starVLA/dataloader/__init__.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/dataloader/__init__.py)

Significant dataloader wiring changes.

Changes:

- adds support for the new `preprocessed_subtask_dataset`
- imports and wires:
  - `PreprocessedSubtaskVLADataset`
  - `PreprocessedSubtaskCollator`
- passes `frame_cache_size` into the new dataset
- optionally builds the Qwen worker collator using model prompt/token information
- clamps worker count for the heavy preprocessed dataset via `safe_num_workers_cap`
- adds `DistributedSampler` support for the preprocessed dataset path
- forces `persistent_workers=False` for the preprocessed path
- keeps dataset statistics save behavior for the new path

#### [`starVLA/dataloader/preprocessed_subtask_dataset.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/dataloader/preprocessed_subtask_dataset.py)

New file.

Implements a new preprocessed-subtask training dataset path with:

- `PreprocessedSubtaskVLADataset`
- `PreprocessedSubtaskCollator`

Dataset features:

- reads episode-level metadata from preprocessed folders
- loads action, state, task, mistake, and complexity labels
- converts raw mistake labels into standardized `mistake = 1.0` semantics
- computes future-label fields
- computes RA-BC progress metadata fields
- loads multi-camera images and video windows
- keeps actions / states in `float32`
- includes bounded in-memory frame cache keyed by `(episode, camera, frame_index)`

Collator features:

- lazily initializes a worker-local Qwen processor
- injects action and embodied-action special tokens
- builds left-padded Qwen inputs with `apply_chat_template`
- stacks videos, actions, states, and scalar supervision fields

#### [`starVLA/dataloader/gr00t_lerobot/data_config.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/dataloader/gr00t_lerobot/data_config.py)

Adds Trossen-specific dataset support.

New class:

- `TrossenAIStationaryDataConfig`

It defines:

- video keys
- state keys
- action keys
- language keys
- state/action normalization transforms

Also adds:

- `trossen_ai_stationary` → `ROBOT_TYPE_CONFIG_MAP`

#### [`starVLA/dataloader/gr00t_lerobot/datasets.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/dataloader/gr00t_lerobot/datasets.py)

Adds broader dataset compatibility and richer label extraction.

Changes:

- adds generic fallback support when a dataset stores actions under a single `action` column
- fills gripper defaults for that generic action-column fallback
- changes video collection logic to keep all listed video streams instead of selectively dropping some
- adds extraction of many step-level metadata fields from `curr_traj_data`, including:
  - `frame_index`
  - `task_id`
  - `reward`
  - `global_complexity_to_go`
  - `local_complexity_to_go`
- adds corresponding future-step metadata fields
- derives standardized `mistake_label` and `future_mistake_label`
- computes RA-BC progress values:
  - `rabc_global_progress`
  - `rabc_future_global_progress`
  - `rabc_global_progress_delta`
  - `rabc_stage_progress`
  - `rabc_future_stage_progress`
  - `rabc_progress_delta`

#### [`starVLA/dataloader/gr00t_lerobot/mixtures.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/dataloader/gr00t_lerobot/mixtures.py)

Adds a new named dataset mixture:

- `trossen_subtask_combined`

#### [`starVLA/dataloader/lerobot_datasets.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/dataloader/lerobot_datasets.py)

Changes LeRobot video backend:

- from `torchvision_av`
- to `decord`

This affects dataset decode behavior and runtime compatibility.

---

### Model Framework

#### [`starVLA/model/framework/VLA_JEPA.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/framework/VLA_JEPA.py)

This is the largest and most important fork delta.

Major changes include:

- integrates the new Qwen 3.5 path
- expands tokenizer handling for action and embodied-action tokens
- adds cached token-id buffers
- adds cached image normalization buffers
- adds cached repeated-diffusion setting
- adds cached Qwen grad-state detection
- adds `refresh_runtime_caches()`
- adds optional channels-last handling for the V-JEPA encoder path
- improves video preprocessing / encoding path
- uses Qwen feature extraction (`forward_features`) instead of full LM logits for the training path
- isolates Qwen `no_grad()` behavior from the world-model branch so `vj_predictor` can train
- isolates frozen JEPA encoding under its own `no_grad()` context
- computes `wm_loss` as a raw latent `L1` loss
- keeps action/state in `float32` while using bf16 autocast around heavy subpaths
- makes action loss return per-sample values so RA-BC weights can be applied correctly
- applies RA-BC weights outside the action head
- returns raw `wm_loss` so trainer-side loss scaling is possible
- fixes inference state dtype to `float32`
- keeps inference output numerically safe when converting back to NumPy
- routes repeated diffusion through the cached trainer value
- threads `use_legacy_rope_bug` into the predictor constructor

Net effect:

- `VLA_JEPA` is no longer just a thin composition layer
- it is the main fork-specific model orchestration point
- it contains many of the training correctness and performance fixes

#### [`starVLA/model/framework/__init__.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/framework/__init__.py)

Minor logging cleanup:

- changes failed auto-import reporting from `logger.log(...)` to `logger.info(...)`

---

### VLM Wrappers

#### [`starVLA/model/modules/vlm/QWen2_5.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/vlm/QWen2_5.py)

Extends configurability and adds feature extraction support.

Changes:

- makes attention implementation configurable via config
- makes device map configurable via config
- adds `forward_features(...)` path that:
  - skips LM head
  - avoids storing hidden-state tuples
  - returns `last_hidden_state`

#### [`starVLA/model/modules/vlm/QWen3.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/vlm/QWen3.py)

Cleanup and configurability improvements.

Changes:

- makes attention implementation configurable
- makes device map configurable
- removes dead debug / `__main__` testing code
- cleans imports

#### [`starVLA/model/modules/vlm/QWen3_5.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/vlm/QWen3_5.py)

New file.

Adds full `Qwen 3.5` multimodal wrapper support.

Capabilities added:

- `Qwen3_5ForConditionalGeneration` integration
- configurable attention implementation
- configurable device map
- optional disabling of unsupported fast linear attention kernels
- `prepare_for_compile()` helper to selectively disable problematic subpaths for `torch.compile`
- `forward_features()` path for efficient hidden-state extraction
- generation wrapper
- multimodal input building via `build_qwenvl_inputs(...)`
- label construction for inserted action-token spans
- optional channels-last conversion for image/video tensors

#### [`starVLA/model/modules/vlm/__init__.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/vlm/__init__.py)

Registry update.

Adds `Qwen3.5` model-name dispatch:

- `"Qwen3.5"` → `_QWen3_5_Interface`

---

### Action Model

#### [`starVLA/model/modules/action_model/GR00T_ActionHeader.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/action_model/GR00T_ActionHeader.py)

Action-loss interface was extended to support trainer-side reweighting.

Changes:

- `FlowmatchingActionHead.forward(...)` now accepts:
  - `reduction="mean"` or `reduction="none"`
- computes per-token loss, then per-sample loss
- supports returning per-sample losses for RA-BC weighting
- retains mean reduction behavior for standard use

Also contains minor cleanup of comments / formatting.

#### [`starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py)

Minor cleanup only:

- removes stray debug comment

#### [`starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py)

Mostly cleanup and log-noise reduction.

Changes:

- removes noisy parameter-count `print()` calls
- removes in-source bug/debug commentary
- cleans wording/comments around `register_to_config` and dtype handling

Architecture note:

- the DiT still alternates self-attention and cross-attention usage by block instead of introducing a new dual-attention redesign

---

### World Model

#### [`starVLA/model/modules/world_model/vj2_modules.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/world_model/vj2_modules.py)

Adds explicit control over the RoPE implementation bug.

Changes:

- `rotate_queries_or_keys(...)` now takes `use_legacy_rope_bug`
- legacy behavior remains available
- corrected interleaved behavior is now available
- threads the flag through:
  - `ACRoPEAttention`
  - `RoPEAttention`
  - `ACBlock`
  - `Block`

This allows new training runs to use the corrected RoPE while keeping compatibility with old checkpoints when needed.

#### [`starVLA/model/modules/world_model/vj2_predictor.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/world_model/vj2_predictor.py)

Predictor cleanup and configurability changes.

Changes:

- adds `use_legacy_rope_bug` argument and stores it
- passes the RoPE flag into predictor blocks
- registers `attn_mask` as a non-persistent buffer when causal masking is enabled
- keeps the attention mask on the right device without unnecessary `.to(...)` copies in `forward`

---

### Training Utilities

#### [`starVLA/training/trainer_utils/trainer_tools.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/training/trainer_utils/trainer_tools.py)

Distributed-safety cleanup.

Changes:

- guards `dist.barrier()` behind `dist.is_initialized()`
- makes rank-0-only behavior safe when distributed training is not initialized
- fixes several rank checks that previously assumed distributed mode was always active
- keeps `_reset_dataloader(...)` calling `sampler.set_epoch(epoch_counter)` when available

---

### Main Trainer

#### [`starVLA/training/train_starvla.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/training/train_starvla.py)

This is the second largest and most important fork delta after `VLA_JEPA.py`.

The trainer changes span:

#### Initialization and scheduling

- resolves `max_train_steps` automatically from epochs when set to `auto`
- resolves `num_warmup_steps` automatically from warmup ratio when set to `auto`
- resolves `save_interval` / `eval_interval` from epoch length when using epoch-style config values
- caches loss scales at init
- validates prepared dataloaders for distributed sampler visibility

#### Accelerator and tracker behavior

- constructs `Accelerator` from config-aware mixed-precision handling
- flattens nested config before passing it into tracker initialization
- switches to `Accelerate`-managed trackers
- removes the old parallel manual TensorBoard logging path

#### Optimizer and config hygiene

- treats `trainer.optimizer.weight_decay` as canonical
- warns when `fused=true` is configured for `AdamW8bit`
- logs optimizer group learning rates more cleanly

#### Data iteration and eval

- creates separate training and eval iterators
- stops eval from consuming the training iterator
- changes eval to run on all ranks and reduce metrics across ranks
- adds optional `eval_before_train`
- renames the misleading eval metric from `mse_score` to `norm_l2_per_element`

#### Logging and performance

- avoids step-0 logging noise
- removes unconditional per-step `.item()` synchronization
- resets peak CUDA memory stats after logging windows so “peak” metrics remain meaningful
- removes redundant top-level `learning_rate` metric in favor of per-group `lr_*`

#### Checkpointing and resume

- saves full `Accelerate` state directories
- writes plain `pytorch_model.pt` alongside full state as a convenience artifact
- fixes checkpoint collectives so all ranks participate correctly
- makes final save compatible with the same state model
- adds save-best-only logic driven by `best_metric_name` / `best_metric_mode`

#### Gradient accumulation and stepping

- fixes accumulation semantics around `zero_grad`
- makes the sync-step update path explicit
- ensures clipping / stepping / scheduler / zeroing happen at the right boundaries

#### Loss handling and RA-BC

- computes trainer-side total loss from:
  - `action_loss_scale`
  - `wm_loss_scale`
- uses standardized mistake-label semantics
- supports nonzero mistake weighting
- warns when mistakes are configured to contribute zero action loss

#### Miscellaneous cleanup

- lazily imports `wandb`
- adds clearer mixed-precision messaging when `Accelerator` AMP is disabled but manual autocast is used
- refreshes model runtime caches after freezing backbones

Net effect:

- `train_starvla.py` in this fork is materially safer, more production-oriented, and more aligned with the intended training objectives than the base version

---

## Files Added by the Fork

New tracked files relative to `upstream/main`:

- [`.gitignore`](/home/mehul/work/vjepa/VLA-JEPA/.gitignore)
- [`assets/vla_jepa_architecture_research.svg`](/home/mehul/work/vjepa/VLA-JEPA/assets/vla_jepa_architecture_research.svg)
- [`docs/vla_jepa_architecture.md`](/home/mehul/work/vjepa/VLA-JEPA/docs/vla_jepa_architecture.md)
- [`scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_5090.yaml`](/home/mehul/work/vjepa/VLA-JEPA/scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_5090.yaml)
- [`starVLA/dataloader/preprocessed_subtask_dataset.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/dataloader/preprocessed_subtask_dataset.py)
- [`starVLA/model/modules/vlm/QWen3_5.py`](/home/mehul/work/vjepa/VLA-JEPA/starVLA/model/modules/vlm/QWen3_5.py)

---

## Removed or Simplified Upstream Behavior

The fork also intentionally removes or de-emphasizes some upstream behavior:

- dead config aliases such as `max_grad_norm`
- noisy print/debug paths in model code
- reliance on raw LM logits where feature extraction is sufficient
- fragile or ambiguous checkpoint behavior
- eval behavior that mutates the training iterator
- implicit assumptions that distributed training is always initialized

---

## Notes

- This document tracks the current branch delta, not a chronological engineering journal.
- If the fork diverges further, regenerate or update this file against `upstream/main`.
- Temporary benchmark logs and throwaway environment experiments were intentionally excluded from the published fork state.
