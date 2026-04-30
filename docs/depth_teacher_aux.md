# Depth Teacher Auxiliary Loss

This note summarizes the MoGe-based geometry teacher auxiliary loss added to VLA-JEPA, why it is implemented this way, how to enable it, and what has been validated so far.

## Goal

The auxiliary path was inspired by LingBot-VLA's direct geometry alignment path, but adapted for VLA-JEPA and Qwen-VL. The important point is that this is feature-embedding distillation, not raw depth or normal regression.

Instead of asking the VLA model to directly regress metric depth or surface normals, the path projects Qwen image-token hidden states into the frozen teacher feature space and applies:

- pooled teacher-feature L1 loss
- teacher/prediction feature-similarity loss

The teacher is `Ruicheng/moge-2-vitl-normal`, which is useful because MoGe-2 provides metric-scale geometry and surface-normal supervision internally. We use its learned feature maps as the distillation target.

## Implementation

The implementation lives in:

- `starVLA/model/modules/geometry_teacher.py`
- `starVLA/model/framework/VLA_JEPA.py`
- `starVLA/training/train_starvla.py`
- `starVLA/training/trainer_utils/trainer_tools.py`

The main pieces are:

- `MoGeGeometryTeacher`: frozen MoGe teacher kept outside the trainable model state dict.
- `DirectGeometryTeacherHead`: trainable MLP that projects Qwen image-token hidden states into MoGe feature space.
- `direct_feature_distillation_loss`: LingBot-style direct feature loss.

The aux path is fully gated by config. With `framework.depth_teacher_aux.enabled: false`, the existing VLA-JEPA model path should remain unaffected.

## Direct-Mode Alignment

Qwen image-token hidden states are extracted from the same image tokens Qwen saw. The branch requires Qwen `image_grid_thw` metadata so target grids are explicit. It does not use the old unsafe fallback that inferred grids from token counts.

The current Qwen input builder passes only the last frame per view into Qwen. The depth teacher path mirrors that behavior by taking `batch_videos[:, :, -1]`, keeping teacher targets one-to-one with Qwen image tokens.

All images in a batch are expected to share the same Qwen image-token grid. If this assumption breaks, the branch raises instead of silently misaligning teacher targets.

## Teacher Feature Dim

MoGe-2 ViT-L normal neck feature dimensions are:

| teacher_feature_level | teacher_feature_dim |
| --- | --- |
| 0 | 1024 |
| 1 | 256 |
| 2 | 128 |
| 3 | 64 |
| 4 | 32 |

The default config uses `teacher_feature_dim: auto`. For `Ruicheng/moge-2-vitl-normal`, the code can derive these dimensions without loading the full model on CPU. If the model is already loaded, the inferred dimension is validated against the loaded module.

If a different teacher model is used and `teacher_feature_dim: auto` cannot be inferred without weights, set `teacher_feature_dim` explicitly or set `eager_load_on_cpu: true`.

## Frame Range Handling

The teacher expects images in `[0, 1]`. The config supports:

- `frame_value_range: uint8` or `0_255`
- `frame_value_range: 0_1`
- `frame_value_range: auto`

The default was set to `uint8` for the current robot training path. This avoids repeatedly syncing min/max from CUDA to CPU every step. `auto` remains available, but it must inspect each batch to avoid incorrectly caching an all-black first batch as `[0, 1]`.

## Warmup And Gradients

The auxiliary head is randomly initialized. To avoid pushing noisy gradients into the VLM at step 0, Qwen image-token hidden states are detached during a configurable warmup.

Supported knobs:

- `detach_vlm_steps`
- `detach_vlm_fraction`

The resolved detach warmup is the larger of the explicit step count and the fraction of `trainer.max_train_steps`. The default uses:

```yaml
detach_vlm_steps: null
detach_vlm_fraction: 0.01
```

This keeps the warmup proportional across short LoRA runs and longer full fine-tunes.

## Config

Default disabled config:

```yaml
framework:
  depth_teacher_aux:
    enabled: false
    mode: direct
    teacher_model: Ruicheng/moge-2-vitl-normal
    input_size: 224
    num_tokens: 256
    teacher_feature_source: neck
    teacher_feature_level: 0
    teacher_feature_dim: auto
    frame_value_range: uint8
    loss_weight: 0.004
    detach_vlm_steps: null
    detach_vlm_fraction: 0.01
    head_hidden_multiplier: 2.0
    head_layer_norm: false
    head_final_init_std: 0.0
    feature_l1_weight: 1.0
    feature_similarity_weight: 1.0
    similarity_max_tokens: 4096
```

For local MoGe checkouts, use one of:

```bash
export STARVLA_MOGE_REPO_PATH=/path/to/moge
```

or:

```yaml
framework:
  depth_teacher_aux:
    moge_repo_path: /path/to/moge
```

Avoid hard-coding machine-local paths in shared configs.

## Checkpoint Compatibility

The checkpoint loader allows:

- old checkpoint without `depth_teacher_aux_head.*` into aux-enabled model
- aux checkpoint with `depth_teacher_aux_head.*` into aux-disabled model

Only `depth_teacher_aux_head.*` is filtered. Other missing or unexpected keys still fail normally.

## Multi-Node Notes

The code relies on Hugging Face Hub's cache locking for downloads. That is acceptable for single-node or shared-cache setups.

For multi-node runs, pre-stage the teacher model instead of letting every rank download it:

```bash
huggingface-cli download Ruicheng/moge-2-vitl-normal --local-dir /shared/moge-2-vitl-normal
```

Then set:

```yaml
framework:
  depth_teacher_aux:
    teacher_model: /shared/moge-2-vitl-normal
```

This avoids simultaneous 1.3 GB downloads from all ranks.

## Validation Performed

Unit tests:

```bash
STARVLA_MOGE_REPO_PATH=/home/mehul/work/vjepa/moge \
  /home/mehul/miniconda3/envs/vla-jepa-vjepa21/bin/python \
  -m pytest tests/test_geometry_teacher.py -q
```

Result:

```text
8 passed
```

The tests cover:

- known MoGe-2 feature-dim inference without loading weights on CPU
- finite loss and finite gradients at zero-init
- synthetic optimization where the loss drives predictions toward teacher features
- deterministic similarity-token subsampling when `train_step` is supplied
- Qwen image-token gather alignment
- feature-dim inference fallback through a residual block
- checkpoint key filtering for enabled and disabled aux modes
- optional real MoGe CUDA feature-shape smoke when CUDA and MoGe are available

Single-GPU smoke tests:

- aux disabled one-step smoke passed with `depth=0.000`
- aux enabled one-step smoke passed with `allow_frozen_qwen=true` for dry-run testing
- 50-step compile/profile smoke passed with:
  - `compile_qwen_model=true`
  - `find_unused_parameters=true`
  - `depth_teacher_aux.enabled=true`
  - `profile_cuda_memory=true`

The 50-step profile completed on the local RTX 5090. At step 10:

```text
CUDA memory after step 10: allocated=6.00 GiB, reserved=7.41 GiB,
max_allocated=7.22 GiB, max_reserved=7.41 GiB
```

This machine exposes one CUDA device, so the 50-step profile validates the compile + aux + memory path on a 5090, but it is not a true multi-GPU DDP validation.

## Remaining Validation Before Long Runs

Before a long or expensive run:

- Run a real multi-GPU DDP smoke with the production launch path.
- If using `compile_qwen_model=true` and `find_unused_parameters=true`, verify no DDP unused-parameter warnings or TorchDynamo internal errors.
- Pre-stage MoGe on multi-node runs.
- Compare a short aux-enabled run against a no-aux baseline with the same seed.
- Track `depth_teacher_feature_l1_loss` and `depth_teacher_feature_similarity_loss` separately.
- Confirm `depth_teacher_vlm_detached` flips from `1.0` to `0.0` at the resolved detach step.

