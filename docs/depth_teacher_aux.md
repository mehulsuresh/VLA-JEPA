# Geometry Teacher Auxiliary Loss: Technical Design

This document describes only the technical implementation of the geometry teacher auxiliary path in VLA-JEPA. It is intended for implementation review.

## Summary

The auxiliary path implements LingBot-style direct feature distillation from a frozen geometry teacher into Qwen image-token hidden states.

It does not regress raw depth, metric depth, point maps, or surface normals. MoGe-2 is used as a frozen geometry-aware feature teacher, and VLA-JEPA learns a projection from Qwen image-token embeddings into MoGe feature space.

The auxiliary branch is disabled by default and is gated by:

```yaml
framework:
  depth_teacher_aux:
    enabled: false
```

When disabled, no teacher is loaded, no auxiliary head is constructed, and no auxiliary loss is added.

## Teacher

Teacher model:

```text
Ruicheng/moge-2-vitl-normal
```

Implementation:

```text
starVLA/model/modules/geometry_teacher.py
```

The teacher wrapper is `MoGeGeometryTeacher`. It is intentionally not an `nn.Module` child of `VLA_JEPA`; the inner MoGe model is kept outside the trainable model state dict and optimizer parameter set.

The teacher is:

- loaded through `moge.model.v2.MoGeModel.from_pretrained`
- set to `requires_grad_(False)`
- set to `eval()`
- moved to the active inference device and dtype on use

Local MoGe source import is controlled by:

```bash
STARVLA_MOGE_REPO_PATH=/path/to/moge
```

or:

```yaml
framework:
  depth_teacher_aux:
    moge_repo_path: /path/to/moge
```

## Teacher Feature Source

The default target is the MoGe neck feature at level 0:

```yaml
teacher_feature_source: neck
teacher_feature_level: 0
teacher_feature_dim: auto
```

For `Ruicheng/moge-2-vitl-normal`, the known neck dimensions are:

| Level | Feature Dim |
| --- | --- |
| 0 | 1024 |
| 1 | 256 |
| 2 | 128 |
| 3 | 64 |
| 4 | 32 |

The implementation can derive these dimensions without loading the full teacher if the model is `Ruicheng/moge-2-vitl-normal` or a local path ending in `moge-2-vitl-normal`.

If the teacher is already loaded, the feature dimension is inferred from the loaded MoGe modules and checked against the configured or known value.

## Student Projection Head

Implementation:

```text
DirectGeometryTeacherHead
```

Input:

```text
[num_images, qwen_image_tokens, qwen_hidden_dim]
```

Output:

```text
[num_images, qwen_image_tokens, teacher_feature_dim]
```

Architecture:

```text
Linear(qwen_hidden_dim, teacher_feature_dim * head_hidden_multiplier)
GELU
Dropout
Linear(teacher_feature_dim * head_hidden_multiplier, teacher_feature_dim)
```

Defaults:

```yaml
head_hidden_multiplier: 2.0
head_layer_norm: false
head_final_init_std: 0.0
```

The final linear layer is zero-initialized by default. This makes the auxiliary head start from a non-random teacher-space output and reduces the chance of sending large initial gradients into the VLM.

A LayerNorm can be enabled with `head_layer_norm: true`, but the default avoids it to stay closer to LingBot direct mode.

## Qwen Image Token Extraction

Implementation:

```text
VLA_JEPA._extract_qwen_image_hidden
```

The branch extracts hidden states only at Qwen image-token positions.

The image token id is resolved from the Qwen processor or tokenizer using:

1. `processor.image_token_id`
2. `processor.image_token`
3. `<|image_pad|>`
4. `<image>`

The implementation requires Qwen `image_grid_thw`. If it is absent, the auxiliary path raises. This is deliberate: inferring image grids from token counts is unsafe when image sizes or view resolutions differ.

The current Qwen input builder passes the last temporal frame from each view. The geometry teacher branch mirrors this by using:

```text
batch_videos[:, :, -1]
```

This keeps Qwen image-token embeddings and MoGe teacher targets one-to-one.

Current alignment assumptions:

- `image_grid_thw.shape[0] == batch_size * num_views`
- every `grid_t == 1`
- all images in the batch share the same token grid
- each sample has exactly `num_views * tokens_per_view` image-token positions

If any of these assumptions fail, the branch raises.

The final gather is vectorized with `torch.gather` instead of per-view indexing loops.

## Frame Preprocessing

Implementation:

```text
MoGeGeometryTeacher._prepare_images
```

MoGe expects images in `[0, 1]`.

Supported input ranges:

```yaml
frame_value_range: uint8   # aliases: 0_255, 0-255
frame_value_range: 0_1
frame_value_range: auto
```

For explicit ranges, the implementation validates min/max once and caches the resolved range. This avoids a CUDA-to-host min/max sync every step.

For `auto`, the implementation checks every batch. This is slower, but avoids incorrectly caching a first all-black uint8 batch as `[0, 1]`.

The default robot config uses:

```yaml
frame_value_range: uint8
```

Frames are resized to `input_size` before teacher inference if needed.

## Teacher Forward

Default config:

```yaml
input_size: 224
num_tokens: 256
teacher_feature_source: neck
teacher_feature_level: 0
```

For a 224x224 input and `num_tokens: 256`, MoGe encoder features are requested on a 16x16 token grid.

For `teacher_feature_source: neck`, the implementation constructs the normalized view-plane UV inputs expected by MoGe's neck and returns the selected neck feature level.

Teacher features are detached before loss computation:

```text
teacher_features: [num_images, teacher_feature_dim, H_teacher, W_teacher]
```

## Loss

Implementation:

```text
direct_feature_distillation_loss
```

Inputs:

```text
predictions:     [num_images, qwen_image_tokens, teacher_feature_dim]
teacher_output:  {"features": [num_images, teacher_feature_dim, H_teacher, W_teacher]}
token_grid_hw:   (H_qwen, W_qwen)
```

The teacher feature map is average-pooled to the Qwen image-token grid:

```text
target = adaptive_avg_pool2d(teacher_features, (H_qwen, W_qwen))
target = flatten_hw_to_tokens(target)
```

The loss has two terms.

Feature L1:

```text
l1_loss = mean(abs(predictions - target))
```

Similarity L1:

```text
pred_norm = normalize(flatten(predictions), dim=-1)
target_norm = normalize(flatten(target), dim=-1)
pred_sim = pred_norm @ pred_norm.T
target_sim = target_norm @ target_norm.T
sim_loss = mean(abs(pred_sim - target_sim))
```

Total auxiliary loss:

```text
depth_teacher_loss =
    feature_l1_weight * l1_loss
  + feature_similarity_weight * sim_loss
```

Trainer total loss contribution:

```text
total_loss += depth_teacher_loss * depth_teacher_loss_scale
```

The default scale is read from:

```yaml
framework.depth_teacher_aux.loss_weight: 0.004
```

or can be overridden with:

```yaml
trainer.loss_scale.depth_teacher
```

## Similarity Subsampling

The similarity matrix is quadratic in token count.

Default:

```yaml
similarity_max_tokens: 4096
```

If the flattened token count exceeds the cap, tokens are sampled with `torch.randperm`.

If `train_step` or `similarity_sample_seed` is supplied, the subsampling generator is seeded as:

```text
seed = similarity_sample_seed + train_step
```

This keeps the estimator unbiased while making loss curves reproducible across diagnostic runs.

Set `similarity_max_tokens: 0` to disable subsampling.

## Gradient Behavior

The auxiliary head is trainable. The MoGe teacher is frozen.

By default, Qwen image-token hidden states are detached for an initial warmup:

```yaml
detach_vlm_steps: null
detach_vlm_fraction: 0.01
```

Resolved detach steps:

```text
max(detach_vlm_steps or 0, ceil(detach_vlm_fraction * trainer.max_train_steps))
```

During detach warmup:

```text
Qwen hidden states -> detach -> auxiliary head -> loss
```

After detach warmup:

```text
Qwen hidden states -> auxiliary head -> loss
```

This lets the projection head learn a reasonable teacher-space mapping before the auxiliary loss can update Qwen or Qwen LoRA parameters.

If Qwen is fully frozen and no LoRA is active, the aux path can still train the projection head. For that dry-run mode, the config must explicitly set:

```yaml
framework:
  depth_teacher_aux:
    allow_frozen_qwen: true
```

Otherwise the trainer raises, because an enabled aux path with no trainable Qwen path is usually a configuration error.

## Checkpoint Semantics

The auxiliary head is part of the VLA-JEPA model state dict:

```text
depth_teacher_aux_head.*
```

The MoGe teacher is not part of the state dict.

Checkpoint loading filters only this known compatibility boundary:

- missing `depth_teacher_aux_head.*` is allowed when loading an old checkpoint into an aux-enabled model
- unexpected `depth_teacher_aux_head.*` is allowed when loading an aux checkpoint into an aux-disabled model
- all other missing or unexpected keys still fail normally

## Config Surface

Relevant config block:

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

Optional operational knobs:

```yaml
moge_repo_path: /path/to/moge
eager_load_on_cpu: false
use_fp16: true
similarity_sample_seed: 0
allow_frozen_qwen: false
allow_zero_loss_scale: false
```

## Known Technical Constraints

The implementation currently assumes Qwen receives one image per view, corresponding to the last temporal frame. If Qwen is changed to consume multiple temporal frames per view, the teacher target path must be updated to pass the same frames through MoGe.

The direct loss assumes all images in a batch share the same Qwen token grid. Mixed-resolution image-token grids are rejected.

The teacher feature target uses average pooling from MoGe grid to Qwen grid. This matches the direct feature-distillation shape requirement, but it is still an approximation when the two grids differ.

For multi-node training, the teacher weights should be pre-staged to avoid every rank downloading the same model from Hugging Face Hub.

`compile_qwen_model: true` with `find_unused_parameters: true` is warned about because `torch.compile` and DDP unused-parameter traversal can interact poorly. This needs a real multi-GPU smoke test before long runs.

