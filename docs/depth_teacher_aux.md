# Geometry Teacher Auxiliary Loss

This document describes the query-only geometry teacher path in VLA-JEPA.

## Goal

The auxiliary branch distills frozen MoGe geometry features into Qwen using dedicated geometry query tokens. It does not regress raw depth pixels or normals.

The training signal is:

```text
Qwen image tokens + contextualized geometry tokens
  -> QueryGeometryTeacherHead
  -> predicted MoGe feature tokens

same RGB frames
  -> frozen MoGe teacher
  -> target MoGe feature tokens

loss = SmoothL1(predicted tokens, target tokens)
```

## Components

- `MoGeGeometryTeacher`: frozen MoGe wrapper kept outside the trainable model state dict.
- `QueryGeometryTeacherHead`: LingBot-style cross-attention resampler.
- `query_feature_distillation_loss`: SmoothL1 over teacher feature tokens.

The branch is disabled by default:

```yaml
framework:
  depth_teacher_aux:
    enabled: false
    mode: query
```

If enabled, only `mode: query` is supported.

## Qwen Tokens

When the branch is enabled, VLA-JEPA adds dedicated geometry tokens to the Qwen prompt:

```text
<|geometry_0|> ... <|geometry_7|>
```

By default this is repeated once per V-JEPA view:

```text
3 views x 8 geometry tokens = 24 geometry tokens per sample
```

These tokens go through Qwen with the real image tokens and become image-conditioned. They are separate from:

- `<|action_i|>` tokens used by the V-JEPA predictor
- `<|embodied_action|>` tokens used by the flow-matching action head

## Head Shape

For each view, the geometry head receives:

```text
Qwen image tokens + 8 contextualized geometry tokens
```

The head uses learned output queries and Perceiver-style cross-attention to produce:

```text
[num_images, 256, teacher_feature_dim]
```

The default config matches LingBot's token counts:

```yaml
num_task_tokens: 8
num_output_tokens: 256
query_num_layers: 1
query_num_heads: 4
query_dim_head: 32
query_ff_mult: 1.0
```

## Teacher Target

The teacher runs on the same last frame per view that Qwen sees:

```text
batch_videos[:, :, -1]
```

MoGe returns feature maps:

```text
[num_images, teacher_feature_dim, H_teacher, W_teacher]
```

Those maps are average-pooled to the query output token grid, usually `16x16 = 256`.

## Loss

The loss is:

```text
depth_teacher_loss = SmoothL1(predicted_256_tokens, teacher_256_tokens)
```

The trainer adds it with:

```text
total_loss += depth_teacher_loss * trainer.loss_scale.depth_teacher
```

The default scale comes from:

```yaml
framework.depth_teacher_aux.loss_weight: 0.004
```

or:

```yaml
trainer.loss_scale.depth_teacher: 0.004
```

## Gradient Behavior

MoGe is always frozen. The query head is trainable.

Qwen hidden states can be detached during an initial warmup:

```yaml
detach_vlm_steps: 5
detach_vlm_fraction: 0.01
```

Resolved detach steps:

```text
max(detach_vlm_steps or 0, ceil(detach_vlm_fraction * trainer.max_train_steps))
```

During warmup, the geometry head learns the teacher feature space before the auxiliary loss updates Qwen or Qwen LoRA parameters.

## Checkpoints

The trainable query head is saved under:

```text
depth_teacher_aux_head.*
```

The MoGe teacher is not saved in VLA-JEPA checkpoints.

Checkpoint loading allows only this known compatibility boundary:

- missing `depth_teacher_aux_head.*` when loading an old checkpoint into an aux-enabled model
- unexpected `depth_teacher_aux_head.*` when loading an aux checkpoint into an aux-disabled model

## Config Surface

```yaml
framework:
  depth_teacher_aux:
    enabled: true
    mode: query
    teacher_model: Ruicheng/moge-2-vits-normal
    moge_repo_path: /path/to/MoGe
    input_size: 224
    num_tokens: 256
    num_task_tokens: 8
    num_output_tokens: 256
    geometry_token_template: <|geometry_{}|>
    query_num_layers: 1
    query_num_heads: 4
    query_dim_head: 32
    query_ff_mult: 1.0
    query_smooth_l1_beta: 1.0
    teacher_feature_source: neck
    teacher_feature_level: 0
    teacher_feature_dim: auto
    frame_value_range: uint8
    loss_weight: 0.004
    detach_vlm_steps: 5
    detach_vlm_fraction: 0.01
    use_fp16: true
    allow_frozen_qwen: false
    allow_zero_loss_scale: false
```

The Qwen prompt must include `{geometry}` when the branch is enabled. If it is missing, VLA-JEPA appends the geometry tokens to the prompt automatically.
