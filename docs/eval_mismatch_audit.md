# Eval Mismatch Audit

Active objective: investigate why LIBERO sim eval did not match training, fix the
cause, and do not close the investigation until rollout evidence and a commit/code
audit support the fix.

## Current High-Confidence Finding

The main train/eval mismatch found so far was future-frame leakage into Qwen
policy conditioning.

Before the fix, Qwen tensor fast path selected the last frame of the training
video tensor. On the LIBERO-plus production config that meant Qwen saw offset
`+2` while the action target starts at offset `0`.

Current production config/dataloader evidence from the A100 cloud image:

```text
video_delta_indices: [0, 1, 2, 3, 4, 5, 6, 7]
action_delta_indices: [0, 1, 2, 3, 4, 5, 6]
video_target_shift_steps: 2
compact_union_offsets: [-5, -4, -3, -2, -1, 0, 1, 2]
context_offsets: [-5, -4, -3, -2, -1, 0]
target_offsets: [-3, -2, -1, 0, 1, 2]
qwen_current_offset: 0
```

The current code now resolves semantic `"current"` to the last frame of the
context clip, so Qwen sees offset `0`, not the future target frame.

Relevant code:

- `starVLA/model/framework/VLA_JEPA.py`
  - `_current_observation_frame_index`
  - `_resolve_video_frame_index`
  - `_frames_for_depth_teacher`
  - `_qwen_observation_frame_index`
  - `_build_qwen_inputs_from_video_tensor`
  - `_split_compact_videos`
- `starVLA/dataloader/gr00t_lerobot/datasets.py`
  - `_build_shifted_video_views`
  - compact decode spec path in `__getitem__`
- `starVLA/dataloader/lerobot_datasets.py`
  - `make_LeRobotSingleDataset` passes configured `video_horizon`.

## Current Run State

Corrected run:

```text
/mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033
```

Last checked:

```text
step: 960
action_loss: 0.12021
wm_loss: 0.684851
depth_teacher_loss: 0.452175
total_loss: 0.267223
wall_step_time: 3.38738 sec
samples_per_sec: 37.7874
epoch: 0.0416197
samples_seen: 122880
checkpoint_count: 0
```

Scheduled `steps_2500` had not been saved at the time of this audit, so the
first corrected rollout below uses an interim `steps_1015` checkpoint created
through the trainer's force-checkpoint hook.

## Corrected Rollout Evidence

An interim corrected checkpoint was forced with the trainer's
`FORCE_CHECKPOINT` hook so the eval path could be tested before scheduled
`steps_2500`.

```text
checkpoint: /mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033/checkpoints/steps_1015
artifact: model.safetensors
trainer_state: present
task: libero_goal / turn_on_the_stove
language: turn on the stove
eval mode: receding
num_ddim_steps: 10
num_trials: 1
max_steps: 1200
result: 1/1 success
video: /mnt/vla-jepa/logs/framefix_steps_1015_turn_on_stove_base_smoke_eglfix/rollout_task0000_turn_on_the_stove_178d22d2_episode0_success.mp4
local mirror: /home/mehul/work/vjepa/eval_videos/framefix_steps_1015_turn_on_stove_base_smoke_eglfix/
```

Visual inspection:

- video length: 71 frames at 10 FPS, about 7.1 seconds
- start: robot above the tabletop with stove knob visible
- middle: arm moves down toward the stove control area
- end: gripper is down at the stove control area and LIBERO marks success

This proves the corrected Qwen-current-frame path can run through the deployed
policy server and complete at least one base LIBERO rollout. It does not by
itself prove full benchmark quality; it is an early step-1015 checkpoint.

Old frame-leaking comparison artifacts already on disk:

```text
old checkpoint/run: /mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_3ep_20260626_082015
old eval checkpoint: /mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_3ep_20260626_082015/eval_steps_50000/pytorch_model.pt
same task success: /mnt/vla-jepa/logs/libero_best_step50000_base_script_20260628_221714/rollout_task0000_turn_on_the_stove_178d22d2_episode0_success.mp4
same task success: /mnt/vla-jepa/logs/libero_best_step50000_fa2_parallel_ddim10_long_20260628_225846/evals/gpu0_p10093_turn_on_the_stove_20260628_230047/rollout_task0000_turn_on_the_stove_178d22d2_episode0_success.mp4
same task failure: /mnt/vla-jepa/logs/libero_best_step50000_fa2_base_ddim10_long_clean_20260628_225006/20260628_225006_turn_on_the_stove/rollout_task0000_turn_on_the_stove_178d22d2_episode0_failure.mp4
same task failure: /mnt/vla-jepa/logs/libero_best_step50000_base_stove_ddim8_20260628_223124/rollout_task0000_turn_on_the_stove_178d22d2_episode0_failure.mp4
same task failure: /mnt/vla-jepa/logs/libero_best_step50000_base_stove_ddim4_20260628_222742/rollout_task0000_turn_on_the_stove_178d22d2_episode0_failure.mp4
```

The old leaked model can sometimes solve this single easy task under the later
eval wrapper, so the comparison is not a clean binary failure/success split.
The stronger evidence for the fix remains the code/data alignment: Qwen no
longer receives the future frame that the action target is supposed to predict.

## Verified So Far

- `git diff --check` on the current worktree passed before adding this note.
- Cloud Docker tests passed:

```text
pytest -q tests/test_checkpoint_paths.py deployment/trossen/test_pipeline.py tests/test_vla_jepa_frame_selection.py
18 passed, 1 warning
```

- Cloud Docker tests after canonical padding fix passed:

```text
pytest -q tests/test_canonical_pyav_threads.py tests/test_training_action_chunking.py tests/test_lerobot_no_base_padding.py tests/test_vla_jepa_frame_selection.py
29 passed, 1 warning
```

- Final focused cloud Docker suite after the preprocessed padding fix passed:

```text
pytest -q tests/test_checkpoint_paths.py deployment/trossen/test_pipeline.py tests/test_vla_jepa_frame_selection.py tests/test_canonical_pyav_threads.py tests/test_preprocessed_optional_labels.py tests/test_training_action_chunking.py tests/test_lerobot_no_base_padding.py tests/test_geometry_teacher.py tests/test_vla_jepa_action_context.py
78 passed, 2 skipped, 1 warning
```

- Local pytest cannot validate this repo because the host environment is not
  installed with `starVLA` or `torch`; use the cloud Docker image for final
  verification.

- Interval checkpoint resolution now supports nested `model.safetensors`.
- `read_mode_config` now finds the run root by searching parent directories for
  `config.yaml`/`dataset_statistics.json`, so nested Accelerate checkpoints work.
- LIBERO production action stats show the gripper dimension is binary and masked:

```text
action.min[-1] = -1.0
action.max[-1] = 1.0
action.mask[-1] = false
```

- Current eval adapter binarizes LIBERO gripper commands as `-1` open, `+1`
  close, matching robosuite/LIBERO command convention.
- V-JEPA world-model path appears aligned:
  - policy/Qwen sees context ending at offset `0`
  - V-JEPA encoder input receives context `[-5..0]`
  - V-JEPA target branch receives future-shifted target `[-3..2]`
  - target encoding is used for world-model loss, not action-head conditioning

## Issues Found During Audit

1. Historical whitespace issue in commit `9041a76127a9abf62fee3e8499af0e21d7a29aa7`.

   `git show --check` reports:

   ```text
   deployment/realman/__init__.py:2: new blank line at EOF.
   ```

   Current worktree fix: removed the extra blank line from
   `deployment/realman/__init__.py`.

2. Historical future-frame bug source outside the requested two-month window.

   Earlier investigation identified commit:

   ```text
   e58ac6be9fcdad10dac0199420ca82f04497633f
   Add cached frame-index path for rank-side video decode
   ```

   That commit introduced hard-coded `-1` frame selection in the tensor fast path.
   It predates `2026-04-29`, but it is the root cause of the LIBERO eval mismatch
   currently being fixed.

3. `2f15cb4` created the LIBERO-plus production configs with
   `video_target_shift_steps: 0`.

   With the old Qwen tensor path, this meant the model built Qwen inputs from the
   last frame of the `[0..7]` observation clip while predicting actions `[0..6]`.
   That is direct future-observation leakage into the action head. The current
   production config changes to `video_target_shift_steps: 2` and
   `qwen_observation_frame_index: current`, which resolves Qwen to offset `0`.

4. Current worktree canonical-streaming padding gap.

   The canonical GCS dataset already edge-clamped near-episode-end action rows,
   but did not return `action_is_pad`, so those repeated tail targets could
   still contribute to action loss. LeRobot/Realman already return
   `action_is_pad` and VLA-JEPA already masks it. Current worktree fix adds the
   same mask to `CanonicalSubsetVLADataset._sample_context` /
   `_sample_from_context` without changing dataset files or action values.

5. Current worktree preprocessed-subtask padding gap.

   `PreprocessedSubtaskVLADataset.__getitem__` also edge-clamped action windows
   near episode ends without returning `action_is_pad`. This affected the older
   preprocessed Trossen-style path, not the active LIBERO-plus run. Current
   worktree fix emits `action_is_pad` and preserves it in
   `PreprocessedSubtaskCollator`, so fixed-size chunks stay fixed while padded
   tail steps are masked from action loss.

6. `git show --check` historical formatting findings.

   The last-two-month commit scan reported only whitespace/EOF issues:

   ```text
   752fe9f docs/depth_teacher_aux.md: new blank line at EOF
   6771dcc scripts/build_decord_gpu.sh: trailing whitespace
   9041a76 deployment/realman/__init__.py: new blank line at EOF
   ```

   Current worktree removes these formatting issues. No functional bug was tied
   to those `git show --check` findings.

## Reviewed Commit Notes

- `2f15cb4` (`Stabilize LIBERO and robot training pipeline`)
  - Created LIBERO-plus full-Qwen configs.
  - Added auto V-JEPA predictor dimension/head resolution.
  - Added PyAV seek fast path and stable stats key ordering.
  - Risk found: the initial LIBERO-plus config used `video_target_shift_steps: 0`,
    which combined with the old Qwen `-1` frame selector created future leakage.
  - No separate action-dim or gripper convention bug found in this commit.

- `5ef0526` (`Prepare LIBERO-plus A100 benchmark training`)
  - Switched production training from monolithic `libero_plus`/`v3.0` to
    `libero_plus_4suite` with auto version detection.
  - Added dataset/trainer timing instrumentation.
  - Live cloud check confirmed this initializes four datasets:
    `libero_plus_10`, `libero_plus_goal`, `libero_plus_object`,
    `libero_plus_spatial`.
  - No frame-ordering change found in the line review. The timing additions do
    not change sample contents.

- `40d6cab` (`Add blockwise VLA action context`) and `80ccf7e`
  (`Clarify action context mask naming`)
  - Added action-head context filtering and blockwise masks on both training and
    inference paths.
  - Current line review: mask orientation is query-by-key and implements
    `key_block <= query_block`, as intended.
  - No separate train/eval mismatch found here.

- `9910a11` (`Align VLA configs with state token flow`)
  - Ensured state/embodied action placeholders are placed before auxiliary
    V-JEPA/geometry tokens, and empty replacements are used when a branch is
    disabled.
  - Current line review: this prevents auxiliary prompt tokens from leaking into
    action-head query slots in the causal Qwen3.5 path.
  - No issue found in current configs.

- `78eddd4` (`Implement RTC action conditioning`)
  - Added RTC time conditioning, prefix masks, and masked loss reduction.
  - Current LIBERO-plus production config has `rtc_training.enabled: false`, so
    RTC is not active in the mismatch under investigation.
  - No disabled-path side effect found in the line review.

- `6771dcc` and `3fc56c8` (canonical streaming dataset / PyAV stability)
  - Added the GCS canonical streaming dataset, bounded PyAV thread count,
    lock-protected downloads, retries, corrupt-video skip/retry handling, and
    dataloader sweep tooling.
  - Current line review found one loss-correctness gap: canonical streaming
    clamped tail action rows but did not emit `action_is_pad`. Fixed in the
    current worktree and covered by `tests/test_canonical_pyav_threads.py`.

- `90ee5f5` and `884f52e` (geometry teacher / query mode)
  - Added the MoGe query teacher, teacher feature-dim validation, and training
    guards for query mode.
  - Current line review: query mode uses dedicated geometry tokens and image
    tokens, not reused action tokens; teacher features are shape-checked against
    config metadata. Inference-only loading now skips MoGe weights but preserves
    enough metadata for policy construction.
  - No V-JEPA/MoGe future leakage found after `_frames_for_depth_teacher` was
    changed to use the semantic current observation frame by default.

- `98a7196` (training launcher cleanup)
  - Centralized launcher environment defaults in `scripts/lib/training_env.sh`.
  - Current line review: no sample, frame, action, or normalization behavior
    changes. Risk is operational only; launchers must still pass the intended
    config and override env vars.

- `8878d5c` and `f71e409` (metric scalar sync deferral)
  - Delayed GPU-to-CPU scalar conversion until logging.
  - Current line review: the VLA-JEPA trainer keeps `.detach()` references and
    `_log_metrics` converts at logging boundaries. The cotrain file modified by
    `f71e409` was later retired by `a224fbb`, so it has no current training
    effect.

- `792520b` and `a224fbb` (legacy training cleanup)
  - Removed stale cotrain entrypoints and old unused code.
  - Current active training entrypoint remains
    `starVLA/training/train_starvla.py`; no current import path depends on the
    retired files.

- `c82e2fc` (missing preprocessed RABC labels)
  - Made optional labels safe when preprocessed metadata does not include RABC
    or mistake fields.
  - Current line review found the preprocessed action-tail `action_is_pad` gap
    described above. Fixed in the current worktree and covered by
    `tests/test_preprocessed_optional_labels.py`.

- `31eecd7` (separate Qwen and V-JEPA camera views)
  - Added canonical camera slot selection so Qwen can receive all available
    views while V-JEPA/geometry can use the fixed view set.
  - Current line review: geometry alignment maps selected Qwen view indices back
    to V-JEPA view indices before auxiliary loss. No duplicated-missing-view
    conditioning bug found.

- `d87edb8` (plain subtask label prompt append)
  - Canonical streaming appends cleaned subtask text to language with a separator
    when labels exist.
  - Current line review: append is idempotent and returns the original task when
    no label exists. It does not affect LIBERO-plus, which has no subtask labels.

- `5fbc7bb` (LeRobot PyAV compatibility)
  - Added PyAV decode support and bounded thread settings.
  - Current line review: action padding from LeRobot transform masks is
    propagated into `action_is_pad`; PyAV fallback/retry paths remain
    conservative and should skip/retry corrupt samples instead of silently
    training on bad frames.

- `306e6ae` (Qwen3-VL backend and blockwise attention)
  - Added Qwen3 routing and FlexAttention blockwise support while keeping
    Qwen3.5 as a config rollback path.
  - Current line review: Qwen3 DeepStack custom feature extraction forwards
    visual metadata; Qwen3.5 rejects Qwen-internal blockwise attention and keeps
    the causal VLM path. Action-head block ids still implement `key_block <=
    query_block`.

- `4d86ba3`, `f0d8734`, `1263647`, and `7b6e11b` (Docker/cloud runtime)
  - Added the Python 3.13 CUDA image, Docker run wrapper, raw DDP A100 launcher,
    scratch/cache mounts, and GCP startup helpers.
  - Current line review: the active run uses raw DDP, no ZeRO-3, no torch
    compile by default, scratch/cache mounts are explicit, and optional fast
    attention builds are controlled by Docker build args. No data-path bug found.

- `84abf0e`, `4a98407`, and `21205d3` (corrupt canonical videos / sidecar tails)
  - Added corrupt-video retries, excluded a known bad AgiBot task, and handled
    sidecar tail windows.
  - Current line review found the canonical `action_is_pad` mask gap described
    above. The current fix masks sidecar tail repeats while keeping fixed-size
    chunks and unchanged dataset files.

- `5be5766` (Qwen3.5 LoRA state-token path)
  - Added prompt-label helpers, state placeholder tokens, LoRA configs, and
    evaluation utilities.
  - Current line review: state token replacement is differentiable when Qwen is
    trainable/LoRA-active; no lingering `torch.no_grad` bug was found in the
    state replacement path.

- `9041a76` (Realman no-base and LIBERO smoke setup)
  - Added Realman 19D no-base action view, deployment expansion to 22D robot
    commands, server logging, guard retries, and LIBERO smoke config.
  - Current line review: model-facing action is
    `source.action[0:16] + source.action[19:22]`; deployment inserts zero base
    velocity at dims `16:19`. The guard is Realman-metadata gated and does not
    affect LIBERO eval.

- `2f15cb4` and `5ef0526` (LIBERO/robot stabilization and A100 benchmark)
  - Added LIBERO-plus configs, interval checkpoint compatibility, eval wrapper
    fixes, timing instrumentation, and the A100 production run setup.
  - Risk found: these configs made the old `-1` Qwen tensor selector especially
    harmful, because Qwen could see a future frame while actions started at the
    current frame. Current worktree fixes the selector and sets
    `video_target_shift_steps: 2` / `qwen_observation_frame_index: current`.

## Commit Window To Review

Commit range requested by the user:

```bash
git log --since='2026-04-29' --reverse --oneline
```

Current list:

```text
90ee5f5 chore: checkpoint before cleanup
98a7196 chore: clean up training launchers
752fe9f docs: document depth teacher auxiliary loss
f78fe07 docs: focus geometry teacher technical design
7e4c90c docs: restore geometry teacher design context
78eddd4 Implement RTC action conditioning
6771dcc Prepare canonical GCS full-training environment
2d7d7c3 Merge pull request #5 from YonduAI/codex/canonical-gcs-full-training-env
3fc56c8 Stabilize canonical streaming dataloader
f16a6f4 Merge pull request #6 from YonduAI/codex/canonical-dataloader-stability
8878d5c Defer VLA-JEPA metric scalar sync to logging
f71e409 Defer cotrain metric scalar sync to logging
792520b Retire stale StarVLA cotrain entrypoint
a224fbb Remove legacy unused training paths
dd50022 Tidy generated artifacts and stale README notes
3b5f0a4 Rewrite README around current training pipeline
09ae128 Merge pull request #7 from YonduAI/codex/canonical-dataloader-stability
649f106 Redo architecture diagram and doc to match current pipeline
c82e2fc Handle missing preprocessed RABC labels
31eecd7 Support separate Qwen and V-JEPA camera views
d87edb8 Append plain subtask labels to canonical prompts
5fbc7bb Add LeRobot PyAV compatibility
3e716b0 Merge pull request #8 from YonduAI/codex/canonical-dataloader-stability
884f52e Add query geometry teacher and Python 3.13 setup
40d6cab Add blockwise VLA action context
80ccf7e Clarify action context mask naming
9910a11 Align VLA configs with state token flow
39a8b3f Merge pull request #9 from YonduAI/codex/canonical-dataloader-stability
306e6ae Add Qwen3-VL backend and blockwise attention
4d86ba3 Add cloud training Docker runtime
f0d8734 Optimize A100 raw DDP training setup
1263647 Fix raw DDP production launcher env
84abf0e Skip corrupt canonical videos during training
4a98407 Exclude corrupt AgiBot brush-water task
21205d3 Handle canonical sidecar tail retries
5be5766 Add Qwen3.5 LoRA state-token training path
9041a76 Add Realman no-base support and LIBERO-plus smoke setup
7b6e11b Mount scratch root in Docker training runner
2f15cb4 Stabilize LIBERO and robot training pipeline
5ef0526 Prepare LIBERO-plus A100 benchmark training
```

## Risk Classification For Line Review

High risk, must be line-reviewed before completion:

- `78eddd4` RTC action conditioning
- `6771dcc` canonical GCS full-training environment
- `3fc56c8` canonical streaming dataloader stability
- `31eecd7` separate Qwen and V-JEPA camera views
- `5fbc7bb` LeRobot PyAV compatibility
- `884f52e` query geometry teacher and Python 3.13 setup
- `40d6cab` blockwise VLA action context
- `9910a11` state token flow configs
- `306e6ae` Qwen3-VL backend and blockwise attention
- `f0d8734` A100 raw DDP setup
- `84abf0e`, `4a98407`, `21205d3` corrupt-video handling
- `5be5766` Qwen3.5 LoRA state-token path
- `9041a76` Realman no-base and LIBERO smoke setup
- `2f15cb4` LIBERO/robot stabilization
- `5ef0526` LIBERO-plus A100 benchmark training

Medium risk:

- `90ee5f5` checkpoint before cleanup
- `98a7196` launcher cleanup
- `8878d5c`, `f71e409` metric sync deferral
- `a224fbb` remove legacy paths
- `c82e2fc` optional RABC labels
- `d87edb8` subtask label prompt append
- `4d86ba3`, `1263647`, `7b6e11b` Docker/cloud launch path changes

Low risk / docs-only unless docs contradict code:

- `752fe9f`, `f78fe07`, `7e4c90c`
- merge commits with no direct diff
- `dd50022`, `3b5f0a4`, `649f106`

## Completion Summary

Completed for this investigation:

- Wrote the sim-eval continuation runbook:
  `docs/libero_sim_eval_agent_runbook.md`.
- Reviewed the current changed code paths and behavior-changing commits in the
  last-two-month window.
- Fixed the canonical streaming `action_is_pad` gap.
- Fixed the preprocessed-subtask `action_is_pad` gap.
- Ran the focused cloud Docker test suite:
  `78 passed, 2 skipped, 1 warning`.
- Forced and evaluated corrected checkpoint `steps_1015`.
- Inspected the corrected rollout video frames.
- Compared against old frame-leaking same-task eval artifacts on disk.

Residual risk:

- The corrected rollout evidence is one easy base LIBERO task from an early
  checkpoint. It proves the train/eval frame-alignment fix can deploy and
  complete a rollout, but not full LIBERO-plus benchmark quality.
- The old leaked step-50000 model also sometimes succeeds on this same easy
  task, so the strongest proof of the fix is the code/data alignment plus the
  absence of future-frame conditioning, not a binary old-fail/new-pass result.
