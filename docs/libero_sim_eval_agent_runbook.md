# LIBERO Sim Eval Agent Runbook

This handoff is for continuing the LIBERO-plus checkpoint/eval investigation on the
A100x8 cloud machine after training starts saving checkpoints.

## Current Context

- Host: `a100x8-colombus.us-east5-a.yondu-general-workspace`
- Cloud repo: `/mnt/vla-jepa/src/VLA-JEPA`
- LIBERO install: `/mnt/vla-jepa/src/LIBERO-plus`
- Docker image: `vla-jepa:py313-cu130-a100-libero-eval`
- Training tmux: `libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033`
- TensorBoard tmux: `libero_plus_tensorboard`
- TensorBoard port: `6008`
- Corrected run root:
  `/mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033`
- Old comparison run root:
  `/mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_3ep_20260626_082015`

The current training run includes the frame-alignment fix:

- Qwen sees the semantic current observation frame, not the last future frame.
- MoGe/depth teacher uses the same semantic current observation frame.
- Shifted compact clips split context/target as `context = [:context_horizon]` and
  `target = [shift:]`.
- Training uses `repeated_diffusion_steps: 8`, `video_target_shift_steps: 2`,
  and `qwen_observation_frame_index: current`.

Do not mark the investigation complete just because the first eval runs. The
completion gate is at the end of this file.

## Check Training Progress

Use TensorBoard scalars instead of scraping the noisy rank progress bars:

```bash
ssh a100x8-colombus.us-east5-a.yondu-general-workspace \
  "docker run --rm --network host \
    -v /mnt/vla-jepa:/mnt/vla-jepa \
    -v /mnt/vla-jepa/src/VLA-JEPA:/workspace/VLA-JEPA \
    -w /workspace/VLA-JEPA \
    vla-jepa:py313-cu130-a100-libero-eval \
    python -c 'from pathlib import Path; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; run=Path(\"/mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033\"); ea=EventAccumulator(str(run/\"starvla\"), size_guidance={\"scalars\":0}); ea.Reload(); tags=ea.Tags().get(\"scalars\", []); [print(f\"{t}: step={ea.Scalars(t)[-1].step} value={ea.Scalars(t)[-1].value:.6g}\") for t in tags if ea.Scalars(t) and any(k in t.lower() for k in [\"loss\",\"mae\",\"samples\",\"step_time\",\"epoch\"])]'"
```

List checkpoints:

```bash
ssh a100x8-colombus.us-east5-a.yondu-general-workspace \
  'run=/mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033; find "$run/checkpoints" -maxdepth 1 -type d -name "steps_*" 2>/dev/null | sort -V | tail -20'
```

An interval checkpoint is usable when it contains at least:

- `model.safetensors`
- `trainer_state.json`

The deployment loader now accepts a run root, `final_model/`, an interval
checkpoint directory, or a concrete `pytorch_model.pt` / `model.safetensors`.

## Start A Policy Server For An Interval Checkpoint

First pick a checkpoint:

```bash
RUN=/mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033
CKPT=$RUN/checkpoints/steps_2500
```

Sanity-check that it resolves:

```bash
ssh a100x8-colombus.us-east5-a.yondu-general-workspace \
  'CKPT=/mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033/checkpoints/steps_2500; \
  docker run --rm --network host \
    -e CKPT="$CKPT" \
    -v /mnt/vla-jepa:/mnt/vla-jepa \
    -v /mnt/vla-jepa/src/VLA-JEPA:/workspace/VLA-JEPA \
    -w /workspace/VLA-JEPA \
    vla-jepa:py313-cu130-a100-libero-eval \
    python -c "import os; from deployment.model_server.checkpoint_utils import resolve_policy_checkpoint; print(resolve_policy_checkpoint(os.environ[\"CKPT\"]))"'
```

If training is still running, check GPU memory first:

```bash
ssh a100x8-colombus.us-east5-a.yondu-general-workspace nvidia-smi
```

The training job uses all 8 GPUs, so an eval policy server can slow training or
OOM if there is no free memory. Prefer waiting for a checkpoint boundary or use
the least busy GPU. Do not pass `--load_training_backbones`; eval should skip
training-only V-JEPA/MoGe teacher backbones.

Start the server in tmux, mapping one physical GPU into the container:

```bash
ssh a100x8-colombus.us-east5-a.yondu-general-workspace
cd /mnt/vla-jepa/src/VLA-JEPA

RUN=/mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033
CKPT=$RUN/checkpoints/steps_2500
PORT=10093
GPU=0
SESSION=libero_policy_steps_2500

tmux new -d -s "$SESSION" "
docker run --rm --gpus device=$GPU --network host \
  -e PYTHONPATH=/workspace/VLA-JEPA:/mnt/vla-jepa/src/LIBERO-plus \
  -e TOKENIZERS_PARALLELISM=false \
  -v /mnt/vla-jepa:/mnt/vla-jepa \
  -v /mnt/vla-jepa/src/VLA-JEPA:/workspace/VLA-JEPA \
  -v /mnt/vla-jepa/src/LIBERO-plus:/mnt/vla-jepa/src/LIBERO-plus \
  -w /workspace/VLA-JEPA \
  vla-jepa:py313-cu130-a100-libero-eval \
  python deployment/model_server/server_policy.py \
    --ckpt_path \"$CKPT\" \
    --host 0.0.0.0 \
    --port \"$PORT\" \
    --cuda 0 \
    --use_bf16
"

tmux capture-pane -pt "$SESSION" -S -80
```

Wait until the server logs show it is listening. Confirm the port:

```bash
timeout 2 bash -c 'cat < /dev/null > /dev/tcp/127.0.0.1/10093' && echo ok
```

## Run A Base No-Perturbation Eval

Start with the simplest known base task:

```bash
cd /mnt/vla-jepa/src/VLA-JEPA

RUN=/mnt/vla-jepa/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033
CKPT=$RUN/checkpoints/steps_2500

POLICY_PORT=10093 \
SUITE_NAME=libero_goal \
TASK_NAME=turn_on_the_stove \
TASK_LANGUAGE="turn on the stove" \
MAX_STEPS=1500 \
NUM_TRIALS=5 \
NUM_DDIM_STEPS=10 \
ACTION_EXECUTION_MODE=receding \
ACTION_ENSEMBLE=true \
RUN_ID=framefix_steps_2500_turn_on_stove_base \
./examples/LIBERO/run_libero_base_eval.sh "$CKPT"
```

Notes:

- `NUM_DDIM_STEPS` is inference denoising. It is separate from training
  `repeated_diffusion_steps: 8`.
- Use `ACTION_EXECUTION_MODE=receding` for first diagnosis because it asks the
  policy every step and avoids stale chunk execution hiding model behavior.
- Use `MAX_STEPS=1500` initially because previous rollouts may need more time.

Inspect outputs:

```bash
OUT=/mnt/vla-jepa/logs/framefix_steps_2500_turn_on_stove_base
tail -200 "$OUT/eval.log"
find "$OUT" -maxdepth 2 -type f | sort
```

Download videos locally if needed:

```bash
mkdir -p /home/mehul/work/vjepa/eval_videos/framefix_steps_2500_turn_on_stove_base
rsync -az a100x8-colombus.us-east5-a.yondu-general-workspace:/mnt/vla-jepa/logs/framefix_steps_2500_turn_on_stove_base/ \
  /home/mehul/work/vjepa/eval_videos/framefix_steps_2500_turn_on_stove_base/
```

## Expand To More Base Tasks

List available base tasks:

```bash
ssh a100x8-colombus.us-east5-a.yondu-general-workspace \
  'find /mnt/vla-jepa/src/LIBERO-plus/libero/libero/init_files/libero_goal -name "*.pruned_init" -printf "%f\n" | sed "s/.pruned_init$//" | sort'
```

Good follow-up tasks:

- `put_the_bowl_on_the_plate` / `put the bowl on the plate`
- `put_the_bowl_on_the_stove` / `put the bowl on the stove`
- `open_the_middle_drawer_of_the_cabinet` / `open the middle drawer of the cabinet`
- `push_the_plate_to_the_front_of_the_stove` / `push the plate to the front of the stove`

Run each task with a unique `RUN_ID`. Keep `NUM_TRIALS=5` for early signal, then
increase only after the first success or clearly improving behavior.

## What Counts As Evidence

For each checkpoint/task, save:

- checkpoint path and resolved artifact path
- policy server command and GPU
- eval command
- `eval.log`
- success count / trial count
- at least one video for success and one video for failure
- short visual read of whether the robot approaches, grasps, moves correctly, or times out

Do not rely only on scalar train loss. The original bug was a train/eval mismatch
caused by future-frame conditioning, so rollout video behavior is the decisive
signal.

## If Eval Still Fails

Check these in order:

1. Confirm policy server loaded the intended checkpoint:
   `deployment.model_server.checkpoint_utils.resolve_policy_checkpoint(CKPT)`.
2. Confirm no teacher models are loaded in eval:
   do not use `--load_training_backbones`.
3. Confirm action convention:
   compare dataset actions, model unnormalized actions, and LIBERO expected
   delta/absolute convention. Do not trust config alone.
4. Confirm observation frame:
   Qwen and MoGe should use semantic current frame, not future target frame.
   Tests: `tests/test_vla_jepa_frame_selection.py`.
5. Confirm state:
   eval passes `--with-state true`; server should receive robot state and Qwen
   state tokens should be enabled by config.
6. Compare against demo action replay:
   if replay succeeds but model rollout fails, physics is probably acceptable
   and the issue is model/data/action alignment.
7. Inspect videos:
   determine if the policy is stationary, moving in the wrong direction,
   opening/closing gripper incorrectly, or timing out after partial progress.

## Tests To Run After Code Changes

Use the cloud Docker image:

```bash
ssh a100x8-colombus.us-east5-a.yondu-general-workspace \
  'docker run --rm --network host \
    -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    -v /mnt/vla-jepa:/mnt/vla-jepa \
    -v /mnt/vla-jepa/src/VLA-JEPA:/workspace/VLA-JEPA \
    -w /workspace/VLA-JEPA \
    vla-jepa:py313-cu130-a100-libero-eval \
    pytest -q \
      tests/test_checkpoint_paths.py \
      deployment/trossen/test_pipeline.py \
      tests/test_vla_jepa_frame_selection.py \
      tests/test_canonical_pyav_threads.py \
      tests/test_preprocessed_optional_labels.py \
      tests/test_training_action_chunking.py \
      tests/test_lerobot_no_base_padding.py \
      tests/test_geometry_teacher.py \
      tests/test_vla_jepa_action_context.py'
```

If the host env is missing `torch` or an editable `starVLA` install, do not use
local pytest as the source of truth. Run the suite inside the cloud Docker image.

## Completion Gate For This Investigation

Do not mark the active goal complete until all of the following are true:

1. At least one corrected checkpoint has been rollout-evaluated in LIBERO sim.
2. Videos and logs have been inspected, not just generated.
3. The corrected run has been compared against the old frame-leaking run on the
   same base task setup.
4. If eval still fails, the failure mode has a concrete root cause or a bounded
   next experiment.
5. The code audit below has been completed and documented.

Required audit before goal completion:

- Review every changed line currently in the worktree.
- Review every commit from the last two months:

```bash
git log --since='2026-04-29' --reverse --oneline
git log --since='2026-04-29' --reverse --format='%H %ad %s' --date=short
```

- For each commit, inspect:

```bash
git show --stat --summary <commit>
git show --check <commit>
git show --find-renames --find-copies <commit>
```

- Focus especially on:
  - future-frame or target leakage into Qwen, MoGe, V-JEPA, or action head inputs
  - action normalization and absolute/delta conventions
  - state-token insertion and state dimensionality
  - blockwise/action-head attention masks
  - `action_is_pad` masking near episode ends
  - Realman no-base action slicing and expansion
  - PyAV/decord decode parity and frame order
  - checkpoint save/load/resume/eval compatibility
  - train/eval config drift

The audit output should be a short written report with file/line references for
every bug or risky shortcut found, and a statement of which targeted tests were
run. Only then is it reasonable to mark the goal complete.
