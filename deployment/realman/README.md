# Realman VLA-JEPA Deployment

The production path has one policy process on the GPU workstation and one
hardware-owning process on the robot:

```text
GPU workstation                                Realman robot
server_policy.py (WebSocket :10093) <-------- robot_unified_teleop.py
  VLA-JEPA checkpoint                           cameras + measured state
  normalized 18D action chunk                   safety limits + action execution
                                                 optional dataset recording
```

Do not run a second standalone robot adapter beside `robot_unified_teleop.py`.
The unified process must remain the sole owner of cameras, arms, grippers, head,
base, lift, takeover, and dataset recording.

## Current Magna Contract

The Magna checkpoint uses:

- cameras, in order: `head`, `wrist_left`, `wrist_right`
- state: 19D `source.observation.state`
- model action: 18D absolute joint targets
- action horizon: 50
- model action layout: 16 arm/gripper values followed by two head values
- robot command: 22D
- deployment expansion: insert zero base velocity at `16:19` and preserve the
  measured lift height at index `21`
- prompt: `reach into the bin, lift the chain, put it in the jig, then remove it
  from the jig and put it in the other bin`

The policy server reports this layout in `realman_action_contract`; the robot
client must reject incompatible action/state dimensions or non-absolute actions.

The image/state wire contract is also explicit in `realman_input_contract`:

- the robot captures the same `640x480` color streams used for data collection
- BGR camera buffers are converted to RGB before policy preprocessing
- each view is resized to `384x384` with OpenCV `INTER_LINEAR`, matching the
  production dataloader
- the three views are sent losslessly as msgpack NumPy `uint8` arrays with shape
  `[B, 3, 384, 384, 3]`
- the server routes `qwen_frames` through the same Qwen tensor fast path used in
  training, including its bilinear `384 -> 224` model resize
- normalized state is finite `float32` with shape `[B, 1, 19]`

Do not pass `--policy-image-size` for the normal Realman path. The robot reads
the required `384` frame size from checkpoint metadata and rejects mismatches.

## Start The Local Server

An interval checkpoint only needs `model.safetensors`. Keep the run-level
`config.yaml` and `dataset_statistics.json` above its `checkpoints/` directory:

```text
checkpoints/<run-id>/
  config.yaml
  dataset_statistics.json
  checkpoints/steps_<N>/model.safetensors
```

Start the server from the repository root. Training-only V-JEPA and MoGe
backbones are skipped by default. Keep the legacy action-output retry heuristic
disabled: it thresholds absolute joint-pose magnitude, so it can bias stochastic
samples but cannot determine whether a trajectory is safe or useful. The robot
client remains responsible for hard limits and per-command step limits.

Do not pass `--use_bf16` for the alignment baseline. Training retains FP32
parameters for the action head and state projector while using BF16 autocast;
the flag casts those parameters themselves to BF16.

```bash
conda run -n vla-jepa-py313-min --no-capture-output \
  python deployment/model_server/server_policy.py \
  --ckpt_path checkpoints/<run-id>/checkpoints/steps_<N> \
  --host 0.0.0.0 \
  --port 10093 \
  --disable_action_guard \
  --policy_output_log_path logs/realman_vlajepa_server.jsonl \
  --policy_input_image_dir logs/realman_vlajepa_inputs
```

Validate metadata without touching the robot:

```bash
conda run -n vla-jepa-py313-min --no-capture-output \
  python deployment/realman/run_realman_policy.py \
  --host 127.0.0.1 \
  --port 10093 \
  --check-only \
  --print-metadata
```

## Robot-Side Dry Run

Use the native VLA-JEPA transport, `realman`. `openpi` and
`--policy-server-address tcp://...` select a different ZMQ protocol and must not
be used for this server.

First request one action chunk without actuating or recording:

```bash
python robot_unified_teleop.py \
  --policy-server-kind realman \
  --policy-host 192.168.10.223 \
  --policy-port 10093 \
  --policy-instruction "reach into the bin, lift the chain, put it in the jig, then remove it from the jig and put it in the other bin" \
  --policy-autostart \
  --policy-dry-run \
  --no-policy-record-dataset \
  --policy-num-steps 1 \
  --policy-fps 20 \
  --policy-chunk-size 1 \
  --policy-max-live-chunk-size 1 \
  --policy-log-path logs/vlajepa_unified_policy_dry_run.jsonl
```

Inspect both workstation and robot JSONL logs. Confirm three fresh, correctly
ordered camera frames; a finite in-range 19D state; an 18D model output; a 22D
expanded robot command; exactly zero base velocity; and unchanged measured lift.
`--policy-dry-run` also disables policy-completion and shutdown safe-parking, so
the validation run must not actuate any subsystem.

The robot JSONL entry includes `policy_input` with frame/state shapes and dtypes.
The workstation JSONL records per-view hashes and optional PNG paths, normalized
state values, and output tensors. It also records action-guard acceptance/retry
details when that heuristic is explicitly enabled.

## Diagnostic Live Rollout

Begin with one action per replan. Keep operator takeover available:

```bash
python robot_unified_teleop.py \
  --policy-server-kind realman \
  --policy-host 192.168.10.223 \
  --policy-port 10093 \
  --policy-instruction "reach into the bin, lift the chain, put it in the jig, then remove it from the jig and put it in the other bin" \
  --policy-autostart \
  --policy-live \
  --policy-fps 20 \
  --policy-chunk-size 1 \
  --policy-max-live-chunk-size 1 \
  --policy-record-dataset \
  --policy-log-path logs/vlajepa_unified_policy.jsonl
```

Use one to four actions per replan while diagnosing model output and tracking.
Plain synchronous rollout starts at action row zero only after inference
finishes, so roughly `latency_seconds * policy_fps` leading rows are stale. Do
not use action horizon 50 as the execution chunk: it would run 2.5 seconds
open-loop at 20 Hz.

After the clean checkpoint passes offline evaluation, measure its high-percentile
server latency and choose a delay `L` with margin. Keep the model horizon at 50,
set both the execution chunk and delayed-unconditioned overlap to `L`, and
require `2 * L <= 50`. The previous FP32 server RPC reached 475 ms before
camera capture, payload construction, decoding, and action preparation, so use
`L=12` as the first measurement point rather than assuming it is sufficient:

```bash
python robot_unified_teleop.py \
  --policy-server-kind realman \
  --policy-host 192.168.10.223 \
  --policy-port 10093 \
  --policy-instruction "reach into the bin, lift the chain, put it in the jig, then remove it from the jig and put it in the other bin" \
  --policy-autostart \
  --policy-num-steps 0 \
  --policy-fps 20 \
  --policy-chunk-size 12 \
  --policy-max-live-chunk-size 12 \
  --policy-async-overlap \
  --policy-async-delay-steps 12 \
  --policy-allow-large-live-chunks \
  --policy-log-path "logs/vlajepa_eval_$(date +%Y%m%d_%H%M%S).jsonl"
```

This overlap mode is for the clean no-RTC checkpoint and sends no RTC payload.
Bootstrap holds and then executes rows `0:L`; later responses execute rows
`L:2L`. The complete observation/RPC/preparation worker must finish before the
absolute `L/fps` boundary with a 12.5 ms activation reserve at 20 Hz. Actual
camera and measured-state timestamps must lie within half a tick of tick zero,
and dispatch completion must remain within half a tick. A fixed
`--policy-action-index L` is a simpler synchronous fallback but is less robust
to latency variation.

If the operator explicitly wants no client-side rollout guards, replace
`--policy-allow-large-live-chunks` with
`--policy-disable-vla-rollout-safety`. That flag bypasses post-unnormalization
hard clipping, slew limits, the live chunk cap/guard, and the latency abort. It
does not disable shape/finite checks, observation freshness, disabled-subsystem
holds, robot-controller limits, or operator stops. It also does not bypass
sensor alignment, the `L/fps` pipeline deadline, activation reserve, or
absolute dispatch cadence. Normalized actions are training-aligned and
unclipped by default; `--policy-clip-normalized-actions` is an explicit
non-training-aligned diagnostic override.

The older `run_realman_policy.py` hardware-adapter mode remains useful for
offline `.npz` tests, but the unified teleop path above is the supported live
deployment path.
