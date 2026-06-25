# Realman Policy Runner

This runner targets the `ogrealman_source_v3` checkpoint schema:

- images: `observation.images.head`, `observation.images.wrist_left`, `observation.images.wrist_right`
- state: `source.observation.state`, 19 dims
- action: `source.action`, 22 dims

Start the generic policy server:

```bash
cd /home/mehul/work/vjepa/VLA-JEPA
python deployment/model_server/server_policy.py \
  --ckpt_path checkpoints/robot_ft_lerobot_ogrealman_source_qwen35_08b_lora_moge_h50_b9_20260605_203520/final_model \
  --port 10093 \
  --use_bf16
```

Validate the server metadata:

```bash
python deployment/realman/run_realman_policy.py \
  --host 127.0.0.1 \
  --port 10093 \
  --check-only \
  --print-metadata
```

Run one dry inference from an `.npz` observation containing the three image arrays and a 19-dim state:

```bash
python deployment/realman/run_realman_policy.py \
  --host 127.0.0.1 \
  --port 10093 \
  --observation-npz /path/to/realman_observation.npz \
  --log-path /tmp/realman_policy.jsonl
```

For live hardware, pass a small adapter factory:

```bash
python deployment/realman/run_realman_policy.py \
  --host 127.0.0.1 \
  --port 10093 \
  --robot-module my_realman_adapter:create_robot \
  --num-steps 100 \
  --chunk-size 1 \
  --live \
  --log-path /tmp/realman_policy_live.jsonl
```

The adapter object must provide `capture_observation()` and may provide `connect()`,
`disconnect()`, and `send_action(...)`. By default, live actions are sent as a split
dictionary with arm, gripper, base, head, and lift fields. Use `--send-format vector`
to send the raw 22-dim vector instead.

## Yondu VR Teleop Bridge

The lightweight Realman bridge reuses the data-collection stack from
`YonduAI/yondu-vr-teleop`:

- observations come from `realman_lerobot.realman_robot.RealmanRobot.capture_observation()`
- RGB frames come from the same shared-memory `LocalRgbFrameSource` used by recording
- policy actions go back through `RealmanRobot.send_action()`, which sends arm joints,
  grippers, base velocity, head joints, and lift height through the teleop session

The teleop camera pipeline must already be publishing RGB shared-memory frames for
`head,left,right`, just like during collection. Point the bridge at the teleop repo:

```bash
export YONDU_VR_TELEOP_ROOT=/path/to/yondu-vr-teleop
```

Then run the policy against the robot:

```bash
python deployment/realman/run_realman_teleop_policy.py \
  --host 127.0.0.1 \
  --port 10093 \
  --num-steps 100 \
  --chunk-size 1 \
  --log-path /tmp/realman_policy_live.jsonl
```

The wrapper defaults to `--live` with confirmation before each replan. Add
`--no-confirm-each-replan` only after checking the first few actions.
