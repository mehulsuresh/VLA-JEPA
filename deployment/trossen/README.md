# Trossen Live Deployment

Use only these files:

- `deployment/trossen/presets/robot_ft_trossen_vjepa21_small_a100x4_weekend_20260404_5ep.env`
- `deployment/trossen/scripts/start_policy_server_weekend.sh`
- `deployment/trossen/scripts/run_live_policy_weekend.sh`

Saved defaults for this checkpoint:

- checkpoint: `/home/mehul/work/vjepa/checkpoints/robot_ft_trossen_vjepa21_small_a100x4_weekend_20260404_5ep/final_model`
- instruction: `Put the shirt in the other bin`
- action mode: `absolute_qpos`
- state normalization: `min_max`
- action normalization: `min_max`
- action horizon: `7`
- live rollout: `fps=20`, `chunk_size=7`, `warmup_steps=10`, `num_steps=120`, `max_relative_target=5`

Why these defaults:

- The raw dataset `action` column is the commanded follower joint goal recorded at each control step, not a true delta action.
- The Trossen recording code stores the goal position directly in the dataset, and the generic LeRobot dataloader normalizes that raw column without converting it to deltas.
- To match training target semantics, interpret outputs as absolute commanded joint goals and keep the default Trossen relative-goal safety cap.
- The saved wrapper now defaults to `chunk_size=7` because the model predicts 7 actions, and `--confirm-each-replan` blocks before capturing the next observation so each Enter gates a fresh 7-action replan.

## What Runs Where

GPU machine:

- Runs the websocket policy server only.
- Needs the checkpoint and CUDA.

Robot control machine:

- Runs the live policy client only.
- Needs access to the Trossen arms, cameras, and the Yondu LeRobot install.

If both are the same machine, open two terminals on that machine and run both commands there.

## Exact Commands

### 1. On the GPU machine

```bash
cd /home/mehul/work/vjepa/VLA-JEPA
conda activate vla-jepa-vjepa21
bash deployment/trossen/scripts/start_policy_server_weekend.sh
```

This starts the policy server on `0.0.0.0:10093`.

### 2. On the robot control machine

If the policy server is on the same machine:

```bash
cd /home/mehul/work/vjepa/VLA-JEPA
conda activate vla-jepa-vjepa21
bash deployment/trossen/scripts/run_live_policy_weekend.sh
```

If the policy server is on a different GPU machine, pass that machine's IP:

```bash
cd /home/mehul/work/vjepa/VLA-JEPA
conda activate vla-jepa-vjepa21
bash deployment/trossen/scripts/run_live_policy_weekend.sh <GPU_MACHINE_IP>
```

Example:

```bash
bash deployment/trossen/scripts/run_live_policy_weekend.sh 192.168.1.10
```

## Optional Station Overrides

If your station does not match the Yondu defaults, pass hardware overrides after the host:

```bash
bash deployment/trossen/scripts/run_live_policy_weekend.sh 192.168.1.10 \
  --left-follower-ip 192.168.1.5 \
  --right-follower-ip 192.168.1.4 \
  --cam-high-serial 130322270184 \
  --cam-left-wrist-serial 218622274938 \
  --cam-right-wrist-serial 128422271347
```

## Output

The live runner writes rollout logs to:

`/home/mehul/work/vjepa/checkpoints/trossen_rollout_logs/live_policy_weekend.jsonl`
