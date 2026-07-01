# VLA-JEPA Adapter For `allenai/vla-evaluation-harness`

This folder keeps the high-throughput benchmark integration separate from the
training code and robot deployment server.

## What It Does

- Runs one VLA-JEPA policy server.
- Accepts many concurrent simulator shards.
- Uses `PredictModelServer.predict_batch()` so simultaneous observations are
  grouped into one model forward pass.
- Uses chunk buffering so one policy call can supply a short action horizon.
- Records aggregate results through the harness without writing thousands of
  MP4 files by default.

## Folder Layout

- `model_server.py`: reusable VLA-JEPA harness adapter.
- `configs/model_server_libero_plus.yaml`: LIBERO/LIBERO-Plus action and
  observation convention.
- `configs/model_server_generic.yaml`: raw starting point for other benchmarks.
- `configs/libero_plus_*.yaml`: LIBERO-Plus benchmark configs.
- `scripts/run_server.sh`: starts only the policy server.
- `scripts/run_sharded_eval.sh`: generic sharded harness runner.
- `scripts/run_libero_plus_sharded.sh`: thin LIBERO-Plus wrapper around the
  generic runner.

## Local Setup

Clone the harness once:

```bash
git clone https://github.com/allenai/vla-evaluation-harness /tmp/vla-evaluation-harness
```

Install the lightweight harness package into both envs, or keep
`HARNESS_ROOT=/tmp/vla-evaluation-harness` and use the provided scripts, which
put `${HARNESS_ROOT}/src` on `PYTHONPATH`.

The model server uses the policy env:

```bash
POLICY_ENV=vla-jepa-py313-min
```

The simulator shards use the LIBERO-Plus env:

```bash
SIM_ENV=libero-plus
```

For local no-Docker LIBERO-Plus runs, the runner automatically uses:

```bash
LIBERO_PLUS_ROOT=/home/mehul/work/vjepa/libero_plus_work/src/LIBERO-plus
```

That source path is required on this machine because the editable `libero`
install is visible to pip but does not import unless the source root is on
`PYTHONPATH`.

If the simulator env is missing harness runtime deps, install the harness into
that env:

```bash
conda run -n libero-plus pip install -e /tmp/vla-evaluation-harness
conda run -n vla-jepa-py313-min pip install -e /tmp/vla-evaluation-harness
```

## Smoke Eval

```bash
cd /home/mehul/work/vjepa/VLA-JEPA
CKPT=/path/to/model.safetensors \
BENCH_CONFIG=eval_harness/vla_jepa/configs/libero_plus_smoke.yaml \
NUM_SHARDS=2 \
MAX_BATCH_SIZE=4 \
./eval_harness/vla_jepa/scripts/run_libero_plus_sharded.sh
```

The same run through the generic script is:

```bash
cd /home/mehul/work/vjepa/VLA-JEPA
CKPT=/path/to/model.safetensors \
BENCH_CONFIG=eval_harness/vla_jepa/configs/libero_plus_smoke.yaml \
SERVER_CONFIG=eval_harness/vla_jepa/configs/model_server_libero_plus.yaml \
NUM_SHARDS=2 \
MAX_BATCH_SIZE=4 \
CHUNK_SIZE=7 \
NUM_DDIM_STEPS=10 \
./eval_harness/vla_jepa/scripts/run_sharded_eval.sh
```

For protocol-only checks, use `libero_plus_tiny.yaml`; it runs one task for 20
steps and is not a meaningful success-rate eval.

## Full LIBERO-Plus Eval

```bash
cd /home/mehul/work/vjepa/VLA-JEPA
CKPT=/path/to/model.safetensors \
NUM_SHARDS=8 \
MAX_BATCH_SIZE=8 \
CHUNK_SIZE=7 \
./eval_harness/vla_jepa/scripts/run_libero_plus_sharded.sh
```

For larger machines, increase `NUM_SHARDS` until the server queue stays healthy,
then increase `MAX_BATCH_SIZE` to the throughput knee. The harness tuning guide
uses the rule `num_shards < 0.8 * model_supply / env_demand`.

## Accuracy/Speed Knobs

- `CHUNK_SIZE=7`: fastest default for the current LIBERO-Plus checkpoint.
- `CHUNK_SIZE=1`: re-query every step; slower, closer to the old receding eval.
- `MAX_BATCH_SIZE`: maximum observations per GPU forward.
- `MAX_WAIT_TIME`: how long the server waits for a partial batch before running
  inference.
- `NUM_DDIM_STEPS`: action-head denoising steps passed to VLA-JEPA.

The full config disables video recording. Use `libero_plus_smoke.yaml` for small
video-producing checks.

## Future Benchmarks

The harness supports more than LIBERO. To add one cleanly:

1. Add a benchmark YAML under `configs/`, using the harness benchmark import
   string and params for that simulator.
2. Start from `configs/model_server_generic.yaml`.
3. Set `image_keys`, `state_keys`, `send_state`, `state_dim`, `chunk_size`, and
   `action_dim` to match the checkpoint and benchmark.
4. Use `benchmark_profile: raw` unless the benchmark has a real convention
   transform. If it does, add a named profile in `model_server.py`.
5. Run `scripts/run_sharded_eval.sh` with `BENCH_CONFIG` and `SERVER_CONFIG`.

Do not reuse the LIBERO profile for another benchmark unless its action format
is exactly `[delta_xyz, axis_angle, close_positive_gripper]` and its masked
gripper dimension should remain normalized instead of min-max unnormalized.
