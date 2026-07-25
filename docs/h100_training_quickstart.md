# Human H100x8 Training Quickstart

This is the supported path for a human starting a RealMan/LeRobot, LIBERO, or
canonical GCS run on one machine with exactly eight NVIDIA H100 GPUs. It does
not require tmux, systemd, a cloud service wrapper, generated environment
files, or an agent.

The default production profile is
[`scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_h100x8_b16_qwen35_2b_full_moge_vitb_vjepa_large.yaml`](../scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_h100x8_b16_qwen35_2b_full_moge_vitb_vjepa_large.yaml).
That YAML is authoritative. Docker and Accelerate are execution mechanisms;
they do not own training settings.

Ready-to-edit H100 entry profiles also exist for
[LIBERO](../scripts/config/h100/vlajepa_robot_ft_libero_plus_h100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml)
and
[canonical GCS](../scripts/config/h100/vlajepa_robot_ft_canonical_full_h100x8_qwen_full_rawddp_moge_vits.yaml).
Those small entry YAMLs use explicit `extends` composition: the named dataset
profile owns model/data/training semantics and
[`h100x8_runtime_base.yaml`](../scripts/config/h100/h100x8_runtime_base.yaml)
owns the container, hardware, helper, and recovery contract. The leaf lists
every intentional override. `plan` prints and hashes the fully resolved YAML;
launch passes that flattened YAML to training. Generic A100 configs without a
`runtime` contract are deliberately rejected by this H100 workflow.

Dataset support is intentionally explicit:

- RealMan/LeRobot and canonical GCS both support the config-owned automatic
  holdout policy: approximately 5–8% of episodes, rounded to a multiple of
  eight, capped at 128 episodes, with exactly 128 balanced eval windows.
- RealMan/LeRobot production profiles use `all_sources_exhaustive`: each
  logical epoch traverses every eligible row from every configured training
  source exactly once before bounded distributed tail padding. Corrupt rows
  fail the run instead of being replaced, and checkpoint resume restores the
  exact data-stream cursor. Canonical GCS keeps its separately validated
  streaming contract and does not claim this exact-cursor behavior.
- LIBERO is supported for H100 training and simulator rollout evaluation, but
  it does not yet have a launcher-managed immutable offline holdout artifact.
  Its profile therefore rejects holdout-policy keys instead of silently
  ignoring them.
- Any other loader/representation combination is rejected until it has a
  reviewed dataset contract.

## 1. Prepare the host

Install the NVIDIA driver, Docker, and the NVIDIA Container Toolkit. Confirm
that `nvidia-smi` shows exactly eight H100s and clone a reviewed, committed
VLA-JEPA revision.

Stage the configured dataset at the resolved plan's data/cache directory. For
the default RealMan profile this is:

```text
/mnt/vla-jepa/datasets/magna_training_data_with_interventions
```

LIBERO uses `/mnt/vla-jepa/datasets/LeRobot`. Canonical GCS uses
`/mnt/vla-jepa/datasets/canonical_gcs` as its local cache and additionally
checks out the exact `dataset-canonicalization` commit declared by its entry
profile. The scratch disk must have enough capacity for the dataset/cache,
Docker layers, Hugging Face cache, and checkpoints. The configured checkpoint
root is `/mnt/vla-jepa/checkpoints`.

For canonical GCS, install Google Cloud SDK on the host (the default expected
root is `/usr/lib/google-cloud-sdk`, or set `GCLOUD_SDK_ROOT`), authenticate
with `gcloud`, and stage a user-readable configuration at
`/mnt/vla-jepa/gcloud-config` (or set `GCLOUD_CONFIG_DIR`). The wrapper mounts
the CLI plus that configuration under the non-root container user's `HOME`,
not under `/root`. A `GOOGLE_APPLICATION_CREDENTIALS` file alone is not enough:
the canonical loader executes `gcloud storage cp`, so activate that service
account in the staged gcloud configuration first. Canonical `setup` and
`check` both require an active account and prove it can read the exact
`datasets.vla_data.gcs_access_probe_object` declared by the entry profile
before training. This bounded probe does not enumerate the whole canonical
bucket.

The Docker wrapper mounts the repository and `runtime.scratch_root`. Therefore
the launcher rejects any run root, dataset/cache root, or helper checkout
outside that scratch root instead of accepting a host path the container
cannot see.

## 2. Build the container and install pinned helpers

From the repository root:

```bash
./scripts/h100_training.sh setup
```

Select another profile by passing the same `--config` to every config-aware
command:

```bash
CONFIG=scripts/config/h100/vlajepa_robot_ft_libero_plus_h100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml
./scripts/h100_training.sh setup --config "$CONFIG"
```

`setup` performs the host hardware/runtime checks, creates the configured
scratch layout, builds the exact image recipe declared in `runtime`, and checks
out MoGe and V-JEPA 2 at the full commits declared in
`runtime.helper_repositories`. To reuse an already-built image:

```bash
./scripts/h100_training.sh setup --skip-build
```

## 3. Read the complete plan

```bash
./scripts/h100_training.sh plan
```

For a selected profile:

```bash
./scripts/h100_training.sh plan --config "$CONFIG"
```

This prints the resolved hardware, image, helper commits, model, policy
representation/dimensions, action horizon, dataset, subtask probability, batch size,
optimizer, learning rates, warmups, checkpoint/eval cadence, selection metric,
and the expected holdout/statistics binding. `plan` deliberately labels
data-contract artifacts `not_checked`, so a brand-new canonical run can be
inspected before its holdout manifest exists. A required config setting that is
absent, mistyped, or incompatible still causes a hard failure. The later
`check` and `start` commands strictly validate artifact bytes and provenance.

If you intentionally changed a RealMan or canonical profile or immutable data
split, rebuild its data contract once, review the result, and commit the
reviewed config/artifacts:

```bash
./scripts/h100_training.sh prepare --yes-rebuild-data-contract
git diff --check
git status --short
```

For a selected non-default profile, preserve the same config explicitly:

```bash
./scripts/h100_training.sh prepare --config "$CONFIG" --yes-rebuild-data-contract
```

Do not run `prepare` merely to make a check pass without reviewing why the
provenance changed.

`prepare` dispatches by dataset family. RealMan rebuilds its episode split and
OpenPI-compatible statistics. Canonical deterministically generates or
verifies its configured 128-observation balanced heldout manifest and refuses
selection drift.
LIBERO currently has no launcher-managed immutable holdout, so `prepare`
fails clearly as unnecessary; its profile disables both eval-before-train and
training-stream checkpoint selection.

## 4. Run the production preflight

The source checkout must now be clean and committed:

```bash
./scripts/h100_training.sh check
```

Use `--config "$CONFIG"` when a non-default profile is selected. `check` also
verifies canonical manifest/source hashes and the pinned canonicalization
checkout for GCS profiles.

This checks the config and artifact hash chain, exact helper commits, dataset
and checkpoint mounts, all eight H100 names and SM90 capabilities, launch port,
CUDA/MoGe imports, FlashAttention, and the Qwen 3.5 fast-linear-attention
backward probe. Training does not start if any gate fails.

## 5. Start training

Foreground mode is the simplest and shows logs directly:

```bash
./scripts/h100_training.sh start
```

For a selected profile:

```bash
./scripts/h100_training.sh start --config "$CONFIG"
```

For a named, detached Docker container:

```bash
./scripts/h100_training.sh start --detach
./scripts/h100_training.sh status
./scripts/h100_training.sh logs
```

An optional run ID may be supplied, but it must keep the config-owned prefix:

```bash
./scripts/h100_training.sh start \
  --run-id robot_ft_lerobot_magna_interventions_h100x8_b16_20260722_120000 \
  --detach
```

The printed Accelerate command receives one trainer argument: a resolved YAML.
There are no hidden dot-list hyperparameter overrides.

## Resume

Resume only from a full-state `steps_N` directory belonging to the original
run. The controller validates model, optimizer, scheduler, trainer state, and
all eight rank RNG states before launch:

```bash
./scripts/h100_training.sh resume \
  --checkpoint /mnt/vla-jepa/checkpoints/RUN_ID/checkpoints/steps_N \
  --detach
```

If a measured loader bottleneck requires changing the worker topology while
resuming, use a reviewed resume-runtime YAML instead of editing the original
training profile or passing a numeric command-line override:

```bash
./scripts/h100_training.sh resume \
  --config scripts/config/h100/vlajepa_robot_ft_lerobot_magna_hq_subtasks_delta_h100x8_b16_qwen35_2b_full_moge_vitb_vjepa_large.yaml \
  --checkpoint /data/mehul-vla-jepa/checkpoints/RUN_ID/checkpoints/steps_N \
  --resume-runtime-config scripts/config/h100/resume_runtime/magna_hq_delta_workers4.yaml \
  --detach
```

The checked-in fresh Magna HQ profile already owns the validated
`num_workers: 4` and `multiprocessing_context: forkserver` settings. The
resume-runtime example is for an older immutable run whose source profile
recorded the former one-worker/spawn topology; it is not needed to start a new
HQ run.

The resume-runtime file is a deliberately narrow operational contract. Schema
version 1 permits exactly the validated worker-topology correction:

```yaml
schema_version: 1
datasets:
  vla_data:
    num_workers: 4
    multiprocessing_context: forkserver
```

The separate `h100_resume_runtime.py` helper first runs the original launcher's
profile, artifact, hardware, and deep runtime checks. It then validates the
checkpoint and the frozen launcher's recorded SHA, loads the run's immutable
`config.yaml`, and only then applies the worker topology. The frozen
`h100_training.py`, `config.yaml`, and `config.json` remain unchanged. The
resulting `resume_invocations/steps_N-*.yaml` records the runtime YAML and
helper paths and SHA-256 values, current Git commit, UTC timestamp, container
image (and Docker image digest when available), plus the previous and resumed
worker count and multiprocessing context. The trainer independently
revalidates that metadata before it loads training state. Any additional key,
either half of the atomic correction, or use without `resume --checkpoint`,
fails closed.

For this Python 3.13 Magna workload, the correction was validated with the
real manifest-bound 360,052-frame training split through Accelerate at eight
ranks, four persistent workers per rank, and two real batches per rank.
`forkserver` avoids the load-sensitive `SemLock._rebuild` failure seen while
directly spawning 32 workers, without the unsafe practice of forking from
CUDA-initialized training ranks.

## Editing the run

Edit the YAML, not the launcher. In particular, the YAML owns:

- Docker image and build features;
- scratch, dataset, and checkpoint paths;
- GPU/process topology and precision;
- helper repository URLs and exact commits;
- dataset family and its loader-specific holdout/normalization contract;
- model, action/state representation, and horizon;
- prompt/subtask probability;
- loader workers and video backend;
- batch size, optimizer, learning rates, schedules, and loss scales;
- checkpoint, evaluation, retention, and best-model selection settings.

The human launcher accepts no numeric training override flags. That is
intentional: the reviewed YAML, holdout manifest, statistics, and source
commit form one auditable training contract. The only exception is the
repo-owned, schema-validated resume-runtime YAML described above; it can alter
only the validated DataLoader worker topology and is recorded separately
without changing the original contract.
