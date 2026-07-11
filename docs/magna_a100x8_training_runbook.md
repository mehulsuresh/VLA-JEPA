# Magna A100x8 Training Runbook

This is the handoff checklist for preparing a fresh GCP `a2-ultragpu-8g` node
and launching the Realman Magna intervention run. Do not start the production
job until the final config and smoke results have been reviewed with the user.

## Agent Handoff Rules

Treat this file as the operational source of truth for this run. Before
provisioning, copying, rebuilding, or launching anything, inspect the existing
node and continue completed work rather than starting it again:

```bash
HOST=columbus-8xa100.us-east5-a.yondu-general-workspace

ssh "$HOST" '
  hostname
  uptime
  nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu,memory.used \
    --format=csv
  df -hT / /mnt/disks/ssd-array /mnt/vla-jepa 2>/dev/null || true
  tmux list-sessions 2>/dev/null || true
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" 2>/dev/null || true
  docker image inspect vla-jepa:py313-cu130-a100 \
    --format "image={{.Id}} created={{.Created}}" 2>/dev/null || true
  test ! -f /mnt/vla-jepa/logs/docker_build_magna_a100.exit || {
    printf "build_exit="
    cat /mnt/vla-jepa/logs/docker_build_magna_a100.exit
  }
  pgrep -af "train_starvla|accelerate launch|docker_build_training|rsync" || true
'
```

Follow these invariants:

- The local checkout is authoritative. The cloud checkout and image are
  disposable replicas; do not overwrite or reset the local worktree.
- Inspect `git status`, the deployed source manifest, tmux sessions, logs, and
  exit markers before inferring that a transfer, build, smoke, or run finished.
- A Docker build snapshots its context when the build starts. If source changes
  during a build, overlay the source again and rerun the build so the final
  image contains the reviewed code; cached dependency and CUDA build layers
  should make the second build short.
- Do not start production training until the user has reviewed the final
  settings, measured 8-GPU smoke throughput, ETA, and checkpoint backup plan.
- Do not stop or delete an A2 node while local SSD contains the only copy of a
  checkpoint. A reboot is different from a stop; verify GCP lifecycle behavior
  before any instance operation.
- Never kill an unfamiliar training, transfer, build, upload, or audit process.
  Establish ownership from its command, tmux session, and log first.
- Leave a final handoff with the exact commit, local diff manifest, image ID,
  dataset checksum result, commands, log paths, run id, and remaining blocker.

### Current Handoff Snapshot (2026-07-11 PDT)

This snapshot records what the previous agent actually completed. It is not a
substitute for checking exit markers and live processes with the command above.
Update this section before handing the machine to another agent.

Verified node state:

```text
GCE machine type              a2-ultragpu-8g
GPUs                          8x A100-SXM4-80GB, full NVLink mesh
Host CPUs / RAM / swap        96 / approximately 1.3 TiB / none
Scratch                       approximately 2.9 TiB RAID0 at /mnt/disks/ssd-array
Docker data root              /mnt/vla-jepa/docker
Runtime code commit           f2aafad (full hash recorded in the cloud manifest)
8-GPU smoke commit            4d263d2ab41df3895d2e46b83a86bc44bbe043bf
Image source commit           f2aafad (full hash recorded in the cloud manifest)
V-JEPA2 helper commit         204698b45b3712590f06245fbfba32d3be539812
MoGe helper commit            07444410f1e33f402353b99d6ccd26bd31e469e8
Last fully baked image ID     sha256:633e1a2a28550726531771b0dc888a83531ec1f599ee17a678f96373b45b6ccc
Image Python / Torch / CUDA   3.13.14 / 2.13.0+cu130 / 13.0
FlashAttention               2.8.3.post1, compiled only for SM80
```

The dataset transfer completed with an empty rsync dry run and a successful
SHA-256 check of every file. The manifest is
`/mnt/vla-jepa/logs/magna_training_data.sha256`; its own SHA-256 is:

```text
02d062e4cc7535b9794cd804f30ea0093b0ce1b4937e64cfacb30c33bebcc49a
```

Completed smoke evidence:

| Probe | Result | Evidence |
| --- | --- | --- |
| Single GPU, batch 1, two optimizer steps | Passed; peak allocated 22.83 GiB | `/mnt/vla-jepa/logs/magna_single_gpu_smoke_retry.log` |
| 8 GPUs, batch 16/rank | Rejected; first backward OOM at approximately 81.1 GiB | `/mnt/vla-jepa/logs/magna_a100x8_b16_smoke.log` |
| 8 GPUs, batch 14/rank | Rejected; step 1 passed, step 2 OOM in action-head SDPA | `/mnt/vla-jepa/logs/magna_a100x8_b14_smoke.log` |
| 8 GPUs, batch 12/rank, eight steps | Passed; observed memory later reached approximately 78.3 GiB | `/mnt/vla-jepa/logs/magna_a100x8_b12_smoke.log` |
| 8 GPUs, batch 13/rank | Rejected; step 1 passed, step 2 exhausted rank 7 at 81,158 MiB | `/mnt/vla-jepa/logs/magna_a100x8_b13_smoke.log` and `.exit` |
| 8 GPUs, batch 12/rank, 30 steps | Passed; steady throughput and loader test | `/mnt/vla-jepa/logs/magna_a100x8_b12_final_smoke_d3821a1_20260711_050611.log` |
| 8 GPUs, batch 12/rank, two-step clean exit | Passed; no resource-tracker warning or new semaphore names | `/mnt/vla-jepa/logs/magna_a100x8_shutdown_smoke_4d263d2_20260711_051742.log` |
| Full-state checkpoint and resume | Passed; saved step 1 and resumed all eight ranks through step 2 | `/mnt/vla-jepa/logs/magna_a100x8_checkpoint_smoke_4d263d2_20260711_052718.log` and `.resume.log` |
| Final 18D action config, 8 GPUs, batch 12/rank | Passed two optimizer steps; action statistics are 18D and state remains 19D | `/mnt/vla-jepa/logs/magna_a100x8_18d_smoke_f2aafad_20260711_0910.result.txt` |

Batch 12 is the measured upper bound: both batch 13 and batch 14 passed their
first step before later sample/token variation exhausted memory. Retain enough
headroom for real sample variation; do not retry a rejected batch merely
because its first step fit.

The verified commit and image contain:

- bounded Arrow-table caching and exact episode slicing for LeRobot v3 shards;
- PyAV nearest-frame conversion, which preserved decoded frame selection while
  reducing a representative decode call from 0.1470 s to 0.0812 s;
- correct epoch propagation through Accelerate's `DataLoaderShard`, removing a
  false sampler warning;
- idempotent DataLoader teardown without calling Python 3.13's private
  `resource_tracker._stop()` while semaphore finalizers are still live.

Local `pytest tests -q` passed with `141 passed, 1 skipped`; the final baked
cloud image passed with `140 passed, 2 skipped`. Use `pytest tests -q`;
bare `pytest -q` also collects optional simulation packages that are not part
of this training image.

The 30-step smoke outlasted the per-rank prefetch queue. Over its final 20
steps it averaged 3.6284 seconds per optimizer step, 26.48 global samples/s,
and 0.00062 seconds of visible data wait. Four workers per rank are therefore
enough; adding workers cannot improve a model-bound step while it does increase
startup time and host memory. The fullest GPUs reached 78,294 MiB in
`nvidia-smi`, leaving 2,883 MiB; PyTorch reported a 75.19 GiB peak allocation.
There were no OOM, NCCL, NaN/Inf, decode, worker-exit, or worker-respawn events.

At 76,644 total optimizer steps, the measured compute-only duration is about
77.25 hours (3.22 days). A production-format resumable checkpoint is 18.53 GB:
11.53 GB of optimizer state plus 7.00 GB of model weights. Saving it added
about 30.6 seconds. At a 2,500-step interval this is approximately 0.34% local
serialization overhead, and retaining three checkpoints uses about 55.6 GB.
The same artifact was loaded with `resume_load_optimizer_state: true`; all
eight ranks restored full state, began at step 1, completed step 2, and exited
cleanly. Network upload time and initial startup still add to the ETA.

The private resource-tracker shutdown hack was proven to cause the noisy
`sem_unlink` tracebacks. The fixed eight-rank exit test had no such warning and
left the POSIX semaphore-name count unchanged at 240. Those 240 stale names
predate the fixed test and include artifacts from force-stopped OOM probes.
They occupy negligible space; remove them only while no training/container or
other multiprocessing job is active.

The final backup preflight used the production uploader and destination prefix:

```text
gs://robotics-datasets-yonduai/gcloud/vla-jepa/checkpoints/preflight/magna_a100x8_checkpoint_smoke_4d263d2_20260711_052718
```

It uploaded checkpoint metadata, TensorBoard logs, and all 12 resumable-state
files. A clean restore downloaded `18,526,901,498` checkpoint bytes at about
`882 MiB/s`; every restored file matched the source SHA-256. Docker's
`model.safetensors` was initially mode `0600`, so commits `c96dc31` and
`f94d28d` moved uploader bookkeeping outside root-owned run directories and
made completed checkpoints/final models host-readable. The uploader now also
rejects unreadable artifacts before beginning a partial transfer.

The VM service account still has only the `devstorage.read_only` OAuth scope,
but `mehul@yonduai.com` is authenticated in the mounted
`/mnt/vla-jepa/gcloud-config`. VM-side write/read/delete and the complete
checkpoint round trip passed against the approved bucket. Do not stop this
local-SSD instance merely to change its service-account scopes.

The only remaining ordered work is:

1. Present the complete setting/throughput/ETA review to the user.
2. Launch production only after the user explicitly approves that review.

## Known Production Inputs

- SSH alias: `columbus-8xa100.us-east5-a.yondu-general-workspace`
- Local dataset:
  `/home/mehul/work/reward_model_small/magna_training_data_with_interventions`
- Cloud dataset:
  `/mnt/vla-jepa/datasets/magna_training_data_with_interventions`
- Cloud source checkout: `/mnt/vla-jepa/src/VLA-JEPA`
- Container image: `vla-jepa:py313-cu130-a100`
- Training config:
  `scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml`
- Launcher:
  `scripts/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh`

The verified dataset snapshot has 2,452,692 frames, 1,638 episodes, 93 video
files, 34.07 hours at 20 FPS, and three H.264 camera streams (`head`,
`wrist_left`, and `wrist_right`). `source.action` is 22D and
`source.observation.state` is 19D. Training deliberately omits source action
dimensions 16:19 (base velocity) and 21 (lift height), producing an 18D action
target while retaining measured lift height in the state input. The main
parquet shards already contain the integrated `subtask_index`, `valid_state`,
`valid_state_source`, and `task_id` columns. The `edits/` directory is
annotation provenance; the production loader does not need it as an overlay.

Expected intervention-label counts for this snapshot:

```text
valid_state=1  2,208,961
valid_state=0    243,731
```

With action horizon 50 and
`action_validity_invalid_run_length: 10`, an exact pass over this snapshot
produced these expected mask statistics:

```text
action_loss_mask_keep_ratio       0.8788677176
action_loss_mask_all_zero_ratio   0.0956765057
full_keep_window_ratio            0.8505107857
mean_kept_action_steps            43.9434 / 50
```

The production config must use the sustained-invalid-prefix mask and keep
RABC disabled. An all-zero action mask suppresses only the action loss for that
sample; V-JEPA/world-model and geometry losses must continue to train. Missing
labels deliberately fall back to an all-ones action mask. Prompt labels come
from `subtask_index`; the sentinel `__unlabeled__` and a genuine null label must
not be appended to the task prompt.

Exact-shape probes rejected batch sizes 16, 14, and 13 per rank. Batch 13 and
14 each passed their first step before a later batch reached approximately
81.1 GiB and OOMed. The measured production candidate is therefore batch 12
per rank, for a global batch of 96. With eight ranks, `drop_last: true`, and
three epochs, this gives `floor(2,452,692 / 96) = 25,548` optimizer steps per
epoch and 76,644 optimizer steps total.

LeRobot v3 low-dimensional data uses a bounded per-worker Arrow shard cache.
Magna sets `lerobot_v3_parquet_cache_size: 5`, matching its five active parquet
shards. Do not remove this setting: the old path converted the complete shared
parquet file to pandas for every shuffled sample, including a 340 MB,
1,785,890-row shard. The corrected path loads each shard as an Arrow table and
converts only the episode slice identified by LeRobot's
`dataset_from_index`/`dataset_to_index`. An exhaustive local check resolved all
1,638 episodes and 2,452,692 rows correctly. Cache size is per worker, is
bounded, and is cleared rather than serialized when workers are spawned.

## 1. Verify The Endpoint

Do not disable host-key checking. First confirm that the SSH alias IP matches
the live GCE instance:

```bash
gcloud compute instances describe columbus-8xa100 \
  --zone us-east5-a \
  --format='value(status,networkInterfaces[0].accessConfigs[0].natIP,machineType.basename())'

ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  columbus-8xa100.us-east5-a.yondu-general-workspace hostname
```

`accept-new` is appropriate only after the GCP lookup confirms the address. It
does not accept a changed key for an existing alias.

## 2. Bootstrap The Fresh A2 Node

Copy and run the committed bootstrap from the local authoritative checkout:

```bash
HOST=columbus-8xa100.us-east5-a.yondu-general-workspace
REPO=/home/mehul/work/vjepa/VLA-JEPA

scp "$REPO/deployment/gcp/startup-a2-training-node.sh" \
  "$HOST:/tmp/startup-a2-training-node.sh"
ssh -tt "$HOST" \
  'sudo env DOCKER_USERS="$USER" bash /tmp/startup-a2-training-node.sh'
```

On a new Debian A2 image, the first pass can install a newer kernel and request
a reboot before installing the NVIDIA driver. If that happens:

1. Wait for the GCE `lastStartTimestamp` and SSH uptime to change.
2. Re-copy the script because `/tmp` is a tmpfs and is empty after reboot.
3. Run the same bootstrap command again.
4. If the driver installation requests another reboot, reboot and rerun the
   bootstrap once more so the NVIDIA Docker runtime is configured.

The script is idempotent. It must leave the machine with:

```text
/dev/md0                         RAID0 over all eight local NVMe SSDs
/mnt/disks/ssd-array             approximately 2.9 TiB ext4
/mnt/vla-jepa                    symlink into the SSD array
/mnt/vla-jepa/docker             Docker data-root
/mnt/vla-jepa/{src,datasets,checkpoints,logs,hf,cache,tmp}
```

The full bootstrap transcript is `/var/log/training-node-bootstrap.log`. Copy
it to scratch after the final pass so it survives root-disk troubleshooting:

```bash
ssh "$HOST" \
  'sudo cp /var/log/training-node-bootstrap.log /mnt/vla-jepa/logs/ && sudo chmod 0644 /mnt/vla-jepa/logs/training-node-bootstrap.log'
```

Open a new SSH login after group membership changes, then verify:

```bash
ssh "$HOST" '
  set -e
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  nvidia-smi topo -m
  df -hT / /mnt/disks/ssd-array /mnt/vla-jepa
  docker info --format "root={{.DockerRootDir}} runtimes={{json .Runtimes}}"
  docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi -L
'
```

Require all eight A100 80 GB devices, a healthy NVLink topology, Docker root on
`/mnt/vla-jepa/docker`, and the `nvidia` runtime before continuing.

## 3. Synchronize The Authoritative Source

The local checkout may contain required uncommitted fixes. A plain clone of
`origin/main` is therefore not sufficient. Start with the matching Git commit,
then overlay every tracked and non-ignored file from the local worktree:

```bash
ssh -A "$HOST" '
  mkdir -p /mnt/vla-jepa/src
  if [[ ! -d /mnt/vla-jepa/src/VLA-JEPA/.git ]]; then
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" \
      git clone git@github.com:YonduAI/VLA-JEPA.git /mnt/vla-jepa/src/VLA-JEPA
  fi
  git -C /mnt/vla-jepa/src/VLA-JEPA fetch origin
  git -C /mnt/vla-jepa/src/VLA-JEPA checkout main
  git -C /mnt/vla-jepa/src/VLA-JEPA reset --hard e8f7be46e1a9cc51050b7d7468f3c0d25d479181
'

cd "$REPO"
git ls-files -co --exclude-standard -z | \
  rsync -a --from0 --files-from=- ./ "$HOST:/mnt/vla-jepa/src/VLA-JEPA/"
```

The `reset --hard` above is only for the fresh disposable cloud clone. Never
run it in the local user checkout. Before using this recipe with a later run,
replace the pinned commit with the current local `git rev-parse HEAD` value.
If local status contains tracked deletions, remove those same paths in the
cloud clone explicitly; `--files-from` cannot represent deletions.
The `-A` forwards the local SSH agent only for this connection; it does not
copy a GitHub key or token to the VM. Confirm `ssh-add -l` and
`ssh -T git@github.com` succeed locally before cloning.

If the private cloud clone cannot fetch because no forwarded GitHub credential
is available, keep credentials off the VM and transfer the missing Git objects
as a bundle after pushing locally. First require a clean cloud worktree and
record its current commit:

```bash
HOST=columbus-8xa100.us-east5-a.yondu-general-workspace
CLOUD_BASE="$(ssh "$HOST" \
  'git -C /mnt/vla-jepa/src/VLA-JEPA rev-parse HEAD')"
test -z "$(ssh "$HOST" \
  'git -C /mnt/vla-jepa/src/VLA-JEPA status --porcelain')"

git push origin main
git bundle create /tmp/vla-jepa-sync.bundle main "^${CLOUD_BASE}"
git bundle verify /tmp/vla-jepa-sync.bundle
scp /tmp/vla-jepa-sync.bundle "$HOST:/mnt/vla-jepa/logs/"

ssh "$HOST" '
  cd /mnt/vla-jepa/src/VLA-JEPA
  git fetch /mnt/vla-jepa/logs/vla-jepa-sync.bundle refs/heads/main
  git reset --hard FETCH_HEAD
  git status --short
  git rev-parse HEAD
'
```

Use `reset --hard` only for this verified disposable cloud replica, after the
local commit is on GitHub and the cloud status check is empty. Confirm local
`HEAD`, local `origin/main`, and cloud `HEAD` are identical afterward.

Record exactly what was deployed:

```bash
mkdir -p /tmp/magna_source_manifest
git rev-parse HEAD > /tmp/magna_source_manifest/base_commit.txt
git status --short > /tmp/magna_source_manifest/status.txt
git diff --binary HEAD > /tmp/magna_source_manifest/worktree.patch
scp -r /tmp/magna_source_manifest "$HOST:/mnt/vla-jepa/logs/"
```

Install helper checkouts at explicit revisions:

```bash
ssh "$HOST" '
  set -e
  test -d /mnt/vla-jepa/src/vjepa2/.git || \
    git clone https://github.com/facebookresearch/vjepa2.git /mnt/vla-jepa/src/vjepa2
  git -C /mnt/vla-jepa/src/vjepa2 fetch origin
  git -C /mnt/vla-jepa/src/vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812

  test -d /mnt/vla-jepa/src/MoGe/.git || \
    git clone https://github.com/microsoft/MoGe.git /mnt/vla-jepa/src/MoGe
  git -C /mnt/vla-jepa/src/MoGe fetch origin
  git -C /mnt/vla-jepa/src/MoGe checkout 07444410f1e33f402353b99d6ccd26bd31e469e8
'
```

## 4. Transfer And Verify The Dataset

The dataset is static before transfer; no annotation process should be writing
to it. Use resumable rsync and preserve hardlinks:

```bash
DATA=/home/mehul/work/reward_model_small/magna_training_data_with_interventions
REMOTE_DATA=/mnt/vla-jepa/datasets/magna_training_data_with_interventions

ssh "$HOST" "mkdir -p '$REMOTE_DATA'"
rsync -aH --partial --info=progress2 "$DATA/" "$HOST:$REMOTE_DATA/"
rsync -aHn --itemize-changes "$DATA/" "$HOST:$REMOTE_DATA/"
```

The second command must print no changed files. Then compare the structural
summary on both machines: 2,452,692 parquet rows, 1,638 episode rows, 93 MP4s,
and no errors in `meta/official_validation_report.json`. For a strict handoff,
generate and compare SHA-256 manifests for `data/`, `videos/`, and the active
top-level files under `meta/` (backup/provenance directories may be listed
separately).

Use this reproducible full-snapshot verification after rsync completes:

```bash
MANIFEST=/tmp/magna_training_data.sha256
(cd "$DATA" && \
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) \
  > "$MANIFEST"
scp "$MANIFEST" "$HOST:/mnt/vla-jepa/logs/magna_training_data.sha256"
ssh "$HOST" "cd '$REMOTE_DATA' && sha256sum -c \
  /mnt/vla-jepa/logs/magna_training_data.sha256"
```

The generated config-specific step cache under `meta/` is part of the snapshot
and must be included in the checksum. If the source dataset is still being
annotated, stop and take an immutable snapshot before computing the manifest;
rsync plus a checksum cannot validate a moving source.

Do not point training at `edits/data`. Confirm the active data shards expose
the labels directly:

```bash
ssh "$HOST" "python3 - <<'PY'
import pyarrow.parquet as pq
p = '$REMOTE_DATA/data/chunk-000/file-000.parquet'
names = set(pq.ParquetFile(p).schema_arrow.names)
required = {'source.action', 'source.observation.state', 'subtask_index', 'valid_state'}
assert required <= names, (required - names)
print('active parquet labels verified')
PY"
```

Run that Python check inside the training container if host Python does not
have PyArrow.

## 5. Build The A100 Image

Build FlashAttention only for SM80. The production Qwen3.5 config requests
FlashAttention 2 and permits an SDPA fallback, but the previous LIBERO probes
showed that the compiled SM80 path is worth retaining for this long run.

```bash
ssh "$HOST" '
  cd /mnt/vla-jepa/src/VLA-JEPA
  IMAGE=vla-jepa:py313-cu130-a100 \
  BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 \
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  INSTALL_DEEPSPEED=1 \
  INSTALL_MOGE=1 \
  INSTALL_FLASH_ATTN=1 \
  FLASH_ATTN_CUDA_ARCH_LIST=8.0 \
  FLASH_ATTN_MAX_JOBS=64 \
  FLASH_ATTN_NVCC_THREADS=1 \
  ./scripts/docker_build_training.sh 2>&1 | \
    tee /mnt/vla-jepa/logs/docker_build_magna_a100.log
'
```

Do not build Decord GPU. This run uses PyAV with one decode thread per worker.
Write an explicit exit marker when building in tmux, and do not treat a missing
tmux session as proof of success:

```bash
ssh "$HOST" '
  if test -f /mnt/vla-jepa/logs/docker_build_magna_a100.exit; then
    printf "build exit: "
    cat /mnt/vla-jepa/logs/docker_build_magna_a100.exit
  else
    echo "build has no exit marker"
  fi
  tail -n 80 /mnt/vla-jepa/logs/docker_build_magna_a100.log
'
```

Require exit code zero and a successful `docker image inspect` before running
preflight. Record the immutable image ID and a package inventory in the handoff:

```bash
ssh "$HOST" '
  docker image inspect vla-jepa:py313-cu130-a100 \
    --format "{{.Id}} {{.Created}}" \
    > /mnt/vla-jepa/logs/magna_image_identity.txt
  cd /mnt/vla-jepa/src/VLA-JEPA
  IMAGE=vla-jepa:py313-cu130-a100 ./scripts/docker_run_training.sh \
    python -m pip freeze \
    > /mnt/vla-jepa/logs/magna_image_pip_freeze.txt
'
```

## 6. Preflight And Smoke Gates

Run the runtime preflight and explicitly verify the native FlashAttention
module:

```bash
ssh "$HOST" '
  cd /mnt/vla-jepa/src/VLA-JEPA
  IMAGE=vla-jepa:py313-cu130-a100 ./scripts/docker_run_training.sh \
    python scripts/preflight_runtime.py \
      --require-cuda \
      --require-moge \
      --config-yaml scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml
  IMAGE=vla-jepa:py313-cu130-a100 ./scripts/docker_run_training.sh \
    python -c "import flash_attn, torch; print(flash_attn.__version__, torch.cuda.get_device_capability())"
'
```

Then run these gates in order:

1. Run `pytest tests -q` in the final image or the established project
   environment. Do not use a bare host Python without project dependencies.
2. A direct dataset sample: verify action `[50, 18]`, state `[1, 19]`, three
   video views, an action mask, finite tensors, and a resolved prompt.
3. A two-step single-GPU forward/backward smoke.
4. A short 8-GPU production-shape smoke long enough to pass model/cache warmup.
5. Verify no OOM, NaN/Inf, worker respawn loop, PyAV thread growth, NCCL error,
   all-zero-mask spike, or rank divergence.
6. Measure steady step time and peak allocated/reserved memory on every GPU.

If GPUs wait on the first batch, send `SIGUSR2` to one worker from inside the
container; workers register a faulthandler for this signal. A stack in
`pandas.read_parquet`/`table_to_dataframe` for the complete v3 shard means the
Arrow episode-slice fix is missing from the deployed source. Do not compensate
for that regression merely by adding workers.

Use a unique smoke run id and disable final-model saving. Example 8-GPU gate:

```bash
ssh "$HOST" '
  cd /mnt/vla-jepa/src/VLA-JEPA
  IMAGE=vla-jepa:py313-cu130-a100 \
  DATA_ROOT=/mnt/vla-jepa/datasets/magna_training_data_with_interventions \
  CHECKPOINT_ROOT=/mnt/vla-jepa/checkpoints \
  RUN_ID=magna_a100x8_smoke_$(date +%Y%m%d_%H%M%S) \
  MAX_TRAIN_STEPS=10 \
  SAVE_INTERVAL=1000 \
  EVAL_INTERVAL=1000 \
  ./scripts/docker_run_training.sh bash -lc \
    './scripts/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh --trainer.save_final_model false'
'
```

## 7. Production Review Gate

Before launch, review and record at least:

- model backbones, frozen/trainable modules, and total/trainable parameters;
- 18D no-base/no-lift action mapping, 19D state input, and absolute-action semantics;
- action horizon, video frames/stride/target shift, and camera order;
- intervention-mask semantics and observed keep/all-zero ratios;
- prompt/subtask-label probability;
- RTC schedule and repeated diffusion samples per optimizer step;
- per-device/global batch, workers, precision, attention backend, and DDP mode;
- optimizer, component learning rates, scheduler, warmup, and loss weights;
- epochs, exact optimizer-step count, measured throughput, and ETA;
- checkpoint cadence/retention, final-model saving, TensorBoard, and GCS backup.

Do not silently change a reviewed setting after the smoke. If the smoke requires
a change, rerun the relevant gate and review the new value.

## 8. Launch, Monitor, And Back Up

Keep logs and checkpoints outside the repo. Launch training and TensorBoard in
separate tmux sessions only after the review gate:

```bash
tmux new-session -d -s magna_train \
  'cd /mnt/vla-jepa/src/VLA-JEPA && <reviewed production command> 2>&1 | tee /mnt/vla-jepa/logs/<run-id>.log'

tmux new-session -d -s magna_tensorboard \
  'cd /mnt/vla-jepa/src/VLA-JEPA && IMAGE=vla-jepa:py313-cu130-a100 ./scripts/docker_run_training.sh tensorboard --logdir /mnt/vla-jepa/checkpoints --host 0.0.0.0 --port 6006'
```

Resume with the original run id and a complete `steps_N` directory. Do not
point `resume_from_checkpoint` at the run root or at `model.safetensors` alone:

```bash
cd /mnt/vla-jepa/src/VLA-JEPA
export RUN_ID=<original-run-id>
export CHECKPOINT_ROOT=/mnt/vla-jepa/checkpoints
export DATA_ROOT_DIR=/mnt/vla-jepa/datasets/magna_training_data_with_interventions
export RESUME_CHECKPOINT="${CHECKPOINT_ROOT}/${RUN_ID}/checkpoints/steps_<N>"

IMAGE=vla-jepa:py313-cu130-a100 \
VLA_JEPA_SCRATCH=/mnt/vla-jepa \
DATA_ROOT="${DATA_ROOT_DIR}" \
MAX_TRAIN_STEPS=76644 \
./scripts/docker_run_training.sh \
  ./scripts/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh \
    --trainer.is_resume true \
    --trainer.resume_from_checkpoint "${RESUME_CHECKPOINT}"
```

Require the log to say `Resumed from checkpoint (full_state)` and verify the
progress bar starts at step `N`. The checkpoint smoke proved this path with
`steps_1`, then completed step 2 on all eight ranks.

View TensorBoard through an SSH tunnel rather than exposing port 6006:

```bash
ssh -N -L 6008:127.0.0.1:6006 \
  columbus-8xa100.us-east5-a.yondu-general-workspace
```

The RAID0 local SSD is ephemeral. A stopped, deleted, or preempted VM can lose
the dataset, logs, and checkpoints. Configure and test a checkpoint destination
before training, then run the committed stable-checkpoint uploader:

First verify both IAM and the VM OAuth scope. Read access is not enough:

```bash
gcloud compute instances describe columbus-8xa100 \
  --zone us-east5-a \
  --format='yaml(serviceAccounts)'

printf 'checkpoint preflight\n' | \
  gcloud storage cp - gs://<approved-bucket>/<approved-prefix>/preflight.txt
gcloud storage rm gs://<approved-bucket>/<approved-prefix>/preflight.txt
```

On this prepared node, authenticate the already verified user credential into
the config directory shared with the training container:

```bash
ssh -t columbus-8xa100.us-east5-a.yondu-general-workspace \
  'CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config \
   gcloud auth login --no-launch-browser'
```

Follow the printed bootstrap flow, then require this command to print
`mehul@yonduai.com` rather than the compute service account:

```bash
ssh columbus-8xa100.us-east5-a.yondu-general-workspace \
  'CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config \
   gcloud auth list --filter=status:ACTIVE --format="value(account)"'
```

An instance created with only `devstorage.read_only` cannot upload even if its
service account has bucket permissions. OAuth scopes cannot be widened on a
running VM. Resolve this before filling local SSD: create the VM with the
`cloud-platform` scope, authenticate an approved user/service account on the
running VM, or attach a non-ephemeral checkpoint disk. Do not stop an already
prepared local-SSD node merely to change scopes; stopping discards local SSD.

```bash
CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config \
POLL_SECONDS=60 STABLE_SECONDS=180 LOG_SYNC_SECONDS=900 \
REMOTE_CHECKPOINT_MAX_TO_KEEP=3 \
  ./scripts/watch_and_upload_checkpoints_gcs.sh \
  /mnt/vla-jepa/checkpoints/<run-id> \
  gs://<approved-bucket>/<approved-prefix>/<run-id>
```

The watcher uploads stable periodic checkpoints, refreshes TensorBoard and
`summary.jsonl` logs, and uploads `final_model` once it is stable. For this run,
it retains the latest three `steps_N` prefixes in GCS; final-model, log, and
metadata paths are never included in that pruning. The bucket's seven-day soft
delete policy preserves a deleted checkpoint temporarily before reclaiming its
storage. A live four-prefix GCS probe verified that retention three deletes
only the oldest `steps_N` prefix. Confirm at least one smoke checkpoint can be
listed and downloaded from the approved destination before relying on the
uploader for production.

For each status handoff, report: run id, source manifest, config path/hash,
current step/epoch, current and rolling step time, samples/s, ETA, latest losses,
mask keep/all-zero ratios, GPU memory/utilization, host RAM/swap/disk, latest
checkpoint, and latest successful cloud upload.

Do not report the node as production-ready merely because training can start.
Production-ready means the dataset checksum passed, the exact final image and
source were smoked on all eight GPUs, batch size was measured rather than
assumed, and a checkpoint was uploaded and downloaded through the same backup
path that the production run will use.
