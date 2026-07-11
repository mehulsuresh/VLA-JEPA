# Magna A100x8 Training Runbook

This is the living operational source of truth for preparing a GCP
`a2-ultragpu-8g` node and launching, monitoring, resuming, and handing off the
Realman Magna intervention run. Do not start a production job until the exact
config, measured smoke result, ETA, and backup plan have been reviewed with the
user.

Last full operational audit: **2026-07-11 18:54 UTC**. The immutable source,
image, dataset, and run identities for the active job are recorded below. Live
step and ETA values are snapshots; always refresh them before acting.

## How To Use And Improve This Runbook

Every agent starts here, even if the request sounds like a simple status check.
First inspect the existing VM and classify the task:

| Situation | Route |
| --- | --- |
| A production process is already running | Inspect it; do not launch, rebuild, resume, or kill anything. Continue at [Live Operations](#8-launch-monitor-resume-and-back-up). |
| The prepared node is idle and a new run is requested | Revalidate source, data, config, image, smoke, and backup gates. Start at [Synchronize Source](#3-synchronize-the-authoritative-source). |
| A complete checkpoint must be resumed | Verify the run identity and full checkpoint directory, then use [Resume](#resume-a-run). |
| The VM is fresh or its Local SSD was recreated | Follow sections 1 through 8 in order. |
| Model, dataset schema, action mapping, GPU type, or dependency image changed | Treat prior batch-size and smoke results as invalid and rerun the affected gates. |

This file is deliberately maintained as procedure plus evidence, not as a
chronological chat log. Every agent who learns something operationally useful
must improve it before handoff:

1. Correct the durable instruction at the point where the issue occurred.
2. Update the bounded **Current Live Handoff Snapshot** with a UTC timestamp.
3. Add one concise row to **Improvement History** only for a reusable lesson,
   failure mode, or changed invariant. Routine step updates do not belong there.
4. Include exact commands, exit markers, hashes, and log paths. Separate
   observed evidence from inference.
5. Never place tokens, passwords, private keys, signed URLs, or credential
   contents in this file, a launch environment, or a run directory.
6. Run the validation commands in **Handoff Completion**, commit the changes,
   push them, and make the cloud checkout match the same commit.

When a command becomes stale, replace it; do not append a second conflicting
recipe. Git history is the archive.

## Agent Handoff Rules

Treat this file as the operational source of truth for this run. Before
provisioning, copying, rebuilding, or launching anything, inspect the existing
node and continue completed work rather than starting it again:

```bash
HOST=columbus-8xa100.us-east5-a.yondu-general-workspace
REPO=/home/mehul/work/vjepa/VLA-JEPA
RUN_ID=magna_interventions_a100x8_qwen35_2b_full_18d_20260711_023223

ssh "$HOST" RUN_ID="$RUN_ID" bash -s <<'REMOTE'
  set -u
  date -u +utc=%Y-%m-%dT%H:%M:%SZ
  hostname
  uptime
  nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu,memory.used \
    --format=csv
  free -h
  df -hT / /mnt/disks/ssd-array /mnt/vla-jepa 2>/dev/null || true
  tmux list-sessions 2>/dev/null || true
  for session in magna_train magna_tensorboard magna_uploader magna_tb_compare; do
    tmux list-panes -t "${session}" -F "${session}|#{pane_start_command}" \
      2>/dev/null || true
  done
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" 2>/dev/null || true
  docker image inspect vla-jepa:py313-cu130-a100 \
    --format "image={{.Id}} created={{.Created}}" 2>/dev/null || true
  git -C /mnt/vla-jepa/src/VLA-JEPA status --short 2>/dev/null || true
  git -C /mnt/vla-jepa/src/VLA-JEPA rev-parse HEAD 2>/dev/null || true
  find /mnt/vla-jepa/logs -maxdepth 1 -type f \
    -name "${RUN_ID}*.exit" -print -exec cat {} \; 2>/dev/null || true
  pgrep -af "train_starvla|accelerate launch|docker_build_training|rsync" || true
REMOTE

git -C "$REPO" status --short
git -C "$REPO" rev-parse HEAD
git -C "$REPO" ls-remote origin refs/heads/main
```

Follow these invariants:

- A clean, pushed Git commit in the local checkout is authoritative. Production
  must not run from an uncommitted overlay. The cloud checkout and image are
  disposable replicas; never overwrite or reset the local worktree.
- Inspect `git status`, the deployed source manifest, tmux sessions, logs, and
  exit markers before inferring that a transfer, build, smoke, or run finished.
- Record **runtime source commit** and **image ID/source commit** separately.
  Training imports code from the bind-mounted cloud checkout; the image supplies
  dependencies and can have been baked from an earlier commit.
- A Docker build snapshots its context when it starts. If source changes during
  a build, cancel or finish it, synchronize the clean commit, and rebuild. Never
  patch a production image or checkout without recording a new commit.
- Do not start production training until the user has reviewed the final
  settings, measured 8-GPU smoke throughput, ETA, and checkpoint backup plan.
- Do not stop or delete an A2 node while local SSD contains the only copy of a
  checkpoint. A guest reboot normally retains Local SSD, while stop/suspend
  discards it by default. GCP offers Local SSD preservation only as a Preview
  option; treat cloud checkpoint verification as mandatory regardless.
- Never kill an unfamiliar training, transfer, build, upload, or audit process.
  Establish ownership from its command, tmux session, and log first.
- Use a unique run ID, named Docker containers, a committed service runner, and
  a non-secret `launch.env`; never rely on shell history to reconstruct a run.
- Leave a final handoff with the runtime commit, image identity, config and data
  hashes, launch environment, commands, log paths, run ID, current metrics,
  latest local and remote checkpoints, and remaining action.

### Current Live Handoff Snapshot (2026-07-11 18:54 UTC)

This snapshot records what was actually observed. It is not a substitute for
the inspection command above. Replace the timestamp and volatile fields before
every handoff; do not stack newer status paragraphs on top of stale ones.

**Current state: production training is active. Do not launch a second job or
resume a checkpoint while `magna_train` or its training container is alive.**

| Field | Observed value |
| --- | --- |
| Run ID | `magna_interventions_a100x8_qwen35_2b_full_18d_20260711_023223` |
| Started | `2026-07-11T09:33:53Z` |
| Runtime source at launch | `cc11e1506e46d7b05ca25bb8d1330c5a990cd89b` from the bind-mounted cloud checkout |
| Dependency image | `sha256:633e1a2a28550726531771b0dc888a83531ec1f599ee17a678f96373b45b6ccc` |
| Image build source | `f2aafad416bffc56132d1e806e860c19543d15e6` |
| Source config SHA-256 | `68fb8ce40e4d0c79860694850333f082669786b0e78f93de01d12ee378a20c42` |
| Resolved run config SHA-256 | `e3bc2dc364163c33d5330f170ac339dceeb50e93ab24eae80a651b3163609587` |
| Dataset manifest SHA-256 | `02d062e4cc7535b9794cd804f30ea0093b0ce1b4937e64cfacb30c33bebcc49a` |
| Step snapshot | 9,010 / 76,644; 864,960 physical samples; epoch 0.353 / 3 |
| Throughput snapshot | 25.81 logged samples/s; 3.672 s median over the latest 100 steps |
| ETA snapshot | Approximately 69.0 hours remaining; `2026-07-14T15:53:49Z` from the latest-100-step median |
| Loss snapshot (latest / mean 100) | total `0.10982 / 0.11421`; action `0.02276 / 0.02911`; world model `0.80982 / 0.79519`; depth `0.37995 / 0.34891` |
| Mask snapshot (latest / mean 100) | keep ratio `0.83500 / 0.89118`; all-zero ratio `0.08333 / 0.08250` |
| Latest checkpoint MAE | `0.082495` at `steps_7500` |
| Checkpoints | `steps_2500`, `steps_5000`, and `steps_7500` exist locally and were uploaded to GCS |
| GCS destination | `gs://robotics-datasets-yonduai/gcloud/vla-jepa/checkpoints/magna_interventions_a100x8_qwen35_2b_full_18d_20260711_023223` |
| Sessions | `magna_train`, `magna_uploader`, `magna_tensorboard`, `magna_tb_compare` |
| Exact current service file | `/mnt/vla-jepa/logs/magna_interventions_a100x8_qwen35_2b_full_18d_20260711_023223.services.sh` |
| Primary logs | `/mnt/vla-jepa/logs/magna_interventions_a100x8_qwen35_2b_full_18d_20260711_023223.log` and `/mnt/vla-jepa/logs/magna_interventions_a100x8_qwen35_2b_full_18d_20260711_023223.gcs_upload.log` |
| Health snapshot | Restart count 0; no OOM, NCCL, decode, worker-exit, or non-finite errors; GPU memory 78,294-78,314 MiB; host RAM approximately 120 GiB used / 1.2 TiB available; no swap; approximately 2.7 TiB free |

The current run predates `scripts/run_cloud_training_service.sh`; its external
service file is preserved above. New runs must use the committed service runner
and an immutable per-run worktree with named containers so process ownership,
source identity, and relaunch behavior are explicit. The shared cloud checkout
may advance for documentation after launch; its current `HEAD` is therefore not
evidence of the code already loaded by this active process.

Verified node state:

```text
GCE machine type              a2-ultragpu-8g
GPUs                          8x A100-SXM4-80GB, full NVLink mesh
Host CPUs / RAM / swap        96 / approximately 1.3 TiB / none
Scratch                       approximately 2.9 TiB RAID0 at /mnt/disks/ssd-array
Docker data root              /mnt/vla-jepa/docker
Runtime source at launch      cc11e1506e46d7b05ca25bb8d1330c5a990cd89b
8-GPU smoke commit            4d263d2ab41df3895d2e46b83a86bc44bbe043bf
Image source commit           f2aafad416bffc56132d1e806e860c19543d15e6
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

The current local audit passed `pytest tests -q` with `147 passed, 1 skipped`.
The final baked cloud image previously passed with `140 passed, 2 skipped`
before the new operational-runner tests were added. Use `pytest tests -q`;
bare `pytest -q` also collects optional simulation packages that are not part
of this training image. Rerun the current suite in every newly built image.

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

The ordered work while this run is active is:

1. Monitor training without modifying the running checkout, container, config,
   dataset, or event files.
2. Verify every new `steps_N` checkpoint becomes stable locally and then appears
   in the GCS uploader log; retain three local and three remote checkpoints.
3. Investigate before restarting if the training container exits. Resume only
   from a complete uploaded or local full-state checkpoint.
4. At completion, verify `final_model`, the final three checkpoints, config,
   launch metadata, TensorBoard logs, and summary are present in GCS before the
   VM is stopped or deleted.

Authoritative evidence on the prepared node:

| Evidence | Path |
| --- | --- |
| Final production preflight | `/mnt/vla-jepa/logs/magna_production_preflight_manifest.txt` |
| Dataset file manifest | `/mnt/vla-jepa/logs/magna_training_data.sha256` |
| Image identity and package inventory | `/mnt/vla-jepa/logs/magna_image_identity.txt`, `/mnt/vla-jepa/logs/magna_image_pip_freeze.txt` |
| Final 18D smoke result | `/mnt/vla-jepa/logs/magna_a100x8_18d_smoke_f2aafad_20260711_0910.result.txt` |
| Full-state save/resume smoke | `/mnt/vla-jepa/logs/magna_a100x8_checkpoint_smoke_4d263d2_20260711_052718.log`, `.resume.log` |
| Production launch definition | `/mnt/vla-jepa/logs/<run-id>.services.sh` |
| Production train/upload logs | `/mnt/vla-jepa/logs/<run-id>.log`, `/mnt/vla-jepa/logs/<run-id>.gcs_upload.log` |

The older `/mnt/vla-jepa/logs/magna_source_manifest/status.txt` describes an
early dirty deployment and is historical only. The clean
`magna_production_preflight_manifest.txt` supersedes it for production identity.

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
- Production image wrapper: `scripts/build_magna_a100_image.sh`
- Train/TensorBoard/uploader service runner:
  `scripts/run_cloud_training_service.sh`
- New-run source worktree:
  `/mnt/vla-jepa/src/run_worktrees/<run-id>` at the reviewed detached commit

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

Production source must be committed, pushed, and clean. Do not deploy a dirty
worktree or use rsync as a source-code overlay: that makes the run impossible to
reproduce from its commit. Commit emergency fixes on an explicit branch first,
review them, and then select the exact production commit.

```bash
HOST=columbus-8xa100.us-east5-a.yondu-general-workspace
REPO=/home/mehul/work/vjepa/VLA-JEPA

cd "$REPO"
test -z "$(git status --porcelain)"
git fetch origin
git push origin main
SOURCE_COMMIT="$(git rev-parse HEAD)"
test "$(git ls-remote origin refs/heads/main | awk '{print $1}')" = "$SOURCE_COMMIT"

ssh -A "$HOST" SOURCE_COMMIT="$SOURCE_COMMIT" bash -s <<'REMOTE'
  set -e
  mkdir -p /mnt/vla-jepa/src
  if [[ ! -d /mnt/vla-jepa/src/VLA-JEPA/.git ]]; then
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new" \
      git clone git@github.com:YonduAI/VLA-JEPA.git /mnt/vla-jepa/src/VLA-JEPA
  fi
  test -z "$(git -C /mnt/vla-jepa/src/VLA-JEPA status --porcelain)"
  git -C /mnt/vla-jepa/src/VLA-JEPA fetch origin main
  git -C /mnt/vla-jepa/src/VLA-JEPA checkout main
  git -C /mnt/vla-jepa/src/VLA-JEPA merge --ff-only origin/main
  test "$(git -C /mnt/vla-jepa/src/VLA-JEPA rev-parse HEAD)" = "$SOURCE_COMMIT"
  test -z "$(git -C /mnt/vla-jepa/src/VLA-JEPA status --porcelain)"
REMOTE
```

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
  git merge --ff-only FETCH_HEAD
  git status --short
  git rev-parse HEAD
'
rm /tmp/vla-jepa-sync.bundle
ssh "$HOST" 'rm /mnt/vla-jepa/logs/vla-jepa-sync.bundle'
```

The bundle path is only a credential-free transport for Git objects. It is not
a substitute for committing. Confirm local `HEAD`, GitHub `main`, and cloud
`HEAD` are identical afterward, then remove the temporary bundle.

Record exactly what was deployed:

```bash
ssh "$HOST" '
  set -e
  cd /mnt/vla-jepa/src/VLA-JEPA
  {
    printf "generated_utc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf "runtime_source_commit=%s\n" "$(git rev-parse HEAD)"
    printf "runtime_source_status_lines=%s\n" "$(git status --porcelain | wc -l)"
    sha256sum \
      scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml \
      scripts/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh \
      scripts/run_cloud_training_service.sh \
      scripts/docker_run_training.sh \
      scripts/watch_and_upload_checkpoints_gcs.sh
  } > /mnt/vla-jepa/logs/magna_runtime_source_manifest.txt
'
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

The config-specific `meta/steps_<key>.pkl` index is derived data, but it affects
startup and is included in the verified snapshot. Generate it with the final
config during the direct-sample preflight, then rerun rsync, the dry run, and
the full checksum if a new cache appeared. Do not let the first production rank
silently mutate what was claimed to be an immutable dataset snapshot. If the
source dataset is still being annotated, stop and take an immutable snapshot
before computing the manifest; rsync plus a checksum cannot validate a moving
source.

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

Run that Python check inside the final training container if host Python does
not have PyArrow. On a fresh node this check may be deferred until section 6,
but it must pass before the smoke gate.

## 5. Build The A100 Image

Build FlashAttention only for SM80. The production Qwen3.5 config requests
FlashAttention 2 and permits an SDPA fallback, but measured A100 probes showed
the compiled SM80 path is worth retaining. Magna uses raw DDP, so the optimized
image wrapper defaults to `INSTALL_DEEPSPEED=0`; enable DeepSpeed only when the
reviewed config actually uses it, then rerun all image and smoke gates.

```bash
ssh "$HOST" '
  set -e
  cd /mnt/vla-jepa/src/VLA-JEPA
  test -z "$(git status --porcelain)"
  test ! -e /mnt/vla-jepa/logs/docker_build_magna_a100.exit || \
    rm /mnt/vla-jepa/logs/docker_build_magna_a100.exit
  test -z "$(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep -x magna_image_build || true)"
  tmux new-session -d -s magna_image_build \
    "cd /mnt/vla-jepa/src/VLA-JEPA && exec ./scripts/build_magna_a100_image.sh"
'
```

Do not build Decord GPU. This run uses PyAV with one decode thread per worker.
The committed wrapper writes the full log, a numeric exit marker, source commit,
and immutable image ID. Never treat a missing tmux session as proof of success:

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

Require exit code zero, a clean source commit in
`magna_image_identity.txt`, and a successful `docker image inspect` before
running preflight. Record a package inventory in the handoff:

```bash
ssh "$HOST" '
  set -e
  test "$(cat /mnt/vla-jepa/logs/docker_build_magna_a100.exit)" = 0
  cat /mnt/vla-jepa/logs/magna_image_identity.txt
  docker image inspect vla-jepa:py313-cu130-a100 --format "{{.Id}} {{.Created}}"
  cd /mnt/vla-jepa/src/VLA-JEPA
  IMAGE=vla-jepa:py313-cu130-a100 DOCKER_GPU_MODE=none MOUNT_GCLOUD=0 \
    ./scripts/docker_run_training.sh \
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
  IMAGE=vla-jepa:py313-cu130-a100 MOUNT_GCLOUD=0 \
    ./scripts/docker_run_training.sh \
    python scripts/preflight_runtime.py \
      --require-cuda \
      --require-moge \
      --config-yaml scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml
  IMAGE=vla-jepa:py313-cu130-a100 MOUNT_GCLOUD=0 \
    ./scripts/docker_run_training.sh \
    python -c "import flash_attn, torch; print(flash_attn.__version__, torch.cuda.get_device_capability())"
'
```

Then run these gates in order:

1. Run `pytest tests -q` in the final image. Do not use a bare host Python
   without project dependencies:

   ```bash
   ssh "$HOST" '
     cd /mnt/vla-jepa/src/VLA-JEPA
     IMAGE=vla-jepa:py313-cu130-a100 VLA_JEPA_SCRATCH=/mnt/vla-jepa \
       MOUNT_GCLOUD=0 \
       ./scripts/docker_run_training.sh pytest tests -q
   '
   ```

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
  set -o pipefail
  cd /mnt/vla-jepa/src/VLA-JEPA
  RUN_ID=magna_a100x8_smoke_$(date +%Y%m%d_%H%M%S)
  LOG=/mnt/vla-jepa/logs/${RUN_ID}.log
  EXIT=/mnt/vla-jepa/logs/${RUN_ID}.exit
  rm -f "${EXIT}"
  IMAGE=vla-jepa:py313-cu130-a100 \
  DOCKER_NAME=${RUN_ID}-train \
  DOCKER_TTY=0 \
  MOUNT_GCLOUD=0 \
  DATA_ROOT_DIR=/mnt/vla-jepa/datasets/magna_training_data_with_interventions \
  CHECKPOINT_ROOT=/mnt/vla-jepa/checkpoints \
  RUN_ID=${RUN_ID} \
  MAX_TRAIN_STEPS=10 \
  SAVE_INTERVAL=1000 \
  EVAL_INTERVAL=1000 \
  ./scripts/docker_run_training.sh bash -lc \
    './scripts/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh --trainer.save_final_model false' \
    2>&1 | tee "${LOG}"
  status=${PIPESTATUS[0]}
  printf "%s\n" "${status}" > "${EXIT}"
  exit "${status}"
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

Create one non-secret launch environment and one preflight manifest. These are
the machine-readable handoff for all tmux services and are uploaded with run
metadata. Do not put credentials or tokens in the environment file. The
service runner requires this file for every non-status mode and rejects
group/world-writable files, shell syntax, variable expansion, and secret-like
variable names:

```bash
ssh "$HOST" '
  set -e
  REPO=/mnt/vla-jepa/src/VLA-JEPA
  RUN_ID=magna_interventions_a100x8_qwen35_2b_full_18d_$(date -u +%Y%m%d_%H%M%S)
  SOURCE_COMMIT=$(git -C "${REPO}" rev-parse HEAD)
  RUN_SOURCE=/mnt/vla-jepa/src/run_worktrees/${RUN_ID}
  RUN_ENV=/mnt/vla-jepa/logs/${RUN_ID}.launch.env
  PREFLIGHT=/mnt/vla-jepa/logs/${RUN_ID}.preflight_manifest.txt
  MAIN_PROCESS_PORT=29641  # choose an unused port and verify it below

  if ss -ltn | awk "{print \$4}" | grep -Eq ":${MAIN_PROCESS_PORT}$"; then
    echo "MAIN_PROCESS_PORT is already in use" >&2
    exit 1
  fi
  test -z "$(git -C "${REPO}" status --porcelain)"
  test ! -e "${RUN_SOURCE}"
  mkdir -p "$(dirname "${RUN_SOURCE}")"
  git -C "${REPO}" worktree add --detach "${RUN_SOURCE}" "${SOURCE_COMMIT}"
  test -z "$(git -C "${RUN_SOURCE}" status --porcelain)"

  cat > "${RUN_ENV}" <<EOF
RUN_ID=${RUN_ID}
RUN_SOURCE=${RUN_SOURCE}
EXPECTED_SOURCE_COMMIT=${SOURCE_COMMIT}
IMAGE=vla-jepa:py313-cu130-a100
VLA_JEPA_SCRATCH=/mnt/vla-jepa
DATA_ROOT_DIR=/mnt/vla-jepa/datasets/magna_training_data_with_interventions
CHECKPOINT_ROOT=/mnt/vla-jepa/checkpoints
LOG_ROOT=/mnt/vla-jepa/logs
TRAIN_LAUNCHER=./scripts/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh
NUM_PROCESSES=8
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}
STARVLA_USE_DEEPSPEED=0
GCS_DEST=gs://robotics-datasets-yonduai/gcloud/vla-jepa/checkpoints/${RUN_ID}
CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config
POLL_SECONDS=60
STABLE_SECONDS=180
LOG_SYNC_SECONDS=900
REMOTE_CHECKPOINT_MAX_TO_KEEP=3
PREFLIGHT_MANIFEST=${PREFLIGHT}
EOF
  chmod 0644 "${RUN_ENV}"

  cd "${RUN_SOURCE}"
  test -z "$(git status --porcelain)"
  {
    printf "generated_utc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf "runtime_source_path=%s\n" "${RUN_SOURCE}"
    printf "runtime_source_commit=%s\n" "${SOURCE_COMMIT}"
    printf "runtime_source_status_lines=0\n"
    printf "launch_env_sha256=%s\n" "$(sha256sum "${RUN_ENV}" | awk "{print \$1}")"
    printf "dataset_manifest_sha256=%s\n" "$(sha256sum /mnt/vla-jepa/logs/magna_training_data.sha256 | awk "{print \$1}")"
    sha256sum scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml
    cat /mnt/vla-jepa/logs/magna_image_identity.txt
    printf "vjepa2_commit=%s\n" "$(git -C /mnt/vla-jepa/src/vjepa2 rev-parse HEAD)"
    printf "moge_commit=%s\n" "$(git -C /mnt/vla-jepa/src/MoGe rev-parse HEAD)"
  } > "${PREFLIGHT}"

  printf "RUN_ENV_FILE=%s\nPREFLIGHT_MANIFEST=%s\n" "${RUN_ENV}" "${PREFLIGHT}"
  cat "${RUN_ENV}"
  cat "${PREFLIGHT}"
'
```

Add the exact final smoke run ID, exit code, throughput, peak memory, and tested
backup destination to the preflight manifest before approval. The source config
hash and the resolved `RUN_DIR/config.yaml` hash will differ because the trainer
adds the run ID, output path, and Accelerate metadata; preserve both.

Do not silently change a reviewed setting after the smoke. If the smoke requires
a change, rerun the relevant gate and review the new value.

## 8. Launch, Monitor, Resume, And Back Up

Keep logs and checkpoints outside the repo. Launch all services from the same
reviewed `RUN_ENV_FILE`; this prevents train, TensorBoard, and uploader paths
from drifting. Start waiting services first and training last, only after user
approval:

```bash
ssh "$HOST" '
  set -e
  RUN_ENV=/mnt/vla-jepa/logs/<run-id>.launch.env
  test -r "${RUN_ENV}"
  set -a; source "${RUN_ENV}"; set +a
  test -x "${RUN_SOURCE}/scripts/run_cloud_training_service.sh"
  test "$(git -C "${RUN_SOURCE}" rev-parse HEAD)" = "${EXPECTED_SOURCE_COMMIT}"
  test -z "$(git -C "${RUN_SOURCE}" status --porcelain)"
  for session in magna_train magna_tensorboard magna_uploader; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "Refusing to replace active tmux session: ${session}" >&2
      exit 1
    fi
  done

  tmux new-session -d -s magna_uploader \
    "RUN_ENV_FILE=${RUN_ENV} exec ${RUN_SOURCE}/scripts/run_cloud_training_service.sh uploader"
  tmux new-session -d -s magna_tensorboard \
    "RUN_ENV_FILE=${RUN_ENV} exec ${RUN_SOURCE}/scripts/run_cloud_training_service.sh tensorboard"
  tmux new-session -d -s magna_train \
    "RUN_ENV_FILE=${RUN_ENV} exec ${RUN_SOURCE}/scripts/run_cloud_training_service.sh train"

  sleep 5
  tmux list-sessions
  for session in magna_train magna_tensorboard magna_uploader; do
    tmux list-panes -t "${session}" -F "${session}|#{pane_start_command}"
  done
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
'
```

The service runner refuses to reuse a nonempty run directory unless resuming,
locks train, TensorBoard, and uploader ownership, validates the per-run source
path and commit, names the train/TensorBoard containers, writes numeric exit markers,
and keeps GCP credentials out of train/TensorBoard containers. The uploader
stages the non-secret launch environment and preflight manifest under
`/mnt/vla-jepa/logs/run_metadata/<run-id>` because Docker-created run
directories are root-owned, publishes that handoff metadata, and resynchronizes
when the trainer creates or changes its resolved config and dataset statistics.

### Resume A Run

Resume with the original run ID and a complete `steps_N` directory. Do not
point `RESUME_CHECKPOINT` at the run root or at `model.safetensors` alone. Stop
and investigate the original failure first, verify no old training container
or rank process remains, then update the original launch environment:

```bash
ssh "$HOST" '
  set -e
  RUN_ENV=/mnt/vla-jepa/logs/<original-run-id>.launch.env
  RESUME=/mnt/vla-jepa/checkpoints/<original-run-id>/checkpoints/steps_<N>
  test -r "${RUN_ENV}"
  set -a; source "${RUN_ENV}"; set +a
  test -d "${RESUME}"
  test -f "${RESUME}/model.safetensors"
  test -z "$(pgrep -af "[t]rain_starvla|[a]ccelerate launch" || true)"
  ! docker ps --format "{{.Names}}" | grep -q -- "-train$"

  tmux new-session -d -s magna_train \
    "RUN_ENV_FILE=${RUN_ENV} RESUME_CHECKPOINT=${RESUME} exec ${RUN_SOURCE}/scripts/run_cloud_training_service.sh train"
'
```

Require the log to say `Resumed from checkpoint (full_state)` and verify the
progress bar starts at step `N`. The checkpoint smoke proved this path with
`steps_1`, then completed step 2 on all eight ranks.

If uploader or TensorBoard sessions also exited, restart only those missing
services with the same `RUN_ENV_FILE`. Do not create a second uploader for the
same run.

### Live Status And Error Triage

Use TensorBoard event data rather than parsing carriage-return progress bars.
The first command reports process/checkpoint identity; the second prints recent
and rolling metrics from the authoritative event file. The active run in the
snapshot predates launch environments and named containers, so the fallback
branch plus the complete `docker ps`/`pgrep` output is authoritative for that
run; its new-run lock lines will correctly be free but do not describe the
legacy process:

```bash
ssh "$HOST" 'bash -s' <<'REMOTE'
set -u
RUN_ID=<run-id>
RUN_ENV=/mnt/vla-jepa/logs/${RUN_ID}.launch.env
if [[ -r "${RUN_ENV}" ]]; then
  set -a; source "${RUN_ENV}"; set +a
  RUN_ENV_FILE="${RUN_ENV}" \
    "${RUN_SOURCE}/scripts/run_cloud_training_service.sh" status
else
  RUN_SOURCE=/mnt/vla-jepa/src/VLA-JEPA
  CHECKPOINT_ROOT=/mnt/vla-jepa/checkpoints
  LOG_ROOT=/mnt/vla-jepa/logs
  IMAGE=vla-jepa:py313-cu130-a100
  RUN_ID="${RUN_ID}" CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    IMAGE="${IMAGE}" \
    "${RUN_SOURCE}/scripts/run_cloud_training_service.sh" status
fi

docker ps --no-trunc \
  --format '{{.ID}}|{{.Names}}|{{.Status}}|{{.Image}}|{{.Command}}'
pgrep -af 'train_starvla|accelerate launch|watch_and_upload_checkpoints_gcs|tensorboard' || true

docker run --rm \
  -v "${CHECKPOINT_ROOT}/${RUN_ID}:/run:ro" \
  "${IMAGE}" python -c '
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("/run/starvla", size_guidance={"scalars": 0})
ea.Reload()
tags = [
    "epoch", "samples_seen", "total_loss", "action_loss", "wm_loss",
    "depth_teacher_loss", "action_loss_mask_keep_ratio",
    "action_loss_mask_all_zero_ratio", "grad_norm", "wall_step_time",
    "avg_samples_per_sec", "mae_score", "rtc_training_probability",
]
for tag in tags:
    if tag not in ea.Tags().get("scalars", []):
        continue
    values = ea.Scalars(tag)
    recent = values[-100:]
    print(f"{tag}: step={values[-1].step} value={values[-1].value:.6g} "
          f"mean_last_{len(recent)}={sum(x.value for x in recent)/len(recent):.6g}")
'

nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu \
  --format=csv
free -h
df -h /mnt/vla-jepa
rg -n -i 'traceback|out of memory|cuda error|nccl.*(error|fail)|non.?finite|worker.*(exit|died)|decode.*(error|fail)|killed' \
  "${LOG_ROOT}/${RUN_ID}.log" || true
tail -n 50 "${LOG_ROOT}/${RUN_ID}.gcs_upload.log"
REMOTE
```

Treat loss values across different datasets/action spaces as trend comparisons,
not directly comparable task quality. Checkpoint MAE is more useful within the
same run; real policy quality still requires held-out inference or rollout eval.

View TensorBoard through an SSH tunnel rather than exposing port 6006:

```bash
ssh -N -L 6008:127.0.0.1:6006 \
  columbus-8xa100.us-east5-a.yondu-general-workspace
```

For comparisons across runs with different global batches, use
`scripts/tensorboard_compare_samples.py` to rewrite display copies onto a
physical-samples-seen axis. Never modify original event files. The current
`magna_tb_compare` service uses 96 samples/step for Magna, 128 for LIBERO+, and
9 for the historical Realman run, and clips historical curves to the live
Magna horizon. In that comparison dashboard, TensorBoard's label `Step` means
physical samples seen.

The RAID0 Local SSD is ephemeral. A reboot normally preserves it, but stop and
suspend discard Local SSD by default, guest-OS shutdown discards it, and
preemption/deletion can lose it. GCP's preserve-on-stop option is Preview and is
not a backup. See Google's [Local SSD persistence documentation](https://cloud.google.com/compute/docs/disks/local-ssd#data_persistence).
Configure and test a durable checkpoint destination before training, then run
the committed stable-checkpoint uploader:

First inspect the VM service-account scopes from an authenticated local gcloud
CLI. Read access is not enough:

```bash
gcloud compute instances describe columbus-8xa100 \
  --zone us-east5-a \
  --format='yaml(serviceAccounts)'
```

Check the credential in the VM's dedicated config directory. If this does not
print the approved user account, authenticate it before continuing:

```bash
HOST=columbus-8xa100.us-east5-a.yondu-general-workspace
ACCOUNT="$(ssh "$HOST" \
  'CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config \
   gcloud auth list --filter=status:ACTIVE --format="value(account)"')"
if [[ "${ACCOUNT}" != "mehul@yonduai.com" ]]; then
  ssh -t "$HOST" \
    'CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config \
     gcloud auth login --no-launch-browser'
fi
test "$(ssh "$HOST" \
  'CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config \
   gcloud auth list --filter=status:ACTIVE --format="value(account)"')" = \
  "mehul@yonduai.com"
```

After authentication, require the account check to print `mehul@yonduai.com`
rather than the compute service account. Then prove write, read, and delete from
the VM itself, using the same gcloud config as the uploader:

```bash
HOST=columbus-8xa100.us-east5-a.yondu-general-workspace
GCS_PROBE=gs://<approved-bucket>/<approved-prefix>/preflight-$(date -u +%Y%m%d_%H%M%S)-$$.txt
ssh "$HOST" GCS_PROBE="$GCS_PROBE" bash -s <<'REMOTE'
set -e
export CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config
printf 'checkpoint preflight\n' | gcloud storage cp - "${GCS_PROBE}"
test "$(gcloud storage cat "${GCS_PROBE}")" = "checkpoint preflight"
gcloud storage rm "${GCS_PROBE}"
REMOTE
```

An instance using metadata-server service-account credentials with only
`devstorage.read_only` cannot upload even if IAM grants bucket permissions.
Access scopes and IAM both constrain that identity; Google recommends the
`cloud-platform` scope with access controlled by IAM. See the official
[Compute Engine service-account guidance](https://cloud.google.com/compute/docs/access/service-accounts).
Scopes cannot be changed while the VM is running. Resolve this before filling
Local SSD: create the VM with `cloud-platform`, authenticate an approved user
in the mounted gcloud config, or attach durable checkpoint storage. Do not rely
on a stop solely to change scopes; default stop behavior discards Local SSD,
and Preview preservation still is not a backup.

The normal path is the `magna_uploader` service launched from `RUN_ENV_FILE`.
For manual recovery only, invoke the same committed watcher with identical
settings and first prove no uploader is already running:

```bash
ssh "$HOST" 'bash -s' <<'REMOTE'
set -e
RUN_ENV=/mnt/vla-jepa/logs/<run-id>.launch.env
set -a; source "${RUN_ENV}"; set +a
test -z "$(pgrep -af '[w]atch_and_upload_checkpoints_gcs' || true)"
cd "${RUN_SOURCE}"
CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config \
POLL_SECONDS=60 STABLE_SECONDS=180 LOG_SYNC_SECONDS=900 \
REMOTE_CHECKPOINT_MAX_TO_KEEP=3 \
  ./scripts/watch_and_upload_checkpoints_gcs.sh \
  "${CHECKPOINT_ROOT}/${RUN_ID}" \
  "${GCS_DEST}"
REMOTE
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

## Proven Failure Modes And Responses

| Symptom or risk | Proven response |
| --- | --- |
| tmux session disappeared but GPU memory remains allocated | tmux is not the process authority. Inspect `docker ps`, container command, and rank PIDs. Stop only the identified named container; killing tmux alone can orphan Docker. |
| Cloud `HEAD` differs from the commit reported for a live run | Use the launch manifest and per-run worktree. The shared checkout can move; current `HEAD` does not rewrite code already loaded by a process. |
| Production source requires an uncommitted overlay | Stop. Commit and review it first. Dirty production source is not reproducible. |
| A larger batch passes step 1 | Keep testing across token/sample variation. Magna batches 13 and 14 later OOMed; batch 12 is the measured ceiling for this exact shape. |
| First batch appears stuck in pandas/parquet conversion | Verify the Arrow episode-slice implementation and bounded shard cache. Do not mask a full-shard regression by adding workers. |
| PyAV creates large thread counts | Keep `video_backend_num_threads: 1`; never use decoder `auto` in multi-worker training. |
| Worker shutdown emits `sem_unlink` or resource-tracker traces | Do not call private `resource_tracker._stop()`. Use the tested idempotent DataLoader teardown. |
| Checkpoint exists only on Local SSD | It is not durable. Wait for stability, verify uploader success, list/download it from GCS, and compare files before relying on it. |
| Host uploader cannot write into a Docker-created run directory | Do not chmod or chown an active run tree. Stage launch/preflight files in `/mnt/vla-jepa/logs/run_metadata/<run-id>` and pass it as `EXTRA_METADATA_DIR`; merge it with readable trainer metadata during upload. |
| GCS read works but writes fail | Check both IAM identity and OAuth scope. Use `cloud-platform` for a new VM or the approved mounted user credential; test write/read/delete. |
| TensorBoard curves from different runs appear misaligned | Compare on physical samples seen with `scripts/tensorboard_compare_samples.py`, not raw optimizer step. |
| Checkpoint saves make occasional long steps | Use median/p95 plus overall samples/s for ETA. The observed approximately 37 s checkpoint outlier is expected and low-overhead at a 2,500-step cadence. |
| Disk usage grows despite checkpoint retention | Check incomplete smoke runs, logs, Docker layers, uploader state, and final-model copies separately. Never delete an active checkpoint or audit artifact without identifying ownership. |

## Handoff Completion

Before ending an agent session:

1. Refresh **Current Live Handoff Snapshot** from event data and UTC time.
2. Verify training/uploader/TensorBoard ownership and health; do not interrupt
   them merely to validate documentation.
3. Record the newest complete local checkpoint and newest confirmed GCS upload.
4. Convert every newly learned failure into a corrected durable instruction or
   a concise failure-mode row.
5. Validate changed scripts and focused tests:

   ```bash
   cd /home/mehul/work/vjepa/VLA-JEPA
   bash -n \
     scripts/build_magna_a100_image.sh \
     scripts/run_cloud_training_service.sh \
     scripts/docker_run_training.sh \
     scripts/watch_and_upload_checkpoints_gcs.sh
   python -m pytest -q \
     tests/test_magna_training_config.py \
     tests/test_checkpoint_uploader.py
   git diff --check
   ```

6. Commit and push intentional changes. Require local `HEAD`, GitHub `main`,
   and the cloud primary checkout to match. Never update or remove an active
   per-run worktree.
7. Leave this compact handoff record in the final response or task log:

   ```text
   observed_utc:
   objective_and_status:
   run_id:
   runtime_source_path_and_commit:
   image_id:
   config_hash_and_dataset_manifest_hash:
   step_epoch_samples_throughput_eta:
   latest_losses_and_mask_ratios:
   gpu_ram_swap_disk:
   latest_local_checkpoint:
   latest_verified_gcs_checkpoint:
   active_sessions_and_named_containers:
   changes_committed:
   evidence_paths:
   next_action_or_blocker:
   ```

## Improvement History

Add a row only when experience changes a reusable instruction or invariant.
The detailed diff remains in Git history.

| UTC date | Experience incorporated | Durable improvement |
| --- | --- | --- |
| 2026-07-10 | Initial fresh A2 setup and Magna intervention-data preparation | Added endpoint, bootstrap, source/data transfer, image, smoke, and launch gates. |
| 2026-07-11 | Batch sweep, PyAV/thread investigation, Arrow cache fix, clean shutdown, and full-state resume test | Recorded batch 12 ceiling, one-thread PyAV rule, bounded episode-slice cache, teardown invariant, and resumable checkpoint evidence. |
| 2026-07-11 | GCS authentication, unreadable container files, and local/remote retention probes | Added credential preflight, host-readable artifact checks, stable upload behavior, restore verification, and three-checkpoint retention. |
| 2026-07-11 | Production launch and live TensorBoard comparison | Corrected runtime-source versus image identity, recorded the active run, added sample-aligned comparisons, and replaced stale launch-pending state. |
| 2026-07-11 | Agent-to-agent operational audit | Added root agent routing; required clean committed source, immutable per-run worktrees, versioned non-secret launch environments, named containers, committed service/build runners, failure-path exit markers, content-aware metadata resynchronization, and mandatory continuous improvement. |
