# VLA-JEPA Repo Environment Setup

This runbook captures the reusable Python environment setup for this repo. It intentionally does not include VM-specific repair work such as NVIDIA driver installation or combining local NVMe disks.

## Target Environment

- Python 3.10 or newer
- PyTorch 2.6.0 with CUDA 12.4 wheels
- torchvision 0.21.0 with CUDA 12.4 wheels
- Dependencies from `requirements.txt`
- Editable install of this repo with `pip install -e .`

On Debian or Ubuntu VMs, install the small system prerequisites first:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential ninja-build openmpi-bin libopenmpi-dev
```

`python3-dev` provides `Python.h`, which Triton/DeepSpeed may need when compiling small runtime helpers. `openmpi-bin` and `libopenmpi-dev` keep `mpi4py` and DeepSpeed MPI discovery behavior predictable.
`ninja-build` lets PyTorch extension builds use parallel compilation; without it, large CUDA extensions may fall back to much slower serial builds.

If the VM does not already have a CUDA toolkit matching the PyTorch wheel CUDA version, install the minimal CUDA 12.4 compiler/runtime-dev packages:

```bash
sudo apt-get install -y cuda-nvcc-12-4 cuda-cudart-dev-12-4 cuda-cccl-12-4
```

Then export:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
```

## One Command Setup

Use a large scratch mount for the environment, package cache, Hugging Face cache, datasets, and checkpoints:

```bash
cd /path/to/VLA-JEPA
VLA_JEPA_SCRATCH=/mnt/vla-jepa ./scripts/setup_repo_env.sh
```

The script creates:

```text
${VLA_JEPA_SCRATCH}/envs/vla-jepa
${VLA_JEPA_SCRATCH}/cache/pip
${VLA_JEPA_SCRATCH}/hf
${VLA_JEPA_SCRATCH}/tmp
${VLA_JEPA_SCRATCH}/datasets
${VLA_JEPA_SCRATCH}/checkpoints
```

It also symlinks the repo-local `.venv` to the scratch-backed environment, so normal commands can use:

```bash
source .venv/bin/activate
```

By default the script installs the repo `dev` extra so smoke tests can run. It also installs bitsandbytes for the optional `AdamW8bit` path.

To skip dev tools:

```bash
VLA_JEPA_INSTALL_DEV=0 VLA_JEPA_SCRATCH=/mnt/vla-jepa ./scripts/setup_repo_env.sh
```

To skip optional bitsandbytes:

```bash
VLA_JEPA_INSTALL_BITSANDBYTES=0 VLA_JEPA_SCRATCH=/mnt/vla-jepa ./scripts/setup_repo_env.sh
```

To rebuild the environment from scratch:

```bash
VLA_JEPA_SCRATCH=/mnt/vla-jepa VLA_JEPA_RECREATE_ENV=1 ./scripts/setup_repo_env.sh
```

## Training Accelerators

### CUDA Decord

The PyPI Decord wheels are CPU-only. That is fine for CPU dataloader video reads, but this repo's rank-side VLA-JEPA GPU video decode path defaults to `gpu_video_decode_backend: decord` and calls `decord.gpu(...)`. Build Decord from source when that path is needed.

Install the additional system packages:

```bash
sudo apt-get install -y \
  cmake pkg-config ffmpeg libffmpeg-nvenc-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libavfilter-dev libavdevice-dev libswresample-dev \
  cuda-nvrtc-dev-12-4 cuda-nvml-dev-12-4
```

Then build and install CUDA-enabled Decord into the existing venv:

```bash
VLA_JEPA_SCRATCH=/mnt/vla-jepa ./scripts/build_decord_gpu.sh
```

This creates a reusable wheel in `${VLA_JEPA_SCRATCH}/wheelhouse`. For an identical Python/CUDA/driver stack, reuse it directly:

```bash
VLA_JEPA_SCRATCH=/mnt/vla-jepa \
VLA_JEPA_DECORD_GPU_WHEEL=/mnt/vla-jepa/wheelhouse/decord-0.6.0-cp311-cp311-linux_x86_64.whl \
./scripts/setup_repo_env.sh
```

To have the main setup script build it from source after installing `requirements.txt`:

```bash
VLA_JEPA_SCRATCH=/mnt/vla-jepa \
VLA_JEPA_INSTALL_DECORD_GPU=1 \
./scripts/setup_repo_env.sh
```

FlashAttention is optional in this fork. The Qwen 3.5 path falls back to SDPA when `flash-attn` is unavailable, and the world-model attention code also has a fallback path.

FlashAttention should be installed as an explicit, machine-specific optimization. Compile only for the GPUs on the VM. For A100, use SM80 only:

```bash
VLA_JEPA_SCRATCH=/mnt/vla-jepa \
VLA_JEPA_INSTALL_FLASH_ATTN=1 \
VLA_JEPA_FLASH_ATTN_CUDA_ARCHS=80 \
VLA_JEPA_FLASH_ATTN_MAX_JOBS=96 \
VLA_JEPA_FLASH_ATTN_NVCC_THREADS=1 \
./scripts/setup_repo_env.sh
```

The script forces a local FlashAttention build when enabled. This avoids accidentally installing an ABI-incompatible cached wheel, but it can take a while. Set `VLA_JEPA_FLASH_ATTN_MAX_JOBS` to the VM's CPU thread count if memory is abundant; lower it on smaller machines.

Ninja does not print a reliable percentage for this build. A rough progress estimate is the number of completed object files under the active `pip-install-*/flash-attn_*` build directory compared with the generated Ninja build graph.

For an identical Python/PyTorch/CUDA stack, reuse a known-good local wheel instead of rebuilding:

```bash
VLA_JEPA_SCRATCH=/mnt/vla-jepa \
VLA_JEPA_INSTALL_FLASH_ATTN=1 \
VLA_JEPA_FLASH_ATTN_WHEEL=/mnt/vla-jepa/wheelhouse/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl \
./scripts/setup_repo_env.sh
```

For the lower-memory optimizer path:

```bash
VLA_JEPA_SCRATCH=/mnt/vla-jepa VLA_JEPA_INSTALL_BITSANDBYTES=1 ./scripts/setup_repo_env.sh
```

## Runtime Exports

Use scratch-backed cache and temp paths for training:

```bash
export HF_HOME=/mnt/vla-jepa/hf
export PIP_CACHE_DIR=/mnt/vla-jepa/cache/pip
export TMPDIR=/mnt/vla-jepa/tmp
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export WANDB_MODE=${WANDB_MODE:-disabled}
```

## Canonical GCS Subset Smoke

The 8x A100 smoke config for this VM is:

```bash
scripts/config/vlajepa_robot_ft_canonical_multi_subset_a100x8_smoke.yaml
```

It uses:

- V-JEPA2 source: `/mnt/vla-jepa/src/vjepa2`
- dataset-canonicalization source: `/mnt/vla-jepa/src/dataset-canonicalization`
- canonical shard cache: `/mnt/vla-jepa/datasets/canonical_gcs`
- checkpoints: `/mnt/vla-jepa/checkpoints`

Clone the two helper repos on a fresh VM:

```bash
mkdir -p /mnt/vla-jepa/src
git clone https://github.com/facebookresearch/vjepa2.git /mnt/vla-jepa/src/vjepa2
git clone https://github.com/YonduAI/dataset-canonicalization.git /mnt/vla-jepa/src/dataset-canonicalization
```

Before launching, verify the VM can read the canonical GCS bucket:

```bash
SID=BAAI-DataCube__robomind_benchmark1_1_release_franka_3rgb_yellow_square_placed_on_ceramic_plate
gcloud storage cp --recursive \
  "gs://robotics-datasets-yonduai/raw/${SID}/main/files/meta" \
  "/mnt/vla-jepa/datasets/canonical_gcs/${SID}/main"
```

The active account needs `storage.objects.get` on the bucket, for example via `roles/storage.objectViewer`. The VM also needs a storage read OAuth scope such as `devstorage.read_only` or `cloud-platform`.

Optionally warm the selected GCS subset once from a single process. This is useful
for smoke tests and first-run validation, but it is not required for full
training:

```bash
source .venv/bin/activate
python scripts/prefill_canonical_gcs_cache.py \
  --config-yaml scripts/config/vlajepa_robot_ft_canonical_multi_subset_a100x8_smoke.yaml
```

The dataloader uses per-file lock files under each cached shard's `.locks/` directory, so multi-rank launches do not race on the same `gcloud storage cp` destination. Sidecar generation is also lock-protected and written atomically. For shared model checkpoints, the V-JEPA torchhub URL cache is warmed by rank 0 and the other ranks wait for a ready sentinel before loading the cached file.

For a full-dataset run, do not prefill the entire corpus unless you explicitly want a cache-warming job. Point every rank at the same scratch cache and let training populate missing files as they are selected.

Recommended full-run canonical GCS settings:

```yaml
datasets:
  vla_data:
    allow_gcs_download: true
    cache_dir: /mnt/vla-jepa/datasets/canonical_gcs
    lazy_cache_shards: true
    index_windows_lazily: true
    prefetch_metadata_across_ranks: true
    max_shards: 0              # unlimited; use a positive value for pilots
    max_windows: 0
    max_windows_per_dataset: 0
    shuffle_shards: true       # deterministic shard-order shuffle
    shuffle: false             # preserves shard locality while cache fills
    reader_cache_size: 128
    sidecar_cache_size: 32
    num_workers: 1             # raise to 2 after confirming GCS/cache headroom
    prefetch_factor: 2
    per_device_batch_size: 16  # 8x A100-80GB tuned default, global batch 128
```

`lazy_cache_shards: true` resolves only metadata and window indexes during dataset construction. `index_windows_lazily: true` keeps the canonical dataset map-style but avoids materializing every stride window as a Python object. `prefetch_metadata_across_ranks: true` splits cold metadata downloads across distributed ranks before the common shard scan, so a full-cache startup uses the 8x GPU node's host/network parallelism instead of making seven ranks wait on each `meta.lock`. This matters for the full compatible GCS corpus, which is tens of millions of training windows.

The first sample from a shard fetches that shard's parquet, builds its sidecar, and fetches only the referenced camera videos under lock. Use `shuffle: false` with `shuffle_shards: true` for the first full pass because it preserves per-shard locality while still avoiding a fixed manifest ordering. Random window shuffling across the whole cold corpus can cause every rank to touch many uncached shards at once. After the cache is warm, `shuffle: true` is reasonable.

The current A100x8 full Qwen fine-tuning launcher is:

```bash
STARVLA_DEEPSPEED_STAGE=3 \
./scripts/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.sh
```

It uses DeepSpeed ZeRO-3, keeps all `torch.compile` flags off, disables Qwen LoRA for full Qwen fine-tuning, keeps the teacher encoders frozen, switches MoGe to `Ruicheng/moge-2-vits-normal`, and turns on training-time RTC with a gradual probability warmup/ramp. Override the long-run size without editing the file:

```bash
MAX_TRAIN_STEPS=30000 \
NUM_WARMUP_STEPS=1000 \
RTC_WARMUP_STEPS=10000 \
RTC_RAMP_STEPS=50000 \
./scripts/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.sh
```

On the 8x A100-SXM4-80GB VM, the focused tuning probes found `PER_DEVICE_BATCH_SIZE=16` with ZeRO-3 and gradient checkpointing on to be the best quick default: steady steps were about 4.8s with global batch 128, reserving about 47 GiB per GPU. That is roughly 26-27 samples/s while leaving memory headroom for full-run variance. The next knobs to retune on another VM are batch size first, then dataloader workers/cache sizes; only revisit ZeRO stage or gradient checkpointing if memory or communication becomes the bottleneck.

The RTC curriculum is intentionally slow for full-corpus training: `warmup_steps=10000` means RTC stays off through step 9999, then `ramp_steps=50000` linearly raises the batch probability to the configured target by step 60000. With global batch 128 and the current compatible corpus size, this reaches full RTC after roughly 8% of a full pass; a 30000-step partial run ends at about 40% RTC probability.

Before committing to a long run on a fresh machine, run the short tuning matrix:

```bash
BENCHMARK_STEPS=30 \
BENCHMARK_MAX_SHARDS=8 \
./scripts/benchmark_full_qwen_a100x8_matrix.sh
```

The matrix keeps the real full-run feature set enabled and varies the expensive machine-specific knobs: DeepSpeed stage, per-GPU batch size, dataloader workers, and gradient checkpointing. It disables interval eval/checkpoint writes and records logs plus a TSV summary under `/mnt/vla-jepa/logs/full_qwen_benchmarks`.

Run the smoke:

```bash
cd /path/to/VLA-JEPA
./scripts/vlajepa_robot_ft_canonical_multi_subset_a100x8_smoke.sh
```

## Verification

After setup:

```bash
source .venv/bin/activate
python - <<'PY'
import torch
import transformers
import deepspeed
import mpi4py
import starVLA

print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```
