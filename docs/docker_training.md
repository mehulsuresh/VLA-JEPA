# Docker Training Runtime

This is the preferred way to bring up VLA-JEPA on cloud GPU machines. The image
keeps Python/PyTorch/PyAV/Qwen/MoGe/DeepSpeed setup consistent while still
letting each cloud choose the right CUDA base image and optional CUDA-extension
builds.

For the end-to-end fresh-node and next-agent handoff checklist used by the
Magna A100x8 run, including the reboot-sensitive bootstrap, source/data
synchronization, recovery inspection, smoke gates, monitoring, and checkpoint
backup, see
[`docs/magna_a100x8_training_runbook.md`](magna_a100x8_training_runbook.md).

## Build

Default build, tested against the Python 3.13 runtime path:

```bash
IMAGE=vla-jepa:py313-cu130 ./scripts/docker_build_training.sh
```

Useful build overrides:

```bash
# H100/A100 cloud image, no FlashAttention dependency.
IMAGE=registry.example.com/vla-jepa:py313-cu130 \
BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
INSTALL_DEEPSPEED=1 \
INSTALL_MOGE=1 \
INSTALL_FLASH_ATTN=0 \
./scripts/docker_build_training.sh

# If a cloud image needs a different CUDA/PyTorch pair.
BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
IMAGE=vla-jepa:py313-cu128 \
./scripts/docker_build_training.sh
```

FlashAttention is intentionally off by default. PyAV handles video decode,
Qwen blockwise attention uses PyTorch FlexAttention, and avoiding GPU-specific
extension wheels makes the image more portable across H100/A100/L40/5090.

If you explicitly want FlashAttention for an H100-only image:

```bash
INSTALL_FLASH_ATTN=1 FLASH_ATTN_SPEC=flash-attn IMAGE=vla-jepa:h100-fa ./scripts/docker_build_training.sh
```

## Fresh A2/A100 Node Bootstrap

On a new GCP A2/A100 training VM, run the host bootstrap before cloning data or
building images:

```bash
sudo DOCKER_USERS="$USER" \
  /path/to/VLA-JEPA/deployment/gcp/startup-a2-training-node.sh
```

The bootstrap is idempotent and may request a reboot while installing the GPU
driver. Re-run it after reboot. It stripes the local NVMe SSDs into
`/mnt/disks/ssd-array`, creates `/data`, creates the VLA-JEPA scratch layout at
`/mnt/vla-jepa`, installs Docker plus the NVIDIA container runtime, and moves
Docker's `data-root` to `/mnt/vla-jepa/docker`. Keeping Docker on the SSD array
matters because the default root disk on these instances is usually too small
for CUDA/PyTorch image layers.

## Preflight

Run this before a long training job:

```bash
./scripts/docker_run_training.sh \
  python scripts/preflight_runtime.py \
    --require-cuda \
    --require-deepspeed \
    --config-yaml scripts/config/vlajepa_robot_ft_canonical_subset_5090_smoke.yaml
```

The run helper defaults to `DOCKER_GPU_MODE=runtime`, which uses
`--runtime=nvidia`, `NVIDIA_VISIBLE_DEVICES=${GPUS:-all}`, and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`. That is the most reliable path on
machines where Docker's `--gpus all` requests extra driver mounts. If a cloud
image requires Docker's native GPU flag instead, set:

```bash
export DOCKER_GPU_MODE=gpus
```

For geometry-teacher configs, the preflight also checks MoGe imports:

```bash
./scripts/docker_run_training.sh \
  python scripts/preflight_runtime.py \
    --require-cuda \
    --require-moge \
    --config-yaml scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml
```

## Run A Smoke

```bash
./scripts/docker_run_training.sh \
  accelerate launch --num_processes 1 --mixed_precision no \
    ./starVLA/training/train_starvla.py \
    --config_yaml scripts/config/vlajepa_robot_ft_canonical_subset_5090_smoke.yaml \
    --trainer.max_train_steps 1 \
    --trainer.save_final_model false
```

## Single-Node 8 GPU Training

```bash
./scripts/docker_run_training.sh \
  accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --config_file starVLA/config/deepseeds/deepspeed_zero2_stable.yaml \
    ./starVLA/training/train_starvla.py \
    --config_yaml scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml
```

## Multi-Node Training

Use the same image tag or digest on every node. Mount datasets/checkpoints at
the same paths on all nodes.

Node 0:

```bash
MASTER_ADDR=10.0.0.10
NNODES=2
GPUS_PER_NODE=8

./scripts/docker_run_training.sh \
  accelerate launch \
    --num_machines "${NNODES}" \
    --machine_rank 0 \
    --num_processes "$((NNODES * GPUS_PER_NODE))" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port 29500 \
    --same_network \
    --mixed_precision bf16 \
    --config_file starVLA/config/deepseeds/deepspeed_zero2_stable.yaml \
    ./starVLA/training/train_starvla.py \
    --config_yaml scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml
```

Node 1 uses the same command with `--machine_rank 1`.

Set these when needed by the cloud network:

```bash
export NCCL_SOCKET_IFNAME=ens8
export GLOO_SOCKET_IFNAME=ens8
export NCCL_IB_DISABLE=0
```

## Mounts And Credentials

The run helper automatically mounts the repo and the local Hugging Face cache
when present. Optional environment variables:

```bash
export DATA_ROOT=/mnt/datasets
export CHECKPOINT_ROOT=/mnt/checkpoints
export HF_HOME=/mnt/hf
export VLA_JEPA_SCRATCH=/mnt/vla-jepa
export GOOGLE_APPLICATION_CREDENTIALS=/mnt/secrets/gcloud.json
```

Set `VLA_JEPA_SCRATCH` when configs reference helper checkouts or caches under a
shared scratch root, for example `/mnt/vla-jepa/src/vjepa2` and
`/mnt/vla-jepa/src/dataset-canonicalization`.

Then run:

```bash
./scripts/docker_run_training.sh bash
```

## Cloud Portability Rules

- Keep Decord GPU out of the image. Use `video_backend: pyav`.
- Keep FlashAttention optional. Build it only for the target GPU family.
- Use the same image digest on every distributed node.
- Keep dataset/checkpoint/cache mount paths identical across nodes.
- Run `scripts/preflight_runtime.py` before long runs.
