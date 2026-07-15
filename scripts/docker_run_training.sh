#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-vla-jepa:py313-cu130}"
GPUS="${GPUS:-all}"
DOCKER_GPU_MODE="${DOCKER_GPU_MODE:-runtime}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
SHM_SIZE="${SHM_SIZE:-64g}"
DOCKER_NETWORK="${DOCKER_NETWORK:-host}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/VLA-JEPA}"

if [[ -z "${VLA_JEPA_SCRATCH:-}" && -d "/mnt/vla-jepa" ]]; then
  VLA_JEPA_SCRATCH="/mnt/vla-jepa"
fi
if [[ -n "${VLA_JEPA_SCRATCH:-}" ]]; then
  mkdir -p \
    "${VLA_JEPA_SCRATCH}/hf" \
    "${VLA_JEPA_SCRATCH}/cache/torch" \
    "${VLA_JEPA_SCRATCH}/cache/pip" \
    "${VLA_JEPA_SCRATCH}/tmp"

  HF_HOME="${HF_HOME:-${VLA_JEPA_SCRATCH}/hf}"
  HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  TORCH_HOME="${TORCH_HOME:-${VLA_JEPA_SCRATCH}/cache/torch}"
  PIP_CACHE_DIR="${PIP_CACHE_DIR:-${VLA_JEPA_SCRATCH}/cache/pip}"
  TMPDIR="${TMPDIR:-${VLA_JEPA_SCRATCH}/tmp}"
fi

DOCKER_ARGS=(
  --rm
  --ipc=host
  --network "${DOCKER_NETWORK}"
  --shm-size "${SHM_SIZE}"
  --ulimit memlock=-1
  --ulimit stack=67108864
  -w "${CONTAINER_WORKDIR}"
  -v "${REPO_ROOT}:${CONTAINER_WORKDIR}"
  -e "WANDB_MODE=${WANDB_MODE:-disabled}"
  -e "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}"
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
  -e "FFMPEG_THREADS=${FFMPEG_THREADS:-1}"
)

DOCKER_MOUNT_TARGETS=("${CONTAINER_WORKDIR}")

if [[ -n "${DOCKER_NAME:-}" ]]; then
  if [[ ! "${DOCKER_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Invalid DOCKER_NAME=${DOCKER_NAME}" >&2
    exit 1
  fi
  DOCKER_ARGS+=(--name "${DOCKER_NAME}")
fi

add_docker_mount() {
  local source_path="$1"
  local target_path="${2:-$1}"
  local mount_suffix="${3:-}"

  if [[ -z "${source_path}" || ! -e "${source_path}" ]]; then
    return 0
  fi

  for existing_target in "${DOCKER_MOUNT_TARGETS[@]}"; do
    if [[ "${existing_target}" == "${target_path}" ]]; then
      return 0
    fi
  done

  DOCKER_ARGS+=(-v "${source_path}:${target_path}${mount_suffix}")
  DOCKER_MOUNT_TARGETS+=("${target_path}")
}

case "${DOCKER_GPU_MODE}" in
  runtime)
    DOCKER_ARGS+=(
      --runtime=nvidia
      -e "NVIDIA_VISIBLE_DEVICES=${GPUS}"
      -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
    )
    ;;
  gpus)
    DOCKER_ARGS+=(
      --gpus "${GPUS}"
      -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
    )
    ;;
  none)
    ;;
  *)
    echo "Invalid DOCKER_GPU_MODE=${DOCKER_GPU_MODE}; expected runtime, gpus, or none" >&2
    exit 1
    ;;
esac

if [[ "${DOCKER_TTY:-auto}" == "1" || ( "${DOCKER_TTY:-auto}" == "auto" && -t 0 && -t 1 ) ]]; then
  DOCKER_ARGS+=(-it)
elif [[ "${DOCKER_TTY:-auto}" != "0" && "${DOCKER_TTY:-auto}" != "auto" ]]; then
  echo "Invalid DOCKER_TTY=${DOCKER_TTY}; expected auto, 0, or 1" >&2
  exit 1
fi

if [[ -n "${VLA_JEPA_SCRATCH:-}" ]]; then
  add_docker_mount "${VLA_JEPA_SCRATCH}"
fi

if [[ -n "${HF_HOME:-}" ]]; then
  DOCKER_ARGS+=(-e "HF_HOME=${HF_HOME}")
  add_docker_mount "${HF_HOME}"
elif [[ -d "${HOME}/.cache/huggingface" ]]; then
  add_docker_mount "${HOME}/.cache/huggingface" "/root/.cache/huggingface"
fi

if [[ -n "${DATA_ROOT:-}" ]]; then
  add_docker_mount "${DATA_ROOT}"
fi

if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  mkdir -p "${CHECKPOINT_ROOT}"
  add_docker_mount "${CHECKPOINT_ROOT}"
fi

if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  DOCKER_ARGS+=(
    -e "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}"
  )
  add_docker_mount "${GOOGLE_APPLICATION_CREDENTIALS}" "${GOOGLE_APPLICATION_CREDENTIALS}" ":ro"
fi

GCLOUD_SDK_ROOT="${GCLOUD_SDK_ROOT:-/usr/lib/google-cloud-sdk}"
if [[ -z "${GCLOUD_CONFIG_DIR:-}" && -d "/mnt/vla-jepa/gcloud-config" ]]; then
  GCLOUD_CONFIG_DIR="/mnt/vla-jepa/gcloud-config"
fi
if [[ "${MOUNT_GCLOUD:-auto}" != "0" && -d "${GCLOUD_SDK_ROOT}" && -n "${GCLOUD_CONFIG_DIR:-}" && -d "${GCLOUD_CONFIG_DIR}" ]]; then
  DOCKER_ARGS+=(
    -e "CLOUDSDK_CONFIG=/root/.config/gcloud"
    -e "PATH=${GCLOUD_SDK_ROOT}/bin:/opt/conda/envs/vla-jepa/bin:/opt/conda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  )
  add_docker_mount "${GCLOUD_SDK_ROOT}" "${GCLOUD_SDK_ROOT}" ":ro"
  add_docker_mount "${GCLOUD_CONFIG_DIR}" "/root/.config/gcloud"
elif [[ "${MOUNT_GCLOUD:-auto}" == "1" ]]; then
  echo "MOUNT_GCLOUD=1 but could not find GCLOUD_SDK_ROOT=${GCLOUD_SDK_ROOT} and GCLOUD_CONFIG_DIR=${GCLOUD_CONFIG_DIR:-<unset>}" >&2
  exit 1
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  DOCKER_ARGS+=(-e "WANDB_API_KEY=${WANDB_API_KEY}")
fi

OPTIONAL_TRAINING_ENV_VARS=(
  RUN_ID
  CONFIG_YAML
  NUM_PROCESSES
  MAIN_PROCESS_PORT
  VLA_JEPA_SCRATCH
  HF_HUB_CACHE
  TORCH_HOME
  PIP_CACHE_DIR
  TMPDIR
  ACCELERATE_BIN
  STARVLA_USE_DEEPSPEED
  STARVLA_ALLOW_TORCH_COMPILE
  STARVLA_DISABLE_TORCH_COMPILE
  STARVLA_H100_LIFECYCLE_TEST
  DATA_ROOT_DIR
  LIBERO_DATA_ROOT
  REALMAN_DATA_ROOT
  PER_DEVICE_BATCH_SIZE
  DATALOADER_NUM_WORKERS
  DATALOADER_PREFETCH_FACTOR
  DATALOADER_TIMEOUT_SECONDS
  DATALOADER_PERSISTENT_WORKERS
  VIDEO_BACKEND
  VIDEO_BACKEND_NUM_THREADS
  EPOCHS
  MAX_TRAIN_STEPS
  NUM_WARMUP_STEPS
  SAVE_INTERVAL
  EVAL_INTERVAL
  LOGGING_FREQUENCY
  FIND_UNUSED_PARAMETERS
  DDP_GRADIENT_AS_BUCKET_VIEW
  DDP_STATIC_GRAPH
  DDP_BUCKET_CAP_MB
  STARVLA_DETAILED_TIMING
  STARVLA_DETAILED_TIMING_FREQUENCY
  STARVLA_DATASET_TIMING
  STARVLA_DATASET_TIMING_EVERY
  STARVLA_DATASET_TIMING_SLOW_SECONDS
)
for env_var in "${OPTIONAL_TRAINING_ENV_VARS[@]}"; do
  if [[ -n "${!env_var:-}" ]]; then
    DOCKER_ARGS+=(-e "${env_var}=${!env_var}")
  fi
done

if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  DOCKER_ARGS+=(-e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}")
fi
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
  DOCKER_ARGS+=(-e "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}")
fi
if [[ -n "${NCCL_IB_DISABLE:-}" ]]; then
  DOCKER_ARGS+=(-e "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}")
fi

if [[ "$#" -eq 0 ]]; then
  set -- bash
fi

exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "$@"
