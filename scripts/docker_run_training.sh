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

if [[ -n "${HF_HOME:-}" ]]; then
  DOCKER_ARGS+=(-e "HF_HOME=${HF_HOME}" -v "${HF_HOME}:${HF_HOME}")
elif [[ -d "${HOME}/.cache/huggingface" ]]; then
  DOCKER_ARGS+=(-v "${HOME}/.cache/huggingface:/root/.cache/huggingface")
fi

if [[ -n "${DATA_ROOT:-}" ]]; then
  DOCKER_ARGS+=(-v "${DATA_ROOT}:${DATA_ROOT}")
fi

if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  mkdir -p "${CHECKPOINT_ROOT}"
  DOCKER_ARGS+=(-v "${CHECKPOINT_ROOT}:${CHECKPOINT_ROOT}")
fi

if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  DOCKER_ARGS+=(
    -e "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}"
    -v "${GOOGLE_APPLICATION_CREDENTIALS}:${GOOGLE_APPLICATION_CREDENTIALS}:ro"
  )
fi

GCLOUD_SDK_ROOT="${GCLOUD_SDK_ROOT:-/usr/lib/google-cloud-sdk}"
if [[ -z "${GCLOUD_CONFIG_DIR:-}" && -d "/mnt/vla-jepa/gcloud-config" ]]; then
  GCLOUD_CONFIG_DIR="/mnt/vla-jepa/gcloud-config"
fi
if [[ "${MOUNT_GCLOUD:-auto}" != "0" && -d "${GCLOUD_SDK_ROOT}" && -n "${GCLOUD_CONFIG_DIR:-}" && -d "${GCLOUD_CONFIG_DIR}" ]]; then
  DOCKER_ARGS+=(
    -v "${GCLOUD_SDK_ROOT}:${GCLOUD_SDK_ROOT}:ro"
    -v "${GCLOUD_CONFIG_DIR}:/root/.config/gcloud"
    -e "CLOUDSDK_CONFIG=/root/.config/gcloud"
    -e "PATH=${GCLOUD_SDK_ROOT}/bin:/opt/conda/envs/vla-jepa/bin:/opt/conda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  )
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
