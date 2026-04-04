export NCCL_IB_DISABLE=1
DEFAULT_SOCKET_IFNAME="${DEFAULT_SOCKET_IFNAME:-lo}"

export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${DEFAULT_SOCKET_IFNAME}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export TMPDIR=/home/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1

export WANDB_MODE=disabled
export STARVLA_USE_DEEPSPEED="${STARVLA_USE_DEEPSPEED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEEPSPEED_STAGE="${STARVLA_DEEPSPEED_STAGE:-2}"
ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml"
EXTRA_TRAIN_ARGS=()
if [[ "${DEEPSPEED_STAGE}" == "3" || "${DEEPSPEED_STAGE,,}" == "zero3" ]]; then
  ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero3.yaml"
  EXTRA_TRAIN_ARGS+=(--framework.qwenvl.device_map null)
fi

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes 8 \
  ./starVLA/training/train_vlajepa_cotrain.py \
  --config_yaml "${REPO_ROOT}/scripts/config/vlajepa_cotrain.yaml" \
  "${EXTRA_TRAIN_ARGS[@]}"
