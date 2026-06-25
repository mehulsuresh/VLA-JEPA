#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/lib/training_env.sh"
starvla_configure_common_training_env ens8

export STARVLA_USE_DEEPSPEED="${STARVLA_USE_DEEPSPEED:-0}"
export VLA_JEPA_SCRATCH="${VLA_JEPA_SCRATCH:-/mnt/vla-jepa}"
export HF_HOME="${HF_HOME:-${VLA_JEPA_SCRATCH}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${VLA_JEPA_SCRATCH}/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${VLA_JEPA_SCRATCH}/cache/pip}"
export TMPDIR="${TMPDIR:-${VLA_JEPA_SCRATCH}/tmp}"
export STARVLA_MOGE_REPO_PATH="${STARVLA_MOGE_REPO_PATH:-${VLA_JEPA_SCRATCH}/src/MoGe}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}"

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || echo "${REPO_ROOT}/.venv/bin/accelerate")}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_libero_plus_a100x8_qwen3_full_moge_vitb_vjepa_large.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29573}"
RUN_ID="${RUN_ID:-robot_ft_libero_plus_a100x8_qwen3_full_moge_vitb_vjepa_large_$(date +%Y%m%d_%H%M%S)}"
starvla_configure_accelerate_cluster_args

STARVLA_EXTRA_TRAIN_ARGS=()
ACCELERATE_LAUNCH_ARGS=(
  --num_processes "${NUM_PROCESSES}"
  "${STARVLA_ACCELERATE_CLUSTER_ARGS[@]}"
  --dynamo_backend no
  --main_process_port "${MAIN_PROCESS_PORT}"
)

if [[ "${STARVLA_USE_DEEPSPEED}" == "1" ]]; then
  STARVLA_DEEPSPEED_STAGE="${STARVLA_DEEPSPEED_STAGE:-2}"
  if [[ -z "${ACCELERATE_CONFIG:-}" ]]; then
    if [[ "${STARVLA_DEEPSPEED_STAGE}" == "3" || "${STARVLA_DEEPSPEED_STAGE,,}" == "zero3" ]]; then
      ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero3.yaml"
    else
      ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2_stable.yaml"
    fi
  fi
  ACCELERATE_LAUNCH_ARGS=(--config_file "${ACCELERATE_CONFIG}" "${ACCELERATE_LAUNCH_ARGS[@]}" --mixed_precision no)
  STARVLA_EXTRA_TRAIN_ARGS+=(
    --trainer.compile_qwen_model false
    --trainer.compile_action_model false
    --trainer.compile_vj_predictor false
    --trainer.compile_vj_encoder false
    --trainer.compile_full_model false
  )
else
  ACCELERATE_LAUNCH_ARGS+=(--mixed_precision bf16)
  STARVLA_EXTRA_TRAIN_ARGS+=(--framework.qwenvl.device_map null)
fi

if [[ "${STARVLA_ALLOW_TORCH_COMPILE:-0}" != "1" ]]; then
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
else
  unset TORCH_COMPILE_DISABLE
  unset TORCHDYNAMO_DISABLE
fi
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"

TRAIN_ARGS=()
add_arg_if_env() {
  local env_name="$1"
  shift
  if [[ -n "${!env_name:-}" ]]; then
    TRAIN_ARGS+=("$@" "${!env_name}")
  fi
}

add_arg_if_env LIBERO_DATA_ROOT --datasets.vla_data.data_root_dir
add_arg_if_env PER_DEVICE_BATCH_SIZE --datasets.vla_data.per_device_batch_size
add_arg_if_env DATALOADER_NUM_WORKERS --datasets.vla_data.num_workers
add_arg_if_env DATALOADER_PREFETCH_FACTOR --datasets.vla_data.prefetch_factor
add_arg_if_env DATALOADER_TIMEOUT_SECONDS --datasets.vla_data.dataloader_timeout_seconds
add_arg_if_env DATALOADER_PERSISTENT_WORKERS --datasets.vla_data.persistent_workers
add_arg_if_env VIDEO_BACKEND --datasets.vla_data.video_backend
add_arg_if_env VIDEO_BACKEND_NUM_THREADS --datasets.vla_data.video_backend_num_threads
add_arg_if_env EPOCHS --trainer.epochs
add_arg_if_env MAX_TRAIN_STEPS --trainer.max_train_steps
add_arg_if_env NUM_WARMUP_STEPS --trainer.num_warmup_steps
add_arg_if_env SAVE_INTERVAL --trainer.save_interval
add_arg_if_env EVAL_INTERVAL --trainer.eval_interval
add_arg_if_env LOGGING_FREQUENCY --trainer.logging_frequency
add_arg_if_env FIND_UNUSED_PARAMETERS --trainer.find_unused_parameters
add_arg_if_env DDP_GRADIENT_AS_BUCKET_VIEW --trainer.ddp_gradient_as_bucket_view
add_arg_if_env DDP_STATIC_GRAPH --trainer.ddp_static_graph
add_arg_if_env DDP_BUCKET_CAP_MB --trainer.ddp_bucket_cap_mb

cd "${REPO_ROOT}"

"${ACCELERATE_BIN}" launch \
  "${ACCELERATE_LAUNCH_ARGS[@]}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${RUN_ID}" \
  "${TRAIN_ARGS[@]}" \
  "${STARVLA_EXTRA_TRAIN_ARGS[@]}" \
  "$@"
