#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/lib/training_env.sh"
starvla_configure_common_training_env eno1
export STARVLA_CLEANUP_SHELL_PID="$$"

starvla_cleanup_stale_training_sidecars
trap starvla_cleanup_stale_training_sidecars EXIT

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || echo "${HOME}/miniconda3/envs/vla-jepa-vjepa21/bin/accelerate")}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_5090_lerobot.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}"
RUN_ID="${RUN_ID:-robot_ft_trossen_vjepa21_small_5090_lerobot_$(date +%Y%m%d_%H%M%S)}"
COMPILE_FULL_MODEL="${COMPILE_FULL_MODEL:-false}"
RUNTIME_TIMING_LOGGING="${RUNTIME_TIMING_LOGGING:-true}"
GPU_VIDEO_DECODE_ON_RANK="${GPU_VIDEO_DECODE_ON_RANK:-false}"
CPU_VIDEO_DECODE_DROP_WORKER_IMAGES="${CPU_VIDEO_DECODE_DROP_WORKER_IMAGES:-true}"
EXTRA_TRAIN_ARGS=(
  --trainer.compile_full_model "${COMPILE_FULL_MODEL}"
  --datasets.vla_data.runtime_timing_logging "${RUNTIME_TIMING_LOGGING}"
  --datasets.vla_data.gpu_video_decode_on_rank "${GPU_VIDEO_DECODE_ON_RANK}"
  --datasets.vla_data.cpu_video_decode_drop_worker_images "${CPU_VIDEO_DECODE_DROP_WORKER_IMAGES}"
)
if [[ -n "${SAVE_BEST_ONLY:-}" ]]; then
  EXTRA_TRAIN_ARGS+=(--trainer.save_best_only "${SAVE_BEST_ONLY}")
fi
if [[ -n "${OPTIMIZER_NAME:-}" ]]; then
  EXTRA_TRAIN_ARGS+=(--trainer.optimizer.name "${OPTIMIZER_NAME}")
  if [[ "${OPTIMIZER_NAME}" == "AdamW" ]]; then
    EXTRA_TRAIN_ARGS+=(--trainer.optimizer.fused true)
  fi
fi

cd "${REPO_ROOT}"

"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --mixed_precision no \
  --dynamo_backend no \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${RUN_ID}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "$@"
