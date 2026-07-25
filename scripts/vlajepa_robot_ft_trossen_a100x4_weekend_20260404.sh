#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/lib/training_env.sh"
starvla_configure_common_training_env ens8
starvla_cleanup_stale_training_sidecars

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || echo "${HOME}/miniconda3/envs/vla-jepa-vjepa21/bin/accelerate")}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_a100x4_weekend_20260404.yaml}"
starvla_configure_deepspeed_launch "${REPO_ROOT}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${STARVLA_DEFAULT_ACCELERATE_CONFIG}}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
starvla_configure_accelerate_cluster_args

cd "${REPO_ROOT}"

"${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  "${STARVLA_ACCELERATE_CLUSTER_ARGS[@]}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  "${STARVLA_EXTRA_TRAIN_ARGS[@]}" \
  "$@"
