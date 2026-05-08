#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/lib/training_env.sh"
starvla_configure_common_training_env ens8

export VLA_JEPA_SCRATCH="${VLA_JEPA_SCRATCH:-/mnt/vla-jepa}"
export HF_HOME="${HF_HOME:-${VLA_JEPA_SCRATCH}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${VLA_JEPA_SCRATCH}/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${VLA_JEPA_SCRATCH}/cache/pip}"
export TMPDIR="${TMPDIR:-${VLA_JEPA_SCRATCH}/tmp}"
export STARVLA_MOGE_REPO_PATH="${STARVLA_MOGE_REPO_PATH:-${VLA_JEPA_SCRATCH}/src/MoGe}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}"

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || echo "${REPO_ROOT}/.venv/bin/accelerate")}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_canonical_multi_subset_a100x8_moge_vjteacher_pilot.yaml}"
starvla_configure_deepspeed_launch "${REPO_ROOT}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${STARVLA_DEFAULT_ACCELERATE_CONFIG}}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
RUN_ID="${RUN_ID:-robot_ft_canonical_multi_subset_a100x8_moge_vjteacher_pilot_$(date +%Y%m%d_%H%M%S)}"
starvla_configure_accelerate_cluster_args

cd "${REPO_ROOT}"

"${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  "${STARVLA_ACCELERATE_CLUSTER_ARGS[@]}" \
  --mixed_precision no \
  --dynamo_backend no \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${RUN_ID}" \
  "${STARVLA_EXTRA_TRAIN_ARGS[@]}" \
  "$@"
