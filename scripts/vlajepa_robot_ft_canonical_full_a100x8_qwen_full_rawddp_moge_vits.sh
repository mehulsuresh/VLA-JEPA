#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/lib/training_env.sh"
starvla_configure_common_training_env ens8
starvla_configure_accelerate_cluster_args

export STARVLA_USE_DEEPSPEED=0
export CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29601}"
export RUN_ID="${RUN_ID:-robot_ft_canonical_full_a100x8_qwen_full_rawddp_b26_moge_vits_$(date +%Y%m%d_%H%M%S)}"

export VLA_JEPA_SCRATCH="${VLA_JEPA_SCRATCH:-/mnt/vla-jepa}"
export HF_HOME="${HF_HOME:-${VLA_JEPA_SCRATCH}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${VLA_JEPA_SCRATCH}/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${VLA_JEPA_SCRATCH}/cache/pip}"
export TMPDIR="${TMPDIR:-${VLA_JEPA_SCRATCH}/tmp}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}"

if [[ "${STARVLA_ALLOW_TORCH_COMPILE:-0}" != "1" ]]; then
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
else
  unset TORCH_COMPILE_DISABLE
  unset TORCHDYNAMO_DISABLE
fi

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || echo "${REPO_ROOT}/.venv/bin/accelerate")}"

cd "${REPO_ROOT}"

"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES}" \
  "${STARVLA_ACCELERATE_CLUSTER_ARGS[@]}" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${RUN_ID}" \
  --framework.qwenvl.device_map null \
  "$@"
