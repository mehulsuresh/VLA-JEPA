#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || echo "${HOME}/miniconda3/envs/vla-jepa-vjepa21/bin/accelerate")}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_canonical_subset_5090_smoke.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29521}"
RUN_ID="${RUN_ID:-robot_ft_canonical_subset_5090_smoke_$(date +%Y%m%d_%H%M%S)}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
export TMPDIR="${TMPDIR:-${HOME}/tmp}"
mkdir -p "${TMPDIR}"

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
  "$@"
