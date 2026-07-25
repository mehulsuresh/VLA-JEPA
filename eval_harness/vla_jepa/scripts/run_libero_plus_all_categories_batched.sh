#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-/home/mehul/work/vjepa/VLA-JEPA}"
RUN_NAME="${RUN_NAME:-libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033}"
RUN_ROOT="${RUN_ROOT:-${VLA_ROOT}/checkpoints/${RUN_NAME}}"
CKPT="${CKPT:-}"
if [[ -z "${CKPT}" ]]; then
    if [[ -f "${RUN_ROOT}/latest_eval_checkpoint.txt" ]]; then
        CKPT="$(<"${RUN_ROOT}/latest_eval_checkpoint.txt")"
    else
        CKPT="$(find "${RUN_ROOT}/checkpoints" -maxdepth 2 -type f -name model.safetensors | sort -V | tail -1)"
    fi
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "Set CKPT or run sync_latest_libero_checkpoint.sh first." >&2
    exit 1
fi

STEP_NAME="$(basename "$(dirname "${CKPT}")")"
BENCH_CONFIG="${BENCH_CONFIG:-${VLA_ROOT}/eval_harness/vla_jepa/configs/libero_plus_all_categories.yaml}"
SERVER_CONFIG="${SERVER_CONFIG:-${VLA_ROOT}/eval_harness/vla_jepa/configs/model_server_libero_plus.yaml}"
NUM_SHARDS="${NUM_SHARDS:-24}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-24}"
MAX_WAIT_TIME="${MAX_WAIT_TIME:-0.05}"
CHUNK_SIZE="${CHUNK_SIZE:-7}"
NUM_DDIM_STEPS="${NUM_DDIM_STEPS:-10}"
PORT="${PORT:-8001}"
CUDA_ID="${CUDA_ID:-0}"
POLICY_ENV="${POLICY_ENV:-vla-jepa-py313-min}"
SIM_ENV="${SIM_ENV:-libero-plus}"
SESSION="${SESSION:-libero_plus_${STEP_NAME}_all_categories_b${MAX_BATCH_SIZE}_eval}"
EVAL_ID="${EVAL_ID:-vlajepa-libero-plus-${STEP_NAME}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/mehul/work/vjepa/eval_videos/harness_libero_plus_${STEP_NAME}_all_categories_b${MAX_BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/launch.env" <<EOF
CKPT=${CKPT}
BENCH_CONFIG=${BENCH_CONFIG}
SERVER_CONFIG=${SERVER_CONFIG}
OUTPUT_DIR=${OUTPUT_DIR}
EVAL_ID=${EVAL_ID}
NUM_SHARDS=${NUM_SHARDS}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE}
MAX_WAIT_TIME=${MAX_WAIT_TIME}
CHUNK_SIZE=${CHUNK_SIZE}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS}
PORT=${PORT}
CUDA_ID=${CUDA_ID}
POLICY_ENV=${POLICY_ENV}
SIM_ENV=${SIM_ENV}
SESSION=${SESSION}
EOF

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 1
fi

tmux new-session -d -s "${SESSION}" "cd '${VLA_ROOT}' && \
  CKPT='${CKPT}' \
  BENCH_CONFIG='${BENCH_CONFIG}' \
  SERVER_CONFIG='${SERVER_CONFIG}' \
  OUTPUT_DIR='${OUTPUT_DIR}' \
  EVAL_ID='${EVAL_ID}' \
  NUM_SHARDS='${NUM_SHARDS}' \
  PORT='${PORT}' \
  CUDA_ID='${CUDA_ID}' \
  MAX_BATCH_SIZE='${MAX_BATCH_SIZE}' \
  MAX_WAIT_TIME='${MAX_WAIT_TIME}' \
  CHUNK_SIZE='${CHUNK_SIZE}' \
  NUM_DDIM_STEPS='${NUM_DDIM_STEPS}' \
  POLICY_ENV='${POLICY_ENV}' \
  SIM_ENV='${SIM_ENV}' \
  ./eval_harness/vla_jepa/scripts/run_libero_plus_sharded.sh"

echo "session=${SESSION}"
echo "output=${OUTPUT_DIR}"
echo "eval_id=${EVAL_ID}"
echo "ckpt=${CKPT}"
