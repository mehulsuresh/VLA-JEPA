#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-/home/mehul/work/vjepa/VLA-JEPA}"
RUN_ROOT="${RUN_ROOT:-${VLA_ROOT}/checkpoints/libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033}"
BENCH_CONFIG="${BENCH_CONFIG:-${VLA_ROOT}/eval_harness/vla_jepa/configs/libero_plus_weak_categories.yaml}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/mehul/work/vjepa/eval_videos}"

clear_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "clearing port ${port}: ${pids//$'\n'/ }"
        # The model server is launched through conda-run; killing the listener
        # is the reliable guard against accidentally evaluating a later
        # checkpoint through a stale earlier server.
        kill ${pids} 2>/dev/null || true
        sleep 3
    fi
    pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "force clearing port ${port}: ${pids//$'\n'/ }"
        kill -9 ${pids} 2>/dev/null || true
        sleep 1
    fi
}

if [[ "$#" -gt 0 ]]; then
    checkpoint_steps=("$@")
else
    checkpoint_steps=(steps_62500 steps_65000 steps_67500)
fi

trap 'clear_port "${PORT}"' EXIT INT TERM

for step_name in "${checkpoint_steps[@]}"; do
    clear_port "${PORT}"

    CKPT="${RUN_ROOT}/checkpoints/${step_name}/model.safetensors"
    if [[ ! -f "${CKPT}" ]]; then
        echo "missing checkpoint: ${CKPT}" >&2
        exit 1
    fi

    stamp="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${OUTPUT_ROOT}/harness_libero_plus_weak_${step_name}_b${MAX_BATCH_SIZE}_${stamp}"
    EVAL_ID="vlajepa-libero-plus-weak-${step_name}-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${OUTPUT_DIR}"

    cat > "${OUTPUT_DIR}/launch.env" <<EOF
CKPT=${CKPT}
RUN_ROOT=${RUN_ROOT}
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
EOF

    echo "== ${step_name} =="
    echo "ckpt=${CKPT}"
    echo "output=${OUTPUT_DIR}"
    echo "eval_id=${EVAL_ID}"

    CKPT="${CKPT}" \
    BENCH_CONFIG="${BENCH_CONFIG}" \
    SERVER_CONFIG="${SERVER_CONFIG}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    EVAL_ID="${EVAL_ID}" \
    NUM_SHARDS="${NUM_SHARDS}" \
    MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
    MAX_WAIT_TIME="${MAX_WAIT_TIME}" \
    CHUNK_SIZE="${CHUNK_SIZE}" \
    NUM_DDIM_STEPS="${NUM_DDIM_STEPS}" \
    PORT="${PORT}" \
    CUDA_ID="${CUDA_ID}" \
    POLICY_ENV="${POLICY_ENV}" \
    SIM_ENV="${SIM_ENV}" \
    "${VLA_ROOT}/eval_harness/vla_jepa/scripts/run_sharded_eval.sh"

    clear_port "${PORT}"
done
