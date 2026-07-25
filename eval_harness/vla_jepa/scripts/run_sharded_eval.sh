#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-/home/mehul/work/vjepa/VLA-JEPA}"
HARNESS_ROOT="${HARNESS_ROOT:-/tmp/vla-evaluation-harness}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/home/mehul/work/vjepa/libero_plus_work/src/LIBERO-plus}"
POLICY_ENV="${POLICY_ENV:-vla-jepa-py313-min}"
SIM_ENV="${SIM_ENV:-libero-plus}"
BENCH_CONFIG="${BENCH_CONFIG:?Set BENCH_CONFIG to a vla-evaluation-harness benchmark YAML}"
SERVER_CONFIG="${SERVER_CONFIG:-${VLA_ROOT}/eval_harness/vla_jepa/configs/model_server_generic.yaml}"
CKPT="${CKPT:?Set CKPT to a VLA-JEPA checkpoint artifact or checkpoint directory}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/mehul/work/vjepa/eval_videos/harness_eval_$(date +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${NUM_SHARDS:-8}"
PORT="${PORT:-8000}"
CUDA_ID="${CUDA_ID:-0}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_WAIT_TIME="${MAX_WAIT_TIME:-0.05}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
NUM_DDIM_STEPS="${NUM_DDIM_STEPS:-}"
EVAL_ID="${EVAL_ID:-vlajepa-harness-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"
export VLA_JEPA_ROOT="${VLA_ROOT}"
export VLA_EVAL_HARNESS_ROOT="${HARNESS_ROOT}"
export VLA_JEPA_EVAL_OUTPUT="${OUTPUT_DIR}"
export PYTHONPATH="${HARNESS_ROOT}/src:${VLA_ROOT}:${PYTHONPATH:-}"
if [[ -d "${LIBERO_PLUS_ROOT}/libero" ]]; then
    export LIBERO_HOME="${LIBERO_HOME:-${LIBERO_PLUS_ROOT}}"
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_PLUS_ROOT}/libero}"
    export PYTHONPATH="${LIBERO_PLUS_ROOT}:${PYTHONPATH}"
fi
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

log() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${OUTPUT_DIR}/run.log"
}

server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

log "output_dir=${OUTPUT_DIR}"
log "eval_id=${EVAL_ID} shards=${NUM_SHARDS} port=${PORT}"
log "bench_config=${BENCH_CONFIG}"
log "server_config=${SERVER_CONFIG}"
log "ckpt=${CKPT}"
log "starting model server"

server_env=(
    "CONFIG=${SERVER_CONFIG}"
    "CKPT=${CKPT}"
    "PORT=${PORT}"
    "CUDA_ID=${CUDA_ID}"
    "MAX_BATCH_SIZE=${MAX_BATCH_SIZE}"
    "MAX_WAIT_TIME=${MAX_WAIT_TIME}"
)
if [[ -n "${CHUNK_SIZE}" ]]; then
    server_env+=("CHUNK_SIZE=${CHUNK_SIZE}")
fi
if [[ -n "${NUM_DDIM_STEPS}" ]]; then
    server_env+=("NUM_DDIM_STEPS=${NUM_DDIM_STEPS}")
fi

env "${server_env[@]}" "${VLA_ROOT}/eval_harness/vla_jepa/scripts/run_server.sh" \
    > "${OUTPUT_DIR}/server.log" 2>&1 &
server_pid=$!

server_ready=0
for i in $(seq 1 240); do
    if python - "${PORT}" <<'PY' >/dev/null 2>&1
import sys, urllib.request
port = int(sys.argv[1])
with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
    raise SystemExit(0 if r.status == 200 else 1)
PY
    then
        log "server_ready_after=${i}s"
        server_ready=1
        break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        log "server_exited"
        tail -200 "${OUTPUT_DIR}/server.log" | tee -a "${OUTPUT_DIR}/run.log"
        exit 1
    fi
    sleep 1
done

if [[ "${server_ready}" != "1" ]]; then
    log "server_health_timeout"
    tail -200 "${OUTPUT_DIR}/server.log" | tee -a "${OUTPUT_DIR}/run.log"
    exit 1
fi

run_vla_eval() {
    conda run --no-capture-output -n "${SIM_ENV}" \
        env PYTHONPATH="${HARNESS_ROOT}/src:${PYTHONPATH:-}" \
            VLA_JEPA_EVAL_OUTPUT="${OUTPUT_DIR}" \
            python -m vla_eval.cli.main "$@"
}

log "launching shards"
pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    run_vla_eval run \
        --no-docker \
        --config "${BENCH_CONFIG}" \
        --server-url "ws://127.0.0.1:${PORT}" \
        --output-dir "${OUTPUT_DIR}" \
        --eval-id "${EVAL_ID}" \
        --shard-id "${shard_id}" \
        --num-shards "${NUM_SHARDS}" \
        > "${OUTPUT_DIR}/shard_${shard_id}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=$((failed + 1))
    fi
done

log "shards_finished failed=${failed}"
log "merging"
if ! run_vla_eval merge --config "${BENCH_CONFIG}" --output-dir "${OUTPUT_DIR}" --eval-id "${EVAL_ID}" \
    > "${OUTPUT_DIR}/merge.log" 2>&1; then
    log "merge_failed"
    tail -200 "${OUTPUT_DIR}/merge.log" | tee -a "${OUTPUT_DIR}/run.log"
    exit 1
fi

log "done"
exit "${failed}"
