#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-/home/mehul/work/vjepa/VLA-JEPA}"
HARNESS_ROOT="${HARNESS_ROOT:-/tmp/vla-evaluation-harness}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/home/mehul/work/vjepa/libero_plus_work/src/LIBERO-plus}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/mehul/work/reward_model_small/pi0}"

POLICY_ENV="${POLICY_ENV:-openpi_realman}"
SIM_ENV="${SIM_ENV:-libero-plus}"
BENCH_CONFIG="${BENCH_CONFIG:-${VLA_ROOT}/eval_harness/vla_jepa/configs/libero_plus_all_categories.yaml}"
PI0_CONFIG_NAME="${PI0_CONFIG_NAME:-pi05_libero}"
PI0_CHECKPOINT="${PI0_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_libero}"

NUM_SHARDS="${NUM_SHARDS:-24}"
PORT="${PORT:-8015}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-224}"
STATE_DIM="${STATE_DIM:-8}"
EVAL_ID="${EVAL_ID:-pi05-libero-plus-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/mehul/work/vjepa/eval_videos/pi05_libero_plus_all_categories_s${NUM_SHARDS}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"

export VLA_JEPA_ROOT="${VLA_ROOT}"
export VLA_EVAL_HARNESS_ROOT="${HARNESS_ROOT}"
export VLA_JEPA_EVAL_OUTPUT="${OUTPUT_DIR}"
export PYTHONPATH="${HARNESS_ROOT}/src:${VLA_ROOT}:${OPENPI_ROOT}/src:${PYTHONPATH:-}"
if [[ -d "${LIBERO_PLUS_ROOT}/libero" ]]; then
    export LIBERO_HOME="${LIBERO_HOME:-${LIBERO_PLUS_ROOT}}"
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_PLUS_ROOT}/libero}"
    export PYTHONPATH="${LIBERO_PLUS_ROOT}:${PYTHONPATH}"
fi
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# OpenPI uses JAX. JAX normally preallocates most GPU memory on first use,
# which can starve cuBLAS and the concurrent EGL simulator shards.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.60}"

cat > "${OUTPUT_DIR}/launch.env" <<EOF
VLA_ROOT=${VLA_ROOT}
HARNESS_ROOT=${HARNESS_ROOT}
LIBERO_PLUS_ROOT=${LIBERO_PLUS_ROOT}
OPENPI_ROOT=${OPENPI_ROOT}
POLICY_ENV=${POLICY_ENV}
SIM_ENV=${SIM_ENV}
BENCH_CONFIG=${BENCH_CONFIG}
PI0_CONFIG_NAME=${PI0_CONFIG_NAME}
PI0_CHECKPOINT=${PI0_CHECKPOINT}
NUM_SHARDS=${NUM_SHARDS}
PORT=${PORT}
CHUNK_SIZE=${CHUNK_SIZE}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION}
STATE_DIM=${STATE_DIM}
XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}
EVAL_ID=${EVAL_ID}
OUTPUT_DIR=${OUTPUT_DIR}
EOF

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
log "checkpoint=${PI0_CHECKPOINT}"
log "starting pi0 model server"

conda run --no-capture-output -n "${POLICY_ENV}" \
    env PYTHONPATH="${PYTHONPATH}" \
        XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}" \
        XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION}" \
    python "${HARNESS_ROOT}/src/vla_eval/model_servers/pi0.py" \
        --args.config_name "${PI0_CONFIG_NAME}" \
        --args.checkpoint "${PI0_CHECKPOINT}" \
        --args.image_key observation/image \
        --args.wrist_image_key observation/wrist_image \
        --args.state_key observation/state \
        --args.state_dim "${STATE_DIM}" \
        --args.image_resolution "${IMAGE_RESOLUTION}" \
        --args.chunk_size "${CHUNK_SIZE}" \
        --port "${PORT}" \
    > "${OUTPUT_DIR}/server.log" 2>&1 &
server_pid=$!

server_ready=0
for i in $(seq 1 900); do
    if python - "${PORT}" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

port = int(sys.argv[1])
with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
    raise SystemExit(0 if response.status == 200 else 1)
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
    if [[ $((i % 60)) -eq 0 ]]; then
        log "waiting_for_server=${i}s"
        tail -20 "${OUTPUT_DIR}/server.log" | tee -a "${OUTPUT_DIR}/run.log"
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
