#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-/home/mehul/work/vjepa/VLA-JEPA}"
LIBERO_HOME="${LIBERO_HOME:-/home/mehul/work/vjepa/libero_plus_work/src/LIBERO-plus}"
CKPT="${CKPT:?Set CKPT to a VLA-JEPA checkpoint artifact}"
ROOT="${ROOT:-/home/mehul/work/vjepa/eval_videos/local_libero_plus_full_$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-10175}"
CUDA_ID="${CUDA_ID:-0}"
POLICY_ENV="${POLICY_ENV:-vla-jepa-py313-min}"
SIM_ENV="${SIM_ENV:-libero-plus}"
NUM_DDIM_STEPS="${NUM_DDIM_STEPS:-10}"
NUM_TRIALS="${NUM_TRIALS:-1}"
TASK_SUITE="${TASK_SUITE:-libero_mix}"
WITH_STATE="${WITH_STATE:-true}"

mkdir -p "${ROOT}"
cd "${VLA_ROOT}"

export DEBUG=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export PYTHONPATH="${VLA_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

log() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${ROOT}/run.log"
}

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

log "root=${ROOT}"
log "ckpt=${CKPT}"
log "task_suite=${TASK_SUITE} num_trials=${NUM_TRIALS} num_ddim_steps=${NUM_DDIM_STEPS}"

conda run --no-capture-output -n "${POLICY_ENV}" \
    python deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --cuda "${CUDA_ID}" \
    --use_bf16 \
    --policy_output_log_path "${ROOT}/policy_io.jsonl" \
    > "${ROOT}/policy_server.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 180); do
    if timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
        log "policy_server_listening_after=${i}s"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "policy_server_exited"
        tail -200 "${ROOT}/policy_server.log" | tee -a "${ROOT}/run.log"
        exit 1
    fi
    sleep 1
done

if ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    log "policy_server_did_not_open_port=${PORT}"
    tail -200 "${ROOT}/policy_server.log" | tee -a "${ROOT}/run.log"
    exit 1
fi

categories=(
    "Background Textures"
    "Camera Viewpoints"
    "Language Instructions"
    "Light Conditions"
    "Objects Layout"
    "Robot Initial States"
    "Sensor Noise"
)

status=0
for category in "${categories[@]}"; do
    safe_name="${category// /_}"
    out="${ROOT}/${safe_name}"
    mkdir -p "${out}"
    log "===== START ${category} ====="
    if conda run --no-capture-output -n "${SIM_ENV}" \
        python examples/LIBERO/eval_libero.py \
        --args.pretrained-path "${CKPT}" \
        --args.host 127.0.0.1 \
        --args.port "${PORT}" \
        --args.task-suite-name "${TASK_SUITE}" \
        --args.num-trials-per-task "${NUM_TRIALS}" \
        --args.video-out-path "${out}" \
        --args.category-value "${category}" \
        --args.with-state "${WITH_STATE}" \
        --args.num-ddim-steps "${NUM_DDIM_STEPS}" \
        > "${out}/eval.log" 2>&1; then
        code=0
    else
        code=$?
        status=${code}
    fi
    successes=$(find "${out}" -maxdepth 1 -type f -name '*success.mp4' | wc -l)
    failures=$(find "${out}" -maxdepth 1 -type f -name '*failure.mp4' | wc -l)
    log "===== END ${category} code=${code} success=${successes} failure=${failures} ====="
done

log "finished status=${status}"
exit "${status}"
