#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-/home/mehul/work/vjepa/VLA-JEPA}"
LIBERO_HOME="${LIBERO_HOME:-/home/mehul/work/vjepa/libero_plus_work/src/LIBERO-plus}"
CKPT="${CKPT:?Set CKPT to a VLA-JEPA checkpoint artifact}"
ROOT="${ROOT:-/home/mehul/work/vjepa/eval_videos/local_libero_plus_parallel_$(date +%Y%m%d_%H%M%S)}"
BASE_PORT="${BASE_PORT:-10250}"
CUDA_IDS="${CUDA_IDS:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
POLICY_ENV="${POLICY_ENV:-vla-jepa-py313-min}"
SIM_ENV="${SIM_ENV:-libero-plus}"
NUM_DDIM_STEPS="${NUM_DDIM_STEPS:-10}"
NUM_TRIALS="${NUM_TRIALS:-1}"
TASK_SUITE="${TASK_SUITE:-libero_mix}"
WITH_STATE="${WITH_STATE:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
skip_existing_args=()
case "${SKIP_EXISTING}" in
    true|True|TRUE|1|yes|Yes|YES)
        skip_existing_args+=(--args.skip-existing)
        ;;
    false|False|FALSE|0|no|No|NO)
        skip_existing_args+=(--args.no-skip-existing)
        ;;
    *)
        echo "Invalid SKIP_EXISTING=${SKIP_EXISTING}; expected true or false" >&2
        exit 2
        ;;
esac

mkdir -p "${ROOT}"
cd "${VLA_ROOT}"

export DEBUG=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export PYTHONPATH="${VLA_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

IFS=',' read -r -a CUDA_ID_LIST <<< "${CUDA_IDS}"

log() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${ROOT}/run.log"
}

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

category_count() {
    local category="$1"
    conda run --no-capture-output -n "${SIM_ENV}" python - "${category}" <<'PY'
import os
import sys

from libero.libero import benchmark

category = sys.argv[1]
suite = benchmark.get_benchmark_dict()["libero_mix"](category_value=category)
print(suite.n_tasks)
PY
}

wait_for_port() {
    local port="$1"
    local server_pid="$2"
    local server_log="$3"
    for i in $(seq 1 180); do
        if timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            log "policy_server_exited port=${port}"
            tail -200 "${server_log}" | tee -a "${ROOT}/run.log"
            return 1
        fi
        sleep 1
    done
    log "policy_server_did_not_open_port=${port}"
    tail -200 "${server_log}" | tee -a "${ROOT}/run.log"
    return 1
}

run_shard() {
    local category="$1"
    local worker_idx="$2"
    local task_start="$3"
    local max_tasks="$4"
    local safe_name="${category// /_}"
    local cuda_idx="${CUDA_ID_LIST[$((worker_idx % ${#CUDA_ID_LIST[@]}))]}"
    local port=$((BASE_PORT + worker_idx))
    local out="${ROOT}/${safe_name}"
    local shard_tag="${safe_name}_start${task_start}_count${max_tasks}_w${worker_idx}"
    local server_log="${ROOT}/policy_server_${shard_tag}.log"
    local eval_log="${out}/eval_${shard_tag}.log"
    local policy_log="${ROOT}/policy_io_${shard_tag}.jsonl"
    local server_pid=""
    local code=0

    mkdir -p "${out}"
    log "START shard=${shard_tag} port=${port} cuda=${cuda_idx}"
    conda run --no-capture-output -n "${POLICY_ENV}" \
        python deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT}" \
        --host 127.0.0.1 \
        --port "${port}" \
        --cuda "${cuda_idx}" \
        --use_bf16 \
        --policy_output_log_path "${policy_log}" \
        > "${server_log}" 2>&1 &
    server_pid=$!

    if wait_for_port "${port}" "${server_pid}" "${server_log}"; then
        if conda run --no-capture-output -n "${SIM_ENV}" \
            python examples/LIBERO/eval_libero.py \
            --args.pretrained-path "${CKPT}" \
            --args.host 127.0.0.1 \
            --args.port "${port}" \
            --args.task-suite-name "${TASK_SUITE}" \
            --args.num-trials-per-task "${NUM_TRIALS}" \
            --args.video-out-path "${out}" \
            --args.category-value "${category}" \
            --args.with-state "${WITH_STATE}" \
            --args.num-ddim-steps "${NUM_DDIM_STEPS}" \
            --args.task-start "${task_start}" \
            --args.max-tasks "${max_tasks}" \
            "${skip_existing_args[@]}" \
            > "${eval_log}" 2>&1; then
            code=0
        else
            code=$?
        fi
    else
        code=1
    fi

    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    local successes failures
    successes=$(find "${out}" -maxdepth 1 -type f -name '*success.mp4' | wc -l)
    failures=$(find "${out}" -maxdepth 1 -type f -name '*failure.mp4' | wc -l)
    log "END shard=${shard_tag} code=${code} category_success=${successes} category_failure=${failures}"
    return "${code}"
}

categories=(
    "Background Textures"
    "Camera Viewpoints"
    "Language Instructions"
    "Light Conditions"
    "Objects Layout"
    "Robot Initial States"
    "Sensor Noise"
)

log "root=${ROOT}"
log "ckpt=${CKPT}"
log "task_suite=${TASK_SUITE} num_trials=${NUM_TRIALS} num_ddim_steps=${NUM_DDIM_STEPS}"
log "num_workers=${NUM_WORKERS} cuda_ids=${CUDA_IDS} skip_existing=${SKIP_EXISTING}"

status=0
for category in "${categories[@]}"; do
    total_tasks="$(category_count "${category}" | tail -n 1)"
    chunk_size=$(( (total_tasks + NUM_WORKERS - 1) / NUM_WORKERS ))
    log "CATEGORY ${category} total_tasks=${total_tasks} chunk_size=${chunk_size}"
    pids=()
    for worker_idx in $(seq 0 $((NUM_WORKERS - 1))); do
        start=$((worker_idx * chunk_size))
        if [[ "${start}" -ge "${total_tasks}" ]]; then
            continue
        fi
        count="${chunk_size}"
        if [[ $((start + count)) -gt "${total_tasks}" ]]; then
            count=$((total_tasks - start))
        fi
        run_shard "${category}" "${worker_idx}" "${start}" "${count}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    pids=()
done

log "finished status=${status}"
exit "${status}"
