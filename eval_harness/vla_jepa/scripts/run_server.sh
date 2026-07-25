#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-/home/mehul/work/vjepa/VLA-JEPA}"
HARNESS_ROOT="${HARNESS_ROOT:-/tmp/vla-evaluation-harness}"
POLICY_ENV="${POLICY_ENV:-vla-jepa-py313-min}"
CONFIG="${CONFIG:-${VLA_ROOT}/eval_harness/vla_jepa/configs/model_server_libero_plus.yaml}"
CKPT="${CKPT:?Set CKPT to a VLA-JEPA checkpoint artifact or checkpoint directory}"
PORT="${PORT:-8000}"
CUDA_ID="${CUDA_ID:-0}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_WAIT_TIME="${MAX_WAIT_TIME:-0.05}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
NUM_DDIM_STEPS="${NUM_DDIM_STEPS:-}"

export VLA_JEPA_ROOT="${VLA_ROOT}"
export VLA_EVAL_HARNESS_ROOT="${HARNESS_ROOT}"
export PYTHONPATH="${HARNESS_ROOT}/src:${VLA_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${VLA_ROOT}"
cmd=(
    conda run --no-capture-output -n "${POLICY_ENV}"
    python eval_harness/vla_jepa/model_server.py \
    --config "${CONFIG}" \
    --port "${PORT}" \
    --args.checkpoint "${CKPT}" \
    --args.cuda "${CUDA_ID}" \
    --args.max_batch_size "${MAX_BATCH_SIZE}" \
    --args.max_wait_time "${MAX_WAIT_TIME}"
)
if [[ -n "${CHUNK_SIZE}" ]]; then
    cmd+=(--args.chunk_size "${CHUNK_SIZE}")
fi
if [[ -n "${NUM_DDIM_STEPS}" ]]; then
    cmd+=(--args.num_ddim_steps "${NUM_DDIM_STEPS}")
fi

exec "${cmd[@]}"
