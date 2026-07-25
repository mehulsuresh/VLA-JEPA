#!/usr/bin/env bash
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-/home/mehul/work/vjepa/VLA-JEPA}"

export BENCH_CONFIG="${BENCH_CONFIG:-${VLA_ROOT}/eval_harness/vla_jepa/configs/libero_plus_all.yaml}"
export SERVER_CONFIG="${SERVER_CONFIG:-${VLA_ROOT}/eval_harness/vla_jepa/configs/model_server_libero_plus.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/mehul/work/vjepa/eval_videos/harness_libero_plus_$(date +%Y%m%d_%H%M%S)}"
export SIM_ENV="${SIM_ENV:-libero-plus}"
export CHUNK_SIZE="${CHUNK_SIZE:-7}"
export NUM_DDIM_STEPS="${NUM_DDIM_STEPS:-10}"

exec "${VLA_ROOT}/eval_harness/vla_jepa/scripts/run_sharded_eval.sh"
