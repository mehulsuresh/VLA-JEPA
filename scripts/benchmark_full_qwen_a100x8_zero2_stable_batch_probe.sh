#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.sh"
STABLE_ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2_stable.yaml"

LOG_ROOT="${ZERO2_STABLE_BATCH_PROBE_LOG_ROOT:-/mnt/vla-jepa/logs/full_qwen_zero2_stable_batch_probe_$(date +%Y%m%d_%H%M%S)}"
STEPS="${ZERO2_STABLE_BATCH_PROBE_STEPS:-24}"
MAX_SHARDS="${ZERO2_STABLE_BATCH_PROBE_MAX_SHARDS:-8}"
BASE_PORT="${ZERO2_STABLE_BATCH_PROBE_BASE_PORT:-29980}"
CASE_TIMEOUT_SECONDS="${ZERO2_STABLE_BATCH_PROBE_CASE_TIMEOUT_SECONDS:-900}"
PYTHON_BIN="${ZERO2_STABLE_BATCH_PROBE_PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
mkdir -p "${LOG_ROOT}"

SUMMARY="${LOG_ROOT}/summary.tsv"
printf "timestamp\tcase\trun_id\texit_code\tstage\tbatch\tglobal_batch\tworkers\tprefetch\treader_cache\tsidecar_cache\tgrad_ckpt\tdeepspeed_config\tsave_interval\teval_interval\tavg_wall_sec\tsamples_per_sec\tlast_wall_sec\tmax_reserved_gb\tlog_file\n" > "${SUMMARY}"

summarize_case() {
  local log_file="$1"
  local batch="$2"
  "${PYTHON_BIN}" - "$log_file" "$batch" <<'PY'
import math
import re
import sys

log_file = sys.argv[1]
batch = int(sys.argv[2])
global_batch = batch * 8
text = open(log_file, "r", encoding="utf-8", errors="replace").read()

avg_values = [float(x) for x in re.findall(r"avg_wall=([0-9]+(?:\.[0-9]+)?)", text)]
wall_values = [float(x) for x in re.findall(r"wall_time=([0-9]+(?:\.[0-9]+)?)", text)]
mem_values = [float(x) for x in re.findall(r"max_reserved=([0-9]+(?:\.[0-9]+)?) GiB", text)]

avg_wall = avg_values[-1] if avg_values else math.nan
last_wall = wall_values[-1] if wall_values else math.nan
max_reserved = mem_values[-1] if mem_values else math.nan
samples_per_sec = global_batch / avg_wall if avg_wall and not math.isnan(avg_wall) else math.nan

def fmt(value):
    return "nan" if math.isnan(value) else f"{value:.4f}"

print("\t".join([fmt(avg_wall), fmt(samples_per_sec), fmt(last_wall), fmt(max_reserved)]))
PY
}

run_case() {
  local case_name="$1"
  local batch="$2"
  local workers="$3"
  local prefetch="$4"
  local reader_cache="$5"
  local sidecar_cache="$6"
  local port="$7"
  local timestamp run_id log_file exit_code metrics global_batch
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_id="z2stable_${case_name}_${timestamp}"
  log_file="${LOG_ROOT}/${run_id}.log"
  global_batch=$((batch * 8))

  echo "=== ${case_name}: stable zero2, batch=${batch}, global_batch=${global_batch}, workers=${workers}, prefetch=${prefetch} ==="
  (
    cd "${REPO_ROOT}" || exit 1
    ACCELERATE_CONFIG="${STABLE_ACCELERATE_CONFIG}" \
    STARVLA_DEEPSPEED_STAGE="2" \
    RUN_ID="${run_id}" \
    MAIN_PROCESS_PORT="${port}" \
    MAX_TRAIN_STEPS="${STEPS}" \
    NUM_WARMUP_STEPS="3" \
    SAVE_INTERVAL="1000000" \
    EVAL_INTERVAL="1000000" \
    LOGGING_FREQUENCY="1" \
    MAX_SHARDS="${MAX_SHARDS}" \
    MAX_SHARDS_PER_DATASET="0" \
    MAX_WINDOWS="0" \
    MAX_WINDOWS_PER_DATASET="0" \
    PER_DEVICE_BATCH_SIZE="${batch}" \
    NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
    NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL}" \
    TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}" \
    TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}" \
    TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}" \
    timeout --signal=INT "${CASE_TIMEOUT_SECONDS}" "${LAUNCHER}" \
      --trainer.eval_interval 1000000 \
      --trainer.save_final_model false \
      --trainer.progress_eta_warmup_steps 3 \
      --trainer.enable_gradient_checkpointing true \
      --datasets.vla_data.per_device_batch_size "${batch}" \
      --datasets.vla_data.num_workers "${workers}" \
      --datasets.vla_data.prefetch_factor "${prefetch}" \
      --datasets.vla_data.reader_cache_size "${reader_cache}" \
      --datasets.vla_data.sidecar_cache_size "${sidecar_cache}" \
      --datasets.vla_data.dataloader_timeout_seconds 420
  ) 2>&1 | tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  metrics="$(summarize_case "${log_file}" "${batch}")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${timestamp}" \
    "${case_name}" \
    "${run_id}" \
    "${exit_code}" \
    "2" \
    "${batch}" \
    "${global_batch}" \
    "${workers}" \
    "${prefetch}" \
    "${reader_cache}" \
    "${sidecar_cache}" \
    "true" \
    "${STABLE_ACCELERATE_CONFIG}" \
    "1000000" \
    "1000000" \
    "${metrics}" \
    "${log_file}" >> "${SUMMARY}"
  echo "=== ${case_name} exit_code=${exit_code}; metrics=${metrics}; log=${log_file} ==="
  sleep 8
}

run_case "b26_w2p2_cache128_32" 26 2 2 128 32 "$((BASE_PORT + 0))"
run_case "b28_w2p2_cache128_32" 28 2 2 128 32 "$((BASE_PORT + 1))"

echo "Stable ZeRO-2 batch probe summary: ${SUMMARY}"
