#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.sh"

LOG_ROOT="${BENCHMARK_LOG_ROOT:-/mnt/vla-jepa/logs/full_qwen_benchmarks}"
STEPS="${BENCHMARK_STEPS:-30}"
MAX_SHARDS="${BENCHMARK_MAX_SHARDS:-8}"
BASE_PORT="${BENCHMARK_BASE_PORT:-29610}"
mkdir -p "${LOG_ROOT}"

SUMMARY="${LOG_ROOT}/summary.tsv"
if [[ ! -s "${SUMMARY}" ]]; then
  printf "timestamp\tcase\trun_id\tstage\tper_device_batch_size\tnum_workers\tprefetch_factor\tgradient_checkpointing\texit_code\tlog_file\n" > "${SUMMARY}"
fi

run_case() {
  local case_name="$1"
  local stage="$2"
  local per_device_batch_size="$3"
  local num_workers="$4"
  local prefetch_factor="$5"
  local gradient_checkpointing="$6"
  local port="$7"

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local run_id="bench_${case_name}_${timestamp}"
  local log_file="${LOG_ROOT}/${run_id}.log"

  echo "=== ${case_name} :: stage=${stage}, bsz=${per_device_batch_size}, workers=${num_workers}, prefetch=${prefetch_factor}, grad_ckpt=${gradient_checkpointing} ==="
  (
    cd "${REPO_ROOT}" || exit 1
    STARVLA_DEEPSPEED_STAGE="${stage}" \
    RUN_ID="${run_id}" \
    MAIN_PROCESS_PORT="${port}" \
    MAX_TRAIN_STEPS="${STEPS}" \
    NUM_WARMUP_STEPS="5" \
    SAVE_INTERVAL="1000000" \
    LOGGING_FREQUENCY="5" \
    MAX_SHARDS="${MAX_SHARDS}" \
    MAX_SHARDS_PER_DATASET="0" \
    MAX_WINDOWS="0" \
    MAX_WINDOWS_PER_DATASET="0" \
    RTC_WARMUP_STEPS="${BENCHMARK_RTC_WARMUP_STEPS:-5}" \
    RTC_RAMP_STEPS="${BENCHMARK_RTC_RAMP_STEPS:-10}" \
    "${LAUNCHER}" \
      --trainer.eval_interval 1000000 \
      --trainer.progress_eta_warmup_steps 5 \
      --trainer.enable_gradient_checkpointing "${gradient_checkpointing}" \
      --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
      --datasets.vla_data.num_workers "${num_workers}" \
      --datasets.vla_data.prefetch_factor "${prefetch_factor}" \
      --datasets.vla_data.sidecar_cache_size "${BENCHMARK_SIDECAR_CACHE_SIZE:-32}" \
      --datasets.vla_data.reader_cache_size "${BENCHMARK_READER_CACHE_SIZE:-128}"
  ) 2>&1 | tee "${log_file}"
  local exit_code="${PIPESTATUS[0]}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${timestamp}" \
    "${case_name}" \
    "${run_id}" \
    "${stage}" \
    "${per_device_batch_size}" \
    "${num_workers}" \
    "${prefetch_factor}" \
    "${gradient_checkpointing}" \
    "${exit_code}" \
    "${log_file}" >> "${SUMMARY}"
  echo "=== ${case_name} exit_code=${exit_code}; log=${log_file} ==="
  sleep 10
}

case_index=0
run_case "zero3_b2_w1_gc_on" "3" "2" "1" "2" "true" "$((BASE_PORT + case_index++))"
run_case "zero3_b4_w1_gc_on" "3" "4" "1" "2" "true" "$((BASE_PORT + case_index++))"
run_case "zero3_b4_w2_gc_on" "3" "4" "2" "2" "true" "$((BASE_PORT + case_index++))"
run_case "zero3_b4_w2_gc_off" "3" "4" "2" "2" "false" "$((BASE_PORT + case_index++))"
run_case "zero2_b4_w2_gc_off" "2" "4" "2" "2" "false" "$((BASE_PORT + case_index++))"

echo "Benchmark summary: ${SUMMARY}"
