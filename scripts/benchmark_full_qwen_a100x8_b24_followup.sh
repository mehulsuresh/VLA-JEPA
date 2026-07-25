#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.sh"

LOG_ROOT="${B24_BENCHMARK_LOG_ROOT:-/mnt/vla-jepa/logs/full_qwen_b24_followup}"
STEPS="${B24_BENCHMARK_STEPS:-16}"
MAX_SHARDS="${B24_BENCHMARK_MAX_SHARDS:-8}"
BASE_PORT="${B24_BENCHMARK_BASE_PORT:-29860}"
CASE_TIMEOUT_SECONDS="${B24_BENCHMARK_CASE_TIMEOUT_SECONDS:-1500}"
PYTHON_BIN="${B24_BENCHMARK_PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
mkdir -p "${LOG_ROOT}"

SUMMARY="${LOG_ROOT}/summary.tsv"
printf "timestamp\tcase\trun_id\texit_code\tstage\tbatch\tglobal_batch\tworkers\tprefetch\treader_cache\tsidecar_cache\tgrad_ckpt\tcompile_flags\tsave_interval\teval_interval\tavg_wall_sec\tsamples_per_sec\tlast_wall_sec\tmax_reserved_gb\tlog_file\n" > "${SUMMARY}"

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
  local stage="$2"
  local batch="$3"
  local workers="$4"
  local prefetch="$5"
  local reader_cache="$6"
  local sidecar_cache="$7"
  local grad_ckpt="$8"
  local save_interval="$9"
  local eval_interval="${10}"
  local port="${11}"
  local compile_label="${12}"
  shift 12
  local extra_args=("$@")

  local timestamp run_id log_file exit_code metrics global_batch
  timestamp="$(date +%Y%m%d_%H%M%S)"
  run_id="b24_${case_name}_${timestamp}"
  log_file="${LOG_ROOT}/${run_id}.log"
  global_batch=$((batch * 8))

  echo "=== ${case_name}: stage=${stage}, batch=${batch}, global_batch=${global_batch}, workers=${workers}, prefetch=${prefetch}, grad_ckpt=${grad_ckpt}, compile=${compile_label}, save=${save_interval}, eval=${eval_interval} ==="
  (
    cd "${REPO_ROOT}" || exit 1
    STARVLA_DEEPSPEED_STAGE="${stage}" \
    RUN_ID="${run_id}" \
    MAIN_PROCESS_PORT="${port}" \
    MAX_TRAIN_STEPS="${STEPS}" \
    NUM_WARMUP_STEPS="3" \
    SAVE_INTERVAL="${save_interval}" \
    LOGGING_FREQUENCY="1" \
    MAX_SHARDS="${MAX_SHARDS}" \
    MAX_SHARDS_PER_DATASET="0" \
    MAX_WINDOWS="0" \
    MAX_WINDOWS_PER_DATASET="0" \
    PER_DEVICE_BATCH_SIZE="${batch}" \
    timeout --signal=INT "${CASE_TIMEOUT_SECONDS}" "${LAUNCHER}" \
      --trainer.eval_interval "${eval_interval}" \
      --trainer.save_final_model false \
      --trainer.progress_eta_warmup_steps 3 \
      --trainer.enable_gradient_checkpointing "${grad_ckpt}" \
      --datasets.vla_data.per_device_batch_size "${batch}" \
      --datasets.vla_data.num_workers "${workers}" \
      --datasets.vla_data.prefetch_factor "${prefetch}" \
      --datasets.vla_data.reader_cache_size "${reader_cache}" \
      --datasets.vla_data.sidecar_cache_size "${sidecar_cache}" \
      "${extra_args[@]}"
  ) 2>&1 | timeout --signal=INT "${CASE_TIMEOUT_SECONDS}" tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  metrics="$(summarize_case "${log_file}" "${batch}")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${timestamp}" \
    "${case_name}" \
    "${run_id}" \
    "${exit_code}" \
    "${stage}" \
    "${batch}" \
    "${global_batch}" \
    "${workers}" \
    "${prefetch}" \
    "${reader_cache}" \
    "${sidecar_cache}" \
    "${grad_ckpt}" \
    "${compile_label}" \
    "${save_interval}" \
    "${eval_interval}" \
    "${metrics}" \
    "${log_file}" >> "${SUMMARY}"
  echo "=== ${case_name} exit_code=${exit_code}; metrics=${metrics}; log=${log_file} ==="
  sleep 8
}

case_index=0
next_port() {
  echo "$((BASE_PORT + case_index))"
  case_index=$((case_index + 1))
}

run_case "z3_b24_w1_gc_on_eager_noio" 3 24 1 2 128 32 true 1000000 1000000 "$(next_port)" "eager"
run_case "z3_b24_w1_gc_off_eager_noio" 3 24 1 2 128 32 false 1000000 1000000 "$(next_port)" "eager"
run_case "z2_b24_w1_gc_on_eager_noio" 2 24 1 2 128 32 true 1000000 1000000 "$(next_port)" "eager"
run_case "z3_b24_w2p2_gc_on_cache128_32_noio" 3 24 2 2 128 32 true 1000000 1000000 "$(next_port)" "eager"
run_case "z3_b24_w4p4_gc_on_cache256_64_noio" 3 24 4 4 256 64 true 1000000 1000000 "$(next_port)" "eager"
run_case "z3_b24_w1_gc_on_compile_action_vjpred" 3 24 1 2 128 32 true 1000000 1000000 "$(next_port)" "action+vjpred" \
  --trainer.allow_compile_with_deepspeed true \
  --trainer.compile_action_model true \
  --trainer.compile_vj_predictor true \
  --trainer.compile_mode reduce-overhead \
  --trainer.compile_dynamic false \
  --trainer.strict_torch_compile true
run_case "z3_b24_w1_gc_on_save_eval_step10" 3 24 1 2 128 32 true 10 10 "$(next_port)" "eager"

echo "Batch-24 follow-up benchmark summary: ${SUMMARY}"
