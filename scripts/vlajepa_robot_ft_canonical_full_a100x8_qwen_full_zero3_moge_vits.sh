#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml}"
export STARVLA_DEEPSPEED_STAGE="${STARVLA_DEEPSPEED_STAGE:-3}"
if [[ -z "${ACCELERATE_CONFIG:-}" ]]; then
  if [[ "${STARVLA_DEEPSPEED_STAGE}" == "3" || "${STARVLA_DEEPSPEED_STAGE,,}" == "zero3" ]]; then
    export ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero3.yaml"
  else
    export ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2_stable.yaml"
  fi
fi
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29561}"
export RUN_ID="${RUN_ID:-robot_ft_canonical_full_a100x8_qwen_full_zero3_b26_moge_vits_$(date +%Y%m%d_%H%M%S)}"

if [[ "${STARVLA_ALLOW_TORCH_COMPILE:-0}" != "1" ]]; then
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
else
  unset TORCH_COMPILE_DISABLE
  unset TORCHDYNAMO_DISABLE
fi

EXCLUDE_DATASET_ARGS=()
if [[ -n "${EXCLUDE_DATASET_IDS_PATH:-}" ]]; then
  EXCLUDE_DATASET_ARGS+=(--datasets.vla_data.exclude_dataset_ids_path "${EXCLUDE_DATASET_IDS_PATH}")
fi

TRAIN_ARGS=()
add_arg_if_env() {
  local env_name="$1"
  shift
  if [[ -n "${!env_name:-}" ]]; then
    TRAIN_ARGS+=("$@" "${!env_name}")
  fi
}

add_arg_if_env RTC_PROB --framework.action_model.rtc_training.rtc_prob
add_arg_if_env RTC_WARMUP_STEPS --framework.action_model.rtc_training.warmup_steps
add_arg_if_env RTC_RAMP_STEPS --framework.action_model.rtc_training.ramp_steps
add_arg_if_env RTC_DISTRIBUTION --framework.action_model.rtc_training.distribution
add_arg_if_env PYAV_THREAD_COUNT --datasets.vla_data.pyav_thread_count
add_arg_if_env PYAV_READER_CACHE_SIZE --datasets.vla_data.pyav_reader_cache_size
add_arg_if_env GCS_DOWNLOAD_TIMEOUT_SECONDS --datasets.vla_data.gcs_download_timeout_seconds
add_arg_if_env GCS_DOWNLOAD_RETRIES --datasets.vla_data.gcs_download_retries
add_arg_if_env GCS_DOWNLOAD_RETRY_BACKOFF_SECONDS --datasets.vla_data.gcs_download_retry_backoff_seconds
add_arg_if_env DATA_FILE_PREFETCH_SHARDS --datasets.vla_data.data_file_prefetch_shards
add_arg_if_env VIDEO_CACHE_MAX_GB --datasets.vla_data.video_cache_max_gb
add_arg_if_env VIDEO_CACHE_PRUNE_INTERVAL_DOWNLOADS --datasets.vla_data.video_cache_prune_interval_downloads
add_arg_if_env VIDEO_CACHE_PRUNE_TARGET_FRACTION --datasets.vla_data.video_cache_prune_target_fraction
add_arg_if_env METADATA_INDEX_CACHE_DIR --datasets.vla_data.metadata_index_cache_dir
add_arg_if_env MAX_SHARDS --datasets.vla_data.max_shards
add_arg_if_env MAX_SHARDS_PER_DATASET --datasets.vla_data.max_shards_per_dataset
add_arg_if_env MAX_WINDOWS --datasets.vla_data.max_windows
add_arg_if_env MAX_WINDOWS_PER_DATASET --datasets.vla_data.max_windows_per_dataset
add_arg_if_env PER_DEVICE_BATCH_SIZE --datasets.vla_data.per_device_batch_size
add_arg_if_env DATALOADER_NUM_WORKERS --datasets.vla_data.num_workers
add_arg_if_env DATALOADER_PREFETCH_FACTOR --datasets.vla_data.prefetch_factor
add_arg_if_env DATALOADER_TIMEOUT_SECONDS --datasets.vla_data.dataloader_timeout_seconds
add_arg_if_env READER_CACHE_SIZE --datasets.vla_data.reader_cache_size
add_arg_if_env SLOW_SAMPLE_LOG_SECONDS --datasets.vla_data.slow_sample_log_seconds
add_arg_if_env EPOCHS --trainer.epochs
add_arg_if_env MAX_TRAIN_STEPS --trainer.max_train_steps
add_arg_if_env NUM_WARMUP_STEPS --trainer.num_warmup_steps
add_arg_if_env SAVE_INTERVAL --trainer.save_interval
add_arg_if_env EVAL_INTERVAL --trainer.eval_interval
add_arg_if_env LOGGING_FREQUENCY --trainer.logging_frequency

exec "${SCRIPT_DIR}/vlajepa_robot_ft_canonical_multi_subset_a100x8_moge_vjteacher_pilot.sh" \
  "${EXCLUDE_DATASET_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "$@"
