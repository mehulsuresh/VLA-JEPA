#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml}"
export STARVLA_DEEPSPEED_STAGE="${STARVLA_DEEPSPEED_STAGE:-2}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2_stable.yaml}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29561}"
export RUN_ID="${RUN_ID:-robot_ft_canonical_full_a100x8_qwen_full_zero2stable_b26_moge_vits_$(date +%Y%m%d_%H%M%S)}"

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

exec "${SCRIPT_DIR}/vlajepa_robot_ft_canonical_multi_subset_a100x8_moge_vjteacher_pilot.sh" \
  --framework.qwenvl.lora.enabled false \
  --framework.qwenvl.strict_attn_implementation true \
  --framework.qwenvl.strict_tensor_input_fast_path true \
  --framework.qwenvl.strict_full_trainable true \
  --framework.qwenvl.enable_fast_linear_attention false \
  --framework.qwenvl.strict_fast_linear_attention false \
  --framework.depth_teacher_aux.teacher_model Ruicheng/moge-2-vits-normal \
  --framework.action_model.rtc_training.enabled true \
  --framework.action_model.rtc_training.condition_dit_tokens true \
  --framework.action_model.rtc_training.rtc_prob "${RTC_PROB:-1.0}" \
  --framework.action_model.rtc_training.warmup_steps "${RTC_WARMUP_STEPS:-100000}" \
  --framework.action_model.rtc_training.ramp_steps "${RTC_RAMP_STEPS:-500000}" \
  --framework.action_model.rtc_training.distribution "${RTC_DISTRIBUTION:-uniform}" \
  --datasets.vla_data.video_decode_backend pyav \
  --datasets.vla_data.pyav_thread_count "${PYAV_THREAD_COUNT:-1}" \
  --datasets.vla_data.pyav_reader_cache_size "${PYAV_READER_CACHE_SIZE:-8}" \
  --datasets.vla_data.gcs_download_timeout_seconds "${GCS_DOWNLOAD_TIMEOUT_SECONDS:-900}" \
  --datasets.vla_data.gcs_download_retries "${GCS_DOWNLOAD_RETRIES:-3}" \
  --datasets.vla_data.gcs_download_retry_backoff_seconds "${GCS_DOWNLOAD_RETRY_BACKOFF_SECONDS:-5}" \
  --datasets.vla_data.data_file_prefetch_shards "${DATA_FILE_PREFETCH_SHARDS:-1}" \
  --datasets.vla_data.video_cache_max_gb "${VIDEO_CACHE_MAX_GB:-500}" \
  --datasets.vla_data.video_cache_prune_interval_downloads "${VIDEO_CACHE_PRUNE_INTERVAL_DOWNLOADS:-8}" \
  --datasets.vla_data.video_cache_prune_target_fraction "${VIDEO_CACHE_PRUNE_TARGET_FRACTION:-0.9}" \
  --datasets.vla_data.lazy_cache_shards true \
  --datasets.vla_data.index_windows_lazily true \
  --datasets.vla_data.prefetch_metadata_across_ranks true \
  --datasets.vla_data.metadata_index_cache true \
  --datasets.vla_data.metadata_index_cache_dir "${METADATA_INDEX_CACHE_DIR:-/mnt/vla-jepa/datasets/canonical_gcs/.canonical_index_cache}" \
  "${EXCLUDE_DATASET_ARGS[@]}" \
  --datasets.vla_data.shuffle_shards true \
  --datasets.vla_data.shuffle false \
  --datasets.vla_data.max_shards "${MAX_SHARDS:-0}" \
  --datasets.vla_data.max_shards_per_dataset "${MAX_SHARDS_PER_DATASET:-0}" \
  --datasets.vla_data.max_windows "${MAX_WINDOWS:-0}" \
  --datasets.vla_data.max_windows_per_dataset "${MAX_WINDOWS_PER_DATASET:-0}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-26}" \
  --datasets.vla_data.num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --datasets.vla_data.prefetch_factor "${DATALOADER_PREFETCH_FACTOR:-1}" \
  --datasets.vla_data.dataloader_timeout_seconds "${DATALOADER_TIMEOUT_SECONDS:-0}" \
  --datasets.vla_data.reader_cache_size "${READER_CACHE_SIZE:-32}" \
  --datasets.vla_data.slow_sample_log_seconds "${SLOW_SAMPLE_LOG_SECONDS:-30}" \
  --trainer.epochs "${EPOCHS:-1}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-auto}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS:-10000}" \
  --trainer.save_interval "${SAVE_INTERVAL:-10000}" \
  --trainer.eval_interval "${EVAL_INTERVAL:-10000}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-10}" \
  --trainer.compile_qwen_model false \
  --trainer.compile_action_model false \
  --trainer.compile_vj_predictor false \
  --trainer.compile_vj_encoder false \
  --trainer.compile_full_model false \
  --trainer.strict_torch_compile true \
  "$@"
