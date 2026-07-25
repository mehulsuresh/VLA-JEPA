#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_canonical_multi_subset_a100x8_moge_vjteacher_pilot.yaml}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29551}"
export RUN_ID="${RUN_ID:-robot_ft_canonical_multi_subset_a100x8_lora_everything_strict_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/vlajepa_robot_ft_canonical_multi_subset_a100x8_moge_vjteacher_pilot.sh" \
  --framework.qwenvl.strict_attn_implementation true \
  --framework.qwenvl.strict_tensor_input_fast_path true \
  --framework.qwenvl.enable_fast_linear_attention false \
  --framework.qwenvl.strict_fast_linear_attention false \
  --framework.qwenvl.lora.strict_trainable true \
  --framework.action_model.rtc_training.enabled true \
  --framework.action_model.rtc_training.condition_dit_tokens true \
  --datasets.vla_data.video_decode_backend decord \
  --datasets.vla_data.lazy_cache_shards true \
  --datasets.vla_data.max_shards "${MAX_SHARDS:-40}" \
  --datasets.vla_data.max_shards_per_dataset "${MAX_SHARDS_PER_DATASET:-2}" \
  --datasets.vla_data.max_windows_per_dataset "${MAX_WINDOWS_PER_DATASET:-192}" \
  --trainer.epochs "${EPOCHS:-2}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-240}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS:-8}" \
  --trainer.save_interval "${SAVE_INTERVAL:-120}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-5}" \
  --trainer.strict_torch_compile true \
  "$@"
