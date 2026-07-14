#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Bounded clean-base pilot for the corrected Magna v3 pipeline. This does not
# load or resume any RealMan/VLA checkpoint. Qwen, V-JEPA, and MoGe retain the
# normal upstream pretrained initializations specified by the base config.
# Pin the config and budget so stale launch-environment variables cannot turn
# this pilot into a continuation of the failed production run.
export CONFIG_YAML="${REPO_ROOT}/scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml"
PILOT_RUN_ID_PREFIX="magna_clean_v3_rtc0_pilot_a100x8_"
PILOT_DATA_ROOT="/mnt/vla-jepa/datasets/magna_training_data_with_interventions"

invalid_runtime_value() {
  local option="$1"
  local value="$2"
  local expected="$3"
  echo "Invalid clean-pilot runtime override ${option}=${value}: expected ${expected}" >&2
  exit 2
}

validate_runtime_value() {
  local option="$1"
  local value="$2"
  local value_kind="$3"
  case "${value_kind}" in
    positive_integer)
      [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
        || invalid_runtime_value "${option}" "${value}" "a positive integer"
      ;;
    nonnegative_integer)
      [[ "${value}" =~ ^[0-9]+$ ]] \
        || invalid_runtime_value "${option}" "${value}" "a non-negative integer"
      ;;
    boolean)
      [[ "${value}" == "true" || "${value}" == "false" ]] \
        || invalid_runtime_value "${option}" "${value}" "true or false"
      ;;
    *)
      echo "Unknown clean-pilot validation kind: ${value_kind}" >&2
      exit 2
      ;;
  esac
}

validate_optional_runtime_env() {
  local env_name="$1"
  local option="$2"
  local value_kind="$3"
  if [[ -n "${!env_name:-}" ]]; then
    validate_runtime_value "${option}" "${!env_name}" "${value_kind}"
  fi
}

# `ACCELERATE_BIN` is executable control, not a performance tuning knob. An
# inherited `/bin/true` previously made this launcher exit successfully without
# training or evaluating anything. Production resolves the installed
# `accelerate` executable itself and rejects every inherited override. Tests use
# a paired, deliberately unforwarded dry-run mode that can only print the final
# command while also skipping the dataset-dependent holdout preflight.
PILOT_TEST_DRY_RUN="${STARVLA_CLEAN_PILOT_TEST_DRY_RUN:-0}"
HOLDOUT_PREFLIGHT_DRY_RUN="${STARVLA_HOLDOUT_PREFLIGHT_DRY_RUN:-0}"
if [[ "${PILOT_TEST_DRY_RUN}" != "0" && "${PILOT_TEST_DRY_RUN}" != "1" ]]; then
  invalid_runtime_value \
    "STARVLA_CLEAN_PILOT_TEST_DRY_RUN" \
    "${PILOT_TEST_DRY_RUN}" \
    "0 or 1"
fi
if [[ "${HOLDOUT_PREFLIGHT_DRY_RUN}" != "0" && "${HOLDOUT_PREFLIGHT_DRY_RUN}" != "1" ]]; then
  invalid_runtime_value \
    "STARVLA_HOLDOUT_PREFLIGHT_DRY_RUN" \
    "${HOLDOUT_PREFLIGHT_DRY_RUN}" \
    "0 or 1"
fi
if [[ -n "${ACCELERATE_BIN:-}" ]]; then
  echo "Clean pilot refuses inherited ACCELERATE_BIN=${ACCELERATE_BIN}" >&2
  exit 2
fi
if [[ "${PILOT_TEST_DRY_RUN}" == "1" ]]; then
  if [[ "${HOLDOUT_PREFLIGHT_DRY_RUN}" != "1" ]]; then
    echo "STARVLA_CLEAN_PILOT_TEST_DRY_RUN=1 requires STARVLA_HOLDOUT_PREFLIGHT_DRY_RUN=1" >&2
    exit 2
  fi
  export ACCELERATE_BIN=/bin/echo
else
  if [[ "${HOLDOUT_PREFLIGHT_DRY_RUN}" == "1" ]]; then
    echo "STARVLA_HOLDOUT_PREFLIGHT_DRY_RUN=1 is allowed only in clean-pilot test dry-run mode" >&2
    exit 2
  fi
  PILOT_ACCELERATE_BIN="$(command -v accelerate 2>/dev/null || true)"
  if [[ -z "${PILOT_ACCELERATE_BIN}" || ! -x "${PILOT_ACCELERATE_BIN}" ]]; then
    echo "Clean pilot requires an executable accelerate on PATH" >&2
    exit 2
  fi
  export ACCELERATE_BIN="${PILOT_ACCELERATE_BIN}"
fi

if [[ -n "${RUN_ID:-}" && "${RUN_ID}" != "${PILOT_RUN_ID_PREFIX}"* ]]; then
  echo "Clean pilot RUN_ID must start with ${PILOT_RUN_ID_PREFIX}; got ${RUN_ID}" >&2
  exit 2
fi
if [[ -n "${DATA_ROOT_DIR:-}" && "${DATA_ROOT_DIR}" != "${PILOT_DATA_ROOT}" ]]; then
  echo "Clean pilot DATA_ROOT_DIR must be ${PILOT_DATA_ROOT}; got ${DATA_ROOT_DIR}" >&2
  exit 2
fi
if [[ -n "${NUM_PROCESSES:-}" && "${NUM_PROCESSES}" != "8" ]]; then
  echo "Clean pilot NUM_PROCESSES must be 8; got ${NUM_PROCESSES}" >&2
  exit 2
fi
if [[ -n "${NUM_MACHINES:-}" && "${NUM_MACHINES}" != "1" ]]; then
  echo "Clean pilot NUM_MACHINES must be 1; got ${NUM_MACHINES}" >&2
  exit 2
fi
if [[ -n "${STARVLA_USE_DEEPSPEED:-}" && "${STARVLA_USE_DEEPSPEED}" != "0" ]]; then
  echo "Clean pilot STARVLA_USE_DEEPSPEED must be 0; got ${STARVLA_USE_DEEPSPEED}" >&2
  exit 2
fi
if [[ -n "${STARVLA_ALLOW_TORCH_COMPILE:-}" && "${STARVLA_ALLOW_TORCH_COMPILE}" != "0" ]]; then
  echo "Clean pilot STARVLA_ALLOW_TORCH_COMPILE must be 0; got ${STARVLA_ALLOW_TORCH_COMPILE}" >&2
  exit 2
fi
if [[ -n "${STARVLA_DISABLE_TORCH_COMPILE:-}" && "${STARVLA_DISABLE_TORCH_COMPILE}" != "1" ]]; then
  echo "Clean pilot STARVLA_DISABLE_TORCH_COMPILE must be 1; got ${STARVLA_DISABLE_TORCH_COMPILE}" >&2
  exit 2
fi
if [[ -n "${PER_DEVICE_BATCH_SIZE:-}" && "${PER_DEVICE_BATCH_SIZE}" != "12" ]]; then
  echo "Clean pilot per-device batch size is fixed by config at 12; got ${PER_DEVICE_BATCH_SIZE}" >&2
  exit 2
fi
if [[ -n "${VIDEO_BACKEND:-}" && "${VIDEO_BACKEND}" != "pyav" ]]; then
  echo "Clean pilot VIDEO_BACKEND must be pyav; got ${VIDEO_BACKEND}" >&2
  exit 2
fi
validate_optional_runtime_env DATALOADER_NUM_WORKERS \
  --datasets.vla_data.num_workers nonnegative_integer
validate_optional_runtime_env DATALOADER_PREFETCH_FACTOR \
  --datasets.vla_data.prefetch_factor positive_integer
validate_optional_runtime_env DATALOADER_TIMEOUT_SECONDS \
  --datasets.vla_data.dataloader_timeout_seconds nonnegative_integer
validate_optional_runtime_env DATALOADER_PERSISTENT_WORKERS \
  --datasets.vla_data.persistent_workers boolean
validate_optional_runtime_env VIDEO_BACKEND_NUM_THREADS \
  --datasets.vla_data.video_backend_num_threads positive_integer
validate_optional_runtime_env LOGGING_FREQUENCY \
  --trainer.logging_frequency positive_integer
validate_optional_runtime_env FIND_UNUSED_PARAMETERS \
  --trainer.find_unused_parameters boolean
validate_optional_runtime_env DDP_GRADIENT_AS_BUCKET_VIEW \
  --trainer.ddp_gradient_as_bucket_view boolean
validate_optional_runtime_env DDP_STATIC_GRAPH \
  --trainer.ddp_static_graph boolean
validate_optional_runtime_env DDP_BUCKET_CAP_MB \
  --trainer.ddp_bucket_cap_mb positive_integer
export RUN_ID="${RUN_ID:-${PILOT_RUN_ID_PREFIX}$(date +%Y%m%d_%H%M%S)}"
export DATA_ROOT_DIR="${PILOT_DATA_ROOT}"
export NUM_PROCESSES=8
export NUM_MACHINES=1
export STARVLA_USE_DEEPSPEED=0
# The downstream generic launcher branches on ALLOW and otherwise preserves
# inherited TORCH_* values. Pin every layer so a stale cloud environment cannot
# change this eager-only pilot's execution mode.
export STARVLA_ALLOW_TORCH_COMPILE=0
export STARVLA_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export VIDEO_BACKEND=pyav
# Do not emit a batch-size dotlist override; the pinned config is authoritative.
unset PER_DEVICE_BATCH_SIZE EPOCHS
export MAX_TRAIN_STEPS=2500
export NUM_WARMUP_STEPS=300
export SAVE_INTERVAL=2500
# The immutable manifest now supplies a separate held-out loader. Evaluate the
# clean initialization once (a fail-fast compatibility check and direct
# learning baseline), then evaluate the same terminal boundary used by this
# bounded pilot.
export EVAL_INTERVAL=2500

# The cloud service translates RESUME_CHECKPOINT into trainer dotlist options.
# Accept only execution-performance knobs whose values cannot change the model,
# data semantics, supervision, optimizer, schedule, or checkpoint lineage.
# Parse option/value pairs together so values are never mistaken for options.
pilot_args=("$@")
arg_index=0
while (( arg_index < ${#pilot_args[@]} )); do
  arg="${pilot_args[arg_index]}"
  if [[ "${arg}" != --* ]]; then
    echo "Refusing orphaned clean-pilot argument: ${arg}" >&2
    exit 2
  fi
  option="${arg%%=*}"
  case "${option}" in
    --datasets.vla_data.num_workers | \
    --datasets.vla_data.prefetch_factor | \
    --datasets.vla_data.dataloader_timeout_seconds | \
    --datasets.vla_data.persistent_workers | \
    --datasets.vla_data.video_backend_num_threads | \
    --datasets.vla_data.worker_torch_threads | \
    --datasets.vla_data.worker_cv2_threads | \
    --datasets.vla_data.lerobot_v3_parquet_cache_size | \
    --datasets.vla_data.pin_memory | \
    --datasets.vla_data.runtime_timing_logging | \
    --trainer.logging_frequency | \
    --trainer.progress_eta_window | \
    --trainer.progress_eta_warmup_steps | \
    --trainer.find_unused_parameters | \
    --trainer.ddp_gradient_as_bucket_view | \
    --trainer.ddp_static_graph | \
    --trainer.ddp_bucket_cap_mb | \
    --trainer.profile_cuda_memory | \
    --trainer.profile_cuda_memory_log_step | \
    --trainer.drop_checkpoint_page_cache | \
    --trainer.trim_process_memory_after_checkpoint)
      ;;
    *)
      echo "Refusing protected clean-pilot override: ${option}" >&2
      exit 2
      ;;
  esac

  if [[ "${arg}" == *=* ]]; then
    value="${arg#*=}"
    next_arg_index=$((arg_index + 1))
  else
    if (( arg_index + 1 >= ${#pilot_args[@]} )) || [[ "${pilot_args[arg_index + 1]}" == --* ]]; then
      echo "Clean-pilot runtime override requires a value: ${option}" >&2
      exit 2
    fi
    value="${pilot_args[arg_index + 1]}"
    next_arg_index=$((arg_index + 2))
  fi

  case "${option}" in
    --datasets.vla_data.num_workers | \
    --datasets.vla_data.dataloader_timeout_seconds | \
    --datasets.vla_data.lerobot_v3_parquet_cache_size | \
    --trainer.progress_eta_warmup_steps | \
    --trainer.profile_cuda_memory_log_step)
      validate_runtime_value "${option}" "${value}" nonnegative_integer
      ;;
    --datasets.vla_data.prefetch_factor | \
    --datasets.vla_data.video_backend_num_threads | \
    --datasets.vla_data.worker_torch_threads | \
    --datasets.vla_data.worker_cv2_threads | \
    --trainer.logging_frequency | \
    --trainer.progress_eta_window | \
    --trainer.ddp_bucket_cap_mb)
      validate_runtime_value "${option}" "${value}" positive_integer
      ;;
    --datasets.vla_data.persistent_workers | \
    --datasets.vla_data.pin_memory | \
    --datasets.vla_data.runtime_timing_logging | \
    --trainer.find_unused_parameters | \
    --trainer.ddp_gradient_as_bucket_view | \
    --trainer.ddp_static_graph | \
    --trainer.profile_cuda_memory | \
    --trainer.drop_checkpoint_page_cache | \
    --trainer.trim_process_memory_after_checkpoint)
      validate_runtime_value "${option}" "${value}" boolean
      ;;
  esac
  arg_index="${next_arg_index}"
done

# Materialize or fail-closed validate one immutable holdout whose episode count
# equals the effective global batch. The generator also binds normalization to
# the train complement, so neither the held-out rows nor full-dataset stats can
# leak into training. Its final prefixed JSON line is the machine-readable path
# passed to every Accelerate rank below.
if [[ "${HOLDOUT_PREFLIGHT_DRY_RUN}" == "1" ]]; then
  # Explicit command-construction tests have no mounted dataset and never
  # launch Python. They may reference only the already generated, checked-in
  # artifact; real training always takes the validating branch below.
  EPISODE_SPLIT_MANIFEST="${REPO_ROOT}/deployment/realman/eval_manifests/magna_internal_holdout_global_batch96_v1.json"
else
  HOLDOUT_GENERATOR_PYTHON="${HOLDOUT_GENERATOR_PYTHON:-$(command -v python)}"
  HOLDOUT_BUILD_OUTPUT="$(
    "${HOLDOUT_GENERATOR_PYTHON}" \
      "${REPO_ROOT}/deployment/realman/build_magna_internal_holdout.py" \
      --dataset-root "${PILOT_DATA_ROOT}" \
      --config "${CONFIG_YAML}" \
      --launcher "${BASH_SOURCE[0]}" \
      --world-size "${NUM_PROCESSES}"
  )"
  printf '%s\n' "${HOLDOUT_BUILD_OUTPUT}"
  HOLDOUT_BUILD_JSON="$(
    printf '%s\n' "${HOLDOUT_BUILD_OUTPUT}" \
      | sed -n 's/^MAGNA_HOLDOUT_BUILD_RESULT=//p' \
      | tail -n 1
  )"
  if [[ -z "${HOLDOUT_BUILD_JSON}" ]]; then
    echo "Holdout generator did not emit MAGNA_HOLDOUT_BUILD_RESULT" >&2
    exit 2
  fi
  EPISODE_SPLIT_MANIFEST="$(
    "${HOLDOUT_GENERATOR_PYTHON}" -c \
      'import json, sys; print(json.loads(sys.argv[1])["manifest_path"])' \
      "${HOLDOUT_BUILD_JSON}"
  )"
fi
if [[ ! -f "${EPISODE_SPLIT_MANIFEST}" ]]; then
  echo "Holdout generator returned a missing manifest: ${EPISODE_SPLIT_MANIFEST}" >&2
  exit 2
fi
export EPISODE_SPLIT_MANIFEST

exec "${SCRIPT_DIR}/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh" \
  "$@" \
  --framework.action_model.rtc_training.enabled false \
  --framework.action_model.rtc_training.max_delay 0 \
  --framework.action_model.rtc_training.rtc_prob 0.0 \
  --framework.action_model.rtc_training.warmup_steps 0 \
  --framework.action_model.rtc_training.ramp_steps 0 \
  --framework.action_model.rtc_training.condition_dit_tokens false \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.dataset_py lerobot_datasets \
  --datasets.vla_data.data_root_dir "${PILOT_DATA_ROOT}" \
  --datasets.vla_data.data_mix magna_source_no_base_no_lift_interventions_v3 \
  --datasets.vla_data.lerobot_version v3.0 \
  --datasets.vla_data.append_task_id_to_prompt false \
  --datasets.vla_data.append_subtask_to_prompt false \
  --datasets.vla_data.task_id_prompt_append_probability 0.0 \
  --datasets.vla_data.subtask_prompt_append_probability 0.0 \
  --datasets.vla_data.qwen_observation_frame_index current \
  --datasets.vla_data.episode_split_manifest "${EPISODE_SPLIT_MANIFEST}" \
  --datasets.vla_data.load_all_data_for_training false \
  --datasets.vla_data.lerobot_statistics_source split_train \
  --datasets.vla_data.require_statistics_frame_count true \
  --trainer.pretrained_checkpoint null \
  --resume_from_checkpoint null \
  --trainer.resume_from_checkpoint null \
  --trainer.is_resume false \
  --trainer.resume_epoch null \
  --trainer.resume_step null \
  --trainer.max_train_steps 2500 \
  --trainer.save_interval 2500 \
  --trainer.checkpoint_max_to_keep 1 \
  --trainer.eval_interval 2500 \
  --trainer.eval_before_train true \
  --trainer.allow_training_stream_eval false \
  --trainer.save_final_model false \
  --trainer.use_rabc false
