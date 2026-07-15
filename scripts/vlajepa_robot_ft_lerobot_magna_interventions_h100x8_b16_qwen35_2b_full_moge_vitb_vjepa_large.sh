#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

H100_CONFIG="${REPO_ROOT}/scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_h100x8_b16_qwen35_2b_full_moge_vitb_vjepa_large.yaml"
H100_DATA_ROOT="/mnt/vla-jepa/datasets/magna_training_data_with_interventions"
H100_MANIFEST="${REPO_ROOT}/deployment/realman/eval_manifests/magna_internal_holdout_global_batch128_v1.json"
H100_RUN_ROOT="/mnt/vla-jepa/checkpoints"
H100_RUN_ID_PREFIX="robot_ft_lerobot_magna_interventions_h100x8_b16_"
H100_LIFECYCLE_TEST="${STARVLA_H100_LIFECYCLE_TEST:-0}"

reject_conflicting_env() {
  local name="$1"
  local expected="$2"
  if [[ -n "${!name:-}" && "${!name}" != "${expected}" ]]; then
    echo "H100 Magna launcher requires ${name}=${expected}; got ${!name}" >&2
    exit 2
  fi
}

reject_conflicting_env CONFIG_YAML "${H100_CONFIG}"
reject_conflicting_env DATA_ROOT_DIR "${H100_DATA_ROOT}"
reject_conflicting_env LIBERO_DATA_ROOT "${H100_DATA_ROOT}"
reject_conflicting_env REALMAN_DATA_ROOT "${H100_DATA_ROOT}"
reject_conflicting_env NUM_PROCESSES 8
reject_conflicting_env NUM_MACHINES 1
reject_conflicting_env CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7
reject_conflicting_env PER_DEVICE_BATCH_SIZE 16
reject_conflicting_env STARVLA_USE_DEEPSPEED 0
reject_conflicting_env STARVLA_ALLOW_TORCH_COMPILE 0
reject_conflicting_env STARVLA_DISABLE_TORCH_COMPILE 1
reject_conflicting_env TORCH_COMPILE_DISABLE 1
reject_conflicting_env TORCHDYNAMO_DISABLE 1
reject_conflicting_env VIDEO_BACKEND pyav
reject_conflicting_env EPOCHS 3
reject_conflicting_env NUM_WARMUP_STEPS 2250
reject_conflicting_env FIND_UNUSED_PARAMETERS false
reject_conflicting_env DDP_GRADIENT_AS_BUCKET_VIEW true
reject_conflicting_env DDP_STATIC_GRAPH false
reject_conflicting_env DDP_BUCKET_CAP_MB 100
reject_conflicting_env DATALOADER_NUM_WORKERS 4
reject_conflicting_env DATALOADER_PREFETCH_FACTOR 2
reject_conflicting_env DATALOADER_TIMEOUT_SECONDS 0
reject_conflicting_env DATALOADER_PERSISTENT_WORKERS true
reject_conflicting_env VIDEO_BACKEND_NUM_THREADS 1
if [[ -n "${ACCELERATE_CONFIG:-}" || -n "${STARVLA_DEEPSPEED_STAGE:-}" ]]; then
  echo "H100 Magna launcher refuses inherited DeepSpeed configuration" >&2
  exit 2
fi
if [[ -n "${ACCELERATE_BIN:-}" ]]; then
  echo "H100 Magna launcher refuses inherited ACCELERATE_BIN=${ACCELERATE_BIN}" >&2
  exit 2
fi
if [[ -n "${RUN_ID:-}" \
  && ! "${RUN_ID}" =~ ^${H100_RUN_ID_PREFIX}[A-Za-z0-9_.-]+$ ]]; then
  echo "H100 Magna RUN_ID must use prefix ${H100_RUN_ID_PREFIX} and a safe non-empty suffix; got ${RUN_ID}" >&2
  exit 2
fi
export RUN_ID="${RUN_ID:-${H100_RUN_ID_PREFIX}$(date +%Y%m%d_%H%M%S)}"
H100_RUN_DIR="${H100_RUN_ROOT}/${RUN_ID}"
if [[ "${H100_LIFECYCLE_TEST}" != "0" && "${H100_LIFECYCLE_TEST}" != "1" ]]; then
  echo "STARVLA_H100_LIFECYCLE_TEST must be 0 or 1; got ${H100_LIFECYCLE_TEST}" >&2
  exit 2
fi
if [[ "${H100_LIFECYCLE_TEST}" == "1" ]]; then
  reject_conflicting_env MAX_TRAIN_STEPS 15
  reject_conflicting_env SAVE_INTERVAL 5
  reject_conflicting_env EVAL_INTERVAL 5
  reject_conflicting_env LOGGING_FREQUENCY 1
  H100_MAX_TRAIN_STEPS=15
  H100_SAVE_INTERVAL=5
  H100_EVAL_INTERVAL=5
  H100_LOGGING_FREQUENCY=1
  H100_LIFECYCLE_ARGS=(--trainer.max_train_steps "${H100_MAX_TRAIN_STEPS}")
else
  reject_conflicting_env SAVE_INTERVAL 1875
  reject_conflicting_env EVAL_INTERVAL 1875
  reject_conflicting_env LOGGING_FREQUENCY 10
  if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
    echo "H100 production launcher refuses inherited MAX_TRAIN_STEPS; production uses config max_train_steps=auto" >&2
    exit 2
  fi
  H100_MAX_TRAIN_STEPS=""
  H100_SAVE_INTERVAL=1875
  H100_EVAL_INTERVAL=1875
  H100_LOGGING_FREQUENCY=10
  H100_LIFECYCLE_ARGS=()
fi

H100_ACCELERATE_BIN="$(command -v accelerate 2>/dev/null || true)"
H100_PYTHON_BIN="$(command -v python 2>/dev/null || true)"
if [[ -z "${H100_ACCELERATE_BIN}" || ! -x "${H100_ACCELERATE_BIN}" ]]; then
  echo "H100 Magna launcher requires an executable accelerate on PATH" >&2
  exit 2
fi
if [[ -z "${H100_PYTHON_BIN}" || ! -x "${H100_PYTHON_BIN}" ]]; then
  echo "H100 Magna launcher requires an executable python on PATH" >&2
  exit 2
fi

invalid_cli_override() {
  local option="$1"
  local detail="${2:-only explicit resume controls are allowed}"
  echo "H100 Magna launcher refuses CLI override ${option}: ${detail}" >&2
  exit 2
}

validate_full_state_checkpoint() {
  local checkpoint_path="$1"
  local required_file rank resolved_checkpoint resolved_run_dir checkpoint_name expected_step
  if [[ ! -d "${checkpoint_path}" ]]; then
    invalid_cli_override "resume_from_checkpoint" "checkpoint directory does not exist: ${checkpoint_path}"
  fi
  resolved_checkpoint="$(realpath -e -- "${checkpoint_path}" 2>/dev/null || true)"
  resolved_run_dir="$(realpath -e -- "${H100_RUN_DIR}" 2>/dev/null || true)"
  if [[ -z "${resolved_checkpoint}" || -z "${resolved_run_dir}" ]]; then
    invalid_cli_override \
      "resume_from_checkpoint" \
      "checkpoint and RUN_ID directory must both resolve to existing paths"
  fi
  checkpoint_name="$(basename -- "${resolved_checkpoint}")"
  if [[ "$(dirname -- "${resolved_checkpoint}")" != "${resolved_run_dir}/checkpoints" \
    || ! "${checkpoint_name}" =~ ^steps_[0-9]+$ ]]; then
    invalid_cli_override \
      "resume_from_checkpoint" \
      "expected ${resolved_run_dir}/checkpoints/steps_N; got ${resolved_checkpoint}"
  fi
  expected_step="${checkpoint_name#steps_}"
  for required_file in model.safetensors optimizer.bin scheduler.bin trainer_state.json; do
    if [[ ! -s "${resolved_checkpoint}/${required_file}" ]]; then
      invalid_cli_override \
        "resume_from_checkpoint" \
        "full-state checkpoint is missing non-empty ${required_file}: ${resolved_checkpoint}"
    fi
  done
  for rank in 0 1 2 3 4 5 6 7; do
    required_file="random_states_${rank}.pkl"
    if [[ ! -s "${resolved_checkpoint}/${required_file}" ]]; then
      invalid_cli_override \
        "resume_from_checkpoint" \
        "full-state checkpoint is missing non-empty ${required_file}: ${resolved_checkpoint}"
    fi
  done
  if ! "${H100_PYTHON_BIN}" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    completed_steps = json.load(handle)["completed_steps"]
if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
    raise TypeError("completed_steps must be an integer")
if completed_steps != int(sys.argv[2]):
    raise ValueError(f"completed_steps={completed_steps} does not match {sys.argv[2]}")
' "${resolved_checkpoint}/trainer_state.json" "${expected_step}" 2>/dev/null; then
    invalid_cli_override \
      "resume_from_checkpoint" \
      "trainer_state.completed_steps must equal checkpoint suffix ${expected_step}: ${resolved_checkpoint}"
  fi
  H100_RESOLVED_CHECKPOINT="${resolved_checkpoint}"
}

validate_resume_cli_args() {
  # Exact CLI allowlist: resume enablement and one checkpoint path, using either
  # the root or trainer spelling. Everything else is profile-owned.
  local args=("$@")
  local arg_index=0
  local arg option value next_arg_index
  local is_resume_value="false"
  local is_resume_count=0
  local checkpoint_path=""
  local checkpoint_count=0
  while (( arg_index < ${#args[@]} )); do
    arg="${args[arg_index]}"
    if [[ "${arg}" == "--" ]]; then
      invalid_cli_override "--" "it would bypass protected H100 profile validation"
    fi
    if [[ "${arg}" != --* ]]; then
      invalid_cli_override "${arg}" "orphaned values are not allowed"
    fi

    option="${arg%%=*}"
    if [[ "${arg}" == *=* ]]; then
      value="${arg#*=}"
      next_arg_index=$((arg_index + 1))
    else
      if (( arg_index + 1 >= ${#args[@]} )) || [[ "${args[arg_index + 1]}" == --* ]]; then
        invalid_cli_override "${option}" "an explicit value is required"
      fi
      value="${args[arg_index + 1]}"
      next_arg_index=$((arg_index + 2))
    fi

    case "${option}" in
      --trainer.is_resume)
        is_resume_count=$((is_resume_count + 1))
        if (( is_resume_count > 1 )); then
          invalid_cli_override "${option}" "duplicate resume enablement is not allowed"
        fi
        if [[ "${value}" != "true" && "${value}" != "false" ]]; then
          invalid_cli_override "${option}" "expected true or false, got ${value}"
        fi
        is_resume_value="${value}"
        ;;
      --resume_from_checkpoint | --trainer.resume_from_checkpoint)
        checkpoint_count=$((checkpoint_count + 1))
        if (( checkpoint_count > 1 )); then
          invalid_cli_override \
            "${option}" \
            "provide exactly one root or trainer resume_from_checkpoint option"
        fi
        if [[ -z "${value}" ]]; then
          invalid_cli_override "${option}" "checkpoint path must not be empty"
        fi
        checkpoint_path="${value}"
        ;;
      *)
        invalid_cli_override "${option}"
        ;;
    esac
    arg_index="${next_arg_index}"
  done

  if [[ "${is_resume_value}" == "true" ]]; then
    if (( checkpoint_count != 1 )); then
      invalid_cli_override \
        "--trainer.is_resume" \
        "true requires exactly one root or trainer resume_from_checkpoint option"
    fi
    validate_full_state_checkpoint "${checkpoint_path}"
    H100_IS_RESUME=true
    H100_RESUME_ARGS=(
      --trainer.is_resume true
      --trainer.resume_from_checkpoint "${H100_RESOLVED_CHECKPOINT}"
    )
  else
    if (( checkpoint_count != 0 )); then
      invalid_cli_override \
        "resume_from_checkpoint" \
        "a checkpoint path requires --trainer.is_resume=true"
    fi
    H100_IS_RESUME=false
    if (( is_resume_count == 1 )); then
      H100_RESUME_ARGS=(--trainer.is_resume false)
    else
      H100_RESUME_ARGS=()
    fi
  fi
}

validate_resume_cli_args "$@"
if [[ "${H100_IS_RESUME}" == "false" \
  && ( -e "${H100_RUN_DIR}" || -L "${H100_RUN_DIR}" ) ]]; then
  invalid_cli_override \
    "RUN_ID" \
    "fresh training refuses existing run directory ${H100_RUN_DIR}; choose a new RUN_ID or resume explicitly"
fi

export CONFIG_YAML="${H100_CONFIG}"
export DATA_ROOT_DIR="${H100_DATA_ROOT}"
unset LIBERO_DATA_ROOT REALMAN_DATA_ROOT ACCELERATE_CONFIG STARVLA_DEEPSPEED_STAGE
export NUM_PROCESSES=8
export NUM_MACHINES=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PER_DEVICE_BATCH_SIZE=16
export STARVLA_USE_DEEPSPEED=0
export STARVLA_ALLOW_TORCH_COMPILE=0
export STARVLA_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export VIDEO_BACKEND=pyav
if [[ -n "${H100_MAX_TRAIN_STEPS}" ]]; then
  export MAX_TRAIN_STEPS="${H100_MAX_TRAIN_STEPS}"
else
  unset MAX_TRAIN_STEPS
fi
export SAVE_INTERVAL="${H100_SAVE_INTERVAL}"
export EVAL_INTERVAL="${H100_EVAL_INTERVAL}"
export LOGGING_FREQUENCY="${H100_LOGGING_FREQUENCY}"
export ACCELERATE_BIN="${H100_ACCELERATE_BIN}"

# Build or validate the immutable holdout from the exact config/launcher pair.
HOLDOUT_BUILD_OUTPUT="$(
  "${H100_PYTHON_BIN}" \
    "${REPO_ROOT}/deployment/realman/build_magna_internal_holdout.py" \
    --dataset-root "${H100_DATA_ROOT}" \
    --config "${H100_CONFIG}" \
    --launcher "${BASH_SOURCE[0]}" \
    --world-size 8
)"
printf '%s\n' "${HOLDOUT_BUILD_OUTPUT}"
HOLDOUT_BUILD_JSON="$(
  printf '%s\n' "${HOLDOUT_BUILD_OUTPUT}" \
    | sed -n 's/^MAGNA_HOLDOUT_BUILD_RESULT=//p' \
    | tail -n 1
)"
if [[ -z "${HOLDOUT_BUILD_JSON}" ]]; then
  echo "H100 holdout generator did not emit MAGNA_HOLDOUT_BUILD_RESULT" >&2
  exit 2
fi
EPISODE_SPLIT_MANIFEST="$(
  "${H100_PYTHON_BIN}" -c \
    'import json, sys; print(json.loads(sys.argv[1])["manifest_path"])' \
    "${HOLDOUT_BUILD_JSON}"
)"
if [[ "${EPISODE_SPLIT_MANIFEST}" != "${H100_MANIFEST}" ]]; then
  echo "H100 holdout generator returned ${EPISODE_SPLIT_MANIFEST}; expected ${H100_MANIFEST}" >&2
  exit 2
fi
if [[ ! -f "${EPISODE_SPLIT_MANIFEST}" ]]; then
  echo "H100 holdout generator returned a missing manifest: ${EPISODE_SPLIT_MANIFEST}" >&2
  exit 2
fi
export EPISODE_SPLIT_MANIFEST

# Only validated resume controls stay live. These final arguments protect the
# b16/global128 H100 data and model semantics; lifecycle mode also pins step 15.
exec "${SCRIPT_DIR}/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh" \
  "${H100_RESUME_ARGS[@]}" \
  --config_yaml "${H100_CONFIG}" \
  --run_id "${RUN_ID}" \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --framework.qwenvl.strict_attn_implementation true \
  --framework.qwenvl.enable_fast_linear_attention true \
  --framework.qwenvl.strict_fast_linear_attention true \
  --framework.action_model.rtc_training.enabled false \
  --framework.action_model.rtc_training.method prefix \
  --framework.action_model.rtc_training.max_delay 0 \
  --framework.action_model.rtc_training.distribution uniform \
  --framework.action_model.rtc_training.rtc_prob 0.0 \
  --framework.action_model.rtc_training.warmup_steps 0 \
  --framework.action_model.rtc_training.ramp_steps 0 \
  --framework.action_model.rtc_training.clean_time 1.0 \
  --framework.action_model.rtc_training.condition_dit_tokens false \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.dataset_py lerobot_datasets \
  --datasets.vla_data.data_root_dir "${H100_DATA_ROOT}" \
  --datasets.vla_data.data_mix magna_source_no_base_no_lift_interventions_v3 \
  --datasets.vla_data.lerobot_version v3.0 \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.qwen_observation_frame_index current \
  --datasets.vla_data.episode_split_manifest "${EPISODE_SPLIT_MANIFEST}" \
  --datasets.vla_data.load_all_data_for_training false \
  --datasets.vla_data.lerobot_statistics_source split_train \
  --datasets.vla_data.require_statistics_frame_count true \
  --datasets.vla_data.append_task_id_to_prompt false \
  --datasets.vla_data.append_subtask_to_prompt false \
  --datasets.vla_data.task_id_prompt_append_probability 0.0 \
  --datasets.vla_data.subtask_prompt_append_probability 0.0 \
  --datasets.vla_data.use_action_validity_prefix_mask true \
  --datasets.vla_data.action_validity_label_key valid_state \
  --datasets.vla_data.action_validity_positive_is_valid true \
  --datasets.vla_data.action_validity_invalid_run_length 10 \
  --datasets.vla_data.video_backend pyav \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.num_warmup_steps 2250 \
  --trainer.loss_scale.wm_warmup_steps 1500 \
  "${H100_LIFECYCLE_ARGS[@]}" \
  --trainer.save_interval "${H100_SAVE_INTERVAL}" \
  --trainer.eval_interval "${H100_EVAL_INTERVAL}" \
  --trainer.checkpoint_max_to_keep 3 \
  --trainer.logging_frequency "${H100_LOGGING_FREQUENCY}" \
  --trainer.eval_before_train true \
  --trainer.allow_training_stream_eval false \
  --trainer.compile_qwen_model false \
  --trainer.compile_qwen_model_dynamic false \
  --trainer.compile_action_model false \
  --trainer.compile_action_model_dynamic false \
  --trainer.compile_vj_predictor false \
  --trainer.compile_vj_predictor_dynamic false \
  --trainer.compile_vj_encoder false \
  --trainer.compile_vj_encoder_dynamic false \
  --trainer.compile_full_model false \
  --trainer.compile_full_model_dynamic false \
  --trainer.compile_dynamic false \
  --trainer.allow_compile_with_deepspeed false \
  --trainer.use_rabc false
