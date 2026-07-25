#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 3 )); then
  echo "Usage: $0 CHECKPOINT_DIR|STEP0_INIT [EVAL_RUN_ID] [LEGACY_ORIGINAL_MANIFEST]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_TARGET="$1"
CHECKPOINT_DIR=""
EVAL_TARGET_ARGS=()
if [[ "${EVAL_TARGET}" == "STEP0_INIT" ]]; then
  if [[ -z "${SOURCE_RUN_CONFIG:-}" ]]; then
    echo "STEP0_INIT requires SOURCE_RUN_CONFIG=/path/to/frozen/run/config.yaml" >&2
    exit 2
  fi
  EVAL_TARGET_LABEL="step0_init"
  EVAL_TARGET_ARGS+=(
    --trainer.eval_only_untrained_initialization true
    --trainer.is_resume false
  )
else
  CHECKPOINT_DIR="$(realpath "${EVAL_TARGET}")"
  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
    exit 2
  fi
  if [[ "$(basename "${CHECKPOINT_DIR}")" != steps_[0-9]* ]]; then
    echo "Checkpoint directory must be named steps_N: ${CHECKPOINT_DIR}" >&2
    exit 2
  fi
  SOURCE_RUN_CONFIG="${SOURCE_RUN_CONFIG:-${CHECKPOINT_DIR}/../../config.yaml}"
  EVAL_TARGET_LABEL="$(basename "${CHECKPOINT_DIR}")"
  EVAL_TARGET_ARGS+=(
    --trainer.eval_only_untrained_initialization false
    --trainer.is_resume true
    --resume_from_checkpoint "${CHECKPOINT_DIR}"
  )
fi

SOURCE_RUN_CONFIG="$(realpath "${SOURCE_RUN_CONFIG}")"
if [[ ! -f "${SOURCE_RUN_CONFIG}" ]]; then
  echo "Frozen source run config does not exist: ${SOURCE_RUN_CONFIG}" >&2
  exit 2
fi
SOURCE_RUN_CONFIG_SHA256="$(sha256sum "${SOURCE_RUN_CONFIG}" | awk '{print $1}')"
CONFIG_YAML="${SOURCE_RUN_CONFIG}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/mnt/vla-jepa/datasets/magna_training_data_with_interventions}"
EVAL_RUN_ROOT="${EVAL_RUN_ROOT:-/mnt/vla-jepa/evals}"
EVAL_RUN_ID="${2:-magna_checkpoint_eval_${EVAL_TARGET_LABEL}_$(date +%Y%m%d_%H%M%S)}"
LEGACY_ORIGINAL_MANIFEST="${3:-}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29671}"

if [[ "${NUM_PROCESSES}" != "8" ]]; then
  echo "Magna checkpoint eval requires NUM_PROCESSES=8; got ${NUM_PROCESSES}" >&2
  exit 2
fi
if [[ -n "${CHECKPOINT_DIR}" && "${EVAL_RUN_ROOT}/${EVAL_RUN_ID}" == "${CHECKPOINT_DIR}"* ]]; then
  echo "Eval output must be separate from the source checkpoint tree" >&2
  exit 2
fi

LEGACY_AUDIT_ARGS=()
if [[ -n "${LEGACY_ORIGINAL_MANIFEST}" ]]; then
  LEGACY_ORIGINAL_MANIFEST="$(realpath "${LEGACY_ORIGINAL_MANIFEST}")"
  if [[ ! -f "${LEGACY_ORIGINAL_MANIFEST}" ]]; then
    echo "Legacy manifest does not exist: ${LEGACY_ORIGINAL_MANIFEST}" >&2
    exit 2
  fi
  # The original manifest binds its original train-only statistics by relative
  # path and SHA-256. The evaluator validates that binding and then removes only
  # historical zero-supervision episode 955, with no replacement.
  LEGACY_AUDIT_ARGS+=(
    --datasets.vla_data.episode_split_manifest "${LEGACY_ORIGINAL_MANIFEST}"
    --trainer.eval_only_legacy_underfilled_holdout true
    --trainer.eval_only_legacy_excluded_zero_valid_episode_ids "[955]"
  )
fi

source "${SCRIPT_DIR}/lib/training_env.sh"
starvla_configure_common_training_env ens8
export STARVLA_USE_DEEPSPEED=0
export STARVLA_ALLOW_TORCH_COMPILE=0
export STARVLA_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export VLA_JEPA_SCRATCH="${VLA_JEPA_SCRATCH:-/mnt/vla-jepa}"
export HF_HOME="${HF_HOME:-${VLA_JEPA_SCRATCH}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${VLA_JEPA_SCRATCH}/cache/torch}"
export TMPDIR="${TMPDIR:-${VLA_JEPA_SCRATCH}/tmp}"
export STARVLA_MOGE_REPO_PATH="${STARVLA_MOGE_REPO_PATH:-${VLA_JEPA_SCRATCH}/src/MoGe}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${TMPDIR}" "${EVAL_RUN_ROOT}"

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || true)}"
if [[ -z "${ACCELERATE_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "An executable accelerate is required" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${ACCELERATE_BIN}" launch \
  --num_processes 8 \
  --num_machines 1 \
  --dynamo_backend no \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision bf16 \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${EVAL_RUN_ID}" \
  --run_root_dir "${EVAL_RUN_ROOT}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}" \
  --framework.qwenvl.device_map null \
  --trainer.eval_only true \
  --trainer.eval_source_training_config_path "${SOURCE_RUN_CONFIG}" \
  --trainer.eval_source_training_config_sha256 "${SOURCE_RUN_CONFIG_SHA256}" \
  --trainer.heldout_focused_eval_enabled true \
  --trainer.heldout_eval_movement_threshold 0.02 \
  --trainer.heldout_focused_eval_required_subtasks "[2,3,4,5,6,7]" \
  --trainer.heldout_focused_eval_min_evaluable_observations_per_subtask 5 \
  --trainer.heldout_focused_eval_transition_coverage_horizon 10 \
  --trainer.heldout_focused_eval_min_arm_movement_elements_h10 1000 \
  --trainer.heldout_focused_eval_min_arm_movement_hold_abs_h10 50.0 \
  --trainer.heldout_focused_eval_min_open_to_close_transitions 16 \
  --trainer.heldout_focused_eval_min_close_to_open_transitions 16 \
  --trainer.heldout_focused_eval_min_open_to_close_windows 16 \
  --trainer.heldout_focused_eval_min_close_to_open_windows 16 \
  --trainer.best_metric_name heldout_focused_eval_task_failure_score_h10 \
  --trainer.best_metric_mode min \
  --trainer.compile_qwen_model false \
  --trainer.compile_action_model false \
  --trainer.compile_vj_predictor false \
  --trainer.compile_vj_encoder false \
  --trainer.compile_full_model false \
  --trainer.resume_load_optimizer_state false \
  --trainer.eval_before_train false \
  "${EVAL_TARGET_ARGS[@]}" \
  "${LEGACY_AUDIT_ARGS[@]}"
