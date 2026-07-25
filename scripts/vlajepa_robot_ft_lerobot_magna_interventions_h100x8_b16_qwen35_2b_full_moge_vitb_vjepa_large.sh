#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

H100_CONFIG="${REPO_ROOT}/scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_h100x8_b16_qwen35_2b_full_moge_vitb_vjepa_large.yaml"

reject_conflicting_env() {
  local name="$1"
  local expected="$2"
  if [[ -n "${!name:-}" && "${!name}" != "${expected}" ]]; then
    echo "H100 Magna launcher requires ${name}=${expected}; got ${!name}" >&2
    exit 2
  fi
}

reject_conflicting_env CONFIG_YAML "${H100_CONFIG}"
reject_conflicting_env NUM_PROCESSES 8
reject_conflicting_env NUM_MACHINES 1
reject_conflicting_env CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7
reject_conflicting_env STARVLA_USE_DEEPSPEED 0
reject_conflicting_env STARVLA_ALLOW_TORCH_COMPILE 0
reject_conflicting_env STARVLA_DISABLE_TORCH_COMPILE 1
reject_conflicting_env TORCH_COMPILE_DISABLE 1
reject_conflicting_env TORCHDYNAMO_DISABLE 1
if [[ -n "${ACCELERATE_CONFIG:-}" || -n "${STARVLA_DEEPSPEED_STAGE:-}" ]]; then
  echo "H100 Magna launcher refuses inherited DeepSpeed configuration" >&2
  exit 2
fi
if [[ -n "${ACCELERATE_BIN:-}" ]]; then
  echo "H100 Magna launcher refuses inherited ACCELERATE_BIN=${ACCELERATE_BIN}" >&2
  exit 2
fi
if [[ -n "${STARVLA_H100_LIFECYCLE_TEST:-}" \
  && "${STARVLA_H100_LIFECYCLE_TEST}" != "0" ]]; then
  echo "H100 lifecycle budgets must use a dedicated YAML config, not launcher overrides" >&2
  exit 2
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

# Read every profile-owned value used by launcher validation from the pinned
# YAML. The launcher selects a profile; it must not maintain a second copy of
# that profile's data, run-directory, or checkpoint-selection contract.
mapfile -t H100_PROFILE_VALUES < <(
  "${H100_PYTHON_BIN}" -c '
from pathlib import Path
import re
import sys
import yaml

config_path = Path(sys.argv[1]).resolve(strict=True)
repo_root = Path(sys.argv[2]).resolve(strict=True)
payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise TypeError("H100 config root must be a mapping")
data = payload.get("datasets", {}).get("vla_data", {})
trainer = payload.get("trainer", {})
runtime = payload.get("runtime", {})
values = {
    "run_id": payload.get("run_id"),
    "run_root_dir": payload.get("run_root_dir"),
    "data_root_dir": data.get("data_root_dir"),
    "episode_split_manifest": data.get("episode_split_manifest"),
    "best_metric_name": trainer.get("best_metric_name"),
    "best_metric_mode": trainer.get("best_metric_mode"),
    "provenance_launcher": runtime.get("provenance_launcher"),
}
for key, value in values.items():
    if not isinstance(value, str) or not value:
        raise ValueError(f"H100 config requires non-empty string {key}")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", values["run_id"]):
    raise ValueError("H100 config run_id is not shell-safe")
run_root = Path(values["run_root_dir"])
data_root = Path(values["data_root_dir"])
if not run_root.is_absolute() or not data_root.is_absolute():
    raise ValueError("H100 run_root_dir and data_root_dir must be absolute")
manifest = Path(values["episode_split_manifest"])
if not manifest.is_absolute():
    manifest = repo_root / manifest
provenance_launcher = Path(values["provenance_launcher"])
if not provenance_launcher.is_absolute():
    provenance_launcher = repo_root / provenance_launcher
provenance_launcher = provenance_launcher.resolve()
if not provenance_launcher.is_relative_to(repo_root) or not provenance_launcher.is_file():
    raise ValueError("runtime.provenance_launcher must be a repository file")
for value in (
    values["run_id"],
    str(run_root),
    str(data_root),
    str(manifest.resolve()),
    values["best_metric_name"],
    values["best_metric_mode"],
    str(provenance_launcher),
):
    print(value)
' "${H100_CONFIG}" "${REPO_ROOT}"
)
if (( ${#H100_PROFILE_VALUES[@]} != 7 )); then
  echo "H100 Magna launcher could not read its complete profile contract from ${H100_CONFIG}" >&2
  exit 2
fi
H100_PROFILE_RUN_ID="${H100_PROFILE_VALUES[0]}"
H100_RUN_ROOT="${H100_PROFILE_VALUES[1]}"
H100_DATA_ROOT="${H100_PROFILE_VALUES[2]}"
H100_MANIFEST="${H100_PROFILE_VALUES[3]}"
H100_BEST_METRIC_NAME="${H100_PROFILE_VALUES[4]}"
H100_BEST_METRIC_MODE="${H100_PROFILE_VALUES[5]}"
H100_PROVENANCE_LAUNCHER="${H100_PROFILE_VALUES[6]}"
H100_RUN_ID_PREFIX="${H100_PROFILE_RUN_ID}_"

if [[ -n "${RUN_ID:-}" \
  && ! "${RUN_ID}" =~ ^${H100_RUN_ID_PREFIX}[A-Za-z0-9_.-]+$ ]]; then
  echo "H100 Magna RUN_ID must use prefix ${H100_RUN_ID_PREFIX} and a safe non-empty suffix; got ${RUN_ID}" >&2
  exit 2
fi
export RUN_ID="${RUN_ID:-${H100_RUN_ID_PREFIX}$(date +%Y%m%d_%H%M%S)}"
H100_RUN_DIR="${H100_RUN_ROOT}/${RUN_ID}"

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
import math
from pathlib import Path
import stat
import sys

checkpoint_path = Path(sys.argv[1])
expected_step = int(sys.argv[2])
expected_metric_name = sys.argv[3]
expected_metric_mode = sys.argv[4]
required_files = [
    "model.safetensors",
    "optimizer.bin",
    "scheduler.bin",
    "trainer_state.json",
    *(f"random_states_{rank}.pkl" for rank in range(8)),
]


def exact_int(value, label):
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def regular_nonempty(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise TypeError(f"checkpoint file must be regular: {path}")
    if info.st_nlink != 1 or info.st_size <= 0:
        raise ValueError(f"checkpoint file must be nonempty and unlinked: {path}")


def load_object(path):
    regular_nonempty(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint JSON must be an object: {path}")
    return payload


def validate_metric_identity(payload, label):
    if payload.get("best_metric_name") != expected_metric_name:
        raise ValueError(f"{label} uses the wrong best metric")
    if payload.get("best_metric_mode") != expected_metric_mode:
        raise ValueError(f"{label} uses the wrong best metric mode")


def validate_selection(payload, containing_step):
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("selection_state.json must use exact integer schema_version=1")
    validate_metric_identity(payload, "selection state")
    best_value = payload.get("best_metric_value")
    best_step = payload.get("best_metric_step")
    best_relative_path = payload.get("checkpoint_relative_path")
    if best_value is None and best_step is None:
        if best_relative_path is not None:
            raise ValueError("empty selection state must not name a checkpoint")
        return None
    if best_value is None or best_step is None:
        raise ValueError("best metric value and step must be set together")
    finite_number(best_value, "best metric value")
    best_step = exact_int(best_step, "best metric step")
    if best_step < 0 or best_step > containing_step:
        raise ValueError("best metric step is outside checkpoint history")
    if best_relative_path != f"checkpoints/steps_{best_step}":
        raise ValueError("best checkpoint path does not match its step")
    return best_step


def validate_legacy_dependency(trainer_state, containing_step):
    value = trainer_state.get("best_metric_value")
    step = trainer_state.get("best_metric_step")
    if value is None and step is None:
        return None
    if value is None:
        raise ValueError("legacy best step has no metric value")
    finite_number(value, "legacy best metric value")
    if step is None:
        return containing_step
    step = exact_int(step, "legacy best metric step")
    if step < 0 or step > containing_step:
        raise ValueError("legacy best metric step is outside checkpoint history")
    return step


validated = set()


def validate_checkpoint(path, step):
    resolved = path.resolve(strict=True)
    if resolved != path or not path.is_dir() or path.name != f"steps_{step}":
        raise ValueError(f"checkpoint dependency path is not canonical: {path}")
    if step in validated:
        return
    for filename in required_files:
        regular_nonempty(path / filename)
    trainer_state = load_object(path / "trainer_state.json")
    completed_steps = exact_int(
        trainer_state.get("completed_steps"), "trainer_state.completed_steps"
    )
    if completed_steps != step:
        raise ValueError("trainer_state.completed_steps does not match checkpoint suffix")
    for field in ("best_metric_name", "best_metric_mode"):
        if field in trainer_state:
            validate_metric_identity(trainer_state, "trainer state")
            break
    marker = trainer_state.get("selection_state_schema_version")
    if marker is not None and (type(marker) is not int or marker != 1):
        raise ValueError("unsupported trainer selection-state schema")
    selection_path = path / "selection_state.json"
    if selection_path.exists() or selection_path.is_symlink():
        dependency_step = validate_selection(load_object(selection_path), step)
    elif marker == 1:
        raise FileNotFoundError(f"required selection state is missing: {selection_path}")
    else:
        dependency_step = validate_legacy_dependency(trainer_state, step)
    validated.add(step)
    if dependency_step is None or dependency_step == step:
        return
    dependency_path = path.parent / f"steps_{dependency_step}"
    validate_checkpoint(dependency_path, dependency_step)


validate_checkpoint(checkpoint_path, expected_step)
' "${resolved_checkpoint}" "${expected_step}" "${H100_BEST_METRIC_NAME}" "${H100_BEST_METRIC_MODE}" 2>/dev/null; then
    invalid_cli_override \
      "resume_from_checkpoint" \
      "checkpoint trainer/selection state is incomplete or inconsistent at step ${expected_step}: ${resolved_checkpoint}"
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
unset DATA_ROOT_DIR LIBERO_DATA_ROOT REALMAN_DATA_ROOT ACCELERATE_CONFIG STARVLA_DEEPSPEED_STAGE
export NUM_PROCESSES=8
export NUM_MACHINES=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export STARVLA_USE_DEEPSPEED=0
export STARVLA_ALLOW_TORCH_COMPILE=0
export STARVLA_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset PER_DEVICE_BATCH_SIZE VIDEO_BACKEND EPOCHS MAX_TRAIN_STEPS NUM_WARMUP_STEPS
unset SAVE_INTERVAL EVAL_INTERVAL LOGGING_FREQUENCY FIND_UNUSED_PARAMETERS
unset DDP_GRADIENT_AS_BUCKET_VIEW DDP_STATIC_GRAPH DDP_BUCKET_CAP_MB
unset DATALOADER_NUM_WORKERS DATALOADER_PREFETCH_FACTOR
unset DATALOADER_TIMEOUT_SECONDS DATALOADER_PERSISTENT_WORKERS
unset VIDEO_BACKEND_NUM_THREADS
export ACCELERATE_BIN="${H100_ACCELERATE_BIN}"
export STARVLA_CONFIG_IS_AUTHORITATIVE=1

# Build or validate the immutable holdout from the exact config/launcher pair.
HOLDOUT_BUILD_OUTPUT="$(
  "${H100_PYTHON_BIN}" \
    "${REPO_ROOT}/deployment/realman/build_magna_internal_holdout.py" \
    --dataset-root "${H100_DATA_ROOT}" \
    --config "${H100_CONFIG}" \
    --launcher "${H100_PROVENANCE_LAUNCHER}" \
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

# The YAML is the sole owner of model, data, optimizer, schedule, evaluation,
# and checkpoint settings. Only validated resume identity is runtime-derived.
exec "${SCRIPT_DIR}/vlajepa_robot_ft_libero_plus_a100x8_qwen3_full_moge_vitb_vjepa_large.sh" \
  "${H100_RESUME_ARGS[@]}"
