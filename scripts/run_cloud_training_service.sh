#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: RUN_ENV_FILE=/path/to/run.env run_cloud_training_service.sh MODE [TRAIN_ARGS...]

Modes:
  train        Launch the configured training job and write its exit marker.
  tensorboard  Serve the run's TensorBoard event directory without GPU access.
  uploader     Mirror stable checkpoints, metadata, logs, and final weights to GCS.
  status       Print process, checkpoint, and exit-marker state for the run.
EOF
}

REQUESTED_MODE="${1:-}"
if [[ -z "${REQUESTED_MODE}" ]]; then
  usage
  exit 2
fi
shift

case "${REQUESTED_MODE}" in
  train|tensorboard|uploader|status) ;;
  *)
    usage
    exit 2
    ;;
esac
readonly REQUESTED_MODE

RUN_ENV_FILE_ARG="${RUN_ENV_FILE:-}"
readonly RUN_ENV_FILE_ARG
RUN_ENV_FILE="${RUN_ENV_FILE_ARG}"
if [[ "${REQUESTED_MODE}" != "status" && -z "${RUN_ENV_FILE}" ]]; then
  echo "RUN_ENV_FILE is required for ${REQUESTED_MODE} mode" >&2
  exit 2
fi
if [[ -n "${RUN_ENV_FILE}" ]]; then
  if [[ ! -f "${RUN_ENV_FILE}" || ! -r "${RUN_ENV_FILE}" ]]; then
    echo "RUN_ENV_FILE is not a readable regular file: ${RUN_ENV_FILE}" >&2
    exit 2
  fi
  launch_env_mode="$(stat -c '%a' "${RUN_ENV_FILE}")"
  if (( (8#${launch_env_mode} & 022) != 0 )); then
    echo "RUN_ENV_FILE must not be group- or world-writable: ${RUN_ENV_FILE}" >&2
    exit 2
  fi
  if grep -Evq '^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_./:@,+-]*$|^[[:space:]]*(#.*)?$' "${RUN_ENV_FILE}"; then
    echo "RUN_ENV_FILE must contain only comments, blanks, and literal KEY=value assignments" >&2
    exit 2
  fi
  if grep -Eiq '^[A-Za-z_][A-Za-z0-9_]*(TOKEN|PASSWORD|PASSWD|SECRET|PRIVATE_KEY|API_KEY|ACCESS_KEY|CREDENTIALS?)[A-Za-z0-9_]*=' "${RUN_ENV_FILE}"; then
    echo "Refusing to use or publish a launch environment containing a secret-like variable" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "${RUN_ENV_FILE}"
  set +a
fi
RUN_ENV_FILE="${RUN_ENV_FILE_ARG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required setting is missing: ${name}" >&2
    exit 2
  fi
}

require_value RUN_ID
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsupported characters: ${RUN_ID}" >&2
  exit 2
fi

IMAGE="${IMAGE:-vla-jepa:py313-cu130-a100}"
VLA_JEPA_SCRATCH="${VLA_JEPA_SCRATCH:-/mnt/vla-jepa}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-${VLA_JEPA_SCRATCH}/datasets/magna_training_data_with_interventions}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${VLA_JEPA_SCRATCH}/checkpoints}"
LOG_ROOT="${LOG_ROOT:-${VLA_JEPA_SCRATCH}/logs}"
RUN_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-./scripts/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.sh}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29573}"
STARVLA_USE_DEEPSPEED="${STARVLA_USE_DEEPSPEED:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
GCS_DEST="${GCS_DEST:-}"
PREFLIGHT_MANIFEST="${PREFLIGHT_MANIFEST:-}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:-}"
RUN_SOURCE="${RUN_SOURCE:-}"
TRAIN_CONTAINER_NAME="${TRAIN_CONTAINER_NAME:-${RUN_ID}-train}"
TENSORBOARD_CONTAINER_NAME="${TENSORBOARD_CONTAINER_NAME:-${RUN_ID}-tensorboard}"
HANDOFF_METADATA_DIR="${HANDOFF_METADATA_DIR:-${LOG_ROOT}/run_metadata/${RUN_ID}}"

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}"

acquire_service_lock() {
  local service_name="$1"
  local lock_path="${LOG_ROOT}/${RUN_ID}.${service_name}.lock"
  exec 9>"${lock_path}"
  if ! flock -n 9; then
    echo "Another ${service_name} service already owns ${lock_path}" >&2
    exit 2
  fi
}

case "${REQUESTED_MODE}" in
  train)
    SERVICE_NAME=train
    SERVICE_EXIT_PATH="${LOG_ROOT}/${RUN_ID}.exit"
    ;;
  tensorboard)
    SERVICE_NAME=tensorboard
    SERVICE_EXIT_PATH="${LOG_ROOT}/${RUN_ID}.tensorboard.exit"
    ;;
  uploader)
    SERVICE_NAME=uploader
    SERVICE_EXIT_PATH="${LOG_ROOT}/${RUN_ID}.gcs_upload.exit"
    ;;
  status)
    SERVICE_NAME=""
    SERVICE_EXIT_PATH=""
    ;;
esac
if [[ -n "${SERVICE_NAME}" ]]; then
  acquire_service_lock "${SERVICE_NAME}"
fi

record_service_exit() {
  local status="$?"
  trap - EXIT
  if [[ -n "${SERVICE_EXIT_PATH}" ]]; then
    printf '%s\n' "${status}" > "${SERVICE_EXIT_PATH}" || true
  fi
  exit "${status}"
}
if [[ -n "${SERVICE_EXIT_PATH}" ]]; then
  rm -f "${SERVICE_EXIT_PATH}"
  trap record_service_exit EXIT
fi

if [[ -n "${RUN_SOURCE}" ]]; then
  if [[ ! -d "${RUN_SOURCE}" ]]; then
    echo "RUN_SOURCE is not a directory: ${RUN_SOURCE}" >&2
    exit 2
  fi
  resolved_run_source="$(realpath -e "${RUN_SOURCE}")"
  if [[ "${resolved_run_source}" != "${REPO_ROOT}" ]]; then
    echo "Runtime source path mismatch: expected ${resolved_run_source}, invoked ${REPO_ROOT}" >&2
    exit 2
  fi
fi

if [[ -n "${EXPECTED_SOURCE_COMMIT}" ]]; then
  actual_source_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  if [[ "${actual_source_commit}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
    echo "Runtime source mismatch: expected ${EXPECTED_SOURCE_COMMIT}, found ${actual_source_commit}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "Runtime source worktree is dirty: ${REPO_ROOT}" >&2
    exit 2
  fi
fi

run_logged() {
  local log_path="$1"
  local exit_path="$2"
  shift 2

  rm -f "${exit_path}"
  set +e
  "$@" 2>&1 | tee "${log_path}"
  local -a pipeline_status=("${PIPESTATUS[@]}")
  set -e
  local status="${pipeline_status[0]}"
  if [[ "${status}" -eq 0 && "${pipeline_status[1]}" -ne 0 ]]; then
    status="${pipeline_status[1]}"
  fi
  printf '%s\n' "${status}" > "${exit_path}"
  return "${status}"
}

wait_for_path() {
  local path="$1"
  while [[ ! -e "${path}" ]]; do
    sleep 5
  done
}

prepare_reproducibility_metadata() {
  mkdir -p "${HANDOFF_METADATA_DIR}"
  if [[ -n "${RUN_ENV_FILE}" ]]; then
    cp -p "${RUN_ENV_FILE}" "${HANDOFF_METADATA_DIR}/launch.env"
    chmod 0644 "${HANDOFF_METADATA_DIR}/launch.env"
  fi
  if [[ -n "${PREFLIGHT_MANIFEST}" ]]; then
    if [[ ! -r "${PREFLIGHT_MANIFEST}" ]]; then
      echo "PREFLIGHT_MANIFEST is not readable: ${PREFLIGHT_MANIFEST}" >&2
      exit 2
    fi
    cp -p "${PREFLIGHT_MANIFEST}" \
      "${HANDOFF_METADATA_DIR}/production_preflight_manifest.txt"
    chmod 0644 "${HANDOFF_METADATA_DIR}/production_preflight_manifest.txt"
  fi
}

case "${REQUESTED_MODE}" in
  train)
    if [[ -d "${RUN_DIR}" && -z "${RESUME_CHECKPOINT}" ]]; then
      existing_run_entry="$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)"
      if [[ -n "${existing_run_entry}" ]]; then
        echo "Run directory is not empty; use a new RUN_ID or set RESUME_CHECKPOINT: ${RUN_DIR}" >&2
        exit 2
      fi
    fi
    if [[ -n "${RESUME_CHECKPOINT}" && ! -d "${RESUME_CHECKPOINT}" ]]; then
      echo "RESUME_CHECKPOINT is not a complete directory: ${RESUME_CHECKPOINT}" >&2
      exit 2
    fi
    if [[ "${TRAIN_LAUNCHER}" == /* ]]; then
      launcher_on_host="${TRAIN_LAUNCHER}"
    else
      launcher_on_host="${REPO_ROOT}/${TRAIN_LAUNCHER#./}"
    fi
    if [[ ! -x "${launcher_on_host}" ]]; then
      echo "Training launcher is not executable: ${launcher_on_host}" >&2
      exit 2
    fi

    train_args=("$@")
    if [[ -n "${RESUME_CHECKPOINT}" ]]; then
      train_args+=(
        --trainer.is_resume true
        --trainer.resume_from_checkpoint "${RESUME_CHECKPOINT}"
      )
    fi

    cd "${REPO_ROOT}"
    run_logged "${LOG_ROOT}/${RUN_ID}.log" "${LOG_ROOT}/${RUN_ID}.exit" \
      env \
        IMAGE="${IMAGE}" \
        DOCKER_NAME="${TRAIN_CONTAINER_NAME}" \
        DOCKER_TTY=0 \
        MOUNT_GCLOUD=0 \
        VLA_JEPA_SCRATCH="${VLA_JEPA_SCRATCH}" \
        DATA_ROOT_DIR="${DATA_ROOT_DIR}" \
        CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
        RUN_ID="${RUN_ID}" \
        NUM_PROCESSES="${NUM_PROCESSES}" \
        MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
        STARVLA_USE_DEEPSPEED="${STARVLA_USE_DEEPSPEED}" \
        ./scripts/docker_run_training.sh \
          "${TRAIN_LAUNCHER}" "${train_args[@]}"
    ;;

  tensorboard)
    wait_for_path "${RUN_DIR}/starvla"
    cd "${REPO_ROOT}"
    run_logged \
      "${LOG_ROOT}/${RUN_ID}.tensorboard.log" \
      "${LOG_ROOT}/${RUN_ID}.tensorboard.exit" \
      env \
        IMAGE="${IMAGE}" \
        DOCKER_NAME="${TENSORBOARD_CONTAINER_NAME}" \
        DOCKER_GPU_MODE=none \
        DOCKER_TTY=0 \
        MOUNT_GCLOUD=0 \
        VLA_JEPA_SCRATCH="${VLA_JEPA_SCRATCH}" \
        ./scripts/docker_run_training.sh \
          tensorboard \
            --logdir "${RUN_DIR}/starvla" \
            --host 0.0.0.0 \
            --port "${TENSORBOARD_PORT:-6006}" \
            --reload_interval 5
    ;;

  uploader)
    require_value GCS_DEST
    if [[ "${GCS_DEST}" != gs://* ]]; then
      echo "GCS_DEST must start with gs://: ${GCS_DEST}" >&2
      exit 2
    fi
    prepare_reproducibility_metadata
    wait_for_path "${RUN_DIR}"
    cd "${REPO_ROOT}"
    run_logged \
      "${LOG_ROOT}/${RUN_ID}.gcs_upload.log" \
      "${LOG_ROOT}/${RUN_ID}.gcs_upload.exit" \
      env \
        CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-${VLA_JEPA_SCRATCH}/gcloud-config}" \
        POLL_SECONDS="${POLL_SECONDS:-60}" \
        STABLE_SECONDS="${STABLE_SECONDS:-180}" \
        LOG_SYNC_SECONDS="${LOG_SYNC_SECONDS:-900}" \
        UPLOAD_FAILURE_BACKOFF_SECONDS="${UPLOAD_FAILURE_BACKOFF_SECONDS:-900}" \
        REMOTE_CHECKPOINT_MAX_TO_KEEP="${REMOTE_CHECKPOINT_MAX_TO_KEEP:-3}" \
        EXTRA_METADATA_DIR="${HANDOFF_METADATA_DIR}" \
        ./scripts/watch_and_upload_checkpoints_gcs.sh "${RUN_DIR}" "${GCS_DEST}"
    ;;

  status)
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'run_id=%s\nrun_dir=%s\nimage=%s\n' "${RUN_ID}" "${RUN_DIR}" "${IMAGE}"
    docker ps --filter "name=^/${TRAIN_CONTAINER_NAME}$" \
      --format 'train_container={{.Names}} status={{.Status}} image={{.Image}}' || true
    docker ps --filter "name=^/${TENSORBOARD_CONTAINER_NAME}$" \
      --format 'tensorboard_container={{.Names}} status={{.Status}} image={{.Image}}' || true
    for service_name in train tensorboard uploader; do
      lock_path="${LOG_ROOT}/${RUN_ID}.${service_name}.lock"
      if flock -n "${lock_path}" true; then
        printf '%s_lock=free\n' "${service_name}"
      else
        printf '%s_lock=held\n' "${service_name}"
      fi
    done
    for marker in \
      "${LOG_ROOT}/${RUN_ID}.exit" \
      "${LOG_ROOT}/${RUN_ID}.tensorboard.exit" \
      "${LOG_ROOT}/${RUN_ID}.gcs_upload.exit"; do
      if [[ -f "${marker}" ]]; then
        printf '%s=%s\n' "$(basename "${marker}")" "$(cat "${marker}")"
      fi
    done
    if [[ -d "${RUN_DIR}/checkpoints" ]]; then
      find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
        -printf 'checkpoint=%f\n' | sort -V
    fi
    ;;

esac
