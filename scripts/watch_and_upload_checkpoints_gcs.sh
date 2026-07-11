#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 RUN_DIR GCS_DEST" >&2
  echo "Example: $0 /mnt/vla-jepa/checkpoints/run gs://bucket/checkpoints/vla-jepa/run" >&2
  exit 2
fi

RUN_DIR="$1"
GCS_DEST="${2%/}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-180}"
UPLOAD_FAILURE_BACKOFF_SECONDS="${UPLOAD_FAILURE_BACKOFF_SECONDS:-900}"
LOG_SYNC_SECONDS="${LOG_SYNC_SECONDS:-900}"
RUN_ONCE="${RUN_ONCE:-0}"
REMOTE_CHECKPOINT_MAX_TO_KEEP="${REMOTE_CHECKPOINT_MAX_TO_KEEP:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
FINAL_MODEL_DIR="${FINAL_MODEL_DIR:-${RUN_DIR}/final_model}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_DIR}/starvla}"
EXTRA_METADATA_DIR="${EXTRA_METADATA_DIR:-}"

if [[ ! "${REMOTE_CHECKPOINT_MAX_TO_KEEP}" =~ ^[0-9]+$ ]]; then
  echo "REMOTE_CHECKPOINT_MAX_TO_KEEP must be a non-negative integer" >&2
  exit 2
fi

if [[ -z "${STATE_DIR:-}" ]]; then
  run_state_name="$(basename "${RUN_DIR}" | tr '/ ' '__' | tr -cd 'A-Za-z0-9._=-')"
  if [[ -n "${UPLOAD_STATE_ROOT:-}" ]]; then
    STATE_DIR="${UPLOAD_STATE_ROOT%/}/${run_state_name}"
  elif [[ -w "${RUN_DIR}" ]]; then
    STATE_DIR="${RUN_DIR}/.upload_state"
  elif [[ -d /mnt/vla-jepa/logs && -w /mnt/vla-jepa/logs ]]; then
    STATE_DIR="/mnt/vla-jepa/logs/checkpoint_upload_state/${run_state_name}"
  else
    STATE_DIR="${RUN_DIR}/.upload_state"
  fi
fi

if [[ -z "${CLOUDSDK_CONFIG:-}" && -d /mnt/vla-jepa/gcloud-config ]]; then
  export CLOUDSDK_CONFIG=/mnt/vla-jepa/gcloud-config
fi

mkdir -p "${STATE_DIR}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

safe_name() {
  printf '%s' "$1" | tr '/ ' '__' | tr -cd 'A-Za-z0-9._=-'
}

is_stable_dir() {
  local dir="$1"
  local newest
  newest="$(find "${dir}" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 || true)"
  if [[ -z "${newest}" ]]; then
    return 1
  fi
  python3 - "$newest" "$STABLE_SECONDS" <<'PY'
import sys, time
newest = float(sys.argv[1])
stable_seconds = float(sys.argv[2])
raise SystemExit(0 if time.time() - newest >= stable_seconds else 1)
PY
}

upload_metadata() {
  local marker="${STATE_DIR}/uploaded_metadata"
  local metadata_dir="${STATE_DIR}/metadata"
  local upload_log="${STATE_DIR}/metadata_upload.log"
  local fingerprint

  mkdir -p "${metadata_dir}"
  find "${metadata_dir}" -mindepth 1 -maxdepth 1 -type f -delete
  for source_dir in "${EXTRA_METADATA_DIR}" "${RUN_DIR}"; do
    if [[ -z "${source_dir}" || ! -d "${source_dir}" ]]; then
      continue
    fi
    find "${source_dir}" -maxdepth 1 -type f \
      \( -name 'config.yaml' -o -name 'config.json' -o -name 'dataset_statistics.json' -o -name 'canonical_subset_summary.json' -o -name 'canonical_subset_manifest.jsonl' -o -name 'launch.env' -o -name 'production_preflight_manifest.txt' \) \
      -exec cp -p {} "${metadata_dir}/" \;
  done
  if ! find "${metadata_dir}" -type f | grep -q .; then
    return 0
  fi

  fingerprint="$({
    cd "${metadata_dir}"
    find . -maxdepth 1 -type f -printf '%P\0' \
      | sort -z \
      | xargs -0 sha256sum
  } | sha256sum | awk '{print $1}')"
  if [[ -e "${marker}" && "$(cat "${marker}")" == "${fingerprint}" ]]; then
    return 0
  fi

  log "Synchronizing changed run metadata -> ${GCS_DEST}/metadata"
  if gcloud storage rsync --recursive "${metadata_dir}" "${GCS_DEST}/metadata" >"${upload_log}" 2>&1; then
    printf '%s\n' "${fingerprint}" > "${marker}"
    log "Synchronized run metadata"
  else
    log "Run metadata upload failed; will retry in ${UPLOAD_FAILURE_BACKOFF_SECONDS}s (details: ${upload_log})"
    return 1
  fi
}

upload_checkpoint_dir() {
  local ckpt_dir="$1"
  local ckpt_name
  local marker
  local upload_log
  ckpt_name="$(basename "${ckpt_dir}")"
  marker="${STATE_DIR}/uploaded_$(safe_name "${ckpt_name}")"
  upload_log="${STATE_DIR}/$(safe_name "${ckpt_name}")_upload.log"

  if [[ -e "${marker}" ]]; then
    return 0
  fi
  if ! is_stable_dir "${ckpt_dir}"; then
    return 0
  fi
  local unreadable_file
  unreadable_file="$(find "${ckpt_dir}" -type f ! -readable -print -quit 2>/dev/null || true)"
  if [[ -n "${unreadable_file}" ]]; then
    log "Checkpoint ${ckpt_name} is not host-readable: ${unreadable_file}"
    return 1
  fi

  log "Uploading checkpoint ${ckpt_name} -> ${GCS_DEST}/checkpoints/${ckpt_name}"
  if gcloud storage rsync --recursive "${ckpt_dir}" "${GCS_DEST}/checkpoints/${ckpt_name}" >"${upload_log}" 2>&1; then
    date -Is > "${marker}"
    log "Uploaded checkpoint ${ckpt_name}"
  else
    log "Checkpoint upload failed for ${ckpt_name}; will retry in ${UPLOAD_FAILURE_BACKOFF_SECONDS}s (details: ${upload_log})"
    return 1
  fi
}

prune_remote_checkpoints() {
  if (( REMOTE_CHECKPOINT_MAX_TO_KEEP <= 0 )); then
    return 0
  fi

  local list_log="${STATE_DIR}/remote_checkpoint_list.log"
  local prune_log="${STATE_DIR}/remote_checkpoint_prune.log"
  local remote_listing
  if ! remote_listing="$(gcloud storage ls "${GCS_DEST}/checkpoints/" 2>"${list_log}")"; then
    if grep -Eqi 'matched no objects|not found|does not exist' "${list_log}"; then
      return 0
    fi
    log "Unable to list remote checkpoints (details: ${list_log})"
    return 1
  fi

  local -a remote_checkpoints=()
  local checkpoint_name
  local listing_path
  mapfile -t remote_checkpoints < <(
    while IFS= read -r listing_path; do
      listing_path="${listing_path%/}"
      checkpoint_name="${listing_path##*/}"
      if [[ "${checkpoint_name}" =~ ^steps_([0-9]+)$ ]]; then
        printf '%020d\t%s/\n' "${BASH_REMATCH[1]}" "${listing_path}"
      fi
    done <<< "${remote_listing}" | sort -n
  )

  local prune_count=$(( ${#remote_checkpoints[@]} - REMOTE_CHECKPOINT_MAX_TO_KEEP ))
  if (( prune_count <= 0 )); then
    return 0
  fi

  : > "${prune_log}"
  local index
  local remote_path
  for (( index = 0; index < prune_count; index++ )); do
    remote_path="${remote_checkpoints[index]#*$'\t'}"
    log "Pruning old remote checkpoint ${remote_path}"
    if ! gcloud storage rm --recursive "${remote_path}" >>"${prune_log}" 2>&1; then
      log "Unable to prune remote checkpoint ${remote_path} (details: ${prune_log})"
      return 1
    fi
  done
}

upload_runtime_logs() {
  local marker="${STATE_DIR}/uploaded_runtime_logs"
  local upload_log="${STATE_DIR}/runtime_logs_upload.log"
  local now
  local last_sync=0

  if [[ ! -d "${TENSORBOARD_DIR}" && ! -f "${RUN_DIR}/summary.jsonl" ]]; then
    return 0
  fi
  if [[ -e "${marker}" ]]; then
    last_sync="$(stat -c '%Y' "${marker}" 2>/dev/null || printf '0')"
  fi
  now="$(date +%s)"
  if (( now - last_sync < LOG_SYNC_SECONDS )); then
    return 0
  fi

  : > "${upload_log}"
  if [[ -d "${TENSORBOARD_DIR}" ]]; then
    log "Synchronizing TensorBoard logs -> ${GCS_DEST}/logs/starvla"
    if ! gcloud storage rsync --recursive "${TENSORBOARD_DIR}" "${GCS_DEST}/logs/starvla" >>"${upload_log}" 2>&1; then
      log "TensorBoard log upload failed; will retry (details: ${upload_log})"
      return 1
    fi
  fi
  if [[ -f "${RUN_DIR}/summary.jsonl" ]]; then
    if ! gcloud storage cp "${RUN_DIR}/summary.jsonl" "${GCS_DEST}/metadata/summary.jsonl" >>"${upload_log}" 2>&1; then
      log "Training summary upload failed; will retry (details: ${upload_log})"
      return 1
    fi
  fi
  date -Is > "${marker}"
}

upload_final_model() {
  local marker="${STATE_DIR}/uploaded_final_model"
  local upload_log="${STATE_DIR}/final_model_upload.log"

  if [[ -e "${marker}" || ! -d "${FINAL_MODEL_DIR}" ]]; then
    return 0
  fi
  if ! is_stable_dir "${FINAL_MODEL_DIR}"; then
    return 0
  fi
  local unreadable_file
  unreadable_file="$(find "${FINAL_MODEL_DIR}" -type f ! -readable -print -quit 2>/dev/null || true)"
  if [[ -n "${unreadable_file}" ]]; then
    log "Final model is not host-readable: ${unreadable_file}"
    return 1
  fi

  log "Uploading final model -> ${GCS_DEST}/final_model"
  if gcloud storage rsync --recursive "${FINAL_MODEL_DIR}" "${GCS_DEST}/final_model" >"${upload_log}" 2>&1; then
    date -Is > "${marker}"
    log "Uploaded final model"
  else
    log "Final model upload failed; will retry in ${UPLOAD_FAILURE_BACKOFF_SECONDS}s (details: ${upload_log})"
    return 1
  fi
}

run_upload_cycle() {
  if ! upload_metadata; then
    return 1
  fi
  if [[ -d "${CHECKPOINT_DIR}" ]]; then
    while IFS= read -r -d '' ckpt_dir; do
      if ! upload_checkpoint_dir "${ckpt_dir}"; then
        return 1
      fi
    done < <(find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -zV)
  fi
  if ! prune_remote_checkpoints; then
    return 1
  fi
  if ! upload_final_model; then
    return 1
  fi
  upload_runtime_logs
}

log "Watching ${CHECKPOINT_DIR}"
log "Destination ${GCS_DEST}"

if [[ "${RUN_ONCE}" == "1" ]]; then
  run_upload_cycle
  exit $?
fi

while true; do
  if ! run_upload_cycle; then
    sleep "${UPLOAD_FAILURE_BACKOFF_SECONDS}"
    continue
  fi
  sleep "${POLL_SECONDS}"
done
