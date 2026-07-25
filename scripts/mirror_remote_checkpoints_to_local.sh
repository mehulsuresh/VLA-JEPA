#!/usr/bin/env bash
set -euo pipefail

# Pull completed checkpoints from a remote training host to durable local storage.
# The remote trainer is never modified; incomplete checkpoints are not copied.
REMOTE_HOST="${REMOTE_HOST:-reward-model-small}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/mnt/vla-jepa/checkpoints/robot_ft_lerobot_magna_interventions_h100x8_b16_rtc0_full_20260715_040438}"
REMOTE_CHECKPOINT_DIR="${REMOTE_CHECKPOINT_DIR:-${REMOTE_RUN_DIR}/checkpoints}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/mnt/vla-jepa/logs}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/data/reward_model_small/checkpoint_mirrors}"
POLL_SECONDS="${POLL_SECONDS:-120}"
STABLE_SECONDS="${STABLE_SECONDS:-180}"
RUN_ONCE="${RUN_ONCE:-0}"

mkdir -p "$LOCAL_ROOT" "$LOCAL_ROOT/.state" "$LOCAL_ROOT/.logs"
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOCAL_ROOT/.logs/mirror.log"; }

rsync_opts=(-a --partial --info=progress2 --human-readable --exclude='*.tmp' --exclude='.upload_state/')

verify_checkpoint() {
  local name="$1" remote_dir="$2" local_dir="$3"
  local remote_manifest local_manifest
  remote_manifest="$(mktemp)"
  local_manifest="$(mktemp)"
  if ! ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" \
    "cd '${remote_dir%/}' && find . -type f -print0 | sort -z | xargs -0 sha256sum" \
    >"$remote_manifest"; then
    rm -f "$remote_manifest" "$local_manifest"
    return 1
  fi
  (
    cd "${local_dir%/}"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
  ) >"$local_manifest"
  if ! cmp -s "$remote_manifest" "$local_manifest"; then
    log "Checksum mismatch for $name; leaving it unverified for retry"
    rm -f "$remote_manifest" "$local_manifest"
    return 1
  fi
  mv "$local_manifest" "$LOCAL_ROOT/.state/$name.sha256"
  rm -f "$remote_manifest"
  return 0
}

sync_once() {
  local listing name remote_ckpt local_ckpt newest now

  log "Syncing run metadata, evaluations, TensorBoard, and launch logs"
  mkdir -p "$LOCAL_ROOT/run" "$LOCAL_ROOT/logs"
  rsync -a --partial --exclude='checkpoints/' --exclude='.upload_state/' \
    "$REMOTE_HOST:$REMOTE_RUN_DIR/" "$LOCAL_ROOT/run/" || true
  local run_id
  run_id="$(basename "$REMOTE_RUN_DIR")"
  rsync -a --partial --include="${run_id}*" --exclude='*' \
    "$REMOTE_HOST:$REMOTE_LOG_DIR/" "$LOCAL_ROOT/logs/" || true

  listing="$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" \
    "find '$REMOTE_CHECKPOINT_DIR' -mindepth 1 -maxdepth 1 -type d -name 'steps_*' -printf '%f\\n' 2>/dev/null | sort -Vr" || true)"
  while IFS= read -r name; do
    [[ "$name" =~ ^steps_[0-9]+$ ]] || continue
    if [[ -f "$LOCAL_ROOT/.state/$name.complete" \
       && -f "$LOCAL_ROOT/.state/$name.sha256" ]]; then
      continue
    fi
    remote_ckpt="$REMOTE_CHECKPOINT_DIR/$name/"
    local_ckpt="$LOCAL_ROOT/checkpoints/$name/"
    mkdir -p "$local_ckpt"
    newest="$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" \
      "find '$remote_ckpt' -type f -printf '%T@\\n' 2>/dev/null | sort -nr | head -1" || true)"
    [[ -n "$newest" ]] || continue
    now="$(date +%s)"
    if ! awk -v n="$newest" -v now="$now" -v stable="$STABLE_SECONDS" 'BEGIN { exit ((now-n)>=stable ? 0 : 1) }'; then
      log "Skipping $name (still changing)"
      continue
    fi
    log "Syncing stable $name"
    rsync "${rsync_opts[@]}" "$REMOTE_HOST:$remote_ckpt" "$local_ckpt" </dev/null
    log "Verifying $name checksums"
    if verify_checkpoint "$name" "$remote_ckpt" "$local_ckpt"; then
      printf '%s\n' "$(date -Is)" > "$LOCAL_ROOT/.state/$name.complete"
      log "Verified complete $name"
      # Rescan after each completed transfer so a newly finalized newest
      # checkpoint always takes priority over older backlog entries.
      break
    fi
  done <<< "$listing"

}

while :; do
  sync_once || log "Mirror pass failed; will retry"
  [[ "$RUN_ONCE" == 1 ]] && break
  sleep "$POLL_SECONDS"
done
