#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-reward-model-small}"
IMAGE="${IMAGE:-vla-jepa:py313-cu130-h100-42262ac-fla}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:7f39b82e06e2c99026b435009b6d38b701782a90f1893f51dc70d85a0b049bf4}"
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/data/reward_model_small/checkpoint_mirrors}"
ARCHIVE="$LOCAL_ROOT/runtime/vla-jepa_py313-cu130-h100-42262ac-fla.tar.zst"
ZSTD="${ZSTD:-/home/mehul/miniconda3/bin/zstd}"

mkdir -p "$LOCAL_ROOT/runtime"
while [[ ! -f "$LOCAL_ROOT/.state/steps_28125.complete" \
      || ! -f "$LOCAL_ROOT/.state/steps_37500.complete" ]]; do
  sleep 60
done

remote_id="$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" \
  "docker image inspect '$IMAGE' --format '{{.Id}}'")"
if [[ "$remote_id" != "$EXPECTED_IMAGE_ID" ]]; then
  echo "Refusing runtime backup: image id $remote_id != $EXPECTED_IMAGE_ID" >&2
  exit 1
fi

if [[ ! -f "$ARCHIVE" ]]; then
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" \
    "docker save '$IMAGE'" | "$ZSTD" -T0 -3 -f -o "$ARCHIVE.partial"
  mv "$ARCHIVE.partial" "$ARCHIVE"
fi
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
