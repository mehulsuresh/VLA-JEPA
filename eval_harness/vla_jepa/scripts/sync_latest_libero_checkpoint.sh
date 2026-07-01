#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-a100x8-colombus.us-east5-a.yondu-general-workspace}"
RUN_NAME="${RUN_NAME:-libero_plus_qwen35_2b_full_b16_sdpa_framefix_shift2_rd8_3ep_20260629_022033}"
REMOTE_RUN="${REMOTE_RUN:-/mnt/vla-jepa/checkpoints/${RUN_NAME}}"
LOCAL_RUN="${LOCAL_RUN:-/home/mehul/work/vjepa/VLA-JEPA/checkpoints/${RUN_NAME}}"

latest_step="$(
    ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
        "find '${REMOTE_RUN}/checkpoints' -maxdepth 1 -type d -name 'steps_*' -printf '%f\n' | sort -V | tail -1"
)"
if [[ -z "${latest_step}" ]]; then
    echo "No steps_* checkpoints found under ${REMOTE_HOST}:${REMOTE_RUN}/checkpoints" >&2
    exit 1
fi

remote_step="${REMOTE_RUN}/checkpoints/${latest_step}"
local_step="${LOCAL_RUN}/checkpoints/${latest_step}"
mkdir -p "${local_step}"

ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "sudo chmod a+r '${remote_step}/model.safetensors' 2>/dev/null || true"

rsync -av "${REMOTE_HOST}:${REMOTE_RUN}/config.yaml" \
    "${REMOTE_HOST}:${REMOTE_RUN}/config.json" \
    "${REMOTE_HOST}:${REMOTE_RUN}/dataset_statistics.json" \
    "${LOCAL_RUN}/"
rsync -av --info=progress2 "${REMOTE_HOST}:${remote_step}/model.safetensors" "${local_step}/"
rsync -av "${REMOTE_HOST}:${remote_step}/trainer_state.json" "${local_step}/" 2>/dev/null || true

ckpt="${local_step}/model.safetensors"
printf '%s\n' "${ckpt}" > "${LOCAL_RUN}/latest_eval_checkpoint.txt"
echo "latest_step=${latest_step}"
echo "ckpt=${ckpt}"
