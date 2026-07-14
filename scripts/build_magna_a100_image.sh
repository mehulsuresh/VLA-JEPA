#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-vla-jepa:py313-cu130-a100}"
LOG_ROOT="${LOG_ROOT:-/mnt/vla-jepa/logs}"
BUILD_LOG="${BUILD_LOG:-${LOG_ROOT}/docker_build_magna_a100.log}"
BUILD_EXIT_FILE="${BUILD_EXIT_FILE:-${LOG_ROOT}/docker_build_magna_a100.exit}"
IMAGE_IDENTITY_FILE="${IMAGE_IDENTITY_FILE:-${LOG_ROOT}/magna_image_identity.txt}"

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

record_exit() {
  local status="$?"
  trap - EXIT
  printf '%s\n' "${status}" > "${BUILD_EXIT_FILE}" || true
  exit "${status}"
}
rm -f "${BUILD_EXIT_FILE}"
trap record_exit EXIT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to build the production image from a dirty worktree" >&2
  exit 2
fi

SOURCE_COMMIT="$(git rev-parse HEAD)"

set +e
env \
  IMAGE="${IMAGE}" \
  BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}" \
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}" \
  INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-0}" \
  INSTALL_MOGE="${INSTALL_MOGE:-1}" \
  INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}" \
  FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn==2.8.3.post1}" \
  FLASH_ATTN_CUDA_ARCH_LIST="${FLASH_ATTN_CUDA_ARCH_LIST:-8.0}" \
  FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-64}" \
  FLASH_ATTN_NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS:-1}" \
  ./scripts/docker_build_training.sh "$@" 2>&1 | tee "${BUILD_LOG}"
pipeline_status=("${PIPESTATUS[@]}")
set -e

status="${pipeline_status[0]}"
if [[ "${status}" -eq 0 && "${pipeline_status[1]}" -ne 0 ]]; then
  status="${pipeline_status[1]}"
fi
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

{
  printf 'source_commit=%s\n' "${SOURCE_COMMIT}"
  docker image inspect "${IMAGE}" --format 'image={{.Id}} created={{.Created}}'
} > "${IMAGE_IDENTITY_FILE}"
