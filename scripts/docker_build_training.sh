#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-vla-jepa:py313-cu130}"
DOCKERFILE="${DOCKERFILE:-${REPO_ROOT}/docker/Dockerfile.py313}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-1}"
INSTALL_MOGE="${INSTALL_MOGE:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn}"
PLATFORM_ARGS=()

if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
  PLATFORM_ARGS+=(--platform "${DOCKER_PLATFORM}")
fi

docker build \
  "${PLATFORM_ARGS[@]}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
  --build-arg "INSTALL_DEEPSPEED=${INSTALL_DEEPSPEED}" \
  --build-arg "INSTALL_MOGE=${INSTALL_MOGE}" \
  --build-arg "INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}" \
  --build-arg "FLASH_ATTN_SPEC=${FLASH_ATTN_SPEC}" \
  "$@" \
  "${REPO_ROOT}"
