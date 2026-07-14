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
FLASH_ATTN_CUDA_ARCH_LIST="${FLASH_ATTN_CUDA_ARCH_LIST:-8.0}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-32}"
FLASH_ATTN_NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS:-2}"
INSTALL_FAST_LINEAR_ATTN="${INSTALL_FAST_LINEAR_ATTN:-0}"
FAST_LINEAR_ATTN_SPEC="${FAST_LINEAR_ATTN_SPEC:-flash-linear-attention[cuda]==0.5.1}"
CAUSAL_CONV1D_SPEC="${CAUSAL_CONV1D_SPEC:-causal-conv1d==1.6.2.post1}"
FAST_LINEAR_ATTN_TRANSFORMERS_SPEC="${FAST_LINEAR_ATTN_TRANSFORMERS_SPEC:-transformers==5.13.1}"
FAST_LINEAR_ATTN_CUDA_ARCH_LIST="${FAST_LINEAR_ATTN_CUDA_ARCH_LIST:-8.0}"
FAST_LINEAR_ATTN_MAX_JOBS="${FAST_LINEAR_ATTN_MAX_JOBS:-32}"
PLATFORM_ARGS=()

require_boolean_build_arg() {
  local name="$1"
  local value="${!name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${name} must be 0 or 1, got: ${value}" >&2
    exit 2
  fi
}

require_boolean_build_arg INSTALL_FAST_LINEAR_ATTN
if [[ "${INSTALL_FAST_LINEAR_ATTN}" == "1" ]]; then
  if [[ ! "${FAST_LINEAR_ATTN_SPEC}" =~ ^flash-linear-attention(\[cuda\])?==[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    echo "FAST_LINEAR_ATTN_SPEC must be an exact flash-linear-attention[CUDA] version pin" >&2
    exit 2
  fi
  if [[ ! "${CAUSAL_CONV1D_SPEC}" =~ ^causal-conv1d==[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    echo "CAUSAL_CONV1D_SPEC must be an exact causal-conv1d version pin" >&2
    exit 2
  fi
  if [[ ! "${FAST_LINEAR_ATTN_TRANSFORMERS_SPEC}" =~ ^transformers==[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    echo "FAST_LINEAR_ATTN_TRANSFORMERS_SPEC must be an exact transformers version pin" >&2
    exit 2
  fi
  if [[ ! "${FAST_LINEAR_ATTN_CUDA_ARCH_LIST}" =~ ^[0-9]+\.[0-9]+([[:space:];]+[0-9]+\.[0-9]+)*$ ]]; then
    echo "FAST_LINEAR_ATTN_CUDA_ARCH_LIST must contain explicit numeric CUDA architectures" >&2
    exit 2
  fi
  if [[ ! "${FAST_LINEAR_ATTN_MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FAST_LINEAR_ATTN_MAX_JOBS must be a positive integer" >&2
    exit 2
  fi
fi

if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
  PLATFORM_ARGS+=(--platform "${DOCKER_PLATFORM}")
fi

docker build \
  "${PLATFORM_ARGS[@]}" \
  "$@" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
  --build-arg "INSTALL_DEEPSPEED=${INSTALL_DEEPSPEED}" \
  --build-arg "INSTALL_MOGE=${INSTALL_MOGE}" \
  --build-arg "INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}" \
  --build-arg "FLASH_ATTN_SPEC=${FLASH_ATTN_SPEC}" \
  --build-arg "FLASH_ATTN_CUDA_ARCH_LIST=${FLASH_ATTN_CUDA_ARCH_LIST}" \
  --build-arg "FLASH_ATTN_MAX_JOBS=${FLASH_ATTN_MAX_JOBS}" \
  --build-arg "FLASH_ATTN_NVCC_THREADS=${FLASH_ATTN_NVCC_THREADS}" \
  --build-arg "INSTALL_FAST_LINEAR_ATTN=${INSTALL_FAST_LINEAR_ATTN}" \
  --build-arg "FAST_LINEAR_ATTN_SPEC=${FAST_LINEAR_ATTN_SPEC}" \
  --build-arg "CAUSAL_CONV1D_SPEC=${CAUSAL_CONV1D_SPEC}" \
  --build-arg "FAST_LINEAR_ATTN_TRANSFORMERS_SPEC=${FAST_LINEAR_ATTN_TRANSFORMERS_SPEC}" \
  --build-arg "FAST_LINEAR_ATTN_CUDA_ARCH_LIST=${FAST_LINEAR_ATTN_CUDA_ARCH_LIST}" \
  --build-arg "FAST_LINEAR_ATTN_MAX_JOBS=${FAST_LINEAR_ATTN_MAX_JOBS}" \
  "${REPO_ROOT}"
