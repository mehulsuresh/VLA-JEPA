#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

scratch_root="${VLA_JEPA_SCRATCH:-${repo_root}}"
env_name="${VLA_JEPA_ENV_NAME:-vla-jepa}"
env_path="${VLA_JEPA_ENV_PATH:-${scratch_root}/envs/${env_name}}"
python_bin="${PYTHON:-python3}"
torch_index_url="${VLA_JEPA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
torch_version="${VLA_JEPA_TORCH_VERSION:-2.6.0}"
torchvision_version="${VLA_JEPA_TORCHVISION_VERSION:-0.21.0}"
install_dev="${VLA_JEPA_INSTALL_DEV:-1}"
install_flash_attn="${VLA_JEPA_INSTALL_FLASH_ATTN:-0}"
install_bitsandbytes="${VLA_JEPA_INSTALL_BITSANDBYTES:-1}"
install_decord_gpu="${VLA_JEPA_INSTALL_DECORD_GPU:-0}"
install_moge="${VLA_JEPA_INSTALL_MOGE:-1}"
decord_gpu_wheel="${VLA_JEPA_DECORD_GPU_WHEEL:-}"
flash_attn_version="${VLA_JEPA_FLASH_ATTN_VERSION:-2.8.3}"
flash_attn_archs="${VLA_JEPA_FLASH_ATTN_CUDA_ARCHS:-}"
flash_attn_max_jobs="${VLA_JEPA_FLASH_ATTN_MAX_JOBS:-}"
flash_attn_nvcc_threads="${VLA_JEPA_FLASH_ATTN_NVCC_THREADS:-1}"
flash_attn_wheel="${VLA_JEPA_FLASH_ATTN_WHEEL:-}"
moge_repo_url="${VLA_JEPA_MOGE_REPO_URL:-https://github.com/microsoft/MoGe.git}"
moge_repo_path="${VLA_JEPA_MOGE_REPO_PATH:-${scratch_root}/src/MoGe}"
moge_repo_ref="${VLA_JEPA_MOGE_REF:-07444410f1e33f402353b99d6ccd26bd31e469e8}"
utils3d_spec="${VLA_JEPA_UTILS3D_SPEC:-utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183}"

cache_dir="${VLA_JEPA_CACHE_DIR:-${scratch_root}/cache/pip}"
tmp_dir="${VLA_JEPA_TMPDIR:-${scratch_root}/tmp}"
hf_home="${HF_HOME:-${scratch_root}/hf}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    echo "Install system prerequisites from docs/repo_environment_setup.md, then rerun." >&2
    exit 1
  fi
}

require_cmd gcc
require_cmd mpicc
require_cmd mpirun
require_cmd ninja
if [[ "${install_moge}" == "1" ]]; then
  require_cmd git
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda-12.4 /usr/local/cuda; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Missing CUDA toolkit nvcc. Expected CUDA_HOME/bin/nvcc." >&2
  echo "For CUDA 12.4 on Debian/Ubuntu, install cuda-nvcc-12-4 and cuda-cudart-dev-12-4." >&2
  exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"

python_include="$("${python_bin}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["include"])
PY
)"
if [[ ! -f "${python_include}/Python.h" ]]; then
  echo "Missing Python headers at ${python_include}/Python.h" >&2
  echo "Install python3-dev or the matching pythonX.Y-dev package, then rerun." >&2
  exit 1
fi

if [[ "${VLA_JEPA_RECREATE_ENV:-0}" == "1" ]]; then
  rm -rf "${env_path}"
fi

mkdir -p "${cache_dir}" "${tmp_dir}" "${hf_home}" "${scratch_root}/checkpoints" "${scratch_root}/datasets"

if [[ ! -x "${env_path}/bin/python" ]]; then
  "${python_bin}" -m venv "${env_path}"
fi

if [[ -e "${repo_root}/.venv" && ! -L "${repo_root}/.venv" ]]; then
  echo "Refusing to replace existing non-symlink ${repo_root}/.venv" >&2
  echo "Move it aside or rerun with VLA_JEPA_ENV_PATH pointing at that environment." >&2
  exit 1
fi
ln -sfn "${env_path}" "${repo_root}/.venv"

export PIP_CACHE_DIR="${cache_dir}"
export TMPDIR="${tmp_dir}"
export HF_HOME="${hf_home}"

"${repo_root}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"${repo_root}/.venv/bin/python" -m pip install \
  --index-url "${torch_index_url}" \
  "torch==${torch_version}" \
  "torchvision==${torchvision_version}"
"${repo_root}/.venv/bin/python" -m pip install -r "${repo_root}/requirements.txt"
editable_spec="${repo_root}"
if [[ "${install_dev}" == "1" ]]; then
  editable_spec="${repo_root}[dev]"
fi
"${repo_root}/.venv/bin/python" -m pip install -e "${editable_spec}"

if [[ "${install_decord_gpu}" == "1" ]]; then
  if [[ -n "${decord_gpu_wheel}" ]]; then
    "${repo_root}/.venv/bin/python" -m pip install --force-reinstall --no-deps "${decord_gpu_wheel}"
  else
    VLA_JEPA_SCRATCH="${scratch_root}" \
    VLA_JEPA_ENV_PYTHON="${repo_root}/.venv/bin/python" \
    "${repo_root}/scripts/build_decord_gpu.sh"
  fi
elif [[ -n "${decord_gpu_wheel}" ]]; then
  "${repo_root}/.venv/bin/python" -m pip install --force-reinstall --no-deps "${decord_gpu_wheel}"
fi

if [[ "${install_flash_attn}" == "1" ]]; then
  if [[ -n "${flash_attn_wheel}" ]]; then
    "${repo_root}/.venv/bin/python" -m pip install --force-reinstall --no-deps "${flash_attn_wheel}"
  else
    if [[ -n "${flash_attn_archs}" ]]; then
      export FLASH_ATTN_CUDA_ARCHS="${flash_attn_archs}"
    fi
    if [[ -n "${flash_attn_max_jobs}" ]]; then
      export MAX_JOBS="${flash_attn_max_jobs}"
    fi
    export NVCC_THREADS="${flash_attn_nvcc_threads}"
    export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"
    "${repo_root}/.venv/bin/python" -m pip install \
      --force-reinstall \
      --no-deps \
      "flash-attn==${flash_attn_version}" \
      --no-build-isolation \
      --no-cache-dir
  fi
fi

if [[ "${install_bitsandbytes}" == "1" ]]; then
  "${repo_root}/.venv/bin/python" -m pip install bitsandbytes
fi

if [[ "${install_moge}" == "1" ]]; then
  mkdir -p "$(dirname "${moge_repo_path}")"
  if [[ ! -d "${moge_repo_path}/.git" ]]; then
    git clone "${moge_repo_url}" "${moge_repo_path}"
  fi
  git -C "${moge_repo_path}" fetch --quiet origin "${moge_repo_ref}"
  git -C "${moge_repo_path}" checkout --quiet "${moge_repo_ref}"
  "${repo_root}/.venv/bin/python" -m pip install --no-deps "${utils3d_spec}"
  "${repo_root}/.venv/bin/python" -m pip install --no-deps -e "${moge_repo_path}"
fi

verify_modules=(
  accelerate
  deepspeed
  decord
  mpi4py
  omegaconf
  peft
  qwen_vl_utils
  starVLA
  torchvision
  transformers
)
if [[ "${install_flash_attn}" == "1" ]]; then
  verify_modules+=(flash_attn)
fi
if [[ "${install_bitsandbytes}" == "1" ]]; then
  verify_modules+=(bitsandbytes)
fi
if [[ "${install_moge}" == "1" ]]; then
  verify_modules+=(moge utils3d)
fi

VLA_JEPA_VERIFY_MODULES="${verify_modules[*]}" "${repo_root}/.venv/bin/python" - <<'PY'
import importlib
import os

import torch

checks = os.environ["VLA_JEPA_VERIFY_MODULES"].split()

print(f"python ok")
print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu_count={torch.cuda.device_count()} first_gpu={torch.cuda.get_device_name(0)}")

missing = []
for name in checks:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if missing:
    raise SystemExit("import checks failed:\n" + "\n".join(missing))

print("import checks ok")
PY

cat <<EOF

VLA-JEPA environment ready.

Activate:
  cd ${repo_root}
  source .venv/bin/activate

Recommended runtime exports:
  export HF_HOME=${hf_home}
  export PIP_CACHE_DIR=${cache_dir}
  export TMPDIR=${tmp_dir}
  export CUDA_HOME=${CUDA_HOME}
  export PATH=${CUDA_HOME}/bin:\$PATH
  export WANDB_MODE=\${WANDB_MODE:-disabled}
  export STARVLA_MOGE_REPO_PATH=${moge_repo_path}
EOF
