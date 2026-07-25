#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

env_name="${VLA_JEPA_ENV_NAME:-vla-jepa-py313-min}"
python_version="${VLA_JEPA_PYTHON_VERSION:-3.13}"
conda_bin="${CONDA_EXE:-$(command -v conda)}"
requirements_file="${VLA_JEPA_REQUIREMENTS_FILE:-${repo_root}/requirements-py313-min.txt}"
install_flash_attn4="${VLA_JEPA_INSTALL_FLASH_ATTN4:-0}"
install_deepspeed="${VLA_JEPA_INSTALL_DEEPSPEED:-0}"
install_wandb="${VLA_JEPA_INSTALL_WANDB:-0}"
install_torchcodec="${VLA_JEPA_INSTALL_TORCHCODEC:-0}"
install_decord="${VLA_JEPA_INSTALL_DECORD:-0}"
decord_wheel="${VLA_JEPA_DECORD_WHEEL:-}"

if [[ -z "${conda_bin}" ]]; then
  echo "Missing conda. Install Miniconda/Mambaforge or create the Python ${python_version} env manually." >&2
  exit 1
fi

if ! "${conda_bin}" env list | awk '{print $1}' | grep -qx "${env_name}"; then
  "${conda_bin}" create -n "${env_name}" "python=${python_version}" -y
fi

run_in_env() {
  "${conda_bin}" run -n "${env_name}" bash -lc "set -euo pipefail; export PYTHONNOUSERSITE=1; $*"
}

run_in_env "python -m pip install --upgrade pip 'setuptools<82' wheel"
run_in_env "python -m pip install --upgrade torch torchvision"
run_in_env "python -m pip install --upgrade -r '${requirements_file}'"
run_in_env "python -m pip install -e '${repo_root}'"

if [[ "${install_flash_attn4}" == "1" ]]; then
  run_in_env "python -m pip install --upgrade --pre \
    https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta4/flash_attn_4-4.0.0b4-py3-none-any.whl"
fi

if [[ "${install_deepspeed}" == "1" ]]; then
  run_in_env "python -m pip install --upgrade deepspeed mpi4py"
fi

if [[ "${install_wandb}" == "1" ]]; then
  run_in_env "python -m pip install --upgrade wandb"
fi

if [[ "${install_torchcodec}" == "1" ]]; then
  run_in_env "python -m pip install --upgrade torchcodec"
fi

if [[ "${install_decord}" == "1" ]]; then
  if [[ -n "${decord_wheel}" ]]; then
    run_in_env "python -m pip install --force-reinstall --no-deps '${decord_wheel}'"
  else
    run_in_env "python -m pip install --upgrade decord"
  fi
fi

run_in_env "python - <<'PY'
import importlib

import torch
import torchvision

checks = [
    'accelerate',
    'albumentations',
    'av',
    'diffusers',
    'gcsfs',
    'omegaconf',
    'peft',
    'pyarrow',
    'qwen_vl_utils',
    'starVLA',
    'timm',
    'transformers',
]

print(f'python ok')
print(f'torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}')
print(f'torchvision={torchvision.__version__}')
if torch.cuda.is_available():
    print(f'gpu_count={torch.cuda.device_count()} first_gpu={torch.cuda.get_device_name(0)}')

missing = []
for name in checks:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f'{name}: {exc}')

if missing:
    raise SystemExit('import checks failed:\\n' + '\\n'.join(missing))
print('import checks ok')
PY"

cat <<EOF

Python ${python_version} minimal VLA-JEPA env ready: ${env_name}

Activate:
  conda activate ${env_name}
  export PYTHONNOUSERSITE=1

Smoke train:
  ACCELERATE_BIN=\$(which accelerate) \\
  RUN_ID=py313_min_smoke_\$(date +%Y%m%d_%H%M%S) \\
  ./scripts/vlajepa_robot_ft_canonical_multi_subset_5090_smoke.sh \\
    --trainer.max_train_steps 1 \\
    --datasets.vla_data.video_decode_backend pyav \\
    --datasets.vla_data.pyav_thread_count 1 \\
    --datasets.vla_data.num_workers 0

Optional extras:
  VLA_JEPA_INSTALL_DEEPSPEED=1 ./scripts/setup_py313_min_env.sh
  VLA_JEPA_INSTALL_FLASH_ATTN4=1 ./scripts/setup_py313_min_env.sh
  VLA_JEPA_INSTALL_WANDB=1 ./scripts/setup_py313_min_env.sh
  VLA_JEPA_INSTALL_TORCHCODEC=1 ./scripts/setup_py313_min_env.sh
  VLA_JEPA_INSTALL_DECORD=1 VLA_JEPA_DECORD_WHEEL=/path/to/decord.whl ./scripts/setup_py313_min_env.sh
EOF
