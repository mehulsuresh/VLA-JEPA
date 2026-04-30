#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

detect_default_ifname() {
  ip route show default 2>/dev/null | awk 'NR==1 {print $5}'
}

sanitize_ld_library_path() {
  python3 - <<'PY'
import os

entries = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
filtered = [entry for entry in entries if entry != "/usr/local/gib/lib64"]
print(":".join(filtered))
PY
}

cleanup_stale_training_sidecars() {
  python3 - <<'PY'
import os
import signal
import subprocess
import time

ENV_PYTHON = os.path.expanduser("~/miniconda3/envs/vla-jepa-vjepa21/bin/python")
patterns = ("from multiprocessing.spawn import spawn_main", "from multiprocessing.resource_tracker import main")

def list_stale_pids():
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,cmd="],
        text=True,
    )
    stale = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_s, ppid_s, cmd = parts
        try:
            pid = int(pid_s)
            ppid = int(ppid_s)
        except ValueError:
            continue
        if ppid != 1:
            continue
        if ENV_PYTHON not in cmd:
            continue
        if not any(pattern in cmd for pattern in patterns):
            continue
        stale.append(pid)
    return stale

stale = list_stale_pids()
if not stale:
    raise SystemExit(0)

print(f"Cleaning up stale multiprocessing sidecars: {stale}", flush=True)
for pid in stale:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

deadline = time.time() + 5.0
while time.time() < deadline:
    remaining = [pid for pid in stale if os.path.exists(f"/proc/{pid}")]
    if not remaining:
        raise SystemExit(0)
    time.sleep(0.2)

for pid in stale:
    if os.path.exists(f"/proc/{pid}"):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY
}

DEFAULT_SOCKET_IFNAME="${DEFAULT_SOCKET_IFNAME:-$(detect_default_ifname)}"
DEFAULT_SOCKET_IFNAME="${DEFAULT_SOCKET_IFNAME:-ens8}"
DEFAULT_SOCKET_IFNAME="${DEFAULT_SOCKET_IFNAME:-lo}"

unset NCCL_NET NCCL_TUNER_CONFIG_PATH NCCL_IB_ADAPTIVE_ROUTING NCCL_IB_FIFO_TC
unset NCCL_IB_QPS_PER_CONNECTION NCCL_IB_TC NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC
unset NCCL_NVLS_CHUNKSIZE NCCL_P2P_NET_CHUNKSIZE
export LD_LIBRARY_PATH="$(sanitize_ld_library_path)"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${DEFAULT_SOCKET_IFNAME}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"
export TMPDIR="${TMPDIR:-${HOME}/tmp}"
export FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export STARVLA_USE_DEEPSPEED="${STARVLA_USE_DEEPSPEED:-1}"

mkdir -p "${TMPDIR}"
cleanup_stale_training_sidecars

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || echo "${HOME}/miniconda3/envs/vla-jepa-vjepa21/bin/accelerate")}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_a100x4_weekend_20260404.yaml}"
DEEPSPEED_STAGE="${STARVLA_DEEPSPEED_STAGE:-2}"
DEFAULT_ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml"
EXTRA_TRAIN_ARGS=()
if [[ "${DEEPSPEED_STAGE}" == "3" || "${DEEPSPEED_STAGE,,}" == "zero3" ]]; then
  DEFAULT_ACCELERATE_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero3.yaml"
  EXTRA_TRAIN_ARGS+=(--framework.qwenvl.device_map null)
fi
if [[ "${STARVLA_USE_DEEPSPEED}" == "1" ]]; then
  if [[ "${STARVLA_ALLOW_COMPILE_WITH_DEEPSPEED:-0}" == "1" ]]; then
    EXTRA_TRAIN_ARGS+=(--trainer.allow_compile_with_deepspeed true)
  else
    EXTRA_TRAIN_ARGS+=(
      --trainer.compile_qwen_model false
      --trainer.compile_action_model false
      --trainer.compile_vj_predictor false
      --trainer.compile_vj_encoder false
      --trainer.compile_full_model false
    )
  fi
fi
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DEFAULT_ACCELERATE_CONFIG}}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

cd "${REPO_ROOT}"

"${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "$@"
