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
TARGET_SHELL_PID = int(os.environ.get("STARVLA_CLEANUP_SHELL_PID", "0") or 0)
patterns = ("from multiprocessing.spawn import spawn_main", "from multiprocessing.resource_tracker import main")

def list_stale_pids():
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,cmd="],
        text=True,
    )
    proc_table = {}
    rows = []
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
        proc_table[pid] = ppid
        rows.append((pid, ppid, cmd))

    def has_ancestor(pid: int, ancestor_pid: int) -> bool:
        if ancestor_pid <= 0:
            return False
        seen = set()
        current = pid
        while current > 0 and current not in seen:
            seen.add(current)
            current = proc_table.get(current, 0)
            if current == ancestor_pid:
                return True
        return False

    stale = []
    for pid, ppid, cmd in rows:
        if ENV_PYTHON not in cmd:
            continue
        if not any(pattern in cmd for pattern in patterns):
            continue
        if ppid != 1 and not has_ancestor(pid, TARGET_SHELL_PID):
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
DEFAULT_SOCKET_IFNAME="${DEFAULT_SOCKET_IFNAME:-eno1}"
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
export STARVLA_CLEANUP_SHELL_PID="$$"

mkdir -p "${TMPDIR}"
cleanup_stale_training_sidecars
trap cleanup_stale_training_sidecars EXIT

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || echo "${HOME}/miniconda3/envs/vla-jepa-vjepa21/bin/accelerate")}"
CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_5090_lerobot.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}"
RUN_ID="${RUN_ID:-robot_ft_trossen_vjepa21_small_5090_lerobot_$(date +%Y%m%d_%H%M%S)}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-AdamW}"
SAVE_BEST_ONLY="${SAVE_BEST_ONLY:-false}"
RUNTIME_TIMING_LOGGING="${RUNTIME_TIMING_LOGGING:-true}"
GPU_VIDEO_DECODE_ON_RANK="${GPU_VIDEO_DECODE_ON_RANK:-false}"
CPU_VIDEO_DECODE_DROP_WORKER_IMAGES="${CPU_VIDEO_DECODE_DROP_WORKER_IMAGES:-true}"
EXTRA_TRAIN_ARGS=(
  --trainer.save_best_only "${SAVE_BEST_ONLY}"
  --trainer.optimizer.name "${OPTIMIZER_NAME}"
  --datasets.vla_data.runtime_timing_logging "${RUNTIME_TIMING_LOGGING}"
  --datasets.vla_data.gpu_video_decode_on_rank "${GPU_VIDEO_DECODE_ON_RANK}"
  --datasets.vla_data.cpu_video_decode_drop_worker_images "${CPU_VIDEO_DECODE_DROP_WORKER_IMAGES}"
)
if [[ "${OPTIMIZER_NAME}" == "AdamW" ]]; then
  EXTRA_TRAIN_ARGS+=(--trainer.optimizer.fused true)
fi

cd "${REPO_ROOT}"

"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --mixed_precision no \
  --dynamo_backend no \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${RUN_ID}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  "$@"
