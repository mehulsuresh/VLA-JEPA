#!/usr/bin/env bash
# Shared helpers for training launchers. This file is sourced by scripts/*.sh.

starvla_detect_default_ifname() {
  if command -v ip >/dev/null 2>&1; then
    ip route show default 2>/dev/null | awk 'NR==1 {print $5}'
    return
  fi
  for candidate in ens9 ens8 eth0 enp0s9; do
    if [[ -d "/sys/class/net/${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
}

starvla_sanitize_ld_library_path() {
  python3 - <<'PY'
import os

entries = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
filtered = [entry for entry in entries if entry != "/usr/local/gib/lib64"]
print(":".join(filtered))
PY
}

starvla_configure_common_training_env() {
  local fallback_ifname="${1:-lo}"
  local default_socket_ifname="${DEFAULT_SOCKET_IFNAME:-$(starvla_detect_default_ifname)}"
  default_socket_ifname="${default_socket_ifname:-${fallback_ifname}}"
  default_socket_ifname="${default_socket_ifname:-lo}"

  unset NCCL_NET NCCL_TUNER_CONFIG_PATH NCCL_IB_ADAPTIVE_ROUTING NCCL_IB_FIFO_TC
  unset NCCL_IB_QPS_PER_CONNECTION NCCL_IB_TC NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC
  unset NCCL_NVLS_CHUNKSIZE NCCL_P2P_NET_CHUNKSIZE
  export LD_LIBRARY_PATH="$(starvla_sanitize_ld_library_path)"

  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${default_socket_ifname}}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"
  export TMPDIR="${TMPDIR:-${HOME}/tmp}"
  export FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"

  mkdir -p "${TMPDIR}"
}

starvla_configure_deepspeed_launch() {
  local repo_root="$1"
  local deepspeed_stage="${STARVLA_DEEPSPEED_STAGE:-2}"

  export STARVLA_USE_DEEPSPEED="${STARVLA_USE_DEEPSPEED:-1}"
  STARVLA_DEFAULT_ACCELERATE_CONFIG="${repo_root}/starVLA/config/deepseeds/deepspeed_zero2.yaml"
  STARVLA_EXTRA_TRAIN_ARGS=()

  if [[ "${deepspeed_stage}" == "3" || "${deepspeed_stage,,}" == "zero3" ]]; then
    STARVLA_DEFAULT_ACCELERATE_CONFIG="${repo_root}/starVLA/config/deepseeds/deepspeed_zero3.yaml"
    STARVLA_EXTRA_TRAIN_ARGS+=(--framework.qwenvl.device_map null)
  fi
  if [[ "${STARVLA_USE_DEEPSPEED}" == "1" ]]; then
    if [[ "${STARVLA_ALLOW_COMPILE_WITH_DEEPSPEED:-0}" == "1" ]]; then
      STARVLA_EXTRA_TRAIN_ARGS+=(--trainer.allow_compile_with_deepspeed true)
    else
      STARVLA_EXTRA_TRAIN_ARGS+=(
        --trainer.compile_qwen_model false
        --trainer.compile_action_model false
        --trainer.compile_vj_predictor false
        --trainer.compile_vj_encoder false
        --trainer.compile_full_model false
      )
    fi
  fi
}

starvla_configure_accelerate_cluster_args() {
  STARVLA_ACCELERATE_CLUSTER_ARGS=(
    --num_machines "${NUM_MACHINES:-1}"
  )

  if [[ -n "${MACHINE_RANK:-}" ]]; then
    STARVLA_ACCELERATE_CLUSTER_ARGS+=(--machine_rank "${MACHINE_RANK}")
  fi
  if [[ -n "${MAIN_PROCESS_IP:-}" ]]; then
    STARVLA_ACCELERATE_CLUSTER_ARGS+=(--main_process_ip "${MAIN_PROCESS_IP}")
  fi
  if [[ -n "${RDZV_BACKEND:-}" ]]; then
    STARVLA_ACCELERATE_CLUSTER_ARGS+=(--rdzv_backend "${RDZV_BACKEND}")
  fi
  if [[ -n "${RDZV_CONF:-}" ]]; then
    STARVLA_ACCELERATE_CLUSTER_ARGS+=(--rdzv_conf "${RDZV_CONF}")
  fi
  if [[ "${SAME_NETWORK:-0}" == "1" ]]; then
    STARVLA_ACCELERATE_CLUSTER_ARGS+=(--same_network)
  fi
}

starvla_cleanup_stale_training_sidecars() {
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
