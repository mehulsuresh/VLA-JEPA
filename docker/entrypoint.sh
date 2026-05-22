#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TMPDIR="${TMPDIR:-/tmp}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

mkdir -p "${TMPDIR}" "${HF_HOME}"

if [[ -n "${STARVLA_MOGE_REPO_PATH:-}" ]]; then
  export PYTHONPATH="${STARVLA_MOGE_REPO_PATH}:${PYTHONPATH:-}"
fi

if [[ "${STARVLA_SANITIZE_LD_LIBRARY_PATH:-1}" == "1" ]]; then
  sanitized_ld_library_path="$(python - <<'PY'
import os

entries = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
filtered = [entry for entry in entries if entry != "/usr/local/gib/lib64"]
print(":".join(filtered))
PY
)"
  export LD_LIBRARY_PATH="${sanitized_ld_library_path}"
fi

if [[ "${STARVLA_RUN_PREFLIGHT:-0}" == "1" ]]; then
  python /workspace/VLA-JEPA/scripts/preflight_runtime.py --require-cuda ${STARVLA_PREFLIGHT_EXTRA_ARGS:-}
fi

exec "$@"
