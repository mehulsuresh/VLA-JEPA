#!/usr/bin/env bash
# Launch TensorBoard for VLA-JEPA training runs.
#
# On the REMOTE machine (this server), run:
#   bash scripts/tb.sh
#
# On YOUR LOCAL machine, open a second terminal and run:
#   ssh -N -L 6006:localhost:6006 mehul_yonduai_com@<server-ip-or-hostname>
#
# Then open http://localhost:6006 in your browser.
#
# Optional env-var overrides:
#   LOGDIR   - path to watch (default: checkpoints/ relative to repo root)
#   TB_PORT  - port to bind on the server (default: 6006)
#   TB_HOST  - interface to bind (default: 0.0.0.0 so the SSH tunnel works)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOGDIR="${LOGDIR:-${REPO_ROOT}/checkpoints}"
TB_PORT="${TB_PORT:-6006}"
TB_HOST="${TB_HOST:-0.0.0.0}"

TB_BIN="${TB_BIN:-$(command -v tensorboard 2>/dev/null || echo "${HOME}/miniconda3/envs/vla-jepa-vjepa21/bin/tensorboard")}"

if [[ ! -x "${TB_BIN}" ]]; then
    echo "ERROR: tensorboard not found at '${TB_BIN}'."
    echo "  Install it with: pip install tensorboard"
    echo "  Or set TB_BIN=/path/to/tensorboard"
    exit 1
fi

echo "==================================================="
echo "  TensorBoard"
echo "  logdir : ${LOGDIR}"
echo "  server : ${TB_HOST}:${TB_PORT}"
echo ""
echo "  SSH tunnel (run this on YOUR local machine):"
echo "    ssh -N -L ${TB_PORT}:localhost:${TB_PORT} \\"
echo "        $(whoami)@$(hostname -I | awk '{print $1}')"
echo ""
echo "  Then open http://localhost:${TB_PORT}"
echo "==================================================="

"${TB_BIN}" \
    --logdir "${LOGDIR}" \
    --host "${TB_HOST}" \
    --port "${TB_PORT}" \
    --reload_interval 10 \
    --reload_multifile true
