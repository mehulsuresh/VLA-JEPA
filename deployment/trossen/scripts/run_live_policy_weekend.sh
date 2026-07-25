#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../presets/robot_ft_trossen_vjepa21_small_a100x4_weekend_20260404_5ep.env"

SERVER_HOST="${1:-$TROSSEN_POLICY_CLIENT_HOST}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$VLA_JEPA_REPO_ROOT"

PYTHONPATH="$VLA_JEPA_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
exec "$VLA_JEPA_PYTHON" -m deployment.trossen.run_trossen_policy \
  --host "$SERVER_HOST" \
  --port "$TROSSEN_POLICY_SERVER_PORT" \
  --instruction "$TROSSEN_POLICY_INSTRUCTION" \
  --action-mode "$TROSSEN_POLICY_ACTION_MODE" \
  --state-norm-mode "$TROSSEN_POLICY_STATE_NORM_MODE" \
  --action-norm-mode "$TROSSEN_POLICY_ACTION_NORM_MODE" \
  --fps "$TROSSEN_POLICY_LIVE_FPS" \
  --chunk-size "$TROSSEN_POLICY_LIVE_CHUNK_SIZE" \
  --num-steps "$TROSSEN_POLICY_LIVE_NUM_STEPS" \
  --warmup-steps "$TROSSEN_POLICY_LIVE_WARMUP_STEPS" \
  --max-relative-target "$TROSSEN_POLICY_MAX_RELATIVE_TARGET" \
  --yondu-lerobot-root "$YONDU_TROSSEN_LEROBOT_ROOT" \
  --live \
  --log-path "$TROSSEN_POLICY_LIVE_LOG_PATH" \
  "$@"
