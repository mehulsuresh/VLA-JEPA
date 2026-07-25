#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../presets/robot_ft_trossen_vjepa21_small_a100x4_weekend_20260404_5ep.env"

cd "$VLA_JEPA_REPO_ROOT"

ARGS=(
  deployment/model_server/server_policy.py
  --ckpt_path "$TROSSEN_POLICY_CKPT"
  --host "$TROSSEN_POLICY_SERVER_HOST"
  --port "$TROSSEN_POLICY_SERVER_PORT"
  --cuda "$TROSSEN_POLICY_CUDA"
)

if [[ "${TROSSEN_POLICY_USE_BF16}" == "1" ]]; then
  ARGS+=(--use_bf16)
fi

exec env DEBUG=false DEBUGPY=false "$VLA_JEPA_PYTHON" "${ARGS[@]}"
