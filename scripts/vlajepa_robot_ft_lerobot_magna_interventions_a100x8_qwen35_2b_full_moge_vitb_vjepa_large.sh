#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

A100_CONFIG="${REPO_ROOT}/scripts/config/vlajepa_robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml"
if [[ -n "${CONFIG_YAML:-}" && "${CONFIG_YAML}" != "${A100_CONFIG}" ]]; then
  echo "A100 Magna launcher uses only its reviewed config: ${A100_CONFIG}" >&2
  exit 2
fi
export CONFIG_YAML="${A100_CONFIG}"
export RUN_ID="${RUN_ID:-robot_ft_lerobot_magna_interventions_a100x8_qwen35_2b_full_moge_vitb_vjepa_large_$(date +%Y%m%d_%H%M%S)}"
if [[ -n "${STARVLA_CONFIG_IS_AUTHORITATIVE:-}" && "${STARVLA_CONFIG_IS_AUTHORITATIVE}" != "1" ]]; then
  echo "A100 Magna production launcher requires STARVLA_CONFIG_IS_AUTHORITATIVE=1" >&2
  exit 2
fi
export STARVLA_CONFIG_IS_AUTHORITATIVE=1

exec "${SCRIPT_DIR}/vlajepa_robot_ft_libero_plus_a100x8_qwen3_full_moge_vitb_vjepa_large.sh" "$@"
