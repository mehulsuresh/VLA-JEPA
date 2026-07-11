#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_lerobot_ogrealman_human_labelled_cloud_a100x8_qwen35_2b_full_moge_vitb_vjepa_large.yaml}"
export RUN_ID="${RUN_ID:-robot_ft_lerobot_ogrealman_human_labelled_cloud_a100x8_qwen35_2b_full_moge_vitb_vjepa_large_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/vlajepa_robot_ft_libero_plus_a100x8_qwen3_full_moge_vitb_vjepa_large.sh" "$@"
