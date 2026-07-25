#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_trossen_vjepa21_small_a100x4_lora.yaml}"

exec "${SCRIPT_DIR}/vlajepa_robot_ft_trossen_a100x4.sh" "$@"
