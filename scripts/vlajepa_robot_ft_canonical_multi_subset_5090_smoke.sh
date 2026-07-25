#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_YAML="${CONFIG_YAML:-${REPO_ROOT}/scripts/config/vlajepa_robot_ft_canonical_multi_subset_5090_smoke.yaml}"
RUN_ID="${RUN_ID:-robot_ft_canonical_multi_subset_5090_smoke_$(date +%Y%m%d_%H%M%S)}"

CONFIG_YAML="${CONFIG_YAML}" RUN_ID="${RUN_ID}" \
  "${REPO_ROOT}/scripts/vlajepa_robot_ft_canonical_subset_5090_smoke.sh" "$@"
