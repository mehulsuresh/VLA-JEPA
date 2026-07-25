#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Centralized environment variables for all child eval scripts.
# You can override them via environment variables when running this script.
: "${sim_python:=/home/dataset-local/SimplerEnv/env/bin/python}"
: "${SimplerEnv_PATH:=/home/dataset-local/SimplerEnv}"
export sim_python
export SimplerEnv_PATH
MODEL_PATH=/home/dataset-local/starVLA_A100/checkpoints/direct_ft/oxe/JEVLA_wo_human/checkpoints/steps_120000_pytorch_model.pt

sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_bridge.sh" "${MODEL_PATH}"

sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_drawer_visual_matching.sh" "${MODEL_PATH}"
sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_move_near_visual_matching.sh" "${MODEL_PATH}"
sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_pick_coke_can_visual_matching.sh" "${MODEL_PATH}"
sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_put_in_drawer_visual_matching.sh" "${MODEL_PATH}"
