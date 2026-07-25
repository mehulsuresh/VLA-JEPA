#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

task="${1:-}"
model_path="${2:-}"
log_dir_root="${3:-}"

if [[ -z "${task}" || -z "${model_path}" ]]; then
  echo "Usage: $(basename "$0") <task|all> <model_path> [log_dir_root]"
  echo "  task: pick_coke_can | move_near | drawer | long_horizon_apple_in_drawer | bridge_put_on | all"
  echo "  model_path: checkpoint file path (.pt)"
  echo "  log_dir_root: optional; defaults to <dirname(model_path)>/google_robot_eval"
  exit 1
fi

run_one() {
  local t="$1"
  if [[ -n "${log_dir_root}" ]]; then
    python "${SCRIPT_DIR}/calc_metrics_evaluation_videos.py" --task "${t}" --model-path "${model_path}" --log-dir-root "${log_dir_root}"
  else
    python "${SCRIPT_DIR}/calc_metrics_evaluation_videos.py" --task "${t}" --model-path "${model_path}"
  fi
}

case "${task}" in
  pick_coke_can|move_near|drawer|long_horizon_apple_in_drawer|bridge_put_on)
    run_one "${task}"
    ;;
  all)
    run_one pick_coke_can
    run_one move_near
    run_one drawer
    run_one long_horizon_apple_in_drawer
    run_one bridge_put_on
    ;;
  *)
    echo "Unknown task: ${task}"
    exit 2
    ;;
esac

