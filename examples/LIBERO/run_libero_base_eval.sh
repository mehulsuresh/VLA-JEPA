#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ckpt_path="${1:-${CKPT_PATH:-}}"
if [[ -z "${ckpt_path}" ]]; then
    echo "Usage: $0 /path/to/checkpoint"
    echo
    echo "The checkpoint can be a run root, final_model/, an interval checkpoint directory,"
    echo "or a concrete pytorch_model.pt/model.safetensors file."
    echo
    echo "Common overrides:"
    echo "  TASK_NAME=turn_on_the_stove"
    echo "  TASK_LANGUAGE='turn on the stove'"
    echo "  SUITE_NAME=libero_goal"
    echo "  MAX_STEPS=1000"
    echo "  NUM_DDIM_STEPS=10"
    echo "  POLICY_PORT=10093"
    exit 2
fi

if [[ ! -e "${ckpt_path}" ]]; then
    echo "Checkpoint does not exist: ${ckpt_path}" >&2
    exit 2
fi

eval_image="${EVAL_IMAGE:-vla-jepa:py313-cu130-a100-libero-eval}"
vla_root="${VLA_ROOT:-/mnt/vla-jepa}"
libero_home="${LIBERO_HOME:-${vla_root}/src/LIBERO-plus}"
policy_host="${POLICY_HOST:-127.0.0.1}"
policy_port="${POLICY_PORT:-10093}"
suite_name="${SUITE_NAME:-libero_goal}"
task_name="${TASK_NAME:-turn_on_the_stove}"
task_language="${TASK_LANGUAGE:-turn on the stove}"
bddl_file="${BDDL_FILE:-${task_name}.bddl}"
init_states_file="${INIT_STATES_FILE:-${task_name}.pruned_init}"
max_steps="${MAX_STEPS:-1000}"
num_ddim_steps="${NUM_DDIM_STEPS:-10}"
num_steps_wait="${NUM_STEPS_WAIT:-10}"
action_execution_mode="${ACTION_EXECUTION_MODE:-receding}"
action_ensemble="${ACTION_ENSEMBLE:-true}"
action_ensemble_horizon="${ACTION_ENSEMBLE_HORIZON:-}"
adaptive_ensemble_alpha="${ADAPTIVE_ENSEMBLE_ALPHA:-0.1}"
num_trials="${NUM_TRIALS:-1}"
seed="${SEED:-7}"
with_state="${WITH_STATE:-true}"
run_id="${RUN_ID:-libero_base_${task_name}_$(date +%Y%m%d_%H%M%S)}"
out_root="${OUT_ROOT:-${vla_root}/logs}"
video_out_path="${VIDEO_OUT_PATH:-${out_root}/${run_id}}"
docker_gpus="${DOCKER_GPUS:-all}"
mujoco_egl_device_id="${MUJOCO_EGL_DEVICE_ID:-0}"

mkdir -p "${video_out_path}"

if ! timeout 2 bash -c "cat < /dev/null > /dev/tcp/${policy_host}/${policy_port}" 2>/dev/null; then
    echo "Policy server is not listening on ${policy_host}:${policy_port}." >&2
    echo "Start deployment/model_server/server_policy.py first, then rerun this script." >&2
    exit 2
fi

echo "Running base LIBERO eval"
echo "  checkpoint: ${ckpt_path}"
echo "  task: ${suite_name}/${task_name}"
echo "  language: ${task_language}"
echo "  max_steps: ${max_steps}"
echo "  num_ddim_steps: ${num_ddim_steps}"
echo "  num_steps_wait: ${num_steps_wait}"
echo "  action_execution_mode: ${action_execution_mode}"
echo "  action_ensemble: ${action_ensemble}"
echo "  policy: ${policy_host}:${policy_port}"
echo "  output: ${video_out_path}"

ensemble_flag="--action-ensemble"
if [[ "${action_ensemble}" == "0" || "${action_ensemble}" == "false" || "${action_ensemble}" == "False" ]]; then
    ensemble_flag="--no-action-ensemble"
fi

ensemble_horizon_args=()
if [[ -n "${action_ensemble_horizon}" ]]; then
    ensemble_horizon_args=(--action-ensemble-horizon "${action_ensemble_horizon}")
fi

docker run --rm -i --gpus "${docker_gpus}" --network host \
    -e PYTHONPATH=/workspace/VLA-JEPA:${libero_home} \
    -e LIBERO_HOME="${libero_home}" \
    -e LIBERO_CONFIG_PATH="${libero_home}/libero" \
    -e MUJOCO_GL=egl \
    -e MUJOCO_EGL_DEVICE_ID="${mujoco_egl_device_id}" \
    -e TOKENIZERS_PARALLELISM=false \
    -v "${vla_root}:${vla_root}" \
    -v "${repo_root}:/workspace/VLA-JEPA" \
    -v "${libero_home}:${libero_home}" \
    -w /workspace/VLA-JEPA \
    "${eval_image}" \
    python examples/LIBERO/eval_libero_base_task.py \
        --pretrained-path "${ckpt_path}" \
        --host "${policy_host}" \
        --port "${policy_port}" \
        --suite-name "${suite_name}" \
        --task-name "${task_name}" \
        --task-language "${task_language}" \
        --bddl-file "${bddl_file}" \
        --init-states-file "${init_states_file}" \
        --video-out-path "${video_out_path}" \
        --num-trials-per-task "${num_trials}" \
        --max-steps "${max_steps}" \
        --num-ddim-steps "${num_ddim_steps}" \
        --num-steps-wait "${num_steps_wait}" \
        --action-execution-mode "${action_execution_mode}" \
        "${ensemble_flag}" \
        "${ensemble_horizon_args[@]}" \
        --adaptive-ensemble-alpha "${adaptive_ensemble_alpha}" \
        --with-state "${with_state}" \
        --seed "${seed}" 2>&1 | tee "${video_out_path}/eval.log"

echo "Eval output: ${video_out_path}"
