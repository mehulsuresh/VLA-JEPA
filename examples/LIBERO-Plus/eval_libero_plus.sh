#!/usr/bin/env bash
set -euo pipefail

export LIBERO_HOME="${LIBERO_HOME:-/home/dataset-local/LIBERO-plus}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}" # let eval_libero find the LIBERO tools
export PYTHONPATH="$(pwd):${PYTHONPATH}" # let LIBERO find the websocket tools from main repo
sim_python="${sim_python:-${LIBERO_HOME}/env/bin/python}"
server_python="${server_python:-python}"

your_ckpt="${your_ckpt:-${1:-}}"
if [[ -z "${your_ckpt}" ]]; then
    echo "Usage: your_ckpt=/path/to/checkpoint $0"
    echo "   or: $0 /path/to/checkpoint"
    exit 2
fi
if [[ ! -e "${your_ckpt}" ]]; then
    echo "Checkpoint does not exist: ${your_ckpt}"
    exit 2
fi
if [[ ! -x "${sim_python}" ]]; then
    echo "Simulator Python is not executable: ${sim_python}"
    echo "Set sim_python=/path/to/LIBERO-plus/env/bin/python"
    exit 2
fi

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

items=("Background Textures" "Camera Viewpoints" "Language Instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise")
if [[ -n "${LIBERO_PLUS_ITEMS:-}" ]]; then
    IFS='|' read -r -a items <<< "${LIBERO_PLUS_ITEMS}"
fi
task_suite_name="${task_suite_name:-libero_mix}"

host="${host:-127.0.0.1}"
base_port="${base_port:-14082}"
index=0
with_state="${with_state:-true}"
num_trials_per_task="${num_trials_per_task:-1}" # must be 1 for perturbation evaluation
wait_for_jobs="${LIBERO_PLUS_WAIT_FOR_JOBS:-true}"
extra_eval_args=()
if [[ -n "${LIBERO_PLUS_MAX_TASKS:-}" ]]; then
    extra_eval_args+=(--args.max-tasks "${LIBERO_PLUS_MAX_TASKS}")
fi
if [[ -n "${LIBERO_PLUS_TASK_START:-}" ]]; then
    extra_eval_args+=(--args.task-start "${LIBERO_PLUS_TASK_START}")
fi
if [[ -n "${LIBERO_PLUS_MAX_STEPS_OVERRIDE:-}" ]]; then
    extra_eval_args+=(--args.max-steps-override "${LIBERO_PLUS_MAX_STEPS_OVERRIDE}")
fi

server_pids=()
eval_pids=()
cleanup() {
    for pid in "${server_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${eval_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

IFS=',' read -r -a CUDA_IDS <<< "${LIBERO_PLUS_CUDA_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
num_gpus="${#CUDA_IDS[@]}"
if [[ "${num_gpus}" -lt 1 ]]; then
    echo "No CUDA devices configured."
    exit 2
fi

for perturbation_name in "${items[@]}"
do
perturbation_file_name=${perturbation_name// /_}
gpu_id="${CUDA_IDS[$((index % num_gpus))]}"
port=$((base_port+index))

${server_python} ./deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    --cuda ${gpu_id} &
server_pids+=("$!")

video_out_path="results/plus_${task_suite_name}/${perturbation_file_name}/${folder_name}"

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}
mkdir -p ${video_out_path}

# export DEBUG=true

MUJOCO_EGL_DEVICE_ID="${LIBERO_PLUS_MUJOCO_EGL_DEVICE_ID:-${MUJOCO_EGL_DEVICE_ID}}" \
${sim_python} ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port ${port} \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" > "${video_out_path}/eval.log" 2>&1 \
    --args.category_value "$perturbation_name" \
    --args.with_state "$with_state" \
    "${extra_eval_args[@]}" &
eval_pids+=("$!")

index=$((index+1))
done

if [[ "${wait_for_jobs}" == "true" ]]; then
    status=0
    for pid in "${eval_pids[@]}"; do
        wait "${pid}" || status=$?
    done
    exit "${status}"
fi
