#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CONTAINER_REPO="/workspace/VLA-JEPA"

usage() {
  cat <<'EOF'
Human-first H100x8 training

Usage:
  ./scripts/h100_training.sh setup --config YAML [--skip-build]
  ./scripts/h100_training.sh plan --config YAML [--json]
  ./scripts/h100_training.sh prepare --config YAML --yes-rebuild-data-contract
  ./scripts/h100_training.sh check --config YAML
  ./scripts/h100_training.sh start --config YAML [--run-id ID] [--detach]
  ./scripts/h100_training.sh resume --config YAML --checkpoint STEPS_DIR [--resume-runtime-config YAML] [--detach]
  ./scripts/h100_training.sh status
  ./scripts/h100_training.sh logs [CONTAINER]

Training hyperparameters are never accepted here. Edit the YAML, run prepare
when its immutable data-contract binding changes, then run check and start.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ $# -ge 1 ]] || { usage; exit 2; }
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi
COMMAND="$1"
shift

CONFIG=""
IMAGE=""
SCRATCH=""
BUILD_ENV=()
RUN_ID=""
CHECKPOINT=""
RESUME_RUNTIME_CONFIG=""
CONTAINER_IMAGE_DIGEST=""
CONTAINER_IMAGE_ID=""
DETACH=0
SKIP_BUILD=0
JSON=0
CONFIRM_PREPARE=0
SEEN_CONFIG=0
POSITIONAL=()

while (( $# > 0 )); do
  case "$1" in
    --config)
      (( $# >= 2 )) || die "--config requires a value"
      CONFIG="$2"
      SEEN_CONFIG=1
      shift 2
      ;;
    --run-id)
      (( $# >= 2 )) || die "--run-id requires a value"
      RUN_ID="$2"
      shift 2
      ;;
    --checkpoint)
      (( $# >= 2 )) || die "--checkpoint requires a value"
      CHECKPOINT="$2"
      shift 2
      ;;
    --resume-runtime-config)
      (( $# >= 2 )) || die "--resume-runtime-config requires a value"
      RESUME_RUNTIME_CONFIG="$2"
      shift 2
      ;;
    --detach)
      DETACH=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --json)
      JSON=1
      shift
      ;;
    --yes-rebuild-data-contract)
      CONFIRM_PREPARE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      die "unknown option $1; training overrides are intentionally unsupported"
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${RESUME_RUNTIME_CONFIG}" && "${COMMAND}" != "resume" ]]; then
  die "--resume-runtime-config is accepted only by resume"
fi
case "${COMMAND}" in
  setup|plan|prepare|check|start|resume)
    [[ "${SEEN_CONFIG}" == "1" && -n "${CONFIG}" ]] \
      || die "${COMMAND} requires an explicit --config YAML"
    ;;
esac

load_bootstrap_runtime() {
  require_command python3
  local resolved runtime_output
  [[ -f "${CONFIG}" && ! -L "${CONFIG}" ]] \
    || die "config must be a regular non-symlink file: ${CONFIG}"
  resolved="$(realpath -e -- "${CONFIG}")" || die "config does not exist: ${CONFIG}"
  runtime_output="$(
    python3 - "${resolved}" "${REPO_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
if not path.is_relative_to(repo_root):
    raise SystemExit(f"config must live inside the repository: {path}")
values = {}
build_values = {}
allowed_build_keys = {
    "DOCKERFILE",
    "BASE_IMAGE",
    "TORCH_INDEX_URL",
    "PYTHON_VERSION",
    "INSTALL_DEEPSPEED",
    "INSTALL_MOGE",
    "INSTALL_FLASH_ATTN",
    "FLASH_ATTN_SPEC",
    "FLASH_ATTN_CUDA_ARCH_LIST",
    "FLASH_ATTN_MAX_JOBS",
    "FLASH_ATTN_NVCC_THREADS",
    "INSTALL_FAST_LINEAR_ATTN",
    "FAST_LINEAR_ATTN_SPEC",
    "CAUSAL_CONV1D_SPEC",
    "FAST_LINEAR_ATTN_TRANSFORMERS_SPEC",
    "FAST_LINEAR_ATTN_TILELANG_SPEC",
    "FAST_LINEAR_ATTN_TVM_FFI_SPEC",
    "FAST_LINEAR_ATTN_CUDA_ARCH_LIST",
    "FAST_LINEAR_ATTN_MAX_JOBS",
}

def unquote(raw):
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1]
    return raw

def extended_paths(config_path, lines):
    result = []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"extends:[ ]*([^#]*?)[ ]*", line)
        if not match:
            continue
        inline = unquote(match.group(1))
        if inline:
            result.append(inline)
        else:
            for child in lines[index + 1:]:
                # Both styles below are valid top-level YAML and are emitted
                # by config tooling in this repository:
                #
                #   extends:
                #     - base.yaml
                #
                #   extends:
                #   - base.yaml
                #
                # Keep this dependency-free bootstrap reader compatible with
                # both so materialized configs work through the human-facing
                # launcher without flattening or launcher-side overrides.
                item = re.fullmatch(r"[ \t]*-[ \t]+([^#]+?)[ \t]*", child)
                if item:
                    result.append(unquote(item.group(1)))
                    continue
                if child and not child.startswith((" ", "\t", "#")):
                    break
        break
    resolved = []
    for value in result:
        if not value or any(token in value for token in ("${", "{{", "!!", "\n", "\x00")):
            raise SystemExit(f"unsafe top-level extends value in {config_path}: {value!r}")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            relative_candidate = config_path.parent / candidate
            candidate = (
                relative_candidate
                if relative_candidate.exists()
                else repo_root / candidate
            )
        candidate = candidate.resolve()
        if not candidate.is_relative_to(repo_root):
            raise SystemExit(
                f"extended config must live inside the repository: {candidate}"
            )
        if not candidate.is_file() or candidate.is_symlink():
            raise SystemExit(f"extended config is not a regular file: {candidate}")
        resolved.append(candidate)
    return resolved

def config_layers(config_path, chain=()):
    if config_path in chain:
        raise SystemExit(
            "config extends cycle: "
            + " -> ".join(str(item) for item in (*chain, config_path))
        )
    lines = config_path.read_text(encoding="utf-8").splitlines()
    layers = []
    for base in extended_paths(config_path, lines):
        layers.extend(config_layers(base, (*chain, config_path)))
    layers.append((config_path, lines))
    return layers

for layer_path, lines in config_layers(path):
    inside = False
    for number, line in enumerate(lines, 1):
        if line == "runtime:":
            if inside:
                raise SystemExit(f"duplicate top-level runtime block in {layer_path}")
            inside = True
            continue
        if inside and line and not line.startswith((" ", "\t", "#")):
            break
        if not inside:
            continue
        match = re.fullmatch(r"  (container_image|scratch_root):[ ]+([^#]+?)[ ]*", line)
        if match:
            key, raw = match.groups()
            raw = unquote(raw)
            if not raw or any(token in raw for token in ("${", "{{", "!!")):
                raise SystemExit(
                    f"unsafe runtime.{key} value at {layer_path}:{number}"
                )
            values[key] = raw
        build_match = re.fullmatch(r"      ([A-Z][A-Z0-9_]+):[ ]+([^#]+?)[ ]*", line)
        if build_match:
            key, raw = build_match.groups()
            raw = unquote(raw)
            if key not in allowed_build_keys:
                raise SystemExit(
                    f"unsupported runtime.container_build argument: {key}"
                )
            if not raw or any(
                token in raw for token in ("${", "{{", "!!", "\n", "\x00")
            ):
                raise SystemExit(
                    f"unsafe runtime.container_build.arguments.{key}"
                )
            build_values[key] = raw
missing = {"container_image", "scratch_root"} - values.keys()
if missing:
    raise SystemExit(f"missing explicit runtime fields: {sorted(missing)}")
required_build = allowed_build_keys
missing_build = required_build - build_values.keys()
if missing_build:
    raise SystemExit(f"missing explicit runtime.container_build fields: {sorted(missing_build)}")
print(values["container_image"])
print(values["scratch_root"])
for key in sorted(build_values):
    print(f"{key}={build_values[key]}")
PY
  )" || die "could not read runtime.container_image and runtime.scratch_root from ${resolved}"
  mapfile -t runtime_values <<<"${runtime_output}"
  [[ ${#runtime_values[@]} -eq 21 ]] \
    || die "config must explicitly define the complete runtime container contract"
  IMAGE="${runtime_values[0]}"
  SCRATCH="${runtime_values[1]}"
  BUILD_ENV=("${runtime_values[@]:2}")
  local index
  for index in "${!BUILD_ENV[@]}"; do
    if [[ "${BUILD_ENV[index]}" == DOCKERFILE=* ]]; then
      local dockerfile dockerfile_path
      dockerfile="${BUILD_ENV[index]#DOCKERFILE=}"
      [[ "${dockerfile}" != /* ]] || die "runtime container DOCKERFILE must be repository-relative"
      [[ -f "${REPO_ROOT}/${dockerfile}" && ! -L "${REPO_ROOT}/${dockerfile}" ]] \
        || die "runtime container DOCKERFILE must be a regular non-symlink file: ${dockerfile}"
      dockerfile_path="$(realpath -e -- "${REPO_ROOT}/${dockerfile}")" \
        || die "runtime container DOCKERFILE does not exist: ${dockerfile}"
      [[ "${dockerfile_path}" == "${REPO_ROOT}/"* ]] \
        || die "runtime container DOCKERFILE escapes the repository: ${dockerfile}"
      [[ -f "${dockerfile_path}" && ! -L "${dockerfile_path}" ]] \
        || die "runtime container DOCKERFILE must be a regular non-symlink file"
      BUILD_ENV[index]="DOCKERFILE=${dockerfile_path}"
    fi
  done
  [[ -n "${IMAGE}" ]] || die "runtime.container_image must not be empty"
  [[ "${SCRATCH}" == /* ]] || die "runtime.scratch_root must be absolute"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required host command: $1"
}

resolve_container_image_id() {
  require_command docker
  CONTAINER_IMAGE_ID="$(
    docker image inspect "${IMAGE}" --format '{{.Id}}' 2>/dev/null
  )" || die "configured container image does not exist locally: ${IMAGE}"
  [[ "${CONTAINER_IMAGE_ID}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "docker returned an invalid Image.Id for ${IMAGE}: ${CONTAINER_IMAGE_ID}"
  # Keep the older variable during the narrow worker-topology resume path.
  # Docker's local Image.Id, not a mutable tag or registry manifest name, is
  # the identity authenticated by both fields.
  CONTAINER_IMAGE_DIGEST="${CONTAINER_IMAGE_ID}"
  echo "Container image ID         : ${CONTAINER_IMAGE_ID}"
}

doctor() {
  require_command docker
  require_command nvidia-smi
  docker info >/dev/null 2>&1 || die "Docker is not usable by ${USER:-the current user}"
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  [[ ${#gpu_names[@]} -eq 8 ]] || die "expected exactly 8 GPUs, found ${#gpu_names[@]}"
  local index
  for index in "${!gpu_names[@]}"; do
    [[ "${gpu_names[index]}" == *H100* ]] \
      || die "GPU ${index} is not an H100: ${gpu_names[index]}"
  done
  docker info --format '{{json .Runtimes}}' | grep -q 'nvidia' \
    || die "Docker NVIDIA runtime is not installed/configured"
  echo "Host hardware              : PASS (8x H100)"
  echo "Docker NVIDIA runtime      : PASS"
}

config_in_container() {
  local resolved relative
  [[ -f "${CONFIG}" && ! -L "${CONFIG}" ]] \
    || die "config must be a regular non-symlink file: ${CONFIG}"
  resolved="$(realpath -e -- "${CONFIG}")" || die "config does not exist: ${CONFIG}"
  [[ "${resolved}" == "${REPO_ROOT}/"* ]] \
    || die "config must live inside the repository: ${resolved}"
  relative="${resolved#${REPO_ROOT}/}"
  printf '%s/%s\n' "${CONTAINER_REPO}" "${relative}"
}

repo_file_in_container() {
  local value="$1"
  local label="$2"
  local resolved relative
  [[ -f "${value}" && ! -L "${value}" ]] \
    || die "${label} must be a regular non-symlink file: ${value}"
  resolved="$(realpath -e -- "${value}")" || die "${label} does not exist: ${value}"
  [[ "${resolved}" == "${REPO_ROOT}/"* ]] \
    || die "${label} must live inside the repository: ${resolved}"
  [[ -f "${resolved}" && ! -L "${resolved}" ]] \
    || die "${label} must be a regular non-symlink file: ${resolved}"
  relative="${resolved#${REPO_ROOT}/}"
  printf '%s/%s\n' "${CONTAINER_REPO}" "${relative}"
}

run_in_container() {
  local docker_name="$1"
  shift
  local auto_remove=1
  local configured_image="${IMAGE}"
  local image_reference="${CONTAINER_IMAGE_ID:-${configured_image}}"
  if [[ "${DETACH}" == "1" ]]; then
    auto_remove=0
  fi
  IMAGE="${image_reference}" \
  VLA_JEPA_SCRATCH="${SCRATCH}" \
  CHECKPOINT_ROOT="${SCRATCH}/checkpoints" \
  STARVLA_CONTAINER_IMAGE="${configured_image}" \
  STARVLA_CONTAINER_IMAGE_ID="${CONTAINER_IMAGE_ID}" \
  STARVLA_CONTAINER_IMAGE_DIGEST="${CONTAINER_IMAGE_DIGEST}" \
  STARVLA_CONFIG_IS_AUTHORITATIVE=1 \
  DOCKER_NAME="${docker_name}" \
  DOCKER_TTY=0 \
  DOCKER_DETACH="${DETACH}" \
  DOCKER_AUTO_REMOVE="${auto_remove}" \
  DOCKER_USER="$(id -u):$(id -g)" \
  DOCKER_HOME=/tmp \
    exec "${SCRIPT_DIR}/docker_run_training.sh" "$@"
}

case "${COMMAND}" in
  setup)
    (( ${#POSITIONAL[@]} == 0 )) || die "setup takes no positional arguments"
    [[ -z "${RUN_ID}" && -z "${CHECKPOINT}" && "${DETACH}" == 0 \
      && "${JSON}" == 0 && "${CONFIRM_PREPARE}" == 0 ]] \
      || die "setup accepts only --config and --skip-build"
    load_bootstrap_runtime
    doctor
    mkdir -p \
      "${SCRATCH}/checkpoints" \
      "${SCRATCH}/hf/hub" \
      "${SCRATCH}/cache/torch" \
      "${SCRATCH}/cache/pip" \
      "${SCRATCH}/tmp" \
      "${SCRATCH}/src" \
      || die "cannot create ${SCRATCH}; create/chown it for ${USER:-this user} first"
    if [[ "${SKIP_BUILD}" == "0" ]]; then
      echo "Building H100 image ${IMAGE}. This can take a while."
      env IMAGE="${IMAGE}" "${BUILD_ENV[@]}" \
        "${SCRIPT_DIR}/docker_build_training.sh"
    else
      docker image inspect "${IMAGE}" >/dev/null 2>&1 \
        || die "--skip-build requested but image does not exist: ${IMAGE}"
    fi
    docker image inspect "${IMAGE}" --format 'Image                     : {{.Id}}'
    config_path="$(config_in_container)"
    echo "Installing config-pinned helper repositories."
    IMAGE="${IMAGE}" \
    VLA_JEPA_SCRATCH="${SCRATCH}" \
    CHECKPOINT_ROOT="${SCRATCH}/checkpoints" \
    STARVLA_CONTAINER_IMAGE="${IMAGE}" \
    STARVLA_CONFIG_IS_AUTHORITATIVE=1 \
    DOCKER_USER="$(id -u):$(id -g)" \
    DOCKER_HOME=/tmp \
    DOCKER_NAME="starvla-h100-setup-$$" \
    DOCKER_TTY=0 \
    DOCKER_GPU_MODE=none \
      "${SCRIPT_DIR}/docker_run_training.sh" \
        python scripts/h100_training.py setup --config "${config_path}"
    printf 'Next: ./scripts/h100_training.sh plan'
    if [[ "${SEEN_CONFIG}" == "1" ]]; then
      printf ' --config %q' "${CONFIG}"
    fi
    printf '\n'
    printf 'If that profile requires a data contract: ./scripts/h100_training.sh prepare'
    if [[ "${SEEN_CONFIG}" == "1" ]]; then
      printf ' --config %q' "${CONFIG}"
    fi
    printf ' --yes-rebuild-data-contract\n'
    printf 'Then: ./scripts/h100_training.sh check'
    if [[ "${SEEN_CONFIG}" == "1" ]]; then
      printf ' --config %q' "${CONFIG}"
    fi
    printf '\n'
    ;;
  plan)
    (( ${#POSITIONAL[@]} == 0 )) || die "plan takes no positional arguments"
    [[ -z "${RUN_ID}" && -z "${CHECKPOINT}" && "${DETACH}" == 0 \
      && "${SKIP_BUILD}" == 0 && "${CONFIRM_PREPARE}" == 0 ]] \
      || die "plan accepts only --config and --json"
    load_bootstrap_runtime
    config_path="$(config_in_container)"
    args=(python scripts/h100_training.py plan --config "${config_path}")
    [[ "${JSON}" == "0" ]] || args+=(--json)
    run_in_container "starvla-h100-plan-$$" "${args[@]}"
    ;;
  prepare)
    (( ${#POSITIONAL[@]} == 0 )) || die "prepare takes no positional arguments"
    [[ -z "${RUN_ID}" && -z "${CHECKPOINT}" && "${DETACH}" == 0 \
      && "${SKIP_BUILD}" == 0 && "${JSON}" == 0 ]] \
      || die "prepare accepts only --config and --yes-rebuild-data-contract"
    load_bootstrap_runtime
    [[ "${CONFIRM_PREPARE}" == "1" ]] \
      || die "prepare requires --yes-rebuild-data-contract"
    config_path="$(config_in_container)"
    DETACH=0
    IMAGE="${IMAGE}" \
    VLA_JEPA_SCRATCH="${SCRATCH}" \
    CHECKPOINT_ROOT="${SCRATCH}/checkpoints" \
    STARVLA_CONTAINER_IMAGE="${IMAGE}" \
    STARVLA_CONFIG_IS_AUTHORITATIVE=1 \
    DOCKER_USER="$(id -u):$(id -g)" \
    DOCKER_HOME=/tmp \
    DOCKER_NAME="starvla-h100-prepare-$$" \
    DOCKER_TTY=0 \
      exec "${SCRIPT_DIR}/docker_run_training.sh" \
        python scripts/h100_training.py prepare \
        --config "${config_path}" \
        --yes-rebuild-data-contract
    ;;
  check)
    (( ${#POSITIONAL[@]} == 0 )) || die "check takes no positional arguments"
    [[ -z "${RUN_ID}" && -z "${CHECKPOINT}" && "${DETACH}" == 0 \
      && "${SKIP_BUILD}" == 0 && "${JSON}" == 0 && "${CONFIRM_PREPARE}" == 0 ]] \
      || die "check accepts only --config"
    load_bootstrap_runtime
    resolve_container_image_id
    config_path="$(config_in_container)"
    run_in_container "starvla-h100-check-$$" \
      python scripts/h100_training.py check --config "${config_path}"
    ;;
  start)
    (( ${#POSITIONAL[@]} == 0 )) || die "start takes no positional arguments"
    [[ "${SKIP_BUILD}" == 0 && "${JSON}" == 0 && "${CONFIRM_PREPARE}" == 0 ]] \
      || die "start accepts only --config, --run-id, and --detach"
    load_bootstrap_runtime
    resolve_container_image_id
    [[ -z "${CHECKPOINT}" ]] || die "use the resume command with --checkpoint"
    config_path="$(config_in_container)"
    container_name="starvla-h100-train-$(date -u +%Y%m%d-%H%M%S)"
    args=(python scripts/h100_training.py launch --config "${config_path}")
    [[ -z "${RUN_ID}" ]] || args+=(--run-id "${RUN_ID}")
    echo "Container                  : ${container_name}"
    [[ "${DETACH}" == "0" ]] \
      || echo "Follow logs                : ./scripts/h100_training.sh logs ${container_name}"
    run_in_container "${container_name}" "${args[@]}"
    ;;
  resume)
    (( ${#POSITIONAL[@]} == 0 )) || die "resume takes no positional arguments"
    [[ "${SKIP_BUILD}" == 0 && "${JSON}" == 0 && "${CONFIRM_PREPARE}" == 0 ]] \
      || die "resume accepts only --config, --checkpoint, and --detach"
    load_bootstrap_runtime
    resolve_container_image_id
    [[ -n "${CHECKPOINT}" ]] || die "resume requires --checkpoint STEPS_DIR"
    [[ -z "${RUN_ID}" ]] || die "resume derives the original run ID from its checkpoint"
    config_path="$(config_in_container)"
    container_name="starvla-h100-resume-$(date -u +%Y%m%d-%H%M%S)"
    if [[ -n "${RESUME_RUNTIME_CONFIG}" ]]; then
      resume_runtime_config_path="$(
        repo_file_in_container \
          "${RESUME_RUNTIME_CONFIG}" \
          "resume runtime config"
      )"
      args=(
        python scripts/h100_resume_runtime.py
        --config "${config_path}"
        --checkpoint "${CHECKPOINT}"
        --resume-runtime-config "${resume_runtime_config_path}"
      )
    else
      args=(
        python scripts/h100_training.py launch
        --config "${config_path}"
        --resume "${CHECKPOINT}"
      )
    fi
    run_in_container "${container_name}" "${args[@]}"
    ;;
  status)
    (( ${#POSITIONAL[@]} == 0 )) || die "status takes no positional arguments"
    [[ "${SEEN_CONFIG}" == 0 && -z "${RUN_ID}" && -z "${CHECKPOINT}" \
      && "${DETACH}" == 0 && "${SKIP_BUILD}" == 0 && "${JSON}" == 0 \
      && "${CONFIRM_PREPARE}" == 0 ]] || die "status accepts no options"
    require_command docker
    docker ps -a \
      --filter 'name=starvla-h100-' \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.CreatedAt}}'
    ;;
  logs)
    (( ${#POSITIONAL[@]} <= 1 )) || die "logs accepts at most one container name"
    [[ "${SEEN_CONFIG}" == 0 && -z "${RUN_ID}" && -z "${CHECKPOINT}" \
      && "${DETACH}" == 0 && "${SKIP_BUILD}" == 0 && "${JSON}" == 0 \
      && "${CONFIRM_PREPARE}" == 0 ]] || die "logs accepts no options"
    require_command docker
    container="${POSITIONAL[0]:-}"
    if [[ -z "${container}" ]]; then
      container="$(docker ps -a --filter 'name=starvla-h100-' --format '{{.Names}}' | head -n 1)"
    fi
    [[ -n "${container}" ]] || die "no starvla-h100 container found"
    exec docker logs -f "${container}"
    ;;
  help)
    usage
    ;;
  *)
    usage >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
