#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
DEFAULT_CONFIG="${REPO_ROOT}/scripts/config/h100/realman_realsource_intervention_hq_curriculum_v1.yaml"
CONTAINER_REPO="/workspace/VLA-JEPA"

usage() {
  printf '%s\n' \
    "Config-owned RealMan H100 curriculum" \
    "" \
    "Usage:" \
    "  ./scripts/h100_curriculum.sh setup [--config YAML] [--skip-build]" \
    "  ./scripts/h100_curriculum.sh plan [--config YAML] [--json]" \
    "  ./scripts/h100_curriculum.sh check [--config YAML]" \
    "  ./scripts/h100_curriculum.sh start [--config YAML] [--run-id ID] [--detach]" \
    "  ./scripts/h100_curriculum.sh resume --config YAML --run-id ORIGINAL [--detach]" \
    "  ./scripts/h100_curriculum.sh status" \
    "  ./scripts/h100_curriculum.sh logs [CONTAINER]" \
    "" \
    "The YAML owns stage order, frozen views, epoch counts, learning rates," \
    "normalization, prompts, and checkpoint handoffs. This wrapper accepts no" \
    "training hyperparameter overrides."
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ $# -ge 1 ]] || { usage; exit 2; }
COMMAND="$1"
shift
CONFIG="${DEFAULT_CONFIG}"
RUN_ID=""
DETACH=0
SKIP_BUILD=0
JSON=0
POSITIONAL=()

while (( $# > 0 )); do
  case "$1" in
    --config)
      (( $# >= 2 )) || die "--config requires a value"
      CONFIG="$2"
      shift 2
      ;;
    --run-id)
      (( $# >= 2 )) || die "--run-id requires a value"
      RUN_ID="$2"
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
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      die "unknown option $1; edit the reviewed YAML instead"
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

load_bootstrap() {
  require_command python3
  local resolved output
  resolved="$(realpath -e -- "${CONFIG}")" || die "config does not exist: ${CONFIG}"
  [[ "${resolved}" == "${REPO_ROOT}/"* ]] \
    || die "curriculum config must live inside the repository"
  CONFIG="${resolved}"
  output="$(
    python3 - "${CONFIG}" "${REPO_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

config = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
values = {}
inside = False
for number, line in enumerate(config.read_text(encoding="utf-8").splitlines(), 1):
    if line == "bootstrap:":
        if inside:
            raise SystemExit("duplicate bootstrap block")
        inside = True
        continue
    if inside and line and not line.startswith((" ", "\t", "#")):
        break
    if not inside:
        continue
    match = re.fullmatch(
        r"  (stage_config|container_image|scratch_root):[ ]+([^#]+?)[ ]*",
        line,
    )
    if match:
        key, value = match.groups()
        value = value.strip().strip("\"'")
        if not value or any(token in value for token in ("${", "{{", "!!", "\x00")):
            raise SystemExit(f"unsafe bootstrap value at line {number}")
        values[key] = value
missing = {"stage_config", "container_image", "scratch_root"} - values.keys()
if missing:
    raise SystemExit(f"missing bootstrap keys: {sorted(missing)}")
stage = Path(values["stage_config"]).expanduser()
if not stage.is_absolute():
    stage = repo / stage
stage = stage.resolve()
if not stage.is_relative_to(repo) or not stage.is_file() or stage.is_symlink():
    raise SystemExit(f"invalid bootstrap stage config: {stage}")
scratch = Path(values["scratch_root"]).expanduser()
if not scratch.is_absolute():
    raise SystemExit("bootstrap.scratch_root must be absolute")
print(stage)
print(values["container_image"])
print(scratch)
PY
  )" || die "could not parse curriculum bootstrap"
  mapfile -t BOOTSTRAP <<<"${output}"
  [[ ${#BOOTSTRAP[@]} -eq 3 ]] || die "invalid curriculum bootstrap output"
  STAGE_CONFIG="${BOOTSTRAP[0]}"
  IMAGE="${BOOTSTRAP[1]}"
  SCRATCH="${BOOTSTRAP[2]}"
  CONFIG_IN_CONTAINER="${CONTAINER_REPO}/${CONFIG#${REPO_ROOT}/}"
}

run_curriculum_container() {
  local name="$1"
  local gpu_mode="$2"
  shift 2
  local auto_remove=1
  [[ "${DETACH}" == "0" ]] || auto_remove=0
  IMAGE="${IMAGE}" \
  VLA_JEPA_SCRATCH="${SCRATCH}" \
  CHECKPOINT_ROOT="${SCRATCH}/checkpoints" \
  STARVLA_CONTAINER_IMAGE="${IMAGE}" \
  DOCKER_NAME="${name}" \
  DOCKER_TTY=0 \
  DOCKER_DETACH="${DETACH}" \
  DOCKER_AUTO_REMOVE="${auto_remove}" \
  DOCKER_GPU_MODE="${gpu_mode}" \
  DOCKER_USER="$(id -u):$(id -g)" \
  DOCKER_HOME=/tmp \
    exec "${SCRIPT_DIR}/docker_run_training.sh" "$@"
}

case "${COMMAND}" in
  setup)
    (( ${#POSITIONAL[@]} == 0 )) || die "setup takes no positional arguments"
    [[ -z "${RUN_ID}" && "${DETACH}" == 0 && "${JSON}" == 0 ]] \
      || die "setup accepts only --config and --skip-build"
    load_bootstrap
    args=(setup --config "${STAGE_CONFIG}")
    [[ "${SKIP_BUILD}" == "0" ]] || args+=(--skip-build)
    exec "${SCRIPT_DIR}/h100_training.sh" "${args[@]}"
    ;;
  plan)
    (( ${#POSITIONAL[@]} == 0 )) || die "plan takes no positional arguments"
    [[ -z "${RUN_ID}" && "${DETACH}" == 0 && "${SKIP_BUILD}" == 0 ]] \
      || die "plan accepts only --config and --json"
    load_bootstrap
    args=(python scripts/h100_curriculum.py plan --config "${CONFIG_IN_CONTAINER}")
    [[ "${JSON}" == "0" ]] || args+=(--json)
    run_curriculum_container "starvla-curriculum-plan-$$" none "${args[@]}"
    ;;
  check)
    (( ${#POSITIONAL[@]} == 0 )) || die "check takes no positional arguments"
    [[ -z "${RUN_ID}" && "${DETACH}" == 0 && "${SKIP_BUILD}" == 0 && "${JSON}" == 0 ]] \
      || die "check accepts only --config"
    load_bootstrap
    run_curriculum_container "starvla-curriculum-check-$$" gpus \
      python scripts/h100_curriculum.py check --config "${CONFIG_IN_CONTAINER}"
    ;;
  start)
    (( ${#POSITIONAL[@]} == 0 )) || die "start takes no positional arguments"
    [[ "${SKIP_BUILD}" == 0 && "${JSON}" == 0 ]] \
      || die "start accepts only --config, --run-id, and --detach"
    load_bootstrap
    name="starvla-curriculum-$(date -u +%Y%m%d-%H%M%S)"
    args=(python scripts/h100_curriculum.py run --config "${CONFIG_IN_CONTAINER}")
    [[ -z "${RUN_ID}" ]] || args+=(--run-id "${RUN_ID}")
    printf 'Container                  : %s\n' "${name}"
    run_curriculum_container "${name}" gpus "${args[@]}"
    ;;
  resume)
    (( ${#POSITIONAL[@]} == 0 )) || die "resume takes no positional arguments"
    [[ "${SKIP_BUILD}" == 0 && "${JSON}" == 0 ]] \
      || die "resume accepts only --config, --run-id, and --detach"
    [[ -n "${RUN_ID}" ]] \
      || die "resume requires the original --run-id"
    load_bootstrap
    name="starvla-curriculum-resume-$(date -u +%Y%m%d-%H%M%S)"
    args=(
      python scripts/h100_curriculum.py run
      --config "${CONFIG_IN_CONTAINER}"
      --run-id "${RUN_ID}"
      --resume
    )
    printf 'Container                  : %s\n' "${name}"
    run_curriculum_container "${name}" gpus "${args[@]}"
    ;;
  status)
    (( ${#POSITIONAL[@]} == 0 )) || die "status takes no positional arguments"
    [[ -z "${RUN_ID}" && "${DETACH}" == 0 && "${SKIP_BUILD}" == 0 && "${JSON}" == 0 ]] \
      || die "status accepts no options"
    require_command docker
    docker ps -a \
      --filter 'name=starvla-curriculum-' \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.CreatedAt}}'
    ;;
  logs)
    (( ${#POSITIONAL[@]} <= 1 )) || die "logs accepts at most one container"
    [[ -z "${RUN_ID}" && "${DETACH}" == 0 && "${SKIP_BUILD}" == 0 && "${JSON}" == 0 ]] \
      || die "logs accepts no options"
    require_command docker
    container="${POSITIONAL[0]:-}"
    if [[ -z "${container}" ]]; then
      container="$(
        docker ps -a \
          --filter 'name=starvla-curriculum-' \
          --format '{{.Names}}' |
          head -n 1
      )"
    fi
    [[ -n "${container}" ]] || die "no curriculum container found"
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
