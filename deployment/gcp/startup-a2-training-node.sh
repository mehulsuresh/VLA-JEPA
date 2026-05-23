#!/usr/bin/env bash
set -Eeuo pipefail

# GCE startup/bootstrap script for Debian 12 A2 Ultra training nodes.
# It installs host runtime tools, NVIDIA drivers, stripes local SSDs into one
# ext4 RAID0 volume, and leaves a /data symlink for datasets, checkpoints,
# logs, and temp files.
#
# Safe to rerun. If Google's GPU installer reboots the VM mid-run, GCE startup
# scripts run again on the next boot and this script continues idempotently.

if [[ ${EUID} -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

export DEBIAN_FRONTEND=noninteractive

LOG_FILE="${LOG_FILE:-/var/log/training-node-bootstrap.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

MOUNT_POINT="${MOUNT_POINT:-/mnt/disks/ssd-array}"
DATA_LINK="${DATA_LINK:-/data}"
RAID_DEVICE="${RAID_DEVICE:-/dev/md0}"
RAID_CHUNK="${RAID_CHUNK:-1024K}"
LOCAL_SSD_GLOB="${LOCAL_SSD_GLOB:-/dev/disk/by-id/google-local-nvme-ssd-*}"

INSTALL_GPU_DRIVER="${INSTALL_GPU_DRIVER:-1}"
DRIVER_BRANCH="${DRIVER_BRANCH:-prod}"
CUDA_INSTALLER_URL="${CUDA_INSTALLER_URL:-https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz}"

INSTALL_DOCKER="${INSTALL_DOCKER:-1}"
INSTALL_NVIDIA_CONTAINER_TOOLKIT="${INSTALL_NVIDIA_CONTAINER_TOOLKIT:-1}"
INSTALL_GIT="${INSTALL_GIT:-1}"
INSTALL_RIPGREP="${INSTALL_RIPGREP:-1}"
DOCKER_USERS="${DOCKER_USERS:-}"

# Fabric Manager is not needed on the GCP A2 Ultra VM shape I tested; NVLink
# and P2P come up healthy without it. Set to 1 only if your image/platform
# exposes manageable NVSwitch devices and you explicitly want it.
INSTALL_FABRIC_MANAGER="${INSTALL_FABRIC_MANAGER:-0}"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

on_error() {
  local exit_code=$?
  log "ERROR: bootstrap failed at line ${BASH_LINENO[0]} with exit code ${exit_code}"
  exit "$exit_code"
}
trap on_error ERR

apt_updated=0
apt_update_once() {
  if [[ "$apt_updated" -eq 0 ]]; then
    log "Updating apt package lists"
    apt-get update
    apt_updated=1
  fi
}

install_packages() {
  apt_update_once
  apt-get install -y --no-install-recommends "$@"
}

install_host_runtime_tools() {
  log "Installing host runtime tools"

  local packages=()
  if [[ "$INSTALL_DOCKER" == "1" ]]; then
    packages+=(docker.io)
  fi
  if [[ "$INSTALL_GIT" == "1" ]]; then
    packages+=(git)
  fi
  if [[ "$INSTALL_RIPGREP" == "1" ]]; then
    packages+=(ripgrep)
  fi

  if [[ "${#packages[@]}" -gt 0 ]]; then
    install_packages "${packages[@]}"
  fi

  if [[ "$INSTALL_DOCKER" == "1" ]] && command -v docker >/dev/null 2>&1; then
    systemctl enable --now docker.service || true

    for docker_user in $DOCKER_USERS; do
      if id "$docker_user" >/dev/null 2>&1; then
        usermod -aG docker "$docker_user"
        log "Added ${docker_user} to docker group"
      else
        log "Docker user ${docker_user} does not exist yet; skipping group add"
      fi
    done
  fi
}

discover_local_ssds() {
  shopt -s nullglob
  local devices=( $LOCAL_SSD_GLOB )
  shopt -u nullglob
  printf '%s\n' "${devices[@]}" | sort -V
}

setup_local_ssd_raid() {
  log "Setting up local SSD RAID0 storage"
  install_packages mdadm

  mapfile -t local_ssds < <(discover_local_ssds)
  if [[ "${#local_ssds[@]}" -eq 0 ]]; then
    log "No Google local NVMe SSDs found; skipping RAID setup"
    return 0
  fi

  log "Found ${#local_ssds[@]} local SSD device(s): ${local_ssds[*]}"
  mkdir -p "$MOUNT_POINT"

  if [[ ! -e "$RAID_DEVICE" ]]; then
    mdadm --assemble --scan || true
  fi

  if [[ ! -e "$RAID_DEVICE" ]]; then
    for dev in "${local_ssds[@]}"; do
      if findmnt --source "$dev" >/dev/null 2>&1; then
        log "$dev is already mounted; refusing to create RAID over mounted storage"
        return 1
      fi
    done

    if wipefs -n "${local_ssds[@]}" | grep -q .; then
      log "Existing filesystem/RAID signatures found on local SSDs."
      log "Set FORCE_RECREATE_LOCAL_SSD_RAID=1 to wipe and recreate them."
      if [[ "${FORCE_RECREATE_LOCAL_SSD_RAID:-0}" != "1" ]]; then
        return 1
      fi
      wipefs -a "${local_ssds[@]}"
    fi

    log "Creating ${RAID_DEVICE} as RAID0 with chunk ${RAID_CHUNK}"
    mdadm --create "$RAID_DEVICE" \
      --level=0 \
      --raid-devices="${#local_ssds[@]}" \
      --chunk="$RAID_CHUNK" \
      "${local_ssds[@]}"
  else
    log "${RAID_DEVICE} already exists; reusing it"
  fi

  if ! fs_type="$(blkid -s TYPE -o value "$RAID_DEVICE" 2>/dev/null)"; then
    log "Formatting ${RAID_DEVICE} as ext4"
    mkfs.ext4 -F -m 0 -L local-ssd-raid0 "$RAID_DEVICE"
    fs_type="ext4"
  fi

  if [[ "$fs_type" != "ext4" ]]; then
    log "${RAID_DEVICE} has filesystem type ${fs_type}; expected ext4"
    return 1
  fi

  local array_line array_uuid mdadm_changed=0
  array_line="$(mdadm --detail --scan "$RAID_DEVICE")"
  array_uuid="$(sed -n 's/.*UUID=//p' <<<"$array_line")"
  if [[ -n "$array_uuid" ]] && ! grep -q "$array_uuid" /etc/mdadm/mdadm.conf; then
    log "Persisting mdadm array metadata"
    printf '%s\n' "$array_line" >> /etc/mdadm/mdadm.conf
    mdadm_changed=1
  fi

  local fs_uuid
  fs_uuid="$(blkid -s UUID -o value "$RAID_DEVICE")"
  if grep -q "[[:space:]]${MOUNT_POINT}[[:space:]]" /etc/fstab; then
    sed -i "\|[[:space:]]${MOUNT_POINT}[[:space:]]|d" /etc/fstab
  fi
  printf 'UUID=%s %s ext4 discard,defaults,nofail 0 2\n' "$fs_uuid" "$MOUNT_POINT" >> /etc/fstab
  systemctl daemon-reload || true

  if [[ "$mdadm_changed" -eq 1 ]]; then
    update-initramfs -u || true
  fi

  if ! mountpoint -q "$MOUNT_POINT"; then
    log "Mounting ${RAID_DEVICE} at ${MOUNT_POINT}"
    mount "$MOUNT_POINT"
  fi

  chmod 0777 "$MOUNT_POINT"
  install -d -m 0777 \
    "$MOUNT_POINT/datasets" \
    "$MOUNT_POINT/checkpoints" \
    "$MOUNT_POINT/logs" \
    "$MOUNT_POINT/.cache" \
    "$MOUNT_POINT/.cache/huggingface" \
    "$MOUNT_POINT/.cache/torch"
  install -d -m 1777 "$MOUNT_POINT/tmp"

  if [[ -L "$DATA_LINK" ]]; then
    ln -sfn "$MOUNT_POINT" "$DATA_LINK"
  elif [[ ! -e "$DATA_LINK" ]]; then
    ln -s "$MOUNT_POINT" "$DATA_LINK"
  else
    log "${DATA_LINK} already exists and is not a symlink; leaving it untouched"
  fi

  cat >/etc/profile.d/training-node-data.sh <<EOF
export TRAINING_DATA_DIR="${DATA_LINK}"
export TMPDIR="${DATA_LINK}/tmp"
export XDG_CACHE_HOME="${DATA_LINK}/.cache"
export HF_HOME="${DATA_LINK}/.cache/huggingface"
export TORCH_HOME="${DATA_LINK}/.cache/torch"
EOF

  log "Storage ready: $(df -hT "$MOUNT_POINT" | tail -1)"
}

gpu_driver_is_ready() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

install_gpu_driver() {
  if [[ "$INSTALL_GPU_DRIVER" != "1" ]]; then
    log "INSTALL_GPU_DRIVER is not 1; skipping GPU driver install"
    return 0
  fi

  if ! lspci | grep -qi nvidia; then
    log "No NVIDIA PCI devices found; skipping GPU driver install"
    return 0
  fi

  if gpu_driver_is_ready; then
    log "NVIDIA driver is already working"
    return 0
  fi

  log "Installing NVIDIA driver with Google GPU installer (${DRIVER_BRANCH} branch)"
  install_packages curl pciutils python3 ca-certificates
  systemctl stop google-cloud-ops-agent || true

  local installer_dir="/var/tmp"
  if [[ -d "${MOUNT_POINT}/tmp" ]]; then
    installer_dir="${MOUNT_POINT}/tmp"
  fi
  mkdir -p "$installer_dir"

  curl -fL "$CUDA_INSTALLER_URL" --output "${installer_dir}/cuda_installer.pyz"
  python3 "${installer_dir}/cuda_installer.pyz" install_driver \
    --installation-mode=repo \
    --installation-branch="$DRIVER_BRANCH"

  # If the installer needed a kernel reboot, the VM may reboot before this point.
  systemctl start google-cloud-ops-agent || true
}

configure_nvidia_runtime() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi is not installed; skipping NVIDIA runtime configuration"
    return 0
  fi

  log "Configuring NVIDIA runtime services"
  modprobe nvidia || true
  modprobe nvidia_uvm || true

  if gpu_driver_is_ready; then
    nvidia-smi -pm 1 || true
    systemctl enable nvidia-persistenced.service || true
    systemctl restart nvidia-persistenced.service || true
  fi

  if [[ "$INSTALL_FABRIC_MANAGER" == "1" ]] && gpu_driver_is_ready; then
    local driver_version
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    log "Installing NVIDIA Fabric Manager ${driver_version}"
    apt_update_once
    apt-get install -yq "nvidia-fabricmanager=${driver_version}-1" || apt-get install -yq nvidia-fabricmanager
    if ! systemctl enable --now nvidia-fabricmanager.service; then
      log "Fabric Manager did not start; disabling failed service and continuing"
      systemctl disable nvidia-fabricmanager.service || true
      systemctl reset-failed nvidia-fabricmanager.service || true
    fi
  fi

  systemctl start google-cloud-ops-agent || true
}

configure_docker_gpu_runtime() {
  if [[ "$INSTALL_DOCKER" != "1" ]]; then
    log "INSTALL_DOCKER is not 1; skipping Docker GPU runtime configuration"
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    log "Docker is not installed; skipping Docker GPU runtime configuration"
    return 0
  fi

  if [[ "$INSTALL_NVIDIA_CONTAINER_TOOLKIT" != "1" ]]; then
    log "INSTALL_NVIDIA_CONTAINER_TOOLKIT is not 1; skipping NVIDIA container toolkit"
    return 0
  fi

  if ! gpu_driver_is_ready; then
    log "NVIDIA driver is not ready; skipping Docker GPU runtime configuration"
    return 0
  fi

  log "Installing and configuring NVIDIA container toolkit for Docker"
  apt_update_once
  if ! apt-cache policy nvidia-container-toolkit | grep -q 'Candidate: (none)'; then
    install_packages nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker.service || true
  else
    log "nvidia-container-toolkit is not available from configured apt repositories"
  fi
}

verify_node() {
  log "Verification summary"
  uname -a
  df -hT / "$MOUNT_POINT" || true
  mdadm --detail --prefer=by-id "$RAID_DEVICE" 2>/dev/null | sed -n '1,80p' || true

  if gpu_driver_is_ready; then
    nvidia-smi --query-gpu=index,name,persistence_mode,memory.total,memory.used,driver_version,compute_mode,mig.mode.current \
      --format=csv,noheader
    nvidia-smi topo -p2p rwnap || true
  else
    log "NVIDIA driver is not ready yet"
  fi

  if command -v git >/dev/null 2>&1; then
    git --version
  fi
  if command -v rg >/dev/null 2>&1; then
    rg --version | head -1
  fi
  if command -v docker >/dev/null 2>&1; then
    docker --version
    docker info --format 'Docker runtimes: {{json .Runtimes}}' 2>/dev/null || true
  fi
  if command -v nvidia-ctk >/dev/null 2>&1; then
    nvidia-ctk --version || true
  fi

  systemctl --failed --no-pager || true
}

main() {
  log "Starting training-node bootstrap"
  install_host_runtime_tools
  setup_local_ssd_raid
  install_gpu_driver
  configure_nvidia_runtime
  configure_docker_gpu_runtime
  apt-get clean || true
  verify_node
  log "Training-node bootstrap complete"
}

main "$@"
