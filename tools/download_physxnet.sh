#!/usr/bin/env bash
# Download the PhysXNet dataset with bounded retries.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="${PHYSXNET_DIR:-${PROJECT_ROOT}/dataset/PhysXNet}"
REPO_ID="${PHYSXNET_REPO_ID:-Caoza/PhysX-3D}"
HF_BIN="${HF_BIN:-hf}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
RETRY_BASE_DELAY="${RETRY_BASE_DELAY:-5}"
RETRY_MAX_DELAY="${RETRY_MAX_DELAY:-300}"
LOG_FILE="${PHYSXNET_LOG:-${PROJECT_ROOT}/download_physxnet.log}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "${LOCAL_DIR}"

log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${message}" | tee -a "${LOG_FILE}"
}

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    log "Download attempt ${attempt}/${MAX_ATTEMPTS}: ${REPO_ID} -> ${LOCAL_DIR}"
    set +e
    "${HF_BIN}" download \
        --repo-type dataset \
        --local-dir "${LOCAL_DIR}" \
        "${REPO_ID}" 2>&1 | tee -a "${LOG_FILE}"
    exit_code=${PIPESTATUS[0]}
    set -e

    if [[ ${exit_code} -eq 0 ]]; then
        log "Download completed successfully."
        exit 0
    fi
    if [[ ${attempt} -eq ${MAX_ATTEMPTS} ]]; then
        break
    fi

    wait_time=$((RETRY_BASE_DELAY * attempt))
    if [[ ${wait_time} -gt ${RETRY_MAX_DELAY} ]]; then
        wait_time=${RETRY_MAX_DELAY}
    fi
    log "Download failed with exit code ${exit_code}; retrying in ${wait_time}s."
    sleep "${wait_time}"
done

log "Download failed after ${MAX_ATTEMPTS} attempts."
exit 1
