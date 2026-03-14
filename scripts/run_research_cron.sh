#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/cron"
LOG_FILE="${LOG_DIR}/research_pipeline.log"
LOCK_FILE="${LOG_DIR}/research_pipeline.lock"

mkdir -p "${LOG_DIR}"

timestamp() {
  date -u "+%Y-%m-%d %H:%M:%S UTC"
}

log_info() {
  printf '[%s] INFO  %s\n' "$(timestamp)" "$*"
}

log_warn() {
  printf '[%s] WARN  %s\n' "$(timestamp)" "$*"
}

log_error() {
  printf '[%s] ERROR %s\n' "$(timestamp)" "$*" >&2
}

exec >>"${LOG_FILE}" 2>&1

cd "${PROJECT_ROOT}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  log_warn "Research cron run skipped because another run is already in progress."
  exit 0
fi

log_info "Research cron run started."
log_info "Project root: ${PROJECT_ROOT}"

VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  if [[ -d "${PROJECT_ROOT}/venv" ]]; then
    VENV_DIR="${PROJECT_ROOT}/venv"
    log_warn "Expected .venv was not found. Falling back to venv."
  else
    log_error "Virtual environment directory not found at ${PROJECT_ROOT}/.venv or ${PROJECT_ROOT}/venv."
    exit 1
  fi
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  log_error "Virtual environment activation script is missing: ${VENV_DIR}/bin/activate"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
log_info "Virtual environment activated from ${VENV_DIR}."

PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  log_error "Python binary not found or not executable: ${PYTHON_BIN}"
  exit 1
fi

run_step() {
  local step_name="$1"
  shift

  log_info "Starting ${step_name}."

  if "$@"; then
    log_info "Completed ${step_name}."
    return 0
  fi

  local exit_code=$?
  log_error "${step_name} failed with exit code ${exit_code}."
  return "${exit_code}"
}

if ! run_step "research comparison pipeline" "${PYTHON_BIN}" -m src.research.run_comparison_pipeline; then
  log_error "Research cron run failed before notifier execution."
  exit 1
fi

if ! run_step "research observational notifier" "${PYTHON_BIN}" -m src.notifications.research_observational_notifier; then
  log_error "Research cron run failed during notifier execution."
  exit 1
fi

log_info "Research cron run finished successfully."
