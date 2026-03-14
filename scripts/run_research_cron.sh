#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/cron"
LOG_FILE="${LOG_DIR}/research_pipeline.log"
LOCK_FILE="${LOG_DIR}/research_pipeline.lock"
JOB_NAME="research_pipeline"
CURRENT_STEP="startup"
PYTHON_BIN=""
ALERT_SENT=0
LOCK_ACQUIRED=0
SKIPPED_DUE_TO_LOCK=0

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

send_failure_alert() {
  local failed_step="$1"
  local exit_code="$2"

  if [[ "${ALERT_SENT}" -eq 1 ]]; then
    return 0
  fi

  local alert_python=""
  if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
    alert_python="${PYTHON_BIN}"
  elif command -v python3 >/dev/null 2>&1; then
    alert_python="$(command -v python3)"
  else
    log_error "Unable to send failure alert because no Python interpreter is available."
    return 1
  fi

  log_info "Sending pipeline failure alert for step ${failed_step}."

  if "${alert_python}" -m src.notifications.pipeline_failure_alert \
    --job-name "${JOB_NAME}" \
    --failed-step "${failed_step}" \
    --exit-code "${exit_code}" \
    --log-file "${LOG_FILE}"; then
    ALERT_SENT=1
    log_info "Pipeline failure alert sent."
    return 0
  fi

  log_error "Pipeline failure alert send failed."
  return 1
}

on_exit() {
  local exit_code=$?

  if [[ "${exit_code}" -eq 0 ]]; then
    return 0
  fi

  if [[ "${SKIPPED_DUE_TO_LOCK}" -eq 1 ]]; then
    return 0
  fi

  log_error "Research cron run exiting with failure at step ${CURRENT_STEP}."
  send_failure_alert "${CURRENT_STEP}" "${exit_code}" || true

  trap - EXIT
  exit "${exit_code}"
}

trap on_exit EXIT

exec >>"${LOG_FILE}" 2>&1

CURRENT_STEP="project root change"
cd "${PROJECT_ROOT}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  SKIPPED_DUE_TO_LOCK=1
  log_warn "Research cron run skipped because another run is already in progress."
  exit 0
fi
LOCK_ACQUIRED=1

log_info "Research cron run started."
log_info "Project root: ${PROJECT_ROOT}"

CURRENT_STEP="virtual environment discovery"
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

CURRENT_STEP="virtual environment activation"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  log_error "Virtual environment activation script is missing: ${VENV_DIR}/bin/activate"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
log_info "Virtual environment activated from ${VENV_DIR}."

CURRENT_STEP="python binary validation"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  log_error "Python binary not found or not executable: ${PYTHON_BIN}"
  exit 1
fi

run_step() {
  local step_name="$1"
  shift

  CURRENT_STEP="${step_name}"
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

CURRENT_STEP="completed"
log_info "Research cron run finished successfully."
