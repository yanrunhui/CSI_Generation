#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_INPUT="${GENERATED_INPUT:-artifacts/generated/b3_scene_id_evaluation.pt}"
MAX_SAMPLES="${MAX_SAMPLES:-2048}"
PROJECTION_DIMENSIONS="${PROJECTION_DIMENSIONS:-32,64,128}"
PROJECTION_SEEDS="${PROJECTION_SEEDS:-0,1,2}"
RANDOM_PHASE_REPEATS="${RANDOM_PHASE_REPEATS:-5}"
MINIMUM_PER_MAP="${MINIMUM_PER_MAP:-8}"
MINIMUM_CONDITION_GROUP="${MINIMUM_CONDITION_GROUP:-6}"
MAXIMUM_CONDITION_GROUPS_PER_STRATUM="${MAXIMUM_CONDITION_GROUPS_PER_STRATUM:-3}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/global_phase_full_csi/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/global_phase_full_csi/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"
REPORT="${REPORT_DIR}/global_phase_full_csi.json"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Global-phase Full-CSI diagnostic completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Diagnostic failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

if [[ ! -f "${GENERATED_INPUT}" ]]; then
  printf 'Required file not found: %s\n' "${GENERATED_INPUT}" >&2
  exit 2
fi

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'generated_input=%s\n' "${GENERATED_INPUT}"
  printf 'report=%s\n' "${REPORT}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START global_phase_full_csi\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/diagnose_global_phase_full_csi.py \
  --generated "${GENERATED_INPUT}" \
  --maximum-samples "${MAX_SAMPLES}" \
  --projection-dimensions "${PROJECTION_DIMENSIONS}" \
  --projection-seeds "${PROJECTION_SEEDS}" \
  --random-phase-repeats "${RANDOM_PHASE_REPEATS}" \
  --minimum-per-map "${MINIMUM_PER_MAP}" \
  --minimum-condition-group "${MINIMUM_CONDITION_GROUP}" \
  --maximum-condition-groups-per-stratum "${MAXIMUM_CONDITION_GROUPS_PER_STRATUM}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --device cuda \
  --output "${REPORT}" \
  2>&1 | tee "${LOG_DIR}/global_phase_full_csi.log"

printf '[%s] DONE  global_phase_full_csi\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/summarize_global_phase_full_csi.py "${REPORT}" \
  | tee "${REPORT_DIR}/summary.json"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${REPORT_DIR}/summary.json" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
