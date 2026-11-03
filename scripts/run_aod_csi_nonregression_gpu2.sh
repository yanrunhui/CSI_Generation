#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -z "${LINK_ARTIFACT:-}" ]]; then
  LINK_ARTIFACT="$(
    find artifacts/generated/aod_link_global -maxdepth 1 -type f \
      -name 'evaluation_*.pt' -print 2>/dev/null | sort | tail -1
  )"
fi
if [[ -z "${RANGE40_ARTIFACT:-}" ]]; then
  RANGE40_ARTIFACT="$(
    find artifacts/generated/aod_range_limit -maxdepth 1 -type f \
      -name 'evaluation_range40_*.pt' -print 2>/dev/null | sort | tail -1
  )"
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_csi_nonregression/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_csi_nonregression/${RUN_TAG}}"
REPORT="${REPORT_DIR}/aod_csi_nonregression.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] AoD CSI non-regression diagnostic completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] AoD CSI non-regression diagnostic failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

for required in "${LINK_ARTIFACT}" "${RANGE40_ARTIFACT}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty auto-detected path>}" >&2
    exit 2
  fi
done

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'link_artifact=%s\n' "${LINK_ARTIFACT}"
  printf 'range40_artifact=%s\n' "${RANGE40_ARTIFACT}"
  printf 'report=%s\n' "${REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START aod_csi_nonregression\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/diagnose_aod_csi_nonregression.py \
  --candidate-artifact "${LINK_ARTIFACT}" \
  --candidate-label link_global \
  --comparison "per_path_range40=${RANGE40_ARTIFACT}" \
  --output "${REPORT}" \
  --device cuda \
  --relative-noninferiority-margin 0.05 \
  2>&1 | tee "${LOG_DIR}/aod_csi_nonregression.log"
"${PYTHON_BIN}" scripts/summarize_aod_csi_nonregression.py "${REPORT}" \
  | tee "${SUMMARY}"
printf '[%s] DONE  aod_csi_nonregression\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
