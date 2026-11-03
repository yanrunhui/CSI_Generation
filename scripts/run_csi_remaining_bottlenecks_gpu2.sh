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
    find artifacts/generated/csi_absolute_quality -maxdepth 1 -type f \
      -name 'link_global_full_test_*.pt' -print 2>/dev/null | sort | tail -1
  )"
fi
if [[ -z "${LINK_ARTIFACT}" || ! -f "${LINK_ARTIFACT}" ]]; then
  printf 'Full-test Link-global artifact not found: %s\n' "${LINK_ARTIFACT:-<empty>}" >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/csi_remaining_bottlenecks/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/csi_remaining_bottlenecks/${RUN_TAG}}"
REPORT="${REPORT_DIR}/csi_remaining_bottlenecks.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Remaining CSI bottleneck diagnostic completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Remaining CSI bottleneck diagnostic failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'link_artifact=%s\n' "${LINK_ARTIFACT}"
  printf 'report=%s\n' "${REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START csi_remaining_bottlenecks\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/diagnose_csi_remaining_bottlenecks.py \
  --artifact "${LINK_ARTIFACT}" \
  --output "${REPORT}" \
  --data-config configs/data.yaml \
  --device cuda \
  --seed "${EVALUATION_SEED:-2026}" \
  --frequency-lags "${FREQUENCY_LAGS:-1,2,4,8,16,32}" \
  --maximum-counterfactual-distance "${MAXIMUM_COUNTERFACTUAL_DISTANCE:-1.0}" \
  --maximum-counterfactual-samples "${MAXIMUM_COUNTERFACTUAL_SAMPLES:-4096}" \
  --batch-size "${BATCH_SIZE:-256}" \
  2>&1 | tee "${LOG_DIR}/csi_remaining_bottlenecks.log"
"${PYTHON_BIN}" scripts/summarize_csi_remaining_bottlenecks.py "${REPORT}" \
  | tee "${SUMMARY}"
printf '[%s] DONE  csi_remaining_bottlenecks\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
