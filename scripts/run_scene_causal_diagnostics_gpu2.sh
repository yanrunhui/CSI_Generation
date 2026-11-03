#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_INPUT="${GENERATED_INPUT:-artifacts/generated/b3_scene_id_evaluation.pt}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.yaml}"
MAX_SYMMETRIC_SAMPLES="${MAX_SYMMETRIC_SAMPLES:-2048}"
MAX_COHERENCE_SAMPLES="${MAX_COHERENCE_SAMPLES:-1024}"
MAX_DONOR_DISTANCE="${MAX_DONOR_DISTANCE:-0.1}"
PROJECTION_DIMENSIONS="${PROJECTION_DIMENSIONS:-32,64,128}"
PROJECTION_SEEDS="${PROJECTION_SEEDS:-0,1,2}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/scene_causal_diagnostics/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/scene_causal_diagnostics/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"
SYMMETRIC_REPORT="${REPORT_DIR}/symmetric_counterfactual.json"
COHERENCE_REPORT="${REPORT_DIR}/complex_csi_coherence.json"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Both causal diagnostics completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Queue failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

for required in "${GENERATED_INPUT}" "${DATA_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required}" >&2
    exit 2
  fi
done

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'generated_input=%s\n' "${GENERATED_INPUT}"
  printf 'report_dir=%s\n' "${REPORT_DIR}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

run_stage 01_symmetric_counterfactual \
  "${PYTHON_BIN}" scripts/diagnose_symmetric_counterfactual.py \
  --generated "${GENERATED_INPUT}" \
  --data-config "${DATA_CONFIG}" \
  --maximum-samples "${MAX_SYMMETRIC_SAMPLES}" \
  --maximum-donor-distance "${MAX_DONOR_DISTANCE}" \
  --device cuda \
  --output "${SYMMETRIC_REPORT}"

run_stage 02_complex_csi_coherence \
  "${PYTHON_BIN}" scripts/diagnose_complex_csi_coherence.py \
  --generated "${GENERATED_INPUT}" \
  --data-config "${DATA_CONFIG}" \
  --maximum-samples "${MAX_COHERENCE_SAMPLES}" \
  --projection-dimensions "${PROJECTION_DIMENSIONS}" \
  --projection-seeds "${PROJECTION_SEEDS}" \
  --device cuda \
  --output "${COHERENCE_REPORT}"

"${PYTHON_BIN}" scripts/summarize_scene_causal_diagnostics.py \
  "${SYMMETRIC_REPORT}" "${COHERENCE_REPORT}" \
  | tee "${REPORT_DIR}/summary.json"

printf '\nReports: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT_DIR}" "${REPORT_DIR}/summary.json" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
