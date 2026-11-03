#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"
CONFIG="${CONFIG:-configs/aod_residual.yaml}"
SEED="${SEED:-2026}"
if [[ -z "${DP_CHECKPOINT:-}" ]]; then
  DP_CHECKPOINT="$(find artifacts/checkpoints/delay_power_residual -mindepth 2 -maxdepth 2 -type f -name best.pt -print 2>/dev/null | sort | tail -1)"
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints/aod_residual/${RUN_TAG}}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_residual/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_residual/${RUN_TAG}}"
ARTIFACT="${ARTIFACT:-artifacts/generated/aod_residual/evaluation_${RUN_TAG}.pt}"
mkdir -p "${CHECKPOINT_DIR}" "${REPORT_DIR}" "${LOG_DIR}" "$(dirname -- "${ARTIFACT}")"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"
REPORT="${REPORT_DIR}/aod_residual.json"
SUMMARY="${REPORT_DIR}/summary.json"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] AoD residual experiment completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] AoD residual experiment failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

for required in "${CACHE}" "${DP_CHECKPOINT}" "${CONFIG}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty DP_CHECKPOINT>}" >&2
    exit 2
  fi
done

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'cache=%s\n' "${CACHE}"
  printf 'dp_checkpoint=%s\n' "${DP_CHECKPOINT}"
  printf 'config=%s\n' "${CONFIG}"
  printf 'checkpoint_dir=%s\n' "${CHECKPOINT_DIR}"
  printf 'artifact=%s\n' "${ARTIFACT}"
  printf 'report=%s\n' "${REPORT}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START 01_train_aod_residual\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/train_aod_residual.py \
  --cache "${CACHE}" \
  --dp-checkpoint "${DP_CHECKPOINT}" \
  --config "${CONFIG}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  2>&1 | tee "${LOG_DIR}/01_train_aod_residual.log"
printf '[%s] DONE  01_train_aod_residual\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

printf '\n[%s] START 02_evaluate_aod_residual\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/evaluate_aod_residual.py \
  --checkpoint "${CHECKPOINT_DIR}/best.pt" \
  --dp-checkpoint "${DP_CHECKPOINT}" \
  --cache "${CACHE}" \
  --output-artifact "${ARTIFACT}" \
  --output-report "${REPORT}" \
  --device cuda \
  --seed "${SEED}" \
  2>&1 | tee "${LOG_DIR}/02_evaluate_aod_residual.log"
"${PYTHON_BIN}" scripts/summarize_aod_residual.py "${REPORT}" | tee "${SUMMARY}"
printf '[%s] DONE  02_evaluate_aod_residual\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

printf '\nArtifact: %s\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${ARTIFACT}" "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
printf 'Independent diagnostics should use GENERATED_INPUT=%s.\n' \
  "${ARTIFACT}" | tee -a "${QUEUE_LOG}"
