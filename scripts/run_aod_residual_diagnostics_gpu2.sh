#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"
SEED="${SEED:-9026}"
EPOCHS="${EPOCHS:-30}"
TRAIN_BATCHES_PER_EPOCH="${TRAIN_BATCHES_PER_EPOCH:-400}"
VALIDATION_BATCHES_PER_EPOCH="${VALIDATION_BATCHES_PER_EPOCH:-100}"

if [[ -z "${AOD_ARTIFACT:-}" ]]; then
  AOD_ARTIFACT="$(
    find artifacts/generated/aod_residual -maxdepth 1 -type f \
      -name 'evaluation_*.pt' -print 2>/dev/null | sort | tail -1
  )"
fi
if [[ -z "${AOD_CHECKPOINT:-}" ]]; then
  AOD_CHECKPOINT="$(
    find artifacts/checkpoints/aod_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null | sort | tail -1
  )"
fi
if [[ -z "${DP_CHECKPOINT:-}" ]]; then
  DP_CHECKPOINT="$(
    find artifacts/checkpoints/delay_power_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null | sort | tail -1
  )"
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_residual_diagnostics/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_residual_diagnostics/${RUN_TAG}}"
CONSTANT_ARTIFACT="${CONSTANT_ARTIFACT:-artifacts/generated/aod_residual/map_constant_offset_${RUN_TAG}.pt}"
REPORT="${REPORT_DIR}/aod_residual_structure.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "$(dirname -- "${CONSTANT_ARTIFACT}")"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] AoD residual diagnostics completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] AoD residual diagnostics failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

for required in "${AOD_ARTIFACT}" "${AOD_CHECKPOINT}" "${DP_CHECKPOINT}" "${CACHE}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty auto-detected path>}" >&2
    exit 2
  fi
done

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'aod_artifact=%s\n' "${AOD_ARTIFACT}"
  printf 'aod_checkpoint=%s\n' "${AOD_CHECKPOINT}"
  printf 'dp_checkpoint=%s\n' "${DP_CHECKPOINT}"
  printf 'cache=%s\n' "${CACHE}"
  printf 'constant_offset_epochs=%s\n' "${EPOCHS}"
  printf 'train_batches_per_epoch=%s\n' "${TRAIN_BATCHES_PER_EPOCH}"
  printf 'validation_batches_per_epoch=%s\n' "${VALIDATION_BATCHES_PER_EPOCH}"
  printf 'report=%s\n' "${REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'constant_artifact=%s\n' "${CONSTANT_ARTIFACT}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START aod_residual_structure_and_constant_offset\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/diagnose_aod_residual_structure.py \
  --artifact "${AOD_ARTIFACT}" \
  --checkpoint "${AOD_CHECKPOINT}" \
  --dp-checkpoint "${DP_CHECKPOINT}" \
  --cache "${CACHE}" \
  --output "${REPORT}" \
  --output-constant-artifact "${CONSTANT_ARTIFACT}" \
  --device cuda \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --train-batches-per-epoch "${TRAIN_BATCHES_PER_EPOCH}" \
  --validation-batches-per-epoch "${VALIDATION_BATCHES_PER_EPOCH}" \
  2>&1 | tee "${LOG_DIR}/01_aod_residual_structure.log"
"${PYTHON_BIN}" scripts/summarize_aod_residual_structure.py "${REPORT}" \
  | tee "${SUMMARY}"
printf '[%s] DONE  aod_residual_structure_and_constant_offset\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

printf '\nReport: %s\nSummary: %s\nConstant artifact: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${CONSTANT_ARTIFACT}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
printf 'For downstream CSI/PDP diagnostics, use GENERATED_INPUT=%s.\n' \
  "${CONSTANT_ARTIFACT}" | tee -a "${QUEUE_LOG}"
