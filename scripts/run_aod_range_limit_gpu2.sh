#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"
CONFIG="${CONFIG:-configs/aod_residual_range40.yaml}"
SEED="${SEED:-2026}"

if [[ -z "${BASE_AOD_ARTIFACT:-}" ]]; then
  BASE_AOD_ARTIFACT="$(
    find artifacts/generated/aod_residual -maxdepth 1 -type f \
      -name 'evaluation_*.pt' -print 2>/dev/null | sort | tail -1
  )"
fi
if [[ -z "${BASE_AOD_CHECKPOINT:-}" ]]; then
  BASE_AOD_CHECKPOINT="$(
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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints/aod_residual_range40/${RUN_TAG}}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_range_limit/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_range_limit/${RUN_TAG}}"
ARTIFACT="${ARTIFACT:-artifacts/generated/aod_range_limit/evaluation_range40_${RUN_TAG}.pt}"
MODEL_REPORT="${REPORT_DIR}/range40_model.json"
COMPARISON_REPORT="${REPORT_DIR}/range_limit_comparison.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${CHECKPOINT_DIR}" "${REPORT_DIR}" "${LOG_DIR}" "$(dirname -- "${ARTIFACT}")"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] AoD range-limit experiment completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] AoD range-limit experiment failed with exit code %s.\n' \
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

for required in \
  "${CACHE}" "${CONFIG}" "${DP_CHECKPOINT}" \
  "${BASE_AOD_ARTIFACT}" "${BASE_AOD_CHECKPOINT}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty auto-detected path>}" >&2
    exit 2
  fi
done

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'cache=%s\n' "${CACHE}"
  printf 'dp_checkpoint=%s\n' "${DP_CHECKPOINT}"
  printf 'base_aod_artifact=%s\n' "${BASE_AOD_ARTIFACT}"
  printf 'base_aod_checkpoint=%s\n' "${BASE_AOD_CHECKPOINT}"
  printf 'wide_config=%s\n' "${CONFIG}"
  printf 'wide_checkpoint_dir=%s\n' "${CHECKPOINT_DIR}"
  printf 'wide_artifact=%s\n' "${ARTIFACT}"
  printf 'comparison_report=%s\n' "${COMPARISON_REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

run_stage 01_train_range40 \
  "${PYTHON_BIN}" scripts/train_aod_residual.py \
    --cache "${CACHE}" \
    --dp-checkpoint "${DP_CHECKPOINT}" \
    --config "${CONFIG}" \
    --checkpoint-dir "${CHECKPOINT_DIR}"

run_stage 02_evaluate_range40 \
  "${PYTHON_BIN}" scripts/evaluate_aod_residual.py \
    --checkpoint "${CHECKPOINT_DIR}/best.pt" \
    --dp-checkpoint "${DP_CHECKPOINT}" \
    --cache "${CACHE}" \
    --output-artifact "${ARTIFACT}" \
    --output-report "${MODEL_REPORT}" \
    --device cuda \
    --seed "${SEED}"

run_stage 03_compare_range40_vs_range20 \
  "${PYTHON_BIN}" scripts/compare_aod_residual_variants.py \
    --candidate-artifact "${ARTIFACT}" \
    --candidate-checkpoint "${CHECKPOINT_DIR}/best.pt" \
    --baseline-artifact "${BASE_AOD_ARTIFACT}" \
    --baseline-checkpoint "${BASE_AOD_CHECKPOINT}" \
    --candidate-label per_path_range40 \
    --baseline-label per_path_range20 \
    --experiment range_limit \
    --output "${COMPARISON_REPORT}" \
    --device cuda

"${PYTHON_BIN}" scripts/summarize_aod_variant_comparison.py \
  "${COMPARISON_REPORT}" | tee "${SUMMARY}"

printf '\nArtifact: %s\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${ARTIFACT}" "${COMPARISON_REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
