#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"

if [[ -z "${LINK_CHECKPOINT:-}" ]]; then
  LINK_CHECKPOINT="$({
    find artifacts/checkpoints/aod_link_global_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
if [[ -z "${ARTIFACT:-}" ]]; then
  ARTIFACT="$({
    find artifacts/generated/csi_absolute_quality -maxdepth 1 -type f \
      -name 'link_global_full_test_*.pt' -print 2>/dev/null || true
  } | sort | tail -1)"
fi
for required in "${ARTIFACT}" "${LINK_CHECKPOINT}" "${CACHE}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_tolerance_tradeoff/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_tolerance_tradeoff/${RUN_TAG}}"
REPORT="${REPORT_DIR}/aod_tolerance_tradeoff.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] AoD tolerance tradeoff completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] AoD tolerance tradeoff failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'artifact=%s\n' "${ARTIFACT}"
  printf 'map_ids=%s\n' "${MAP_IDS:-10}"
  printf 'path_counts=%s\n' "${PATH_COUNTS:-5}"
  printf 'tolerances_deg=%s\n' "${TOLERANCES_DEG:-0,5,10,15}"
  printf 'candidate_multipliers=%s\n' "${CANDIDATE_MULTIPLIERS:-5,10,20}"
  printf 'requested_samples=%s\n' "${REQUESTED_SAMPLES:-8}"
  printf 'report=%s\n' "${REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START aod_tolerance_tradeoff\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/diagnose_aod_tolerance_tradeoff.py \
  --artifact "${ARTIFACT}" \
  --link-checkpoint "${LINK_CHECKPOINT}" \
  --cache "${CACHE}" \
  --output "${REPORT}" \
  --map-ids "${MAP_IDS:-10}" \
  --path-counts "${PATH_COUNTS:-5}" \
  --target-angles="${TARGET_ANGLES:--150,-90,-30,30,90,150}" \
  --tolerances-deg "${TOLERANCES_DEG:-0,5,10,15}" \
  --candidate-multipliers "${CANDIDATE_MULTIPLIERS:-5,10,20}" \
  --requested-samples "${REQUESTED_SAMPLES:-8}" \
  --pool-trials "${POOL_TRIALS:-200}" \
  --minimum-fidelity-samples "${MINIMUM_FIDELITY_SAMPLES:-8}" \
  --maximum-fidelity-samples "${MAXIMUM_FIDELITY_SAMPLES:-64}" \
  --device cuda \
  2>&1 | tee "${LOG_DIR}/aod_tolerance_tradeoff.log"
printf '[%s] DONE  aod_tolerance_tradeoff\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/summarize_aod_tolerance_tradeoff.py "${REPORT}" \
  | tee "${SUMMARY}"
printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
