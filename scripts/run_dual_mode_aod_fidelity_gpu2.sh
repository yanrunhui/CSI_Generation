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
    find artifacts/generated/aod_link_global -maxdepth 1 -type f \
      -name 'evaluation_*.pt' -print 2>/dev/null || true
  } | sort | tail -1)"
fi
if [[ -z "${ARTIFACT}" || ! -f "${ARTIFACT}" ]]; then
  printf 'Link-global evaluation artifact not found: %s\n' "${ARTIFACT:-<empty>}" >&2
  exit 2
fi
if [[ -z "${LINK_CHECKPOINT}" || ! -f "${LINK_CHECKPOINT}" ]]; then
  printf 'Link-global checkpoint not found: %s\n' "${LINK_CHECKPOINT:-<empty>}" >&2
  exit 2
fi
if [[ ! -f "${CACHE}" ]]; then
  printf 'B3 cache not found: %s\n' "${CACHE}" >&2
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/dual_mode_aod_fidelity/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/dual_mode_aod_fidelity/${RUN_TAG}}"
REPORT="${REPORT_DIR}/dual_mode_aod_fidelity.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Dual-mode AoD fidelity completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Dual-mode AoD fidelity failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'artifact=%s\n' "${ARTIFACT}"
  printf 'link_checkpoint=%s\n' "${LINK_CHECKPOINT}"
  printf 'cache=%s\n' "${CACHE}"
  printf 'map_ids=%s\n' "${MAP_IDS:-10}"
  printf 'path_counts=%s\n' "${PATH_COUNTS:-5}"
  printf 'report=%s\n' "${REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START dual_mode_aod_fidelity\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/evaluate_dual_mode_aod_fidelity.py \
  --artifact "${ARTIFACT}" \
  --link-checkpoint "${LINK_CHECKPOINT}" \
  --cache "${CACHE}" \
  --output "${REPORT}" \
  --map-ids "${MAP_IDS:-10}" \
  --path-counts "${PATH_COUNTS:-5}" \
  --target-angles="${TARGET_ANGLES:--150,-90,-30,30,90,150}" \
  --angle-half-width-deg "${ANGLE_HALF_WIDTH_DEG:-15}" \
  --minimum-group-samples "${MINIMUM_GROUP_SAMPLES:-24}" \
  --maximum-group-samples "${MAXIMUM_GROUP_SAMPLES:-128}" \
  --small-projection-threshold-deg "${SMALL_PROJECTION_THRESHOLD_DEG:-45}" \
  --device cuda \
  2>&1 | tee "${LOG_DIR}/dual_mode_aod_fidelity.log"
printf '[%s] DONE  dual_mode_aod_fidelity\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/summarize_dual_mode_aod_fidelity.py "${REPORT}" \
  | tee "${SUMMARY}"
printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
