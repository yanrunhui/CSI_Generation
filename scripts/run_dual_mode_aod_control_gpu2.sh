#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
B3_CHECKPOINT="${B3_CHECKPOINT:-artifacts/checkpoints/b3_seen_map_baseline/best.pt}"
SUPPORT_CACHE="${SUPPORT_CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"

if [[ -z "${DP_CHECKPOINT:-}" ]]; then
  DP_CHECKPOINT="$({
    find artifacts/checkpoints/delay_power_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
if [[ -z "${LINK_CHECKPOINT:-}" ]]; then
  LINK_CHECKPOINT="$({
    find artifacts/checkpoints/aod_link_global_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/dual_mode_aod_control/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/dual_mode_aod_control/${RUN_TAG}}"
REPORT="${REPORT_DIR}/dual_mode_aod_control.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

for required in \
  "${B3_CHECKPOINT}" "${DP_CHECKPOINT}" "${LINK_CHECKPOINT}" "${SUPPORT_CACHE}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty auto-detected path>}" >&2
    exit 2
  fi
done

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Dual-mode AoD control diagnostic completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Dual-mode AoD control diagnostic failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'b3_checkpoint=%s\n' "${B3_CHECKPOINT}"
  printf 'dp_checkpoint=%s\n' "${DP_CHECKPOINT}"
  printf 'link_checkpoint=%s\n' "${LINK_CHECKPOINT}"
  printf 'support_cache=%s\n' "${SUPPORT_CACHE}"
  printf 'map_id=%s\n' "${MAP_ID:-10}"
  printf 'report=%s\n' "${REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START dual_mode_aod_control\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/diagnose_dual_mode_aod_control.py \
  --b3-checkpoint "${B3_CHECKPOINT}" \
  --dp-checkpoint "${DP_CHECKPOINT}" \
  --link-checkpoint "${LINK_CHECKPOINT}" \
  --support-cache "${SUPPORT_CACHE}" \
  --output "${REPORT}" \
  --map-id "${MAP_ID:-10}" \
  --samples-per-condition "${SAMPLES_PER_CONDITION:-128}" \
  --generation-seeds "${GENERATION_SEEDS:-2026,2027,2028}" \
  --target-angles="${TARGET_ANGLES:--180,-135,-90,-45,0,45,90,135}" \
  --minimum-mean-concentration "${MINIMUM_MEAN_CONCENTRATION:-0.10}" \
  --strongest-near-tie-margin "${STRONGEST_NEAR_TIE_MARGIN:-0.02}" \
  --device cuda \
  2>&1 | tee "${LOG_DIR}/dual_mode_aod_control.log"
printf '[%s] DONE  dual_mode_aod_control\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/summarize_dual_mode_aod_control.py "${REPORT}" \
  | tee "${SUMMARY}"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
