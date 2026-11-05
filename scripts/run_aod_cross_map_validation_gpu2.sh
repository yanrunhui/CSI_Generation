#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${LINK_CHECKPOINT:-}" ]]; then
  LINK_CHECKPOINT="$({
    find artifacts/checkpoints/aod_link_global_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
if [[ -z "${CACHE:-}" ]]; then
  CACHE="$({
    find artifacts/generated/delay_power_residual -maxdepth 1 -type f \
      -name 'b3_cache_seed*.pt' -print 2>/dev/null || true
  } | sort | tail -1)"
fi

ARTIFACT_LIST=()
if [[ -n "${ARTIFACTS:-}" ]]; then
  IFS=':' read -r -a ARTIFACT_LIST <<< "${ARTIFACTS}"
elif [[ -n "${ARTIFACT:-}" ]]; then
  ARTIFACT_LIST=("${ARTIFACT}")
else
  while IFS= read -r artifact; do
    ARTIFACT_LIST+=("${artifact}")
  done < <(
    find artifacts/generated/csi_absolute_quality -maxdepth 1 -type f \
      -name 'link_global_full_test_*.pt' -print 2>/dev/null | sort | tail -1
  )
fi

if [[ ${#ARTIFACT_LIST[@]} -eq 0 ]]; then
  printf 'No evaluation artifact found. Set ARTIFACT or colon-separated ARTIFACTS.\n' >&2
  exit 2
fi
for required in "${LINK_CHECKPOINT}" "${CACHE}" "${ARTIFACT_LIST[@]}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty>}" >&2
    exit 2
  fi
done

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_cross_map_validation/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_cross_map_validation/${RUN_TAG}}"
REPORT="${REPORT_DIR}/aod_cross_map_validation.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Cross-Map AoD validation completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Cross-Map AoD validation failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'artifacts=%s\n' "${ARTIFACT_LIST[*]}"
  printf 'map_ids=%s\n' "${MAP_IDS:-all}"
  printf 'path_counts=%s\n' "${PATH_COUNTS:-5}"
  printf 'modes=%s\n' "${MODES:-mean}"
  printf 'target_angles=%s\n' "${TARGET_ANGLES:--150,-90,-30,30,90,150}"
  printf 'tolerances_deg=%s\n' "${TOLERANCES_DEG:-10}"
  printf 'candidate_multipliers=%s\n' "${CANDIDATE_MULTIPLIERS:-10,20,40}"
  printf 'report=%s\nsummary=%s\nlog_dir=%s\n' \
    "${REPORT}" "${SUMMARY}" "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

AGGREGATE_ARGUMENTS=()
artifact_number=0
for artifact in "${ARTIFACT_LIST[@]}"; do
  artifact_number=$((artifact_number + 1))
  label="seed_${artifact_number}_$(basename "${artifact}" .pt)"
  partial_report="${REPORT_DIR}/${label}.json"
  printf '\n[%s] START %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${label}" | tee -a "${QUEUE_LOG}"
  "${PYTHON_BIN}" scripts/diagnose_aod_tolerance_tradeoff.py \
    --artifact "${artifact}" \
    --link-checkpoint "${LINK_CHECKPOINT}" \
    --cache "${CACHE}" \
    --output "${partial_report}" \
    --map-ids "${MAP_IDS:-all}" \
    --path-counts "${PATH_COUNTS:-5}" \
    --modes "${MODES:-mean}" \
    --target-angles="${TARGET_ANGLES:--150,-90,-30,30,90,150}" \
    --tolerances-deg "${TOLERANCES_DEG:-10}" \
    --candidate-multipliers "${CANDIDATE_MULTIPLIERS:-10,20,40}" \
    --requested-samples "${REQUESTED_SAMPLES:-8}" \
    --pool-trials "${POOL_TRIALS:-200}" \
    --minimum-fidelity-samples "${MINIMUM_FIDELITY_SAMPLES:-8}" \
    --maximum-fidelity-samples "${MAXIMUM_FIDELITY_SAMPLES:-32}" \
    --swd-projections "${SWD_PROJECTIONS:-64}" \
    --projection-dimension "${PROJECTION_DIMENSION:-64}" \
    --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
    --seed "$(( ${SEED:-2026} + artifact_number - 1 ))" \
    --device cuda \
    2>&1 | tee "${LOG_DIR}/${label}.log"
  printf '[%s] DONE  %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${label}" | tee -a "${QUEUE_LOG}"
  AGGREGATE_ARGUMENTS+=(--report "${label}=${partial_report}")
done

printf '\n[%s] START aggregate\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/aggregate_aod_cross_map_validation.py \
  "${AGGREGATE_ARGUMENTS[@]}" \
  --output "${REPORT}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
  --maximum-relative-degradation "${MAXIMUM_RELATIVE_DEGRADATION:-0.05}" \
  2>&1 | tee "${LOG_DIR}/aggregate.log"
"${PYTHON_BIN}" scripts/summarize_aod_cross_map_validation.py "${REPORT}" \
  | tee "${SUMMARY}"
printf '[%s] DONE  aggregate\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
