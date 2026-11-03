#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_INPUT="${GENERATED_INPUT:-artifacts/generated/b3_scene_id_evaluation.pt}"
MINIMUM_PER_MAP="${MINIMUM_PER_MAP:-12}"
MINIMUM_LOCAL_GROUP="${MINIMUM_LOCAL_GROUP:-4}"
MAXIMUM_LOCAL_GROUPS_PER_STRATUM="${MAXIMUM_LOCAL_GROUPS_PER_STRATUM:-4}"
MINIMUM_COMMON_PATH_COUNT="${MINIMUM_COMMON_PATH_COUNT:-4}"
PROJECTIONS="${PROJECTIONS:-128}"
PROJECTION_SEEDS="${PROJECTION_SEEDS:-0,1,2}"
WEIGHTED_QUANTILES="${WEIGHTED_QUANTILES:-512}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
SEED="${SEED:-2026}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/per_map_power_aod_fidelity/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/per_map_power_aod_fidelity/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"
REPORT="${REPORT_DIR}/per_map_power_aod_fidelity.json"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Per-Map Power-AoD fidelity diagnostic completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Diagnostic failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

if [[ ! -f "${GENERATED_INPUT}" ]]; then
  printf 'Required generated artifact not found: %s\n' "${GENERATED_INPUT}" >&2
  exit 2
fi

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'generated_input=%s\n' "${GENERATED_INPUT}"
  printf 'report=%s\n' "${REPORT}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START per_map_power_aod_fidelity\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/diagnose_per_map_power_aod_fidelity.py \
  --generated "${GENERATED_INPUT}" \
  --minimum-per-map "${MINIMUM_PER_MAP}" \
  --minimum-local-group "${MINIMUM_LOCAL_GROUP}" \
  --maximum-local-groups-per-stratum "${MAXIMUM_LOCAL_GROUPS_PER_STRATUM}" \
  --minimum-common-path-count "${MINIMUM_COMMON_PATH_COUNT}" \
  --projections "${PROJECTIONS}" \
  --projection-seeds "${PROJECTION_SEEDS}" \
  --weighted-quantiles "${WEIGHTED_QUANTILES}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --device cuda \
  --seed "${SEED}" \
  --output "${REPORT}" \
  2>&1 | tee "${LOG_DIR}/per_map_power_aod_fidelity.log"

printf '[%s] DONE  per_map_power_aod_fidelity\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/summarize_per_map_power_aod_fidelity.py "${REPORT}" \
  | tee "${REPORT_DIR}/summary.json"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${REPORT_DIR}/summary.json" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
