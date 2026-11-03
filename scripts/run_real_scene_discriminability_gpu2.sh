#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
SCENE_PLAN="${SCENE_PLAN:-artifacts/splits/scene_id_seen_map_v1.json}"
PREPARED_ROOT="${PREPARED_ROOT:-artifacts/prepared_d2los}"
SPLIT="${SPLIT:-test}"
ANCHOR_MAP="${ANCHOR_MAP:-10}"
SPECIFIC_MAP="${SPECIFIC_MAP:-25}"
MAXIMUM_ANCHOR_PAIRS="${MAXIMUM_ANCHOR_PAIRS:-0}"
MAXIMUM_GENERAL_PAIRS="${MAXIMUM_GENERAL_PAIRS:-50}"
MAXIMUM_SAMPLES_PER_MAP="${MAXIMUM_SAMPLES_PER_MAP:-256}"
SAMPLES_PER_COMPARISON="${SAMPLES_PER_COMPARISON:-48}"
REPEATS="${REPEATS:-20}"
MINIMUM_SUCCESSFUL_REPEATS="${MINIMUM_SUCCESSFUL_REPEATS:-10}"
MAXIMUM_ATTEMPTS_PER_REPEAT="${MAXIMUM_ATTEMPTS_PER_REPEAT:-200}"
MAXIMUM_CONDITION_DISTANCE="${MAXIMUM_CONDITION_DISTANCE:-1.0}"
DIAGNOSTIC_PROJECTIONS="${DIAGNOSTIC_PROJECTIONS:-64}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SEED="${SEED:-2026}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/real_scene_discriminability/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/real_scene_discriminability/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"
REPORT="${REPORT_DIR}/real_scene_discriminability.json"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Real-to-real scene discriminability completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Diagnostic failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

if [[ ! -f "${SCENE_PLAN}" ]]; then
  printf 'Required scene plan not found: %s\n' "${SCENE_PLAN}" >&2
  exit 2
fi
if [[ ! -d "${PREPARED_ROOT}" ]]; then
  printf 'Prepared RayVerse root not found: %s\n' "${PREPARED_ROOT}" >&2
  exit 2
fi

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'scene_plan=%s\n' "${SCENE_PLAN}"
  printf 'prepared_root=%s\n' "${PREPARED_ROOT}"
  printf 'split=%s\n' "${SPLIT}"
  printf 'anchor_map=%s\n' "${ANCHOR_MAP}"
  printf 'specific_map=%s\n' "${SPECIFIC_MAP}"
  printf 'report=%s\n' "${REPORT}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START real_scene_discriminability\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/diagnose_real_scene_discriminability.py \
  --scene-plan "${SCENE_PLAN}" \
  --prepared-root "${PREPARED_ROOT}" \
  --split "${SPLIT}" \
  --anchor-map "${ANCHOR_MAP}" \
  --specific-map "${SPECIFIC_MAP}" \
  --maximum-anchor-pairs "${MAXIMUM_ANCHOR_PAIRS}" \
  --maximum-general-pairs "${MAXIMUM_GENERAL_PAIRS}" \
  --maximum-samples-per-map "${MAXIMUM_SAMPLES_PER_MAP}" \
  --samples-per-comparison "${SAMPLES_PER_COMPARISON}" \
  --repeats "${REPEATS}" \
  --minimum-successful-repeats "${MINIMUM_SUCCESSFUL_REPEATS}" \
  --maximum-attempts-per-repeat "${MAXIMUM_ATTEMPTS_PER_REPEAT}" \
  --maximum-condition-distance "${MAXIMUM_CONDITION_DISTANCE}" \
  --diagnostic-projections "${DIAGNOSTIC_PROJECTIONS}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --device cuda \
  --seed "${SEED}" \
  --output "${REPORT}" \
  2>&1 | tee "${LOG_DIR}/real_scene_discriminability.log"

printf '[%s] DONE  real_scene_discriminability\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/summarize_real_scene_discriminability.py "${REPORT}" \
  | tee "${REPORT_DIR}/summary.json"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${REPORT_DIR}/summary.json" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
