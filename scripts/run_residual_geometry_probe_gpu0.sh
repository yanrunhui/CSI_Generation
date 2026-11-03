#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARED_ROOT="${PREPARED_ROOT:-artifacts/prepared_d2los}"
SCENE_PLAN="${SCENE_PLAN:-artifacts/splits/scene_id_seen_map_v1.json}"
GEOMETRY_FEATURES="${GEOMETRY_FEATURES:-artifacts/geometry/map_global_building_features.json}"
SEEDS="${SEEDS:-2026,2027,2028}"
NUM_WORKERS="${NUM_WORKERS:-4}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/residual_geometry_probe/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/residual_geometry_probe/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

timestamp() {
  date --iso-8601=seconds 2>/dev/null || date
}

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Residual geometry probe completed successfully.\n' "$(timestamp)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Residual geometry probe failed with exit code %s.\n' \
      "$(timestamp)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'Required file not found: %s\n' "${path}" >&2
    exit 2
  fi
}

require_file "${PREPARED_ROOT}/manifest.json"
require_file "${SCENE_PLAN}"
require_file "${GEOMETRY_FEATURES}"

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'geometry_features=%s\n' "${GEOMETRY_FEATURES}"
  printf 'report_dir=%s\n' "${REPORT_DIR}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

printf '\n[%s] START residual_geometry_probe\n' "$(timestamp)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/run_residual_geometry_probe.py \
  --prepared-root "${PREPARED_ROOT}" \
  --scene-plan "${SCENE_PLAN}" \
  --geometry-features "${GEOMETRY_FEATURES}" \
  --protocol both \
  --device cuda \
  --num-workers "${NUM_WORKERS}" \
  --seeds "${SEEDS}" \
  --geometry-dropout 0.20 \
  --maximum-gate 0.50 \
  --initial-gate 0.05 \
  --correction-penalty 0.05 \
  --output "${REPORT_DIR}/residual_geometry_probe.json" \
  2>&1 | tee "${LOG_DIR}/residual_geometry_probe.log"
printf '[%s] DONE  residual_geometry_probe\n' "$(timestamp)" | tee -a "${QUEUE_LOG}"

printf '\nReport: %s\nLogs: %s\n' \
  "${REPORT_DIR}/residual_geometry_probe.json" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
