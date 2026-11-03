#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${RAW_ROOT:-}"
PREPARED_ROOT="${PREPARED_ROOT:-artifacts/prepared_d2los}"
SCENE_PLAN="${SCENE_PLAN:-artifacts/splits/scene_id_seen_map_v1.json}"
GEOMETRY_FEATURES="${GEOMETRY_FEATURES:-artifacts/geometry/map_global_building_features.json}"
SEEDS="${SEEDS:-2026,2027,2028}"
NUM_WORKERS="${NUM_WORKERS:-4}"
REBUILD_GEOMETRY="${REBUILD_GEOMETRY:-0}"

if [[ -z "${RAW_ROOT}" && -f "${PREPARED_ROOT}/manifest.json" ]]; then
  RAW_ROOT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["raw_root"])' "${PREPARED_ROOT}/manifest.json")"
fi
if [[ -z "${RAW_ROOT}" ]]; then
  printf 'RAW_ROOT is required and must point to D2Los_Data on the server.\n' >&2
  printf 'Example: RAW_ROOT=/data/D2Los_Data bash scripts/run_geometry_probe_gpu0.sh\n' >&2
  exit 2
fi
if [[ ! -d "${RAW_ROOT}" ]]; then
  printf 'Raw data directory not found: %s\n' "${RAW_ROOT}" >&2
  printf 'Override the prepared-manifest path with RAW_ROOT=/server/path/to/D2Los_Data.\n' >&2
  exit 2
fi
GEOMETRY_ROOT="${GEOMETRY_ROOT:-${RAW_ROOT}/buildings_complete}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/geometry_probe/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/geometry_probe/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "$(dirname -- "${GEOMETRY_FEATURES}")"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

timestamp() {
  date --iso-8601=seconds 2>/dev/null || date
}

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Geometry probe completed successfully.\n' "$(timestamp)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Geometry probe failed with exit code %s.\n' "$(timestamp)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' "$(timestamp)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' "$(timestamp)" "${name}" | tee -a "${QUEUE_LOG}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'Required file not found: %s\n' "${path}" >&2
    exit 2
  fi
}

require_file "${PREPARED_ROOT}/manifest.json"
require_file "${SCENE_PLAN}"
if [[ ! -d "${GEOMETRY_ROOT}" ]]; then
  printf 'Geometry directory not found: %s\n' "${GEOMETRY_ROOT}" >&2
  exit 2
fi

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'raw_root=%s\n' "${RAW_ROOT}"
  printf 'geometry_root=%s\n' "${GEOMETRY_ROOT}"
  printf 'geometry_features=%s\n' "${GEOMETRY_FEATURES}"
  printf 'report_dir=%s\n' "${REPORT_DIR}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

if [[ "${REBUILD_GEOMETRY}" == "1" || ! -f "${GEOMETRY_FEATURES}" ]]; then
  run_stage 01_extract_geometry \
    "${PYTHON_BIN}" scripts/extract_geometry_features.py \
    --raw-root "${RAW_ROOT}" \
    --geometry-root "${GEOMETRY_ROOT}" \
    --map-size 256 \
    --orientation-bins 12 \
    --layout-grid-size 4 \
    --output "${GEOMETRY_FEATURES}"
else
  printf '\n[%s] SKIP 01_extract_geometry; reusing %s\n' \
    "$(timestamp)" "${GEOMETRY_FEATURES}" | tee -a "${QUEUE_LOG}"
fi

run_stage 02_run_geometry_probe \
  "${PYTHON_BIN}" scripts/run_geometry_probe.py \
  --prepared-root "${PREPARED_ROOT}" \
  --scene-plan "${SCENE_PLAN}" \
  --geometry-features "${GEOMETRY_FEATURES}" \
  --protocol both \
  --device cuda \
  --num-workers "${NUM_WORKERS}" \
  --seeds "${SEEDS}" \
  --output "${REPORT_DIR}/geometry_probe.json"

printf '\nReport: %s\nLogs: %s\n' \
  "${REPORT_DIR}/geometry_probe.json" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
