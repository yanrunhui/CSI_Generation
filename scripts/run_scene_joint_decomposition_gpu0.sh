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
GENERATED_INPUT="${GENERATED_INPUT:-artifacts/generated/b3_scene_id_evaluation.pt}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.yaml}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2048}"
SUPPORT_RADIUS="${SUPPORT_RADIUS:-1.0}"
MIN_DENSITY_GROUP="${MIN_DENSITY_GROUP:-64}"
MIN_PER_MAP_DENSITY_GROUP="${MIN_PER_MAP_DENSITY_GROUP:-8}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
NUM_WORKERS="${NUM_WORKERS:-4}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/scene_joint_decomposition/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/scene_joint_decomposition/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"
REPORT_PATH="${REPORT_DIR}/scene_joint_decomposition.json"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Joint-distribution diagnostic completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Diagnostic failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
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

require_file "${GENERATED_INPUT}"
require_file "${SCENE_PLAN}"
require_file "${PREPARED_ROOT}/manifest.json"
require_file "${DATA_CONFIG}"

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'generated_input=%s\n' "${GENERATED_INPUT}"
  printf 'scene_plan=%s\n' "${SCENE_PLAN}"
  printf 'support_radius=%s\n' "${SUPPORT_RADIUS}"
  printf 'report=%s\n' "${REPORT_PATH}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

command=(
  "${PYTHON_BIN}" scripts/diagnose_scene_joint_decomposition.py
  --generated "${GENERATED_INPUT}"
  --scene-plan "${SCENE_PLAN}"
  --prepared-root "${PREPARED_ROOT}"
  --data-config "${DATA_CONFIG}"
  --maximum-samples "${MAX_EVAL_SAMPLES}"
  --support-radius "${SUPPORT_RADIUS}"
  --minimum-density-group "${MIN_DENSITY_GROUP}"
  --minimum-per-map-density-group "${MIN_PER_MAP_DENSITY_GROUP}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --num-workers "${NUM_WORKERS}"
  --device cuda
  --output "${REPORT_PATH}"
)

printf '\n[%s] START scene_joint_decomposition\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${command[@]}" 2>&1 | tee "${LOG_DIR}/scene_joint_decomposition.log"
printf '[%s] DONE  scene_joint_decomposition\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

"${PYTHON_BIN}" scripts/summarize_scene_joint_decomposition.py "${REPORT_PATH}" \
  | tee "${REPORT_DIR}/summary.json"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT_PATH}" "${REPORT_DIR}/summary.json" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
