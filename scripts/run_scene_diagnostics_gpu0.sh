#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# tmux inherits the launching shell's environment.  CUDA_VISIBLE_DEVICES makes
# physical GPU 0 appear as cuda:0 inside every Python process in this queue.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARED_ROOT="${PREPARED_ROOT:-artifacts/prepared_d2los}"
SCENE_PLAN="${SCENE_PLAN:-artifacts/splits/scene_id_seen_map_v1.json}"
SCENE_CHECKPOINT="${SCENE_CHECKPOINT:-artifacts/checkpoints/b3_scene_id/best_calibrated.pt}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-artifacts/checkpoints/b3_seen_map_baseline/best.pt}"
GENERATED_INPUT="${GENERATED_INPUT:-artifacts/generated/b3_scene_id_evaluation.pt}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2048}"
LEARNABILITY_SEEDS="${LEARNABILITY_SEEDS:-2026,2027,2028}"
MAP_QUERY_LOS="${MAP_QUERY_LOS:-map_10,-90,100,5,1}"
MAP_QUERY_NLOS="${MAP_QUERY_NLOS:-map_10,-95,150,nan,0}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/scene_diagnostics/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/scene_diagnostics/${RUN_TAG}}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "$(dirname -- "${GENERATED_INPUT}")"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] All diagnostics completed successfully.\n' "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Queue failed with exit code %s.\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'Required file not found: %s\n' "${path}" >&2
    exit 2
  fi
}

require_file "${SCENE_PLAN}"
require_file "${PREPARED_ROOT}/manifest.json"

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'scene_plan=%s\n' "${SCENE_PLAN}"
  printf 'generated_input=%s\n' "${GENERATED_INPUT}"
  printf 'report_dir=%s\n' "${REPORT_DIR}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

coverage_command=(
  "${PYTHON_BIN}" scripts/diagnose_map_condition_coverage.py
  --prepared-root "${PREPARED_ROOT}"
  --scene-plan "${SCENE_PLAN}"
  --bins 5
  --radii 0.5,1.0,2.0
  --query "${MAP_QUERY_LOS}"
  --query "${MAP_QUERY_NLOS}"
  --output "${REPORT_DIR}/map_condition_coverage.json"
)
run_stage 01_map_condition_coverage "${coverage_command[@]}"

learnability_command=(
  "${PYTHON_BIN}" scripts/diagnose_scene_feature_learnability.py
  --scene-plan "${SCENE_PLAN}"
  --prepared-root "${PREPARED_ROOT}"
  --target all
  --device cuda
  --seeds "${LEARNABILITY_SEEDS}"
  --output "${REPORT_DIR}/scene_feature_learnability.json"
)
run_stage 02_scene_feature_learnability "${learnability_command[@]}"

if [[ ! -f "${GENERATED_INPUT}" ]]; then
  if [[ ! -f "${SCENE_CHECKPOINT}" && -f artifacts/checkpoints/b3_scene_id/best.pt ]]; then
    SCENE_CHECKPOINT=artifacts/checkpoints/b3_scene_id/best.pt
  fi
  require_file "${SCENE_CHECKPOINT}"
  evaluation_command=(
    "${PYTHON_BIN}" scripts/evaluate_scene_id_protocol.py
    --scene-checkpoint "${SCENE_CHECKPOINT}"
    --prepared-root "${PREPARED_ROOT}"
    --metrics-device cuda
    --max-samples "${MAX_EVAL_SAMPLES}"
    --output "${REPORT_DIR}/b3_scene_id_protocol.json"
    --generated-output "${GENERATED_INPUT}"
  )
  if [[ -f "${BASELINE_CHECKPOINT}" ]]; then
    evaluation_command+=(--baseline-checkpoint "${BASELINE_CHECKPOINT}")
  fi
  run_stage 03_generate_evaluation_payload "${evaluation_command[@]}"
else
  printf '\n[%s] SKIP 03_generate_evaluation_payload; reusing %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${GENERATED_INPUT}" | tee -a "${QUEUE_LOG}"
fi

attribution_command=(
  "${PYTHON_BIN}" scripts/diagnose_physical_parameter_attribution.py
  --generated "${GENERATED_INPUT}"
  --candidate correct_map
  --maximum-samples "${MAX_EVAL_SAMPLES}"
  --device cuda
  --output "${REPORT_DIR}/b3_scene_id_parameter_attribution.json"
)
run_stage 04_physical_parameter_attribution "${attribution_command[@]}"

printf '\nReports: %s\nLogs: %s\n' "${REPORT_DIR}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
