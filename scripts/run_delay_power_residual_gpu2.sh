#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
# Matched Map-independent B3: same seen-Map spatial split and 41-epoch budget
# as the B3-S-ID experiment, without any scene input.
B3_CHECKPOINT="${B3_CHECKPOINT:-artifacts/checkpoints/b3_seen_map_baseline/best.pt}"
SCENE_PLAN="${SCENE_PLAN:-artifacts/splits/scene_id_seen_map_v1.json}"
PREPARED_ROOT="${PREPARED_ROOT:-artifacts/prepared_d2los}"
CONFIG="${CONFIG:-configs/delay_power_residual.yaml}"
SEED="${SEED:-2026}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-256}"
CACHE_WORKERS="${CACHE_WORKERS:-2}"
CACHE_SAMPLING_STEPS="${CACHE_SAMPLING_STEPS:-100}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CACHE="${CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed${SEED}.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints/delay_power_residual/${RUN_TAG}}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/delay_power_residual/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/delay_power_residual/${RUN_TAG}}"
ARTIFACT="${ARTIFACT:-artifacts/generated/delay_power_residual/evaluation_${RUN_TAG}.pt}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "$(dirname -- "${CACHE}")" "$(dirname -- "${ARTIFACT}")"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"
REPORT="${REPORT_DIR}/delay_power_residual.json"
SUMMARY="${REPORT_DIR}/summary.json"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] DP residual experiment completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] DP residual experiment failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

for required in "${B3_CHECKPOINT}" "${SCENE_PLAN}" "${CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required}" >&2
    exit 2
  fi
done
if [[ ! -d "${PREPARED_ROOT}" ]]; then
  printf 'Prepared RayVerse root not found: %s\n' "${PREPARED_ROOT}" >&2
  exit 2
fi

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'b3_checkpoint=%s\n' "${B3_CHECKPOINT}"
  printf 'scene_plan=%s\n' "${SCENE_PLAN}"
  printf 'prepared_root=%s\n' "${PREPARED_ROOT}"
  printf 'cache=%s\n' "${CACHE}"
  printf 'checkpoint_dir=%s\n' "${CHECKPOINT_DIR}"
  printf 'artifact=%s\n' "${ARTIFACT}"
  printf 'report=%s\n' "${REPORT}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

if [[ -f "${CACHE}" ]]; then
  printf '\n[%s] SKIP  01_cache_b3 (existing cache)\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
else
  printf '\n[%s] START 01_cache_b3\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  "${PYTHON_BIN}" scripts/cache_b3_dp_residual.py \
    --b3-checkpoint "${B3_CHECKPOINT}" \
    --scene-plan "${SCENE_PLAN}" \
    --prepared-root "${PREPARED_ROOT}" \
    --output "${CACHE}" \
    --batch-size "${CACHE_BATCH_SIZE}" \
    --num-workers "${CACHE_WORKERS}" \
    --sampling-steps "${CACHE_SAMPLING_STEPS}" \
    --device cuda \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/01_cache_b3.log"
  printf '[%s] DONE  01_cache_b3\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
fi

printf '\n[%s] START 02_train_dp_residual\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/train_delay_power_residual.py \
  --cache "${CACHE}" \
  --config "${CONFIG}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  2>&1 | tee "${LOG_DIR}/02_train_dp_residual.log"
printf '[%s] DONE  02_train_dp_residual\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

printf '\n[%s] START 03_evaluate_dp_residual\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
"${PYTHON_BIN}" scripts/evaluate_delay_power_residual.py \
  --checkpoint "${CHECKPOINT_DIR}/best.pt" \
  --cache "${CACHE}" \
  --output-artifact "${ARTIFACT}" \
  --output-report "${REPORT}" \
  --device cuda \
  --seed "${SEED}" \
  2>&1 | tee "${LOG_DIR}/03_evaluate_dp_residual.log"
"${PYTHON_BIN}" scripts/summarize_delay_power_residual.py "${REPORT}" \
  | tee "${SUMMARY}"
printf '[%s] DONE  03_evaluate_dp_residual\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

printf '\nArtifact: %s\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${ARTIFACT}" "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
printf '\nFor the existing paper diagnostics, use GENERATED_INPUT=%s.\n' \
  "${ARTIFACT}" | tee -a "${QUEUE_LOG}"
