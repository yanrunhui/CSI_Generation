#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
GENERATE_FULL_ARTIFACT=0
if [[ -z "${LINK_ARTIFACT:-}" ]]; then
  GENERATE_FULL_ARTIFACT=1
  LINK_ARTIFACT="artifacts/generated/csi_absolute_quality/link_global_full_test_${RUN_TAG}.pt"
  LINK_MODEL_REPORT="artifacts/reports/csi_absolute_quality/${RUN_TAG}/link_global_full_test_generation.json"
  if [[ -z "${LINK_CHECKPOINT:-}" ]]; then
    LINK_CHECKPOINT="$(
      find artifacts/checkpoints/aod_link_global_residual -mindepth 2 -maxdepth 2 \
        -type f -name best.pt -print 2>/dev/null | sort | tail -1
    )"
  fi
  if [[ -z "${DP_CHECKPOINT:-}" ]]; then
    DP_CHECKPOINT="$(
      find artifacts/checkpoints/delay_power_residual -mindepth 2 -maxdepth 2 \
        -type f -name best.pt -print 2>/dev/null | sort | tail -1
    )"
  fi
else
  LINK_MODEL_REPORT="<not generated; existing artifact supplied>"
fi

REPORT_DIR="${REPORT_DIR:-artifacts/reports/csi_absolute_quality/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/csi_absolute_quality/${RUN_TAG}}"
REPORT="${REPORT_DIR}/csi_absolute_quality.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}" "$(dirname -- "${LINK_ARTIFACT}")"

if (( GENERATE_FULL_ARTIFACT == 1 )); then
  for required in "${CACHE}" "${LINK_CHECKPOINT}" "${DP_CHECKPOINT}"; do
    if [[ -z "${required}" || ! -f "${required}" ]]; then
      printf 'Required file not found: %s\n' "${required:-<empty auto-detected path>}" >&2
      exit 2
    fi
  done
elif [[ ! -f "${LINK_ARTIFACT}" ]]; then
  printf 'Link-global evaluation artifact not found: %s\n' "${LINK_ARTIFACT}" >&2
  exit 2
fi

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] CSI absolute-quality evaluation completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] CSI absolute-quality evaluation failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'generate_full_artifact=%s\n' "${GENERATE_FULL_ARTIFACT}"
  printf 'cache=%s\n' "${CACHE}"
  printf 'link_checkpoint=%s\n' "${LINK_CHECKPOINT:-<existing artifact>}"
  printf 'dp_checkpoint=%s\n' "${DP_CHECKPOINT:-<existing artifact>}"
  printf 'link_artifact=%s\n' "${LINK_ARTIFACT}"
  printf 'report=%s\n' "${REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

if (( GENERATE_FULL_ARTIFACT == 1 )); then
  run_stage 01_generate_full_test_artifact \
    "${PYTHON_BIN}" scripts/evaluate_aod_residual.py \
      --checkpoint "${LINK_CHECKPOINT}" \
      --dp-checkpoint "${DP_CHECKPOINT}" \
      --cache "${CACHE}" \
      --output-artifact "${LINK_ARTIFACT}" \
      --output-report "${LINK_MODEL_REPORT}" \
      --split test \
      --maximum-samples-per-map "${MAXIMUM_SAMPLES_PER_MAP:-1000000}" \
      --device cuda \
      --seed "${ARTIFACT_SELECTION_SEED:-2026}"
fi

run_stage 02_csi_absolute_quality \
  "${PYTHON_BIN}" scripts/evaluate_csi_absolute_quality.py \
    --artifact "${LINK_ARTIFACT}" \
    --output "${REPORT}" \
    --device cuda \
    --seed "${EVALUATION_SEED:-2026}" \
    --minimum-pairs-per-map "${MINIMUM_PAIRS_PER_MAP:-12}" \
    --maximum-condition-distance "${MAXIMUM_CONDITION_DISTANCE:-2.5}" \
    --local-minimum-samples "${LOCAL_MINIMUM_SAMPLES:-8}" \
    --projection-dimension "${PROJECTION_DIMENSION:-64}" \
    --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
    --swd-projections "${SWD_PROJECTIONS:-128}" \
    --coverage-k "${COVERAGE_K:-3}"
"${PYTHON_BIN}" scripts/summarize_csi_absolute_quality.py "${REPORT}" \
  | tee "${SUMMARY}"
printf '[%s] DONE  csi_absolute_quality\n' \
  "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
