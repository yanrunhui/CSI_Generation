#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-artifacts/generated/delay_power_residual/b3_cache_seed2026.pt}"
CONFIG="${CONFIG:-configs/aod_relative_cluster_residual.yaml}"
SEED="${SEED:-2026}"

if [[ -z "${DP_CHECKPOINT:-}" ]]; then
  DP_CHECKPOINT="$(
    find artifacts/checkpoints/delay_power_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null | sort | tail -1
  )"
fi
if [[ -z "${LINK_CHECKPOINT:-}" ]]; then
  LINK_CHECKPOINT="$(
    find artifacts/checkpoints/aod_link_global_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null | sort | tail -1
  )"
fi
if [[ -z "${LINK_ARTIFACT:-}" ]]; then
  LINK_ARTIFACT="$(
    find artifacts/generated/aod_link_global -maxdepth 1 -type f \
      -name 'evaluation_*.pt' -print 2>/dev/null | sort | tail -1
  )"
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints/aod_relative_cluster_residual/${RUN_TAG}}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_relative_cluster/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_relative_cluster/${RUN_TAG}}"
ARTIFACT="${ARTIFACT:-artifacts/generated/aod_relative_cluster/evaluation_${RUN_TAG}.pt}"
MODEL_REPORT="${REPORT_DIR}/relative_cluster_model.json"
MODEL_SUMMARY="${REPORT_DIR}/relative_cluster_summary.json"
CSI_REPORT="${REPORT_DIR}/csi_nonregression.json"
CSI_SUMMARY="${REPORT_DIR}/csi_nonregression_summary.json"
mkdir -p "${CHECKPOINT_DIR}" "${REPORT_DIR}" "${LOG_DIR}" "$(dirname -- "${ARTIFACT}")"

QUEUE_LOG="${LOG_DIR}/queue.log"
STATUS_FILE="${LOG_DIR}/status.txt"

on_exit() {
  local code=$?
  if (( code == 0 )); then
    printf 'SUCCESS\n' > "${STATUS_FILE}"
    printf '\n[%s] Relative-AoD cluster experiment completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Relative-AoD cluster experiment failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

run_stage() {
  local name="$1"
  shift
  printf '\n[%s] START %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  printf '[%s] DONE  %s\n' \
    "$(date --iso-8601=seconds 2>/dev/null || date)" "${name}" | tee -a "${QUEUE_LOG}"
}

for required in \
  "${CACHE}" "${CONFIG}" "${DP_CHECKPOINT}" "${LINK_CHECKPOINT}" "${LINK_ARTIFACT}"; do
  if [[ -z "${required}" || ! -f "${required}" ]]; then
    printf 'Required file not found: %s\n' "${required:-<empty auto-detected path>}" >&2
    exit 2
  fi
done

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'cache=%s\n' "${CACHE}"
  printf 'dp_checkpoint=%s\n' "${DP_CHECKPOINT}"
  printf 'link_checkpoint=%s\n' "${LINK_CHECKPOINT}"
  printf 'link_artifact=%s\n' "${LINK_ARTIFACT}"
  printf 'config=%s\n' "${CONFIG}"
  printf 'checkpoint_dir=%s\n' "${CHECKPOINT_DIR}"
  printf 'artifact=%s\n' "${ARTIFACT}"
  printf 'report_dir=%s\n' "${REPORT_DIR}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

run_stage 01_train_relative_cluster \
  "${PYTHON_BIN}" scripts/train_aod_residual.py \
    --cache "${CACHE}" \
    --dp-checkpoint "${DP_CHECKPOINT}" \
    --link-global-checkpoint "${LINK_CHECKPOINT}" \
    --config "${CONFIG}" \
    --checkpoint-dir "${CHECKPOINT_DIR}"

run_stage 02_evaluate_relative_cluster \
  "${PYTHON_BIN}" scripts/evaluate_aod_residual.py \
    --checkpoint "${CHECKPOINT_DIR}/best.pt" \
    --dp-checkpoint "${DP_CHECKPOINT}" \
    --cache "${CACHE}" \
    --output-artifact "${ARTIFACT}" \
    --output-report "${MODEL_REPORT}" \
    --device cuda \
    --seed "${SEED}"

run_stage 03_summarize_relative_cluster \
  "${PYTHON_BIN}" scripts/summarize_aod_residual.py "${MODEL_REPORT}"
"${PYTHON_BIN}" scripts/summarize_aod_residual.py "${MODEL_REPORT}" \
  > "${MODEL_SUMMARY}"

run_stage 04_csi_nonregression \
  "${PYTHON_BIN}" scripts/diagnose_aod_csi_nonregression.py \
    --candidate-artifact "${ARTIFACT}" \
    --candidate-label relative_cluster \
    --comparison "link_global=${LINK_ARTIFACT}" \
    --output "${CSI_REPORT}" \
    --device cuda \
    --relative-noninferiority-margin 0.05

run_stage 05_summarize_csi_nonregression \
  "${PYTHON_BIN}" scripts/summarize_aod_csi_nonregression.py "${CSI_REPORT}"
"${PYTHON_BIN}" scripts/summarize_aod_csi_nonregression.py "${CSI_REPORT}" \
  > "${CSI_SUMMARY}"

printf '\nArtifact: %s\nModel report: %s\nModel summary: %s\nCSI report: %s\nCSI summary: %s\nLogs: %s\n' \
  "${ARTIFACT}" "${MODEL_REPORT}" "${MODEL_SUMMARY}" \
  "${CSI_REPORT}" "${CSI_SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"

