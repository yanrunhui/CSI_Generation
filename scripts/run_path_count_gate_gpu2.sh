#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARED_ROOT="${PREPARED_ROOT:-artifacts/prepared_d2los}"
SCENE_PLAN="${SCENE_PLAN:-artifacts/splits/scene_id_seen_map_v1.json}"
B3_CHECKPOINT="${B3_CHECKPOINT:-artifacts/checkpoints/b3_seen_map_baseline/best.pt}"
GENERATION_SEEDS="${GENERATION_SEEDS:-2026,2027,2028,2029,2030}"
ALPHAS="${ALPHAS:-0,0.5,1,1.5}"

if [[ -z "${DP_CHECKPOINT:-}" ]]; then
  DP_CHECKPOINT="$({
    find artifacts/checkpoints/delay_power_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi
if [[ -z "${LINK_CHECKPOINT:-}" ]]; then
  LINK_CHECKPOINT="$({
    find artifacts/checkpoints/aod_link_global_residual -mindepth 2 -maxdepth 2 \
      -type f -name best.pt -print 2>/dev/null || true
  } | sort | tail -1)"
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR="${CACHE_DIR:-artifacts/generated/path_count_gate}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/path_count_gate/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/path_count_gate/${RUN_TAG}}"
REPORT="${REPORT_DIR}/path_count_gate.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${CACHE_DIR}" "${REPORT_DIR}" "${LOG_DIR}"

for required in \
  "${PREPARED_ROOT}" "${SCENE_PLAN}" "${B3_CHECKPOINT}" \
  "${DP_CHECKPOINT}" "${LINK_CHECKPOINT}"; do
  if [[ -z "${required}" || ! -e "${required}" ]]; then
    printf 'Required path not found: %s\n' "${required:-<empty auto-detected path>}" >&2
    exit 2
  fi
done

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
    printf '\n[%s] Path-count-aware gate experiment completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] Path-count-aware gate experiment failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'generation_seeds=%s\n' "${GENERATION_SEEDS}"
  printf 'alphas=%s\n' "${ALPHAS}"
  printf 'b3_checkpoint=%s\n' "${B3_CHECKPOINT}"
  printf 'dp_checkpoint=%s\n' "${DP_CHECKPOINT}"
  printf 'link_checkpoint=%s\n' "${LINK_CHECKPOINT}"
  printf 'report=%s\n' "${REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

IFS=',' read -r -a SEEDS <<< "${GENERATION_SEEDS}"
CACHE_ARGS=()
for raw_seed in "${SEEDS[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  if [[ -z "${seed}" ]]; then
    continue
  fi
  if [[ "${seed}" == "2026" \
        && -f artifacts/generated/delay_power_residual/b3_cache_seed2026.pt ]]; then
    cache="artifacts/generated/delay_power_residual/b3_cache_seed2026.pt"
  else
    cache="${CACHE_DIR}/b3_validation_test_seed${seed}.pt"
    if [[ ! -f "${cache}" ]]; then
      run_stage "cache_b3_seed${seed}" \
        "${PYTHON_BIN}" scripts/cache_b3_dp_residual.py \
          --b3-checkpoint "${B3_CHECKPOINT}" \
          --scene-plan "${SCENE_PLAN}" \
          --prepared-root "${PREPARED_ROOT}" \
          --output "${cache}" \
          --device cuda \
          --seed "${seed}" \
          --batch-size "${CACHE_BATCH_SIZE:-256}" \
          --num-workers "${CACHE_NUM_WORKERS:-2}" \
          --splits validation,test
    else
      printf '[%s] REUSE cache_b3_seed%s %s\n' \
        "$(date --iso-8601=seconds 2>/dev/null || date)" "${seed}" "${cache}" \
        | tee -a "${QUEUE_LOG}"
    fi
  fi
  CACHE_ARGS+=(--cache "${seed}=${cache}")
done

if (( ${#CACHE_ARGS[@]} < 4 )); then
  printf 'At least two generation seeds are required.\n' >&2
  exit 2
fi

run_stage evaluate_path_count_gate \
  "${PYTHON_BIN}" scripts/evaluate_path_count_gate.py \
    "${CACHE_ARGS[@]}" \
    --dp-checkpoint "${DP_CHECKPOINT}" \
    --link-checkpoint "${LINK_CHECKPOINT}" \
    --output "${REPORT}" \
    --alphas "${ALPHAS}" \
    --device cuda \
    --selection-seed "${SELECTION_SEED:-2026}" \
    --maximum-samples-per-split "${MAXIMUM_SAMPLES_PER_SPLIT:-2048}" \
    --minimum-map-samples "${MINIMUM_MAP_SAMPLES:-24}" \
    --batch-size "${EVALUATION_BATCH_SIZE:-256}" \
    --swd-projections "${SWD_PROJECTIONS:-128}" \
    --map-swd-projections "${MAP_SWD_PROJECTIONS:-64}" \
    --projection-dimension "${PROJECTION_DIMENSION:-64}" \
    --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
    --maximum-nonregression "${MAXIMUM_NONREGRESSION:-0.05}" \
    --maximum-primary-regression "${MAXIMUM_PRIMARY_REGRESSION:-0.02}" \
    --minimum-primary-improvement "${MINIMUM_PRIMARY_IMPROVEMENT:-0.02}" \
    --minimum-seed-wins "${MINIMUM_SEED_WINS:-4}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}"

"${PYTHON_BIN}" scripts/summarize_path_count_gate.py "${REPORT}" | tee "${SUMMARY}"

printf '\nReport: %s\nSummary: %s\nLogs: %s\n' \
  "${REPORT}" "${SUMMARY}" "${LOG_DIR}" | tee -a "${QUEUE_LOG}"
