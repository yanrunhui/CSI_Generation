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
GENERATION_SEEDS="${GENERATION_SEEDS:-2026,2027,2028}"

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
CACHE_DIR="${CACHE_DIR:-artifacts/generated/aod_spatial_seed_validation}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/generated/aod_spatial_seed_validation}"
REPORT_DIR="${REPORT_DIR:-artifacts/reports/aod_spatial_seed_validation/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-artifacts/logs/aod_spatial_seed_validation/${RUN_TAG}}"
AGGREGATE_REPORT="${REPORT_DIR}/aod_cross_map_seed_validation.json"
LOCALIZATION_REPORT="${REPORT_DIR}/spatial_failure_localization.json"
SUMMARY="${REPORT_DIR}/summary.json"
mkdir -p "${CACHE_DIR}" "${ARTIFACT_DIR}" "${REPORT_DIR}" "${LOG_DIR}"

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
    printf '\n[%s] AoD spatial seed validation completed successfully.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${QUEUE_LOG}"
  else
    printf 'FAILED exit_code=%s\n' "${code}" > "${STATUS_FILE}"
    printf '\n[%s] AoD spatial seed validation failed with exit code %s.\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${code}" | tee -a "${QUEUE_LOG}"
  fi
}
trap on_exit EXIT

{
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'generation_seeds=%s\n' "${GENERATION_SEEDS}"
  printf 'map_ids=%s\n' "${MAP_IDS:-all}"
  printf 'path_counts=%s\n' "${PATH_COUNTS:-5}"
  printf 'target_angles=%s\n' "${TARGET_ANGLES:--150,-90,-30,30,90,150}"
  printf 'tolerances_deg=%s\n' "${TOLERANCES_DEG:-10}"
  printf 'aggregate_report=%s\n' "${AGGREGATE_REPORT}"
  printf 'localization_report=%s\n' "${LOCALIZATION_REPORT}"
  printf 'summary=%s\n' "${SUMMARY}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
} | tee "${QUEUE_LOG}"

IFS=',' read -r -a SEEDS <<< "${GENERATION_SEEDS}"
REPORT_ARGUMENTS=()
valid_seed_count=0
for raw_seed in "${SEEDS[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  if [[ -z "${seed}" ]]; then
    continue
  fi
  valid_seed_count=$((valid_seed_count + 1))
  if [[ "${seed}" == "2026" \
        && -f artifacts/generated/delay_power_residual/b3_cache_seed2026.pt ]]; then
    cache="artifacts/generated/delay_power_residual/b3_cache_seed2026.pt"
  elif [[ "${FORCE_REGENERATE:-0}" != "1" \
          && -f "artifacts/generated/path_count_gate/b3_validation_test_seed${seed}.pt" ]]; then
    cache="artifacts/generated/path_count_gate/b3_validation_test_seed${seed}.pt"
    printf '[%s] REUSE path-count diagnostic cache seed=%s %s\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${seed}" "${cache}" \
      | tee -a "${QUEUE_LOG}"
  else
    cache="${CACHE_DIR}/b3_test_seed${seed}.pt"
    if [[ ! -f "${cache}" || "${FORCE_REGENERATE:-0}" == "1" ]]; then
      run_stage "01_cache_b3_seed${seed}" \
        "${PYTHON_BIN}" scripts/cache_b3_dp_residual.py \
          --b3-checkpoint "${B3_CHECKPOINT}" \
          --scene-plan "${SCENE_PLAN}" \
          --prepared-root "${PREPARED_ROOT}" \
          --output "${cache}" \
          --batch-size "${CACHE_BATCH_SIZE:-256}" \
          --num-workers "${CACHE_NUM_WORKERS:-2}" \
          --sampling-steps "${CACHE_SAMPLING_STEPS:-100}" \
          --splits test \
          --device cuda \
          --seed "${seed}"
    else
      printf '[%s] REUSE cache seed=%s %s\n' \
        "$(date --iso-8601=seconds 2>/dev/null || date)" "${seed}" "${cache}" \
        | tee -a "${QUEUE_LOG}"
    fi
  fi

  artifact="${ARTIFACT_DIR}/link_global_test_seed${seed}.pt"
  model_report="${REPORT_DIR}/link_global_generation_seed${seed}.json"
  if [[ ! -f "${artifact}" || "${FORCE_REGENERATE:-0}" == "1" ]]; then
    run_stage "02_generate_link_global_seed${seed}" \
      "${PYTHON_BIN}" scripts/evaluate_aod_residual.py \
        --checkpoint "${LINK_CHECKPOINT}" \
        --dp-checkpoint "${DP_CHECKPOINT}" \
        --cache "${cache}" \
        --output-artifact "${artifact}" \
        --output-report "${model_report}" \
        --split test \
        --maximum-samples-per-map "${MAXIMUM_SAMPLES_PER_MAP:-1000000}" \
        --batch-size "${EVALUATION_BATCH_SIZE:-256}" \
        --device cuda \
        --seed "${ARTIFACT_SELECTION_SEED:-2026}"
  else
    printf '[%s] REUSE artifact seed=%s %s\n' \
      "$(date --iso-8601=seconds 2>/dev/null || date)" "${seed}" "${artifact}" \
      | tee -a "${QUEUE_LOG}"
  fi

  tolerance_report="${REPORT_DIR}/aod_tolerance_seed${seed}.json"
  run_stage "03_diagnose_seed${seed}" \
    "${PYTHON_BIN}" scripts/diagnose_aod_tolerance_tradeoff.py \
      --artifact "${artifact}" \
      --link-checkpoint "${LINK_CHECKPOINT}" \
      --cache "${cache}" \
      --output "${tolerance_report}" \
      --map-ids "${MAP_IDS:-all}" \
      --path-counts "${PATH_COUNTS:-5}" \
      --modes "${MODES:-mean}" \
      --target-angles="${TARGET_ANGLES:--150,-90,-30,30,90,150}" \
      --tolerances-deg "${TOLERANCES_DEG:-10}" \
      --candidate-multipliers "${CANDIDATE_MULTIPLIERS:-10,20,40}" \
      --requested-samples "${REQUESTED_SAMPLES:-8}" \
      --pool-trials "${POOL_TRIALS:-200}" \
      --minimum-fidelity-samples "${MINIMUM_FIDELITY_SAMPLES:-8}" \
      --maximum-fidelity-samples "${MAXIMUM_FIDELITY_SAMPLES:-32}" \
      --swd-projections "${SWD_PROJECTIONS:-64}" \
      --projection-dimension "${PROJECTION_DIMENSION:-64}" \
      --projection-seeds "${PROJECTION_SEEDS:-0,1,2}" \
      --seed "${DIAGNOSTIC_SEED:-2026}" \
      --device cuda
  REPORT_ARGUMENTS+=(--report "seed${seed}=${tolerance_report}")
done

if (( valid_seed_count < 3 )); then
  printf 'At least three generation seeds are required; got %s.\n' "${valid_seed_count}" >&2
  exit 2
fi

run_stage 04_aggregate \
  "${PYTHON_BIN}" scripts/aggregate_aod_cross_map_validation.py \
    "${REPORT_ARGUMENTS[@]}" \
    --output "${AGGREGATE_REPORT}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
    --maximum-relative-degradation "${MAXIMUM_RELATIVE_DEGRADATION:-0.05}"

run_stage 05_localize_spatial_failures \
  "${PYTHON_BIN}" scripts/localize_aod_spatial_failures.py \
    "${REPORT_ARGUMENTS[@]}" \
    --output "${LOCALIZATION_REPORT}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-10000}" \
    --minimum-seeds 3

"${PYTHON_BIN}" scripts/summarize_aod_spatial_seed_validation.py \
  --aggregate "${AGGREGATE_REPORT}" \
  --localization "${LOCALIZATION_REPORT}" \
  | tee "${SUMMARY}"

printf '\nAggregate: %s\nLocalization: %s\nSummary: %s\nLogs: %s\n' \
  "${AGGREGATE_REPORT}" "${LOCALIZATION_REPORT}" "${SUMMARY}" "${LOG_DIR}" \
  | tee -a "${QUEUE_LOG}"
