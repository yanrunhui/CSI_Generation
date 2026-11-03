# Scene-representation diagnostics before B3-S-Geo

These diagnostics are designed to answer three questions before training a
geometry-conditioned diffusion model:

1. Does each Map have enough samples near a requested physical condition?
2. Can a small non-diffusion model learn Map-specific statistics on a spatial
   holdout?
3. Which path-parameter discrepancies drive the generated spatial-statistics
   error?

All three commands are analysis-only. They do not update a B3 or B3-S-ID
checkpoint. Use the same immutable seen-Map spatial plan as the B3-S-ID
experiment so that train, validation, and test regions remain comparable.

## 1. Map × physical-condition coverage

```bash
python scripts/diagnose_map_condition_coverage.py \
  --prepared-root artifacts/prepared_d2los \
  --scene-plan artifacts/splits/scene_id_seen_map_v1.json \
  --bins 5 \
  --radii 0.5,1.0,2.0 \
  --query map_10,-90,100,5,1 \
  --query map_10,-95,150,nan,0 \
  --output artifacts/reports/map_condition_coverage.json
```

The joint cells use pooled training quantiles for total power, delay spread,
and valid K-factor. LoS and K-validity are exact strata. Local distances use
training-set standard deviations and also require exact LoS and K-validity.
For NLoS queries, pass `nan` as K-factor.

The most useful fields are:

- `explicit_queries[*].neighbors_by_radius`, which directly answers how many
  training CSI samples from Map 10 are near a requested condition;
- `fraction_of_samples_in_sparse_cells`, which measures how much of a Map lies
  in joint cells with fewer than `--minimum-cell-samples` examples;
- `training_local_neighborhoods`, which summarizes leave-one-out effective
  sample counts rather than raw Map totals.

Quantile cells are descriptive and should not be interpreted as independent
physical regimes. The continuous-neighborhood counts are the primary result.

## 2. Map-specific feature learnability without diffusion

```bash
python scripts/diagnose_scene_feature_learnability.py \
  --scene-plan artifacts/splits/scene_id_seen_map_v1.json \
  --prepared-root artifacts/prepared_d2los \
  --target all \
  --device cuda \
  --seeds 2026,2027,2028 \
  --output artifacts/reports/scene_feature_learnability.json
```

The script trains two small conditional-mean MLPs on real data only:

- physical conditions → AoD/spatial summaries;
- physical conditions + Map embedding → the same summaries.

It evaluates the correct Map, a cyclically wrong Map, and the physics-only
model on receiver-spatially held-out real data. AoD targets use Cartesian unit
vectors, power-weighted concentration, and RMS angular spread, avoiding the
azimuth wrap discontinuity. Spatial targets are complex Tx-correlation values
at several horizontal, vertical, and diagonal UPA offsets.

Evidence that scene information is learnable requires a stable positive
`mse_reduction_correct_map_vs_physics_only` and
`mse_reduction_correct_map_vs_wrong_map` across seeds, preferably both for the
AoD and spatial-correlation target groups. A null result is not proof that
geometry is irrelevant: sparse condition support and irreducible conditional
variation can also make a conditional mean difficult to predict.

For a cheaper first pass, use `--target aod`. For the direct spatial test, use
`--target spatial`.

## 3. Physical-parameter and spatial-error attribution

First produce the saved tensors from the existing unified B3-S-ID evaluation:

```bash
python scripts/evaluate_scene_id_protocol.py \
  --scene-checkpoint artifacts/checkpoints/b3_scene_id/best_calibrated.pt \
  --baseline-checkpoint artifacts/checkpoints/b3_seen_map_baseline/best.pt \
  --metrics-device cuda \
  --max-samples 2048 \
  --output artifacts/reports/b3_scene_id_vs_b3.json \
  --generated-output artifacts/generated/b3_scene_id_evaluation.pt
```

Then run the attribution:

```bash
python scripts/diagnose_physical_parameter_attribution.py \
  --generated artifacts/generated/b3_scene_id_evaluation.pt \
  --candidate correct_map \
  --device cuda \
  --output artifacts/reports/b3_scene_id_parameter_attribution.json
```

The first section compares real and generated distributions for:

- unweighted 3D AoD directions;
- power-weighted AoD link statistics and within-link clustering;
- the joint relative-delay/log-power distribution;
- circular path phase and delay–phase coupling;
- complex frequency correlation;
- the full complex Tx spatial-correlation matrix.

The second section performs counterfactual repairs. Each generated link is
matched to the nearest real link with the same path count, LoS state, and
K-validity. One parameter group at a time is copied from the matched real link,
and CSI is rebuilt with the validated physical synthesizer. A positive
`improvement_baseline_minus_repaired` means that repair moved spatial
correlation closer to real data.

Important limitations:

- donor reuse is allowed, so the repair is a sensitivity diagnostic rather
  than an unbiased estimator of a causal contribution;
- paths are compared in the repository's delay order and exact path count is
  required;
- joint repairs reveal interactions and are not additive;
- every repair reports physical-condition drift, since replacing power or
  delay can move a sample away from its original target;
- spatial-correlation phase is an aggregate property affected by AoD, power,
  array convention, and their coupling. It is not equivalent to marginal path
  phase error.

To analyze the matched physics-only B3 or the wrong-Map ablation, change
`--candidate` to `physics_only_baseline` or `shuffled_map`.

## Decision rule for B3-S-Geo

A clean motivation for B3-S-Geo is present when:

1. local Map × condition support is sparse or highly uneven;
2. the small Map-aware predictor nevertheless beats both physics-only and
   wrong-Map controls on held-out regions; and
3. the learned difference appears in geometry-sensitive AoD/spatial targets,
   not only in total power or delay.

If the first condition holds but the second does not, more data or a less noisy
statistical target should be tested before attributing the failure specifically
to Map-ID representation. If the second holds while B3-S-ID fails, the result
points more directly to the diffusion structure, objective, or optimization.

## Queue all diagnostics in tmux on GPU 0

The queue wrapper runs the three diagnostics sequentially, reusing the unified
generated tensor artifact when it already exists. If the artifact is absent,
it first runs `evaluate_scene_id_protocol.py` to create it. Reports and logs
are stored in timestamped directories.

```bash
tmux new-session -d -s scene_diag_gpu0 \
  "cd '$PWD' && CUDA_VISIBLE_DEVICES=0 bash scripts/run_scene_diagnostics_gpu0.sh"
```

Attach to the session or follow the queue log with:

```bash
tmux attach-session -t scene_diag_gpu0
tail -f artifacts/logs/scene_diagnostics/*/queue.log
```

Override server paths without editing the script:

```bash
SCENE_CHECKPOINT=/path/to/best.pt \
BASELINE_CHECKPOINT=/path/to/baseline.pt \
PREPARED_ROOT=/path/to/prepared_d2los \
SCENE_PLAN=/path/to/scene_id_seen_map_v1.json \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_scene_diagnostics_gpu0.sh
```
