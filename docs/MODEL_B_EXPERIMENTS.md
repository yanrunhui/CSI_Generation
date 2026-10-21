# Model B structured-physics ablations and condition-generalization protocols

This extension preserves the original Model B as B0 and separates the proposed
changes so they can be attributed rather than bundled without evidence.

## Model ablations

| Name | Configuration | Change |
| --- | --- | --- |
| B0 | `configs/multipath_diffusion_b0.yaml` | Original count loss and post-hoc projection |
| B1 | `configs/multipath_diffusion_b1.yaml` | Smoothed count balancing plus prior-corrected probability sampling |
| B2 | `configs/multipath_diffusion_b2.yaml` | B1 plus RMS-normalized relative-delay shape |
| B3 | `configs/multipath_diffusion.yaml` | B2 plus analytic total-power and LoS K-factor parameterization |
| B3-S-ID | `configs/multipath_diffusion_scene_id.yaml` | B3 plus categorical RayVerse Map ID conditioning for seen-map statistical generation |

B3-S-ID uses a different, within-map spatial holdout and therefore must be
compared with the matched `multipath_diffusion_seen_map_baseline.yaml` run, not
with a B3 checkpoint trained on the original cross-map split. See
[`B3_SCENE_ID.md`](B3_SCENE_ID.md) for the split, training, generation and
Map Shuffle evaluation protocol.

All versions retain the same masked Path Transformer.  Delay, power, phase and
angles remain members of one jointly denoised path table.  B2/B3 do not create
independent generators for physical parameter families.

Because B1--B3 use a weighted count term, their scalar training loss is not on
the same scale as B0. Do not rank ablations by cross-run training loss; use the
common held-out distribution and physical metrics below.

The new path-count loss uses a clipped square-root inverse-frequency weight,
not full inverse frequency.  Since weighted cross entropy changes the learned
class prior, inference subtracts `log(class_weight)` before temperature
calibration and categorical sampling.  `--path-count L` bypasses the count head
only for a diagnostic of the continuous diffusion branch.

## Path-count censoring audit

Run the raw-data audit over all source files before interpreting class 5:

```bash
python scripts/audit_deepmimo.py --max-files 0 \
  --output artifacts/audits/deepmimo_all.json
```

`max_path_class_is_censored=true` means class 5 contains both true five-path
links and links truncated from more than five rays.  New prepared shards also
store `original_path_count` and `path_count_truncated`; re-preparation is needed
to populate those fields.  The default `overflow_policy: error` prevents silent
truncation.

## Pre-registering a continuous condition region

First audit support without looking at model outcomes:

```bash
python scripts/audit_condition_support.py \
  --field delay_spread_ns --bins 20 \
  --output artifacts/experiments/delay_support.json
```

Choose one interval with enough train and independent test examples, and record
that choice before training.  The example below uses `[40, 60)` ns only as a
placeholder; it must be replaced if the support audit says it is unsuitable.

Create matched plans for the same interval:

```bash
python scripts/create_condition_experiment.py \
  --name delay_40_60_id --mode id \
  --field delay_spread_ns --lower 40 --upper 60 \
  --output artifacts/experiments/delay_40_60_id.json

python scripts/create_condition_experiment.py \
  --name delay_40_60_sparse_64 --mode sparse --sparse-retain-count 64 \
  --field delay_spread_ns --lower 40 --upper 60 \
  --output artifacts/experiments/delay_40_60_sparse_64.json

python scripts/create_condition_experiment.py \
  --name delay_40_60_ood --mode ood \
  --field delay_spread_ns --lower 40 --upper 60 \
  --output artifacts/experiments/delay_40_60_ood.json
```

For a sparse-condition curve, repeat with nested retained counts such as
`16, 64, 256, 1024`. The same seed produces nested subsets. All three plans use
the same outside-region validation view; OOD additionally removes the region
from training. Normalization statistics are
recomputed from each plan's training view and tagged with the plan fingerprint,
so a full-training statistics cache cannot leak held-out conditions.

## Training

Every ablation/plan pair must use its own checkpoint directory:

```bash
python scripts/train_multipath_diffusion.py \
  --config configs/multipath_diffusion.yaml \
  --experiment-plan artifacts/experiments/delay_40_60_ood.json \
  --checkpoint-dir artifacts/checkpoints/b3_delay_40_60_ood
```

Do not resume a B0 checkpoint into B2/B3: the learned delay coordinate has a
different physical meaning.  Resume only within the same checkpoint-embedded
configuration and experiment plan.

After training, calibrate the count posterior on that experiment's validation
view without changing model weights:

```bash
python scripts/calibrate_path_count.py \
  --checkpoint artifacts/checkpoints/b3_delay_40_60_ood/best.pt \
  --output artifacts/reports/b3_delay_40_60_ood_count_calibration.json \
  --output-checkpoint artifacts/checkpoints/b3_delay_40_60_ood/best_calibrated.pt
```

## Evaluation protocol A: per-condition generation

Each independent test condition is used once to draw one generated channel.
The two populations therefore have the same condition distribution.  Paired
NMSE is intentionally not computed.

```bash
python scripts/evaluate_model_b_protocol.py \
  --checkpoint artifacts/checkpoints/b3_delay_40_60_ood/best_calibrated.pt \
  --protocol per_condition --max-samples 2048 \
  --output artifacts/reports/b3_delay_40_60_ood_per_condition.json
```

Add `--force-reference-path-count` as a diagnostic only.  The difference
between free-count and forced-count results separates count-head failure from
continuous-parameter diffusion failure.

## Evaluation protocol B: fixed dense condition

The evaluator chooses a real medoid-like condition inside the pre-registered
test region and reports the standardized reference-condition radius:

```bash
python scripts/evaluate_model_b_protocol.py \
  --checkpoint artifacts/checkpoints/b3_delay_40_60_id/best_calibrated.pt \
  --protocol fixed_region --max-samples 512 \
  --output artifacts/reports/b3_delay_40_60_id_fixed.json
```

Use fewer references than candidates for a genuinely local comparison.  Do not
describe a large-radius comparison as a fixed-condition test.

## Training-only baselines

Both baselines obey the same experiment plan, so OOD references cannot leak
from the test region into their training reservoir:

```bash
python scripts/evaluate_condition_baseline.py \
  --experiment-plan artifacts/experiments/delay_40_60_ood.json \
  --baseline resampling --protocol per_condition \
  --output artifacts/reports/resampling_delay_40_60_ood.json

python scripts/evaluate_condition_baseline.py \
  --experiment-plan artifacts/experiments/delay_40_60_ood.json \
  --baseline statistical --protocol per_condition \
  --output artifacts/reports/statistical_delay_40_60_ood.json
```

Report Model B, conditional resampling and the statistical multipath baseline
with the same seed, reference indices, condition region and sample budget.

Aggregate the matched reports into the sparse-support curve:

```bash
python scripts/compare_condition_experiments.py \
  artifacts/reports/b3_delay_40_60_ood_per_condition.json \
  artifacts/reports/b3_delay_40_60_sparse_16_per_condition.json \
  artifacts/reports/b3_delay_40_60_sparse_64_per_condition.json \
  artifacts/reports/b3_delay_40_60_sparse_256_per_condition.json \
  artifacts/reports/b3_delay_40_60_id_per_condition.json \
  --output artifacts/reports/delay_40_60_support_curve.json \
  --csv artifacts/reports/delay_40_60_support_curve.csv
```

## Required reporting

- raw and retained path-count histograms, censoring status and per-class recall;
- count NLL/ECE before and after temperature calibration;
- constraint safety residuals separately from analytic parameterization changes;
- full Delay–Power–Phase–AoA/AoD joint MMD/SW and active-path correlation MAE;
- CSI summary MMD/SW, frequency/spatial correlation and diversity;
- condition support count and standardized neighborhood radius;
- ID, sparse-count curve and OOD reports using an untouched test reference;
- no paired CSI NMSE for stochastic conditional draws.
