# Physics-Controllable CSI Generation

This project generates complex MIMO-OFDM CSI by learning a conditional
distribution over explicit multipath parameters and passing generated paths
through a differentiable physics synthesizer.

The original baseline is a conditional VAE, not a deterministic CSI regressor:

`(random latent z, physical conditions c) -> multipath parameters -> constraints -> CSI`.

The repository also contains two independent conditional diffusion branches:

- Model B generates path count and explicit multipath parameters, then applies
  the existing physics constraints and differentiable CSI synthesizer.
- Model C generates a reversible complex Tx angle-delay representation and
  converts it directly back to wideband CSI.

Their architecture and physical-control limitations are documented in
[the diffusion architecture audit](docs/DIFFUSION_ARCHITECTURE.md).
The structured Model B ablations and matched ID/sparse/OOD protocols are in
[the Model B experiment guide](docs/MODEL_B_EXPERIMENTS.md).

## Data and physical definition

The configured dataset is the local RayVerse/D2Los tree:

`/Users/yanrunhui/CSI_model/deepmimo_scenarios/D2Los_Data`

PROP files contain ray parameters rather than stored CSI. Preprocessing converts
path loss to dimensionless path gain, derives carrier phase from propagation
distance, and synthesizes CSI under the configured array and OFDM system.

Default output shape is `[batch, 1, 64, 64]`:

- Rx: 1x1
- Tx: 8x8 UPA
- OFDM subcarriers: 64
- carrier: 3.5 GHz
- bandwidth: 100 MHz (an explicit experiment setting, not PROP metadata)

See [physics conventions](docs/PHYSICS_CONVENTIONS.md), the
[data audit](docs/DATA_AUDIT.md), and the [reference audit](docs/REFERENCE_AUDIT.md).

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[evaluation,test]'
```

Install the optional `pgc` extra only when loading the legacy sibling baseline:

```bash
pip install -e '.[pgc]'
```

## Workflow

Audit raw files:

```bash
python scripts/audit_deepmimo.py --max-files 100
```

Prepare sharded train/validation/test data and synthesize CSI:

```bash
python scripts/prepare_dataset.py
```

The checked-in configuration prepares 100,000 map-balanced samples and selects
at most 32 receivers per source. Change `max_samples` and
`max_receivers_per_source` deliberately for a larger server-side run. Also
update `data.raw_root` if the server mounts D2Los at a different path.

Cross-check the PyTorch synthesizer against an independent NumPy implementation:

```bash
python scripts/verify_synthesizer.py
```

Train or resume the CVAE:

```bash
python scripts/train_cvae.py
python scripts/train_cvae.py --resume artifacts/checkpoints/last.pt
```

Train the diffusion models independently:

```bash
python scripts/train_multipath_diffusion.py \
  --config configs/multipath_diffusion.yaml

python scripts/train_angle_delay_diffusion.py \
  --config configs/angle_delay_diffusion.yaml
```

On first use, either command computes shared normalization statistics from the
training split only and stores them under `artifacts/statistics/`. Checkpoints
include those statistics, model/optimizer/scheduler/scaler state, EMA weights,
RNG state, configuration, history, epoch, and global step. Resume with
`--resume artifacts/checkpoints/<model>/last.pt`.

Diagnose a reconstruction plateau without updating model parameters:

```bash
python scripts/diagnose_checkpoint.py \
  --checkpoint artifacts/checkpoints/last.pt \
  --split validation \
  --batch-size 64 \
  --num-batches 8 \
  --output artifacts/diagnostics/checkpoint_diagnosis.json
```

The diagnostic compares posterior sampling with posterior-mean reconstruction,
checks latent usage and CSI-only gradients, and performs oracle ablations by
replacing predicted phase, angles, delays, or powers with their ground-truth
values before re-synthesizing CSI. The largest oracle NMSE reduction identifies
the dominant path-parameter error source.

Generate multiple samples under one condition:

```bash
python scripts/generate_csi.py \
  --checkpoint artifacts/checkpoints/best.pt \
  --output artifacts/generated/los_example.pt \
  --num-samples 1000 \
  --total-power-db -90 \
  --delay-spread-ns 100 \
  --k-factor-db 5 \
  --los-status 1
```

The equivalent diffusion generation commands are:

```bash
python scripts/generate_multipath_diffusion.py \
  --checkpoint artifacts/checkpoints/multipath_diffusion/best.pt \
  --output artifacts/generated/multipath_diffusion.pt \
  --num-samples 1000 --total-power-db -90 --delay-spread-ns 100 \
  --k-factor-db 5 --los-status 1

python scripts/generate_angle_delay_diffusion.py \
  --checkpoint artifacts/checkpoints/angle_delay_diffusion/best.pt \
  --output artifacts/generated/angle_delay_diffusion.pt \
  --num-samples 1000 --total-power-db -90 --delay-spread-ns 100 \
  --k-factor-db 5 --los-status 1
```

Model C defaults to soft conditional generation. Its optional
`--calibration csi_energy_proxy` rescales average CSI energy only; it is not
reported as exact path-total-power, K-factor, or LoS control.

For NLoS, K-factor is invalid under the default direct-to-remaining definition;
the supplied numeric value is stored with a false validity flag and is not
enforced.

Evaluate against condition-neighbor samples from the independent test split:

```bash
python scripts/evaluate.py \
  --generated artifacts/generated/los_example.pt \
  --output artifacts/evaluation/los_example.json
```

For either diffusion artifact, use the common condition-matched distribution
evaluation and then combine the reports:

```bash
python scripts/evaluate_diffusion.py \
  --generated artifacts/generated/multipath_diffusion.pt \
  --output artifacts/reports/model_B.json

python scripts/evaluate_diffusion.py \
  --generated artifacts/generated/angle_delay_diffusion.pt \
  --output artifacts/reports/model_C.json

python scripts/compare_generators.py \
  --model-b artifacts/reports/model_B.json \
  --model-c artifacts/reports/model_C.json
```

The common comparison uses distribution distances, correlations and diversity;
it deliberately does not score a random generated sample against an arbitrary
test sample with paired NMSE.

Generate the training-only conditional resampling baseline with the same output
schema, then pass it to the same evaluator:

```bash
python scripts/conditional_resampling.py \
  --output artifacts/generated/resampling_los_example.pt \
  --method kernel \
  --num-samples 1000 \
  --total-power-db -90 \
  --delay-spread-ns 100 \
  --k-factor-db 5 \
  --los-status 1
```

## Important safeguards

- Cross-map splitting is the default and its map manifest is persisted.
- Single-map mode assigns complete spatial tiles, not individual neighboring
  receiver points, to splits.
- Conditional resampling only accepts an explicitly supplied reference payload;
  use the training split for baseline experiments.
- Missing path phase is not presented as a measured label.
- Absolute receive power in dBm is not inferred from path loss without a full
  link-budget definition.
- The legacy PGC-VAE remains an independent narrowband baseline; its outputs are
  not repeated across subcarriers to fabricate wideband CSI.

## Local generation status

The repository includes tests under `tests/`, but no training or tests were run
during the local code-generation task. Execute them in the intended server
environment after installing the declared dependencies.

