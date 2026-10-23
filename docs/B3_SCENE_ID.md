# B3-S-ID: scene-ID-conditioned multipath diffusion

B3-S-ID extends B3 from `p(P | c)` to `p(P | c, map_id)`. It remains a
statistical channel generator, not a position-to-channel predictor. The
existing structured delay, total-power, K-factor, LoS, phase, angle and CSI
synthesis code is shared with B3.

## What is new

- The categorical RayVerse Map ID is mapped to a stable contiguous index.
- `MapEmbedding` is fused with the physical-condition embedding by an MLP.
- The fused embedding conditions both the path-count predictor and every
  diffusion denoising step.
- The stable mapping and the complete split plan are embedded in checkpoints.
- Unknown Map IDs raise an error; they never receive a random embedding.
- Training logs scene-branch gradient norm and nonzero rate.
- Generation reports empirical condition support for the requested map.
- Evaluation includes per-map statistics and a same-noise Map Shuffle
  ablation, with an optional matched physics-only B3 baseline.

## Fair seen-map split

The original prepared shards use a cross-map split. B3-S-ID must not be judged
on Map IDs absent from its embedding table. `create_scene_id_split.py`
therefore reuses all prepared records and groups them by
`(map_id, receiver_tile_x, receiver_tile_y)`. Each map contributes disjoint
tiles to train, validation and test. Records in the same receiver tile stay in
the same split even when they originate from different transmitters.

The plan includes a source-overlap audit. Source IDs may span receiver regions
because source position is not an input to this model; the spatial receiver
regions themselves never overlap. Set `--guard-band-tiles 1` if an experiment
requires an empty tile band between training and held-out regions (this drops
near-boundary training samples).

## Commands

Create one immutable plan and use it for both models:

```bash
python scripts/create_scene_id_split.py \
  --prepared-root artifacts/prepared_d2los \
  --output artifacts/splits/scene_id_seen_map_v1.json \
  --tile-size-m 10 \
  --guard-band-tiles 0
```

Train the matched physics-only baseline and B3-S-ID independently:

```bash
python scripts/train_scene_id_diffusion.py \
  --config configs/multipath_diffusion_seen_map_baseline.yaml \
  --scene-plan artifacts/splits/scene_id_seen_map_v1.json

python scripts/train_scene_id_diffusion.py \
  --config configs/multipath_diffusion_scene_id.yaml \
  --scene-plan artifacts/splits/scene_id_seen_map_v1.json
```

Optionally initialize non-scene modules from an existing B3 checkpoint. This
starts a new optimizer and experiment history, because it is fine-tuning rather
than exact resume:

```bash
python scripts/train_scene_id_diffusion.py \
  --config configs/multipath_diffusion_scene_id.yaml \
  --scene-plan artifacts/splits/scene_id_seen_map_v1.json \
  --initialize-from-b3 artifacts/checkpoints/multipath_diffusion/best.pt \
  --checkpoint-dir artifacts/checkpoints/b3_scene_id_from_b3
```

Calibrate path-count temperature on the embedded spatial validation split:

```bash
python scripts/calibrate_path_count.py \
  --checkpoint artifacts/checkpoints/b3_scene_id/best.pt \
  --output artifacts/reports/b3_scene_id_calibration.json \
  --output-checkpoint artifacts/checkpoints/b3_scene_id/best_calibrated.pt
```

Generate from one known Map ID and physical condition:

```bash
python scripts/generate_scene_id_diffusion.py \
  --checkpoint artifacts/checkpoints/b3_scene_id/best_calibrated.pt \
  --map-id map_10 \
  --total-power-db -90 \
  --delay-spread-ns 100 \
  --k-factor-db 10 \
  --los-status 1 \
  --num-samples 1000 \
  --output artifacts/generated/b3_scene_id_map_10.pt
```

Run the unified independent-test comparison:

```bash
python scripts/evaluate_scene_id_protocol.py \
  --scene-checkpoint artifacts/checkpoints/b3_scene_id/best_calibrated.pt \
  --baseline-checkpoint artifacts/checkpoints/b3_seen_map_baseline/best.pt \
  --metrics-device cuda \
  --max-samples 2048 \
  --output artifacts/reports/b3_scene_id_vs_b3.json \
  --generated-output artifacts/generated/b3_scene_id_evaluation.pt
```

Audit whether Map ID is scientifically necessary after matching physical
conditions (this operates only on real held-out data). The report also compares
cross-Map distances against condition-matched splits from within the same Map,
which acts as a finite-sample null:

```bash
python scripts/audit_scene_separation.py \
  --scene-plan artifacts/splits/scene_id_seen_map_v1.json \
  --device cuda \
  --output artifacts/reports/real_map_separation.json
```

For the paper-level audit, use the strict protocol. It enforces one-to-one
matching without replacement, exact LoS and K-validity matching, 20 repeated
draws, Map-cluster bootstrap confidence intervals, coordinate/Tx-layout
checks, and both global-coordinate and receiver-PCA-aligned spatial metrics:

```bash
python scripts/audit_scene_separation_strict.py \
  --scene-plan artifacts/splits/scene_id_seen_map_v1.json \
  --prepared-root artifacts/prepared_d2los \
  --split test \
  --maximum-condition-distance 1.0 \
  --samples-per-comparison 64 \
  --repeats 20 \
  --maximum-map-pairs 100 \
  --maximum-samples-per-map 256 \
  --map-pair-selection random \
  --cluster-bootstrap-samples 2000 \
  --device cuda \
  --seed 2026 \
  --output artifacts/reports/real_map_scene_structure_strict.json
```

Rotation alignment is a coordinate-sensitivity analysis. The prepared records
contain Tx/Rx positions but no per-transmitter antenna yaw/pitch/roll, so the
report explicitly records that physical array orientation cannot be verified.

After unified evaluation has saved generated tensors, diagnose whether the
correct Map ID improves AoD direction, angular clustering, power-angle
coupling, the full spatial correlation matrix, and normalized complex CSI:

```bash
python scripts/diagnose_scene_conditioning.py \
  --generated artifacts/generated/b3_scene_id_evaluation.pt \
  --device cuda \
  --output artifacts/reports/b3_scene_id_structure_diagnostics.json
```

The central claim is supported only when the correct Map ID is closer to that
map's held-out real distribution than both the shuffled Map ID and the matched
physics-only baseline. Output change alone is not evidence of better fidelity.
