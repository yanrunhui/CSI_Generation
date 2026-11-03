# B3-S-Geo-lite: Geometry Feature Extraction and Probe

This is the low-cost gate before implementing geometry-conditioned diffusion.
It uses real data only and answers whether map-global building statistics add
predictive value for AoD, Power–AoD coupling, and complex Tx spatial
correlation.

## Geometry features

`extract_geometry_features.py` reads `buildings_complete/<map_id>.json`. Each
file is treated as a list of building-footprint polygons in the same 0–256
coordinate frame as RayVerse. No path, CSI, LoS, power, delay, Tx, or Rx label
is used to construct geometry features.

The first version extracts:

- summed footprint fraction, building-count density, and perimeter density;
- log-area, log-perimeter, edge-length, and compactness distributions;
- a length-weighted axial edge-orientation histogram over 0–180 degrees;
- a 4×4 area-weighted building-centroid layout histogram.

The coverage feature is the sum of polygon areas divided by map area. It is
not an exact polygon union and therefore assumes footprints do not overlap.

## Probe models

All targets are standardized using training data only.

- **P0:** physical conditions plus a learned global null context;
- **P1:** physical conditions plus a categorical Map embedding;
- **P2:** physical conditions plus standardized geometry and a two-layer
  Geometry Encoder.

Targets contain circularity-safe AoD summaries, power-weighted AoD second
moments and strongest-path direction, and complex Tx correlations at several
horizontal, vertical, and diagonal UPA offsets.

Correct P2 geometry is also compared with cyclically shuffled geometry. This
separates genuine geometry use from the effect of adding network parameters.

## Protocols

The command runs two independent protocols:

1. `seen_map_spatial` reuses the B3-S-ID receiver-spatial plan and compares
   P0, P1, P2, wrong Map ID, and shuffled geometry.
2. `unseen_map` reuses the original prepared cross-map splits. Validation and
   test Maps are completely absent from training. Only P0 and P2 are defined;
   a categorical P1 embedding cannot represent an unseen Map.

Geometry normalization gives every training Map equal weight. No validation
or test geometry enters the normalization statistics.

## Server command

Activate the intended Python environment and launch on physical GPU 0. The
wrapper first reads `raw_root` from the prepared manifest:

```bash
tmux new-session -d -s geo_probe_gpu0 \
  "cd '$PWD' && CUDA_VISIBLE_DEVICES=0 PYTHON_BIN='$(command -v python)' bash scripts/run_geometry_probe_gpu0.sh"
```

If the manifest records a path from another machine, override it explicitly:

```bash
tmux new-session -d -s geo_probe_gpu0 \
  "cd '$PWD' && RAW_ROOT='/server/path/to/D2Los_Data' CUDA_VISIBLE_DEVICES=0 PYTHON_BIN='$(command -v python)' bash scripts/run_geometry_probe_gpu0.sh"
```

Monitor it with:

```bash
tmux attach-session -t geo_probe_gpu0
tail -f artifacts/logs/geometry_probe/*/queue.log
```

The wrapper extracts geometry once, caches it under `artifacts/geometry/`, and
runs both protocols for seeds 2026, 2027, and 2028. Set
`REBUILD_GEOMETRY=1` to deliberately replace the cache after source changes.

Manual commands are:

```bash
python scripts/extract_geometry_features.py \
  --raw-root /path/to/D2Los_Data \
  --output artifacts/geometry/map_global_building_features.json

python scripts/run_geometry_probe.py \
  --prepared-root artifacts/prepared_d2los \
  --scene-plan artifacts/splits/scene_id_seen_map_v1.json \
  --geometry-features artifacts/geometry/map_global_building_features.json \
  --protocol both \
  --device cuda \
  --output artifacts/reports/geometry_probe.json
```

## Decision gate

Proceed to B3-S-Geo-lite diffusion only when all of the following are observed
across seeds:

1. P2 beats P0 for AoD and Power–AoD targets;
2. correct geometry beats shuffled geometry;
3. the improvement also appears in map-macro metrics rather than being driven
   by a few maps;
4. P2 shows a positive result on completely held-out test Maps;
5. complex spatial-correlation targets do not materially regress.

Seen-map P2 can use the geometry vector as a map fingerprint, so the unseen-map
protocol is the stronger evidence that the selected statistics encode shared
geometric structure rather than merely memorizing Map identity.

## Residual Geometry Probe

If direct P2 uses correct geometry but still underperforms P0 on unseen Maps,
run the residual probe before considering diffusion:

```text
y_hat = frozen_P0(c) + alpha * tanh(Delta_geo(c, g))
```

For every seed, P0 is trained once and frozen. The geometry residual has a
zero-initialized output layer, starts from `alpha=0.05`, is bounded by
`alpha<=0.5`, applies 20% geometry dropout, and penalizes correction energy.
It therefore begins exactly at P0 and can only learn a limited scene-specific
correction.

Three nested geometry variants are evaluated:

- **G1:** density, footprint/edge size distributions, and compactness;
- **G2:** G1 plus length-weighted edge orientation;
- **G3:** G2 plus the absolute 4×4 layout histogram.

Each variant reports its nearest-training-Map geometry distance for every
held-out Map, maximum feature z-score, and the correlation between OOD distance
and residual improvement.

Launch the residual probe on physical GPU 0:

```bash
tmux new-session -d -s residual_geo_gpu0 \
  "cd '$PWD' && CUDA_VISIBLE_DEVICES=0 PYTHON_BIN='$(command -v python)' bash scripts/run_residual_geometry_probe_gpu0.sh"
```

Monitor it with:

```bash
tmux attach-session -t residual_geo_gpu0
tail -f artifacts/logs/residual_geometry_probe/*/queue.log
```

The diffusion gate is passed only when at least one G variant beats both frozen
P0 and shuffled geometry across seeds on completely held-out Maps, with
positive AoD, Power–AoD, and spatial-correlation reductions.
