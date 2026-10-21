# Conditional diffusion extension

This extension keeps the original physics-conditioned CVAE as a baseline and
adds two independent conditional diffusion models.  It does not change the
verified array response, constraint projection, or MIMO-OFDM synthesis
conventions.

## Audit decisions

- D2Los PROP files contain multipath geometry rather than native CSI.  The
  prepared dataset and `MIMOOFDMSynthesizer` remain the single source of truth
  for CSI construction.
- Model B generates a distribution over delay-sorted path lists.  Path count is
  a separate categorical variable.  A structured, masked representation avoids
  treating padded paths as observations and avoids phase/azimuth wrap
  discontinuities.
- Model C generates the real and imaginary parts of a reversible Tx
  angle-delay transform.  It uses one training-set scale instead of per-sample
  normalization so that absolute channel energy is not discarded.
- Both models consume condition values together with validity flags.  Classifier
  free guidance drops only the physical condition; structural information such
  as path count remains available.
- Training selects checkpoints by validation diffusion loss.  Paired CSI NMSE
  is not used as the primary objective because these are conditional
  distribution generators, not deterministic paired reconstructions.

## Existing code audit and implementation map

Directly reused modules:

- `physics_csi.data.dataset`: the persisted cross-map train/validation/test
  split and sharded loader.
- `physics_csi.data.physical_labels`: canonical four-condition definitions and
  validity flags.
- `physics_csi.physics.constraints`: total-power, delay-spread, K-factor, and
  LoS projections for Model B.
- `physics_csi.physics.synthesizer` and `physics_csi.physics.arrays`: the
  verified complex MIMO-OFDM forward model.
- Existing distribution, diversity, rare-condition, and cross-map evaluation
  utilities where their input assumptions remain valid.

New modules are isolated below `physics_csi.models.diffusion`, with a separate
trainer, YAML files, train/generate scripts, and a common comparison layer.
The existing CVAE files, entry points, checkpoints, and loss remain untouched.

The generic diffusion reference provided a useful check for cosine schedules,
epsilon prediction, DDPM/DDIM equations, EMA, and classifier-free guidance.
The wireless cDDIM reference motivated a two-channel complex representation,
but its hard-coded shapes, per-sample maximum normalization, and unclear local
license make its implementation unsuitable for reuse.

Primary risks and mitigations:

- Variable path count: model it categorically and mask every continuous loss.
- Path identity/order: retain delay ordering and sort decoded paths jointly.
- Periodic variables: use sine/cosine pairs for phase and azimuth.
- First-path absolute delay: diffuse it separately in the first slot rather
  than forcing it to zero.
- Constraint-induced distribution shift: save raw values and per-sample
  pre/post projection changes.
- Model C identifiability: do not claim exact path-level K/LoS recovery from a
  finite CSI tensor; label external-estimator results as estimator-dependent.
- Information leakage: compute and cache all normalization constants from the
  training split only, then embed them in every checkpoint.

## Model B: multipath parameter diffusion

The path-count predictor learns `p(L | c)`. Conditional diffusion learns
`p(x | L, c)` for a maximum-length path table. The table stores first delay,
non-negative relative-delay shape, normalized log-power, circular
phase/azimuth coordinates, bounded elevation coordinates, and a direct-path
flag. Loss is normalized by the number of valid feature elements.

The original B0 representation learns relative delay in ns and applies the
legacy projection afterwards. The structured B2/B3 representation divides the
training relative-delay vector by its power-weighted RMS spread. At generation
time the target spread restores the physical scale analytically. B3 likewise
maps learned positive power proportions to target total power and, for the
project-specific LoS definition, allocates direct/scattered power analytically
from K. `PhysicsConstraintLayer` remains a final safety check; its residual
change is reported separately from the analytic parameterization.

Count training optionally uses clipped square-root inverse-frequency weights.
Inference corrects the induced class-prior shift, applies a validation-fitted
temperature, and samples categorically. A forced-L path exists only for
diagnosing `p(x | L,c)` independently of `p(L | c)`.

## Model C: angle-delay diffusion

For the current single-Rx configuration, CSI `[B, 1, T, F]` is reshaped with
the configured Tx array dimensions, transformed by an orthonormal spatial FFT
and frequency IFFT, and packed as `[B, 2, TxRows, TxCols, F]`.  The inverse is
covered by a round-trip test.  Default generation is soft-conditioned (C1): no
post-hoc claim is made that K-factor or LoS can be exactly recovered from a
channel tensor.  An optional energy-proxy rescaling is explicitly labelled as
a CSI-energy proxy, not path-total-power calibration.

## Provenance

The generic DDPM/DDIM equations and cosine schedule follow the public diffusion
literature and were cross-checked against the MIT-licensed
`denoising-diffusion-pytorch` reference repository.  The local implementation
is project-specific and written around arbitrary-rank tensors.  The cDDIM
reference was used only to compare representation choices; code was not copied
because no repository-level license was present in the supplied checkout.

