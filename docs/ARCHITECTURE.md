# Architecture

```text
RayVerse PROP v2
      |
      v
streaming adapter -- units/provenance --> sharded multipath dataset
      |                                      |
      |                                      +--> map/spatial split manifest
      v
physical labels [power, delay spread, K validity, LoS]
      |
      v
conditional path encoder --> q(z | paths, conditions)
                                  |
                                  v
conditions --> embedding --> conditional multipath decoder
                                  |
                  path count + validity + explicit path parameters
                                  |
                                  v
                 differentiable physics constraints
                                  |
                                  v
                    MIMO-OFDM synthesizer
                                  |
                                  v
                     complex CSI [B, 1, 64, 64]
```

The decoder learns unspecified joint statistics. Hard constraints only project
quantities whose definitions are explicit and feasible. Conditions with missing
semantics, especially NLoS Rician K-factor, carry validity masks rather than
fabricated targets.

Training reconstruction uses delay-sorted ground-truth paths, avoiding a
combinatorial path assignment in the first version. Generated path count is
sampled from a conditional categorical head, and the first `L` delay-sorted
slots become active.

