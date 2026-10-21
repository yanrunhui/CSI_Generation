# PGC-VAE code audit

The sibling reference repository was inspected directly rather than inferred
from filenames.

## `vae_model.py`

- `WeightsVAE` receives real/imaginary beamspace channel images shaped
  `[batch, 2, antennas, antennas]`.
- Five strided convolution blocks are followed by hard-coded fully connected
  layers. For the supplied 16-by-16 setup, the flattened feature size is 2048.
- The posterior is diagonal Gaussian, parameterized by `fc_mu` and `fc_var`.
- The training script defaults to a latent dimension of 64.
- The decoder maps a latent sample to a real angular grid of shape
  `[batch, 1, resolution, resolution]`; it does not output delays, phases or
  explicit path records.
- Loss combines channel NMSE, an L11/group-sparsity penalty on angular weights,
  and KL divergence to a standard normal prior.

## `train_vae.py`

- Loads pickled train/validation beamspace channels and immediately moves each
  sample to CUDA.
- Builds an AoA/AoD response dictionary and maps decoder weights back to a
  narrowband channel.
- Uses Adam with learning rate `2e-5` by default. A StepLR object is created but
  is not stepped in the supplied loop.
- Saves only a model state dictionary, not optimizer/scheduler/scaler state;
  resume training is not implemented.
- Random generation samples a standard Gaussian latent vector and decodes an
  angular-weight grid.
- The script references `wasserstein_dist` without a local definition and does
  not isolate validation with `eval()`/`no_grad()`.

## `sim_model.py` and `simulator.py`

- `sim_model.py` contains a deterministic channel-to-angular-weight network and
  an unused neural simulator; it is not the physical dictionary itself.
- `simulator.AntPanel` constructs normalized array responses from angular
  direction cosines.
- `Model.getChannelfromCombiArrayResponse` contracts angular weights with a
  precomputed joint AoA/AoD dictionary.
- The active path uses one subcarrier. A legacy method contains a delay phase
  term but is not used by the VAE training pipeline.
- DeepMIMO preprocessing retains power and four angles, while phase and delay
  are generated as fixed zeros in the default path.

## Reuse decision

The reusable concepts are normalized steering vectors, a differentiable
physics dictionary, Gaussian reparameterization, sparsity-aware generation and
channel-domain NMSE. The new project replaces hard-coded CUDA, fixed tensor
sizes, narrowband synthesis, implicit units and incomplete checkpointing.

