# Physics and unit conventions

## Raw quantities

- RayVerse `path_loss_db` is attenuation in dB, not received power.
- `path_power_linear = 10 ** (-path_loss_db / 10)` is a dimensionless path-gain ratio.
- `total_power_db = 10 * log10(sum(path_power_linear))` uses that dimensionless reference.
- Absolute dBm requires a transmit power and a complete link-budget convention; it is not inferred.
- Delays are read in ns and converted to seconds only inside the OFDM synthesizer.
- Stored azimuth/elevation angles are radians and are converted to degrees by the adapter.

## Path phase

RayVerse does not store a complex path coefficient or an independent phase.
The initial path phase is therefore defined as

`phase_rad = wrap(-2*pi*carrier_hz*distance_m/c)`.

The OFDM frequency-selective phase uses centered baseband subcarrier offsets:

`exp(-j*2*pi*f_offset[k]*delay_s)`.

Keeping carrier phase and baseband delay phase separate avoids applying the
carrier term twice.

The delay-spread projection scales excess delays around the earliest generated
path while preserving that earliest absolute delay. It therefore controls RMS
spread without silently converting absolute delays into relative delays.

## Arrays

Azimuth is measured in the x-y plane from +x; elevation is measured from the
x-y plane toward +z. The direction vector is

`[cos(el)cos(az), cos(el)sin(az), sin(el)]`.

Array positions are expressed in wavelengths. The default UPA lies in the x-y
plane with half-wavelength spacing and centered element coordinates. Steering
vectors are unit norm, so path power is the energy across all antenna pairs,
not per-element power.

## LoS and K-factor

A zero-interaction normal path is treated as a direct path. A rooftop
diffraction path (`path_type == -1`) is never marked direct. Rician K-factor is
defined only when a direct component and at least one remaining component both
have positive power:

`K = direct_power / remaining_power`.

NLoS samples carry `k_factor_valid = false`. An optional dominant-path ratio
may be evaluated separately, but is not labeled as Rician K-factor.

