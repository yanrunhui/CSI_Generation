# D2Los / RayVerse data audit

Dataset root:

`/Users/yanrunhui/CSI_model/deepmimo_scenarios/D2Los_Data`

Observed organization:

- 101 `map_*` directories (`map_0` through `map_100`).
- 97 non-empty map directories and 9648 `.propbin.gz` source files.
- `map_56`, `map_62`, `map_63` and `map_67` contain no propbin files.
- `map_48` contains 48 source files; the other non-empty maps observed contain 100.
- `buildings_complete` contains geometry JSON files, which are intentionally not
  used as first-stage model inputs.

The files use RayVerse PROP version 2. Each source file contains a 32-byte
header, a receiver table, a compact FP16 path table and an interaction-chain
table. Available path fields are path loss, distance, delay, AoD/AoA azimuth
and elevation, interaction count, path type and interaction offset.

The format does not contain a precomputed CSI tensor, OFDM bandwidth, antenna
array definition, transmit power or independent path phase. CSI is generated
from the path table under an explicitly selected system configuration.

An inspected `map_0/source_0` file contains 46,431 receiver records and 224,897
path records. Every receiver has 1--5 paths in that file. The preparation code
still audits every selected source and fails by default if a link exceeds the
configured path capacity.

