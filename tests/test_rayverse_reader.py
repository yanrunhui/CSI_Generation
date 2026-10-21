from __future__ import annotations

import gzip
import struct

import numpy as np

from physics_csi.data.rayverse import read_propbin


def test_reads_compact_prop_v2(tmp_path):
    path = tmp_path / "map_0" / "special_points_propbin_3.5GHz" / "source_0.propbin.gz"
    path.parent.mkdir(parents=True)
    header = struct.pack("<II3fIII", 0x50524F50, 2, 1.0, 2.0, 3.0, 1, 1, 0)
    receiver = struct.pack("<i3f2fHHI", 7, 4.0, 5.0, 1.5, -80.0, 103.0, 1, 0, 0)
    path_record = struct.pack(
        "<eeeeeeeHbBHI",
        103.0,
        100.0,
        333.5,
        0.1,
        0.2,
        -0.1,
        -0.2,
        0,
        0,
        0,
        0,
        0,
    )
    with gzip.open(path, "wb") as stream:
        stream.write(header + receiver + path_record)

    propbin = read_propbin(path)
    link = next(propbin.iter_links())
    assert propbin.version == 2
    assert propbin.carrier_hz == 3.5e9
    assert link.receiver_id == 7
    assert link.path_count == 1
    np.testing.assert_allclose(link.delay_ns, [333.5])

