from struct import pack

import numpy as np

from flora_core.demonstrations.spacemouse import _read_motion


def test_spacemouse_reads_split_and_combined_reports():
    class Device:
        def __init__(self, reports):
            self.reports = iter(reports)

        def read(self, size):
            return next(self.reports, b"")

    axes = np.zeros(6)
    translation, rotation = _read_motion(
        Device([pack("<B3h", 1, 35, 70, -105), pack("<B3h", 2, 140, 175, 210)]),
        axes,
    )
    np.testing.assert_allclose(translation, [0.2, 0.1, 0.3])
    np.testing.assert_allclose(rotation, [0.5, 0.4, -0.6])

    translation, rotation = _read_motion(Device([pack("<B6h", 1, 0, 0, 0, 0, 0, 0)]), axes)
    np.testing.assert_array_equal(translation, np.zeros(3))
    np.testing.assert_array_equal(rotation, np.zeros(3))
