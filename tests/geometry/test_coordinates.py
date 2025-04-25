import pytest

import numpy as np

from pchandler.geometry.coordinates import CoordinateSet3D


class TestCoordinates3D:
    arr: CoordinateSet3D = CoordinateSet3D(np.random.randn(100, 3))

    def test_num_pts(self):
        assert len(self.arr) == 100
        assert self.arr.num_points == 100
