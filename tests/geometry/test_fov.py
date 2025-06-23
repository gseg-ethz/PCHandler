import pytest

import numpy as np

from src.pchandler.v2.geometry.coordinates import CartesianCoordinates, SphericalCoordinates
from src.pchandler.v2.geometry.fov import FoV

@pytest.fixture
def fov():
    return FoV(right = 1.23, top = -0.45, left = 0.12, bottom = 2.78)


class TestFov:
    def test_properties(self, fov):

        for name in ('top', 'bottom', 'left', 'right', 'horizontal_min', 'horizontal_max',
                     'vertical_min', 'vertical_max'):

            assert hasattr(fov, name)
            assert isinstance(getattr(fov, name), (float, int, np.ndarray))

        assert fov.right == 1.23
        assert fov.top == -0.45
        assert fov.left == 0.12
        assert fov.bottom == 2.78

    def test_iter(self, fov):
        for func in (list, tuple, np.asarray):
            vals = func(fov)
            if func == np.asarray:
                isinstance(vals, np.ndarray)
            else:
                isinstance(vals, func)

            for i, expected_value in enumerate((1.23, -0.45, 0.12, 2.78)):
                assert vals[i] == expected_value

        set_ = set(vals)

        for i in (1.23, -0.45, 0.12, 2.78):
            assert i in set_

    def test_crosses_pi(self):
        raise NotImplementedError

    def test_from_spherical(self):
        raise NotImplementedError

    def test_width(self):
        raise NotImplementedError

    def test_height(self):
        raise NotImplementedError

    def test_extent(self):
        raise NotImplementedError

    def test_center(self):
        raise NotImplementedError

    def test_from_center_with_extent(self):
        raise NotImplementedError

    def test_union(self):
        raise NotImplementedError

    def test_intersect(self):
        raise NotImplementedError

    def test_ratio(self):
        raise NotImplementedError

    def test_repr(self):
        raise NotImplementedError

    def test_extend_to_ratio(self):
        raise NotImplementedError

    def test_split(self):
        raise NotImplementedError

    def test_equal_tiles(self):
        raise NotImplementedError

    def test_tile(self):
        raise NotImplementedError

    def test_quadrants(self):
        raise NotImplementedError

    def test_merge(self):
        raise NotImplementedError

    def test_old_properties(self):
        raise NotImplementedError

