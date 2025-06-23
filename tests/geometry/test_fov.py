import pytest

import numpy as np
import math

from src.pchandler.v2.geometry.core import PointCloudData
from src.pchandler.v2.geometry.fov import FoV
from src.pchandler.v2.constants import PI


@pytest.fixture(scope="function", autouse=True)
def pcd():
    return PointCloudData(np.random.rand(10000, 3))


class TestFov:
    def test_properties(self):
        fov = FoV(right = 1.23, top = 0.45, left = -0.12, bottom = 2.78)

        for name in ('top', 'bottom', 'left', 'right', 'horizontal_min', 'horizontal_max',
                     'elevation_min', 'elevation_max'):

            assert hasattr(fov, name)
            assert isinstance(getattr(fov, name), (float, int, np.ndarray))

        assert fov.right == 1.23
        assert fov.top == 0.45
        assert fov.left == -0.12
        assert fov.bottom == 2.78

    def test_iter(self):
        fov = FoV(right = 1.23, top = 0.45, left = -0.12, bottom = 2.78)
        for func in (list, tuple, np.asarray):
            vals = func(fov)
            if func == np.asarray:
                isinstance(vals, np.ndarray)
            else:
                isinstance(vals, func)

            for i, expected_value in enumerate((1.23, 0.45, -0.12, 2.78)):
                assert vals[i] == expected_value

        set_ = set(vals)

        for i in (1.23, 0.45, -0.12, 2.78):
            assert i in set_

    def test_crosses_pi(self):
        fov_crosses_pi = FoV(right = 3.01, left=-2.8, top = 0.3, bottom=2.78)
        assert fov_crosses_pi.crosses_pi

    def test_from_spherical(self, pcd):
        spher_array = pcd.spher

        fov = FoV.from_spherical(spher_array)
        assert isinstance(fov, FoV)
        assert fov.left == spher_array[:, 1].max()
        assert fov.right == spher_array[:, 1].min()
        assert fov.top == spher_array[:, 2].min()
        assert fov.bottom == spher_array[:, 2].max()

        spher_obj = pcd.to_spherical()
        fov2 = FoV.from_spherical(spher_obj)
        assert isinstance(fov2, FoV)
        assert fov2.left == spher_array[:, 1].max()
        assert fov2.right == spher_array[:, 1].min()
        assert fov2.top == spher_array[:, 2].min()
        assert fov2.bottom == spher_array[:, 2].max()

    def test_width(self):
        # Normal case
        fov = FoV(top=0, bottom=2.45, left=1.3, right=0.2)

        assert fov.width() == 1.1

        right = PI - 1
        left = -PI + 1
        fov = FoV(0, 2.45, left=left, right=right)
        assert fov.width() == 2

    def test_height(self):
        fov = FoV(top=0.4, bottom=2.45, left=1.3, right=0.2)
        assert math.isclose(fov.height(), 2.05)

    def test_extent(self):
        fov = FoV(top=0.4, bottom=2.45, left=1.3, right=0.2)
        extent = fov.extent()
        assert extent[0] == fov.width()
        assert extent[1] == fov.height()
        assert math.isclose(extent[0], 1.1)
        assert math.isclose(extent[1], 2.05)

    def test_center(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)
        center = fov.center()

        assert math.isclose(center[0], 0.8)
        assert math.isclose(center[1], 1.4)

    def test_from_center_with_extent(self):
        fov = FoV.from_center_with_extent(centerpoint=(0.2, 0.2), extent=(0.5, 0.9))
        assert math.isclose(fov.right, -0.05)
        assert math.isclose(fov.left, 0.45)
        assert math.isclose(fov.bottom, 0.65)
        assert math.isclose(fov.top, -0.25)

    def test_union(self):

        fov1 = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)
        fov2 = FoV(top=0.2, bottom=2.2, left=1.5, right=-1)

        union = fov1.union(fov2)

        assert isinstance(union, FoV)
        assert math.isclose(union.left, 1.5)
        assert math.isclose(union.right, -1)
        assert math.isclose(union.top, 0.2)
        assert math.isclose(union.bottom, 2.4)
        # Not testing for crossing of PI / TWO_PI

    def test_intersect(self):

        fov1 = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)
        fov2 = FoV(top=0.2, bottom=2.2, left=1.5, right=-1)

        intersect = fov1.intersect(fov2)

        assert isinstance(intersect, FoV)
        assert math.isclose(intersect.left, 1.3)
        assert math.isclose(intersect.right, 0.3)
        assert math.isclose(intersect.top, 0.4)
        assert math.isclose(intersect.bottom, 2.2)

    def test_ratio(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)
        ratio = fov.ratio()
        assert isinstance(ratio, (float, int))
        assert math.isclose(ratio, 0.5)

    def test_repr(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)
        assert isinstance(repr(fov), str)

    def test_extend_to_ratio(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)
        ref_ratio = 1

        fov2 = fov.extend_to_ratio(ref_ratio)
        assert math.isclose(fov2.top, 0.4)
        assert math.isclose(fov2.bottom, 2.4)

        fov = FoV(top=0.0, bottom=0.5, left=0.5, right=0.0)
        ref_ratio = 2

        fov2 = fov.extend_to_ratio(ref_ratio)
        assert math.isclose(fov2.top, 0.0)
        assert math.isclose(fov2.bottom, 0.5)
        assert math.isclose(fov2.left, 1.0)
        assert math.isclose(fov2.right, 0.0)

    def test_split(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)
        splits = fov.split((2, 4))

        assert len(splits) == 8
        for i, split in enumerate(splits):
            assert math.isclose(split.width(), 0.5)
            assert math.isclose(split.height(), 0.5)
            if i > 0:
                if i == 4:
                    assert split.left > splits[i-1].left
                    assert split.right > splits[i-1].right

                if i % 4:
                    assert split.top > splits[i-1].top
                    assert split.bottom > splits[i-1].bottom

    def test_equal_tiles(self):
        fov = FoV(top=0.1, bottom=1.2, left=0.7, right=0.4)

        tiles = fov.equal_tiles(height=0.4, width=0.2)
        assert len(tiles) == 6

    def test_tile(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)

        fov_by_extent = FoV.from_center_with_extent((0, 0), (0.2, 0.2))

        tiles = fov.tile(fov_by_extent)

        assert len(tiles) * len(tiles[0]) == 50
        for rows in tiles:
            for i in rows:
                assert math.isclose(i.width(), 0.2)
                assert math.isclose(i.height(), 0.2)
                assert i.center() != (0, 0)


    def test_quadrants(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)
        splits = fov.split((2, 2))

        quadrants = fov.quadrants()

        for i, split in enumerate(splits):
            assert split == quadrants[i]

    def test_merge(self):
        fov1 = FoV(top=0.3, bottom=0.5, left=2.7, right=2.4)
        fov2 = FoV(top=0.3, bottom=0.5, left=2.6, right=2.1)
        fov3 = FoV(top=0.7, bottom=0.9, left=2.5, right=2.2)
        fov4 = FoV(top=0.8, bottom=1.5, left=2.5, right=0.9)

        fov = FoV.merge([fov1, fov2, fov3, fov4])
        assert math.isclose(fov.left, 2.7)
        assert math.isclose(fov.right, 0.9)
        assert math.isclose(fov.top, 0.3)
        assert math.isclose(fov.bottom, 1.5)

        for i in (fov2, fov3, fov4):
            fov1 = fov1.union(i)

        assert fov1 == fov



    def test_old_properties(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)

        assert fov.horizontal_min == fov.right
        assert fov.horizontal_max == fov.left
        assert fov.elevation_max == fov.bottom
        assert fov.elevation_min == fov.top


class TestFoVTree:
    def test_initialisation(self):
        raise NotImplementedError