import pytest

import numpy as np
import math

from pchandler.spherical import Angle
from pchandler.geometry.core import PointCloudData
from pchandler.geometry.fov import FoV, _OldFoV, FoVTree, _OldFoVTree
from pchandler.constants import PI, EPS


@pytest.fixture(scope="function", autouse=True)
def pcd():
    return PointCloudData(np.random.rand(10000, 3))


class TestFov:
    def test_properties(self):
        fov = FoV(right = 1.23, top = 0.45, left = -0.12, bottom = 2.78)

        for name in ('top', 'bottom', 'left', 'right', 'horizontal_min', 'horizontal_max',
                     'elevation_min', 'elevation_max'):

            assert hasattr(fov, name)
            assert isinstance(getattr(fov, name), Angle)

        assert fov.right == 1.23
        assert fov.top == 0.45
        assert fov.left == -0.12
        assert fov.bottom == 2.78

    def test_instantiation(self):
        fov = FoV(left=Angle(-0.12), right=Angle(1.23), top=Angle(0.45), bottom=Angle(2.78))
        assert fov.right == 1.23
        assert fov.top == 0.45
        assert fov.left == -0.12
        assert fov.bottom == 2.78

        fov = FoV(left="0deg", right="90deg", top="0rad", bottom="200gon")

        assert math.isclose(fov.left, Angle(0))
        assert math.isclose(fov.right, Angle(np.pi/2))
        assert math.isclose(fov.top, Angle(0))
        assert math.isclose(fov.bottom, Angle(np.pi))

    def test_iter(self):
        fov = FoV(left = -0.12, top = 0.45, right = 1.23, bottom = 2.78)
        for func in (list, tuple):
            vals = func(fov)
            assert isinstance(vals, func)

            for i, expected_value in enumerate((-0.12, 0.45, 1.23, 2.78)):
                assert vals[i] == expected_value

        set_ = set(vals)

        for i in (-0.12, 0.45, 1.23, 2.78):
            assert i in set_

    def test_crosses_pi(self):
        fov_crosses_pi = FoV(left=3.01, right=-2.8, top = 0.3, bottom=2.78)
        assert fov_crosses_pi.crosses_pi


    def test_width(self):
        # Normal case
        fov = FoV(top=0, bottom=2.45, right=1.3, left=0.2)

        assert fov.width() == 1.1

        left = PI - 1
        right = -PI + 1
        fov = FoV(top=0, bottom=2.45, left=left, right=right)
        assert fov.width() == 2

    def test_height(self):
        fov = FoV(top=0.4, bottom=2.45, right=1.3, left=0.2)
        assert math.isclose(fov.height(), 2.05)

    def test_extent(self):
        fov = FoV(top=0.4, bottom=2.45, right=1.3, left=0.2)
        extent = fov.extent()
        assert extent[0] == fov.width()
        assert extent[1] == fov.height()
        assert math.isclose(extent[0], 1.1)
        assert math.isclose(extent[1], 2.05)

    def test_center(self):
        fov = FoV(top=0.4, bottom=2.4, right=1.3, left=0.3)
        center = fov.center()

        assert math.isclose(center[0], 0.8)
        assert math.isclose(center[1], 1.4)

    def test_from_center_with_extent(self):
        fov = FoV.from_center_with_extent(centerpoint=(0.2, 1.0), extent=(0.5, 0.9))
        assert math.isclose(fov.left, -0.05)
        assert math.isclose(fov.right, 0.45)
        assert math.isclose(fov.bottom, 1.45)
        assert math.isclose(fov.top, 0.55)

    def test_union(self):

        fov1 = FoV(top=0.4, bottom=2.4, right=1.3, left=0.3)
        fov2 = FoV(top=0.2, bottom=2.2, right=1.5, left=-1)

        union = fov1.union(fov2)

        assert isinstance(union, FoV)
        assert math.isclose(union.right, 1.5)
        assert math.isclose(union.left, -1)
        assert math.isclose(union.top, 0.2)
        assert math.isclose(union.bottom, 2.4)
        # Not testing for crossing of PI / TWO_PI

    def test_intersect(self):

        fov1 = FoV(top=0.4, bottom=2.4, right=1.3, left=0.3)
        fov2 = FoV(top=0.2, bottom=2.2, right=1.5, left=-1)

        intersect = fov1.intersect(fov2)

        assert isinstance(intersect, FoV)
        assert math.isclose(intersect.right, 1.3)
        assert math.isclose(intersect.left, 0.3)
        assert math.isclose(intersect.top, 0.4)
        assert math.isclose(intersect.bottom, 2.2)

    def test_repr(self):
        fov = FoV(top=0.4, bottom=2.4, right=1.3, left=0.3)
        assert isinstance(repr(fov), str)

    def test_ratio(self):
        fov = FoV(top=0, bottom=1/3, left=0, right=1)
        ratio = fov.ratio()
        assert isinstance(ratio, (float, int))
        assert math.isclose(ratio, 3)

    def test_extend_to_ratio(self):
        n = 10_000
        np.random.seed(42)
        top_samples = np.random.uniform(low=0.0, high=np.pi/2, size=n)
        bottom_samples = np.random.uniform(low=np.pi/2, high=np.pi, size=n)
        left_samples = np.random.uniform(low=-np.pi, high=0, size=n)
        right_samples = np.random.uniform(low=0, high=np.pi, size=n)
        ratios = np.random.randint(low=1, high=10, size=n) / np.random.randint(low=1, high=10, size=n)
        angle_samples = zip(left_samples,right_samples,top_samples,bottom_samples, ratios)


        for angles in angle_samples:
            fov = FoV(left=angles[0], right=angles[1], top=angles[2], bottom=angles[3])
            fov_extended = fov.extend_to_ratio(angles[4])

            assert math.isclose(fov_extended.ratio(), angles[4])
            assert fov_extended.width() - fov.width() >= -EPS and fov_extended.height() - fov.height() >= -EPS



    def test_split(self):
        fov = FoV(top="20deg", bottom=2.4, right=1.3, left="40gon")
        splits = fov.split((2, 4))

        assert len(splits) == 8
        # assert np.allclose([split.height() for split in splits])
        for i, split in enumerate(splits):
            assert math.isclose(split.width(), (fov.right - fov.left)/2)
            assert math.isclose(split.height(), (fov.bottom - fov.top)/4)
            if i > 0:
                if i == 4:
                    assert split.right > splits[i-1].right
                    assert split.left > splits[i-1].left

                if i % 4:
                    assert split.top > splits[i-1].top
                    assert split.bottom > splits[i-1].bottom

    def test_equal_tiles(self):
        fov = FoV(top=0.1, bottom=1.2, right=0.7, left=0.4)

        tiles = fov.equal_tiles(height=0.4, width=0.2)
        assert len(tiles) == 6

    def test_tile(self):
        fov = FoV(top="0.4rad", bottom=2.4, right=1.3, left=0.3)

        fov_by_extent = FoV(left="0deg", top="0gon", right=0.2, bottom=0.2)

        tiles = fov.tile(fov_by_extent)

        assert len(tiles) * len(tiles[0]) == 50
        for rows in tiles:
            for i in rows:
                assert math.isclose(i.width(), 0.2)
                assert math.isclose(i.height(), 0.2)
                assert i.center() != (0, 0)


    def test_quadrants(self):
        fov = FoV(top=0.4, bottom=2.4, right=1.3, left=0.3)
        splits = fov.split((2, 2))

        quadrants = fov.quadrants()

        for i, split in enumerate(splits):
            assert split == quadrants[i]

    def test_merge(self):
        fov1 = FoV(top=0.3, bottom=0.5, right=2.7, left=2.4)
        fov2 = FoV(top=0.3, bottom=0.5, right=2.6, left=2.1)
        fov3 = FoV(top=0.7, bottom=0.9, right=2.5, left=2.2)
        fov4 = FoV(top=0.8, bottom=1.5, right=2.5, left=0.9)

        fov = FoV.merge([fov1, fov2, fov3, fov4])
        assert math.isclose(fov.right, 2.7)
        assert math.isclose(fov.left, 0.9)
        assert math.isclose(fov.top, 0.3)
        assert math.isclose(fov.bottom, 1.5)

        for i in (fov2, fov3, fov4):
            fov1 = fov1.union(i)

        assert fov1 == fov



    def test_old_properties(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)

        assert fov.horizontal_min == fov.left
        assert fov.elevation_min == fov.top
        assert fov.horizontal_max == fov.right
        assert fov.elevation_max == fov.bottom



@pytest.fixture(scope='function')
def new_fov() -> FoV:
    return FoV(left=0.3, top=0.4, right=1.3, bottom=2.4)

@pytest.fixture(scope='function')
def new_fov2() -> FoV:
    return FoV(left=0.5, top=0.6, right=1.7, bottom=2.2)

@pytest.fixture(scope='function')
def old_fov() -> _OldFoV:
    return _OldFoV(horizontal_min=0.3, elevation_min=0.4, horizontal_max=1.3, elevation_max=2.4, unit='rad')

@pytest.fixture(scope='function')
def old_fov2() -> _OldFoV:
    return _OldFoV(horizontal_min=0.5, elevation_min=0.6, horizontal_max=1.7, elevation_max=2.2, unit='rad')

@pytest.fixture(scope='function')
def fov_center() -> tuple[float, float]:
    return 0.5, 0.7

@pytest.fixture(scope='function')
def fov_extent() -> tuple[float, float]:
    return 0.2, 1.2


@pytest.fixture(scope='function')
def new_extent_as_fov() -> FoV:
    return FoV(left=0.3, top=0.4, right=0.4, bottom=0.6)

@pytest.fixture(scope='function')
def old_extent_as_fov() -> _OldFoV:
    return _OldFoV(horizontal_min=0.3, elevation_min=0.4, horizontal_max=0.4, elevation_max=0.6, unit='rad')


class TestCompareOldVsNew:
    def test_attributes(self, new_fov, old_fov):
        for attr in ('horizontal_min', 'horizontal_max', 'elevation_min', 'elevation_max'):
            assert getattr(new_fov, attr) == getattr(old_fov, attr)

    def test_properties(self, new_fov, old_fov):
        for method_name in ('width', 'height', 'extent', 'center', 'ratio'):
            assert getattr(new_fov, method_name)() == getattr(old_fov, method_name)()

    def test_multi_fov_methods(self, new_fov, new_fov2, old_fov, old_fov2):
        assert tuple(new_fov.intersect(new_fov2)) == old_fov.intersect(old_fov2).as_tuple()    #type: ignore
        assert tuple(new_fov.union(new_fov2)) == old_fov.union(old_fov2).as_tuple()            #type: ignore

    def test_center_with_extent(self, fov_center, fov_extent):
        assert (tuple(FoV.from_center_with_extent(fov_center, fov_extent)) ==
                _OldFoV.from_center_with_extent(fov_center, fov_extent).as_tuple())  #type: ignore

    def test_extend_to_ratio(self, new_fov, old_fov):
        ratio = 1.6
        assert tuple(new_fov.extend_to_ratio(ratio)) == old_fov.extend_to_ratio(ratio).as_tuple()
        ratio = 0.3
        assert tuple(new_fov.extend_to_ratio(ratio)) == old_fov.extend_to_ratio(ratio).as_tuple()

    def test_split(self, new_fov, old_fov):
        split_1 = new_fov.split((3,4))
        split_2 = old_fov.split((3,4))

        assert len(split_1) == len(split_2)

        for i, item in enumerate(split_1):
            assert tuple(item) == split_2[i].as_tuple()

    def test_equal_tiles(self, new_fov, old_fov, old_extent_as_fov, new_extent_as_fov):
        eq_tiles_1 = new_fov.tile(new_extent_as_fov)
        eq_tiles_2 = old_fov.tile(old_extent_as_fov)

        for i, items in enumerate(eq_tiles_1):
            for j, item in enumerate(items):

                assert tuple(item) == eq_tiles_2[i][j].as_tuple()


    def test_quadrants(self, new_fov, old_fov):
        quadrant_1 = new_fov.quadrants()
        quadrant_2 = old_fov.quadrants()

        assert len(quadrant_1) == len(quadrant_2)

        for i, item in enumerate(quadrant_1):
            assert tuple(item) == quadrant_2[i].as_tuple()

    def test_merge(self, new_fov, new_fov2, old_fov, old_fov2):
        merged_new = FoV.merge([new_fov2, new_fov])
        merged_old = _OldFoV.merge([old_fov2, old_fov])

        assert tuple(merged_new) == merged_old.as_tuple()


class TestFoVTree:
    def test_initialisation(self, new_fov: FoV):
        new_tree = FoVTree.build_from_tiles(new_fov.tile(FoV(left=0, right=0.1, top=1.3, bottom=1.4)))
        assert isinstance(new_tree, FoVTree)
        assert len(new_tree.children) == 4

    def test_compare_w_old(self, new_fov: FoV, old_fov: _OldFoV):
        new_tree = FoVTree.build_from_tiles(new_fov.tile(FoV(left=0, right=0.1, top=1.3, bottom=1.4)))
        old_tree = _OldFoVTree.build_from_tiles(new_fov.tile(_OldFoV(horizontal_min=0, horizontal_max=0.1, elevation_min=1.3, elevation_max=1.4, unit='rad')))

        assert len(new_tree.children) == len(old_tree.children)
        assert new_tree.depth() == old_tree.depth()

        for i, node in enumerate(new_tree.to_list()):
            assert node == new_tree.to_list()[i]
