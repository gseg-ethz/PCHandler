import pytest

import numpy as np

from pchandler.core import PointCloudData
from pchandler.geometry.util import get_outline_polygon, MinMaxPoints

@pytest.fixture(scope='function')
def pcd_simple():
    return PointCloudData([[0,0,0], [1,1,1], [2,2,2]])

@pytest.fixture(scope='function')
def minmax():
    return MinMaxPoints(np.array([0,1,2]), np.array([3,4,5]))

@pytest.fixture(scope='function')
def xy_plane_points():
    return PointCloudData([[-1,-1,0], [-1,1,0], [1,1,0], [1,-1,0], [0.5,0.5,0]])

@pytest.fixture(scope='function')
def xz_plane_points():
    return PointCloudData([[0,0,0], [1,0,1], [1,0,0], [0,0,1], [0.5,0,0.5]])

@pytest.fixture(scope='function')
def yz_plane_points():
    return PointCloudData([[0,0,0], [0,1,1], [0,0,1], [0,1,0], [0, 0.5, 0.5]])


class TestMinMaxPoints:
    def test_init_positional(self):
        min_max_points = MinMaxPoints(np.array([0, 0, 0]), np.array([1, 1, 1]))
        assert np.all(min_max_points.minimum == [0, 0, 0])
        assert np.all(min_max_points.maximum == [1, 1, 1])

    def test_init_kwarg(self):
        min_max_points = MinMaxPoints(maximum=np.array([2, 2, 2]), minimum=np.array([1, 1, 1]))
        assert np.all(min_max_points.minimum == [1, 1, 1])
        assert np.all(min_max_points.maximum == [2, 2, 2])

    def test_init_from_points(self, pcd_simple ):
        min_max_points = MinMaxPoints.from_points(pcd_simple.arr)
        assert np.all(min_max_points.minimum == [0, 0, 0])
        assert np.all(min_max_points.maximum == [2, 2, 2])

        minmax_shifted = MinMaxPoints.from_points(pcd_simple.arr, already_applied_shift_vec=np.ones(3))
        assert np.all(minmax_shifted.minimum == [1, 1, 1])
        assert np.all(minmax_shifted.maximum == [3, 3, 3])

        null_minmax = MinMaxPoints.from_points(np.array([]))
        assert np.all(null_minmax.minimum == [0, 0, 0])
        assert np.all(null_minmax.maximum == [0, 0, 0])

    def test_init_from_bboxes(self, pcd_simple):
        bbox1 = MinMaxPoints.from_points(pcd_simple.arr)
        bbox2 = MinMaxPoints.from_points(pcd_simple.arr+1)
        bbox3 = MinMaxPoints.from_points(pcd_simple.arr+2)
        bbox4 = MinMaxPoints.from_points(pcd_simple.arr+3)

        minmax_limits = MinMaxPoints.from_minmax_points([bbox1, bbox2, bbox3, bbox4])
        assert np.all(minmax_limits.minimum == [0, 0, 0])
        assert np.all(minmax_limits.maximum == [5, 5, 5])

    def test_init_from_self(self, minmax):
        min_max_points = MinMaxPoints.from_minmax_points(minmax)
        assert np.all(min_max_points.minimum == [0,1,2])
        assert np.all(min_max_points.maximum == [3,4,5])

    def test_central_point(self, minmax):
        assert np.all(minmax.central_point == [1.5, 2.5, 3.5])

    def test_extents(self, minmax):
        assert np.all(minmax.extents == [3, 3, 3])

    def test___array__(self, minmax):
        assert np.all(np.array(minmax) == [[0,1,2], [3, 4, 5]])


class TestGetOutlinePolygon:
    def test_xy_plane(self):
        a = PointCloudData((np.random.rand(10000, 3) * 8) + np.array([-3, -1, 2]))
        outline = get_outline_polygon(a, 'xy')
        assert np.allclose(np.array(outline.bounds), [-3, -1, 5, 7], atol=1e-1)

    def test_xz_plane(self):
        a = PointCloudData((np.random.rand(10000, 3) * 8) + np.array([-3, -1, 2]))
        outline = get_outline_polygon(a, 'xz')
        assert np.allclose(np.array(outline.bounds), [-3, 2, 5, 10], atol=1e-1)

    def test_yz_plane(self):
        a = PointCloudData((np.random.rand(10000, 3) * 8) + np.array([-3, -1, 2]))
        outline = get_outline_polygon(a, 'yz')
        assert np.allclose(np.array(outline.bounds), [-1, 2, 7, 10], atol=1e-1)

    def test_invalid_plane_input(self):
        with pytest.raises(ValueError):
            get_outline_polygon(PointCloudData([[0,0,0], [1,1,1], [2,2,2]]), plane='invalid')