import pytest

from shapely.geometry import box, Polygon
from pydantic import ValidationError
import numpy as np

from pchandler.core import PointCloudData
from pchandler.filters.gpu import SphericalPolygonFilterGPU, PolygonFilterGPU


@pytest.fixture(scope="function")
def simple_array():
    return np.array([[-2, -2, -2],
                     [-1, -1, -1],
                     [-0.4, -0.3, -0.2],
                     [1, 1, 1],
                     [2, 2, 2]])

@pytest.fixture(scope="function")
def polygon():
    return np.array([[0, 0],])

@pytest.fixture(scope="function")
def sphere_filter():
    return SphericalPolygonFilterGPU(np.array([0, 1, 0.5]))


@pytest.fixture(scope="function")
def polygon_filter():
    polygon = box(-0.31, -0.31, 1.3, 1.3)
    return PolygonFilterGPU(polygon, 'xy')


class TestGPUSpherePolygonFilter:
    def test_init(self, sphere_filter):
        assert hasattr(sphere_filter, "polygon")

    def test_invalid_init(self, sphere_filter):
        with pytest.raises(ValueError):
            SphericalPolygonFilterGPU(np.array([0, 1, 1, 1]), 1.2)

        with pytest.raises(ValidationError):
            SphericalPolygonFilterGPU(np.array([0, 1, 1]), "Fasb")

    @pytest.mark.parametrize('kwargs', ({}, {'optimized_shift': None}))
    def test_mask(self, sphere_filter, simple_array, kwargs):
        pcd = PointCloudData(simple_array, **kwargs)
        mask = sphere_filter.mask(pcd)

        assert mask.shape == (pcd.shape[0],)
        assert mask.dtype == np.bool_
        assert isinstance(mask, np.ndarray)
        assert np.all(mask == np.array([False, False, False, True, False]))


class TestSphericalPolygonFilterGPU:
    @pytest.fixture
    def simple_pcd(self):
        # Create a simple point cloud with known spherical coordinates
        hz = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        v = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        r = np.ones_like(hz)
        xyz = np.column_stack([r * np.cos(v) * np.cos(hz),
                               r * np.cos(v) * np.sin(hz),
                               r * np.sin(v)])
        return PointCloudData(xyz=xyz)

    @pytest.fixture
    def test_polygon(self):
        # Create a polygon that includes points in the middle of our test data
        vertices = [(0.0, 0.3), (0.7, 0.3), (0.7, 0.9), (0.0, 0.9)]
        return Polygon(vertices)

    def test_initialization(self, test_polygon):
        filter_gpu = SphericalPolygonFilterGPU(test_polygon)
        assert isinstance(filter_gpu.polygon, Polygon)
        assert filter_gpu.polygon.equals(test_polygon)

    def test_mask_computation(self, simple_pcd, test_polygon):
        filter_gpu = SphericalPolygonFilterGPU(test_polygon)
        mask = filter_gpu.mask(simple_pcd)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
        assert mask.shape == (len(simple_pcd),)

        # The polygon should include the points with hz between 0.0 and 0.7
        # and v between 0.3 and 0.9
        expected_mask = np.array([False, True, True, False, False])
        assert np.array_equal(mask, expected_mask)

    def test_invalid_polygon(self):
        with pytest.raises(Exception):  # Should raise some form of validation error
            SphericalPolygonFilterGPU([(0, 0), (1, 1)])  # Invalid polygon format

    def test_empty_point_cloud(self, test_polygon):
        empty_pcd = PointCloudData(xyz=np.empty((0, 3)))
        filter_gpu = SphericalPolygonFilterGPU(test_polygon)
        mask = filter_gpu.mask(empty_pcd)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
        assert mask.shape == (0,)
