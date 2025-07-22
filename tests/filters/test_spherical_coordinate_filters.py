import pytest

import numpy as np
from shapely.geometry.polygon import Polygon

from pchandler.geometry import PointCloudData
from pchandler.filters import SphericalPolygonFilter, RangeFilter

@pytest.fixture(scope='function')
def pcd():
    return PointCloudData(
        np.random.rand(100000,3)*10)

class TestRangeFilter:
    def test_range_filter(self, pcd):
        range_filter = RangeFilter(low=10, high = 70)
        pcd_filtered = range_filter.extract(pcd)
        assert len(pcd_filtered) < 100000
        assert np.all(pcd_filtered.r >= 10)
        assert np.all(pcd_filtered.r <= 70)

    def test_invalid_filter(self, pcd):
        sf_filter = RangeFilter(low=1000, high=1200)
        pcd_filtered = sf_filter.extract(pcd)
        assert pcd_filtered is None


class TestSphericalPolygonFilter:
    def test_spherical_polygon_filter(self, pcd):
        polygon = Polygon([[0.1, 0.6], [0.5, 0.6], [0.5, 1.2], [0.1, 1.2]])
        polygon_filter = SphericalPolygonFilter(polygon)
        pcd_filtered = polygon_filter.extract(pcd)
        assert len(pcd_filtered) < 100000
        assert np.isclose(pcd_filtered.hz.min(), 0.1, atol=0.0002)
        assert np.isclose(pcd_filtered.hz.max(), 0.5, atol=0.0002)
        assert np.isclose(pcd_filtered.v.min(), 0.6, atol=0.0002)
        assert np.isclose(pcd_filtered.v.max(), 1.2, atol=0.0002)
