import pytest

import numpy as np
from shapely.geometry.polygon import Polygon

from pchandler.geometry.core import PointCloudData
from pchandler.geometry.coordinates import rhv2xyz
from pchandler.geometry.fov import FoV
from pchandler.filters import SphericalPolygonFilter, RangeFilter, FoVFilter

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
        assert len(pcd_filtered) == 0


class TestSphericalPolygonFilter:
    def test_spherical_polygon_filter(self, pcd):
        polygon = Polygon([[0.1, 0.6], [0.5, 0.6], [0.5, 1.2], [0.1, 1.2]])
        polygon_filter = SphericalPolygonFilter(polygon)
        pcd_filtered = polygon_filter.extract(pcd)
        assert len(pcd_filtered) < 100000
        assert pcd_filtered.hz.min() >= 0.1
        assert pcd_filtered.hz.max() <= 0.5
        assert pcd_filtered.v.min() >= 0.6
        assert pcd_filtered.v.max() <= 1.2

class TestFoVFilter:
    def test_fov_filter(self, pcd):
        hz = np.linspace(-np.pi, np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        r = np.ones(10)
        rhv = np.stack([r, hz, v], axis=1)
        xyz = rhv2xyz(rhv)
        pcd = PointCloudData(xyz=xyz, numerical_optimization_shift=None)

        fov = FoV(left=-1.5, top=0.2, right=0.7, bottom=1.9)

        fov_filter = FoVFilter(fov)
        expected_mask = np.zeros(10, dtype=np.bool_)
        expected_mask[[3,4,5]] = True
        mask = fov_filter.mask(pcd)
        assert np.all(mask == expected_mask)
        pcd_filtered = fov_filter.extract(pcd)
        assert len(pcd_filtered) == 3