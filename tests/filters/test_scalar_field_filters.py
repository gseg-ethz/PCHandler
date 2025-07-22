import pytest

import numpy as np

from pchandler.geometry import PointCloudData
from pchandler.filters import ScalarFieldFilter, ScalarFieldPercentileFilter


@pytest.fixture(scope='function')
def pcd():
    return PointCloudData(
        np.random.rand(10,3)*100,
        intensity=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int16),
        reflectance=np.array([0.5, 0.2, 0.7, 0.1, .3, .4, .5,.8, 9.2, 1.0]),
        rgb=np.random.randint(0, 256, (10,3), dtype=np.uint8),
        normals=np.random.rand(10,3))

class TestScalarFieldFilter:
    def test_intensity_filter(self, pcd):
        sf_filter = ScalarFieldFilter('intensity', upper_bound=7, lower_bound=3)
        pcd_filtered = sf_filter.extract(pcd)
        assert len(pcd_filtered.intensity) == 5
        assert np.all(pcd_filtered.intensity == np.array([3, 4, 5, 6, 7]))

    def test_reflectance_filter(self, pcd):
        sf_filter = ScalarFieldFilter('reflectance', upper_bound=0.8, lower_bound=0.2)
        pcd_filtered = sf_filter.extract(pcd)
        assert len(pcd_filtered.reflectance) == 7
        assert np.allclose(pcd_filtered.reflectance, np.array([0.5, 0.2, 0.7, 0.3, 0.4, 0.5, 0.8]))

    def test_invalid_filter(self, pcd):
        sf_filter = ScalarFieldFilter('intensity', upper_bound=100, lower_bound=100)
        pcd_filtered = sf_filter.extract(pcd)
        assert pcd_filtered is None

class TestScalarFieldPercentileFilter:
    def test_intensity_filter(self, pcd):
        sf_filter = ScalarFieldPercentileFilter('intensity', lower_percentile=25, upper_percentile=75)
        pcd_filtered = sf_filter.extract(pcd)
        assert len(pcd_filtered.intensity) == 4
        assert np.all(pcd_filtered.intensity == np.array([3, 4, 5, 6]))

    def test_reflectance_filter(self, pcd):
        sf_filter = ScalarFieldPercentileFilter('reflectance', lower_percentile=25, upper_percentile=75)
        pcd_filtered = sf_filter.extract(pcd)
        assert len(pcd_filtered.reflectance) == 4
        assert np.allclose(pcd_filtered.reflectance, np.array([0.5, 0.7, 0.4, 0.5]))

    def test_invalid_filter(self, pcd):
        sf_filter = ScalarFieldPercentileFilter('intensity', lower_percentile=98, upper_percentile=99)
        pcd_filtered = sf_filter.extract(pcd)
        assert pcd_filtered is None