import pytest

import numpy as np
from shapely.geometry import Polygon

from pchandler.v2.filters.cartesian_filters import *


@pytest.fixture(scope="function")
def box_filter():
    return BoxFilter(-np.ones(3), np.ones(3))


class TestBoxFilter:
    def test_init(self, box_filter: BoxFilter):
        assert hasattr(BoxFilter, "extents")
        assert hasattr(box_filter, "minimum")
        assert hasattr(box_filter, "maximum")
        assert np.all(box_filter.extents == 2)
        assert np.all(box_filter.minimum == -1)
        assert np.all(box_filter.maximum == 1)

    @pytest.mark.parametrize('maximum', (np.zeros(3), np.array([2, 0, 1])))
    def test_invalid_init(self, box_filter: BoxFilter, maximum):
        with pytest.raises(ValueError):
            BoxFilter(np.ones(3), maximum)

    @pytest.mark.parametrize('kwargs', ({}, {'optimized_shift': None}))
    def test_mask(self, box_filter: BoxFilter, kwargs):
        array = np.array([[-2, -2, -2],
                          [-0.4, -0.3, -0.2],
                          [1, 1, 1],
                          [2, 2, 2]])

        array = PointCloudData(array, **kwargs)
        mask = box_filter.mask(array)
        assert mask.shape == (array.shape[0],)
        assert mask.dtype == np.bool_
        assert isinstance(mask, np.ndarray)
        assert np.all(mask == np.array([False, True, True, False]))


class TestSphereFilter:
    def test_init(self):
        raise NotImplementedError

    def test_mask(self):
        raise NotImplementedError


class TestPolygonFilter:
    def test_init(self):
        raise NotImplementedError

    def test_mask(self):
        raise NotImplementedError