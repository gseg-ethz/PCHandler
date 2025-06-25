import pytest

from shapely.geometry import box

from pchandler.v2.filters.cartesian_filters import *


@pytest.fixture(scope="function")
def simple_array():
    return np.array([[-2, -2, -2],
                     [-1, -1, -1],
                     [-0.4, -0.3, -0.2],
                     [1, 1, 1],
                     [2, 2, 2]])

@pytest.fixture(scope="function")
def box_filter():
    return BoxFilter(-np.ones(3), np.ones(3))

@pytest.fixture(scope="function")
def sphere_filter():
    return SphereFilter(np.array([0.5, 0.5, 0.5]), 1.3)

@pytest.fixture(scope="function")
def polygon_filter():
    polygon = box(-0.31, -0.31, 1.3, 1.3)
    return PolygonFilter(polygon, 'xy')


class TestBoxFilter:
    def test_init(self, box_filter: BoxFilter):
        assert hasattr(BoxFilter, "extents")
        assert hasattr(box_filter, "minimum")
        assert hasattr(box_filter, "maximum")
        assert np.all(box_filter.extents == 2)
        assert np.all(box_filter.minimum == -1)
        assert np.all(box_filter.maximum == 1)

    @pytest.mark.parametrize('maximum', (np.zeros(3), np.array([2, 0, 1])))
    def test_invalid_init(self, maximum):
        with pytest.raises(ValueError):
            BoxFilter(np.ones(3), maximum)

    @pytest.mark.parametrize('kwargs', ({}, {'optimized_shift': None}))
    def test_mask(self, box_filter: BoxFilter, kwargs, simple_array):
        array = PointCloudData(simple_array, **kwargs)
        mask = box_filter.mask(array)
        assert mask.shape == (array.shape[0],)
        assert mask.dtype == np.bool_
        assert isinstance(mask, np.ndarray)
        assert np.all(mask == np.array([False, True, True, True, False]))


class TestSphereFilter:
    def test_init(self, sphere_filter):
        assert hasattr(sphere_filter, "sphere_center")
        assert hasattr(sphere_filter, "radius")
        assert np.all(sphere_filter.sphere_center == 0.5)
        assert np.all(sphere_filter.radius == 1.3)

    def test_invalid_init(self, sphere_filter):
        with pytest.raises(ValueError):
            SphereFilter(np.array([0, 1, 1, 1]), 1.2)

        with pytest.raises(TypeError):
            SphereFilter(np.array([0, 1, 1]), "Fasb")

    @pytest.mark.parametrize('kwargs', ({}, {'optimized_shift': None}))
    def test_mask(self, sphere_filter, simple_array, kwargs):
        pcd = PointCloudData(simple_array, **kwargs)
        mask = sphere_filter.mask(pcd)

        assert mask.shape == (pcd.shape[0],)
        assert mask.dtype == np.bool_
        assert isinstance(mask, np.ndarray)
        assert np.all(mask == np.array([False, False, False, True, False]))


class TestPolygonFilter:
    def test_init(self, polygon_filter):
        assert hasattr(polygon_filter, "polygon")
        assert hasattr(polygon_filter, "plane")
        assert np.all(polygon_filter.polygon == box(-0.31, -0.31, 1.3, 1.3))
        assert np.all(polygon_filter.plane == 'xy')

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            PolygonFilter(box(0, 0, 1, 1), 'zzz')

        with pytest.raises(ValueError):
            PolygonFilter('asdsa', 'xy')


    @pytest.mark.parametrize('kwargs', ({}, {'optimized_shift': None}))
    def test_mask(self, polygon_filter, simple_array, kwargs):
        pcd = PointCloudData(simple_array, **kwargs)
        mask = polygon_filter.mask(pcd)

        assert mask.shape == (pcd.shape[0],)
        assert mask.dtype == np.bool_
        assert isinstance(mask, np.ndarray)
        assert np.all(mask == np.array([False, False, False, True, False]))

        # Only yz plane will be different as x is the larget value to be filtered
        polygon_filter.plane = 'xz'
        mask2 = polygon_filter.mask(pcd)
        assert np.allclose(mask2, mask)

        polygon_filter.plane = 'yz'
        mask3 = polygon_filter.mask(pcd)
        assert not np.allclose(mask3, mask)