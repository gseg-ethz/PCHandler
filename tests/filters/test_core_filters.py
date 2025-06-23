import pytest

import numpy as np

from pchandler.v2.geometry.core import PointCloudData
from pchandler.v2.filters.core import *

@pytest.fixture(scope='function', autouse=True)
def pcd_all():
    xyz = np.random.rand(100,3)
    rgb = np.random.randint(0, 256, (100,3), dtype=np.uint8)
    normals = np.random.rand(100,3)
    intensity = np.random.randint(0, 1000, (100,), dtype=np.uint16)
    pcd = PointCloudData(xyz, rgb=rgb, normals=normals, intensity=intensity)
    return pcd


@pytest.fixture(scope='function', autouse=True)
def pcd_only_coords():
    return PointCloudData(np.random.rand(100,3))

class TestAbstractPointCloudFilter:
    def test_abstract_methods(self):
        for name in ('mask', 'reduce', 'sample'):
            assert hasattr(PointCloudFilter, name)


class TestGenericFieldFilter:
    def test_initialisation(self):
        field_filter = GenericFieldFilter('dummy', lambda x: x > 5)

        assert field_filter.field == 'dummy'
        assert isinstance(field_filter.filter_func, Callable)


    @pytest.mark.parametrize('name,attr', (('spherical_coordinates', 'spher'),
                                           ('intensity', 'intensity'),
                                           ('cartesian_coordinates', 'xyz'),
                                           ('xyz', 'xyz'),
                                           ('rgb', 'rgb')))
    def test_mask_method(self, name, attr, pcd_all, pcd_only_coords):
        field_filter = GenericFieldFilter(name, lambda x: np.ones_like(x, dtype=np.bool_))
        mask = field_filter.mask(pcd_all)
        assert mask.shape == getattr(pcd_all, attr).shape
        assert mask.dtype == np.bool_


    def test_invalid_mask(self, pcd_only_coords):
        field_filter = GenericFieldFilter('rgb', lambda x: np.ones_like(x, dtype=np.bool_))
        with pytest.raises(ValueError):
            field_filter.mask(pcd_only_coords)


