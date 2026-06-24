from typing import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from pchandler import PointCloudData
from pchandler.filters import GenericFieldFilter, PointCloudFilter


@pytest.fixture(scope="function", autouse=True)
def pcd_all():
    rng = np.random.default_rng(0)
    xyz = rng.random((100, 3))
    rgb = rng.integers(0, 256, (100, 3), dtype=np.uint8)
    normals = rng.random((100, 3))
    intensity = rng.integers(0, 1000, (100,), dtype=np.uint16)
    pcd = PointCloudData(xyz, rgb=rgb, normals=normals, intensity=intensity)
    return pcd


@pytest.fixture(scope="function", autouse=True)
def pcd_only_coords():
    rng = np.random.default_rng(1)
    return PointCloudData(rng.random((100, 3)))


class TestAbstractPointCloudFilter:
    def test_abstract_methods(self, pcd_only_coords):
        for name in ("mask", "reduce", "sample"):
            assert hasattr(PointCloudFilter, name)

        with pytest.raises(TypeError):
            PointCloudFilter()

        # Not implemented Methods
        assert PointCloudFilter.mask(None, pcd_only_coords) == None  # type: ignore


class TestGenericFieldFilter:
    def test_initialisation(self):
        field_filter = GenericFieldFilter("dummy", lambda x: x > 5)

        assert field_filter.field_label == "dummy"
        assert isinstance(field_filter.filter_func, Callable)

    @pytest.mark.parametrize(
        "name,attr",
        (
            ("spherical_coordinates", "spher"),
            ("intensity", "intensity"),
            ("cartesian_coordinates", "xyz"),
            ("xyz", "xyz"),
            ("rgb", "rgb"),
        ),
    )
    def test_mask_method(self, name, attr, pcd_all, pcd_only_coords):
        field_filter = GenericFieldFilter(name, lambda x: np.ones_like(x, dtype=np.bool_))
        mask = field_filter.mask(pcd_all)
        assert mask.shape == getattr(pcd_all, attr).shape
        assert mask.dtype == np.bool_

    def test_all_methods(self, pcd_only_coords):
        field_filter = GenericFieldFilter("sf1", lambda x: x < 0.5)
        pcd_only_coords.scalar_fields.create_field("sf1", np.linspace(0, 1, len(pcd_only_coords)))
        pcd1 = pcd_only_coords.copy()
        pcd2 = pcd_only_coords.copy()

        mask = field_filter.mask(pcd2)
        sample = field_filter.sample(pcd2)
        extracted = field_filter.extract(pcd2)
        field_filter.reduce(pcd1)

        assert np.sum(mask) == 50
        assert len(sample) == 50
        assert len(extracted) == 50
        assert len(pcd1) == 50

        assert np.all(sample == extracted)
        assert np.all(pcd1 == field_filter.sample(pcd_only_coords))

    def test_invalid_mask(self, pcd_only_coords):
        field_filter = GenericFieldFilter("rgb", lambda x: np.ones_like(x, dtype=np.bool_))

        with pytest.raises(ValueError):
            field_filter.mask(pcd_only_coords)

        with pytest.raises(ValidationError):
            GenericFieldFilter([1, 2, 3], lambda x: x)

        with pytest.raises(ValidationError):
            GenericFieldFilter("rgb", 23)

        with pytest.raises(ValueError):
            field_filter.field_label = "non-existant"
            field_filter.mask(pcd_only_coords)
