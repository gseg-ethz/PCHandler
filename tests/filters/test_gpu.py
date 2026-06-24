import numpy as np
import pytest
from shapely.geometry import Polygon

from pchandler import PointCloudData
from pchandler._optional import is_gpu_available
from pchandler.filters import PolygonFilterGPU, SphericalPolygonFilterGPU
from pchandler.geometry.coordinates import rhv2xyz

# D-06: skip cleanly if the smoke probe says no GPU at runtime, even when cudf is importable.
pytestmark = pytest.mark.skipif(
    not is_gpu_available(),
    reason="GPU support not available (cudf+cuspatial+geopandas importable AND smoke kernel passes).",
)


@pytest.fixture(scope="function", autouse=True)
def sphere_pcd_():
    # D-10 — was unseeded uniform calls; now seeded default_rng per TEST-05 pattern (test_downsample.py:17-29)
    rng = np.random.default_rng(0)
    rhv = np.column_stack(
        (
            rng.uniform(low=5, high=10, size=100),
            rng.uniform(low=np.pi / 2, high=2 * np.pi / 3, size=100),
            rng.uniform(low=0, high=np.pi / 2, size=100),
        )
    )
    rhv = np.vstack(
        (
            rhv,
            np.array([[10, -np.pi / 2, np.pi / 2]]),
        )
    )
    pcd = PointCloudData(xyz=rhv2xyz(rhv))
    return pcd


@pytest.fixture(scope="function", autouse=True)
def sphere_polygon_():
    return Polygon([[0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi], [0.0, np.pi]])


@pytest.fixture(scope="function", autouse=True)
def xyz_pcd_():
    # D-10 — was unseeded uniform calls; now seeded default_rng(1) per TEST-05 pattern
    rng = np.random.default_rng(1)
    xyz = np.column_stack(
        (
            rng.uniform(low=0.1, high=9.9, size=100),
            rng.uniform(low=0.1, high=4.9, size=100),
            rng.uniform(low=0, high=100, size=100),
        )
    )
    xyz = np.vstack(
        (
            xyz,
            np.array([[-10, -10, 10]]),
        )
    )
    pcd = PointCloudData(xyz=xyz)
    return pcd


@pytest.fixture(scope="function", autouse=True)
def xy_polygon_():
    return Polygon([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])


class TestSphericalPolygonFilterGPU:
    def test_init(self, sphere_polygon_, sphere_pcd_):
        sphere_filter = SphericalPolygonFilterGPU(sphere_polygon_)
        mask = sphere_filter.mask(sphere_pcd_)
        assert all(mask[:-1])
        assert not mask[-1]


class TestPolygonFilterGPU:
    def test_init(self, xy_polygon_, xyz_pcd_):
        polygon_filter = PolygonFilterGPU(xy_polygon_, "xy")
        assert polygon_filter.plane == "xy"
        mask = polygon_filter.mask(xyz_pcd_)
        assert all(mask[:-1])
        assert not mask[-1]

    def test_invalid_init(self, xy_polygon_):
        with pytest.raises(ValueError):
            polygon_filter = PolygonFilterGPU(xy_polygon_, "abc")
