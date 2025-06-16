import weakref

import numpy as np
import pytest

from pchandler.v2.geometry import OSM_Manager, PointCloudData


def random_coordinates(scale: float, offset: float) -> np.ndarray:
    xyz_base = np.random.randn(100, 3)
    return xyz_base * scale + offset


@pytest.fixture(scope="session", autouse=True)
def small_coordinates(scale_small: float, offset_small: float) -> np.ndarray:
    # this is to ensure the conversion resolves this but no shift applied
    xyz_base = np.random.randn(100, 3).astype(np.float64)
    return xyz_base * scale_small + offset_small


@pytest.fixture(scope="session", autouse=True)
def large_coordinates(scale_large, offset_large) -> np.ndarray:
    xyz = random_coordinates(scale_large, offset_large)
    assert np.min_scalar_type(xyz) == np.float64
    return xyz


@pytest.fixture(scope="session", autouse=True)
def scale_large() -> float:
    return float(2**33)


@pytest.fixture(scope="session", autouse=True)
def scale_small() -> float:
    return float(10**3)


@pytest.fixture(scope="session", autouse=True)
def offset_large() -> float:
    return float(2**49)


@pytest.fixture(scope="session", autouse=True)
def offset_small() -> float:
    return float(10**3)


def check_origin_exists(pcd: PointCloudData):
    assert isinstance(pcd.spherical_coordinates_origin, np.ndarray) == True


def check_global_shift_need(xyz, expected: bool):
    assert PointCloudData._needs_global_shift(xyz) == expected


def check_global_shift_not_applied(pcd: PointCloudData):
    assert pcd.global_coordinate_shift is None or (
        isinstance(pcd.global_coordinate_shift, np.ndarray) and np.all(pcd.global_coordinate_shift == 0)
    )


def check_global_shift_applied(pcd: PointCloudData):
    assert pcd.global_coordinate_shift is not None
    assert isinstance(pcd.global_coordinate_shift, np.ndarray)
    assert np.any(pcd.global_coordinate_shift != 0)


def test_global_shifted(large_coordinates, rgb_, normals_, intensities_, offset_large):
    xyz = large_coordinates
    pcd = PointCloudData(xyz=xyz, rgb=rgb_, normals=normals_, intensity=intensities_)

    check_global_shift_need(xyz, True)
    check_global_shift_applied(pcd)
    check_origin_exists(pcd)
    assert isinstance(pcd, PointCloudData)

    # Data should not be close after shift
    assert not np.all(np.isclose(pcd.xyz, xyz))

    # Global shift should be close to the offset
    assert np.all(np.isclose(pcd.spherical_coordinates_origin, np.array([-offset_large, -offset_large, -offset_large])))

    # 'Scalar fields' should be identical
    assert np.all(rgb_ == pcd.rgb)
    assert np.all(normals_ == pcd.normals)
    assert np.all(intensities_ == pcd.scalar_fields["intensity"])

    assert pcd.nbPoints == 100
    assert pcd.xyz.dtype == np.float32


def test_non_shifted_cloud(small_coordinates, rgb_, normals_, intensities_, offset_small):
    xyz = small_coordinates
    pcd = PointCloudData(xyz, rgb=rgb_, normals=normals_, sfm={"intensity": intensities_})

    check_global_shift_need(xyz, False)
    check_global_shift_not_applied(pcd)
    check_origin_exists(pcd)
    assert np.all(pcd.spherical_coordinates_origin == np.zeros(3))
    assert pcd.nbPoints == 100

    # Scalar fields
    assert np.all(rgb_ == pcd.rgb)
    assert np.all(normals_ == pcd.normals)
    assert np.all(intensities_ == pcd.scalar_fields["intensity"])

    # Points should match now
    assert np.all(np.isclose(pcd.xyz, xyz))
    assert pcd.xyz.dtype == np.float32


def test_register():
    a = PointCloudData(xyz=np.random.rand(100, 3) * 100, optimised=True)
    assert any(id(wref) == id(a) for wref in OSM_Manager.point_clouds)
    OSM_Manager.reset()


def test_weakref_list():
    pcd_list = list()
    pcd_list.append(PointCloudData(arr=np.random.rand(100, 3) * 100, optimised=True))
    pcd_list.append(PointCloudData(arr=np.random.rand(100, 3) * 200, optimised=True))
    pcd_list.append(PointCloudData(arr=np.random.rand(100, 3) * 400, optimised=True))

    assert len(OSM_Manager.point_clouds) == 3

    for i, pcd in enumerate(pcd_list):
        assert any(id(wref) == id(pcd) for wref in OSM_Manager.point_clouds)  # ensure pcd is in the set
        assert np.allclose(pcd.min(), pcd_list[i].min())
        assert np.allclose(pcd.max(), pcd_list[i].max())

    del_id = id(pcd_list[0])
    del pcd_list[0]

    assert not any(id(wref) == del_id for wref in OSM_Manager.point_clouds)
    assert len(OSM_Manager.point_clouds) == 2
    OSM_Manager.reset()


def test_optimal_shift_max_size():
    pcd_list = list()
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 100, optimised=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 200, optimised=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 400, optimised=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 800, optimised=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 1000, optimised=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 2000, optimised=True))
    pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 5000, optimised=True))

    assert len(OSM_Manager.point_clouds) == 7

    with pytest.raises(ValueError):
        PointCloudData(xyz=np.random.rand(100, 3) * 20000, optimised=True)

    OSM_Manager.reset()


def test_optimal_shift_reset():
    a = PointCloudData(xyz=np.random.rand(100, 3) * 100, optimised=True)
    b = PointCloudData(xyz=np.random.rand(100, 3) * 200, optimised=True)
    c = PointCloudData(xyz=np.random.rand(100, 3) * 400, optimised=True)
    d = PointCloudData(xyz=np.random.rand(100, 3) * 800, optimised=True)
    assert len(OSM_Manager.point_clouds) == 4
    OSM_Manager.reset()

    assert len(OSM_Manager.point_clouds) == 0
    assert OSM_Manager.current_bbox is None
    assert OSM_Manager.bounding_boxes is None
    assert np.allclose(OSM_Manager.optimal_shift, np.zeros(3))
    assert isinstance(OSM_Manager.point_clouds, weakref.WeakSet)


def test_adapting_optimal_shift():
    # Test that on each addition, the new optimal shift center is calculated.
    # When it reaches max size, throws error

    # Define offsets for the optimal shift
    a_origin = np.array([100, -400, 100], dtype=np.float32)
    b_origin = np.array([3400, -1200, -100], dtype=np.float32)
    c_origin = np.array([-4000, -6500, -7000], dtype=np.float32)
    d_origin = np.array([-6350, 5320, -8200], dtype=np.float32)
    e_origin = np.array([20000]).astype(np.float32)

    # Update the coordinates
    a_ = (np.random.rand(10, 3) * 400).astype(np.float32) + a_origin
    b_ = (np.random.rand(10, 3) * 400).astype(np.float32) + b_origin
    c_ = (np.random.rand(10, 3) * 400).astype(np.float32) + c_origin
    d_ = (np.random.rand(10, 3) * 400).astype(np.float32) + d_origin
    e_ = (np.random.rand(10, 3) * 400).astype(np.float32) - e_origin

    # Initialise the first, opt shift is too close to origin (< 500) so shouldn't shift
    a = PointCloudData(xyz=a_.copy(), socs_origin=a_origin.copy(), optimised=True)
    assert OSM_Manager.optimal_shift.shape == (3,)
    assert np.allclose(OSM_Manager.optimal_shift, np.zeros(3))
    last_shift = OSM_Manager.optimal_shift.copy()
    last_a_socs = a.socs_origin.copy()

    # Initialise 2nd, optimal shift should be computed and applied
    b = PointCloudData(xyz=b_.copy(), socs_origin=b_origin.copy(), optimised=True)
    assert not np.allclose(OSM_Manager.optimal_shift, last_shift)
    assert not np.allclose(last_a_socs, a.socs_origin)
    last_a_socs = a.socs_origin.copy()
    last_shift = OSM_Manager.optimal_shift.copy()
    # TODO implement some basic test

    # Initialise 3rd, a new optimal shift should be computed
    c = PointCloudData(xyz=c_.copy(), socs_origin=c_origin.copy(), optimised=True)
    assert not np.allclose(OSM_Manager.optimal_shift, last_shift)
    assert not np.allclose(last_a_socs, a.socs_origin)
    last_a_socs = a.socs_origin.copy()
    last_shift = OSM_Manager.optimal_shift.copy()
    # TODO implement more tests to a,b and c

    # Initialise 4th
    d = PointCloudData(xyz=d_.copy(), socs_origin=d_origin.copy(), optimised=True)
    assert not np.allclose(OSM_Manager.optimal_shift, last_shift)
    assert not np.allclose(last_a_socs, a.socs_origin)

    with pytest.raises(ValueError):
        PointCloudData(xyz=e_.copy().astype(np.float32), optimised=True)

    # check that all the socs are different
    assert not np.allclose(a_origin, a.socs_origin)
    assert not np.allclose(b_origin, b.socs_origin)
    assert not np.allclose(c_origin, c.socs_origin)
    assert not np.allclose(d_origin, d.socs_origin)
    assert np.allclose(a_, a + OSM_Manager.optimal_shift, atol=0.0005)
    assert np.allclose(b_, b + OSM_Manager.optimal_shift, atol=0.0005)
    assert np.allclose(c_, c + OSM_Manager.optimal_shift, atol=0.0005)
    assert np.allclose(d_, d + OSM_Manager.optimal_shift, atol=0.0005)
    assert np.allclose(a_origin, a.socs_origin + OSM_Manager.optimal_shift)
    assert np.allclose(b_origin, b.socs_origin + OSM_Manager.optimal_shift)
    assert np.allclose(c_origin, c.socs_origin + OSM_Manager.optimal_shift)
    assert np.allclose(d_origin, d.socs_origin + OSM_Manager.optimal_shift)

    OSM_Manager.reset()
