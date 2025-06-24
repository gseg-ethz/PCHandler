import weakref

import numpy as np
import pytest

from pchandler.v2.geometry.coordinates import CartesianCoordinates
from pchandler.v2.geometry import PointCloudData
from pchandler.v2.geometry.optimal_shift import OptimizedShiftManager, OptimizedShift

# FIXME
OSM_Manager = None


def random_coordinates(scale: float, offset: float) -> np.ndarray:
    xyz_base = np.random.randn(100, 3).astype(np.float32)
    return (xyz_base * np.float32(scale) + np.float32(offset)).astype(np.float64)

@pytest.fixture(scope="function", autouse=True)
def scale_large() -> float:
    return float(2**33)


@pytest.fixture(scope="function", autouse=True)
def scale_small() -> float:
    return float(10**3)


@pytest.fixture(scope="function", autouse=True)
def offset_large() -> float:
    return float(2**49)


@pytest.fixture(scope="function", autouse=True)
def offset_small() -> float:
    return float(10**3)

@pytest.fixture(scope="function", autouse=True)
def coords_no_shift(scale_small: float):
    xyz = random_coordinates(scale_small, 0)
    return xyz

@pytest.fixture(scope="function", autouse=True)
def coords_shift(scale_small: float, offset_large: float) -> np.ndarray:
    return random_coordinates(scale_small, offset_large)

@pytest.fixture(scope="function", autouse=True)
def coords_unshiftable(scale_large, offset_small) -> np.ndarray:
    return random_coordinates(scale_large, offset_small)

@pytest.fixture(scope="function", autouse=True)
def coords_unshiftable2(scale_large, offset_large) -> np.ndarray:
    return random_coordinates(scale_large, offset_large)

@pytest.fixture(autouse=True)
def clear_instantiated_osm():
    yield
    # Teardown logic
    OptimizedShiftManager._instances = {}
    assert len(OptimizedShiftManager._instances) == 0


@pytest.fixture(scope="function")
def osm() -> OptimizedShiftManager:
    return OptimizedShiftManager()

class TestOptimizedShift:
    def test_class_attributes(self):
        raise NotImplementedError

    def test_initialisation(self):
        raise NotImplementedError

    def test_optimal_shift_property(self):
        raise NotImplementedError

    def test_register_method(self):
        raise NotImplementedError

    def test_can_add_without_change_method(self):
        raise NotImplementedError

    def test_add_member_method(self):
        raise NotImplementedError

    def test_expand_and_add_method(self):
        raise NotImplementedError

    def test_compute_shift_method(self):
        raise NotImplementedError

    def test_apply_shift_delta_method(self):
        raise NotImplementedError

    def test_restart_or_fail_method(self):
        raise NotImplementedError

    class TestMinMaxPoints:
        def test_class_attributes(self):
            min_max_pts = OptimizedShift.MinMaxPoints(np.zeros(3), np.ones(3))
            assert hasattr(min_max_pts, 'minimum')
            assert hasattr(min_max_pts, 'maximum')
            assert np.all(min_max_pts.minimum, 0)
            assert np.all(min_max_pts.maximum, 1)

        def test_from_points_method(self):
            raise NotImplementedError

        def test_from_minmax_points_method(self):
            raise NotImplementedError

        def test_central_point_property(self):
            raise NotImplementedError

        def test_extents_property(self):
            raise NotImplementedError



class TestOptimizedShiftManager:
    def test_custom_error(self):
        with pytest.raises(OptimizedShiftManager.ShiftNotFeasibleError):
            raise OptimizedShiftManager.ShiftNotFeasibleError("Shift not possible")

    def test_class_variables(self, osm):
        for name in ('_optimized_shifts', '_maximum_decimal_places'):
            assert hasattr(osm, name)

    def test_has_class_methods(self):
        for name in (
                'register', 'new_shift', 'all_shifts', 'maximum_decimal_places', 'is_shift_needed', 'is_shift_possible'
        ):
            assert hasattr(OptimizedShiftManager, name)

    def test_empty_initialisation(self, osm):
        assert isinstance(osm, OptimizedShiftManager)
        assert len(osm._optimized_shifts) == 0
        assert osm._maximum_decimal_places == 4

        osm2: OptimizedShiftManager = OptimizedShiftManager(5)
        assert osm2.maximum_decimal_places != 5

    def test_maximal_decimal_setter(self):
        osm_ = OptimizedShiftManager(maximum_decimal_places=10)
        assert osm_._maximum_decimal_places == 10

    def test_singleton(self, osm):
        osm2 = OptimizedShiftManager(5)
        osm3 = OptimizedShiftManager(6)
        assert id(osm) == id(osm2)
        assert id(osm) == id(osm3)

        assert osm.maximum_decimal_places == 4

    def test_new_shift_and_register_methods(self, osm):
        shift1 = osm.new_shift()
        assert isinstance(shift1, OptimizedShift)
        assert len(osm) == 1
        shift2 = osm.new_shift()
        assert id(shift1) != id(shift2)
        assert len(osm) == 2
        assert id(shift1) in [id(i) for i in osm._optimized_shifts]
        assert id(shift2) in [id(i) for i in osm._optimized_shifts]
        assert id(shift1) != id(shift2)

    def test_all_shifts_method(self, osm):
        added_shifts = []
        for _ in range(5):
            shift = osm.new_shift()
            added_shifts.append(shift)

        assert len(osm) == 5
        assert len(osm._optimized_shifts) == 5
        for shift in osm.all_shifts:
            assert shift in added_shifts
            added_shifts.pop(added_shifts.index(shift))

        assert len(added_shifts) == 0

    def test_is_shift_needed_method(self, osm, coords_shift, coords_no_shift, coords_unshiftable, coords_unshiftable2):
        assert osm.is_shift_needed(coords_shift)
        assert osm.is_shift_needed(coords_unshiftable)
        assert osm.is_shift_needed(coords_unshiftable2)
        assert not osm.is_shift_needed(coords_no_shift)

    def test_is_shift_possible(self, osm, coords_shift, coords_no_shift, coords_unshiftable, coords_unshiftable2):
        assert osm.is_shift_possible(coords_shift)
        assert osm.is_shift_possible(coords_no_shift)
        assert not osm.is_shift_possible(coords_unshiftable)
        assert not osm.is_shift_possible(coords_unshiftable2)

    @pytest.mark.parametrize('array_type', (CartesianCoordinates, PointCloudData))
    def test_checks_on_array_like(self, osm, coords_shift: CartesianCoordinates, array_type):
        obj = array_type(xyz=coords_shift)
        assert osm.is_shift_possible(obj)
        # Case where optimized shift is on the PointCloudData object
        if isinstance(obj, PointCloudData):
            assert not osm.is_shift_needed(obj)
            assert obj.optimized_shift is not None
        else:
            assert osm.is_shift_needed(obj)


class TestPracticalUsage:
    def test_no_shift_required(self):
        raise NotImplementedError

    def test_single_pcd_shift_required(self):
        raise NotImplementedError

    def test_multiple_pcd_single_shift(self):
        raise NotImplementedError

    def test_multiple_pcd_multiple_shifts(self):
        raise NotImplementedError

    def test_no_shift_possible(self):
        raise NotImplementedError
#
# def check_global_shift_applied(pcd: PointCloudData):
#     assert pcd.global_coordinate_shift is not None
#     assert isinstance(pcd.global_coordinate_shift, np.ndarray)
#     assert np.any(pcd.global_coordinate_shift != 0)
#
#
# def test_global_shifted(coordinates_unshiftable, rgb_, normals_, intensities_, offset_large):
#     xyz = coordinates_unshiftable
#     pcd = PointCloudData(xyz=xyz, rgb=rgb_, normals=normals_, intensity=intensities_)
#
#     check_global_shift_need(xyz, True)
#     check_global_shift_applied(pcd)
#     check_origin_exists(pcd)
#     assert isinstance(pcd, PointCloudData)
#
#     # Data should not be close after shift
#     assert not np.all(np.isclose(pcd.xyz, xyz))
#
#     # Global shift should be close to the offset
#     assert np.all(np.isclose(pcd.spherical_coordinates_origin, np.array([-offset_large, -offset_large, -offset_large])))
#
#     # 'Scalar fields' should be identical
#     assert np.all(rgb_ == pcd.rgb)
#     assert np.all(normals_ == pcd.normals)
#     assert np.all(intensities_ == pcd.scalar_fields["intensity"])
#
#     assert pcd.nbPoints == 100
#     assert pcd.xyz.dtype == np.float32
#
#
# def test_non_shifted_cloud(small_coordinates, rgb_, normals_, intensities_, offset_small):
#     xyz = small_coordinates
#     pcd = PointCloudData(xyz, rgb=rgb_, normals=normals_, sfm={"intensity": intensities_})
#
#     check_global_shift_need(xyz, False)
#     check_global_shift_not_applied(pcd)
#     check_origin_exists(pcd)
#     assert np.all(pcd.spherical_coordinates_origin == np.zeros(3))
#     assert pcd.nbPoints == 100
#
#     # Scalar fields
#     assert np.all(rgb_ == pcd.rgb)
#     assert np.all(normals_ == pcd.normals)
#     assert np.all(intensities_ == pcd.scalar_fields["intensity"])
#
#     # Points should match now
#     assert np.all(np.isclose(pcd.xyz, xyz))
#     assert pcd.xyz.dtype == np.float32
#
#
# def test_register():
#     a = PointCloudData(xyz=np.random.rand(100, 3) * 100, optimised=True)
#     assert any(id(wref) == id(a) for wref in OSM_Manager.point_clouds)
#     OSM_Manager.reset()
#
#
# def test_weakref_list():
#     pcd_list = list()
#     pcd_list.append(PointCloudData(arr=np.random.rand(100, 3) * 100, optimised=True))
#     pcd_list.append(PointCloudData(arr=np.random.rand(100, 3) * 200, optimised=True))
#     pcd_list.append(PointCloudData(arr=np.random.rand(100, 3) * 400, optimised=True))
#
#     assert len(OSM_Manager.point_clouds) == 3
#
#     for i, pcd in enumerate(pcd_list):
#         assert any(id(wref) == id(pcd) for wref in OSM_Manager.point_clouds)  # ensure pcd is in the set
#         assert np.allclose(pcd.min(), pcd_list[i].min())
#         assert np.allclose(pcd.max(), pcd_list[i].max())
#
#     del_id = id(pcd_list[0])
#     del pcd_list[0]
#
#     assert not any(id(wref) == del_id for wref in OSM_Manager.point_clouds)
#     assert len(OSM_Manager.point_clouds) == 2
#     OSM_Manager.reset()
#
#
# def test_optimal_shift_max_size():
#     pcd_list = list()
#     pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 100, optimised=True))
#     pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 200, optimised=True))
#     pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 400, optimised=True))
#     pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 800, optimised=True))
#     pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 1000, optimised=True))
#     pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 2000, optimised=True))
#     pcd_list.append(PointCloudData(xyz=np.random.rand(100, 3) * 5000, optimised=True))
#
#     assert len(OSM_Manager.point_clouds) == 7
#
#     with pytest.raises(ValueError):
#         PointCloudData(xyz=np.random.rand(100, 3) * 20000, optimised=True)
#
#     OSM_Manager.reset()
#
#
# def test_optimal_shift_reset():
#     a = PointCloudData(xyz=np.random.rand(100, 3) * 100, optimised=True)
#     b = PointCloudData(xyz=np.random.rand(100, 3) * 200, optimised=True)
#     c = PointCloudData(xyz=np.random.rand(100, 3) * 400, optimised=True)
#     d = PointCloudData(xyz=np.random.rand(100, 3) * 800, optimised=True)
#     assert len(OSM_Manager.point_clouds) == 4
#     OSM_Manager.reset()
#
#     assert len(OSM_Manager.point_clouds) == 0
#     assert OSM_Manager.current_bbox is None
#     assert OSM_Manager.bounding_boxes is None
#     assert np.allclose(OSM_Manager.optimal_shift, np.zeros(3))
#     assert isinstance(OSM_Manager.point_clouds, weakref.WeakSet)
#
#
# def test_adapting_optimal_shift():
#     # Test that on each addition, the new optimal shift center is calculated.
#     # When it reaches max size, throws error
#
#     # Define offsets for the optimal shift
#     a_origin = np.array([100, -400, 100], dtype=np.float32)
#     b_origin = np.array([3400, -1200, -100], dtype=np.float32)
#     c_origin = np.array([-4000, -6500, -7000], dtype=np.float32)
#     d_origin = np.array([-6350, 5320, -8200], dtype=np.float32)
#     e_origin = np.array([20000]).astype(np.float32)
#
#     # Update the coordinates
#     a_ = (np.random.rand(10, 3) * 400).astype(np.float32) + a_origin
#     b_ = (np.random.rand(10, 3) * 400).astype(np.float32) + b_origin
#     c_ = (np.random.rand(10, 3) * 400).astype(np.float32) + c_origin
#     d_ = (np.random.rand(10, 3) * 400).astype(np.float32) + d_origin
#     e_ = (np.random.rand(10, 3) * 400).astype(np.float32) - e_origin
#
#     # Initialise the first, opt shift is too close to origin (< 500) so shouldn't shift
#     a = PointCloudData(xyz=a_.copy(), socs_origin=a_origin.copy(), optimised=True)
#     assert OSM_Manager.optimal_shift.shape == (3,)
#     assert np.allclose(OSM_Manager.optimal_shift, np.zeros(3))
#     last_shift = OSM_Manager.optimal_shift.copy()
#     last_a_socs = a.socs_origin.copy()
#
#     # Initialise 2nd, optimal shift should be computed and applied
#     b = PointCloudData(xyz=b_.copy(), socs_origin=b_origin.copy(), optimised=True)
#     assert not np.allclose(OSM_Manager.optimal_shift, last_shift)
#     assert not np.allclose(last_a_socs, a.socs_origin)
#     last_a_socs = a.socs_origin.copy()
#     last_shift = OSM_Manager.optimal_shift.copy()
#     # TODO implement some basic test
#
#     # Initialise 3rd, a new optimal shift should be computed
#     c = PointCloudData(xyz=c_.copy(), socs_origin=c_origin.copy(), optimised=True)
#     assert not np.allclose(OSM_Manager.optimal_shift, last_shift)
#     assert not np.allclose(last_a_socs, a.socs_origin)
#     last_a_socs = a.socs_origin.copy()
#     last_shift = OSM_Manager.optimal_shift.copy()
#     # TODO implement more tests to a,b and c
#
#     # Initialise 4th
#     d = PointCloudData(xyz=d_.copy(), socs_origin=d_origin.copy(), optimised=True)
#     assert not np.allclose(OSM_Manager.optimal_shift, last_shift)
#     assert not np.allclose(last_a_socs, a.socs_origin)
#
#     with pytest.raises(ValueError):
#         PointCloudData(xyz=e_.copy().astype(np.float32), optimised=True)
#
#     # check that all the socs are different
#     assert not np.allclose(a_origin, a.socs_origin)
#     assert not np.allclose(b_origin, b.socs_origin)
#     assert not np.allclose(c_origin, c.socs_origin)
#     assert not np.allclose(d_origin, d.socs_origin)
#     assert np.allclose(a_, a + OSM_Manager.optimal_shift, atol=0.0005)
#     assert np.allclose(b_, b + OSM_Manager.optimal_shift, atol=0.0005)
#     assert np.allclose(c_, c + OSM_Manager.optimal_shift, atol=0.0005)
#     assert np.allclose(d_, d + OSM_Manager.optimal_shift, atol=0.0005)
#     assert np.allclose(a_origin, a.socs_origin + OSM_Manager.optimal_shift)
#     assert np.allclose(b_origin, b.socs_origin + OSM_Manager.optimal_shift)
#     assert np.allclose(c_origin, c.socs_origin + OSM_Manager.optimal_shift)
#     assert np.allclose(d_origin, d.socs_origin + OSM_Manager.optimal_shift)
#
#     OSM_Manager.reset()
