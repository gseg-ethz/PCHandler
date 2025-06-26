import weakref

import numpy as np
import pytest

from pchandler.v2.geometry.coordinates import CartesianCoordinates
from pchandler.v2.geometry import PointCloudData
from pchandler.v2.geometry.optimal_shift import OptimizedShiftManager, OptimizedShift
from pchandler.v2.geometry.util import MinMaxPoints

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
    return float(10**4) * 2.3

@pytest.fixture(scope="function", autouse=True)
def coords_no_shift(scale_small: float):
    xyz = random_coordinates(scale_small, 0)
    return xyz

@pytest.fixture(scope="function", autouse=True)
def coords_shift(scale_small: float, offset_small: float) -> np.ndarray:
    return random_coordinates(scale_small, offset_small)

@pytest.fixture(scope="function", autouse=True)
def coords_shift2(scale_small: float, offset_large: float) -> np.ndarray:
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
def pcd(coords_shift) -> PointCloudData:
    return PointCloudData(coords_shift, optimized_shift=None)


@pytest.fixture(scope="function")
def osm() -> OptimizedShiftManager:
    return OptimizedShiftManager()

@pytest.fixture(scope="function")
def opt_shift(osm) -> OptimizedShift:
    return osm.new_shift()

class TestOptimizedShift:
    @pytest.mark.parametrize('attr', ('_optimal_shift', '_member_pcds', '_member_pcds_unshifted_bbox'))
    def test_class_attributes(self, opt_shift, attr):
        assert hasattr(opt_shift, attr)
        assert getattr(opt_shift, attr) is not None

    def test_initialisation(self, opt_shift):
        assert len(opt_shift._member_pcds) == 0
        assert len(opt_shift._member_pcds_unshifted_bbox) == 0
        assert np.all(opt_shift.optimal_shift == [0, 0, 0])
        assert len(OptimizedShiftManager()) == 1
        assert opt_shift in OptimizedShiftManager().all_shifts

    def test_optimal_shift_property(self, opt_shift):
        assert hasattr(OptimizedShift, 'optimal_shift')
        assert np.all(opt_shift.optimal_shift == 0)

    def test_register_method(self, pcd, opt_shift):
        xyz_original = pcd.xyz.copy()
        pcd2 = pcd.copy() + 2.0
        pcd3 = pcd2.copy() + 3.0
        assert pcd2 is not pcd
        assert pcd2 is not pcd3
        assert pcd2.xyz is not pcd.xyz
        assert pcd2.xyz is not pcd3.xyz

        # Iterate through pcd's and registering each one
        for i, item in enumerate((pcd, pcd2, pcd3)):
            # Main logic that once registered, the pcd item should be optimized
            assert item.optimized_shift is None
            opt_shift.register(item, item.xyz)
            # FIXME: THIS IS THE LINE OF FAILURE
            assert item.optimized_shift is not None
            assert len(opt_shift) == i+1
            assert item in set(opt_shift._member_pcds)

            assert xyz_original is not item.xyz
            if i > 0:
                assert not np.all(np.isclose(xyz_original, item.xyz))

    def test_can_add_without_change_method(self, pcd, opt_shift):
        pcd2 = pcd + 0.5
        opt_shift.register(pcd, pcd.xyz)
        assert opt_shift._can_add_without_change(pcd2.xyz)
        pcd3 = pcd + 50000
        assert not opt_shift._can_add_without_change(pcd3.xyz)

    def test_add_member_method(self, pcd, opt_shift):
        pcd2 = pcd.copy() + 2.0
        pcd3 = pcd2.copy() + 3.0

        for item in (pcd, pcd2, pcd3):
            opt_shift._add_member(item, item.xyz)

        assert len(opt_shift) == 3
        assert len(opt_shift._member_pcds) == 3
        assert len(opt_shift._member_pcds_unshifted_bbox) == 3

    def test_expand_and_add_method(self, opt_shift, pcd):
        original_xyz = pcd.xyz.copy()
        pcd2 = pcd - 60000

        opt_shift.register(pcd, pcd.xyz)
        assert not opt_shift._can_add_without_change(pcd2.xyz)
        shift_1 = opt_shift.optimal_shift.copy()
        shifted_xyz_1 = pcd.xyz.copy()

        opt_shift.register(pcd2, pcd2.xyz)
        shift_2 = opt_shift.optimal_shift.copy()
        shifted_xyz_2 = pcd.xyz.copy()

        assert np.all(shift_1 != shift_2)
        assert np.all(original_xyz != shifted_xyz_1)
        assert np.all(original_xyz != shifted_xyz_2)

    def test_compute_new_shift_method(self, opt_shift, pcd):
        pcd2 = pcd - 3000
        opt_shift.register(pcd, pcd.xyz)

        assert np.any(opt_shift.optimal_shift != 0)

        new_shift = opt_shift._compute_new_shift(pcd2.xyz)
        assert np.all((new_shift - 3000/2) == opt_shift.optimal_shift)

        with pytest.raises(OptimizedShiftManager.ShiftNotFeasibleError):
            opt_shift._compute_new_shift(pcd2.xyz - 1000000)

    def test_apply_shift_delta_method(self, opt_shift, pcd):
        original_xyz = pcd.xyz.copy()
        pcd2 = pcd + np.random.rand(100,3)*40
        original_xyz2 = pcd2.xyz.copy()

        opt_shift.register(pcd, pcd.xyz)
        opt_shift.register(pcd2, pcd2.xyz)

        registered_xyz = pcd.xyz.copy()
        registered_xyz2 = pcd2.xyz.copy()

        assert not np.allclose(registered_xyz, original_xyz)
        assert not np.allclose(registered_xyz2, original_xyz2)

        delta = np.array([-500, 1000, 3000])

        opt_shift._apply_shift_delta(delta)

        assert not np.allclose(registered_xyz, pcd.xyz)
        assert not np.allclose(registered_xyz2, pcd2.xyz)

        assert np.allclose(pcd.xyz + delta, registered_xyz)
        assert np.allclose(pcd2.xyz + delta, registered_xyz2)


    def test_restart_or_fail_method(self):
        raise NotImplementedError

    class TestMinMaxPoints:
        def test_class_attributes(self):
            min_max_pts = MinMaxPoints(np.zeros(3), np.ones(3))

            assert hasattr(min_max_pts, 'minimum')
            assert hasattr(min_max_pts, 'maximum')
            assert np.all(min_max_pts.minimum == 0)
            assert np.all(min_max_pts.maximum == 1)

            assert isinstance(min_max_pts.minimum, np.ndarray)
            assert isinstance(min_max_pts.maximum, np.ndarray)

            assert len(min_max_pts.minimum) == 3
            assert len(min_max_pts.maximum) == 3

        def test_initialise(self):
            low = np.ones(3)
            high = np.full_like(low, 15)

            min_max = MinMaxPoints(low, high)
            assert isinstance(min_max, MinMaxPoints)
            assert np.allclose(min_max.minimum, low)
            assert np.allclose(min_max.maximum, high)

        def test_from_points_method(self):
            array = np.random.rand(100,3)
            min_max = MinMaxPoints.from_minmax_points(array)

            assert isinstance(min_max, MinMaxPoints)
            assert np.all(array.min(axis=0) == min_max.minimum)
            assert np.all(array.max(axis=0) == min_max.maximum)

        def test_from_minmax_points_method(self):
            pts_minmax = []
            for i in range(7):
                pts_minmax.append(
                    MinMaxPoints(minimum=np.ones(3) * -i, maximum=np.ones(3) * i)
                )

            minmax = MinMaxPoints.from_minmax_points(pts_minmax)
            assert np.all(minmax.minimum == -6)
            assert np.all(minmax.maximum == 6)

        def test_central_point_property(self):

            low = np.array([0, 1, 2])
            high = np.array([3, 14, 20])
            min_max = MinMaxPoints(low, high)

            center = min_max.central_point

            assert len(center) == 3
            assert isinstance(center, np.ndarray)
            assert np.all(center == [1.5, 7.5, 11])

        def test_extents_property(self):
            low = np.array([0, 1, 2])
            high = np.array([3, 14, 20])
            min_max = MinMaxPoints(low, high)

            extents = min_max.extents

            assert np.all(extents == [3, 13, 18])


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

    def test_is_shift_needed_method(
            self, osm, coords_shift, coords_shift2, coords_no_shift, coords_unshiftable, coords_unshiftable2):
        assert osm.is_shift_needed(coords_shift)
        assert osm.is_shift_needed(coords_shift2)
        assert osm.is_shift_needed(coords_unshiftable)
        assert osm.is_shift_needed(coords_unshiftable2)
        assert not osm.is_shift_needed(coords_no_shift)

    def test_is_shift_possible(
            self, osm, coords_shift, coords_shift2, coords_no_shift, coords_unshiftable, coords_unshiftable2):
        assert osm.is_shift_possible(coords_shift)
        assert osm.is_shift_possible(coords_shift2)
        assert osm.is_shift_possible(coords_no_shift)
        assert not osm.is_shift_possible(coords_unshiftable)
        assert not osm.is_shift_possible(coords_unshiftable2)

    @pytest.mark.parametrize('array_type', (CartesianCoordinates, PointCloudData))
    def test_checks_on_array_like(self, osm, coords_shift: CartesianCoordinates, array_type):
        obj = array_type(xyz=coords_shift)
        assert osm.is_shift_possible(obj)
        # Case where optimized shift is on the PointCloudData object
        if array_type is PointCloudData:
            assert not osm.is_shift_needed(obj)
        else:
            assert osm.is_shift_needed(obj)


def test_pcd_optimized_shift_kwargs():
    osm = OptimizedShiftManager()
    xyz = np.random.rand(100, 3)

    pcd = PointCloudData(xyz, optimized_shift=OptimizedShift(np.array([-5000, 0, 0])))
    shift_1 = pcd.optimized_shift.optimal_shift.copy()

    xyz2 = np.random.rand(100, 3) + np.array([6000,-1000, 2000])
    pcd2 = PointCloudData(xyz2, optimized_shift=pcd.optimized_shift)
    shift_2 = pcd.optimized_shift.optimal_shift.copy()

    xyz2 = np.random.rand(100, 3) + [100000, 2000, 400]
    pcd3 = PointCloudData(xyz2, optimized_shift=pcd.optimized_shift)
    shift_3 = pcd.optimized_shift.optimal_shift.copy()

    assert pcd.optimized_shift is pcd2.optimized_shift is pcd3.optimized_shift

    assert np.all(shift_1 != shift_2)
    assert np.all(shift_3 != shift_2)
    assert np.all(shift_3 != shift_1)

