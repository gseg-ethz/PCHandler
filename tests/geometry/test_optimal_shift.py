import copy
import gc
import pickle
import uuid
import weakref

import numpy as np
import pytest
from GSEGUtils.singleton import SingletonMeta

from pchandler import PointCloudData
from pchandler.geometry import OptimizedShift, OptimizedShiftManager
from pchandler.geometry.coordinates import CartesianCoordinates
from pchandler.geometry.util import MinMaxPoints


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
    # Teardown: remove this singleton from the shared SingletonMeta registry.
    # Assigning to OptimizedShiftManager._instances would create a shadow attribute on
    # the subclass instead of clearing the metaclass-level dict (CR-01, post-COUPLE-01).
    SingletonMeta._instances.pop(OptimizedShiftManager, None)
    assert OptimizedShiftManager not in SingletonMeta._instances


@pytest.fixture(scope="function")
def pcd(coords_shift) -> PointCloudData:
    return PointCloudData(coords_shift, numerical_optimization_shift=None)


@pytest.fixture(scope="function")
def osm() -> OptimizedShiftManager:
    return OptimizedShiftManager()


@pytest.fixture(scope="function")
def opt_shift() -> OptimizedShift:
    return OptimizedShift(np.array([1, 2, 3]))


class TestOptimizedShift:
    def test_initialisation(self):
        SingletonMeta._instances.pop(OptimizedShiftManager, None)
        opt_shift = OptimizedShift(np.array([1, 2, 3]))
        assert isinstance(opt_shift.uuid, uuid.UUID)
        assert isinstance(opt_shift._shift, np.ndarray)
        assert isinstance(opt_shift._member_coordinate_sets, weakref.WeakSet)

        assert len(opt_shift._member_coordinate_sets) == 0
        assert len(OptimizedShiftManager()) == 1
        assert np.all(opt_shift.value == [1, 2, 3])
        assert opt_shift in OptimizedShiftManager().all_shifts

    def test_new_value_assignment(self, opt_shift):
        xyz = np.random.rand(100, 3)
        pcd = PointCloudData(xyz, numerical_optimization_shift=opt_shift)
        initial_value = opt_shift.value
        initial_uuid = opt_shift.uuid
        assert np.allclose(pcd.xyz, xyz - initial_value)

        new_value = np.array([10, 20, 30])
        opt_shift.value = new_value
        assert opt_shift.uuid != initial_uuid
        assert pcd._shift_applied_by.uuid == opt_shift.uuid
        assert np.allclose(pcd.xyz, xyz - new_value)

    def test_invalid_value_assignment(self, opt_shift):
        xyz = np.random.rand(100, 3)
        pcd = PointCloudData(xyz, numerical_optimization_shift=opt_shift)
        initial_value = opt_shift.value
        initial_uuid = opt_shift.uuid
        assert np.allclose(pcd.xyz, xyz - initial_value)

        new_value = np.array([100_000, 200_000, 300_000])
        with pytest.raises(OptimizedShiftManager.ShiftNotFeasibleError):
            opt_shift.value = new_value

    def test___contains__(self, opt_shift):
        pcd = PointCloudData(np.random.rand(100, 3), numerical_optimization_shift=opt_shift)
        assert len(opt_shift._member_coordinate_sets) == 1
        assert pcd in opt_shift

        pcd2 = PointCloudData(np.random.rand(100, 3) + 1000, numerical_optimization_shift=opt_shift)
        assert len(opt_shift._member_coordinate_sets) == 2
        assert pcd2 in opt_shift

    def test___len__(self, opt_shift):
        pcds = []
        for i in range(17):
            xyz = PointCloudData(np.random.rand(100, 3), numerical_optimization_shift=opt_shift)
            pcds.append(xyz)

        assert len(opt_shift) == 17
        assert len(opt_shift._member_coordinate_sets) == 17

    def test___hash__(self, opt_shift):
        assert id(opt_shift) != hash(opt_shift)

    def test___eq__(self, opt_shift):
        xyz = PointCloudData(np.random.rand(100, 3), numerical_optimization_shift=opt_shift)
        func, state = xyz.__reduce__()
        obj = PointCloudData.model_construct(**state[0])
        assert id(obj) != id(xyz)
        assert opt_shift.__eq__(obj.numerical_optimization_shift)

    def test___reduce__(self, opt_shift):
        xyz = PointCloudData(np.random.rand(100, 3), numerical_optimization_shift=opt_shift)
        func, state = xyz.__reduce__()
        assert func == type(xyz)._reconstruct
        dumped = xyz.model_dump()
        for k, v in state[0].items():
            if isinstance(v, np.ndarray):
                assert np.all(v == dumped[k])
            else:
                assert v == dumped[k]

    def test_deepcopy(self, opt_shift):
        opt2 = copy.deepcopy(opt_shift)
        assert opt2 is not opt_shift
        assert hash(opt2) != hash(opt_shift)
        assert np.all(opt2.value == opt_shift.value)

    def test_uuid(self, opt_shift):
        assert isinstance(opt_shift.uuid, uuid.UUID)

        with pytest.raises(AttributeError):
            # noinspection PyPropertyAccess
            opt_shift.uuid = uuid.uuid4()

    def test___array_interface___(self, opt_shift):
        assert np.all(np.add(opt_shift, 1) == [2, 3, 4])

        result_copy = np.array(opt_shift)
        assert np.all(opt_shift.value == [1, 2, 3])
        assert np.all(result_copy == [1, 2, 3])

        assert id(result_copy) != id(opt_shift)
        assert id(result_copy) != id(opt_shift.value)
        assert not np.shares_memory(result_copy, opt_shift.value)

        # Same object
        result_ref = np.asarray(opt_shift)

        assert np.all(result_ref == [1, 2, 3])
        assert id(result_ref) != id(opt_shift)
        assert id(result_ref) != id(opt_shift.value)
        assert np.shares_memory(result_ref, opt_shift.value)

    def test_register_simple(self):
        pcd1 = PointCloudData(np.array([[0, 0, 0], [1, 1, 1]]))
        opt_shift = pcd1.numerical_optimization_shift

        # Case 0: Register shift
        opt_shift.register(pcd1)
        assert pcd1 in opt_shift
        assert len(opt_shift) == 1
        assert pcd1.numerical_optimization_shift is opt_shift

        # Case 1: Try to register self again, no change
        opt_shift.register(pcd1)
        assert pcd1 in opt_shift
        assert len(opt_shift) == 1
        assert pcd1.numerical_optimization_shift is opt_shift

        # Case 2: Add without change to optimal shift, length change
        value_before = pcd1.numerical_optimization_shift.value.copy()
        pcd2 = PointCloudData(pcd1 + 2.0)
        opt_shift.register(pcd2)
        assert pcd2 in opt_shift
        assert len(opt_shift) == 2
        assert np.all(pcd2.numerical_optimization_shift.value == value_before)

    def test_register_invalid_attempted(self):
        a = PointCloudData(
            np.array([[0, 0, 0], [1, 1, 1]]),
            numerical_optimization_shift=OptimizedShift(np.array([-9000, -9000, -9000])),
        )
        with pytest.raises(OptimizedShiftManager.ShiftNotFeasibleError):
            a.numerical_optimization_shift.register(PointCloudData(a - 1_045_000.456))

    def test_unregister(self):
        pcd1 = PointCloudData(np.array([[0, 0, 0], [1, 1, 1]]))
        pcd2 = PointCloudData(pcd1 + 2.0)
        pcd1.numerical_optimization_shift.register(pcd2)

        assert len(pcd1.numerical_optimization_shift) == 2

        pcd1.numerical_optimization_shift.unregister(pcd2)

        assert len(pcd1.numerical_optimization_shift) == 1

    def test_can_add_without_change(self, pcd, opt_shift):
        pcd2 = pcd + 0.5
        opt_shift.register(pcd)
        assert opt_shift._can_add_without_change(pcd2.xyz)
        pcd3 = pcd + 50000
        assert not opt_shift._can_add_without_change(pcd3.xyz)

    def test_add_member_method(self, pcd, opt_shift):
        pcd2 = pcd.copy() + 2.0
        pcd3 = pcd2.copy() + 3.0

        for item in (pcd, pcd2, pcd3):
            opt_shift._add_member(item)

        assert len(opt_shift) == 3
        assert len(opt_shift._member_coordinate_sets) == 3

    def test_compute_new_shift(self):
        a = PointCloudData(
            np.array([[0, 0, 0], [1, 1, 1]]),
            numerical_optimization_shift=OptimizedShift(np.array([-9000, -9000, -9000])),
        )
        b = PointCloudData(
            np.array([[0, 0, 0], [1, 1, 1]]) + 5000,
            numerical_optimization_shift=OptimizedShift(np.array([14_000, 14_000, 14_000])),
        )

        shift_a = a.numerical_optimization_shift._shift.copy()
        shift_b = b.numerical_optimization_shift._shift.copy()

        a.numerical_optimization_shift.register(b)
        new_shift = a.numerical_optimization_shift._compute_new_shift()

        assert not a.numerical_optimization_shift._can_add_without_change(b.arr)

        assert np.all(shift_a != new_shift)
        assert np.all(shift_b != new_shift)

    def test_compute_and_apply_shift_delta(self, opt_shift, coords_shift):
        a = PointCloudData(
            np.array([[0, 0, 0], [1, 1, 1]]),
            socs_origin=np.zeros(3),
            numerical_optimization_shift=OptimizedShift(np.array([-9000, -9000, -9000])),
        )
        b = PointCloudData(
            np.array([[0, 0, 0], [1, 1, 1]]) + 5000,
            numerical_optimization_shift=OptimizedShift(np.array([14_000, 14_000, 14_000])),
        )

        coords_a = a.arr.copy()
        shift_a = a.numerical_optimization_shift._shift.copy()

        new_shift = a.numerical_optimization_shift._compute_new_shift(b.unshifted_bbox)
        a.numerical_optimization_shift._compute_and_apply_shift_delta(new_shift)

        assert np.all(a.socs_origin == -new_shift)
        assert np.all(coords_a - new_shift == a.arr - a.numerical_optimization_shift)
        assert np.all(a.numerical_optimization_shift._shift == shift_a)  # Shift is not updated yet

    def test_reconstruct_no_change(self, osm):
        shift = OptimizedShift(np.array([1, 2, 3]))
        func, state = shift.__reduce__()
        assert len(osm) == 1

        # Case 1 - no change
        reconstructed = func(*state)
        assert reconstructed.uuid is shift.uuid
        assert id(reconstructed) == id(shift)
        assert len(osm) == 1

    def test_reconstruct_after_shift_deleted(self, osm):
        shift = OptimizedShift(np.array([1, 2, 3]))
        func, state = shift.__reduce__()

        old_id = id(shift)
        old_uuid = shift.uuid

        del shift
        gc.collect()

        reconstructed = func(*state)
        assert id(reconstructed) != old_id
        assert reconstructed.uuid == old_uuid

    def test_reconstruct_after_shift_moved(self, osm):
        shift = OptimizedShift(np.array([1, 2, 3]))
        func, state = shift.__reduce__()

        old_id = id(shift)
        old_uuid = shift.uuid

        old_value = shift.value
        shift.value = old_value + np.array([1000, 1000, 1000])

        reconstructed_shift = func(*state)

        assert id(reconstructed_shift) != old_id
        assert reconstructed_shift.uuid == old_uuid
        assert reconstructed_shift.uuid != shift.uuid
        assert np.allclose(reconstructed_shift.value, old_value)

    def test_reconstruct_pcd_after_shift_moved(self, opt_shift):
        xyz = np.random.rand(100, 3)
        pcd = PointCloudData(xyz, numerical_optimization_shift=opt_shift)
        initial_value = opt_shift.value
        initial_uuid = opt_shift.uuid

        pcd_pickled = pickle.dumps(pcd)
        opt_shift.value = np.array([0, 0, 0])

        pcd_unpickled = pickle.loads(pcd_pickled)
        assert np.allclose(xyz, pcd.xyz - opt_shift.value, atol=1e-5, rtol=1e-5)
        assert np.allclose(pcd_unpickled, xyz - initial_value, atol=1e-5, rtol=1e-5)
        assert pcd_unpickled._shift_applied_by.uuid == initial_uuid


class TestMinMaxPoints:
    def test_class_attributes(self):
        min_max_pts = MinMaxPoints(np.zeros(3), np.ones(3))

        assert hasattr(min_max_pts, "minimum")
        assert hasattr(min_max_pts, "maximum")
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
        array = np.random.rand(100, 3)
        min_max = MinMaxPoints.from_minmax_points(array)

        assert isinstance(min_max, MinMaxPoints)
        assert np.all(array.min(axis=0) == min_max.minimum)
        assert np.all(array.max(axis=0) == min_max.maximum)

    def test_from_minmax_points_method(self):
        pts_minmax = []
        for i in range(7):
            pts_minmax.append(MinMaxPoints(minimum=np.ones(3) * -i, maximum=np.ones(3) * i))

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
    def test_custom_errors(self):
        with pytest.raises(OptimizedShiftManager.ShiftNotFeasibleError):
            raise OptimizedShiftManager.ShiftNotFeasibleError("Shift not possible")

        with pytest.raises(OptimizedShiftManager.ShiftUUIDAlreadyTaken):
            raise OptimizedShiftManager.ShiftUUIDAlreadyTaken("UUID taken")

        with pytest.raises(OptimizedShiftManager.ShiftUUIDNotFound):
            raise OptimizedShiftManager.ShiftUUIDNotFound("UUID not found")

    def test_empty_initialisation(self):
        osm = OptimizedShiftManager()
        assert isinstance(osm, OptimizedShiftManager)
        assert len(osm.all_shifts) == 0
        assert osm._minimum_decimal_places == 3

    def test_minimal_decimal_places_kwarg(self):
        osm_ = OptimizedShiftManager(minimum_decimal_places=10)
        assert osm_._minimum_decimal_places == 10

    def test_singleton(self, osm):
        osm2 = OptimizedShiftManager(5)
        osm3 = OptimizedShiftManager(6)
        assert id(osm) == id(osm2)
        assert id(osm) == id(osm3)

        assert osm._minimum_decimal_places == 3

    def test_new_shift_and_register_methods(self, osm):
        shift1 = OptimizedShift()
        assert isinstance(shift1, OptimizedShift)
        assert len(osm) == 1

        shift2 = OptimizedShift()
        assert id(shift1) != id(shift2)

        assert len(osm) == 2

        assert id(shift1) in [id(i) for i in osm.all_shifts]
        assert id(shift2) in [id(i) for i in osm.all_shifts]

    def test_all_shifts_method(self, osm):
        added_shifts = []
        for _ in range(5):
            shift = OptimizedShift()
            added_shifts.append(shift)

        assert len(osm) == 5
        assert len(osm.all_shifts) == 5
        for shift in osm.all_shifts:
            assert shift in added_shifts
            added_shifts.pop(added_shifts.index(shift))

        assert len(added_shifts) == 0

    def test_is_shift_needed_method(
        self, osm, coords_shift, coords_shift2, coords_no_shift, coords_unshiftable, coords_unshiftable2
    ):
        assert osm.is_shift_needed(coords_shift)
        assert osm.is_shift_needed(coords_shift2)
        assert osm.is_shift_needed(coords_unshiftable)
        assert osm.is_shift_needed(coords_unshiftable2)
        assert not osm.is_shift_needed(coords_no_shift)

    def test_is_shift_possible(
        self, osm, coords_shift, coords_shift2, coords_no_shift, coords_unshiftable, coords_unshiftable2
    ):
        assert osm.is_shift_possible(coords_shift)
        assert osm.is_shift_possible(coords_shift2)
        assert osm.is_shift_possible(coords_no_shift)
        assert not osm.is_shift_possible(coords_unshiftable)
        assert not osm.is_shift_possible(coords_unshiftable2)

    def test_checks_on_pointclouddata_object(self, osm, coords_shift: CartesianCoordinates):
        obj = PointCloudData(coords_shift)
        assert osm.is_shift_possible(obj)
        assert not osm.is_shift_needed(obj)


def test_pcd_optimized_shift_general(osm):
    xyz = np.random.rand(100, 3)

    # add, no change
    pcd = PointCloudData(xyz, numerical_optimization_shift=OptimizedShift(np.array([-5000, 0, 0])))
    shift_1 = pcd.numerical_optimization_shift.value.copy()

    assert pcd.xyz.dtype == np.float32
    assert len(osm) == 1
    assert len(pcd.numerical_optimization_shift) == 1

    # Add and change
    xyz2 = np.random.rand(100, 3) + np.array([6000, -1000, 2000])
    pcd2 = PointCloudData(xyz2, numerical_optimization_shift=pcd.numerical_optimization_shift)
    shift_2 = pcd2.numerical_optimization_shift.value.copy()

    assert pcd2.xyz.dtype == np.float32
    assert len(osm) == 1
    assert len(pcd.numerical_optimization_shift) == 2
    assert np.any(shift_1 != shift_2)

    # Not feasible, add new group
    xyz2 = np.random.rand(100, 3) + [100000, 2000, 400]
    pcd3 = PointCloudData(xyz2, numerical_optimization_shift=pcd.numerical_optimization_shift)
    shift_3 = pcd3.numerical_optimization_shift.value.copy()

    assert pcd3.xyz.dtype == np.float32
    assert len(osm) == 2
    assert len(pcd.numerical_optimization_shift) == 2
    assert len(pcd3.numerical_optimization_shift) == 1

    assert pcd.numerical_optimization_shift is pcd2.numerical_optimization_shift
    assert pcd.numerical_optimization_shift is not pcd3.numerical_optimization_shift

    assert np.any(shift_3 != shift_2)
