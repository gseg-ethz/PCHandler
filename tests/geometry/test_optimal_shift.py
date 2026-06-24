import copy
import gc
import logging
import multiprocessing as mp
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
from pchandler.scalar_fields import ScalarFieldManager


def _child_unpickle_and_assert(blob: bytes, queue) -> None:
    """Run in a spawn-child: unpickle PCD, compute world frame, ``queue.put`` it.

    MUST be module-level (RESEARCH Pitfall 3): spawn-child re-imports this
    module, so class-method or nested-function helpers raise ``AttributeError``
    on the child side.
    """
    # Re-import inside the child for spawn-context isolation. The import below
    # is intentional even if the parent already imported the same names — spawn
    # creates a fresh interpreter that re-executes module top-level code.
    import pickle as _pickle

    import numpy as _np

    pcd_revived = _pickle.loads(blob)
    if pcd_revived.numerical_optimization_shift is not None:
        world = pcd_revived.xyz.astype(_np.float64) + pcd_revived.numerical_optimization_shift.value
    else:
        world = pcd_revived.xyz.astype(_np.float64)
    queue.put(world.tolist())


def random_coordinates(scale: float, offset: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz_base = rng.standard_normal((100, 3)).astype(np.float32)
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
    # Bracket: clean BEFORE yield (so a sibling test that mutated the singleton
    # in a prior run can't leak state into this test) AND after yield (keeps
    # the original teardown behaviour). Phase 9 D-12 F1.
    # Assigning to OptimizedShiftManager._instances would create a shadow attribute on
    # the subclass instead of clearing the metaclass-level dict (CR-01, post-COUPLE-01).
    SingletonMeta._instances.pop(OptimizedShiftManager, None)
    assert OptimizedShiftManager not in SingletonMeta._instances
    yield
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
        rng = np.random.default_rng(1)
        xyz = rng.random((100, 3))
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
        rng = np.random.default_rng(2)
        xyz = rng.random((100, 3))
        pcd = PointCloudData(xyz, numerical_optimization_shift=opt_shift)
        initial_value = opt_shift.value
        initial_uuid = opt_shift.uuid
        assert np.allclose(pcd.xyz, xyz - initial_value)

        new_value = np.array([100_000, 200_000, 300_000])
        with pytest.raises(OptimizedShiftManager.ShiftNotFeasibleError):
            opt_shift.value = new_value

    def test___contains__(self, opt_shift):
        rng = np.random.default_rng(3)
        pcd = PointCloudData(rng.random((100, 3)), numerical_optimization_shift=opt_shift)
        assert len(opt_shift._member_coordinate_sets) == 1
        assert pcd in opt_shift

        pcd2 = PointCloudData(rng.random((100, 3)) + 1000, numerical_optimization_shift=opt_shift)
        assert len(opt_shift._member_coordinate_sets) == 2
        assert pcd2 in opt_shift

    def test___len__(self, opt_shift):
        rng = np.random.default_rng(4)
        pcds = []
        for i in range(17):
            xyz = PointCloudData(rng.random((100, 3)), numerical_optimization_shift=opt_shift)
            pcds.append(xyz)

        assert len(opt_shift) == 17
        assert len(opt_shift._member_coordinate_sets) == 17

    def test___hash__(self, opt_shift):
        assert id(opt_shift) != hash(opt_shift)

    def test___eq__(self, opt_shift):
        rng = np.random.default_rng(5)
        xyz = PointCloudData(rng.random((100, 3)), numerical_optimization_shift=opt_shift)
        func, state = xyz.__reduce__()
        obj = PointCloudData.model_construct(**state[0])
        assert id(obj) != id(xyz)
        assert opt_shift.__eq__(obj.numerical_optimization_shift)

    def test___reduce__(self, opt_shift):
        rng = np.random.default_rng(6)
        xyz = PointCloudData(rng.random((100, 3)), numerical_optimization_shift=opt_shift)
        func, state = xyz.__reduce__()
        assert func == type(xyz)._reconstruct
        dumped = xyz.model_dump()
        for k, v in state[0].items():
            if isinstance(v, np.ndarray):
                assert np.all(v == dumped[k])
            elif isinstance(v, ScalarFieldManager):
                # model_dump returns a detached copy (CR-01 fix); compare by type only.
                assert isinstance(dumped[k], ScalarFieldManager)
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
        rng = np.random.default_rng(7)
        xyz = rng.random((100, 3))
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
        rng = np.random.default_rng(8)
        array = rng.random((100, 3))
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
    rng = np.random.default_rng(9)
    xyz = rng.random((100, 3))

    # add, no change
    pcd = PointCloudData(xyz, numerical_optimization_shift=OptimizedShift(np.array([-5000, 0, 0])))
    shift_1 = pcd.numerical_optimization_shift.value.copy()

    assert pcd.xyz.dtype == np.float32
    assert len(osm) == 1
    assert len(pcd.numerical_optimization_shift) == 1

    # Add and change
    xyz2 = rng.random((100, 3)) + np.array([6000, -1000, 2000])
    pcd2 = PointCloudData(xyz2, numerical_optimization_shift=pcd.numerical_optimization_shift)
    shift_2 = pcd2.numerical_optimization_shift.value.copy()

    assert pcd2.xyz.dtype == np.float32
    assert len(osm) == 1
    assert len(pcd.numerical_optimization_shift) == 2
    assert np.any(shift_1 != shift_2)

    # Not feasible, add new group
    xyz2 = rng.random((100, 3)) + [100000, 2000, 400]
    pcd3 = PointCloudData(xyz2, numerical_optimization_shift=pcd.numerical_optimization_shift)
    shift_3 = pcd3.numerical_optimization_shift.value.copy()

    assert pcd3.xyz.dtype == np.float32
    assert len(osm) == 2
    assert len(pcd.numerical_optimization_shift) == 2
    assert len(pcd3.numerical_optimization_shift) == 1

    assert pcd.numerical_optimization_shift is pcd2.numerical_optimization_shift
    assert pcd.numerical_optimization_shift is not pcd3.numerical_optimization_shift

    assert np.any(shift_3 != shift_2)


def test_optimized_shift_per_instance_feasibility():
    """BUG-04: per-instance _minimum_decimal_places gates feasibility independently of the manager.

    Notes
    -----
    Values are chosen so that the per-axis range (200) falls between the two
    instances' representable spans:
      - tight: ``_minimum_decimal_places=9`` → max_repr = 10**(7-9) = 0.01  → 200 > 0.01, rejected.
      - loose: manager default 3            → max_repr = 10**(7-3) = 10000 → 200 < 10000, accepted.

    The verbatim "1e10 magnitude" values from the plan/research draft were
    impossible to differentiate (both instances reject 1e10); corrected here
    per executor Rule 1 — Bug.
    """
    # Per-axis range = 200 (max=100, min=-100). See docstring math.
    bbox_values = np.array([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])

    tight_shift = OptimizedShift(np.zeros(3))
    tight_shift._minimum_decimal_places = 9
    assert not tight_shift._is_shift_possible(bbox_values), (
        "Tight (9 decimals) shift instance must reject range-200 values"
    )

    loose_shift = OptimizedShift(np.zeros(3))
    # _minimum_decimal_places stays None → falls through to manager default (3)
    assert loose_shift._is_shift_possible(bbox_values), (
        "Loose (manager default) shift instance must accept range-200 values"
    )


def test_optimized_shift_eq_returns_notimplemented_on_non_shift():
    """WR-04: OptimizedShift.__eq__ does not crash on non-OptimizedShift operand.

    ``__eq__`` is exercised by hash-set deduplication and by
    ``Optional[OptimizedShift]`` comparisons throughout merge/un-shift
    surfaces, where ``None`` is a routine operand. The previous form
    raised ``AttributeError`` on the first ``None`` (``None.uuid`` does
    not exist); returning :data:`NotImplemented` for unsupported types
    is the Python equality protocol's documented contract and lets the
    runtime fall back to identity comparison.
    """
    shift = OptimizedShift(np.array([1.0, 2.0, 3.0]))

    # None operand: previously raised AttributeError. Must compare False.
    assert (shift == None) is False  # noqa: E711 - explicit None equality test
    assert (None == shift) is False  # noqa: E711
    assert (shift != None) is True  # noqa: E711

    # Other non-shift operands: also False (not a TypeError, not a crash).
    assert (shift == 42) is False
    assert (shift == "shift") is False

    # set() deduplication of Optional[OptimizedShift]: must not crash.
    bag = {shift, None, shift}
    assert len(bag) == 2 and None in bag and shift in bag


def test_optimized_shift_register_accepts_unshifted_world_frame():
    """BUG-06 positive: register takes world-frame (unshifted) coordinates as documented."""
    # Build a small set of world-frame coordinates clustered around (100, 200, 300)
    world_xyz = np.array(
        [[100.0, 200.0, 300.0], [100.1, 200.1, 300.1], [100.2, 200.2, 300.2]],
        dtype=np.float64,
    )
    cc = CartesianCoordinates(world_xyz)
    # Register at a shift close to the centroid so the shift is feasible.
    shift = OptimizedShift(np.array([100.0, 200.0, 300.0]))
    shift.register(cc)
    assert cc in shift, "register(unshifted_coords) must put coords in the shift's member set"


def _make_pcd_with_nos():
    """Build a small PCD with a non-trivial NOS for pickle tests."""
    world_xyz = np.array(
        [[100.0, 200.0, 300.0], [100.1, 200.1, 300.1], [100.2, 200.2, 300.2]],
        dtype=np.float64,
    )
    nos = OptimizedShift(np.array([100.0, 200.0, 300.0]))
    return PointCloudData(world_xyz, numerical_optimization_shift=nos)


def test_pcd_unpickle_fresh_manager_preserves_world_frame():
    """FRAG-01 scenario 1: unpickle on a fresh manager preserves world frame."""
    pcd = _make_pcd_with_nos()
    source_world_frame = pcd.xyz + pcd.numerical_optimization_shift.value
    blob = pickle.dumps(pcd)

    # Simulate fresh destination manager (Phase 1 hot-patch ae401dd pattern)
    SingletonMeta._instances.pop(OptimizedShiftManager, None)

    restored = pickle.loads(blob)
    restored_world_frame = restored.xyz + restored.numerical_optimization_shift.value
    np.testing.assert_array_equal(source_world_frame, restored_world_frame)


def test_pcd_unpickle_uuid_match_vector_match_returns_existing():
    """FRAG-01 scenario 2: same-UUID-same-vector → returns the existing manager instance."""
    pcd = _make_pcd_with_nos()
    pre_existing = pcd.numerical_optimization_shift  # already registered at construction
    blob = pickle.dumps(pcd)
    # Manager state unchanged between pickle.dumps and pickle.loads
    restored = pickle.loads(blob)
    assert restored.numerical_optimization_shift is pre_existing


def test_pcd_unpickle_uuid_match_vector_mismatch_mints_new():
    """FRAG-01 scenario 3: same-UUID-different-vector → mints fresh UUID, preserves world frame."""
    pcd = _make_pcd_with_nos()
    source_uuid = pcd.numerical_optimization_shift._uuid
    source_vec = pcd.numerical_optimization_shift.value
    source_world_frame = pcd.xyz + source_vec
    blob = pickle.dumps(pcd)

    # Mutate destination manager: clear, then register a bogus shift under the
    # source UUID but with a divergent vector. Use _construct_with_uuid (the
    # private staticmethod) to bypass __init__'s auto-mint behaviour.
    SingletonMeta._instances.pop(OptimizedShiftManager, None)
    bogus = OptimizedShift._construct_with_uuid(source_uuid, np.array([999.0, 999.0, 999.0]))
    assert OptimizedShiftManager().get_by_uuid(source_uuid) is bogus

    restored = pickle.loads(blob)
    # Mint-new-UUID branch fired: destination NOS has a different UUID
    assert restored.numerical_optimization_shift._uuid != source_uuid
    # The pickled vector is preserved on the new shift
    np.testing.assert_array_equal(restored.numerical_optimization_shift.value, source_vec)
    # And the world frame is preserved (most important assertion)
    np.testing.assert_array_equal(
        restored.xyz + restored.numerical_optimization_shift.value,
        source_world_frame,
    )


# ---------------------------------------------------------------------------
# Phase 6 TEST-01 — cross-process unpickle + UUID-collision contract
# (D-04 / ROADMAP SC #1 — Plan 06-01)
# ---------------------------------------------------------------------------


@pytest.fixture
def assert_pchandler_caplog_propagation(caplog: pytest.LogCaptureFixture) -> None:
    """Smoke-fixture: verify ``caplog`` receives ``pchandler.*`` INFO records.

    Mitigates RESEARCH §A1 [ASSUMED] (pchandler logger propagation to caplog).
    Triggers a known-emitting INFO path on the ``pchandler`` logger by hitting
    ``OptimizedShift._reconstruct``'s divergent-vector branch and asserts the
    INFO record reaches ``caplog``. If propagation is broken, TEST-01's
    caplog assertions would silently pass vacuously.

    Note: ``_by_uuid`` is a ``WeakValueDictionary``; the pre-registered shift
    must be kept alive via a local strong reference for ``get_by_uuid`` to
    return it during ``_reconstruct``.
    """
    caplog.set_level(logging.INFO, logger="pchandler")
    SingletonMeta._instances.pop(OptimizedShiftManager, None)
    shift_vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    bogus_uuid = uuid.uuid4()
    _alive = OptimizedShift._construct_with_uuid(bogus_uuid, shift_vec)  # noqa: F841
    # Reconstruct under same UUID with divergent vector → INFO emitter
    OptimizedShift._reconstruct(bogus_uuid, np.array([10.0, 20.0, 30.0], dtype=np.float64))
    assert caplog.records, (
        "pchandler logger propagation broken — investigate. "
        "Expected an INFO record from OptimizedShift._reconstruct divergent-vector "
        "branch; caplog received zero records."
    )
    caplog.clear()
    # Reset OSM so the test gets a clean manager.
    del _alive
    SingletonMeta._instances.pop(OptimizedShiftManager, None)


class TestReconstructCrossProcess:
    """TEST-01 / D-04 / SC #1: cross-process pickle + UUID-collision three-branch state machine.

    The three-branch state machine is documented at
    ``src/pchandler/geometry/optimal_shift.py:789-835``. The class also asserts
    spawn-child fidelity for the no-collision branch (the spawn child runs in
    a freshly-instantiated ``OptimizedShiftManager``, so the pickled UUID is
    free and ``_reconstruct`` takes case 1).
    """

    def test_subprocess_unpickle_preserves_world_frame(self):
        """Scenario 1 (cross-process): spawn child unpickles PCD, world frame preserved.

        Per RESEARCH §D-04: ``mp.get_context("spawn")`` matches CI's macOS/Win
        default and produces a known-clean subinterpreter (no fork-inherited
        OSM state, Pitfall 1).
        """
        rng = np.random.default_rng(0)
        # Large-coord world frame tightly clustered around a shift origin so
        # the shift is feasible (range << maximum_number_representable).
        origin = np.array([1e6, 2e6, 3e6], dtype=np.float64)
        xyz_world_frame = origin + rng.random((100, 3)) * 10.0
        nos = OptimizedShift(origin)
        pcd = PointCloudData(xyz_world_frame, numerical_optimization_shift=nos)
        assert pcd.numerical_optimization_shift is not None, (
            "Test fixture broken: expected the NOS to be feasible for this coord cluster."
        )
        parent_world = pcd.xyz.astype(np.float64) + pcd.numerical_optimization_shift.value

        blob = pickle.dumps(pcd)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        p = ctx.Process(target=_child_unpickle_and_assert, args=(blob, queue))
        p.start()
        try:
            child_world = np.array(queue.get(timeout=30))
        finally:
            p.join(timeout=30)
        assert p.exitcode == 0, f"spawn child exited with {p.exitcode}"
        np.testing.assert_allclose(child_world, parent_world, atol=1e-6)

    def test_reconstruct_uuid_collision_same_vec_reuses(self):
        """Scenario 2 (within-process, branch 2 of _reconstruct): same-UUID-same-vector reuses existing.

        Per Phase 3 D-18 branch 2: the unpickled NOS IS the pre-existing
        manager-held instance (object identity, not just equality).

        Note: ``_by_uuid`` is a ``WeakValueDictionary``; the pre-registered
        shift survives because the constructed PCD holds a strong reference.
        """
        SingletonMeta._instances.pop(OptimizedShiftManager, None)
        shift_vec = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        known_uuid = uuid.uuid4()
        # Pre-register a shift under known_uuid with shift_vec; this is the
        # "existing same-vector" instance the unpickle is expected to reuse.
        existing = OptimizedShift._construct_with_uuid(known_uuid, shift_vec)

        # Build a PCD whose NOS already IS that existing shift, then pickle.
        world_xyz = np.array(
            [[100.0, 200.0, 300.0], [100.1, 200.1, 300.1], [100.2, 200.2, 300.2]],
            dtype=np.float64,
        )
        pcd = PointCloudData(world_xyz, numerical_optimization_shift=existing)
        blob = pickle.dumps(pcd)
        unpickled = pickle.loads(blob)

        # Branch 2: object identity preserved across pickle round-trip on
        # same-UUID-same-vector.
        assert unpickled.numerical_optimization_shift is existing
        assert unpickled.numerical_optimization_shift._uuid == known_uuid
        # Keep PCD alive across the assertions; without it the WeakValue could
        # be GC'd between pickle and assert in aggressive-GC interpreters.
        del pcd

    def test_reconstruct_uuid_collision_divergent_vec_mints(
        self,
        caplog: pytest.LogCaptureFixture,
        assert_pchandler_caplog_propagation,
    ):
        """Scenario 3 (within-process, branch 3 of _reconstruct): divergent-vector mints fresh UUID.

        Per Phase 3 D-18 branch 3: a UUID collision with a divergent vector
        mints a fresh ``uuid.uuid4()`` and registers a new shift. World-frame
        coordinates are preserved at the cost of UUID continuity. The
        ``pchandler`` INFO record "UUID collision with divergent vector" must
        fire on the destination side.
        """
        caplog.set_level(logging.DEBUG, logger="pchandler")

        SingletonMeta._instances.pop(OptimizedShiftManager, None)
        known_uuid = uuid.uuid4()
        vec_a = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        vec_b = np.array([900.0, 900.0, 900.0], dtype=np.float64)
        # Build a PCD whose NOS carries (known_uuid, vec_b); pickle records that pair.
        nos_b = OptimizedShift._construct_with_uuid(known_uuid, vec_b)
        world_xyz = np.array(
            [[900.0, 900.0, 900.0], [900.1, 900.1, 900.1], [900.2, 900.2, 900.2]],
            dtype=np.float64,
        )
        pcd = PointCloudData(world_xyz, numerical_optimization_shift=nos_b)
        source_world = pcd.xyz.astype(np.float64) + pcd.numerical_optimization_shift.value
        blob = pickle.dumps(pcd)

        # Reset destination manager and pre-register (known_uuid, vec_a) — a
        # divergent vector under the same UUID. ``_by_uuid`` is a
        # WeakValueDictionary; bind the registered shift to a local so it
        # survives until ``_reconstruct`` queries the manager.
        SingletonMeta._instances.pop(OptimizedShiftManager, None)
        _alive_a = OptimizedShift._construct_with_uuid(known_uuid, vec_a)  # noqa: F841

        unpickled = pickle.loads(blob)
        unpickled_world = unpickled.xyz.astype(np.float64) + unpickled.numerical_optimization_shift.value

        # Branch 3: fresh UUID minted; vec_b preserved on the new shift.
        assert unpickled.numerical_optimization_shift._uuid != known_uuid
        np.testing.assert_array_equal(unpickled.numerical_optimization_shift._shift, vec_b)
        # World-frame preserved despite UUID mint.
        np.testing.assert_array_equal(unpickled_world, source_world)
        # INFO log fired.
        assert any(
            "UUID collision with divergent vector" in r.message and r.levelname == "INFO" for r in caplog.records
        ), (
            "Expected INFO log 'UUID collision with divergent vector' on branch 3 of "
            f"_reconstruct; got records: {[(r.levelname, r.message) for r in caplog.records]}"
        )
        # DEBUG log fired (folded in10 / Phase 6 D-23 — landed by Plan 06-04 Task 7).
        assert any("shift collision detail" in r.message and r.levelname == "DEBUG" for r in caplog.records), (
            "Expected DEBUG log 'shift collision detail' on branch 3 of "
            f"_reconstruct (folded in10); got records: {[(r.levelname, r.message) for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# Phase 5 API-06 regression tests (Plan 05-05)
# ---------------------------------------------------------------------------


class TestOptimizedShiftManagerApiCompletion:
    """API-06: manager setter removed; getter + init kwarg preserved (D-16, D-19 T01-T02)."""

    def test_setter_removed(self):
        """T01: Writing to OSM().minimum_decimal_places raises AttributeError (D-16)."""
        osm = OptimizedShiftManager()
        with pytest.raises(AttributeError, match="(has no setter|can't set attribute|read.only)"):
            osm.minimum_decimal_places = 5

    def test_init_kwarg_then_getter(self):
        """T02: OSM getter returns the value supplied via the init kwarg (D-16)."""
        # clear_instantiated_osm autouse fixture resets the singleton per-function
        osm = OptimizedShiftManager(minimum_decimal_places=4)
        assert osm.minimum_decimal_places == 4


class TestOptimizedShiftApiCompletion:
    """API-06: per-instance setter + init kwarg on OptimizedShift (D-17, D-18, D-19 T03-T06)."""

    def test_per_instance_setter_happy(self):
        """T03: Assigning a feasible precision to an OptimizedShift succeeds (D-17).

        Note: the PointCloudData must be kept alive so its CartesianCoordinates
        remains in the WeakSet during the setter call.
        """
        rng = np.random.default_rng(42)
        shift = OptimizedShift(np.zeros(3, dtype=np.float64))
        # Register a small coord set (range ~1; well within maximum_number_representable=10^4)
        xyz = rng.random((10, 3)).astype(np.float64)
        pcd = PointCloudData(xyz, numerical_optimization_shift=shift)  # noqa: F841

        shift.minimum_decimal_places = 4
        assert shift.minimum_decimal_places == 4
        del pcd  # allow GC after test

    def test_per_instance_setter_reverts_on_infeasible(self):
        """T04: Setter reverts _minimum_decimal_places on ShiftNotFeasibleError (D-17).

        A precision of 18 is far above the float64 ceiling (~15 significant
        digits); maximum_number_representable = 10**(7-18) = 1e-11.  Even a
        coord range of ~1 in float64 exceeds that limit, so _is_shift_possible
        returns False.

        Note: the PointCloudData must be kept alive so its CartesianCoordinates
        remains in the WeakSet; without a reference the coord set is GC'd before
        the setter walks the member list.
        """
        rng = np.random.default_rng(42)
        shift = OptimizedShift(np.zeros(3, dtype=np.float64))
        # Keep the PCD alive so the coord set stays in shift._member_coordinate_sets
        xyz = rng.random((10, 3)).astype(np.float64) + 1.0  # range ~1
        pcd = PointCloudData(xyz, numerical_optimization_shift=shift)  # noqa: F841

        old_value = shift.minimum_decimal_places  # either None→manager or explicitly set

        with pytest.raises(OptimizedShiftManager.ShiftNotFeasibleError):
            shift.minimum_decimal_places = 18  # impossibly tight

        # Revert invariant: value must equal the old observable minimum_decimal_places
        assert shift.minimum_decimal_places == old_value
        del pcd  # allow GC after test

    def test_init_kwarg(self):
        """T05: OptimizedShift(shift_vec=..., minimum_decimal_places=5) stores the value (D-18)."""
        shift = OptimizedShift(
            shift_vec=np.array([1000.0, 2000.0, 3000.0], dtype=np.float64),
            minimum_decimal_places=5,
        )
        assert shift.minimum_decimal_places == 5

    def test_init_kwarg_fallback_to_manager(self):
        """T06: OptimizedShift() without kwarg falls back to the manager default (Phase 3 D-08)."""
        shift = OptimizedShift(shift_vec=np.array([1000.0, 2000.0, 3000.0], dtype=np.float64))
        assert shift.minimum_decimal_places == OptimizedShiftManager().minimum_decimal_places
