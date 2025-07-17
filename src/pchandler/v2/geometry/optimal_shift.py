from __future__ import annotations

import threading
import uuid
import weakref
from typing import TYPE_CHECKING, ClassVar, Self, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import PointCloudData
    from .core import CartesianCoordinates

from .util import MinMaxPoints
from ..base_types import Vector_3_T, Array_Nx3_T
from ..constants import validate_variables

class SingletonMeta(type):
    _instances: ClassVar[dict[type, object]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __call__(cls, *args, **kwargs) -> Self:
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]


class OptimizedShiftManager(metaclass=SingletonMeta):
    _by_uuid: weakref.WeakValueDictionary[uuid.UUID, OptimizedShift]
    _maximum_decimal_places: int
    FLOAT32_DECIMAL_PRECISION: int = 7

    class ShiftNotFeasibleError(Exception):
        pass

    class ShiftUUIDAlreadyTaken(Exception):
        pass

    class ShiftUUIDNotFound(Exception):
        pass

    def __init__(self, minimum_decimal_places: int = 3) -> None:
        self._minimum_decimal_places = minimum_decimal_places
        self._by_uuid = weakref.WeakValueDictionary()

    def register_shift(self, shift: OptimizedShift) -> None:
        if shift.uuid in self._by_uuid and id(shift) is not id(self._by_uuid[shift.uuid]):
            raise OptimizedShiftManager.ShiftUUIDAlreadyTaken()

        self._by_uuid[shift.uuid] = shift

    def register_with(self, coordinates: CartesianCoordinates, shift: OptimizedShift | uuid.UUID) -> OptimizedShift:
        if isinstance(shift, uuid.UUID):
            try:
                shift = self._by_uuid[shift]
            except KeyError:
                raise OptimizedShiftManager.ShiftUUIDNotFound()
        while True:
            try:
                shift.register(coordinates)
                return shift
            except OptimizedShiftManager.ShiftNotFeasibleError:
                if len(shift) >= 1: # Allow for one more attempt on a fresh OptimizedShift
                    shift = OptimizedShift()
                else:
                    break
        raise OptimizedShiftManager.ShiftNotFeasibleError()

    def __len__(self):
        return len(self._by_uuid)

    @property
    def all_shifts(self) -> list[OptimizedShift]:
        return list(self._by_uuid.values())

    @property
    def minimum_decimal_places(self) -> int:
        return self._minimum_decimal_places

    @minimum_decimal_places.setter
    def minimum_decimal_places(self, maximum_decimal_places: int) -> None:
        pass # Todo: Should probably check all registered GlobalShifts and their pcds if they conform to new limit

    @property
    def maximum_number_representable(self) -> float:
        return 10**(self.FLOAT32_DECIMAL_PRECISION - self.minimum_decimal_places)

    def get_by_uuid(self, u: uuid.UUID) -> Optional[OptimizedShift]:
        return self._by_uuid.get(u)

    def is_shift_needed(self, values: Array_Nx3_T) -> bool:
        return np.any(np.abs(values) >= self.maximum_number_representable)

    def is_shift_possible(self, values: Array_Nx3_T) -> bool:
        return np.all(np.subtract(np.max(values, axis=0), np.min(values, axis=0)) < self.maximum_number_representable)

class OptimizedShift:
    _uuid: uuid.UUID
    _shift: Vector_3_T
    _member_coordinate_sets: weakref.WeakSet[CartesianCoordinates]
    _member_coordinate_sets_unshifted_bbox: weakref.WeakKeyDictionary[CartesianCoordinates, MinMaxPoints]

    @validate_variables
    def __init__(self, shift_vec: Optional[Vector_3_T] = None) -> None:
        self._uuid = uuid.uuid4()
        self._shift = np.zeros((3,), dtype=np.float64) if shift_vec is None else shift_vec
        self._member_coordinate_sets = weakref.WeakSet()
        self._member_coordinate_sets_unshifted_bbox = weakref.WeakKeyDictionary()
        OptimizedShiftManager().register_shift(self)

    def __len__(self):
        return len(self._member_coordinate_sets)

    @property
    def uuid(self):
        return self._uuid

    @property
    def value(self) -> NDArray[np.float64]:
        return self._shift

    def register(self, coordinate_set: CartesianCoordinates) -> None:
        """
        Try to add `pcd` (with its point‐cloud `points`) into this shift.
        Returns self if successful;
        returns a brand‐new OptimizedShift if this set can’t fit but we had others;
        or None if even a singleton can’t fit.
        """

        if self._can_add_without_change(coordinate_set.arr):
            self._add_member(coordinate_set)
            return
        try:
            self._expand_and_add(coordinate_set)
            return
        except OptimizedShiftManager.ShiftNotFeasibleError:
            raise OptimizedShiftManager.ShiftNotFeasibleError()

    def check_addibility(self, points: Array_Nx3_T) -> bool:
        try:
            _ = self._compute_new_shift(points)
            return True
        except OptimizedShiftManager.ShiftNotFeasibleError:
            return False


    def _can_add_without_change(self, pts: Array_Nx3_T) -> bool:
        shifted = np.subtract(pts, self._shift)
        return not OptimizedShiftManager().is_shift_needed(shifted)

    def _add_member(self, coordinate_set: CartesianCoordinates) -> None:
        """Add the point‐cloud under the existing shift."""
        self._member_coordinate_sets.add(coordinate_set)
        self._member_coordinate_sets_unshifted_bbox[coordinate_set] = MinMaxPoints.from_points(coordinate_set.arr)

    def _expand_and_add(self, coordinate_set: CartesianCoordinates) -> None:
        """
        Compute a new optimal shift to cover existing + this pts,
        apply it and then add the new pcd.
        """
        new_shift = self._compute_new_shift(coordinate_set.arr)
        self._apply_shift_delta(new_shift)
        self._shift = new_shift
        self._add_member(coordinate_set)

    def _compute_new_shift(self, points: Array_Nx3_T) -> Vector_3_T:
        # build up the combined bounding‐box
        all_bboxes = list(self._member_coordinate_sets_unshifted_bbox.values())
        all_bboxes.append(MinMaxPoints.from_points(points))
        combined = MinMaxPoints.from_minmax_points(all_bboxes)

        if not OptimizedShiftManager().is_shift_possible(np.vstack((combined.minimum, combined.maximum))):
            raise OptimizedShiftManager.ShiftNotFeasibleError()

        # round the center to keep ints
        return np.round(combined.central_point, decimals=-2)

    def _apply_shift_delta(self, new_shift: Vector_3_T):
        """Move every registered coordinate set by the change from old to new shift."""
        delta = self._shift - new_shift
        for pcd in self._member_coordinate_sets:
            pcd.update_shift(delta)


    def __reduce__(self):
        return self._reconstruct, (self._uuid, self._shift)

    @staticmethod
    def _reconstruct(u: uuid.UUID, shift_vec: Vector_3_T) -> "OptimizedShift":
        mgr = OptimizedShiftManager()
        existing = mgr.get_by_uuid(u)
        if existing is not None and np.allclose(existing.value, shift_vec):
            return existing

        new = object.__new__(OptimizedShift)
        new._uuid = u
        new._shift = shift_vec
        new._member_coordinate_sets = weakref.WeakSet()
        new._member_coordinate_sets_unshifted_bbox = weakref.WeakKeyDictionary()
        mgr.register_shift(new)
        return new

    def reattach_member(self, coordinate_set: CartesianCoordinates) -> None:
        """
       Register `pcd` under this shift, *without* changing any existing member PCDs.
       `unshifted_pts` should be the original float64 points before shifting.
       """
        # add to the weakref set
        self._member_coordinate_sets.add(coordinate_set)
        # store its unshifted bounding box
        self._member_coordinate_sets_unshifted_bbox[coordinate_set] = (
            MinMaxPoints.from_points(coordinate_set.xyz, already_applied_shift_vec = self._shift)
        )

