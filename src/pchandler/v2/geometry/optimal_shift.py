from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING, ClassVar, Self, Literal, NamedTuple, cast, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import PointCloudData

from ..base_types import Vector_3_T, Array_Nx3_T

class SingletonMeta(type):
    _instances: ClassVar[dict[type, object]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __call__(cls, *args, **kwargs) -> Self:
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]


class OptimizedShiftManager(metaclass=SingletonMeta):
    _optimized_shifts: weakref.WeakSet[OptimizedShift]
    _maximum_decimal_places: int

    class ShiftNotFeasibleError(Exception):
        pass

    def __init__(self, maximum_decimal_places: int = 4) -> None:
        self._optimized_shifts = weakref.WeakSet()
        self._maximum_decimal_places = maximum_decimal_places

    def register(self, shift: OptimizedShift) -> None:
        self._optimized_shifts.add(shift)

    # FIXME: Has no usages?
    @staticmethod
    def new_shift() -> OptimizedShift:
        return OptimizedShift()

    @property
    def all_shifts(self) -> list[OptimizedShift]:
        return list(self._optimized_shifts)

    @property
    def maximum_decimal_places(self) -> int:
        return self._maximum_decimal_places

    @maximum_decimal_places.setter
    def maximum_decimal_places(self, maximum_decimal_places: int) -> None:
        pass # Todo: Should probably check all registered GlobalShifts and their pcds if they conform to new limit

    def is_shift_needed(self, values: NDArray[np.floating] | Array_Nx3_T | "PointCloudData") -> bool:
        return np.any(np.abs(values) >= 10 ** self._maximum_decimal_places)

    def is_shift_possible(self, values: NDArray[np.floating] | Array_Nx3_T | "PointCloudData") -> bool:
        return np.all(np.subtract(np.max(values, axis=0), np.min(values, axis=0)) < 10 ** self._maximum_decimal_places)

class OptimizedShift:

    class MinMaxPoints(NamedTuple):
        minimum: Vector_3_T
        maximum: Vector_3_T

        @classmethod
        def from_points(cls, points: Array_Nx3_T) -> Self:
            min_point = np.min(points, axis=0)
            max_point = np.max(points, axis=0)
            return cls(min_point, max_point)

        @classmethod
        def from_minmax_points(cls, minmax_points: Iterable[Self]) -> Self:
            arr = Array_Nx3_T(np.vstack(tuple(minmax_points)))
            return cls.from_points(arr)

        @property
        def central_point(self) -> Vector_3_T:
            return Vector_3_T(np.mean(np.vstack((self.minimum, self.maximum)), axis=0))

        @property
        def extents(self) -> Vector_3_T:
            return Vector_3_T(np.subtract((self.maximum, self.minimum)), axis=0)

    _optimal_shift: Vector_3_T
    _member_pcds: weakref.WeakSet["PointCloudData"]
    _member_pcds_unshifted_bbox: weakref.WeakKeyDictionary["PointCloudData", MinMaxPoints]

    def __init__(self, optimal_shift: Optional[NDArray[np.floating] | Vector_3_T] = None) -> None:
        self._optimal_shift = Vector_3_T(np.zeros(3)) if optimal_shift is None else optimal_shift
        self._member_pcds = weakref.WeakSet()
        self._member_pcds_unshifted_bbox = weakref.WeakKeyDictionary()
        OptimizedShiftManager().register(self)

    @property
    def optimal_shift(self) -> NDArray[np.float64]:
        return self._optimal_shift

    def register(self, pcd: PointCloudData, points: Array_Nx3_T) -> Optional[Self]:
        """
        Try to add `pcd` (with its point‐cloud `points`) into this shift.
        Returns self if successful;
        returns a brand‐new OptimizedShift if this set can’t fit but we had others;
        or None if even a singleton can’t fit.
        """

        if self._can_add_without_change(points):
            return self._add_member(pcd, points)

        try:
            return self._expand_and_add(pcd, points)
        except OptimizedShiftManager.ShiftNotFeasibleError:
            return self._restart_or_fail(pcd, points)


    def _can_add_without_change(self, pts: Array_Nx3_T) -> bool:
        shifted = np.subtract(pts, self._optimal_shift)
        return not OptimizedShiftManager().is_shift_needed(shifted)

    def _add_member(self, pcd, pts):
        """Add the point‐cloud under the existing shift."""
        self._member_pcds.add(pcd)
        self._member_pcds_unshifted_bbox[pcd] = OptimizedShift.MinMaxPoints.from_points(pts)
        return self

    def _expand_and_add(self, pcd, pts):
        """
        Compute a new optimal shift to cover existing + this pts,
        apply it and then add the new pcd.
        """
        new_shift = self._compute_new_shift(pts)
        self._apply_shift_delta(new_shift)
        self._optimal_shift = new_shift
        return self._add_member(pcd, pts)

    def _compute_new_shift(self, pts) -> Vector_3_T:
        # build up the combined bounding‐box
        all_boxes = list(self._member_pcds_unshifted_bbox.values())
        all_boxes.append(OptimizedShift.MinMaxPoints.from_points(pts))
        combined = OptimizedShift.MinMaxPoints.from_minmax_points(all_boxes)

        if not OptimizedShiftManager().is_shift_possible(np.vstack((combined.minimum, combined.maximum))):
            raise OptimizedShiftManager.ShiftNotFeasibleError()

        # round the center to keep ints
        return Vector_3_T(np.round(combined.central_point, decimals=-(OptimizedShiftManager().maximum_decimal_places- 1)))

    def _apply_shift_delta(self, new_shift):
        """Move every registered PCD by the change from old→new shift."""
        delta = self._optimal_shift - new_shift
        for pcd in self._member_pcds:
            pcd.update_shift(delta)

    def _restart_or_fail(self, pcd, pts):
        if self._member_pcds:
            # start a new group with this single pcd
            return OptimizedShift().register(pcd, pts)
        return None

    # def register(self, pcd: "PointCloudData", values: Array_Nx3_T) -> Optional[Self]:
    #     if self.does_current_shift_fit(values):
    #         self._member_pcds.add(pcd)
    #         self._member_pcds_unshifted_bbox[pcd] = OptimizedShift.MinMaxPoints.from_points(values)
    #         return self
    #     try:
    #         new_shift = self.calculate_new_shift(values)
    #         self.inform_members_change(new_shift)
    #         self._optimal_shift = new_shift
    #         self._member_pcds.add(pcd)
    #         self._member_pcds_unshifted_bbox[pcd] = OptimizedShift.MinMaxPoints.from_points(values)
    #         return self
    #     except OptimizedShiftManager.ShiftNotFeasibleError:
    #         # In this case the new pcd can't be combined with the already registered pcds
    #         if len(self._member_pcds) > 0:
    #             new_optimal_shift = OptimizedShift()
    #             new_shift = new_optimal_shift.register(pcd, values)
    #             return new_shift
    #         else:
    #             return None
    #
    # def inform_members_change(self, new_shift: Vector_3_T) -> None:
    #     shift_delta = self._optimal_shift - new_shift
    #     for pcd in self._member_pcds:
    #         pcd.update_shift(shift_delta)
    #
    # def does_current_shift_fit(self, values: Array_Nx3_T | NDArray[np.floating]) -> bool:
    #     return not bool(OptimizedShiftManager().is_shift_needed(np.subtract(values, self._optimal_shift)))
    #
    # def calculate_new_shift(self, values: Array_Nx3_T| NDArray[np.floating]) -> Vector_3_T:
    #     minmax = OptimizedShift.MinMaxPoints.from_points(values)
    #     minmax_points = [mmp for mmp in self._member_pcds_unshifted_bbox.values()]
    #     minmax_points.append(minmax)
    #     new_bbox = OptimizedShift.MinMaxPoints.from_minmax_points(minmax_points)
    #
    #     if not OptimizedShiftManager().is_shift_possible(np.vstack((new_bbox.minimum, new_bbox.maximum))):
    #         raise OptimizedShiftManager().ShiftNotFeasibleError()
    #
    #     return np.round(new_bbox.central_point)
