# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

from __future__ import annotations

import copy
import logging
import threading
import uuid
import weakref
from typing import TYPE_CHECKING, ClassVar, Optional, cast
from uuid import UUID

import numpy as np
from GSEGUtils.base_types import Array_Nx3_T, Vector_3_T
from GSEGUtils.constants import validate_variables

from pchandler.geometry.util import MinMaxPoints

if TYPE_CHECKING:
    from pchandler.geometry.coordinates import CartesianCoordinates

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    _instances: ClassVar[dict[type, object]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]


class OptimizedShiftManager(metaclass=SingletonMeta):
    _by_uuid: weakref.WeakValueDictionary[uuid.UUID, OptimizedShift]
    _minimum_decimal_places: int
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

    def __len__(self):
        return len(self._by_uuid)

    @property
    def all_shifts(self) -> list[OptimizedShift]:
        return list(self._by_uuid.values())

    @property
    def minimum_decimal_places(self) -> int:
        return self._minimum_decimal_places

    @minimum_decimal_places.setter
    def minimum_decimal_places(self, value: int) -> None:  # TODO input value name correct?
        pass  # Todo: Should probably check all registered GlobalShifts and their pcds if they conform to new limit

    @property
    def maximum_number_representable(self) -> float:
        return 10 ** (self.FLOAT32_DECIMAL_PRECISION - self.minimum_decimal_places)

    def get_by_uuid(self, u: UUID) -> Optional[OptimizedShift]:
        return self._by_uuid.get(u)

    def is_shift_needed(self, values: Array_Nx3_T) -> bool:
        return bool(np.any(np.abs(values) >= self.maximum_number_representable))

    def is_shift_possible(self, values: Array_Nx3_T) -> bool:
        return bool(
            np.all(np.subtract(np.max(values, axis=0), np.min(values, axis=0)) < self.maximum_number_representable)
        )

    def update_uuid(self, old_uuid: uuid.UUID, new_uuid: uuid.UUID) -> None:
        if old_uuid not in self._by_uuid:
            raise OptimizedShiftManager.ShiftUUIDNotFound(f"Shift with uuid: {old_uuid} not found.")
        if new_uuid in self._by_uuid:
            raise OptimizedShiftManager.ShiftUUIDAlreadyTaken(f"Shift with uuid: {new_uuid} already exists.")
        self._by_uuid[new_uuid] = self._by_uuid.pop(old_uuid)

    def register_shift(self, shift: OptimizedShift) -> None:
        if shift.uuid in self._by_uuid and id(shift) is not id(self._by_uuid[shift.uuid]):
            raise OptimizedShiftManager.ShiftUUIDAlreadyTaken()

        self._by_uuid[shift.uuid] = shift

    def register_coordinates_to_shift(
        self, coordinates: CartesianCoordinates, shift: OptimizedShift | uuid.UUID
    ) -> OptimizedShift:
        if isinstance(shift, OptimizedShift):
            current_shift = shift
        else:
            try:
                current_shift: OptimizedShift = self._by_uuid[shift]
            except KeyError:
                raise OptimizedShiftManager.ShiftUUIDNotFound()

        while True:
            try:
                current_shift.register(coordinates)
                return current_shift

            except OptimizedShiftManager.ShiftNotFeasibleError:
                if len(current_shift) >= 1:  # Allow for one more attempt on a fresh OptimizedShift
                    current_shift = OptimizedShift()
                else:
                    break

        raise OptimizedShiftManager.ShiftNotFeasibleError()


class OptimizedShift:
    _uuid: uuid.UUID
    _shift: Vector_3_T
    _member_coordinate_sets: weakref.WeakSet[CartesianCoordinates]
    # _member_coordinate_sets_unshifted_bbox: weakref.WeakKeyDictionary[CartesianCoordinates, MinMaxPoints]

    @validate_variables
    def __init__(self, shift_vec: Optional[Vector_3_T] = None) -> None:
        self._uuid = uuid.uuid4()
        self._shift = np.zeros((3,), dtype=np.float64) if shift_vec is None else shift_vec
        self._member_coordinate_sets = weakref.WeakSet()
        # self._member_coordinate_sets_unshifted_bbox = weakref.WeakKeyDictionary()
        OptimizedShiftManager().register_shift(self)

    def __contains__(self, value: CartesianCoordinates) -> bool:
        return any(value is pcd for pcd in self._member_coordinate_sets)

    def __len__(self):
        return len(self._member_coordinate_sets)

    def __hash__(self) -> int:
        return hash(self._uuid)

    def __eq__(self, other) -> bool:
        return self.uuid == other.uuid

    def __reduce__(self):
        return self._reconstruct, (self._uuid, self._shift)

    def __deepcopy__(self, memo):
        # Construct new
        new_vec = copy.deepcopy(self._shift, memo)
        new_shift = type(self)(new_vec)
        memo[id(self)] = new_shift

        return new_shift

    def __repr__(self) -> str:
        return f"OptimizedShift(uuid={self.uuid}, value={self.value}, num_registered_pcds={len(self)})"

    @property
    def uuid(self) -> uuid.UUID:
        return self._uuid

    @property
    def value(self) -> Vector_3_T:
        return self._shift

    @value.setter
    @validate_variables
    def value(self, new_shift: Vector_3_T) -> None:
        # check if new shift can be used for registered points
        all_bboxes = [member.unshifted_bbox for member in self._member_coordinate_sets]
        if all_bboxes:
            combined = MinMaxPoints.from_minmax_points(all_bboxes)

            if OptimizedShiftManager().is_shift_needed(np.vstack((combined.minimum, combined.maximum)) - new_shift):
                raise OptimizedShiftManager.ShiftNotFeasibleError(
                    f"The provided shift {new_shift} is not feasible for the coordinate sets registered to this shift."
                )

        self._compute_and_apply_shift_delta(new_shift)
        new_uuid = uuid.uuid4()
        OptimizedShiftManager().update_uuid(self._uuid, new_uuid)
        self._uuid = new_uuid
        self._shift = new_shift

        logger.debug(f"Updated shift to {new_shift}. New uuid: {new_uuid}.")

    @property
    def __array_interface__(self):
        return self._shift.__array_interface__

    def register(self, coordinate_set: CartesianCoordinates) -> None:
        """
        Try to add `pcd` (with its point‐cloud `points`) into this shift.
        Returns self if successful;
        returns a brand‐new OptimizedShift if this set can’t fit but we had others;
        or None if even a singleton can’t fit.
        """
        if coordinate_set in self:
            return
        # TODO: Think how to check already shifted coordinates

        unshifted_bbox = coordinate_set.unshifted_bbox

        if self._can_add_without_change(np.array(unshifted_bbox)):
            self._add_member(coordinate_set)
            return

        try:
            self._expand_and_add(coordinate_set)
            return
        except OptimizedShiftManager.ShiftNotFeasibleError:
            raise OptimizedShiftManager.ShiftNotFeasibleError()

    def unregister(self, coordinate_set: CartesianCoordinates) -> None:
        if coordinate_set not in self:
            return
        for wr in list(self._member_coordinate_sets.data):  # type: ignore[attr-defined]
            if wr() is coordinate_set:
                self._member_coordinate_sets.data.discard(wr)  # type: ignore[attr-defined]
                break

    def check_addibility(self, unshifted_pts: Array_Nx3_T) -> bool:
        # TODO: Check usage [can points be shifted and unshifted]
        try:
            _ = self._compute_new_shift(unshifted_pts)
            return True
        except OptimizedShiftManager.ShiftNotFeasibleError:
            return False

    def _can_add_without_change(self, unshifted_pts: Array_Nx3_T) -> bool:
        # TODO: Check usage [can points be shifted and unshifted]
        shifted = np.subtract(unshifted_pts, self._shift)
        return not OptimizedShiftManager().is_shift_needed(shifted)

    def _add_member(self, coordinate_set: CartesianCoordinates) -> None:
        """Add the point‐cloud under the existing shift."""
        self._member_coordinate_sets.add(coordinate_set)
        # self._member_coordinate_sets_unshifted_bbox[coordinate_set] = MinMaxPoints.from_points(coordinate_set.arr)

    def _expand_and_add(self, coordinate_set: CartesianCoordinates) -> None:
        """
        Compute a new optimal shift to cover existing coordinate sets and the new pts,
        apply it, generate new uuid, and then add the new pcd.
        """
        new_shift = self._compute_new_shift(coordinate_set.unshifted_bbox)
        new_uuid = uuid.uuid4()
        OptimizedShiftManager().update_uuid(self._uuid, new_uuid)
        logger.debug(f"Updated shift to {new_shift}. New uuid: {new_uuid}.")
        self._uuid = new_uuid
        self._compute_and_apply_shift_delta(new_shift)
        self._shift = new_shift
        self._add_member(coordinate_set)

    def _compute_new_shift(self, additional_bbox: Optional[MinMaxPoints | Array_Nx3_T] = None) -> Vector_3_T:
        # build up the combined bounding‐box
        all_bboxes = [member.unshifted_bbox for member in self._member_coordinate_sets]
        if additional_bbox is not None:
            all_bboxes.append(additional_bbox)
        combined = MinMaxPoints.from_minmax_points(all_bboxes)

        if not OptimizedShiftManager().is_shift_possible(np.vstack((combined.minimum, combined.maximum))):
            raise OptimizedShiftManager.ShiftNotFeasibleError()

        # round the center to keep ints
        return np.round(combined.central_point, decimals=-2)

    def _compute_and_apply_shift_delta(self, new_shift: Vector_3_T):
        """Move every registered coordinate set by the change from old to new shift."""
        delta = self._shift - new_shift
        for pcd in self._member_coordinate_sets:
            pcd.update_shift(delta)

    @staticmethod
    def _reconstruct(u: UUID, shift_vec: Vector_3_T) -> "OptimizedShift":
        mgr = OptimizedShiftManager()
        existing = mgr.get_by_uuid(u)

        if existing is not None:
            return existing

        # TODO implement the creation of a new shift object if ever there's a difference with the existing one
        new = object.__new__(OptimizedShift)
        new._uuid = u
        new._shift = shift_vec
        new._member_coordinate_sets = weakref.WeakSet()
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
        # self._member_coordinate_sets_unshifted_bbox[coordinate_set] = (
        #     MinMaxPoints.from_points(coordinate_set.xyz, already_applied_shift_vec = self._shift)
        # )
