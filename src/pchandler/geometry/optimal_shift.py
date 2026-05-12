# pchandler – Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2022–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""Numerical-precision shift management for Cartesian coordinates."""

from __future__ import annotations

import copy
import logging
import uuid
import weakref
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

import numpy as np
from GSEGUtils.base_types import Array_Nx3_T, Vector_3_T
from GSEGUtils.constants import validate_variables
from GSEGUtils.singleton import SingletonMeta

from pchandler.geometry.util import MinMaxPoints

if TYPE_CHECKING:
    from pchandler.geometry.coordinates import CartesianCoordinates

__all__ = ["OptimizedShiftManager", "OptimizedShift"]

logger = logging.getLogger(__name__)


class OptimizedShiftManager(metaclass=SingletonMeta):
    """Manager for optimized shifts.

    This class allows registration, retrieval, and management of shifts
    while maintaining a global configuration for numerical precision.

    Parameters
    ----------
    FLOAT32_DECIMAL_PRECISION : int
        Default decimal precision for floating point calculations.
    """

    _by_uuid: weakref.WeakValueDictionary[uuid.UUID, OptimizedShift]
    _minimum_decimal_places: int
    FLOAT32_DECIMAL_PRECISION: int = 7

    class ShiftNotFeasibleError(Exception):
        """Raised when no feasible shift exists for the requested coordinate set."""

    class ShiftUUIDAlreadyTaken(Exception):
        """Raised when registering a UUID that is already known to the manager."""

    class ShiftUUIDNotFound(Exception):
        """Raised when retrieving an unknown UUID from the manager."""

    def __init__(self, minimum_decimal_places: int = 3) -> None:
        """Initialize the manager singleton.

        Parameters
        ----------
        minimum_decimal_places : int, optional
            The minimum number of decimal places to be used. Default is 3.
        """
        self._minimum_decimal_places = minimum_decimal_places
        self._by_uuid = weakref.WeakValueDictionary()

    def __len__(self):
        """Return the number of registered shifts."""
        return len(self._by_uuid)

    @property
    def all_shifts(self) -> list[OptimizedShift]:
        """Return all currently-registered optimized shifts.

        Returns
        -------
        list[OptimizedShift]
            All shifts currently held by the manager (weak-ref values only).
        """
        return list(self._by_uuid.values())

    @property
    def minimum_decimal_places(self) -> int:
        """Return the minimum number of decimal places supported by the manager.

        Returns
        -------
        int
            Configured minimum decimal places.
        """
        return self._minimum_decimal_places

    @minimum_decimal_places.setter
    def minimum_decimal_places(self, value: int) -> None:  # TODO input value name correct?
        """Set the minimum number of decimal places supported by the manager.

        Parameters
        ----------
        value : int
            New minimum decimal places.
        """
        pass  # Todo: Should probably check all registered GlobalShifts and their pcds if they conform to new limit

    @property
    def maximum_number_representable(self) -> float:
        """Return the maximum number representable with the configured float precision.

        Returns
        -------
        float
            ``10 ** (FLOAT32_DECIMAL_PRECISION - minimum_decimal_places)``.
        """
        return 10 ** (self.FLOAT32_DECIMAL_PRECISION - self.minimum_decimal_places)

    def get_by_uuid(self, u: UUID) -> OptimizedShift | None:
        """Retrieve an :class:`OptimizedShift` by its UUID.

        Parameters
        ----------
        u : UUID
            UUID of the shift to retrieve.

        Returns
        -------
        OptimizedShift | None
            The matching shift, or ``None`` if no shift is registered for ``u``.
        """
        return self._by_uuid.get(u)

    def is_shift_needed(self, values: Array_Nx3_T) -> bool:
        """Check whether ``values`` exceed the maximum number representable in float32.

        Parameters
        ----------
        values : Array_Nx3_T
            Candidate coordinates to evaluate.

        Returns
        -------
        bool
            ``True`` if at least one component would lose precision without a shift.
        """
        return bool(np.any(np.abs(values) >= self.maximum_number_representable))

    def is_shift_possible(self, values: Array_Nx3_T) -> bool:
        """Check whether the range of ``values`` is within the representable limit.

        Parameters
        ----------
        values : Array_Nx3_T
            Candidate coordinates to evaluate.

        Returns
        -------
        bool
            ``True`` if a finite shift can bring all values inside the float32
            representable range.
        """
        return bool(
            np.all(np.subtract(np.max(values, axis=0), np.min(values, axis=0)) < self.maximum_number_representable)
        )

    def update_uuid(self, old_uuid: uuid.UUID, new_uuid: uuid.UUID) -> None:
        """Replace a registered shift's UUID with a new one.

        An error is raised if the old UUID does not exist, or if the new UUID
        is already in use.

        Parameters
        ----------
        old_uuid : uuid.UUID
            The UUID of the existing shift to be updated.
        new_uuid : uuid.UUID
            The new UUID to assign to the shift.

        Raises
        ------
        OptimizedShiftManager.ShiftUUIDNotFound
            If ``old_uuid`` is not registered.
        OptimizedShiftManager.ShiftUUIDAlreadyTaken
            If ``new_uuid`` is already registered.
        """
        if old_uuid not in self._by_uuid:
            raise OptimizedShiftManager.ShiftUUIDNotFound(f"Shift with uuid: {old_uuid} not found.")
        if new_uuid in self._by_uuid:
            raise OptimizedShiftManager.ShiftUUIDAlreadyTaken(f"Shift with uuid: {new_uuid} already exists.")
        self._by_uuid[new_uuid] = self._by_uuid.pop(old_uuid)

    def register_shift(self, shift: OptimizedShift) -> None:
        """Register a shift in the manager.

        If a shift with the same UUID already exists but refers to a
        different object instance, a :class:`ShiftUUIDAlreadyTaken` is raised.

        Parameters
        ----------
        shift : OptimizedShift
            Shift to register.

        Raises
        ------
        OptimizedShiftManager.ShiftUUIDAlreadyTaken
            Shift with the same UUID already exists in the manager.
        """
        if shift.uuid in self._by_uuid and id(shift) is not id(self._by_uuid[shift.uuid]):
            raise OptimizedShiftManager.ShiftUUIDAlreadyTaken()

        self._by_uuid[shift.uuid] = shift

    def register_coordinates_to_shift(
        self, coordinates: CartesianCoordinates, shift: OptimizedShift | uuid.UUID
    ) -> OptimizedShift:
        """Register coordinates to an existing optimized shift.

        Associates the given coordinates with an :class:`OptimizedShift`. If
        the registration fails, this method attempts to use a fresh
        :class:`OptimizedShift` before raising an error.

        Parameters
        ----------
        coordinates : CartesianCoordinates
            Coordinate set to register.
        shift : OptimizedShift or uuid.UUID
            Existing shift (or its UUID) to attempt registration with.

        Returns
        -------
        OptimizedShift
            The shift the coordinates were ultimately registered to (may be a
            new shift if the input one could not absorb them).

        Raises
        ------
        OptimizedShiftManager.ShiftUUIDNotFound
            UUID not found in the manager.
        OptimizedShiftManager.ShiftNotFeasibleError
            Shift not feasible for the given coordinates.
        """
        if isinstance(shift, OptimizedShift):
            current_shift = shift
        else:
            try:
                current_shift: OptimizedShift = self._by_uuid[shift]
            except KeyError as err:
                raise OptimizedShiftManager.ShiftUUIDNotFound() from err

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
    """A numerical-precision shift shared by one or more coordinate sets.

    Holds the translation vector and weak references to the
    :class:`CartesianCoordinates` instances linked to it.

    Parameters
    ----------
    shift_vec : Vector_3_T, optional
        A vector representing the shift. Defaults to ``(0, 0, 0)``.
    """

    _uuid: uuid.UUID
    _shift: Vector_3_T
    _member_coordinate_sets: weakref.WeakSet[CartesianCoordinates]
    # _member_coordinate_sets_unshifted_bbox: weakref.WeakKeyDictionary[CartesianCoordinates, MinMaxPoints]

    @validate_variables
    def __init__(self, shift_vec: Optional[Vector_3_T] = None) -> None:
        """Initialize an :class:`OptimizedShift`.

        Parameters
        ----------
        shift_vec : Vector_3_T, optional
            A vector representing the shift. If not provided, defaults to
            ``(0, 0, 0)``.
        """
        self._uuid = uuid.uuid4()
        self._shift = np.zeros((3,), dtype=np.float64) if shift_vec is None else shift_vec
        self._member_coordinate_sets = weakref.WeakSet()
        # self._member_coordinate_sets_unshifted_bbox = weakref.WeakKeyDictionary()
        OptimizedShiftManager().register_shift(self)

    def __contains__(self, value: CartesianCoordinates) -> bool:
        """Return ``True`` if ``value`` is one of the registered coordinate sets."""
        return any(value is pcd for pcd in self._member_coordinate_sets)

    def __len__(self):
        """Return the number of coordinate sets currently registered to this shift."""
        return len(self._member_coordinate_sets)

    def __hash__(self) -> int:
        """Return a hash based on the shift's UUID."""
        return hash(self._uuid)

    def __eq__(self, other) -> bool:
        """Two shifts compare equal when their UUIDs are equal."""
        return self.uuid == other.uuid

    def __reduce__(self):
        """Return the (callable, state) tuple used by :mod:`pickle` to reconstruct ``self``."""
        return self._reconstruct, (self._uuid, self._shift)

    def __deepcopy__(self, memo):
        """Deep-copy returns a fresh :class:`OptimizedShift` carrying a copy of the shift vector."""
        # Construct new
        new_vec = copy.deepcopy(self._shift, memo)
        new_shift = type(self)(new_vec)
        memo[id(self)] = new_shift

        return new_shift

    def __repr__(self) -> str:
        """Return a debug-friendly representation of the shift."""
        return f"OptimizedShift(uuid={self.uuid}, value={self.value}, num_registered_pcds={len(self)})"

    @property
    def uuid(self) -> uuid.UUID:
        """Return the shift's UUID.

        Returns
        -------
        uuid.UUID
            The shift's stable UUID.
        """
        return self._uuid

    @property
    def value(self) -> Vector_3_T:
        """Return the current 3D shift vector.

        Returns
        -------
        Vector_3_T
            The translation vector applied by this shift.
        """
        return self._shift

    @value.setter
    @validate_variables
    def value(self, new_shift: Vector_3_T) -> None:
        """Set a new shift vector, recomputing all registered coordinate sets.

        Checks whether the provided new shift is feasible for the registered
        coordinate sets. If feasible, computes and applies the shift delta to
        all linked coordinate sets.

        Parameters
        ----------
        new_shift : Vector_3_T
            New shift vector.

        Raises
        ------
        OptimizedShiftManager.ShiftNotFeasibleError
            If ``new_shift`` cannot be applied to all registered coordinate sets.
        """
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
    def __array_interface__(self) -> dict[str, Any]:
        """Expose the shift vector as a NumPy ``__array_interface__``.

        Returns
        -------
        dict[str, Any]
            The underlying shift vector's array-interface descriptor.
        """
        return self._shift.__array_interface__

    def register(self, coordinate_set: CartesianCoordinates) -> None:
        """Register a :class:`CartesianCoordinates` instance against this shift.

        Tries to add ``coordinate_set`` to this shift. If the coordinates fit
        without modification, they are added directly; otherwise the method
        attempts to expand the shift via :meth:`_expand_and_add`. If even that
        fails, a :class:`ShiftNotFeasibleError` is raised.

        Parameters
        ----------
        coordinate_set : CartesianCoordinates
            The CartesianCoordinates object to register.

        Raises
        ------
        OptimizedShiftManager.ShiftNotFeasibleError
            If the coordinates cannot fit into this shift and cannot be made
            to fit by expansion.
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
        except OptimizedShiftManager.ShiftNotFeasibleError as err:
            raise OptimizedShiftManager.ShiftNotFeasibleError() from err

    def unregister(self, coordinate_set: CartesianCoordinates) -> None:
        """Remove a coordinate set from this shift.

        Parameters
        ----------
        coordinate_set : CartesianCoordinates
            Coordinate set to unregister; no-op if not currently registered.
        """
        if coordinate_set not in self:
            return
        for wr in list(self._member_coordinate_sets.data):  # type: ignore[attr-defined]
            if wr() is coordinate_set:
                self._member_coordinate_sets.data.discard(wr)  # type: ignore[attr-defined]
                break

    def check_addibility(self, unshifted_pts: Array_Nx3_T) -> bool:
        """Check whether ``unshifted_pts`` can be absorbed into this shift.

        Parameters
        ----------
        unshifted_pts : Array_Nx3_T
            Candidate points (in their original, unshifted frame).

        Returns
        -------
        bool
            ``True`` if a feasible shift can absorb the points.
        """
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
        """Attempt to expand the current optimised shift to include the new coordinate set.

        If so, it is applied, and a new UUID is generated.

        Parameters
        ----------
        coordinate_set : CartesianCoordinates
        """
        # Compute shift based on new coords
        new_shift = self._compute_new_shift(coordinate_set.unshifted_bbox)

        # Set a new UUID
        new_uuid = uuid.uuid4()
        OptimizedShiftManager().update_uuid(self._uuid, new_uuid)
        logger.debug(f"Updated shift to {new_shift}. New uuid: {new_uuid}.")

        # Apply shift delta to linked point clouds
        self._uuid = new_uuid
        self._compute_and_apply_shift_delta(new_shift)
        self._shift = new_shift
        self._add_member(coordinate_set)

    def _compute_new_shift(self, additional_bbox: Optional[MinMaxPoints | Array_Nx3_T] = None) -> Vector_3_T:
        """Compute a new shift vector covering all registered members plus an optional extra bbox.

        Parameters
        ----------
        additional_bbox : MinMaxPoints or Array_Nx3_T, optional
            An additional bounding box to include in the calculation.

        Returns
        -------
        Vector_3_T
            New shift vector (rounded to whole hundreds).

        Raises
        ------
        OptimizedShiftManager.ShiftNotFeasibleError
            If no feasible shift exists for the combined coordinate range.
        """
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
        """Move all registered coordinate sets by the delta between the old and the new shift.

        Parameters
        ----------
        new_shift : Vector_3_T
            New shift vector to apply.
        """
        delta = self._shift - new_shift
        for pcd in self._member_coordinate_sets:
            pcd.update_shift(delta)

    @staticmethod
    def _reconstruct(u: UUID, shift_vec: Vector_3_T) -> "OptimizedShift":
        """Reconstruct an :class:`OptimizedShift` from previously-pickled state.

        Parameters
        ----------
        u : UUID
            UUID of the shift.
        shift_vec : Vector_3_T
            Shift vector.

        Returns
        -------
        OptimizedShift
            Either the existing manager-held shift with matching UUID, or a
            freshly-constructed shift registered with the manager.
        """
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
        """Reattach a coordinate set to this shift without changing the other members.

        ``unshifted_pts`` should be the original ``float64`` points before
        shifting.

        Parameters
        ----------
        coordinate_set : CartesianCoordinates
            Coordinate set to reattach.
        """
        # add to the weakref set
        self._member_coordinate_sets.add(coordinate_set)
        # store its unshifted bounding box
        # self._member_coordinate_sets_unshifted_bbox[coordinate_set] = (
        #     MinMaxPoints.from_points(coordinate_set.xyz, already_applied_shift_vec = self._shift)
        # )
