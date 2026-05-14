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
from typing import TYPE_CHECKING, Any, NewType, Optional
from uuid import UUID

import numpy as np
import numpy.typing as npt
from GSEGUtils.base_types import Array_Nx3_T, Vector_3_T
from GSEGUtils.constants import validate_variables
from GSEGUtils.singleton import SingletonMeta

from pchandler.geometry.util import MinMaxPoints

if TYPE_CHECKING:
    from pchandler.geometry.coordinates import CartesianCoordinates

# BUG-06 type tags (Phase 3 D-09 / D-28):
#   All public-method inputs on OptimizedShift / OptimizedShiftManager that
#   take coordinate arrays SHOULD annotate as ``Unshifted`` (world-frame,
#   caller's original frame). Internal ``_``-prefixed helpers may take
#   ``Shifted`` (after subtracting ``self._shift``); the method's parameter
#   annotation names the convention.
#
# The NewType is over ``npt.NDArray[np.float64]`` (NOT numpydantic's
# ``Array_Nx3_T``) because mypy strict rejects ``NewType(name, Array_Nx3_T)``
# as "must be subclassable" (RESEARCH OPEN-Q2 + CONTEXT D-28 ratification).
# Runtime shape/dtype validation continues via numpydantic on the ``_shift``
# field and ``validate_variables`` on ``__init__`` / ``value.setter``.
_CoordsF = npt.NDArray[np.float64]
Unshifted = NewType("Unshifted", _CoordsF)
Shifted = NewType("Shifted", _CoordsF)

__all__ = ["OptimizedShiftManager", "OptimizedShift", "Unshifted", "Shifted"]

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
        # WR-02 (Phase 3 code review): the previous form
        # ``id(shift) is not id(self._by_uuid[shift.uuid])`` compared two
        # freshly-minted ``int`` objects with ``is``. Since CPython only
        # interns ints in [-5, 256], two ``id()`` calls on real memory
        # addresses almost always satisfy ``is not`` — the comparison was
        # essentially "always True" and would incorrectly reject a no-op
        # re-register of the same instance. ``object is object`` is the
        # unambiguous identity check.
        if shift.uuid in self._by_uuid and self._by_uuid[shift.uuid] is not shift:
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

    All public methods on this class take ``Unshifted`` (world-frame,
    caller's original frame) coordinate arrays. Internal ``_``-prefixed
    helpers may take ``Shifted`` coordinates (after subtracting
    ``self._shift``), and name the convention in their parameter
    annotation. ``Unshifted`` / ``Shifted`` are :func:`typing.NewType`
    aliases over ``npt.NDArray[np.float64]``; the tag is a static-check
    signal only. Runtime shape/dtype validation is delegated to
    numpydantic on the ``_shift`` field and to ``validate_variables`` on
    public-method inputs.

    Parameters
    ----------
    shift_vec : Vector_3_T, optional
        A vector representing the shift. Defaults to ``(0, 0, 0)``.
    minimum_decimal_places : int, optional
        Per-instance precision floor. When ``None`` (default), falls back to
        :attr:`OptimizedShiftManager.minimum_decimal_places`. Settable
        post-init via the validating setter on the same property.
    """

    _uuid: uuid.UUID
    _shift: Vector_3_T
    _member_coordinate_sets: weakref.WeakSet[CartesianCoordinates]
    _minimum_decimal_places: Optional[int] = None

    @validate_variables
    def __init__(
        self,
        shift_vec: Optional[Vector_3_T] = None,
        *,
        minimum_decimal_places: Optional[int] = None,
    ) -> None:
        """Initialize an :class:`OptimizedShift`.

        Parameters
        ----------
        shift_vec : Vector_3_T, optional
            A vector representing the shift. If not provided, defaults to
            ``(0, 0, 0)``.
        minimum_decimal_places : int, optional
            Per-instance precision floor. When ``None`` (default), falls back
            to :attr:`OptimizedShiftManager.minimum_decimal_places`. Settable
            post-init via the validating setter on the same property.
        """
        self._uuid = uuid.uuid4()
        self._shift = np.zeros((3,), dtype=np.float64) if shift_vec is None else shift_vec
        self._minimum_decimal_places = minimum_decimal_places
        self._member_coordinate_sets = weakref.WeakSet()
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

    def __eq__(self, other: object) -> bool:
        """Two shifts compare equal when their UUIDs are equal.

        WR-04 (Phase 3 code review): return :data:`NotImplemented` (which the
        Python runtime treats as "fall back to the other operand's
        ``__eq__`` or default to identity") on a non-:class:`OptimizedShift`
        operand rather than raising :class:`AttributeError`. ``__eq__`` is
        called by hash-set deduplication and by ``Optional[OptimizedShift]``
        comparisons throughout the merge / un-shift surfaces, where ``None``
        is a routine operand; the previous form crashed on the first ``None``.
        """
        if not isinstance(other, OptimizedShift):
            return NotImplemented
        return self.uuid == other.uuid

    def __reduce__(self):
        """Return the (callable, state) tuple used by :mod:`pickle` to reconstruct ``self``."""
        return self._reconstruct, (self._uuid, self._shift)

    def __deepcopy__(self, memo: dict) -> OptimizedShift:
        """Deep-copy returns a fresh :class:`OptimizedShift` carrying a copy of the shift vector.

        The copy gets its own UUID and is registered with :class:`OptimizedShiftManager`
        as an independent shift.  Member coordinate sets are *not* copied because they
        are weak references to live objects — the caller is responsible for re-linking
        any :class:`CartesianCoordinates` instances that should belong to the copy.

        The ``memo`` dict is populated *before* deepcopying sub-objects to prevent
        infinite recursion if there are circular references back to ``self``.
        """
        # Allocate via __new__ so we can populate memo before __init__ triggers
        # any recursive deepcopy calls.
        new_shift: OptimizedShift = object.__new__(type(self))
        memo[id(self)] = new_shift

        # Now safe to deepcopy the scalar shift vector (no cycles expected but memo
        # is already set, so a cycle would resolve correctly).
        new_shift._shift = copy.deepcopy(self._shift, memo)
        new_shift._uuid = uuid.uuid4()
        new_shift._minimum_decimal_places = self._minimum_decimal_places
        new_shift._member_coordinate_sets = weakref.WeakSet()
        OptimizedShiftManager().register_shift(new_shift)

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

            if self._is_shift_needed(np.vstack((combined.minimum, combined.maximum)) - new_shift):
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
    def minimum_decimal_places(self) -> int:
        """Per-instance precision threshold; falls through to the manager default when unset.

        Returns
        -------
        int
            This shift's configured minimum decimal places, or the
            :class:`OptimizedShiftManager` default if not overridden.
        """
        if self._minimum_decimal_places is not None:
            return self._minimum_decimal_places
        return OptimizedShiftManager().minimum_decimal_places

    @minimum_decimal_places.setter
    def minimum_decimal_places(self, value: int) -> None:
        """Set the per-instance precision floor.

        Walks ``self._member_coordinate_sets``; if any registered set would
        become infeasible under the new precision (via
        ``self._is_shift_possible`` — Phase 3 D-08), raises
        :class:`OptimizedShiftManager.ShiftNotFeasibleError` listing the
        offending coordinate-set UUIDs.

        Reverts ``self._minimum_decimal_places`` to the previous value **only**
        if ``_is_shift_possible`` flags an infeasible coord set
        (:class:`OptimizedShiftManager.ShiftNotFeasibleError`). Other
        exceptions propagate without revert so the caller sees the real
        root cause.

        Notes
        -----
        Iteration uses a list-snapshot of ``self._member_coordinate_sets`` to
        guard against WeakSet mutation under concurrent GC
        (RESEARCH §"Pitfall 5").
        """
        old_value = self._minimum_decimal_places
        self._minimum_decimal_places = value
        try:
            infeasible: list[str] = []
            for coord_set in list(self._member_coordinate_sets):
                if not self._is_shift_possible(coord_set.xyz):
                    cs_id = getattr(coord_set, "_uuid", None) or repr(coord_set)
                    infeasible.append(str(cs_id))
            if infeasible:
                raise OptimizedShiftManager.ShiftNotFeasibleError(
                    f"Setting minimum_decimal_places={value} would make registered coordinate sets "
                    f"infeasible: {infeasible}"
                )
        except OptimizedShiftManager.ShiftNotFeasibleError:
            self._minimum_decimal_places = old_value
            raise

    @property
    def maximum_number_representable(self) -> float:
        """Maximum number representable in float32 given this instance's threshold.

        Returns
        -------
        float
            ``10 ** (FLOAT32_DECIMAL_PRECISION - self.minimum_decimal_places)``.
        """
        return 10 ** (OptimizedShiftManager.FLOAT32_DECIMAL_PRECISION - self.minimum_decimal_places)

    def _is_shift_needed(self, values: Unshifted) -> bool:
        """Check whether ``values`` exceed this instance's representable range.

        Parameters
        ----------
        values : Unshifted
            Candidate world-frame coordinates to evaluate against this
            shift's threshold.

        Returns
        -------
        bool
            ``True`` if at least one component would lose precision without a shift.
        """
        return bool(np.any(np.abs(values) >= self.maximum_number_representable))

    def _is_shift_possible(self, values: Unshifted) -> bool:
        """Check whether the range of ``values`` fits this instance's representable span.

        Parameters
        ----------
        values : Unshifted
            Candidate world-frame coordinates to evaluate against this
            shift's threshold.

        Returns
        -------
        bool
            ``True`` if a finite shift can bring all values inside the
            float32 representable range for this instance.
        """
        return bool(
            np.all(np.subtract(np.max(values, axis=0), np.min(values, axis=0)) < self.maximum_number_representable)
        )

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

        Notes
        -----
        WR-05 (Phase 3 code review): we deliberately do NOT use
        :meth:`weakref.WeakSet.discard` / :meth:`~weakref.WeakSet.remove` /
        :meth:`~weakref.WeakSet.__contains__` here because internally those
        call ``set.discard(weakref.ref(item))``, and ``set`` lookups confirm
        bucket membership via ``__eq__``. :class:`CartesianCoordinates`
        inherits its ``__eq__`` from :class:`numpy.ndarray` (via numpydantic
        / :class:`GSEGUtils.base_arrays.BaseArray`), which returns an array
        rather than a scalar — every public ``WeakSet`` method that touches
        ``__eq__`` therefore raises :class:`ValueError`
        ("truth value of an array with more than one element is ambiguous").

        The identity-based manual loop below is the workaround: iterate the
        underlying ``set[weakref.ref]`` and match referents via ``is`` (object
        identity), bypassing ``__eq__`` entirely. The ``type: ignore`` annotates
        the documented CPython attribute access; ``__contains__`` on
        :class:`OptimizedShift` itself is also identity-based for the same
        reason (see :meth:`__contains__`).
        """
        for wr in list(self._member_coordinate_sets.data):  # type: ignore[attr-defined]
            if wr() is coordinate_set:
                self._member_coordinate_sets.data.discard(wr)  # type: ignore[attr-defined]
                break

    def check_addibility(self, unshifted_pts: Unshifted) -> bool:
        """Check whether ``unshifted_pts`` can be absorbed into this shift.

        Parameters
        ----------
        unshifted_pts : Unshifted
            Candidate points in their original, world-frame (un-shifted)
            frame.

        Returns
        -------
        bool
            ``True`` if a feasible shift can absorb the points.
        """
        try:
            _ = self._compute_new_shift(unshifted_pts)
            return True
        except OptimizedShiftManager.ShiftNotFeasibleError:
            return False

    def _can_add_without_change(self, unshifted_pts: Unshifted) -> bool:
        """Return ``True`` if ``unshifted_pts`` fits this shift unchanged.

        Parameters
        ----------
        unshifted_pts : Unshifted
            Candidate points in their world-frame (un-shifted) frame.
            ``self._shift`` is subtracted internally to produce a local
            ``Shifted`` array before the feasibility check.
        """
        shifted = np.subtract(unshifted_pts, self._shift)
        # Cast: subtracting self._shift puts the points in this shift's
        # local frame; _is_shift_needed semantically operates on Unshifted
        # arrays (representable-range check), and the same range test is
        # valid for the shifted local frame here.
        return not self._is_shift_needed(shifted)

    def _add_member(self, coordinate_set: CartesianCoordinates) -> None:
        """Add the point‐cloud under the existing shift."""
        self._member_coordinate_sets.add(coordinate_set)

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

    def _compute_new_shift(self, additional_bbox: Optional[MinMaxPoints | Unshifted] = None) -> Vector_3_T:
        """Compute a new shift vector covering all registered members plus an optional extra bbox.

        Parameters
        ----------
        additional_bbox : MinMaxPoints or Unshifted, optional
            An additional bounding box (or world-frame coordinate array) to
            include in the calculation.

        Returns
        -------
        Vector_3_T
            New shift vector (rounded to whole hundreds). This is a
            translation offset in world-frame coordinates, NOT an
            ``Unshifted`` or ``Shifted`` coordinate array — kept as
            ``Vector_3_T`` per RESEARCH Open Issue #1.

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

        if not self._is_shift_possible(np.vstack((combined.minimum, combined.maximum))):
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
    def _construct_with_uuid(u: UUID, shift_vec: Vector_3_T) -> "OptimizedShift":
        """Build a fresh OptimizedShift carrying ``u`` + ``shift_vec`` and register it.

        Used by :meth:`_reconstruct` only — for the bypass-``__init__`` pickle
        reconstruction path. Normal construction goes via :meth:`__init__` and
        never routes through this helper.

        Parameters
        ----------
        u : UUID
            The UUID to assign to the new shift. Caller is responsible for
            ensuring no collision in the manager registry (either passed-from-
            pickled-state when the manager has no entry under ``u``, or a
            freshly-minted ``uuid.uuid4()`` on the mint-new-UUID branch).
        shift_vec : Vector_3_T
            The shift vector from the pickled state.

        Returns
        -------
        OptimizedShift
            A new shift, already registered with the singleton manager.
        """
        new = object.__new__(OptimizedShift)
        new._uuid = u
        new._shift = shift_vec
        new._member_coordinate_sets = weakref.WeakSet()
        OptimizedShiftManager().register_shift(new)
        return new

    @staticmethod
    def _reconstruct(u: UUID, shift_vec: Vector_3_T) -> "OptimizedShift":
        """Reconstruct an :class:`OptimizedShift` from previously-pickled state.

        The destination-process :class:`OptimizedShiftManager` may already hold
        a shift under UUID ``u``. Three cases (FRAG-01 / Phase 3 D-18):

        1. **No existing shift under** ``u``: construct + register with the
           pickled UUID. Cross-pickle UUID continuity is preserved.
        2. **Existing shift, same vector**: return the existing instance. The
           pickled stream's references collapse onto the destination instance.
        3. **Existing shift, divergent vector**: mint a fresh ``uuid.uuid4()``
           and register a NEW shift carrying the pickled ``shift_vec``. The
           pre-existing shift under ``u`` is left untouched. World-frame
           coordinates are preserved at the cost of UUID continuity; cross-PCD
           merge on the destination now requires explicit reframing.

        Parameters
        ----------
        u : UUID
            UUID from the pickled state.
        shift_vec : Vector_3_T
            Shift vector from the pickled state.

        Returns
        -------
        OptimizedShift
            Either the pre-existing manager instance (case 2) or a freshly-
            constructed instance routed through :meth:`_construct_with_uuid`
            (cases 1 and 3).
        """
        mgr = OptimizedShiftManager()
        existing = mgr.get_by_uuid(u)
        if existing is not None:
            if np.array_equal(existing._shift, shift_vec):
                return existing
            # Case 3: divergent vector → mint fresh UUID
            old_u = u
            u = uuid.uuid4()
            logger.info(
                "OptimizedShift._reconstruct: UUID collision with divergent vector "
                "(old UUID %s); minted fresh UUID %s for the pickled shift vector. "
                "World-frame coordinates preserved; cross-PCD merge requires "
                "explicit reframing.",
                old_u,
                u,
            )
            logger.debug(
                "shift collision detail: existing shift vector=%s, incoming=%s, magnitude diff=%.6f",
                existing._shift,
                shift_vec,
                float(np.linalg.norm(existing._shift - shift_vec)),
            )
        return OptimizedShift._construct_with_uuid(u, shift_vec)

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
