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

"""Fields of View (FoVs) and hierarchical FoV trees for spherical-coordinate spatial partitioning.

This module provides classes and methods for defining and manipulating Fields of View (FoVs) and hierarchical FoV trees.
It is designed to facilitate the spatial partitioning, tiling, and merging of 3D regions based on angular constraints.
The module supports flexible representation of angular units and integrates with external tools to enable hierarchical
partitioning of FoVs.

Key Features:
-------------
- **FoV Class**:
  - Represents rectangular angular regions in 3D space.
  - Supports unit conversion between radians, degrees, and gradians (gon).
  - Provides methods for splitting, merging, and calculating geometric properties such as aspect ratios and centers.

- **FoVTree Class**:
  - Implements a hierarchical tree structure for managing FoVs.
  - Enables efficient spatial partitioning, depth-based querying, and merging operations.
  - Compatible with tile-based FoV organization for large-scale datasets.

- **Utility Methods**:
  - Split a single FoV into multiple tiles or quadrants.
  - Convert between tuple, dictionary, or NumPy array representations of FoV boundaries.
  - Calculate optimal partitioning schemes for FoVs based on aspect ratios and angular extents.

Dependencies:
-------------
- ``numpy``: For numerical computations.
- ``pchandler.util``: Provides utilities for angle unit conversion and numerical constants.

Usage:
------
Example: Create an FoV and convert it between different representations:

.. code-block:: python

    from pchandler.fov import FoV

    # Define a field of view in degrees
    fov = FoV(horizontal_min=0, horizontal_max=90, elevation_min=-30, elevation_max=30, unit="deg")

    # Convert to radians
    fov_rad = fov.as_tuple(unit="rad")
    print("FoV in radians:", fov_rad)

    # Split the FoV into a 2x2 grid
    sub_fovs = fov.split(shape=(2, 2))
    print("Sub-FoVs:", sub_fovs)


Example: Use a hierarchical FoV tree for spatial partitioning:

.. code-block:: python

    from pchandler.fov import FoV, FoVTree

    # Create a base FoV
    base_fov = FoV(horizontal_min=0, horizontal_max=90, elevation_min=-30, elevation_max=30, unit="deg")

    # Split into tiles and build a tree
    tiles = base_fov.tile(FoV(horizontal_min=0, horizontal_max=30, elevation_min=-10, elevation_max=10))
    fov_tree = FoVTree.build_from_tiles(tiles)

    # Query the depth of the tree
    print("Tree depth:", fov_tree.depth())
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import chain
from typing import Any, Generator, Iterable, Optional, Self, TypeAlias, cast

import numpy as np
from GSEGUtils.base_types import Vector_Bool_T, Vector_Float_T, VectorT
from GSEGUtils.constants import DEFAULT_CONFIG, EPS, PI, TWO_PI, validate_variables
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    field_validator,
    model_validator,
    validate_call,
)

from pchandler.geometry.spherical import Angle, AngleArray

__all__ = ["FoV", "FoVTree"]

logger = logging.getLogger(__name__.split(".")[0])

AngleLikeT: TypeAlias = Angle | float | str
v_limits = (float(0 - EPS), float(PI + EPS))
hz_limits = (float(-PI - EPS), float(PI + EPS))


class FoV(BaseModel):
    """A rectangular angular region (Field of View) in spherical coordinates.

    Defined by left/right horizontal bounds and top/bottom vertical bounds.
    Horizontal angles wrap on the ``+/- pi`` discontinuity (left > right
    indicates a wrapping FoV).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    left: Angle = Field(..., description="Hz ∈ [–π, +π]")
    right: Angle = Field(..., description="Hz ∈ [–π, +π]")
    top: Angle = Field(..., description="V ∈ [0, +π]")
    bottom: Angle = Field(..., description="V ∈ [0, +π]")

    def __init__(self, *, left: AngleLikeT, top: AngleLikeT, right: AngleLikeT, bottom: AngleLikeT):
        """Build an :class:`FoV` from four angular limits.

        Parameters
        ----------
        left : Angle
            Hz ∈ [–π, +π].
        right : Angle
            Hz ∈ [–π, +π].
        top : Angle
            V ∈ [0, +π].
        bottom : Angle
            V ∈ [0, +π].
        """
        super().__init__(left=left, top=top, right=right, bottom=bottom)

    @field_validator("left", "right", mode="after")
    @classmethod
    def _check_hz(cls, hz: Angle) -> Angle:
        if not -np.pi - EPS <= hz.internal_value <= np.pi + EPS:
            raise ValueError(f"Horizontal angle {hz.radians} not in [-π, π]")
        return hz

    @field_validator("top", "bottom", mode="after")
    @classmethod
    def _check_elevation(cls, v: Angle) -> Angle:
        if not 0 - EPS <= v.internal_value <= np.pi + EPS:
            raise ValueError(f"Top angle {v.radians} not in [0, π]")
        return v

    @model_validator(mode="after")
    def _check_bottom_and_top(self) -> Self:
        if self.top > self.bottom:
            raise ValueError(f"Top ({self.top.radians}) must be smaller than bottom ({self.bottom.radians})")
        return self

    @classmethod
    def construct_without_bounds_check(
        cls, *, left: AngleLikeT, right: AngleLikeT, top: AngleLikeT, bottom: AngleLikeT
    ) -> Self:
        """Construct a FoV without bounds check.

        Enables construction of an :class:`FoV` whose angular limits cross over
        the standard wraparound boundary (e.g., a horizontal FoV spanning
        ``350`` to ``10`` degrees).

        Parameters
        ----------
        left : AngleLikeT
            Left horizontal bound. May be greater than ``right`` to denote a
            wraparound region.
        right : AngleLikeT
            Right horizontal bound.
        top : AngleLikeT
            Top vertical bound.
        bottom : AngleLikeT
            Bottom vertical bound. May be less than ``top`` for wraparound.

        Returns
        -------
        FoV
            A new :class:`FoV` constructed via
            :meth:`pydantic.BaseModel.model_construct`, bypassing the
            ``_check_elevation`` and ``_check_bottom_and_top`` model validators.

        Notes
        -----
        This method bypasses the ``_check_elevation`` and
        ``_check_bottom_and_top`` model validators via :meth:`model_construct`.
        It exists for legitimate use cases -- FoVs that cross the angular
        wraparound (e.g., a horizontal FoV spanning ``350`` to ``10`` degrees).
        The caller is responsible for ensuring the
        (``left``, ``right``, ``top``, ``bottom``) quadruple represents a
        meaningful angular region; downstream FoV operations may produce
        surprising results if the invariant is violated.

        This is FRAG-06 documentation (Plan 02-06 / D-22): the
        ``model_construct`` call below is deliberate and the design is correct
        as-is. The method is **not** a security bypass --
        ``construct_without_bounds_check`` is callable only by code that
        already has full control over the FoV construction, and the validators
        it skips are domain-shape checks (angular ordering), not security-shape
        checks.
        """
        new_instance = cls.model_construct(
            _fields_set={"left", "right", "top", "bottom"},
            left=Angle.parse(left),
            top=Angle.parse(top),
            right=Angle.parse(right),
            bottom=Angle.parse(bottom),
        )
        return new_instance

    @classmethod
    def from_angles(cls, horizontal: VectorT | AngleArray, vertical: VectorT | AngleArray) -> Self:
        """Construct a FoV from horizontal and vertical angles.

        Parameters
        ----------
        horizontal: VectorT | AngleArray
        vertical: VectorT | AngleArray

        Returns
        -------
        FoV
        """
        # Create a continuous representation of hz angles in range from 0 -> 2π
        hz_shifted = horizontal.copy()
        hz_shifted[hz_shifted < 0] = TWO_PI + hz_shifted[hz_shifted < 0]

        # Get the extent from max-min
        extent_shifted = hz_shifted.max() - hz_shifted.min()
        extent = horizontal.max() - horizontal.min()

        # The smallest extent represents the most likely
        if extent_shifted < extent and not np.isclose(extent_shifted, extent):
            left = hz_shifted.min()
            right = hz_shifted.max() - TWO_PI
        else:
            left = horizontal.min()
            right = horizontal.max()

        fov = cls(left=left, right=right, top=vertical.min(), bottom=vertical.max())

        # Check that the points do in fact lie in the smaller extent
        if not np.all(fov.find_points_inside(horizontal, vertical)):
            fov = cls(left=right, right=left, top=vertical.min(), bottom=vertical.max())

        # assert np.all(fov.find_points_inside(horizontal, vertical))

        return fov

    def __iter__(self) -> Generator[tuple[str, Angle], None, None]:
        """Yield ``(name, Angle)`` pairs for each of the four bounds, in left-top-right-bottom order."""
        yield "left", self.left
        yield "top", self.top
        yield "right", self.right
        yield "bottom", self.bottom

    @property
    def crosses_pi(self) -> bool:
        """Check whether the FoV horizontal range crosses the ``+/- pi`` boundary.

        Returns
        -------
        bool
            ``True`` if ``left > right`` (wrapping FoV).
        """
        return self.left > self.right

    def width(self) -> Angle:
        """Return the angular width (horizontal extent) of the FoV.

        Returns
        -------
        Angle
            The horizontal extent (accounting for wrap at ``+/- pi``).
        """
        if self.crosses_pi:
            return Angle(TWO_PI) - (self.left - self.right)
        return self.right - self.left

    def height(self) -> Angle:
        """Return the angular height (vertical extent) of the FoV.

        Returns
        -------
        Angle
            The vertical extent.
        """
        return self.bottom - self.top

    def extent(self) -> tuple[Angle, Angle]:
        """Return the angular extent ``(width, height)`` of the FoV.

        Returns
        -------
        tuple[Angle, Angle]
            ``(width, height)`` pair.
        """
        return self.width(), self.height()

    def center(self) -> tuple[Angle, Angle]:
        """Return the center of the FoV.

        Returns
        -------
        tuple[Angle, Angle]
        """
        half_width = self.width() / 2
        half_height = self.height() / 2

        hz_center = self.left + half_width
        v_center = self.top + half_height

        if not self.crosses_pi:
            return hz_center, v_center

        if not (0 <= hz_center.radians <= PI):
            hz_center = self.right - half_width

        return hz_center, v_center

    @classmethod
    @validate_call(config=cast(ConfigDict, DEFAULT_CONFIG | {"validate_return_type": False}))
    def from_center_with_extent(
        cls, centerpoint: tuple[Angle | float, Angle | float], extent: tuple[Angle | float, Angle | float]
    ) -> Self:
        """Build an :class:`FoV` from a center point and angular extent.

        Parameters
        ----------
        centerpoint : tuple[float, float]
            ``(horizontal_center, vertical_center)`` for the FoV.
        extent : tuple[float, float]
            ``(width, height)`` extent for the FoV.

        Returns
        -------
        FoV
            FoV centered on ``centerpoint`` with the requested extent.
        """
        new_instance = cls.construct_without_bounds_check(
            left=centerpoint[0] - extent[0] / 2,
            right=centerpoint[0] + extent[0] / 2,
            top=centerpoint[1] - extent[1] / 2,
            bottom=centerpoint[1] + extent[1] / 2,
        )
        return new_instance

    def _non_wrapping_segments(self) -> tuple[FoV, FoV]:
        part_a = type(self)(left=self.left, right=PI, top=self.top, bottom=self.bottom)
        part_b = type(self)(left=-PI, right=self.right, top=self.top, bottom=self.bottom)
        return part_a, part_b

    @staticmethod
    def _interval_intersection_1d(a0: Angle, a1: Angle, b0: Angle, b1: Angle) -> tuple[Angle, Angle] | None:
        """Compute the intersection of two 1D closed intervals on the real number line (no wrapping).

        Parameters
        ----------
        a0 : Angle
            Minimum of the first interval.
        a1 : Angle
            Maximum of the first interval.
        b0 : Angle
            Minimum of the second interval.
        b1 : Angle
            Maximum of the second interval.

        Returns
        -------
        tuple[float, float] | None
            Intersection ``(low, high)`` or ``None`` if the intervals do not
            overlap.
        """
        low = max(a0, b0)  # Left or top components
        high = min(a1, b1)  # Right or bottom components
        return (low, high) if low < high or np.isclose(low, high) else None

    @classmethod
    def _get_intersections_from_one_wrapping_fov(cls, wrapping_fov: FoV, fov2: FoV) -> tuple[Angle, Angle] | None:
        # get the intersections on either non-crossing part
        part_a, part_b = wrapping_fov._non_wrapping_segments()
        part_a_intersect = cls._interval_intersection_1d(part_a.left, part_a.right, fov2.left, fov2.right)
        part_b_intersect = cls._interval_intersection_1d(part_b.left, part_b.right, fov2.left, fov2.right)

        # Case 2: Still no intersections
        if (part_a_intersect is None) and (part_b_intersect is None):
            return None

        # Case 3 + 4: Get the intersection from the respective side
        elif part_a_intersect is None:
            left = part_b_intersect[0]
            right = part_b_intersect[1]
        elif part_b_intersect is None:
            left = part_a_intersect[0]
            right = part_a_intersect[1]
        # Case 5 + 6: Intersection on both sides / full circle encompassing
        else:
            if np.isclose(fov2.left, -PI) and np.isclose(fov2.right, PI):
                return wrapping_fov.left, wrapping_fov.right

            raise ValueError("Intersections on both sides of the wrapping FoV detected and not supported")
            # Two cases:
            #  - Where the non-wrapping FoV is a full circle (encasing) the wrapping FoV
            #  - Where the non-wrapping FoV touches/overlaps both edges of the wrapping FoV. Creating two intersections

        return left, right

    @staticmethod
    def _get_unions_from_one_wrapping_fov(wrapping_fov: FoV, non_wrapping_fov: FoV) -> tuple[Angle, Angle]:
        """Split a wrapping FoV into two segments and union each against a non-wrapping FoV."""
        part_a, part_b = wrapping_fov._non_wrapping_segments()
        part_a_union = part_a.union(non_wrapping_fov)
        part_b_union = part_b.union(non_wrapping_fov)

        if part_a_union.width() < part_b_union.width():
            left = part_a_union.left
            right = part_b.right
        else:
            left = part_a.left
            right = part_b_union.right

        return left, right

    def union(self, fov2: Self) -> Self:
        """Return the union of this FoV with another (handles wrap at ``+/- pi``).

        Parameters
        ----------
        fov2 : FoV
            The other FoV to union with.

        Returns
        -------
        FoV
            The smallest FoV containing both inputs.
        """
        # split into non-overlapping FoVs -
        # Part a represents part less than +PI and part b represents part greater than -PI
        # The smallest width made from the unions represents the side to extend

        if self.crosses_pi and not fov2.crosses_pi:
            left, right = self._get_unions_from_one_wrapping_fov(self, fov2)

        elif not self.crosses_pi and fov2.crosses_pi:
            left, right = self._get_unions_from_one_wrapping_fov(fov2, self)

        # Case 1 and 4: Both non-wrapping or both wrapping
        else:
            left = min(self.left, fov2.left)
            right = max(self.right, fov2.right)

        return type(self)(
            left=left,
            right=right,
            top=min(self.top, fov2.top),
            bottom=max(self.bottom, fov2.bottom),
        )

    def intersect(self, fov2: Self) -> Self | None:
        """Return the intersection of this FoV with another, or ``None`` if disjoint.

        Parameters
        ----------
        fov2 : FoV
            The other FoV to intersect with.

        Returns
        -------
        FoV or None
            The intersection FoV, or ``None`` if the two are disjoint.
        """
        # Case 1: no vertical intersection, therefore no intersection.
        vertical_intersection = self._interval_intersection_1d(a0=self.top, a1=self.bottom, b0=fov2.top, b1=fov2.bottom)

        if vertical_intersection is None:
            return None
        else:
            top = vertical_intersection[0]
            bottom = vertical_intersection[1]

        if self.crosses_pi and not fov2.crosses_pi:
            left, right = self._get_intersections_from_one_wrapping_fov(self, fov2)
        elif not self.crosses_pi and fov2.crosses_pi:
            left, right = self._get_intersections_from_one_wrapping_fov(fov2, self)
        elif self.crosses_pi and fov2.crosses_pi:
            left = max(self.left, fov2.left)
            right = min(self.right, fov2.right)
        else:
            horizontal_intersection = self._interval_intersection_1d(self.left, self.right, fov2.left, fov2.right)
            if horizontal_intersection is None:
                return None

            left = horizontal_intersection[0]
            right = horizontal_intersection[1]

        return type(self)(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        )

    def encompasses(self, fov2: Self) -> bool:
        """Check whether ``self`` fully surrounds ``fov2``.

        Parameters
        ----------
        fov2 : FoV
            Candidate inner FoV.

        Returns
        -------
        bool
            ``True`` if ``fov2`` is fully contained inside ``self`` (within
            ``EPS`` tolerance).
        """
        left_chk = self.left <= fov2.left + EPS
        top_chk = self.top <= fov2.top + EPS
        right_chk = self.right >= fov2.right - EPS
        bottom_chk = self.bottom >= fov2.bottom - EPS

        return left_chk and top_chk and right_chk and bottom_chk

    def find_points_inside(self, horizontal: Vector_Float_T, vertical: Vector_Float_T) -> Vector_Bool_T:
        """Return a boolean mask of the input points that fall inside the FoV."""
        v_indices = np.logical_and(vertical >= self.top, vertical <= self.bottom)

        if self.crosses_pi:
            # Combines the ranges [left, π] and [-π, right]
            hz_indices = np.logical_or(horizontal >= self.left, horizontal <= self.right)
        else:
            # Range of [left, right]
            hz_indices = np.logical_and(horizontal >= self.left, horizontal <= self.right)

        return np.logical_and(v_indices, hz_indices)

    @validate_variables
    def ratio(self) -> NonNegativeFloat:
        """Return the width-to-height ratio of the FoV.

        Returns
        -------
        NonNegativeFloat
            ``width() / height()``.
        """
        return self.width() / self.height()

    @validate_call(config=DEFAULT_CONFIG)
    def extend_to_ratio(self, ratio: float) -> Self:
        """Extend the FoV to match a specified width-to-height ratio.

        Parameters
        ----------
        ratio : float
            Target width-to-height ratio.

        Returns
        -------
        FoV
            New FoV matching the requested ratio (or ``self`` if already at
            ratio).
        """
        if self.ratio() - ratio > EPS:
            target_vertical_extent = self.width() / ratio

            return type(self).construct_without_bounds_check(
                left=self.left,
                top=self.top,
                right=self.right,
                bottom=self.top + target_vertical_extent,
            )

        elif ratio - self.ratio() > EPS:
            target_horizontal_extent = self.height() * ratio

            return type(self).construct_without_bounds_check(
                left=self.left,
                top=self.top,
                right=self.left + target_horizontal_extent,
                bottom=self.bottom,
            )

        else:
            return self

    @validate_call(config=DEFAULT_CONFIG)
    def split(self, shape: tuple[int, int]) -> list[Self]:
        """Split the FoV into smaller FoVs based on a grid shape.

        Parameters
        ----------
        shape : tuple[int, int]
            The number of horizontal and vertical splits respectively.

        Returns
        -------
        list[FoV]
        """
        assert shape[0] > 0 and shape[1] > 0
        if shape[0] == shape[1] == 1:
            return [self]

        horizontal_borders = AngleArray(
            np.linspace(
                start=self.left.radians, stop=self.right.radians, num=shape[0] + 1, endpoint=True, retstep=False
            )
        )

        vertical_borders = AngleArray(
            np.linspace(
                start=self.top.radians, stop=self.bottom.radians, num=shape[1] + 1, endpoint=True, retstep=False
            )
        )

        fov_splits = [
            type(self)(left=hor_min, top=elev_min, right=hor_max, bottom=elev_max)
            for hor_min, hor_max in zip(horizontal_borders[:-1], horizontal_borders[1:], strict=True)
            for elev_min, elev_max in zip(vertical_borders[:-1], vertical_borders[1:], strict=True)
        ]
        for fov_split in fov_splits:
            fov_split.left.display_unit = self.left.display_unit
            fov_split.right.display_unit = self.right.display_unit
            fov_split.top.display_unit = self.top.display_unit
            fov_split.bottom.display_unit = self.bottom.display_unit
        return fov_splits

    @validate_call(config=DEFAULT_CONFIG)
    def equal_tiles(self, width: Angle | float, height: Angle | float) -> list[Self]:
        """Divides a region into equal tiles based on a specified width and height.

        Parameters
        ----------
        width : Angle or float
        height : Angle or float

        Returns
        -------
        list[FoV]
        """
        assert float(width) > 0 and float(height) > 0
        # assert any(target < own for target, own in zip(target_extent[0], self.extent(target_extent[1])))

        return self.split(
            shape=(
                int(np.ceil(self.width() / width)),
                int(np.ceil(self.height() / height)),
            )
        )

    def tile(self, target_extent: Self, expand_to_integer_multiple: bool = False) -> list[list[Self]]:
        """Divides the current field of view (FOV) into smaller tiles based on the specified target extent/FoV.

        If `expand_to_integer_multiple` is True, the method ensures that the field of view is expanded to the nearest
        integer multiple of the target_extent dimensions before tiling.

        Parameters
        ----------
        target_extent : FoV
            FoV object representing the target extent for tiling.

        expand_to_integer_multiple : bool, optional

        Returns
        -------
        list[list[FoV]]
        """
        if expand_to_integer_multiple:
            width_int = int(np.ceil(self.width() / target_extent.width()))
            height_int = int(np.ceil(self.height() / target_extent.height()))

            width_target = width_int * target_extent.width()
            height_target = height_int * target_extent.height()

            extended_fov = type(self).from_center_with_extent(self.center(), (width_target, height_target))
            return extended_fov.tile(target_extent, False)

        horizontal_steps = AngleArray(np.append(np.arange(self.left, self.right, target_extent.width()), self.right))
        elevation_steps = AngleArray(np.append(np.arange(self.top, self.bottom, target_extent.height()), self.bottom))

        horizontal_bins = list(zip(horizontal_steps[:-1], horizontal_steps[1:], strict=True))
        vertical_bins = list(zip(elevation_steps[:-1], elevation_steps[1:], strict=True))

        tiles = []
        for left_edge, right_edge in horizontal_bins:
            if right_edge - left_edge <= 0:
                continue

            horizontal_tiles = []
            for top_edge, bottom_edge in vertical_bins:
                if bottom_edge - top_edge <= 0:
                    continue

                new_fov = type(self).construct_without_bounds_check(
                    left=left_edge,
                    top=top_edge,
                    right=right_edge,
                    bottom=bottom_edge,
                )

                new_fov.left.display_unit = self.left.display_unit
                new_fov.right.display_unit = self.right.display_unit
                new_fov.top.display_unit = self.top.display_unit
                new_fov.bottom.display_unit = self.bottom.display_unit

                if all(e > EPS for e in new_fov.extent()):
                    horizontal_tiles.append(new_fov)

            if horizontal_tiles:
                tiles.append(horizontal_tiles)

        return tiles

    def quadrants(self) -> tuple[Self, ...]:
        """Split the FoV into four equal quadrants.

        Returns
        -------
        tuple of Self
            The four quadrant FoVs (top-left, top-right, bottom-left,
            bottom-right).
        """
        # Keep for legacy
        return tuple(self.split(shape=(2, 2)))

    @classmethod
    def merge(cls, fovs: Iterable[Self]) -> Self:
        """Merge multiple FoVs into one that encompasses the total area covered.

        Parameters
        ----------
        fovs : Iterable[FoV]
            All FoV objects to be merged.

        Returns
        -------
        FoV
            Smallest FoV containing all inputs.
        """
        horizontals = []
        verticals = []
        for fov in fovs:
            horizontals.extend([fov.left, fov.right])
            verticals.extend([fov.top, fov.bottom])

        return cls.from_angles(np.array(horizontals), np.array(verticals))

    @property
    def horizontal_min(self) -> Angle:
        """Horizontal minimum angle value. Equivalent to the left attribute.

        Warnings
        --------
        DeprecationWarning
            This property is deprecated. Use the 'left' property instead.

        Returns
        -------
        Angle
        """
        warnings.warn(
            "elevation_min property has been deprecated. Please use the 'top' property",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.left

    @property
    def horizontal_max(self) -> Angle:
        """Horizontal maximum angle value. Equivalent to the right attribute.

        Warnings
        --------
        DeprecationWarning
            This property is deprecated. Use the 'right' property instead.

        Returns
        -------
        Angle
        """
        warnings.warn(
            "horizontal_max property has been deprecated. Please use the 'top' property",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.right

    @property
    def elevation_min(self) -> Angle:
        """Elevation minimum angle value. Equivalent to the top attribute.

        Warnings
        --------
        DeprecationWarning
            This property is deprecated. Use the 'top' property instead.

        Returns
        -------
        Angle
        """
        warnings.warn(
            "elevation_min property has been deprecated. Please use the 'top' property",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.top

    @property
    def elevation_max(self) -> Angle:
        """Elevation maximum angle value. Equivalent to the bottom attribute.

        Warnings
        --------
        DeprecationWarning
            This property is deprecated. Use the 'bottom' property instead.

        Returns
        -------
        Angle
        """
        warnings.warn(
            "elevation_max property has been deprecated. Please use the 'bottom' property",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.bottom

    def __str__(self):
        """Return a human-readable representation of the FoV in display units."""
        left, top, right, bottom = self.left, self.top, self.right, self.bottom
        return f"{self.__class__.__name__}({left=!s}, {right=!s}, {top=!s}, {bottom=!s})"

    def __repr__(self):
        """Return the debug representation (same as ``__str__``)."""
        return str(self)


@dataclass(init=True, frozen=True)
class FoVTree:
    """A hierarchical tree structure for spatial partitioning of FoVs.

    Parameters
    ----------
    identifier : str
        Unique identifier for this tree node.
    node : FoV
        The FoV associated with this tree node.
    children : dict[str, FoVTree] | None
        Dictionary of child nodes, if any.
    """

    identifier: str
    node: FoV
    children: Optional[dict[str, Self]] = field(default_factory=dict)

    @staticmethod
    def add_identifier(fovs: list[FoV], shape: tuple[int, int]) -> tuple[tuple[str | Any, FoV], ...]:
        """Add unique identifier(s) to each field of view (FoV) in the list.

        Parameters
        ----------
        fovs : list[FoV]
        shape : tuple[int, int]
            Shape dimensions used to calculate the identifier length.

        Returns
        -------
        tuple[tuple[str | Any, FoV], ...]
            A tuple containing tuples, each consisting of a unique identifier
            and the corresponding FoV object.
        """
        identifier_length = np.ceil(math.log(shape[0] * shape[1], 16)).astype(int)
        return tuple(
            [(((identifier_length - len(hex_str := f"{i:x}")) * "0" + hex_str), fov) for i, fov in enumerate(fovs)]
        )

    def depth(self) -> int:
        """Return the depth of the FoVTree from the current node.

        Leaf nodes have a depth of 1.

        Returns
        -------
        int
        """
        if self.is_leaf() or self.children is None:
            return 1
        return max([c.depth() for c in self.children.values()]) + 1

    def __len__(self) -> int:
        """Return the number of leaf nodes in the subtree rooted at ``self``."""
        if self.is_leaf() or self.children is None:
            return 1
        return sum((len(c) for c in self.children.values()))

    def to_list(self) -> list[tuple[str, FoV]]:
        """Convert tree structure into a flattened list.

        Returns
        -------
        list[(str, FoV)]
            Each tuple consists of the identifier and the FoV node
        """
        if self.is_leaf() or self.children is None:
            return [(self.identifier, self.node)]

        children_lists = [c.to_list() for c in self.children.values()]

        return list(chain.from_iterable(children_lists))

    @classmethod
    def build_from_tiles(cls, tiles: list[list[FoV]], min_children: int = 4, identifier: str = "") -> Self | None:
        """Construct a tree from a 2D grid of FoVs by recursive quad-splitting.

        Parameters
        ----------
        tiles : list[list[FoV]]
            Grid of FoVs.
        min_children : int, default=4
            The minimum number of children to avoid further splitting.
        identifier : str, default=""
            Unique identifier for this node.

        Returns
        -------
        FoVTree
            Root of the constructed tree, or ``None`` if ``tiles`` is empty.
        """
        assert min_children > 1
        if not tiles or not tiles[0]:
            return None

        # Todo: Check this logic!
        if len(tiles) == 1 and len(tiles[0]) == 1:
            if identifier == "":
                return cls("root", tiles[0][0], None)
            return cls(identifier, tiles[0][0], None)

        fov = FoV.construct_without_bounds_check(
            left=tiles[0][0].left,
            top=tiles[0][0].top,
            right=tiles[-1][-1].right,
            bottom=tiles[-1][-1].bottom,
        )

        if len(tiles) * len(tiles[0]) <= min_children:
            flat_tiles = [tile for row in tiles for tile in row]
            fov_children: dict[str, FoVTree] = {
                str(i): cls(identifier + str(i), tile, {}) for i, tile in enumerate(flat_tiles)
            }
            return cls(identifier, fov, fov_children)  # TODO: BUG! Rebuild identifier function to work 2D

        q0 = tiles[: len(tiles) // 2]
        q1 = tiles[len(tiles) // 2 :]
        q00 = [row[: len(row) // 2] for row in q0]
        q01 = [row[len(row) // 2 :] for row in q0]
        q10 = [row[: len(row) // 2] for row in q1]
        q11 = [row[len(row) // 2 :] for row in q1]

        fov_children = {
            "0": cast(FoVTree, cls.build_from_tiles(q00, min_children, identifier=(identifier + "0"))),
            "1": cast(FoVTree, cls.build_from_tiles(q01, min_children, identifier=(identifier + "1"))),
            "2": cast(FoVTree, cls.build_from_tiles(q10, min_children, identifier=(identifier + "2"))),
            "3": cast(FoVTree, cls.build_from_tiles(q11, min_children, identifier=(identifier + "3"))),
        }

        fov_children = {k: v for k, v in fov_children.items() if v is not None}

        return cls(identifier, fov, fov_children)

    # TODO - Reimplement Low priority / Historical
    # @classmethod
    # def build_by_splitting(cls, fov: FoV, target_ratio: float, target_fov_extent: tuple[tuple[float, float], str],
    #                        max_denominator: int, identifier: str = "") -> Self:
    #     # TODO: Rework stopping criteria
    #     assert target_fov_extent[1] in ("rad", "gon", "deg")
    #
    #     target_extent = target_fov_extent[0]
    #     angle_unit = target_fov_extent[1]
    #
    #     shape = cls.calculate_optimal_shape(fov, target_ratio, target_fov_extent, max_denominator)
    #
    #     if (fov.extent(unit=angle_unit)[0] < target_extent[0] * shape[0] or
    #             fov.extent(unit=angle_unit)[1] < target_extent[1] * shape[1]):
    #         fov_tiles = fov.equal_tiles(target_fov_extent)
    #         shape = (len(fov_tiles), 1)
    #         fov_children = {child_identifier: cls(identifier + child_identifier, child, {})
    #                         for child_identifier, child in cls.add_identifier(fov_tiles, shape)}
    #     else:
    #         fov_splits = fov.split(shape)
    #         fov_children = {
    #             child_identifier: cls.build_by_splitting(
    #                 child, target_ratio, target_fov_extent,
    #                 max_denominator * 2, identifier + child_identifier,
    #             )
    #             for child_identifier, child in cls.add_identifier(fov_splits, shape)
    #         }
    #
    #     return cls(identifier, fov, fov_children)

    # @classmethod
    # def build_by_tiling(cls, fov: FoV, target_fov_extent: tuple[tuple[float, float], str],
    #                     identifier: str = "") -> Self:
    #     fov_tiles = fov.equal_tiles(target_fov_extent)
    #
    #     identifier_length = np.ceil(math.log(len(fov_tiles), 16)).astype(int)
    #     fov_with_identifier = tuple([(((identifier_length - len(hex_str := f"{i:x}")) * "0" + hex_str), fov)
    #                                  for i, fov in enumerate(fov_tiles)])
    #
    #     fov_children = {child_identifier: cls(identifier + child_identifier, child, {})
    #                     for child_identifier, child in fov_with_identifier}
    #
    #     return cls(identifier, fov, fov_children)

    # @classmethod
    # def build(cls, fov: FoV, target_fov_extent: FoV):
    #     tiles = fov.tile((target_fov_extent.extent("rad"), "rad"))
    #     pass

    def __getitem__(self, identifier: str) -> Self:
        """Look up a child by ``identifier`` (descending into nested children for long ids)."""
        # TODO: extend to complete for full string
        if not identifier or identifier == "root" or self.children is None:
            return self

        child_identifier_length = np.ceil(math.log(len(self.children), 16)).astype(int)

        if len(identifier) > child_identifier_length:
            return self.children[identifier[:child_identifier_length]][identifier[child_identifier_length:]]

        return self.children[identifier]

    def is_leaf(self):
        """Check whether this FoVTree node is a leaf (has no child nodes).

        Returns
        -------
        bool
            ``True`` if the node has no children, ``False`` otherwise.
        """
        return not self.children

    @staticmethod
    def calculate_optimal_shape(fov: FoV, target_ratio: float, max_denominator: float) -> tuple[int, int]:
        """Calculate the optimal shape based on a given field of view (FoV), target ratio, and maximum denominator.

        Parameters
        ----------
        fov : FoV
        target_ratio : float
        max_denominator : float
            Maximum number of FoV tree should be split into along a direction

        Returns
        -------
        tuple[int, int]
        """
        shape = Fraction(fov.ratio() / target_ratio).limit_denominator(np.round(max_denominator).astype(int))
        new_shape = (
            shape.numerator,
            shape.denominator,
        )
        if new_shape[0] == new_shape[1] == 1:
            new_shape = (2, 2)

        return new_shape
