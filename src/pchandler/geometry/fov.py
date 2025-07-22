"""
``pchandler.fov``

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
from typing import (
    Iterable, Optional, cast, Self, TYPE_CHECKING, Any, Union,
    Generator
)

import numpy as np
from numba.core.extending import overload
from pydantic import Field, NonNegativeFloat, validate_call, BaseModel, field_validator, ConfigDict

from pchandler.constants import EPS, TWO_PI, PI, validate_variables, DEFAULT_CONFIG
from pchandler.util import AngleUnit, convert_angles
from pchandler.spherical.angle import Angle, AngleArray

if TYPE_CHECKING:
    from pchandler.base_types import Vector_3_T

logger = logging.getLogger(__name__.split(".")[0])



class FoV(BaseModel):
    """
    Field of View is defined by the spherical angles from the scan origin coordinate system (SOCS).

    Azimuths or Horizontal angles are in the range of [-PI, +PI] with +PI being a right rotation from the X-axis
    Zenith / Vertical angles are in the range of [0, +PI] with 0 being at the zenith

    This is designed to be more compatible with spherical angle projections to image coordinate systems.

    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    left: Angle = Field(..., description="Hz ∈ [–π, +π]")
    right: Angle = Field(..., description="Hz ∈ [–π, +π]")
    top: Angle = Field(..., description="V ∈ [0, +π]")
    bottom: Angle = Field(..., description="V ∈ [0, +π]")

    @overload
    def __init__(
        self,
        *,
        left: Union[Angle, float, str],
        right: Union[Angle, float, str],
        top: Union[Angle, float, str],
        bottom: Union[Angle, float, str],
    ) -> None: ...  # pragma: no cover

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def construct_without_bounds_check(
            cls,
            *,
            left: Union[Angle, float, str],
            right: Union[Angle, float, str],
            top: Union[Angle, float, str],
            bottom: Union[Angle, float, str],
    ) -> Self:
        left_a = Angle.parse(left)
        right_a = Angle.parse(right)
        top_a = Angle.parse(top)
        bottom_a = Angle.parse(bottom)
        new_instance = cls.model_construct(
            _fields_set={'left', 'right', 'top', 'bottom'},
            left=left_a,
            right=right_a,
            top=top_a,
            bottom=bottom_a,
        )
        return new_instance


    @field_validator("left", "right", mode="after")
    def _check_hz(cls, hz: Angle) -> Angle:
        if not isinstance(hz, Angle):
            raise TypeError("left/right must be an Angle or float")
        if not -np.pi-EPS <= hz.internal_value <= np.pi+EPS:
            raise ValueError(f"Horizontal angle {hz.radians} not in [-π, π]")
        return hz

    @field_validator("top", mode="after")
    def _check_top(cls, v: Angle, info) -> Angle:
        if not isinstance(v, Angle):
            raise TypeError("Top must be an Angle or float")
        if not 0-EPS <= v.internal_value <= np.pi+EPS:
            raise ValueError(f"Top angle {v.radians} not in [0, π]")
        bottom = info.data.get("bottom")
        if bottom is not None and v > bottom:
            raise ValueError(f"Top ({v.radians}) must be smaller than bottom ({bottom.radians})")
        return v

    @field_validator("bottom", mode="after")
    def _check_top(cls, v: Angle, info) -> Angle:
        if not isinstance(v, Angle):
            raise TypeError("Bottom must be an Angle or float")
        if not 0-EPS <= v.internal_value <= np.pi+EPS:
            raise ValueError(f"Bottom angle {v.radians} not in [0, π]")
        top = info.data.get("top")
        if top is not None and v < top:
            raise ValueError(f"Bottom ({v.radians}) must be larger than top ({top.radians})")
        return v

    # TODO: Write tests
    @classmethod
    def from_angles(cls, horizontal: Vector_3_T | AngleArray, vertical: Vector_3_T | AngleArray) -> Self:
        return cls(left=horizontal.min(), top=vertical.min(), right=horizontal.max(), bottom=vertical.max())

    def __iter__(self) -> Generator[Angle]:
        yield self.left
        yield self.top
        yield self.right
        yield self.bottom

    # DISCUSS Would a change to left/right create a clash with previous logic based on min/max values -
    #  especially where it wraps at TWO_PI?
    @property
    def crosses_pi(self) -> bool:
        return self.left > self.right


    def width(self) -> Angle:
        if self.crosses_pi:
            return TWO_PI - (self.left - self.right)
        return self.right - self.left

    def height(self) -> Angle:
        return self.bottom - self.top

    def extent(self) -> tuple[Angle, Angle]:
        return self.width(), self.height()

    def center(self) -> tuple[Angle, Angle]:
        horizontal_center = (self.left + self.right) / 2
        elevation_center = (self.top + self.bottom) / 2
        if self.crosses_pi:
            horizontal_center = (horizontal_center + PI) % PI
        return horizontal_center, elevation_center

    @classmethod
    @validate_call(config=DEFAULT_CONFIG | {"validate_return_type": False})
    def from_center_with_extent(cls, centerpoint: tuple[Angle, Angle], extent: tuple[Angle, Angle]) -> Self:
        """
        Creates an FoV instance from a center point and angular extent.

        Parameters
        ----------
        centerpoint : tuple[float, float]
            The (horizontal_angle, vertical_angle) center of the FoV in the specified unit.
        extent : tuple[float, float]
            The angular extent (width, height) of the FoV in the specified unit.

        Returns
        -------
        FoV
            A new FoV instance.
        """
        # hz_min = centerpoint[0] - extent[0] / 2
        # hz_max = centerpoint[0] + extent[0] / 2
        # v_min = centerpoint[1] - extent[1] / 2
        """
        Creates an FoV instance from a center point and angular extent.

        Parameters
        ----------
        centerpoint : tuple[float, float]
            The (horizontal_angle, vertical_angle) center of the FoV in the specified unit.
        extent : tuple[float, float]
            The angular extent (width, height) of the FoV in the specified unit.

        Returns
        -------
        FoV
            A new FoV instance.
        """
        new_instance = cls.construct_without_bounds_check(
            left=centerpoint[0] - extent[0] / 2,
            right=centerpoint[0] + extent[0] / 2,
            top=centerpoint[1] - extent[1] / 2,
            bottom=centerpoint[1] + extent[1] / 2)
        return new_instance

        # def union(self, fov2: Self) -> Self:


    def union(self, fov2: Self) -> Self:
        """
        Computes the union of this FoV with another.

        Parameters
        ----------
        fov2 : FoV
            Another FoV to compute the union with.

        Returns
        -------
        FoV
            The smallest FoV enclosing both.
        """
        return FoV(
            left=min(self.left, fov2.left),
            top=min(self.top, fov2.top),
            right=max(self.right, fov2.right),
            bottom=max(self.bottom, fov2.bottom)
        )

    def intersect(self, fov2: Self) -> Self:
        """
        Computes the intersection of this FoV with another.

        Parameters
        ----------
        fov2 : FoV
            Another FoV to compute the intersection with.

        Returns
        -------
        FoV
            The largest FoV contained within both.
        """
        return FoV(
            left=max(self.left, fov2.left),
            top=max(self.top, fov2.top),
            right=min(self.right, fov2.right),
            bottom=min(self.bottom, fov2.bottom),
        )

    @validate_variables
    def ratio(self) -> NonNegativeFloat:
        """
        Computes the width-to-height ratio of the FoV.

        Returns
        -------
        float
            The aspect ratio (width/height) of the FoV.
        """
        return self.width() / self.height()


    @validate_call(config=DEFAULT_CONFIG)
    def extend_to_ratio(self, ratio: float) -> Self:
        if self.ratio() - ratio > EPS:
            target_vertical_extent = self.width() / ratio
            new_fov = FoV.construct_without_bounds_check(
                left=self.left,
                top=self.top,
                right=self.right,
                bottom=self.top + target_vertical_extent,
            )
        elif ratio - self.ratio() > EPS:
            target_horizontal_extent = self.height() * ratio
            new_fov = FoV.construct_without_bounds_check(
                left=self.left,
                top=self.top,
                right=self.left + target_horizontal_extent,
                bottom=self.bottom,
            )
        else:
            new_fov = self
        return new_fov

    @validate_call(config=DEFAULT_CONFIG)
    def split(self, shape: tuple[int, int]) -> list[Self]:
        """
        Splits the FoV into smaller FoVs based on a grid shape.

        Parameters
        ----------
        shape : tuple[int, int]
            The number of horizontal and vertical splits.

        Returns
        -------
        list[FoV]
            A list of smaller FoVs.
        """
        assert shape[0] > 0 and shape[1] > 0
        if shape[0] == shape[1] == 1:
            return [self]

        horizontal_borders = Angle(np.linspace(
            start=self.left.radians,
            stop=self.right.radians,
            num=shape[0] + 1,
            endpoint=True,
            retstep=False
        ))
        vertical_borders = Angle(np.linspace(
            start=self.top.radians,
            stop=self.bottom.radians,
            num=shape[1] + 1,
            endpoint=True,
            retstep=False
        ))
        # horizontal_borders.display_unit = self.left.display_unit
        # vertical_borders.display_unit = self.top.display_unit

        fov_splits = [
            FoV(left=hor_min, top=elev_min, right=hor_max, bottom=elev_max)
            for hor_min, hor_max in zip(horizontal_borders[:-1], horizontal_borders[1:])
            for elev_min, elev_max in zip(vertical_borders[:-1], vertical_borders[1:])
        ]
        for fov_split in fov_splits:
            fov_split.left.display_unit = self.left.display_unit
            fov_split.right.display_unit = self.right.display_unit
            fov_split.top.display_unit = self.top.display_unit
            fov_split.bottom.display_unit = self.bottom.display_unit
        return fov_splits

    @validate_call(config=DEFAULT_CONFIG)
    def equal_tiles(self, width: Angle, height: Angle) -> list[Self]:
        assert width > 0 and height > 0
        # assert any(target < own for target, own in zip(target_extent[0], self.extent(target_extent[1])))

        return self.split(
            shape=(
                np.ceil(self.width() / width).astype(int),
                np.ceil(self.height() / height).astype(int),
            )
        )

    def tile(self, target_extent: Self) -> list[list[Self]]:
        horizontal_steps = Angle(np.append(
            np.arange(self.left, self.right, target_extent.width()), self.right
        ))

        elevation_steps = Angle(np.append(
            np.arange(self.top, self.bottom, target_extent.height()), self.bottom
        ))

        horizontal_bins = list(zip(horizontal_steps[:-1], horizontal_steps[1:]))
        vertical_bins = list(zip(elevation_steps[:-1], elevation_steps[1:]))

        tiles = []
        for hor_bin in horizontal_bins:
            if hor_bin[-1] - hor_bin[0] <= 0:
                continue
            horizontal_tiles = []
            for vert_bin in vertical_bins:
                if vert_bin[-1] - vert_bin[0] <= 0:
                    continue
                new_fov = FoV(
                    left=hor_bin[0],
                    top=vert_bin[0],
                    right=hor_bin[1],
                    bottom=vert_bin[1],
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
        # Keep for legacy
        return tuple(self.split(shape=(2, 2)))

    @classmethod
    def merge(cls, fovs: Iterable[Self]) -> Self:
        return cls(
            left=min(fovs, key=lambda fov: fov.left).left,
            top=min(fovs, key=lambda fov: fov.top).top,
            right=max(fovs, key=lambda fov: fov.right).right,
            bottom=max(fovs, key=lambda fov: fov.bottom).bottom,
        )

    @property
    def horizontal_min(self) -> Angle:
        warnings.warn("elevation_min property has been deprecated. Please use the 'top' property",
                      DeprecationWarning, stacklevel=2)
        return self.left

    @property
    def horizontal_max(self) -> Angle:
        warnings.warn("horizontal_max property has been deprecated. Please use the 'top' property",
                      DeprecationWarning, stacklevel=2)
        return self.right

    @property
    def elevation_min(self) -> Angle:
        warnings.warn("elevation_min property has been deprecated. Please use the 'top' property",
                      DeprecationWarning, stacklevel=2)
        return self.top

    @property
    def elevation_max(self) -> Angle:
        warnings.warn("elevation_max property has been deprecated. Please use the 'bottom' property",
                      DeprecationWarning, stacklevel=2)
        return self.bottom

    def __repr__(self):
        left, top, right, bottom = self.left, self.top, self.right, self.bottom
        return (
            f"{self.__class__.__name__}({left=!r}, {right=!r}, {top=!r}, {bottom=!r})"
        )

    def __str__(self):
        left, top, right, bottom = self.left, self.top, self.right, self.bottom
        return (
            f"{self.__class__.__name__}({left=!s}, {right=!s}, {top=!s}, {bottom=!s})"
        )

@dataclass(init=True, frozen=True)
class FoVTree:
    """
    Represents a hierarchical tree structure for spatial partitioning of FoVs.

    Attributes
    ----------
    identifier : str
        A unique identifier for this tree node.
    node : FoV
        The FoV associated with this tree node.
    children : Optional[dict[str, FoVTree]]
        A dictionary of child nodes, if any.
    """

    identifier: str
    node: FoV
    children: Optional[dict[str, Self]] = field(default_factory=dict)

    @staticmethod
    def add_identifier(fovs: list[FoV], shape: tuple[int, int]) -> tuple[tuple[str | Any, FoV], ...]:
        identifier_length = np.ceil(math.log(shape[0] * shape[1], 16)).astype(int)
        return tuple(
            [(((identifier_length - len(hex_str := f"{i:x}")) * "0" + hex_str), fov) for i, fov in enumerate(fovs)]
        )

    def depth(self) -> int:
        if not self.children:
            return 1
        return max([c.depth() for c in self.children.values()]) + 1

    def to_list(self) -> list[tuple[str, FoV]]:
        if self.is_leaf():
            return [(self.identifier, self.node)]

        children_lists = [c.to_list() for c in self.children.values()]

        return list(chain.from_iterable(children_lists))

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
    #         fov_children = {child_identifier: cls.build_by_splitting(child, target_ratio, target_fov_extent,
    #                                                                  max_denominator * 2, identifier + child_identifier)
    #                         for child_identifier, child in cls.add_identifier(fov_splits, shape)}
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

    @classmethod
    def build_from_tiles(cls, tiles: list[list[FoV]], min_children: int = 4, identifier: str = "") -> Self:
        """
        Constructs a tree from a grid of FoVs.

        Parameters
        ----------
        tiles : list[list[FoV]]
            A grid of FoVs to organize into a tree.
        min_children : int, default=4
            The minimum number of children to avoid further splitting.
        identifier : str, default=""
            The identifier for the root node.

        Returns
        -------
        FoVTree
            The root of the constructed tree.
        """
        assert min_children > 1
        if not tiles or not tiles[0]:
            return None

        # Todo: Check this logic!
        if len(tiles) == 1 and len(tiles[0]) == 1:
            if identifier == "":
                return cls("root", tiles[0][0], None)
            return cls(identifier, tiles[0][0], None)

        fov = FoV(
            left=tiles[0][0].left,
            top=tiles[0][0].top,
            right=tiles[-1][-1].right,
            bottom=tiles[-1][-1].bottom,
        )

        if len(tiles) * len(tiles[0]) <= min_children:
            flat_tiles = [tile for row in tiles for tile in row]
            fov_children = {str(i): cls(identifier + str(i), tile, {}) for i, tile in enumerate(flat_tiles)}
            return cls(identifier, fov, fov_children)  # TODO: BUG! Rebuild identifier function to work 2D

        q0 = tiles[: len(tiles) // 2]
        q1 = tiles[len(tiles) // 2 :]
        q00 = [row[: len(row) // 2] for row in q0]
        q01 = [row[len(row) // 2 :] for row in q0]
        q10 = [row[: len(row) // 2] for row in q1]
        q11 = [row[len(row) // 2 :] for row in q1]

        fov_children = {
            "0": cls.build_from_tiles(q00, min_children, identifier=(identifier + "0")),
            "1": cls.build_from_tiles(q01, min_children, identifier=(identifier + "1")),
            "2": cls.build_from_tiles(q10, min_children, identifier=(identifier + "2")),
            "3": cls.build_from_tiles(q11, min_children, identifier=(identifier + "3")),
        }

        fov_children = {k: v for k, v in fov_children.items() if v is not None}

        return cls(identifier, fov, fov_children)

    # @staticmethod
    # def quadrant_split(tiles: list[list[FoV]]):
    #     # q1 = tiles[:len(tiles) // 2]
    #     # q2 = tiles[len(tiles) // 2:]
    #     # q11 = [row[:len(row)//2] for row in q1]
    #     # q12 = [row[len(row)//2:] for row in q1]
    #     # q21 = [row[:len(row) // 2] for row in q2]
    #     # q22 = [row[len(row) // 2:] for row in q2]
    #     # quadrant_FoV = FoV(horizontal_min=q11[0][0].horizontal_min,
    #     #                    elevation_min=q11[0][0].elevation_min,
    #     #                    horizontal_max=q22[-1][-1].horizontal_max,
    #     #                    elevation_max=q22[-1][-1].elevation_max)
    #
    #     FoVTree.quadrant_split(q11)
    #     FoVTree.quadrant_split(q12)
    #     FoVTree.quadrant_split(q21)
    #     FoVTree.quadrant_split(q22)
    #
    #     pass

    def __getitem__(self, identifier: str) -> Self:
        # TODO: extend to complete for full string
        if not identifier or identifier == "root":
            return self
        child_identifier_length = np.ceil(math.log(len(self.children), 16)).astype(int)
        if len(identifier) > child_identifier_length:
            return self.children[identifier[:child_identifier_length]][identifier[child_identifier_length:]]

        return self.children[identifier]

    def is_leaf(self):
        return not self.children

    @staticmethod
    def calculate_optimal_shape(
        fov: FoV, target_ratio: float, max_denominator: float
    ) -> tuple[int, int]:

        shape = Fraction(fov.ratio() / target_ratio).limit_denominator(np.round(max_denominator).astype(int))
        shape = (
            shape.numerator,
            shape.denominator,
        )
        if shape[0] == shape[1] == 1:
            shape = (2, 2)

        return shape


@dataclass(init=False, frozen=True)
class _OldFoV:
    """
    Represents a rectangular angular region in 3D space with defined horizontal and elevation bounds.

    Attributes
    ----------
    horizontal_min : float
        The minimum horizontal angle (in radians by default).
    elevation_min : float
        The minimum elevation angle (in radians by default).
    horizontal_max : float
        The maximum horizontal angle (in radians by default).
    elevation_max : float
        The maximum elevation angle (in radians by default).
    """

    # TODO: Rework quadrants as special case of generalized split on shape tuple.
    horizontal_min: float
    elevation_min: float
    horizontal_max: float
    elevation_max: float
    # _defined_units: tuple[str, ...] = ("rad", "gon", "deg")
    __internal_angular_unit: AngleUnit = AngleUnit.RAD

    def __init__(
        self,
        *,
        horizontal_min: float,
        elevation_min: float,
        horizontal_max: float,
        elevation_max: float,
        unit: [str | AngleUnit] = "rad",
    ):
        """
        Initializes an FoV instance.

        Parameters
        ----------
        horizontal_min : float
            The minimum horizontal angle in the specified unit.
        elevation_min : float
            The minimum elevation angle in the specified unit.
        horizontal_max : float
            The maximum horizontal angle in the specified unit.
        elevation_max : float
            The maximum elevation angle in the specified unit.
        unit : str or AngleUnit, default="rad"
            The angular unit of the input values ("rad", "gon", or "deg").

        Raises
        ------
        AssertionError
            If the maximum angles are less than the minimum angles.
        """
        assert horizontal_max >= horizontal_min  # TODO: Rethink in the context describing shortest path
        assert elevation_max >= elevation_min

        input_unit = AngleUnit(unit)
        values = np.array([horizontal_min, elevation_min, horizontal_max, elevation_max], dtype=float)
        convert_angles(values, source_unit=input_unit, target_unit=self.__internal_angular_unit, out=values)

        object.__setattr__(self, "horizontal_min", values[0])
        object.__setattr__(self, "elevation_min", values[1])
        object.__setattr__(self, "horizontal_max", values[2])
        object.__setattr__(self, "elevation_max", values[3])

    @classmethod
    def from_center_with_extent(
        cls, centerpoint: tuple[float, float], extent: tuple[float, float], unit: [str | AngleUnit] = "rad"
    ) -> Self:
        """
        Creates an FoV instance from a center point and angular extent.

        Parameters
        ----------
        centerpoint : tuple[float, float]
            The (horizontal, elevation) center of the FoV in the specified unit.
        extent : tuple[float, float]
            The angular extent (width, height) of the FoV in the specified unit.
        unit : str or AngleUnit, default="rad"
            The angular unit of the input values ("rad", "gon", or "deg").

        Returns
        -------
        FoV
            A new FoV instance.
        """
        fov_min = np.array(centerpoint) - np.array(extent) / 2
        fov_max = np.array(centerpoint) + np.array(extent) / 2
        return cls(
            horizontal_min=fov_min[0],
            horizontal_max=fov_max[0],
            elevation_min=fov_min[1],
            elevation_max=fov_max[1],
            unit=unit,
        )

    def as_numpy(self, unit: [str | AngleUnit] = "rad") -> np.ndarray:
        """
        Converts the FoV to a NumPy array.

        Parameters
        ----------
        unit : str or AngleUnit, default="rad"
            The angular unit of the output values ("rad", "gon", or "deg").

        Returns
        -------
        np.ndarray
            A NumPy array of the FoV boundaries.
        """
        unit = AngleUnit(unit)
        values = np.array([self.horizontal_min, self.elevation_min, self.horizontal_max, self.elevation_max])
        convert_angles(values, source_unit=self.__internal_angular_unit, target_unit=unit, out=values)
        return values

    def as_tuple(self, unit: [str | AngleUnit] = "rad") -> tuple[float, float, float, float]:
        """
        Converts the FoV to a tuple.

        Parameters
        ----------
        unit : str or AngleUnit, default="rad"
            The angular unit of the output values ("rad", "gon", or "deg").

        Returns
        -------
        tuple[float, float, float, float]
            A tuple of the FoV boundaries.
        """
        return tuple(self.as_numpy(unit=unit))

    def as_dict(self, unit: [str | AngleUnit] = "rad") -> dict[str, float]:
        """
        Converts the FoV to a dictionary.

        Parameters
        ----------
        unit : str or AngleUnit, default="rad"
            The angular unit of the output values ("rad", "gon", or "deg").

        Returns
        -------
        dict[str, float]
            A dictionary of the FoV boundaries.
        """
        values = self.as_numpy(unit=unit)
        return {
            "horizontal_min": cast(float, values[0]),
            "elevation_min": cast(float, values[1]),
            "horizontal_max": cast(float, values[2]),
            "elevation_max": cast(float, values[3]),
        }

    def width(self, unit: [str | AngleUnit] = "rad") -> float:
        """
        Computes the width of the FoV.

        Parameters
        ----------
        unit : str or AngleUnit, default="rad"
            The angular unit of the output.

        Returns
        -------
        float
            The width of the FoV.
        """
        values = self.as_dict(unit=unit)
        return values["horizontal_max"] - values["horizontal_min"]

    def height(self, unit: [str | AngleUnit] = "rad") -> float:
        """
        Computes the height of the FoV.

        Parameters
        ----------
        unit : str or AngleUnit, default="rad"
            The angular unit of the output.

        Returns
        -------
        float
            The height of the FoV.
        """
        values = self.as_dict(unit=unit)
        return values["elevation_max"] - values["elevation_min"]

    def extent(self, unit: [str | AngleUnit] = "rad") -> tuple[float, float]:
        return self.width(unit=unit), self.height(unit=unit)

    def center(self, unit: [str | AngleUnit] = "rad") -> tuple[float, float]:
        values = self.as_dict(unit=unit)
        elevation_center = (values["elevation_min"] + values["elevation_max"]) / 2
        horizontal_center = (values["horizontal_min"] + values["horizontal_max"]) / 2
        return horizontal_center, elevation_center

    def union(self, fov2: Self) -> Self:
        """
        Computes the union of this FoV with another.

        Parameters
        ----------
        fov2 : FoV
            Another FoV to compute the union with.

        Returns
        -------
        FoV
            The smallest FoV enclosing both.
        """
        return type(self)(
            horizontal_min=min(self.horizontal_min, fov2.horizontal_min),
            elevation_min=min(self.elevation_min, fov2.elevation_min),
            horizontal_max=max(self.horizontal_max, fov2.horizontal_max),
            elevation_max=max(self.elevation_max, fov2.elevation_max),
        )

    def intersect(self, fov2: Self) -> Self:
        """
        Computes the intersection of this FoV with another.

        Parameters
        ----------
        fov2 : FoV
            Another FoV to compute the intersection with.

        Returns
        -------
        FoV
            The largest FoV contained within both.
        """
        return type(self)(
            horizontal_min=max(self.horizontal_min, fov2.horizontal_min),
            elevation_min=max(self.elevation_min, fov2.elevation_min),
            horizontal_max=min(self.horizontal_max, fov2.horizontal_max),
            elevation_max=min(self.elevation_max, fov2.elevation_max),
        )

    def ratio(self) -> float:
        """
        Computes the width-to-height ratio of the FoV.

        Returns
        -------
        float
            The aspect ratio (width/height) of the FoV.
        """
        return self.extent()[0] / self.extent()[1]

    def __repr__(self):
        return (
            f"({self.horizontal_min:0.4f}, {self.elevation_min:0.4f}, "
            f"{self.horizontal_max:0.4f}, {self.elevation_max:0.4f})"
        )

    def extend_to_ratio(self, ratio: float) -> Self:
        if self.ratio() - ratio > EPS:
            target_vertical_extent = self.extent()[0] / ratio
            new_fov = type(self)(
                horizontal_min=self.horizontal_min,
                elevation_min=self.elevation_min,
                horizontal_max=self.horizontal_max,
                elevation_max=self.elevation_min + target_vertical_extent,
            )
        elif ratio - self.ratio() > EPS:
            target_horizontal_extent = self.extent()[1] * ratio
            new_fov = type(self)(
                horizontal_min=self.horizontal_min,
                elevation_min=self.elevation_min,
                horizontal_max=self.horizontal_min + target_horizontal_extent,
                elevation_max=self.elevation_max,
            )
        else:
            new_fov = self

        return new_fov

    def split(self, shape: tuple[int, int]) -> list[Self]:
        """
        Splits the FoV into smaller FoVs based on a grid shape.

        Parameters
        ----------
        shape : tuple[int, int]
            The number of horizontal and vertical splits.

        Returns
        -------
        list[FoV]
            A list of smaller FoVs.
        """
        assert shape[0] > 0 and shape[1] > 0
        if shape[0] == shape[1] == 1:
            return [self]

        horizontal_borders = np.linspace(
            start=self.horizontal_min, stop=self.horizontal_max, num=shape[0] + 1, endpoint=True, retstep=False
        )
        elevation_borders = np.linspace(
            start=self.elevation_min, stop=self.elevation_max, num=shape[1] + 1, endpoint=True, retstep=False
        )

        fov_splits = [
            type(self)(horizontal_min=hor_min, elevation_min=elev_min, horizontal_max=hor_max, elevation_max=elev_max)
            for hor_min, hor_max in zip(horizontal_borders[:-1], horizontal_borders[1:])
            for elev_min, elev_max in zip(elevation_borders[:-1], elevation_borders[1:])
        ]

        return fov_splits

    def equal_tiles(
        self,
        target_extent: tuple[tuple[float, float], str],
    ) -> list[Self]:
        assert target_extent[0][0] > 0 and target_extent[0][1]
        # assert any(target < own for target, own in zip(target_extent[0], self.extent(target_extent[1])))

        return self.split(
            shape=(
                np.ceil(self.extent(target_extent[1])[0] / target_extent[0][0]).astype(int),
                np.ceil(self.extent(target_extent[1])[1] / target_extent[0][1]).astype(int),
            )
        )

    # def tile(self, target_extent: tuple[tuple[float, float], str], ) -> list[list[Self], ...]:
    #     #TODO: Update to take a FoV
    #     assert target_extent[0][0] > 0 and target_extent[0][1] > 0
    #     # assert all(target < own for target, own in zip(target_extent[0], self.extent(target_extent[1])))
    #
    #     assert target_extent[1] == "rad"
    #
    #     horizontal_steps = np.append(np.arange(self.horizontal_min, self.horizontal_max, target_extent[0][0]),
    #                                  self.horizontal_max)
    #
    #     elevation_steps = np.append(np.arange(self.elevation_min, self.elevation_max, target_extent[0][1]),
    #                                 self.elevation_max)
    #
    #     horizontal_bins = list(zip(horizontal_steps[:-1], horizontal_steps[1:]))
    #     elevation_bins = list(zip(elevation_steps[:-1], elevation_steps[1:]))
    #
    #     tiles = []
    #     for hor_bin in horizontal_bins:
    #         horizontal_tiles = []
    #         for elev_bin in elevation_bins:
    #             horizontal_tiles.append(FoV(horizontal_min=hor_bin[0],
    #                                         elevation_min=elev_bin[0],
    #                                         horizontal_max=hor_bin[1],
    #                                         elevation_max=elev_bin[1]))
    #         tiles.append(horizontal_tiles)
    #     return tiles
    #
    def tile(self, target_extent: Self) -> list[list[Self]]:

        horizontal_steps = np.append(
            np.arange(self.horizontal_min, self.horizontal_max, target_extent.width()), self.horizontal_max
        )

        elevation_steps = np.append(
            np.arange(self.elevation_min, self.elevation_max, target_extent.height()), self.elevation_max
        )

        horizontal_bins = list(zip(horizontal_steps[:-1], horizontal_steps[1:]))
        elevation_bins = list(zip(elevation_steps[:-1], elevation_steps[1:]))

        tiles = []
        for hor_bin in horizontal_bins:
            if hor_bin[-1] - hor_bin[0] <= 0:
                continue
            horizontal_tiles = []
            for elev_bin in elevation_bins:
                if elev_bin[-1] - elev_bin[0] <= 0:
                    continue
                new_fov = type(self)(
                    horizontal_min=hor_bin[0],
                    elevation_min=elev_bin[0],
                    horizontal_max=hor_bin[1],
                    elevation_max=elev_bin[1],
                )
                if all(e > EPS for e in new_fov.extent()):
                    horizontal_tiles.append(new_fov)
            if horizontal_tiles:
                tiles.append(horizontal_tiles)
        return tiles

    def quadrants(self):
        # Keep for legacy
        return tuple(self.split(shape=(2, 2)))

    @classmethod
    def merge(cls, fovs: Iterable[Self]) -> Self:
        min_horizontal_min = min(fovs, key=lambda fov: fov.horizontal_min).horizontal_min
        min_elevation_min = min(fovs, key=lambda fov: fov.elevation_min).elevation_min
        max_horizontal_max = max(fovs, key=lambda fov: fov.horizontal_max).horizontal_max
        max_elevation_max = max(fovs, key=lambda fov: fov.elevation_max).elevation_max

        return cls(
            horizontal_min=min_horizontal_min,
            elevation_min=min_elevation_min,
            horizontal_max=max_horizontal_max,
            elevation_max=max_elevation_max,
        )


@dataclass(init=True, frozen=True)
class _OldFoVTree:
    """
    Represents a hierarchical tree structure for spatial partitioning of FoVs.

    Attributes
    ----------
    identifier : str
        A unique identifier for this tree node.
    node : FoV
        The FoV associated with this tree node.
    children : Optional[dict[str, FoVTree]]
        A dictionary of child nodes, if any.
    """

    identifier: str
    node: FoV
    children: Optional[dict[str, Self]] = field(default_factory=dict)

    @staticmethod
    def add_identifier(fovs: list[FoV], shape: tuple[int, int]):
        identifier_length = np.ceil(math.log(shape[0] * shape[1], 16)).astype(int)
        return tuple(
            [(((identifier_length - len(hex_str := f"{i:x}")) * "0" + hex_str), fov) for i, fov in enumerate(fovs)]
        )

    def depth(self) -> int:
        if not self.children:
            return 1
        return max([c.depth() for c in self.children.values()]) + 1

    def to_list(self) -> list[tuple[str, FoV]]:
        if self.is_leaf():
            return [(self.identifier, self.node)]

        children_lists = [c.to_list() for c in self.children.values()]

        return list(chain.from_iterable(children_lists))

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
    #         fov_children = {child_identifier: cls.build_by_splitting(child, target_ratio, target_fov_extent,
    #                                                                  max_denominator * 2, identifier + child_identifier)
    #                         for child_identifier, child in cls.add_identifier(fov_splits, shape)}
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

    @classmethod
    def build_from_tiles(cls, tiles: list[list[FoV]], min_children: int = 4, identifier: str = "") -> Self:
        """
        Constructs a tree from a grid of FoVs.

        Parameters
        ----------
        tiles : list[list[FoV]]
            A grid of FoVs to organize into a tree.
        min_children : int, default=4
            The minimum number of children to avoid further splitting.
        identifier : str, default=""
            The identifier for the root node.

        Returns
        -------
        FoVTree
            The root of the constructed tree.
        """
        assert min_children > 1
        if not tiles or not tiles[0]:
            return None

        # Todo: Check this logic!
        if len(tiles) == 1 and len(tiles[0]) == 1:
            if identifier == "":
                return cls("root", tiles[0][0], None)
            return cls(identifier, tiles[0][0], None)

        fov = _OldFoV(
            horizontal_min=tiles[0][0].horizontal_min,
            elevation_min=tiles[0][0].elevation_min,
            horizontal_max=tiles[-1][-1].horizontal_max,
            elevation_max=tiles[-1][-1].elevation_max,
        )

        if len(tiles) * len(tiles[0]) <= min_children:
            flat_tiles = [tile for row in tiles for tile in row]
            fov_children = {str(i): cls(identifier + str(i), tile, {}) for i, tile in enumerate(flat_tiles)}
            return cls(identifier, fov, fov_children)  # TODO: BUG! Rebuild identifier function to work 2D

        q0 = tiles[: len(tiles) // 2]
        q1 = tiles[len(tiles) // 2 :]
        q00 = [row[: len(row) // 2] for row in q0]
        q01 = [row[len(row) // 2 :] for row in q0]
        q10 = [row[: len(row) // 2] for row in q1]
        q11 = [row[len(row) // 2 :] for row in q1]

        fov_children = {
            "0": cls.build_from_tiles(q00, min_children, identifier=(identifier + "0")),
            "1": cls.build_from_tiles(q01, min_children, identifier=(identifier + "1")),
            "2": cls.build_from_tiles(q10, min_children, identifier=(identifier + "2")),
            "3": cls.build_from_tiles(q11, min_children, identifier=(identifier + "3")),
        }

        fov_children = {k: v for k, v in fov_children.items() if v is not None}

        return cls(identifier, fov, fov_children)

    # @staticmethod
    # def quadrant_split(tiles: list[list[FoV]]):
    #     # q1 = tiles[:len(tiles) // 2]
    #     # q2 = tiles[len(tiles) // 2:]
    #     # q11 = [row[:len(row)//2] for row in q1]
    #     # q12 = [row[len(row)//2:] for row in q1]
    #     # q21 = [row[:len(row) // 2] for row in q2]
    #     # q22 = [row[len(row) // 2:] for row in q2]
    #     # quadrant_FoV = FoV(horizontal_min=q11[0][0].horizontal_min,
    #     #                    elevation_min=q11[0][0].elevation_min,
    #     #                    horizontal_max=q22[-1][-1].horizontal_max,
    #     #                    elevation_max=q22[-1][-1].elevation_max)
    #
    #     FoVTree.quadrant_split(q11)
    #     FoVTree.quadrant_split(q12)
    #     FoVTree.quadrant_split(q21)
    #     FoVTree.quadrant_split(q22)
    #
    #     pass

    def __getitem__(self, identifier: str) -> Self:
        # TODO: extend to complete for full string
        if not identifier or identifier == "root":
            return self
        child_identifier_length = np.ceil(math.log(len(self.children), 16)).astype(int)
        if len(identifier) > child_identifier_length:
            return self.children[identifier[:child_identifier_length]][identifier[child_identifier_length:]]

        return self.children[identifier]

    def is_leaf(self):
        return not self.children

    @staticmethod
    def calculate_optimal_shape(
        fov: FoV, target_ratio: float, target_extent: tuple[float, float], max_denominator: float
    ) -> tuple[int, int]:

        shape = Fraction(fov.ratio() / target_ratio).limit_denominator(np.round(max_denominator).astype(int))
        shape = (
            shape.numerator,
            shape.denominator,
        )
        if shape[0] == shape[1] == 1:
            shape = (2, 2)

        return shape