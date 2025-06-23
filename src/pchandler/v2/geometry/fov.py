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

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import chain
from typing import Iterable, Optional, cast, Self, Annotated, NamedTuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from pydantic import Field

from ..constants import EPS, TWO_PI, PI, HALF_PI
from ..util import AngleUnit, convert_angles

if TYPE_CHECKING:
    from .coordinates import SphericalCoordinates, CartesianCoordinates

logger = logging.getLogger(__name__.split(".")[0])

NumT = float|int|np.number|npt.NDArray

HzAngleT = Annotated[NumT, Field(ge=0, le=TWO_PI)]
ElevAngleT = Annotated[NumT, Field(ge=0, le=PI)]


# DISCUSS - do we want to keep angular units or use radians? Distance is also arbitrary
class FoV(NamedTuple):
    top: ElevAngleT
    bottom: ElevAngleT
    left: HzAngleT
    right: HzAngleT

    def __iter__(self) -> Iterable[HzAngleT|ElevAngleT]:
        yield self.right
        yield self.top
        yield self.left
        yield self.bottom

    @property
    def crosses_pi(self):
        return self.right > self.left

    @classmethod
    def from_spherical(cls, coordinates: SphericalCoordinates|CartesianCoordinates) -> Self:
        return cls(left=coordinates.hz.max(),
                   right=coordinates.hz.min(),
                   top=coordinates.v.min(),
                   bottom=coordinates.v.max())

    def width(self) -> HzAngleT:
        if self.crosses_pi:
            return TWO_PI - (self.right - self.left)
        return self.left - self.right

    def height(self) -> ElevAngleT:
        return self.bottom - self.top

    def extent(self) -> tuple[HzAngleT, ElevAngleT]:
        return self.width(), self.height()

    def center(self) -> tuple[HzAngleT, ElevAngleT]:
        horizontal_center = (self.left + self.right) / 2
        elevation_center = (self.top + self.bottom) / 2
        return horizontal_center, elevation_center

    @classmethod
    def from_center_with_extent(cls, centerpoint: tuple[float, float], extent: tuple[float, float]) -> Self:
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
        return cls(left=fov_min[0], right=fov_max[0], top=fov_min[1], bottom=fov_max[1] )

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
            right=min(self.right, fov2.right),
            left=min(self.left, fov2.left),
            bottom=max(self.bottom, fov2.bottom),
            top=max(self.top, fov2.top),
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
            f"({self.left:0.4f}, {self.right:0.4f}, "
            f"{self.top:0.4f}, {self.right:0.4f})"
        )

    def extend_to_ratio(self, ratio: float) -> Self:
        if self.ratio() - ratio > EPS:
            target_vertical_extent = self.extent()[0] / ratio
            new_fov = FoV(
                left=self.left,
                top=self.top,
                right=self.right,
                bottom=self.bottom + target_vertical_extent,
            )
        elif ratio - self.ratio() > EPS:
            target_horizontal_extent = self.extent()[1] * ratio
            new_fov = FoV(
                left=self.left,
                top=self.top,
                right=self.right + target_horizontal_extent,
                bottom=self.bottom,
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
            start=self.right, stop=self.left, num=shape[0] + 1, endpoint=True, retstep=False
        )
        elevation_borders = np.linspace(
            start=self.top, stop=self.bottom, num=shape[1] + 1, endpoint=True, retstep=False
        )

        fov_splits = [
            FoV(right=hor_min, top=elev_min, left=hor_max, bottom=elev_max)
            for hor_min, hor_max in zip(horizontal_borders[:-1], horizontal_borders[1:])
            for elev_min, elev_max in zip(elevation_borders[:-1], elevation_borders[1:])
        ]

        return fov_splits

    # DISCUSS is this still specialised from pc2img?
    def equal_tiles(self, target_extent: tuple[tuple[float, float], str]) -> list[Self]:
        assert target_extent[0][0] > 0 and target_extent[0][1]
        # assert any(target < own for target, own in zip(target_extent[0], self.extent(target_extent[1])))

        return self.split(
            shape=(
                np.ceil(self.extent(target_extent[1])[0] / target_extent[0][0]).astype(int),
                np.ceil(self.extent(target_extent[1])[1] / target_extent[0][1]).astype(int),
            )
        )

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
                new_fov = FoV(
                    right=hor_bin[0],
                    top=elev_bin[0],
                    left=hor_bin[1],
                    bottom=elev_bin[1],
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
        left = min(fovs, key=lambda fov: fov.left).left
        top = min(fovs, key=lambda fov: fov.top).top
        right = max(fovs, key=lambda fov: fov.right).right
        bottom = max(fovs, key=lambda fov: fov.bottom).bottom

        return cls(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        )

    @property
    def horizontal_min(self):
        return self.right

    @property
    def horizontal_max(self):
        return self.left

    @property
    def elevation_min(self):
        return self.top

    @property
    def elevation_max(self):
        return self.bottom

@dataclass(init=False, frozen=True)
class OldFoV:
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

        fov = FoV(
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
        fov: FoV, target_ratio: float, target_extent: tuple[tuple[float, float], str], max_denominator: float
    ) -> tuple[int, int]:

        shape = Fraction(fov.ratio() / target_ratio).limit_denominator(np.round(max_denominator).astype(int))
        shape = (
            shape.numerator,
            shape.denominator,
        )
        if shape[0] == shape[1] == 1:
            shape = (2, 2)

        return shape
