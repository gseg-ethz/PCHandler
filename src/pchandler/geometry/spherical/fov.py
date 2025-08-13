from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import chain
from typing import Any, Generator, Iterable, Optional, Self, TypeAlias, cast

import numpy as np
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

from GSEGUtils.base_types import VectorT
from pchandler.geometry.spherical import Angle, AngleArray

__all__ = ['FoV', 'FoVTree']

logger = logging.getLogger(__name__.split(".")[0])

AngleLikeT: TypeAlias = Angle | float | str
v_limits = (float(0 - EPS), float(PI + EPS))
hz_limits = (float(-PI - EPS), float(PI + EPS))


class FoV(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    left: Angle = Field(..., description="Hz ∈ [–π, +π]")
    right: Angle = Field(..., description="Hz ∈ [–π, +π]")
    top: Angle = Field(..., description="V ∈ [0, +π]")
    bottom: Angle = Field(..., description="V ∈ [0, +π]")

    def __init__(self, *, left: AngleLikeT, top: AngleLikeT, right: AngleLikeT, bottom: AngleLikeT):
        """
        Class containing the angular limits defining a field of view based in spherical coordinates.

        Parameters
        ----------
        left: Angle
            Hz ∈ [–π, +π]
        right: Angle
            Hz ∈ [–π, +π]
        top: Angle
            V ∈ [0, +π]
        bottom: Angle
            V ∈ [0, +π]
        """
        super().__init__(left=left, top=top, right=right, bottom=bottom)

    @field_validator("left", "right", "top", "bottom", mode="before")
    @classmethod
    def _coerce_to_angle(cls, value: AngleLikeT) -> Angle:
        return Angle.parse(value)

    @field_validator("left", "right", mode="after")
    def _check_hz(cls, hz: Angle) -> Angle:
        if not -np.pi - EPS <= hz.internal_value <= np.pi + EPS:
            raise ValueError(f"Horizontal angle {hz.radians} not in [-π, π]")
        return hz

    @field_validator("top", "bottom", mode="after")
    def _check_elevation(cls, v: Angle) -> Angle:
        if not 0 - EPS <= v.internal_value <= np.pi + EPS:
            raise ValueError(f"Top angle {v.radians} not in [0, π]")
        return v

    @model_validator(mode="after")
    def _check_bottom_and_top(self):
        if self.top > self.bottom:
            raise ValueError(f"Top ({self.top.radians}) must be smaller than bottom ({self.bottom.radians})")

    @classmethod
    def construct_without_bounds_check(
        cls, *, left: AngleLikeT, right: AngleLikeT, top: AngleLikeT, bottom: AngleLikeT
    ) -> Self:
        """Constuct a FoV without bounds check.

        Enables construction of FoV which cross over the FoV angular bounds

        Parameters
        ----------
        left : AngleLikeT
        right : AngleLikeT
        top : AngleLikeT
        bottom : AngleLikeT

        Returns
        -------
        FoV
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
        return cls(left=horizontal.min(), top=vertical.min(), right=horizontal.max(), bottom=vertical.max())

    def __iter__(self) -> Generator[tuple[str, Angle], None, None]:
        yield "left", self.left
        yield "top", self.top
        yield "right", self.right
        yield "bottom", self.bottom

    @property
    def crosses_pi(self) -> bool:
        """Check if the FoV horizontal range crosses the boundary +/- PI

        Returns
        -------
        bool
        """
        return self.left > self.right

    def width(self) -> Angle:
        """Return the angular width (horizontal angle) of the FoV

        Returns
        -------
        Angle
        """
        if self.crosses_pi:
            return Angle(TWO_PI) - (self.left - self.right)
        return self.right - self.left

    def height(self) -> Angle:
        """Return the angular height (vertical angle) of the FoV

        Returns
        -------
        Angle
        """
        return self.bottom - self.top

    def extent(self) -> tuple[Angle, Angle]:
        """Return the angular extent (width, height) of the FoV

        Returns
        -------
        tuple[Angle, Angle]
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
        """Creates an FoV instance from a center point and angular extent.

        Parameters
        ----------
        centerpoint: tuple[float, float]
        extent: tuple[float, float]

        Returns
        -------
        FoV
        """
        # hz_min = centerpoint[0] - extent[0] / 2
        # hz_max = centerpoint[0] + extent[0] / 2
        # v_min = centerpoint[1] - extent[1] / 2
        new_instance = cls.construct_without_bounds_check(
            left=centerpoint[0] - extent[0] / 2,
            right=centerpoint[0] + extent[0] / 2,
            top=centerpoint[1] - extent[1] / 2,
            bottom=centerpoint[1] + extent[1] / 2,
        )
        return new_instance

        # def union(self, fov2: Self) -> Self:

    def union(self, fov2: Self) -> Self:
        """Returns the union of this FoV with another.

        Parameters
        ----------
        fov2 : FoV

        Returns
        -------
        FoV
        """
        return type(self)(
            left=min(self.left, fov2.left),
            top=min(self.top, fov2.top),
            right=max(self.right, fov2.right),
            bottom=max(self.bottom, fov2.bottom),
        )

    def intersect(self, fov2: Self) -> Self:
        """Returns the intersection of this FoV with another.

        Parameters
        ----------
        fov2 : FoV

        Returns
        -------
        FoV
        """
        return type(self)(
            left=max(self.left, fov2.left),
            top=max(self.top, fov2.top),
            right=min(self.right, fov2.right),
            bottom=min(self.bottom, fov2.bottom),
        )

    @validate_variables
    def ratio(self) -> NonNegativeFloat:
        """Returns the width-to-height ratio of the FoV.

        Returns
        -------
        NonNegativeFloat
        """
        return self.width() / self.height()

    @validate_call(config=DEFAULT_CONFIG)
    def extend_to_ratio(self, ratio: float) -> Self:
        """Extends the FoV to a specified ratio.

        Parameters
        ----------
        ratio: float

        Returns
        -------
        FoV
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

        horizontal_borders = cast(
            AngleArray,
            Angle(
                np.linspace(
                    start=self.left.radians, stop=self.right.radians, num=shape[0] + 1, endpoint=True, retstep=False
                )
            ),
        )
        vertical_borders = cast(
            AngleArray,
            Angle(
                np.linspace(
                    start=self.top.radians, stop=self.bottom.radians, num=shape[1] + 1, endpoint=True, retstep=False
                )
            ),
        )
        # TODO check if a bug on AngleArray initialisation when used in place above
        # horizontal_borders.display_unit = self.left.display_unit
        # vertical_borders.display_unit = self.top.display_unit

        fov_splits = [
            type(self)(left=hor_min, top=elev_min, right=hor_max, bottom=elev_max)
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

        # TODO check if these functions should actually be AngleArray and default units
        horizontal_steps = cast(
            AngleArray, Angle(np.append(np.arange(self.left, self.right, target_extent.width()), self.right))
        )

        elevation_steps = cast(
            AngleArray, Angle(np.append(np.arange(self.top, self.bottom, target_extent.height()), self.bottom))
        )

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
                new_fov = type(self)(
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
        """Splits the current object into four quadrants.

        Returns
        -------
        tuple of Self
        """
        # Keep for legacy
        return tuple(self.split(shape=(2, 2)))

    @classmethod
    def merge(cls, fovs: Iterable[Self]) -> Self:
        """Merges multiple FoV returning a single FoV that encompasses the total area covered
        Parameters
        ----------
        fovs : Iterable[FoV]
            All FoV objects to be merged.

        Returns
        -------
        FoV
        """
        return cls(
            left=min(fovs, key=lambda fov: fov.left).left,
            top=min(fovs, key=lambda fov: fov.top).top,
            right=max(fovs, key=lambda fov: fov.right).right,
            bottom=max(fovs, key=lambda fov: fov.bottom).bottom,
        )

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
        left, top, right, bottom = self.left, self.top, self.right, self.bottom
        return f"{self.__class__.__name__}({left=!s}, {right=!s}, {top=!s}, {bottom=!s})"

    def __repr__(self):
        return str(self)


@dataclass(init=True, frozen=True)
class FoVTree:
    """A hierarchical tree structure for spatial partitioning of FoVs

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
        """Constructs a tree from a grid of FoVs.

        Parameters
        ----------
        tiles : list[list[FoV]]
            Grid of FoVs
        min_children : int, default=4
            The minimum number of children to avoid further splitting.
        identifier : str, default=""
            Unique identifier for this node.

        Returns
        -------
        FoVTree
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

    def __getitem__(self, identifier: str) -> Self:
        # TODO: extend to complete for full string
        if not identifier or identifier == "root" or self.children is None:
            return self

        child_identifier_length = np.ceil(math.log(len(self.children), 16)).astype(int)

        if len(identifier) > child_identifier_length:
            return self.children[identifier[:child_identifier_length]][identifier[child_identifier_length:]]

        return self.children[identifier]

    def is_leaf(self):
        """Check if the FoVTree node is a leaf (no child nodes)

        Returns
        -------
        bool
            True if the node has no children, False otherwise.
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
