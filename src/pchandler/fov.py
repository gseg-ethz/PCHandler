import sys

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import cast
if sys.version[0] == 3 and sys.version_info[1] >= 11:
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np

from pchandler.util import AngleUnit, convert_angles




@dataclass(init=False, frozen=True)
class FoV:
    # TODO: Rework quadrants as special case of generalized split on shape tuple.
    horizontal_min: float
    elevation_min: float
    horizontal_max: float
    elevation_max: float
    # _defined_units: tuple[str, ...] = ("rad", "gon", "deg")
    __internal_angular_unit: AngleUnit = AngleUnit.RAD

    def __init__(self, *, horizontal_min: float, elevation_min: float, horizontal_max: float, elevation_max: float,
                 unit: [str | AngleUnit] = "rad"):
        assert horizontal_max > horizontal_min  # TODO: Rethink in the context describing shortest path
        assert elevation_max > elevation_min

        input_unit = AngleUnit(unit)
        values = np.array([horizontal_min, elevation_min, horizontal_max, elevation_max], dtype=float)
        convert_angles(values, source_unit=input_unit, target_unit=self.__internal_angular_unit, out=values)

        object.__setattr__(self, "horizontal_min", values[0])
        object.__setattr__(self, "elevation_min", values[1])
        object.__setattr__(self, "horizontal_max", values[2])
        object.__setattr__(self, "elevation_max", values[3])

    def as_numpy(self, unit: [str | AngleUnit] = "rad") -> np.ndarray:
        unit = AngleUnit(unit)
        values = np.array([self.horizontal_min, self.elevation_min, self.horizontal_max, self.elevation_max])
        convert_angles(values, source_unit=self.__internal_angular_unit, target_unit=unit, out=values)
        return values

    def as_tuple(self, unit: [str | AngleUnit] = "rad") -> tuple[float, float, float, float]:
        return tuple(self.as_numpy(unit=unit))

    def as_dict(self, unit: [str | AngleUnit] = "rad") -> dict[str, float]:
        values = self.as_numpy(unit=unit)
        return {"horizontal_min": cast(float, values[0]),
                "elevation_min": cast(float, values[1]),
                "horizontal_max": cast(float, values[2]),
                "elevation_max": cast(float, values[3])}

    def width(self, unit: [str | AngleUnit] = "rad") -> float:
        values = self.as_dict(unit=unit)
        return values["horizontal_max"] - values["horizontal_min"]

    def height(self, unit: [str | AngleUnit] = "rad") -> float:
        values = self.as_dict(unit=unit)
        return values["elevation_max"] - values["elevation_min"]

    def extent(self, unit: [str | AngleUnit] = "rad") -> tuple[float, float]:
        return self.width(unit=unit), self.height(unit=unit)

    def union(self, fov2: Self) -> Self:
        return FoV(horizontal_min=min(self.horizontal_min, fov2.horizontal_min),
                   elevation_min=min(self.elevation_min, fov2.elevation_min),
                   horizontal_max=max(self.horizontal_max, fov2.horizontal_max),
                   elevation_max=max(self.elevation_max, fov2.elevation_max))

    def ratio(self) -> float:
        return self.extent()[0] / self.extent()[1]

    def __repr__(self):
        values = self.as_dict(unit=AngleUnit.GON)
        return f"({values['horizontal_min']:0.4f}, {values['elevation_min']:0.4f}, " \
               f"{values['horizontal_max']:0.4f}, {values['elevation_max']:0.4f})"

    def extend_to_ratio(self, ratio: float) -> Self:
        if self.ratio() > ratio:
            target_vertical_extent = self.extent()[0] / ratio
            new_fov = FoV(horizontal_min=self.horizontal_min,
                          elevation_min=self.elevation_min,
                          horizontal_max=self.horizontal_max,
                          elevation_max=self.elevation_min + target_vertical_extent)
        elif self.ratio() < ratio:
            target_horizontal_extent = self.extent()[1] * ratio
            new_fov = FoV(horizontal_min=self.horizontal_min,
                          elevation_min=self.elevation_min,
                          horizontal_max=self.horizontal_min + target_horizontal_extent,
                          elevation_max=self.elevation_max)
        else:
            new_fov = self

        return new_fov

    def split(self, shape: tuple[int, int]) -> list[Self]:
        assert shape[0] > 0 and shape[1] > 0
        if shape[0] == shape[1] == 1:
            return [self]

        horizontal_borders = np.linspace(start=self.horizontal_min,
                                         stop=self.horizontal_max,
                                         num=shape[0] + 1,
                                         endpoint=True,
                                         retstep=False)
        elevation_borders = np.linspace(start=self.elevation_min,
                                        stop=self.elevation_max,
                                        num=shape[1] + 1,
                                        endpoint=True,
                                        retstep=False)

        fov_splits = [FoV(horizontal_min=hor_min,
                          elevation_min=elev_min,
                          horizontal_max=hor_max,
                          elevation_max=elev_max)
                      for hor_min, hor_max in zip(horizontal_borders[:-1], horizontal_borders[1:])
                      for elev_min, elev_max in zip(elevation_borders[:-1], elevation_borders[1:])]

        return fov_splits

    def equal_tiles(self, target_extent: tuple[tuple[float, float], str], ) -> list[Self]:
        assert target_extent[0][0] > 0 and target_extent[0][1]
        # assert any(target < own for target, own in zip(target_extent[0], self.extent(target_extent[1])))

        return self.split(shape=(np.ceil(self.extent(target_extent[1])[0] / target_extent[0][0]).astype(int),
                                 np.ceil(self.extent(target_extent[1])[1] / target_extent[0][1]).astype(int)))

    def tile(self, target_extent: tuple[tuple[float, float], str], ) -> list[list[Self], ...]:
        #TODO: Update to take a FoV
        assert target_extent[0][0] > 0 and target_extent[0][1] > 0
        # assert all(target < own for target, own in zip(target_extent[0], self.extent(target_extent[1])))

        assert target_extent[1] == "rad"

        horizontal_steps = np.append(np.arange(self.horizontal_min, self.horizontal_max, target_extent[0][0]),
                                     self.horizontal_max)

        elevation_steps = np.append(np.arange(self.elevation_min, self.elevation_max, target_extent[0][1]),
                                    self.elevation_max)

        horizontal_bins = list(zip(horizontal_steps[:-1], horizontal_steps[1:]))
        elevation_bins = list(zip(elevation_steps[:-1], elevation_steps[1:]))

        tiles = []
        for hor_bin in horizontal_bins:
            horizontal_tiles = []
            for elev_bin in elevation_bins:
                horizontal_tiles.append(FoV(horizontal_min=hor_bin[0],
                                            elevation_min=elev_bin[0],
                                            horizontal_max=hor_bin[1],
                                            elevation_max=elev_bin[1]))
            tiles.append(horizontal_tiles)
        return tiles

    def quadrants(self):
        # Keep for legacy
        return tuple(self.split(shape=(2, 2)))


@dataclass(init=True, frozen=True)
class FoVTree:
    identifier: str
    node: FoV
    children: dict[str, Self] = field(default_factory=dict)

    @staticmethod
    def add_identifier(fovs: list[FoV], shape: tuple[int, int]):
        identifier_length = np.ceil(math.log(shape[0] * shape[1], 16)).astype(int)
        return tuple([(((identifier_length - len(hex_str := f"{i:x}")) * "0" + hex_str), fov)
                      for i, fov in enumerate(fovs)])

    def depth(self) -> int:
        if not self.children:
            return 1
        return max([c.depth() for c in self.children.values()]) + 1

    @classmethod
    def build_by_splitting(cls, fov: FoV, target_ratio: float, target_fov_extent: tuple[tuple[float, float], str],
                           max_denominator: int, identifier: str = "") -> Self:
        # TODO: Rework stopping criteria
        assert target_fov_extent[1] in ("rad", "gon", "deg")

        target_extent = target_fov_extent[0]
        angle_unit = target_fov_extent[1]

        shape = cls.calculate_optimal_shape(fov, target_ratio, target_fov_extent, max_denominator)

        if (fov.extent(unit=angle_unit)[0] < target_extent[0] * shape[0] or
                fov.extent(unit=angle_unit)[1] < target_extent[1] * shape[1]):
            fov_tiles = fov.equal_tiles(target_fov_extent)
            shape = (len(fov_tiles), 1)
            fov_children = {child_identifier: cls(identifier + child_identifier, child, {})
                            for child_identifier, child in cls.add_identifier(fov_tiles, shape)}
        else:
            fov_splits = fov.split(shape)
            fov_children = {child_identifier: cls.build_by_splitting(child, target_ratio, target_fov_extent,
                                                                     max_denominator * 2, identifier + child_identifier)
                            for child_identifier, child in cls.add_identifier(fov_splits, shape)}

        return cls(identifier, fov, fov_children)

    @classmethod
    def build_by_tiling(cls, fov: FoV, target_fov_extent: tuple[tuple[float, float], str],
                        identifier: str = "") -> Self:
        fov_tiles = fov.equal_tiles(target_fov_extent)

        identifier_length = np.ceil(math.log(len(fov_tiles), 16)).astype(int)
        fov_with_identifier = tuple([(((identifier_length - len(hex_str := f"{i:x}")) * "0" + hex_str), fov)
                                     for i, fov in enumerate(fov_tiles)])

        fov_children = {child_identifier: cls(identifier + child_identifier, child, {})
                        for child_identifier, child in fov_with_identifier}

        return cls(identifier, fov, fov_children)

    # @classmethod
    # def build(cls, fov: FoV, target_fov_extent: FoV):
    #     tiles = fov.tile((target_fov_extent.extent("rad"), "rad"))
    #     pass

    @classmethod
    def build_from_tiles(cls, tiles: list[list[FoV]], min_children: int = 4, identifier: str = "") -> Self:
        assert min_children > 1
        if not tiles or not tiles[0]:
            return None

        fov = FoV(horizontal_min=tiles[0][0].horizontal_min,
                  elevation_min=tiles[0][0].elevation_min,
                  horizontal_max=tiles[-1][-1].horizontal_max,
                  elevation_max=tiles[-1][-1].elevation_max)

        if len(tiles) * len(tiles[0]) <= min_children:
            flat_tiles = [tile for row in tiles for tile in row]
            fov_children = {str(i): cls(identifier+str(i), tile, {})
                            for i, tile in enumerate(flat_tiles)}
            return cls(identifier, fov, fov_children) #TODO: BUG! Rebuild identifier function to work 2D

        q0 = tiles[:len(tiles) // 2]
        q1 = tiles[len(tiles) // 2:]
        q00 = [row[:len(row) // 2] for row in q0]
        q01 = [row[len(row) // 2:] for row in q0]
        q10 = [row[:len(row) // 2] for row in q1]
        q11 = [row[len(row) // 2:] for row in q1]

        fov_children = {"0": cls.build_from_tiles(q00, min_children, identifier=(identifier + "0")),
                        "1": cls.build_from_tiles(q01, min_children, identifier=(identifier + "1")),
                        "2": cls.build_from_tiles(q10, min_children, identifier=(identifier + "2")),
                        "3": cls.build_from_tiles(q11, min_children, identifier=(identifier + "3")),
                        }

        fov_children = {k: v for k, v in fov_children.items() if v is not None}

        return cls(identifier, fov, fov_children)

    @staticmethod
    def quadrant_split(tiles: list[list[FoV]]):
        # q1 = tiles[:len(tiles) // 2]
        # q2 = tiles[len(tiles) // 2:]
        # q11 = [row[:len(row)//2] for row in q1]
        # q12 = [row[len(row)//2:] for row in q1]
        # q21 = [row[:len(row) // 2] for row in q2]
        # q22 = [row[len(row) // 2:] for row in q2]
        # quadrant_FoV = FoV(horizontal_min=q11[0][0].horizontal_min,
        #                    elevation_min=q11[0][0].elevation_min,
        #                    horizontal_max=q22[-1][-1].horizontal_max,
        #                    elevation_max=q22[-1][-1].elevation_max)

        FoVTree.quadrant_split(q11)
        FoVTree.quadrant_split(q12)
        FoVTree.quadrant_split(q21)
        FoVTree.quadrant_split(q22)

        pass

    def __getitem__(self, identifier: str):
        # TODO: extend to complete for full string
        child_identifier_length = np.ceil(math.log(len(self.children), 16)).astype(int)
        if len(identifier) > child_identifier_length:
            return self.children[identifier[:child_identifier_length]][identifier[child_identifier_length:]]

        return self.children[identifier]

    def is_leaf(self):
        return not self.children

    @staticmethod
    def calculate_optimal_shape(fov: FoV, target_ratio: float,
                                target_extent: tuple[tuple[float, float], str],
                                max_denominator: float) -> tuple[int, int]:

        shape = Fraction(fov.ratio() / target_ratio).limit_denominator(np.round(max_denominator).astype(int))
        shape = (shape.numerator, shape.denominator,)
        if shape[0] == shape[1] == 1:
            shape = (2, 2)

        return shape
