# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

from typing import NamedTuple, Optional

__all__ = ['RGB_NAMES', 'NORMAL_NAMES', 'XYZ_NAMES', 'INTENSITY_NAMES', 'REFLECTANCE_NAMES',
           'COMMON_FIELD_NAMES', 'COMMON_FIELD_BASES']


TripletT = tuple[str, str, str]


class _NameConstantsSingle(NamedTuple):
    base: str
    char: Optional[str] = None
    extra_names: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.base,) + self.extra_names

    @property
    def all(self) -> tuple[str, ...]:
        if self.char:
            return self.names + (self.char,)
        return self.names


class _NameConstantsTriplet(NamedTuple):
    base: str
    char: TripletT
    extra_names: tuple[str, ...] = ()
    words: TripletT = ("", "", "")
    float: TripletT = ("", "", "")
    reverse: Optional[str] = None

    @property
    def names(self) -> tuple[str, ...]:
        return (self.base,) + self.extra_names

    @property
    def triplets(self) -> tuple[TripletT, ...]:
        triplets: tuple[TripletT, ...] = (self.char,)

        if self.words:
            triplets += (self.words,)

        if self.float:
            triplets += (self.float,)

        return triplets

    @property
    def scalars(self) -> tuple[str, ...]:
        return self.char + self.words + self.float

    @property
    def all(self) -> tuple[TripletT | str, ...]:
        if self.reverse:
            return self.names + self.scalars + (self.reverse,)
        return self.names + self.scalars

    @property
    def positional(self) -> tuple[tuple[str, ...], ...]:
        groups: list[list[str]] = [[], [], []]
        for triple in self.triplets:
            for i, value in enumerate(triple):
                groups[i].append(value)

        return tuple(tuple(group) for group in groups)

    def get_position(self, name):
        for i, positional_names in enumerate(self.positional):
            if name in positional_names:
                return i

        raise ValueError("Could not find name in positional names")

#: Pre-defined names for RGB field access
RGB_NAMES = _NameConstantsTriplet(
    base = "rgb",
    char = ("r", "g", "b"),
    words = ("red", "green", "blue"),
    float = ("rf", "gf", "bf"),
    extra_names = ("colour", "colours", "color", "colors"),
    reverse = "bgr"
)

#: Pre-defined names for Normals field access
NORMAL_NAMES = _NameConstantsTriplet(
    base="normals",
    char=("nx", "ny", "nz"),
    words=("normalx", "normaly", "normalz"),
    extra_names=("normal", "normal_fields", "nxnynz"),
    reverse="nznynx",
)

#: Pre-defined names for XYZ coordinate access
XYZ_NAMES = _NameConstantsTriplet(
    base="xyz",
    char=("x", "y", "z"),
    extra_names=("cartesian", "cartesians", "coordinates", "coordinate"),
    reverse="zyx",
)

#: Pre-defined names for Intensity field access
INTENSITY_NAMES = _NameConstantsSingle(
    base="intensity",
    char="i",
    extra_names=("intensities",))

#: Pre-defined names for Reflectance field access
REFLECTANCE_NAMES = _NameConstantsSingle(
    base="reflectance",
)

COMMON_FIELD_NAMES: tuple[_NameConstantsSingle | _NameConstantsTriplet, ...] = \
    (RGB_NAMES, NORMAL_NAMES, INTENSITY_NAMES, REFLECTANCE_NAMES)

COMMON_FIELD_BASES = (field.base for field in COMMON_FIELD_NAMES)
