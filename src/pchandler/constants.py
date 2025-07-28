from typing import Optional, NamedTuple

from numpy import finfo, float32, pi
from pydantic import ConfigDict, validate_call

EPS = finfo(float32).eps

PI = pi
HALF_PI = pi * 0.5
TWO_PI = pi * 2

TripletT = tuple[str, str, str]


class NameConstantsSingle(NamedTuple):
    base: str
    char: Optional[str] = None
    extra_names: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.base, ) + self.extra_names

    @property
    def all(self) -> tuple[str, ...]:
        if self.char:
            return self.names + (self.char,)
        return self.names


class NameConstantsTriplet(NamedTuple):
    base: str
    char: TripletT
    extra_names: tuple[str, ...] = ()
    words: TripletT = ("","","")
    float: TripletT = ("","","")
    reverse: Optional[str] = None

    @property
    def names(self) -> tuple[str, ...]:
        return (self.base,) + self.extra_names

    @property
    def triplets(self) -> tuple[TripletT, ...]:
        triplets: tuple[TripletT, ...] = (self.char, )

        if self.words:
            triplets += (self.words, )

        if self.float:
            triplets += (self.float, )

        return triplets

    @property
    def scalars(self) -> tuple[str, ...]:
        return self.char + self.words + self.float

    @property
    def all(self) -> tuple[TripletT|str, ...]:
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


RGB_NAMES = NameConstantsTriplet(
    base = "rgb",
    char = ("r", "g", "b"),
    words = ("red", "green", "blue"),
    float = ("rf", "gf", "bf"),
    extra_names = ("colour", "colours", "color", "colors"),
    reverse = "bgr"
)
NORMAL_NAMES = NameConstantsTriplet(
    base="normals",
    char=("nx", "ny", "nz"),
    words=("normalx", "normaly", "normalz"),
    extra_names=("normal", "normal_fields", "nxnynz"),
    reverse="nznynx"
)

XYZ_NAMES = NameConstantsTriplet(
    base="xyz",
    char=("x", "y", "z"),
    extra_names=("cartesian", "cartesians", "coordinates", "coordinate"),
)

INTENSITY_NAMES = NameConstantsSingle(
    base="intensity",
    char="i",
    extra_names=("intensities",)
)

REFLECTANCE_NAMES = NameConstantsSingle(
    base="reflectance",
)

COMMON_FIELD_NAMES: tuple[NameConstantsSingle|NameConstantsTriplet, ...] = \
    (RGB_NAMES, NORMAL_NAMES, INTENSITY_NAMES, REFLECTANCE_NAMES)

COMMON_FIELD_BASES = (field.base for field in COMMON_FIELD_NAMES)

# TODO determine if str_to_lower should be in default config for validation functions/method
DEFAULT_CONFIG = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, str_to_lower=True)
VALIDATE_RETURN_CONFIG = DEFAULT_CONFIG | {'validate_return': True}

validate_variables = validate_call(config=VALIDATE_RETURN_CONFIG)
