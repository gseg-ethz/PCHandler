from numpy import finfo, float32, pi
from pydantic import ConfigDict, validate_call

EPS = finfo(float32).eps

PI = pi
HALF_PI = pi * 0.5
TWO_PI = pi * 2

RGB_FIELD = "rgb"
NORMALS_FIELD = "normals"
INTENSITY_FIELD = "intensity"
REFLECTANCE_FIELD = "reflectance"

XYZ_FIELDS = ('x', 'y', 'z')

RGB_CHAR = ("r", "g", "b")
RGB_FLOAT = ("rf", "gf", "bf")
RGB_WORD = ("red", "green", "blue")
RGB_PARTIAL_NAMES = RGB_CHAR + RGB_WORD + RGB_FLOAT
RGB_FULL_NAMES = (RGB_FIELD, 'colour', 'colours', 'color', 'colors')
RGB_ALL_NAMES = RGB_FULL_NAMES + RGB_PARTIAL_NAMES

NORMALS_CHAR = ("nx", "ny", "nz")
NORMALS_WORD = ("normalx", "normaly", "normalz")
NORMAL_PARTIAL_NAMES = NORMALS_CHAR + NORMALS_WORD
NORMAL_FULL_NAMES = (NORMALS_FIELD, "normal", "normal_fields", "nxnynz")
NORMAL_ALL_NAMES = NORMAL_FULL_NAMES + NORMAL_PARTIAL_NAMES

INTENSITY_ALL_NAMES = (INTENSITY_FIELD, "intensities", "i")

REFLECTANCE_ALL_NAMES = (REFLECTANCE_FIELD,)

# TODO determine if str_to_lower should be in default config for validation functions/method
DEFAULT_CONFIG = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, str_to_lower=True)
VALIDATE_RETURN_CONFIG = DEFAULT_CONFIG | {'validate_return': True}

validate_variables = validate_call(config=VALIDATE_RETURN_CONFIG)
