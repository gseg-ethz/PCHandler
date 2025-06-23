from numpy import finfo, float32, pi
from pydantic import ConfigDict

EPS = finfo(float32).eps

PI = pi
HALF_PI = pi * 0.5
TWO_PI = pi * 2

RGB_FIELD = "rgb"
NORMALS_FIELD = "normals"
INTENSITY_FIELD = "intensity"
REFLECTANCE_FIELD = "reflectance"

RGB_CHAR = ("r", "g", "b")
RGB_WORD = ("red", "green", "blue")
RGB_PARTIAL_NAMES = RGB_CHAR + RGB_WORD
RGB_ALL_POTENTIAL_NAMES = (RGB_FIELD, "bgr", "rgba") + RGB_PARTIAL_NAMES
NORMALS_CHAR = ("nx", "ny", "nz")
NORMALS_WORD = ("normalx", "normaly", "normalz")
NORMAL_PARTIAL_NAMES = NORMALS_CHAR + NORMALS_WORD
NORMAL_POTENTIAL_NAMES = (NORMALS_FIELD, "normal", "normal_fields") + NORMAL_PARTIAL_NAMES
INTENSITY_POTENTIAL_NAMES = (INTENSITY_FIELD, "intensities", "i")
REFLECTANCE_POTENTIAL_NAMES = (REFLECTANCE_FIELD, )

DEFAULT_CONFIG = ConfigDict(arbitrary_types_allowed=True)
