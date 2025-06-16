from numpy import pi, finfo, float32

from pydantic import ConfigDict

EPS = finfo(float32).eps

PI = pi
HALF_PI = pi * 0.5
TWO_PI = pi * 2

RGB_FIELD = 'rgb'
NORMALS_FIELD = 'normals'
RGB_POTENTIAL_NAMES = ('r', 'g', 'b', 'rgb', 'bgr', 'red', 'green', 'blue', 'rgba')
NORMAL_POTENTIAL_NAMES = ('normal', 'normals', 'normal_fields', 'nx', 'ny', 'nz')

DEFAULT_CONFIG = ConfigDict(arbitrary_types_allowed=True)
