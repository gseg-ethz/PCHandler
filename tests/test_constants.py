import numpy as np
import pytest

from pchandler.v2.constants import (
    DEFAULT_CONFIG,
    EPS,
    HALF_PI,
    NORMAL_ALL_NAMES,
    NORMALS_FIELD,
    PI,
    RGB_FIELD,
    RGB_ALL_NAMES,
    TWO_PI, INTENSITY_FIELD, REFLECTANCE_FIELD,
)


def test_eps():
    assert np.finfo(np.float32).eps == EPS


def test_PI_values():
    assert PI == np.pi
    assert HALF_PI == np.pi / 2
    assert TWO_PI == np.pi * 2


def test_default_pydantic_config():
    assert DEFAULT_CONFIG.get("arbitrary_types_allowed", False)


def test_field_names():
    assert RGB_FIELD == "rgb"
    assert NORMALS_FIELD == "normals"
    assert INTENSITY_FIELD == "intensity"
    assert REFLECTANCE_FIELD == "reflectance"
