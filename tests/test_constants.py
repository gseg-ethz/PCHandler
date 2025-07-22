import pytest
import numpy as np

from pchandler.constants import (
    DEFAULT_CONFIG,
    EPS,
    HALF_PI,
    PI,
    RGB_NAMES,
    NORMAL_NAMES,
    INTENSITY_NAMES,
    REFLECTANCE_NAMES,
    TWO_PI
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
    assert RGB_NAMES.base == "rgb"
    assert NORMAL_NAMES.base == "normals"
    assert INTENSITY_NAMES.base == "intensity"
    assert REFLECTANCE_NAMES.base == "reflectance"
