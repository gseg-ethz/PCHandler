from pchandler.constants import (
    INTENSITY_NAMES,
    NORMAL_NAMES,
    REFLECTANCE_NAMES,
    RGB_NAMES,
)


def test_field_names():
    assert RGB_NAMES.base == "rgb"
    assert NORMAL_NAMES.base == "normals"
    assert INTENSITY_NAMES.base == "intensity"
    assert REFLECTANCE_NAMES.base == "reflectance"
