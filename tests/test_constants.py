from pchandler.constants import (
    RGB_NAMES,
    NORMAL_NAMES,
    INTENSITY_NAMES,
    REFLECTANCE_NAMES,
)


def test_field_names():
    assert RGB_NAMES.base == "rgb"
    assert NORMAL_NAMES.base == "normals"
    assert INTENSITY_NAMES.base == "intensity"
    assert REFLECTANCE_NAMES.base == "reflectance"
