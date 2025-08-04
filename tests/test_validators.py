import pytest
from abc import ABC
from typing import Callable

import numpy as np

from GSEGUtils.constants import HALF_PI, PI, TWO_PI
from GSEGUtils.validators import (
    validate_azimuth_angles,
    validate_inclination_angles,
    validate_spherical_angles,
    validate_horizontal_angles,
    validate_zenith_angles,
    validate_radius,
    coerce_wrapped_azimuth_angles,
    coerce_wrapped_horizontal_angles,
    validate_transposed_2d_array,
    convert_slice_to_integer_range,
    validate_in_range,
normalize_uint8,
normalize_uint16,
normalize_int8,
normalize_int16,
normalize_int32,
normalize_int64
)

COORDINATE_3D_PROPERTIES = ("x", "y", "z", "r", "v", "hz", "rho", "theta", "phi", "xyz", "spher")


class BaseAngleTestClass(ABC):
    main_test: Callable | None = None

    @pytest.mark.parametrize(
        "values, func",
        [
            ("str", main_test),
            ({"angle": 74}, main_test),
            ({1.3, 0.2, -1.3}, main_test),
        ],
    )
    def test_invalid_types(self, values: np.ndarray, func: Callable):
        with pytest.raises(TypeError):
            func(values)


class TestValidation(BaseAngleTestClass):
    class TestHzAngle(BaseAngleTestClass):
        main_test: Callable = validate_horizontal_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, np.pi, 1.7, -np.pi]), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 2 * np.pi, 1.7, -np.pi - 3]), main_test),
                (float(72), main_test),
                (int(-44), main_test),
                (np.array([0, 1.3, np.pi, np.pi * 1.5, np.pi*2]), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestAzimuthAngle(BaseAngleTestClass):
        main_test: Callable = validate_azimuth_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, np.pi, 1.7, 2 * np.pi]), main_test),
                (np.linspace(0, TWO_PI, 1000, endpoint=True), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 3 * np.pi, 1.7, -np.pi - 3]), main_test),
                (np.linspace(-HALF_PI, HALF_PI, 1000, endpoint=True), main_test),
                (float(72), main_test),
                (int(-44), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestZenithAngles(BaseAngleTestClass):
        main_test: Callable = validate_zenith_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, np.pi, 1.7, np.pi]), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 3 * np.pi, 1.7, -np.pi - 3]), main_test),
                ("not an array", main_test),
                (float(72), main_test),
                (np.array([-1.1, 0.5, 1.3]), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestInclinationAngles(BaseAngleTestClass):
        main_test: Callable = validate_inclination_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 1.3, -np.pi / 2, np.pi / 2]), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, np.pi, 1.7, -np.pi]), main_test),
                (np.array([0, np.pi, 1.7, 2*np.pi]), main_test),
                (float(72), main_test),
                (int(-44), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestRadiusDistance(BaseAngleTestClass):
        main_test: Callable = validate_radius

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([0, 1.3, 200, 300000.21323]), main_test),
                (np.random.rand(100, 100), main_test),
                (np.zeros((100, 100)), main_test),
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (np.array([-1.3, -2, np.inf, -np.pi]), main_test),
                (np.random.rand(100, 100) * -1, main_test),
                ([1.3, 2000, 23445.123], main_test),
                ((1.3, 2000, 23445.123), main_test),
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestSphericalCoordinates(BaseAngleTestClass):
        main_test: Callable = validate_spherical_angles

        @pytest.mark.parametrize(
            "values, func",
            [
                (
                    np.array([[np.random.rand(10) * 100, np.random.rand(10) * TWO_PI - PI, np.random.rand(10) * PI]]),
                    main_test,
                )
            ],
        )
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize(
            "values, func",
            [
                (
                    np.array(
                        [
                            np.random.rand(10) * 100 - 50,
                            np.random.rand(10) * TWO_PI - PI,
                            np.random.rand(10) * PI,
                        ]
                    ),
                    main_test,
                ),
                (
                    np.array(
                        [
                            np.random.rand(10) * 100,
                            np.random.rand(10) * TWO_PI - PI,
                            np.random.rand(10) * TWO_PI,
                        ]
                    ),
                    main_test,
                ),
                (
                    np.array(
                        [
                            np.random.rand(10) * 100,
                            np.random.rand(10) * TWO_PI,
                            np.random.rand(10) * PI,
                        ]
                    ),
                    main_test,
                ),
                (
                    np.array(
                        [
                            np.random.rand(10) * 100,
                            np.random.rand(10) * -TWO_PI,
                            np.random.rand(10) * PI,
                        ]
                    ),
                    main_test,
                ),
                (
                    "Not an array", main_test,
                )
            ],
        )
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises(Exception) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)


def test_coerce_wrapped_azimuths():
    original = np.linspace(0, TWO_PI, 1000000, endpoint=False)
    offset = 0.3455 * PI
    array = original + offset
    coerced_array = coerce_wrapped_azimuth_angles(array)
    assert validate_azimuth_angles(coerced_array) is not None
    coerced_array -= offset
    coerced_original_coordinates = coerce_wrapped_azimuth_angles(coerced_array)
    assert validate_azimuth_angles(coerced_original_coordinates) is not None
    assert np.allclose(coerced_original_coordinates, original)


def test_coerce_wrapped_horizontal_angles():
    original = np.linspace(PI, -PI, 1000000, endpoint=False)
    offset = 0.3455 * PI
    array = original + offset
    coerced_array = coerce_wrapped_horizontal_angles(array)
    assert validate_horizontal_angles(coerced_array) is not None
    coerced_array -= offset
    coerced_original_coordinates = coerce_wrapped_horizontal_angles(coerced_array)
    assert validate_horizontal_angles(coerced_original_coordinates) is not None
    assert np.allclose(coerced_original_coordinates, original)


@pytest.mark.parametrize("array", (np.random.rand(3, 10), np.random.rand(10, 3)))
def test_validate_Nx3_transposed(array):
    original = array.copy()
    if array.shape != (10, 3):
        original = original.T
    array = validate_transposed_2d_array(array, cols=3)
    assert array.shape == (10, 3)
    assert np.allclose(array, original)


@pytest.mark.parametrize("array", (np.random.rand(2, 10), np.random.rand(10, 2)))
def test_validate_Nx2_transposed(array):
    original = array.copy()
    if array.shape != (10, 2):
        original = original.T
    array = validate_transposed_2d_array(array, cols=2)
    assert array.shape == (10, 2)
    assert np.allclose(array, original)


def test_invalid_transposed_2d():
    a = np.random.rand(100, 100, 100)
    b = np.random.rand(100, 100)
    with pytest.raises(ValueError):
        validate_transposed_2d_array(a, cols=3)



@pytest.mark.parametrize(("slice_obj", "expected"), (
                         (slice(None, None, None), [i for i in range(10)]),
                         (slice(0, None, None), [i for i in range(10)]),
                         (slice(0, 5, None), [i for i in range(5)]),
                         (slice(3, 8, None), [i for i in range(3, 8)]),
                         (slice(3, 8, 2), [3, 5, 7]),
                         (slice(3, 9, 2), [3, 5, 7]),
                         (slice(3, 9, -1), []),
                         (slice(9, 3, -1), [9, 8, 7, 6, 5, 4]),
                         (slice(None, None, 3), [0, 3, 6, 9]),
                         (slice(-1, -3, -1), [9, 8]),
                         (slice(-1, -3, None), []),
                         (slice(-2, -10, -2), [8, 6, 4, 2]),
                         (slice(-2, None, -2), [8, 6, 4, 2, 0]),
                         (slice(None, 4, None), [0, 1, 2, 3]),
))
def test_slice_to_integer_range(slice_obj, expected):
    result = convert_slice_to_integer_range(slice_obj, 10)
    assert np.all(result == expected)

@pytest.mark.parametrize(("value", "v_min", "v_max"), (
    (np.full(2, 2), 1, 4),
    (np.arange(100), -1, 100),
    (np.linspace(-np.pi, np.pi, 100, endpoint=True), -np.pi-0.0001, np.pi+0.0001),
    ([-2, 7, 1004, 200.43], -34, 10000),
    (1, 0, 2)
))
def test_validate_in_range(value, v_min, v_max):
    assert validate_in_range(value, v_min, v_max) is None


def test_validate_in_range_invalid():
    # Both bounds broken
    with pytest.raises(ValueError):
        validate_in_range(np.arange(100), 20, 60)

    # Lower bound broken
    with pytest.raises(ValueError):
        validate_in_range(np.array([-100]), 10, 100)

    # Upper bound broken
    with pytest.raises(ValueError):
        validate_in_range(np.array([100123]), 10, 100)


@pytest.mark.parametrize(("func", "dtype"), (
                         (normalize_uint8, np.uint8),
                         (normalize_uint16, np.uint16),
                         (normalize_int8, np.int8),
                         (normalize_int16, np.int16),
                         (normalize_int32, np.int32),
                         (normalize_int64, np.int64)
))
def test_normalize_to_dedicated_int_dtype_funcs(func, dtype: np.dtype):
    values = np.random.rand(100).astype(np.float64)
    lower = np.iinfo(dtype).min
    upper = np.iinfo(dtype).max

    width = upper - lower

    values = np.ceil(values * width + lower).astype(dtype)
    computed = func(values)

    assert np.allclose(computed, values)

