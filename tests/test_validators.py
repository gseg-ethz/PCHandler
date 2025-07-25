from abc import ABC
from typing import Callable

import pytest

from pchandler.geometry.scalar_fields import ScalarField
from pchandler.base_arrays import BaseArray
from pchandler.validators import *

COORDINATE_3D_PROPERTIES = ("x", "y", "z", "r", "v", "hz", "rho", "theta", "phi", "xyz", "spher")


class BaseAngleTestClass(ABC):
    main_test: Callable = None

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
