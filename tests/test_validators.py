from abc import ABC
from typing import Callable

import pytest

from pchandler.v2.constants import HALF_PI, PI, TWO_PI
from pchandler.v2.geometry.scalar_fields import ScalarField
from pchandler.v2.base_arrays import BaseArray
from pchandler.v2.validators import *

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


@pytest.mark.parametrize(
    "array",
    (
        BaseArray(arr=np.ones(10)),
        np.ones(10),
        (np.ones(10),),
        ScalarField(arr=np.ones(10), name="red"),
        {"arr": np.ones(10), "dummy": True},
    ),
)
def test_extract_array_valid(array):
    array = extract_array(array)
    assert np.all(array == np.ones(10))


@pytest.mark.parametrize(
    "array",
    (
        "false",
        (
            np.ones(10),
            np.ones(10),
        ),
        ("avs",),
        {
            "false",
        },
        {"false": 2},
        False,
    ),
)
def test_extract_array_invalid(array):
    with pytest.raises(TypeError):
        extract_array(array)


@pytest.mark.parametrize("array", (np.random.rand(3, 10), np.random.rand(10, 3)))
def test_validate_Nx3_transposed(array):
    original = array.copy()
    if array.shape != (10, 3):
        original = original.T
    array = validate_n_by_3_transposed(array)
    assert array.shape == (10, 3)
    assert np.allclose(array, original)


@pytest.mark.parametrize("array", (np.random.rand(2, 10), np.random.rand(10, 2)))
def test_validate_Nx2_transposed(array):
    original = array.copy()
    if array.shape != (10, 2):
        original = original.T
    array = validate_n_by_2_transposed(array)
    assert array.shape == (10, 2)
    assert np.allclose(array, original)


@pytest.mark.parametrize(
    "array",
    (
        np.arange(10).reshape((1, 10)),
        np.arange(10).reshape((1, 1, 10)),
        np.arange(10).reshape((1, 10, 1)),
        np.arange(10).reshape((1, 1, 10, 1, 1)),
    ),
)
def test_validate_transposed_vector(array):
    array = validate_transposed_vector(array)
    assert array.shape == (10,)
    assert np.allclose(array, np.arange(10))


def test_invalid_transpose():
    a = np.random.rand(100, 100, 100)
    b = np.random.rand(100, 100)
    with pytest.raises(ValueError):
        validate_n_by_3_transposed(a)
