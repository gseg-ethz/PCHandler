import pytest
from typing import Callable
from abc import ABC

from src.pchandler.base_arrays import BaseArray
from src.pchandler.geometry.scalar_fields import ScalarField
from src.pchandler.validators import *

from pchandler.constants import PI, TWO_PI, HALF_PI

COORDINATE_3D_PROPERTIES = ('x', 'y', 'z', 'r', 'v', 'hz', 'rho', 'theta', 'phi', 'xyz', 'spher')


class BaseAngleTestClass(ABC):
    main_test: Callable = None

    @pytest.mark.parametrize("values, func", [
        ("str", main_test),
        ({"angle": 74}, main_test),
        ({1.3, 0.2, -1.3}, main_test),
    ])
    def test_invalid_types(self, values: np.ndarray, func: Callable):
        with pytest.raises(TypeError):
            func(values)


class TestValidation(BaseAngleTestClass):
    class TestHzAngle(BaseAngleTestClass):
        main_test: Callable = validate_horizontal_angles

        @pytest.mark.parametrize("values, func", [
            (np.array([0, np.pi, 1.7, -np.pi]), main_test),
        ])
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize("values, func", [
            (np.array([0, 2*np.pi, 1.7, -np.pi-3]), main_test),
            (float(72), main_test),
            (int(-44), main_test),
        ])
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises( Exception ) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)


    class TestAzimuthAngle(BaseAngleTestClass):
        main_test: Callable = validate_azimuth_angles

        @pytest.mark.parametrize("values, func", [
            (np.array([0, np.pi, 1.7, 2*np.pi]), main_test),
            (np.linspace(0, TWO_PI, 1000, endpoint=True), main_test),
        ])
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize("values, func", [
            (np.array([0, 3*np.pi, 1.7, -np.pi-3]), main_test),
            (np.linspace(-HALF_PI, HALF_PI, 1000, endpoint=True), main_test),
            (float(72), main_test),
            (int(-44), main_test),
        ])
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises( Exception ) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)


    class TestZenithAngles(BaseAngleTestClass):
        main_test: Callable = validate_zenith_angles

        @pytest.mark.parametrize("values, func", [
            (np.array([0, np.pi, 1.7, np.pi]), main_test),
        ])
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize("values, func", [
            (np.array([0, 3*np.pi, 1.7, -np.pi-3]), main_test),
        ])
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises( Exception ) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)

    class TestInclinationAngles(BaseAngleTestClass):
        main_test: Callable = validate_inclination_angles

        @pytest.mark.parametrize("values, func", [
            (np.array([0, 1.3, -np.pi/2, np.pi/2]), main_test),
        ])
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize("values, func", [
            (np.array([0, np.pi, 1.7, -np.pi]), main_test),
            (float(72), main_test),
            (int(-44), main_test),
        ])
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises( Exception ) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)


    class TestRadiusDistance(BaseAngleTestClass):
        main_test: Callable = validate_radius

        @pytest.mark.parametrize("values, func", [
            (np.array([0, 1.3, 200, 300000.21323]), main_test),
            (np.random.rand(100, 100), main_test),
            (np.zeros((100, 100)), main_test),
        ])
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None

        @pytest.mark.parametrize("values, func", [
            (np.array([-1.3, -2, np.inf, -np.pi]), main_test),
            (np.random.rand(100, 100) * -1, main_test),
            ([1.3, 2000, 23445.123], main_test),
            ((1.3, 2000, 23445.123), main_test),
        ])
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises( Exception ) as e:
                func(values)
            assert type(e.value) in (TypeError, ValueError)


    class TestSphericalCoordinates(BaseAngleTestClass):
        main_test: Callable = validate_spherical_angles

        @pytest.mark.parametrize("values, func", [
            (np.array([
                [np.random.rand(10)*100,
                np.random.rand(10)*TWO_PI - PI,
                np.random.rand(10)*PI
            ]]), main_test)
        ])
        def test_valid_values(self, values: np.ndarray, func: Callable):
            assert func(values) is not None


        @pytest.mark.parametrize("values, func", [
            (np.array([
                np.random.rand(10)*100-50,
                np.random.rand(10)*TWO_PI - PI,
                np.random.rand(10)*PI,
            ]), main_test),
            (np.array([
                np.random.rand(10) * 100,
                np.random.rand(10) * TWO_PI - PI,
                np.random.rand(10) * TWO_PI,
            ]), main_test),
            (np.array([
                np.random.rand(10) * 100,
                np.random.rand(10) * TWO_PI,
                np.random.rand(10) * PI,
            ]), main_test),
            (np.array([
                np.random.rand(10) * 100,
                np.random.rand(10) * -TWO_PI,
                np.random.rand(10) * PI,
            ]), main_test)
        ])
        def test_invalid_values(self, values: np.ndarray, func: Callable):
            with pytest.raises( Exception ) as e:
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

@pytest.mark.parametrize('array', (BaseArray(arr=np.ones(10)),
                                   np.ones(10),
                                   (np.ones(10), ),
                                   ScalarField(arr=np.ones(10), name='red'),
                                   {'arr': np.ones(10), 'dummy': True}))
def test_extract_array_valid(array):
    array = extract_array(array)
    assert np.all(array == np.ones(10))

@pytest.mark.parametrize('array', ('false', (np.ones(10), np.ones(10), ),
                                   ("avs",), {'false', }, {'false': 2}, False), )
def test_extract_array_invalid(array):
    with pytest.raises(TypeError):
        extract_array(array)


example_vectors = tuple([
    np.linspace(0, 2 ** 8 - 1, 255, dtype=np.uint8, endpoint=True),
    np.linspace(0, 2 ** 16 - 1, 255, dtype=np.uint16, endpoint=True),
    np.linspace(0, 2 ** 32 - 1, 255, dtype=np.uint32, endpoint=True),
    np.linspace(-2 ** 7, 2 ** 7 - 1, 255, dtype=np.int8, endpoint=True),
    np.linspace(-2 ** 15, 2 ** 15 - 1, 255, dtype=np.int16, endpoint=True),
    np.linspace(-2 ** 31, 2 ** 31 - 1, 255, dtype=np.int32, endpoint=True),
    np.linspace(0, 1, 255, dtype=np.float32, endpoint=True),
])

pairs = []
for i in example_vectors:
    for j in example_vectors:
        pairs.append((i, j))

@pytest.mark.parametrize(['original', 'target'], pairs)
def test_linear_map_dtype(original: np.ndarray, target: np.ndarray):
    """
    Test the linear mapping function between dtypes to ensure no clipping of values.
    255 samples used to avoid having to accommodate any rounding of digits in the logical comparison.
    These would be a loss of precision
    """
    # Adjust comparison tolerances based on bit sizes
    original_bits = original.dtype.itemsize * 8
    target_bits = target.dtype.itemsize * 8

    # Ensure that the manually written min/max values match the iinfo
    if np.issubdtype(original.dtype, np.integer):
        assert np.iinfo(original.dtype).min == original.min()
        assert np.iinfo(original.dtype).max == original.max()

    mapped_array = linear_map_dtype(original, target.dtype)

    # Testing of np.clip led to erroneous mapping results and often limits clipped to 0
    if np.issubdtype(target.dtype, np.floating):
        # Ensure start and end are exact. The others just need to be withi
        # n the precision
        atol = 1 / (2 ** original_bits)
        assert np.allclose(0, mapped_array[0])
        assert np.allclose(1, mapped_array[-1])
        assert np.allclose(target, mapped_array, atol=atol)
        print(f'from {original.dtype} to {target.dtype} with{atol=}')

    else:
        expected_min = np.iinfo(target.dtype).min
        expected_max = np.iinfo(target.dtype).max
        atol = (2 ** target_bits // 2 ** original_bits)
        assert np.allclose(expected_min, mapped_array[0])
        assert np.allclose(expected_max, mapped_array[-1])
        assert np.allclose(target, mapped_array, atol=atol)
        print(f'from {original.dtype} to {target.dtype} with {atol=}')

def test_invalid_linear_map_dtype():
    array = np.ones(14, dtype=np.complex64)
    array2 = np.ones(14, dtype=np.uint8)
    with pytest.raises(TypeError):
        linear_map_dtype(array, np.uint8)

    with pytest.raises(TypeError):
        linear_map_dtype(array2, np.complex64)


@pytest.mark.parametrize('array', (np.random.rand(3, 10), np.random.rand(10,3)))
def test_validate_Nx3_transposed(array):
    original = array.copy()
    if array.shape != (10, 3):
        original = original.T
    array = validate_n_by_3_transposed(array)
    assert array.shape == (10, 3)
    assert np.allclose(array, original)


@pytest.mark.parametrize('array', (np.random.rand(2, 10), np.random.rand(10,2)))
def test_validate_Nx2_transposed(array):
    original = array.copy()
    if array.shape != (10, 2):
        original = original.T
    array = validate_n_by_2_transposed(array)
    assert array.shape == (10, 2)
    assert np.allclose(array, original)


@pytest.mark.parametrize('array', (np.arange(10).reshape((1, 10)), np.arange(10).reshape((1, 1, 10)),
                                   np.arange(10).reshape((1, 10, 1)), np.arange(10).reshape((1, 1, 10,1 ,1))))
def test_validate_transposed_vector(array):
    array = validate_transposed_vector(array)
    assert array.shape == (10, )
    assert np.allclose(array, np.arange(10))

def test_invalid_transpose():
    a = np.random.rand(100, 100, 100)
    b = np.random.rand(100, 100)
    with pytest.raises(ValueError):
        validate_n_by_3_transposed(a)
