import pytest

import numpy as np

from src.pchandler.v2.base_arrays import BaseArray
from src.pchandler.v2.geometry.scalar_fields import ScalarField
from src.pchandler.v2.validators import *


def test_validate_radius():
    valid_array = np.random.rand(100, 100)
    valid_array2 = np.zeros(100)
    invalid_array = np.random.rand(100, 100) * -1
    with pytest.raises(ValueError):
        validate_radius(invalid_array)

    array = validate_radius(valid_array)
    assert np.all(array == valid_array)
    array2 = validate_radius(valid_array2)
    assert np.all(array2 == valid_array2)

def test_validate_azimuthal_angles():
    valid_array = np.linspace(-PI, PI, 1000, endpoint=True)
    invalid_array = np.linspace(-TWO_PI, PI, 1000, endpoint=True)
    invalid_array2 = np.linspace(0, TWO_PI, 1000, endpoint=True) # range [0, 2PI]

    with pytest.raises(ValueError):
        validate_azimuthal_angles(invalid_array)

    with pytest.raises(ValueError):
        validate_azimuthal_angles(invalid_array2)

    validated_array = validate_azimuthal_angles(valid_array)
    assert np.all(validated_array == valid_array)

def test_validate_zenith_angles():
    valid_array = np.linspace(0, PI, 1000, endpoint=True)
    invalid_array = np.linspace(-HALF_PI, HALF_PI, 1000, endpoint=True)
    invalid_array2 = np.linspace(0, 2*PI, 1000, endpoint=True) # range [0, 2PI]

    with pytest.raises(ValueError):
        validate_zenith_angles(invalid_array)

    with pytest.raises(ValueError):
        validate_zenith_angles(invalid_array2)

    validated_array = validate_azimuthal_angles(valid_array)
    assert np.all(validated_array == valid_array)

def test_spherical_angles():
    valid_array = np.empty((1000, 3))
    valid_array[:, 0] = np.linspace(0, 1000, 1000, endpoint=True)
    valid_array[:, 1] = np.linspace(-PI, PI, 1000, endpoint=True)
    valid_array[:, 2] = np.linspace(0, PI, 1000, endpoint=True)

    array = validate_spherical_angles(valid_array)
    assert np.all(array == valid_array)

    # Bad radius
    invalid_array1 = np.empty((1000, 3))
    invalid_array1[:, 0] = np.linspace(0, -1000, 1000, endpoint=True)
    invalid_array1[:, 1] = np.linspace(-PI, PI, 1000, endpoint=True)
    invalid_array1[:, 2] = np.linspace(0, PI, 1000, endpoint=True)

    # Bad azimuth
    invalid_array2 = np.empty((1000, 3))
    invalid_array2[:, 0] = np.linspace(0, 1000, 1000, endpoint=True)
    invalid_array2[:, 1] = np.linspace(0, TWO_PI, 1000, endpoint=True)
    invalid_array2[:, 2] = np.linspace(0, PI, 1000, endpoint=True)

    # Bad zenith
    invalid_array3 = np.empty((1000, 3))
    invalid_array3[:, 0] = np.linspace(0, 1000, 1000, endpoint=True)
    invalid_array3[:, 1] = np.linspace(-PI, PI, 1000, endpoint=True)
    invalid_array3[:, 2] = np.linspace(-HALF_PI, HALF_PI, 1000, endpoint=True)

    for arr in (invalid_array1, invalid_array2, invalid_array3):
        with pytest.raises(ValueError):
            validate_spherical_angles(arr)

def test_coerce_wrapped_azimuths():
    original = np.linspace(PI, -PI, 1000000, endpoint=False)
    offset = 0.3455 * PI
    array = original + offset
    assert validate_azimuthal_angles(coerce_wrapped_azimuths(array)) is not None
    array2 = array - offset
    assert validate_azimuthal_angles(coerce_wrapped_azimuths(array2)) is not None
    assert np.allclose(array2, original)

@pytest.mark.parametrize('array', (BaseArray(arr=np.ones(10)),
                                   np.ones(10), (np.ones(10), ),
                                   ScalarField(arr=np.ones(10), name='red'),
                                   (ScalarField(arr=np.ones(10), name='red'),),
                                   {'arr': np.ones(10), 'dummy': True}))
def test_extract_array_valid(array):
    array = extract_array(array)
    assert np.all(array == np.ones(10))

@pytest.mark.parametrize('array', ('false', (np.ones(10), np.ones(10), ),
                                   ("avs",), {'false', }, {'false': 2}, False))
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
    array = validate_Nx3_transposed(array)
    assert array.shape == (10, 3)
    assert np.allclose(array, original)


@pytest.mark.parametrize('array', (np.random.rand(2, 10), np.random.rand(10,2)))
def test_validate_Nx2_transposed(array):
    original = array.copy()
    if array.shape != (10, 2):
        original = original.T
    array = validate_Nx2_transposed(array)
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
        validate_Nx3_transposed(a)
    with pytest.raises(ValueError):
        validate_Nx3_transposed(b)
