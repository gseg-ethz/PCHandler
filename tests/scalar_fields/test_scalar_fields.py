import numpy as np
import pytest
from GSEGUtils.validators import linear_map_dtype, normalize_self
from pydantic import ValidationError

from pchandler.constants import NORMAL_NAMES, RGB_NAMES
from pchandler.scalar_fields.scalar_fields import (
    DtypeState,
    NormalFields,
    NormalisedInt16ScalarField,
    RGBFields,
    ScalarField,
    ScalarFieldBoolean,
    ScalarFieldFloat32,
    ScalarFieldUint8,
    SegmentationMap,
)

N = 100


@pytest.fixture(scope="function")
def nx3_uint8():
    return RGBFields(np.random.randint(0, 256, (N, 3), dtype=np.uint8))


@pytest.fixture(scope="function")
def normals_float32():
    vals = np.random.rand(N, 3).astype(np.float32)
    vals = vals / np.linalg.norm(vals, axis=1).reshape(-1, 1)
    return vals


def test_lower_str_annotation():
    """
        Test name conversion to lowercase

    Returns
    -------

    """
    upper_name = "STEVE"
    sf = ScalarField(np.ones(N), name=upper_name)

    assert isinstance(sf.name, str)
    assert sf.name != upper_name
    assert sf.name == "steve"


class TestDtypeState:
    def test_initialise(self):
        state = DtypeState(np.int16, 3, 5)
        assert state.dtype == np.int16
        assert state.lower == 3
        assert state.upper == 5

    def test_generate_method(self):
        array = np.array([2, 3, 4, 8]).astype(np.uint8)
        state = DtypeState.generate(array)
        assert state.dtype == np.uint8
        assert state.lower == 2
        assert state.upper == 8

    def test_validate(self):
        with pytest.raises(ValueError):
            a = DtypeState(np.uint8, 10, 5)
            DtypeState.validate(a)


class TestScalarFieldClass:
    def test_keyword_init(self):
        a = ScalarField(
            arr=np.ones(N).astype(np.float32),
            name="Allen",
            origin_dtype=DtypeState(dtype=np.int32, lower=0, upper=1),
        )

        assert a.name == "allen"
        assert np.all(a == 1)
        assert a.origin_dtype.dtype == np.int32
        assert a.origin_dtype.dtype != a.arr.dtype
        assert isinstance(a, ScalarField)

    def test_positional_init(self):
        a = ScalarField(np.ones(N), name="steve")

        assert a.name == "steve"
        assert np.all(a == np.ones(N))

    @pytest.mark.parametrize(
        ["array", "name", "origin_dtype"],
        (
            (np.ones(N), True, None),
            (np.ones(N), {"sas"}, None),
            (np.ones(N), 1234.123, None),
            (np.ones((N, 2)), "sty", None),
            (np.ones((N, 4)), "steve", None),
            (np.ones((N, 3, 3)), "asd", None),
            (np.ones(N), "steve", True),
            (np.ones(N), "steve", {"a": 2}),
        ),
    )
    def test_invalid_values(self, array, name, origin_dtype):
        with pytest.raises(Exception) as e:
            _ = ScalarField(array, name=name, origin_dtype=origin_dtype)  # type: ignore

        assert type(e.value) in (ValidationError, ValueError, TypeError)

    @pytest.mark.parametrize("array", (np.ones(1), np.ones(1000), np.ones((1000, 1)), np.ones((1, 1, 1, 1, 1))))
    def test_valid_shape(self, array):
        a = ScalarField(array, name="a")
        assert a.arr.shape == (array.size,)
        assert np.all(a.arr == array.squeeze())

    def test_init_from_self(self):
        b = ScalarField(np.ones(N), name="test")
        c = ScalarField(b, name="second")
        assert np.all(b == c)
        assert b.name != c.name

        e = ScalarField(c)
        assert id(e) != id(c)
        assert id(e.arr) == id(c.arr)  # Pass by reference expected
        assert e.name == c.name
        assert np.all(e.arr == c.arr)

    @pytest.mark.parametrize("array", (np.ones((N, 2)), np.ones((N, 3))))
    def test_invalid_shape(self, array):
        with pytest.raises(Exception) as e:
            ScalarField(array, name="a")

        assert type(e.value) in (ValueError, TypeError, ValidationError)

    def test_get_original_data(self):
        a = np.random.randint(-1000, 1000, 2000, dtype=np.int16)
        b = normalize_self(a)
        b = linear_map_dtype(b, np.uint8)
        field = ScalarField(b, name="converted", origin_dtype=DtypeState(a.dtype, a.min(), a.max()))

        c = field.get_original_data()

        atol = np.ceil((a.max() - a.min()) / 2**8)
        assert np.allclose(a, c, atol=atol)

        d = np.random.randint(0, 233, 2000, dtype=np.uint8)
        dtype_state = DtypeState.generate(d)
        sf = ScalarField(d, name="converted", origin_dtype=dtype_state)
        e = sf.get_original_data()
        assert np.all(d == e)
        assert isinstance(e, np.ndarray)


class TestTypeDefinedScalarFields:
    def test_normalised_int16(self):
        array = np.random.randint(-(2**14), 2**13, N, dtype=np.int16)
        b = NormalisedInt16ScalarField(array, name="temp")
        assert isinstance(b, NormalisedInt16ScalarField)
        assert np.all(b == array)

        as_uint8 = b.to_uint8()
        assert isinstance(as_uint8, ScalarField)
        assert as_uint8.dtype == np.uint8

    def test_bool_valid(self):
        array = np.random.randint(0, 1, N, dtype=np.bool_)
        b = ScalarFieldBoolean(array, name="temp")
        assert isinstance(b, ScalarFieldBoolean)
        assert np.all(b == array)

    def test_bool_invalid(self):
        array = np.random.randint(-128, 127, N, dtype=np.int8)
        with pytest.raises(Exception) as e:
            ScalarFieldBoolean(array)
        assert type(e.value) in (ValidationError, ValueError, TypeError)

    def test_scalar_uint8(self):
        array = np.random.randint(0, 255, N, dtype=np.uint8)
        b = ScalarFieldUint8(array, name="temp")
        assert isinstance(b, ScalarFieldUint8)
        assert np.all(b == array)
        assert b.dtype == np.uint8

    def test_scalar_float32(self):
        array = np.random.rand(N).astype(np.float32)
        b = ScalarFieldFloat32(array, name="temp")
        assert isinstance(b, ScalarFieldFloat32)
        assert np.all(b == array)
        assert b.dtype == np.float32


class TestRgbField:
    def test_positional_init(self):
        data = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
        a = RGBFields(data)

        assert a.name == RGB_NAMES.base
        assert np.all(a == data)

        a = RGBFields(data, name="not_rgb")
        assert a.name == RGB_NAMES.base
        assert a.name != "not_rgb"

    def test_keyword_init(self):
        data = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
        a = RGBFields(arr=data)

        assert a.name == RGB_NAMES.base
        assert np.all(a == data)

        a = RGBFields(arr=data, name="not_rgb")
        assert a.name == RGB_NAMES.base
        assert a.name != "not_rgb"

    def test_invalid_shapes(self):
        data = np.random.randint(0, 255, N, dtype=np.uint8)
        with pytest.raises(ValidationError):
            RGBFields(data)

    def test_invalid_dtypes(self):
        data = np.array([[1 + 2j, 3 + 4j, 5 + 6j], [1 + 2j, 3 + 4j, 5 + 6j], [1 + 2j, 3 + 4j, 5 + 6j]])
        with pytest.raises(Exception) as e:
            RGBFields(data)

        assert type(e.value) in (ValidationError, TypeError)

    def test_properties(self):
        data = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
        a = RGBFields(data)

        assert np.all(a.r == data[:, 0])
        assert np.all(a.g == data[:, 1])
        assert np.all(a.b == data[:, 2])

    def test_initialise_field_class_method(self):
        data: np.ndarray = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
        rgb1 = RGBFields.initialize(data.shape[0])
        assert np.all(rgb1 == np.zeros_like(data))
        assert np.uint8 == rgb1.dtype

        rgb2 = RGBFields.initialize(data.shape[0], data)
        assert not np.allclose(rgb2, np.zeros_like(data))
        assert np.uint8 == rgb2.dtype
        assert np.all(rgb2 == data)

    @pytest.mark.parametrize(
        "arr",
        (
            # COUPLE-05 D-17 site 2 ratified contract: RGB float input MUST be in [0, 1].
            # Out-of-`[0, 1]` float parametrizations were removed in 04-06b because the
            # explicit `source_range=(0.0, 1.0)` kwarg makes [-1, 1] / [0, 255] / [-128, 127]
            # float inputs clip-and-saturate (D-12) rather than auto-rescale by min/max.
            # Integer inputs still round-trip via the D-15 iinfo bypass.
            np.random.rand(10000, 3),
            np.random.randint(-(2**7), 2**7, (10000, 3), dtype=np.int8),
            np.random.randint(0, 2**16, (10000, 3), dtype=np.uint16),
            np.random.randint(0, 2**8, (10000, 3), dtype=np.uint8),
        ),
    )
    def test_normalized(self, arr):
        rgb = RGBFields(arr)
        original: np.ndarray = rgb.get_original_data()

        original_bits = arr.dtype.itemsize * 8
        target_bits = rgb.dtype.itemsize * 8
        if np.issubdtype(arr.dtype, np.floating):
            atol = 1
        else:
            atol = 2**original_bits // 2**target_bits
            atol += 1

        assert np.allclose(arr, original, atol=atol)


def test_rgb_float_to_uint8_clip_and_saturate():
    """COUPLE-05 D-17 site 2 regression: RGBFields([0.0, 0.5, 1.0]) -> [0, 128, 255].

    The explicit ``source_range=(0.0, 1.0)`` kwarg threaded through
    ``RGBFields._normalise_to_uint8`` ratifies the float-RGB contract. Endpoints
    and the mid-value clip-and-saturate per CONTEXT D-12.
    """
    rgb = RGBFields(np.array([[0.0, 0.5, 1.0]]))
    assert rgb.arr.dtype == np.uint8
    assert rgb.arr[0, 0] == 0
    assert rgb.arr[0, -1] == 255


def test_normalised_int16_to_uint8_integer_bypass():
    """COUPLE-05 D-17 site 3 regression: int16 -> uint8 conversion (D-15 bypass).

    ``NormalisedInt16ScalarField.arr`` is int16 by class invariant; the
    ``source_range=(0.0, 1.0)`` kwarg threaded through ``to_uint8`` is
    documentational only because the integer-input path bypasses the clip
    step (D-15).
    """
    sf = NormalisedInt16ScalarField(np.array([-100, 0, 100], dtype=np.int16), name="x")
    out = sf.to_uint8()
    assert out.arr.dtype == np.uint8


class TestNormalsField:
    def test_positional_init(self, normals_float32):
        a = NormalFields(normals_float32)

        assert a.name == NORMAL_NAMES.base
        assert np.all(a == normals_float32)

        a = NormalFields(normals_float32, name="not_normals")
        assert a.name == NORMAL_NAMES.base
        assert a.name != "not_normals"

    def test_keyword_init(self, normals_float32):
        a = NormalFields(arr=normals_float32)

        assert a.name == NORMAL_NAMES.base
        assert np.all(a == normals_float32)

        a = NormalFields(arr=normals_float32, name="not_normals")
        assert a.name == NORMAL_NAMES.base
        assert a.name != "not_normals"

    def test_invalid_shapes(self):
        data = np.random.rand(N).astype(np.float32)
        with pytest.raises(ValidationError):
            NormalFields(data)

    def test_invalid_dtypes(self):
        data = np.random.randint(0, 2, (N, 3), dtype=np.bool_)
        with pytest.raises(Exception) as e:
            NormalFields(data)

        assert type(e.value) in (ValidationError, TypeError)

    def test_properties(self, normals_float32):
        a = NormalFields(normals_float32)

        assert np.all(a.nx == normals_float32[:, 0])
        assert np.all(a.ny == normals_float32[:, 1])
        assert np.all(a.nz == normals_float32[:, 2])

    def test_initialise_field_class_method(self):
        data = np.random.rand(N, 3).astype(np.float32)
        data /= np.linalg.norm(data, axis=1).reshape(-1, 1)
        check_data = np.zeros_like(data)
        check_data[:, 2] = 1
        normals1 = NormalFields.initialize(data.shape[0])
        assert np.all(normals1 == check_data)
        assert np.float32 == normals1.dtype

        normals2 = NormalFields.initialize(data.shape[0], data)
        assert not np.allclose(normals2, np.zeros_like(data))
        assert np.float32 == normals2.dtype
        assert np.all(normals2 == data)


class TestSegmentationField:
    def test_valid_shapes(self):
        a = np.arange(10, dtype=np.uint8)
        a2 = np.arange(10, dtype=np.uint8)
        a3 = np.arange(1000097, dtype=np.uint8)
        a4 = np.random.randint(0, 10, (10, 1), dtype=np.uint8)
        b = SegmentationMap(arr=a, name="Segmentation")
        b2 = SegmentationMap(arr=a2, name="Segmentation")
        b3 = SegmentationMap(arr=a3, name="Segmentation")
        b4 = SegmentationMap(arr=a4, name="Segmentation")

        assert isinstance(b, SegmentationMap)
        assert len(a) == len(b)
        assert len(a2) == len(b2)
        assert len(a3) == len(b3)
        assert np.all(a == b)
        assert np.all(a2 == b2)
        assert b4.shape == (10,)

    def test_invalid_shapes(self):
        a = np.random.randint(0, 100, (N, 3), dtype=np.uint8)
        b = np.random.randint(0, 100, (N, 5), dtype=np.uint8)

        with pytest.raises(ValueError):
            SegmentationMap(arr=a, name="Segmentation")
        with pytest.raises(ValueError):
            SegmentationMap(arr=b, name="Segmentation")

    @pytest.mark.parametrize(
        "array",
        (
            np.random.randint(0, 2**14, N, dtype=np.int16),
            np.random.randint(0, 100, N, dtype=np.uint32),
            np.random.rand(N).astype(np.float32),
            np.ones((N, 5), dtype=np.bool_),
        ),
    )
    def test_dtypes(self, array):
        with pytest.raises(Exception) as e:
            SegmentationMap(array, name="Segmentation")
            assert type(e.value) in (ValueError, TypeError, ValidationError)

    def test_initialize_method(self):
        sizes_small = [10 for _ in range(120)]
        sizes_large = [100 for _ in range(259)]
        sizes_too_large = [100 for _ in range(2**20)]

        result = SegmentationMap.initialize("small", sizes_small)
        assert result.size == sum(sizes_small)
        assert result.dtype == np.uint8
        assert np.all(np.zeros(result.size) == result)

        result2 = SegmentationMap.initialize("largs", sizes_large)
        assert result2.size == sum(sizes_large)
        assert result2.dtype == np.uint16
        assert np.all(np.zeros(result2.size) == result2)

        assert result.shape != result2.shape

        with pytest.raises(ValueError):
            SegmentationMap.initialize("fail", sizes_too_large)


example_vectors = tuple(
    [
        np.linspace(0, 2**8 - 1, 255, dtype=np.uint8, endpoint=True),
        np.linspace(0, 2**16 - 1, 255, dtype=np.uint16, endpoint=True),
        np.linspace(0, 2**32 - 1, 255, dtype=np.uint32, endpoint=True),
        np.linspace(-(2**7), 2**7 - 1, 255, dtype=np.int8, endpoint=True),
        np.linspace(-(2**15), 2**15 - 1, 255, dtype=np.int16, endpoint=True),
        np.linspace(-(2**31), 2**31 - 1, 255, dtype=np.int32, endpoint=True),
        np.linspace(0, 1, 255, dtype=np.float32, endpoint=True),
    ]
)

pairs = []
for i in example_vectors:
    for j in example_vectors:
        pairs.append((i, j))


@pytest.mark.parametrize(["original", "target"], pairs)
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
        # Ensure start and end are exact. The others just need to be within the precision
        atol = 1 / (2**original_bits)
        expected_min, expected_max = 0, 1

    else:
        atol = (2**target_bits // 2**original_bits) or 1
        expected_min, expected_max = np.iinfo(target.dtype).min, np.iinfo(target.dtype).max

    assert np.allclose(expected_min, mapped_array[0])
    assert np.allclose(expected_max, mapped_array[-1])
    assert np.allclose(target, mapped_array, atol=atol)
    # print(f'From {original.dtype} to {target.dtype} with {atol=}')


def test_invalid_linear_map_dtype():
    array = np.ones(14, dtype=np.complex64)
    array2 = np.ones(14, dtype=np.uint8)
    with pytest.raises(TypeError):
        linear_map_dtype(array, np.uint8)

    with pytest.raises(TypeError):
        linear_map_dtype(array2, np.complex64)


def test_normalise_self_valid():
    array = np.random.randint(13, 144, 1000, np.uint8)
    normalised = normalize_self(array)

    assert not np.allclose(array, normalised)
    assert normalised.min() == 0
    assert normalised.max() == 255


def test_normalise_self_invalid():
    array = np.random.rand(1000) * 244 - 50

    normalised = normalize_self(array)
    assert not np.allclose(array, normalised)
    assert normalised.min() == 0
    assert normalised.max() == 1
    assert normalised.min() != array.min()
    assert normalised.max() != array.max()
