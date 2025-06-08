import pytest

from pydantic import BaseModel, ValidationError

from src.pchandler.geometry.scalar_fields import *


class TestGlobals:
    def test_fixed_key_names(self):
        assert RGB_FIELD == 'rgb'
        assert NORMALS_FIELD == 'normals'

    def test_lower_str_annotation(self):
        class A(BaseModel):
            name: LowerStr

        test_name = "STEVE"
        b = A(name=test_name)
        assert isinstance(b.name, str)
        assert b.name != test_name
        assert b.name == 'steve'

        test2 = "  Allen Steve  "
        c = A(name=test2)
        assert c.name == 'allen steve'

class TestScalarFieldClass:
    def test_init(self):
        a = ScalarField(arr=np.ones(10).astype(np.float32),
                        name='Allen',
                        operations_performed=['Nothing', 'Yet'],
                        original_state=DataRange(dtype=np.int32, lower=0, upper=1),
                        )

        assert a.name == 'allen'
        assert np.all(a.arr == 1)
        assert a.operations_performed == ['Nothing', 'Yet']
        assert a.original_state.dtype != np.int32
        assert a.original_state.dtype == a.arr.dtype
        assert isinstance(a, ScalarField)

    def test_defaults(self):
        a = ScalarField(arr=np.ones(10), name='a')
        assert a.operations_performed is None
        assert a.original_state.dtype == np.float64
        assert a.original_state.lower == 1
        assert a.original_state.upper == 1

    ones = np.ones(10)

    @pytest.mark.parametrize(['array', 'name', 'operations', 'original_state'], (
        (ones, True, None, None),
        (ones, {'sas'}, None, None),
        (ones, 1234.123, None, None),
        (np.ones((10,2)), 'sty', None, None),
        (np.ones((10,4)), "steve", None, None),
        (np.ones((10,3,3)), 'asd', None, None),
        (ones, 'steve', True, None),
        (ones, 'steve', {'a': 2}, None)
    ))
    def test_invalid_values(self, array, name, operations, original_state):
        with pytest.raises(Exception) as e:
            _ = ScalarField(arr=array, name=name, original_state=original_state, operations_performed=operations)

        assert type(e.value) in (ValidationError, ValueError, TypeError)

    @pytest.mark.parametrize('array', ( np.ones(1), np.ones(1000), np.ones((1000, 1)), np.ones((1,1,1,1,1))))
    def test_valid_shape(self, array):
        a = ScalarField(arr=array, name='a')
        assert a.arr.shape == (array.size,)
        assert np.all(a.arr == array.squeeze())

    def test_init_from_self(self):
        b = ScalarField(arr=np.ones(10), name='test'),
        c = ScalarField(arr=b, name='second')
        assert np.all(b == c)
        assert b.name != c.name

        d = ScalarField(**c.model_dump())
        assert id(d) != id(c)
        assert id(d.arr) != id(c.arr)
        assert d.name == c.name
        assert np.all(d.arr == c.arr)

    @pytest.mark.parametrize('array', (np.ones((10,2)), np.ones((10,3)), np.array([0]).squeeze(), np.ones((1000, 1))))
    def test_invalid_shape(self, array):
        a = ScalarField(arr=array, name='a')
        assert a.arr.shape == array.squeeze().shape
        assert np.all(a.arr == array)

    def test_normalise(self):
        raise NotImplementedError

    def test_normalise_dtype(self):
        kwargs = {'arr': np.arange(255).astype(np.int16),
                  'name': 'steve',}
        a = ScalarField.normalize_based_on_original_dtype(kwargs)
        assert np.any(a != kwargs['arr'])
        assert np.allclose(a.min(), np.iinfo(np.int16).min)
        assert np.allclose(a.max(), np.iinfo(np.int16).max)
        raise NotImplementedError

    def test_unpack_dtypes_method(self):
        raise NotImplementedError

    def test_get_npydantic_dtype(self):
        raise NotImplementedError

    def test_coerce_to_target_type(self):
        raise NotImplementedError

    def test_get_original_dtype(self):
        raise NotImplementedError

    def test_create_rollback(self):
        raise NotImplementedError

class TestScalarFieldTriplets:
    def test_valid_shapes(self):
        raise NotImplementedError

    def test_invalid_shapes(self):
        raise NotImplementedError

class TestIntensityField:
    def test_valid_shapes(self):
        raise NotImplementedError

    def test_invalid_shapes(self):
        raise NotImplementedError

    def test_valid_dtypes(self):
        raise NotImplementedError

    def test_invalid_dtypes(self):
        raise NotImplementedError

    def test_initialize_method(self):
        raise NotImplementedError


class TestRgbField:
    def test_valid_shapes(self):
        raise NotImplementedError

    def test_invalid_shapes(self):
        raise NotImplementedError

    def test_valid_dtypes(self):
        raise NotImplementedError

    def test_invalid_dtypes(self):
        raise NotImplementedError

    def test_initialize_method(self):
        raise NotImplementedError

    def test_properties(self):
        raise NotImplementedError


class TestNormalsField:
    def test_valid_shapes(self):
        raise NotImplementedError

    def test_invalid_shapes(self):
        raise NotImplementedError

    def test_valid_dtypes(self):
        raise NotImplementedError

    def test_invalid_dtypes(self):
        raise NotImplementedError

    def test_initialize_method(self):
        raise NotImplementedError

    def test_properties(self):
        raise NotImplementedError


class TestBooleanField:
    def test_valid_shapes(self):
        raise NotImplementedError

    def test_invalid_shapes(self):
        raise NotImplementedError

    def test_dtypes(self):
        raise NotImplementedError

class TestSegmentationField:
    def test_valid_shapes(self):
        a = np.arange(10, dtype=np.uint8)
        a2 = np.arange(10, dtype=np.uint8)
        a3 = np.arange(1000097, dtype=np.uint8)
        a4 = np.random.randint(0, 10, (10,1), dtype=np.uint8)
        b = SegmentationMap(arr=a, name='Segmentation')
        b2 = SegmentationMap(arr=a2, name='Segmentation')
        b3 = SegmentationMap(arr=a3, name='Segmentation')
        b4 = SegmentationMap(arr=a4, name='Segmentation')

        assert isinstance(b, SegmentationMap)
        assert len(a) == len(b)
        assert len(a2) == len(b2)
        assert len(a3) == len(b3)
        assert np.all(a == b)
        assert np.all(a2 == b2)
        assert b4.shape == (10,)
        
    def test_invalid_shapes(self):
        a = np.random.randint(0, 100, (100, 3), dtype=np.uint8)
        b = np.random.randint(0, 100, (100, 5), dtype=np.uint8)

        with pytest.raises(ValueError):
            c = SegmentationMap(arr=a, name='Segmentation')
        with pytest.raises(ValueError):
            d = SegmentationMap(arr=b, name='Segmentation')

    def test_dtypes(self):
        a = np.random.randint(0, 2**15, 100, dtype=np.uint16)
        b = np.random.randint(0, 100, 100, dtype=np.uint32)
        c = np.random.rand(100).astype(np.float32)
        d = np.ones((100, 5), dtype=np.bool_)

        for pcd in (a, b, c, d):
            with pytest.raises(TypeError):
                c = SegmentationMap(arr=pcd, name='Segmentation')


