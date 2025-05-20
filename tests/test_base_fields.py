import pytest
import numpy as np

from src.pchandler.base_fields import (ValidatedAttribute, ValidatedArrayAttribute, VectorAttribute, Array2dAttribute,
                                       Array3dAttribute, PointSet2D, PointSet3D, ValidatedArray, ValidatedArrayNx2,
                                       ReadOnlyArrayAttribute, ValidatedVector, ValidatedArray2D)


def is_even(x):
    if not x % 2 == 0:
        raise ValueError("Not an even number")


class CoercedField(ValidatedAttribute):
    def __init__(self, *args, **kwargs):
        kwargs |= {'coerce': True}
        super().__init__(*args, **kwargs)


class OptionalField(ValidatedAttribute):
    def __init__(self, *args, **kwargs):
        kwargs |= {'optional': True}
        super().__init__(*args, **kwargs)


class IsEvenField(ValidatedAttribute):
    def __init__(self, *args, **kwargs):
        kwargs |= {'validators': [is_even]}
        super().__init__(*args, **kwargs)


class CustomFields:
    coerced_string: CoercedField = CoercedField(str)
    optional_float: OptionalField = OptionalField(float)
    even_number: IsEvenField = IsEvenField(int)


class CustomObject:
    name: ValidatedAttribute = ValidatedAttribute[str](str, coerce=True)
    length: ValidatedAttribute = ValidatedAttribute[float](float, optional=True, default=None)
    frozen: ValidatedAttribute = ValidatedAttribute(float, freezable=True)
    weight: ValidatedAttribute = ValidatedAttribute(int, validators=[is_even])
    height: ValidatedAttribute = ValidatedAttribute(float, default=22.0, coerce=True)

    def __init__(self, name, length, weight, frozen, height=None):
        self.name = name
        self.length = length
        self.weight = weight
        self.frozen = frozen
        self.height = height
        return


class TestCustomFields:
    def test_base_field_parameters(self):
        custom_obj = CustomObject(name='abc', length=None, weight=22, frozen=12.0)
        custom_obj.name = 123
        assert custom_obj.name == "123"
        custom_obj.length = None
        assert custom_obj.length is None
        custom_obj.length = 123.45

        with pytest.raises(AttributeError):
            custom_obj.frozen = 23.4

        custom_obj.weight = 22
        assert custom_obj.weight == 22

        with pytest.raises(ValueError):
            custom_obj.weight = 21

        assert custom_obj.height == 22.0

        custom_obj.height = 245.1
        assert custom_obj.height == 245.1

    def test_field_param_overwrites(self):
        obj = CustomObject(name='abc', length=None, weight=22, frozen=12.3)

        assert obj.name == 'abc'
        assert obj.length is None
        assert obj.weight == 22
        assert obj.frozen == 12.3

        # Test coercion field
        obj.name = 123
        assert obj.name == '123'

        obj.length = 2.0
        assert obj.length == 2.0
        obj.length = None
        assert obj.length is None

        obj.weight = 124
        assert obj.weight == 124
        with pytest.raises(ValueError):
            obj.weight = 123

        with pytest.raises(AttributeError):
            obj.frozen = 1234567

    def test_non_class_attribute(self):
        a = CustomFields.optional_float
        assert hasattr(a, 'name')
        assert hasattr(a, 'private_name')
        assert hasattr(a, 'options_name')

        assert getattr(a, 'name') == 'optional_float'
        assert getattr(a, 'private_name') == '_optional_float'
        assert getattr(a, 'options_name') == '_optional_float_options'

    @pytest.mark.parametrize('invalid_value', (False, {'abc': 123}, (1, 2, 3), 'not_valid'))
    def test_invalid_type(self, invalid_value):
        obj: CustomObject = CustomObject('abc', None, 122, frozen=123.4)
        with pytest.raises(TypeError):
            obj.length = invalid_value

    def test_optional_field(self):
        with pytest.raises(ValueError):
            CustomObject(name='abc', weight=None, length=None, frozen=12)

        obj: CustomObject = CustomObject('abc', None, 122, frozen=123.4)
        with pytest.warns(Warning):
            obj.height = None

    def test_coercion_field(self):
        obj: CustomObject = CustomObject('abc', None, 122, frozen=123.4)
        with pytest.raises(TypeError):
            obj.height = np.array([1, 2, 3])

    def test_delete(self):
        obj: CustomObject = CustomObject('abc', None, 122, frozen=123.4)

        del obj.length
        assert not hasattr(obj, '_length')

        with pytest.raises(ValueError):
            del obj.frozen


class CustomNpyVectorField(ValidatedArrayAttribute):
    __ndim__ = 1


class CustomNpyTripletArray(ValidatedArrayAttribute):
    __ndim__ = 2
    __shape__ = (None, 3)


class CustomTransformMatrixField(ValidatedArrayAttribute):
    __ndim__ = 2
    __shape__ = (3,3)


class CustomReadOnly(ValidatedArrayAttribute):
    def __init__(self, *args, **kwargs):
        kwargs |= {'freezable': True}
        super().__init__(*args, **kwargs)


class CustomNpyFieldTestObject:
    vector: CustomNpyVectorField = CustomNpyVectorField(coerce=True)
    triplet_array: CustomNpyTripletArray = CustomNpyTripletArray()
    transform_matrix: CustomTransformMatrixField = CustomTransformMatrixField()
    read_only: CustomReadOnly = CustomReadOnly()
    # shallow_to_overwrite: CustomReadOnly = CustomReadOnly('shallow_to_overwrite', frozen=FrozenEnum.SHALLOW)

    def __init__(self,
                 vector = np.array([1, 2, 3]),
                 triplet_array = np.array([[1, 2, 3], [4, 5, 6]]),
                 transform_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                 read_only = np.ones((10,10))):

        print(f'ID: {id(self)}')
        self.vector = vector
        self.triplet_array = triplet_array
        self.transform_matrix = transform_matrix
        self.read_only = read_only

@pytest.fixture(scope="function")
def custom_npy_fld_obj():
    return CustomNpyFieldTestObject()

class TestCustomNumpyFields:
    def test_initialisation(self, custom_npy_fld_obj):
        vector = np.array([1, 2, 3])
        triplet_array = np.array([[1, 2, 3], [4, 5, 6]])
        transform_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        read_only = np.ones((10, 10))
        # shallow_to_overwrite = np.ones((10, 10))

        assert np.all(custom_npy_fld_obj.vector == vector)
        assert np.all(custom_npy_fld_obj.triplet_array == triplet_array)
        assert np.all(custom_npy_fld_obj.transform_matrix == transform_matrix)

        assert np.all(custom_npy_fld_obj.read_only == read_only)
        # assert np.all(custom_npy_fld_obj.shallow_to_overwrite == shallow_to_overwrite)

    def test_array_coercion(self, custom_npy_fld_obj):
        vector = [1, 2, 3]
        custom_npy_fld_obj.vector = vector
        assert np.all(custom_npy_fld_obj.vector == np.array(vector))
        with pytest.raises(TypeError):
            custom_npy_fld_obj.vector = {'abc', 123}


    def test_ndim_vector(self, custom_npy_fld_obj):
        custom_npy_fld_obj.vector = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError):
            custom_npy_fld_obj.vector = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])


    def test_ndim_triplet_array(self, custom_npy_fld_obj):
        custom_npy_fld_obj.triplet_array = np.random.rand(10, 3)
        with pytest.raises(ValueError):
            custom_npy_fld_obj.triplet_array = np.random.rand(10, 4)


    def test_ndim_transform_matrix(self, custom_npy_fld_obj):
        custom_npy_fld_obj.transform_matrix = np.random.rand(3, 3)

        with pytest.raises(ValueError):
            custom_npy_fld_obj.transform_matrix = np.random.rand(4, 4)


    def test_read_only(self, custom_npy_fld_obj):
        with pytest.raises(AttributeError):
            custom_npy_fld_obj.read_only = np.random.rand(2, 2)

        with pytest.raises(ValueError):
            custom_npy_fld_obj.read_only[0, 0] = 1

        assert np.all(custom_npy_fld_obj.read_only == np.ones((10, 10)))

        custom_npy_fld_obj.triplet_array[0, 0] = 1234567
        assert custom_npy_fld_obj.triplet_array[0, 0] == 1234567


class TestDataArray:
    def test_mutability(self):
        array = np.random.rand(5,3)
        array2 = np.random.rand(5,3)
        test_data = ValidatedArray(array)
        test_id = id(test_data)
        np.testing.assert_array_almost_equal(test_data, array)
        test_data.arr = array2
        assert np.all(test_data == array2)
        assert np.all(test_data != array)
        assert test_id == id(test_data)


    def test_immutability(self) -> None:
        test_data = ValidatedArray(np.random.rand(5,3))
        array2 = ReadOnlyArrayAttribute(np.random.randn(5,3))
        with pytest.raises( AttributeError ):
            array2.arr = test_data.arr

        test_data.arr[0, 0] = 3
        # This blocks the user from assigning directly to the array as well
        with pytest.raises( ValueError ):
            array2.arr[0, 0] = 3

    def test_numpy_functionality(self):
        a: ValidatedArray = ValidatedArray(np.ones((5,3)))
        b: ValidatedArray = a + np.ones(3) * 3
        assert np.all(a != b)
        c: ValidatedArray = b - np.ones(3) * 3
        np.testing.assert_array_almost_equal(a, c)
        d: np.ndarray|ValidatedArray = a != b
        assert isinstance(d, np.ndarray)
        assert np.issubdtype(d.dtype, np.bool_)
        assert np.mean(b) == 4

    def test_helper_properties(self):
        a: ValidatedArray = ValidatedArray(np.ones((5,3)))
        assert a.ndim == 2
        assert a.shape == (5,3)
        assert a.dtype == np.float64
        assert a.size == np.prod(a.shape)
        assert a.base is None

    @pytest.mark.parametrize("values, error_type", [
        (('str', {'23': 1}, {1, 3, 5}, False), TypeError),
        ((None,), ValueError)])
    def test_validation_func_invalid_values(self, values, error_type):
        for val in values:
            print(f'Trying {val=}')
            with pytest.raises(error_type):
                ValidatedArray(val)

    @pytest.mark.parametrize("values", [
        np.random.randn(5).astype('f8'),
        np.random.randn(1000,10000).astype('f4'),
        np.random.randn(1000, 20, 3),
        np.random.randint(0, 100, (30, 3), dtype=np.int64),
        np.random.randint(0, 100, (30, 3), dtype=np.int32),
        np.random.randint(0, 254, (30, 3), dtype=np.uint8)
    ])
    def test_validation_func_valid_values(self, values):
        assert isinstance(ValidatedArray(values), ValidatedArray)

    def test_get_set_item(self):
        test_data = ValidatedArray(np.random.rand(10,3))
        temp = test_data
        assert id(temp) == id(test_data)
        temp2 = test_data.copy()
        temp2_deep = test_data.copy(deep=True)
        assert id(temp2) != id(test_data)
        assert id(temp2.arr) == id(test_data.arr)
        assert id(temp2_deep.arr) != id(test_data.arr)
        test_data[0, 0] = 111
        assert np.all(temp == test_data)
        assert np.all(temp2 == test_data) # change in test_data will be observed in the view of temp2

        assert np.all(temp == test_data)


    def test_copying(self):
        test_data = ValidatedArray(np.random.rand(10,3))
        a = test_data       # id(a) == id(test_data) and id(a.arr) == id(test_data.arr)
        assert id(a) == id(test_data)
        assert id(a.arr) == id(test_data.arr)
        b = test_data.copy() # array is still a view, but different id
        assert id(b) != id(test_data)
        assert id(b.arr) == id(test_data.arr)
        c = test_data.copy(deep=True)
        assert id(c) != id(test_data)
        assert id(c.arr) != id(test_data.arr)
        a[0, 0] = 100
        np.testing.assert_array_equal(a, b)
        assert a[0, 0] != c[0, 0]
        assert np.all(a[1, :] == c[1, :])



class TestDataArray1D:
    def test_initialisation(self):
        a: ValidatedVector = ValidatedVector(np.ones((5,)))
        assert a.ndim == 1

        b: ValidatedVector = ValidatedVector(np.ones((5,1)))
        assert b.ndim == 1

        c: ValidatedVector = ValidatedVector(np.ones((1, 5)))
        assert c.ndim == 1

        # with pytest.raises( ValueError ):
        assert np.array(42).ndim == 0
        with pytest.raises(ValueError):
            d: ValidatedVector = ValidatedVector(np.array(42))

        with pytest.raises( ValueError ):
            ValidatedVector(np.random.randn(5, 3, 3))

        assert np.all(ValidatedVector(np.array([[1, 2]])) == np.array([[1, 2]]))


class TestDataArray2D:
    def test_initialisation(self):
        a: ValidatedArray2D = ValidatedArray2D(np.ones((5,3)))
        assert a.ndim == 2

        with pytest.raises( TypeError ):
            ValidatedArray2D('bacs')

class TestDataArrayNx2:
    def test_initialisation(self):
        a: ValidatedArrayNx2 = ValidatedArrayNx2(np.ones((5,2)))
        assert a.ndim == 2
        assert a.shape[1] == 2
        assert len(a) == 5

    def test_invalid_values(self):
        for val in (np.random.rand(5, 3), np.random.rand(10, 2, 4), np.random.rand(5, 1)):
            with pytest.raises(ValueError):
                ValidatedArrayNx2(val)

