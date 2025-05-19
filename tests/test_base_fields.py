import pytest
import numpy as np

from src.pchandler.base_fields import ValidatedField
from src.pchandler.base_fields import ValidatedNumpyField


def is_even(x):
    if not x % 2 == 0:
        raise ValueError("Not an even number")
    return x


class CustomCoerceField(ValidatedField):
    def __init__(self, *args, **kwargs):
        kwargs |= {'coerce': True}
        super().__init__(*args, **kwargs)


class CustomOptionalField(ValidatedField):
    def __init__(self, *args, **kwargs):
        kwargs |= {'optional': True}
        super().__init__(*args, **kwargs)


class CustomValidatorField(ValidatedField):
    def __init__(self, *args, **kwargs):
        kwargs |= {'validators': [is_even]}
        super().__init__(*args, **kwargs)


class CustomFields:
    name: CustomCoerceField = CustomCoerceField(str)
    length: CustomOptionalField = CustomOptionalField(float)
    weight: CustomValidatorField = CustomValidatorField(int)


class CustomObject:
    name: ValidatedField = ValidatedField(str, coerce=True)
    length: ValidatedField = ValidatedField(float, optional=True, default=None)
    frozen: ValidatedField = ValidatedField(float, freezable=True)
    weight: ValidatedField = ValidatedField(int, validators=[is_even])
    height: ValidatedField = ValidatedField(float, default=22.0, coerce=True)

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
        a = CustomFields.length
        assert hasattr(a, 'name')
        assert hasattr(a, 'private_name')
        assert hasattr(a, 'options_name')

        assert getattr(a, 'name') == 'length'
        assert getattr(a, 'private_name') == '_length'
        assert getattr(a, 'options_name') == '_length_options'

    @pytest.mark.parametrize('invalid_value', (False, {'abc': 123}, (1, 2, 3), 'asdasd'))
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


class CustomNpyVectorField(ValidatedNumpyField):
    __ndim__ = 1


class CustomNpyTripletArray(ValidatedNumpyField):
    __ndim__ = 2
    __shape__ = (None, 3)


class CustomTransformMatrixField(ValidatedNumpyField):
    __ndim__ = 2
    __shape__ = (3,3)


class CustomReadOnly(ValidatedNumpyField):
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


