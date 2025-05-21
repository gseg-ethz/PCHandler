import pytest
import numpy as np

from src.pchandler.base_descriptors import Descriptor, CoerceDescriptor, OptionalDescriptor, ArrayDescriptor, FrozenDescriptor


def is_even(x):
    if not x % 2 == 0:
        raise ValueError("Not an even number")



class IsEvenField(Descriptor):
    def __init__(self, *args, **kwargs):
        kwargs |= {'validators': [is_even]}
        super().__init__(*args, **kwargs)


class MainObject:
    name: Descriptor = CoerceDescriptor(str)
    length: Descriptor = OptionalDescriptor(float, optional=False, default=None)
    frozen: Descriptor = FrozenDescriptor(float)
    weight: Descriptor = IsEvenField(int)
    height: Descriptor = CoerceDescriptor(float, default=22.0)

    def __init__(self, name, length, weight, frozen, height=None):
        self.name = name
        self.length = length
        self.weight = weight
        self.frozen = frozen
        self.height = height
        return


class TestCustomDescriptors:
    def test_base_field_parameters(self):
        custom_obj = MainObject(name='abc', length=None, weight=22, frozen=12.0)
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
        obj = MainObject(name='abc', length=None, weight=22, frozen=12.3)

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
        a = MainObject.length
        assert hasattr(a, 'name')
        assert hasattr(a, 'private_name')
        assert hasattr(a, 'options_name')

        assert getattr(a, 'name') == 'length'
        assert getattr(a, 'private_name') == '_length'
        assert getattr(a, 'options_name') == '_length_options'

    @pytest.mark.parametrize('invalid_value', (False, {'abc': 123}, (1, 2, 3), 'not_valid'))
    def test_invalid_type(self, invalid_value):
        obj: MainObject = MainObject('abc', None, 122, frozen=123.4)
        with pytest.raises(TypeError):
            obj.length = invalid_value

    def test_optional_field(self):
        with pytest.raises(ValueError):
            MainObject(name='abc', weight=None, length=None, frozen=12)

        obj: MainObject = MainObject('abc', None, 122, frozen=123.4)
        with pytest.warns(Warning):
            obj.height = None

    def test_coercion_field(self):
        obj: MainObject = MainObject('abc', None, 122, frozen=123.4)
        with pytest.raises(TypeError):
            obj.height = np.array([1, 2, 3])

    def test_delete(self):
        obj: MainObject = MainObject('abc', None, 122, frozen=123.4)

        del obj.length
        assert not hasattr(obj, '_length')

        with pytest.raises(ValueError):
            del obj.frozen


class CustomReadOnly(ArrayDescriptor):
    def __init__(self, *args, **kwargs):
        kwargs |= {'freezable': True}
        super().__init__(*args, **kwargs)


class CustomNpyFieldTestObject:
    vector: ArrayDescriptor = ArrayDescriptor(np.ndarray, coerce=True)
    read_only: ArrayDescriptor = ArrayDescriptor(np.ndarray, freezable=True)

    def __init__(self, vector=np.array([1, 2, 3]), read_only = np.ones((10,10))):
        self.vector = vector
        self.read_only = read_only

@pytest.fixture(scope="function")
def custom_npy_fld_obj():
    return CustomNpyFieldTestObject()

class TestCustomNumpyFields:
    def test_initialisation(self, custom_npy_fld_obj):
        read_only = np.ones((10, 10))

        assert np.all(custom_npy_fld_obj.read_only == read_only)

    def test_array_coercion(self, custom_npy_fld_obj):
        vector = [1, 2, 3]
        custom_npy_fld_obj.vector = vector
        assert np.all(np.isclose(custom_npy_fld_obj.vector, np.array(vector)))
        with pytest.raises(TypeError):
            custom_npy_fld_obj.vector = {'abc', 123}


    def test_read_only(self, custom_npy_fld_obj):
        with pytest.raises(AttributeError):
            custom_npy_fld_obj.read_only = np.random.rand(2, 2)

        with pytest.raises(ValueError):
            custom_npy_fld_obj.read_only[0, 0] = 1

        assert np.all(custom_npy_fld_obj.read_only == np.ones((10, 10)))


    def test_manual_override(self, custom_npy_fld_obj):
        custom_npy_fld_obj.__dict__['_read_only_options']._frozen = False


        custom_npy_fld_obj.read_only = np.array([1, 2, 3, 4, 5])
        assert np.all(custom_npy_fld_obj.read_only == np.array([1, 2, 3, 4, 5]))

