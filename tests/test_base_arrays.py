# noinspection PyPackageRequirements
import pytest
from abc import ABC, abstractmethod

from typing import Any

# noinspection PyPackageRequirements
import numpy as np
# noinspection PyPackageRequirements
from numpydantic import NDArray, Shape
# noinspection PyPackageRequirements
from pydantic import BaseModel

from pchandler.v2.base_arrays import BaseArray, make_ndarray_type, NpMixinT


@pytest.fixture
def array():
    return BaseArray(arr=np.random.rand(100,100))

class TestMakeNpydanticType:
    def test_shape_definitions(self):
        assert NDArray[Shape['*'],Any] == make_ndarray_type(None)
        assert NDArray[Shape['*, *'],Any] == make_ndarray_type(None, None)
        assert NDArray[Shape['*, *, *'],Any] == make_ndarray_type(None, None, None)
        assert NDArray[Shape['8'],Any] == make_ndarray_type(8)
        assert NDArray[Shape['8'],Any] == make_ndarray_type('8')
        assert NDArray[Shape['2, 4, 7, 9'],Any] == make_ndarray_type(2, 4, 7, 9)
        assert NDArray[Shape['*, ...'], Any] == make_ndarray_type()

    def test_type_definitions(self):
        assert NDArray[Shape['*'],np.float32] == make_ndarray_type(None, dtype=np.float32)
        assert NDArray[Shape['*'],np.float64] == make_ndarray_type(None, dtype=np.float64)
        assert NDArray[Shape['*'],np.uint8] == make_ndarray_type(None, dtype=np.uint8)
        assert NDArray[Shape['*'],np.uint16] == make_ndarray_type(None, dtype=np.uint16)
        assert NDArray[Shape['*'],np.int32] == make_ndarray_type(None, dtype=np.int32)
        assert NDArray[Shape['*'],np.bool_] == make_ndarray_type(None, dtype=np.bool_)
        assert NDArray[Shape['*, ...'], np.bool_] == make_ndarray_type(dtype=np.bool_)


class BaseTests(ABC):
    cls = None

    def test_class_level_definition(self):
        assert 'arr' in self.cls.model_fields
        assert hasattr(self.cls, 'model_fields')    # ensure we stick with the 'arr' attribute naming convention
        assert hasattr(self.cls, 'model_config')
        assert isinstance(self.cls.model_fields['arr'].annotation, NDArray)
        assert issubclass(self.cls, BaseModel)
        assert issubclass(self.cls, NpMixinT)

    def test_class_level_config_defaults(self):
        # These should be defined but flexible on deciding the default values
        assert 'frozen' in self.cls.model_config
        assert 'extra' in self.cls.model_config

        # These should always exist to ensure validation and functions with custom Types
        #   - serialisation may not be guaranteed.
        assert self.cls.model_config['arbitrary_types_allowed'] == True
        assert self.cls.model_config['revalidate_instances'] == 'always'
        assert self.cls.model_config['validate_assignment'] == True

    def test_has_methods(self):
        for name in ('coerce', 'freeze', '__array_interface__', '__setitem__', '__getitem__',
                     'shape', 'dtype', 'ndim', 'base', 'size', '__len__'):
            assert hasattr(self.cls, name)

    def test_array_ufunc(self):
        pass

    def test_array_signature(self):
        pass

    @abstractmethod
    def test_initialisation(self):
        raise NotImplementedError()

    @abstractmethod
    def test_np_object_methods(self):
        pass

    @abstractmethod
    def test_np_class_methods(self):
        pass

    @abstractmethod
    def test_np_operator_mixins(self):
        pass


class TestBaseArray(BaseTests):
    cls = BaseArray

    def test_initialisation(self):
        a = self.cls(arr=np.random.rand(100, 100))
        assert isinstance(a, self.cls)

    class TestObjectMethods:
        cls = BaseArray

        def test_numpy_properties(self):
            a = self.cls(arr=np.random.rand(100, 100, 100, 100))
            assert a.shape == (100, 100, 100, 100)
            assert a.dtype == np.float64
            assert a.ndim == 4
            assert a.base is None
            assert a.size == 100**a.ndim

        def test_len(self):
            a = self.cls(arr=np.random.rand(100))
            with pytest.raises(NotImplementedError):
                len(a)

        def test_get_set_item(self):
            a = self.cls(arr=np.random.rand(100,100))
            # Test getitem
            assert np.all(a[:, 0] == a.arr[:, 0])

            # Test setitem
            a[0, 0] = 13
            assert a[0,0] == 13

        def test_array_assignment(self):
            a = self.cls(arr=np.random.rand(100,100))
            assert a.shape == (100, 100)

            # Test set
            a.arr = np.random.rand(10, 10).astype(np.float32)

            assert a.shape == (10, 10)
            assert a.dtype == np.float32
            assert a.ndim == 2
            assert a.base is None
            assert a.size == 10**a.ndim

        def test_array_copy(self):
            a = self.cls(arr=np.random.rand(100,100))
            b = a.copy_array()
            assert np.all(np.isclose(b, a))
            assert id(a) != id(b)

        def test_get_view(self):
            a = self.cls(arr=np.random.rand(100,100))
            b = a.get_view()
            assert id(a) != id(b)
            assert b.base is not None

    def test_initialised_obj_not_a_view(self):
        a = np.ones((10,10))
        obj1 = self.cls(arr=a)
        obj2 = self.cls(arr=a)

        assert id(obj1) != id(a)
        assert id(obj1.arr) != id(a)

        assert id(obj2) != id(a)
        assert id(obj2.arr) != id(a)

        assert id(obj1) != id(obj2)
        assert id(obj1.arr) != id(obj2.arr)

    @pytest.mark.parametrize('array', (([1, 2, 3]), (1, 2, 3)))
    def test_coerce_to_array(self, array):
        a = self.cls(arr=array) # type: ignore
        assert a.dtype == np.int64
        assert a.ndim == 1
        assert a.shape == (3,)
        assert np.all(np.isclose(a, np.array([1, 2, 3])))

    @pytest.mark.parametrize('array', ({'dict': 'Should fail'}, 'String is bad', {1, 2, 3}, True, 1.3, None))
    def test_incorrect_types(self, array):
        with pytest.raises(TypeError):
            self.cls(arr=array) # type: ignore

    def test_np_class_methods(self):
        a = self.cls(arr=np.ones((100, 3)))
        b = np.zeros_like(a)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert b.ndim == 2
        assert a.shape == (100, 3)
        assert np.all(b == 0)

        b = np.reshape(a, (3, 100)) # a view
        assert a.shape == (100, 3)
        assert b.shape == a.T.shape
        assert b.base is not None
        assert isinstance(a, self.cls)
        assert isinstance(b, np.ndarray)

        b = np.reshape(a, (5, 60))
        assert a.shape != (5, 60)
        assert b.shape == (5, 60)

        assert np.sum(b) == 300 == np.sum(a)

    def test_np_operator_mixins(self):
        a = self.cls(arr=np.ones((100, 3)))

        # Subtract
        a = a - 1.0
        assert np.all(np.isclose(a, 0))
        assert isinstance(a, self.cls)    # change after assignment

        # Add
        a = a + 2.0
        assert np.all(np.isclose(a, 2))
        assert isinstance(a, self.cls)    # change after assignment

        # Divide
        a = a / 0.5
        assert np.all(np.isclose(a, 4))
        assert isinstance(a, self.cls)    # change after assignment

        # Multiply
        a = a * 3
        assert np.all(np.isclose(a, 12))
        assert isinstance(a, self.cls)    # change after assignment

    def test_np_object_methods(self):
        a = self.cls(arr=np.random.rand(10, 3))
        assert a.min() >= 0
        assert a.max() <= 1

