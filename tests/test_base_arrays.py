# noinspection PyPackageRequirements
import pytest
from abc import ABC, abstractmethod

from typing import Any, get_args

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
        a = self.cls(arr=np.random.rand(100, 100, 100, 100))
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
            assert np.all(b == a)
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

    def test_coerce_to_array(self):
        a = self.cls(arr=[1, 2, 3])
        assert a.dtype == np.int64
        assert a.ndim == 1
        assert a.shape == (3,)
        assert np.all(a == np.array([1, 2, 3]))

        a = self.cls(arr=(1, 2, 3))
        assert a.dtype == np.int64
        assert a.ndim == 1
        assert a.shape == (3,)
        assert np.all(a == np.array([1, 2, 3]))

        with pytest.raises(TypeError):
            self.cls(arr={'flag': 'Should not coerce'})

        with pytest.raises(TypeError):
            self.cls(arr='String should not work')

        with pytest.raises(TypeError):
            self.cls(arr=True)

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
        a.arr = a - 1.0
        assert np.all(np.isclose(a, 0))

        # Add
        a.arr = a + 2.0
        assert np.all(np.isclose(a, 2))

        # Divide
        a.arr = a / 0.5
        assert np.all(np.isclose(a, 4))

        # Multiply
        a.arr = a * 3
        assert np.all(np.isclose(a, 12))

    def test_np_object_methods(self):
        a = self.cls(arr=np.random.rand(10, 3))
        assert a.min() >= 0
        assert a.max() <= 1





#
# class TestDataArray:
#     def test_mutability(self):
#         a = np.random.rand(5,3)
#         b = np.random.rand(5,3)
#         a_array = ValidatedArray(a)
#         original_id = id(a_array)
#         np.testing.assert_array_almost_equal(a_array, a)
#         a_array._arr = b
#         assert np.all(a_array == b)                 # Check the new values are assigned
#         assert np.all(a_array != a)                 # Check that the values have changed
#         assert original_id == id(a_array)           # Check that it's still the same object
#         assert isinstance(a_array, NpMixinT)        # Check it's still one of these classes
#
#     def test_with_descriptor(self):
#         a = TempAbc(np.array([[1, 2, 3], [2, 3, 4]]))
#         assert isinstance(a.abc, ArrayNx3)
#         assert np.all(a.abc == np.array([[1, 2, 3], [2, 3, 4]]))
#
#         b = TempAbc()
#         assert isinstance(b.abc, ArrayNx3)
#         assert np.all(b.abc == np.ones((5, 3)))
#
#         with pytest.raises(ValueError):
#             TempAbc(np.random.rand(10,10,10))
#
#     def test_immutability(self) -> None:
#         test_data = ValidatedArray(np.random.rand(5,3))
#         array2 = ReadOnlyArray(np.random.randn(5,3))
#         with pytest.raises( AttributeError ):
#             array2._arr = test_data._arr
#
#         test_data._arr[0, 0] = 3
#         # This blocks the user from assigning directly to the array as well
#         with pytest.raises( ValueError ):
#             array2._arr[0, 0] = 3
#
#     def test_numpy_functionality(self):
#         a: ValidatedArray = ValidatedArray(np.ones((5,3)))
#         b: ValidatedArray = a + np.ones(3) * 3
#         assert np.all(a != b)
#         c: ValidatedArray = b - np.ones(3) * 3
#         np.testing.assert_array_almost_equal(a, c)
#         d: np.ndarray|ValidatedArray = a != b
#         assert isinstance(d, np.ndarray)
#         assert np.issubdtype(d.dtype, np.bool_)
#         assert np.mean(b) == 4
#
#     def test_helper_properties(self):
#         a: ValidatedArray = ValidatedArray(np.ones((5,3)))
#         assert a.ndim == 2
#         assert a.shape == (5,3)
#         assert a.dtype == np.float64
#         assert a.size == np.prod(a.shape)
#         assert a.base is None
#
#     @pytest.mark.parametrize("values, error_type", [
#         (('str', {'23': 1}, {1, 3, 5}, False), TypeError),
#         ((None,), ValueError)])
#     def test_validation_func_invalid_values(self, values, error_type):
#         for val in values:
#             print(f'Trying {val=}')
#             with pytest.raises(error_type):
#                 ValidatedArray(val)
#
#     @pytest.mark.parametrize("values", [
#         np.random.randn(5).astype('f8'),
#         np.random.randn(1000,10000).astype('f4'),
#         np.random.randn(1000, 20, 3),
#         np.random.randint(0, 100, (30, 3), dtype=np.int64),
#         np.random.randint(0, 100, (30, 3), dtype=np.int32),
#         np.random.randint(0, 254, (30, 3), dtype=np.uint8)
#     ])
#     def test_validation_func_valid_values(self, values):
#         assert isinstance(ValidatedArray(values), ValidatedArray)
#
#     def test_get_set_item(self):
#         test_data = ValidatedArray(np.random.rand(10,3))
#         temp = test_data
#         assert id(temp) == id(test_data)
#         temp2 = test_data.copy()
#         temp2_deep = test_data.copy(deep=True)
#         assert id(temp2) != id(test_data)
#         assert id(temp2._arr) == id(test_data._arr)
#         assert id(temp2_deep._arr) != id(test_data._arr)
#         test_data[0, 0] = 111
#         assert np.all(temp == test_data)
#         assert np.all(temp2 == test_data) # change in test_data will be observed in the view of temp2
#
#         assert np.all(temp == test_data)
#
#
#     def test_copying(self):
#         test_data = ValidatedArray(np.random.rand(10,3))
#         a = test_data       # id(a) == id(test_data) and id(a.arr) == id(test_data.arr)
#         assert id(a) == id(test_data)
#         assert id(a._arr) == id(test_data._arr)
#         b = test_data.copy() # array is still a view, but different id
#         assert id(b) != id(test_data)
#         assert id(b._arr) == id(test_data._arr)
#         c = test_data.copy(deep=True)
#         assert id(c) != id(test_data)
#         assert id(c._arr) != id(test_data._arr)
#         a[0, 0] = 100
#         np.testing.assert_array_equal(a, b)
#         assert a[0, 0] != c[0, 0]
#         assert np.all(a[1, :] == c[1, :])
#
#
#
# class TestDataArray1D:
#     def test_initialisation(self):
#         a: VectorN = VectorN(np.ones((5,)))
#         assert a.ndim == 1
#
#         b: VectorN = VectorN(np.ones((5, 1)))
#         assert b.ndim == 1
#
#         c: VectorN = VectorN(np.ones((1, 5)))
#         assert c.ndim == 1
#
#         # with pytest.raises( ValueError ):
#         assert np.array(42).ndim == 0
#         with pytest.raises(ValueError):
#             d: VectorN = VectorN(np.array(42))
#
#         with pytest.raises( ValueError ):
#             VectorN(np.random.randn(5, 3, 3))
#
#         assert np.all(VectorN(np.array([[1, 2]])) == np.array([[1, 2]]))
#
#
# class TestDataArray2D:
#     def test_initialisation(self):
#         a: Array2d = Array2d(np.ones((5, 3)))
#         assert a.ndim == 2
#
#         with pytest.raises( TypeError ):
#             Array2d('bacs')
#
# class TestDataArrayNx2:
#     def test_initialisation(self):
#         a: ArrayNx2 = ArrayNx2(np.ones((5, 2)))
#         assert a.ndim == 2
#         assert a.shape[1] == 2
#         assert len(a) == 5
#
#     def test_invalid_values(self):
#         for val in (np.random.rand(5, 3), np.random.rand(10, 2, 4), np.random.rand(5, 1)):
#             with pytest.raises(ValueError):
#                 ArrayNx2(val)
#
# class TestDefaults:
#     def test_vector_2d(self):
#         a = Vector2()
#         assert np.all(a == 0)
#
#     def test_vector_3d(self):
#         b = Vector3()
#         assert np.all(b == 0)
#
#     def test_4x4_matrix(self):
#         c = TransformArray4x4()
#         assert np.all(c == np.eye(4))
#


