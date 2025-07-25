from abc import ABC, abstractmethod
from typing import Any
import warnings

import numpy as np
import numpy.typing as npt
import pytest
from numpydantic import NDArray, Shape
from numpydantic.exceptions import DtypeError, ShapeError
from pydantic import ConfigDict, ValidationError

from pchandler.base_arrays import (
    ArrayNx2,
    ArrayNx3,
    BaseArray,
    BaseVector,
    FixedLengthArray,
    HomogeneousArray,
    SampleArray,
    make_ndarray_type,
)


@pytest.fixture(scope="function")
def random_1() -> npt.NDArray[np.float32]:
    return np.random.rand(10,3).astype(np.float32)

@pytest.fixture(scope="function")
def random_2() -> npt.NDArray[np.float64]:
    return np.random.rand(10,3).astype(np.float64)

@pytest.fixture(scope="function")
def array_1(random_1: npt.NDArray[np.float32]) -> BaseArray:
    return BaseArray(arr=random_1)

@pytest.fixture(scope="function")
def array_2(random_2: npt.NDArray[np.float64]) -> BaseArray:
    return BaseArray(arr=random_2)

@pytest.fixture(scope="function")
def random_view(random_1: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return random_1.view()

class TestNpydanticType:
    """
    Tests to support the make_ndarray_type function as a simple helper function
    """
    def test_shape_definitions(self) -> None:
        """
        Tests the use of the make_ndarray_type function for creating numpydantic type objects
        Returns
        -------

        """
        assert NDArray[Shape["*"], Any] == make_ndarray_type(None)
        assert NDArray[Shape["*, *"], Any] == make_ndarray_type(None, None)
        assert NDArray[Shape["*, *, *"], Any] == make_ndarray_type(None, None, None)
        assert NDArray[Shape["8"], Any] == make_ndarray_type(8)
        assert NDArray[Shape["8"], Any] == make_ndarray_type("8")
        assert NDArray[Shape["2, 4, 7, 9"], Any] == make_ndarray_type(2, 4, 7, 9)
        assert NDArray[Shape["*, ..."], Any] == make_ndarray_type()

    def test_type_definitions(self) -> None:
        assert NDArray[Shape["*"], np.float32] == make_ndarray_type(None, dtype=np.float32)
        assert NDArray[Shape["*"], np.float64] == make_ndarray_type(None, dtype=np.float64)
        assert NDArray[Shape["*"], np.uint8] == make_ndarray_type(None, dtype=np.uint8)
        assert NDArray[Shape["*"], np.uint16] == make_ndarray_type(None, dtype=np.uint16)
        assert NDArray[Shape["*"], np.int32] == make_ndarray_type(None, dtype=np.int32)
        assert NDArray[Shape["*"], np.bool_] == make_ndarray_type(None, dtype=np.bool_)
        assert NDArray[Shape["*, ..."], np.bool_] == make_ndarray_type(dtype=np.bool_)

    def test_numpydantic_type_usage(self) -> None:
        # Test type validation
        float_64_validator = make_ndarray_type(None, dtype=np.float64)

        with pytest.raises(DtypeError):
            float_64_validator(np.random.rand(10,3).astype(np.float32))

        # Test shape validation
        shape_10_3 = make_ndarray_type(10, 3)
        with pytest.raises(ShapeError):
            shape_10_3(np.random.rand(10, 4))
        with pytest.raises(ShapeError):
            shape_10_3(np.random.rand(3, 3))


class TestBaseArray:
    cls = BaseArray

    def test_initialisation(self) -> None:
        data = np.random.rand(100, 100)

        # Tests on default initialisation (value passed is a reference)
        a = BaseArray(arr=data)
        assert isinstance(a, self.cls)
        assert a is not data                    # base id does not match
        assert a.arr is data                    # array id matches
        assert np.all(a == data)                # Values are identical
        assert a.arr.base is None               # ensure it's not a view

        # Tests on a copy that is passed
        a_copied = BaseArray(arr=data.copy())
        assert a is not a_copied                # New object
        assert a_copied is not data             # Data differs to new array
        assert a_copied.arr is not data         # Array is a new object, not a copy
        assert a_copied.arr.base is None        # Ensure it's a copy and not a view
        assert np.all(a_copied == data)         # Values are identical

        # Tests on a view that is passed
        a_view = BaseArray(arr=data.view())
        assert a is not a_view                  # New object
        assert a_view is not data               # Also a new object
        assert a_view.arr is not data           # View creates a new object..
        assert a_view.arr.base is data          # But it does have a base which is the data object
        assert np.all(a_view == data)           # Values are identical

    @pytest.mark.parametrize("input_values", ({"dict": "Should fail"}, "String is bad", {1, 2, 3}, None))
    def test_invalid_input_types(self, input_values: Any) -> None:
        with pytest.raises(ValidationError):
            self.cls(arr=input_values)

    def test_general_base_model(self) -> None:
        assert "arr" in self.cls.model_fields
        assert hasattr(self.cls, "model_fields")  # ensure we stick with the 'arr' attribute naming convention
        assert hasattr(self.cls, "model_config")
        assert isinstance(self.cls.model_fields["arr"].annotation, NDArray)
        assert issubclass(self.cls, BaseArray)

    def test_model_config(self) -> None:
        assert self.cls.model_config["arbitrary_types_allowed"] == True
        assert self.cls.model_config["validate_assignment"] == True
        assert self.cls.model_config["revalidate_instances"] == "never"
        assert self.cls.model_config["validate_default"] == True
        assert self.cls.model_config["strict"] == True
        assert self.cls.model_config["frozen"] == False
        assert self.cls.model_config["extra"] == "ignore"
        assert self.cls.model_config["serialize_by_alias"] == False
        assert self.cls.model_config["populate_by_name"] == False

        # Test validate assignment
        # noinspection PyArgumentList
        a = BaseArray(arr=[1, 2, 3], dummy_field=2)
        with pytest.raises(ValidationError):
            # noinspection PyTypeChecker
            a.arr = "False"

        # Test frozen is false
        a.arr = np.array([2, 2])
        assert np.all(a.arr == [2, 2])

        # Test extra fields ignore
        assert not hasattr(a, "dummy_field")

        # Without the appropriate fields, cannot test for:
        # - revalidate instances
        # - strict
        # - frozen (frozen assignment)
        # - serialize_by_alias
        # - populate_by_name

    def test_coerce_to_numpy_valid(self) -> None:
        # Case value is True / 1
        # noinspection PyTypeChecker
        a = BaseArray(arr=True)
        assert a.shape == (1, )
        assert a.dtype == np.bool_
        assert a.arr == True

        # Case value is a list of ints
        # noinspection PyTypeChecker
        b = BaseArray(arr=[1, 3, 2, 4, -1])
        assert b.shape == (5,)
        assert b.dtype == np.int64
        assert np.all(b == [1, 3, 2, 4, -1])

        # Case of tuple of floats
        # noinspection PyTypeChecker
        c = BaseArray(arr=(2.3, 2.2, 4.5))
        assert c.shape == (3,)
        assert c.dtype == np.float64
        assert np.all(c == [2.3, 2.2, 4.5])

        # Case of BaseArray type
        # noinspection PyTypeChecker
        d = BaseArray(arr=c)
        assert d.shape == (3,)
        assert d.dtype == np.float64
        assert np.all(d == [2.3, 2.2, 4.5])

    def test_coerce_to_numpy_invalid(self) -> None:
        # Case using a set is an object type
        with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
            BaseArray(arr={1, 2, 3})

        # Fails on unsupported objects
        with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
            BaseArray(arr=object())

        # Dict doesn't work
        with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
            BaseArray(arr={'abc': 124})

        # Not supporting complex arrays at this point
        with pytest.raises(ValidationError):
            BaseArray(arr=np.ones(3, dtype=np.complex128))

        # Fail on a list of strings
        with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
            BaseArray(arr=['a', '23'])

    def test_freeze(self) -> None:
        warnings.warn("This method is deprecated", DeprecationWarning)
        assert True

    def test_array_interface(self, random_1: npt.NDArray[np.float32], random_2: npt.NDArray[np.float64]) -> None:
        # The array interface should be identical to the input if it is not a copy
        value = BaseArray(arr=random_1)
        assert random_1.__array_interface__ == value.__array_interface__
        assert random_1.__array_function__ == value.arr.__array_function__

        # Case of a copy
        value_copy = BaseArray(arr=random_2.copy())
        assert random_2.__array_interface__ != value_copy.__array_interface__

    def test_transpose(self, random_1: npt.NDArray[np.float32]) -> None:
        # Test the transpose function matches that of numpy
        value = BaseArray(arr=random_1)

        assert np.all(random_1.T == value.T)
        assert value.T.shape == (3, 10)

    def test_shape(self, random_1: npt.NDArray[np.float32]) -> None:
        # Test that the shape returns the same as the original numpy object
        value = BaseArray(arr=random_1)
        assert value.shape == random_1.shape

    def test_dtype(self, random_1: npt.NDArray[np.float32]) -> None:
        # Test that the dtype matches
        value = BaseArray(arr=random_1)
        assert value.dtype == random_1.dtype

    def test_ndim(self, random_1: npt.NDArray[np.float32]) -> None:
        # Test the NDIM match
        value = BaseArray(arr=random_1)
        assert value.ndim == random_1.ndim == 2

    def test_base(self, random_view: npt.NDArray[np.float32]) -> None:
        # Test the base property to show if a view
        assert random_view.base is not None

        value = BaseArray(arr=random_view)
        assert value.base is not None
        assert value.base.shape == random_view.shape
        assert value.base.dtype == random_view.dtype

    def test_size(self, array_1: BaseArray) -> None:
        assert array_1.size == np.prod(np.array(array_1.shape))

    def test_min(self) -> None:
        # noinspection PyTypeChecker
        value = BaseArray(arr=[[1, 2, 3, 4], [2, -4, 5, -23.2]])
        assert value.min() == -23.2
        assert np.min(value) == -23.2

    def test_max(self) -> None:
        # noinspection PyTypeChecker
        value = BaseArray(arr=[[1, 2, 3, 4], [2, -4, 5, -23.2]])
        assert value.max() == 5
        assert np.max(value) == 5

    def test_len(self, array_1: BaseArray) -> None:
        assert len(array_1) == 10

    def test_copy_shallow(self, array_1: BaseArray) -> None:
        arr_2 = array_1.copy(deep=False)
        assert arr_2 is not array_1         # The base object differs
        assert arr_2.arr is array_1.arr     # But the array object is just a reference
        assert np.all(arr_2 == array_1)     # Copy should have exactly equal values (no precision loss)

    def test_copy_deep(self, array_1: BaseArray) -> None:
        # Basic case
        arr_2 = array_1.copy(deep=True)
        assert arr_2 is not array_1                     # The base object differs
        assert id(arr_2.arr) is not id(array_1.arr)     # The array object is also new
        assert np.all(arr_2 == array_1)

        # Test on passing of positional object
        rand = np.random.rand(100,4)
        arr_3 = array_1.copy(rand, deep=True)
        assert np.all(arr_3 == rand)
        assert arr_3 is not rand
        assert id(arr_3.arr) is not id(rand)

        # Test on passing of positional object
        rand2 = np.random.rand(100,4)
        arr_4 = array_1.copy(update={'arr': rand2}, deep=True)
        assert np.all(arr_4 == rand2)
        assert arr_4 is not rand2
        assert id(arr_4.arr) is not id(rand2)

    def test_copy_update(self, array_1: BaseArray) -> None:
        # Test the positional array update object
        # numpy array
        rand = np.random.rand(100,3)
        array_3 = array_1.copy(rand)
        assert np.all(array_3.arr == rand)
        assert array_3.arr is rand
        assert array_3.arr is not array_1.arr
        assert array_3 is not array_1

        # As BaseArray
        array_4 = array_1.copy(BaseArray(arr=rand))
        assert np.all(array_4.arr == rand)
        assert array_4.arr is rand
        assert array_4.arr is not array_1.arr
        assert array_4 is not array_1

        # As a list
        array_5 = array_1.copy([2, 4, 5])
        assert np.all(array_5.arr == [2, 4, 5])
        assert array_5.arr is not array_1.arr
        assert array_5 is not array_1

        # And as an update kwarg
        array_6 = array_1.copy(update={'arr': rand})
        assert np.all(array_6.arr == rand)
        assert array_6.arr is rand
        assert array_6.arr is not array_1.arr
        assert array_6 is not array_1

    def test_numpy_functions(self) -> None:
        """
        Test a number of numpy based functions
        """
        a = BaseArray(arr=np.ones((5, 3)))
        b = np.add(a, 1)
        assert np.all(b == 2)

        c = np.mean(a, axis=0)
        assert np.all(c == 1.0)
        assert c.shape == (3,)
        assert isinstance(b, np.ndarray)
        assert np.allclose(a, 1)

    def test_getitem(self, array_1: npt.NDArray[np.floating | np.integer| np.bool_]) -> None:
        data = [[1, 2, 3,], [4, 5, 6], [-1.2, -5.6, -9.4], [2, -2, 2]]
        array_np = np.array(data)
        # noinspection PyTypeChecker
        array_base = BaseArray(arr=data)

        # Case 1 - Single indexing
        result = array_base[3]
        assert np.all(array_np[3] == result)
        assert array_np[3] is not result
        assert isinstance(result, BaseArray)

        # Case 2 - Multi-sampling, no duplicates
        result = array_base[[0, 3]]
        assert np.all(array_np[[0, 3]] == result)
        assert array_np[[0, 3]] is not result
        assert isinstance(result, BaseArray)

        # Case 3 - Multi-sampling, duplicates
        result = array_base[[0, 1, 1, 3, 3]]
        assert np.all(array_np[[0, 1, 1, 3, 3]] == result)
        assert len(result) == 5
        assert result.shape == (5,3)
        assert id(array_np[[0, 1, 1, 3, 3]]) is not id(result)
        assert isinstance(result, BaseArray)

        # Case 4 - Slice
        result = array_base[::2]
        assert np.all(array_np[::2] == result)
        assert np.all(result == array_base[[0, 2]])
        assert isinstance(result, BaseArray)

        # Case 4 - Multi-dimension indexing
        # noinspection PyTypeChecker
        result = array_base[[0, 1, 1, 3], [2, 1, 2, 1]]
        assert np.all(array_np[[0, 1, 1, 3], [2, 1, 2, 1]] == result)
        assert np.all(result == [3, 5, 6, -2])
        assert isinstance(result, BaseArray)

        # Case 5 - Unsupported type (string)
        with pytest.raises(IndexError):
            # noinspection PyTypeChecker
            b = array_base["1"]

    def test_setitem(self) -> None:
        data = np.zeros((10, 3))
        array = BaseArray(arr=data)

        # Case 1 - Single indexing
        a = array.copy()
        # Row indexing
        a[3] = 24
        assert np.all(a[3] == 24)
        assert np.all(a[2] == 0)
        # Element indexing
        a[1, 1] = 22
        assert a[1, 1] == 22
        assert a[1, 0] == 0
        assert isinstance(a, BaseArray)

        # Case 2 - Multi-indexing, constant
        a = array.copy()
        a[[1, 3, 4]] = 3
        for i in (1, 3):
            assert np.all(a[i] == 3)
        assert isinstance(a, BaseArray)

        # Case 3 - Multi-indexing, array
        a = array.copy()
        # Rows
        a[[1, 3, 4]] = [[0,1,2], [1, 2, 3], [2, 3, 4]]
        assert np.all(a[[1, 3, 4]] == [[0,1,2], [1, 2, 3], [2, 3, 4]])
        assert isinstance(a, BaseArray)

        # Case 4 - Slice
        a = array.copy()
        a[::4] = [[0,1,2], [1, 2, 3], [2, 3, 4]]
        assert np.all(a[::4] == [[0,1,2], [1, 2, 3], [2, 3, 4]])
        assert isinstance(a, BaseArray)

        # Case 4 - Multi-dimension indexing
        a = array.copy()
        # noinspection PyTypeChecker
        a[[0, 3, 5, 7], [1, 0, 1, 2]] = [2, 4, 6, 8]
        # noinspection PyTypeChecker
        assert np.all(a[[0, 3, 5, 7], [1, 0, 1, 2]] == [2, 4, 6, 8])

        # Case 5 - Unsupported type (string)
        with pytest.raises(IndexError):
            a['1'] = 3

    def test_mixins(self, random_1: npt.NDArray[np.float32], random_2: npt.NDArray[np.float64]) -> None:
        # Test that the behaviour results in the same logic as that of numpy
        a = BaseArray(arr=random_1)
        b = BaseArray(arr=random_2)
        assert np.all((random_1 < random_2) == (a < b))
        assert np.all((random_1 <= random_2) == (a <= b))
        assert np.all((random_1 > random_2) == (a > b))
        assert np.all((random_1 >= random_2) == (a >= b))
        assert np.all((random_1 == random_2) == (a == b))
        assert np.all((random_1 != random_2) == (a != b))

        a = self.cls(arr=np.random.rand(100, 100))

        # Base doesn't support inplace or math operators by default
        with pytest.raises(TypeError):
            a += 1
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            a = a - 1.0
        with pytest.raises(TypeError):
            a = a + 2.0
        with pytest.raises(TypeError):
            a = a / 0.5
        with pytest.raises(TypeError):
            a = a * 3

        # Ensure logical checks produce a boolean type
        b = a > 3
        assert b.dtype == np.bool_
        b = a < 3
        assert b.dtype == np.bool_
        b = a <= 3
        assert b.dtype == np.bool_
        b = a >= 3
        assert b.dtype == np.bool_
        b = a == 3
        assert b.dtype == np.bool_
        b = a != 3
        assert b.dtype == np.bool_
        b = 3 > a
        assert b.dtype == np.bool_
        b = 3 < a
        assert b.dtype == np.bool_
        b = 3 <= a
        assert b.dtype == np.bool_
        b = 3 >= a
        assert b.dtype == np.bool_
        # noinspection PyTypeChecker
        b = 3 == a
        # noinspection PyUnresolvedReferences
        assert b.dtype == np.bool_
        # noinspection PyTypeChecker
        b = 3 != a
        # noinspection PyUnresolvedReferences
        assert b.dtype == np.bool_

    def test_reshape(self) -> None:
        a = self.cls(arr=np.ones((100, 3)))
        b = np.zeros_like(a).copy()

        # Ensure a new object but all shapes and values match
        assert a is not b
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert b.ndim == 2
        assert a.shape == (100, 3)
        assert np.all(b == 0)
        assert a.size == 300

        # Compare against a view
        b = np.reshape(a, (3, 100))     # a view
        assert a.shape == (100, 3)            # Original shape has not changed
        assert b.shape == (3, 100)            # New shape is different
        assert b.base is not a.arr            # The view base is a copy
        assert isinstance(a, self.cls)        # Base is not changed
        assert isinstance(b, np.ndarray)      # Reshape returns a numpy array object
        assert b.shape == a.T.shape           # Transpose should match

        # Test another shape more complex
        b = np.reshape(a.arr, (5, 60))
        assert a.shape != (5, 60)
        assert b.shape == (5, 60)

        # Check that all values are equal
        assert np.sum(b) == 300 == np.sum(a)
        assert b.base is a.arr                  # This time the array is the base


class TestSamplingArray:
    cls = SampleArray

    def test_initialisation(self) -> None:
        a = self.cls(arr=np.random.rand(100, 4))

        assert isinstance(a.arr, np.ndarray)
        assert isinstance(a, self.cls)
        assert a.shape == (100, 4)

    def test_methods(self) -> None:
        for name in ("create_mask", "sample", "reduce", "extract"):
            hasattr(self.cls, name)

    def test_slice_extract_reduce_1d(self) -> None:

        index_s = slice(0, 4, 1)
        index_b = [True, True, True, True, False, False, False, False, False, False]
        (index_np_b := np.zeros(10, dtype=np.bool_))[index_b] = True
        index_np_i = np.array([0, 1, 2, 3])
        a = np.random.rand(
            10,
        )

        for index in (index_s, index_b, index_np_b, index_np_i):
            b = self.cls(arr=a.copy())
            mask = b.create_mask(index)
            assert np.all(mask == np.array([True, True, True, True, False, False, False, False, False, False]))

            assert mask.shape == a.shape == b.shape

            sampled = b.sample(index)
            assert np.all(sampled.arr == a[index])

            assert sampled is not b
            assert sampled.arr is not a
            assert sampled.arr is not b

            c = b.copy()
            d = c.extract(index)

            b.reduce(~mask)

            # Extract should be close to the invert reduced
            assert np.allclose(b, c)
            assert np.allclose(sampled, d)
            assert c.shape[0] == 6
            assert b.shape[0] == 6
            assert a.shape[0] == 10
            assert d.shape[0] == 4
            print("Finished")

    def test_slice_extract_reduce_2d(self) -> None:
        a = np.random.rand(2, 5)

        index_s = slice(0, 3, 1), slice(0, 3, 1)
        index_b = [[True, True, True, False, False], [True, True, True, False, False]]
        (index_np_b := np.zeros((2, 5), dtype=np.bool_))[index_b] = True
        index_np_i = np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 2, 0, 1, 2])

        b = self.cls(arr=a.copy())

        for index in (index_s, index_b, index_np_b, index_np_i):
            mask = b.create_mask(index)
            assert np.all(mask == np.array([[True, True, True, False, False], [True, True, True, False, False]]))

        with pytest.raises(IndexError):
            b.create_mask("asdasd")

        with pytest.raises(ValidationError):
            a = self.cls(arr=np.zeros((10, 3), dtype=np.complex128))

        assert mask.shape == a.shape == b.shape

        for index in (index_s, index_b, index_np_b, index_np_i):
            b = self.cls(arr=a.copy())

            sampled = b.sample(index)

            assert sampled is not b
            assert sampled.arr is not a
            assert sampled.arr is not b

            c = b.copy()
            d = c.extract(index)

            b.reduce(~mask)

            # Extract should be close to the invert reduced
            assert np.allclose(b, c)
            assert np.allclose(sampled, d)
            assert c.shape[0] == 4  # Remainder after extract
            assert b.shape[0] == 4  # Reduced
            assert a.shape[0] == 2  # Original
            assert a.size == 10  # Original
            assert d.shape[0] == 6  # Sampled / Extract
            print("Finished")

    def test_numpy_funcs(self) -> None:
        a = self.cls(arr=np.random.rand(100, 4))
        assert np.mean(a) is not None

    def test_properties(self) -> None:
        assert self.cls(arr=np.random.rand(100, 4)).ndim == 2

    def test_np_operator_mixins(self) -> None:
        a = self.cls(arr=np.random.rand(100, 4))
        with pytest.raises(TypeError):
            a += 1

    def test_view(self) -> None:
        a = self.cls(arr=np.random.rand(100, 4))
        assert a.base is None

    def test_other(self) -> None:
        pass


class TestFixedLengthAndMixins:
    cls = FixedLengthArray

    def test_initialisation(self):
        a = self.cls(arr=np.random.rand(10, 3))
        assert isinstance(a, self.cls)

    def test_properties(self):
        pass

    def test_numpy_funcs(self):
        pass

    def test_other(self):
        pass

    def test_view(self):
        pass

    def test_methods(self):
        # __len__
        a = self.cls(arr=np.random.rand(10, 3))
        assert len(a) == 10

        # __iter__
        i = 0
        for row in a:
            assert np.allclose(row, a[i])
            i += 1
        assert i == 10

    def test_vector_mask(self):
        a = np.random.rand(20, 3)
        b = self.cls(arr=a.copy())

        mask = b.create_mask([0, 1, 2])

        assert np.sum(mask) == 3
        assert np.all(mask[:3])
        assert not np.all(mask[3:])
        assert mask.shape == (20,)

        mask = b.create_mask([[0, 1, 2], [0, 3, 4]])
        assert np.sum(mask) == 5
        assert np.all(mask[:5])
        assert not np.all(mask[5:])
        assert mask.shape == (20,)

        with pytest.raises(IndexError):
            b.create_mask((slice(0, 4, 1), slice(0, 8, 2)))

        with pytest.raises(ValueError):
            b.create_mask(np.ones_like(a, dtype=np.bool_))

        # Still create mask from multi-dimension array as np supports it
        b.create_mask(np.array([[0, 4, 5, 6], [2, 3, 4, 5]]))

    def test_sample_reduce_extract(self):
        a = np.random.rand(20, 3)
        b = self.cls(arr=a.copy())

        mask = b.create_mask([0, 2, 4, 6])
        sampled = b.sample(mask)

        assert sampled is not b
        assert np.all(sampled.arr == b.arr[mask, :])
        assert np.all(sampled.arr == b.arr[[0, 2, 4, 6], :])
        assert sampled.shape == (4, 3)

        c = b.copy()
        d = b.copy()

        extract = b.extract(mask)
        c.reduce(mask)
        d.reduce(~mask)

        assert np.all(extract.arr == sampled.arr)
        assert np.all(d.arr == b.arr)
        assert np.all(c.arr == sampled.arr)

    def test_np_operator_mixins(self):
        a = np.random.rand(10, 3)
        f = self.cls(arr=a + 3)
        g = self.cls(arr=np.full_like(a, 2))
        b = self.cls(arr=a.copy())

        # Left unary operators
        c = b + 1
        assert np.allclose(c, a + 1)
        assert np.all(b != c)
        assert isinstance(c, self.cls)
        c = b - 1
        assert np.allclose(c, a - 1)
        assert np.all(b != c)
        assert isinstance(c, self.cls)
        c = b * 2
        assert np.allclose(c, a * 2)
        assert np.all(b != c)
        assert isinstance(c, self.cls)
        c = b / 2
        assert np.allclose(c, a / 2)
        assert np.all(b != c)
        assert isinstance(c, self.cls)
        c = f // 1
        assert np.allclose(c, 3)
        assert isinstance(c, self.cls)
        c = f % 1
        assert np.allclose(c, a)
        assert isinstance(c, self.cls)
        c = g**3
        assert np.allclose(c, 8)
        assert isinstance(c, self.cls)

        c = 1 + b
        assert np.allclose(c, a + 1)
        assert np.all(b != c)
        assert isinstance(c, self.cls)
        c = 1 - b
        assert np.allclose(c, 1 - a)
        assert np.all(b != c)
        assert isinstance(c, self.cls)
        c = 2 * b
        assert np.allclose(c, 2 * a)
        assert np.all(b != c)
        assert isinstance(c, self.cls)
        c = 2 / b
        assert np.allclose(c, 2 / a)
        assert np.all(c > a)
        assert isinstance(c, self.cls)
        c = 3.5 // g
        assert np.allclose(c, 1)
        assert isinstance(c, self.cls)
        c = 4 % f
        assert np.allclose(c, 1 - a)
        assert isinstance(c, self.cls)
        c = 3**g
        assert np.allclose(c, 9)
        assert isinstance(c, self.cls)

        # In place operators
        b += 1
        assert np.allclose(b, a + 1)
        assert isinstance(c, self.cls)
        b -= 1
        assert np.allclose(b, a)
        assert isinstance(c, self.cls)
        b *= 3
        assert np.allclose(b, a * 3)
        assert isinstance(c, self.cls)
        b /= 3
        assert np.allclose(b, a)
        assert isinstance(c, self.cls)
        b += 1
        b %= 1
        assert np.allclose(b, a)
        assert isinstance(c, self.cls)
        b //= 1
        assert np.allclose(b, np.zeros_like(a))
        assert isinstance(c, self.cls)
        b.arr = np.full_like(b.arr, 3)
        b **= 3
        assert np.allclose(b, 27)
        assert isinstance(c, self.cls)


class TestBaseVector:
    def test_initialisation(self):
        with pytest.raises(ValidationError):
            BaseVector(arr=np.random.rand(10, 3))
        with pytest.raises(ValidationError):
            BaseVector(arr=np.random.rand(10, 2))

        vec = BaseVector(arr=np.random.rand(10))

        assert vec.size == 10
        assert len(vec) == 10
        assert np.sum(vec) < 10
        assert vec.shape == (10,)


class TestHomogeneousAndMixins:
    cls = HomogeneousArray

    def test_initialisation(self):
        a = HomogeneousArray(arr=np.zeros((10, 3)))
        assert isinstance(a, HomogeneousArray)
        assert np.all(a.arr == np.zeros((10, 3)))

    def test_properties(self):
        a = np.random.rand(10, 3)
        b = self.cls(arr=a)
        c = b.H
        assert a.shape != c.shape
        assert a.shape[0] == c.shape[0]
        assert np.all(c[:, 3] == 1)
        assert c.shape[1] == 4

    def test_view(self):
        pass

    def test_methods(self):
        a = np.random.rand(10, 3)
        b = self.cls(arr=a)
        c = b.H
        assert isinstance(c, np.ndarray)
        assert c.shape == (10, 4)
        assert np.all(b.arr == c[:, :3])

    def test_numpy_funcs(self):
        pass

    def test_np_operator_mixins(self):
        pass

    def test_other(self):
        pass


class TestArrayNx2:
    def test_initialisation(self):
        with pytest.raises(ValidationError):
            ArrayNx2(arr=np.random.rand(10, 3))

        array = ArrayNx2(arr=np.random.rand(10, 2))

        assert array.size == 20
        assert len(array) == 10
        assert np.sum(array) < 20
        assert array.shape == (10, 2)


class TestArrayNx3:
    def test_initialisation(self):
        with pytest.raises(ValidationError):
            ArrayNx3(arr=np.random.rand(10, 2))

        array = ArrayNx3(arr=np.random.rand(10, 3))

        assert array.size == 30
        assert len(array) == 10
        assert np.sum(array) < 30
        assert array.shape == (10, 3)


# class TestReadOnlyArray:
#     def test_initialisation(self):
#         array = ReadOnlyArray(arr=np.random.rand(10, 2))
#         with pytest.raises(ValidationError):
#             array.arr = np.random.randn(10, 2)
#
#
# class TestReadOnlyVector:
#     def test_initialisation(self):
#         array = ReadOnlyVector(arr=np.random.rand(10))
#
#         with pytest.raises(ValidationError):
#             array.arr = np.random.randn(10)


# class TestImageLike(BaseTests):
#     def test_initialisation(self): ...
#
#     def test_properties(self): ...
#
#     def test_methods(self): ...
#
#     def test_numpy_funcs(self): ...
#
#     def test_other(self): ...
#
#     def test_view(self): ...
#
#     def test_np_operator_mixins(self): ...
