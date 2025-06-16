from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pytest
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, ValidationError

from pchandler.v2.base_arrays import (
    ArrayNx2,
    ArrayNx3,
    BaseArray,
    BaseVector,
    FixedLengthArray,
    HomoegeneousArray,
    ReadOnlyArray,
    ReadOnlyVector,
    SampleArray,
    make_ndarray_type,
)

base_config_fields = (
    "arbitrary_types_allowed",
    "validate_assignment",
    "revalidate_instances",
    "validate_default",
    "frozen",
    "extra",
)


class TestMakeNpydanticType:
    def test_shape_definitions(self):
        assert NDArray[Shape["*"], Any] == make_ndarray_type(None)
        assert NDArray[Shape["*, *"], Any] == make_ndarray_type(None, None)
        assert NDArray[Shape["*, *, *"], Any] == make_ndarray_type(None, None, None)
        assert NDArray[Shape["8"], Any] == make_ndarray_type(8)
        assert NDArray[Shape["8"], Any] == make_ndarray_type("8")
        assert NDArray[Shape["2, 4, 7, 9"], Any] == make_ndarray_type(2, 4, 7, 9)
        assert NDArray[Shape["*, ..."], Any] == make_ndarray_type()

    def test_type_definitions(self):
        assert NDArray[Shape["*"], np.float32] == make_ndarray_type(None, dtype=np.float32)
        assert NDArray[Shape["*"], np.float64] == make_ndarray_type(None, dtype=np.float64)
        assert NDArray[Shape["*"], np.uint8] == make_ndarray_type(None, dtype=np.uint8)
        assert NDArray[Shape["*"], np.uint16] == make_ndarray_type(None, dtype=np.uint16)
        assert NDArray[Shape["*"], np.int32] == make_ndarray_type(None, dtype=np.int32)
        assert NDArray[Shape["*"], np.bool_] == make_ndarray_type(None, dtype=np.bool_)
        assert NDArray[Shape["*, ..."], np.bool_] == make_ndarray_type(dtype=np.bool_)


class BaseTests(ABC):
    cls = None

    def test_class_level_definition(self):
        if self.cls is not None:
            assert "arr" in self.cls.model_fields
            assert hasattr(self.cls, "model_fields")  # ensure we stick with the 'arr' attribute naming convention
            assert hasattr(self.cls, "model_config")
            assert isinstance(self.cls.model_fields["arr"].annotation, NDArray)
            assert issubclass(self.cls, BaseModel)

    def test_class_level_config_defaults(self):
        if self.cls is not None:
            # These should be defined but flexible on deciding the default values
            assert "frozen" in self.cls.model_config
            assert "extra" in self.cls.model_config

            # These should always exist to ensure validation and functions with custom Types
            #   - serialisation may not be guaranteed.
            assert self.cls.model_config["arbitrary_types_allowed"] == True
            assert self.cls.model_config["revalidate_instances"] == "always"
            assert self.cls.model_config["validate_assignment"] == True
            assert self.cls.model_config["validate_default"] == True

    def test_has_methods(self):
        if self.cls is not None:

            for name in (
                "freeze",
                "__array_interface__",
                "update_copy",
                "copy",
                "view",
                "T",
                "shape",
                "dtype",
                "ndim",
                "base",
                "size",
                "min",
                "max",
                "__len__",
                "__getitem__",
                "__setitem__",
            ):
                assert hasattr(self.cls, name)

    @abstractmethod
    def test_initialisation(self): ...

    @abstractmethod
    def test_properties(self): ...

    @abstractmethod
    def test_methods(self): ...

    @abstractmethod
    def test_numpy_funcs(self): ...

    # Goal is to test that the object is treated like a numpy object

    @abstractmethod
    def test_other(self): ...

    @abstractmethod
    def test_view(self): ...

    @abstractmethod
    def test_np_operator_mixins(self):
        a = self.cls(arr=np.random.rand(100, 100))

        with pytest.raises(TypeError):
            a += 1
        with pytest.raises(TypeError):
            a = a - 1.0
        with pytest.raises(TypeError):
            a = a + 2.0
        with pytest.raises(TypeError):
            a = a / 0.5
        with pytest.raises(TypeError):
            a = a * 3

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
        b = 3 == a
        assert b.dtype == np.bool_
        b = 3 != a
        assert b.dtype == np.bool_


class TestBaseArray(BaseTests):
    cls = BaseArray

    def test_other(self):
        pass

    def test_numpy_funcs(self):
        a = self.cls(arr=np.ones((5, 3)))
        b = np.add(a, 1)
        assert np.all(b == 2)

        c = np.mean(a, axis=0)
        assert np.all(c == 1.0)
        assert c.shape == (3,)
        assert isinstance(b, np.ndarray)
        assert np.allclose(a, 1)

    def test_initialisation(self):
        data = np.random.rand(100, 100)
        a = self.cls(arr=data)

        assert isinstance(a, self.cls)
        assert a is not data
        assert a.arr is not data
        assert np.all(a == a)

    def test_view(self):
        a = self.cls(arr=np.random.rand(100, 100))

        with pytest.raises(NotImplementedError):
            a.view()

    @pytest.mark.parametrize(
        "input_values", ({"dict": "Should fail"}, "String is bad", {1, 2, 3}, True, 1.4, None, [1, 2, 3], (3, 4, 5))
    )
    def test_incorrect_types(self, input_values):

        with pytest.raises(Exception) as e:
            self.cls(arr=input_values)  # type: ignore

        assert type(e.value) in (ValueError, TypeError, ValidationError)

    def test_properties(self):
        a = self.cls(arr=np.ones((100, 3)))
        b = np.zeros_like(a).copy()
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert b.ndim == 2
        assert a.shape == (100, 3)
        assert np.all(b == 0)
        assert a.size == 300

        b = np.reshape(a, (3, 100))  # a view
        assert a.shape == (100, 3)
        assert b.base is not None
        assert isinstance(a, self.cls)
        assert isinstance(b, np.ndarray)

        # Test the translation
        assert b.shape == a.T.shape
        assert isinstance(a, self.cls)

        b = np.reshape(a, (5, 60))
        assert a.shape != (5, 60)
        assert b.shape == (5, 60)

        assert np.sum(b) == 300 == np.sum(a)

    def test_methods(self):
        a = self.cls(arr=np.random.rand(10, 3))
        assert np.all(a == a.arr)
        assert a.min() >= 0
        assert a.max() <= 1

    def test_np_operator_mixins(self):
        super().test_np_operator_mixins()

    class TestSubclassMethods:
        cls = BaseArray

        def test_frozen(self):
            class _FrozenArray(BaseArray):
                model_config = ConfigDict(frozen=True)

            a = _FrozenArray(arr=np.random.rand(100, 100))

            with pytest.raises(ValidationError):
                a.arr = np.zeros(3)
            with pytest.raises(ValueError):
                a.arr[0] = 3

            # Ensure that the dict is not being
            for field in base_config_fields:
                assert field in a.model_config

        def test_array_interface(self):
            a = np.random.rand(100, 100)
            array = self.cls(arr=a)
            assert array.__array_interface__ == array.arr.__array_interface__
            # Every object is a "copy" to avoid any chance of a reference
            assert array.__array_interface__ != a.__array_interface__

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
            a = self.cls(arr=np.random.rand(100, 100))
            # Test getitem
            assert np.all(a[:, 0] == a.arr[:, 0])

            b = a.copy()

            # Test setitem
            a[0, 0] = 13
            assert a[0, 0] == 13
            assert b[0, 0] != 13

            b[0:3, :] = a[0:3, :]
            assert b[0, 0] == 13
            assert np.allclose(b[0:3, :], a[0:3, :])

        def test_array_assignment(self):
            a = np.random.rand(100, 100)
            array = self.cls(arr=a)
            assert a.shape == (100, 100)

            # Test set
            b = np.random.rand(10, 10).astype(np.float32)
            array.arr = b

            # Check the array has been updated
            assert array.shape == (10, 10)
            assert array.dtype == np.float32
            assert array.ndim == 2
            assert array.base is None
            assert array.size == 10**array.ndim

            assert np.all(array.arr == b)

        def test_copy(self):
            a = self.cls(arr=np.random.rand(100, 100))
            b = a.copy()

            # Check that the object copied has no references to the original
            assert a is not b
            assert a.arr is not b.arr
            assert np.all(np.isclose(b, a))

            with pytest.raises(NotImplementedError):
                a.copy(deep=False)

        def test_update_copy(self):
            a = np.random.rand(100, 100)
            b = np.random.rand(10, 10)
            c = np.random.rand(5, 3)

            array = self.cls(arr=a)
            array_2 = array.update_copy(b)
            array_3 = array.update_copy(update={"arr": c})

            with pytest.raises(TypeError):
                array.update_copy(update={"arr": "asdasd"})

            # Show the original object hasn't changed
            assert array.shape == a.shape
            assert np.allclose(array.arr, a)

            # Show the underlying data has been changed with the new copy
            assert array_2.shape != array.shape
            assert array_3.shape != array_2.shape
            assert array_3.shape != array.shape

            # Updated copies are new objects
            assert array is not array_2
            assert array is not array_3
            assert array_2 is not array_3

            assert array is not a
            assert array_2 is not b
            assert array_3 is not c

            # Show that the array data is also a copy and not a reference
            assert array.arr is not a
            assert array_2.arr is not b
            assert array_3.arr is not c

        def test_min_max(self):
            a = np.random.rand(100, 100)
            array = self.cls(arr=a.copy())

            assert np.min(a) == array.min()
            assert np.max(a) == array.max()


class TestSamplingArray(BaseTests):
    cls = SampleArray

    def test_initialisation(self):
        a = self.cls(arr=np.random.rand(100, 4))

        assert isinstance(a.arr, np.ndarray)
        assert isinstance(a, self.cls)
        assert a.shape == (100, 4)

    def test_methods(self):
        for name in ("create_mask", "sample", "reduce", "extract"):
            hasattr(self.cls, name)

    def test_slice_extract_reduce_1d(self):

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

    def test_slice_extract_reduce_2d(self):
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

    def test_numpy_funcs(self):
        a = self.cls(arr=np.random.rand(100, 4))
        assert np.mean(a) is not None

    def test_properties(self):
        assert self.cls(arr=np.random.rand(100, 4)).ndim == 2

    def test_np_operator_mixins(self):
        a = self.cls(arr=np.random.rand(100, 4))
        with pytest.raises(TypeError):
            a += 1

    def test_view(self):
        a = self.cls(arr=np.random.rand(100, 4))
        assert a.base is None

    def test_other(self):
        pass


class TestFixedLengthAndMixins(BaseTests):
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


class TestHomogeneousAndMixins(BaseTests):
    cls = HomoegeneousArray

    def test_initialisation(self):
        a = HomoegeneousArray(arr=np.zeros((10, 3)))
        assert isinstance(a, HomoegeneousArray)
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


class TestReadOnlyArray:
    def test_initialisation(self):
        array = ReadOnlyArray(arr=np.random.rand(10, 2))
        with pytest.raises(ValidationError):
            array.arr = np.random.randn(10, 2)


class TestReadOnlyVector:
    def test_initialisation(self):
        array = ReadOnlyVector(arr=np.random.rand(10))

        with pytest.raises(ValidationError):
            array.arr = np.random.randn(10)


class TestImageLike(BaseTests):
    def test_initialisation(self): ...

    def test_properties(self): ...

    def test_methods(self): ...

    def test_numpy_funcs(self): ...

    def test_other(self): ...

    def test_view(self): ...

    def test_np_operator_mixins(self): ...
