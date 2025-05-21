import pytest

import numpy as np

from pchandler.base_descriptors import ArrayDescriptor, Descriptor
from src.pchandler.base_arrays import (ValidatedArray, VectorN, ArrayNx3, ArrayNx2, Array2d, NpMixinT,
                                       Vector2, Vector3, TransformArray4x4)


class ReadOnlyArray(ValidatedArray):
    _arr: np.ndarray = ArrayDescriptor(freezable=True, coerce=True)



class TempAbc:
    abc: ArrayNx3 = Descriptor(ArrayNx3, default=np.ones((5, 3)), coerce=True)

    def __init__(self, abc=None):
        self.abc = abc

class TestDataArray:
    def test_mutability(self):
        a = np.random.rand(5,3)
        b = np.random.rand(5,3)
        a_array = ValidatedArray(a)
        original_id = id(a_array)
        np.testing.assert_array_almost_equal(a_array, a)
        a_array._arr = b
        assert np.all(a_array == b)                 # Check the new values are assigned
        assert np.all(a_array != a)                 # Check that the values have changed
        assert original_id == id(a_array)           # Check that it's still the same object
        assert isinstance(a_array, NpMixinT)        # Check it's still one of these classes

    def test_with_descriptor(self):
        a = TempAbc(np.array([[1, 2, 3], [2, 3, 4]]))
        assert isinstance(a.abc, ArrayNx3)
        assert np.all(a.abc == np.array([[1, 2, 3], [2, 3, 4]]))

        b = TempAbc()
        assert isinstance(b.abc, ArrayNx3)
        assert np.all(b.abc == np.ones((5, 3)))

        with pytest.raises(ValueError):
            TempAbc(np.random.rand(10,10,10))

    def test_immutability(self) -> None:
        test_data = ValidatedArray(np.random.rand(5,3))
        array2 = ReadOnlyArray(np.random.randn(5,3))
        with pytest.raises( AttributeError ):
            array2._arr = test_data._arr

        test_data._arr[0, 0] = 3
        # This blocks the user from assigning directly to the array as well
        with pytest.raises( ValueError ):
            array2._arr[0, 0] = 3

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
        assert id(temp2._arr) == id(test_data._arr)
        assert id(temp2_deep._arr) != id(test_data._arr)
        test_data[0, 0] = 111
        assert np.all(temp == test_data)
        assert np.all(temp2 == test_data) # change in test_data will be observed in the view of temp2

        assert np.all(temp == test_data)


    def test_copying(self):
        test_data = ValidatedArray(np.random.rand(10,3))
        a = test_data       # id(a) == id(test_data) and id(a.arr) == id(test_data.arr)
        assert id(a) == id(test_data)
        assert id(a._arr) == id(test_data._arr)
        b = test_data.copy() # array is still a view, but different id
        assert id(b) != id(test_data)
        assert id(b._arr) == id(test_data._arr)
        c = test_data.copy(deep=True)
        assert id(c) != id(test_data)
        assert id(c._arr) != id(test_data._arr)
        a[0, 0] = 100
        np.testing.assert_array_equal(a, b)
        assert a[0, 0] != c[0, 0]
        assert np.all(a[1, :] == c[1, :])



class TestDataArray1D:
    def test_initialisation(self):
        a: VectorN = VectorN(np.ones((5,)))
        assert a.ndim == 1

        b: VectorN = VectorN(np.ones((5, 1)))
        assert b.ndim == 1

        c: VectorN = VectorN(np.ones((1, 5)))
        assert c.ndim == 1

        # with pytest.raises( ValueError ):
        assert np.array(42).ndim == 0
        with pytest.raises(ValueError):
            d: VectorN = VectorN(np.array(42))

        with pytest.raises( ValueError ):
            VectorN(np.random.randn(5, 3, 3))

        assert np.all(VectorN(np.array([[1, 2]])) == np.array([[1, 2]]))


class TestDataArray2D:
    def test_initialisation(self):
        a: Array2d = Array2d(np.ones((5, 3)))
        assert a.ndim == 2

        with pytest.raises( TypeError ):
            Array2d('bacs')

class TestDataArrayNx2:
    def test_initialisation(self):
        a: ArrayNx2 = ArrayNx2(np.ones((5, 2)))
        assert a.ndim == 2
        assert a.shape[1] == 2
        assert len(a) == 5

    def test_invalid_values(self):
        for val in (np.random.rand(5, 3), np.random.rand(10, 2, 4), np.random.rand(5, 1)):
            with pytest.raises(ValueError):
                ArrayNx2(val)

class TestDefaults:
    def test_vector_2d(self):
        a = Vector2()
        assert np.all(a == 0)

    def test_vector_3d(self):
        b = Vector3()
        assert np.all(b == 0)

    def test_4x4_matrix(self):
        c = TransformArray4x4()
        assert np.all(c == np.eye(4))



