import pytest
import numpy as np

from pchandler.base_classes import (
    DataArray, DataArray2D, DataArray3D, DataArrayNx2, DataArrayNx3, DataArrayMxNx3, DataArray4D, DataArray1D)


class TestDataArray:
    def test_mutability(self):
        array = np.random.rand(5,3)
        array2 = np.random.rand(5,3)
        test_data = DataArray(array)
        test_id = id(test_data)
        np.testing.assert_array_almost_equal(test_data, array)
        test_data.arr = array2
        assert np.all(test_data == array2)
        assert np.all(test_data != array)
        assert test_id == id(test_data)


    def test_immutability(self) -> None:
        test_data = DataArray(np.random.rand(5,3))
        array2 = DataArray(np.random.randn(5,3), immutable=True)
        with pytest.raises( AttributeError ):
            array2.arr = test_data.arr

        test_data.arr[0, 0] = 3
        # This blocks the user from assigning directly to the array as well
        with pytest.raises( ValueError ):
            array2.arr[0, 0] = 3

    def test_set_immutable(self):
        array1 = DataArray(np.random.rand(5,3))
        array2 = DataArray(np.random.rand(5,3), immutable=True)
        with pytest.raises( AttributeError ):
            array2.arr = array1.arr

        array2.set_immutability(False)
        array2.arr = array1.arr
        np.testing.assert_array_equal(array1, array2)

        array2.set_immutability(True)
        array3 = DataArray(np.random.rand(5,3), immutable=True)
        with pytest.raises( AttributeError ):
            array2.arr = array3.arr

        with pytest.raises( AttributeError ):
            array2.arr = array1.arr

        with pytest.raises( AttributeError ):
            array2.arr = array1   # type: ignore

    def test_numpy_functionality(self):
        a: DataArray = DataArray(np.ones((5,3)))
        b: DataArray = a + np.ones(3) * 3
        assert np.all(a != b)
        c: DataArray = b - np.ones(3) * 3
        np.testing.assert_array_almost_equal(a, c)
        d: np.ndarray|DataArray = a != b
        assert isinstance(d, np.ndarray)
        assert np.issubdtype(d.dtype, np.bool_)
        assert np.mean(b) == 4

    def test_helper_properties(self):
        a: DataArray = DataArray(np.ones((5,3)))
        assert a.ndim == 2
        assert a.shape == (5,3)
        assert a.dtype == np.float64
        assert a.size == np.prod(a.shape)
        assert a.base is None

    @pytest.mark.parametrize("values, error_type", [
        (('str', {'23': 1}, {1, 3, 5}, False, None,
          np.empty(3, dtype=np.dtype('c16')), 1j * np.arange(5)), TypeError),
        ((np.array([]), np.empty(0, dtype=np.dtype('f4'))), ValueError)])
    def test_validation_func_invalid_values(self, values, error_type):
        for val in values:
            with pytest.raises(error_type):
                DataArray(val)

    @pytest.mark.parametrize("values", [
        np.random.randn(5).astype('f8'),
        np.random.randn(1000,10000).astype('f4'),
        np.random.randn(1000, 20, 3),
        np.random.randint(0, 100, (30, 3), dtype=np.int64),
        np.random.randint(0, 100, (30, 3), dtype=np.int32),
        np.random.randint(0, 254, (30, 3), dtype=np.uint8)
    ])
    def test_validation_func_valid_values(self, values):
        assert isinstance(DataArray(values), DataArray)

    def test_get_set_item(self):
        test_data = DataArray(np.random.rand(10,3))
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

        test_data.set_immutability(True)
        # Test's creates a deep copy
        temp3: np.ndarray = test_data[:]  # should create a copy
        assert id(temp3) != id(test_data.arr)
        assert temp3.base is None
        assert np.all(temp3 == test_data)

        test_data.set_immutability(False)

        test_data[0, :] = np.ones(3)
        assert np.all(temp == test_data)
        assert np.all(temp3[0, :] != test_data[0, :])
        test_data[0, :] = temp3[0, :]

        assert np.all(temp == test_data)
        assert np.all(temp3 == test_data)

        # Because temp3 is a deep copy, an assignment shouldn't change data_array_test
        temp3[0] = 3
        assert temp3[0,0] != test_data[0, 0]
        temp[0] = 5 #but as this is a view created with writing capabilities...
        assert temp[0, 0] == test_data[0, 0]
        assert temp3[0, 0] != test_data[0, 0]

    def test_copying(self):
        test_data = DataArray(np.random.rand(10,3))
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

        test_data.set_immutability(True)    # copy should become deep copy to avoid views assigning values
        d = test_data.copy(deep=False)
        assert id(d) != id(test_data)
        assert id(d.arr) != id(test_data.arr)


class TestDataArray1D:
    def test_initialisation(self):
        a: DataArray1D = DataArray1D(np.ones((5,)))
        assert a.ndim == 1

        b: DataArray1D = DataArray1D(np.ones((5,1)))
        assert b.ndim == 1


        c: DataArray1D = DataArray1D(np.ones((1, 5)))
        assert c.ndim == 1

        assert np.array(42).ndim == 0
        d: DataArray1D = DataArray1D(np.array(42))
        assert d.ndim == 1

        with pytest.raises( ValueError ):
            DataArray1D(np.random.randn(5, 3, 3))

        DataArray1D._num_rows = 3
        e: DataArray1D = DataArray1D(np.array([1, 2, 3]))

        with pytest.raises( ValueError ):
            DataArray1D(np.array([1, 2]))


class TestDataArray2D:
    def test_initialisation(self):
        a: DataArray2D = DataArray2D(np.ones((5,3)))
        assert a.ndim == 2

        with pytest.raises( ValueError ):
            DataArray2D(np.random.randn(5,3, 3))

class TestDataArrayNx2:
    def test_initialisation(self):
        a: DataArrayNx2 = DataArrayNx2(np.ones((5,2)))
        assert a.ndim == 2
        assert a.shape[1] == 2
        assert len(a) == 5

    def test_invalid_values(self):
        for val in (np.random.rand(5, 3), np.random.rand(10, 2, 4), np.random.rand(5, 1)):
            with pytest.raises(ValueError):
                DataArrayNx2(val)

class TestDataArrayNx3:
    def test_initialisation(self):
        a: DataArrayNx3 = DataArrayNx3(np.ones((5,3)))
        assert a.ndim == 2
        assert a.shape[1] == 3

    def test_invalid_values(self):
        for val in (np.random.rand(5, 2), np.random.rand(10, 2, 4), np.random.rand(5, 1)):
            with pytest.raises(ValueError):
                DataArrayNx3(val)

class TestDataArray3D:
    def test_initialisation(self):
        a: DataArray3D = DataArray3D(np.ones((5,3,8)))
        assert a.ndim == 3

    def test_invalid_values(self):
        for val in (np.random.rand(5, 2), np.random.rand(10, 2, 4, 4), np.random.rand(5)):
            with pytest.raises(ValueError):
                DataArray3D(val)

class TestDataArrayMxNx3:
    def test_initialisation(self):
        a: DataArrayMxNx3 = DataArrayMxNx3(np.ones((5,3,3)))
        assert a.ndim == 3

    def test_invalid_values(self):
        for val in (np.random.rand(5, 2, 2), np.random.rand(10, 2, 4), np.random.rand(5, 1, 1)):
            with pytest.raises(ValueError):
                DataArrayMxNx3(val)


class TestDataArray4D:
    def test_initialisation(self):
        a: DataArray4D = DataArray4D(np.ones((5,3,8, 4)))
        assert a.ndim == 4

    def test_invalid_values(self):
        for val in (np.random.rand(5, 2), np.random.rand(10, 2, 4), np.random.rand(5), np.random.rand(5,5,5,5,5)):
            with pytest.raises(ValueError):
                DataArray4D(val)