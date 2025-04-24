import pytest
from typing import Callable
from abc import ABC

import numpy as np

from pchandler.geometry.core_experimental import (
    check_hz_angles, check_zenith_angles, check_azimuth_angles, check_radial_distances, check_inclination_angles,
    check_spherical_coordinates, DataArray, Coordinates3D
)

PI = np.pi
TWO_PI = 2 * PI
HALF_PI = PI / 2


class BaseAngleTestClass(ABC):
    main_test: Callable = None

    @pytest.mark.parametrize("values, func", [
        ("str", main_test),
        ({"angle": 74}, main_test),
        ({1.3, 0.2, -1.3}, main_test),
    ])
    def test_invalid_types(self, values: np.ndarray, func: Callable):
        with pytest.raises(TypeError):
            func(values)


class TestHzAngleValidation(BaseAngleTestClass):
    main_test: Callable = check_hz_angles

    @pytest.mark.parametrize("values, func", [
        (np.array([0, np.pi, 1.7, -np.pi]), main_test),
        (float(1.3), main_test),
        (int(1), main_test),
        ([1.3, 0.2, -1.3], main_test),
        ((1.3, 0.2, -1.3), main_test),
    ])
    def test_valid_values(self, values: np.ndarray, func: Callable):
        assert func(values) is None

    @pytest.mark.parametrize("values, func", [
        (np.array([0, 2*np.pi, 1.7, -np.pi-3]), main_test),
        (float(72), main_test),
        (int(-44), main_test),
    ])
    def test_invalid_values(self, values: np.ndarray, func: Callable):
        with pytest.raises( ValueError ):
            func(values)


class TestAzimuthAngleValidation(BaseAngleTestClass):
    main_test: Callable = check_azimuth_angles

    @pytest.mark.parametrize("values, func", [
        (np.array([0, np.pi, 1.7, 2*np.pi]), main_test),
        (float(5.3), main_test),
        (int(4), main_test),
        ([1.3, 0.2, 3.5], main_test),
        ((1.3, 0.2, 4.3), main_test),
    ])
    def test_valid_values(self, values: np.ndarray, func: Callable):
        assert func(values) is None

    @pytest.mark.parametrize("values, func", [
        (np.array([0, 3*np.pi, 1.7, -np.pi-3]), main_test),
        (float(72), main_test),
        (int(-44), main_test),
    ])
    def test_invalid_values(self, values: np.ndarray, func: Callable):
        with pytest.raises( ValueError ):
            func(values)


class TestZenithAngles(BaseAngleTestClass):
    main_test: Callable = check_zenith_angles

    @pytest.mark.parametrize("values, func", [
        (np.array([0, np.pi, 1.7, np.pi]), main_test),
        (float(2.3), main_test),
        (int(2), main_test),
        ([1.3, 0.2, np.pi], main_test),
        ((1.3, 0.2, np.pi), main_test),
    ])
    def test_valid_values(self, values: np.ndarray, func: Callable):
        assert func(values) is None

    @pytest.mark.parametrize("values, func", [
        (np.array([0, 3*np.pi, 1.7, -np.pi-3]), main_test),
        (float(72), main_test),
        (int(-44), main_test),
    ])
    def test_invalid_values(self, values: np.ndarray, func: Callable):
        with pytest.raises( ValueError ):
            func(values)


class TestInclinationAngles(BaseAngleTestClass):
    main_test: Callable = check_inclination_angles

    @pytest.mark.parametrize("values, func", [
        (np.array([0, 1.3, -np.pi/2, np.pi/2]), main_test),
        (float(1.3), main_test),
        (int(-1), main_test),
        ([1.3, 0.2, -np.pi/2], main_test),
        ((-1.3, 0.2, np.pi/2), main_test),
    ])
    def test_valid_values(self, values: np.ndarray, func: Callable):
        assert func(values) is None

    @pytest.mark.parametrize("values, func", [
        (np.array([0, np.pi, 1.7, -np.pi]), main_test),
        (float(72), main_test),
        (int(-44), main_test),
    ])
    def test_invalid_values(self, values: np.ndarray, func: Callable):
        with pytest.raises( ValueError ):
            func(values)


class TestRadiusDistance(BaseAngleTestClass):
    main_test: Callable = check_radial_distances

    @pytest.mark.parametrize("values, func", [
        (np.array([0, 1.3, 200, 300000.21323]), main_test),
        (float(156.32), main_test),
        (int(126), main_test),
        ([1.3, 2000, 23445.123], main_test),
        ((1.3, 2000, 23445.123), main_test),
    ])
    def test_valid_values(self, values: np.ndarray, func: Callable):
        assert func(values) is None

    @pytest.mark.parametrize("values, func", [
        (np.array([-1.3, -2, np.inf, -np.pi]), main_test),
        (float(-72.3), main_test),
        (int(-44), main_test),
    ])
    def test_invalid_values(self, values: np.ndarray, func: Callable):
        with pytest.raises( ValueError ):
            func(values)


class TestSphericalCoordinates(BaseAngleTestClass):
    main_test: Callable = check_spherical_coordinates

    @pytest.mark.parametrize("values, func", [
        (np.array([
            [np.random.rand(10)*100,
            np.random.rand(10)*PI,
            np.random.rand(10)*TWO_PI - PI
        ]]), main_test)
    ])
    def test_valid_values(self, values: np.ndarray, func: Callable):
        assert func(values) is None


    @pytest.mark.parametrize("values, func", [
        (np.array([
            np.random.rand(10)*100-50,
            np.random.rand(10)*PI,
            np.random.rand(10)*TWO_PI - PI,
        ]), main_test),
        (np.array([
            np.random.rand(10) * 100,
            np.random.rand(10) * TWO_PI,
            np.random.rand(10) * TWO_PI - PI,
        ]), main_test),
        (np.array([
            np.random.rand(10) * 100,
            np.random.rand(10) * PI,
            np.random.rand(10) * TWO_PI,
        ]), main_test),
        (np.array([
            np.random.rand(10) * 100,
            np.random.rand(10) * PI,
            np.random.rand(10) * -TWO_PI,
        ]), main_test)
    ])
    def test_invalid_values(self, values: np.ndarray, func: Callable):
        with pytest.raises( ValueError ):
            func(values)


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

    @pytest.mark.parametrize("values, error_type", [
        (('str', {'23': 1}, {1, 3, 5}, False, None,
          np.empty(3, dtype=np.dtype('c16')), 1j * np.arange(5)), TypeError),
        ((np.array([]), np.random.rand(3,3,3,3)), ValueError)])
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
        assert id(temp2) != id(test_data)
        assert np.all(temp2 == test_data)
        assert temp2.base is None
        test_data.set_immutability(True)
        # Test's creates a deep copy
        temp3 = test_data  # should create a copy
        assert temp3.base is None
        assert np.all(temp3 == test_data)

        test_data.set_immutability(False)

        test_data[0, :] = np.ones(3)
        assert np.all(temp == test_data)
        assert np.all(temp3 != test_data)
        test_data[0, :] = temp3.arr

        assert np.all(temp == test_data)
        assert np.all(temp3 == test_data)

        # Because temp3 is a deep copy, assignment shouldn't change data_array_test
        temp3[0] =3
        assert temp3[0] != test_data[0, 0]
        temp[0] = 5 #but as this is a view created with write capabilities...
        assert temp[0] == test_data[0, 0]
        assert temp3[0] == test_data[0, 0]

    def test_copying(self):
        test_data = DataArray(np.random.rand(10,3))
        a = test_data       # id(a) == id(test_data)
        assert id(a) == id(test_data)
        b = test_data.copy() # array is still a view, but different id
        assert id(b) != id(test_data)
        c = test_data.copy(deep=True)
        a[0, 0] = 100
        np.testing.assert_array_equal(a, b)
        assert a[0, 0] != c[0, 0]
        assert np.all(a[1, :] == c[1, :])



COORDINATE_3D_PROPERTIES = ('x', 'y', 'z', 'r', 'v', 'hz', 'rho', 'theta', 'phi', 'xyz', 'spher')


class TestCoordinates3D:
    arr: Coordinates3D = Coordinates3D(np.random.randn(100, 3))

    def test_num_pts(self):
        assert len(self.arr) == 100
        assert self.arr.num_points == 100

    @pytest.mark.parametrize("attr", list(COORDINATE_3D_PROPERTIES))
    def test_not_implemented_properties(self, attr):
        with pytest.raises( NotImplementedError ):
            getattr(self.arr, attr)

    @pytest.mark.parametrize("attr", ['to_spherical', 'to_cartesian'])
    def test_not_implemented_methods(self, attr):
        with pytest.raises( NotImplementedError ):
            getattr(self.arr, attr)()

    @pytest.mark.parametrize("attr", ['from_spherical', 'from_cartesian'])
    def test_not_implemented_class_methods(self, attr):
        with pytest.raises( NotImplementedError ):
            getattr(self.arr, attr)(self.arr.arr)


class TestCartesianCoordinates:
    pass

class TestBasePointCloud:
    pass

class TestSphericalPointCloud:
    pass

class TestTlsPointCloud:
    pass

class TestMultiScanCloud:
    pass
