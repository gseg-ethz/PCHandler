import pytest
from typing import Callable
from abc import ABC

import numpy as np

from pchandler.geometry.validation import (
    check_hz_angles, check_zenith_angles, check_azimuth_angles, check_radial_distances, check_inclination_angles,
    check_spherical_coordinates)


PI = np.pi
TWO_PI = 2 * PI
HALF_PI = PI / 2

COORDINATE_3D_PROPERTIES = ('x', 'y', 'z', 'r', 'v', 'hz', 'rho', 'theta', 'phi', 'xyz', 'spher')


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


class TestValidation(BaseAngleTestClass):
    class TestHzAngle(BaseAngleTestClass):
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


    class TestAzimuthAngle(BaseAngleTestClass):
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
