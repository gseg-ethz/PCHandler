import pytest
import sys

import numpy as np

from pchandler.v2.geometry.coordinates import (
    CartesianCoordinates, Abstract3dCoordinates, SphericalCoordinates, ArrayNx3, rhv2xyz, xyz2rhv
)

PI = np.pi
PI_2 = np.pi * 2
HALF_PI = np.pi * 0.5

@pytest.fixture(scope='function')
def known_spher():
    return np.array([
        [1, 0, HALF_PI],
        [1, HALF_PI, HALF_PI],
        [1, 0, 0],
        [1, PI, HALF_PI],
        [1, -HALF_PI, HALF_PI],
        [1, 0, PI],
        [np.sqrt(3), np.arctan2(1, 1), np.arctan2(np.sqrt(2), 1)]
    ]).astype(np.float64)

@pytest.fixture(scope='function')
def small_xyz():
    return np.random.rand(100, 3)

@pytest.fixture(scope='function')
def known_xyz():
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [1, 1, 1]
    ]).astype(np.float64)

@pytest.fixture(scope='function')
def large_xyz():
    return np.random.rand(1_000_000, 3)

@pytest.fixture(scope='function')
def cart_obj(large_xyz):
    return CartesianCoordinates(large_xyz)

class TestConversions:
    def test_cartesian_to_spherical(self, known_xyz, known_spher):
        rhv = xyz2rhv(known_xyz)
        assert np.allclose(rhv, known_spher)
        xyz = rhv2xyz(known_spher)
        assert np.allclose(xyz, known_xyz)

    def test_forward_backward(self, small_xyz, large_xyz):
        for arr in (small_xyz, large_xyz):
            xyz2 = rhv2xyz(xyz2rhv(arr))
            assert np.allclose(xyz2, arr)


class TestCartesianCoordinates:
    def test_small_array(self, small_xyz, large_xyz, known_xyz):
        for array in (small_xyz, large_xyz):
            a = CartesianCoordinates(array)
            assert isinstance(a, CartesianCoordinates)
            assert isinstance(a, Abstract3dCoordinates)
            assert isinstance(a, ArrayNx3)

    def test_cached_properties(self, cart_obj):
        assert 'spher' not in cart_obj.__dict__
        num_items_before = len(cart_obj.__dict__)

        _ = cart_obj.r
        num_items_after = len(cart_obj.__dict__)

        assert 'spher' in cart_obj.__dict__
        assert num_items_after > num_items_before

        del cart_obj.spher

        num_items_after = len(cart_obj.__dict__)

        assert num_items_after == num_items_before


    def test_has_properties(self, cart_obj):
        assert hasattr(cart_obj, 'xyz')
        assert hasattr(cart_obj, 'spher')
        assert hasattr(cart_obj, 'x')
        assert hasattr(cart_obj, 'y')
        assert hasattr(cart_obj, 'z')
        assert hasattr(cart_obj, 'yxz')
        assert hasattr(cart_obj, 'rhv')
        assert hasattr(cart_obj, 'r')
        assert hasattr(cart_obj, 'hz')
        assert hasattr(cart_obj, 'v')
        assert hasattr(cart_obj, 'to_spherical')
        assert hasattr(cart_obj, 'from_spherical')

    def test_cartesian_properties(self, cart_obj):
        assert np.all(cart_obj.x == cart_obj[:, 0])
        assert np.all(cart_obj.y == cart_obj[:, 1])
        assert np.all(cart_obj.z == cart_obj[:, 2])
        assert np.all(cart_obj.x == cart_obj._arr[:, 0])
        assert np.all(cart_obj.y == cart_obj._arr[:, 1])
        assert np.all(cart_obj.z == cart_obj._arr[:, 2])

        assert id(cart_obj._arr) == id(cart_obj.xyz)

    def test_spherical_properties(self, cart_obj):
        assert np.all(cart_obj.r == cart_obj.spher[:, 0])
        assert np.all(cart_obj.hz == cart_obj.spher[:, 1])
        assert np.all(cart_obj.v == cart_obj.spher[:, 2])
        assert id(cart_obj.rhv) == id(cart_obj.spher)

    def test_conversion_functions(self, cart_obj):
        spherical = cart_obj.to_spherical()
        assert isinstance(spherical, SphericalCoordinates)

        assert 'spher' not in cart_obj.__dict__
        # Check the id but also generate the cached property
        assert id(spherical) != id(cart_obj.spher)

        spherical2 = cart_obj.to_spherical()

        assert id(spherical2) != id(spherical) != id(cart_obj.spher)
        assert np.allclose(spherical, spherical2)
        assert np.allclose(cart_obj.spher, spherical2)
        assert np.allclose(spherical, cart_obj.spher)






