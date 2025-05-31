import copy

import pytest
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from pchandler.v2.base_arrays import ArrayNx3
from pchandler.v2.geometry.coordinates import (
    CartesianCoordinates, Abstract3dCoordinates, SphericalCoordinates, rhv2xyz, xyz2rhv, AbstractCoordinates
)
from pchandler.v2.geometry.transforms import Transform

PI = np.pi
PI_2 = np.pi * 2
HALF_PI = np.pi * 0.5

_known_spher = np.array([
    [1, 0, HALF_PI],
    [1, HALF_PI, HALF_PI],
    [1, 0, 0],
    [1, PI, HALF_PI],
    [1, -HALF_PI, HALF_PI],
    [1, 0, PI],
    [np.sqrt(3), np.arctan2(1, 1), np.arctan2(np.sqrt(2), 1)]
]).astype(np.float64)

@pytest.fixture(scope='function')
def known_spher():
    return _known_spher

_small_xyz = np.random.rand(100, 3)

@pytest.fixture(scope='function')
def small_xyz():
    return _small_xyz

_known_xyz = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [1, 1, 1]
]).astype(np.float64)

@pytest.fixture(scope='function')
def known_xyz():
    return _known_xyz

_large_xyz = np.random.rand(1_000_000, 3)

@pytest.fixture(scope='function')
def large_xyz():
    return _large_xyz

@pytest.fixture(scope='function')
def cart_obj(large_xyz) -> CartesianCoordinates:
    return CartesianCoordinates(arr=large_xyz)


class TestCartesianCoordinates:
    @pytest.mark.parametrize('array', (_small_xyz, _large_xyz, _known_xyz))
    def test_instantiation(self, array):
        a = CartesianCoordinates(arr=array)
        assert isinstance(a, CartesianCoordinates)
        assert isinstance(a, Abstract3dCoordinates)
        assert isinstance(a, AbstractCoordinates)
        assert isinstance(a, ArrayNx3)

    def test_cached_properties(self, cart_obj):
        assert 'spher' not in cart_obj.__dict__
        num_items_before = len(cart_obj.__dict__)

        _ = cart_obj.r
        num_items_after = len(cart_obj.__dict__)

        assert 'spher' in cart_obj.__dict__
        assert num_items_after > num_items_before

        del cart_obj.__dict__['spher']

        num_items_after = len(cart_obj.__dict__)

        assert num_items_after == num_items_before

    @pytest.mark.parametrize('attr', ('arr', 'socs_origin', 'transform_ledger', 'is_at_socs'))
    def test_has_attributes(self, cart_obj, attr):
        assert attr in cart_obj.__dict__.keys()
        if attr == 'socs_origin':
            assert np.allclose(cart_obj.socs_origin, np.zeros(3))

    @pytest.mark.parametrize('prop', ('xyz' ,'spher', 'x', 'y', 'z', 'yxz', 'rhv', 'r', 'hz', 'v'))
    def test_has_properties(self, cart_obj, prop):
        # These properties are access methods and shouldn't be in the base dict
        assert prop in type(cart_obj).__dict__ and prop not in cart_obj.__dict__

    @pytest.mark.parametrize('method', ('to_spherical', 'from_spherical', 'transform'))
    def test_has_methods(self, cart_obj, method):
        assert callable(getattr(cart_obj, method))

    def test_cartesian_properties(self, cart_obj):
        # Check xyz and arr are the same object
        assert id(cart_obj.arr) == id(cart_obj.xyz)
        assert np.all(cart_obj == cart_obj.arr)
        with pytest.raises(IndexError):
            assert np.all(cart_obj.x == cart_obj[:, 0])
        assert np.all(cart_obj.x == cart_obj.arr[:, 0])
        assert np.all(cart_obj.y == cart_obj.arr[:, 1])
        assert np.all(cart_obj.z == cart_obj.arr[:, 2])

        assert np.any(cart_obj.yxz != cart_obj.arr)
        assert np.all(cart_obj.yxz[:, 0] == cart_obj.arr[:, 1])
        assert np.all(cart_obj.yxz[:, 1] == cart_obj.arr[:, 0])
        assert np.all(cart_obj.yxz[:, 2] == cart_obj.arr[:, 2])

    def test_spherical_properties(self, cart_obj):
        assert id(cart_obj.arr) != id(cart_obj.spher)
        assert id(cart_obj.rhv) == id(cart_obj.spher)
        assert np.all(cart_obj.r == cart_obj.spher[:, 0])
        assert np.all(cart_obj.hz == cart_obj.spher[:, 1])
        assert np.all(cart_obj.v == cart_obj.spher[:, 2])

    def test_conversion_functions(self, cart_obj):
        spherical = cart_obj.to_spherical()
        assert isinstance(spherical, SphericalCoordinates)
        assert np.all(spherical.socs_origin == cart_obj.socs_origin)
        assert spherical.transform_ledger == cart_obj.transform_ledger
        assert spherical.is_at_socs == cart_obj.is_at_socs

        # check that the cart_obj cleans up the cached spherical coordinates
        # reasoning is that if a separate object is created, these are not needed
        assert 'spher' not in cart_obj.__dict__

        # Check the id and generate the cached property simultaneously
        assert id(spherical) != id(cart_obj.spher)

        spherical2 = cart_obj.to_spherical()

        # Check that all objects are different
        assert id(spherical2) != id(spherical) != id(cart_obj.spher)

        cart2 = spherical.to_cartesian()
        assert id(cart2) != id(cart_obj)
        assert id(cart2) != id(spherical)
        assert np.allclose(cart2.arr, cart_obj.arr)
        assert np.allclose(cart2.spher, spherical)

    def test_transform_method(self, cart_obj):
        xyz = cart_obj.xyz.copy()
        rotation = Rotation.from_euler(seq='zyx', angles=np.array([0, 1.3, 1.4])).as_matrix()
        scale = np.array([0.5, 0.5, 0.5])
        translation = np.array([3, 3, 3])

        t_mat = np.eye(4)
        t_mat[:3, :3] *= rotation
        t_mat[[0, 1, 2], [0, 1, 2]] *= scale
        t_mat[:3, 3] += translation

        cart_obj.transform(rotation=rotation, scale=scale, translation=translation)

        assert 'AFFINE' in cart_obj.transform_ledger[-1][0]


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

    class TestSphericalOrigin:
        @staticmethod
        def arrays_setup(fixed_xyz):
            xyz = CartesianCoordinates(arr=fixed_xyz)
            xyz_shift = copy.deepcopy(xyz)
            xyz_shift.socs_origin = np.ones(3)
            rhv_shift = xyz_shift.to_spherical()
            xyz2_shift = rhv_shift.to_cartesian()
            return xyz, xyz_shift, rhv_shift, xyz2_shift

        def test_xyz_coords(self, known_xyz):
            xyz, xyz_shift, rhv_shift, xyz2_shift = self.arrays_setup(known_xyz)
            # Show that xyz coordinates remain the same
            assert np.allclose(xyz, xyz_shift)
            assert np.allclose(xyz.xyz, rhv_shift.xyz)
            assert np.allclose(xyz, known_xyz)
            assert np.allclose(xyz, xyz2_shift.xyz)

        def test_spherical_origin(self, known_xyz):
            xyz, xyz_shift, rhv_shift, xyz2_shift = self.arrays_setup(known_xyz)
            # Origins should be different
            assert np.all(xyz.socs_origin != xyz_shift.socs_origin)
            assert np.all(xyz.socs_origin != rhv_shift.socs_origin)
            assert np.all(xyz.socs_origin != xyz2_shift.socs_origin)

            # Origins should be maintained
            assert np.all(xyz_shift.socs_origin == rhv_shift.socs_origin)
            assert np.all(xyz_shift.socs_origin == xyz2_shift.socs_origin)

        def test_spherical_coordinates(self, known_xyz, known_spher):
            xyz, xyz_shift, rhv_shift, xyz2_shift = self.arrays_setup(known_xyz)

            assert np.allclose(xyz.spher, known_spher)                # Reference is still the same
            assert np.allclose(xyz_shift.spher, rhv_shift)   # shifted objects still match
            assert np.allclose(xyz_shift.spher, xyz2_shift.spher)   # shifted objects still match
            assert np.any(xyz.spher != xyz_shift.spher)     # Different origins should yield diff results

