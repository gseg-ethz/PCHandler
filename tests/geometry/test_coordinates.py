import copy
import warnings

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from pchandler.base_arrays import ArrayNx3
from pchandler.constants import HALF_PI, PI
from pchandler.geometry.coordinates import (
    Abstract3dCoordinates,
    AbstractCoordinates,
    CartesianCoordinates,
    # SphericalCoordinates,
    Transform,
    rhv2xyz,
    xyz2rhv,
)
from pchandler.geometry.fov import FoV

# Radius, Horizontal, Vertical (Zenith)
_known_spher = np.array(
    [
        [1, 0, HALF_PI],
        [1, HALF_PI, HALF_PI],
        [1, 0, 0],
        [1, PI, HALF_PI],
        [1, -HALF_PI, HALF_PI],
        [1, 0, PI],
        [np.sqrt(3), np.arctan2(1, 1), np.arctan2(np.sqrt(2), 1)],
    ], dtype=np.float64
)

# X, Y, Z
_known_xyz = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [1, 1, 1]
    ], dtype=np.float64
)

_small_xyz = np.random.rand(100, 3)
_large_xyz = np.random.rand(1_000_000, 3)

@pytest.fixture(scope="function")
def known_spher():
    return _known_spher

@pytest.fixture(scope="function")
def small_xyz():
    return _small_xyz

@pytest.fixture(scope="function")
def known_xyz():
    return _known_xyz

@pytest.fixture(scope="function")
def large_xyz():
    return _large_xyz

@pytest.fixture(scope="function")
def cart_obj(large_xyz) -> CartesianCoordinates:
    return CartesianCoordinates(arr=large_xyz)


class TestCartesianCoordinates:
    @pytest.mark.parametrize("array", (_small_xyz, _large_xyz, _known_xyz))
    def test_instantiation(self, array):
        a = CartesianCoordinates(arr=array)
        assert isinstance(a, CartesianCoordinates)
        assert isinstance(a, Abstract3dCoordinates)
        assert isinstance(a, AbstractCoordinates)
        assert isinstance(a, ArrayNx3)

    def test_cached_properties(self, cart_obj):
        assert "spher" not in cart_obj.__dict__
        num_items_before = len(cart_obj.__dict__)

        _ = cart_obj.r
        num_items_after = len(cart_obj.__dict__)

        assert "spher" in cart_obj.__dict__
        assert num_items_after > num_items_before

        del cart_obj.__dict__["spher"]

        num_items_after = len(cart_obj.__dict__)

        assert num_items_after == num_items_before

    @pytest.mark.parametrize("attr", ("arr", "socs_origin"))
    def test_has_attributes(self, cart_obj, attr):
        assert attr in cart_obj.__dict__.keys()
        if attr == "socs_origin":
            if cart_obj.socs_origin is not None:
                assert np.allclose(cart_obj.socs_origin, np.zeros(3))

    @pytest.mark.parametrize("prop", ("xyz", "spher", "x", "y", "z", "yxz", "rhv", "r", "hz", "v"))
    def test_has_properties(self, cart_obj, prop):
        # These properties are access methods and shouldn't be in the base dict
        assert prop in type(cart_obj).__dict__ and prop not in cart_obj.__dict__

    # @pytest.mark.parametrize("method", ("to_spherical", "from_spherical", "transform"))
    @pytest.mark.parametrize("method", ("from_spherical", "transform"))
    def test_has_methods(self, cart_obj, method):
        assert callable(getattr(cart_obj, method))

    def test_cartesian_properties(self, cart_obj):
        # Check xyz and arr are the same object
        assert id(cart_obj.arr) == id(cart_obj.xyz)
        assert np.all(cart_obj == cart_obj.arr)
        assert np.all(cart_obj.x == cart_obj[:, 0])
        assert np.all(cart_obj.y == cart_obj[:, 1])
        assert np.all(cart_obj.z == cart_obj[:, 2])

        assert np.any(cart_obj.yxz != cart_obj.arr)
        assert np.all(cart_obj.yxz[:, 0] == cart_obj[:, 1])
        assert np.all(cart_obj.yxz[:, 1] == cart_obj[:, 0])
        assert np.all(cart_obj.yxz[:, 2] == cart_obj[:, 2])

    def test_spherical_properties(self, cart_obj):
        assert id(cart_obj.arr) != id(cart_obj.spher)
        assert id(cart_obj.rhv) == id(cart_obj.spher)
        assert np.all(cart_obj.r == cart_obj.spher[:, 0])
        assert np.all(cart_obj.hz == cart_obj.spher[:, 1])
        assert np.all(cart_obj.v == cart_obj.spher[:, 2])

    def test_from_spherical(self, cart_obj):
        # This is based on the implementation when there is no SphericalCoordinates
        cart_2 = type(cart_obj).from_spherical(cart_obj.spher)
        assert isinstance(cart_2, CartesianCoordinates)

        if cart_obj.dtype == np.float64 and cart_obj.spher.dtype == np.float64:
            assert np.allclose(cart_obj, cart_2)
        elif cart_obj.dtype == np.float32:
            assert np.allclose(cart_2, cart_obj, atol=1e-6)
            # TODO Further investigation should be made why the precision is so bad

        # # check that the cart_obj cleans up the cached spherical coordinates
        # # reasoning is that if a separate object is created, these are not needed
        # assert "spher" not in cart_obj.__dict__
        #
        # # Check the id and generate the cached property simultaneously
        # assert id(spherical) != id(cart_obj.spher)
        #
        # spherical2 = cart_obj.to_spherical()
        # cart2 = CartesianCoordinates.from_spherical(spherical2)
        # assert np.allclose(cart_obj.xyz, cart2.xyz)
        #
        # # Check that all objects are different
        # assert id(spherical2) != id(spherical) != id(cart_obj.spher)
        #
        # cart2 = spherical.to_cartesian()
        # assert id(cart2) != id(cart_obj)
        # assert id(cart2) != id(spherical)
        # assert np.allclose(cart2.arr, cart_obj.arr)
        # assert np.allclose(cart2.spher, spherical)

    def test_transform(self, cart_obj):
        affine = np.eye(4) * 2
        xyz2 = cart_obj.copy()
        xyz2.transform(affine)

        assert np.allclose(cart_obj * 2, xyz2)
        #
        # rotation = Rotation.from_euler(seq="zyx", angles=np.array([0, 1.3, 1.4])).as_matrix()
        # scale = np.array([0.5, 0.5, 0.5])
        # translation = np.array([3, 3, 3])
        #
        # t_mat = np.eye(4)
        # t_mat[:3, :3] *= rotation
        # t_mat[[0, 1, 2], [0, 1, 2]] *= scale
        # t_mat[:3, 3] += translation
        #
        # cart_obj.transform(rotation=rotation, scale=scale, translation=translation)
        #
        # # TODO reimplement transform ledger then retest
        # # assert 'AFFINE' in cart_obj.transform_ledger[-1][0]

    def test_scale(self, cart_obj):
        xyz2 = cart_obj.copy()
        xyz2.scale([3, 3, 3])
        assert np.allclose(cart_obj * 3, xyz2)

    def test_rotate(self, cart_obj):
        xyz2 = cart_obj.copy()
        rot_forward = Rotation.from_euler(seq='zyx', angles=[90, 45, 30], degrees=True ).as_matrix()
        xyz2.rotate(rot_forward)
        temp = xyz2.copy()
        temp.rotate(np.linalg.inv(rot_forward))
        assert np.allclose(cart_obj, temp)
        assert not np.any(xyz2 == cart_obj)

    def test_translate(self, cart_obj):
        xyz2 = cart_obj.copy()
        xyz2.translate([3, 3, 3])
        assert np.allclose(cart_obj + 3, xyz2)

    def test_left_matrix_multiplication(self, cart_obj):
        rand_3x3 = np.random.rand(3, 3)
        with pytest.raises(NotImplementedError):
            cart_obj @ rand_3x3

    def test_right_matrix_multiplication(self, cart_obj):
        rand_3x3 = np.random.rand(3, 3)
        b = rand_3x3 @ cart_obj.T
        assert np.any(b.T != cart_obj)
        assert b.T.shape == cart_obj.shape

        arr = cart_obj.__rmatmul__(rand_3x3)
        assert isinstance(arr, type(cart_obj))
        assert arr.shape == cart_obj.shape

        same_shape_array = np.ones_like(cart_obj.arr)
        b = cart_obj.__rmatmul__(same_shape_array.T)
        assert isinstance(b, np.ndarray)
        assert b.shape != cart_obj.shape

    def test_inplace_matrix_multiplication(self, cart_obj):
        rand_3x3 = np.random.rand(3, 3)
        with pytest.raises(NotImplementedError):
            cart_obj @= rand_3x3

    def test_homogeneous_matrix_multiplication(self, cart_obj):
        rand_4x4 = np.random.rand(4, 4)
        # Will not work with numpy arrays due to left matrix multiplication approach
        with pytest.raises(ValueError):
            rand_4x4 @ cart_obj

        b = rand_4x4 @ cart_obj.H.T
        assert np.any(b.T[:, :3] != cart_obj)
        assert b.T.shape != cart_obj.shape

        transform = Transform(arr=rand_4x4)
        # c is automatically transformed into a Nx3 array
        c = transform @ cart_obj
        assert c.shape == cart_obj.shape
        assert np.allclose(b.T[:, :3], c)

    def test_get_item(self, cart_obj):
        a = cart_obj[0:10]
        assert isinstance(a, CartesianCoordinates)

        assert len(a) == 10
        assert a.shape == (10, 3)
        assert np.all(a.arr[0:10, :] == cart_obj.arr[0:10, :])

        # Indexing an object in the typical numpy fashion is possible.
        # But if an error throws, will return the numpy array
        c = cart_obj[0:10, :2]
        assert isinstance(c, np.ndarray)
        assert not isinstance(c, CartesianCoordinates)
        assert c.shape == (10, 2)

        with pytest.raises(IndexError):
            mask = np.ones((10, 3), dtype=np.bool_)
            cart_obj[mask]

    def test_fov(self, cart_obj):
        assert isinstance(cart_obj.fov, FoV)


# class TestSphericalCoordinates:
#     def test_instantiation(self, known_spher, known_xyz):
#         spherical = SphericalCoordinates(arr=known_spher)
#
#         assert isinstance(spherical, SphericalCoordinates)
#         assert np.all(spherical.spher == known_spher)
#
#         with pytest.raises(ValidationError):
#             SphericalCoordinates(arr=known_xyz)
#
#     def test_from_cartesian(self, known_spher, known_xyz):
#         cart_obj = CartesianCoordinates(arr=known_xyz)
#         spher_obj = SphericalCoordinates.from_cartesian(cart_obj)
#         assert np.allclose(spher_obj.arr, known_spher)
#
#     # # TODO uncomment when FOV re-implemented
#     # def test_fov(self, known_spher):
#     #     SphericalCoordinates(arr=known_spher).fov


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

    # TODO implement conversion tests from Cartesian and PointCloudData types
    # def test_supported_types(self, known_xyz, known_spher):
    #     xyz = CartesianCoordinates(arr=known_xyz)
    #     rhv = xyz2rhv(xyz)
    #     assert np.allclose(rhv, known_spher)
    #
    #     xyz = PointCloudData(xyz=xyz)
    #     rhv = xyz2rhv(xyz)
    #     assert np.allclose(rhv, known_spher)

    class TestSphericalOrigin:
        @staticmethod
        def arrays_setup(fixed_xyz):
            xyz = CartesianCoordinates(arr=fixed_xyz)
            xyz_shift = copy.deepcopy(xyz)
            xyz_shift.socs_origin = np.ones(3)
            rhv_shift = xyz_shift.to_spherical()
            xyz2_shift = rhv_shift.to_cartesian()
            return xyz, xyz_shift, rhv_shift, xyz2_shift

        def test_xyz_coordinates(self, known_xyz):
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

            assert np.allclose(xyz.spher, known_spher)  # Reference is still the same
            assert np.allclose(xyz_shift.spher, rhv_shift)  # shifted objects still match
            assert np.allclose(xyz_shift.spher, xyz2_shift.spher)  # shifted objects still match
            assert np.any(xyz.spher != xyz_shift.spher)  # Different origins should yield diff results
