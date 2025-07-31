import copy
import uuid
import logging
import pickle

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
from pchandler.geometry.optimal_shift import OptimizedShift
from pchandler.geometry.util import MinMaxPoints

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

N = 100
_small_xyz = np.random.rand(N, 3)
_large_xyz = np.random.rand(1_000_000, 3)


def random_coordinates(scale: float, offset: float) -> np.ndarray:
    xyz_base = np.random.randn(N, 3)
    return xyz_base * scale + offset

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
    return CartesianCoordinates(arr=large_xyz, socs_origin=np.array([0, 0, 0]))

@pytest.fixture(scope="function", autouse=True)
def sfs_():
    array = np.random.rand(N)
    return {"test": array}

@pytest.fixture(scope="function", autouse=True)
def nos_() -> OptimizedShift:
    return OptimizedShift(np.array([50000, 10000, 0]))

@pytest.fixture(scope="function", autouse=True)
def nos_mini_() -> OptimizedShift:
    return OptimizedShift(np.array([1,2,3]))


@pytest.fixture(scope="function", autouse=True)
def xyz_() -> np.ndarray:
    return random_coordinates(10, 0)

@pytest.fixture(scope="function", autouse=True)
def xyz_local_() -> np.typing.NDArray:
    return random_coordinates(1, 0)

@pytest.fixture(scope="function", autouse=True)
def xyz_global_() -> np.typing.NDArray:
    return random_coordinates(1, 100_000)

@pytest.fixture(scope="function", autouse=True)
def xyz_huge_() -> np.typing.NDArray:
    return random_coordinates(100_000, 0)


@pytest.fixture(scope="function")
def pcd_shifted(nos_) -> CartesianCoordinates:
    xyz = random_coordinates(1, 0)
    xyz += nos_.value
    return CartesianCoordinates(
        xyz=xyz,
        numerical_optimization_shift= nos_,
    )


class TestCartesianCoordinates:
    class TestAbstractCoordinates:
        def test_uuid(self, small_xyz):
            a = AbstractCoordinates(arr=small_xyz)
            assert isinstance(a.id, uuid.UUID)

            # id can be set via it's alias
            b = AbstractCoordinates(arr=small_xyz, _id=a.id)
            assert isinstance(b.id, uuid.UUID)
            assert a.id == b.id
            assert a.id is b.id

            # id is a "hidden" attribute and cannot be set with the id kwarg
            c = AbstractCoordinates(arr=small_xyz, id=a.id)
            assert isinstance(c.id, uuid.UUID)
            assert a.id != c.id
            assert a.id is not c.id

    class TestAbstract3dCoord:
        def test_default_attributes(self, small_xyz):
            # Test defaults
            xyz = CartesianCoordinates(arr=small_xyz)
            assert xyz.project_transformation is None
            assert xyz.socs_origin is None

        @pytest.mark.parametrize("socs_origin", (np.array([1, 2, 3]), (2, 3, 4), [5, -2.3, 4.879]))
        def test_valid_socs_origin(self, socs_origin, small_xyz):
            xyz = CartesianCoordinates(arr=small_xyz, socs_origin=socs_origin)

            np.all(xyz.socs_origin == np.array(socs_origin))

        @pytest.mark.parametrize("socs_origin", (
                1, "a", True, [1, 2, 3, 4], [1, 2], [1, 2, 3, 4, 5], np.ones((3,2)), {-32.4, -45.3, -2}
        ))
        def test_invalid_socs_origin(self,socs_origin, small_xyz):
            with pytest.raises(ValidationError):
                CartesianCoordinates(arr=small_xyz, socs_origin=socs_origin)

        @pytest.mark.parametrize("project_transformation", (np.eye(4), np.eye(4).tolist(),
                                                            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
                                                            ))
        def test_valid_project_transformation(self, project_transformation, small_xyz):
            xyz = CartesianCoordinates(arr=small_xyz, project_transformation=project_transformation)
            np.all(xyz.project_transformation == np.array(project_transformation))

        @pytest.mark.parametrize("project_transformation", (
                1, "a", True, [1, 2, 3, 4, 5], np.ones((3,2)), np.ones((3,3)), np.ones((4,3)), np.ones((3,3)).tolist()
        ))
        def test_invalid_project_transformation(self, project_transformation, small_xyz):
            with pytest.raises(ValidationError):
                CartesianCoordinates(arr=small_xyz, project_transformation=project_transformation)

        def test_abstract_xyz_spher_methods(self, small_xyz):
            with pytest.raises(TypeError) as e:
                Abstract3dCoordinates(arr=small_xyz)

            assert "spher" in str(e.value)
            assert "xyz" in str(e.value)

        def test_matmul(self, cart_obj):
            with pytest.raises(NotImplementedError):
                cart_obj @ np.random.rand(3, 3)

        def test_rmatmul(self, cart_obj):
            rand_3x3 = np.random.rand(3, 3)
            rand_4x4 = np.eye(4)
            rand_4x4[:3, :3] = rand_3x3.copy()
            rand_4x4[:3, 3] = 1

            # Case 1 - Left matmul method used (e.g. numpy array)
            b = rand_3x3 @ cart_obj.T
            assert np.any(b.T != cart_obj)
            assert b.T.shape == cart_obj.shape
            assert isinstance(b, np.ndarray)
            assert not isinstance(b, type(cart_obj))

            # Case 2 - Right matmul method called with 3x3 matrix
            c = cart_obj.__rmatmul__(rand_3x3)
            assert isinstance(c, type(cart_obj))
            assert not isinstance(c, np.ndarray)
            assert c.shape == cart_obj.shape
            assert np.allclose(b.T,c)

            # Case 3 - Another shaped array with matching inner coordinates
            same_shape_array = np.ones_like(cart_obj.arr)
            d = cart_obj.__rmatmul__(same_shape_array.T)
            assert isinstance(d, np.ndarray)
            assert d.shape != cart_obj.shape
            assert d.shape[0] == d.shape[1]

            # Case 4 - Right matmul with 4x4 matrix
            e = cart_obj.__rmatmul__(rand_4x4)
            assert isinstance(e, type(cart_obj))
            assert not isinstance(e, np.ndarray)
            assert e.shape == cart_obj.shape
            assert np.allclose(e-1, c)


        def test_imatmul(self, cart_obj):
            with pytest.raises(NotImplementedError):
                cart_obj @= np.random.rand(3, 3)

    @pytest.mark.parametrize("array", (_small_xyz, _large_xyz, _known_xyz))
    def test_kwarg_init(self, array):
        a = CartesianCoordinates(arr=array)
        b = CartesianCoordinates(xyz=array)
        assert np.all(a == b)
        assert a is not b
        assert a.arr is not b.arr   # numerical shift "changes" the coordinates

        for temp in (a, b):
            assert isinstance(temp, CartesianCoordinates)
            assert isinstance(temp, Abstract3dCoordinates)
            assert isinstance(temp, AbstractCoordinates)
            assert isinstance(temp, ArrayNx3)

        a = CartesianCoordinates(arr=array, numerical_optimization_shift=None)
        b = CartesianCoordinates(xyz=array, numerical_optimization_shift=None)
        assert a.arr is b.arr   #When no numerical shift, the coordinates are the same object
        assert a is not b

    def test_positional_init(self, known_xyz):
        a = CartesianCoordinates(known_xyz, numerical_optimization_shift=None)
        b = CartesianCoordinates(arr=known_xyz, numerical_optimization_shift=None)

        assert isinstance(a, CartesianCoordinates)
        assert np.all(a == b)
        assert a.arr is b.arr
        assert a is not b

    def test_unshifted_bbox_attr(self, known_xyz):
        xyz = CartesianCoordinates(arr=known_xyz)

        assert np.all(xyz.unshifted_bbox.minimum == -1)
        assert np.all(xyz.unshifted_bbox.maximum == 1)

        xyz = CartesianCoordinates(arr=known_xyz, unshifted_bbox=None)

        assert np.all(xyz.unshifted_bbox.minimum == -1)
        assert np.all(xyz.unshifted_bbox.maximum == 1)

        xyz = CartesianCoordinates(arr=known_xyz, unshifted_bbox=MinMaxPoints(minimum=[1,2,3], maximum=[4,5,6]))

        assert np.all(xyz.unshifted_bbox.minimum == [1, 2, 3])
        assert np.all(xyz.unshifted_bbox.maximum == [4, 5, 6])

    def test_compute_unshifted_bbox(self, known_xyz):
        # Initialise and get original bbox
        xyz = CartesianCoordinates(arr=known_xyz, unshifted_bbox=None)
        old = copy.deepcopy(xyz.unshifted_bbox)

        # Reset bbox to None
        xyz.unshifted_bbox = None
        assert xyz.unshifted_bbox is None

        # Run the computation and check to old
        xyz.compute_unshifted_bbox()
        assert np.all(np.array(xyz.unshifted_bbox) == np.array(old))

        # Update the coordinates, check it hasn't changed
        # TODO this case is needing handling in the future
        xyz.arr = np.random.rand(10,3)*40 - 10  # New coordinates
        xyz.compute_unshifted_bbox()    # No change
        assert np.all(np.array(xyz.unshifted_bbox) == np.array(old))

        # As coordinates have changed, this should be recomputed and no longer equal
        xyz.compute_unshifted_bbox(overwrite=True)
        assert not np.all(np.array(xyz.unshifted_bbox) == np.array(old))

    def test_process_shift(self, known_xyz):
        # TODO add checks to weakrefs in OSM
        # Case 1 - prev_shift is None and NOS is None
        #   Basic init, register to NOS
        case_1 = CartesianCoordinates(arr=known_xyz, numerical_optimization_shift=None)
        assert case_1._shift_applied_by is None
        assert case_1.numerical_optimization_shift is None
        assert np.all(case_1 == known_xyz)
        assert not case_1.numerically_optimized

        # Case 2 - prev_shift is None and NOS exists
        #   Revert to prev_shift, convert to float64, unregister prev_shift
        case_2 = CartesianCoordinates(arr=known_xyz, numerical_optimization_shift=OptimizedShift(np.array([1, 1, 1])))
        assert case_2._shift_applied_by is not None
        assert case_2.numerical_optimization_shift is not None
        assert case_2._shift_applied_by is case_2.numerical_optimization_shift
        assert np.all(case_2.numerical_optimization_shift.value == 1)
        assert np.all(case_2 + 1 == known_xyz)
        assert case_2.numerically_optimized

        # Case 3 - prev_shift exists and NOS is None
        #   Updates coordinates and removes the optimised shift
        case_3 = CartesianCoordinates(arr=known_xyz+2,
                                      _shift_applied_by=OptimizedShift(np.array([-2, -2, -2])),
                                      numerical_optimization_shift=None)
        assert case_3._shift_applied_by is None
        assert case_3.numerical_optimization_shift is None
        assert case_3._shift_applied_by is case_3.numerical_optimization_shift
        assert np.all(case_3 == known_xyz)
        assert not case_3.numerically_optimized

        # Case 4 - prev_shift exists and NOS exists
        #   If same, register to NOS. Else, apply difference, unregister prev, register NOS
        case_4 = CartesianCoordinates(arr=known_xyz+2,
                                      _shift_applied_by=OptimizedShift(np.array([-2, -2, -2])),
                                      numerical_optimization_shift=OptimizedShift(np.array([1, 1, 1])))

        assert case_4.numerical_optimization_shift is case_4._shift_applied_by
        assert case_4.numerical_optimization_shift is not None
        assert np.all(case_4.numerical_optimization_shift.value == 1)
        assert np.all(case_4 + 1 == known_xyz)
        assert case_4.numerically_optimized

    def test_reduce(self):
        xyz = CartesianCoordinates(arr=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]))
        xyz2 = copy.deepcopy(xyz)
        xyz.reduce([0, 3, 4])
        assert len(xyz) == 3
        assert len(xyz2) == 6
        assert np.all(xyz[0, :] == xyz2[0, :])
        assert np.all(xyz[1, :] == xyz2[3, :])
        assert np.all(xyz[2, :] == xyz2[4, :])
        assert not np.all(np.array(xyz.unshifted_bbox) == np.array(xyz2.unshifted_bbox))
        assert np.all(xyz.unshifted_bbox.minimum == xyz2.unshifted_bbox.minimum )
        assert np.all(xyz.unshifted_bbox.maximum == 4 )
        assert np.all(xyz2.unshifted_bbox.maximum == 5 )

    def test_sample(self):
        xyz = CartesianCoordinates(arr=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]))
        sample = xyz.sample([0, 3, 4])
        assert len(sample) == 3
        assert len(xyz) == 6
        assert np.all(sample[0, :] == xyz[0, :])
        assert np.all(sample[1, :] == xyz[3, :])
        assert np.all(sample[2, :] == xyz[4, :])
        assert not np.all(np.array(sample.unshifted_bbox) == np.array(xyz.unshifted_bbox))
        assert np.all(sample.unshifted_bbox.minimum == xyz.unshifted_bbox.minimum )
        assert np.all(sample.unshifted_bbox.maximum == 4 )
        assert np.all(xyz.unshifted_bbox.maximum == 5 )

    def test_update_shift(self, known_xyz):
        xyz = CartesianCoordinates(arr=known_xyz,
                                   socs_origin=np.array([-1, 4, 2.5]),
                                   _shift_applied_by=OptimizedShift(np.array([-2, -2, -2])),
                                   numerical_optimization_shift=OptimizedShift(np.array([1, 1, 1])))

        assert np.all(xyz + 3 == known_xyz)
        assert np.all(xyz.socs_origin == np.array([-4, 1, -0.5]))
        assert xyz.arr.dtype == np.float32
        assert xyz.socs_origin.dtype == np.float32


        xyz = CartesianCoordinates(arr=known_xyz,
                                   socs_origin=np.array([-1, 4, 2.5]),
                                   _shift_applied_by=OptimizedShift(np.array([-2, -2, -2])),
                                   numerical_optimization_shift=None)

        assert np.all(xyz + 2 == known_xyz)
        assert np.all(xyz.socs_origin == np.array([-3, 2, 0.5]))
        assert xyz.arr.dtype == np.float64
        assert xyz.socs_origin.dtype == np.float64
        assert xyz._shift_applied_by is None
        assert xyz.numerical_optimization_shift is None

        xyz2 = CartesianCoordinates(arr=xyz.arr,
                                    socs_origin=np.array([5,6,7]),
                                    _shift_applied_by=xyz.numerical_optimization_shift,
                                    numerical_optimization_shift=xyz.numerical_optimization_shift)

        assert np.all(xyz2.socs_origin == [5,6,7])
        assert xyz2._shift_applied_by is xyz.numerical_optimization_shift
        assert xyz2._shift_applied_by is xyz2.numerical_optimization_shift

    def test_register_with_shift_at_osm(self, known_xyz):
        xyz = CartesianCoordinates(arr=known_xyz + [14_123.842, 0, 9_000])
        xyz = CartesianCoordinates(arr=known_xyz + [14_123.842, 0, 9_000])
        assert xyz.numerical_optimization_shift is not None
        raise NotImplementedError

    def test_setattr(self):
        raise NotImplementedError

    def test_hash(self):
        raise NotImplementedError

    def test_model_dump(self):
        raise NotImplementedError

    def test__reduce__(self):
        raise NotImplementedError

    def test_reconstruct(self):
        raise NotImplementedError

    def test_merge(self):
        raise NotImplementedError

    def test_numerically_optimized(self):
        raise NotImplementedError

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
        assert np.all(cart_obj._hz_v == cart_obj.spher[:, 1:])

    def test_fov(self):
        assert isinstance(cart_obj.fov, FoV)
        raise NotImplementedError

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

    def test_rotate(self, cart_obj):
        xyz2 = type(cart_obj)(arr=cart_obj.copy())
        rot_forward = Rotation.from_euler(seq='zyx', angles=[90, 45, 30], degrees=True).as_matrix()
        xyz2.rotate(rot_forward)
        temp = type(xyz2)(arr=xyz2.copy())
        temp.rotate(np.linalg.inv(rot_forward))
        assert np.allclose(cart_obj, temp, atol=1e-6)
        assert not np.any(xyz2 == cart_obj)

    def test_translate(self, cart_obj):
        xyz2 = type(cart_obj)(arr=cart_obj)
        xyz2.translate(np.array([3, 3, 3]))
        assert np.allclose(cart_obj + 3, xyz2)

    def test_scale(self, cart_obj):
        xyz2 = type(cart_obj)(arr=cart_obj)
        xyz2.scale(np.array([3, 3, 3]))
        assert np.allclose(cart_obj * 3, xyz2)

    def test_transform(self, cart_obj):
        xyz2 = type(cart_obj)(arr=cart_obj)
        affine = np.eye(4) * 2
        affine[:3, 3] = 1
        xyz2.transform(affine)

        assert np.allclose(cart_obj * 2 + 1, xyz2)
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
        # assert 'AFFINE' in cart_obj.transform_ledger[-1][0]

    def test_homogeneous_matrix_multiplication(self, cart_obj):
        rand_4x4 = np.random.rand(4, 4)
        # Will not work with numpy arrays due to left matrix multiplication approach
        with pytest.raises(ValueError):
            rand_4x4 @ cart_obj

        b = rand_4x4 @ cart_obj.H.T
        assert np.any(b.T[:, :3] != cart_obj)
        assert b.T.shape != cart_obj.shape

        transform = Transform(arr=rand_4x4)
        with pytest.raises(ValueError):
            transform @ cart_obj

        valid_transform = transform.copy()
        valid_transform[3, :3] = 0
        valid_transform[3, 3] = 1

        # c is automatically transformed into a Nx3 array
        c = valid_transform @ cart_obj
        assert c.shape == cart_obj.shape
        assert np.allclose(b.T[:, :3], c)

    class TestNOS:
        class TestNOSInstantiation:
            def test_defaults_to_float32_and_zero_shift(self, xyz_):
                pcd = CartesianCoordinates(xyz=xyz_)
                assert pcd.xyz.dtype == np.float32
                assert np.allclose(pcd.numerical_optimization_shift.value, [0, 0, 0])
                assert np.allclose(pcd.xyz, xyz_)
                assert np.allclose(pcd.unshifted_bbox.minimum, xyz_.min(axis=0))
                assert np.allclose(pcd.unshifted_bbox.maximum, xyz_.max(axis=0))

            def test_instantiation_without_nos_keeps_float64_and_none_shift(self, xyz_):
                pcd_no_nos = CartesianCoordinates(xyz=xyz_, numerical_optimization_shift=None)
                assert pcd_no_nos.xyz.dtype == np.float64
                assert pcd_no_nos.numerical_optimization_shift is None
                assert np.allclose(pcd_no_nos.xyz, xyz_)
                assert np.allclose(pcd_no_nos.unshifted_bbox.minimum, xyz_.min(axis=0))
                assert np.allclose(pcd_no_nos.unshifted_bbox.maximum, xyz_.max(axis=0))

            def test_auto_shift_applied_when_points_are_large(self, xyz_local_, nos_):
                xyz_shifted = xyz_local_ + nos_.value
                pcd_nos = CartesianCoordinates(xyz=xyz_shifted)
                assert pcd_nos.xyz.dtype == np.float32
                assert pcd_nos.numerical_optimization_shift is not nos_
                assert np.allclose(pcd_nos.numerical_optimization_shift.value, nos_.value)
                assert np.allclose(pcd_nos.xyz, xyz_local_) # This only holds because xyz_shifted is very close to 0,0,0
                assert np.allclose(pcd_nos.unshifted_bbox.minimum, xyz_shifted.min(axis=0))
                assert np.allclose(pcd_nos.unshifted_bbox.maximum, xyz_shifted.max(axis=0))

            def test_predefined_shift_when_points_are_large(self, xyz_local_, nos_):
                xyz_shifted = xyz_local_ + nos_.value
                pcd_nos_expected = CartesianCoordinates(xyz=xyz_shifted, numerical_optimization_shift=nos_)
                assert pcd_nos_expected.xyz.dtype == np.float32
                assert pcd_nos_expected.numerical_optimization_shift is nos_
                assert np.allclose(pcd_nos_expected.xyz, xyz_local_)
                assert np.allclose(pcd_nos_expected.unshifted_bbox.minimum, xyz_shifted.min(axis=0))
                assert np.allclose(pcd_nos_expected.unshifted_bbox.maximum, xyz_shifted.max(axis=0))

            def test_unsuitable_shift(self, xyz_, nos_, caplog):
                caplog.set_level(logging.DEBUG)
                pcd_nos = CartesianCoordinates(xyz=xyz_ + nos_.value, numerical_optimization_shift=nos_) #Needed to bind nos_

                pcd_nos_unexpected = CartesianCoordinates(xyz=xyz_, numerical_optimization_shift=nos_)
                assert pcd_nos_unexpected.xyz.dtype == np.float32
                assert pcd_nos_unexpected.numerical_optimization_shift is not nos_
                assert "provided numerical_optimization_shift was not feasible" in caplog.text
                assert np.allclose(pcd_nos_unexpected.xyz, xyz_ - pcd_nos_unexpected.numerical_optimization_shift.value)
                assert np.allclose(pcd_nos_unexpected.unshifted_bbox.minimum, xyz_.min(axis=0))
                assert np.allclose(pcd_nos_unexpected.unshifted_bbox.maximum, xyz_.max(axis=0))


            def test_unsuitable_coordinates_for_shift(self, xyz_huge_, caplog):
                caplog.set_level(logging.DEBUG)
                pcd_huge = CartesianCoordinates(xyz=xyz_huge_)
                assert pcd_huge.xyz.dtype == np.float64
                assert pcd_huge.numerical_optimization_shift is None
                assert "No numerical_optimization_shift was feasible." in caplog.text
                assert np.allclose(pcd_huge.unshifted_bbox.minimum, xyz_huge_.min(axis=0))
                assert np.allclose(pcd_huge.unshifted_bbox.maximum, xyz_huge_.max(axis=0))

            # xyz_global = random_coordinates(1,100_000)
            # pcd_global = PointCloudData(xyz=xyz_global)
            # assert pcd_global.xyz.dtype == np.float32
            # assert np.allclose(pcd_global.numerical_optimization_shift.value, 3*[100_000,])
            # assert np.allclose(pcd_global.xyz, xyz_global - np.array(3*[100_000,]))
            # assert np.allclose(pcd_global.unshifted_bbox.minimum, xyz_global.min(axis=0))
            # assert np.allclose(pcd_global.unshifted_bbox.maximum, xyz_global.max(axis=0))

        class TestNOSChange:
            def test_updating_from_default_to_predefined_shift_logs_and_keeps_bbox(self, xyz_local_, nos_, caplog):
                caplog.set_level(logging.DEBUG)

                pcd = CartesianCoordinates(xyz=xyz_local_)
                pcd.numerical_optimization_shift = nos_
                assert "Updating shift" in caplog.text
                assert np.allclose(pcd.numerical_optimization_shift.value, [0,0,0])
                assert np.allclose(pcd.unshifted_bbox.minimum, xyz_local_.min(axis=0))
                assert np.allclose(pcd.unshifted_bbox.maximum, xyz_local_.max(axis=0))


            def test_applying_small_shift_after_default_instantiation_adjusts_coords(self, xyz_local_, nos_mini_):

                pcd_local = CartesianCoordinates(xyz=xyz_local_)
                assert np.allclose(pcd_local.numerical_optimization_shift.value, [0, 0, 0])
                assert np.allclose(pcd_local.xyz, xyz_local_)
                assert np.allclose(pcd_local.unshifted_bbox.minimum, xyz_local_.min(axis=0))
                assert np.allclose(pcd_local.unshifted_bbox.maximum, xyz_local_.max(axis=0))

                pcd_local.numerical_optimization_shift = nos_mini_
                assert np.allclose(pcd_local.xyz, xyz_local_ - nos_mini_.value)
                assert np.allclose(pcd_local.unshifted_bbox.minimum, xyz_local_.min(axis=0))
                assert np.allclose(pcd_local.unshifted_bbox.maximum, xyz_local_.max(axis=0))


            def test_changing_from_none_to_predefined_shift_changes_dtype_and_coords(self, xyz_local_, nos_mini_):
                # Case: changing from None to predefined shift; additional checks on change in coordinate dtype
                pcd_unshifted = CartesianCoordinates(xyz=xyz_local_, numerical_optimization_shift=None)
                assert pcd_unshifted.xyz.dtype == np.float64
                assert np.allclose(pcd_unshifted.unshifted_bbox.minimum, xyz_local_.min(axis=0))
                assert np.allclose(pcd_unshifted.unshifted_bbox.maximum, xyz_local_.max(axis=0))

                pcd_unshifted.numerical_optimization_shift = nos_mini_
                assert np.allclose(pcd_unshifted.xyz, xyz_local_ - nos_mini_.value)
                assert pcd_unshifted.xyz.dtype == np.float32
                assert np.allclose(pcd_unshifted.unshifted_bbox.minimum, xyz_local_.min(axis=0))
                assert np.allclose(pcd_unshifted.unshifted_bbox.maximum, xyz_local_.max(axis=0))


            def test_changing_from_predefined_shift_to_none_restores_dtype_and_coords(self, xyz_local_, nos_mini_):
                # Case: changing from predefined shift to None; additional checks on change in coordinate dtype
                pcd_shifted = CartesianCoordinates(xyz=xyz_local_, numerical_optimization_shift=nos_mini_)
                pcd_shifted.numerical_optimization_shift = None
                assert pcd_shifted.xyz.dtype == np.float64
                assert np.allclose(pcd_shifted.xyz, xyz_local_, rtol=1e-5, atol=1e-6) # Due to the conversion to float32 and back
                assert pcd_shifted not in nos_mini_
                assert pcd_shifted.numerical_optimization_shift is None
                assert np.allclose(pcd_shifted.unshifted_bbox.minimum, xyz_local_.min(axis=0))
                assert np.allclose(pcd_shifted.unshifted_bbox.maximum, xyz_local_.max(axis=0))

            def test_copying_predefined_shift_creates_distinct_uuid_but_same_value(self, nos_):
                # Case: copying predefined shift
                nos2 = copy.deepcopy(nos_)
                assert nos_.uuid != nos2.uuid
                assert np.allclose(nos_.value, nos2.value)

            def test_initial_large_points_then_apply_provided_shift_recovers_original(self, xyz_local_, nos_):
                # Case: changing from initial (large) to predefined shift
                xyz_shifted = xyz_local_ + nos_.value
                pcd_shifted = CartesianCoordinates(xyz=xyz_shifted)
                pcd_shifted.numerical_optimization_shift = nos_
                assert np.allclose(pcd_shifted.xyz, xyz_local_)
                assert np.allclose(pcd_shifted.unshifted_bbox.minimum, xyz_shifted.min(axis=0))
                assert np.allclose(pcd_shifted.unshifted_bbox.maximum, xyz_shifted.max(axis=0))

    class TestCopy:
        def test_deepcopy(self, pcd_shifted):
            pcd_copy = pcd_shifted.copy(deep=True, link_to_same_NOS=True)
            assert isinstance(pcd_copy, CartesianCoordinates)
            assert id(pcd_shifted.numerical_optimization_shift) == id(pcd_copy.numerical_optimization_shift)
            assert np.allclose(pcd_copy.xyz, pcd_shifted.xyz)

            # TODO: Rework the deepcopy of NOS
            pcd_copy = pcd_shifted.copy(deep=True, link_to_same_NOS=False)
            assert id(pcd_shifted.numerical_optimization_shift) != id(pcd_copy.numerical_optimization_shift)
            assert np.allclose(pcd_copy.xyz, pcd_shifted.xyz)

    class TestPickle:
        def test_pcd_pickle(self, pcd_shifted):
            pickle_pcd = pickle.dumps(pcd_shifted)
            unpickled_pcd = pickle.loads(pickle_pcd)

            assert isinstance(unpickled_pcd, CartesianCoordinates)
            assert np.allclose(unpickled_pcd.xyz, pcd_shifted.xyz)
            assert unpickled_pcd.id == pcd_shifted.id
            assert unpickled_pcd in pcd_shifted.numerical_optimization_shift

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

    def test_supported_types(self, known_xyz, known_spher):
        xyz = CartesianCoordinates(known_xyz)
        rhv = xyz2rhv(xyz)
        assert np.allclose(rhv, known_spher)

        xyz = CartesianCoordinates(xyz)
        rhv = xyz2rhv(xyz)
        assert np.allclose(rhv, known_spher)

        # Show that a translation shift is the same
        xyz += 1
        xyz.socs_origin = np.ones(3)
        rhv = xyz2rhv(xyz, xyz.socs_origin)
        assert np.allclose(rhv, known_spher)


    class TestSphericalOrigin:
        def test_socs_origin(self, known_xyz, known_spher):
            # Different origin but same underlying coordinates
            xyz = CartesianCoordinates(arr=known_xyz)
            xyz_shift: CartesianCoordinates = CartesianCoordinates(arr=known_xyz, socs_origin=np.ones(3))
            assert np.all(xyz == xyz_shift)
            assert xyz.socs_origin is not xyz_shift.socs_origin
            assert xyz.socs_origin is None
            assert np.all(xyz_shift.socs_origin == 1)

            # Spherical coordinates will differ
            assert not np.allclose(xyz.spher, xyz_shift.spher)
            assert not np.allclose(known_spher, xyz_shift.spher)

            # Shifting the coordinates to mimic the relative position of the socs origin
            xyz2 = xyz.copy(deep=True)
            xyz2 -= 1
            assert np.allclose(xyz2.spher, xyz_shift.spher)
            assert xyz2.socs_origin is None
            assert xyz_shift.socs_origin is not None

            xyz_shift2: CartesianCoordinates = CartesianCoordinates(arr=known_xyz, socs_origin=np.array([-1, -2, -3]))

            assert np.any(xyz.spher != xyz_shift2.spher)  # Different origins should yield diff results
