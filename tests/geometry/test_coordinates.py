import copy
import logging
import pickle
import uuid

import numpy as np
import pytest
from GSEGUtils.base_arrays import ArrayNx3
from GSEGUtils.constants import HALF_PI, PI
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from pchandler.geometry import OptimizedShift
from pchandler.geometry.coordinates import (  # SphericalCoordinates,
    Abstract3dCoordinates,
    AbstractCoordinates,
    CartesianCoordinates,
    Transform,
    rhv2xyz,
    xyz2rhv,
)
from pchandler.geometry.spherical import FoV
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
    ],
    dtype=np.float64,
)

# X, Y, Z
_known_xyz = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 1]], dtype=np.float64
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
    return OptimizedShift(np.array([1, 2, 3]))


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
        numerical_optimization_shift=nos_,
    )


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

        assert id(a) != a.id


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

    @pytest.mark.parametrize(
        "socs_origin", (1, "a", True, [1, 2, 3, 4], [1, 2], [1, 2, 3, 4, 5], np.ones((3, 2)), {-32.4, -45.3, -2})
    )
    def test_invalid_socs_origin(self, socs_origin, small_xyz):
        with pytest.raises(ValidationError):
            CartesianCoordinates(arr=small_xyz, socs_origin=socs_origin)

    @pytest.mark.parametrize(
        "project_transformation",
        (np.eye(4), np.eye(4).tolist(), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))),
    )
    def test_valid_project_transformation(self, project_transformation, small_xyz):
        xyz = CartesianCoordinates(arr=small_xyz, project_transformation=project_transformation)
        np.all(xyz.project_transformation == np.array(project_transformation))

    @pytest.mark.parametrize(
        "project_transformation",
        (1, "a", True, [1, 2, 3, 4, 5], np.ones((3, 2)), np.ones((3, 3)), np.ones((4, 3)), np.ones((3, 3)).tolist()),
    )
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
        assert np.allclose(b.T, c)

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
        assert np.allclose(e - 1, c)

    def test_imatmul(self, cart_obj):
        with pytest.raises(NotImplementedError):
            cart_obj @= np.random.rand(3, 3)


# TODO implement base class for tests
class BaseTestCartesianCoordinates:
    cls = CartesianCoordinates

    @pytest.mark.parametrize("array", (_small_xyz, _large_xyz, _known_xyz))
    def test_kwarg_init(self, array):
        a = self.cls(arr=array)
        b = self.cls(xyz=array)
        assert np.all(a == b)
        assert a is not b
        assert a.arr is not b.arr  # numerical shift "changes" the coordinates

        for temp in (a, b):
            assert isinstance(temp, self.cls)
            assert isinstance(temp, Abstract3dCoordinates)
            assert isinstance(temp, AbstractCoordinates)
            assert isinstance(temp, ArrayNx3)

        a = self.cls(arr=array, numerical_optimization_shift=None)
        b = self.cls(xyz=array, numerical_optimization_shift=None)
        assert a.arr is b.arr  # When no numerical shift, the coordinates are the same object
        assert a is not b

        with pytest.raises(ValueError):
            a = self.cls(arr=array, xyz=array)

    def test_positional_init(self, known_xyz):
        a = self.cls(known_xyz, numerical_optimization_shift=None)
        b = self.cls(arr=known_xyz, numerical_optimization_shift=None)

        assert isinstance(a, self.cls)
        assert np.all(a == b)
        assert a.arr is b.arr
        assert a is not b

    def test_unshifted_bbox_attr(self, known_xyz):
        xyz = self.cls(arr=known_xyz)

        assert np.all(xyz.unshifted_bbox.minimum == -1)
        assert np.all(xyz.unshifted_bbox.maximum == 1)

        xyz = self.cls(arr=known_xyz, unshifted_bbox=None)

        assert np.all(xyz.unshifted_bbox.minimum == -1)
        assert np.all(xyz.unshifted_bbox.maximum == 1)

        xyz = self.cls(
            arr=known_xyz, unshifted_bbox=MinMaxPoints(minimum=np.array([1, 2, 3]), maximum=np.array([4, 5, 6]))
        )

        assert np.all(xyz.unshifted_bbox.minimum == [1, 2, 3])
        assert np.all(xyz.unshifted_bbox.maximum == [4, 5, 6])

    def test_compute_unshifted_bbox(self, known_xyz):
        # Initialise and get original bbox
        xyz = self.cls(arr=known_xyz, unshifted_bbox=None)
        old = copy.deepcopy(xyz.unshifted_bbox)

        # Reset bbox to None
        xyz.unshifted_bbox = None
        assert xyz.unshifted_bbox is None

        # Run the computation and check to old
        xyz.compute_unshifted_bbox()
        assert np.all(np.array(xyz.unshifted_bbox) == np.array(old))

        # Update the coordinates, check it hasn't changed
        xyz.arr = np.random.rand(10, 3) * 40 - 10  # New coordinates
        xyz.compute_unshifted_bbox()  # No change
        assert np.all(np.array(xyz.unshifted_bbox) == np.array(old))

        # As coordinates have changed, this should be recomputed and no longer equal
        xyz.compute_unshifted_bbox(overwrite=True)
        assert not np.all(np.array(xyz.unshifted_bbox) == np.array(old))

    def test_process_shift(self, known_xyz):
        # Case 1 - prev_shift is None and NOS is None
        #   Basic init, register to NOS
        case_1 = self.cls(arr=known_xyz, numerical_optimization_shift=None)
        assert case_1._shift_applied_by is None
        assert case_1.numerical_optimization_shift is None
        assert np.all(case_1 == known_xyz)
        assert not case_1.numerically_optimized

        # Case 2 - prev_shift is None and NOS exists
        #   Revert to prev_shift, convert to float64, unregister prev_shift
        case_2 = self.cls(arr=known_xyz, numerical_optimization_shift=OptimizedShift(np.array([1, 1, 1])))
        assert case_2._shift_applied_by is not None
        assert case_2.numerical_optimization_shift is not None
        assert case_2._shift_applied_by is case_2.numerical_optimization_shift
        assert np.all(case_2.numerical_optimization_shift.value == 1)
        assert np.all(case_2 + 1 == known_xyz)
        assert case_2.numerically_optimized

        # Case 3 - prev_shift exists and NOS is None
        #   Updates coordinates and removes the optimised shift
        case_3 = self.cls(
            arr=known_xyz + 2,
            _shift_applied_by=OptimizedShift(np.array([-2, -2, -2])),
            numerical_optimization_shift=None,
        )
        assert case_3._shift_applied_by is None
        assert case_3.numerical_optimization_shift is None
        assert case_3._shift_applied_by is case_3.numerical_optimization_shift
        assert np.all(case_3 == known_xyz)
        assert not case_3.numerically_optimized

        # Case 4 - prev_shift exists and NOS exists
        #   If same, register to NOS. Else, apply difference, unregister prev, register NOS
        case_4 = self.cls(
            arr=known_xyz + 2,
            _shift_applied_by=OptimizedShift(np.array([-2, -2, -2])),
            numerical_optimization_shift=OptimizedShift(np.array([1, 1, 1])),
        )

        assert case_4.numerical_optimization_shift is case_4._shift_applied_by
        assert case_4.numerical_optimization_shift is not None
        assert np.all(case_4.numerical_optimization_shift.value == 1)
        assert np.all(case_4 + 1 == known_xyz)
        assert case_4.numerically_optimized

    def test_reduce(self):
        xyz = self.cls(arr=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]))
        xyz2 = copy.deepcopy(xyz)
        xyz.reduce([0, 3, 4])
        assert len(xyz) == 3
        assert len(xyz2) == 6
        assert np.all(xyz[0, :] == xyz2[0, :])
        assert np.all(xyz[1, :] == xyz2[3, :])
        assert np.all(xyz[2, :] == xyz2[4, :])
        assert not np.all(np.array(xyz.unshifted_bbox) == np.array(xyz2.unshifted_bbox))
        assert np.all(xyz.unshifted_bbox.minimum == xyz2.unshifted_bbox.minimum)
        assert np.all(xyz.unshifted_bbox.maximum == 4)
        assert np.all(xyz2.unshifted_bbox.maximum == 5)

    def test_sample(self):
        xyz = self.cls(arr=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]))
        sample = xyz.sample([0, 3, 4])
        assert len(sample) == 3
        assert len(xyz) == 6
        assert np.all(sample[0, :] == xyz[0, :])
        assert np.all(sample[1, :] == xyz[3, :])
        assert np.all(sample[2, :] == xyz[4, :])
        assert not np.all(np.array(sample.unshifted_bbox) == np.array(xyz.unshifted_bbox))
        assert np.all(sample.unshifted_bbox.minimum == xyz.unshifted_bbox.minimum)
        assert np.all(sample.unshifted_bbox.maximum == 4)
        assert np.all(xyz.unshifted_bbox.maximum == 5)

    def test_update_shift(self, known_xyz):
        xyz = self.cls(
            arr=known_xyz,
            socs_origin=np.array([-1, 4, 2.5]),
            _shift_applied_by=OptimizedShift(np.array([-2, -2, -2])),
            numerical_optimization_shift=OptimizedShift(np.array([1, 1, 1])),
        )

        assert np.all(xyz + 3 == known_xyz)
        assert np.all(xyz.socs_origin == np.array([-4, 1, -0.5]))
        assert xyz.arr.dtype == np.float32
        assert xyz.socs_origin.dtype == np.float32

        xyz = self.cls(
            arr=known_xyz,
            socs_origin=np.array([-1, 4, 2.5]),
            _shift_applied_by=OptimizedShift(np.array([-2, -2, -2])),
            numerical_optimization_shift=None,
        )

        assert np.all(xyz + 2 == known_xyz)
        assert np.all(xyz.socs_origin == np.array([-3, 2, 0.5]))
        assert xyz.arr.dtype == np.float64
        assert xyz.socs_origin.dtype == np.float64
        assert xyz._shift_applied_by is None
        assert xyz.numerical_optimization_shift is None

        xyz2 = self.cls(
            arr=xyz.arr,
            socs_origin=np.array([5, 6, 7]),
            _shift_applied_by=xyz.numerical_optimization_shift,
            numerical_optimization_shift=xyz.numerical_optimization_shift,
        )

        assert np.all(xyz2.socs_origin == [5, 6, 7])
        assert xyz2._shift_applied_by is xyz.numerical_optimization_shift
        assert xyz2._shift_applied_by is xyz2.numerical_optimization_shift

    @staticmethod
    def test_setattr(cart_obj):
        new_xyz = cart_obj.copy()
        with pytest.raises(AttributeError):
            new_xyz._shift_applied_by = OptimizedShift(np.array([1, 1, 1]))

        new_xyz.numerical_optimization_shift = OptimizedShift(np.array([1, 1, 1]))
        assert np.allclose((new_xyz + new_xyz.numerical_optimization_shift), cart_obj.xyz, rtol=1e-5, atol=1e-6)

        new_xyz.socs_origin = [2, 3, 3]
        assert np.all(new_xyz.socs_origin == [2, 3, 3])

        with pytest.raises(ValidationError):
            new_xyz.socs_origin = "invalid_type"

    @staticmethod
    def test_hash(cart_obj):
        assert id(cart_obj) == cart_obj.__hash__()

    @staticmethod
    def test_model_dump(cart_obj):
        dump = cart_obj.model_dump()
        for attr in (
            "arr",
            "id",
            "project_transformation",
            "socs_origin",
            "unshifted_bbox",
            "_shift_applied_by",
            "numerical_optimization_shift",
        ):
            assert attr in dump
            if isinstance(dump[attr], np.ndarray):
                assert np.all(dump[attr] == getattr(cart_obj, attr))
            else:
                assert dump[attr] == getattr(cart_obj, attr)

    @staticmethod
    def test__reduce__(cart_obj):
        func, state = cart_obj.__reduce__()
        assert callable(func)
        assert isinstance(state, tuple)
        assert isinstance(state[0], dict)
        assert "arr" in state[0]

    @staticmethod
    def test_reconstruct(cart_obj):
        func, state = cart_obj.__reduce__()
        new_cart_obj = func(state[0])
        assert isinstance(new_cart_obj, type(cart_obj))
        assert new_cart_obj.id == cart_obj.id
        assert new_cart_obj is not cart_obj
        assert new_cart_obj.arr is cart_obj.arr  # still has the same reference as the object wasn't deleted

    @staticmethod
    def test_reconstruct_pickle_round_trip_validates_positive(cart_obj):
        """Plan 02-02 / SEC-02 / D-13 positive regression.

        ``pickle.loads(pickle.dumps(coord))`` round-trips with per-field
        validation: every field passes its TypeAdapter check before reaching
        ``model_construct``. Verifies the validated path produces an equal
        :class:`CartesianCoordinates`, including ``_shift_applied_by``.
        """
        blob = pickle.dumps(cart_obj)
        revived = pickle.loads(blob)
        assert isinstance(revived, type(cart_obj))
        assert revived.id == cart_obj.id
        np.testing.assert_array_equal(np.asarray(revived.arr), np.asarray(cart_obj.arr))
        # _shift_applied_by may be None for an unshifted PCD, OR an OptimizedShift
        # instance for a shifted one — match the original.
        if cart_obj._shift_applied_by is None:
            assert revived._shift_applied_by is None
        else:
            assert isinstance(revived._shift_applied_by, type(cart_obj._shift_applied_by))

    @staticmethod
    def test_reconstruct_rejects_wrong_dtype_state():
        """Plan 02-02 / SEC-02 / D-13 negative — security regression.

        A hand-crafted state dict with ``arr=np.array(["a", "b"])`` (wrong
        dtype + wrong shape) MUST raise :class:`pydantic.ValidationError`
        when passed through :meth:`CartesianCoordinates._reconstruct`.
        Per-field :class:`TypeAdapter.validate_python` rejects the malformed
        ``arr`` before ``model_construct`` runs.

        Without the SEC-02 fix, ``model_construct(**state)`` would have
        accepted the malformed array, producing a broken
        :class:`CartesianCoordinates` that downstream filters would then
        crash on at use time (or worse, silently propagate).
        """
        malicious_state = {
            "arr": np.array(["a", "b"]),  # wrong dtype + wrong shape
            "id": None,
            "project_transformation": None,
            "socs_origin": None,
            "unshifted_bbox": None,
            "numerical_optimization_shift": None,
            "_shift_applied_by": None,
        }
        with pytest.raises(ValidationError):
            CartesianCoordinates._reconstruct(malicious_state)

    @staticmethod
    def test_reconstruct_field_set_pinned(cart_obj):
        """Plan 02-02 / regression guard: model_dump() field set is stable.

        If a future field is added to Abstract3dCoordinates /
        CartesianCoordinates without a corresponding entry in
        ``_FIELD_VALIDATORS``, this test fails — forcing the maintainer to
        update the validator dict.

        ``unshifted_bbox`` is the only computed field deliberately skipped
        from ``_FIELD_VALIDATORS`` (W-7 / RESEARCH §"Open Question 1"
        Rec. (b)); other fields MUST have a validator entry.
        """
        from pchandler.geometry.coordinates import _FIELD_VALIDATORS

        dumped = cart_obj.model_dump()
        # All dumped fields except `unshifted_bbox` (computed, W-7) MUST appear in
        # _FIELD_VALIDATORS. `id` is now also keyed (W-2).
        exempt = {"unshifted_bbox"}
        unvalidated = set(dumped.keys()) - set(_FIELD_VALIDATORS.keys()) - exempt
        assert not unvalidated, (
            f"New fields in CartesianCoordinates.model_dump() must be added to "
            f"_FIELD_VALIDATORS (or explicitly exempted): {unvalidated}"
        )

    @staticmethod
    def test_copy_default(cart_obj):
        # Deep copy, new ID must be generated, no refs maintained except link to same NOS
        cart_obj.project_transformation = np.eye(4)
        new_xyz = cart_obj.copy()

        for new_xyz in (cart_obj.copy(), cart_obj.copy(deep=True)):
            # Both reference and UUID should be different
            assert id(new_xyz) != id(cart_obj)
            assert new_xyz.id != cart_obj.id

            assert new_xyz.arr is not cart_obj.arr
            assert np.all(new_xyz.arr == cart_obj.arr)

            # Default links back to same NOS
            assert new_xyz.numerical_optimization_shift is cart_obj.numerical_optimization_shift

            assert new_xyz._shift_applied_by is cart_obj._shift_applied_by
            assert new_xyz.project_transformation is not cart_obj.project_transformation
            assert np.all(new_xyz.project_transformation == cart_obj.project_transformation)

            assert new_xyz.socs_origin is not cart_obj.socs_origin
            assert np.all(new_xyz.socs_origin == cart_obj.socs_origin)

            assert new_xyz.unshifted_bbox is not cart_obj.unshifted_bbox
            assert np.all(new_xyz.unshifted_bbox.minimum == cart_obj.unshifted_bbox.minimum)
            assert np.all(new_xyz.unshifted_bbox.maximum == cart_obj.unshifted_bbox.maximum)

    @staticmethod
    def test_copy_array(cart_obj):
        new_xyz = cart_obj.copy([[1, 2, 3], [4, 5, 6]])
        assert np.all(new_xyz.arr == np.array([[1, 2, 3], [4, 5, 6]]))

        with pytest.raises(ValidationError):
            cart_obj.copy("Not a valid array")

    @staticmethod
    def test_copy_update(cart_obj):
        socs = np.array([-1, 4, 2.5])
        proj_transform = np.eye(4)
        new_xyz = cart_obj.copy(update={"socs_origin": socs, "project_transformation": proj_transform})

        assert new_xyz.socs_origin is not cart_obj.socs_origin
        assert new_xyz.socs_origin is socs

        assert new_xyz.project_transformation is not cart_obj.project_transformation
        assert new_xyz.project_transformation is proj_transform

    @staticmethod
    def test_copy_dont_link_to_same_nos(cart_obj):
        # Should still be the same NOS as the coordinates are the same
        new_xyz = cart_obj.copy(link_to_same_NOS=False)
        assert new_xyz.numerical_optimization_shift is not cart_obj.numerical_optimization_shift
        assert np.all(new_xyz.numerical_optimization_shift.value == cart_obj.numerical_optimization_shift.value)

    def test_copy_dont_link_to_same_nos_and_array(self, cart_obj):
        shifted_coords = np.array([[1000880.456, 208800.534, 0], [1000880, 208800, -234]])
        expected = self.cls(shifted_coords)
        new_xyz = cart_obj.copy(shifted_coords, link_to_same_NOS=False)
        assert new_xyz.numerical_optimization_shift is not cart_obj.numerical_optimization_shift
        assert np.all(new_xyz.numerical_optimization_shift.value != expected.numerical_optimization_shift.value)

    @staticmethod
    def test_merge_single(cart_obj):
        merged = type(cart_obj).merge(cart_obj)
        assert len(merged) == len(cart_obj)
        assert np.all(merged == cart_obj)
        assert merged.arr is not cart_obj.arr

    def test_merge_multiple_same_nos(self):
        a = self.cls(np.ones((2, 3)) * 2)
        b = self.cls(np.ones((2, 3)) * 8)
        c = self.cls(np.ones((2, 3)) * -5)
        assert np.all(a.numerical_optimization_shift.value == b.numerical_optimization_shift.value)
        assert np.all(a.numerical_optimization_shift.value == c.numerical_optimization_shift.value)

        merged = self.cls.merge(a, b, c)
        assert np.all(merged.numerical_optimization_shift.value == a.numerical_optimization_shift.value)

        assert len(merged) == 6
        assert merged.shape == (6, 3)
        assert np.all(merged[[0, 1]] == 2)
        assert np.all(merged[[2, 3]] == 8)
        assert np.all(merged[[4, 5]] == -5)

    def test_merge_multiple_different_nos(self):
        # TODO see comment on merge method
        a = self.cls([[2, 2, 2], [2, 2, 2]], numerical_optimization_shift=OptimizedShift(np.array([1, 1, 1])))
        b = self.cls([[8, 8, 8], [8, 8, 8]], numerical_optimization_shift=OptimizedShift(np.array([8, 8, 8])))
        c = self.cls(
            [[-5, -5, -5], [-5, -5, -5]], numerical_optimization_shift=OptimizedShift(np.array([1000, 1000, 1000]))
        )

        assert np.all(a.numerical_optimization_shift.value != b.numerical_optimization_shift.value)
        assert np.all(a.numerical_optimization_shift.value != c.numerical_optimization_shift.value)

        merged = self.cls.merge(a, b, c)
        assert merged.numerical_optimization_shift is a.numerical_optimization_shift
        assert len(merged) == 6
        assert np.all(merged[[0, 1]] + a.numerical_optimization_shift.value == 2)
        assert np.all(merged[[2, 3]] + a.numerical_optimization_shift.value == 8)
        assert np.all(merged[[4, 5]] + a.numerical_optimization_shift.value == -5)

    def test_merge_different_nos_recomputed(self):
        a = self.cls(
            np.array([[0, 0, 0], [1, 1, 1]]),
            numerical_optimization_shift=OptimizedShift(np.array([-9000, -9000, -9000])),
        )
        b = self.cls(
            np.array([[0, 0, 0], [1, 1, 1]]) + 5000,
            numerical_optimization_shift=OptimizedShift(np.array([14_000, 14_000, 14_000])),
        )

        a_shift = copy.deepcopy(a.numerical_optimization_shift)
        b_shift = copy.deepcopy(b.numerical_optimization_shift)

        c = self.cls.merge(a, b)

        assert np.all(c.numerical_optimization_shift.value != a_shift.value)
        assert np.all(c.numerical_optimization_shift.value != b_shift.value)

    def test_merge_pcds_without_nos(self):
        a = self.cls(np.array([[0, 0, 0], [1, 1, 1]]), numerical_optimization_shift=None)
        b = self.cls(np.array([[5000, 2000, 1000], [2540, 123.2, 999.9]]), numerical_optimization_shift=None)

        c = self.cls.merge(a, b)
        assert c.numerical_optimization_shift is None
        assert np.all(c[[0, 1]] == a.arr)
        assert np.all(c[[2, 3]] == b.arr)

    def test_merge_incompatible_nos(self):
        a = self.cls(np.ones((2, 3)) * 123_456.789)
        b = self.cls(np.ones((2, 3)) * -123_456.789)

        assert a.numerical_optimization_shift is not None
        assert b.numerical_optimization_shift is not None
        assert a.numerical_optimization_shift is not b.numerical_optimization_shift

        merged = self.cls.merge(a, b)

        assert merged.numerical_optimization_shift is None
        assert np.allclose(merged[[0, 1]], a + a.numerical_optimization_shift)
        assert np.allclose(merged[[2, 3]], b + b.numerical_optimization_shift)
        assert merged.dtype == np.float64

    def test_merge_invalid_empty(self):
        with pytest.raises(ValueError):
            self.cls.merge()

    @staticmethod
    def test_numerically_optimized(cart_obj):
        cart_obj.numerical_optimization_shift = OptimizedShift(np.array([1, 1, 1]))
        assert cart_obj.numerically_optimized
        cart_obj.numerical_optimization_shift = None
        assert not cart_obj.numerically_optimized
        cart_obj.numerical_optimization_shift = OptimizedShift(np.array([0, 0, 0]))
        assert not cart_obj.numerically_optimized
        cart_obj.numerical_optimization_shift = OptimizedShift(np.array([1e-10, 1e-10, 1e-10]))
        assert not cart_obj.numerically_optimized

    @staticmethod
    def test_cached_properties(cart_obj):
        assert "spher" not in cart_obj.__dict__
        num_items_before = len(cart_obj.__dict__)

        _ = cart_obj.r
        num_items_after = len(cart_obj.__dict__)

        assert "spher" in cart_obj.__dict__
        assert num_items_after > num_items_before

        # spher should not propogate to copide object
        copy_obj = cart_obj.copy()
        assert "spher" not in copy_obj.__dict__

        # Ensure the deletion returns to original
        del cart_obj.__dict__["spher"]
        num_items_after = len(cart_obj.__dict__)
        assert num_items_after == num_items_before

    @staticmethod
    def test_cartesian_properties(cart_obj):
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

    @staticmethod
    def test_spherical_properties(cart_obj):
        assert id(cart_obj.arr) != id(cart_obj.spher)
        assert id(cart_obj.rhv) == id(cart_obj.spher)
        assert np.all(cart_obj.r == cart_obj.spher[:, 0])
        assert np.all(cart_obj.hz == cart_obj.spher[:, 1])
        assert np.all(cart_obj.v == cart_obj.spher[:, 2])
        assert np.all(cart_obj._hz_v == cart_obj.spher[:, 1:])

    def test_spher_w_nos_and_socs_combined(self):
        coords = np.random.rand(100, 3) + 100_000
        pcd0 = self.cls(coords, numerical_optimization_shift=None)
        pcd1 = self.cls(coords)
        pcd2 = self.cls(coords, numerical_optimization_shift=OptimizedShift([200, 200, 200]))
        pcd3 = self.cls(coords + 10, socs_origin=[10, 10, 10])
        pcd4 = self.cls(
            coords + 10, socs_origin=[10, 10, 10], numerical_optimization_shift=OptimizedShift([200, 200, 200])
        )
        pcd5 = self.cls(coords + 10, socs_origin=[10, 10, 10], numerical_optimization_shift=None)
        pcd6 = pcd4.copy()
        pcd6.numerical_optimization_shift = OptimizedShift([100, 100, 100])

        assert np.allclose(pcd0.spher, pcd1.spher)
        assert np.allclose(pcd1.spher, pcd2.spher)
        assert np.allclose(pcd2.spher, pcd3.spher)
        assert np.allclose(pcd3.spher, pcd4.spher)
        assert np.allclose(pcd4.spher, pcd5.spher)
        assert np.allclose(pcd5.spher, pcd6.spher)

    @staticmethod
    def test_fov(cart_obj):
        assert isinstance(cart_obj.fov, FoV)

    def test_from_spherical(self, cart_obj):
        # This is based on the implementation when there is no SphericalCoordinates
        cart_2 = type(cart_obj).from_spherical(cart_obj.spher)
        assert isinstance(cart_2, self.cls)

        if cart_obj.dtype == np.float64 and cart_obj.spher.dtype == np.float64:
            assert np.allclose(cart_obj, cart_2)
        elif cart_obj.dtype == np.float32:
            assert np.allclose(cart_2, cart_obj, rtol=1e-5, atol=1e-6)

    @staticmethod
    def test_rotate(cart_obj):
        xyz2 = type(cart_obj)(arr=cart_obj.copy())
        rot_forward = Rotation.from_euler(seq="zyx", angles=[90, 45, 30], degrees=True).as_matrix()
        xyz2.rotate(rot_forward)
        temp = type(xyz2)(arr=xyz2.copy())
        temp.rotate(np.linalg.inv(rot_forward))
        assert np.allclose(cart_obj, temp, atol=1e-6)
        assert not np.any(xyz2 == cart_obj)

    @staticmethod
    def test_translate(cart_obj):
        xyz2 = type(cart_obj)(arr=cart_obj)
        xyz2.translate(np.array([3, 3, 3]))
        assert np.allclose(cart_obj + 3, xyz2)

    @staticmethod
    def test_scale(cart_obj):
        xyz2 = type(cart_obj)(arr=cart_obj)
        xyz2.scale(np.array([3, 3, 3]))
        assert np.allclose(cart_obj * 3, xyz2)

    @staticmethod
    def test_transform(cart_obj):
        xyz2 = type(cart_obj)(arr=cart_obj)
        affine = np.eye(4) * 2
        affine[:3, 3] = 1
        xyz2.transform(affine)
        assert np.allclose(cart_obj * 2 + 1, xyz2)

    @staticmethod
    def test_homogeneous_matrix_multiplication(cart_obj):
        rand_4x4 = np.random.rand(4, 4)
        # Will not work with numpy arrays due to the left matrix multiplication approach
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

    def test_deepcopy(self, pcd_shifted):
        pcd_copy = pcd_shifted.copy(deep=True, link_to_same_NOS=True)
        assert isinstance(pcd_copy, self.cls)
        assert id(pcd_shifted.numerical_optimization_shift) == id(pcd_copy.numerical_optimization_shift)
        assert np.allclose(pcd_copy.xyz, pcd_shifted.xyz)

        # TODO: Rework the deepcopy of NOS
        pcd_copy = pcd_shifted.copy(deep=True, link_to_same_NOS=False)
        assert id(pcd_shifted.numerical_optimization_shift) != id(pcd_copy.numerical_optimization_shift)
        assert np.allclose(pcd_copy.xyz, pcd_shifted.xyz)

    def test_pickle(self, pcd_shifted):
        pickle_pcd = pickle.dumps(pcd_shifted)
        unpickled_pcd = pickle.loads(pickle_pcd)

        assert isinstance(unpickled_pcd, self.cls)
        assert np.allclose(unpickled_pcd.xyz, pcd_shifted.xyz)
        assert unpickled_pcd.id == pcd_shifted.id
        assert unpickled_pcd in pcd_shifted.numerical_optimization_shift


class BaseTestNOSInit:
    cls = CartesianCoordinates

    def test_defaults_to_float32_and_zero_shift(self, xyz_):
        pcd = self.cls(xyz=xyz_)
        assert pcd.xyz.dtype == np.float32
        assert np.allclose(pcd.numerical_optimization_shift.value, [0, 0, 0])
        assert np.allclose(pcd.xyz, xyz_)
        assert np.allclose(pcd.unshifted_bbox.minimum, xyz_.min(axis=0))
        assert np.allclose(pcd.unshifted_bbox.maximum, xyz_.max(axis=0))

    def test_instantiation_without_nos_keeps_float64_and_none_shift(self, xyz_):
        pcd_no_nos = self.cls(xyz=xyz_, numerical_optimization_shift=None)
        assert pcd_no_nos.xyz.dtype == np.float64
        assert pcd_no_nos.numerical_optimization_shift is None
        assert np.allclose(pcd_no_nos.xyz, xyz_)
        assert np.allclose(pcd_no_nos.unshifted_bbox.minimum, xyz_.min(axis=0))
        assert np.allclose(pcd_no_nos.unshifted_bbox.maximum, xyz_.max(axis=0))

    def test_auto_shift_applied_when_points_are_large(self, xyz_local_, nos_):
        xyz_shifted = xyz_local_ + nos_.value
        pcd_nos = self.cls(xyz=xyz_shifted)
        assert pcd_nos.xyz.dtype == np.float32
        assert pcd_nos.numerical_optimization_shift is not nos_
        assert np.allclose(pcd_nos.numerical_optimization_shift.value, nos_.value)
        assert np.allclose(pcd_nos.xyz, xyz_local_)  # This only holds because xyz_shifted is very close to 0,0,0
        assert np.allclose(pcd_nos.unshifted_bbox.minimum, xyz_shifted.min(axis=0))
        assert np.allclose(pcd_nos.unshifted_bbox.maximum, xyz_shifted.max(axis=0))

    def test_predefined_shift_when_points_are_large(self, xyz_local_, nos_):
        xyz_shifted = xyz_local_ + nos_.value
        pcd_nos_expected = self.cls(xyz=xyz_shifted, numerical_optimization_shift=nos_)
        assert pcd_nos_expected.xyz.dtype == np.float32
        assert pcd_nos_expected.numerical_optimization_shift is nos_
        assert np.allclose(pcd_nos_expected.xyz, xyz_local_)
        assert np.allclose(pcd_nos_expected.unshifted_bbox.minimum, xyz_shifted.min(axis=0))
        assert np.allclose(pcd_nos_expected.unshifted_bbox.maximum, xyz_shifted.max(axis=0))

    def test_unsuitable_shift(self, xyz_, nos_, caplog):
        caplog.set_level(logging.DEBUG)
        pcd_nos = self.cls(xyz=xyz_ + nos_.value, numerical_optimization_shift=nos_)  # Needed to bind nos_

        pcd_nos_unexpected = self.cls(xyz=xyz_, numerical_optimization_shift=nos_)
        assert pcd_nos_unexpected.xyz.dtype == np.float32
        assert pcd_nos_unexpected.numerical_optimization_shift is not nos_
        assert "input numerical_optimization_shift was not feasible" in caplog.text
        assert np.allclose(pcd_nos_unexpected.xyz, xyz_ - pcd_nos_unexpected.numerical_optimization_shift.value)
        assert np.allclose(pcd_nos_unexpected.unshifted_bbox.minimum, xyz_.min(axis=0))
        assert np.allclose(pcd_nos_unexpected.unshifted_bbox.maximum, xyz_.max(axis=0))

    def test_unsuitable_coordinates_for_shift(self, xyz_huge_, caplog):
        caplog.set_level(logging.DEBUG)
        pcd_huge = self.cls(xyz=xyz_huge_)
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


class BaseTestNOSChange:
    cls = CartesianCoordinates

    def test_updating_from_default_to_predefined_shift_logs_and_keeps_bbox(self, xyz_local_, nos_, caplog):
        caplog.set_level(logging.DEBUG)

        pcd = self.cls(xyz=xyz_local_)
        pcd.numerical_optimization_shift = nos_
        assert "Updated shift" in caplog.text
        assert np.allclose(pcd.numerical_optimization_shift.value, [0, 0, 0])
        assert np.allclose(pcd.unshifted_bbox.minimum, xyz_local_.min(axis=0))
        assert np.allclose(pcd.unshifted_bbox.maximum, xyz_local_.max(axis=0))

    def test_applying_small_shift_after_default_instantiation_adjusts_coords(self, xyz_local_, nos_mini_):
        pcd_local = self.cls(xyz_local_)
        assert np.allclose(pcd_local.numerical_optimization_shift.value, [0, 0, 0])
        assert np.allclose(pcd_local.xyz, xyz_local_)
        assert np.allclose(pcd_local.unshifted_bbox.minimum, xyz_local_.min(axis=0))
        assert np.allclose(pcd_local.unshifted_bbox.maximum, xyz_local_.max(axis=0))

        pcd_local.numerical_optimization_shift = nos_mini_
        assert np.allclose(pcd_local.xyz, xyz_local_ - nos_mini_.value, rtol=1e-5, atol=1e-6)
        assert np.allclose(pcd_local.unshifted_bbox.minimum, xyz_local_.min(axis=0))
        assert np.allclose(pcd_local.unshifted_bbox.maximum, xyz_local_.max(axis=0))

    def test_changing_from_none_to_predefined_shift_changes_dtype_and_coords(self, xyz_local_, nos_mini_):
        # Case: changing from None to predefined shift; additional checks on change in coordinate dtype
        pcd_unshifted = self.cls(xyz=xyz_local_, numerical_optimization_shift=None)
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
        pcd_shifted = self.cls(xyz=xyz_local_, numerical_optimization_shift=nos_mini_)
        pcd_shifted.numerical_optimization_shift = None
        assert pcd_shifted.xyz.dtype == np.float64
        assert np.allclose(pcd_shifted.xyz, xyz_local_, rtol=1e-5, atol=1e-6)  # Due to the conversion to float32
        assert pcd_shifted not in nos_mini_
        assert pcd_shifted.numerical_optimization_shift is None
        assert np.allclose(pcd_shifted.unshifted_bbox.minimum, xyz_local_.min(axis=0))
        assert np.allclose(pcd_shifted.unshifted_bbox.maximum, xyz_local_.max(axis=0))

    @staticmethod
    def test_copying_predefined_shift_creates_distinct_uuid_but_same_value(nos_):
        # Case: copying predefined shift
        nos2 = copy.deepcopy(nos_)
        assert nos_.uuid != nos2.uuid
        assert np.allclose(nos_.value, nos2.value)

    def test_initial_large_points_then_apply_provided_shift_recovers_original(self, xyz_local_, nos_):
        # Case: changing from initial (large) to predefined shift
        xyz_shifted = xyz_local_ + nos_.value
        pcd_shifted = self.cls(xyz=xyz_shifted)
        pcd_shifted.numerical_optimization_shift = nos_
        assert np.allclose(pcd_shifted.xyz, xyz_local_)
        assert np.allclose(pcd_shifted.unshifted_bbox.minimum, xyz_shifted.min(axis=0))
        assert np.allclose(pcd_shifted.unshifted_bbox.maximum, xyz_shifted.max(axis=0))


class BaseTestConversions:
    cls = CartesianCoordinates

    @staticmethod
    def test_cartesian_to_spherical(known_xyz, known_spher):
        rhv = xyz2rhv(known_xyz)
        assert np.allclose(rhv, known_spher)

        xyz = rhv2xyz(known_spher)
        assert np.allclose(xyz, known_xyz)

    @staticmethod
    def test_forward_backward(small_xyz, large_xyz):
        for arr in (small_xyz, large_xyz):
            xyz2 = rhv2xyz(xyz2rhv(arr))
            assert np.allclose(xyz2, arr)

    def test_supported_types(self, known_xyz, known_spher):
        xyz = self.cls(known_xyz)
        rhv = xyz2rhv(xyz)
        assert np.allclose(rhv, known_spher)

        xyz = self.cls(xyz)
        rhv = xyz2rhv(xyz)
        assert np.allclose(rhv, known_spher)

        # Show that a translation shift is the same
        xyz += 1
        xyz.socs_origin = np.ones(3)
        rhv = xyz2rhv(xyz, xyz.socs_origin)
        assert np.allclose(rhv, known_spher)

    def test_socs_origin(self, known_xyz, known_spher):
        # Different origin but same underlying coordinates
        xyz = self.cls(arr=known_xyz)
        xyz_shift = self.cls(arr=known_xyz, socs_origin=np.ones(3))
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

        xyz_shift2 = self.cls(arr=known_xyz, socs_origin=np.array([-1, -2, -3]))

        assert np.any(xyz.spher != xyz_shift2.spher)  # Different origins should yield diff results


class TestCartesianCoordinates(BaseTestCartesianCoordinates):
    pass


class TestCartesianNOSInit(BaseTestNOSInit):
    pass


class TestCartesianNOSChange(BaseTestNOSChange):
    pass


class TestCartesianConversions(BaseTestConversions):
    pass
