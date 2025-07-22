import copy
import logging
import pickle

import numpy as np
import pytest

import open3d as o3d
from pydantic import ValidationError
from pchandler.geometry import PointCloudData
from pchandler.geometry.scalar_fields import ScalarField, RGBFields, NormalFields
from pchandler.geometry.scalar_field_manager import ScalarFieldManager
from pchandler.geometry.optimal_shift import OptimizedShift

N = 100

def random_coordinates(scale: float, offset: float) -> np.ndarray:
    xyz_base = np.random.randn(N, 3)
    return xyz_base * scale + offset


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

@pytest.fixture(scope="function", autouse=True)
def normals_() -> np.ndarray:
    return np.random.rand(N, 3).astype(np.float32)


@pytest.fixture(scope="function", autouse=True)
def intensity_() -> np.ndarray:
    return np.random.rand(N).astype(np.float32)


@pytest.fixture(scope="function", autouse=True)
def reflectance_() -> np.ndarray:
    return np.random.rand(N).astype(np.float32)


@pytest.fixture(scope="function", autouse=True)
def scalar_field() -> ScalarField:
    return ScalarField(np.random.rand(N), name='test')


@pytest.fixture(scope="function", autouse=True)
def rgb_():
    return np.random.randint(0, 255, (N, 3), dtype=np.uint8)


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

@pytest.fixture(scope="function")
def pcd(rgb_, normals_, intensity_, reflectance_) -> PointCloudData:
    return PointCloudData(
        xyz=random_coordinates(1, 0),
        rgb=rgb_,
        normals=normals_,
        intensity=intensity_,
        reflectance=reflectance_)


@pytest.fixture(scope="function")
def pcd2(rgb_, normals_, intensity_) -> PointCloudData:
    return PointCloudData(
        xyz=random_coordinates(1, 0),
        rgb=rgb_,
        normals=normals_,
        intensity=intensity_)


@pytest.fixture(scope="function")
def pcd3(rgb_, normals_, reflectance_) -> PointCloudData:
    return PointCloudData(
        xyz=random_coordinates(1, 0),
        rgb=rgb_,
        normals=normals_,
        reflectance=reflectance_)

@pytest.fixture(scope="function")
def pcd_shifted(rgb_, normals_, reflectance_, nos_) -> PointCloudData:
    xyz = random_coordinates(1, 0)
    xyz += nos_.value
    return PointCloudData(
        xyz=xyz,
        rgb=rgb_,
        normals=normals_,
        reflectance=reflectance_,
        numerical_optimization_shift= nos_,
    )

@pytest.fixture(scope="function")
def pcd_o3d():
    xyz = np.random.rand(10,3)
    rgb = np.ones((10,3))
    normals = np.random.rand(10,3)
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


@pytest.fixture(scope="function")
def pcd_o3d_tensor():
    xyz = (np.random.rand(10, 3) -0.5) * 50 + 20
    rgb = np.ones((10, 3)).astype(np.float32)
    intensity = np.random.rand(10) * 255
    normals = np.random.rand(10, 3)
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(xyz)
    pcd.point.rgb = o3d.core.Tensor(rgb)
    pcd.point.normals = o3d.core.Tensor(normals)
    pcd.point.intensity = o3d.core.Tensor(intensity)

    return pcd


class TestPointCloudData:

    class TestInitialisation:
        def test_positional_coordinate_arg(self, xyz_):
            pcd = PointCloudData(xyz_, numerical_optimization_shift=None)
            assert np.all(pcd == xyz_)
            assert np.all(pcd.arr == xyz_)

        def test_alias_coordinate_arg(self, xyz_):
            pcd = PointCloudData(xyz=xyz_, numerical_optimization_shift=None)
            assert np.all(pcd == xyz_)
            assert np.all(pcd.arr == xyz_)
            pcd2 = PointCloudData(arr=xyz_, numerical_optimization_shift=None)
            assert np.all(pcd2 == xyz_)
            assert np.all(pcd2.arr == xyz_)

        def test_keyword_coordinate_arg(self, xyz_):
            pcd = PointCloudData(xyz=xyz_, numerical_optimization_shift=None)
            assert np.all(pcd == xyz_)
            assert np.all(pcd.arr == xyz_)

        def test_rgb_keyword(self, xyz_, rgb_):
            pcd = PointCloudData(xyz_, rgb=rgb_)
            assert "rgb" in pcd.scalar_fields
            assert np.allclose(pcd.rgb, rgb_)

        def test_normals_keyword(self, xyz_, normals_):
            pcd = PointCloudData(xyz_, normals=normals_)
            assert "normals" in pcd.scalar_fields
            assert np.allclose(pcd.normals, normals_)

        def test_intensity_keyword(self, xyz_, intensity_):
            pcd = PointCloudData(xyz_, intensity=intensity_)
            vals = ScalarField(intensity_, name='test')
            assert "intensity" in pcd.scalar_fields
            assert np.allclose(pcd.intensity, vals)

        def test_reflectance_keyword(self, xyz_, intensity_):
            pcd = PointCloudData(xyz_, reflectance=intensity_)

            vals = ScalarField(intensity_, name='test')
            assert "reflectance" in pcd.scalar_fields
            assert np.allclose(pcd.reflectance, vals)

        def test_all_scalar_fields(self, xyz_, rgb_, normals_, intensity_):
            pcd = PointCloudData(xyz_, rgb=rgb_, normals=normals_, intensity=intensity_, reflectance=intensity_)

            for name in ("rgb", "normals", "intensity", "reflectance"):
                assert name in pcd.scalar_fields

        # TODO: Discuss what the optimized flag should represent
        # def test_optimized_keyword(self, xyz_):
        #     pcd = PointCloudData(xyz_)
        #     assert not pcd.optimized

            # pcd = PointCloudData(xyz_, numerical_optimization_shift=OptimizedShift())
            # assert pcd.optimized is not None

        def test_socs_origin_keyword(self, xyz_):
            pcd = PointCloudData(xyz_)
            assert pcd.socs_origin is None

            pcd = PointCloudData(xyz_, socs_origin=None)
            assert pcd.socs_origin is None

            pcd = PointCloudData(xyz_, socs_origin=np.ones(3))
            assert np.allclose(pcd.socs_origin, np.ones(3))

        def test_scalar_fields(self, xyz_, sfs_):
            pcd = PointCloudData(xyz_)
            assert pcd.scalar_fields == {}

            pcd = PointCloudData(xyz_, scalar_fields=sfs_)
            assert "test" in pcd.scalar_fields
            assert np.all(pcd.scalar_fields["test"] == sfs_["test"])
            assert isinstance(pcd.scalar_fields["test"], ScalarField)

            test_array = np.random.rand(N)
            pcd = PointCloudData(xyz_, scalar_fields={"test": ScalarField(test_array, name="test2")})
            assert "test" in pcd.scalar_fields
            assert np.all(pcd.scalar_fields["test"] == test_array)
            assert pcd.scalar_fields["test"].name != "test2"
            assert isinstance(pcd.scalar_fields["test"], ScalarField)


    class TestInvalidValues:
        def test_xyz(self, rgb_, normals_, sfs_):
            # XYZ non_array object
            with pytest.raises(TypeError):
                PointCloudData({"asb": 123}, rgb=rgb_, normals=normals_, scalar_fields=sfs_)

            # Too many columns
            with pytest.raises(ValueError):
                data = np.random.rand(N, 4).astype(np.float32)
                PointCloudData(data, rgb=rgb_, normals=normals_, scalar_fields=sfs_)

            # Too many dimensions
            with pytest.raises(ValueError):
                data = np.random.rand(N, 4, 3).astype(np.float32)
                PointCloudData(data, rgb=rgb_, normals=normals_, scalar_fields=sfs_)

            # Too fed dimensions
            with pytest.raises(ValueError):
                data = np.random.rand(N, 2).astype(np.float32)
                PointCloudData(data, rgb=rgb_, normals=normals_, scalar_fields=sfs_)

        def test_rgb(self, xyz_, normals_, sfs_):
            # rgb non_array
            with pytest.raises(Exception) as e:
                PointCloudData(xyz=xyz_, rgb="NotAnNdarray", normals=normals_, scalar_fields=sfs_)
                assert type(e.value) in (ValueError, TypeError, AttributeError)

        def test_normals(self, xyz_, rgb_, sfs_):
            # normals non_array
            with pytest.raises(Exception) as e:
                PointCloudData(xyz=xyz_, rgb=rgb_, normals=1, scalar_fields=sfs_)
            assert type(e.value) in (ValueError, TypeError, AttributeError, ValidationError)

        def test_intensity(self, xyz_, rgb_, normals_, reflectance_, sfs_):
            # normals non_array
            with pytest.raises(Exception) as e:
                PointCloudData(xyz=xyz_, rgb=rgb_, normals=normals_, reflectance=reflectance_, intensity=1, scalar_fields=sfs_)
            assert type(e.value) in (ValueError, TypeError, AttributeError)

        def test_reflectance(self, xyz_, rgb_, normals_, intensity_, sfs_):
            # normals non_array
            with pytest.raises(Exception) as e:
                PointCloudData(xyz=xyz_, rgb=rgb_, normals=normals_, reflectance=1, intensity=intensity_, scalar_fields=sfs_)
            assert type(e.value) in (ValueError, TypeError, AttributeError)

        def test_socs_origin(self, xyz_, rgb_, normals_, sfs_):
            # spherical_coordinates_origin
            with pytest.raises(Exception) as e:
                PointCloudData(
                    xyz=xyz_,
                    rgb=rgb_,
                    normals=normals_,
                    scalar_fields=sfs_,
                    socs_origin="NotAnOrigin",  #type: ignore
                )
            assert type(e.value) in (ValueError, TypeError, AttributeError, ValidationError)

    class TestProperties:
        def test_rgb_getter(self, pcd):
            assert isinstance(pcd.rgb, RGBFields)
            assert pcd.rgb.shape[1] == 3
            assert pcd.rgb.dtype == np.uint8

        def test_normals_getter(self, pcd):
            assert isinstance(pcd.normals, NormalFields)
            assert pcd.normals.shape[1] == 3
            assert pcd.normals.dtype == np.float32

        def test_intensity_getter(self, pcd):
            assert isinstance(pcd.intensity, ScalarField)
            assert pcd.intensity.ndim == 1
            assert pcd.intensity.dtype == np.float32

        def test_reflectance_getter(self, pcd):
            assert isinstance(pcd.reflectance, ScalarField)
            assert pcd.reflectance.ndim == 1
            assert pcd.reflectance.dtype == np.float32

        def test_rgb_setter(self, pcd):
            new_data = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
            pcd.rgb = new_data
            assert isinstance(pcd.rgb, RGBFields)
            assert pcd.rgb.shape[1] == 3
            assert pcd.rgb.dtype == np.uint8
            assert np.all(pcd.rgb == new_data)

        def test_normals_setter(self, pcd):
            new_data = np.random.rand(N, 3).astype(np.float32)
            pcd.normals = new_data
            assert isinstance(pcd.normals, NormalFields)
            assert pcd.normals.shape[1] == 3
            assert pcd.normals.dtype == np.float32
            assert np.all(pcd.normals == new_data / np.linalg.norm(new_data, axis=1).reshape(-1, 1))

        def test_intensity_setter(self, pcd):
            new_data = np.random.rand(N).astype(np.float32)
            pcd.intensity = new_data
            assert isinstance(pcd.intensity, ScalarField)
            assert pcd.intensity.ndim == 1
            assert pcd.intensity.dtype == np.float32
            assert np.allclose(pcd.intensity.get_original_data(), new_data, atol=1/2**16)

        def test_reflectance_setter(self, pcd):
            new_data = np.random.rand(N)
            pcd.reflectance = new_data.astype(np.float32)
            assert isinstance(pcd.reflectance, ScalarField)
            assert pcd.reflectance.ndim == 1
            assert pcd.reflectance.dtype == np.float32
            assert np.allclose(pcd.reflectance.get_original_data(), new_data, atol=1/2**16)

        def test_sf_input_as_scalar_fields(self, xyz_, rgb_, normals_):
            scalar_fields = ScalarFieldManager()
            pcd = PointCloudData(xyz=xyz_, scalar_fields=scalar_fields)

            assert isinstance(pcd.scalar_fields, ScalarFieldManager)
            assert len(pcd.scalar_fields) == 0

            pcd2 = PointCloudData(xyz=xyz_, rgb=rgb_, normals=normals_, scalar_fields=scalar_fields)
            assert isinstance(pcd2.scalar_fields, ScalarFieldManager)
            assert len(pcd2.scalar_fields) == 2

    class TestNOS:
        class TestNOSInstantiation:
            def test_defaults_to_float32_and_zero_shift(self, xyz_):
                pcd = PointCloudData(xyz=xyz_)
                assert pcd.xyz.dtype == np.float32
                assert np.allclose(pcd.numerical_optimization_shift.value, [0, 0, 0])
                assert np.allclose(pcd.xyz, xyz_)
                assert np.allclose(pcd.unshifted_bbox.minimum, xyz_.min(axis=0))
                assert np.allclose(pcd.unshifted_bbox.maximum, xyz_.max(axis=0))

            def test_instantiation_without_nos_keeps_float64_and_none_shift(self, xyz_):
                pcd_no_nos = PointCloudData(xyz=xyz_, numerical_optimization_shift=None)
                assert pcd_no_nos.xyz.dtype == np.float64
                assert pcd_no_nos.numerical_optimization_shift is None
                assert np.allclose(pcd_no_nos.xyz, xyz_)
                assert np.allclose(pcd_no_nos.unshifted_bbox.minimum, xyz_.min(axis=0))
                assert np.allclose(pcd_no_nos.unshifted_bbox.maximum, xyz_.max(axis=0))

            def test_auto_shift_applied_when_points_are_large(self, xyz_local_, nos_):
                xyz_shifted = xyz_local_ + nos_.value
                pcd_nos = PointCloudData(xyz=xyz_shifted)
                assert pcd_nos.xyz.dtype == np.float32
                assert pcd_nos.numerical_optimization_shift is not nos_
                assert np.allclose(pcd_nos.numerical_optimization_shift.value, nos_.value)
                assert np.allclose(pcd_nos.xyz, xyz_local_) # This only holds because xyz_shifted is very close to 0,0,0
                assert np.allclose(pcd_nos.unshifted_bbox.minimum, xyz_shifted.min(axis=0))
                assert np.allclose(pcd_nos.unshifted_bbox.maximum, xyz_shifted.max(axis=0))

            def test_predefined_shift_when_points_are_large(self, xyz_local_, nos_):
                xyz_shifted = xyz_local_ + nos_.value
                pcd_nos_expected = PointCloudData(xyz=xyz_shifted, numerical_optimization_shift=nos_)
                assert pcd_nos_expected.xyz.dtype == np.float32
                assert pcd_nos_expected.numerical_optimization_shift is nos_
                assert np.allclose(pcd_nos_expected.xyz, xyz_local_)
                assert np.allclose(pcd_nos_expected.unshifted_bbox.minimum, xyz_shifted.min(axis=0))
                assert np.allclose(pcd_nos_expected.unshifted_bbox.maximum, xyz_shifted.max(axis=0))

            def test_unsuitable_shift(self, xyz_, nos_, caplog):
                caplog.set_level(logging.DEBUG)
                pcd_nos = PointCloudData(xyz=xyz_ + nos_.value, numerical_optimization_shift=nos_) #Needed to bind nos_

                pcd_nos_unexpected = PointCloudData(xyz=xyz_, numerical_optimization_shift=nos_)
                assert pcd_nos_unexpected.xyz.dtype == np.float32
                assert pcd_nos_unexpected.numerical_optimization_shift is not nos_
                assert "provided numerical_optimization_shift was not feasible" in caplog.text
                assert np.allclose(pcd_nos_unexpected.xyz, xyz_ - pcd_nos_unexpected.numerical_optimization_shift.value)
                assert np.allclose(pcd_nos_unexpected.unshifted_bbox.minimum, xyz_.min(axis=0))
                assert np.allclose(pcd_nos_unexpected.unshifted_bbox.maximum, xyz_.max(axis=0))


            def test_unsuitable_coordinates_for_shift(self, xyz_huge_, caplog):
                caplog.set_level(logging.DEBUG)
                pcd_huge = PointCloudData(xyz=xyz_huge_)
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

                pcd = PointCloudData(xyz=xyz_local_)
                pcd.numerical_optimization_shift = nos_
                assert "Updating shift" in caplog.text
                assert np.allclose(pcd.numerical_optimization_shift.value, [0,0,0])
                assert np.allclose(pcd.unshifted_bbox.minimum, xyz_local_.min(axis=0))
                assert np.allclose(pcd.unshifted_bbox.maximum, xyz_local_.max(axis=0))


            def test_applying_small_shift_after_default_instantiation_adjusts_coords(self, xyz_local_, nos_mini_):

                pcd_local = PointCloudData(xyz=xyz_local_)
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
                pcd_unshifted = PointCloudData(xyz=xyz_local_, numerical_optimization_shift=None)
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
                pcd_shifted = PointCloudData(xyz=xyz_local_, numerical_optimization_shift=nos_mini_)
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
                pcd_shifted = PointCloudData(xyz=xyz_shifted)
                pcd_shifted.numerical_optimization_shift = nos_
                assert np.allclose(pcd_shifted.xyz, xyz_local_)
                assert np.allclose(pcd_shifted.unshifted_bbox.minimum, xyz_shifted.min(axis=0))
                assert np.allclose(pcd_shifted.unshifted_bbox.maximum, xyz_shifted.max(axis=0))


    class TestCopy:
        def test_deepcopy(self, pcd_shifted):
            pcd_copy = pcd_shifted.copy(deep=True, link_to_same_NOS=True)
            assert isinstance(pcd_copy, PointCloudData)
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

            assert isinstance(unpickled_pcd, PointCloudData)
            assert np.allclose(unpickled_pcd.xyz, pcd_shifted.xyz)
            assert unpickled_pcd.id == pcd_shifted.id
            assert unpickled_pcd in pcd_shifted.numerical_optimization_shift



    class TestReduceSampleExtractMerge:
        class TestMaskGeneration:
            def test_ndarray(self, pcd):
                mask = pcd.create_mask(np.arange(0, 10))
                assert mask.ndim == 1
                assert mask.shape[0] == len(pcd)
                assert mask.dtype == np.bool_

            def test_indices(self, pcd):
                mask = pcd.create_mask([0, 1, 3])

                assert mask.ndim == 1
                assert mask.shape[0] == len(pcd)
                assert mask.dtype == np.bool_

                other_mask = np.zeros_like(mask, dtype=np.bool_)
                other_mask[0] = True
                other_mask[1] = True
                other_mask[3] = True

                assert np.all(mask == other_mask)

            def test_slice(self, pcd):
                mask = pcd.create_mask(slice(1, 6, 2))

                assert mask.ndim == 1
                assert mask.shape[0] == len(pcd)
                assert mask.dtype == np.bool_
                assert mask[1] == True
                assert mask[3] == True
                assert mask[5] == True

            def test_mask(self, pcd):
                mask_in = np.zeros(len(pcd), dtype=np.bool_)
                mask_in[0::2] = True
                mask = pcd.create_mask(mask_in)

                assert mask.ndim == 1
                assert mask.shape[0] == len(pcd)
                assert mask.dtype == np.bool_
                assert np.all(mask_in == mask)

            def test_invalid_bool_shape(self, pcd):
                bad_mask = np.zeros(55, dtype=np.bool_)
                with pytest.raises(Exception) as e:
                    pcd.create_mask(bad_mask)

                assert type(e.value) in (TypeError, IndexError, ValueError)

            def test_invalid_float_array(self, pcd):
                bad_mask = np.random.rand(len(pcd))
                with pytest.raises(Exception) as e:
                    pcd.create_mask(bad_mask)   #type: ignore

                assert type(e.value) in (TypeError, IndexError, ValueError)

            def test_invalid_selection_type(self, pcd):
                bad_index_type = "NotAMask"
                with pytest.raises(Exception) as e:
                    pcd.create_mask(bad_index_type) #type: ignore

                assert type(e.value) in (TypeError, IndexError, ValueError)

        def test_reduce(self, pcd):
            start_id = id(pcd)
            base_xyz = pcd.xyz.copy()
            base_rgb = pcd.rgb.copy()
            base_normal = pcd.normals.copy()
            base_intensities = copy.deepcopy(pcd.scalar_fields["intensity"])

            mask = np.zeros(len(pcd), dtype=np.bool_)
            mask[0:10] = True

            pcd.reduce(mask)

            end_id = id(pcd)

            # check if same object
            assert start_id == end_id

            assert np.allclose(base_xyz[mask, :], pcd.xyz)
            assert np.allclose(base_rgb[mask], pcd.rgb)
            assert np.allclose(base_normal[mask], pcd.normals)
            assert np.allclose(base_intensities[mask], pcd.scalar_fields["intensity"].arr)

        def test_sample(self, pcd):
            start_id = id(pcd)
            base_xyz = pcd.xyz.copy()
            base_rgb = pcd.rgb.copy()
            base_normal = pcd.normals.copy()
            base_intensities = copy.deepcopy(pcd.scalar_fields["intensity"])

            mask = np.zeros(len(pcd), dtype=np.bool_)
            mask[0:10] = True

            new_pcd = pcd.sample(mask)

            end_id = id(new_pcd)

            # Ensure completely different objects
            assert start_id != end_id
            assert id(pcd.xyz) != id(new_pcd.xyz)
            assert id(pcd.rgb) != id(new_pcd.rgb)
            assert id(pcd.normals) != id(new_pcd.normals)
            assert id(pcd.scalar_fields) != id(new_pcd.scalar_fields)
            assert id(pcd.spher) != id(new_pcd.spher)

            assert np.allclose(base_xyz[mask, :], new_pcd.xyz)
            assert np.allclose(base_rgb[mask], new_pcd.rgb)
            assert np.allclose(base_normal[mask], new_pcd.normals)
            assert np.allclose(base_intensities[mask], new_pcd.scalar_fields["intensity"])
            # assert pcd.optimized == new_pcd.optimized
            if isinstance(pcd.socs_origin, np.ndarray):
                assert np.all(pcd.socs_origin == new_pcd.socs_origin)
            else:
                assert pcd.socs_origin == new_pcd.socs_origin
            assert np.any(pcd.spher[mask, :] == new_pcd.spher)

        def test_extract(self, pcd):
            start_id = id(pcd)
            base_xyz = pcd.xyz.copy()
            base_rgb = pcd.rgb.copy()
            base_normal = pcd.normals.copy()
            base_intensities = copy.deepcopy(pcd.scalar_fields["intensity"])

            mask = np.zeros(len(pcd), dtype=np.bool_)
            mask[0:10] = True

            sampled_pcd = pcd.sample(~mask)
            extracted_pcd = pcd.extract(mask)

            end_id = id(extracted_pcd)

            # Ensure completely different objects
            assert start_id != end_id
            assert id(pcd.xyz) != id(sampled_pcd.xyz)
            assert id(pcd.rgb) != id(sampled_pcd.rgb)
            assert id(pcd.normals) != id(sampled_pcd.normals)
            assert id(pcd.scalar_fields) != id(sampled_pcd.scalar_fields)
            assert id(pcd.spher) != id(sampled_pcd.spher)

            assert np.all(base_xyz[~mask, :] == sampled_pcd.xyz)
            assert np.all(base_rgb[~mask] == sampled_pcd.rgb)
            assert np.allclose(base_normal[~mask], sampled_pcd.normals)
            assert np.all(base_intensities[~mask] == sampled_pcd.intensity)
            assert np.all(pcd.socs_origin == sampled_pcd.socs_origin)
            # assert np.all(pcd.optimized == sampled_pcd.optimized)
            assert np.all(pcd.spher == sampled_pcd.spher)

            assert np.all(base_xyz[mask, :] == extracted_pcd.xyz)
            assert np.all(base_rgb[mask] == extracted_pcd.rgb)
            assert np.all(base_normal[mask] == extracted_pcd.normals)
            assert np.all(base_intensities[mask] == extracted_pcd.intensity)
            assert np.all(pcd.socs_origin == extracted_pcd.socs_origin)
            # assert np.all(pcd.optimized == extracted_pcd.optimized)

            assert len(extracted_pcd) + len(sampled_pcd) == base_xyz.shape[0]
            assert len(extracted_pcd) == 10

        def test_extract_one_element(self, pcd):
            mask = np.zeros(len(pcd), dtype=np.bool_)
            mask[0] = True

            pcd_extract = pcd.extract(mask)
            assert pcd_extract.arr.shape == (1, 3)
            assert pcd_extract.rgb.shape == (1, 3)
            assert pcd_extract.normals.shape == (1, 3)
            assert pcd_extract.intensity.shape == (1,)


        def test_none_fields(self, xyz_):
            ref_pcd = PointCloudData(xyz_, scalar_fields=ScalarFieldManager(parent=None))
            this_pcd = ref_pcd.copy()

            mask = np.zeros(len(this_pcd), dtype=np.bool_)
            mask[0:10] = True

            sample_pcd = ref_pcd.sample(mask)
            # this_pc becomes reduced after the extraction
            extracted_pcd = this_pcd.extract(mask)
            reduced_pcd = this_pcd

            assert id(reduced_pcd) == id(this_pcd)
            assert id(extracted_pcd) != id(ref_pcd)
            assert id(this_pcd) != id(ref_pcd)
            assert id(sample_pcd) != id(ref_pcd)

            assert np.all(len(sample_pcd) == len(extracted_pcd))
            assert np.all(reduced_pcd.socs_origin == extracted_pcd.socs_origin)
            assert np.all(ref_pcd.socs_origin == extracted_pcd.socs_origin)

            # These should be None
            # assert ref_pcd.optimized == extracted_pcd.optimized
            # assert reduced_pcd.optimized == extracted_pcd.optimized

        def test_merge(self, pcd, pcd2, pcd3):
            merged_1 = PointCloudData.merge(pcd, pcd2, pcd3)

            assert len(merged_1) == len(pcd) + len(pcd2) + len(pcd3)

            assert 'rgb' in merged_1.scalar_fields
            assert 'normals' in merged_1.scalar_fields
            assert 'intensity' not in merged_1.scalar_fields
            assert 'reflectance' not in merged_1.scalar_fields

    # def test_immutability(self, xyz_, rgb_, normals_, intensity_):
    #     pcd = PointCloudData(xyz=xyz_, rgb=rgb_, normals=normals_, scalar_fields={"intensity": intensity_}, frozen=True)
    #
    #     with pytest.raises(Exception) as e:
    #         pcd.xyz = np.random.rand(N, 3)
    #
    #     assert type(e.value) in (AttributeError, ValidationError, TypeError, ValueError)
    #
    #     # TODO add future feature to have a READ ONLY point cloud object where no fields or attributes can be changed


class TestOpen3DSupport:
    def test_to_o3d(self, pcd: PointCloudData) -> None:
        obj = pcd.to_o3d(as_tensor=False)
        assert isinstance(obj, o3d.geometry.PointCloud)
        for name in ('points', 'colors', 'normals'):
            assert hasattr(obj, name)

        assert np.allclose(np.asarray(obj.points), pcd.xyz)
        assert np.allclose(np.asarray(obj.normals), pcd.normals)
        assert np.allclose(np.asarray(obj.colors), pcd.rgb.as_normalised_float32())

        assert not hasattr(obj, 'intensity')
        assert not hasattr(obj, 'reflectance')

    def test_to_o3d_tensor(self, pcd: PointCloudData) -> None:
        obj = pcd.to_o3d(as_tensor=True)

        assert isinstance(obj, o3d.t.geometry.PointCloud)
        assert np.allclose(pcd.xyz, obj.point.positions.numpy())
        for attr in ('normals', 'rgb', 'reflectance', 'intensity'): # Todo: Reflectance doesn't work right
            assert getattr(pcd, attr) is None or hasattr(obj.point, attr)
            assert getattr(pcd, attr) is None or np.allclose(getattr(pcd, attr), getattr(obj.point, attr).numpy())

    def test_from_o3d(self, pcd_o3d: o3d.geometry.PointCloud):
        pcd = PointCloudData.from_o3d(pcd_o3d)
        assert isinstance(pcd, PointCloudData)
        assert pcd.intensity is None
        assert pcd.reflectance is None
        assert hasattr(pcd, 'rgb')
        assert hasattr(pcd, 'normals')

        assert np.allclose(pcd.rgb, 255)
        assert np.allclose(pcd.xyz, np.asarray(pcd_o3d.points))
        assert np.allclose(pcd.normals, np.asarray(pcd_o3d.normals))

    def test_from_o3d_tensor(self, pcd_o3d_tensor: o3d.t.geometry.PointCloud) -> None:
        pcd = PointCloudData.from_o3d(pcd_o3d_tensor)
        assert isinstance(pcd, PointCloudData)
        assert hasattr(pcd, 'intensity')
        assert hasattr(pcd, 'reflectance')
        assert hasattr(pcd, 'rgb')
        assert hasattr(pcd, 'normals')

        assert np.allclose(pcd.rgb, 255)
        assert pcd.intensity.max() <= 2**15
        assert pcd.intensity.min() >= -2**15
        assert np.allclose(pcd.intensity.get_original_data(), pcd_o3d_tensor.point.intensity.numpy(), atol=1/255)
        assert np.allclose(pcd.xyz, pcd_o3d_tensor.point.positions.numpy())
        assert np.allclose(pcd.normals, pcd_o3d_tensor.point.normals.numpy())