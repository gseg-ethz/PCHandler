import copy

import numpy as np
import pytest

from pchandler.v2.geometry import (
    PointCloudData,
    RGBFields,
    ScalarField,
    ScalarFieldManager,
)


def random_coordinates(scale: float, offset: float) -> np.ndarray:
    xyz_base = np.random.randn(100, 3)
    return xyz_base * scale + offset


@pytest.fixture(scope="function", autouse=True)
def xyz_() -> np.ndarray:
    return random_coordinates(10, 0)


@pytest.fixture(scope="function", autouse=True)
def normals_() -> np.ndarray:
    return np.random.rand(100, 3).astype(np.float32)


@pytest.fixture(scope="function", autouse=True)
def intensities_() -> np.ndarray:
    return np.random.rand(100).astype(np.float32)


@pytest.fixture(scope="function", autouse=True)
def rgb_():
    return np.random.randint(0, 255, (100, 3), dtype=np.uint8)


@pytest.fixture(scope="function", autouse=True)
def sfs_():
    array = np.random.rand(100)
    return {"test": array}


@pytest.fixture(scope="function")
def pcd(xyz_, rgb_, normals_, intensities_) -> PointCloudData:
    return PointCloudData(xyz_, rgb=rgb_, normals=normals_, intensity=intensities_)


class TestPointCloudData:

    class TestInitialisation:
        def test_positional_coordinate_arg(self, xyz_):
            pcd = PointCloudData(xyz_)
            assert np.all(pcd == xyz_)
            assert np.all(pcd.arr == xyz_)

        def test_keyword_coordinate_arg(self, xyz_):
            pcd = PointCloudData(xyz=xyz_)
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

        def test_intensity_keyword(self, xyz_, intensities_):
            pcd = PointCloudData(xyz_, intensity=intensities_)
            assert "intensity" in pcd.scalar_fields
            assert np.allclose(pcd.intensity, intensities_)

        def test_reflectance_keyword(self, xyz_, intensities_):
            pcd = PointCloudData(xyz_, reflectance=intensities_)
            assert "reflectance" in pcd.scalar_fields
            assert np.allclose(pcd.reflectance, intensities_)

        def test_all_scalar_fields(self, xyz_, rgb_, normals_, intensities_):
            pcd = PointCloudData(xyz_, rgb=rgb_, normals=normals_, intensity=intensities_, reflectance=intensities_)

            for name in ("rgb", "normals", "intensity", "reflectance"):
                assert name in pcd.scalar_fields

        def test_optimised_keyword(self, xyz_):
            pcd = PointCloudData(xyz_)
            assert not pcd.optimised

            pcd = PointCloudData(xyz_, optimised=True)
            assert pcd.optimised

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

            test_array = np.random.rand(100)
            pcd = PointCloudData(xyz_, scalar_fields={"test": ScalarField(test_array, name="test2")})
            assert "test" in pcd.scalar_fields
            assert np.all(pcd.scalar_fields["test"] == test_array)
            assert pcd.scalar_fields["test"].name != "test2"
            assert isinstance(pcd.scalar_fields["test"], ScalarField)

    class TestInvalidValues:
        def test_xyz(self, rgb_, normals_, sfs_):
            # XYZ non_array object
            with pytest.raises(TypeError):
                PointCloudData({"asb": 123}, rgb=rgb_, normals=normals_, sfm=sfs_)

            # Too many columns
            with pytest.raises(ValueError):
                data = np.random.rand(100, 4).astype(np.float32)
                PointCloudData(data, rgb=rgb_, normals=normals_, scalar_fields=sfs_)

            # Too many dimensions
            with pytest.raises(ValueError):
                data = np.random.rand(100, 4, 3).astype(np.float32)
                PointCloudData(data, rgb=rgb_, normals=normals_, scalar_fields=sfs_)

            # Too fed dimensions
            with pytest.raises(ValueError):
                data = np.random.rand(100, 2).astype(np.float32)
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
                assert type(e.value) in (ValueError, TypeError, AttributeError)

        def test_socs_origin(self, xyz_, rgb_, normals_, sfs_):
            # spherical_coordinates_origin
            with pytest.raises(Exception) as e:
                PointCloudData(
                    xyz=xyz_,
                    rgb=rgb_,
                    normals=normals_,
                    scalar_fields=sfs_,
                    socs_origin="NotAnOrigin",
                )
                assert type(e.value) in (ValueError, TypeError, AttributeError)

    class TestProperties:
        def test_rgb(self, pcd):
            assert isinstance(pcd.rgb, RGBFields)
            assert pcd.rgb.shape[1] == 3
            assert pcd.rgb.dtype == np.uint8

        def test_sf_input_as_scalar_fields(self, xyz_, rgb_, normals_, intensities_):
            scalar_fields = ScalarFieldManager()
            pcd = PointCloudData(xyz=xyz_, scalar_fields=scalar_fields)

            assert isinstance(pcd.scalar_fields, ScalarFieldManager)
            assert len(pcd.scalar_fields) == 0

            pcd2 = PointCloudData(xyz=xyz_, rgb=rgb_, normals=normals_, scalar_fields=scalar_fields)
            assert isinstance(pcd2.scalar_fields, ScalarFieldManager)
            assert len(pcd2.scalar_fields) == 2

    class TestReduceSampleExtract:
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
                    pcd.create_mask(bad_mask)

                assert type(e.value) in (TypeError, IndexError, ValueError)

            def test_invalid_selection_type(self, pcd):
                bad_index_type = "NotAMask"
                with pytest.raises(Exception) as e:
                    pcd.create_mask(bad_index_type)

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

            assert np.all(base_xyz[mask, :] == pcd.xyz)
            assert np.all(base_rgb[mask] == pcd.rgb)
            assert np.all(base_normal[mask] == pcd.normals)
            assert np.all(base_intensities[mask] == pcd.scalar_fields["intensity"].arr)

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

            assert np.all(base_xyz[mask, :] == new_pcd.xyz)
            assert np.all(base_rgb[mask] == new_pcd.rgb)
            assert np.all(base_normal[mask] == new_pcd.normals)
            assert np.all(base_intensities[mask] == new_pcd.scalar_fields["intensity"])
            assert pcd.optimised == new_pcd.optimised
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
            assert np.all(base_normal[~mask] == sampled_pcd.normals)
            assert np.all(base_intensities[~mask] == sampled_pcd.intensity)
            assert np.all(pcd.socs_origin == sampled_pcd.socs_origin)
            assert np.all(pcd.optimised == sampled_pcd.optimised)
            assert np.all(pcd.spher == sampled_pcd.spher)

            assert np.all(base_xyz[mask, :] == extracted_pcd.xyz)
            assert np.all(base_rgb[mask] == extracted_pcd.rgb)
            assert np.all(base_normal[mask] == extracted_pcd.normals)
            assert np.all(base_intensities[mask] == extracted_pcd.intensity)
            assert np.all(pcd.socs_origin == extracted_pcd.socs_origin)
            assert np.all(pcd.optimised == extracted_pcd.optimised)

            assert len(extracted_pcd) + len(sampled_pcd) == base_xyz.shape[0]
            assert len(extracted_pcd) == 10

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
            assert ref_pcd.optimised == extracted_pcd.optimised
            assert reduced_pcd.optimised == extracted_pcd.optimised

    def test_immutability(self, xyz_, rgb_, normals_, intensities_):
        pcd = PointCloudData(xyz=xyz_, rgb=rgb_, normals=normals_, scalar_fields={"intensity": intensities_})

        with pytest.raises(AttributeError):
            pcd.xyz = np.random.rand(100, 3)

        with pytest.raises(AttributeError):
            pcd.rgb = np.random.randint(0, 255, (100, 3), dtype=np.uint8)

        with pytest.raises(AttributeError):
            pcd.normals = np.random.rand(100, 3)

        # TODO should this be so easily overwriteable? Should a setter function be defined to go with?
        #  Habit says yes so it's more interoperable with other code E/g/ Pcd.xyz, Pcd.rgb, Pcd.intensity, Pcd.normals
        # with pytest.raises(AttributeError):
        #     pcd.scalar_fields['intensity'] = np.random.rand(100)
