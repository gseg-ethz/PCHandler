import copy

import numpy as np
import pytest

from src.pchandler.v2.geometry.core import PointCloudData
from src.pchandler.v2.geometry.scalar_field_manager import ScalarFieldManager


@pytest.fixture(scope="session", autouse=True)
def scale_large() -> float:
    return float(2**33)


@pytest.fixture(scope="session", autouse=True)
def scale_small() -> float:
    return float(10**3)


@pytest.fixture(scope="session", autouse=True)
def offset_large() -> float:
    return float(2**49)


@pytest.fixture(scope="session", autouse=True)
def offset_small() -> float:
    return float(10**3)


def random_coords(scale: float, offset: float) -> np.ndarray:
    xyz_base = np.random.randn(100, 3)
    return xyz_base * scale + offset


@pytest.fixture(scope="session", autouse=True)
def small_coordinates(scale_small: float, offset_small: float) -> np.ndarray:
    # this is to ensure the conversion resolves this but no shift applied
    xyz_base = np.random.randn(100, 3).astype(np.float64)
    return xyz_base * scale_small + offset_small


@pytest.fixture(scope="session", autouse=True)
def large_coordinates(scale_large, offset_large) -> np.ndarray:
    xyz = random_coords(scale_large, offset_large)
    assert np.min_scalar_type(xyz) == np.float64
    return xyz


@pytest.fixture(scope="session", autouse=True)
def normals() -> np.ndarray:
    return np.random.rand(100, 3).astype(np.float32)


@pytest.fixture(scope="session", autouse=True)
def intensities() -> np.ndarray:
    return np.random.rand(100).astype(np.float32)


@pytest.fixture(scope="session", autouse=True)
def scalar_fields(intensities) -> dict:
    return {"intensity": intensities}


@pytest.fixture(scope="session", autouse=True)
def rgb():
    return np.random.randint(0, 255, (100, 3), dtype=np.uint8)


@pytest.fixture(scope="function")
def pcd(small_coordinates, rgb, normals, scalar_fields, intensities):
    return PointCloudData(xyz=small_coordinates, rgb=rgb, normals=normals, scalar_fields=scalar_fields)

# TODO need to define a common name for colours / intensities to get called
@pytest.fixture(scope="function")
def pcd_shifted(large_coordinates, rgb, normals, scalar_fields, intensities):
    return PointCloudData(xyz=large_coordinates, rgb=rgb, normals=normals, scalar_fields=scalar_fields)


class TestPointCloudData:

    class TestInitialisation:
        @staticmethod
        def check_global_shift_need(xyz, expected: bool):
            assert PointCloudData._needs_global_shift(xyz) == expected

        @staticmethod
        def check_global_shift_not_applied(pcd: PointCloudData):
            assert pcd.global_coordinate_shift is None or (
                isinstance(pcd.global_coordinate_shift, np.ndarray) and np.all(pcd.global_coordinate_shift == 0)
            )

        @staticmethod
        def check_global_shift_applied(pcd: PointCloudData):
            assert pcd.global_coordinate_shift is not None
            assert isinstance(pcd.global_coordinate_shift, np.ndarray)
            assert np.any(pcd.global_coordinate_shift != 0)

        @staticmethod
        def check_origin_exists(pcd: PointCloudData):
            assert isinstance(pcd.spherical_coordinates_origin, np.ndarray) == True

        @staticmethod
        def check_spherical_coordinates_calculated(pcd: PointCloudData):
            assert not getattr(pcd, "_spherical_coordinates_calculated")

        def test_global_shifted(self, large_coordinates, rgb, normals, intensities, offset_large):
            xyz = large_coordinates
            pcd = PointCloudData(xyz=xyz, rgb=rgb, normals=normals, intensity=intensities)

            self.check_global_shift_need(xyz, True)
            self.check_global_shift_applied(pcd)
            self.check_origin_exists(pcd)
            assert isinstance(pcd, PointCloudData)

            # Data should not be close after shift
            assert not np.all(np.isclose(pcd.xyz, xyz))

            # Global shift should be close to the offset
            assert np.all(
                np.isclose(pcd.spherical_coordinates_origin, np.array([-offset_large, -offset_large, -offset_large]))
            )

            # 'Scalar fields' should be identical
            assert np.all(rgb == pcd.color)
            assert np.all(normals == pcd.normals)
            assert np.all(intensities == pcd.scalar_fields["intensity"])

            assert pcd.nbPoints == 100
            assert pcd.xyz.dtype == np.float32

        def test_non_shifted_cloud(self, small_coordinates, rgb, normals, intensities, offset_small):
            xyz = small_coordinates
            pcd = PointCloudData(xyz=xyz, color=rgb, normals=normals, scalar_fields={"intensity": intensities})

            self.check_global_shift_need(xyz, False)
            self.check_global_shift_not_applied(pcd)
            self.check_origin_exists(pcd)
            assert np.all(pcd.spherical_coordinates_origin == np.zeros(3))
            assert pcd.nbPoints == 100

            # Scalar fields
            assert np.all(rgb == pcd.color)
            assert np.all(normals == pcd.normals)
            assert np.all(intensities == pcd.scalar_fields["intensity"])

            # Points should match now
            assert np.all(np.isclose(pcd.xyz, xyz))
            assert pcd.xyz.dtype == np.float32

    class TestInvalidValues:
        def test_xyz(self, rgb, normals, scalar_fields):
            # XYZ non_array object
            with pytest.raises(TypeError):
                PointCloudData(xyz={"asb": 123}, color=rgb, normals=normals, scalar_fields=scalar_fields)

            # Too many columns
            with pytest.raises(ValueError):
                PointCloudData(
                    xyz=np.random.rand(100, 4).astype(np.float32),
                    color=rgb,
                    normals=normals,
                    scalar_fields=scalar_fields,
                )

            # Too many dimensions
            with pytest.raises(ValueError):
                PointCloudData(
                    xyz=np.random.rand(100, 4, 3).astype(np.float32),
                    color=rgb,
                    normals=normals,
                    scalar_fields=scalar_fields,
                )

            # Too fed dimensions
            with pytest.raises(ValueError):
                PointCloudData(
                    xyz=np.random.rand(100, 2).astype(np.float32),
                    color=rgb,
                    normals=normals,
                    scalar_fields=scalar_fields,
                )

        def test_color(self, small_coordinates, normals, scalar_fields):
            # color non_array
            with pytest.raises(TypeError):
                PointCloudData(
                    xyz=small_coordinates, color="NotAnNdarray", normals=normals, scalar_fields=scalar_fields
                )

        def test_normals(self, small_coordinates, rgb, scalar_fields):
            # normals non_array
            with pytest.raises(TypeError):
                PointCloudData(xyz=small_coordinates, color=rgb, normals=1, scalar_fields=scalar_fields)

        def test_global_coordinate_shift(self, small_coordinates, rgb, normals, scalar_fields):
            # global_coordinate_shift
            with pytest.raises(TypeError):
                PointCloudData(
                    xyz=small_coordinates,
                    color=rgb,
                    normals=normals,
                    scalar_fields=scalar_fields,
                    global_coordinate_shift="NotAShift",
                )

        def test_spherica_coordinate_origin(self, small_coordinates, rgb, normals, scalar_fields):
            # spherical_coordinates_origin
            with pytest.raises(TypeError):
                PointCloudData(
                    xyz=small_coordinates,
                    color=rgb,
                    normals=normals,
                    scalar_fields=scalar_fields,
                    spherical_coordinates_origin="NotAnOrigin",
                )

    class TestProperties:
        def test_color(self, pcd):
            assert isinstance(pcd.color, np.ndarray)
            assert pcd.color.shape[1] == 3
            assert pcd.color.dtype == np.uint8

        def test_sf_input_as_vector(self, small_coordinates, rgb, normals, intensities):
            pcd = PointCloudData(xyz=small_coordinates, color=rgb, normals=normals, scalar_fields=intensities)

            assert pcd.scalar_fields["scalar_fields"] is not None
            assert np.all(pcd.scalar_fields["scalar_fields"] == intensities)
            assert isinstance(pcd.scalar_fields, ScalarFieldManager)
            assert len(pcd.scalar_fields) == 1

        def test_sf_input_as_triplet(self, small_coordinates, rgb, normals, intensities):
            with pytest.raises(TypeError, ValueError, IndexError):
                pcd = PointCloudData(xyz=small_coordinates, color=rgb, normals=normals, scalar_fields=rgb)

            assert isinstance(pcd.scalar_fields, ScalarFieldManager)
            assert pcd.scalar_fields["scalar_fields"] is not None
            assert len(pcd.scalar_fields) == 1

        def test_sf_input_as_sfm(self, small_coordinates, rgb, normals, intensities):
            sfm = ScalarFieldManager(expected_length=small_coordinates.shape[0])
            pcd = PointCloudData(xyz=small_coordinates, color=rgb, normals=normals, scalar_fields=sfm)

            assert isinstance(pcd.scalar_fields, ScalarFieldManager)
            assert len(pcd.scalar_fields) == 0

    class TestReduceSampleExtract:
        class TestMaskGeneration:
            def test_ndarray(self, pcd):
                mask = pcd._convert_indexing_to_mask(np.arange(0, 10))
                assert mask.ndim == 1
                assert mask.shape[0] == pcd.nbPoints
                assert mask.dtype == np.bool_

            def test_indices(self, pcd):
                mask = pcd._convert_indexing_to_mask([0, 1, 3])

                assert mask.ndim == 1
                assert mask.shape[0] == pcd.nbPoints
                assert mask.dtype == np.bool_

                other_mask = np.zeros_like(mask, dtype=np.bool_)
                other_mask[0] = True
                other_mask[1] = True
                other_mask[3] = True

                assert np.all(mask == other_mask)

            def test_slice(self, pcd):
                mask = pcd._convert_indexing_to_mask(slice(1, 6, 2))

                assert mask.ndim == 1
                assert mask.shape[0] == pcd.nbPoints
                assert mask.dtype == np.bool_
                assert mask[1] == True
                assert mask[3] == True
                assert mask[5] == True

            def test_mask(self, pcd):
                mask_in = np.zeros(pcd.nbPoints, dtype=np.bool_)
                mask_in[0::2] = True
                mask = pcd._convert_indexing_to_mask(mask_in)

                assert mask.ndim == 1
                assert mask.shape[0] == pcd.nbPoints
                assert mask.dtype == np.bool_
                assert np.all(mask_in == mask)

            def test_invalid_bool_shape(self, pcd):
                bad_mask = np.zeros(55, dtype=np.bool_)
                with pytest.raises(ValueError):
                    pcd._convert_indexing_to_mask(bad_mask)

            def test_invalid_float_array(self, pcd):
                bad_mask = np.random.rand(pcd.nbPoints)
                with pytest.raises(TypeError):
                    pcd._convert_indexing_to_mask(bad_mask)

            def test_invalid_selection_type(self, pcd):
                bad_index_type = "NotAMask"
                with pytest.raises(TypeError):
                    pcd._convert_indexing_to_mask(bad_index_type)

        def test_reduce(self, pcd):
            start_id = id(pcd)
            base_xyz = pcd.xyz.copy()
            base_rgb = pcd.color.copy()
            base_normal = pcd.normals.copy()
            base_intensities = copy.deepcopy(pcd.scalar_fields["intensity"])

            mask = np.zeros(pcd.nbPoints, dtype=np.bool_)
            mask[0:10] = True

            pcd.reduce(mask)

            end_id = id(pcd)

            # check if same object
            assert start_id == end_id

            assert np.all(base_xyz[mask, :] == pcd.xyz)
            assert np.all(base_rgb[mask, :] == pcd.color)
            assert np.all(base_normal[mask, :] == pcd.normals)
            assert np.all(base_intensities[mask] == pcd.scalar_fields["intensity"].data)

        def test_sample(self, pcd_shifted):
            a = pcd_shifted.spherical_coordinates
            start_id = id(pcd_shifted)
            base_xyz = pcd_shifted.xyz.copy()
            base_rgb = pcd_shifted.color.copy()
            base_normal = pcd_shifted.normals.copy()
            base_intensities = copy.deepcopy(pcd_shifted.scalar_fields["intensity"])

            mask = np.zeros(pcd_shifted.nbPoints, dtype=np.bool_)
            mask[0:10] = True

            new_pcd = pcd_shifted.sample(mask)

            end_id = id(new_pcd)

            # Ensure completely different objects
            assert start_id != end_id
            assert id(pcd_shifted.xyz) != id(new_pcd.xyz)
            assert id(pcd_shifted.color) != id(new_pcd.color)
            assert id(pcd_shifted.normals) != id(new_pcd.normals)
            assert id(pcd_shifted.scalar_fields) != id(new_pcd.scalar_fields)
            assert id(pcd_shifted.spherical_coordinates_origin) != id(new_pcd.spherical_coordinates_origin)
            assert id(pcd_shifted.global_coordinate_shift) != id(new_pcd.global_coordinate_shift)
            assert id(pcd_shifted.spherical_coordinates) != id(new_pcd.spherical_coordinates)

            assert np.all(base_xyz[mask, :] == new_pcd.xyz)
            assert np.all(base_rgb[mask, :] == new_pcd.color)
            assert np.all(base_normal[mask, :] == new_pcd.normals)
            assert np.all(base_intensities[mask] == new_pcd.scalar_fields["intensity"].data)
            assert np.all(pcd_shifted.global_coordinate_shift == new_pcd.global_coordinate_shift)
            assert np.all(pcd_shifted.spherical_coordinates_origin == new_pcd.spherical_coordinates_origin)
            assert np.all(pcd_shifted.spherical_coordinates == new_pcd.spherical_coordinates)

        def test_extract(self, pcd_shifted):
            a = pcd_shifted.spherical_coordinates
            start_id = id(pcd_shifted)
            base_xyz = pcd_shifted.xyz.copy()
            base_rgb = pcd_shifted.color.copy()
            base_normal = pcd_shifted.normals.copy()
            base_intensities = copy.deepcopy(pcd_shifted.scalar_fields["intensity"])

            mask = np.zeros(pcd_shifted.nbPoints, dtype=np.bool_)
            mask[0:10] = True

            sampled_pcd = pcd_shifted.sample(~mask)
            extracted_pcd = pcd_shifted.extract(mask)

            end_id = id(extracted_pcd)

            # Ensure completely different objects
            assert start_id != end_id
            assert id(pcd_shifted.xyz) != id(sampled_pcd.xyz)
            assert id(pcd_shifted.color) != id(sampled_pcd.color)
            assert id(pcd_shifted.normals) != id(sampled_pcd.normals)
            assert id(pcd_shifted.scalar_fields) != id(sampled_pcd.scalar_fields)
            assert id(pcd_shifted.spherical_coordinates_origin) != id(sampled_pcd.spherical_coordinates_origin)
            assert id(pcd_shifted.global_coordinate_shift) != id(sampled_pcd.global_coordinate_shift)
            assert id(pcd_shifted.spherical_coordinates) != id(sampled_pcd.spherical_coordinates)

            assert np.all(base_xyz[~mask, :] == sampled_pcd.xyz)
            assert np.all(base_rgb[~mask, :] == sampled_pcd.color)
            assert np.all(base_normal[~mask, :] == sampled_pcd.normals)
            assert np.all(base_intensities[~mask] == sampled_pcd.scalar_fields["intensity"].data)
            assert np.all(pcd_shifted.global_coordinate_shift == sampled_pcd.global_coordinate_shift)
            assert np.all(pcd_shifted.spherical_coordinates_origin == sampled_pcd.spherical_coordinates_origin)
            assert np.all(pcd_shifted.spherical_coordinates == sampled_pcd.spherical_coordinates)

            assert np.all(base_xyz[mask, :] == extracted_pcd.xyz)
            assert np.all(base_rgb[mask, :] == extracted_pcd.color)
            assert np.all(base_normal[mask, :] == extracted_pcd.normals)
            assert np.all(base_intensities[mask] == extracted_pcd.scalar_fields["intensity"].data)
            assert np.all(pcd_shifted.global_coordinate_shift == extracted_pcd.global_coordinate_shift)
            assert np.all(pcd_shifted.spherical_coordinates_origin == extracted_pcd.spherical_coordinates_origin)
            assert np.all(pcd_shifted.spherical_coordinates == extracted_pcd.spherical_coordinates)

            assert extracted_pcd.nbPoints + sampled_pcd.nbPoints == base_xyz.shape[0]
            assert extracted_pcd.nbPoints == 10

        def test_none_fields(self, small_coordinates):
            ref_pcd = PointCloudData(small_coordinates, scalar_fields=ScalarFieldManager())
            this_pcd = ref_pcd.copy()

            mask = np.zeros(this_pcd.nbPoints, dtype=np.bool_)
            mask[0:10] = True

            sample_pcd = ref_pcd.sample(mask)
            # this_pc becomes reduced after the extraction
            extracted_pcd = this_pcd.extract(mask)
            reduced_pcd = this_pcd

            assert id(reduced_pcd) == id(this_pcd)
            assert id(extracted_pcd) != id(ref_pcd)
            assert id(this_pcd) != id(ref_pcd)
            assert id(sample_pcd) != id(ref_pcd)

            assert np.all(sample_pcd.nbPoints == extracted_pcd.nbPoints)
            assert np.all(reduced_pcd.spherical_coordinates_origin == extracted_pcd.spherical_coordinates_origin)
            assert np.all(ref_pcd.spherical_coordinates_origin == extracted_pcd.spherical_coordinates_origin)

            # These should be None
            assert ref_pcd.global_coordinate_shift == extracted_pcd.global_coordinate_shift
            assert reduced_pcd.global_coordinate_shift == extracted_pcd.global_coordinate_shift

    def test_immutability(self, small_coordinates, rgb, normals, intensities, offset_small):
        xyz = small_coordinates
        rgb = rgb
        normals = normals
        intensities = intensities

        pcd = PointCloudData(xyz=xyz, color=rgb, normals=normals, scalar_fields={"intensity": intensities})

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
