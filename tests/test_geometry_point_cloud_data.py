import pytest

import numpy as np

from src.pchandler.geometry import PointCloudData

@pytest.fixture
def scale_large():
    return float(2**33)

@pytest.fixture
def scale_small():
    return float(10**3)

@pytest.fixture
def offset_large():
    return float(2**49)

@pytest.fixture
def offset_small():
    return float(10**3)

def random_coords(scale: float, offset: float) -> np.ndarray:
    xyz_base = np.random.randn(100, 3)
    return xyz_base * scale + offset

@pytest.fixture
def small_coordinates(scale_small: float, offset_small: float) -> np.ndarray:
    # this is to ensure the conversion resolves this but no shift applied
    xyz_base = np.random.randn(100, 3).astype(np.float64)
    return xyz_base * scale_small + offset_small

@pytest.fixture
def large_coordinates(scale_large, offset_large):
    xyz = random_coords(scale_large, offset_large)
    assert np.min_scalar_type(xyz) == np.float64
    return xyz

@pytest.fixture
def random_normals():
    return np.random.rand(100, 3).astype(np.float32)

@pytest.fixture
def random_intensities():
    return np.random.rand(100).astype(np.float32)

@pytest.fixture
def random_colour():
    return np.random.randint(0, 255, (100, 3), dtype=np.uint8)


class TestPointCloudData:
    @staticmethod
    def check_global_shift_need(xyz, expected: bool):
        assert PointCloudData.__dict__['_PointCloudData__check_for_need_of_global_shift'](xyz) == expected

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
        assert not getattr(pcd, '_spherical_coordinates_calculated')

    def test_initialise_global_shifted(
            self, large_coordinates, random_colour, random_normals, random_intensities, offset_large):
        xyz = large_coordinates
        rgb = random_colour
        normals = random_normals
        intensities = random_intensities

        pcd = PointCloudData(
            xyz=xyz, color=rgb, normals=normals, scalar_fields={'intensity': intensities}
        )

        self.check_global_shift_need(xyz, True)
        self.check_global_shift_applied(pcd)
        self.check_origin_exists(pcd)
        assert isinstance(pcd, PointCloudData)

        # Data should not be close after shift
        assert not np.all(np.isclose(pcd.xyz, xyz))

        # Global shift should be close to the offset
        assert np.all(np.isclose(pcd.spherical_coordinates_origin, np.array([-offset_large, -offset_large, -offset_large])))

        # 'Scalar fields' should be identical
        assert np.all(rgb == pcd.color)
        assert np.all(normals == pcd.normals)
        assert np.all(intensities == pcd.scalar_fields['intensity'])

        assert pcd.nbPoints == 100
        assert pcd.xyz.dtype == np.float32

    def test_initialise_non_shifted_cloud(
            self, small_coordinates, random_colour, random_normals, random_intensities, offset_small):
        xyz = small_coordinates
        rgb = random_colour
        normals = random_normals
        intensities = random_intensities

        pcd = PointCloudData(
            xyz=xyz, color=rgb, normals=normals, scalar_fields={'intensity': intensities}
        )


        self.check_global_shift_need(xyz, False)
        self.check_global_shift_not_applied(pcd)
        self.check_origin_exists(pcd)
        assert np.all(pcd.spherical_coordinates_origin == np.zeros(3))
        assert pcd.nbPoints == 100

        # Scalar fields
        assert np.all(rgb == pcd.color)
        assert np.all(normals == pcd.normals)
        assert np.all(intensities == pcd.scalar_fields['intensity'])

        # Points should match now
        assert np.all(np.isclose(pcd.xyz, xyz))
        assert pcd.xyz.dtype == np.float32

    def test_immutability(
            self, small_coordinates, random_colour, random_normals, random_intensities, offset_small
    ):
        xyz = small_coordinates
        rgb = random_colour
        normals = random_normals
        intensities = random_intensities

        pcd = PointCloudData(
            xyz=xyz, color=rgb, normals=normals, scalar_fields={'intensity': intensities}
        )

        with pytest.raises(AttributeError):
            pcd.xyz = np.random.rand(100,3)

        with pytest.raises(AttributeError):
            pcd.rgb = np.random.randint(0, 255, (100,3), dtype=np.uint8)

        with pytest.raises(AttributeError):
            pcd.normals = np.random.rand(100,3)

        # TODO should this be so easily overwriteable - this can be overwritten
        #  with pytest.raises(AttributeError):
        #      pcd.scalar_fields['intensity'] = np.random.rand(100)






        
