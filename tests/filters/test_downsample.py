import pytest

from pydantic import ValidationError

from pchandler.v2.filters.downsample import *

N = 100000

@pytest.fixture(scope='function', autouse=True)
def pcd_all():
    xyz = np.random.rand(N,3)
    rgb = np.random.randint(0, 256, (N, 3), dtype=np.uint8)
    normals = np.random.rand(N, 3)
    intensity = np.random.randint(0, 1000, (N,), dtype=np.uint16)
    pcd = PointCloudData(xyz, rgb=rgb, normals=normals, intensity=intensity)
    return pcd


@pytest.fixture(scope='function', autouse=True)
def pcd_only_coords():
    return PointCloudData(np.random.rand(100,3))

@pytest.fixture(scope="function")
def random_downsample_filter():
    return RandomDownsampleFilter(0.5)

@pytest.fixture(scope="function")
def voxel_downsample():
    return VoxelDownsample(0.1, 'constant')

@pytest.fixture(scope="function")
def angle_bin_downsample():
    return AngleBinDownsample(0.1, 'linear')


class TestRandomDownSampleFilter:
    def test_init(self, random_downsample_filter):
        assert hasattr(random_downsample_filter, "size")
        assert hasattr(RandomDownsampleFilter, "sample")
        assert np.all(random_downsample_filter.size == 0.5)

    @pytest.mark.parametrize('size', (1.2, -1, 1000, False))
    def test_invalid_init(self, random_downsample_filter, size):
        with pytest.raises(ValueError):
            RandomDownsampleFilter(size)


    def test_mask(self, random_downsample_filter, pcd_all):
        expected_size = int(np.round(0.5*len(pcd_all)))
        mask = random_downsample_filter.mask(pcd_all)

        assert mask.shape == (pcd_all.shape[0],)
        assert mask.dtype == np.bool_
        assert isinstance(mask, np.ndarray)

        assert np.sum(mask) == expected_size


class TestVoxelDownsampleFilter:
    def test_init(self, voxel_downsample):
        assert hasattr(voxel_downsample, "voxel_size")
        assert hasattr(voxel_downsample, "weighting_method")
        assert hasattr(VoxelDownsample, "sample")
        assert np.all(voxel_downsample.voxel_size == 0.1)
        assert np.all(voxel_downsample.weighting_method == "constant")

    def test_invalid_init(self, voxel_downsample):
        with pytest.raises(TypeError):
            voxel_downsample('abc')

        with pytest.raises(ValidationError):
            VoxelDownsample(0.1, 'asdasd')

        with pytest.raises(ValueError):
            VoxelDownsample(0, 'constant')

        with pytest.raises(ValueError):
            VoxelDownsample(-1.3, 'constant')

        with pytest.raises(ValueError):
            VoxelDownsample(1.3, True)

    def test_mask(self, voxel_downsample, pcd_all):
        expected_size = ((1 / voxel_downsample.voxel_size)+1) ** 3
        pcd = voxel_downsample.sample(pcd_all)

        assert 1331 == expected_size

        assert len(pcd) == expected_size

        assert isinstance(pcd, PointCloudData)

        # TODO test for spacing between voxel centers


class TestAngleBinDownsample:
    def test_init(self, angle_bin_downsample):
        assert hasattr(angle_bin_downsample, "angle_bin_size")
        assert hasattr(angle_bin_downsample, "weighting_method")
        assert hasattr(AngleBinDownsample, "sample")
        assert np.all(angle_bin_downsample.angle_bin_size == 0.1)
        assert np.all(angle_bin_downsample.weighting_method == "linear")

    def test_invalid_init(self, angle_bin_downsample):
        with pytest.raises(TypeError):
            angle_bin_downsample('abc')

        with pytest.raises(ValidationError):
            AngleBinDownsample(0.1, 'asdasd')

        with pytest.raises(ValueError):
            AngleBinDownsample(0, 'constant')

        with pytest.raises(ValueError):
            AngleBinDownsample(-1.3, 'constant')

        with pytest.raises(ValueError):
            AngleBinDownsample(1.3, True)

    def test_mask(self, angle_bin_downsample, pcd_all):
        pcd = angle_bin_downsample.sample(pcd_all)

        raise NotImplementedError("Need to finish angle bin tests - e.g. meshgrid values")