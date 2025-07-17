import pytest

from pathlib import Path

import numpy as np

from pchandler.v2.geometry import PointCloudData
from pchandler.v2.data_io import Las as LAS

base_directory = Path(__file__).resolve().parent

@pytest.fixture(scope='function')
def pcd():
    return PointCloudData(
        np.random.rand(1000, 3) * 100,
        rgb=np.random.randint(0, 256, (1000, 3), dtype=np.uint8),
        intensity=np.random.randint(0, 256, (1000,), dtype=np.uint8),
        reflectance=np.random.randint(0, 256, (1000,), dtype=np.uint8),
        normals=np.random.rand(1000, 3).astype(np.float32),
    )


class TestLasHandler:
    rgb_file = base_directory / ".." / "data" / "test_target_intensity_normals_rgb.las"
    out_path = base_directory / ".." / "data" / "test_target_rgb_temp.las"

    def test_load(self):
        pcd = LAS.load(self.rgb_file)
        assert len(pcd) == 43577
        assert 'intensity' in pcd.scalar_fields
        assert 'normals' in pcd.scalar_fields
        assert 'rgb' in pcd.scalar_fields

    def test_save(self):
        original_pcd = LAS.load(self.rgb_file)
        LAS.save(self.out_path,original_pcd)
        new_pcd = LAS.load(self.out_path)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz, atol=0.0001)
        assert np.allclose(original_pcd.normals, new_pcd.normals, atol=0.0001)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields