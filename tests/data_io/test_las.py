import pytest

from pathlib import Path

import numpy as np

from pchandler.v2.data_io.las import LasHandler


class TestLasHandler:
    rgb_file = Path(r"D:\Python\pchandler\tests\data\test_target_intensity_normals_rgb.las")

    def test_load(self):
        pcd = LasHandler.load(self.rgb_file)
        assert len(pcd) == 43577
        assert 'intensity' in pcd.scalar_fields
        assert 'normals' in pcd.scalar_fields
        assert 'rgb' in pcd.scalar_fields

    def test_save(self):
        out_path = Path(r"D:\Python\pchandler\tests\data\test_target_rgb_temp.las")
        original_pcd = LasHandler.load(self.rgb_file)
        LasHandler.save(original_pcd, out_path)
        new_pcd = LasHandler.load(out_path)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz, atol=0.0001)
        assert np.allclose(original_pcd.normals, new_pcd.normals, atol=0.0001)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields