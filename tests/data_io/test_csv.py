import pytest

from pathlib import Path

import numpy as np

from pchandler.v2.data_io.csv import CsvHandler

base_directory = Path(__file__).resolve().parent

class TestCsvHandler:
    rgb_file = base_directory / ".." / "data" / "test_target_intensity_normals_rgbfloat.txt"
    out_path = base_directory / ".." / "data" / "test_target_rgb_temp.csv"

    def test_load(self):
        pcd = CsvHandler.load(self.rgb_file)
        assert len(pcd) == 43577
        assert 'rgb' in pcd.scalar_fields
        assert pcd.rgb.dtype == np.uint8

    def test_save(self):
        original_pcd = CsvHandler.load(self.rgb_file)
        CsvHandler.save(original_pcd, self.out_path)
        new_pcd = CsvHandler.load(self.out_path)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz)
        assert np.allclose(original_pcd.rgb, new_pcd.rgb)

        assert np.allclose(original_pcd.normals, new_pcd.normals)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields