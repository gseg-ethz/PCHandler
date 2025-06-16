from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

import numpy as np

from pchandler.v2.data_io import load_e57, load_ply, load_csv, load_laz, save_ply, save_csv

class TestPlyHandler:
    file = Path(r"D:\Python\pchandler\tests\data\test_target_scalar_field.ply")
    rgb_file = Path(r"D:\Python\pchandler\tests\data\test_target_rgb.ply")


    def test_load(self):
        pcd = load_ply(self.file, retain_colors=True, retain_normals=True)
        assert len(pcd) == 43577
        assert 'rgb' not in pcd.scalar_fields

        pcd = load_ply(self.rgb_file, retain_colors=True, retain_normals=True)
        assert len(pcd) == 43577
        assert 'rgb' in pcd.scalar_fields

    def test_save(self):
        out_path = Path(r"D:\Python\pchandler\tests\data\test_target_rgb_temp.ply")
        original_pcd = load_ply(self.rgb_file, retain_colors=True, retain_normals=True)
        save_ply(out_path, original_pcd)
        new_pcd = load_ply(out_path, retain_colors=True, retain_normals=True)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz)
        assert np.allclose(original_pcd.rgb, new_pcd.rgb)
        assert np.allclose(original_pcd.normals, new_pcd.normals)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields


class TestCsvHandler:
    file = Path(r"D:\Python\pchandler\tests\data\test_target_scalar_field.csv")
    rgb_file = Path(r"D:\Python\pchandler\tests\data\test_target_rgb.csv")


    def test_load(self):
        pcd = load_csv(self.rgb_file)
        assert len(pcd) == 43577
        assert 'rgb' in pcd.scalar_fields

    def test_save(self):
        out_path = Path(r"D:\Python\pchandler\tests\data\test_target_rgb_temp.csv")
        original_pcd = load_csv(self.rgb_file)
        save_csv(out_path, original_pcd)
        new_pcd = load_csv(out_path)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz)
        assert np.allclose(original_pcd.rgb, new_pcd.rgb)
        assert np.allclose(original_pcd.normals, new_pcd.normals)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields