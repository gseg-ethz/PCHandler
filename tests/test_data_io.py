from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

import numpy as np

from pchandler.v2.data_io.ply import PlyHandler

class TestPlyHandler:
    file = Path(r"D:\Python\pchandler\tests\data\test_target_scalar_field.ply")
    rgb_file = Path(r"D:\Python\pchandler\tests\data\test_target_rgb.ply")


    def test_load(self):
        pcd = PlyHandler.load(self.file, keep_rgb=False, keep_normals=True)
        assert len(pcd) == 43577
        assert 'rgb' not in pcd.scalar_fields
        assert 'normals' in pcd.scalar_fields
        assert len(pcd.scalar_fields) == 1

        pcd = PlyHandler.load(self.rgb_file, keep_rgb=True, keep_normals=True)
        assert len(pcd) == 43577
        assert 'rgb' in pcd.scalar_fields

    def test_save(self):
        out_path = Path(r"D:\Python\pchandler\tests\data\test_target_rgb_temp.ply")
        original_pcd = PlyHandler.load(self.rgb_file, keep_rgb=True, keep_normals=True)
        PlyHandler.save(original_pcd, out_path, keep_rgb=True, keep_normals=True)
        new_pcd = PlyHandler.load(out_path, keep_rgb=True, keep_normals=True)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz)
        assert np.allclose(original_pcd.rgb, new_pcd.rgb)
        assert np.allclose(original_pcd.normals, new_pcd.normals)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields
#
#
# class TestCsvHandler:
#     file = Path(r"D:\Python\pchandler\tests\data\test_target_scalar_field.csv")
#     rgb_file = Path(r"D:\Python\pchandler\tests\data\test_target_rgb.csv")
#
#
#     def test_load(self):
#         pcd = load_csv(self.rgb_file)
#         assert len(pcd) == 43577
#         assert 'rgb' in pcd.scalar_fields
#
#     def test_save(self):
#         out_path = Path(r"D:\Python\pchandler\tests\data\test_target_rgb_temp.csv")
#         original_pcd = load_csv(self.rgb_file)
#         save_csv(out_path, original_pcd)
#         new_pcd = load_csv(out_path)
#
#         assert np.allclose(original_pcd.xyz, new_pcd.xyz)
#         assert np.allclose(original_pcd.rgb, new_pcd.rgb)
#         assert np.allclose(original_pcd.normals, new_pcd.normals)
#
#         for name in original_pcd.scalar_fields.keys():
#             assert name in new_pcd.scalar_fields