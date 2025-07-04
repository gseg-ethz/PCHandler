import pytest
from pathlib import Path

import numpy as np

from pchandler.v2.data_io.ply import PlyHandler

base_directory = Path(__file__).resolve().parent

FILE_BINARY = base_directory / ".." / "data" / "test_target_intensity_normals_rgb.ply"
FILE_ASCII = base_directory / ".." / "data" / "test_target_intensity_normals_rgb_ascii.ply"


class TestPlyHandler:
    out_path = base_directory / ".." / "data" / "test_target_rgb_temp.ply"

    @pytest.mark.parametrize('file', (FILE_BINARY, FILE_ASCII))
    def test_load(self, file):
        pcd = PlyHandler.load(file, keep_rgb=False, keep_normals=False, keep_intensity=False)
        assert len(pcd) == 43577
        assert 'rgb' not in pcd.scalar_fields
        assert 'normals' not in pcd.scalar_fields
        assert 'intensity' not in pcd.scalar_fields
        assert len(pcd.scalar_fields) == 0

    @pytest.mark.parametrize('file', (FILE_BINARY, FILE_ASCII))
    def test_save(self, file):
        original_pcd = PlyHandler.load(file, keep_rgb=True, keep_normals=True, keep_intensity=True)
        PlyHandler.save(original_pcd, self.out_path, keep_rgb=True, keep_normals=True, keep_intensity=True)
        new_pcd = PlyHandler.load(self.out_path, keep_rgb=True, keep_normals=True, keep_intensity=True)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz)
        assert np.allclose(original_pcd.rgb, new_pcd.rgb)
        assert np.allclose(original_pcd.normals, new_pcd.normals)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields