import pytest
from pathlib import Path

import numpy as np

from pchandler.v2.data_io.ply import PlyHandler

base_directory = Path(__file__).resolve().parent

FILE_BINARY = base_directory / ".." / "data" / "test_target_intensity_normals_rgb.ply"
FILE_ASCII = base_directory / ".." / "data" / "test_target_intensity_normals_rgb_ascii.ply"
FILE_SCALAR_PREFIXED = base_directory / ".." / "data" / "2019July_Refl-10to0_rot_extract.ply"

class TestPlyHandler:
    out_path = base_directory / ".." / "data" / "test_target_rgb_temp.ply"

    @pytest.mark.parametrize('file', (FILE_BINARY, FILE_ASCII))
    def test_load(self, file):
        pcd = PlyHandler.load(file)
        assert len(pcd) == 43577
        assert 'rgb' in pcd.scalar_fields
        assert 'normals' in pcd.scalar_fields
        assert 'intensity' not in pcd.scalar_fields
        assert len(pcd.scalar_fields) == 3

    @pytest.mark.parametrize('file', (FILE_BINARY, FILE_ASCII))
    def test_save(self, file):
        original_pcd = PlyHandler.load(file,)
        PlyHandler.save(original_pcd, self.out_path)
        new_pcd = PlyHandler.load(self.out_path)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz)
        assert np.allclose(original_pcd.rgb, new_pcd.rgb)
        assert np.allclose(original_pcd.normals, new_pcd.normals)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields

    def test_scalar_prefix(self):
        pcd = PlyHandler.load(FILE_SCALAR_PREFIXED)
        assert hasattr(pcd, 'intensity')
