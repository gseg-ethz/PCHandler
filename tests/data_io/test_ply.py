import pytest
from pathlib import Path

import numpy as np

from pchandler.v2.data_io.ply import PlyHandler

base_directory = Path(__file__).resolve().parent

FILE_BINARY = base_directory / ".." / "data" / "test_target_intensity_normals_rgb.ply"
FILE_ASCII = base_directory / ".." / "data" / "test_target_intensity_normals_rgb_ascii.ply"
FILE_SCALAR_PREFIXED = base_directory / ".." / "data" / "2019July_Refl-10to0_rot_extract.ply"

class TestPlyHandler:
    @pytest.mark.parametrize('file', (FILE_BINARY, FILE_ASCII))
    def test_load(self, file):
        pcd = PlyHandler.load(file)
        assert len(pcd) == 43577
        assert 'rgb' in pcd.scalar_fields
        assert 'normals' in pcd.scalar_fields
        assert 'intensity' not in pcd.scalar_fields
        assert len(pcd.scalar_fields) == 3

    # TODO implement temp file
    @pytest.mark.parametrize('file', (FILE_BINARY, FILE_ASCII))
    def test_save(self, file):
        out_path = Path(file)
        out_path = out_path.parent / (out_path.stem + "_temp" + out_path.suffix)
        original_pcd = PlyHandler.load(file, remove_prefix=False)
        PlyHandler.save(original_pcd, out_path, add_prefix=False)
        new_pcd = PlyHandler.load(out_path, remove_prefix=False)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz)
        assert np.allclose(original_pcd.rgb, new_pcd.rgb)
        assert np.allclose(original_pcd.normals, new_pcd.normals)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields

    def test_scalar_prefix(self):
        pcd = PlyHandler.load(FILE_SCALAR_PREFIXED, remove_prefix=True)
        assert hasattr(pcd, 'intensity')


    def test_write_ascii(self):
        pcd = PlyHandler.load(FILE_SCALAR_PREFIXED, remove_prefix=False)
        out_path = Path(FILE_SCALAR_PREFIXED)
        out_path = out_path.parent / ('temp_binary' + out_path.suffix)
        PlyHandler.save(pcd, out_path, add_prefix=False, as_ascii=False)
        out_path2 = out_path.parent / ('temp_ascii' + out_path.suffix)
        if not out_path2.is_file(): # Only create on demand due to slow computation time
            PlyHandler.save(pcd, out_path, add_prefix=False, as_ascii=True)

        loaded_1 = PlyHandler.load(out_path, remove_prefix=False)
        loaded_2 = PlyHandler.load(out_path2, remove_prefix=False)

        assert np.all(loaded_1.xyz == loaded_2.xyz)
        assert np.all(loaded_1.rgb == loaded_2.rgb)
        assert np.all(loaded_1.normals == loaded_2.normals)
        assert np.all(loaded_1.scalar_fields['scalar_intensity'] == loaded_2.scalar_fields['scalar_intensity'])
