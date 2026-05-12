from pathlib import Path

import pytest

from pchandler.data_io import E57 as E57Handler
from tests.data_io.test_core import BaseLoadSave

base_directory = Path(__file__).resolve().parent.parent


class TestE57Handler(BaseLoadSave):
    cls = E57Handler
    folder = BaseLoadSave.folder / "E57"
    reference = folder / "XYZ_RGB_Normals_Intensity.e57"

    def test_save(self, tmp_path):
        with pytest.raises(NotImplementedError):
            super()._save(tmp_path)

    def test_load_all(self):
        super()._load_all()


# class TestE57Handler:
#     file_1 = base_directory / ".."/ "data" / "test_target_intensity_normals_rgb.e57"
#     out_path = base_directory / ".." / "data" / "test_target_rgb_temp.e57"
#
#
#
#
#     def test_load_all(self):
#         pcd = E57Handler.load(self.file_1, retain_intensity=True, retain_rgb=True)
#         assert len(pcd) == 43577
#         assert 'intensity' in pcd.scalar_fields
#         assert 'rgb' in pcd.scalar_fields
#         assert 'normals' not in pcd.scalar_fields
#
#     def test_load_none(self):
#         pcd = E57Handler.load(self.file_1, retain_intensity=False, retain_rgb=False)
#         assert len(pcd) == 43577
#         assert 'intensity' not in pcd.scalar_fields
#         assert 'rgb' not in pcd.scalar_fields
#         assert len(pcd) == 43577
#
#     def test_save(self):
#         assert True
#
#         # original_pcd = E57Handler.load(self.file_1)
#         # E57Handler.save(original_pcd, self.out_path)
#         # new_pcd = E57Handler.load(self.out_path)
#         #
#         # assert np.allclose(original_pcd.xyz, new_pcd.xyz)
#         # assert np.allclose(original_pcd.intensity, new_pcd.intensity)
#         #
#         # for name in original_pcd.scalar_fields.keys():
#         #     assert name in new_pcd.scalar_fields
