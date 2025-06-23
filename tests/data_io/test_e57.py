import pytest

from pathlib import Path

import numpy as np

from pchandler.v2.data_io.e57 import E57Handler


class TestCsvHandler:
    file_1 = Path(r"/home/jon-allemand/PycharmProjects/pchandler/tests/data/pumpAGroupByLineRowColumnIndex.e57")

    def test_load(self):
        pcd = E57Handler.load(self.file_1)
        assert len(pcd) == 155201
        assert 'rgb' in pcd.scalar_fields

    def test_save(self):
        out_path = Path(r"D:\Python\pchandler\tests\data\test_target_rgb_temp" + E57Handler.FORMATS[0])
        original_pcd = E57Handler.load(self.file_1)
        E57Handler.save(original_pcd, out_path)
        new_pcd = E57Handler.load(out_path)

        assert np.allclose(original_pcd.xyz, new_pcd.xyz)
        assert np.allclose(original_pcd.rgb, new_pcd.rgb)
        assert np.allclose(original_pcd.normals, new_pcd.normals)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields