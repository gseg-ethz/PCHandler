import pytest
from pathlib import Path

from pchandler.data_io import Ply as PlyHandler
from tests.data_io.test_core import BaseLoadSave

base_directory = Path(__file__).resolve().parent.parent


class TestPlyAsciiHandler(BaseLoadSave):
    cls = PlyHandler
    folder = BaseLoadSave.folder / 'PLY_ASCII'
    reference = folder / 'XYZ_RGB_Normals_Intensity_SFs.ply'

    def test_load_all(self):
        super().test_load_all()

    def test_save(self):
        super().test_save()

class TestPlyBinaryHandler(BaseLoadSave):
    cls = PlyHandler
    folder = BaseLoadSave.folder / 'PLY_Binary'
    reference = folder / 'XYZ_RGB_Normals_Intensity_SFs.ply'

    def test_load_all(self):
        super().test_load_all()

    def test_save(self):
        super().test_save()

    def test_binary_vs_ascii_write(self):
        pass
