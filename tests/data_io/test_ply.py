import pytest
from pathlib import Path

from pchandler.data_io import Ply as PlyHandler
from tests.data_io.test_core import TestLoadSave

base_directory = Path(__file__).resolve().parent.parent


class TestPlyAsciiHandler(TestLoadSave):
    cls = PlyHandler
    folder = TestLoadSave.folder / 'PLY_ASCII'
    reference: Path = folder / 'XYZ_Only.ply'
    all_fields_file = folder / 'XYZ_RGB_Normals_Intensity_SFs.ply'


class TestPlyBinaryHandler(TestLoadSave):
    cls = PlyHandler
    folder = TestLoadSave.folder / 'PLY_Binary'
    reference: Path = folder / 'XYZ_Only.ply'
    all_fields_file = folder / 'XYZ_RGB_Normals_Intensity_SFs.ply'

    def test_binary_vs_ascii_write(self):
        pass
