import pytest

from pathlib import Path

from pchandler.data_io.las import LasHandler
from tests.data_io.test_core import TestLoadSave

base_directory = Path(__file__).resolve().parent.parent


class TestLasHandler(TestLoadSave):
    cls = LasHandler
    folder = TestLoadSave.folder / 'LAS'
    reference: Path = folder / 'XYZ_Only.las'
    all_fields_file = folder / 'XYZ_RGB_Normals_Intensity_SFs.las'

