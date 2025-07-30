import pytest

from pathlib import Path

from pchandler.data_io.las import LasHandler
from tests.data_io.test_core import BaseLoadSave

base_directory = Path(__file__).resolve().parent.parent


class TestLasHandler(BaseLoadSave):
    cls = LasHandler
    folder = BaseLoadSave.folder / 'LAS'
    reference = folder / 'XYZ_RGB_Normals_Intensity_SFs.las'

    def test_intensity_normalisation(self):
        pass
