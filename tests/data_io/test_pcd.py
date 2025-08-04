import pytest

from pathlib import Path

from pchandler.data_io.pcd import PcdHandler
from pchandler.core import PointCloudData
from tests.data_io.test_core import BaseLoadSave

base_directory = Path(__file__).resolve().parent.parent


class TestLasHandler(BaseLoadSave):
    cls = PcdHandler
    folder = BaseLoadSave.folder / 'PCD'
    reference = folder / 'XYZ_RGB_Normals_Intensity_SFs.pcd'

    def test_save(self):
        with pytest.raises(NotImplementedError):
            self.cls.save(PointCloudData([[0,1,2],[2, 3, 4]]), self.reference)

    def test_load_all(self):
        with pytest.raises(NotImplementedError):
            self.cls.load(self.reference)
