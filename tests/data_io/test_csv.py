from pathlib import Path

from pchandler.data_io import Csv as CsvHandler
from tests.data_io.test_core import TestLoadSave

base_directory = Path(__file__).resolve().parent.parent


class TestCsvHandler(TestLoadSave):
    cls = CsvHandler
    folder = TestLoadSave.folder / 'TXT'
    reference: Path = folder / 'XYZ_Only.txt'
    all_fields_file = folder / 'XYZ_RGB_Normals_Intensity_SFs.txt'

    def test_no_number_points_line(self):
        pass

    def test_sniff_file(self):
        pass

    def test_get_ascii_load_dtype(self):
        pass

    def test_column_names_row_load(self):
        pass

    def test_comment_on_load(self):
        pass

    def test_delimiters(self):
        pass
