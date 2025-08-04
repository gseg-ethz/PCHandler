import pytest
from pathlib import Path

import numpy as np

from pchandler.data_io.csv import CsvHandler, _get_header, _delimiter_sniffer, sniff_file, generate_ascii_load_dtype
from tests.data_io.test_core import BaseLoadSave

base_directory = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='session')
def no_points_number_line(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('//X Y Z\n')
        f.write('1,2,3\n')
        f.write('4,5,6\n')

    yield file
    file.unlink(missing_ok=True)

@pytest.fixture(scope='session')
def fake_number_points(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('//X Y Z\n')
        f.write('200\n')
        f.write('1,2,3\n')
        f.write('4,5,6\n')

    yield file
    file.unlink(missing_ok=True)


@pytest.fixture(scope='session')
def invalid_delimiter(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('//X Y Z\n')
        f.write('200\n')
        for i in range(20):
            f.write(f'{i}-{i+1}-{i*2}\n')
    return file

@pytest.fixture(scope='session')
def delim_space(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('//X Y Z\n')
        for i in range(20):
            f.write(f'{i} {i+1} {i*2}\n')
    return file

@pytest.fixture(scope='session')
def delim_semicolon(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('//X Y Z\n')
        for i in range(20):
            f.write(f'{i};{i+1};{i*2}\n')
    return file

@pytest.fixture(scope='session')
def delim_comma(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('//X Y Z\n')
        for i in range(20):
            f.write(f'{i},{i+1},{i*2}\n')
    return file

@pytest.fixture(scope='session')
def delim_tab(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('//X Y Z\n')
        for i in range(20):
            f.write(f'{i}\t{i+1}\t{i*2}\n')
    return file


@pytest.fixture(scope='session')
def inconsistent_column_number(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('//X Y Z\n')
        f.write('20\n')
        f.write('1,2,3\n')
        f.write('4,5,6,7,8\n')

        for i in range(18):
            f.write(f'\n{i},{i+1},{i*2}')

    return file

@pytest.fixture(scope='session')
def column_names_not_last_row(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('//X Y Z\n')
        f.write('// GeneralComment \n')
        f.write('// Another Comment with punctuation which should break it. \n')
        for i in range(20):
            f.write(f'{i},{i+1},{i*2}\n')

    return file

@pytest.fixture(scope='session')
def hash_commented_file(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / 'temp.txt'
    with open(file, 'w') as f:
        f.write('#X Y Z red green blue\n')
        f.write('# GeneralComment \n')
        f.write('# Another Comment with punctuation which should break it. \n')
        for _ in range(20):
            f.write(f'0.24,1.1,2.3,4,8,12\n')

    return file


class TestCsvHandler(BaseLoadSave):
    cls = CsvHandler
    folder = BaseLoadSave.folder / 'TXT'
    reference = folder / 'XYZ_RGB_Normals_Intensity_SFs.txt'

    def test_save(self, tmp_path):
        super()._save(tmp_path)

    def test_load_all(self):
        super()._load_all()

    def test_number_points_line(self, no_points_number_line: Path, fake_number_points: Path):
        _, number_points = _get_header(no_points_number_line)
        assert number_points is None

        _, number_points = _get_header(fake_number_points)
        assert number_points == 200

    def test_get_ascii_load_dtype(self):
        f64 = np.float64().dtype.str
        f32 = np.float32().dtype.str
        u8 = np.uint8().dtype.str

        assert generate_ascii_load_dtype(['x','y','z']) == np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': [f64, f64, f64]
        })
        assert generate_ascii_load_dtype(['x','y','z','r','g','b']) == np.dtype({
            'names': ['x','y','z','r','g','b'],
            'formats': [f64, f64, f64, u8, u8, u8]
        })
        assert generate_ascii_load_dtype(['x','y','z','red','green','blue']) == np.dtype({
            'names': ['x','y','z','red','green','blue'],
            'formats': [f64, f64, f64, u8, u8, u8]
        })
        assert generate_ascii_load_dtype(['x','y','z','rf','gf','bf']) == np.dtype({
            'names': ['x','y','z','rf','gf','bf'],
            'formats': [f64, f64, f64, f32, f32, f32]
        })
        assert generate_ascii_load_dtype(['x','y','z','nx','ny','nz']) == np.dtype({
            'names': ['x','y','z','nx','ny','nz'],
            'formats': [f64, f64, f64, f32, f32, f32]
        })
        assert generate_ascii_load_dtype(['x','y','z','nx','ny','nz','r','g','b','sf1','sf2']) == np.dtype({
            'names': ['x','y','z','nx','ny','nz','r','g','b','sf1','sf2'],
            'formats': [f64, f64, f64, f32, f32, f32, u8, u8, u8, f32, f32]
        })

    def test_column_names_row_load(self, column_names_not_last_row):
        # with pytest.raises(ValueError):
        info = sniff_file(column_names_not_last_row)
        assert info.fields == []

        info = sniff_file(column_names_not_last_row, field_names_row_index=0)
        assert len(info.header) == 3
        assert info.delimiter == ','
        assert info.fields == ['X', 'Y', 'Z']
        assert info.num_fields == 3
        assert info.num_points is None

    def test_comment_on_load(self, hash_commented_file):
        pcd = CsvHandler.load(hash_commented_file, comment='#', column_names_row=0)
        assert np.allclose(pcd.x, 0.24)
        assert np.allclose(pcd.y, 1.1)
        assert np.allclose(pcd.z, 2.3)
        assert np.allclose(pcd.rgb.r, 4)
        assert np.allclose(pcd.rgb.g, 8)
        assert np.allclose(pcd.rgb.b, 12)
        assert len(pcd) == 20

    def test_delimiters(self,
                        invalid_delimiter: Path,
                        delim_space: Path,
                        delim_semicolon: Path,
                        delim_comma: Path,
                        delim_tab: Path):
        with pytest.raises(ValueError):
            _delimiter_sniffer(invalid_delimiter)

        delim, num = _delimiter_sniffer(invalid_delimiter, delimiters='-')
        assert delim == '-'
        assert num == 3

        delim, num = _delimiter_sniffer(delim_space)
        assert delim == ' '
        assert num == 3

        # Also perform coverage on reaching a break line
        delim, num = _delimiter_sniffer(delim_semicolon, lines_to_check=1000)
        assert delim == ';'
        assert num == 3

        delim, num = _delimiter_sniffer(delim_comma)
        assert delim == ','
        assert num == 3

        delim, num = _delimiter_sniffer(delim_tab)
        assert delim == '\t'
        assert num == 3

    def test_get_field_count(self, delim_comma, inconsistent_column_number):

        with pytest.raises(ValueError):
            _delimiter_sniffer(delim_comma, minimum_columns=5)

        with pytest.raises(ValueError):
            _, num = _delimiter_sniffer(delim_comma, comment='#')

        with pytest.raises(ValueError):
            _delimiter_sniffer(inconsistent_column_number)

