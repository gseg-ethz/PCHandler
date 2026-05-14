from pathlib import Path

import numpy as np
import pytest

from pchandler.data_io import Csv as CsvHandler
from pchandler.data_io.csv import (
    _delimiter_sniffer,
    _get_header,
    generate_ascii_load_dtype,
    sniff_file,
)
from pchandler.geometry import OptimizedShift
from tests.data_io.test_core import BaseLoadSave

base_directory = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def no_points_number_line(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("//X Y Z\n")
        f.write("1,2,3\n")
        f.write("4,5,6\n")

    yield file
    file.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def fake_number_points(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("//X Y Z\n")
        f.write("200\n")
        f.write("1,2,3\n")
        f.write("4,5,6\n")

    yield file
    file.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def invalid_delimiter(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("//X Y Z\n")
        f.write("200\n")
        for i in range(20):
            f.write(f"{i}-{i + 1}-{i * 2}\n")
    return file


@pytest.fixture(scope="session")
def delim_space(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("//X Y Z\n")
        for i in range(20):
            f.write(f"{i} {i + 1} {i * 2}\n")
    return file


@pytest.fixture(scope="session")
def delim_semicolon(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("//X Y Z\n")
        for i in range(20):
            f.write(f"{i};{i + 1};{i * 2}\n")
    return file


@pytest.fixture(scope="session")
def delim_comma(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("//X Y Z\n")
        for i in range(20):
            f.write(f"{i},{i + 1},{i * 2}\n")
    return file


@pytest.fixture(scope="session")
def delim_tab(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("//X Y Z\n")
        for i in range(20):
            f.write(f"{i}\t{i + 1}\t{i * 2}\n")
    return file


@pytest.fixture(scope="session")
def inconsistent_column_number(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("//X Y Z\n")
        f.write("20\n")
        f.write("1,2,3\n")
        f.write("4,5,6,7,8\n")

        for i in range(18):
            f.write(f"\n{i},{i + 1},{i * 2}")

    return file


@pytest.fixture(scope="session")
def column_names_not_last_row(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("//X Y Z\n")
        f.write("// GeneralComment \n")
        f.write("// Another Comment with punctuation which should break it. \n")
        for i in range(20):
            f.write(f"{i},{i + 1},{i * 2}\n")

    return file


@pytest.fixture(scope="session")
def hash_commented_file(tmp_path_factory):
    file = tmp_path_factory.mktemp("data") / "temp.txt"
    with open(file, "w") as f:
        f.write("#X Y Z red green blue\n")
        f.write("# GeneralComment \n")
        f.write("# Another Comment with punctuation which should break it. \n")
        for _ in range(20):
            f.write("0.24,1.1,2.3,4,8,12\n")

    return file


class TestCsvHandler(BaseLoadSave):
    cls = CsvHandler
    folder = BaseLoadSave.folder / "TXT"
    reference = folder / "XYZ_RGB_Normals_Intensity_SFs.txt"

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

        assert generate_ascii_load_dtype(["x", "y", "z"]) == np.dtype(
            {"names": ["x", "y", "z"], "formats": [f64, f64, f64]}
        )
        assert generate_ascii_load_dtype(["x", "y", "z", "r", "g", "b"]) == np.dtype(
            {"names": ["x", "y", "z", "r", "g", "b"], "formats": [f64, f64, f64, u8, u8, u8]}
        )
        assert generate_ascii_load_dtype(["x", "y", "z", "red", "green", "blue"]) == np.dtype(
            {"names": ["x", "y", "z", "red", "green", "blue"], "formats": [f64, f64, f64, u8, u8, u8]}
        )
        assert generate_ascii_load_dtype(["x", "y", "z", "rf", "gf", "bf"]) == np.dtype(
            {"names": ["x", "y", "z", "rf", "gf", "bf"], "formats": [f64, f64, f64, f32, f32, f32]}
        )
        assert generate_ascii_load_dtype(["x", "y", "z", "nx", "ny", "nz"]) == np.dtype(
            {"names": ["x", "y", "z", "nx", "ny", "nz"], "formats": [f64, f64, f64, f32, f32, f32]}
        )
        assert generate_ascii_load_dtype(["x", "y", "z", "nx", "ny", "nz", "r", "g", "b", "sf1", "sf2"]) == np.dtype(
            {
                "names": ["x", "y", "z", "nx", "ny", "nz", "r", "g", "b", "sf1", "sf2"],
                "formats": [f64, f64, f64, f32, f32, f32, u8, u8, u8, f32, f32],
            }
        )

    def test_column_names_row_load(self, column_names_not_last_row):
        # with pytest.raises(ValueError):
        info = sniff_file(column_names_not_last_row)
        assert info.fields == []

        info = sniff_file(column_names_not_last_row, field_names_row_index=0)
        assert len(info.header) == 3
        assert info.delimiter == ","
        assert info.fields == ["X", "Y", "Z"]
        assert info.num_fields == 3
        assert info.num_points is None

    def test_comment_on_load(self, hash_commented_file):
        pcd = CsvHandler.load(hash_commented_file, comment="#", column_names_row=0)
        assert np.allclose(pcd.x, 0.24)
        assert np.allclose(pcd.y, 1.1)
        assert np.allclose(pcd.z, 2.3)
        assert np.allclose(pcd.rgb.r, 4)
        assert np.allclose(pcd.rgb.g, 8)
        assert np.allclose(pcd.rgb.b, 12)
        assert len(pcd) == 20

    def test_delimiters(
        self, invalid_delimiter: Path, delim_space: Path, delim_semicolon: Path, delim_comma: Path, delim_tab: Path
    ):
        with pytest.raises(ValueError):
            _delimiter_sniffer(invalid_delimiter)

        delim, num = _delimiter_sniffer(invalid_delimiter, delimiters="-")
        assert delim == "-"
        assert num == 3

        delim, num = _delimiter_sniffer(delim_space)
        assert delim == " "
        assert num == 3

        # Also perform coverage on reaching a break line
        delim, num = _delimiter_sniffer(delim_semicolon, lines_to_check=1000)
        assert delim == ";"
        assert num == 3

        delim, num = _delimiter_sniffer(delim_comma)
        assert delim == ","
        assert num == 3

        delim, num = _delimiter_sniffer(delim_tab)
        assert delim == "\t"
        assert num == 3

    def test_get_field_count(self, delim_comma, inconsistent_column_number):

        with pytest.raises(ValueError):
            _delimiter_sniffer(delim_comma, minimum_columns=5)

        with pytest.raises(ValueError):
            _, num = _delimiter_sniffer(delim_comma, comment="#")

        with pytest.raises(ValueError):
            _delimiter_sniffer(inconsistent_column_number)


@pytest.mark.parametrize("suffix", [".txt", ".csv", ".xyz", ".asc", ".ascii", ".pts"])
class TestCsvPcdKwargs:
    """Tests that pcd_kwargs are propagated through CsvHandler.load (resolves csv.py TODO)."""

    def test_propagate_nos(self, tmp_path, suffix):
        """numerical_optimization_shift kwarg is forwarded to PointCloudData constructor."""
        rng = np.random.default_rng(42)
        from pchandler import PointCloudData

        xyz = rng.random((100, 3)) * 100.0
        pcd_orig = PointCloudData(xyz=xyz.astype(np.float64))
        out_path = tmp_path / f"in{suffix}"
        CsvHandler.save(pcd_orig, out_path)

        nos = OptimizedShift(np.array([1000.0, 2000.0, 3000.0]))
        loaded = CsvHandler.load(out_path, numerical_optimization_shift=nos)
        assert np.allclose(loaded.numerical_optimization_shift.value, [1000.0, 2000.0, 3000.0])

    def test_propagate_parametrized(self, tmp_path, suffix):
        """Additional pcd_kw (socs_origin) is forwarded to PointCloudData constructor."""
        rng = np.random.default_rng(7)
        from pchandler import PointCloudData

        xyz = rng.random((20, 3)) * 50.0
        pcd_orig = PointCloudData(xyz=xyz.astype(np.float64))
        out_path = tmp_path / f"in{suffix}"
        CsvHandler.save(pcd_orig, out_path)

        socs = np.array([10.0, 20.0, 30.0])
        loaded = CsvHandler.load(out_path, socs_origin=socs)
        assert np.allclose(loaded.socs_origin, socs)


class TestCsvStrictByName:
    """Tests for strict-by-name field-selection contract (API-04 / D-12)."""

    @pytest.fixture
    def csv_with_header(self, tmp_path):
        """CSV with header ['x', 'y', 'z', 'intensity', 'classification'] and 10 rows."""
        rng = np.random.default_rng(99)
        path = tmp_path / "strict_test.csv"
        xyz = rng.random((10, 3)) * 100.0
        intensity = rng.random(10).astype(np.float32)
        classification = rng.integers(0, 5, size=10).astype(np.float32)
        with open(path, "w") as f:
            f.write("// x,y,z,intensity,classification\n")
            for i in range(10):
                f.write(f"{xyz[i, 0]:.6f},{xyz[i, 1]:.6f},{xyz[i, 2]:.6f},{intensity[i]:.6f},{classification[i]:.6f}\n")
        return path, xyz, intensity, classification

    def test_field_present(self, csv_with_header):
        """Load with a known field succeeds and data matches the original column."""
        path, xyz, intensity, _ = csv_with_header
        loaded = CsvHandler.load(path, scalar_fields=["intensity"])
        assert np.allclose(loaded.scalar_fields["intensity"], intensity, atol=1e-5)

    def test_single_missing_raises(self, csv_with_header):
        """Load with a field not in the header raises ValueError with the documented message."""
        path, _, _, _ = csv_with_header
        with pytest.raises(ValueError, match="requested scalar fields not in header"):
            CsvHandler.load(path, scalar_fields=["missing"])

    def test_partial_match_lists_missing(self, csv_with_header):
        """Load with one present + one absent field raises ValueError listing only the absent name."""
        path, _, _, _ = csv_with_header
        with pytest.raises(ValueError) as exc_info:
            CsvHandler.load(path, scalar_fields=["intensity", "missing"])
        msg = str(exc_info.value)
        # The portion between "not in header: " and ". Available" should list only "missing"
        not_in_header_part = msg.split("not in header: ")[1].split(". Available")[0]
        assert "missing" in not_in_header_part
        assert "intensity" not in not_in_header_part

    def test_positional_fallback_no_header(self, tmp_path):
        """File without a header falls back to positional column mapping."""
        rng = np.random.default_rng(55)
        path = tmp_path / "no_header.csv"
        xyz = rng.random((15, 3)) * 10.0
        intensity = rng.random(15).astype(np.float32)
        with open(path, "w") as f:
            for i in range(15):
                f.write(f"{xyz[i, 0]:.6f},{xyz[i, 1]:.6f},{xyz[i, 2]:.6f},{intensity[i]:.6f}\n")
        # No header in file — sniff_file returns fields=[] — positional fallback applies
        loaded = CsvHandler.load(path, scalar_fields=["intensity"])
        # Column 3 (0-indexed) should map to intensity via positional fallback
        assert np.allclose(loaded.scalar_fields["intensity"], intensity, atol=1e-5)
