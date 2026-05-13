import tempfile
from pathlib import Path

import laspy
import numpy as np
from GSEGUtils.validators import normalize_uint16

from pchandler import PointCloudData
from pchandler.data_io import Las as LasHandler
from pchandler.geometry import OptimizedShift
from tests.data_io.test_core import BaseLoadSave

base_directory = Path(__file__).resolve().parent.parent


class TestLasHandler(BaseLoadSave):
    cls = LasHandler
    folder = BaseLoadSave.folder / "LAS"
    reference = folder / "XYZ_RGB_Normals_Intensity_SFs.las"

    def test_save(self, tmp_path):
        super()._save(tmp_path)

    def test_load_all(self):
        super()._load_all()

    def test_save_no_optimal_shift(self):
        # Test for coverage
        pcd = PointCloudData(np.round(np.random.rand(100, 3) * 1000, decimals=3), numerical_optimization_shift=None)

        las_file = tempfile.NamedTemporaryFile(suffix=".las", delete_on_close=False)
        with las_file as fp:
            LasHandler.save(pcd, Path(fp.name))
            las_pcd = LasHandler.load(Path(fp.name))

        assert np.allclose(pcd.xyz, las_pcd.xyz)

    def test_load_no_optimal_shift(self):
        pcd_w_shift = LasHandler.load(self.reference)
        pcd_no_shift = LasHandler.load(self.reference, force_no_numerical_shift=True)

        assert len(pcd_w_shift) == len(pcd_no_shift)
        assert not np.allclose(pcd_w_shift.xyz, pcd_no_shift.xyz)
        assert np.allclose(
            pcd_w_shift.xyz + pcd_w_shift.numerical_optimization_shift.value, pcd_no_shift.xyz, atol=0.0001
        )

    def test_intensity_normalisation(self):
        i = np.random.rand(100)
        pcd = PointCloudData(np.random.rand(100, 3), intensity=i.copy())

        las_file = tempfile.NamedTemporaryFile(suffix=".las", delete_on_close=False)
        with las_file as fp:
            LasHandler.save(pcd, Path(fp.name))
            las_pcd = LasHandler.load(Path(fp.name))

        assert not np.allclose(las_pcd.intensity, i)
        assert np.all(las_pcd.intensity == normalize_uint16(i))

    def test_las_load_with_nos_in_pcd_kw(self):
        """BUG-08: caller's NOS in **pcd_kw wins over the LAS-header default; no crash."""
        custom_shift = OptimizedShift(np.array([1.0, 2.0, 3.0]))
        # Pass via the kwarg path (which routes into pcd_kw inside the loader).
        pcd = LasHandler.load(self.reference, numerical_optimization_shift=custom_shift)
        assert pcd.numerical_optimization_shift is custom_shift

    def test_las_load_without_nos_uses_header_default(self):
        """BUG-08: no NOS in pcd_kw -> LAS-header default applies (no regression)."""
        pcd = LasHandler.load(self.reference)
        assert pcd.numerical_optimization_shift is not None
        # The default is OptimizedShift(las.header.offsets); compare values.
        with laspy.open(self.reference) as f:
            expected_offsets = np.array(f.header.offsets, dtype=np.float64)
        np.testing.assert_array_equal(pcd.numerical_optimization_shift.value, expected_offsets)
