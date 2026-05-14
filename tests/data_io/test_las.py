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

        # COUPLE-05 D-17 site 1 regression: explicit source_range matches implicit default
        assert np.array_equal(normalize_uint16(i), normalize_uint16(i, source_range=(0.0, 1.0)))

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

    def test_save_add_prefix(self, tmp_path):
        """API-02 T01: save with add_prefix=True writes 'scalar_' prefixed extra-dim."""
        rng = np.random.default_rng(42)
        n = 10
        xyz = rng.random((n, 3)).astype(np.float32)
        pcd = PointCloudData(xyz, numerical_optimization_shift=None)
        pcd.scalar_fields["classification"] = np.arange(n, dtype=np.uint8)

        out = tmp_path / "test.las"
        LasHandler.save(pcd, out, add_prefix=True, prefix="scalar_")

        # On-disk extra-dim must be prefixed
        extra_dim_names = list(laspy.read(str(out)).point_format.extra_dimension_names)
        assert "scalar_classification" in extra_dim_names

        # Reload: remove_prefix=True (default) must strip the prefix so key is "classification"
        reloaded = LasHandler.load(out, remove_prefix=True)
        assert "classification" in reloaded.scalar_fields

    def test_save_no_prefix(self, tmp_path):
        """API-02 T02: save with add_prefix=False writes bare extra-dim without prefix."""
        rng = np.random.default_rng(42)
        n = 10
        xyz = rng.random((n, 3)).astype(np.float32)
        pcd = PointCloudData(xyz, numerical_optimization_shift=None)
        pcd.scalar_fields["classification"] = np.arange(n, dtype=np.uint8)

        out = tmp_path / "test.las"
        LasHandler.save(pcd, out, add_prefix=False)

        # On-disk extra-dim must NOT be prefixed
        extra_dim_names = list(laspy.read(str(out)).point_format.extra_dimension_names)
        assert "classification" in extra_dim_names
        assert "scalar_classification" not in extra_dim_names

        # Reload with remove_prefix=False: key must appear as "classification"
        reloaded = LasHandler.load(out, remove_prefix=False)
        assert "classification" in reloaded.scalar_fields
        np.testing.assert_array_equal(
            np.asarray(reloaded.scalar_fields["classification"]),
            np.arange(n, dtype=np.uint8),
        )

    def test_save_revert_sf_types(self, tmp_path):
        """API-02 T03: save with revert_sf_types=True casts back to the tracked origin dtype."""
        from pchandler.scalar_fields.scalar_fields import ScalarField

        rng = np.random.default_rng(42)
        n = 10
        xyz = rng.random((n, 3)).astype(np.float32)
        pcd = PointCloudData(xyz, numerical_optimization_shift=None)

        # Establish uint8 as origin dtype
        pcd.scalar_fields["classification"] = np.arange(n, dtype=np.uint8)
        origin = pcd.scalar_fields["classification"].origin_dtype

        # Now reassign as float32 but preserve origin dtype (simulates mid-pipeline cast)
        pcd.scalar_fields["classification"] = ScalarField(
            np.arange(n, dtype=np.float32), name="classification", origin_dtype=origin
        )
        assert pcd.scalar_fields["classification"].dtype == np.float32
        assert pcd.scalar_fields["classification"].origin_dtype.dtype == np.uint8

        out = tmp_path / "test.las"
        LasHandler.save(pcd, out, add_prefix=False, revert_sf_types=True)

        # On-disk dtype must be uint8 (reverted to origin)
        saved = laspy.read(str(out))
        assert "classification" in saved.point_format.extra_dimension_names
        assert saved.point_format["classification"].dtype == np.dtype("uint8")

    def test_save_no_revert(self, tmp_path):
        """API-02 T04: save with revert_sf_types=False keeps current in-memory dtype."""
        from pchandler.scalar_fields.scalar_fields import ScalarField

        rng = np.random.default_rng(42)
        n = 10
        xyz = rng.random((n, 3)).astype(np.float32)
        pcd = PointCloudData(xyz, numerical_optimization_shift=None)

        # Establish uint8 as origin dtype then cast to float32 in-memory
        pcd.scalar_fields["classification"] = np.arange(n, dtype=np.uint8)
        origin = pcd.scalar_fields["classification"].origin_dtype
        pcd.scalar_fields["classification"] = ScalarField(
            np.arange(n, dtype=np.float32), name="classification", origin_dtype=origin
        )

        out = tmp_path / "test.las"
        LasHandler.save(pcd, out, add_prefix=False, revert_sf_types=False)

        # On-disk dtype must be float32 (current in-memory dtype preserved)
        saved = laspy.read(str(out))
        assert "classification" in saved.point_format.extra_dimension_names
        assert saved.point_format["classification"].dtype == np.dtype("float32")

    def test_save_long_name_raises(self, tmp_path):
        """API-02 T05: extra-dim name > 31 chars raises ValueError before any disk write."""
        rng = np.random.default_rng(42)
        n = 3
        xyz = rng.random((n, 3)).astype(np.float32)
        pcd = PointCloudData(xyz, numerical_optimization_shift=None)

        # 32-char field name: bare name already exceeds the 31-char LAS limit
        long_name = "a" * 32
        pcd.scalar_fields[long_name] = np.arange(n, dtype=np.uint8)

        out = tmp_path / "test.las"
        import pytest

        with pytest.raises(ValueError, match="31"):
            LasHandler.save(pcd, out, add_prefix=False)

        # File must not have been created (guard fires before disk write)
        assert not out.exists()
