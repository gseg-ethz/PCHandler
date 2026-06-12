from pathlib import Path

import numpy as np
import pye57
import pytest

from pchandler import PointCloudData
from pchandler.data_io import E57 as E57Handler
from tests.data_io.test_core import BaseLoadSave

base_directory = Path(__file__).resolve().parent.parent


@pytest.fixture
def simple_pcd():
    rng = np.random.default_rng(42)
    xyz = rng.uniform(low=-100, high=100, size=(1000, 3))
    rgb = rng.integers(low=0, high=255, size=(1000, 3), dtype=np.uint8)
    intensity = rng.uniform(low=0, high=1, size=(1000,)).astype(np.float32)
    return PointCloudData(xyz, rgb=rgb, intensity=intensity, numerical_optimization_shift=None)


class TestE57Handler(BaseLoadSave):
    cls = E57Handler
    folder = BaseLoadSave.folder / "E57"
    reference = folder / "XYZ_RGB_Normals_Intensity.e57"

    def test_load_all(self):
        super()._load_all()


def test_e57_save_round_trip_no_shift(simple_pcd, tmp_path):
    out = tmp_path / "test.e57"
    E57Handler.save(simple_pcd, out)
    loaded = E57Handler.load(out)
    assert np.allclose(loaded.xyz, simple_pcd.xyz, atol=1e-6)
    assert np.array_equal(np.asarray(loaded.rgb), np.asarray(simple_pcd.rgb))
    assert np.allclose(np.asarray(loaded.intensity), np.asarray(simple_pcd.intensity), atol=1e-6)
    # simple_pcd has no shift; loaded NOS should be None or a zero-value shift (depends on load path).
    if loaded.numerical_optimization_shift is not None:
        assert np.allclose(loaded.numerical_optimization_shift.value, np.zeros(3), atol=1e-9)


def test_e57_save_round_trip_with_shift_embed_true(tmp_path):
    rng = np.random.default_rng(43)
    # Make XYZ that requires a shift (large magnitudes)
    xyz = rng.uniform(low=2_700_000, high=2_700_100, size=(100, 3))
    pcd = PointCloudData(xyz)
    assert pcd.numerical_optimization_shift is not None
    shift_value = pcd.numerical_optimization_shift.value.copy()
    out = tmp_path / "test.e57"
    E57Handler.save(pcd, out, embed_shift_in_transform=True)
    loaded = E57Handler.load(out, read_transform=True)
    assert np.allclose(loaded.xyz, pcd.xyz, atol=1e-4)
    assert loaded.numerical_optimization_shift is not None
    assert np.allclose(loaded.numerical_optimization_shift.value, shift_value, atol=1e-4)


def test_e57_save_round_trip_with_shift_embed_false(tmp_path):
    rng = np.random.default_rng(44)
    xyz = rng.uniform(low=2_700_000, high=2_700_100, size=(100, 3))
    pcd = PointCloudData(xyz)
    assert pcd.numerical_optimization_shift is not None
    shift_value = pcd.numerical_optimization_shift.value.copy()
    out = tmp_path / "test.e57"
    E57Handler.save(pcd, out, embed_shift_in_transform=False)
    loaded = E57Handler.load(out, read_transform=False)
    # World-frame coords written on disk: (pcd.xyz + shift).
    # PointCloudData re-applies its own shift on load, so compare world-frame:
    # loaded.xyz + loaded_shift == pcd.xyz + original_shift
    world_original = np.asarray(pcd.xyz) + shift_value
    if loaded.numerical_optimization_shift is not None:
        world_loaded = np.asarray(loaded.xyz) + loaded.numerical_optimization_shift.value
    else:
        world_loaded = np.asarray(loaded.xyz)
    assert np.allclose(world_loaded, world_original, atol=1.0)  # float32 storage = ~0.25 precision for 2.7M coords


def test_e57_save_iterable_input(simple_pcd, tmp_path):
    pcds = [simple_pcd, simple_pcd.copy(), simple_pcd.copy()]
    out = tmp_path / "multi.e57"
    E57Handler.save(pcds, out)
    loaded_gen = E57Handler.load(out)  # multi-scan returns generator
    loaded_list = list(loaded_gen)
    assert len(loaded_list) == 3


def test_e57_save_iterable_single_open(simple_pcd, tmp_path, monkeypatch):
    """Assert pye57.E57 is opened exactly once across an iterable input."""
    open_count = {"n": 0}
    original_init = pye57.E57.__init__

    def spy_init(self, path, mode="r", *args, **kwargs):
        open_count["n"] += 1
        return original_init(self, path, mode, *args, **kwargs)

    monkeypatch.setattr(pye57.E57, "__init__", spy_init)

    pcds = [simple_pcd, simple_pcd.copy(), simple_pcd.copy()]
    out = tmp_path / "multi_single_open.e57"
    E57Handler.save(pcds, out)
    assert open_count["n"] == 1, f"Expected exactly 1 pye57.E57 open, got {open_count['n']}"


def test_e57_save_skip_warn_unsupported_scalar_field(simple_pcd, tmp_path, caplog):
    simple_pcd.scalar_fields["reflectance"] = np.zeros(len(simple_pcd), dtype=np.float32)
    out = tmp_path / "test.e57"
    with caplog.at_level("WARNING"):
        E57Handler.save(simple_pcd, out, strict=False)
    assert "reflectance" in caplog.text


def test_e57_save_strict_raises_on_unsupported(simple_pcd, tmp_path):
    simple_pcd.scalar_fields["reflectance"] = np.zeros(len(simple_pcd), dtype=np.float32)
    out = tmp_path / "test.e57"
    with pytest.raises(ValueError, match="reflectance"):
        E57Handler.save(simple_pcd, out, strict=True)
