from pathlib import Path

import numpy as np

from pchandler import PointCloudData
from pchandler.data_io import Ply as PlyHandler
from tests.data_io.test_core import BaseLoadSave

base_directory = Path(__file__).resolve().parent.parent


class TestPlyAsciiHandler(BaseLoadSave):
    cls = PlyHandler
    folder = BaseLoadSave.folder / "PLY_ASCII"
    reference = folder / "XYZ_RGB_Normals_Intensity_SFs.ply"

    def test_load_all(self):
        super()._load_all()

    def test_save(self, tmp_path):
        super()._save(tmp_path, as_ascii=True)


class TestPlyBinaryHandler(BaseLoadSave):
    cls = PlyHandler
    folder = BaseLoadSave.folder / "PLY_Binary"
    reference = folder / "XYZ_RGB_Normals_Intensity_SFs.ply"

    def test_load_all(self):
        super()._load_all()

    def test_save(self, tmp_path):
        super()._save(tmp_path, as_ascii=False)


def test_binary_vs_ascii_write(tmp_path):
    xyz = np.random.rand(100, 3)
    intensity = np.random.rand(100)
    rgb = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
    normals = np.random.rand(100, 3).astype(np.float32)
    sf1 = np.random.rand(100)
    pcd = PointCloudData(
        xyz=xyz,
        intensity=intensity,
        rgb=rgb,
        normals=normals,
        scalar_fields={"sf1": sf1},
        numerical_optimization_shift=None,
    )

    ascii_file = tmp_path / "ascii.ply"
    binary_file = tmp_path / "binary.ply"
    PlyHandler.save(pcd, ascii_file, as_ascii=True)
    pcd_asc = PlyHandler.load(ascii_file)

    PlyHandler.save(pcd, binary_file, as_ascii=False)
    pcd_bin = PlyHandler.load(binary_file)

    assert np.allclose(xyz, pcd_bin.xyz)
    assert np.allclose(rgb, pcd_bin.rgb)
    assert np.allclose(intensity, pcd_bin.intensity)
    assert np.allclose(sf1, pcd_bin.scalar_fields["sf1"])

    assert np.allclose(xyz, pcd_asc.xyz)
    assert np.allclose(rgb, pcd_asc.rgb)
    assert np.allclose(intensity, pcd_asc.intensity)
    assert np.allclose(sf1, pcd_asc.scalar_fields["sf1"])

    # Normals are converted to unit vectors
    assert np.allclose(normals / np.linalg.norm(normals, axis=1).reshape(-1, 1), pcd_bin.normals)
    assert np.allclose(normals / np.linalg.norm(normals, axis=1).reshape(-1, 1), pcd_asc.normals)
