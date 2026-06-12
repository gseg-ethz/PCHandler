"""Stub-drift smoke test — every symbol declared in 41_pchandler/stubs/*/__init__.pyi
must resolve at runtime when the backend is installed.
"""

import pytest


@pytest.mark.parametrize(
    "module_name, symbols",
    [
        ("laspy", ["read", "create", "ExtraBytesParams", "LasData"]),
        ("pye57", ["E57", "ScanHeader"]),
        ("pye57.e57", ["SUPPORTED_POINT_FIELDS"]),
        ("plyfile", ["PlyData", "PlyElement"]),
        ("open3d.geometry", ["PointCloud"]),
        ("open3d.utility", ["Vector3dVector"]),
        ("open3d.core", ["Tensor"]),
        ("open3d.t.geometry", ["PointCloud"]),
        ("py4dgeo", ["Epoch"]),
    ],
)
def test_stub_symbols_resolve_at_runtime(module_name: str, symbols: list[str]) -> None:
    mod = pytest.importorskip(module_name)
    for sym in symbols:
        assert hasattr(mod, sym), f"{module_name}.{sym} missing at runtime — stub drift detected"
