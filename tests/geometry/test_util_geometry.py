import alphashape
import numpy as np
import pytest

from pchandler import PointCloudData
from pchandler.geometry import util as geom_util
from pchandler.geometry.util import MinMaxPoints, get_outline_polygon


@pytest.fixture(scope="function")
def pcd_simple():
    return PointCloudData([[0, 0, 0], [1, 1, 1], [2, 2, 2]])


@pytest.fixture(scope="function")
def minmax():
    return MinMaxPoints(np.array([0, 1, 2]), np.array([3, 4, 5]))


@pytest.fixture(scope="function")
def xy_plane_points():
    return PointCloudData([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0], [0.5, 0.5, 0]])


@pytest.fixture(scope="function")
def xz_plane_points():
    return PointCloudData([[0, 0, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0.5, 0, 0.5]])


@pytest.fixture(scope="function")
def yz_plane_points():
    return PointCloudData([[0, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0.5, 0.5]])


class TestMinMaxPoints:
    def test_init_positional(self):
        min_max_points = MinMaxPoints(np.array([0, 0, 0]), np.array([1, 1, 1]))
        assert np.all(min_max_points.minimum == [0, 0, 0])
        assert np.all(min_max_points.maximum == [1, 1, 1])

    def test_init_kwarg(self):
        min_max_points = MinMaxPoints(maximum=np.array([2, 2, 2]), minimum=np.array([1, 1, 1]))
        assert np.all(min_max_points.minimum == [1, 1, 1])
        assert np.all(min_max_points.maximum == [2, 2, 2])

    def test_init_from_points(self, pcd_simple):
        min_max_points = MinMaxPoints.from_points(pcd_simple.arr)
        assert np.all(min_max_points.minimum == [0, 0, 0])
        assert np.all(min_max_points.maximum == [2, 2, 2])

        minmax_shifted = MinMaxPoints.from_points(pcd_simple.arr, already_applied_shift_vec=np.ones(3))
        assert np.all(minmax_shifted.minimum == [1, 1, 1])
        assert np.all(minmax_shifted.maximum == [3, 3, 3])

        null_minmax = MinMaxPoints.from_points(np.array([]))
        assert np.all(null_minmax.minimum == [0, 0, 0])
        assert np.all(null_minmax.maximum == [0, 0, 0])

    def test_init_from_bboxes(self, pcd_simple):
        bbox1 = MinMaxPoints.from_points(pcd_simple.arr)
        bbox2 = MinMaxPoints.from_points(pcd_simple.arr + 1)
        bbox3 = MinMaxPoints.from_points(pcd_simple.arr + 2)
        bbox4 = MinMaxPoints.from_points(pcd_simple.arr + 3)

        minmax_limits = MinMaxPoints.from_minmax_points([bbox1, bbox2, bbox3, bbox4])
        assert np.all(minmax_limits.minimum == [0, 0, 0])
        assert np.all(minmax_limits.maximum == [5, 5, 5])

    def test_init_from_self(self, minmax):
        min_max_points = MinMaxPoints.from_minmax_points(minmax)
        assert np.all(min_max_points.minimum == [0, 1, 2])
        assert np.all(min_max_points.maximum == [3, 4, 5])

    def test_central_point(self, minmax):
        assert np.all(minmax.central_point == [1.5, 2.5, 3.5])

    def test_extents(self, minmax):
        assert np.all(minmax.extents == [3, 3, 3])

    def test___array__(self, minmax):
        assert np.all(np.array(minmax) == [[0, 1, 2], [3, 4, 5]])


class TestGetOutlinePolygon:
    def test_xy_plane(self):
        a = PointCloudData((np.random.rand(10000, 3) * 8) + np.array([-3, -1, 2]))
        outline = get_outline_polygon(a, "xy")
        assert np.allclose(np.array(outline.bounds), [-3, -1, 5, 7], atol=1e-1)

    def test_xz_plane(self):
        a = PointCloudData((np.random.rand(10000, 3) * 8) + np.array([-3, -1, 2]))
        outline = get_outline_polygon(a, "xz")
        assert np.allclose(np.array(outline.bounds), [-3, 2, 5, 10], atol=1e-1)

    def test_yz_plane(self):
        a = PointCloudData((np.random.rand(10000, 3) * 8) + np.array([-3, -1, 2]))
        outline = get_outline_polygon(a, "yz")
        assert np.allclose(np.array(outline.bounds), [-1, 2, 7, 10], atol=1e-1)

    def test_invalid_plane_input(self):
        with pytest.raises(ValueError):
            get_outline_polygon(PointCloudData([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), plane="invalid")

    # Phase 4 PERF-02 D-25 #1: deterministic by default (seed=None → default_rng(0)).
    def test_determinism_default_seed(self):
        # Small jitter-sensitive synthetic fixture: 500 points in a unit cube.
        # Pre-fix this test would be flaky across runs (global np.random RNG);
        # post-fix the default seed=None resolves to default_rng(0).
        rng = np.random.default_rng(123)
        pcd = PointCloudData(rng.standard_normal((500, 3)))
        a = get_outline_polygon(pcd, "xy")
        b = get_outline_polygon(pcd, "xy")
        # Shapely's geometric equality (D-25 #1).
        assert a.equals(b)

    # Phase 4 PERF-02 D-25 #2: deterministic with explicit seed.
    def test_determinism_explicit_seed(self):
        rng = np.random.default_rng(7)
        pcd = PointCloudData(rng.standard_normal((500, 3)))
        a = get_outline_polygon(pcd, "xy", seed=42)
        b = get_outline_polygon(pcd, "xy", seed=42)
        assert a.equals(b)
        # Different seeds should yield distinguishable polygons. The jitter is
        # 1e-6 — small but enough on a 500-pt cloud to perturb the alpha-shape
        # decision boundary. If a future numpy or alphashape change makes this
        # case insensitive, swap to a larger nb_points override; for now the
        # researcher's discretionary verification confirms inequality.
        c = get_outline_polygon(pcd, "xy", seed=0)
        assert not a.equals(c)

    # Phase 4 PERF-02 D-25 #3: auto-cap triggers when nb_points=-1 and input
    # exceeds _DEFAULT_OUTLINE_MAX_POINTS. We monkeypatch the constant low and
    # capture alphashape's input shape via a wrapper (D-25 Claude's Discretion).
    def test_auto_cap_triggers(self, monkeypatch):
        monkeypatch.setattr(geom_util, "_DEFAULT_OUTLINE_MAX_POINTS", 10)
        rng = np.random.default_rng(0)
        pcd = PointCloudData(rng.standard_normal((100, 3)))

        captured: dict[str, tuple[int, ...]] = {}
        real_alphashape = alphashape.alphashape

        def _capture(arr, *args, **kwargs):
            captured["shape"] = np.asarray(arr).shape
            return real_alphashape(arr, *args, **kwargs)

        monkeypatch.setattr(alphashape, "alphashape", _capture)
        _ = get_outline_polygon(pcd, "xy")
        # Post-trim, alphashape sees 10 rows (the monkeypatched cap), not 100.
        assert captured["shape"][0] == 10

    # Phase 4 PERF-02 D-25 #4: explicit nb_points overrides the auto-cap.
    def test_auto_cap_respects_explicit_nb_points(self, monkeypatch):
        # Even with a tiny default cap, an explicit nb_points wins.
        monkeypatch.setattr(geom_util, "_DEFAULT_OUTLINE_MAX_POINTS", 10)
        rng = np.random.default_rng(0)
        pcd = PointCloudData(rng.standard_normal((1000, 3)))

        captured: dict[str, tuple[int, ...]] = {}
        real_alphashape = alphashape.alphashape

        def _capture(arr, *args, **kwargs):
            captured["shape"] = np.asarray(arr).shape
            return real_alphashape(arr, *args, **kwargs)

        monkeypatch.setattr(alphashape, "alphashape", _capture)
        _ = get_outline_polygon(pcd, "xy", nb_points=50)
        # Explicit nb_points=50 overrides the auto-cap (10).
        assert captured["shape"][0] == 50
