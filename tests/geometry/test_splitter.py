import joblib
import numpy as np
import pytest
from pydantic import ValidationError

from pchandler import PointCloudData
from pchandler.geometry.spherical import FoV, FoVTree
from pchandler.geometry.splitter import (
    FoVTreePointCloudSplitter,
    PointCloudSplitter,
    check_number_jobs,
    split_pc_with_fov_tree,
)


def _build_n_fov_tree(n_fovs: int) -> FoVTree:
    """Construct a 1xN-tile FoVTree containing approximately ``n_fovs`` FoVs.

    The Phase 4 PERF-03 D-29 tests need both small (<= _SERIAL_THRESHOLD = 16)
    and larger (> 16) trees. The existing :func:`new_tree` fixture only
    provides a 200-FoV tree; this helper builds the deterministic
    intermediate sizes (3 / 5 / 8 / 20 ...) used by the equivalence
    suite. A 1xN row keeps :meth:`FoV.tile` stable across requested
    counts (a 2D grid would auto-round into the nearest near-square
    layout via ``FoVTree.build_from_tiles``).
    """
    new_fov = FoV(left=0.3, top=0.4, right=1.3, bottom=2.4)
    # Width 1.0 / N → ~N tiles in a single row. Small epsilon so the
    # endpoint inclusion in `FoV.tile` doesn't over-shoot.
    target_width = 1.0 / n_fovs + 1e-6
    tile_extent = FoV(left=0, right=target_width, top=0.0, bottom=2.5)
    tiles = new_fov.tile(tile_extent)
    tree = FoVTree.build_from_tiles(tiles)
    assert tree is not None
    actual = len(tree.to_list())
    # Sanity: the helper must produce the requested count (otherwise the
    # threshold reasoning in downstream tests is wrong).
    assert actual == n_fovs, f"_build_n_fov_tree({n_fovs}) produced {actual} FoVs"
    return tree


@pytest.fixture(scope="function", autouse=True)
def pcd_() -> PointCloudData:
    """Seeded synthetic PCD for the splitter test suite (D-08 / TEST-05).

    ``np.random.default_rng(0)`` matches the project deterministic-by-default
    convention (Phase 4 D-23). Required by ``test_split``'s TEST-02 contract
    (Phase 6 D-05): direct/iterative shared-key equality is only reproducible
    on a fixed seed.
    """
    rng = np.random.default_rng(0)
    return PointCloudData(
        rng.random((100, 3)) * 100,
        intensity=rng.integers(-200, 200, (100,), dtype=np.int16),
        rgb=rng.integers(0, 256, (100, 3), dtype=np.uint8),
        normals=rng.random((100, 3)),
    )


@pytest.fixture(scope="function")
def new_fov() -> FoV:
    return FoV(left=0.3, top=0.4, right=1.3, bottom=2.4)


@pytest.fixture(scope="function")
def tile_extent() -> FoV:
    return FoV(left=0, right=0.1, top=1.3, bottom=1.4)


@pytest.fixture(scope="function")
def new_tree(new_fov, tile_extent) -> FoVTree:
    return FoVTree.build_from_tiles(new_fov.tile(tile_extent))


class TestAbstractPcdSplitter:
    def test_abstract_methods(self):
        assert hasattr(PointCloudSplitter, "split")


class TestFoVTreePointCloudSplitter:
    def test_initialisation(self, new_tree):

        initialised_splitter = FoVTreePointCloudSplitter(new_tree)

        assert initialised_splitter.fov_tree is new_tree
        assert initialised_splitter.remove_empty is True
        assert initialised_splitter.n_jobs == -1

    def test_invalid_initialisations(self, new_tree):
        with pytest.warns(UserWarning):
            FoVTreePointCloudSplitter(new_tree, n_jobs=100)

        with pytest.raises(TypeError):
            FoVTreePointCloudSplitter(new_tree, n_jobs="NotANumber")  # type: ignore

        with pytest.raises(ValidationError):
            FoVTreePointCloudSplitter(new_tree, remove_empty=123)  # type: ignore

        with pytest.raises(ValidationError):
            FoVTreePointCloudSplitter(new_tree, method=123)

        with pytest.raises(ValidationError):
            FoVTreePointCloudSplitter(new_tree, method="Not_valid")

    def test_split(self, pcd_, new_tree):
        """TEST-02 / D-05 / SC #2: direct-vs-iterative cross-method invariant.

        ``_iterative_split`` returns leaf FoVs only; ``_direct_split`` returns
        ALL FoVs (including non-leaves). The two key-sets are intentionally
        different (researcher repro: 200 entries direct vs 48 iterative on the
        seeded fixture). The contract under test is the SHARED-KEY INTERSECTION:
        on every FoV present in both outputs, per-point arrays must byte-match
        within float32 tolerance. See ``_direct_split`` / ``_iterative_split``
        Notes blocks in ``src/pchandler/geometry/splitter.py`` for the pinned
        contract.

        Both ``iterative.split`` and ``direct.split`` mutate their input PCD
        (iterative via cascading ``FoVFilter.extract``; direct via the trailing
        ``pcd.reduce(zeros)``); each split therefore runs on a fresh
        ``pcd_.copy()``.
        """
        pcd_original = pcd_.copy()
        iterative_splitter = FoVTreePointCloudSplitter(new_tree, method="iterative")
        direct_splitter = FoVTreePointCloudSplitter(new_tree, method="direct", n_jobs=1)

        # Both splits mutate pcd_; give each a fresh copy.
        pcd_iter_input = pcd_.copy()
        pcd_direct_input = pcd_.copy()
        splits_iter = iterative_splitter.split(pcd_iter_input)
        merged_pcd = PointCloudData.merge(*[v for v in splits_iter.values()])
        merged_pcd = PointCloudData.merge(pcd_iter_input, merged_pcd)

        assert len(pcd_original) == len(merged_pcd)
        assert np.allclose(pcd_original.unshifted_bbox, merged_pcd.unshifted_bbox)

        splits_direct = direct_splitter.split(pcd_direct_input)

        # TEST-02 contract: direct/iterative may produce divergent key sets
        # (iterative returns leaves only; direct returns all FoVs including
        # non-leaves). On the shared-key intersection, per-point arrays MUST
        # byte-match within float32 tolerance.
        shared_keys = set(splits_iter) & set(splits_direct)
        assert len(shared_keys) > 0, (
            f"No shared keys between iterative {list(splits_iter)[:5]}... "
            f"and direct {list(splits_direct)[:5]}... — contract regression"
        )
        for k in shared_keys:
            np.testing.assert_allclose(
                splits_iter[k].xyz,
                splits_direct[k].xyz,
                atol=1e-6,
                err_msg=f"xyz divergence at shared key {k!r}",
            )
            np.testing.assert_allclose(
                splits_iter[k].rgb,
                splits_direct[k].rgb,
                atol=1e-6,
                err_msg=f"rgb divergence at shared key {k!r}",
            )
            np.testing.assert_allclose(
                splits_iter[k].intensity,
                splits_direct[k].intensity,
                atol=1e-6,
                err_msg=f"intensity divergence at shared key {k!r}",
            )
            np.testing.assert_allclose(
                splits_iter[k].normals,
                splits_direct[k].normals,
                atol=1e-6,
                err_msg=f"normals divergence at shared key {k!r}",
            )

    def test_invalid_split_mode(self, pcd_, new_tree):
        pcd_original = pcd_.copy()
        with pytest.raises(ValidationError):
            iterative_splitter = FoVTreePointCloudSplitter(new_tree, method="new_moe")
            splits_1 = iterative_splitter.split(pcd_)

        with pytest.raises(ValueError):
            check_number_jobs(0)

    def test_split_validate_variables_negative_path(self, new_tree):
        """DEBT-08 / Phase 6 D-17 — split() raises ValidationError on bad pcd type.

        Re-applied ``@validate_variables`` decorator at splitter.py:~168 catches
        the type mismatch BEFORE entry into the body. Verifies the runtime
        gate; live-reproduction in RESEARCH §D-17 (2026-05-14) confirmed the
        historical "weird errors" TODO no longer reproduces under current
        Pydantic 2.11 + numpydantic 1.6.
        """
        splitter = FoVTreePointCloudSplitter(new_tree)
        with pytest.raises(ValidationError):
            splitter.split("not a pcd")  # type: ignore[arg-type]

    def test_split_pc_with_fov_tree_validate_variables_negative_path(self, new_tree):
        """DEBT-08 / Phase 6 D-17 — free-function variant raises ValidationError on bad pcd."""
        with pytest.raises(ValidationError):
            split_pc_with_fov_tree("not a pcd", new_tree)  # type: ignore[arg-type]

    # --- PERF-03 D-29 #1-#4 equivalence + #5 auto-dispatch threshold -----

    @pytest.mark.parametrize("prefer", ["serial", "processes", "auto"])
    def test_direct_split_equivalence_small_tree(self, pcd_, prefer):
        """`_direct_split` returns identical per-FoV xyz under every ``prefer``.

        Small (3-FoV) tree exercises the sub-threshold path under
        ``prefer="auto"`` (3 <= _SERIAL_THRESHOLD = 16).
        """
        tree = _build_n_fov_tree(3)
        # Use a fresh copy per dispatch path because `_direct_split` calls
        # `pcd.reduce(np.zeros(...))` and would zero out the shared fixture.
        pcd_serial = pcd_.copy()
        pcd_run = pcd_.copy()

        ref = FoVTreePointCloudSplitter(tree, prefer="serial", n_jobs=1)._direct_split(pcd_serial, tree)
        out = FoVTreePointCloudSplitter(tree, prefer=prefer, n_jobs=1)._direct_split(pcd_run, tree)

        assert set(out.keys()) == set(ref.keys())
        for key in ref:
            assert np.array_equal(out[key].xyz, ref[key].xyz), f"xyz mismatch on FoV {key} for prefer={prefer}"

    @pytest.mark.parametrize("prefer", ["serial", "processes", "auto"])
    def test_direct_split_equivalence_larger_tree(self, pcd_, prefer):
        """`_direct_split` returns identical per-FoV xyz on a larger (20-FoV) tree.

        20 > _SERIAL_THRESHOLD = 16 — exercises the above-threshold auto
        path (parallel) alongside explicit ``serial`` / ``processes``.
        """
        tree = _build_n_fov_tree(20)
        pcd_serial = pcd_.copy()
        pcd_run = pcd_.copy()

        ref = FoVTreePointCloudSplitter(tree, prefer="serial", n_jobs=1)._direct_split(pcd_serial, tree)
        out = FoVTreePointCloudSplitter(tree, prefer=prefer, n_jobs=1)._direct_split(pcd_run, tree)

        assert set(out.keys()) == set(ref.keys())
        for key in ref:
            assert np.array_equal(out[key].xyz, ref[key].xyz), f"xyz mismatch on FoV {key} for prefer={prefer}"

    @pytest.mark.parametrize("prefer", ["serial", "processes", "auto"])
    def test_iterative_split_equivalence_small_tree(self, pcd_, prefer):
        """`_iterative_split` matches across ``prefer`` on a small (3-FoV) tree."""
        tree = _build_n_fov_tree(3)
        pcd_serial = pcd_.copy()
        pcd_run = pcd_.copy()

        ref = FoVTreePointCloudSplitter(tree, prefer="serial", n_jobs=1)._iterative_split(pcd_serial, tree)
        out = FoVTreePointCloudSplitter(tree, prefer=prefer, n_jobs=1)._iterative_split(pcd_run, tree)

        assert set(out.keys()) == set(ref.keys())
        for key in ref:
            assert np.array_equal(out[key].xyz, ref[key].xyz), f"xyz mismatch on FoV {key} for prefer={prefer}"

    @pytest.mark.parametrize("prefer", ["serial", "processes", "auto"])
    def test_iterative_split_equivalence_larger_tree(self, pcd_, prefer):
        """`_iterative_split` matches across ``prefer`` on a 20-FoV tree (> threshold)."""
        tree = _build_n_fov_tree(20)
        pcd_serial = pcd_.copy()
        pcd_run = pcd_.copy()

        ref = FoVTreePointCloudSplitter(tree, prefer="serial", n_jobs=1)._iterative_split(pcd_serial, tree)
        out = FoVTreePointCloudSplitter(tree, prefer=prefer, n_jobs=1)._iterative_split(pcd_run, tree)

        assert set(out.keys()) == set(ref.keys())
        for key in ref:
            assert np.array_equal(out[key].xyz, ref[key].xyz), f"xyz mismatch on FoV {key} for prefer={prefer}"

    def test_auto_dispatch_threshold(self, pcd_, monkeypatch):
        """Below `_SERIAL_THRESHOLD=16`, ``prefer="auto"`` MUST NOT invoke joblib.Parallel.

        Monkeypatches `joblib.Parallel.__call__` to record invocations; asserts
        zero invocations on a 5-FoV tree and ≥1 invocation on a 20-FoV tree.
        Locks in the conservative dispatch contract for PERF-03 D-26 / D-28.
        """
        call_count = {"n": 0}
        real_call = joblib.Parallel.__call__

        def _count(self, *args, **kwargs):
            call_count["n"] += 1
            return real_call(self, *args, **kwargs)

        monkeypatch.setattr(joblib.Parallel, "__call__", _count)
        # Module-level alias used inside splitter.py also resolves to the
        # same class; the monkeypatch above patches both.

        # Below threshold (5 ≤ 16): no joblib invocation expected.
        small_tree = _build_n_fov_tree(5)
        pcd_small = pcd_.copy()
        FoVTreePointCloudSplitter(small_tree, prefer="auto", n_jobs=1)._direct_split(pcd_small, small_tree)
        assert call_count["n"] == 0, f"prefer='auto' on 5-FoV tree should not invoke Parallel (got {call_count['n']})"

        # Above threshold (20 > 16): joblib MUST be invoked at least once.
        call_count["n"] = 0
        large_tree = _build_n_fov_tree(20)
        pcd_large = pcd_.copy()
        FoVTreePointCloudSplitter(large_tree, prefer="auto", n_jobs=1)._direct_split(pcd_large, large_tree)
        assert call_count["n"] >= 1, f"prefer='auto' on 20-FoV tree should invoke Parallel (got {call_count['n']})"

    def test_prefer_serial_never_invokes_parallel(self, pcd_, monkeypatch):
        """`prefer="serial"` must NOT call joblib.Parallel regardless of tree size.

        Locks in the explicit escape-hatch contract: even with a tree well
        above ``_SERIAL_THRESHOLD``, ``prefer="serial"`` forces the inline
        loop. Companion to ``test_auto_dispatch_threshold``.
        """
        call_count = {"n": 0}
        real_call = joblib.Parallel.__call__

        def _count(self, *args, **kwargs):
            call_count["n"] += 1
            return real_call(self, *args, **kwargs)

        monkeypatch.setattr(joblib.Parallel, "__call__", _count)

        large_tree = _build_n_fov_tree(32)  # well above threshold
        pcd_large = pcd_.copy()
        FoVTreePointCloudSplitter(large_tree, prefer="serial", n_jobs=1)._direct_split(pcd_large, large_tree)
        assert call_count["n"] == 0, (
            f"prefer='serial' on 32-FoV tree should not invoke Parallel (got {call_count['n']})"
        )

    def test_prefer_default_is_auto(self, new_tree):
        """`prefer` defaults to ``"auto"`` per D-27."""
        s = FoVTreePointCloudSplitter(new_tree)
        assert s.prefer == "auto"

    def test_prefer_forwarded_to_free_function(self, pcd_, monkeypatch):
        """`split_pc_with_fov_tree(..., prefer="serial")` skips joblib.Parallel.

        Companion regression test for D-27 — the free function must honour
        the same ``prefer`` kwarg as the class.
        """
        call_count = {"n": 0}
        real_call = joblib.Parallel.__call__

        def _count(self, *args, **kwargs):
            call_count["n"] += 1
            return real_call(self, *args, **kwargs)

        monkeypatch.setattr(joblib.Parallel, "__call__", _count)

        tree = _build_n_fov_tree(20)
        split_pc_with_fov_tree(pcd_.copy(), tree, prefer="serial")
        assert call_count["n"] == 0, (
            f"split_pc_with_fov_tree(prefer='serial') should not invoke Parallel (got {call_count['n']})"
        )


def test_split_pc_with_fov_tree(pcd_, new_tree):
    result = split_pc_with_fov_tree(pcd_, new_tree)
