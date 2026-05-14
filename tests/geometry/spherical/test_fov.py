import math

import numpy as np
import pytest
from GSEGUtils.constants import EPS, PI
from GSEGUtils.util import AngleUnit, _deg2rad, _rad2deg, _rad2gon
from pydantic import ValidationError

from pchandler import PointCloudData
from pchandler.geometry.spherical import Angle, FoV, FoVTree


@pytest.fixture(scope="function", autouse=True)
def pcd():
    return PointCloudData(np.random.rand(10000, 3))


@pytest.fixture(scope="function")
def fov_rad_values():
    return np.array([-PI / 2, PI / 2, PI / 8, (7 / 8) * PI])


@pytest.fixture(scope="function")
def fov_deg_values(fov_rad_values):
    return _rad2deg(fov_rad_values)


@pytest.fixture(scope="function")
def fov_gon_values(fov_rad_values):
    return _rad2gon(fov_rad_values)


@pytest.fixture(scope="function")
def fov_():
    """Minimal valid FoV used for deprecation-warning assertions (BUG-03)."""
    return FoV(
        left=Angle(0, unit=AngleUnit.RAD),
        right=Angle(1, unit=AngleUnit.RAD),
        top=Angle(0, unit=AngleUnit.RAD),
        bottom=Angle(1, unit=AngleUnit.RAD),
    )


class TestFov:
    def test_init(self, fov_rad_values, fov_deg_values, fov_gon_values):
        for unit, values in zip(
            (AngleUnit.DEGREE, AngleUnit.RAD, AngleUnit.GON), (fov_deg_values, fov_rad_values, fov_gon_values)
        ):
            left = Angle(values[0], unit=unit)
            right = Angle(values[1], unit=unit)
            top = Angle(values[2], unit=unit)
            bottom = Angle(values[3], unit=unit)
            fov = FoV(left=left, right=right, top=top, bottom=bottom)

            assert math.isclose(fov.left.radians, -PI / 2)
            assert math.isclose(fov.right.radians, PI / 2)
            assert math.isclose(fov.top.radians, PI / 8)
            assert math.isclose(fov.bottom.radians, (7 / 8) * PI)

        # Init from floats -> Assumed to be radians
        fov = FoV(
            left=float(fov_rad_values[0]),
            right=float(fov_rad_values[1]),
            top=float(fov_rad_values[2]),
            bottom=float(fov_rad_values[3]),
        )

        assert math.isclose(fov.left.radians, -PI / 2)
        assert math.isclose(fov.right.radians, PI / 2)
        assert math.isclose(fov.top.radians, PI / 8)
        assert math.isclose(fov.bottom.radians, (7 / 8) * PI)

        # Init from strings -> Coerces to Angle types
        fov = FoV(left="0deg", right="90deg", top="0rad", bottom="200gon")

        assert math.isclose(fov.left, Angle(0))
        assert math.isclose(fov.right, Angle(np.pi / 2))
        assert math.isclose(fov.top, Angle(0))
        assert math.isclose(fov.bottom, Angle(np.pi))

    def test_init_invalid_values(self):
        # Bottom under top
        with pytest.raises(ValueError):
            FoV(left=0, right=1, top=2, bottom=0)

        # Horizontal not in range
        with pytest.raises(ValueError):
            FoV(left=100, right=1, top=2, bottom=0)

        # Elevation angle not in range
        with pytest.raises(ValueError):
            FoV(left=0, right=1, top=100, bottom=200)

        # Type checks
        with pytest.raises(ValidationError):
            FoV(left=0, right=1, top=2, bottom={"200deg": 1000})

    def test_construct_without_bounds(self, fov_rad_values):
        fov = FoV.construct_without_bounds_check(left=100, right=300, top=200, bottom=199)

        assert fov.left.radians == 100
        assert fov.right.radians == 300
        assert fov.top.radians == 200
        assert fov.bottom.radians == 199

    def test_construct_without_bounds_invalid_inputs(self):
        with pytest.raises(ValueError):
            FoV.construct_without_bounds_check(left=100, right=200, top=200, bottom={"Not a": "Valid Angle"})

    def test_from_angles(self):
        # Case not crossing pi narrow
        hz = np.linspace(-0.2, 0.3, 1000, endpoint=True)
        v = np.linspace(0.2, 1.45, 1000, endpoint=True)

        fov = FoV.from_angles(horizontal=hz, vertical=v)

        assert np.isclose(fov.left, -0.2)
        assert np.isclose(fov.right, 0.3)
        assert np.isclose(fov.top, 0.2)
        assert np.isclose(fov.bottom, 1.45)

        # Case not crossing pi wide
        hz = np.linspace(-2.1, 1.74, 1000, endpoint=True)
        fov = FoV.from_angles(horizontal=hz, vertical=v)

        assert np.isclose(fov.left, -2.1)
        assert np.isclose(fov.right, 1.74)
        assert np.isclose(fov.top, 0.2)
        assert np.isclose(fov.bottom, 1.45)

        # Case crossing pi, narrow
        hz = np.concat((np.linspace(3.0, PI, 500, endpoint=False), np.linspace(-PI, -3.0, 500, endpoint=True)))
        fov = FoV.from_angles(horizontal=hz, vertical=v)
        assert np.isclose(fov.left, 3.0)
        assert np.isclose(fov.right, -3.0)

    def test_iter(self):
        fov = FoV(left=-0.12, top=0.45, right=1.23, bottom=2.78)

        expected_values = {"left": -0.12, "top": 0.45, "right": 1.23, "bottom": 2.78}

        for key, val in fov:
            assert expected_values[key] == val

    def test_crosses_pi(self):
        fov_crosses_pi = FoV(left=3.01, right=-2.8, top=0.3, bottom=2.78)
        assert fov_crosses_pi.crosses_pi

    def test_width(self):
        fov = FoV(top=0, bottom=2.45, right=1.3, left=0.2)
        assert fov.width() == 1.1

        left = PI - 1
        right = -PI + 1
        fov = FoV(top=0, bottom=2.45, left=left, right=right)
        assert fov.width() == 2

    def test_height(self):
        fov = FoV(top=0.4, bottom=2.45, right=1.3, left=0.2)
        assert math.isclose(fov.height(), 2.05)

    def test_extent(self):
        fov = FoV(top=0.4, bottom=2.45, right=1.3, left=0.2)
        extent = fov.extent()
        assert extent[0] == fov.width()
        assert extent[1] == fov.height()
        assert math.isclose(extent[0], 1.1)
        assert math.isclose(extent[1], 2.05)

    def test_center(self):
        fov = FoV(left=0.3, right=1.3, top=0.4, bottom=2.4)
        center = fov.center()
        assert math.isclose(center[0], 0.8)
        assert math.isclose(center[1], 1.4)

        # Test crossing pi
        fov = FoV(left=PI - 0.1, right=-PI + 0.3, top=0.4, bottom=2.4)
        center = fov.center()
        assert math.isclose(center[0], -PI + 0.1)
        assert math.isclose(center[1], 1.4)

        fov = FoV(left="170deg", right="-150deg", top=0.4, bottom=2.4)
        center = fov.center()
        assert math.isclose(center[0], _deg2rad(-170))
        assert math.isclose(center[1], 1.4)

    def test_from_center_with_extent(self):
        # Ignores bounds
        fov = FoV.from_center_with_extent(centerpoint=(0.2, 1.0), extent=(7, 10))
        assert math.isclose(fov.left, -3.3)
        assert math.isclose(fov.right, 3.7)
        assert math.isclose(fov.top, -4.0)
        assert math.isclose(fov.bottom, 6.0)

    def test_intersect(self):
        fov1 = FoV(top=0.4, bottom=2.4, right=1.3, left=0.3)
        fov2 = FoV(top=0.2, bottom=2.2, right=1.5, left=-1)

        intersect = fov1.intersect(fov2)

        assert isinstance(intersect, FoV)
        assert math.isclose(intersect.right, 1.3)
        assert math.isclose(intersect.left, 0.3)
        assert math.isclose(intersect.top, 0.4)
        assert math.isclose(intersect.bottom, 2.2)

    def test_ratio(self):
        fov = FoV(top=0, bottom=1 / 3, left=0, right=1)
        ratio = fov.ratio()
        assert isinstance(ratio, (float, int))
        assert math.isclose(ratio, 3)

    def test_extend_to_ratio(self):
        n = 10_000
        np.random.seed(42)
        top_samples = np.random.uniform(low=0.0, high=np.pi / 2, size=n)
        bottom_samples = np.random.uniform(low=np.pi / 2, high=np.pi, size=n)
        left_samples = np.random.uniform(low=-np.pi, high=0, size=n)
        right_samples = np.random.uniform(low=0, high=np.pi, size=n)
        ratios = np.random.randint(low=1, high=10, size=n) / np.random.randint(low=1, high=10, size=n)
        angle_samples = zip(left_samples, right_samples, top_samples, bottom_samples, ratios)

        for angles in angle_samples:
            fov = FoV(left=angles[0], right=angles[1], top=angles[2], bottom=angles[3])
            fov_extended = fov.extend_to_ratio(angles[4])

            assert math.isclose(fov_extended.ratio(), angles[4])
            assert fov_extended.width() - fov.width() >= -EPS and fov_extended.height() - fov.height() >= -EPS

        fov = FoV(left=1, right=2, top=1, bottom=1.2)
        extended = fov.extend_to_ratio(fov.ratio())
        assert fov is extended

    def test_split(self):
        fov = FoV(top="20deg", bottom=2.4, right=1.3, left="40gon")
        splits = fov.split((2, 4))

        assert len(splits) == 8
        # assert np.allclose([split.height() for split in splits])
        for i, split in enumerate(splits):
            assert math.isclose(split.width(), (fov.right - fov.left) / 2)
            assert math.isclose(split.height(), (fov.bottom - fov.top) / 4)
            if i > 0:
                if i == 4:
                    assert split.right > splits[i - 1].right
                    assert split.left > splits[i - 1].left

                if i % 4:
                    assert split.top > splits[i - 1].top
                    assert split.bottom > splits[i - 1].bottom

        splits = fov.split((1, 1))
        assert splits[0] is fov

    def test_equal_tiles(self):
        fov = FoV(top=0.1, bottom=1.2, right=0.7, left=0.4)

        tiles = fov.equal_tiles(height=0.4, width=0.2)
        assert len(tiles) == 6

    def test_tile(self):
        fov = FoV(top="0.4rad", bottom=2.4, right=1.3, left=0.3)
        fov_by_extent = FoV(left="0deg", top="0gon", right=0.2, bottom=0.2)

        tiles = fov.tile(fov_by_extent)

        assert len(tiles) * len(tiles[0]) == 50
        for rows in tiles:
            for i in rows:
                assert math.isclose(i.width(), 0.2)
                assert math.isclose(i.height(), 0.2)
                assert i.center() != (0, 0)

        fov = FoV(top="0.4rad", bottom=2.4, right=1.3, left=0.3)
        fov_by_extent = FoV(left="0deg", top="0gon", right=0.3, bottom=0.45)

        tiles = fov.tile(fov_by_extent, expand_to_integer_multiple=True)
        # Check left to right first
        assert len(tiles) == 4
        # Check top to bottom
        assert len(tiles[0]) == 5
        for rows in tiles:
            for i in rows:
                assert math.isclose(i.width(), 0.3)
                assert math.isclose(i.height(), 0.45)

    def test_quadrants(self):
        fov = FoV(top=0.4, bottom=2.4, right=1.3, left=0.3)
        splits = fov.split((2, 2))

        quadrants = fov.quadrants()

        for i, split in enumerate(splits):
            assert split == quadrants[i]

    def test_merge(self):
        fov1 = FoV(top=0.3, bottom=0.5, right=2.7, left=2.4)
        fov2 = FoV(top=0.3, bottom=0.5, right=2.6, left=2.1)
        fov3 = FoV(top=0.7, bottom=0.9, right=2.5, left=2.2)
        fov4 = FoV(top=0.8, bottom=1.5, right=2.5, left=0.9)

        fov = FoV.merge([fov1, fov2, fov3, fov4])
        assert math.isclose(fov.right, 2.7)
        assert math.isclose(fov.left, 0.9)
        assert math.isclose(fov.top, 0.3)
        assert math.isclose(fov.bottom, 1.5)

        for i in (fov2, fov3, fov4):
            fov1 = fov1.union(i)

        assert fov1 == fov

    def test_old_properties(self):
        fov = FoV(top=0.4, bottom=2.4, left=1.3, right=0.3)

        assert fov.horizontal_min == fov.left
        assert fov.elevation_min == fov.top
        assert fov.horizontal_max == fov.right
        assert fov.elevation_max == fov.bottom

    def test_str(self):
        fov = FoV(top=0.4, bottom=2.4, right=1.3, left=0.3)
        assert isinstance(str(fov), str)

    def test_repr(self):
        fov = FoV(top=0.4, bottom=2.4, right=1.3, left=0.3)
        print(fov)
        assert isinstance(repr(fov), str)
        assert repr(fov) == str(fov)
        assert repr(fov).startswith("FoV(")


@pytest.fixture(scope="function")
def new_fov() -> FoV:
    return FoV(left=0.3, top=0.4, right=1.3, bottom=2.4)


@pytest.fixture(scope="function")
def new_fov2() -> FoV:
    return FoV(left=0.5, top=0.6, right=1.7, bottom=2.2)


@pytest.fixture(scope="function")
def fov_center() -> tuple[float, float]:
    return 0.5, 0.7


@pytest.fixture(scope="function")
def fov_extent() -> tuple[float, float]:
    return 0.2, 1.2


@pytest.fixture(scope="function")
def new_extent_as_fov() -> FoV:
    return FoV(left=0.3, top=0.4, right=0.4, bottom=0.6)


class TestFoVTree:
    def test_initialisation(self, new_fov: FoV):
        new_tree = FoVTree.build_from_tiles(new_fov.tile(FoV(left=0.0, right=0.1, top=1.3, bottom=1.4)))
        assert isinstance(new_tree, FoVTree)
        assert len(new_tree.children) == 4


def test_fovtree_flat_tiles_unique_identifiers(fov_):
    """BUG-05: flat-tiles fallback produces unique identifiers across siblings."""
    # fov_ extent is 1x1 (left=0, right=1, top=0, bottom=1); a 0.5x0.5 target
    # tile yields a 2x2 grid -> total 4 children <= min_children=4 -> flat-tiles path.
    tiles = fov_.tile(FoV(left=0.0, right=0.5, top=0.0, bottom=0.5))
    tree = FoVTree.build_from_tiles(tiles, min_children=4)
    assert tree is not None
    ids = [ident for ident, _ in tree.to_list()]
    assert len(ids) == len(set(ids)), f"duplicate identifiers in flat-tiles tree: {ids}"


def test_fovtree_mixed_recursive_and_flat_unique_identifiers(fov_):
    """BUG-05: mixed recursive + flat-tiles paths produce globally unique identifiers."""
    # 4x4 grid (target tile 0.25x0.25 over a 1x1 fov_) yields 16 > min_children=4,
    # forcing the recursive quad-split; sub-quadrants (each 2x2 == 4 <= 4) hit the
    # flat-tiles fallback. Exercises both code paths in the same tree.
    tiles = fov_.tile(FoV(left=0.0, right=0.25, top=0.0, bottom=0.25))
    tree = FoVTree.build_from_tiles(tiles, min_children=4)
    assert tree is not None
    ids = [ident for ident, _ in tree.to_list()]
    assert len(ids) == len(set(ids)), f"duplicate identifiers in mixed tree: {ids}"


def test_fovtree_getitem_flat_tile_lookup(fov_):
    """CR-02: __getitem__ resolves the flat-tile 3-char ``"<r>-<c>"`` identifiers."""
    # 2x2 grid -> flat-tile branch; sibling ids are "0-0", "0-1", "1-0", "1-1".
    tiles = fov_.tile(FoV(left=0.0, right=0.5, top=0.0, bottom=0.5))
    tree = FoVTree.build_from_tiles(tiles, min_children=4)
    assert tree is not None
    assert tree.children is not None
    for key in ("0-0", "0-1", "1-0", "1-1"):
        assert tree[key] is tree.children[key]


def test_fovtree_getitem_nested_recursive_lookup(fov_):
    """CR-02: __getitem__ descends into nested ``"<r>-<c>"`` paths concatenated by build_from_tiles."""
    # 4x4 grid -> recursive quad-split at root, flat-tile branch in each
    # quadrant. Every leaf identifier is the concatenation of a 3-char
    # parent id + 3-char child id, e.g. ``"0-00-0"``.
    tiles = fov_.tile(FoV(left=0.0, right=0.25, top=0.0, bottom=0.25))
    tree = FoVTree.build_from_tiles(tiles, min_children=4)
    assert tree is not None
    # Materialise every leaf identifier and round-trip through __getitem__.
    for leaf_id, leaf_fov in tree.to_list():
        resolved = tree[leaf_id]
        # The leaf's stored identifier is the *full* concatenated path; verify
        # we landed on the same node by comparing identifier strings.
        assert resolved.identifier == leaf_id, (
            f"tree[{leaf_id!r}] resolved to identifier={resolved.identifier!r}, expected {leaf_id!r}"
        )
        assert resolved.node is leaf_fov


def test_fovtree_getitem_root_and_empty_passthrough(fov_):
    """CR-02: ``""`` and ``"root"`` and a leaf return self (back-compat with previous behaviour)."""
    tiles = fov_.tile(FoV(left=0.0, right=0.5, top=0.0, bottom=0.5))
    tree = FoVTree.build_from_tiles(tiles, min_children=4)
    assert tree is not None
    assert tree[""] is tree
    assert tree["root"] is tree
    # Leaf: children is None, any identifier returns self.
    leaf = tree["0-0"]
    assert leaf["whatever"] is leaf


def test_fov_check_elevation_uses_field_name_in_error():
    """IN-01: ``_check_elevation`` validator reports the offending field name.

    The validator is bound to both ``top`` and ``bottom``; previously it
    hard-coded "Top angle ..." for both, mis-attributing bottom-bound
    violations.  Use ``ValidationInfo.field_name`` to surface the correct
    label.
    """
    with pytest.raises(ValidationError) as bottom_err:
        FoV(left=0, right=1, top=1, bottom=-1)
    assert "Bottom angle" in str(bottom_err.value)

    with pytest.raises(ValidationError) as top_err:
        FoV(left=0, right=1, top=-1, bottom=1)
    assert "Top angle" in str(top_err.value)


def test_fovtree_getitem_unknown_segment_raises(fov_):
    """CR-02: unknown child segment raises KeyError with a contextual message."""
    tiles = fov_.tile(FoV(left=0.0, right=0.5, top=0.0, bottom=0.5))
    tree = FoVTree.build_from_tiles(tiles, min_children=4)
    assert tree is not None
    with pytest.raises(KeyError, match="No child '9-9' under"):
        _ = tree["9-9"]


class TestFoVTreeGetitem:
    """Regression tests pinning FoVTree.__getitem__ behaviour (API-03 / SC-3).

    Identifiers emitted by ``build_from_tiles`` are ``_``-separated between
    tree levels and ``<r>-<c>`` within a single level. The parser in
    ``__getitem__`` partitions on the first ``_`` to descend one level at a
    time, so arbitrary r/c widths (incl. multi-digit indices) resolve
    correctly. Quick task 260514-noz added the multi-digit regression after
    the original Phase 3 CR-02 4x4 grid failed to exercise the wide-grid path.
    """

    @pytest.fixture(scope="function")
    def tree(self, fov_: FoV) -> FoVTree:
        """Depth-2 FoVTree built from a 4x4 grid.

        A 0.25x0.25 target tile over the 1x1 fov_ produces 16 children
        (> min_children=4), forcing a recursive quad-split at the root.
        Each quadrant has 4 children (<= min_children=4), hitting the flat-tile
        branch. The resulting tree has depth 2 with leaf identifiers of the
        form "<root-child-id>_<leaf-id>" e.g. "0-0_0-0".
        """
        tiles = fov_.tile(FoV(left=0.0, right=0.25, top=0.0, bottom=0.25))
        result = FoVTree.build_from_tiles(tiles, min_children=4)
        assert result is not None
        return result

    def test_getitem_full_identifier(self, tree: FoVTree) -> None:
        """tree["0-0_0-0"] resolves to the leaf node with identifier "0-0_0-0"."""
        # "0-0" picks the first-level recursive quadrant; "0-0" picks its
        # flat-tile child -> the full path is "_"-joined: "0-0_0-0".
        resolved = tree["0-0_0-0"]
        assert resolved.identifier == "0-0_0-0"

    def test_getitem_root(self, tree: FoVTree) -> None:
        """tree["root"] returns the tree itself (back-compat shortcut)."""
        assert tree["root"] is tree

    def test_getitem_invalid(self, tree: FoVTree) -> None:
        """tree["nonexistent"] raises KeyError with a contextual message."""
        with pytest.raises(KeyError, match="No child"):
            _ = tree["nonexistent"]

    def test_getitem_multi_digit_tile_grid(self, fov_: FoV) -> None:
        """Wide grids (r or c >= 10) resolve correctly through __getitem__.

        Regression for the multi-digit identifier bug (quick task 260514-noz):
        a 12x12 flat-tile grid produces identifiers like "10-3", "11-11".
        Under the old 3-char fixed-stride parser, "11-3" was split as
        head="11-" / rest="3" and raised a misleading KeyError. Under the
        new "_"-aware parser, partition on "_" yields head="11-3" / rest="",
        which resolves directly to the matching child.
        """
        # 1/12 ≈ 0.08333 target tile over the 1x1 fov_ -> 12x12 = 144 tiles.
        # min_children=200 forces the flat-tile branch directly at the root,
        # so child identifiers are exactly "<r>-<c>" with r, c in [0, 11].
        step = 1.0 / 12
        tiles = fov_.tile(FoV(left=0.0, right=step, top=0.0, bottom=step))
        tree = FoVTree.build_from_tiles(tiles, min_children=200)
        assert tree is not None

        # Sanity: we are exercising the wide-grid path -> at least one child
        # identifier has a multi-digit segment.
        assert any(any(int(p) >= 10 for p in cid.split("-")) for cid in tree.children), (
            f"expected multi-digit child id; got {sorted(tree.children)}"
        )

        # Multi-digit lookup must succeed (this raised KeyError pre-fix).
        resolved = tree["11-3"]
        assert resolved.identifier == "11-3"

        # Adjacent single-digit lookup must also succeed.
        resolved_low = tree["3-3"]
        assert resolved_low.identifier == "3-3"

        # Unknown multi-digit segment still raises a contextual KeyError.
        with pytest.raises(KeyError, match="No child '12-0' under"):
            _ = tree["12-0"]


def _fov_are_equal(fov1: FoV, fov2: FoV):
    array_1 = np.array([fov1.left, fov1.right, fov1.top, fov1.bottom])
    array_2 = np.array([fov2.left, fov2.right, fov2.top, fov2.bottom])

    assert np.allclose(array_1, array_2)
    return True


class TestFoVUnion:
    def test_joint_both(self):
        fov1 = FoV(top="20deg", bottom="80deg", left="-40deg", right="80deg")
        fov2 = FoV(top="12deg", bottom="64deg", left="-20deg", right="130deg")
        expected = FoV(top="12deg", bottom="80deg", left="-40deg", right="130deg")
        union = fov1.union(fov2)
        assert _fov_are_equal(expected, union)

    def test_joint_first_fov_wraps(self):
        fov1 = FoV(top="20deg", bottom="80deg", left="160deg", right="-100deg")
        fov2 = FoV(top="12deg", bottom="64deg", left="-120deg", right="30deg")
        expected = FoV(top="12deg", bottom="80deg", left="160deg", right="30deg")
        union = fov1.union(fov2)
        assert _fov_are_equal(expected, union)

    def test_joint_second_fov_wraps(self):
        fov1 = FoV(top="20deg", bottom="80deg", left="50deg", right="160deg")
        fov2 = FoV(top="12deg", bottom="64deg", left="120deg", right="-130deg")
        expected = FoV(top="12deg", bottom="80deg", left="50deg", right="-130deg")
        union = fov1.union(fov2)
        assert _fov_are_equal(expected, union)

    def test_joint_both_wrap(self):
        fov1 = FoV(top="20deg", bottom="80deg", left="150deg", right="-160deg")
        fov2 = FoV(top="12deg", bottom="64deg", left="170deg", right="-140deg")
        expected = FoV(top="12deg", bottom="80deg", left="150deg", right="-140deg")
        union = fov1.union(fov2)
        assert _fov_are_equal(expected, union)

    def test_disjointed(self):
        fov1 = FoV(top="20deg", bottom="50deg", left="-140deg", right="-20deg")
        fov2 = FoV(top="80deg", bottom="120deg", left="30deg", right="120deg")
        expected = FoV(top="20deg", bottom="120deg", left="-140deg", right="120deg")
        union = fov1.union(fov2)
        assert _fov_are_equal(expected, union)

    def test_basic(self):
        fov1 = FoV(top="20deg", bottom=2.4, right=1.3, left=0.3)
        fov2 = FoV(top=0.2, bottom=2.2, right=1.5, left=-1)

        union = fov1.union(fov2)

        assert isinstance(union, FoV)
        assert math.isclose(union.right, 1.5)
        assert math.isclose(union.left, -1)
        assert math.isclose(union.top, 0.2)
        assert math.isclose(union.bottom, 2.4)
        # Not testing for crossing of PI / TWO_PI


class TestFoVIntersection:
    def test_overlap_hz_and_v(self):
        fov1 = FoV(top="20deg", bottom="80deg", left="-40deg", right="80deg")
        fov2 = FoV(top="12deg", bottom="64deg", left="-20deg", right="130deg")
        expected = FoV(top="20deg", bottom="64deg", left="-20deg", right="80deg")
        intersect = fov1.intersect(fov2)
        assert _fov_are_equal(expected, intersect)

    def test_joint_first_fov_wraps(self):
        fov1 = FoV(top="20deg", bottom="80deg", left="160deg", right="-100deg")
        fov2 = FoV(top="12deg", bottom="64deg", left="-120deg", right="30deg")
        expected = FoV(top="20deg", bottom="64deg", left="-120deg", right="-100deg")
        intersect = fov1.intersect(fov2)
        assert _fov_are_equal(expected, intersect)

    def test_joint_second_fov_wraps(self):
        fov1 = FoV(top="20deg", bottom="80deg", left="50deg", right="160deg")
        fov2 = FoV(top="12deg", bottom="64deg", left="120deg", right="-130deg")
        expected = FoV(top="20deg", bottom="64deg", left="120deg", right="160deg")
        intersect = fov1.intersect(fov2)
        assert _fov_are_equal(expected, intersect)

    def test_joint_both_wrap(self):
        fov1 = FoV(top="20deg", bottom="80deg", left="150deg", right="-160deg")
        fov2 = FoV(top="12deg", bottom="64deg", left="170deg", right="-140deg")
        expected = FoV(top="20deg", bottom="64deg", left="170deg", right="-160deg")
        intersect = fov1.intersect(fov2)
        assert _fov_are_equal(expected, intersect)

    def test_disjointed_vertically(self):
        fov1 = FoV(top="20deg", bottom="50deg", left="-140deg", right="-20deg")
        fov2 = FoV(top="80deg", bottom="120deg", left="-130deg", right="-120deg")
        intersect = fov1.intersect(fov2)
        assert intersect is None

    def test_disjointed_horizontally(self):
        fov1 = FoV(top="20deg", bottom="50deg", left="-140deg", right="-20deg")
        fov2 = FoV(top="10deg", bottom="120deg", left="5deg", right="20deg")
        intersect = fov1.intersect(fov2)
        assert intersect is None

    def test_intersect_both_hz_sides(self):
        fov1 = FoV(top="20deg", bottom="50deg", left="140deg", right="-140deg")
        fov2 = FoV(top="10deg", bottom="120deg", left="-160deg", right="160deg")

        with pytest.raises(ValueError):
            fov1.intersect(fov2)

    def test_intersect_full_circle_non_wrapping(self):
        fov1 = FoV(top="20deg", bottom="50deg", left="-140deg", right="-20deg")
        fov2 = FoV(top="10deg", bottom="120deg", left=-PI, right=PI)
        expected = FoV(top="20deg", bottom="50deg", left="-140deg", right="-20deg")
        intersect = fov1.intersect(fov2)
        assert _fov_are_equal(expected, intersect)

    def test_intersect_full_circle_wrapping(self):
        fov1 = FoV(top="20deg", bottom="50deg", left="140deg", right="-20deg")
        fov2 = FoV(top="10deg", bottom="120deg", left=-PI, right=PI)
        expected = FoV(top="20deg", bottom="50deg", left="140deg", right="-20deg")
        intersect = fov1.intersect(fov2)
        assert _fov_are_equal(expected, intersect)

    def test_basic(self):
        fov1 = FoV(top=0.4, bottom=2.4, left=0.3, right=1.3)
        fov2 = FoV(top=0.2, bottom=2.2, left=-1, right=1.5)

        intersect = fov1.intersect(fov2)

        assert isinstance(intersect, FoV)
        assert math.isclose(intersect.right, 1.3)
        assert math.isclose(intersect.left, 0.3)
        assert math.isclose(intersect.top, 0.4)
        assert math.isclose(intersect.bottom, 2.2)
        # Not testing for crossing of PI / TWO_PI


@pytest.mark.parametrize(
    "prop_name, expected_msg",
    [
        ("horizontal_min", "horizontal_min property has been deprecated. Please use the 'left' property"),
        ("horizontal_max", "horizontal_max property has been deprecated. Please use the 'right' property"),
        ("elevation_min", "elevation_min property has been deprecated. Please use the 'top' property"),
        ("elevation_max", "elevation_max property has been deprecated. Please use the 'bottom' property"),
    ],
)
def test_fov_deprecated_property_warning_message(fov_, prop_name, expected_msg):
    """BUG-03: each deprecated FoV property emits the correct warning text."""
    with pytest.warns(DeprecationWarning) as record:
        _ = getattr(fov_, prop_name)
    assert len(record) == 1
    assert str(record[0].message) == expected_msg
