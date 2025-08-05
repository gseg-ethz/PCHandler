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


@pytest.fixture(scope="function", autouse=True)
def pcd_() -> PointCloudData:
    return PointCloudData(
        np.random.rand(100, 3) * 100,
        intensity=np.random.randint(-200, 200, (100,), dtype=np.int16),
        rgb=np.random.randint(0, 256, (100, 3), dtype=np.uint8),
        normals=np.random.rand(100, 3),
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
        # TODO this is throwing warnings
        pcd_original = pcd_.copy()
        iterative_splitter = FoVTreePointCloudSplitter(new_tree, method="iterative")
        direct_splitter = FoVTreePointCloudSplitter(new_tree, method="direct", n_jobs=1)

        splits_1 = iterative_splitter.split(pcd_)
        merged_pcd = PointCloudData.merge(*[v for v in splits_1.values()])
        merged_pcd = PointCloudData.merge(pcd_, merged_pcd)

        assert len(pcd_original) == len(merged_pcd)
        assert np.allclose(pcd_original.unshifted_bbox, merged_pcd.unshifted_bbox)

        splits_2 = direct_splitter.split(pcd_)

        # TODO add tests supporting the direct splitter
        with pytest.raises(ValueError):
            for k, v in splits_1.items():
                assert np.allclose(v.xyz, splits_2[k].xyz)
                assert np.allclose(v.rgb, splits_2[k].rgb)
                assert np.allclose(v.intensity, splits_2[k].intensity)
                assert np.allclose(v.normals, splits_2[k].normals)

    def test_invalid_split_mode(self, pcd_, new_tree):
        pcd_original = pcd_.copy()
        with pytest.raises(ValidationError):
            iterative_splitter = FoVTreePointCloudSplitter(new_tree, method="new_moe")
            splits_1 = iterative_splitter.split(pcd_)

        with pytest.raises(ValueError):
            check_number_jobs(0)


def test_split_pc_with_fov_tree(pcd_, new_tree):
    result = split_pc_with_fov_tree(pcd_, new_tree)
