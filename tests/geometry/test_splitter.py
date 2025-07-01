import pytest

import numpy as np
from pydantic import ValidationError

from pchandler.v2.geometry import PointCloudData
from pchandler.v2.geometry.fov import FoV, FoVTree
from pchandler.v2.geometry.splitter import FoVTreePointCloudSplitter, split_pc_with_fov_tree, PointCloudSplitter


pcd = PointCloudData(np.random.rand(100,3)*100,
                          intensity=np.random.randint(-200, 200, (100,), dtype=np.int16),
                          rgb=np.random.randint(0, 256, (100,3), dtype=np.uint8),
                          normals=np.random.rand(100,3))

@pytest.fixture(scope='function')
def new_fov() -> FoV:
    return FoV(left=0.3, top=0.4, right=1.3, bottom=2.4)

@pytest.fixture(scope='function')
def tile_extent() -> FoV:
    return FoV(left=0, right=0.1, top=1.3, bottom=1.4)

@pytest.fixture(scope='function')
def new_tree(new_fov, tile_extent) -> FoVTree:
    return FoVTree.build_from_tiles(new_fov.tile(tile_extent))


class TestAbstractPcdSplitter:
    def test_abstract_methods(self):
        assert hasattr(PointCloudSplitter, 'split')


class TestFoVTreePointCloudSplitter:
    def test_initialisation(self, new_tree):

        initialised_splitter = FoVTreePointCloudSplitter(new_tree)

        assert initialised_splitter.fov_tree is new_tree
        assert initialised_splitter.remove_empty is True
        assert initialised_splitter.n_jobs == -1

    def test_invalid_initialisations(self, new_tree):
        with pytest.raises(ValidationError):
            FoVTreePointCloudSplitter(new_tree, n_jobs=-2)

        with pytest.raises(ValidationError):
            FoVTreePointCloudSplitter(new_tree, n_jobs=100)

        with pytest.raises(TypeError):
            FoVTreePointCloudSplitter(new_tree, n_jobs="NotANumber")    # type: ignore

        with pytest.raises(ValidationError):
            FoVTreePointCloudSplitter(new_tree, remove_empty=123)       # type: ignore

        with pytest.raises(ValidationError):
            FoVTreePointCloudSplitter(new_tree, method=123)

        with pytest.raises(ValidationError):
            FoVTreePointCloudSplitter(new_tree, method='Not_valid')

    def test_split(self, new_tree):
        iterative_splitter = FoVTreePointCloudSplitter(new_tree, method='iterative')
        direct_splitter = FoVTreePointCloudSplitter(new_tree, method='direct')

        splits_1 = iterative_splitter.split(pcd)
        splits_2 = direct_splitter.split(pcd)

        for k, v in splits_1.items():
            assert np.allclose(v.xyz, splits_2[k].xyz)
            assert np.allclose(v.rgb, splits_2[k].rgb)
            assert np.allclose(v.intensity, splits_2[k].intensity)
            assert np.allclose(v.normals, splits_2[k].normals)
