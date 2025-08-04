import pytest

import numpy as np

from pchandler.geometry.core import PointCloudData
from pchandler.geometry.coordinates import rhv2xyz

from pchandler.filters.outlier_filter import SphericalOutlierFilter, CartesianOutlierFilter


class TestSphericalOutlierFilter:
    def test_spherical_outlier_filter(self):
        r = np.random.rand(100_000) * 10 + 20
        h = np.linspace(-0.1, 1.5, 100_000)
        v = np.linspace(0.1, 0.3, 100_000)

        num_outliers = 100
        r_out = np.random.rand(num_outliers) * 200 + 34
        h_out = np.random.rand(num_outliers) * 1.1 - 1.8
        v_out = np.random.rand(num_outliers) + 1.3
        rhv = np.stack([r, h, v], axis=1)
        rhv_out = np.stack([r_out, h_out, v_out], axis=1)
        main_cluster = rhv2xyz(rhv)
        xyz_out = rhv2xyz(rhv_out)
        points_w_outliers = np.vstack((main_cluster, xyz_out))

        xyz = PointCloudData(
            points_w_outliers,
            numerical_optimization_shift=None
        )

        filter_spherical = SphericalOutlierFilter(std_ratio=0.95, number_of_neighbours=3)
        filtered_pcd = filter_spherical.extract(xyz)

        assert len(filtered_pcd) == len(main_cluster)
        assert np.allclose(filtered_pcd.xyz, main_cluster)

        assert len(xyz) == num_outliers
        assert np.allclose(xyz, xyz_out)

class TestCartesianOutlierFilter:
    def test_cartesian_outlier_filter(self):
        main_cluster = np.random.rand(100000, 3) * 10
        num_outliers = 100  # Number of outliers
        outliers = np.random.uniform(low=-50, high=50, size=(num_outliers, 3))

        points_with_outliers = np.vstack((main_cluster, outliers))

        xyz = PointCloudData(
            points_with_outliers,
            numerical_optimization_shift=None
        )

        filter_spherical = CartesianOutlierFilter(std_ratio=0.95, number_of_neighbours=3)
        filtered_pcd = filter_spherical.extract(xyz)

        assert len(filtered_pcd) == len(main_cluster)
        assert np.allclose(filtered_pcd.xyz, main_cluster)

        assert len(xyz) == num_outliers
        assert np.allclose(xyz, outliers)