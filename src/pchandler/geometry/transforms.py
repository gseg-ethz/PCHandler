"""
Transforms module for pchandler.geometry.

Provides helper functions for transforming point clouds and converting between coordinate systems.
"""

import numpy as np
from numpy.typing import NDArray

from pchandler.core import PointCloudData


def transform_point_cloud(pcd: PointCloudData, transformation_matrix: np.ndarray) -> None:
    """
    Applies a 4x4 transformation matrix to the given point cloud.

    Parameters
    ----------
    pcd : PointCloudData
        The point cloud to transform.
    transformation_matrix : np.ndarray
        A (4 x 4) transformation matrix.
    """
    pcd.transform(transformation_matrix)


def translate(pcd: PointCloudData, translation: NDArray[np.floating]) -> PointCloudData:
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 3] = translation
    return pcd.transform(transformation_matrix)


def scale(pcd: PointCloudData, scale: float) -> PointCloudData:
    scale_matrix = np.eye(4)
    scale_matrix[:3, 3] = scale
    return pcd.transform(scale_matrix)


# Todo: Add rotation and more complex

# def rotate(pcd: PointCloudData, rotation: np.ndarray) -> PointCloudData:
