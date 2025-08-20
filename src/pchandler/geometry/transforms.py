"""
Transforms module for pchandler.geometry.

Provides helper functions for transforming point clouds and converting between coordinate systems.
"""

import numpy as np
from numpy.typing import NDArray
from .core import PointCloudData
from typing import Literal

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


def toggle_socs2prcs(pcd: PointCloudData) -> PointCloudData:
    
    if pcd.xyz_is_prcs == True:
        print("Point cloud already in PRCS, you called transformation that was unecessary.")
    else:
        # Transform pcd.xyz and pcd.spherical_coordinates_origin from SOCS to PRCS
        transformation_matrix = pcd.tmat_socs2prcs
        pcd.transform(transformation_matrix)
        object.__setattr__(pcd, 'xyz_is_prcs', True)
        object.__setattr__(pcd, 'tmat_socs2prcs', transformation_matrix)
    return pcd


def toggle_prcs2socs(pcd: PointCloudData) -> PointCloudData:

    if pcd.xyz_is_prcs == False:
        print("Point cloud already in SOCS, you called transformation that was unecessary.")
    else:
        # Transform pcd.xyz and pcd.spherical_coordinates_origin from SOCS to PRCS
        transformation_matrix = pcd.tmat_socs2prcs
        transformation_matrix_inv = np.linalg.inv(transformation_matrix)
        pcd.transform(transformation_matrix_inv)
        object.__setattr__(pcd,'xyz_is_prcs',False)
        object.__setattr__(pcd, 'tmat_socs2prcs', transformation_matrix)
    return pcd


def modify_tmat_socs2prcs(pcd: PointCloudData, transformation_matrix: np.ndarray, mode: Literal['chain', 'replace'] = 'chain') -> PointCloudData:
    if mode == 'chain':
        chained_transformation = transformation_matrix @ pcd.tmat_socs2prcs
        pcd.__setattr__(pcd, 'tmat_socs2prcs', chained_transformation)
        if pcd.xyz_is_prcs:
            object.transform(transformation_matrix)
    elif mode == 'replace':
        object.__setattr__(pcd, 'tmat_socs2prcs', transformation_matrix)
        if pcd.xyz_is_prcs:
            tmat_diff = transformation_matrix @ np.linalg.inv(pcd.tmat_socs2prcs)
            pcd.transform(tmat_diff)

    return pcd

def lazy_global_shift_change(pcd:PointCloudData, new_global_shift = np.ndarray) -> PointCloudData:
    diff_global_shift = pcd.global_coordinate_shift - new_global_shift
    shifted_xyz = pcd.xyz + diff_global_shift
    shifted_sco = pcd.spherical_coordinates_origin + diff_global_shift
    object.__setattr__(pcd, 'xyz', shifted_xyz)
    object.__setattr__(pcd, 'global_coordinate_shift', new_global_shift)
    object.__setattr__(pcd, 'spherical_coordinates_origin', shifted_sco)
    return pcd



#TODO (TOMISLAV) - check interactions with NiMe's transformation functions:
#   a) if apply_math_to_xyz() and transformation_matrix is not None -> warning ("Broken, set to None")
#   b) if transform(pcd, 4x4) applied and transformation_matrix is not None -> warning ("Broken, set to None, if want to keep transform, modify transform_mat")

# Existing prototypes in: \\wsl.localhost\Ubuntu-22.04\scratch\projects\old_projects_and_dependencies\PCHandler\src\pchandler\geometry.py
#                          lines 231 - 239


