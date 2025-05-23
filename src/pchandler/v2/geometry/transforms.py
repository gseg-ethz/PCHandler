"""
Transforms module for pchandler.geometry.

Provides helper functions for transforming point clouds and converting between coordinate systems.
"""
from __future__ import annotations
from typing import Optional, overload
import copy


import numpy as np
from numpy.typing import NDArray
from collections import OrderedDict, MutableMapping
from pydantic import BaseModel, model_validator, ValidationError, ConfigDict, validate_call


from .core import PointCloudData
from ..base_arrays import Array_4x4_T, Array4x4, BaseVector

class TransformRecord(BaseModel):
    forward: Optional[Array_4x4_T] = None
    backward: Optional[Array_4x4_T] = None

    @model_validator(mode='after')
    def update_matrices(self):
        if self.forward is None and self.backward is None:
            raise ValidationError('TransformRecord must receive at least one forward or backward transformation')

        if self.forward is None:
            self.forward = np.linalg.inv(self.backward)

        if self.backward is None:
            self.backward = np.linalg.inv(self.forward)


class TransformLedger(OrderedDict, MutableMapping   [str, TransformRecord]):
    def __int__(self):
        super(TransformLedger, self).__init__()

    @overload
    def __getitem__(self, index: int) -> tuple[str, TransformRecord]:
        ...

    @overload
    def __getitem__(self, index: str) -> TransformRecord:
        ...

    def __getitem__(self, key: str|int) -> TransformRecord|tuple[str, TransformRecord]:
        if isinstance(key, int):
            return list(super(TransformLedger, self).items())[key][1]
        return super(TransformLedger, self).__getitem__(key)

    def __setitem__(self, key: str|int, value: np.ndarray|Array4x4|TransformRecord):
        if isinstance(value, TransformRecord):
            if isinstance(key, int):
                key = list(super(TransformLedger, self).items())[key]
            if key in super(TransformLedger, self).keys():
                super(TransformLedger, self).__setitem__(key, value)

            # Create a new record appended with the number entry if entry name doesn't exist
            super(TransformLedger, self).__setitem__(f'{key}_{len(self)}', value)
        else:
            raise ValueError(f'Cannot set transform record value of type: {type(value)}')

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def rollback_record(self, index: int) -> tuple[np.ndarray, TransformLedger]:
        previous_history = type(self)()
        remaining_transforms: TransformLedger[str, TransformRecord] = copy.deepcopy(self)

        for i in range(0, index):
            name, record = remaining_transforms.popitem(last=False)
            previous_history[name] = record

        key, first_record = remaining_transforms.popitem(last=False)
        chained_transform: np.ndarray = first_record.backward
        for record in remaining_transforms.values():
            chained_transform @= record.backward

        return chained_transform, previous_history


class GlobalShift(BaseVector):
    model_config = ConfigDict(extra='forbid')

    def as_record(self):
        forward = np.eye(4); forward[:3, :3] = -self.arr
        backward = np.eye(4); backward[:3, :3] = self.arr
        return TransformRecord(forward=forward, backward=backward)



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
