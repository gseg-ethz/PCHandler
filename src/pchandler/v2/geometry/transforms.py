"""
Transforms module for pchandler.geometry.

Provides helper functions for transforming point clouds and converting between coordinate systems.
"""
from __future__ import annotations

import copy
from typing import Optional, overload, MutableMapping
from collections import OrderedDict


import numpy    as np
from pydantic   import BaseModel, model_validator, ValidationError, ConfigDict, validate_call

from ..base_arrays import Array_4x4_T, AffineArray, BaseVector, Array_3x3_T

class Transform(AffineArray):
    @classmethod
    def from_translation(cls, vector: BaseVector) -> Transform:
        return cls.generate(translation=vector)

    @classmethod
    def from_rotation(cls, matrix: Array_3x3_T) -> Transform:
        return cls.generate(rotation=matrix)

    @classmethod
    def from_matrix(cls, matrix: Array_4x4_T) -> Transform:
        return cls(arr=matrix)

    @classmethod
    def generate(cls, rotation=None, translation=None, scale=None):
        """
        Takes the form x0 = (T @ R * s + t) @ x1

        Rotation  Translation     Scale:
        | R 0 |     | I t |      | s 0 |
        | 0 1 |     | 0 1 |      | 0 1 |
        """
        if rotation is None and translation is None and scale is None:
            raise ValueError('TransformRecord must have at least one of the arguments')

        affine = np.eye(4)
        if rotation:
            affine[:3, :3] @= rotation

        if translation:
            affine[:3, 3] += translation

        if scale:
            affine[np.eye(3)] *= scale

        return cls.from_matrix(affine)

    def as_record(self, forward=True):
        if forward:
            return TransformRecord(forward=self.arr)
        return TransformRecord(backward=self.arr)

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

    def __getitem__(self, key: str|int) -> TransformRecord|tuple[str, TransformRecord]:
        if isinstance(key, int):
            return list(super(TransformLedger, self).items())[key]
        return super(TransformLedger, self).__getitem__(key)

    def __setitem__(self, key: str|int, value: np.ndarray|AffineArray|TransformRecord):
        if isinstance(value, TransformRecord):
            if isinstance(key, int):
                key = list(super(TransformLedger, self).items())[key]
            if key in super(TransformLedger, self).keys():
                super(TransformLedger, self).__setitem__(key, value)

            # Create a new record appended with the number entry if entry name doesn't exist
            super(TransformLedger, self).__setitem__(f'{key}_{len(self)}', value)
        else:
            raise ValueError(f'Cannot set transform record value of type: {type(value)}')

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
        (forward := np.eye(4))[:3, 3] = -self.arr
        (backward := np.eye(4))[:3, 3] = self.arr
        return TransformRecord(forward=forward, backward=backward)
