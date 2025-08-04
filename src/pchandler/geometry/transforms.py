"""
Transforms module for pchandler.geometry.

Provides helper functions for transforming point clouds and converting between coordinate systems.
"""

from __future__ import annotations

from typing import Any, Self, Literal

import numpy as np
from pydantic import (
    Field,
)

from GSEGUtils.base_arrays import BaseArray, FixedLengthArray, NumericMixins
from GSEGUtils.base_types import Array_3x3_T, Array_4x4_T, Array_Float_T, Vector_3_T


class _TransformArray(NumericMixins):
    def __matmul__(self, other: Array_Float_T | BaseArray) -> Any:
        if isinstance(other, FixedLengthArray) and other.ndim > 1:  # Coordinates and PointCloudData
            return other.__rmatmul__(self)

        other = np.asarray(other)

        if self.shape == other.shape: # Same shaped objects treated like another transformation
            return super().__matmul__(other)

        else: # return numpy for numpy input
            return self.arr @ other

    def __imatmul__(self, other: Array_Float_T | BaseArray) -> Self:
        self.arr @= other
        return self


class _Transform3x3(_TransformArray):
    arr: Array_3x3_T = Field(default_factory=lambda: np.eye(3))


class _Transform4x4(_TransformArray):
    arr: Array_4x4_T = Field(default_factory=lambda: np.eye(4))


class Transform(_Transform4x4):

    def __init__(self, arr: Array_Float_T | Array_4x4_T | Self, **kwargs):
        super().__init__(arr=arr, **kwargs)

    @classmethod
    def from_translation(cls, vector: Vector_3_T) -> Transform:
        return cls._generate(vector, mode='translate')

    @classmethod
    def from_rotation(cls, matrix: Array_3x3_T) -> Transform:
        return cls._generate(matrix, mode='rotate')

    @classmethod
    def from_affine(cls, matrix: Array_4x4_T) -> Transform:
        return cls._generate(matrix, mode='affine')

    @classmethod
    def from_scale(cls, vector: Vector_3_T|float) -> Transform:
        return cls._generate(vector, mode='scale')

    @classmethod
    def generate(cls,
                 rotation: Array_3x3_T = np.eye(3),
                 translation: Vector_3_T = np.zeros(3),
                 scale: Vector_3_T|float = 1, ):
        """
                Takes the form x0 = (R * s + t) @ x1

                Rotation  Translation     Scale:
                | R 0 |     | I t |      | s 0 |
                | 0 1 |     | 0 1 |      | 0 1 |
                """
        affine = np.eye(4).astype(np.float32)
        affine[:3, :3] = rotation.astype(np.float32)
        affine[:3, 3] = translation.astype(np.float32)
        affine[np.diag_indices(3)] *= scale
        return cls(affine)

    @classmethod
    def _generate(cls, values=Array_Float_T, mode: Literal['translate', 'rotate', 'scale', 'affine'] = 'affine', ):

        affine = np.eye(4)
        values = np.asarray(values)
        if mode == 'translate':
            affine[:3, 3] += values
        elif mode == 'rotate':
            affine[:3, :3] @= values
        elif mode == 'scale':
            affine[np.diag_indices(3)] *= values
        elif mode == 'affine':
            affine = values
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return cls(affine)

#     def as_record(self, forward=True):
#         if forward:
#             return TransformRecord(forward=self.arr)
#         return TransformRecord(backward=self.arr)
#
#
# class TransformRecord(BaseModel):
#     forward: Optional[Array_4x4_T] = None
#     backward: Optional[Array_4x4_T] = None
#
#     @model_validator(mode="after")
#     def update_matrices(self):
#         if self.forward is None and self.backward is None:
#             raise ValidationError("TransformRecord must receive at least one forward or backward transformation")
#
#         if self.forward is None:
#             self.forward = np.linalg.inv(self.backward)
#
#         if self.backward is None:
#             self.backward = np.linalg.inv(self.forward)


# class TransformLedger(OrderedDict[str, TransformRecord], MutableMapping[str, TransformRecord]):
#     def __int__(self):
#         super(TransformLedger, self).__init__()
#
#     def __getitem__(self, key: str | int) -> tuple[str, TransformRecord]:   # type: ignore[override]
#         """Use either an index or named key to access a record"""
#         if isinstance(key, int):
#             return list(super(TransformLedger, self).items())[key]
#
#         return key, super(TransformLedger, self).__getitem__(key)
#
#     def __setitem__(self, key: str | int, value: np.ndarray | _Transform4x4 | TransformRecord):
#         if isinstance(value, TransformRecord):
#             if isinstance(key, int):
#                 key = list(super(TransformLedger, self).items())[key]
#             if key in super(TransformLedger, self).keys():
#                 super(TransformLedger, self).__setitem__(key, value)
#
#             # Create a new record appended with the number entry if entry name doesn't exist
#             super(TransformLedger, self).__setitem__(f"{key}_{len(self)}", value)
#         else:
#             raise ValueError(f"Cannot set transform record value of type: {type(value)}")
#
#     def rollback_record(self, index: int) -> tuple[np.ndarray, TransformLedger]:
#         previous_history = type(self)()
#         remaining_transforms: TransformLedger[str, TransformRecord] = copy.deepcopy(self)
#
#         for i in range(0, index):
#             name, record = remaining_transforms.popitem(last=False)
#             previous_history[name] = record
#
#         key, first_record = remaining_transforms.popitem(last=False)
#         chained_transform: np.ndarray = first_record.backward
#         for record in remaining_transforms.values():
#             chained_transform @= record.backward
#
#         return chained_transform, previous_history


