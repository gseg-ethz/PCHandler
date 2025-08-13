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
    """
    Represents a transformation array with special handling for matrix multiplication.

    Enables advanced matrix multiplication features, particularly for specific array types like
    FixedLengthArray or numpy arrays. Provides support for both standard and in-place matrix
    multiplication.

    Parameters
    ----------
    arr : Array_Float_T | BaseArray
        Internal array storing the values to be transformed.
    shape : tuple
        Tuple representing the shape of the array.
    """
    def __matmul__(self, other: Array_Float_T | BaseArray) -> Any:
        """
        Perform matrix multiplication.

        This method supports matrix multiplication between the current object
        and another object or array-like structure. It handles objects of the same
        shape or different shapes appropriating compatibility checks.

        Parameters
        ----------
        other : Array_Float_T or BaseArray
            The other array-like or compatible type for matrix multiplication.

        Returns
        -------
        Any
            Resulting product of the matrix multiplication. The returned type may
            vary depending on the type of the input (e.g., numpy array or object-specific type).
        """
        if isinstance(other, FixedLengthArray) and other.ndim > 1:  # Coordinates and PointCloudData
            return other.__rmatmul__(self)

        other = np.asarray(other)

        if self.shape == other.shape: # Same shaped objects treated like another transformation
            return super().__matmul__(other)

        else: # return numpy for numpy input
            return self.arr @ other

    def __imatmul__(self, other: Array_Float_T | BaseArray) -> Self:
        """
        Performs in-place matrix multiplication.

        The method updates the existing object by performing matrix multiplication
        with another array or compatible object.

        Parameters
        ----------
        other : Array_Float_T or BaseArray
            The array or compatible object to be used in the matrix multiplication.

        Returns
        -------
        Self
            The updated object after in-place matrix multiplication.
        """
        self.arr @= other
        return self


class _Transform3x3(_TransformArray):
    """
    Represents a 3x3 transformation array.

    This class encapsulates a 3x3 transformation matrix and provides functionalities
    to work with such matrices. It serves as a fundamental representation for
    various kinds of 3x3 transformations in numerical computations.

    Parameters
    ----------
    arr : Array_3x3_T
        The 3x3 transformation matrix, initialized to the identity matrix by
        default.
    """
    arr: Array_3x3_T = Field(default_factory=lambda: np.eye(3))


class _Transform4x4(_TransformArray):
    """
    Represents a 4x4 transformation matrix.

    This class encapsulates a 4x4 transformation matrix, which is commonly
    used in geometric transformations, including translations, rotations,
    scaling, and projections. The default matrix is an identity matrix.

    Parameters
    ----------
    arr : Array_4x4_T
        A 4x4 matrix representing the transformation. Default is an
        identity matrix.
    """
    arr: Array_4x4_T = Field(default_factory=lambda: np.eye(4))


class Transform(_Transform4x4):
    """
    Represents a 4x4 transformation matrix for operations like translation, rotation,
    scaling, and affine transformations.

    Provides methods for creating transformations based on translation vectors, rotation
    matrices, scaling factors, and affine matrices.

    Parameters
    ----------
    arr : Array_Float_T | Array_4x4_T | Self
        Underlying data representing the transformation matrix.
    """

    def __init__(self, arr: Array_Float_T | Array_4x4_T | Self, **kwargs):
        """
        Initializes an instance of the class with the given array and additional keyword arguments.

        Parameters
        ----------
        arr : Array_Float_T or Array_4x4_T or Self
            Input array or object used for initialization.
        kwargs : dict
            Additional options or settings for initialization.
        """
        super().__init__(arr=arr, **kwargs)

    @classmethod
    def from_translation(cls, vector: Vector_3_T) -> Transform:
        """
        Creates a transform object based on a translation vector.

        This class method generates a transformation that represents a translation
        operation, using the given translation vector as input. The implementation
        is designed to abstract the generation process internally.

        Parameters
        ----------
        vector : Vector_3_T
            The 3D vector representing the translation magnitude for each axis.

        Returns
        -------
        Transform
            A new Transform object representing the translation.
        """
        return cls._generate(vector, mode='translate')

    @classmethod
    def from_rotation(cls, matrix: Array_3x3_T) -> Transform:
        """
        Creates a Transform object using a given rotation matrix.

        Parameters
        ----------
        matrix : Array_3x3_T
            A 3x3 rotation matrix.

        Returns
        -------
        Transform
            A new Transform object initialized with the rotation matrix.
        """
        return cls._generate(matrix, mode='rotate')

    @classmethod
    def from_affine(cls, matrix: Array_4x4_T) -> Transform:
        """
        Creates a Transform instance from a 4x4 affine matrix.

        Parameters
        ----------
        matrix : Array_4x4_T
            A 4x4 transformation matrix that represents the affine transform.

        Returns
        -------
        Transform
            A Transform object created using the provided affine matrix.
        """
        return cls._generate(matrix, mode='affine')

    @classmethod
    def from_scale(cls, vector: Vector_3_T|float) -> Transform:
        """
        Create a Transform instance based on scaling by the provided vector.

        This method generates a Transformation object that applies scaling transformations
        determined by the specified vector or scalar factor.

        Parameters
        ----------
        vector : Vector_3_T or float
            The scaling factor(s). If a scalar (float) is provided, uniform scaling
            is applied to all dimensions. If a 3D vector is provided, non-uniform scaling
            is applied based on its components.

        Returns
        -------
        Transform
            A Transform object representing the scaling transformation.
        """
        return cls._generate(vector, mode='scale')

    @classmethod
    def generate(cls,
                 rotation: Array_3x3_T = np.eye(3),
                 translation: Vector_3_T = np.zeros(3),
                 scale: Vector_3_T|float = 1, ):
        """
        Generates an affine transformation matrix based on the given rotation,
        translation, and scale parameters. The output matrix follows the form:
        x0 = (R * s + t) @ x1, where R is the rotation, s is the scale, and t
        is the translation.

        Parameters
        ----------
        rotation : Array_3x3_T, optional
            A 3x3 rotation matrix. Defaults to the identity matrix.
        translation : Vector_3_T, optional
            A 3-dimensional translation vector. Defaults to a zero vector.
        scale : Vector_3_T or float, optional
            Scale factor(s) applied to the axes. Accepts a scalar or a
            3-dimensional vector. Defaults to 1.

        Returns
        -------
        cls
            An instance of the class with the generated affine transformation
            matrix.
        """
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
        """
        Generates a transformation matrix using specified values and mode.

        Parameters
        ----------
        values : Array_Float_T
            Array-like input values used for the transformation.
        mode : {'translate', 'rotate', 'scale', 'affine'}, default='affine'
            Transformation mode. It can be one of the following:
            - 'translate': Values represent translation offsets.
            - 'rotate': Values represent rotation matrix.
            - 'scale': Values represent scaling factors.
            - 'affine': Values represent a complete affine matrix.

        Returns
        -------
        cls
            A new instance of the class initialized with the generated transformation
            matrix.
        """

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


