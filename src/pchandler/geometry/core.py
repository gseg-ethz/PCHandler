from __future__ import annotations

import uuid
from functools import wraps
from typing import Optional, TypedDict, Unpack, Self, Union, Any

from pydantic import Field, model_validator, ValidationError, field_validator
import numpy as np

from .transforms import Transform, TransformLedger, TransformRecord
from .coordinates import CartesianCoordinates
from .optimal_shift import OSM_Manager
from .scalar_field_manager import ScalarFieldManager
from .scalar_fields import (
    ScalarField, RGBFields, NormalFields,
    RGB_FIELD, NORMALS_FIELD)


def update_transformation_ledger(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(instance: PointCloudData, *args, **kwargs):
            result = func(instance, *args, **kwargs)

            result.transform_ledger[name] = TransformRecord(forward=args[0])
        return wrapper
    return decorator


class PointCloudConfig(TypedDict):
    scalar_fields: ScalarFieldManager|dict[str, ScalarField]
    socs_origin: Union[np.ndarray, None]
    is_at_socs: bool
    project_transform: Union[np.ndarray, Transform, None]
    rgb: Union[np.ndarray, RGBFields]
    normals: Union[np.ndarray, NormalFields]


@OSM_Manager.register_point_cloud
class PointCloudData(CartesianCoordinates):
    transform_ledger: TransformLedger[str, [Transform]] = Field(default_factory=TransformLedger)
    optimal: bool = Field(default=False, exclude=True)
    sfm: ScalarFieldManager|dict[str, ScalarField]|None = None
    uuid: str = Field(default_factory= lambda: str(uuid.uuid4()), exclude=True)

    @model_validator(mode='before')
    @classmethod
    def validate_initial_scalar_fields(cls, kwargs: Unpack[PointCloudConfig]) -> dict[str, Any]:
        sfm = ScalarFieldManager(None, fields=kwargs.pop('sfm', {}))

        rgb = kwargs.pop(RGB_FIELD, None)
        if rgb is not None:
            sfm.add_field(
                RGBFields.initialize(
                    size = rgb.shape[0],
                    value = rgb
                ))

        normals = kwargs.pop(NORMALS_FIELD, None)
        if normals is not None:
            sfm.add_field(
                NormalFields.initialize(
                    size = normals.shape[0],
                    value = normals
                ))


        kwargs['sfm'] = ScalarFieldManager(None, fields=sfm)
        return kwargs

    @model_validator(mode='before')
    @classmethod
    def validate_initial_coordinates(cls, kwargs: Unpack[PointCloudConfig]) -> dict[str, Any]:
        key = {'arr', 'xyz'} & set(kwargs.keys())
        if len(key) != 1:
            raise ValidationError(f"Invalid keyword arguments. Only accepts 'xyz' OR 'arr', not both.")
        xyz = kwargs.pop(list(key)[0])

        # Override the passed point cloud with any input kwargs
        if isinstance(xyz, cls):
            kwargs = xyz.model_dump() | kwargs
        elif isinstance(xyz, np.ndarray):
            kwargs = kwargs | {'arr': xyz}
        else:
            raise TypeError(f'Unsupported object type passed as xyz for PointCloudData: {type(xyz)}')

        # Ensure is_at_socs if project_transform is set (assumed transform from project to socs)
        if kwargs.get('project_transform', None) is not None:
            kwargs['is_at_socs'] = True
        return kwargs


    @field_validator('transform_ledger', mode='before')
    @classmethod
    def initialise_empty_ledger(cls, value: dict | TransformLedger):
        if isinstance(value, dict):
            return TransformLedger(**value)
        return value

    @model_validator(mode='after')
    def validate_model(self) -> None:
        """Revalidate model to ensure that the weakref points to the correct object"""
        if isinstance(self.sfm, ScalarFieldManager):
            self.sfm.parent = self

        elif isinstance(self.sfm, dict):
            self.sfm = ScalarFieldManager(parent=self, fields=self.sfm)

        elif self.sfm is None:
            self.sfm = ScalarFieldManager(parent=self)

    @property
    def normals(self): return self.sfm.normals
    @property
    def rgb(self): return self.sfm.rgb
    @property
    def intensity(self): return self.sfm.intensity
    @property
    def reflectance(self): return self.sfm.reflectance

    def __hash__(self) -> int:
        return hash(self.uuid)

    def __getitem__(self, item):
        return self.sample(self.create_mask(item))

    def __setitem__(self, key, value: PointCloudData):
        raise IndexError(f'Setting items in PointCloudData is not supported. Consider using the update_copy or '
                         f'dump data to a dict and reinstantiate.')

    def __eq__(self, other: PointCloudData|np.ndarray) -> bool:
        if isinstance(other, PointCloudData) and self.shape == other.shape:
            if np.allclose(self.arr, other.arr) and self.uuid == other.uuid:
                return True
            return np.allclose(self.arr, other.arr)
        else:
            if isinstance(self, CartesianCoordinates):
                return CartesianCoordinates.__eq__(self, other)
            else:
                raise NotImplementedError(f'Equality function not implemented of object type {type(self)}.')

    def sample(self, mask):
        mask = self.create_mask(mask)
        return self.update_copy(self.arr[mask, :], update={'sfm': self.sfm.sample(mask)})

    def reduce(self, mask):
        super().reduce(mask)
        self.sfm.reduce(mask)

    def extract(self, mask):
        extracted = super().extract(mask)
        extracted.sfm._fields = self.sfm.extract(mask)
        return extracted

    def merge(self):
        raise NotImplementedError

    def to_o3d(self):
        raise NotImplementedError

    @classmethod
    def from_o3d(cls, o3d):
        raise NotImplementedError

    def to_py4dgeo(self):
        raise NotImplementedError

    @classmethod
    def from_py4dgeo(cls, py4dgeo):
        raise NotImplementedError

