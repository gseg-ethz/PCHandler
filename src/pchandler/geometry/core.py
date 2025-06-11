from __future__ import annotations

import uuid
from functools import wraps
from typing import Optional, TypedDict, Unpack, Self, Union, Any, Callable

from pydantic import Field, model_validator, ValidationError, field_validator
from pydantic.dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from .transforms import Transform, TransformLedger, TransformRecord
from .coordinates import CartesianCoordinates
from ..constants import RGB_FIELD, NORMALS_FIELD
from .optimal_shift import OSM_Manager
from .scalar_field_manager import ScalarFieldManager
from .scalar_fields import ScalarField, RGBFields, NormalFields
from ..validators import extract_array


def update_transformation_ledger(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(instance: PointCloudData, *args, **kwargs):
            result = func(instance, *args, **kwargs)

            result.transform_ledger[name] = TransformRecord(forward=args[0])
        return wrapper
    return decorator


@OSM_Manager.register_point_cloud
class PointCloudData(CartesianCoordinates):
    transform_ledger: TransformLedger[str, [Transform]] = Field(default_factory=TransformLedger)
    sfm: ScalarFieldManager|dict[str, ScalarField]|None = Field(default=None, alias='scalar_fields')

    def __init__(
            self,
            *args,
            xyz: np.ndarray | CartesianCoordinates = None,
            rgb: npt.NDArray[Any, np.uint8]|RGBFields = None,
            normals: npt.NDArray[Any, np.float32]|NormalFields = None,
            intensity: npt.NDArray|ScalarField = None,
            reflectance: npt.NDArray|ScalarField = None,
            optimised: bool = False,
            socs_origin: np.ndarray|None = None,
            scalar_fields: dict|None = None,
            **kwargs):

        if scalar_fields is None:
            scalar_fields = {}

        sfm = ScalarFieldManager(None, fields={**scalar_fields, **kwargs.pop('sfm', {})})

        if isinstance(rgb, np.ndarray):
            rgb = RGBFields(rgb)

        if isinstance(normals, np.ndarray):
            normals = NormalFields(normals)

        if isinstance(intensity, np.ndarray):
            intensity = ScalarField(intensity, name='intensity')

        if isinstance(reflectance, np.ndarray):
            reflectance = ScalarField(reflectance, name='reflectance')

        for field in (rgb, normals, intensity, reflectance):
            if field is not None:
                sfm.add_field(field)

        if xyz is not None:
            kwargs['arr'] = extract_array(xyz)
        elif len(args) == 1:
            kwargs['arr'] = extract_array(args[0])
        elif 'arr' in kwargs or 'xyz' in kwargs:
            pass
        else:
            raise ValueError('No coordinates were passed into PointCloudData as either the first positional argument '
                             'or as the keyword argument "xyz"')

        if not kwargs.get('transform_ledger', False):
            kwargs['transform_ledger'] = TransformLedger()

        kwargs['sfm'] = sfm
        kwargs['optimised'] = optimised
        kwargs['socs_origin'] = socs_origin

        super(CartesianCoordinates, self).__init__(**kwargs)

    @model_validator(mode='before')
    @classmethod
    def validate_initial_coordinates(cls, kwargs ) -> dict[str, Any]:
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
        # TODO reimpliment when base is done
        # if kwargs.get('project_transform', None) is not None:
        #     kwargs['is_at_socs'] = True
        return kwargs


    @field_validator('transform_ledger', mode='before')
    @classmethod
    def initialise_empty_ledger(cls, value: dict | TransformLedger):
        if isinstance(value, dict):
            return TransformLedger(**value)
        return value

    @model_validator(mode='after')
    def validate_model(self) -> Self:
        """Revalidate model to ensure that the weakref points to the correct object"""
        if isinstance(self.sfm, ScalarFieldManager):
            self.sfm.parent = self

        elif isinstance(self.sfm, dict):
            self.sfm = ScalarFieldManager(parent=self, fields=self.sfm)

        elif self.sfm is None:
            self.sfm = ScalarFieldManager(parent=self)

        return self

    @property
    def normals(self):
        return self.sfm.normals
    @property
    def rgb(self):
        return self.sfm.rgb
    @property
    def intensity(self):
        return self.sfm.intensity
    @property
    def reflectance(self):
        return self.sfm.reflectance

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

    def copy(self, *, deep: bool = True, **kwargs) -> Self:
        """
        Produce a deep or shallow copy of the model. Updates the model also if parameter is parsed.
        """
        if not deep:
            raise NotImplementedError(f'Shallow copy is not implemented on this class: {type(self)}')

        update = kwargs.get('update', {})
        update |= self.model_dump(exclude=set(update.keys()))

        result = type(self)(**update)

        return result.model_validate(result, strict=True)

    def sample(self, mask):
        mask = self.create_mask(mask)
        return self.update_copy(self.arr[mask, :], update={'sfm': self.sfm.sample(mask)})

    def reduce(self, mask):
        super().reduce(mask)
        self.sfm.reduce(mask)

    def extract(self, mask):
        extracted = super().extract(mask)
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

