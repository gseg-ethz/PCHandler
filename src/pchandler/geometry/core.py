from __future__ import annotations

from functools import wraps
from typing import Optional, Self, Any, Callable

from pydantic import Field, model_validator, field_validator
import numpy as np
import numpy.typing as npt

from .transforms import Transform, TransformLedger, TransformRecord
from .coordinates import CartesianCoordinates
from ..base_types import Array_Nx3_T, Array_4x4_T
from .optimal_shift import OSM_Manager
from .scalar_field_manager import ScalarFieldManager
from .scalar_fields import ScalarField, RGBFields, NormalFields
from ..validators import extract_array

# TODO decide on this artifact
def update_transformation_ledger(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(instance: PointCloudData, *args, **kwargs):
            result = func(instance, *args, **kwargs)

            result.transform_ledger[name] = TransformRecord(forward=args[0])
        return wrapper
    return decorator


class PointCloudData(CartesianCoordinates):
    arr: Array_Nx3_T = Field(alias='xyz')
    transform_ledger: TransformLedger[str, [Transform]] = Field(default_factory=TransformLedger)
    scalar_fields: ScalarFieldManager | dict[str, ScalarField] | None = Field(default=None, alias='scalar_fields')

    # TODO only have the args and kwargs necessary. No open *args, **kwargs no
    def __init__(
            self,
            xyz: np.ndarray | CartesianCoordinates = None,
            *,
            rgb: Optional[npt.NDArray[Any, np.uint8]|RGBFields] = None,
            normals: Optional[npt.NDArray[Any, np.float32]|NormalFields] = None,
            intensity: Optional[npt.NDArray|ScalarField] = None,
            reflectance: Optional[npt.NDArray|ScalarField] = None,
            optimised: bool = False,
            socs_origin: Optional[np.ndarray] = None,
            scalar_fields: Optional[ScalarFieldManager|dict] = None,
            project_transformation: Optional[Array_4x4_T] = None,
            transform_ledger: Optional[TransformLedger] = None,
            **kwargs
            ):


        if scalar_fields is None:
            scalar_fields = {}

        scalar_fields = ScalarFieldManager(None, fields=scalar_fields)

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
                scalar_fields.add_field(field)
        #
        # if transform_ledger is not None:
        #     kwargs['transform_ledger'] = TransformLedger()

        if xyz is not None:
            kwargs['arr'] = extract_array(xyz)

        kwargs['scalar_fields'] = scalar_fields
        kwargs['optimised'] = optimised
        kwargs['socs_origin'] = socs_origin
        kwargs['project_transformation'] = project_transformation
        kwargs['transform_ledger'] =  TransformLedger()     # FIXME

        super(CartesianCoordinates, self).__init__(**kwargs)

    # @model_validator(mode='before')
    # @classmethod
    # def validate_initial_coordinates(cls, kwargs ) -> dict[str, Any]:
    #     key = {'arr', 'xyz'} & set(kwargs.keys())
    #     if len(key) != 1:
    #         raise ValidationError(f"Invalid keyword arguments. Only accepts 'xyz' OR 'arr', not both.")
    #     xyz = kwargs.pop(list(key)[0])
    #
    #     # Override the passed point cloud with any input kwargs
    #     if isinstance(xyz, cls):
    #         kwargs = xyz.model_dump() | kwargs
    #     elif isinstance(xyz, np.ndarray):
    #         kwargs = kwargs | {'arr': xyz}
    #     else:
    #         raise TypeError(f'Unsupported object type passed as xyz for PointCloudData: {type(xyz)}')

        # TODO Resolve this with the global shift logic once the base is done
        # if kwargs.get('project_transform', None) is not None:
        #     kwargs['is_at_socs'] = True
        # return kwargs

    # TODO Also reimplement this
    @field_validator('transform_ledger', mode='before')
    @classmethod
    def initialise_empty_ledger(cls, value: dict | TransformLedger):
        if isinstance(value, dict):
            return TransformLedger(**value)
        return value

    @model_validator(mode='after')
    def validate_model(self) -> Self:
        """Revalidate model to ensure that the weakref points to the correct object"""
        if isinstance(self.scalar_fields, ScalarFieldManager):
            self.scalar_fields.parent = self

        elif isinstance(self.scalar_fields, dict):
            self.scalar_fields = ScalarFieldManager(parent=self, fields=self.scalar_fields)

        elif self.scalar_fields is None:
            self.scalar_fields = ScalarFieldManager(parent=self)

        return self

    @property
    def normals(self):
        return self.scalar_fields.normals

    @normals.setter
    def normals(self, value: np.ndarray|NormalFields):
        self.scalar_fields.normals = value

    @property
    def rgb(self):
        return self.scalar_fields.rgb

    @rgb.setter
    def rgb(self, value: np.ndarray|RGBFields):
        self.scalar_fields.rgb = value

    @property
    def intensity(self):
        return self.scalar_fields.intensity

    @intensity.setter
    def intensity(self, value: np.ndarray|ScalarField):
        self.scalar_fields.intensity = value

    @property
    def reflectance(self):
        return self.scalar_fields.reflectance

    @reflectance.setter
    def reflectance(self, value: np.ndarray|ScalarField):
        self.scalar_fields.reflectance = value

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

        result = type(self)(update.pop('arr'), **update)

        return result.model_validate(result, strict=True)

    def sample(self, mask):
        mask = self.create_mask(mask)
        return self.update_copy(self.arr[mask, :], update={'scalar_fields': self.scalar_fields.sample(mask)})

    def reduce(self, mask):
        super().reduce(mask)
        self.scalar_fields.reduce(mask)

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

