from __future__ import annotations

import copy
from functools import wraps
from typing import Annotated, Any, Callable, Mapping, Optional, Self

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator, model_validator

from ..base_types import Array_4x4_T, Array_Nx3_T
from ..validators import extract_array
from .coordinates import CartesianCoordinates
from .optimal_shift import OptimizedShift

# from .optimal_shift import OSM_Manager
from .scalar_field_manager import ScalarFieldManager
from .scalar_fields import NormalFields, RGBFields, ScalarField
from .transforms import Transform, TransformLedger, TransformRecord


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
    arr: Array_Nx3_T = Field(alias="xyz")
    transform_ledger: TransformLedger[str, [Transform]] = Field(default_factory=TransformLedger)
    scalar_fields: ScalarFieldManager | dict[str, ScalarField] = Field(default_factory=ScalarFieldManager)

    # TODO: Check if this should move to the CartesianCoordinates
    optimal_shift: Optional[OptimizedShift]

    def __init__(
        self,
        xyz: npt.NDArray[np.floating] | CartesianCoordinates = None,
        *,
        rgb: Optional[npt.NDArray[Any, np.uint8] | RGBFields] = None,
        normals: Optional[npt.NDArray[Any, np.float32] | NormalFields] = None,
        intensity: Optional[npt.NDArray | ScalarField] = None,
        reflectance: Optional[npt.NDArray | ScalarField] = None,
        optimal_shift: OptimizedShift | ellipsis | None = Ellipsis,
        socs_origin: Optional[np.ndarray] = None,
        scalar_fields: Optional[ScalarFieldManager | dict] = None,
        project_transformation: Optional[Array_4x4_T] = None,
        transform_ledger: Optional[TransformLedger] = None,
    ):

        kwargs = {}

        if scalar_fields is None:
            scalar_fields = {}

        scalar_fields = ScalarFieldManager(None, fields=scalar_fields)

        if isinstance(rgb, np.ndarray):
            rgb = RGBFields(rgb)

        if isinstance(normals, np.ndarray):
            normals = NormalFields(normals)

        if isinstance(intensity, np.ndarray):
            intensity = ScalarField(intensity, name="intensity")

        if isinstance(reflectance, np.ndarray):
            reflectance = ScalarField(reflectance, name="reflectance")

        for field in (rgb, normals, intensity, reflectance):
            if field is not None:
                scalar_fields.add_field(field)
        #
        # if transform_ledger is not None:
        #     kwargs['transform_ledger'] = TransformLedger()

        if xyz is not None:
            kwargs["xyz"] = extract_array(xyz)

        if optimal_shift is Ellipsis:
            optimal_shift = OptimizedShift()

        optimal_shift.register(self)

        kwargs["scalar_fields"] = scalar_fields
        # TODO update logic inline with the global optimisation
        kwargs["optimised"] = optimal_shift is not None
        kwargs["optimal_shift"] = optimal_shift
        kwargs["socs_origin"] = socs_origin
        kwargs["project_transformation"] = project_transformation
        # TODO update logic once global shift and corresponding transforms supported
        kwargs["transform_ledger"] = transform_ledger if transform_ledger is not None else TransformLedger()

        # if kwargs.get('project_transform', None) is not None:
        #     kwargs['is_at_socs'] = True
        # return kwargs

        super().__init__(**kwargs)

    def __hash__(self) -> int:
        return id(self)

    # TODO Also reimplement this
    @field_validator("transform_ledger", mode="before")
    @classmethod
    def initialise_empty_ledger(cls, value: dict | TransformLedger):
        if isinstance(value, dict):
            return TransformLedger(**value)
        return value

    @model_validator(mode="after")
    def update_parent_weakref(self) -> Self:
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

    @property
    def rgb(self):
        return self.scalar_fields.rgb

    @property
    def intensity(self):
        return self.scalar_fields.intensity

    @property
    def reflectance(self):
        return self.scalar_fields.reflectance

    def __getitem__(self, item):
        return self.sample(self.create_mask(item))

    def __setitem__(self, key, value: PointCloudData):
        raise IndexError(
            f"Setting items in PointCloudData is not supported. Consider using the update_copy or "
            f"dump data to a dict and reinstantiate."
        )

    def update_copy(
        self, array: npt.NDArray | Self | None = None, *, deep: bool = True, update: Mapping[str, Any] = None
    ) -> Self:
        """
        This function is designed to be more efficient by not dumping the memory heavy array if it's to be updated in
        the new instance.
        E.g. if 'arr' is in the update dict {'arr': np.random.rand(10000, 3)}, don't dump the existing, just add this
        new value.
        """
        update = update or {}

        if array is not None:
            update["xyz"] = array.arr if isinstance(array, PointCloudData) else array

        return self.copy(deep=deep, update=update)

    # TODO explicitly state the
    def copy(self, *, deep: bool = True, update: dict = None) -> Self:
        """
        Produce a deep or shallow copy of the model. Updates the model also if parameter is parsed.
        """
        if not deep:
            raise NotImplementedError(f"Shallow copy is not implemented on this class: {type(self)}")

        if update is None:
            update = self.model_dump()
        else:
            update |= self.model_dump(exclude=(set(update.keys()) | {"arr"}))
        update = copy.deepcopy(update)
        return type(self)(update.pop("xyz"), **update)

    def sample(self, mask):
        mask = self.create_mask(mask)
        return self.update_copy(self.arr[mask, :], update={"scalar_fields": self.scalar_fields.sample(mask)})

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
