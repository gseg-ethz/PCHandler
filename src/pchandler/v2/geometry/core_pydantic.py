from __future__ import annotations

from typing import Optional, TypedDict, Unpack, Self, Union

from pydantic import Field, model_validator, ValidationError
import numpy as np

from .transforms import Transform
from .coordinates import CartesianCoordinates
from .optimal_shift import OSM_Manager
from .scalar_fields_pydantic import ScalarFieldManager, ScalarField


class PointCloudConfig(TypedDict):
    scalar_fields: ScalarFieldManager|dict[str, ScalarField]
    socs_origin: Union[np.ndarray, None]
    is_at_socs: bool
    project_transform: Union[np.ndarray, Transform, None]


@OSM_Manager.register_point_cloud
class PointCloudData(CartesianCoordinates):
    optimal: bool = Field(default=False, exclude=True)
    sfm: Optional[ScalarFieldManager] = Field(default_factory=ScalarFieldManager)

    @model_validator(mode='before')
    @classmethod
    def validate_init_params(cls, kwargs: Unpack[PointCloudConfig]) -> PointCloudData:
        key = {'arr', 'xyz'} & set(kwargs.keys())
        if len(key) != 1:
            raise ValidationError(f"Invalid keyword arguments. Only accepts 'xyz' OR 'arr', not both.")
        xyz = kwargs.pop(list(key)[0])

        # Override the passed point cloud with any input kwargs
        if isinstance(xyz, cls):
            kwargs = xyz.model_dump() | kwargs

        # Pass in the data with any set parameters
        elif isinstance(xyz, np.ndarray):
            kwargs = kwargs | {'arr': xyz}

        else:
            raise TypeError(f'Unsupported object type passed as xyz for PointCloudData: {type(xyz)}')

        # Ensure is_at_socs if project_transform is set (assumed transform from project to socs)
        if kwargs.get('project_transform', None) is not None:
            kwargs['is_at_socs'] = True

        return kwargs

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        # TODO raise error that you cannot change individual or selection of points in place and to use
        #  sample/extract/reduce
        pass

    def sample(self, mask):
        raise NotImplementedError

    def reduce(self, mask):
        raise NotImplementedError

    def extract(self, mask):
        raise NotImplementedError

    def merge(self):
        raise NotImplementedError

    def to_o3d(self):
        raise NotImplementedError

    @classmethod
    def from_o3d(self, o3d):
        raise NotImplementedError

    def to_py4dgeo(self):
        raise NotImplementedError

    @classmethod
    def from_py4dgeo(self, py4dgeo):
        raise NotImplementedError

    @property
    def normals(self): return self.sfm.normals
    @property
    def rgb(self): return self.sfm.rgb
    @property
    def intensity(self): return self.sfm.intensity
    @property
    def reflectance(self): return self.sfm.reflectance
