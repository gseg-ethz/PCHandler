from typing import Optional, TypedDict, Unpack

from pydantic import Field
import numpy as np

from .coordinates import CartesianCoordinates, TLSCoordinates
from .scalar_fields_pydantic import ScalarFieldManager



class ConfigPointCloud(TypedDict):
    project_optimal_center: np.ndarray
    optimal: bool
    read_only: bool
    transform_to_project_coords:


class PointCloudFactory:
    def __new__(cls, pcds: list[...], *args, **kwargs) -> BasePointCloud|TLSCloud|list[BasePointCloud]|list[TLSCloud]:

class BasePointCloud(CartesianCoordinates):
    sfm: Optional[ScalarFieldManager] = Field(default_factory=ScalarFieldManager)

    def __new__(cls, arbitrary=False, force_unoptimised=False, project_transformation=False, *args, **kwargs):
        # TODO logic for handling the different point cloud types based on parameters
        if arbitrary:
        """

        Parameters
        ----------
        args
        kwargs
        """

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

    def sample(self, *index, sub_ok=True):
        self.get_copy(self.sample(*index), update={
            'sfm': self.sfm[*index]
        })

    def reduce(self, *index):
        pass

    def extract(self):
        pass

    def merge(self):
        pass

    def to_o3d(self):
        pass

    @classmethod
    def from_o3d(self, o3d):
        pass

    def to_py4dgeo(self):
        pass

    @classmethod
    def from_py4dgeo(self, py4dgeo):
        pass

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
