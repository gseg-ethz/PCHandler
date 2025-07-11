import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from ..constants import validate_variables
from ..geometry.core import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class PointCloudFilter(ABC):
    """
    Abstract base class for filters on a PointCloudData.

    Subclasses should implement the mask() method to return a boolean mask
    that selects the desired points.
    """

    @abstractmethod
    def mask(self, pcd: PointCloudData) -> npt.NDArray[np.bool_]:
        """
        Compute and return a boolean mask for the provided point cloud.

        Parameters:
            pcd (PointCloudData): The point cloud to filter.

        Returns:
            A numpy boolean array with shape (N,) where N is the number of points.
        """
        pass

    @validate_variables
    def reduce(self, pcd: PointCloudData) -> None:
        """
        Reduces the point cloud in-place to only the points where mask() is True.

        Parameters:
            pcd (PointCloudData): The point cloud to reduce.

        Returns:
            The modified point cloud.
        """
        m = self.mask(pcd)
        # if np.sum(m) == 0:
        #     warnings.warn('Filter produced no values to index, thus returning full point cloud')
        # else:
        #     pcd.reduce(m)
        pcd.reduce(m)

    def extract(self, pcd: PointCloudData) -> Optional[PointCloudData]:
        """
        Extracts points where mask() is True: returns a new point cloud with those points,
        and removes them from the original.

        Parameters:
            pcd (PointCloudData): The point cloud to extract points from.

        Returns:
            A new PointCloudData instance containing the extracted points.
        """
        m = self.mask(pcd)

        # if np.sum(m) == 0:
        #     warnings.warn('Filter produced no values to index, thus nothing has been extracted')
        #     return None
        new_pcd = pcd.extract(m)
        return new_pcd

    def sample(self, pcd: PointCloudData) -> Optional[PointCloudData]:
        """
        Returns a new point cloud with only the points where mask() is True, leaving
        the original point cloud untouched.

        Parameters:
            pcd (PointCloudData): The point cloud to sample from.

        Returns:
            A new PointCloudData instance containing only the sampled points.
        """
        m = self.mask(pcd)
        # if np.sum(m) == 0:
        #     warnings.warn('Filter produced no values to index, thus nothing has been sampled')
        #     return None
        return pcd.sample(m)


class GenericFieldFilter(PointCloudFilter):
    """
    A generic filter that uses a user-supplied function to generate a mask
    from a given field.

    Parameters:
        field_label: The field (attribute or scalar field key) on which to operate.
        filter_func: A callable that takes the field data and returns a boolean mask.
    """

    @validate_variables
    def __init__(self, field_label: str, filter_func: Callable) -> None:
        self.field_label = field_label
        self.filter_func = filter_func

    def mask(self, pcd: PointCloudData) -> npt.NDArray[np.bool_]:
        """
        Retrieves the field data from the point cloud, applies the filter function,
        and returns the resulting boolean mask.
        """
        if self.field_label == "cartesian_coordinates":
            data = pcd.xyz
        elif self.field_label == "spherical_coordinates":
            data = pcd.spher
        elif self.field_label in pcd.scalar_fields:
            data = pcd.scalar_fields[self.field_label].arr
        elif hasattr(pcd, self.field_label):
            data = getattr(pcd, self.field_label)
        else:
            raise ValueError(f"Field '{self.field_label}' does not exist in the point cloud.")

        if data is None:
            raise ValueError(f"Field '{self.field_label}' does not exist in the point cloud.")

        return self.filter_func(data)
