import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..core import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class PointCloudFilter(ABC):
    """
    Abstract base class for filters on a PointCloudData.

    Subclasses should implement the mask() method to return a boolean mask
    that selects the desired points.
    """

    @abstractmethod
    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Compute and return a boolean mask for the provided point cloud.

        Parameters:
            pcd (PointCloudData): The point cloud to filter.

        Returns:
            A numpy boolean array with shape (N,) where N is the number of points.
        """
        pass

    def reduce(self, pcd: PointCloudData) -> None:
        """
        Reduces the point cloud in-place to only the points where mask() is True.

        Parameters:
            pcd (PointCloudData): The point cloud to reduce.

        Returns:
            The modified point cloud.
        """
        m = self.mask(pcd)
        pcd.reduce(m)

    def extract(self, pcd: PointCloudData) -> PointCloudData:
        """
        Extracts points where mask() is True: returns a new point cloud with those points,
        and removes them from the original.

        Parameters:
            pcd (PointCloudData): The point cloud to extract points from.

        Returns:
            A new PointCloudData instance containing the extracted points.
        """
        m = self.mask(pcd)
        new_pcd = pcd.extract(m)
        return new_pcd

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        """
        Returns a new point cloud with only the points where mask() is True, leaving
        the original point cloud untouched.

        Parameters:
            pcd (PointCloudData): The point cloud to sample from.

        Returns:
            A new PointCloudData instance containing only the sampled points.
        """
        m = self.mask(pcd)
        return pcd.sample()


class GenericFieldFilter(PointCloudFilter):
    """
    A generic filter that uses a user-supplied function to generate a mask
    from a given field.
    """

    def __init__(self, field_label: str, filter_func: Callable):
        """
        Parameters:
            field_label: The field (attribute or scalar field key) on which to operate.
            filter_func: A callable that takes the field data and returns a boolean mask.
        """
        self.field = field_label
        self.filter_func = filter_func

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Retrieves the field data from the point cloud, applies the filter function,
        and returns the resulting boolean mask.
        """
        if self.field == "spherical_coordinates":
            _ = pcd.spherical_coordinates  # Ensure it's computed.
            data = pcd.spherical_coordinates
        elif self.field in pcd.scalar_fields:
            data = pcd.scalar_fields[self.field].data
        elif hasattr(pcd, self.field):
            data = getattr(pcd, self.field)
        else:
            raise ValueError(f"Field '{self.field}' does not exist in the point cloud.")
        return self.filter_func(data)
