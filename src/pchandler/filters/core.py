from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Annotated, Callable, Sequence, cast

import numpy as np
import numpy.typing as npt
from GSEGUtils.constants import validate_variables
from pydantic import BeforeValidator
from shapely import Polygon

from pchandler import PointCloudData
from pchandler.scalar_fields import SF_T

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
        pcd.reduce(self.mask(pcd))

    def extract(self, pcd: PointCloudData) -> PointCloudData:
        """
        Extracts points where mask() is True: returns a new point cloud with those points,
        and removes them from the original.

        Parameters:
            pcd (PointCloudData): The point cloud to extract points from.

        Returns:
            A new PointCloudData instance containing the extracted points.
        """
        return pcd.extract(self.mask(pcd))

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        """
        Returns a new point cloud with only the points where mask() is True, leaving
        the original point cloud untouched.

        Parameters:
            pcd (PointCloudData): The point cloud to sample from.

        Returns:
            A new PointCloudData instance containing only the sampled points.
        """
        return pcd.sample(self.mask(pcd))


class GenericFieldFilter(PointCloudFilter):
    """
    A generic filter that uses a user-supplied function to _generate a mask
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
            data = cast(SF_T, pcd.scalar_fields[self.field_label]).arr
        elif hasattr(pcd, self.field_label):
            data = getattr(pcd, self.field_label)
        else:
            raise ValueError(f"Field '{self.field_label}' does not exist in the point cloud.")

        if data is None:
            raise ValueError(f"Field '{self.field_label}' does not exist in the point cloud.")

        return self.filter_func(data)


ValidatedPolygonT = Annotated[
    Sequence | npt.NDArray[np.floating | np.integer] | Polygon, BeforeValidator(lambda x: Polygon(x))
]
