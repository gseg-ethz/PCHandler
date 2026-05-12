# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""
Contains the abstract filter class and a generic filter class
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Annotated, Callable, Sequence, cast

import numpy as np
import numpy.typing as npt
from GSEGUtils.base_types import Vector_Bool_T
from GSEGUtils.constants import validate_variables
from pydantic import BeforeValidator
from shapely import Polygon

from pchandler import PointCloudData
from pchandler.scalar_fields import SF_T

logger = logging.getLogger(__name__.split(".")[0])


class PointCloudFilter(ABC):
    """Abstract base class for PointCloudData filters"""

    @abstractmethod
    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """
        Compute and return a boolean mask for the provided point cloud.

        Parameters
        ----------
        pcd: Point

        Returns
        -------
        Vector_Bool_T
        """
        pass

    @validate_variables
    def reduce(self, pcd: PointCloudData) -> None:
        """Reduces the point to only the points defined by the mask.

        Parameters
        ----------
        pcd: PointCloudData

        Returns
        -------
        PointCloudData
        """
        pcd.reduce(self.mask(pcd))

    def extract(self, pcd: PointCloudData) -> PointCloudData:
        """Returns a new point cloud from the points selected by the mask.

        The existing point cloud is reduced to only the points defined by the negated mask.

        Parameters
        ----------
        pcd: PointCloudData

        Returns
        -------
        PointCloudData
        """
        return pcd.extract(self.mask(pcd))

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        """Returns a copy of the point cloud sampled using the boolean mask.

        Parameters
        ----------
        pcd: PointCloudData

        Returns
        -------
        PointCloudData
        """
        return pcd.sample(self.mask(pcd))


class GenericFieldFilter(PointCloudFilter):
    @validate_variables
    def __init__(self, field_label: str, filter_func: Callable[[npt.NDArray], Vector_Bool_T]) -> None:
        """Generic filter class that will perform a custom filter on a defined field.

        Parameters
        ----------
        field_label : str
            A label or name for the field.
        filter_func : Callable[[NDArray], Vector_Bool_T]
            A callable function used to perform filtering logic. Must return a boolean mask.
        """
        self.field_label = field_label
        self.filter_func = filter_func

    # TODO check if this is in use anywhere before changing the default names
    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask from the defined field and function

        Parameters
        ----------
        pcd: PointCloudData

        Returns
        -------
        Vector_Bool_T
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
