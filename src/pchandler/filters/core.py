# pchandler – Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2022–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""Abstract filter base class plus a generic field-based filter."""

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
    """Abstract base class for :class:`PointCloudData` filters."""

    @abstractmethod
    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Compute and return a boolean mask for the provided point cloud.

        Parameters
        ----------
        pcd : PointCloudData
            Point cloud to be evaluated.

        Returns
        -------
        Vector_Bool_T
            Boolean mask, ``True`` for points kept by the filter.
        """
        pass

    @validate_variables
    def reduce(self, pcd: PointCloudData) -> None:
        """Reduce the point cloud in place to the points selected by the mask.

        Parameters
        ----------
        pcd : PointCloudData
            Point cloud mutated in place.
        """
        pcd.reduce(self.mask(pcd))

    def extract(self, pcd: PointCloudData) -> PointCloudData:
        """Return a new point cloud built from the points selected by the mask.

        The existing point cloud is reduced to only the points defined by the
        negated mask.

        Parameters
        ----------
        pcd : PointCloudData
            Source point cloud (mutated: keeps only the negated-mask points).

        Returns
        -------
        PointCloudData
            New point cloud carrying the selected points.
        """
        return pcd.extract(self.mask(pcd))

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        """Return a copy of the point cloud sampled using the boolean mask.

        Parameters
        ----------
        pcd : PointCloudData
            Source point cloud (not mutated).

        Returns
        -------
        PointCloudData
            New point cloud carrying the sampled points.
        """
        return pcd.sample(self.mask(pcd))


class GenericFieldFilter(PointCloudFilter):
    """Generic filter that evaluates a callable on a named field."""

    @validate_variables
    def __init__(self, field_label: str, filter_func: Callable[[npt.NDArray], Vector_Bool_T]) -> None:
        """Build a generic filter that operates on a single named field.

        Parameters
        ----------
        field_label : str
            A label or name for the field.
        filter_func : Callable[[NDArray], Vector_Bool_T]
            A callable function used to perform filtering logic. Must return a
            boolean mask.
        """
        self.field_label = field_label
        self.filter_func = filter_func

    # TODO check if this is in use anywhere before changing the default names
    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask from the defined field and function.

        Parameters
        ----------
        pcd : PointCloudData
            Point cloud to be evaluated.

        Returns
        -------
        Vector_Bool_T
            Boolean mask returned by ``filter_func`` evaluated on the named
            field.
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
