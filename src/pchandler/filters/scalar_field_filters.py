# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""Scalar-field-based filters."""

import logging
from typing import Annotated, cast

import numpy as np
from GSEGUtils.base_types import Vector_Bool_T
from GSEGUtils.constants import validate_variables
from pydantic import Field, NonNegativeFloat

from pchandler import PointCloudData
from pchandler.filters import PointCloudFilter
from pchandler.scalar_fields import SF_T

logger = logging.getLogger(__name__.split(".")[0])

PercentileT = Annotated[NonNegativeFloat, Field(lt=100.0)]


class ScalarFieldFilter(PointCloudFilter):
    """Filter points by an absolute value range over a named scalar field."""

    @validate_variables
    def __init__(self, field_label: str, lower_bound: float = -np.inf, upper_bound: float = np.inf) -> None:
        """Filter points based on a value range for a particular scalar field.

        Parameters
        ----------
        field_label : str
            Name of the scalar field to evaluate.
        lower_bound : float, default=-np.inf
            Lower (inclusive) bound on the scalar value.
        upper_bound : float, default=np.inf
            Upper (inclusive) bound on the scalar value.
        """
        self.field_label = field_label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask for the values within the specified scalar field range.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        Vector_Bool_T
        """
        if self.field_label not in pcd.scalar_fields.keys():
            raise KeyError(f"Scalar field '{self.field_label}' is not defined.")
        scalar_field_data = cast(SF_T, pcd.scalar_fields[self.field_label])

        return np.logical_and(scalar_field_data >= self.lower_bound, scalar_field_data <= self.upper_bound)


class ScalarFieldPercentileFilter(PointCloudFilter):
    """Filter points by a percentile range over a named scalar field."""

    @validate_variables
    def __init__(
        self, field_label: str, lower_percentile: PercentileT = 0, upper_percentile: PercentileT = 100
    ) -> None:
        """Filter points based on percentile ranges for a given scalar field.

        Parameters
        ----------
        field_label : str
            Name of the scalar field to evaluate.
        lower_percentile : PercentileT, default=0
            Lower percentile (0 ≤ ``lower_percentile`` < 100).
        upper_percentile : PercentileT, default=100
            Upper percentile (``lower_percentile`` ≤ ``upper_percentile`` < 100).
        """
        if lower_percentile > upper_percentile:
            raise ValueError(
                f"Lower percentile value ({lower_percentile}) must be less than the "
                f"upper percentile value ({upper_percentile}"
            )

        self.field_label = field_label
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask from the target percentile range.

        Parameters
        ----------
        pcd : PointCloudData
            Source point cloud.

        Returns
        -------
        Vector_Bool_T
            Boolean mask, ``True`` for points whose value falls inside the
            percentile range.
        """
        if self.field_label not in pcd.scalar_fields.keys():
            raise KeyError(f"Scalar field '{self.field_label}' is not defined.")

        sf = cast(SF_T, pcd.scalar_fields[self.field_label])
        lower_bound, upper_bound = np.percentile(sf.arr, [self.lower_percentile, self.upper_percentile])

        return np.logical_and(sf >= lower_bound, sf <= upper_bound)
