# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

import logging
from typing import Annotated, cast

import numpy as np
from GSEGUtils.constants import validate_variables
from numpy.typing import NDArray
from pydantic import Field, NonNegativeFloat

from pchandler import PointCloudData
from pchandler.filters import PointCloudFilter
from pchandler.scalar_fields import SF_T

logger = logging.getLogger(__name__.split(".")[0])

PercentileT = Annotated[NonNegativeFloat, Field(lt=100.0)]


class ScalarFieldFilter(PointCloudFilter):
    @validate_variables
    def __init__(self, field_label: str, lower_bound: float = -np.inf, upper_bound: float = np.inf) -> None:
        self.field_label = field_label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        if self.field_label not in pcd.scalar_fields.keys():
            raise KeyError(f"Scalar field '{self.field_label}' is not defined.")
        scalar_field_data = cast(SF_T, pcd.scalar_fields[self.field_label])

        return np.logical_and(scalar_field_data >= self.lower_bound, scalar_field_data <= self.upper_bound)


class ScalarFieldPercentileFilter(PointCloudFilter):
    @validate_variables
    def __init__(
        self, field_label: str, lower_percentile: PercentileT = 0, upper_percentile: PercentileT = 100
    ) -> None:

        if lower_percentile > upper_percentile:
            raise ValueError(
                f"Lower percentile value ({lower_percentile}) must be less than the "
                f"upper percentile value ({upper_percentile}"
            )

        self.field_label = field_label
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        if self.field_label not in pcd.scalar_fields.keys():
            raise KeyError(f"Scalar field '{self.field_label}' is not defined.")

        sf = cast(SF_T, pcd.scalar_fields[self.field_label])
        lower_bound, upper_bound = np.percentile(sf.arr, [self.lower_percentile, self.upper_percentile])

        return np.logical_and(sf >= lower_bound, sf <= upper_bound)
