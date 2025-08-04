import logging
from typing import Annotated, cast

import numpy as np
from numpy.typing import NDArray

from pydantic import NonNegativeFloat, Field

from pchandler.core import PointCloudData

from GSEGUtils.constants import validate_variables

from .core import PointCloudFilter
from ..scalar_fields.sf_types import SF_T


logger = logging.getLogger(__name__.split(".")[0])

PercentileT = Annotated[NonNegativeFloat, Field(lt=100.0)]

class ScalarFieldFilter(PointCloudFilter):
    @validate_variables
    def __init__(self,
                 field_label: str,
                 lower_bound: float = -np.inf,
                 upper_bound: float = np.inf) -> None:
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
    def __init__(self,
                 field_label: str,
                 lower_percentile: PercentileT = 0,
                 upper_percentile: PercentileT = 100) -> None:

        if lower_percentile > upper_percentile:
            raise ValueError(f"Lower percentile value ({lower_percentile}) must be less than the "
                             f"upper percentile value ({upper_percentile}")

        self.field_label = field_label
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        if self.field_label not in pcd.scalar_fields.keys():
            raise KeyError(f"Scalar field '{self.field_label}' is not defined.")

        sf = cast(SF_T, pcd.scalar_fields[self.field_label])
        lower_bound, upper_bound = np.percentile(sf.arr, [self.lower_percentile, self.upper_percentile])

        return np.logical_and(sf >= lower_bound, sf <= upper_bound)
