import logging

import numpy as np
from numpy.typing import NDArray

from .core import PointCloudFilter
from ..core import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])

class ScalarFieldFilter(PointCloudFilter):
    def __init__(self, field_label: str, lower_bound: float = -np.inf, upper_bound: float = np.inf):
        self.field_label = field_label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        assert self.field_label in pcd.scalar_fields.keys(), f"Field {self.field_label} is not defined."
        scalar_field_data = pcd.scalar_fields[self.field_label].data

        return np.logical_and(scalar_field_data >= self.lower_bound,
                              scalar_field_data <= self.upper_bound)

class ScalarFieldPercentileFilter(PointCloudFilter):
    def __init__(self, field_label: str, lower_percentile: float = 0.0, upper_percentile: float = 100.0):
        assert 0.0 <= lower_percentile <= upper_percentile <= 100.0
        self.field_label = field_label
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        assert self.field_label in pcd.scalar_fields.keys(), f"Field {self.field_label} is not defined."
        lower_bound, upper_bound = np.percentile(pcd.scalar_fields[self.field_label], [self.lower_percentile, self.upper_percentile])
        scalar_field_data = pcd.scalar_fields[self.field_label].data

        return np.logical_and(scalar_field_data >= lower_bound,
                              scalar_field_data <= upper_bound)