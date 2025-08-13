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
    """
    Initializes a bounded validation field with a label, lower bound, and
    upper bound.

    Parameters
    ----------
    field_label : str
        A label that identifies the field being validated.
    lower_bound : float, optional
        The lower boundary for the field value. Default is negative infinity.
    upper_bound : float, optional
        The upper boundary for the field value. Default is positive infinity.
    """
    @validate_variables
    def __init__(self, field_label: str, lower_bound: float = -np.inf, upper_bound: float = np.inf) -> None:
        """
        Initializes a bounded validation field with a label, lower bound, and upper bound.

        Parameters
        ----------
        field_label : str
            A label that identifies the field being validated.
        lower_bound : float, optional
            The lower boundary for the field value. Default is negative infinity.
        upper_bound : float, optional
            The upper boundary for the field value. Default is positive infinity.
        """
        self.field_label = field_label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Masks the input point cloud based on a scalar field.

        This method checks if a specified scalar field exists in the point cloud
        data. It applies a logical condition to filter the points based on the
        scalar field's values, returning a boolean mask. The mask specifies which
        points in the cloud meet the condition defined by `lower_bound` and
        `upper_bound`.

        Parameters
        ----------
        pcd : PointCloudData
             The point cloud data containing scalar fields.

        Returns
        -------
        numpy.ndarray of numpy.bool_
            Boolean array where `True` indicates points satisfying the scalar
            field range conditions and `False` otherwise.
        """
        if self.field_label not in pcd.scalar_fields.keys():
            raise KeyError(f"Scalar field '{self.field_label}' is not defined.")
        scalar_field_data = cast(SF_T, pcd.scalar_fields[self.field_label])

        return np.logical_and(scalar_field_data >= self.lower_bound, scalar_field_data <= self.upper_bound)


class ScalarFieldPercentileFilter(PointCloudFilter):
    """
    Filter point cloud data based on scalar field percentile range.

    Applies a filtering mechanism to mask points in a point cloud whose scalar field values
    fall within a specified percentile range. Designed to work with point cloud data that contains
    scalar fields, this filter is configurable using field labels and percentile thresholds.

    Parameters
    ----------
    field_label : str
        Label identifying the scalar field to filter on.
    lower_percentile : PercentileT
        Minimum percentile threshold for filtering.
    upper_percentile : PercentileT
        Maximum percentile threshold for filtering.
    """
    @validate_variables
    def __init__(
        self, field_label: str, lower_percentile: PercentileT = 0, upper_percentile: PercentileT = 100
    ) -> None:
        """
        Initialize object with field label and percentile values.

        Parameters
        ----------
        field_label : str
            Label for the field.
        lower_percentile : PercentileT, optional
            Minimum percentile value, by default 0.
        upper_percentile : PercentileT, optional
            Maximum percentile value, by default 100.

        Raises
        ------
        ValueError
            If `lower_percentile` is greater than `upper_percentile`.
        """

        if lower_percentile > upper_percentile:
            raise ValueError(
                f"Lower percentile value ({lower_percentile}) must be less than the "
                f"upper percentile value ({upper_percentile}"
            )

        self.field_label = field_label
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Generate a boolean mask by filtering points in a point cloud based on scalar field
        values within a defined percentile range.

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud data containing scalar fields for filtering.

        Returns
        -------
        NDArray[np.bool_]
            Boolean array where points within the given percentile range are marked as True.
        """
        if self.field_label not in pcd.scalar_fields.keys():
            raise KeyError(f"Scalar field '{self.field_label}' is not defined.")

        sf = cast(SF_T, pcd.scalar_fields[self.field_label])
        lower_bound, upper_bound = np.percentile(sf.arr, [self.lower_percentile, self.upper_percentile])

        return np.logical_and(sf >= lower_bound, sf <= upper_bound)
