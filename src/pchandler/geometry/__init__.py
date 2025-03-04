"""
Public API for pchandler.geometry.
Re-exports key classes and functions from the submodules.
"""

from .core import PointCloudData
from . import (
    filters,
    gpu,
    scalar_fields
)
# from .filters import (
#     filter_by_field,
#     box_cut,
#     filter_range,
#     extract_range,
#     filter_to_polygon,
#     random_subsample,
#     sample,
#     extract,
# )
# from .transforms import (
#     transform_point_cloud,
#     cartesian_to_spherical,
#     spherical_to_cartesian,
# )
# from .gpu import (
#     filter_to_polygon_gpu,
#     filter_spherical_polygon_gpu,
# )
# from .merging import (
#     merge_pcd,
#     split_pc_with_fov_tree,
# )
# from .scalar_fields import (
#     ScalarFieldManager,
#     ScalarField
# )

__all__ = [
    "PointCloudData",
    "filters",
    "gpu",
    "scalar_fields",

    # "filter_by_field",
    # "box_cut",
    # "filter_range",
    # "extract_range",
    # "filter_to_polygon",
    # "transform_point_cloud",
    # "cartesian_to_spherical",
    # "spherical_to_cartesian",
    # "random_subsample",
    # "sample",
    # "extract",
    # "filter_to_polygon_gpu",
    # "filter_spherical_polygon_gpu",
    # "merge_pcd",
    # "split_pc_with_fov_tree",
    # "ScalarField",
    # "ScalarFieldManager",

]
