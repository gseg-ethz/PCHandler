from pathlib import Path
from typing import Optional
import logging

import numpy as np
import numpy.typing as npt
import laspy                # type: ignore[import-untyped]

from pchandler.data_io.core import AbstractIOHandler, _get_rgb_or_normal_field_names
from pchandler.constants import RGB_NAMES, NORMAL_NAMES, INTENSITY_NAMES, XYZ_NAMES
from pchandler.geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


# TODO add functionality to handle the extra fields?
class LasHandler(AbstractIOHandler):
    FORMATS = ['.las', '.laz']

    @classmethod
    def load(cls,
             path: str | Path, /,
             scalar_fields: Optional[list[str]] = None,
             remove_prefix: bool = True,
             prefix: str = 'scalar_',
             **config
             ) -> PointCloudData:

        logger.info(f"Loading LAZ file: {path}")

        las = laspy.read(path)
        header_names = las._points.array.dtype.names

        data = las._points.array
        field_names = cls._validate_field_selection(scalar_fields, header_names, remove_prefix, prefix)

        pcd = PointCloudData(las.xyz)
        cls.extract_scalar_fields(pcd, data, data.size, field_names)

        logger.info(f"Successfully loaded LAZ file: {path}")
        return pcd

    @classmethod
    def save(cls,
             pcd: PointCloudData,
             path: str | Path,
             /,
             scalar_fields: Optional[list[str]] = None,
             add_prefix: bool = True,
             prefix: str = 'scalar_',
             revert_sf_types: bool = False,
             **config) -> None:

        logger.info(f"Attempting to write to LAS/LAZ file: {path}")

        # TODO should this be linked to the optimal shift?
        offsets: npt.NDArray[np.float32|np.float64] = pcd.min(axis=0)
        # TODO extend config parameters for these scale values
        scales: npt.NDArray[np.float32|np.float64] = np.array([0.0001, 0.0001, 0.0001])

        if scalar_fields is None:
            scalar_fields = list(pcd.scalar_fields.keys())
        elif not isinstance(scalar_fields, list):
            scalar_fields = list(scalar_fields)

        # Can use the check on the base name as named scalar_fields should match the pcd object names
        if RGB_NAMES.base in scalar_fields:
            index = scalar_fields.index(RGB_NAMES.base)
            scalar_fields = scalar_fields[:index] + list(RGB_NAMES.char) + scalar_fields[index+1:]

        if NORMAL_NAMES.base in scalar_fields:
            index = scalar_fields.index(NORMAL_NAMES.base)
            scalar_fields = scalar_fields[:index] + list(NORMAL_NAMES.char) + scalar_fields[index+1:]

        # Set the base coordinates of the LAS point cloud as well as offsets and scales
        las = laspy.create()
        las.header.offsets = offsets
        las.header.scales = scales

        las.X = (pcd.x - offsets[0]) / scales[0]
        las.Y = (pcd.y - offsets[1]) / scales[1]
        las.Z = (pcd.z - offsets[2]) / scales[2]

        # RGB values
        if (rgb_fields := _get_rgb_or_normal_field_names(scalar_fields, RGB_NAMES)) and pcd.rgb:
            for field in rgb_fields:
                index = RGB_NAMES.get_position(field)
                setattr(las, RGB_NAMES.words[index], getattr(pcd.rgb, RGB_NAMES.char[index]))

        # Intensities
        if (intensity_fields := set(scalar_fields).intersection(INTENSITY_NAMES.all)) and pcd.intensity:
            las.intensity = pcd.intensity

        # Clear the previous sfs used
        for name in (XYZ_NAMES.char + tuple(rgb_fields) + tuple(intensity_fields)):
            if name in scalar_fields:
                scalar_fields.remove(name)

        # Remaining fields including normals as these are extra dimensions
        for field in scalar_fields:
            value = np.asarray(pcd.scalar_fields[field])

            if not hasattr(las, field):
                las.add_extra_dim(laspy.ExtraBytesParams(field, value.dtype))

            setattr(las, field, value)

        las.write(path)

        logger.info(f"Successfully wrote to  LAS/LAZ file: {path}")

