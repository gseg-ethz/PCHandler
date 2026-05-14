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

"""E57 file-format handler class."""

import logging
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Unpack

import numpy as np
import pye57  # type: ignore[import-untyped]

from pchandler import PointCloudData
from pchandler.constants import INTENSITY_NAMES, RGB_NAMES
from pchandler.data_io.core import AbstractIOHandler, PointCloudDataKW

logger = logging.getLogger(__name__.split(".")[0])

__all__ = ["E57Handler"]


class E57Handler(AbstractIOHandler):
    """Handles E57 file input and output.

    Currently limited to reading and writing single scans with and only RGB and intensity scalar field types.
    All other scalar fields are ignored (due to library limitations).

    Supported file extensions:

    * .e57

    Notes
    -----
    ``save`` supports both a single :class:`~pchandler.PointCloudData` and an iterable of them.
    Arbitrary scalar fields outside ``pye57.e57.SUPPORTED_POINT_FIELDS`` are skipped with a
    ``logger.warning`` by default.  Pass ``strict=True`` to raise ``ValueError`` instead.
    The ``SUPPORTED_SCALAR_FIELDS_MAP`` limitation remains in place for all write operations.
    """

    FORMATS = [".e57"]

    SUPPORTED_FIELDS = {"x": "cartesianX", "y": "cartesianY", "z": "cartesianZ"}

    SUPPORTED_SCALAR_FIELDS_MAP = {"intensity": "intensity", "r": "colorRed", "g": "colorGreen", "b": "colorBlue"}

    @classmethod
    def load(
        cls,  # type: ignore[override]
        path: str | Path,
        /,
        retain_rgb: bool = True,
        retain_intensity: bool = True,
        pcd_index: Optional[int] = None,
        read_transform: bool = True,
        ignore_missing_fields: bool = True,
        **pcd_kw: Unpack[PointCloudDataKW],
    ) -> PointCloudData | Generator[PointCloudData, None, None]:
        """Load one or more point cloud from an E57 file.

        Parameters
        ----------
        path: str or Path
        retain_rgb: bool, default=True
            Flag if RGB values should be loaded (if exists)
        retain_intensity: bool, default=True
            Flag if intensity values should be loaded (if exists)
        pcd_index: int or None, default=None
            Index of the specific point cloud to load. If None, loads all scans,
            by default None.
        read_transform: bool, default=True
            Indicates if the transformation information should be read, by default True.
        ignore_missing_fields: bool, default=True
            If true, no errors are raised if fields are missing from the point cloud.
        kwargs: dict

        Returns
        -------
        PointCloudData or Generator[PointCloudData, None, None]
            Returns a single point cloud or a generator of point clouds depending
            on the value of `pcd_index`.

        Raises
        ------
        ValueError
            If `pcd_index` is provided and is out of the range [0, num_scans).

        Notes
        -----
        This is a class method intended for loading E57 point cloud data based on
        the provided parameters.
        """
        path = Path(path)
        kwargs = {
            "retain_rgb": retain_rgb,
            "retain_intensity": retain_intensity,
            "pcd_index": pcd_index,
            "ignore_missing_fields": ignore_missing_fields,
            "read_transform": read_transform,
        }

        logger.info(f"Loading E57 file: {path}")

        e57 = pye57.E57(str(path), mode="r")
        number_of_scans = e57.scan_count
        e57.close()

        point_cloud_index = 0 if number_of_scans == 1 else pcd_index

        if point_cloud_index is None:
            logger.debug(f"Loading {number_of_scans} scans from E57 file.")
            return cls._load_all_e57_scans(path, **kwargs, **pcd_kw)

        elif 0 <= point_cloud_index < number_of_scans:
            logger.debug(f"Loading scan index {point_cloud_index} from E57 file.")
            kwargs["pcd_index"] = point_cloud_index
            return cls._load_single_e57(path, **kwargs, **pcd_kw)

        else:
            raise ValueError(f"Input point cloud index passed is outside of the range [0, num_scans). Got {pcd_index}")

    @classmethod
    def save(
        cls,
        pcd: PointCloudData | Iterable[PointCloudData],
        path: str | Path,
        /,
        *,
        embed_shift_in_transform: bool = True,
        strict: bool = False,
        **config: Any,
    ) -> None:
        """Save one or more point clouds to an E57 file.

        Parameters
        ----------
        pcd : PointCloudData or Iterable[PointCloudData]
            Single point cloud or an iterable of point clouds. Each element is written
            as a separate scan inside the same E57 file.
        path : str or Path
            Output file path.  Always opened with ``mode="w"`` (truncate or create).
        embed_shift_in_transform : bool, default=True
            When ``True`` (default) and the point cloud carries a numerical-optimisation
            shift, the shifted XYZ coordinates are written as cartesianX/Y/Z and the
            shift vector is stored in the per-scan E57 translation pose.  On reload with
            ``read_transform=True``, :class:`~pchandler.geometry.OptimizedShift` is
            reconstructed automatically.

            When ``False``, the world-frame coordinates (``pcd.xyz + shift``) are
            written with an identity pose — useful for consumers that do not honour E57
            scan transforms.
        strict : bool, default=False
            When ``True``, any scalar field whose name is not in
            ``pye57.e57.SUPPORTED_POINT_FIELDS`` raises ``ValueError`` listing all
            unsupported names.  When ``False`` (default), unsupported fields are skipped
            with a ``logger.warning``.
        **config : dict
            Additional keyword arguments (reserved for future use).

        Returns
        -------
        None
        """
        logger.info(f"Saving E57 file: {path}")
        if isinstance(pcd, PointCloudData):
            pcd_iter: list[PointCloudData] = [pcd]
        else:
            pcd_iter = list(pcd)

        with pye57.E57(str(path), mode="w") as e57:
            for single_pcd in pcd_iter:
                cls._save_single_e57(
                    e57, single_pcd, embed_shift_in_transform=embed_shift_in_transform, strict=strict, **config
                )

    # TODO implement tests on file with multiple scans
    @classmethod
    def _load_all_e57_scans(cls, path, **kwargs) -> Generator[PointCloudData, None, None]:
        """Load all E57 scans from a file.

        Parameters
        ----------
        path : str
        **kwargs : dict

        Yields
        ------
        Generator[PointCloudData, None, None]
        """
        logger.debug(f"Loading multiple scans from E57 file: {path}")
        e57 = pye57.E57(str(path), mode="r")
        number_of_scans = e57.scan_count

        for i in range(number_of_scans):
            kwargs["pcd_index"] = i
            yield cls._load_single_e57(path, **kwargs)

    @classmethod
    def _load_single_e57(
        cls,  # type: ignore[override]
        path: str | Path,
        /,
        retain_rgb: bool = True,
        retain_intensity: bool = True,
        pcd_index: Optional[int] = None,
        read_transform: bool = True,
        ignore_missing_fields: bool = True,
        **pcd_kw: Unpack[PointCloudDataKW],
    ) -> PointCloudData:
        """Load a single scan from an E57 file as a :class:`PointCloudData` object."""
        logger.debug(f"Loading single scan {pcd_index} from E57 file: {path}")

        logger.debug(
            f"Reading fields:"
            f"{'\n    ' + RGB_NAMES.base if retain_rgb else ''}"
            f"{'\n    ' + INTENSITY_NAMES.base if retain_intensity else ''}"
        )

        with pye57.E57(str(path), mode="r") as e57:
            header = e57.get_header(pcd_index)

            expected_fields: tuple = tuple()
            if retain_rgb:
                expected_fields += ("colorRed", "colorGreen", "colorBlue")
            if retain_intensity:
                expected_fields += ("intensity",)

            unsupported_fields = set(header.point_fields).difference(expected_fields)

            if len(unsupported_fields) > 0:
                logger.warning(
                    f"Fields discovered in file but are not supported by pye57 and will not be loaded: "
                    f"{('\n' + field for field in unsupported_fields)}"
                )

            data = e57.read_scan(
                pcd_index,
                ignore_missing_fields=ignore_missing_fields,
                intensity=retain_intensity,
                colors=retain_rgb,
                transform=read_transform,
            )

            pcd = PointCloudData(
                np.column_stack((data["cartesianX"], data["cartesianY"], data["cartesianZ"])), **pcd_kw
            )

            if retain_rgb:
                if "colorRed" in data:
                    pcd.rgb = np.column_stack((data["colorRed"], data["colorGreen"], data["colorBlue"]))
                else:
                    logger.warning("Could not read colour information from point cloud")

            if retain_intensity:
                if "intensity" in data:
                    pcd.intensity = data["intensity"]
                else:
                    logger.warning("Could not read intensity information from point cloud")

            logger.info(f"Successfully loaded scan {pcd_index} from E57 file: {path}")

        return pcd

    @classmethod
    def _save_single_e57(  # noqa: C901  # Multi-branch E57 write — shift-mode + scalar-field policy + rgb/intensity; deferred to Phase 6 refactor.
        cls,
        e57: pye57.E57,
        pcd: PointCloudData,
        /,
        *,
        embed_shift_in_transform: bool,
        strict: bool,
        **config: Any,
    ) -> None:
        """Write a single :class:`~pchandler.PointCloudData` scan to an open pye57.E57 handle.

        Parameters
        ----------
        e57 : pye57.E57
            An already-open pye57 file handle (``mode="w"``).
        pcd : PointCloudData
            Point cloud to write.
        embed_shift_in_transform : bool
            See :meth:`save`.
        strict : bool
            See :meth:`save`.
        **config : dict
            Reserved for future use.
        """
        from pye57.e57 import SUPPORTED_POINT_FIELDS as _PYE57_SUPPORTED_POINT_FIELDS

        _supported_pye57_keys = frozenset(_PYE57_SUPPORTED_POINT_FIELDS.keys())

        # ------------------------------------------------------------------ #
        # 1. Determine pose (rotation quaternion + translation)               #
        # ------------------------------------------------------------------ #
        # Identity quaternion [w, x, y, z] per RESEARCH §"Pitfall 1".
        identity_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if pcd.numerical_optimization_shift is None:
            rotation = identity_rotation
            translation = np.zeros(3, dtype=np.float64)
            xyz_to_write = pcd.xyz
        elif embed_shift_in_transform:
            # Embed the shift in the E57 per-scan translation pose.
            # Write shifted (float32-friendly) coords; load reconstructs OptimizedShift.
            rotation = identity_rotation
            translation = pcd.numerical_optimization_shift.value.astype(np.float64)
            xyz_to_write = pcd.xyz
        else:
            # Write world-frame coordinates; identity pose (no shift in file).
            rotation = identity_rotation
            translation = np.zeros(3, dtype=np.float64)
            xyz_to_write = pcd.xyz + pcd.numerical_optimization_shift.value

        # ------------------------------------------------------------------ #
        # 2. Build the data dict for write_scan_raw                          #
        # ------------------------------------------------------------------ #
        xyz_arr = np.asarray(xyz_to_write)
        data: dict[str, np.ndarray] = {
            "cartesianX": xyz_arr[:, 0].astype(np.float64),
            "cartesianY": xyz_arr[:, 1].astype(np.float64),
            "cartesianZ": xyz_arr[:, 2].astype(np.float64),
        }

        if pcd.rgb is not None:
            rgb_arr = np.asarray(pcd.rgb)
            data["colorRed"] = rgb_arr[:, 0]
            data["colorGreen"] = rgb_arr[:, 1]
            data["colorBlue"] = rgb_arr[:, 2]

        if pcd.intensity is not None:
            data["intensity"] = np.asarray(pcd.intensity).astype(np.float32)

        # ------------------------------------------------------------------ #
        # 3. Scalar-field skip-and-warn policy (D-05)                        #
        # ------------------------------------------------------------------ #
        # Skip RGB/intensity — handled above in step 2 already.
        # pcd.scalar_fields stores rgb as 'rgb', intensity as 'intensity'.
        # SUPPORTED_SCALAR_FIELDS_MAP has 'r'/'g'/'b'/'intensity'; the base
        # name 'rgb' is not a pye57 key, so we must skip it explicitly.
        _skip_keys = frozenset({"r", "g", "b", "rgb", "intensity"})
        unsupported: list[str] = []

        for sf_name in pcd.scalar_fields.keys():
            if sf_name in _skip_keys:
                continue
            mapped_key = cls.SUPPORTED_SCALAR_FIELDS_MAP.get(sf_name, sf_name)
            if mapped_key in _supported_pye57_keys:
                data[mapped_key] = np.asarray(pcd.scalar_fields[sf_name])
            else:
                unsupported.append(sf_name)

        if unsupported:
            if strict:
                raise ValueError(
                    f"E57Handler.save: scalar fields not supported by pye57: {unsupported}. "
                    f"Supported: {sorted(_PYE57_SUPPORTED_POINT_FIELDS.keys())}"
                )
            for sf_name in unsupported:
                logger.warning(
                    "E57Handler.save: skipping unsupported scalar field %r (pye57 only supports %s)",
                    sf_name,
                    sorted(_PYE57_SUPPORTED_POINT_FIELDS.keys()),
                )

        # ------------------------------------------------------------------ #
        # 4. Write the scan                                                   #
        # ------------------------------------------------------------------ #
        e57.write_scan_raw(data, rotation=rotation, translation=translation)
