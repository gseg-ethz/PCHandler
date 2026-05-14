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

"""Project-local string and dtype type aliases shared across :mod:`pchandler`.

Provides :data:`LowerStr` (a lower-cased, stripped ``str`` for scalar-field
names), :data:`SfNameT` (the optional variant used as the canonical
``scalar_field`` name annotation), :class:`DtypeDict` (the structured-array
schema accepted by NumPy ``dtype`` constructors), and :class:`PointCloudDataKW`
(PEP 692 ``Unpack``-friendly TypedDict of every optional kwarg accepted by
:class:`pchandler.PointCloudData`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, NotRequired, Optional, Sequence, TypedDict

import numpy.typing as npt
from pydantic import StringConstraints

if TYPE_CHECKING:
    # All type references in :class:`PointCloudDataKW` below are forward-ref
    # strings so the runtime module stays cycle-free and import-light. The
    # pre-commit mypy isolated venv lacks ``GSEGUtils`` (see
    # ``.pre-commit-config.yaml`` mirrors-mypy ``additional_dependencies``
    # comment) — ``[mypy-GSEGUtils.base_types] ignore_missing_imports`` in
    # ``mypy.ini`` reconciles the two environments without re-enabling the
    # broader ``[mypy-GSEGUtils.*]`` override dropped in Plan 01-03b.
    from GSEGUtils.base_types import (  # noqa: F401
        Array_Nx3_Float_T,
        Array_Nx3_T,
        Array_Nx3_Uint8_T,
        Vector_3_Float_T,
        VectorT,
    )
    from numpy.typing import NDArray  # noqa: F401

    from pchandler.geometry import OptimizedShift  # noqa: F401
    from pchandler.geometry.util import MinMaxPoints  # noqa: F401
    from pchandler.scalar_fields import (  # noqa: F401
        NormalFields,
        RGBFields,
        ScalarField,
        ScalarFieldManager,
        ScalarFieldTriplet,
    )

LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True)]
SfNameT = Optional[LowerStr]


class DtypeDict(TypedDict):
    """Structured-array dtype description compatible with :func:`numpy.dtype`.

    Attributes
    ----------
    names : list[LowerStr]
        Field names in declaration order.
    formats : list[npt.DTypeLike]
        Per-field dtype specifications, parallel to ``names``.
    """

    names: list[LowerStr]
    formats: list[npt.DTypeLike]


class PointCloudDataKW(TypedDict, total=False):
    """Optional keyword arguments accepted by :class:`pchandler.PointCloudData`.

    PEP 692 ``Unpack``-friendly TypedDict mirroring the full optional kwarg
    surface of :class:`pchandler.PointCloudData.__init__`. Structurally a
    superset of :class:`pchandler.geometry.coordinates.CartesianKwFull` so the
    PCD ``**kwargs`` annotation can drop the prior
    ``# type: ignore[call-overload]`` previously needed when forwarding to
    :class:`CartesianCoordinates`.

    Used as ``**kwargs: Unpack[PointCloudDataKW]`` on:

    * :class:`pchandler.PointCloudData.__init__`
    * :meth:`pchandler.PointCloudData.merge`
    * :meth:`pchandler.data_io.PlyHandler.load`
    * :meth:`pchandler.data_io.CsvHandler.load`
    * :meth:`pchandler.data_io.LasHandler.load`
    * :meth:`pchandler.data_io.E57Handler.load` (and ``_load_single_e57``;
      the Phase 5 D-24 exemplar).

    ``xyz`` is intentionally NOT included — it remains an explicit
    positional-only parameter on ``PointCloudData.__init__``. ``arr`` IS
    included for structural-superset symmetry with
    :class:`CartesianKwFull`, even though
    :class:`pchandler.geometry.coordinates.CartesianCoordinates` handles it
    inside ``__init__`` rather than as a user-facing PCD kwarg — the
    structural-superset relationship lets the PCD ``**kwargs`` annotation
    drop ``# type: ignore[call-overload]`` cleanly.

    Attributes
    ----------
    rgb : RGBFields | Array_Nx3_Float_T | Array_Nx3_Uint8_T | None
        Optional RGB colour per point.
    normals : NormalFields | Array_Nx3_Float_T | None
        Per-point normal vectors (normalised to unit length on assignment).
    intensity : ScalarField | VectorT | None
        Optional intensity scalar field.
    reflectance : ScalarField | VectorT | None
        Optional reflectance scalar field.
    scalar_fields : ScalarFieldManager | dict[...] | None
        Additional custom scalar fields (manager or name->data mapping).
    socs_origin : Vector_3_Float_T | None
        Scan original coordinate-system origin (used for conversion to
        spherical coordinates).
    project_transformation : NDArray | None
        4x4 affine transform from scan coordinates to project coordinates.
    numerical_optimization_shift : OptimizedShift | None
        Pre-existing numerical-precision shift to attach to the cloud.
    unshifted_bbox : MinMaxPoints | None
        Cached unshifted bounding box (computed field; typically not
        user-supplied).
    _shift_applied_by : OptimizedShift | None
        Bookkeeping field used by :class:`CartesianCoordinates` to track the
        shift instance currently applied to ``arr``.
    arr : Array_Nx3_T
        Underlying ``(N, 3)`` coordinate array. Present for structural-superset
        symmetry with :class:`CartesianKwFull`; not a typical PCD kwarg.
    """

    rgb: NotRequired[Optional["RGBFields | Array_Nx3_Float_T | Array_Nx3_Uint8_T"]]
    normals: NotRequired[Optional["NormalFields | Array_Nx3_Float_T"]]
    intensity: NotRequired[Optional["ScalarField | VectorT"]]
    reflectance: NotRequired[Optional["ScalarField | VectorT"]]
    scalar_fields: NotRequired[
        Optional[
            "ScalarFieldManager | dict[str, ScalarField | ScalarFieldTriplet | Array_Nx3_T | VectorT | Sequence[Any]]"
        ]
    ]
    socs_origin: NotRequired[Optional["Vector_3_Float_T"]]
    project_transformation: NotRequired[Optional["NDArray[Any]"]]
    numerical_optimization_shift: NotRequired[Optional["OptimizedShift"]]
    unshifted_bbox: NotRequired[Optional["MinMaxPoints"]]
    _shift_applied_by: NotRequired[Optional["OptimizedShift"]]
    arr: NotRequired["Array_Nx3_T"]
