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

"""Unit-aware angle scalar and array wrappers (radians / degrees / gon)."""

from __future__ import annotations

import numbers
import re
from collections.abc import Sequence
from functools import total_ordering
from typing import Any, Generator, Self, cast

import numpy as np
from GSEGUtils.base_types import Array_Float_T, ArrayT
from GSEGUtils.util import AngleUnit, convert_angles
from numpy.typing import DTypeLike, NDArray


def _rebuild_angle(cls, internal_value, display_unit):
    """Reconstruct an :class:`Angle` / :class:`AngleArray` from pickled state.

    Parameters
    ----------
    cls : type[Angle] | type[AngleArray]
        Concrete class to reconstruct.
    internal_value : float | ArrayT
        Internal angle value (radians).
    display_unit : AngleUnit
        Original display unit.

    Returns
    -------
    Angle | AngleArray
        The reconstructed angle.
    """
    # internal_value already in INTERNAL_UNIT (rad)
    # Slotted-class pickle reconstructor: AngleBase uses __slots__ (not Pydantic),
    # so object.__new__ + manual __init__ is the standard pattern -- not a bypass.
    obj = object.__new__(cls)
    AngleBase.__init__(obj, internal_value, display_unit)
    return obj


@total_ordering
class AngleBase:
    """Base class providing storage, unit conversion, comparison and arithmetic for angles.

    Angles are always stored internally in radians; the ``display_unit``
    controls the unit used for string formatting and ``display_value``.
    """

    __slots__ = ("_internal_value", "_display_unit")

    _INTERNAL_UNIT = AngleUnit.RAD

    def __init__(self, value: float | ArrayT, unit: AngleUnit = AngleUnit.RAD):
        """Initialize an :class:`AngleBase`.

        Internally the angle is stored in radians; ``unit`` specifies the
        display unit (and the input unit of ``value``).

        Parameters
        ----------
        value : float | ArrayT
            Numeric value(s) interpreted in ``unit``.
        unit : AngleUnit, default=AngleUnit.RAD
            Unit in which ``value`` is supplied and in which the angle will be
            displayed.
        """
        value = np.array(value, dtype=float)
        # noinspection PyTypeChecker
        convert_angles(value, source_unit=unit, target_unit=self._INTERNAL_UNIT, out=value)
        self._internal_value = value
        self._display_unit = unit

    def to(self, unit: AngleUnit) -> Array_Float_T | float:
        """Convert the stored radians to ``unit`` and return the numeric value.

        Parameters
        ----------
        unit : AngleUnit
            Target unit.

        Returns
        -------
        Array_Float_T | float
            Numeric value(s) in ``unit``.
        """
        arr = self._internal_value.copy()
        # noinspection PyTypeChecker
        convert_angles(arr, source_unit=self._INTERNAL_UNIT, target_unit=unit, out=arr)
        return arr.item() if arr.ndim == 0 else arr

    def in_unit(self, unit: AngleUnit) -> Self:
        """Return a view of the angle with its display unit changed to ``unit``.

        Parameters
        ----------
        unit : AngleUnit
            Target display unit.

        Returns
        -------
        Angle | AngleArray
            New instance sharing the underlying radian array.
        """
        # 1) Allocate a new empty instance of the same class
        # Slotted-class fast-path: share the underlying radian array without re-running
        # convert_angles. AngleBase uses __slots__ (not Pydantic) -- not a bypass.
        new = object.__new__(type(self))
        # 2) Shallow‐share the internal rad‐array and swap the unit
        new._internal_value = self._internal_value
        new._display_unit = unit
        return new

    @property
    def display_unit(self) -> AngleUnit:
        """Return the display unit currently set for the angle(s).

        Returns
        -------
        AngleUnit
            The current display unit.
        """
        return self._display_unit

    @display_unit.setter
    def display_unit(self, unit: AngleUnit) -> None:
        """Set the display unit for the angle(s).

        Parameters
        ----------
        unit : AngleUnit
            New display unit.
        """
        self._display_unit = unit

    @property
    def internal_value(self) -> float | Array_Float_T:
        """Return the underlying stored angle value(s), always in radians."""
        return self._internal_value

    @property
    def display_value(self) -> Array_Float_T:
        """Return the angle value(s) converted into the current display unit."""
        out = self._internal_value.copy()
        # noinspection PyTypeChecker
        convert_angles(out, self._INTERNAL_UNIT, self._display_unit, out=out)
        return out

    @property
    def degrees(self) -> float | Array_Float_T:
        """Return a copy of the angle value(s) in degrees.

        Returns
        -------
        float | Array_Float_T
            Angle(s) in degrees.
        """
        return self.to(AngleUnit.DEGREE)

    def in_degrees(self) -> Self:
        """Return a view of the angle(s) with display unit set to degrees.

        Returns
        -------
        Angle | AngleArray
            New instance sharing the underlying radian array.
        """
        return self.in_unit(AngleUnit.DEGREE)

    @property
    def radians(self) -> float | Array_Float_T:
        """Return a copy of the angle value(s) in radians.

        Returns
        -------
        float | Array_Float_T
            Angle(s) in radians.
        """
        return self.to(AngleUnit.RAD)

    def in_radians(self) -> Self:
        """Return a view of the angle(s) with display unit set to radians.

        Returns
        -------
        Angle | AngleArray
            New instance sharing the underlying radian array.
        """
        return self.in_unit(AngleUnit.RAD)

    @property
    def gon(self) -> float | Array_Float_T:
        """Return a copy of the angle value(s) in gradians (gon).

        Returns
        -------
        float | Array_Float_T
            Angle(s) in gon.
        """
        return self.to(AngleUnit.GON)

    def in_gon(self) -> Self:
        """Return a view of the angle(s) with display unit set to gradians (gon).

        Returns
        -------
        Angle | AngleArray
            New instance sharing the underlying radian array.
        """
        return self.in_unit(AngleUnit.GON)

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray:
        """Expose the angle(s) as a numpy array (radians) for array operations.

        Parameters
        ----------
        dtype : DTypeLike, optional
            Numpy dtype for the returned array.

        Returns
        -------
        NDArray
            The internal radian array (copied to ``dtype`` if requested).
        """
        return np.array(self._internal_value, dtype=dtype)

    # DISCUSS should we return an angle object? That way the user can access the value in unit of choice
    def min(self) -> Any:
        """Return the minimum value of the angle(s) in radians."""
        return np.array(self).min()

    def max(self):
        """Return the maximum value of the angle(s) in radians."""
        return np.array(self).max()

    def __add__(self, other) -> Self:
        r"""Add another angle or a real-number / ndarray operand (interpreted as radians).

        Notes
        -----
        Per Phase 3 D-29 the ``+`` / ``-`` operators ARE expanded to allow
        ``AngleArray + AngleArray`` with matching shapes (element-wise);
        mismatched shapes raise ``ValueError`` naming both shapes. The
        previously-bare ``except Exception`` was masking these failure modes
        (and internal bugs like ``AttributeError`` on ``internal_value``) as
        misleading ``NotImplementedError``\ s; the explicit isinstance ladder
        below distinguishes the three failure modes (unsupported operand,
        shape mismatch, internal bug) and uses Python's ``return
        NotImplemented`` sentinel for the unsupported-operand path.
        """
        if isinstance(other, AngleBase):
            try:
                new_val = self.internal_value + other.internal_value
            except ValueError as err:  # numpy broadcast / shape mismatch
                raise ValueError(
                    f"AngleBase add: shape {np.shape(self.internal_value)} and "
                    f"{np.shape(other.internal_value)} do not broadcast"
                ) from err
        elif isinstance(other, (numbers.Real, np.ndarray)):
            new_val = self.radians + other
        else:
            return NotImplemented  # Python protocol; reflected dunder gets a chance

        new_instance = type(self)(new_val, self._INTERNAL_UNIT)
        new_instance.display_unit = self.display_unit
        return new_instance

    def __radd__(self, other) -> Any:
        """Right-side add (FRAG-05): delegate to ``other.__add__`` with our radian value.

        Returns ``NotImplemented`` when ``other`` is neither another
        :class:`AngleBase` nor a numeric scalar / :class:`numpy.ndarray`, so
        Python's reflected-op protocol can fall through to raising
        :class:`TypeError`.
        """
        if isinstance(other, AngleBase):
            try:
                new_val = self.internal_value + other.internal_value
            except ValueError as err:  # numpy broadcast / shape mismatch
                raise ValueError(
                    f"AngleBase radd: shape {np.shape(self.internal_value)} and "
                    f"{np.shape(other.internal_value)} do not broadcast"
                ) from err
        elif isinstance(other, (numbers.Real, np.ndarray)):
            new_val = self.radians + other
        else:
            return NotImplemented

        new_instance = type(self)(new_val, self._INTERNAL_UNIT)
        new_instance.display_unit = self.display_unit
        return new_instance

    def __sub__(self, other):
        """Subtract another angle or a real-number / ndarray operand (interpreted as radians).

        Notes
        -----
        See :meth:`__add__` notes (Phase 3 D-29). The asymmetry between
        ``__sub__`` (uses ``Angle(...)``) and ``__add__`` (uses
        ``type(self)(...)``) is preserved deliberately for Phase 3; a future
        angles-API phase reconciles via ``_wrap_result`` (CONTEXT Deferred
        Ideas / RESEARCH note).
        """
        if isinstance(other, AngleBase):
            try:
                new_val = self.internal_value - other.internal_value
            except ValueError as err:
                raise ValueError(
                    f"AngleBase sub: shape {np.shape(self.internal_value)} and "
                    f"{np.shape(other.internal_value)} do not broadcast"
                ) from err
        elif isinstance(other, (numbers.Real, np.ndarray)):
            new_val = self.radians - other
        else:
            return NotImplemented

        # NOTE (Phase 3 D-26 / Deferred Ideas): __sub__ uses Angle(...) while
        # __add__ uses type(self)(...). Asymmetry preserved deliberately for
        # Phase 3; a future angles-API phase reconciles via `_wrap_result`.
        new_instance = Angle(new_val, self._INTERNAL_UNIT)
        new_instance.display_unit = self.display_unit
        return new_instance

    def __rsub__(self, other):
        """Right-side subtract (FRAG-05): explicit isinstance ladder; ``NotImplemented`` for unsupported."""
        if isinstance(other, AngleBase):
            try:
                new_val = other.internal_value - self.internal_value
            except ValueError as err:
                raise ValueError(
                    f"AngleBase rsub: shape {np.shape(self.internal_value)} and "
                    f"{np.shape(other.internal_value)} do not broadcast"
                ) from err
        elif isinstance(other, (numbers.Real, np.ndarray)):
            new_val = other - self.radians
        else:
            return NotImplemented

        new_instance = Angle(new_val, self._INTERNAL_UNIT)
        new_instance.display_unit = self.display_unit
        return new_instance

    def __mul__(self, other):
        """Multiply by a scalar; double-AngleBase multiplication is forbidden (Phase 3 D-26)."""
        if isinstance(other, AngleBase):
            return NotImplemented  # forbid: deferred to future angles-API phase
        return Angle(self.display_value * other, self.display_unit)

    def __rmul__(self, other):
        """Right-side multiplication; same semantics as :meth:`__mul__`."""
        if isinstance(other, AngleBase):
            return NotImplemented  # forbid: deferred to future angles-API phase
        return Angle(self.display_value * other, self.display_unit)

    def __truediv__(self, other):
        """Divide by a scalar; double-AngleBase division forbidden (Phase 3 D-26)."""
        if isinstance(other, AngleBase):
            return NotImplemented  # forbid: deferred to future angles-API phase
        return Angle(self.display_value / other, self.display_unit)

    def __rtruediv__(self, other):
        """Reject division when the divisor is an :class:`AngleBase` (Phase 3 D-26)."""
        if isinstance(other, AngleBase):
            return NotImplemented  # forbid: deferred to future angles-API phase
        return NotImplemented  # other / Angle is undefined for any operand

    def __mod__(self, other: Any) -> Self | float:
        """Compute ``self % other`` (double-AngleBase modulo forbidden; Phase 3 D-26)."""
        if isinstance(other, AngleBase):
            return NotImplemented  # forbid: deferred to future angles-API phase
        # return self.__binary_op(other, np.mod)
        mod_val = self.display_value % other
        return type(self)(mod_val, self.display_unit)

    def __rmod__(self, other):
        """Reject modulo when the divisor is an :class:`AngleBase` (Phase 3 D-26)."""
        if isinstance(other, AngleBase):
            return NotImplemented  # forbid: deferred to future angles-API phase
        return NotImplemented  # other % Angle is undefined for any operand

    # def __divmod__(self, other):
    #     if isinstance(other, AngleBase):
    #         raise NotImplementedError(f"Modulo not defined between two AngleBase types.")
    #     return self.__binary_op(other, divmod)

    def _compare(self, other, op):
        """Compare the current angle with ``other`` using a numpy ufunc.

        Parameters
        ----------
        other : AngleBase | float
            Right-hand operand.
        op : ufunc
            A numpy universal function such as :func:`numpy.equal` or
            :func:`numpy.less`.

        Returns
        -------
        bool or NDArray[bool]
            Element-wise comparison result.
        """
        # Pull out the raw float/array to compare
        if isinstance(other, AngleBase):
            if self._INTERNAL_UNIT == other._INTERNAL_UNIT:
                lhs = self._internal_value
                rhs = other._internal_value
            else:
                lhs = self._internal_value
                rhs = other.to(self._INTERNAL_UNIT)
        else:
            lhs = self.display_value
            rhs = other
        # op is a ufunc like np.equal, np.less, etc.
        return op(lhs, rhs)

    def __eq__(self, other):
        """Element-wise equality, reduced via ``np.all`` to a single bool."""
        return bool(np.all(self._compare(other, np.equal)))

    def __lt__(self, other):
        """Element-wise less-than comparison (used by :func:`total_ordering`)."""
        return self._compare(other, np.less)

    # __le__, __gt__, __ge__, __ne__ provided by total_ordering

    def __hash__(self):
        """Hash by quantized radian value (10 decimal places) for stability."""
        # 1) get a canonical float
        rad = self.to(AngleUnit.RAD)
        # 2) either hash the float directly (exact), or
        #    quantize it to avoid weird float artifacts:
        quant = np.round(rad, 10)
        return hash(quant)

    def __reduce__(self):
        """Return the (callable, state) tuple used by :mod:`pickle`."""
        return _rebuild_angle, (type(self), self.display_value, self._display_unit)


class Angle(AngleBase):
    """Represent a scalar (or, via ``__new__`` dispatch, array) angle in any supported unit.

    Supports flexible construction from numbers, strings (``"45deg"``,
    ``"0.5 rad"``, ``"200.4gon"``), and array-like inputs. Array inputs are
    upcast to :class:`AngleArray`.

    Parameters
    ----------
    value : float
        The numerical representation of the angle.
    unit : AngleUnit
        The unit of the angle, such as radians or degrees.
    """

    def __new__(cls, value: float | Array_Float_T | str, unit: AngleUnit = AngleUnit.RAD):
        """Return a new :class:`Angle` or :class:`AngleArray` based on the input shape.

        Parameters
        ----------
        value : float, Array_Float_T, or str
            The angle value(s). If a string, it is parsed to create an
            instance.
        unit : AngleUnit, default=AngleUnit.RAD
            The unit of the angle value(s).

        Returns
        -------
        Angle or AngleArray
            An instance of either :class:`Angle` or :class:`AngleArray`
            depending on the dimension of the input value.
        """
        if isinstance(value, str):
            return cls.parse(value)

        arr = np.array(value, dtype=float)

        target = Angle if arr.ndim == 0 else AngleArray
        inst = super().__new__(target)
        AngleBase.__init__(inst, arr, unit)
        return inst

    @classmethod
    def parse(cls, value: Any) -> Self:  # noqa: C901  # Multi-format parser — branching tracks supported input shapes; refactor deferred to Phase 6.
        """Build an :class:`Angle` / :class:`AngleArray` from a variety of input formats.

        Supported formats include:

        * Angle | AngleArray
        * (value, unit) tuple
        * single string ``"45deg"``, ``"0.5 rad"``, ``"200.4gon"``, etc.
        * numpy scalar
        * int | float
        * numpy array
        * list | tuple of numbers

        Parameters
        ----------
        value : Any
            Input value to parse into an angle.

        Returns
        -------
        Angle | AngleArray
            The parsed angle.

        Raises
        ------
        ValueError
            If ``value`` cannot be parsed as an angle.
        """
        # 0) is already Angle
        if isinstance(value, AngleBase):
            return cast(Self, value)
        # 1) (value, unit) tuple
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], AngleUnit):
            return cls(value[0], value[1])

        # 2) single string "45deg", "0.5 rad", etc.
        if isinstance(value, str):
            m = re.match(r"^\s*([+-]?[0-9]*\.?[0-9]+)\s*(°|deg|rad|gon)\s*$", value, re.I)
            if m:
                val, suf = float(m.group(1)), m.group(2).lower()
                unit = {
                    "°": AngleUnit.DEGREE,
                    "deg": AngleUnit.DEGREE,
                    "rad": AngleUnit.RAD,
                    "gon": AngleUnit.GON,
                }[suf]
                return cls(val, unit)
            else:
                raise ValueError(f"Cannot parse angle from {value!r}")

        # 3) numpy scalar
        if isinstance(value, np.generic):
            return cls(float(value), cls._INTERNAL_UNIT)

        # 4) bare Python scalar
        if isinstance(value, (int, float)):
            return cls(value, cls._INTERNAL_UNIT)

        # 5) numpy array
        if isinstance(value, np.ndarray):
            return cls(value, cls._INTERNAL_UNIT)

        # 6) any other Sequence (list or tuple) but *not* a string
        if isinstance(value, Sequence):
            # empty sequence → treat as empty array
            if len(value) == 0:
                return cls(np.array([], dtype=float), cls._INTERNAL_UNIT)

            # all‐numbers → straightforward array
            if all(isinstance(x, (int, float, np.generic)) for x in value):
                return cls(np.array(value, dtype=float), cls._INTERNAL_UNIT)

            # all‐strings → parse each element and collect radians
            if all(isinstance(x, str) for x in value):
                parsed = [Angle.parse(x).internal_value for x in value]
                return cls(np.array(parsed, dtype=float), cls._INTERNAL_UNIT)

            # mixed or unsupported → error out
            raise ValueError(f"Cannot parse angle sequence: {value!r}")

        # 7) give up
        raise ValueError(f"Cannot parse angle from {value!r}")

    @classmethod
    def __get_validators__(cls) -> Generator:
        """Yield the pydantic validators used to coerce inputs into an :class:`Angle`."""
        yield cls._validate

    @classmethod
    def _validate(cls, v: Any, field: Any) -> Self:
        """Pydantic-style validator: delegate to :meth:`parse`."""
        return cls.parse(v)

    def __float__(self) -> float:
        """Allow ``float(Angle)`` for scalar cases (returns the stored radian value)."""
        # allow float(Angle) for scalar cases
        return float(self._internal_value)

    def __repr__(self):
        """Return a debug-friendly representation: ``Angle(value, unit=...)``."""
        val = self.to(self._display_unit)
        return f"{self.__class__.__name__}({val:.4f}, unit={self._display_unit.name})"

    def __str__(self):
        """Return a human-friendly string: ``"<value> <unit>"`` in the display unit."""
        val = self.to(self._display_unit)
        return f"{val:.4f} {self._display_unit.value}"


class AngleArray(AngleBase):
    """Multi-dimensional array of angles, supporting indexing, iteration and unit conversion.

    Parameters
    ----------
    arr : ArrayT
        The array of angle values.
    unit : AngleUnit, default=AngleUnit.RAD
        The unit in which ``arr`` is supplied (also the initial display unit).
    """

    def __new__(cls, arr: ArrayT, unit: AngleUnit = AngleUnit.RAD):
        """Return a new :class:`AngleArray` carrying ``arr`` (a copy as float64)."""
        arr = np.array(arr, dtype=float)
        inst = super().__new__(cls)
        AngleBase.__init__(inst, arr, unit)
        return inst

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying array.

        Returns
        -------
        tuple[int, ...]
            Shape tuple, matching :attr:`numpy.ndarray.shape`.
        """
        return self._internal_value.shape

    def __len__(self) -> int:
        """Return the length of the first dimension of the underlying array."""
        return self._internal_value.shape[0]

    def __getitem__(self, idx):
        """Index into the array; scalar indices return an :class:`Angle`, slices return an :class:`AngleArray`."""
        # return either an Angle (if scalar) or AngleArray
        sub = self.internal_value[idx]
        if np.ndim(sub) == 0:
            new_angle = Angle(float(sub), self._INTERNAL_UNIT)
            new_angle.display_unit = self._display_unit
            return new_angle
        new_angle_array = AngleArray(sub, self._INTERNAL_UNIT)
        new_angle_array.display_unit = self._display_unit
        return new_angle_array

    def __iter__(self):
        """Iterate over scalar :class:`Angle` instances, one per element."""
        for x in self._internal_value:
            iter_angle = Angle(float(x), self._INTERNAL_UNIT)
            iter_angle.display_unit = self._display_unit
            yield iter_angle

    def __repr__(self):
        """Return a debug-friendly representation including shape, unit and a value preview."""
        vals = self.to(self._display_unit)
        preview = np.array2string(vals, threshold=4)
        return (
            f"{self.__class__.__name__}(shape={self._internal_value.shape}, "
            f"unit={self._display_unit.name}, values={preview})"
        )

    def __str__(self):
        """Return a human-friendly array preview in the current display unit."""
        vals = self.to(self._display_unit)
        preview = np.array2string(vals, threshold=4)
        return f"{preview} {self.display_unit.name}"

    def __eq__(self, other):
        """Element-wise equality against another :class:`AngleBase` (radian compare)."""
        if isinstance(other, AngleBase):
            return self.internal_value == other.internal_value
        raise NotImplementedError()

    def __lt__(self, other):
        """Element-wise comparison is not implemented for :class:`AngleArray`."""
        raise NotImplementedError()

    def __ne__(self, other):
        """Element-wise comparison is not implemented for :class:`AngleArray`."""
        raise NotImplementedError()

    def __le__(self, other):
        """Element-wise comparison is not implemented for :class:`AngleArray`."""
        raise NotImplementedError()

    def __gt__(self, other):
        """Element-wise comparison is not implemented for :class:`AngleArray`."""
        raise NotImplementedError()

    def __ge__(self, other):
        """Element-wise comparison is not implemented for :class:`AngleArray`."""
        raise NotImplementedError()
