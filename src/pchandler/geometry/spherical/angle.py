# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import total_ordering
from typing import Any, Generator, Self, cast

import numpy as np
from GSEGUtils.base_types import Array_Float_T, ArrayT
from GSEGUtils.util import AngleUnit, convert_angles
from numpy.typing import DTypeLike, NDArray


def _rebuild_angle(cls, internal_value, display_unit):
    """Reconstruct method for __reduce__

    Parameters
    ----------
    cls: type[Angle] | type[AngleArray]
    internal_value: float | ArrayT
    display_unit: AngleUnit

    Returns
    -------
    Angle | AngleArray
    """
    # internal_value already in INTERNAL_UNIT (rad)
    obj = object.__new__(cls)
    AngleBase.__init__(obj, internal_value, display_unit)
    return obj


@total_ordering
class AngleBase:
    __slots__ = ("_internal_value", "_display_unit")

    _INTERNAL_UNIT = AngleUnit.RAD

    def __init__(self, value: float | ArrayT, unit: AngleUnit = AngleUnit.RAD):
        """AngleBase class provides the basic functions for interacting with angles

        These include storage, unit conversion, comparison and numerical operations.

        Internally, the angles are stored in radians.

        Parameters
        ----------
        value: float | ArrayT
        unit: AngleUnit
            Unit which the angle should be displayed in.
        """
        value = np.array(value, dtype=float)
        # noinspection PyTypeChecker
        convert_angles(value, source_unit=unit, target_unit=self._INTERNAL_UNIT, out=value)
        self._internal_value = value
        self._display_unit = unit

    def to(self, unit: AngleUnit) -> Array_Float_T | float:
        """Convert stored radians → `unit`.

        Parameters
        ----------
        unit: AngleUnit

        Returns
        -------
        Array_Float_T | float
        """
        arr = self._internal_value.copy()
        # noinspection PyTypeChecker
        convert_angles(arr, source_unit=self._INTERNAL_UNIT, target_unit=unit, out=arr)
        return arr.item() if arr.ndim == 0 else arr

    def in_unit(self, unit: AngleUnit) -> Self:
        """Return a view of the object converted to the specified unit.

        Parameters
        ----------
        unit: AngleUnit

        Returns
        -------
        Angle | AngleArray
        """
        # 1) Allocate a new empty instance of the same class
        new = object.__new__(type(self))
        # 2) Shallow‐share the internal rad‐array and swap the unit
        new._internal_value = self._internal_value
        new._display_unit = unit
        return new

    @property
    def display_unit(self) -> AngleUnit:
        """Display unit set for this angle(s)

        Returns
        -------
        AngleUnit
        """
        return self._display_unit

    @display_unit.setter
    def display_unit(self, unit: AngleUnit) -> None:
        """Set the display unit for this angle(s)

        Parameters
        ----------
        unit

        Returns
        -------

        """
        self._display_unit = unit

    @property
    def internal_value(self) -> float | Array_Float_T:
        """Returns the underlying stored angle(s) (radians)"""
        return self._internal_value

    @property
    def display_value(self) -> Array_Float_T:
        """Returns the angle(s) in the display unit (degrees, radians or gon)"""
        out = self._internal_value.copy()
        # noinspection PyTypeChecker
        convert_angles(out, self._INTERNAL_UNIT, self._display_unit, out=out)
        return out

    @property
    def degrees(self) -> float | Array_Float_T:
        """Returns a copy of the angle(s) in degrees

        Returns
        -------
        float | Array_Float_T
        """
        return self.to(AngleUnit.DEGREE)

    def in_degrees(self) -> Self:
        """Returns a view of the angle(s) in degrees

        Returns
        -------
        Angle | AngleArray
        """
        return self.in_unit(AngleUnit.DEGREE)

    @property
    def radians(self) -> float | Array_Float_T:
        """Returns a copy of the angle(s) in radians

        Returns
        -------
        float | Array_Float_T
        """
        return self.to(AngleUnit.RAD)

    def in_radians(self) -> Self:
        """Returns a view of the angle(s) in radians

        Returns
        -------
        Angle | AngleArray
        """
        return self.in_unit(AngleUnit.RAD)

    @property
    def gon(self) -> float | Array_Float_T:
        """Returns a copy of the angle(s) in gradians(gon)

        Returns
        -------
        float | Array_Float_T
        """
        return self.to(AngleUnit.GON)

    def in_gon(self) -> Self:
        """Returns a view of the angle(s) in gradians(gon)

        Returns
        -------
        Angle | AngleArray
        """
        return self.in_unit(AngleUnit.GON)

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray:
        """Enables numpy array operations on the angle(s)

        Parameters
        ----------
        dtype: DTypeLike | None

        Returns
        -------
        NDArray
        """
        return np.array(self._internal_value, dtype=dtype)

    # DISCUSS should we return an angle object? That way the user can access the value in unit of choice
    def min(self) -> Any:
        """Returns the minimum value of the angle(s) in radians"""
        return np.array(self).min()

    def max(self):
        """Returns the maximum value of the angle(s) in radians"""
        return np.array(self).max()

    def __add__(self, other) -> Self:
        try:
            if isinstance(other, AngleBase):
                new_val = self.internal_value + other.internal_value
            else:
                new_val = self.radians + other
        except Exception as err:
            raise NotImplementedError(f"Add not defined between type: {type(other)} and {type(self)}") from err

        new_instance = type(self)(new_val, self._INTERNAL_UNIT)
        new_instance.display_unit = self.display_unit
        return new_instance

    def __radd__(self, other) -> Any:
        try:
            return other.__add__(self.radians)
        except Exception as err:
            raise NotImplementedError(f"Add not defined between type: {type(other)} and {type(self)}") from err

    def __sub__(self, other):
        try:
            if isinstance(other, AngleBase):
                new_val = self.internal_value - other.internal_value
            else:
                new_val = self.radians - other
        except Exception as err:
            raise NotImplementedError(f"Subtraction not defined between type: {type(other)} and {type(self)}") from err

        new_instance = Angle(new_val, self._INTERNAL_UNIT)
        new_instance.display_unit = self.display_unit
        return new_instance

    def __rsub__(self, other):
        try:
            return other.__sub__(self.radians)
        except Exception as err:
            raise NotImplementedError(
                f"Subtraction not defined between types: {type(other)} and {type(self)}."
            ) from err

    def __mul__(self, other):
        if isinstance(other, AngleBase):
            raise NotImplementedError("Multiplication not defined between two AngleBase types.")
        return Angle(self.display_value * other, self.display_unit)

    def __rmul__(self, other):
        return self.__mul__(other)
        # if isinstance(other, AngleBase):
        #     raise NotImplementedError(f"Multiplication not defined between two AngleBase types.")
        # return self.__binary_op(other, np.multiply)

    def __truediv__(self, other):
        if isinstance(other, AngleBase):
            return self.internal_value / other.internal_value
        return Angle(self.display_value / other, self.display_unit)

    def __rtruediv__(self, other):
        raise NotImplementedError("Division not with AngleBase as divisor.")
        # other / self
        # if isinstance(other, AngleBase):
        #     vals = np.divide(other.to(other.display_unit), self.to(other.display_unit))
        #     unit = other._display_unit
        #     return Angle(vals, unit)
        # else:
        #     vals = np.divide(other, self.to(self.display_unit))
        #     return Angle(vals, self.display_unit)

    def __mod__(self, other: Any) -> Self | float:
        if isinstance(other, AngleBase):
            raise NotImplementedError("Modulo not defined between two AngleBase types.")
        # return self.__binary_op(other, np.mod)
        mod_val = self.display_value % other
        return type(self)(mod_val, self.display_unit)

    def __rmod__(self, other):
        raise NotImplementedError("Modulo not defined for divisor as AngleBase.")

    # def __divmod__(self, other):
    #     if isinstance(other, AngleBase):
    #         raise NotImplementedError(f"Modulo not defined between two AngleBase types.")
    #     return self.__binary_op(other, divmod)

    def _compare(self, other, op):
        """Compares the current object with another using a specified operator.

        Parameters
        ----------
        other : AngleBase | float
        op : ufunc
            A numpy universal function such as `np.equal` or `np.less`

        Returns
        -------
        bool or NDArray[bool]
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
        return bool(np.all(self._compare(other, np.equal)))

    def __lt__(self, other):
        return self._compare(other, np.less)

    # __le__, __gt__, __ge__, __ne__ provided by total_ordering

    def __hash__(self):
        # 1) get a canonical float
        rad = self.to(AngleUnit.RAD)
        # 2) either hash the float directly (exact), or
        #    quantize it to avoid weird float artifacts:
        quant = np.round(rad, 10)
        return hash(quant)

    def __reduce__(self):
        return _rebuild_angle, (type(self), self.display_value, self._display_unit)


class Angle(AngleBase):
    """
    Represents an angle in various units, such as degrees, radians, or gons. Provides functionality to
    create an angle instance from different data types and formats.

    The class supports scalar and array-based input, with automatic differentiation between single values
    and sequences. It also enables flexible parsing from strings, tuples, and other formats.

    Parameters
    ----------
    value : float
        The numerical representation of the angle.
    unit : AngleUnit
        The unit of the angle, such as radians or degrees.
    """

    def __new__(cls, value: float | Array_Float_T | str, unit: AngleUnit = AngleUnit.RAD):
        """Returns a new instance of Angle or AngleArray based on the input value and unit.

        Parameters
        ----------
        value : float, Array_Float_T, or str
            The angle value(s). If a string, it is parsed to create an instance.

        unit : AngleUnit, default AngleUnit.RAD
            The unit of the angle value(s). Defaults to radians.

        Returns
        -------
        Angle or AngleArray
            An instance of either Angle or AngleArray depending on the dimension
            of the input value.
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
        """Create an angle from a variety of input formats.

        Supported formats include:

        * Angle | AngleArray
        * (value, unit) tuple
        * single string "45deg", "0.5 rad", "200.4gon", etc.
        * numpy scalar
        * int | float
        * numpy array
        * list | tuple of numbers

        Parameters
        ----------
        value: Any

        Returns
        -------
        Angle | AngleArray
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
        yield cls._validate

    @classmethod
    def _validate(cls, v: Any, field: Any) -> Self:
        return cls.parse(v)

    def __float__(self) -> float:
        # allow float(Angle) for scalar cases
        return float(self._internal_value)

    def __repr__(self):
        val = self.to(self._display_unit)
        return f"{self.__class__.__name__}({val:.4f}, unit={self._display_unit.name})"

    def __str__(self):
        val = self.to(self._display_unit)
        return f"{val:.4f} {self._display_unit.value}"


class AngleArray(AngleBase):
    """
    Represents a multi-dimensional array of angle values.

    Provides functionality for working with arrays of angles, including indexing,
    iterability, and unit conversions.

    Parameters
    ----------
    unit : AngleUnit
        The unit of the angle array internally stored.
    display_unit : AngleUnit
        The unit in which the angle values are displayed.
    """

    def __new__(cls, arr: ArrayT, unit: AngleUnit = AngleUnit.RAD):
        arr = np.array(arr, dtype=float)
        inst = super().__new__(cls)
        AngleBase.__init__(inst, arr, unit)
        return inst

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the array

        Returns
        -------
        tuple[int, ...]
        """
        return self._internal_value.shape

    def __len__(self) -> int:
        return self._internal_value.shape[0]

    def __getitem__(self, idx):
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
        for x in self._internal_value:
            iter_angle = Angle(float(x), self._INTERNAL_UNIT)
            iter_angle.display_unit = self._display_unit
            yield iter_angle

    def __repr__(self):
        vals = self.to(self._display_unit)
        preview = np.array2string(vals, threshold=4)
        return (
            f"{self.__class__.__name__}(shape={self._internal_value.shape}, "
            f"unit={self._display_unit.name}, values={preview})"
        )

    def __str__(self):
        vals = self.to(self._display_unit)
        preview = np.array2string(vals, threshold=4)
        return f"{preview} {self.display_unit.name}"

    def __eq__(self, other):
        if isinstance(other, AngleBase):
            return self.internal_value == other.internal_value
        raise NotImplementedError()

    def __lt__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        raise NotImplementedError()

    def __le__(self, other):
        raise NotImplementedError()

    def __gt__(self, other):
        raise NotImplementedError()

    def __ge__(self, other):
        raise NotImplementedError()
