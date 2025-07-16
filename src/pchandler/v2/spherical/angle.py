import re
from collections.abc import Sequence
from functools import total_ordering

import numpy as np
from narwhals.stable.v1 import exclude
from numpy.typing import NDArray
from typing import Any, Generator, Self, overload
from ..util import AngleUnit, convert_angles
from ..base_types import ArrayT

@total_ordering
class AngleBase:
    __slots__ = ("_internal_value", "_display_unit")

    _INTERNAL_UNIT = AngleUnit.RAD

    def __init__(self, value: float | ArrayT, unit: AngleUnit):
        # convert_angles(rad,
        #                source_unit=unit,
        #                target_unit=AngleUnit.RAD,
        #                out=rad)
        arr = np.array(value, dtype=float)
        convert_angles(arr, source_unit=unit, target_unit=self._INTERNAL_UNIT, out=arr)
        self._internal_value = arr
        self._display_unit = unit


    def to(self, unit: AngleUnit) -> float | NDArray[np.floating]:
        """
        Convert stored radians → `unit`.
        Returns either a scalar float (if input was scalar)
        or an ndarray of floats.
        """
        arr = self._internal_value.copy()
        convert_angles(arr,
                       source_unit=self._INTERNAL_UNIT,
                       target_unit=unit,
                       out=arr)
        return arr.item() if arr.ndim == 0 else arr

    def in_unit(self, unit: AngleUnit) -> Self:
        """
        Return a *view* of this object with the same underlying
        ._rad array in radians, but reporting a different unit.
        """
        # 1) Allocate a new empty instance of the same class
        new = object.__new__(type(self))
        # 2) Shallow‐share the internal rad‐array and swap the unit
        new._internal_value  = self._internal_value
        new._display_unit = unit
        return new

    @property
    def display_unit(self) -> AngleUnit:
        return self._display_unit

    @display_unit.setter
    def display_unit(self, unit: AngleUnit) -> None:
        self._display_unit = unit

    @property
    def internal_value(self) -> float | NDArray[np.floating]:
        return self._internal_value

    @property
    def display_value(self) -> float | NDArray[np.floating]:
        out = self._internal_value.copy()
        convert_angles(out,self._INTERNAL_UNIT,self._display_unit,out=out)
        return out

    @property
    def degrees(self) -> Self:
        return self.to(AngleUnit.DEGREE)

    def in_degrees(self) -> Self:
        return self.in_unit(AngleUnit.DEGREE)

    @property
    def radians(self):
        return self.to(AngleUnit.RAD)

    def in_radians(self) -> Self:
        return self.in_unit(AngleUnit.RAD)

    @property
    def gon(self):
        return self.to(AngleUnit.GON)

    def in_gon(self) -> Self:
        return self.in_unit(AngleUnit.GON)


    def __array__(self, dtype=None) -> NDArray:
        """
        This makes np.asarray(angle) produce the raw radian array,
        so most numpy functions will operate on ._rad directly.
        """
        return np.array(self._internal_value, dtype=dtype)

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    #     """
    #     Intercept numpy ufuncs (like add, sin, etc).
    #     We extract raw radians, apply the ufunc, and wrap back in Angle.
    #     """
    #     # Extract `out` keyword
    #     out = kwargs.pop("out", None)
    #
    #
    #
    #     # Run trigonometric functions on RADIANS
    #     TRIG_FUNC = {
    #         np.sin, np.cos, np.tan,
    #         np.sinh, np.cosh, np.tanh,
    #     }
    #     if ufunc in TRIG_FUNC:
    #         return getattr(ufunc, method)(inputs[0].radians, **kwargs)
    #
    #     # extract raw arrays from any Angle inputs
    #     args = []
    #     all_inputs_AngleBase = all(isinstance(x, AngleBase) for x in inputs)
    #     display_units = {x.display_unit for x in inputs if isinstance(x, AngleBase)}
    #     if len(display_units) != 1:
    #         raise NotImplementedError(f"When mixing non-Angle and AngleBase inputs, all AngleBase inputs must share the same display_unit.")
    #     unit_to_use = inputs[0].display_unit if all_inputs_AngleBase else display_units.pop()
    #
    #     for x in inputs:
    #         if isinstance(x, AngleBase):
    #             args.append(x.to(unit_to_use))
    #         else:
    #             args.append(x)
    #     result = getattr(ufunc, method)(*args, **kwargs)
    #     # If ufunc returns a tuple, wrap each; else wrap single
    #     if isinstance(result, tuple):
    #         return tuple(Angle(r, unit_to_use) for r in result)
    #     return Angle(result, unit_to_use)
    #
    # def __binary_op(self, other, ufunc):
    #     if isinstance(other, AngleBase):
    #         vals = ufunc(self.to(self.display_unit), other.to(self.display_unit))
    #         unit = self.display_unit
    #     else:
    #         vals = ufunc(self.to(self.display_unit), other)
    #         unit = self.display_unit
    #
    #     return Angle(vals, unit)

    # def _add_subtract(self, other, op):


    def __add__(self, other) -> Self:
        try:
            if isinstance(other, AngleBase):
                new_val = self.internal_value + other.internal_value
            else:
                new_val = self.radians + other
        except Exception:
            raise NotImplementedError(f"Add not defined between type: {type(other)} and {type(self)}")

        new_instance = Angle(new_val, self._INTERNAL_UNIT)
        new_instance.display_unit = self.display_unit
        return new_instance

    def __radd__(self, other) -> Any:
        try:
            return other.__add__(self.radians)
        except Exception:
            raise NotImplementedError(f"Add not defined between type: {type(other)} and {type(self)}")

    def __sub__(self, other):
        try:
            if isinstance(other, AngleBase):
                new_val = self.internal_value - other.internal_value
            else:
                new_val = self.radians - other
        except Exception:
            raise NotImplementedError(f"Subtraction not defined between type: {type(other)} and {type(self)}")

        new_instance = Angle(new_val, self._INTERNAL_UNIT)
        new_instance.display_unit = self.display_unit
        return new_instance

    def __rsub__(self, other):
        try:
            return other.__sub__(self.radians)
        except Exception:
            raise NotImplementedError(f"Subtraction not defined between types: {type(other)} and {type(self)}.")

    def __mul__(self, other):
        if isinstance(other, AngleBase):
            raise NotImplementedError(f"Multiplication not defined between two AngleBase types.")
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
        raise NotImplementedError(f"Division not with AngleBase as divisor.")
        # other / self
        # if isinstance(other, AngleBase):
        #     vals = np.divide(other.to(other.display_unit), self.to(other.display_unit))
        #     unit = other._display_unit
        #     return Angle(vals, unit)
        # else:
        #     vals = np.divide(other, self.to(self.display_unit))
        #     return Angle(vals, self.display_unit)

    @overload
    def __mod__(self, other: Self) -> float: ...


    def __mod__(self, other: Any) -> Self | float:
        if isinstance(other, AngleBase):
            raise NotImplementedError(f"Modulo not defined between two AngleBase types.")
        # return self.__binary_op(other, np.mod)
        mod_val = self.display_value % other
        return Angle(mod_val, self.display_unit)

    def __rmod__(self, other):
        raise NotImplementedError(f"Modulo not defined for divisor as AngleBase.")

    # def __divmod__(self, other):
    #     if isinstance(other, AngleBase):
    #         raise NotImplementedError(f"Modulo not defined between two AngleBase types.")
    #     return self.__binary_op(other, divmod)


    def _compare(self, other, op):
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
        quant = round(rad, 10)
        return hash(quant)

class Angle(AngleBase):

    def __new__(cls, value: float | ArrayT, unit: AngleUnit = AngleUnit.RAD):
        """
            Selects Angle or AngleArray based on if value is scalar or an array.
        """

        arr = np.array(value, dtype=float)

        target = Angle if arr.ndim == 0 else AngleArray
        inst = super().__new__(target)
        AngleBase.__init__(inst, arr, unit)
        return inst

    def __init__(self, value, unit=AngleUnit.RAD):
        super().__init__(value, unit)


    @classmethod
    def parse(cls, v: Any) -> Self:
        # 0) is already Angle
        if isinstance(v, AngleBase):
            return v
        # 1) (value, unit) tuple
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], AngleUnit):
            return cls(v[0], v[1])

        # 2) single string "45deg", "0.5 rad", etc.
        if isinstance(v, str):
            m = re.match(r"^\s*([+-]?[0-9]*\.?[0-9]+)\s*(°|deg|rad|gon)\s*$", v, re.I)
            if m:
                val, suf = float(m.group(1)), m.group(2).lower()
                unit = {
                    "°":   AngleUnit.DEGREE,
                    "deg": AngleUnit.DEGREE,
                    "rad": AngleUnit.RAD,
                    "gon": AngleUnit.GON,
                }[suf]
                return cls(val, unit)
            else:
                raise ValueError(f"Cannot parse angle from {v!r}")

        # 3) numpy scalar
        if isinstance(v, np.generic):
            return cls(float(v), cls._INTERNAL_UNIT)

        # 4) bare Python scalar
        if isinstance(v, (int, float)):
            return cls(v, cls._INTERNAL_UNIT)

        # 5) numpy array
        if isinstance(v, np.ndarray):
            return cls(v, cls._INTERNAL_UNIT)

        # 6) any other Sequence (list or tuple) but *not* a string
        if isinstance(v, Sequence):
            # empty sequence → treat as empty array
            if len(v) == 0:
                return cls(np.array([], dtype=float), cls._INTERNAL_UNIT)

            # all‐numbers → straightforward array
            if all(isinstance(x, (int, float, np.generic)) for x in v):
                return cls(np.array(v, dtype=float), cls._INTERNAL_UNIT)

            # all‐strings → parse each element and collect radians
            if all(isinstance(x, str) for x in v):
                parsed = [Angle.parse(x).internal_value for x in v]
                return cls(np.array(parsed, dtype=float), cls._INTERNAL_UNIT)

            # mixed or unsupported → error out
            raise ValueError(f"Cannot parse angle sequence: {v!r}")

        # 7) give up
        raise ValueError(f"Cannot parse angle from {v!r}")

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
    """Represents an N‑D array of angles."""

    def __init__(self, value, unit=AngleUnit.RAD):
        pass

    def __new__(cls, arr: ArrayT, unit: AngleUnit = AngleUnit.RAD):
        arr = np.array(arr, dtype=float)
        inst = super().__new__(cls)
        AngleBase.__init__(inst, arr, unit)
        return inst


    @property
    def shape(self) -> tuple[int, ...]:
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
        return (f"{self.__class__.__name__}(shape={self._internal_value.shape}, "
                f"unit={self._display_unit.name}, values={preview})")

    def __str__(self):
        vals = self.to(self._display_unit)
        preview = np.array2string(vals, threshold=4)
        return f"{preview} {self.display_unit.name}"


    def _compare(self, other, op):
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
