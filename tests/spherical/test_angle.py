import pickle

import pytest
import numpy as np

from pchandler.spherical.angle import Angle, AngleArray
from pchandler.util import AngleUnit

import math

class TestAngle:

    @pytest.mark.parametrize("value, unit, expected_rad", [
        (1.0, AngleUnit.RAD, 1.0),
        (180.0, AngleUnit.DEGREE, np.pi),
        (400.0, AngleUnit.GON, 400 * np.pi / 200.0),
    ])
    def test_scalar_initialization_and_internal(self, value, unit, expected_rad):
        a = Angle(value, unit)
        # internal_value always in radians
        assert np.isclose(a.internal_value, expected_rad)
        # float conversion
        assert float(a) == pytest.approx(expected_rad)
        # default display unit is the one passed in
        assert a.display_unit == unit

    def test_array_initialization_and_internal(self):
        arr = [0, 90, 180]
        a = Angle(arr, AngleUnit.DEGREE)
        # internal_value stored in radians
        expected = np.array([0, np.pi/2, np.pi])
        np.testing.assert_allclose(a.internal_value, expected, rtol=1e-6)
        # display_unit remains degree
        assert a.display_unit == AngleUnit.DEGREE
        # type is AngleArray
        assert isinstance(a, AngleArray)

    def test_parse_various_inputs(self):
        # tuple form
        a = Angle.parse((90, AngleUnit.DEGREE))
        assert a.display_unit == AngleUnit.DEGREE
        assert float(a) == pytest.approx(np.pi/2)
        # bare scalar
        b = Angle.parse(2.5)
        assert b.display_unit == AngleUnit.RAD
        assert float(b) == pytest.approx(2.5)
        # numpy scalar
        c = Angle.parse(np.float64(1.57))
        assert isinstance(c, Angle)
        assert c.display_unit == AngleUnit.RAD
        # list of floats
        d = Angle.parse([0, np.pi])
        assert isinstance(d, AngleArray)
        np.testing.assert_allclose(d.internal_value, [0, np.pi], rtol=1e-6)
        # list of strings
        e = Angle.parse(["0deg", "180deg"])
        assert isinstance(e, AngleArray)
        np.testing.assert_allclose(e.internal_value, [0, np.pi], rtol=1e-6)
        # invalid
        with pytest.raises(ValueError):
            Angle.parse(["bad", 1.0])

    @pytest.mark.parametrize("init_val, init_unit, target_unit, expected", [
        (np.pi/2, AngleUnit.RAD, AngleUnit.DEGREE, 90.0),
        (100.0, AngleUnit.GON, AngleUnit.DEGREE, 90.0),
        ([0, 90, 180], AngleUnit.DEGREE, AngleUnit.RAD, [0, np.pi/2, np.pi]),
    ])
    def test_to_conversion(self, init_val, init_unit, target_unit, expected):
        a = Angle(init_val, init_unit)
        out = a.to(target_unit)
        if isinstance(expected, list):
            np.testing.assert_allclose(out, expected, rtol=1e-6)
        else:
            assert out == pytest.approx(expected)

    def test_in_unit_and_display_properties(self):
        a = Angle(1.0, AngleUnit.RAD)
        b = a.in_degrees()
        assert isinstance(b, Angle)
        assert b.display_unit == AngleUnit.DEGREE
        # internal not changed
        assert float(a.internal_value) == float(b.internal_value)
        # changing display_unit in place
        a.display_unit = AngleUnit.GON
        assert a.display_unit == AngleUnit.GON

    def test_repr_scalar(self):
        a = Angle(np.pi, AngleUnit.RAD)
        rep = repr(a)
        assert "Angle(" in rep and "unit=RAD" in rep and "3.1416" in rep

    def test_numpy_protocol_and_ufunc(self):
        a = Angle([0, np.pi/2, np.pi], AngleUnit.RAD)
        arr = np.asarray(a)
        np.testing.assert_array_equal(arr, [0, np.pi/2, np.pi])
        # unary ufunc
        sin_a = np.sin(a)
        assert isinstance(sin_a, AngleArray)
        np.testing.assert_allclose(sin_a.to(AngleUnit.RAD), np.sin(arr), rtol=1e-6)
        # binary ufunc
        x = Angle(np.pi/4, AngleUnit.RAD)
        y = Angle(np.pi/6, AngleUnit.RAD)
        z = np.add(x, y)
        assert isinstance(z, Angle)
        assert float(z) == pytest.approx(np.pi/4 + np.pi/6)
        # tuple-returning ufunc
        q, r = np.divmod(x, y)
        assert all(isinstance(v, Angle) for v in (q, r))
        # recombine
        rec = q * y + r
        assert float(rec) == pytest.approx(float(x))


class TestAngleArrayDirect:

    def test_direct_anglearray_construction(self):
        arr = AngleArray([0, 180], AngleUnit.DEGREE)
        assert isinstance(arr, AngleArray)
        # internal_value in radians
        np.testing.assert_allclose(arr.internal_value, [0, np.pi], rtol=1e-6)
        # default display_unit is degree
        assert arr.display_unit == AngleUnit.DEGREE
        scalar = AngleArray(180, AngleUnit.DEGREE)
        assert isinstance(scalar, AngleArray)
        np.testing.assert_allclose(scalar.internal_value, [np.pi], rtol=1e-6)

    def test_anglearray_getitem_and_iter(self):
        arr = AngleArray([0, 90, 180], AngleUnit.DEGREE)
        # __len__
        assert len(arr) == 3
        # getitem scalar
        a0 = arr[0]
        assert isinstance(a0, Angle)
        assert float(a0) == pytest.approx(0.0)
        # getitem slice
        sub = arr[1:3]
        assert isinstance(sub, AngleArray)
        np.testing.assert_allclose(sub.internal_value, [np.pi/2, np.pi], rtol=1e-6)
        # iteration
        vals = [float(x) for x in arr]
        np.testing.assert_allclose(vals, [0, np.pi/2, np.pi], rtol=1e-6)

    def test_anglearray_repr(self):
        arr = AngleArray([0, 45, 90], AngleUnit.DEGREE)
        rep = repr(arr)
        assert "AngleArray(" in rep and "shape=(3,)" in rep and "unit=DEGREE" in rep

class TestAngleComparison:

    def test_scalar_eq_ne(self):
        a = Angle(90, AngleUnit.DEGREE)
        b = Angle(np.pi/2, AngleUnit.RAD)
        # equality and inequality between Angles
        assert a == b
        assert not (a != b)
        # equality and inequality against bare float
        assert a == 90
        assert math.isclose(b, np.pi/2)

    @pytest.mark.parametrize("lhs, rhs, lt_expected", [
        (Angle(90, AngleUnit.DEGREE), Angle(180, AngleUnit.DEGREE), True),        # 90° < 180°
        (Angle(np.pi/2, AngleUnit.RAD), np.pi, True),                             # π/2 < π
        (Angle(100, AngleUnit.GON), Angle(np.pi/2, AngleUnit.RAD), False),                      # 100 gon == π/2, so not < π/2
        (Angle(100, AngleUnit.GON), Angle(2*np.pi, AngleUnit.RAD), True),
        (Angle(200, AngleUnit.DEGREE), Angle(100, AngleUnit.DEGREE), False),      # 200° !< 100°
    ])
    def test_scalar_ordering(self, lhs, rhs, lt_expected):
        # < and <=
        assert (lhs < rhs) == lt_expected
        assert (lhs <= rhs) == lt_expected or (lhs == rhs)
        # > and >= (inverse relations)
        assert (rhs > lhs) == lt_expected
        assert (rhs >= lhs) == lt_expected or (lhs == rhs)

    # def test_array_comparisons(self):
    #     arr1 = Angle([0,  90, 180], AngleUnit.DEGREE)
    #     arr2 = Angle([45, 90, 135], AngleUnit.DEGREE)
    #     # element‑wise <
    #     lt = arr1 < arr2
    #     assert isinstance(lt, np.ndarray)
    #     np.testing.assert_array_equal(lt, [True, False, False])
    #     # element‑wise < against a Python sequence
    #     lt2 = arr1 < [45, 90, 200]
    #     np.testing.assert_array_equal(lt2, [True, False, True])
    #     # == on arrays collapses to a single bool
    #     assert (arr1 == arr2) is False
    #     # verify <= on arrays returns element‑wise
    #     le = arr1 <= arr2
    #     np.testing.assert_array_equal(le, [True, True, False])

    def test_hash_and_set_behavior(self):
        a = Angle(90, AngleUnit.DEGREE)
        b = Angle(np.pi/2, AngleUnit.RAD)
        # equivalent angles must have the same hash
        assert hash(a) == hash(b)
        # putting both in a set yields only one unique element
        s = {a, b}
        assert len(s) == 1


class TestAngleMath:

    @pytest.mark.parametrize("angle1_val, angle1_unit, angle2_val, angle2_unit", [
        (np.pi/2, AngleUnit.RAD, 180, AngleUnit.DEGREE),
        (100.0, AngleUnit.GON, 90.0, AngleUnit.DEGREE),
        ([0, 90, 180], AngleUnit.DEGREE, [0, np.pi / 2, np.pi], AngleUnit.RAD),
        ([0, 90, 180], AngleUnit.DEGREE, np.pi, AngleUnit.RAD),
        (np.pi, AngleUnit.DEGREE, [0, 90, 180], AngleUnit.RAD),
    ])
    def test_angle_angle_math(self, angle1_val, angle1_unit, angle2_val, angle2_unit):
        angle1 = Angle(angle1_val, angle1_unit)
        angle2 = Angle(angle2_val, angle2_unit)

        angle_add = angle1 + angle2
        assert isinstance(angle_add, Angle | AngleArray)
        assert angle_add.display_unit == angle1_unit
        assert np.allclose(Angle(angle1.internal_value+angle2.internal_value),
                           angle_add)

        angle_sub = angle1 - angle2
        assert isinstance(angle_add, Angle | AngleArray)
        assert angle_add.display_unit == angle1_unit
        assert np.allclose(Angle(angle1.internal_value-angle2.internal_value),
                           angle_sub)

class TestAnglePickle:

    @pytest.mark.parametrize("angle_val, angle_unit", [
        (np.pi / 2, AngleUnit.RAD),
        (180, AngleUnit.DEGREE),
        (100.0, AngleUnit.GON),
        (90.0, AngleUnit.DEGREE),
        ([0, 90, 180], AngleUnit.DEGREE),
        ([0, np.pi / 2, np.pi], AngleUnit.RAD),
    ])
    def test_angle_pickle(self, angle_val, angle_unit):
        a = Angle(angle_val, angle_unit)

        pickled_a = pickle.dumps(a)
        unpickled_a = pickle.loads(pickled_a)

        assert np.allclose(a, unpickled_a)
        assert unpickled_a.display_unit == angle_unit










