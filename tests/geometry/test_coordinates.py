import pytest

from dataclasses import dataclass
import copy

import numpy as np

from pchandler.geometry.coordinates import CoordinateSet3D, GeneralCoordinates, cartesian2spherical, spherical2cartesian, cart2spher_vec, spher2cart_vec, CoordSysEnum
from pchandler.geometry.validation import check_spherical_coordinates
from tests.utils import numpy_in_dict_equality_check


@dataclass(frozen=True)
class Example:
    xyz: np.ndarray

PI = np.pi
TWO_PI = np.pi*2
HALF_PI = np.pi/2

known_cartesian: np.ndarray = np.array([[0, 0, 1],
                                                [1, 0, 0],
                                                [0, 1, 0],
                                                [-1, 0, 0],
                                                [0, -1, 0], ], dtype=np.float64)

expected_spher: np.ndarray = np.array([[1, 0, 0],
                                       [1, HALF_PI, 0],
                                       [1, HALF_PI, HALF_PI],
                                       [1, HALF_PI, PI],
                                       [1, HALF_PI, -HALF_PI], ], dtype=np.float64)

class TestConversions:
    def test_cartesian2spherical_array(self):

        spher = cartesian2spherical(known_cartesian)
        cart = spherical2cartesian(expected_spher)
        np.testing.assert_array_equal(spher, expected_spher)
        np.testing.assert_array_almost_equal(cart, known_cartesian)

        xyz = np.random.rand(100,3)*100
        temp_spher = cartesian2spherical(xyz)

        assert check_spherical_coordinates(temp_spher) is None
        xyz2 = spherical2cartesian(temp_spher)
        np.testing.assert_array_almost_equal(xyz2, xyz)

    @pytest.mark.parametrize("xyz, expected",
                             ([(known_cartesian[i, :], expected_spher[i,:]) for i in range(known_cartesian.shape[0])]))
    def test_spherical2cartesian_vector(self, xyz, expected):
        r, v, hz = cart2spher_vec(xyz[0], xyz[1], xyz[2])
        assert r == expected[0]
        assert v == expected[1]
        assert hz == expected[2]
        x, y, z = spher2cart_vec(r, v, hz)
        assert x == xyz[0]
        assert y == xyz[1]
        assert z == xyz[2]


class TestGeneralisedCoordinates:
    arr: CoordinateSet3D = CoordinateSet3D(np.random.randn(100, 3))

    def test_num_pts(self):
        assert len(self.arr) == 100
        assert self.arr.num_points == 100

    def test_dataclass_immutability(self):
        a = np.random.randn(100,3)
        example_dataclass = Example(a.copy())
        example_coordinates = GeneralCoordinates(a.copy(), immutable=True)
        print(f"{example_coordinates.immutable=}")

        old_val = example_dataclass.xyz[0, 0]
        example_dataclass.xyz[0, 0] = 25

        assert example_dataclass.xyz[0, 0] != old_val   # Value has changed on an immutable dataclass
        with pytest.raises(AttributeError):
            # noinspection PyDataclass
            example_dataclass.xyz = np.random.rand(100,3)

        with pytest.raises(ValueError):
            example_coordinates.xyz[0, 0] = 25

    def test_properties(self):
        a = np.random.randn(100,3)
        b = cartesian2spherical(a)
        coords = GeneralCoordinates(a.copy())

        assert coords.coord_system == CoordSysEnum.CART
        assert np.all(a == coords)
        assert np.all(a[:, 0] == coords.x)
        assert np.all(a[:, 1] == coords.y)
        assert np.all(a[:, 2] == coords.z)
        dict_cart_before = copy.deepcopy(coords.__dict__)

        assert np.all(coords == coords.xyz)
        # if xyz coords, cached_property should not be initialised
        assert numpy_in_dict_equality_check(dict_cart_before, coords.__dict__) == True

        assert np.all(a == coords.spher)
        assert np.all(b == coords.spher)
        assert np.all(b[:, 0] == coords.r)
        assert np.all(b[:, 1] == coords.v)
        assert np.all(b[:, 2] == coords.hz)
        assert np.all(b[:, 0] == coords.rho)
        assert np.all(b[:, 1] == coords.theta)
        assert np.all(b[:, 2] == coords.phi)

        assert np.any(coords.arr != a)

        assert numpy_in_dict_equality_check(dict_cart_before, coords.__dict__) == False

        assert "_spher" in coords.__dict__
        coords.to_spherical()
        assert np.all(b == coords.spher)
        assert np.all(b == coords.arr)
        dict_spher_before = copy.deepcopy(coords.__dict__)

        assert numpy_in_dict_equality_check(dict_cart_before, coords.__dict__) == True

        assert np.any(coords.arr != a)
        assert np.all(a == coords.xyz)

        assert dict_spher_before != coords.__dict__

        assert numpy_in_dict_equality_check(dict_cart_before, coords.__dict__) == False

        assert "_xyz" in coords.__dict__
        coords.invalidate_cache()
        assert "_xyz" not in coords.__dict__
        assert coords.coord_system == CoordSysEnum.SPHER


        c = GeneralCoordinates(a.copy(), coord_system=CoordSysEnum.SPHER)
        assert c.coord_system == CoordSysEnum.SPHER
        assert np.all


