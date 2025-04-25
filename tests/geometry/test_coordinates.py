
from pchandler.geometry.coordinates import CoordinateSet3D, CartesianCoordinates, SphericalCoordinates


class TestCoordinates3D:
    arr: CoordinateSet3D = CoordinateSet3D(np.random.randn(100, 3))

    def test_num_pts(self):
        assert len(self.arr) == 100
        assert self.arr.num_points == 100

    @pytest.mark.parametrize("attr", list(COORDINATE_3D_PROPERTIES))
    def test_not_implemented_properties(self, attr):
        with pytest.raises( NotImplementedError ):
            getattr(self.arr, attr)

    @pytest.mark.parametrize("attr", ['to_spherical', 'to_cartesian'])
    def test_not_implemented_methods(self, attr):
        with pytest.raises( NotImplementedError ):
            getattr(self.arr, attr)()

    @pytest.mark.parametrize("attr", ['from_spherical', 'from_cartesian'])
    def test_not_implemented_class_methods(self, attr):
        with pytest.raises( NotImplementedError ):
            getattr(self.arr, attr)(self.arr.arr)


class TestCartesianCoordinates:
    pass

class TestBasePointCloud:
    pass

class TestSphericalPointCloud:
    pass

class TestTlsPointCloud:
    pass

class TestMultiScanCloud:
    pass
