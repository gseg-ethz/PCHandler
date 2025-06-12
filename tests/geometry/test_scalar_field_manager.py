import pytest
import numpy as np

from weakref import ReferenceType

from pchandler.geometry.scalar_fields import ScalarField, RGBFields, NormalFields
from pchandler.geometry.scalar_field_manager import ScalarFieldManager
from pchandler.geometry.core import PointCloudData

N = 40

@pytest.fixture(scope="function", autouse=True)
def rgb_field() -> RGBFields:
    return RGBFields(np.random.randint(0, 255, (N,3), dtype=np.uint8))

@pytest.fixture(scope="function", autouse=True)
def normals_field() -> NormalFields:
    return NormalFields(np.random.rand(N,3).astype(np.float32))

@pytest.fixture(scope="function", autouse=True)
def scalar_field() -> ScalarField:
    return ScalarField(np.random.rand(N), name='test')

@pytest.fixture(scope="function", autouse=True)
def invalid_size() -> ScalarField:
    return ScalarField(np.random.rand(N+10), name='test')

@pytest.fixture(scope="function", autouse=True)
def intensity_field() -> ScalarField:
    return ScalarField(np.random.rand(N), name='intensity')

@pytest.fixture(scope="function", autouse=True)
def reflectance_field() -> ScalarField:
    return ScalarField(np.random.rand(N), name='reflectance')

@pytest.fixture(scope="function", autouse=True)
def pcd() -> PointCloudData:
    return PointCloudData(np.random.rand(N,3))

@pytest.fixture(scope="function", autouse=True)
def empty_sfm(pcd) -> ScalarFieldManager:
    return ScalarFieldManager(
        parent=pcd,
        fields={}
    )
@pytest.fixture(scope="function", autouse=True)
def base_sfm(rgb_field, normals_field, scalar_field, intensity_field, reflectance_field, pcd) -> ScalarFieldManager:
    return ScalarFieldManager(
        parent=pcd,
        fields={
            intensity_field.name: intensity_field,
            rgb_field.name: rgb_field,
            normals_field.name: normals_field,
            reflectance_field.name: reflectance_field,
            scalar_field.name: scalar_field,
        }
    )

class TestSfmInitialisation:

    def test_empty(self):
        sfm = ScalarFieldManager()
        assert isinstance(sfm, ScalarFieldManager)
        assert len(sfm.fields) == 0
        assert sfm._parent is None

    def test_empty_w_parent(self, pcd):
        sfm = ScalarFieldManager(parent=pcd)

        assert isinstance(sfm, ScalarFieldManager)
        assert len(sfm.fields) == 0
        assert isinstance(sfm._parent, ReferenceType)
        assert sfm._parent() is pcd

    def test_fields_without_parent(self, scalar_field, rgb_field, normals_field):

        sfs = {sf.name: sf for sf in (scalar_field, rgb_field, normals_field)}
        sfm = ScalarFieldManager(fields=sfs)

        for name in ('rgb', 'normals', 'test'):
            assert name in sfm

        assert np.allclose(sfm.normals, normals_field)
        assert np.allclose(sfm.rgb, rgb_field)
        assert np.allclose(sfm['test'], scalar_field)

    def test_fields_with_parent(self, scalar_field, rgb_field, normals_field, pcd):
        sfs = {sf.name: sf for sf in (scalar_field, rgb_field, normals_field)}
        sfm = ScalarFieldManager(parent=pcd, fields=sfs)

        for name in ('rgb', 'normals', 'test'):
            assert name in sfm

        assert len(sfm) == 3
        assert isinstance(sfm._parent, ReferenceType)
        assert np.allclose(sfm.normals, normals_field)
        assert np.allclose(sfm.rgb, rgb_field)
        assert np.allclose(sfm['test'], scalar_field)

    def test_fields_with_self(self, scalar_field, rgb_field, normals_field):
        sfs = {sf.name: sf for sf in (scalar_field, rgb_field, normals_field)}
        sfm = ScalarFieldManager(parent=None, fields=sfs)

        sfm2 = ScalarFieldManager(parent=None, fields = sfm)
        assert np.allclose(sfm.normals, sfm2.normals)
        assert np.allclose(sfm.rgb, sfm2.rgb)
        assert np.allclose(sfm['test'], sfm2['test'])
        assert len(sfm) == len(sfm2)

    def test_fields_with_ndarray(self):
        sfs = {'nums': np.random.rand(100)}
        sfm = ScalarFieldManager(fields=sfs)
        assert np.allclose(sfm['nums'], sfs['nums'])

    @pytest.mark.parametrize('parent, fields', (
            (None, 3),
            (None, 'fail'),
            (None, True),
            (None, ('a', 2)),
            (3, {}),
            (True, {}),
            ('asdasd', {}),
            (None, {'abc': True}),
            (None, np.random.rand(100))))
    def test_invalid(self, parent, fields):
        with pytest.raises(Exception) as e:
            ScalarFieldManager(parent=parent, fields=fields)

        assert type(e.value) in (TypeError, ValueError)


class TestSfmDunderMethods:
    def test_iter(self, base_sfm):
        for name, value in base_sfm.items():
            assert name in ('intensity', 'rgb', 'normals', 'reflectance', 'test')
            assert isinstance(value, ScalarField)
            assert value.shape[0] == N


    def test_contains(self, base_sfm):
        assert 'intensity' in base_sfm
        assert 'test' in base_sfm
        assert 'rgb' in base_sfm
        assert 'normals' in base_sfm
        assert 'reflectance' in base_sfm
        assert 'REFLECTANCE' in base_sfm

    def test_len(self, base_sfm):
        assert len(base_sfm) == 5

    def test_getitem_str(self, base_sfm, rgb_field, normals_field, scalar_field, intensity_field, reflectance_field):
        for sf in (rgb_field, normals_field, scalar_field, intensity_field, reflectance_field):
            assert np.all(base_sfm[sf.name] == sf)

    def test_getitem_index(self, base_sfm, rgb_field, normals_field, scalar_field, intensity_field, reflectance_field):
        sampled = base_sfm[0:10]

        for sf in (rgb_field, normals_field, scalar_field, intensity_field, reflectance_field):
            assert np.all(sampled[sf.name] == sf[0:10])

    def test_setitem(self, base_sfm, invalid_size, scalar_field):
        original = base_sfm['intensity'].copy()
        base_sfm['intensity'] = scalar_field
        assert np.all(base_sfm.intensity == scalar_field)
        assert not np.allclose(base_sfm.intensity, original)

    def test_delitem(self, base_sfm):
        assert 'intensity' in base_sfm
        assert base_sfm.intensity is not None
        assert len(base_sfm) == 5

        del base_sfm['intensity']

        assert 'intensity' not in base_sfm
        assert base_sfm.intensity is None
        assert len(base_sfm) == 4

        del base_sfm['rgb']

        assert 'rgb' not in base_sfm
        assert base_sfm.rgb is None
        assert len(base_sfm) == 3

    def test_or_symbol_merge(self):
        raise NotImplementedError('')

class TestMutableMappingMethods:
    def test_keys(self, base_sfm):
        for key in base_sfm.keys():
            assert isinstance(key, str)
            assert key in base_sfm
            assert base_sfm.fields[key] is not None

        assert list(base_sfm.keys()) == list(base_sfm.fields.keys())

    def test_values(self, base_sfm):
        assert len(base_sfm.values()) == 5
        for value in base_sfm.values():
            assert isinstance(value, ScalarField)
            assert value.shape[0] == N

    def test_items(self, base_sfm):
        for i, item in enumerate(base_sfm.items()):
            k, v = item
            assert k == base_sfm.keys()[i]
            assert np.all(v == list(base_sfm.values())[i])

class TestNamedFieldPropertyGetters:
    def test_rgb_getter(self, base_sfm):
        assert hasattr(base_sfm, 'rgb')
        assert np.all(base_sfm.rgb == base_sfm['rgb'])
        assert base_sfm.rgb.dtype == np.uint8

    def test_normals_getter(self, base_sfm):
        assert hasattr(base_sfm, 'normals')
        assert np.all(base_sfm.normals == base_sfm['normals'])
        assert base_sfm.normals.dtype == np.float32

    def test_intensity_getter(self, base_sfm):
        assert hasattr(base_sfm, 'intensity')
        assert np.all(base_sfm.intensity == base_sfm['intensity'])
        assert base_sfm.intensity.dtype == np.float64

    def test_reflectance_getter(self, base_sfm):
        assert hasattr(base_sfm, 'reflectance')
        assert np.all(base_sfm.reflectance == base_sfm['reflectance'])
        assert base_sfm.reflectance.dtype == np.float64

    def test_rgb_setter(self, empty_sfm):
        array = np.random.randint(0, 255, (N,3), dtype=np.uint8)
        empty_sfm.rgb = array
        assert hasattr(empty_sfm, 'rgb')
        assert np.all(empty_sfm.rgb == array)
        assert empty_sfm.rgb.dtype == np.uint8

    def test_normals_setter(self, empty_sfm):
        array = np.random.rand(N,3).astype(np.float32)
        empty_sfm.normals = array
        assert hasattr(empty_sfm, 'normals')
        assert np.all(empty_sfm.normals == array)
        assert empty_sfm.normals.dtype == np.float32

    def test_intensity_setter(self, empty_sfm):
        array = np.random.rand(N)
        empty_sfm.intensity = array
        assert hasattr(empty_sfm, 'intensity')
        assert np.all(empty_sfm.intensity == array)
        assert empty_sfm.intensity.dtype == np.float64

    def test_reflectance_setter(self, empty_sfm):
        array = np.random.rand(N)
        empty_sfm.reflectance = array

        assert hasattr(empty_sfm, 'reflectance')
        assert np.all(empty_sfm.reflectance == array)
        assert empty_sfm.reflectance.dtype == np.float64


class TestShapeHelpers:
    # DISCUSS should the RGB equate to multiple fields and be reflected in the shape?
    def test_shape(self, base_sfm):
        assert base_sfm.shape == (N, len(base_sfm))

    def test_num_points(self, base_sfm):
        assert base_sfm.num_points == N

class TestAdditionalFieldMethods:
    def test_add_field(self, base_sfm, scalar_field):
        data = scalar_field.copy(deep=True)
        data.name = 'NewField'
        base_sfm.add_field(data)
        assert 'newfield' in base_sfm
        assert np.all(base_sfm['newfield'] == data)

    def test_create_field(self, base_sfm):
        data = np.random.rand(N)
        base_sfm.create_field('new_field', data)

        assert 'new_field' in base_sfm
        assert np.all(base_sfm['new_field'] == data)
        assert isinstance(base_sfm['new_field'], ScalarField)

    def test_remove_field(self, base_sfm):
        assert 'rgb' in base_sfm
        base_sfm.remove_field('rgb')
        assert 'rgb' not in base_sfm
        assert base_sfm.rgb is None

class TestNamedFieldHandlers:
    def test_handle_rgb(self):
        raise NotImplementedError('')

    def test_handle_normals(self):
        raise NotImplementedError('')

class TestSampleExtractReduce:
    def test_sample(self, base_sfm):
        raise NotImplementedError('')

    def test_reduce(self):
        raise NotImplementedError('')

    def test_extract(self):
        raise NotImplementedError('')

class TestMerge:
    def test_generate_point_cloud_indexes(self):
        raise NotImplementedError('')

    def test_merge(self):
        raise NotImplementedError