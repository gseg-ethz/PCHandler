import copy
import gc
import logging
from weakref import ReferenceType

import numpy as np
import pytest

from pchandler import PointCloudData
from pchandler.scalar_fields import ScalarFieldManager
from pchandler.scalar_fields.scalar_fields import (
    AbstractScalarField,
    NormalFields,
    NormalisedInt16ScalarField,
    RGBFields,
    ScalarField,
)

N = 40


@pytest.fixture(scope="function", autouse=True)
def rgb_field() -> RGBFields:
    return RGBFields(np.random.randint(0, 255, (N, 3), dtype=np.uint8))


@pytest.fixture(scope="function", autouse=True)
def normals_field() -> NormalFields:
    return NormalFields(np.random.rand(N, 3).astype(np.float32))


@pytest.fixture(scope="function", autouse=True)
def intensity_field() -> ScalarField:
    return ScalarField(np.random.rand(N), name="intensity")


@pytest.fixture(scope="function", autouse=True)
def reflectance_field() -> ScalarField:
    return ScalarField(np.random.rand(N), name="reflectance")


@pytest.fixture(scope="function", autouse=True)
def scalar_field() -> ScalarField:
    return ScalarField(np.random.rand(N), name="test")


@pytest.fixture(scope="function", autouse=True)
def invalid_size() -> ScalarField:
    return ScalarField(np.random.rand(N + 10), name="test")


@pytest.fixture(scope="function", autouse=True)
def pcd() -> PointCloudData:
    return PointCloudData(np.random.rand(N, 3))


@pytest.fixture(scope="function", autouse=True)
def empty_sfm(pcd) -> ScalarFieldManager:
    return ScalarFieldManager(parent=pcd, fields={})


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
        },
    )


@pytest.fixture(scope="function", autouse=True)
def no_parent_sfm(rgb_field, normals_field, scalar_field, intensity_field, reflectance_field) -> ScalarFieldManager:
    return ScalarFieldManager(
        fields={
            intensity_field.name: intensity_field,
            rgb_field.name: rgb_field,
            normals_field.name: normals_field,
            reflectance_field.name: reflectance_field,
            scalar_field.name: scalar_field,
        },
    )


class TestSfmInitialisation:

    def test_init_no_kwargs(self):
        sfm = ScalarFieldManager()
        assert isinstance(sfm, ScalarFieldManager)
        assert len(sfm.fields) == 0
        assert sfm._parent is None

    def test_empty_w_parent_kwarg(self, pcd):
        sfm = ScalarFieldManager(parent=pcd)

        assert isinstance(sfm, ScalarFieldManager)
        assert len(sfm.fields) == 0
        assert isinstance(sfm._parent, ReferenceType)
        assert sfm._parent() is pcd

    def test_positional_fields(self, scalar_field, rgb_field, normals_field):
        sfs = {sf.name: sf for sf in (scalar_field, rgb_field, normals_field)}
        sfm = ScalarFieldManager(sfs)

        for name in ("rgb", "normals", "test"):
            assert name in sfm

        assert np.allclose(sfm.normals, normals_field)
        assert np.allclose(sfm.rgb, rgb_field)
        assert np.allclose(sfm["test"], scalar_field)

    def test_fields_without_parent_and_positional(self, scalar_field, rgb_field, normals_field):
        sfs = {sf.name: sf for sf in (scalar_field, rgb_field, normals_field)}

        for sfm in (ScalarFieldManager(fields=sfs), ScalarFieldManager(sfs)):
            for name in ("rgb", "normals", "test"):
                assert name in sfm

            assert np.allclose(sfm.normals, normals_field)
            assert np.allclose(sfm.rgb, rgb_field)
            assert np.allclose(sfm["test"], scalar_field)

    def test_fields_with_parent(self, scalar_field, rgb_field, normals_field, pcd):
        sfs = {sf.name: sf for sf in (scalar_field, rgb_field, normals_field)}
        sfm = ScalarFieldManager(parent=pcd, fields=sfs)

        for name in ("rgb", "normals", "test"):
            assert name in sfm

        assert len(sfm) == 3
        assert isinstance(sfm._parent, ReferenceType)
        assert np.allclose(sfm.normals, normals_field)
        assert np.allclose(sfm.rgb, rgb_field)
        assert np.allclose(sfm["test"], scalar_field)

    def test_fields_with_self(self, scalar_field, rgb_field, normals_field):
        sfs = {sf.name: sf for sf in (scalar_field, rgb_field, normals_field)}
        sfm = ScalarFieldManager(parent=None, fields=sfs)

        sfm2 = ScalarFieldManager(parent=None, fields=sfm)
        assert np.allclose(sfm.normals, sfm2.normals)
        assert np.allclose(sfm.rgb, sfm2.rgb)
        assert np.allclose(sfm["test"], sfm2["test"])
        assert len(sfm) == len(sfm2)
        assert id(sfm) != id(sfm2)

        # DECIDE - should the fields be shared? Test below passes
        assert id(sfm.rgb) == id(sfm2.rgb)
        assert id(sfm.normals) == id(sfm2.normals)
        assert id(sfm["test"]) == id(sfm2["test"])

    def test_fields_with_ndarray(self):
        rand_vals = np.random.rand(100)
        sfs = {"nums": rand_vals}
        sfm = ScalarFieldManager(fields=sfs)

        assert np.allclose(sfm["nums"], rand_vals)
        assert not isinstance(sfm["nums"], np.ndarray)
        assert isinstance(sfm["nums"], ScalarField)

        # DECIDE - Same as above, data is shared
        assert sfm["nums"].arr is rand_vals

    @pytest.mark.parametrize(
        "parent, fields",
        (
            (None, 3),
            (None, "fail"),
            (None, True),
            (None, ("a", 2)),
            (3, {}),
            (True, {}),
            ("asdasd", {}),
            (None, {"abc": True}),
            (None, np.random.rand(100)),
        ),
    )
    def test_invalid(self, parent, fields):
        with pytest.raises(Exception) as e:
            ScalarFieldManager(parent=parent, fields=fields)

        assert type(e.value) in (TypeError, ValueError)


class TestSfmValidators:
    def test_validate_lengths(self, pcd, base_sfm, no_parent_sfm, invalid_size, caplog):
        # General case where the parent object is set
        caplog.set_level(logging.INFO)
        assert base_sfm.validate_lengths() is None

        # Case with no parents
        no_parent_sfm.validate_lengths()
        assert "No parent point cloud to validate scalar field lengths against" in caplog.text

        # Case throws error due to mismatch field lengths
        invalid_fields = {"rgb": base_sfm.rgb, "invalid": invalid_size}
        sfm = ScalarFieldManager(fields=invalid_fields)

        # Throws error when assigned and lengths don'e match
        with pytest.raises(ValueError):
            sfm.parent = pcd


class TestSfmDunderMethods:
    def test_len(self, base_sfm):
        assert len(base_sfm) == 5

    def test_iter(self, base_sfm):
        for name, value in base_sfm.items():
            assert name in ("intensity", "rgb", "normals", "reflectance", "test")
            assert isinstance(value, AbstractScalarField)
            assert value.shape[0] == N

    def test_contains(self, base_sfm):
        assert "intensity" in base_sfm
        assert "test" in base_sfm
        assert "rgb" in base_sfm
        assert "normals" in base_sfm
        assert "reflectance" in base_sfm
        assert "REFLECTANCE" in base_sfm

    def test_getitem_str(self, base_sfm, rgb_field, normals_field, scalar_field, intensity_field, reflectance_field):
        for sf in (rgb_field, normals_field, scalar_field, intensity_field, reflectance_field):
            assert np.all(base_sfm[sf.name] == sf)

    def test_getitem_index(self, base_sfm, rgb_field, normals_field, scalar_field, intensity_field, reflectance_field):
        sampled = base_sfm[0:10]

        for sf in (rgb_field, normals_field, scalar_field, intensity_field, reflectance_field):
            assert np.all(sampled[sf.name] == sf[0:10])

    def test_setitem(self, base_sfm, invalid_size, scalar_field):
        base_sfm["intensity"] = scalar_field
        assert np.allclose(base_sfm.intensity, scalar_field)

        # Set a new scalar field
        assert "custom1" not in base_sfm
        len_before = len(base_sfm)

        base_sfm["custom1"] = scalar_field
        assert "custom1" in base_sfm
        assert len(base_sfm) == len_before + 1

        # Clear field by setting it to None
        base_sfm["custom1"] = None
        # DECIDE should this be a keyerror or return None
        with pytest.raises(KeyError):
            assert base_sfm.fields["custom1"] is None
        assert len(base_sfm) == len_before

        # Sequence is used for the values
        list_obj = np.ones(N).tolist()
        base_sfm["seq1"] = list_obj
        assert np.all(base_sfm["seq1"] == 1)
        assert isinstance(base_sfm["seq1"], ScalarField)

        # Invalid length sf
        with pytest.raises(ValueError):
            base_sfm["invalid"] = np.ones(14)

    def test_delitem(self, base_sfm):
        assert "intensity" in base_sfm
        assert base_sfm.intensity is not None
        assert len(base_sfm) == 5

        del base_sfm["intensity"]

        assert "intensity" not in base_sfm
        assert base_sfm.intensity is None
        assert len(base_sfm) == 4

        del base_sfm["rgb"]

        assert "rgb" not in base_sfm
        assert base_sfm.rgb is None
        assert len(base_sfm) == 3


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
            assert isinstance(value, AbstractScalarField)
            assert value.shape[0] == N

    def test_items(self, base_sfm):
        for i, item in enumerate(base_sfm.items()):
            k, v = item
            assert k == list(base_sfm.keys())[i]
            assert np.all(v == list(base_sfm.values())[i])


class TestGeneralProperties:
    def test_parent_getter(self, base_sfm, no_parent_sfm):
        assert base_sfm.parent is not None
        assert no_parent_sfm.parent is None

    def test_parent_setter(self, pcd, base_sfm):
        assert base_sfm.parent is pcd
        assert id(base_sfm.parent) == id(pcd)

    def test_shape(self, base_sfm):
        assert base_sfm.shape == (N, len(base_sfm))

    def test_num_points(self, empty_sfm, base_sfm, no_parent_sfm):
        assert no_parent_sfm.num_points == -1
        this_pcd = PointCloudData(np.random.rand(100, 3), scalar_field={"abc": np.random.rand(100)})
        sfm = copy.deepcopy(this_pcd.scalar_fields)
        del this_pcd
        gc.collect()
        assert sfm.num_points == -1


class TestSpecialNamedProperties:
    def test_rgb_getter(self, base_sfm):
        assert hasattr(base_sfm, "rgb")
        assert np.all(base_sfm.rgb == base_sfm["rgb"])
        assert base_sfm.rgb.dtype == np.uint8

    def test_rgb_setter(self, empty_sfm):
        array = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
        empty_sfm.rgb = array
        assert hasattr(empty_sfm, "rgb")
        assert np.all(empty_sfm.rgb == array)
        assert empty_sfm.rgb.dtype == np.uint8

    def test_normals_getter(self, base_sfm):
        assert hasattr(base_sfm, "normals")
        assert np.all(base_sfm.normals == base_sfm["normals"])
        assert base_sfm.normals.dtype == np.float32

    def test_normals_setter(self, empty_sfm):
        array = np.random.rand(N, 3).astype(np.float32)
        array = array / np.linalg.norm(array, axis=1).reshape(-1, 1)
        empty_sfm.normals = array
        assert hasattr(empty_sfm, "normals")
        assert np.allclose(empty_sfm.normals, array)
        assert empty_sfm.normals.dtype == np.float32

    def test_intensity_getter(self, base_sfm):
        assert hasattr(base_sfm, "intensity")
        assert np.all(base_sfm.intensity == base_sfm["intensity"])
        assert base_sfm.intensity.dtype == np.float64

    def test_intensity_setter(self, empty_sfm):
        array = np.random.rand(N)
        empty_sfm.intensity = array
        assert hasattr(empty_sfm, "intensity")
        assert np.allclose(empty_sfm.intensity, array)
        assert empty_sfm.intensity.dtype == array.dtype

    def test_reflectance_getter(self, base_sfm):
        assert hasattr(base_sfm, "reflectance")
        assert np.all(base_sfm.reflectance == base_sfm["reflectance"])
        assert base_sfm.reflectance.dtype == np.float64

    def test_reflectance_setter(self, empty_sfm):
        array = np.random.rand(N)
        empty_sfm.reflectance = array

        assert hasattr(empty_sfm, "reflectance")
        assert np.allclose(empty_sfm.reflectance, array)
        assert empty_sfm.reflectance.dtype == array.dtype


class TestSampleExtractReduce:
    def test_sample(self, base_sfm):
        index = slice(0, 10, 1)
        sample = base_sfm.sample(index)

        assert len(sample.rgb) == 10
        assert len(sample) == len(base_sfm)
        assert np.all(sample.rgb == base_sfm.rgb[index])
        assert np.allclose(sample.normals, base_sfm.normals[index])

    def test_reduce(self, base_sfm):
        index = slice(0, 10, 1)
        original = copy.deepcopy(base_sfm)

        base_sfm.reduce(index)

        assert len(base_sfm.rgb) == 10
        assert len(base_sfm) == 5

        sample = original.sample(index)

        assert np.all(base_sfm.rgb == sample.rgb)
        assert np.all(base_sfm.normals == sample.normals)
        assert np.all(base_sfm.intensity == sample.intensity)

        assert id(base_sfm.rgb) != id(sample.rgb)
        assert id(base_sfm.normals) != id(sample.normals)
        assert id(base_sfm.intensity) != id(sample.intensity)

    def test_extract(self, base_sfm, empty_sfm, no_parent_sfm):
        index = slice(0, 10, 1)
        original = copy.deepcopy(base_sfm)
        extracted = base_sfm.extract(index)
        sample = original.sample(index)

        assert len(sample) == len(extracted)
        assert len(sample) == len(original)
        assert len(base_sfm.rgb) == len(original.rgb) - len(extracted.rgb)
        assert np.all(sample.rgb == extracted.rgb)

        # Returns self if it's empty
        emptied = empty_sfm.extract(index)
        assert len(emptied) == 0
        assert id(emptied) != id(empty_sfm)

        # use the create_mask method from sf if the parent is not defined
        no_parent_copy = copy.deepcopy(no_parent_sfm)
        extracted = no_parent_copy.extract(index)
        sample = no_parent_sfm.sample(index)

        assert len(sample) == len(extracted)
        assert len(sample) == len(no_parent_copy)
        assert len(no_parent_copy.rgb) == len(no_parent_sfm.rgb) - len(extracted.rgb)
        assert np.all(sample.rgb == extracted.rgb)


class TestExtraFieldMethods:
    def test_add_field(self, base_sfm, scalar_field):
        data = scalar_field.copy(deep=True)
        data.name = "NewField"
        base_sfm.add_field(data)
        assert "newfield" in base_sfm
        assert np.all(base_sfm["newfield"] == data)

    def test_create_field(self, base_sfm):
        data = np.random.rand(N)
        base_sfm.create_field("new_field", data)

        assert "new_field" in base_sfm
        assert np.all(base_sfm["new_field"] == data)
        assert isinstance(base_sfm["new_field"], ScalarField)

    def test_remove_field(self, base_sfm):
        assert "rgb" in base_sfm
        base_sfm.remove_field("rgb")
        assert "rgb" not in base_sfm
        assert base_sfm.rgb is None


class TestNamedFieldHandlers:
    def test_rgb_handlers(self, base_sfm, empty_sfm):
        r = ScalarField(np.random.randint(0, 255, N, dtype=np.uint8), name="r")
        red = ScalarField(np.random.randint(0, 255, N, dtype=np.uint8), name="red")
        green = ScalarField(np.random.randint(0, 255, N, dtype=np.uint8), name="green")
        g = ScalarField(np.random.randint(0, 255, N, dtype=np.uint8), name="g")
        blue = ScalarField(np.random.randint(0, 255, N, dtype=np.uint8), name="blue")
        b = ScalarField(np.random.randint(0, 255, N, dtype=np.uint8), name="b")
        color = RGBFields(np.random.randint(0, 255, (N, 3), dtype=np.uint8), name="color")
        colour = RGBFields(np.random.randint(0, 255, (N, 3), dtype=np.uint8), name="colour")
        colors = RGBFields(np.random.randint(0, 255, (N, 3), dtype=np.uint8), name="colors")
        colours = RGBFields(np.random.randint(0, 255, (N, 3), dtype=np.uint8), name="colours")
        float_vec_target = np.linspace(0, 255, N, endpoint=True).astype(np.float32)
        rf = ScalarField(float_vec_target / 255, name="rf")
        gf = ScalarField(float_vec_target / 255, name="gf")
        bf = ScalarField(float_vec_target / 255, name="bf")
        order = ("r", "g", "b")

        base_sfm.rgb = None
        for i, sf in enumerate((r, g, b)):
            base_sfm.add_field(sf)
            assert hasattr(base_sfm, "rgb")
            assert not hasattr(base_sfm, sf.name)
            assert sf.name not in base_sfm
            assert np.all(getattr(base_sfm.rgb, order[i]) == sf)

        base_sfm.rgb = None
        for i, sf in enumerate((red, green, blue)):
            base_sfm.add_field(sf)
            assert not hasattr(base_sfm, sf.name)
            assert sf.name not in base_sfm
            assert np.all(getattr(base_sfm.rgb, order[i]) == sf)

        base_sfm.rgb = None
        for i, sf in enumerate((rf, gf, bf)):
            base_sfm.add_field(sf)
            assert not hasattr(base_sfm, sf.name)
            assert sf.name not in base_sfm
            assert np.allclose(getattr(base_sfm.rgb, order[i]), np.ceil(sf * 255), atol=1)

        for sf in (color, colors, colour, colours):
            base_sfm.add_field(sf)
            assert np.all(sf == base_sfm.rgb)

        # Test reversed
        base_sfm["bgr"] = colour
        assert np.all(base_sfm.rgb == colour[:, [2, 1, 0]])
        assert np.all(base_sfm["bgr"] == colour)

        # Test access of scalar on empty
        assert empty_sfm["r"] is None

        with pytest.raises(KeyError):
            base_sfm._get_rgb("invalid")
        with pytest.raises(KeyError):
            base_sfm._set_rgb("invalid", colour)

    def test_normal_handlers(self, base_sfm, empty_sfm):
        nx = ScalarField(np.random.rand(N).astype(np.float32), name="nx")
        ny = ScalarField(np.random.rand(N).astype(np.float32), name="ny")
        nz = ScalarField(np.random.rand(N).astype(np.float32), name="nz")
        nxnynz = NormalFields(np.random.rand(N, 3).astype(np.float32), name="nxnynz")
        normal = NormalFields(np.random.rand(N, 3).astype(np.float32), name="normal")

        order = ("nx", "ny", "nz")

        base_sfm.normals = None
        for i, sf in enumerate((nx, ny, nz)):
            base_sfm.add_field(sf)
            assert hasattr(base_sfm, "normals")
            assert sf.name not in base_sfm
            assert np.all(getattr(base_sfm.normals, order[i]) == sf)

        base_sfm.normals = None
        for sf in (nxnynz, normal):
            base_sfm.add_field(sf)
            assert hasattr(base_sfm, "normals")
            assert np.allclose(sf, base_sfm.normals)

        # Test reversed
        base_sfm["nznynx"] = normal
        assert np.all(base_sfm.normals == normal.arr[:, [2, 1, 0]])
        assert np.all(base_sfm["nznynx"] == normal)

        # Test access of scalar on empty
        assert empty_sfm["nx"] is None

        with pytest.raises(KeyError):
            base_sfm._get_normals("invalid")
        with pytest.raises(KeyError):
            base_sfm._set_normals("invalid", normal)


class TestClassMethods:
    def test_merge_valid(self, intensity_field, rgb_field, reflectance_field, normals_field, scalar_field):
        cloud1 = ScalarFieldManager(
            parent=None, fields={"intensity": intensity_field, "rgb": rgb_field, "reflectance": reflectance_field}
        )

        cloud2 = ScalarFieldManager(
            parent=None,
            fields={"rgb": rgb_field, "reflectance": reflectance_field, "normals": normals_field, "test": scalar_field},
        )

        merged = ScalarFieldManager.merge([cloud1, cloud2])

        assert "rgb" in merged
        assert "reflectance" in merged
        assert "intensity" not in merged
        assert "normals" not in merged
        assert "test" not in merged

        assert len(merged.rgb) == (len(cloud1.rgb) + len(cloud2.rgb))
        assert len(merged) == 2

    def test_merge_invalid(self, intensity_field, rgb_field, reflectance_field, normals_field, scalar_field):
        with pytest.raises(ValueError):
            ScalarFieldManager.merge([])

        # TODO test for the other cases?
        # DECIDE do we want to handle different names with SF objects to the fields keys?
