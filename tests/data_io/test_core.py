from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
import warnings

import pytest
import numpy as np

from pchandler.constants import RGB_NAMES, INTENSITY_NAMES, NORMAL_NAMES, REFLECTANCE_NAMES
from pchandler.scalar_fields.scalar_fields import RGBFields, NormalFields
from pchandler.geometry.core import PointCloudData
from pchandler.data_io.core import (
    find_point_cloud_in_directory,
    AbstractIOHandler,
    _get_rgb_or_normal_field_names,
    _get_sf_dtype,
    _clean_field_names,
    _clean_string,
    _clean_header_name
)

base_directory = Path(__file__).resolve().parent.parent
test_data_dir = base_directory / "data"

N = 100

@pytest.fixture(scope='function')
def valid_normals():
    vals = np.random.rand(N, 3).astype(np.float32)
    vals = vals / np.linalg.norm(vals, axis=1).reshape(-1, 1)
    return vals

@pytest.fixture(scope="function")
def struct_array():
    struct_dtype = np.dtype({
        'names': ['x', 'y', 'z', 'r', 'g', 'b', 'nx', 'ny', 'nz', 'intensity', 'sf1'],
        'formats': [np.float32, np.float32, np.float32, np.uint8, np.uint8, np.uint8, np.float32, np.float32,
                    np.float32, np.uint8, np.float64]
    })
    array = np.empty((N,), dtype=struct_dtype)
    return array

class TestFindPointCloudFunction:
    def test_directory_path_input(self):
        # Test the likely types of inputs from the user
        assert len(find_point_cloud_in_directory(test_data_dir, include_subdirectories=False)) == 0

        assert len(find_point_cloud_in_directory(test_data_dir / 'E57')) == 6
        assert len(find_point_cloud_in_directory(test_data_dir)) == 48

        # File passed instead of directory
        with pytest.raises(IOError):
            find_point_cloud_in_directory(test_data_dir / 'E57' / 'XYZ_Only.e57')

        # Non-existent directory
        with pytest.raises(IOError):
            find_point_cloud_in_directory(test_data_dir / 'Not_a_dir')

    def test_file_types(self):
        # Verify it searches and finds the right number of files of that format
        assert len(find_point_cloud_in_directory(test_data_dir, ['.e57'])) == 6
        assert len(find_point_cloud_in_directory(test_data_dir, ['.txt'])) == 8
        assert len(find_point_cloud_in_directory(test_data_dir / 'PLY_Binary', ['.ply'])) == 8
        assert len(find_point_cloud_in_directory(test_data_dir / 'PLY_ASCII', ['.ply'])) == 8
        assert len(find_point_cloud_in_directory(test_data_dir, ['.las'])) == 8
        assert len(find_point_cloud_in_directory(test_data_dir, ['.laz'])) == 8
        assert len(find_point_cloud_in_directory(test_data_dir, ['.csv'])) == 1
        assert len(find_point_cloud_in_directory(test_data_dir, ['.pts'])) == 1
        assert len(find_point_cloud_in_directory(test_data_dir, ['.pts', '.las', '.ply'])) == 25

    def test_include_subdirectories(self):
        # Test the flag for including subdirs
        assert len(find_point_cloud_in_directory(test_data_dir, include_subdirectories=False)) == 0
        assert len(find_point_cloud_in_directory(test_data_dir, include_subdirectories=True)) == 48

class TestAbstractIOMethods:
    def test_validate_field_selection_case1(self):
        # Case 1 - User provides no input
        user_fields = None
        header_fields = ['x', 'y', 'z', 'R', 'G', 'B', 'nx', 'ny', 'nz', 'scalar_sf1']

        final_fields = AbstractIOHandler._validate_field_selection(user_fields, header_fields, True, 'scalar_')

        assert list(final_fields.keys()) == ['r', 'g', 'b', 'nx', 'ny', 'nz', 'sf1']
        assert list(final_fields.values()) == ['R', 'G', 'B', 'nx', 'ny', 'nz', 'scalar_sf1']

    def test_validate_field_selection_case2(self):
        # Case 2 - User provides no input but file reading could not resolve the header / field names
        user_fields = None
        header_fields = []

        with pytest.raises(ValueError):
            AbstractIOHandler._validate_field_selection(user_fields, header_fields, False, '')

    def test_validate_field_selection_case3(self):
        # Case 3 - User enters an empty list, indicating ignore all fields (XYZ only)
        user_fields = []
        header_fields = ['x', 'y', 'z', 'R', 'G', 'B', 'nx', 'ny', 'nz', 'scalar_sf1']

        final_fields = AbstractIOHandler._validate_field_selection(user_fields, header_fields, True, 'scalar_')

        assert final_fields == {}

    def test_validate_field_selection_case4(self):
        # Case 4 - User provides a subset of fields available
        user_fields = ['r', 'g', 'b', 'sf1']
        header_fields = ['x', 'y', 'z', 'R', 'G', 'B', 'nx', 'ny', 'nz', 'scalar_sf1']

        final_fields = AbstractIOHandler._validate_field_selection(user_fields, header_fields, True, 'scalar_')

        assert list(final_fields.keys()) == ['r', 'g', 'b', 'sf1']
        assert list(final_fields.values()) == ['R', 'G', 'B', 'scalar_sf1']


    def test_validate_field_selection_case5(self):
        # Case 5 - User provides a subset of fields but in the name format of that used in the file
        user_fields = ['NORMALX', 'NORMALY', 'NORMALZ', 'scalar_sf1']
        header_fields = ['x', 'y', 'z', 'R', 'G', 'B', 'NORMALX', 'NORMALY', 'NORMALZ', 'scalar_sf1']

        final_fields = AbstractIOHandler._validate_field_selection(user_fields, header_fields, True, 'scalar_')

        assert list(final_fields.keys()) == ['normalx', 'normaly', 'normalz', 'sf1']
        assert list(final_fields.values()) == ['NORMALX', 'NORMALY', 'NORMALZ', 'scalar_sf1']


    def test_validate_field_selection_case6(self):
        # Case 6 - Invalid / unrecognised field is passed
        user_fields = ['normalx', 'normaly', 'normalz', 'sf1', 'sf_unknown']
        header_fields = ['x', 'y', 'z', 'R', 'G', 'B', 'NORMALX', 'NORMALY', 'NORMALZ', 'scalar_sf1']

        with pytest.raises(ValueError):
            AbstractIOHandler._validate_field_selection(user_fields, header_fields, True, 'scalar_')

    def test_extract_xyz(self, struct_array):
        # Extraction of the xyz from a struct array
        xyz = np.random.rand(100, 3).astype(np.float32)
        struct_array['x'] = xyz[:, 0]
        struct_array['y'] = xyz[:, 1]
        struct_array['z'] = xyz[:, 2]

        xyz_ = AbstractIOHandler.extract_xyz(struct_array, N)
        assert xyz_.dtype == np.float32
        assert xyz_.shape == (N, 3)
        assert np.all(xyz_ == xyz)

    def test_extract_scalar_fields(self, struct_array, valid_normals):
        pcd = PointCloudData(np.random.rand(N, 3).astype(np.float32))
        rgb = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
        intensity = np.random.randint(0, 255, N, dtype=np.uint8)

        header_fields = ['x', 'y', 'z', 'r', 'g', 'b', 'nx', 'ny', 'nz', 'intensity', 'sf1']
        final_fields = AbstractIOHandler._validate_field_selection(None, header_fields, True, 'scalar_')

        struct_array['x'] = pcd[:, 0]
        struct_array['y'] = pcd[:, 1]
        struct_array['z'] = pcd[:, 2]
        struct_array['r'] = rgb[:, 0]
        struct_array['g'] = rgb[:, 1]
        struct_array['b'] = rgb[:, 2]
        struct_array['nx'] = valid_normals[:, 0]
        struct_array['ny'] = valid_normals[:, 1]
        struct_array['nz'] = valid_normals[:, 2]
        struct_array['intensity'] = intensity
        struct_array['sf1'] = intensity.astype(np.float64)

        AbstractIOHandler.extract_scalar_fields(pcd, struct_array, num_points=N, field_names=final_fields)

        assert np.all(pcd.intensity == intensity)
        assert np.all(pcd.rgb == rgb)
        assert np.allclose(pcd.normals, valid_normals)
        assert np.all(pcd.scalar_fields['sf1'] == intensity.astype(np.float64))

    def test_extract_scalar_triplets(self, struct_array):

        normals = np.random.rand(N, 3).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1).reshape(-1,1)
        intensity = np.random.randint(0, 255, N, dtype=np.uint8)
        struct_array['nx'] = normals[:, 0]
        struct_array['ny'] = normals[:, 1]
        struct_array['nz'] = normals[:, 2]
        struct_array['intensity'] = intensity

        field_names = {'r': 'r', 'g': 'g', 'b': 'b', 'nx': 'nx', 'ny': 'ny', 'nz': 'nz', 'sf1': 'sf1'}

        rgb_fields = _get_rgb_or_normal_field_names(['x', 'y', 'z', 'r', 'g', 'b', 'sf1'], RGB_NAMES)
        rgb = AbstractIOHandler._extract_scalar_field_triplet(struct_array, N, rgb_fields, RGBFields, field_names)

        assert np.all(rgb[:, 0] == struct_array['r'])
        assert np.all(rgb[:, 1] == struct_array['g'])
        assert np.all(rgb[:, 2] == struct_array['b'])

        normal_fields = _get_rgb_or_normal_field_names(['nx', 'ny', 'nz', 'r', 'g', 'b', 'sf1'], NORMAL_NAMES)
        normals = AbstractIOHandler._extract_scalar_field_triplet(struct_array, N, normal_fields, NormalFields, field_names)

        assert np.allclose(normals[:, 0], struct_array['nx'])
        assert np.allclose(normals[:, 1], struct_array['ny'])
        assert np.allclose(normals[:, 2], struct_array['nz'])

    def test_generate_struct_dtype(self):
        sfs = ['r', 'g', 'b', 'nx', 'ny', 'nz', 'intensity', 'sf1']

        expected_optimised = {
            'names': ['x', 'y', 'z', 'r', 'g', 'b', 'nx', 'ny', 'nz', 'intensity', 'sf1'],
            'formats': [np.float32, np.float32, np.float32, np.uint8, np.uint8, np.uint8,
                        np.float32, np.float32, np.float32, np.uint8, np.float64]
        }

        expected_not_optimised = {
            'names': ['x', 'y', 'z', 'r', 'g', 'b', 'nx', 'ny', 'nz', 'intensity', 'sf1'],
            'formats': [np.float64, np.float64, np.float64, np.uint8, np.uint8, np.uint8,
                        np.float32, np.float32, np.float32, np.uint8, np.float64]
        }

        expected_vals = (expected_optimised, expected_not_optimised)

        pcd1 = PointCloudData(np.random.rand(N, 3))
        pcd2 = PointCloudData(np.random.rand(N, 3), numerical_optimization_shift=None)
        for i, pcd in enumerate([pcd1, pcd2]):
            pcd.rgb = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
            pcd.normals = np.random.rand(N, 3).astype(np.float32)
            pcd.intensity = np.random.randint(0, 255, N, dtype=np.uint8)
            pcd.scalar_fields.create_field('sf1', np.random.rand(N).astype(np.float64))

            pcd_dt = AbstractIOHandler._generate_struct_dtype(pcd, sfs, False)

            assert pcd_dt == expected_vals[i]

    def test_generate_struct_array(self):
        for i in range(2):
            if i == 0:
                pcd = PointCloudData(np.random.rand(N, 3))
            else:
                pcd = PointCloudData(np.random.rand(N, 3), numerical_optimization_shift=None)

            pcd.rgb = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
            pcd.normals = np.random.rand(N, 3).astype(np.float32)
            pcd.intensity = np.random.randint(0, 255, N, dtype=np.uint8)
            pcd.scalar_fields.create_field('sf1', np.random.rand(N).astype(np.float64))

            struct_arr = AbstractIOHandler._generate_structured_array(pcd, scalar_fields=None, add_prefix=True, prefix='dummy_', revert_sf_types=False)

            assert isinstance(struct_arr, np.ndarray)
            assert len(struct_arr) == N

            assert np.all(struct_arr['dummy_sf1'] == pcd.scalar_fields['sf1'])
            assert np.all(struct_arr['dummy_intensity'] == pcd.scalar_fields['intensity'])
            assert np.all(struct_arr['r'] == pcd.rgb.r)
            assert np.all(struct_arr['g'] == pcd.rgb.g)
            assert np.all(struct_arr['b'] == pcd.rgb.b)
            assert np.allclose(struct_arr['nx'], pcd.normals.nx)
            assert np.allclose(struct_arr['ny'], pcd.normals.ny)
            assert np.allclose(struct_arr['nz'], pcd.normals.nz)

            struct_arr = AbstractIOHandler._generate_structured_array(pcd, scalar_fields=['rf', 'gf', 'bf', 'sf1'], add_prefix=False, prefix='', revert_sf_types=False)

            assert isinstance(struct_arr, np.ndarray)
            assert len(struct_arr) == N

            assert np.all(struct_arr['sf1'] == pcd.scalar_fields['sf1'])
            assert 'intensity' not in struct_arr.dtype.names
            assert 'rf' in struct_arr.dtype.names
            assert 'gf' in struct_arr.dtype.names
            assert 'bf' in struct_arr.dtype.names
            assert 'nx' not in struct_arr.dtype.names
            assert 'ny' not in struct_arr.dtype.names
            assert 'nz' not in struct_arr.dtype.names
            assert 'x' in struct_arr.dtype.names
            assert 'y' in struct_arr.dtype.names
            assert 'z' in struct_arr.dtype.names

            if 'rf' in struct_arr.dtype.names:
                assert struct_arr['rf'].dtype == np.float32
                assert struct_arr['gf'].dtype == np.float32
                assert struct_arr['bf'].dtype == np.float32
                assert np.isclose(struct_arr['rf'].min(), 0, atol=0.5/256)
                assert np.isclose(struct_arr['gf'].min(), 0, atol=0.5/256)
                assert np.isclose(struct_arr['bf'].min(), 0, atol=0.5/256)
                assert np.all(struct_arr['rf'] <= 1)
                assert np.all(struct_arr['gf'] <= 1)
                assert np.all(struct_arr['bf'] <= 1)


    def test_get_field_names(self):
        # Fields by character
        all_char_names = ['x', 'y', 'z', 'r', 'g', 'b', 'nx', 'ny', 'nz', 'sf1', 'sf2']
        rgb_names = _get_rgb_or_normal_field_names(all_char_names, RGB_NAMES)
        normal_names = _get_rgb_or_normal_field_names(all_char_names, NORMAL_NAMES)
        assert rgb_names == list(RGB_NAMES.char)
        assert normal_names == list(NORMAL_NAMES.char)

        # Partial_field_names
        partial_names = ['red', 'nx', 'nz', 'sf1', 'sf2']
        rgb_names = _get_rgb_or_normal_field_names(partial_names, RGB_NAMES)
        normal_names = _get_rgb_or_normal_field_names(partial_names, NORMAL_NAMES)
        assert rgb_names == []
        assert normal_names == []

        # Fields by full 'word'
        all_word_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'normalx', 'normaly', 'normalz', 'sf1', 'sf2']
        rgb_names = _get_rgb_or_normal_field_names(all_word_names, RGB_NAMES)
        normal_names = _get_rgb_or_normal_field_names(all_word_names, NORMAL_NAMES)
        assert rgb_names == list(RGB_NAMES.words)
        assert normal_names == list(NORMAL_NAMES.words)

        # Floating point rgb fields
        float_names = ['x', 'y', 'z', 'rf', 'gf', 'bf', 'sf1', 'sf2']
        rgb_names = _get_rgb_or_normal_field_names(float_names, RGB_NAMES)
        assert rgb_names == list(RGB_NAMES.float)

        # Error when using Intensity or reflectance names
        field_names = ['r', 'g', 'b']
        for ref_names in (REFLECTANCE_NAMES, INTENSITY_NAMES):
            with pytest.raises(ValueError):
                _get_rgb_or_normal_field_names(field_names, ref_names)

        base_names = ['a', 'd', 'e']
        result = _get_rgb_or_normal_field_names(base_names, RGB_NAMES)
        assert result == []

    def test_get_sf_dtype(self):
        # Get the original or current scalar_field dtype
        rgb = RGBFields(np.random.rand(100,3).astype(np.float32))
        assert _get_sf_dtype(rgb, False) == np.uint8
        assert _get_sf_dtype(rgb, True) == np.float32

    def test_clean_field_names(self):
        # Remove X,Y,Z and scalar prefixes from a list of field names
        scalar_fields = ['scalar_a', 'scalar_b',' c  ', 'd', 'Scalar_E']
        all_fields = ['x', 'Y','Z'] + scalar_fields
        cleaned_sfs = _clean_field_names(scalar_fields, _clean_header_name, prefix='scalar_')
        cleaned_all_fields = _clean_field_names(all_fields, _clean_header_name, prefix='scalar_')
        assert list(cleaned_sfs.keys()) == ['a', 'b', 'c', 'd', 'e']
        assert list(cleaned_all_fields.keys()) == ['a', 'b', 'c', 'd', 'e']
        assert list(cleaned_sfs.values()) == scalar_fields
        assert list(cleaned_all_fields.values()) == scalar_fields
        assert list(_clean_field_names(['x', 'Y', 'Z'], _clean_header_name, prefix='')) == ['x', 'y', 'z']

    def test_clean_strings(self):
        # Remove white space and convert to lowercase
        assert _clean_string("   abcd   ") == "abcd"
        assert _clean_string("AbCdEfG") == "abcdefg"
        assert _clean_string("   BobTheBuilder") == "bobthebuilder"

    def test_clean_header_names(self):
        # Recursively remove prefixes from the field name
        assert _clean_header_name("scalar_scalar_scalar_abcd", prefix="scalar_") == "abcd"
        assert _clean_header_name("Scalar_Scalar_Scalar_abcd", prefix="scalar_") == "abcd"
        assert _clean_header_name("scalar_Scalar_scalar_abcd", prefix="Scalar_") == "abcd"
        assert _clean_header_name("scalar_abcd", prefix="scalar_") == "abcd"
        assert _clean_header_name("scalar abcd", prefix="scalar") == "abcd"
        assert _clean_header_name(" scalar abcd ", prefix="scalar") == "abcd"
        assert _clean_header_name("scalar_", prefix="scalar_") == "scalar_"

class BaseLoadSave:
    cls: type[AbstractIOHandler] = AbstractIOHandler
    folder: Path = test_data_dir
    reference: Path = folder / 'replace_this_path.txt'

    def test_find_pcds(self):
        num_files = len(self.cls.find_pcds_in_directory(self.folder))
        if '.e57' in self.cls.FORMATS:
            assert num_files == 6
        elif '.txt' in self.cls.FORMATS:
            assert num_files == 10
        elif '.pcd' in self.cls.FORMATS:
            assert num_files == 1
        else:
            assert num_files == 8

    def _load_all(self, **kwargs):
        reference = self.cls.load(self.reference, remove_prefix=True, **kwargs)

        for fmt in self.cls.FORMATS:
            for file in find_point_cloud_in_directory(self.folder, [fmt], True):
                pcd = self.cls.load(file, remove_prefix=True)
                assert len(pcd) == 54202
                assert isinstance(pcd, PointCloudData)

                if 'XYZ' in file.name:
                    assert np.allclose(reference, pcd.xyz)

                if 'RGB' in file.name:
                    assert np.allclose(reference.rgb, pcd.rgb)
                else:
                    assert pcd.rgb is None

                if 'Normals' in file.name:
                    if file.suffix == '.e57':
                        warnings.warn(UserWarning('pye57 does not yet support normals'))
                        assert pcd.normals is None
                    else:
                        assert np.allclose(reference.normals, pcd.normals)
                else:
                    assert pcd.normals is None

                if 'Intensity' in file.name:
                    assert np.allclose(reference.intensity, pcd.intensity)
                else:
                    assert pcd.intensity is None

                if 'sfs' in file.name.lower():
                    assert np.allclose(reference.scalar_fields['custom1'], pcd.scalar_fields['custom1'])
                    assert np.allclose(reference.scalar_fields['sqrt(custom1)'], pcd.scalar_fields['sqrt(custom1)'])
                else:
                    assert 'custom1' not in pcd.scalar_fields
                    assert 'sqrt(custom1)' not in pcd.scalar_fields

    def _save(self, tmp_path, **kwargs):
        temp_file = tmp_path / ('tmp' + self.cls.FORMATS[0])
        original_pcd = self.cls.load(self.reference, remove_prefix=True)
        self.cls.save(original_pcd, temp_file, add_prefix=False, **kwargs)
        new_pcd = self.cls.load(temp_file, remove_prefix=False)

        if '.csv' in self.cls.FORMATS:
            # Adapt precision due to the number of digits written to file
            assert np.allclose(original_pcd.xyz, new_pcd.xyz, atol=1e-06)
        elif '.las' in self.cls.FORMATS:
            # Adapt precision due to the "scale" factor as this rounds digits
            assert np.allclose(original_pcd.xyz, new_pcd.xyz, atol=1e-04)
        else:
            assert np.allclose(original_pcd.xyz, new_pcd.xyz)

        assert np.allclose(original_pcd.rgb, new_pcd.rgb, atol=1)

        assert np.allclose(original_pcd.normals, new_pcd.normals)

        for name in original_pcd.scalar_fields.keys():
            assert name in new_pcd.scalar_fields
