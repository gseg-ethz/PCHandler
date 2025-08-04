import pytest

import numpy as np
from pchandler.geometry.transforms import Transform


@pytest.fixture
def identity_transform():
    """Fixture for creating an identity Transform."""
    return Transform(np.eye(4))

@pytest.fixture
def random_transform():
    """Fixture for creating a random affine transform."""
    rotation = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float32)
    translation = np.array([5, 10, 15], dtype=np.float32)
    scale = np.array([2, 2, 2], dtype=np.float32)
    return Transform.generate(rotation=rotation, translation=translation, scale=scale)


def test_identity_transform(identity_transform):
    """Test that the identity transform behaves as expected."""
    assert np.allclose(identity_transform.arr, np.eye(4)), "Identity transform should be the 4x4 identity matrix."


def test_transform_from_translation():
    """Test creating a transform from a translation vector."""
    translation = np.array([10, 20, 30])
    transform = Transform.from_translation(translation)
    expected = np.eye(4)
    expected[:3, 3] = translation
    assert np.allclose(transform.arr, expected), "Transform from translation failed."


def test_transform_from_rotation():
    """Test creating a transform from a rotation matrix."""
    rotation = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    transform = Transform.from_rotation(rotation)
    expected = np.eye(4)
    expected[:3, :3] = rotation
    assert np.allclose(transform.arr, expected), "Transform from rotation failed."


def test_transform_from_scale():
    """Test creating a transform from scaling factors."""
    scale = np.array([2, 2, 2])
    transform = Transform.from_scale(scale)
    expected = np.eye(4)
    np.fill_diagonal(expected[:3, :3], scale)
    assert np.allclose(transform.arr, expected), "Transform from scale failed."


def test_transform_from_affine():
    """Test creating a transform from an affine matrix."""
    affine = np.eye(4) * 2
    transform = Transform.from_affine(affine)
    assert np.allclose(transform.arr, affine), "Transform from affine failed."


def test_transform_matrix_multiplication(identity_transform, random_transform):
    """Test matrix multiplication between transforms."""
    result = identity_transform @ random_transform
    assert np.allclose(result.arr, random_transform.arr), "Multiplication with identity transform failed."

    result_self = random_transform @ random_transform
    assert result_self.shape == random_transform.shape
    # Verify it is valid or transpose expected data

    np_data = np.random.rand(4, 100)
    result = random_transform @ np_data
    assert result.shape == (4, 100)


def test_generate_transform():
    """Test creating a composite transform."""
    rotation = np.eye(3)
    translation = np.array([10, 20, 30])
    scale = 2
    transform = Transform.generate(rotation=rotation, translation=translation, scale=scale)

    expected = np.eye(4)
    expected[:3, :3] = rotation * scale
    expected[:3, 3] = translation
    assert np.allclose(transform.arr, expected), "Composite transformation generation failed."


def test_invalid_mode_in_generate():
    """Test passing an invalid mode to the _generate method."""
    with pytest.raises(ValueError, match="Invalid mode"):
        Transform._generate(values=np.eye(4), mode="invalid_mode")


def test_transform_inplace_multiplication(random_transform):
    """Test in-place matrix multiplication."""
    original_transform = random_transform.arr.copy()
    random_transform @= np.eye(4)
    assert np.allclose(random_transform.arr, original_transform), "In-place multiplication with identity failed."


def test_transform_equivalence():
    """Test equivalence of different transforms using the matrix."""
    t1 = Transform.generate(rotation=np.eye(3), translation=np.zeros(3), scale=1.0)
    t2 = Transform(np.eye(4))
    assert np.allclose(t1.arr, t2.arr), "Equivalence of generated identity transform and direct initialization failed."


def test_transform_invalid_inputs():
    """Test for invalid inputs to the Transform class."""
    with pytest.raises(Exception):
        Transform(np.zeros((3, 3)))  # Invalid affine shape
