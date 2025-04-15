"""Unit testing for quaternions module"""

import pytest
import numpy as np

from polycrystal.orientations import parameterizations


class Names:
    quats = "quaternions"
    avecs = "axial-vectors"


@pytest.fixture
def identity_matrix():
    return np.identity(3)


@pytest.fixture
def identity_quaternion():
    return np.array([1.0, 0, 0, 0])


@pytest.fixture
def identity_axial_vector():
    return np.array([0.0, 0, 0])


def test_identity(identity_matrix, identity_quaternion, identity_axial_vector):
    rmat = parameterizations.to_rmats(identity_quaternion, Names.quats)
    assert np.allclose(rmat, identity_matrix)

    rmat = parameterizations.to_rmats(identity_axial_vector, Names.avecs)
    assert np.allclose(rmat, identity_matrix)


def test_convert(identity_quaternion, identity_axial_vector):
    avec = parameterizations.convert(identity_quaternion, Names.quats, Names.avecs)
    assert np.allclose(avec, identity_axial_vector)

    avec = parameterizations.convert(identity_axial_vector, Names.avecs, Names.quats)
    assert np.allclose(avec, identity_quaternion)


# Now test axial-vectors with multiple rotations.


@pytest.fixture
def avec_90deg():
    p2 = np.pi / 2.0
    return np.array([[p2, 0, 0], [0, p2, 0], [0, 0, p2]])


@pytest.fixture
def rmat_90deg():
    """Rotation matrices for 90-degree rotations about each axis"""
    return np.array(
        [
            [[1.0, 0, 0], [0, 0, -1], [0, 1, 0]],
            [[0, 0, 1], [0, 1.0, 0], [-1, 0, 0]],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1.0]],
        ]
    )


def test_axial_vectors(avec_90deg, rmat_90deg):

    rmat = parameterizations.to_rmats(avec_90deg, Names.avecs)
    assert np.allclose(rmat, rmat_90deg)
