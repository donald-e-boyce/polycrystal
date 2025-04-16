"""Unit testing for quaternions module"""

import pytest
import numpy as np

from polycrystal.orientations import parameterizations


np.set_printoptions(precision=2, suppress=True)


class Names:
    quats = "quaternions"
    avecs = "axial-vectors"
    eulZXZ = "euler:ZXZ-deg"


@pytest.fixture
def identity_matrix():
    return np.identity(3)


@pytest.fixture
def identity_quaternion():
    return np.array([1.0, 0, 0, 0])


@pytest.fixture
def identity_axial_vector():
    return np.array([0.0, 0, 0])


@pytest.fixture
def identity_euler_ZXZ_deg():
    return np.array([0.0, 0, 0])


def test_identity(
        identity_matrix, identity_quaternion, identity_axial_vector,
        identity_euler_ZXZ_deg
):
    rmat = parameterizations.to_rmats(identity_quaternion, Names.quats)
    assert np.allclose(rmat, identity_matrix)

    rmat = parameterizations.to_rmats(identity_axial_vector, Names.avecs)
    assert np.allclose(rmat, identity_matrix)

    rmat = parameterizations.to_rmats(identity_euler_ZXZ_deg, Names.eulZXZ)
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


# Test Euler angles.


@pytest.fixture
def euler_ZXZ_in():
    return np.array(
        [
            [90, 0, 0],
            [0, 90, 0],
            [0, 0, 90],
            [90, 90, 0],
            [90, 0, 90],
            [0, 90, 90],
            [90, 90, 90]
        ],
        dtype=float
    )


@pytest.fixture
def euler_ZXZ_out():
    return np.array(
        [
            [[0,-1,0],[1,0,0],[0,0,1]],
            [[1,0,0],[0,0,-1],[0,1,0]],
            [[0,-1,0],[1,0,0],[0,0,1]],
            [[0,0,1],[1,0,0],[0,1,0]],
            [[-1,0,0],[0,-1,0],[0,0,1]],
            [[0,-1,0],[0,0,-1],[1,0,0]],
            [[0,0,1],[0,-1,0],[1,0,0]]
        ],
        dtype=float
    )


def test_euler_ZXZ_deg(euler_ZXZ_in, euler_ZXZ_out):

    rmat = parameterizations.to_rmats(euler_ZXZ_in, Names.eulZXZ)
    assert np.allclose(rmat, euler_ZXZ_out)
