"""Unit testing for quaternions module"""

import pytest
import numpy as np

from polycrystal.orientations import parameterizations


class Names:
    quats = 'quaternions'
    avecs = 'axial_vectors'


@pytest.fixture
def identity_matrix():
    return np.identity(3)


@pytest.fixture
def identity_quaternion():
    return np.array([1., 0, 0., 0])


def test_identity(identity_quaternion, identity_matrix):
    rmat = parameterizations.to_rmats(identity_quaternion, Names.quats)
    assert np.allclose(rmat, identity_matrix)
