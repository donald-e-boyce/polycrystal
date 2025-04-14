"""Convert among parameterizations"""
from scipy.spatial.transform import Rotation as ScipyRotation

from .baseclass import Parameterization
from ._registry import registry


def names():
    """return list of orientation parameterizatons"""
    return list(registry.keys())


def to_rmats(orientations, parameterization):
    """paramert list of orientations to rotation matrices

    Parameters
    ----------
    orientations: array (n, m)
       array of `n` parameter `m`-vectors
    parameterization: str
       name of orientation parameterization

    Returns
    -------
    array (n, 3, 3)
       array of `n` 3x3 matrices
"""
    return registry[parameterization]().to_rmats(orientations)


def from_rmats(rmats, parameterization):
    """paramert list of rotation matrices to list of orientations

    Parameters
    ----------
    rmats: array (n, 3, 3)
       array of `n` 3x3 matrices
    parameterization: str
       name of orientation parameterization

    Returns
    -------
    array (n, m)
        array of `n` parameter `m`-vectors
    """
    return registry[parameterization]().from_rmats(rmats)


def convert(from_ori, from_param, to_param):
    """paramert between orientation parameterizations

    Parameters
    ----------
    from_ori: array (n, m1)
       input array of orientation parameters
    from_param: str
       name for the input orientation parameterization
    to_param: str
       name of the output  parameterization

    Returns
    -------
    array (n, m2)
       array of orientation parameters in the new parameterization
"""
    return from_rmats(to_rmats(from_ori, from_param), to_param)


# Below are the actual conversion classes.


class Quaternion(Parameterization):
    """Quaternion"""
    name = 'quaternions'

    def to_rmats(self, a):
        return Rotation.as_matrix(Rotation.from_quat(a))

    def from_rmats(self, r):
        return Rotation.as_quat(Rotation.from_matrix(r))


class Exponential(Parameterization):
    """Exponential map using axial vector of skew matrix"""
    name = 'exponentials'

    def to_rmats(self, a):
        return Rotation.as_matrix(Rotation.from_rotvec(a))

    def from_rmats(self, r):
        return Rotation.as_rotvec(Rotation.from_matrix(r))
