"""Convert among parameterizations"""

import abc

from scipy.spatial.transform import Rotation


# This is the registry for conversion classes.
_registry = dict()


# First is the abstract base class for parameterization conversions.


class _ParameterizationABC(abc.ABC):

    @abc.abstractmethod
    def to_rmats(self, a):
        """convert from parameterization to rotation matrices

        Parameters
        ----------
        a: array (n, m)
           array of `n` parameter arrays, each of length `m`

        Returns
        -------
        array (n, 3, 3)
           array of `n` 3x3 matrices
        """

    @abc.abstractmethod
    def from_rmats(self, r):
        """convert to parameterization from rotation matrices

        Parameters
        ----------
        r: array (n, 3, 3)
           array of `n` 3x3 matrices

        Returns
        -------
        array (n, m)
           array of `n` parameter arrays, each of length `m`
        """


# Below is the registry that metaclass that updates the dictionary of conversion classes
# during class instantiation.


class _ParameterizationRegistry(abc.ABCMeta):
    """Keep a dictionary of parameterizations by name"""

    def __init__(cls, name, bases, attrs):
        type.__init__(cls, name, bases, attrs)

        if hasattr(cls, "name"):
            _registry[cls.name] = cls


# Here is the base class for parameterizations, which automatically registers
# conversion classes.


class _Parameterization(_ParameterizationABC, metaclass=_ParameterizationRegistry):
    """Base class for parameterizations"""


# Next is the actual module API.


def names():
    """return list of orientation parameterizatons"""
    return list(_registry.keys())


def to_rmats(orientations, parameterization):
    """paramert list of orientations to rotation matrices

    Parameters
    ----------
    orientations: array (n, m)
       array of `n` parameter sets, each of length `m`
    parameterization: str
       name of orientation parameterization

    Returns
    -------
    array (n, 3, 3)
       array of `n` 3x3 matrices
    """
    return _registry[parameterization]().to_rmats(orientations)


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
    return _registry[parameterization]().from_rmats(rmats)


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


class _Quaternion(_Parameterization):
    """Quaternion"""

    name = "quaternions"

    def to_rmats(self, a):
        return Rotation.as_matrix(Rotation.from_quat(a))

    def from_rmats(self, r):
        return Rotation.as_quat(Rotation.from_matrix(r))


class _Exponential(_Parameterization):
    """Exponential map using axial vector of skew matrix"""

    name = "exponentials"

    def to_rmats(self, a):
        return Rotation.as_matrix(Rotation.from_rotvec(a))

    def from_rmats(self, r):
        return Rotation.as_rotvec(Rotation.from_matrix(r))
