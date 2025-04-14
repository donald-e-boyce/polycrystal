"""Orientation Parmaeterizations

The primary interface here is through the `to_rmats()`, `from_rmats()` and
`convert()` functions. To add a new parameterization, make a subclass of the
`Parameterization` class in `baseclass.py`.  Add the parameterization name in the
`parameterization` attribute. Then implement the `to_rmats()` and `from_rmats()`
methods for that class. The parameterization will be automatically registered here.

The parameterizations here are not meant to be exhaustive. They are mainly for
internal use, particularly the quaternions. The `scipy` Rotation class
is a good choice for more complete conversions.

See Also
---------

* `scipy.transform.Rotation`_

.. _scipy.transform.Rotation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
"""
import pkgutil
import importlib

from ._registry import registry

## Import all  modules that define a parameterization.
#IGNORE = set(('_registry', 'baseclass', 'parameterizationabc'))
#for loader, name, ispkg in pkgutil.iter_modules(__path__):
#    if name not in IGNORE:
#        importlib.import_module(f'.{name}', __package__)


def parameterizatons():
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
