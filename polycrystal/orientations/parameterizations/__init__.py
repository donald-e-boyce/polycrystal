"""Orientation Parmaeterizations

The primary interface here is through the `to_rmats()`, `from_rmats()` and
`convert()` functions. To add a new parameterization, make a subclass of the
`Parameterization` class in `baseclass.py`.  Add the parameterization name in the
`parameterization` attribute. Then implement the `to_rmats()` and `from_rmats()`
methods for that class. The parameterization will be automatically registered here.

The parameterizations here are not meant to be exhaustive. They are mainly for
internal use, particularly the quaternions. They are based on the `scipy` Rotation
class when possible.

See Also
---------

* `scipy.spatial.transform.Rotation`_

.. _scipy.transform.Rotation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
"""

from ._convert import names, to_rmats, from_rmats, convert
