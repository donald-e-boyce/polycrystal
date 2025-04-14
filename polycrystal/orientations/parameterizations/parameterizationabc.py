"""Abstract Base Class for orientation parameterizations"""
import abc


class ParameterizationABC(abc.ABC):

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
        pass

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
        pass
