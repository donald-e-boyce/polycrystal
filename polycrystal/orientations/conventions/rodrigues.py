"""Rodrigues parameters"""
import numpy as np

from polycrystal.orientations import quaternions


def volume_element(r):
    """Compute volume element for Rodrigues parameters

    PARAMETERS
    ----------
    r: aray(n, 3)
       input array of Rodrigues parameters

    RETURNS
    -------
    array(n):
       volume element at each point
    """
    return 1/(1 + np.sum(r * r, 1)) ** 2


def to_quaternions(r):
    """Convert to Rodrigues parameters to quaternions

    PARAMETERS
    ----------
    r: aray(n, 3)
       input array of Rodrigues parameters

    RETURNS
    -------
    array(n, 4):
       array of quaterions
    """
    r = np.atleast_2d(r)
    n = len(r)
    cphiby2   = np.cos(np.arctan(np.sqrt(np.sum(r*r,1)))).reshape(n,1)
    q = np.hstack((cphiby2, cphiby2*r))

    return q


def distance(r1, r2):
    """Distance between Rodrigues parameters

    r1 - array (nx3 or 3) of Rodrigues parameters
    r2 - array (nx3 or 3) of Rodrigues parameters

    RETURNS
    d - distance between r1 and r2
        if either has length 3, it is intepreted as a single Rodrigues parameter
        and distance is found between taht
    """
    q1 = to_quaternion(r1)
    q2 = quaternions.inverse(to_quaternion(r2))
    t = quaternions.multiply(q1, q2)

    return 2*np.arccos(t[:,0])
