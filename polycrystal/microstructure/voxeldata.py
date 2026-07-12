"""Voxel data based on regular grid"""
from polycrystal.microstructure import\
    ConstantGrainOrientationMicrostructure as CgoMicrostructure

import numpy as np


class VoxelData(CgoMicrostructure):
    """Voxel data based on regular grid

    Parameters
    ----------
    grain_ids: int array (l, m, n)
       array of grain IDs. IDs need not be contiguous — gaps are permitted so
       that a coarser voxel dataset can reuse the same IDs as a finer one, and
       so that ID 0 can be reserved for unknown or unassigned voxels.
       `num_grains` returns ``grain_ids.max() + 1``, which is the number of
       slots required in `orientation_list`, not necessarily the count of
       distinct grains present in the array.
    orientation_list: array (num_grains, 3, 3)
       orientations indexed by grain ID
    voxel_dims: 3-tuple
       the voxel dimensions in each direction
    origin: tuple | array, default = (0,0,0)
       lower left corner of box
    direction: 3-tuple of bools, default = (True, True, True)
       voxel ordering per axis; True means the voxel index increases with
       coordinate value, False means it decreases
    """

    def __init__(
            self, grain_ids, orientation_list, voxel_dims,
                 origin=(0,0,0), direction=3 * (True,)
    ):
        self.gids = grain_ids
        self.shape = grain_ids.shape
        self._num_grains = self.gids.max() + 1
        self.vdims = np.array(voxel_dims)
        self.origin = np.array(origin)
        self.lowleft = self.origin
        self.upright = self.origin + self.shape * self.vdims
        self._orientations = orientation_list
        self.direction = direction

    @property
    def num_grains(self):
        return self._num_grains

    @property
    def num_phases(self):
        return 1

    @property
    def direction(self):
        """Voxel order direction (increasing/decreasing) per axis"""
        return self._direction

    @direction.setter
    def direction(self, v):
        self._direction = v
        self.v0 = np.where(v, self.lowleft, self.upright)
        self.dv = np.where(v, 1, -1) * self.vdims

    @property
    def num_cells(self):
        """number of cells"""
        return np.prod(self.shape)

    def grain(self, x):
        """grain ID by position

        Parameters
        ----------
        x: array (n, 3)
           array of `n` points

        Returns
        -------
        int array (n)
           grain IDs for each point; points outside the grid are clamped to
           the nearest boundary cell
        """
        dx = x - self.v0
        vox = np.clip((dx / self.dv).astype(int), 0, np.array(self.shape) - 1)
        return self.gids[vox[:, 0], vox[:, 1], vox[:, 2]]

    def phase(self, g):
        """phase ID for grains `g`

        Parameters
        ----------
        g: int array (n)
           array of grain IDs

        Returns
        -------
        int array (n)
           phase IDs (always 0; VoxelData is single-phase)
        """
        return np.zeros(len(g), dtype=int)

    @property
    def orientation_list(self):
        return self._orientations

    def grain_orientation(self, g):
        """orientation of grains

        Parameters
        ----------
        g: int array (n)
           array of grain IDs

        Returns
        -------
        array (n, 3, 3)
           rotation matrices for each grain
        """
        return self.orientation_list[g]
