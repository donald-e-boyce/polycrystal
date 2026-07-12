"""Tests for the voxeldata module"""
import numpy as np
import pytest

from polycrystal.microstructure import voxeldata


def identity_array(n):
    return np.tile(np.identity(3), (n, 1, 1))


@pytest.fixture
def vd():
    """4x3x2 grid with unit voxels, default origin and direction.

    gids[i, :, :] == i so grain ID equals the first-axis index.
    """
    shp = (4, 3, 2)
    w6 = np.ones(6)
    gids = np.hstack([i * w6 for i in range(4)]).reshape(shp).astype(int)
    oris = identity_array(4)
    return voxeldata.VoxelData(gids, oris, np.ones(3))


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

def test_attributes(vd):
    assert vd.shape == (4, 3, 2)
    assert np.all(vd.vdims == 1)
    assert np.all(vd.origin == 0)
    assert np.all(vd.lowleft == 0)
    assert np.all(vd.upright == (4, 3, 2))
    assert vd.num_cells == 24
    assert np.all(vd.orientation_list == identity_array(4))
    assert vd.direction == (True, True, True)
    assert vd.num_grains == 4
    assert vd.num_phases == 1


def test_num_grains_with_gaps():
    """num_grains == max(grain_id) + 1, not the count of distinct IDs."""
    shp = (2, 2, 2)
    gids = np.zeros(shp, dtype=int)
    gids[1, :, :] = 4          # IDs 0 and 4; 1, 2, 3 are absent
    vd = voxeldata.VoxelData(gids, identity_array(5), np.ones(3))
    assert vd.num_grains == 5


# ---------------------------------------------------------------------------
# grain()
# ---------------------------------------------------------------------------

def test_grain_along_x(vd):
    x = np.array([[0.5, 1.0, 1.0], [1.5, 1.0, 1.0],
                  [2.5, 1.0, 1.0], [3.5, 1.0, 1.0]])
    assert np.all(vd.grain(x) == [0, 1, 2, 3])


def test_grain_along_y_and_z(vd):
    """Varying y and z within the same x-slab should all return the same grain."""
    x = np.array([[0.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 2.5, 1.5]])
    assert np.all(vd.grain(x) == 0)


def test_grain_non_unit_voxels():
    shp = (2, 2, 2)
    gids = np.zeros(shp, dtype=int)
    gids[1, :, :] = 1
    vd = voxeldata.VoxelData(gids, identity_array(2), voxel_dims=(2.0, 2.0, 2.0))
    x = np.array([[0.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
    assert np.all(vd.grain(x) == [0, 1])


def test_grain_non_default_origin():
    shp = (2, 2, 2)
    gids = np.zeros(shp, dtype=int)
    gids[1, :, :] = 1
    vd = voxeldata.VoxelData(gids, identity_array(2), voxel_dims=(1, 1, 1),
                              origin=(10, 10, 10))
    x = np.array([[10.5, 10.5, 10.5], [11.5, 10.5, 10.5]])
    assert np.all(vd.grain(x) == [0, 1])


def test_grain_boundary_clamping(vd):
    """Points outside the grid clamp to the nearest cell."""
    x_low  = np.array([[-5.0, 1.0, 1.0]])   # far below origin
    x_high = np.array([[99.0, 1.0, 1.0]])    # far above upright
    assert vd.grain(x_low)[0]  == 0          # clamps to first x-slab
    assert vd.grain(x_high)[0] == 3          # clamps to last x-slab


# ---------------------------------------------------------------------------
# direction option
# ---------------------------------------------------------------------------

def test_grain_direction_all_false():
    """direction=False on all axes reverses index order on every axis."""
    shp = (3, 1, 1)
    gids = np.arange(3).reshape(shp)   # gids[i,0,0] == i
    vd = voxeldata.VoxelData(gids, identity_array(3), np.ones(3),
                              direction=(False, True, True))
    # With direction[0]=False: voxel 0 is at the HIGH physical-x end (x near 3),
    # voxel 2 is at the LOW end (x near 0).
    x = np.array([[2.5, 0.5, 0.5],   # high-x → voxel index 0 → grain 0
                  [1.5, 0.5, 0.5],   # mid-x  → voxel index 1 → grain 1
                  [0.5, 0.5, 0.5]])  # low-x  → voxel index 2 → grain 2
    assert np.all(vd.grain(x) == [0, 1, 2])


def test_grain_direction_mixed():
    """direction=(True, False, True): y-axis is reversed."""
    shp = (1, 3, 1)
    gids = np.arange(3).reshape(shp)   # gids[0,j,0] == j
    vd = voxeldata.VoxelData(gids, identity_array(3), np.ones(3),
                              direction=(True, False, True))
    # y increases physically from 0→3, but voxel index 0 is at high-y end.
    x = np.array([[0.5, 2.5, 0.5],   # high-y → voxel index 0 → grain 0
                  [0.5, 1.5, 0.5],   # mid-y  → voxel index 1 → grain 1
                  [0.5, 0.5, 0.5]])  # low-y  → voxel index 2 → grain 2
    assert np.all(vd.grain(x) == [0, 1, 2])


def test_direction_clamping_reversed(vd):
    """Boundary clamping works correctly when direction is False."""
    shp = (3, 1, 1)
    gids = np.arange(3).reshape(shp)
    vd = voxeldata.VoxelData(gids, identity_array(3), np.ones(3),
                              direction=(False, True, True))
    x_past_high = np.array([[99.0, 0.5, 0.5]])   # beyond physical high end
    x_past_low  = np.array([[-1.0, 0.5, 0.5]])   # beyond physical low end
    assert vd.grain(x_past_high)[0] == 0   # clamps to voxel 0 (high-x end)
    assert vd.grain(x_past_low)[0]  == 2   # clamps to voxel 2 (low-x end)


# ---------------------------------------------------------------------------
# phase() and orientation
# ---------------------------------------------------------------------------

def test_phase(vd):
    g = np.array([0, 1, 2, 3])
    assert np.all(vd.phase(g) == 0)


def test_grain_orientation(vd):
    g = np.array([0, 1, 2, 3])
    result = vd.grain_orientation(g)
    assert result.shape == (4, 3, 3)
    assert np.allclose(result, identity_array(4))
