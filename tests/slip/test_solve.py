"""Unit tests for MaterialData, StressData, StressSequence, and solvers in polycrystal.slip.solve."""
import pytest
import numpy as np

from polycrystal.slip.solve import (
    MaterialData,
    StressData,
    StressSequence,
    Pointwise,
    Euler,
    Block,
)
from polycrystal.slip.slipcrystal import SlipCrystal
from polycrystal.slip.slip_groups import get_group
from polycrystal.slip.slip_models import (
    AF_ZeroBackStress,
    AF_ZeroBackStressParameters,
    AF_SingleHardness,
    AF_SingleHardnessParameters,
)


class StubMaterial:
    """Minimal stub material model used for basic testing."""

    @property
    def num_statevar(self):
        return 17


@pytest.fixture
def fcc_afzb_material():
    """Create an FCC slip crystal with AFZB model."""
    params = AF_ZeroBackStressParameters(
        gamma_dot_0=0.00897,
        m=1 / 19.77,
        H=6391.49,
        H_d=18.06,
        q12=1.2,
    )
    return SlipCrystal([get_group("fcc")], AF_ZeroBackStress(params))


def test_materialdata_basic(tmp_path):
    """Test MaterialData construction, static helpers, properties, and round-trip I/O."""
    rmats = MaterialData.random_orientations(10)
    assert rmats.shape == (10, 3, 3)

    s1 = MaterialData.constant_state_af(5, 2.9, 1)
    assert s1.shape == (5,)
    assert s1[0] == 2.9

    s12 = MaterialData.constant_state_af(7, 3.2, 12)
    assert s12.shape == (7, 12)
    assert s12[0, 0] == 3.2

    s24 = MaterialData.constant_state_af(3, 1.4, 24)
    assert s24.shape == (3, 24)
    assert s24[0, 0] == 1.4
    assert s24[0, 12] == 0.0

    defg = MaterialData.constant_defgrad(8)
    assert defg.shape == (8, 3, 3)
    assert np.allclose(defg[0], np.eye(3))

    m = StubMaterial()
    md = MaterialData(m, rmats, s12, defg[:10])
    assert md.num_pts == 10
    assert md.num_sv == 17

    p = tmp_path / "out.npz"
    md.save_data(p)

    md2 = MaterialData.from_file(m, p)
    assert np.array_equal(md.orient, md2.orient)
    assert np.array_equal(md.state, md2.state)
    assert np.array_equal(md.def_grad, md2.def_grad)


def test_stressdata_basic(tmp_path):
    """Test StressData construction, round-trip I/O, interpolation, and frame rotation."""
    csig0 = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 0]])
    csig1 = np.array([[0.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
    T = 5.2
    sd = StressData(csig0, csig1, T)
    assert sd.T == 5.2

    p = tmp_path / "sd.npz"
    sd.save(p)

    sd2 = StressData.from_file(p)
    assert np.allclose(sd.csig0, sd2.csig0)
    assert np.allclose(sd.csigT, sd2.csigT)
    assert sd.T == sd2.T

    cmid = 0.5 * (csig0 + csig1)
    assert np.allclose(cmid, sd2.cstress_t(T / 2))

    # 90 degrees about x: [x, y, z] -> [x, z, -y]
    orient = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).reshape(1, 3, 3)
    s_sig = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]).reshape(1, 3, 3)
    c_sig = StressData.cstress_from_sample(s_sig, orient)
    assert s_sig[0, 1, 1] == c_sig[0, 2, 2]
    assert s_sig[0, 2, 2] == c_sig[0, 1, 1]


def test_stress_sequence(tmp_path):
    """Test StressSequence creation, field access, and round-trip I/O."""
    csig = np.zeros((3, 5, 3, 3))
    csig[1, :, 0, 0] = 100.0
    times = np.array([0.0, 10.0, 20.0])

    sseq = StressSequence(csig=csig, times=times)
    assert len(sseq.csig) == 3
    assert len(sseq.times) == 3

    p = tmp_path / "sseq.npz"
    sseq.save(p)

    sseq2 = StressSequence.from_file(p)
    assert np.allclose(sseq.csig, sseq2.csig)
    assert np.allclose(sseq.times, sseq2.times)


def test_pointwise_solver(fcc_afzb_material):
    """Test Pointwise ODE integration and plastic deformation gradient integration."""
    matl = fcc_afzb_material
    numpts = 4
    g0 = 345.63

    rmats = MaterialData.random_orientations(numpts)
    state = MaterialData.constant_state_af(numpts, g0, matl.num_statevar)
    mdata = MaterialData(matl, rmats, state)

    ssig0 = np.zeros((3, 3))
    ssig1 = np.diag([400.0, -200.0, -200.0])

    csig0 = StressData.cstress_from_sample(ssig0, rmats)
    csig1 = StressData.cstress_from_sample(ssig1, rmats)
    dtime = 5.0
    sdata = StressData(csig0, csig1, dtime)

    solver = Pointwise(mdata, sdata)

    # 1. Final state only
    final_state = solver.run(time_history=False)
    assert final_state.shape == (numpts, matl.num_statevar)
    # Hardness should increase under plastic deformation
    assert np.all(final_state >= state)

    # 2. Time history
    thist = solver.run(time_history=True)
    assert len(thist) == numpts
    for ipt in range(numpts):
        assert thist[ipt].shape[0] == 1 + matl.num_statevar
        assert thist[ipt][0, 0] == 0.0
        assert np.isclose(thist[ipt][0, -1], dtime)

    # 3. Integrate Fp
    fp = solver.integrate_fp(thist)
    assert fp.shape == (numpts, 3, 3)
    for ipt in range(numpts):
        # Determinant of plastic deformation gradient should be close to 1 (isochoric plastic flow)
        assert np.isclose(np.linalg.det(fp[ipt]), 1.0, atol=1e-3)


def test_euler_and_block_solvers(fcc_afzb_material):
    """Test Euler and Block solvers."""
    matl = fcc_afzb_material
    numpts = 3
    g0 = 345.63

    rmats = MaterialData.random_orientations(numpts)
    state = MaterialData.constant_state_af(numpts, g0, matl.num_statevar)
    mdata = MaterialData(matl, rmats, state)

    ssig0 = np.zeros((3, 3))
    ssig1 = np.diag([300.0, -150.0, -150.0])

    csig0 = StressData.cstress_from_sample(ssig0, rmats)
    csig1 = StressData.cstress_from_sample(ssig1, rmats)
    sdata = StressData(csig0, csig1, 2.0)

    # Test Euler
    euler_solver = Euler(mdata, sdata)
    e_state = euler_solver.run(nsteps=50, time_history=False)
    assert e_state.shape == (numpts, matl.num_statevar)

    t, y_hist = euler_solver.run(nsteps=50, time_history=True)
    assert len(t) == 51
    assert y_hist.shape == (numpts, 51, matl.num_statevar)

    # Test Block
    block_solver = Block(mdata, sdata)
    b_state = block_solver.run()
    assert b_state.shape == (numpts, matl.num_statevar)
