"""Unit tests for single crystal elasticity"""
import numpy as np

from polycrystal.elasticity import single_crystal


TOL = 1e-14


def maxdiff(a, b):
    return np.max(np.abs(a - b))


class TestSingleCrystal:

    ID_6X6 = np.identity(6)

    def test_identity(self):

        # From K, G
        sx = single_crystal.SingleCrystal.from_K_G(1/3, 1/2)
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

        # From E, nu
        sx = single_crystal.SingleCrystal.from_E_nu(1, 0)
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

        # Isotropic: c11, c12
        sx = single_crystal.SingleCrystal(
            'isotropic', [1.0,  0.0]
        )
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

        # Cubic
        sx = single_crystal.SingleCrystal(
            'cubic', [1.0,  0.0,  0.5]
        )
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

        # Hexagonal
        sx = single_crystal.SingleCrystal(
            'hexagonal', [1.0,  0.0,  0.0,  1.0,  0.5]
        )
        assert maxdiff(sx.stiffness, self.ID_6X6) < TOL

"""
class TestUtilities(unittest.TestCase):
    def test_matrix_rep(self):
        a6 = np.array([2.0, 3.1, 5.2, 7.3, 11.4, 13.5])
        m3x3 = sx_elas.to_3x3(a6)
        b6 = sx_elas.to_6vec(m3x3)
        for i in range(len(a6)):
            self.assertEqual(a6[i], b6[i])

    @unittest.skip("under development")
    def test_rotation(self):
        i3 = np.identity(3)
        R3 = np.vstack((i3[2], i3[0], i3[1]))
        LR = sx_elas.rotation_operator(R3)

        M6 = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # print np.dot(LR, M6)
"""
