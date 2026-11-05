"""Small analytic and integration checks; no research archives are required."""

import unittest
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit

from chi_tools.constants import s0, sx, sy, sz
from chi_tools.models import RhombohedralMultilayer
from chi_tools.calculate_chi import calculate_chi_ph, calculate_chi_pp
from chi_tools.vertex import tensor, ph4_t2m, ph_t2m, ph_m2t, pp_t2m, pp_m2t
from chi_tools.phase_diagram import classify_instability
from chi_tools.hartree_fock import hartree_fock, calculate_hf_band


def small_model(twist=False):
    model = RhombohedralMultilayer(nlayer=2 if twist else 3, V=20,
                                   theta=1.0 if twist else 0, twist=twist, Qcut=1)
    model.init_mesh(N=2 if twist else 200)
    model.calculate_bandstructure()
    model.calculate_mu(occ=0, T=4.2)
    return model


class NumericalChecks(unittest.TestCase):
    def test_pauli_algebra(self):
        assert_allclose(sx @ sy, 1j * sz)
        for sigma in (sx, sy, sz):
            assert_array_equal(sigma @ sigma, s0)

    def test_twisted_and_untwisted_hermiticity_and_time_reversal(self):
        for twist in (False, True):
            model = small_model(twist)
            k = np.array([0.013, -0.007])
            h = model.hamiltonian(k, 1)
            assert_allclose(h, h.conj().T, atol=1e-14)
            # The plane-wave q basis reverses as well as k under time reversal.
            reverse_q = np.arange(model.Ndim).reshape(model.nlayer, model.Nqvec, 2)[:, ::-1].ravel()
            hm = model.hamiltonian(-k, -1)[np.ix_(reverse_q, reverse_q)]
            assert_allclose(h, hm.conj(), atol=1e-14)

    def test_density_and_noninteracting_hf(self):
        model = small_model()
        occ = 0.1
        mu = model.calculate_mu(occ=occ, T=4.2)
        density = 2 * model.factor * np.sum(expit(-model.beta * (model.e - mu)) - 0.5)
        self.assertAlmostEqual(density, occ, places=8)
        hartree_fock(model, tensor(0), T=4.2, occ=occ,
                              sigma_seed=np.zeros((4, 4)), verbose=False)
        calculate_hf_band(model)
        self.assertAlmostEqual(model.mu_hf, mu, places=10)
        assert_allclose(model.sigma, 0, atol=1e-14)
        expected = np.sort(np.concatenate([model.e[0], model.e[1]] * 2, axis=1), axis=1)
        assert_allclose(model.e_hf, expected, atol=1e-13)

    def test_hf_ivc_bands_against_full_orbital_hamiltonian(self):
        model = small_model()
        sigma = np.kron(np.eye(2), 0.001 * sx)
        model.sigma = sigma
        calculate_hf_band(model)
        got = model.e_hf
        expected = []
        for k in range(model.nk):
            h = np.kron(sigma, np.eye(model.Ndim)).astype(complex)
            for a, valley in enumerate((0, 1, 0, 1)):
                block = slice(a * model.Ndim, (a + 1) * model.Ndim)
                h[block, block] += model.h[valley, k]
            expected.append(np.linalg.eigvalsh(h))
        assert_allclose(got, expected, atol=1e-13)

    def test_degenerate_bubble_limits_and_custom_model(self):
        # One flat band per valley exactly at mu: PH = beta/4, PP = -beta/4.
        model = SimpleNamespace(e=np.zeros((2, 1, 1)), u=np.ones((2, 1, 1, 1)),
                                e_inv=np.zeros((2, 1, 1)), u_inv=np.ones((2, 1, 1, 1)),
                                mu=0.0, beta=8.0, N=1)
        ph = calculate_chi_ph(model)
        pp = calculate_chi_pp(model)
        expected = np.zeros((2, 2, 2, 2))
        for a in range(2):
            for b in range(2):
                expected[a, b, b, a] = 2.0
        assert_array_equal(ph, expected)
        assert_array_equal(pp, -expected)

    def test_pp_uses_explicit_inverse_bands(self):
        model = small_model(True)
        model.calculate_inverse_bandstructure()
        result = calculate_chi_pp(model)
        self.assertEqual(model.e_inv.shape, model.e.shape)
        self.assertEqual(model.u_inv.shape, model.u.shape)
        self.assertTrue(np.isfinite(result).all())
        for valley, tau in enumerate((-1, 1)):
            for ik, k in enumerate(model.k.T):
                expected = np.linalg.eigvalsh(model.hamiltonian(-k, tau))
                assert_allclose(model.e_inv[valley, ik], expected, atol=1e-13)

    def test_vertex_permutations_and_zero_bubble_limit(self):
        model = RhombohedralMultilayer(twist=False)
        U4 = tensor(1.2, V1=0.7, J=-0.2)
        assert_array_equal(ph_m2t(ph_t2m(U4)), U4)
        assert_array_equal(pp_m2t(pp_t2m(U4)), U4)
        bare = model.split_vertex(U4, vertex='bare')
        zero = np.zeros((2, 2, 2, 2))
        model.chi_ph = model.chi_pp = zero
        dressed = model.split_vertex(U4, vertex='dressed')
        for channel in bare:
            assert_allclose(dressed[channel], bare[channel], atol=1e-14)
        del model.chi_pp
        with self.assertRaises(AttributeError):
            model.split_vertex(U4, vertex='dressed')
        for channel, value in model.split_vertex(U4, vertex='bare').items():
            assert_allclose(value, bare[channel], atol=1e-14)
        with self.assertRaises(ValueError):
            model.split_vertex(U4, vertex='unknown')

    def test_phase_thresholds_broadcasting(self):
        chi = np.zeros((2, 2, 2, 2))
        for a in range(2):
            for b in range(2):
                chi[a, b, b, a] = 1
        channels = {key: np.zeros((2, 4, 4)) for key in ('Vd', 'Vm', 'Vs', 'Vt')}
        channels['Vd'][:, 1, 1] = [0.9, 1.1]
        model = RhombohedralMultilayer(twist=False)
        model.chi_ph = chi
        assert_array_equal(classify_instability(model.kernel(channels)), [0, 2])

    def test_batched_vertices_match_separate_points(self):
        model = RhombohedralMultilayer(twist=False)
        U4 = tensor(0.4, J=-0.1)
        ph = np.zeros((3, 2, 2, 2, 2))
        for a in range(2):
            for b in range(2):
                ph[:, a, b, b, a] = [0.1, 0.2, 0.3]
        bare = model.split_vertex(U4)
        model.chi_ph, model.chi_pp = ph, -ph
        batch = model.split_vertex(U4, vertex='dressed')
        for key, value in model.split_vertex(U4, vertex='bare').items():
            assert_allclose(value, bare[key], atol=1e-14)
        for i in range(3):
            model.chi_ph, model.chi_pp = ph[i], -ph[i]
            point = model.split_vertex(U4, vertex='dressed')
            for key in batch:
                assert_allclose(batch[key][i], point[key], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
