"""Checks of full-band HF, explicit seeds, filling, and intervalley mixing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import expit
from scipy.optimize import brentq

from chi_tools import hartree_fock as hf_module
from chi_tools.constants import kB, s0, sx, sy, sz
from chi_tools.hartree_fock import hartree_fock, calculate_hf_band, sigma_from_density
from chi_tools.models import RhombohedralMultilayer
from chi_tools.phase_diagram import kernel
from chi_tools.vertex import tensor, split_vertex


def model_fixture(window=False):
    """Return a small two-valley model with complex orbital mixing."""
    e = np.array([[[-.3, .01, .4], [-.04, .03, .2], [.1, .3, .5]],
                  [[-.25, -.02, .35], [-.16, .015, .22], [.12, .32, .55]]])
    rng = np.random.default_rng(211)
    u = np.empty((2, 3, 3, 3), dtype=complex)
    for v in range(2):
        for k in range(3):
            u[v, k], _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    model = SimpleNamespace(e=e, u=u, nk=3, N=2, factor=.1)
    if window:
        RhombohedralMultilayer.energy_window(model, cutoff=80)
    return model


def mixed_seed():
    """Return a traceless self-energy seed with spin and intervalley components."""
    sigma = np.kron(sz, s0) + .2 * np.kron(s0, sx) + .1 * np.kron(sx, sy)
    return 1e-3 * sigma / np.linalg.norm(sigma)


class HartreeFockChecks(unittest.TestCase):
    def test_bands_against_orbital_hamiltonian(self):
        sigma = .001 * (np.kron(sz, s0) + np.kron(s0, sy) + .2 * np.kron(sx, sx))
        for window in (False, True):
            model = model_fixture(window)
            model.sigma = sigma
            calculate_hf_band(model)
            got = model.e_hf
            for k in range(model.nk):
                h = np.kron(sigma, np.eye(3)).astype(complex)
                for a, v in enumerate((0, 1, 0, 1)):
                    block = slice(3 * a, 3 * a + 3)
                    h[block, block] += model.u[v, k] @ np.diag(model.e[v, k]) @ model.u[v, k].conj().T
                assert_allclose(got[k], np.linalg.eigvalsh(h), atol=1e-14)

    def test_bands_use_eigenvalues_only(self):
        model = model_fixture(True)
        with patch.object(np.linalg, 'eigh', side_effect=AssertionError('Unneeded eigenvectors')):
            model.sigma = np.zeros((4, 4))
            self.assertIsNone(calculate_hf_band(model))
        self.assertEqual(model.e_hf.shape, (3, 12))

    def test_energy_window_does_not_change_hf(self):
        model = model_fixture()
        U4 = tensor(.1, V1=.08, J=-.02)
        args = dict(T=70., occ=.02, sigma_seed=mixed_seed(), max_iter=5, verbose=False)
        hartree_fock(model, U4, **args)
        keys = ('sigma', 'rho', 'mu_hf')
        expected = {key: np.copy(getattr(model, key)) for key in keys}
        calculate_hf_band(model)
        bands = model.e_hf.copy()
        for cutoff in (80, 0):
            RhombohedralMultilayer.energy_window(model, cutoff=cutoff)
            self.assertLess(model.band_mask.sum(), model.e.size)
            # HF does not need selection metadata, even with prepared chi states.
            del model.band_mask
            hartree_fock(model, U4, **args)
            for key in keys:
                assert_allclose(getattr(model, key), expected[key], atol=1e-14)
            calculate_hf_band(model)
            assert_allclose(model.e_hf, bands, atol=1e-14)

    def test_iteration_reuses_hamiltonian_and_preserves_filling(self):
        model = model_fixture(True)
        U4, T, occ = tensor(.1, V1=.08, J=-.02), 70., .02
        with patch.object(hf_module, '_hamiltonian', wraps=hf_module._hamiltonian) as prepare:
            hartree_fock(model, U4, T, occ, mixed_seed(), max_iter=5, tol=1e-30, verbose=False)
            self.assertEqual(prepare.call_count, 1)
        calculate_hf_band(model)
        density = model.factor * np.sum(expit(-(model.e_hf - model.mu_hf) / (kB * T)) - .5)
        self.assertAlmostEqual(density, occ, places=10)
        sigma_new = sigma_from_density(U4, model.rho)
        sigma_new -= np.trace(sigma_new) * np.eye(4) / 4
        self.assertGreater(np.max(np.abs(sigma_new - model.sigma)), 1e-30)

    def test_noninteracting_limit_uses_full_spectrum(self):
        model = model_fixture(True)
        hartree_fock(model, tensor(0), 70., .02, np.zeros((4, 4)), verbose=False)
        calculate_hf_band(model)
        expected = np.sort(model.e[[0, 1, 0, 1]].transpose(1, 0, 2).reshape(3, 12), axis=-1)
        assert_allclose(model.e_hf, expected, atol=1e-14)
        density = 2 * model.factor * np.sum(expit(-(model.e - model.mu_hf) / (kB * 70.)) - .5)
        self.assertAlmostEqual(density, .02, places=10)

    def test_diagonal_hf_matches_energy_shift_loop(self):
        model = model_fixture()
        U4, T, occ, mixing = tensor(.1, V1=.08, J=-.02), 70., .02, .3
        seed = 1e-3 * np.kron(sz, sz)
        sigma = seed.copy()
        for iteration in range(5):
            energies = model.e[[0, 1, 0, 1]].transpose(1, 0, 2) + np.diag(sigma)[None, :, None]
            mu = brentq(lambda mu: model.factor * np.sum(expit(-(energies - mu) / (kB * T)) - .5) - occ,
                        energies.min() - 1, energies.max() + 1)
            rho = np.diag(np.sum(expit(-(energies - mu) / (kB * T)) - .5, axis=(0, 2))) / model.N**2
            updated = sigma_from_density(U4, rho)
            updated -= np.trace(updated) * np.eye(4) / 4
            if iteration < 4:
                sigma = (1 - mixing) * sigma + mixing * updated
        hartree_fock(model, U4, T, occ, seed, mixing=mixing,
                             max_iter=5, tol=1e-30, verbose=False)
        calculate_hf_band(model)
        assert_allclose(model.sigma, sigma, atol=1e-14)
        assert_allclose(model.rho, rho, atol=1e-14)
        assert_allclose(model.e_hf, np.sort(energies.reshape(3, 12), axis=-1), atol=1e-14)
        self.assertAlmostEqual(model.mu_hf, mu, places=12)
        assert_allclose(seed, 1e-3 * np.kron(sz, sz), atol=0)

    def test_ivc_seed_produces_coherent_density(self):
        T, amplitude = 100., .001
        model = SimpleNamespace(e=np.zeros((2, 1, 1)), u=np.ones((2, 1, 1, 1)),
                                N=1, nk=1, factor=1.)
        for spin in (s0, sz):
            mode = np.kron(spin, sy)
            hartree_fock(model, tensor(.02), T, 0., amplitude * mode,
                                 max_iter=1, verbose=False)
            expected = -.5 * np.tanh(amplitude / (2 * kB * T)) * mode
            assert_allclose(model.rho, expected, atol=1e-14)
            assert_allclose(model.sigma, amplitude * mode, atol=1e-14)

    def test_explicit_density_modes_match_linearized_response(self):
        T = 100.
        model = SimpleNamespace(e=np.zeros((2, 1, 1)), u=np.ones((2, 1, 1, 1)),
                                N=1, nk=1, factor=1., chi_ph=np.eye(4) / (4 * kB * T))
        U4 = tensor(.02, V1=.03, J=-.005)
        kernels = kernel(model, split_vertex(model, U4))
        for key, spin in (('Kd', s0), ('Km', sz)):
            for valley_mode in (sz, sx, sy):
                rho_seed = 1e-7 * np.kron(spin, valley_mode)
                sigma_seed = sigma_from_density(U4, rho_seed)
                hartree_fock(model, U4, T, 0., sigma_seed, max_iter=1, verbose=False)
                response = (kernels[key] @ valley_mode.ravel()).reshape(2, 2)
                assert_allclose(model.rho, 1e-7 * np.kron(spin, response), rtol=1e-7, atol=1e-14)

    def test_minimal_state_preserves_bare_arrays_and_mu(self):
        model = model_fixture()
        model.mu = .012
        before = set(vars(model))
        e, u = model.e.copy(), model.u.copy()
        result = hartree_fock(model, tensor(.1), 70., .02, mixed_seed(), max_iter=2, verbose=False)
        self.assertIsNone(result)
        self.assertEqual(set(vars(model)) - before,
                         {'sigma', 'rho', 'mu_hf'})
        assert_allclose(model.e, e, atol=0)
        assert_allclose(model.u, u, atol=0)
        self.assertEqual(model.mu, .012)
        calculate_hf_band(model)
        self.assertEqual(model.e_hf.shape, (3, 12))

    def test_restart_needs_only_sigma(self):
        model, uninterrupted = model_fixture(), model_fixture()
        U4 = tensor(.1, V1=.08, J=-.02)
        args = dict(T=70., occ=.02, mixing=.3, tol=1e-30, verbose=False)
        hartree_fock(uninterrupted, U4, sigma_seed=mixed_seed(), max_iter=8, **args)
        hartree_fock(model, U4, sigma_seed=mixed_seed(), max_iter=5, **args)
        # Restart evaluates the stored self-energy again before the next update.
        hartree_fock(model, U4, sigma_seed=model.sigma, max_iter=4, **args)
        for name in ('sigma', 'rho', 'mu_hf'):
            assert_allclose(getattr(model, name), getattr(uninterrupted, name), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
