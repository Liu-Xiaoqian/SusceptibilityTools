"""Check windowed sums against explicit band pairs and the T=0 filling."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit

from chi_tools.models import RhombohedralMultilayer
from chi_tools.calculate_chi import calculate_chi, calculate_chi_ph, calculate_chi_pp


def synthetic_model():
    model = RhombohedralMultilayer(nlayer=2, twist=False)
    model.N, model.nk = 2, 3
    model.e = np.array([
        [[-.6, -.25, .15, .7], [-.7, -.4, .2, .5], [-.5, -.2, .35, .6]],
        [[-.5, -.3, .2, .8], [-.6, -.15, .3, .7], [-.8, -.35, .25, .5]],
    ]) + .07
    model.e_inv = model.e[:, ::-1].copy() + .03
    rng = np.random.default_rng(12)
    model.u = np.linalg.qr(rng.normal(size=(2, 3, 4, 4)) +
                           1j * rng.normal(size=(2, 3, 4, 4)))[0]
    model.u_inv = np.linalg.qr(rng.normal(size=(2, 3, 4, 4)) +
                               1j * rng.normal(size=(2, 3, 4, 4)))[0]
    model.mu, model.beta, model.factor = .12, 20., .1
    return model


def explicit_bubbles(model):
    ph = np.zeros((2, 2, 2, 2))
    pp = ph.copy()
    mask = np.ones_like(model.e, dtype=bool)
    inverse = np.ones_like(model.e_inv, dtype=bool)
    bounds = getattr(model, 'energy_bounds', None)
    if bounds is not None:
        mask = (model.e >= bounds[0]) & (model.e <= bounds[1])
        inverse = (model.e_inv >= bounds[0]) & (model.e_inv <= bounds[1])
    for a in range(2):
        for b in range(2):
            for k in range(model.nk):
                for n in range(model.e.shape[-1]):
                    if not mask[a, k, n]:
                        continue
                    ea = model.e[a, k, n] - model.mu
                    fa = expit(-model.beta * ea)
                    ua = model.u[a, k, :, n]
                    for m in range(model.e.shape[-1]):
                        if mask[b, k, m]:
                            eb = model.e[b, k, m] - model.mu
                            fb = expit(-model.beta * eb)
                            factor = ((fa-fb)/(eb-ea) if abs(eb-ea) >= 1e-10
                                      else model.beta*fa*(1-fa))
                            ph[a, b, b, a] += factor * abs(np.vdot(ua, model.u[b, k, :, m]))**2
                        if inverse[b, k, m]:
                            eb = model.e_inv[b, k, m] - model.mu
                            fb = expit(-model.beta * eb)
                            factor = ((1-fa-fb)/(-ea-eb) if abs(ea+eb) >= 1e-10
                                      else -model.beta/2*(fa*(1-fa)+fb*(1-fb)))
                            pp[a, b, b, a] += factor * abs(ua @ model.u_inv[b, k, :, m])**2
    return ph/model.N**2, pp/model.N**2


class WindowChecks(unittest.TestCase):
    def test_neutrality_uses_global_half_filling_and_inclusive_bounds(self):
        model = synthetic_model()
        model.e = np.array([[[0., 10.], [1., 2.]]] * 2)
        model.e_inv = model.e.copy()
        model.u = np.broadcast_to(np.eye(2), (2, 2, 2, 2)).copy()
        model.u_inv = model.u.copy()
        assert_array_equal(model.energy_window(cutoff=500), [1., 2.])
        self.assertEqual(model.mu0, 1.5)
        self.assertEqual(np.count_nonzero(model.e < model.mu0), model.e.size // 2)
        assert_array_equal(model.band_mask, (model.e == 1.) | (model.e == 2.))
        model.mu = 123.
        assert_array_equal(model.energy_window(cutoff=500), [1., 2.])
        assert_allclose(model.energy_window(), [model.mu0 - .6, model.mu0 + .6])
        self.assertEqual(model.energy_cutoff, 600)

    def test_full_and_windowed_sums(self):
        model = synthetic_model()
        expected = explicit_bubbles(model)
        assert_allclose(calculate_chi(model), expected, atol=1e-13)
        assert_allclose((calculate_chi_ph(model), calculate_chi_pp(model)),
                        expected, atol=1e-13)
        for cutoff in (300, 1, 0, 600):
            model.energy_window(cutoff)
            expected = explicit_bubbles(model)
            combined = calculate_chi(model)
            separate = calculate_chi_ph(model), calculate_chi_pp(model)
            for i in range(2):
                assert_allclose(combined[i], expected[i], atol=1e-13)
                assert_allclose(separate[i], expected[i], atol=1e-13)

    def test_full_arrays_and_density_preserved(self):
        model = synthetic_model()
        arrays = [model.e.copy(), model.u.copy(), model.e_inv.copy(), model.u_inv.copy()]
        mu = model.calculate_mu(T=50., occ=.1)
        model.energy_window(300)
        calculate_chi(model)
        self.assertEqual(mu, model.calculate_mu(T=50., occ=.1))
        for before, after in zip(arrays, (model.e, model.u, model.e_inv, model.u_inv)):
            assert_array_equal(before, after)

    def test_combined_reuses_occupations(self):
        model = synthetic_model()
        model.energy_window(300)
        with patch('chi_tools.calculate_chi.expit', wraps=expit) as fermi:
            calculate_chi_ph(model)
            calculate_chi_pp(model)
            separate_calls = fermi.call_count
            fermi.reset_mock()
            calculate_chi(model)
            self.assertEqual(fermi.call_count * 3, separate_calls * 2)

    def test_ph_does_not_require_inverse_bands(self):
        model = synthetic_model()
        del model.e_inv, model.u_inv
        model.energy_window(300)
        with patch.object(model, 'calculate_inverse_bandstructure', side_effect=AssertionError):
            calculate_chi_ph(model)

    def test_explicit_inverse_preparation_and_window_refresh(self):
        model = RhombohedralMultilayer(nlayer=2, twist=True, theta=1., Qcut=1)
        model.init_mesh(N=2)
        model.calculate_bandstructure()
        model.calculate_mu(T=4.2)
        model.energy_window(300)
        with self.assertRaises(ValueError):
            calculate_chi(model)
        model.calculate_inverse_bandstructure()
        # Newly calculated inverse bands must be selected explicitly too.
        with self.assertRaises(ValueError):
            calculate_chi(model)
        model.energy_window(300)
        with patch.object(model, 'calculate_inverse_bandstructure', side_effect=AssertionError):
            calculate_chi(model)
        model.init_mesh(N=3)
        # Band methods must not call the window method as a side effect.
        with patch.object(model, 'energy_window', side_effect=AssertionError):
            model.calculate_bandstructure()
            model.calculate_inverse_bandstructure()
        model.energy_window(300)
        self.assertEqual(model.band_mask.shape, model.e.shape)
        model.calculate_mu(T=4.2)
        calculate_chi(model)
        self.assertEqual(model.band_mask_inv.shape, model.e_inv.shape)

    def test_prepared_states_have_variable_sizes_and_full_orbital_vectors(self):
        model = synthetic_model()
        bounds = model.energy_window(300)
        sizes = [len(ek) for valley in model.e_active for ek in valley]
        self.assertGreater(len(set(sizes)), 1)
        for e, u, ea, ua in ((model.e, model.u, model.e_active, model.u_active),
                              (model.e_inv, model.u_inv, model.e_active_inv, model.u_active_inv)):
            for v in range(len(e)):
                for k in range(model.nk):
                    keep = (e[v, k] >= bounds[0]) & (e[v, k] <= bounds[1])
                    assert_array_equal(ea[v][k], e[v, k, keep])
                    assert_array_equal(ua[v][k], u[v, k][:, keep])

    def test_chi_uses_prepared_states_without_reading_masks_or_bounds(self):
        model = synthetic_model()
        model.energy_window(300)
        expected = explicit_bubbles(model)
        del model.band_mask, model.band_mask_inv, model.energy_bounds
        with patch.object(model, 'energy_window', side_effect=AssertionError):
            assert_allclose(calculate_chi(model), expected, atol=1e-13)

    def test_invalid_cutoffs(self):
        model = synthetic_model()
        for cutoff in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                model.energy_window(cutoff)


if __name__ == '__main__':
    unittest.main()
