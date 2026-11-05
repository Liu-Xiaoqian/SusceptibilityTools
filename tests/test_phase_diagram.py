"""Analytic checks of channel selection, labels, thresholds, and broadcasting."""

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from chi_tools.models import RhombohedralMultilayer
from chi_tools.phase_diagram import classify_instability
from chi_tools.vertex import tensor, ph4_t2m, ph4_m2t, pp4_t2m, pp4_m2t


def unit_bubble():
    """Return a PH bubble whose matrix representation is the identity."""
    chi = np.zeros((2, 2, 2, 2))
    for a in range(2):
        for b in range(2):
            chi[a, b, b, a] = 1
    return chi


class ClassificationChecks(unittest.TestCase):
    def setUp(self):
        self.model = RhombohedralMultilayer(twist=False)
        self.model.chi_ph = unit_bubble()

    def test_dressed_modes_and_bare_grouping(self):
        self.model.chi_pp = -self.model.chi_ph
        for channel, (key, coefficient) in enumerate(zip(
                ('Vd', 'Vm', 'Vs', 'Vt'), (1., -1., .5, -.5))):
            for off_diagonal in (0, 1):
                vertices = {key: np.zeros((4, 4)) for key in ('Vd', 'Vm', 'Vs', 'Vt')}
                vertices[key][off_diagonal, off_diagonal] = 1.2 / coefficient
                expected = 1 + 2 * channel + off_diagonal
                self.assertEqual(classify_instability(self.model.kernel(vertices, channels='dressed'), channels='dressed'), expected)
                self.assertEqual(classify_instability(self.model.kernel(vertices)),
                                 1 + off_diagonal if channel < 2 else 0)

    def test_bare_broadcasting_without_pairing_inputs(self):
        vertices = {'Vd': np.zeros((2, 1, 4, 4)), 'Vm': np.zeros((1, 3, 4, 4))}
        vertices['Vd'][:, 0, 0, 0] = [.5, 1.5]
        vertices['Vm'][0, 1, 0, 0] = -1.2
        vertices['Vm'][0, 2, 1, 1] = -1.4
        assert_array_equal(classify_instability(self.model.kernel(vertices)), [[0, 1, 2], [1, 1, 1]])

    def test_threshold_and_tie_priority(self):
        self.model.chi_pp = -self.model.chi_ph
        vertices = {key: np.zeros((4, 4)) for key in ('Vd', 'Vm', 'Vs', 'Vt')}
        vertices['Vd'][0, 0] = 1 - 2e-12
        self.assertEqual(classify_instability(self.model.kernel(vertices)), 0)
        vertices['Vd'][0, 0] = 1 - .5e-12
        self.assertEqual(classify_instability(self.model.kernel(vertices)), 1)
        vertices['Vd'][0, 0] = 1.2
        vertices['Vm'][1, 1] = -1.2 - .5e-12
        self.assertEqual(classify_instability(self.model.kernel(vertices, channels='dressed'), channels='dressed'), 1)
        vertices['Vm'][1, 1] = -1.2 - 2e-12
        self.assertEqual(classify_instability(self.model.kernel(vertices, channels='dressed'), channels='dressed'), 4)

    def test_channel_selection_errors(self):
        vertices = {'Vd': np.eye(4), 'Vm': np.eye(4)}
        with self.assertRaises(ValueError):
            self.model.kernel(vertices, channels='unknown')
        with self.assertRaises(AttributeError):
            classify_instability(self.model.kernel(vertices, channels='dressed'), channels='dressed')
        with self.assertRaises(ValueError):
            classify_instability({'Kd': np.eye(4), 'Km': np.eye(4)}, channels='unknown')

    def test_classifier_accepts_independent_kernels(self):
        kernels = {'Kd': np.diag([.1, 1.2, .1, .1]), 'Km': np.zeros((4, 4))}
        self.assertEqual(classify_instability(kernels), 2)

    def test_explicit_kernel_signs(self):
        self.model.chi_pp = -self.model.chi_ph
        vertices = {key: np.arange(16).reshape(4, 4) for key in ('Vd', 'Vm', 'Vs', 'Vt')}
        kernels = self.model.kernel(vertices, channels='dressed')
        assert_allclose(kernels['Kd'], vertices['Vd'])
        assert_allclose(kernels['Km'], -vertices['Vm'])
        assert_allclose(kernels['Ks'], .5 * vertices['Vs'])
        assert_allclose(kernels['Kt'], -.5 * vertices['Vt'])

    def test_single_point_and_multidimensional_grids(self):
        rng = np.random.default_rng(531)
        U4 = tensor(.4, V1=.3, J=-.1)
        point = RhombohedralMultilayer(twist=False)
        for shape in ((), (3,), (2, 3), (2, 1, 3)):
            ph = np.zeros(shape + (2, 2, 2, 2))
            pp = np.zeros_like(ph)
            for a in range(2):
                for b in range(2):
                    ph[..., a, b, b, a] = rng.uniform(.1, .3, shape)
                    pp[..., a, b, b, a] = -rng.uniform(.1, .3, shape)
            self.model.chi_ph, self.model.chi_pp = ph, pp
            for mode in ('bare', 'dressed'):
                vertices = self.model.split_vertex(U4, vertex=mode)
                kernels = self.model.kernel(vertices, channels=mode)
                phases = classify_instability(kernels, channels=mode)
                self.assertEqual(phases.shape, shape)
                for ind in np.ndindex(shape):
                    point.chi_ph, point.chi_pp = ph[ind], pp[ind]
                    vpoint = point.split_vertex(U4, vertex=mode)
                    kpoint = point.kernel(vpoint, channels=mode)
                    for key in vertices:
                        actual = vertices[key] if mode == 'bare' else vertices[key][ind]
                        assert_allclose(actual, vpoint[key], atol=1e-14)
                    for key in kernels:
                        assert_allclose(kernels[key][ind], kpoint[key], atol=1e-14)
                    self.assertEqual(phases[ind], classify_instability(kpoint, channels=mode))

    def test_ph_pp_grids_broadcast_independently(self):
        U4 = tensor(.4, V1=.3, J=-.1)
        ph_scale = np.array([.1, .2]).reshape(2, 1, 1)
        pp_scale = -np.linspace(.1, .3, 6).reshape(1, 3, 2)
        self.model.chi_ph = ph_scale[..., None, None, None, None] * unit_bubble()
        self.model.chi_pp = pp_scale[..., None, None, None, None] * unit_bubble()
        vertices = self.model.split_vertex(U4, vertex='dressed')
        kernels = self.model.kernel(vertices, channels='dressed')
        phase = classify_instability(kernels, channels='dressed')
        self.assertEqual(phase.shape, (2, 3, 2))
        point = RhombohedralMultilayer(twist=False)
        for i, j, k in np.ndindex(phase.shape):
            point.chi_ph = self.model.chi_ph[i, 0, 0]
            point.chi_pp = self.model.chi_pp[0, j, k]
            vpoint = point.split_vertex(U4, vertex='dressed')
            kpoint = point.kernel(vpoint, channels='dressed')
            for key in kernels:
                assert_allclose(kernels[key][i, j, k], kpoint[key], atol=1e-14)
            self.assertEqual(phase[i, j, k], classify_instability(kpoint, channels='dressed'))

    def test_response_matrices_match_four_index_tensors(self):
        rng = np.random.default_rng(721)
        U4 = tensor(.4, V1=.3, J=-.1)
        matrix_model = RhombohedralMultilayer(twist=False)
        for shape in ((), (2, 3), (41, 61), (2, 1, 3)):
            ph = rng.normal(scale=.01, size=shape + (2, 2, 2, 2))
            pp = rng.normal(scale=.01, size=shape + (2, 2, 2, 2))
            self.model.chi_ph, self.model.chi_pp = ph, pp
            matrix_model.chi_ph = ph4_t2m(ph)
            matrix_model.chi_pp = pp4_t2m(pp)
            assert_array_equal(ph4_m2t(matrix_model.chi_ph), ph)
            assert_array_equal(pp4_m2t(matrix_model.chi_pp), pp)
            for mode in ('bare', 'dressed'):
                expected_v = self.model.split_vertex(U4, vertex=mode)
                actual_v = matrix_model.split_vertex(U4, vertex=mode)
                expected_k = self.model.kernel(expected_v, channels=mode)
                actual_k = matrix_model.kernel(actual_v, channels=mode)
                for key in expected_v:
                    assert_allclose(actual_v[key], expected_v[key], atol=1e-14)
                for key in expected_k:
                    assert_allclose(actual_k[key], expected_k[key], atol=1e-14)
                assert_array_equal(classify_instability(actual_k, channels=mode),
                                   classify_instability(expected_k, channels=mode))

    def test_invalid_susceptibility_shape(self):
        for value in (1., np.zeros((3, 3))):
            with self.assertRaises(ValueError):
                ph4_t2m(value)
            with self.assertRaises(ValueError):
                pp4_t2m(value)

    def test_tensor_is_independent_of_model(self):
        original = tensor(1.2, V1=.7, J=-.2)
        self.assertEqual(original.shape, (4, 4, 4, 4))
        self.assertEqual(original[0, 2, 2, 0], 1.2)
        assert_array_equal(tensor(0), np.zeros_like(original))
        self.assertTrue(np.any(original != 0))


if __name__ == '__main__':
    unittest.main()
