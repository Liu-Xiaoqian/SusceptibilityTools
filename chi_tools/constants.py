"""Unit conversions and Pauli matrices (energies in eV)."""

import numpy as np

# Boltzmann constant in eV/K; beta = 1 / (kB * T).
kB = 8.617328149740566e-5

s0 = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
pauli = [s0, sx, sy, sz]
