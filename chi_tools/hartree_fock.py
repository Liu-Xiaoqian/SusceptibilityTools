"""Self-consistent Hartree-Fock calculations for Stoner and IVC order."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .constants import kB


def sigma_from_density(U4, rho):
    """Return the Hermitian self-energy Sigma[b,a] = -Gamma[a,b,c,d] rho[c,d]."""
    sigma = -np.einsum("abcd,cd->ba", U4, rho)
    return (sigma + sigma.conj().T) / 2


def _hamiltonian(model):
    """Reconstruct the full spin-valley Hamiltonian from model.e and model.u."""
    n = model.e.shape[-1]
    bare = (model.u * model.e[..., None, :]) @ model.u.conj().swapaxes(-1, -2)
    h = np.zeros((model.nk, 4 * n, 4 * n), dtype=complex)
    # Flavor order: (up K-, up K+, down K-, down K+).
    for a, valley in enumerate((0, 1, 0, 1)):
        block = slice(a * n, (a + 1) * n)
        h[:, block, block] = bare[valley]
    return h


def calculate_hf_band(model):
    """Store full-band HF energies in model.e_hf using model.sigma."""
    h = _hamiltonian(model)
    model.e_hf = np.linalg.eigvalsh(h + np.kron(model.sigma, np.eye(model.e.shape[-1])))


def hartree_fock(model, U4, T, occ, sigma_seed, mixing=0.2, max_iter=200,
                 tol=1e-8, verbose=True):
    """Solve full-band HF from a self-energy seed in eV and store results on model."""
    beta = 1 / (kB * T)
    h = _hamiltonian(model)
    orbital_identity = np.eye(model.e.shape[-1])
    sigma = np.array(sigma_seed, dtype=complex, copy=True)
    sigma -= np.trace(sigma) * np.eye(4) / 4

    for iteration in range(max_iter):
        energies, states = np.linalg.eigh(h + np.kron(sigma, orbital_identity))

        def density_error(mu):
            """Return the excess density relative to the requested filling."""
            return model.factor * np.sum(expit(-beta * (energies - mu)) - 0.5) - occ

        mu = brentq(density_error, energies.min() - 1, energies.max() + 1)
        occupations = expit(-beta * (energies - mu))
        states = states.reshape(model.nk, 4, model.e.shape[-1], energies.shape[-1])
        rho = np.einsum("kn,kain,kbin->ab", occupations - 0.5, states, states.conj()) / model.N**2
        sigma_new = sigma_from_density(U4, rho)
        sigma_new -= np.trace(sigma_new) * np.eye(4) / 4
        error = np.max(np.abs(sigma_new - sigma))

        # Optional progress output: iteration, mu (meV), and self-energy error (eV).
        if verbose:
            print(f"{iteration:3d}  mu={mu*1e3: .6f} meV  error={error:.3e}")
        # Store the self-energy used to calculate the final density and mu.
        if error < tol or iteration == max_iter - 1:
            break
        sigma = (1 - mixing) * sigma + mixing * sigma_new

    model.sigma = sigma
    model.rho = rho
    model.mu_hf = mu
