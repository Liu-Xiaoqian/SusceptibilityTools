"""Self-consistent Hartree-Fock calculations for Stoner and IVC order."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from .constants import kB


VALLEY = (0, 1, 0, 1)  # (up K-, up K+, down K-, down K+).


def sigma_from_density(U4, rho):
    """Return the Hermitian self-energy Sigma[b,a] = -Gamma[a,b,c,d] rho[c,d]."""
    sigma = -np.einsum("abcd,cd->ba", U4, rho)
    return (sigma + sigma.conj().T) / 2


def _prepare_basis(model):
    """Prepare band overlaps and frozen states for the selected HF energy window."""
    if hasattr(model, "e_active"):
        e, u, mask = model.e_active, model.u_active, model.band_mask
    else:
        e, u, mask = model.e, model.u, np.ones_like(model.e, dtype=bool)
    inactive = [model.e[v][~mask[v]] for v in range(2)]
    prepared = []
    for k in range(model.nk):
        basis = [u[v][k] for v in range(2)]
        bare = np.concatenate([e[v][k] for v in VALLEY])
        edges = np.r_[0, np.cumsum([len(e[v][k]) for v in VALLEY])]
        blocks = [slice(edges[a], edges[a + 1]) for a in range(4)]
        overlaps = [[basis[a].conj().T @ basis[b] for b in range(2)] for a in range(2)]
        frozen = np.concatenate([model.e[v, k, ~mask[v, k]] for v in VALLEY])
        prepared.append((bare, basis, blocks, overlaps, frozen))
    return prepared, inactive


def _hf_matrix(point, sigma):
    """Construct the HF matrix in the selected band basis at one momentum."""
    bare, basis, blocks, overlaps, frozen = point
    h = np.diag(bare).astype(complex)
    for a, va in enumerate(VALLEY):
        for b, vb in enumerate(VALLEY):
            h[blocks[a], blocks[b]] += sigma[a, b] * overlaps[va][vb]
    return h


def _spectrum(point, sigma):
    """Return energies and flavor-density weights without retaining eigenvectors."""
    energy, coeff = np.linalg.eigh(_hf_matrix(point, sigma))
    _, basis, blocks, _, _ = point
    states = np.array([basis[v] @ coeff[blocks[a]] for a, v in enumerate(VALLEY)])
    weights = np.einsum("ain,bin->nab", states, states.conj())
    return energy, weights


def _band_energies(point, sigma):
    """Return sorted selected HF and frozen bare energies at one momentum."""
    return np.sort(np.r_[np.linalg.eigvalsh(_hf_matrix(point, sigma)), point[4]])


def calculate_hf_band(model, workers=1):
    """Store HF and frozen bare energies, parallelizing momenta across workers."""
    prepared, _ = _prepare_basis(model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        model.e_hf = np.array(list(pool.map(partial(_band_energies, sigma=model.sigma), prepared)))


def hartree_fock(model, U4, T, occ, sigma_seed, mixing=0.2, max_iter=200,
                 tol=1e-8, verbose=True, workers=1, callback=None, start_iter=0):
    """Solve HF in the selected bands, including frozen bands in density and filling."""
    if not 0 <= start_iter < max_iter:
        raise ValueError("Require 0 <= start_iter < max_iter (total iteration limit).")
    beta = 1 / (kB * T)
    prepared, inactive = _prepare_basis(model)
    frozen = np.concatenate([inactive[v] for v in VALLEY])
    sigma = np.array(sigma_seed, dtype=complex, copy=True)
    sigma -= np.trace(sigma) * np.eye(4) / 4

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for iteration in range(start_iter, max_iter):
            start = perf_counter()
            spectra = list(pool.map(partial(_spectrum, sigma=sigma), prepared))
            energies = np.concatenate([energy for energy, _ in spectra] + [frozen])

            def density_error(mu):
                """Return the excess density relative to the requested filling."""
                return model.factor * np.sum(expit(-beta * (energies - mu)) - 0.5) - occ

            mu = brentq(density_error, energies.min() - 1, energies.max() + 1)
            rho = np.diag([np.sum(expit(-beta * (inactive[v] - mu)) - 0.5)
                           for v in VALLEY]).astype(complex)
            for energy, weights in spectra:
                f = expit(-beta * (energy - mu))
                rho += np.einsum("n,nab->ab", f - 0.5, weights)
            del spectra
            rho /= model.N**2
            sigma_new = sigma_from_density(U4, rho)
            sigma_new -= np.trace(sigma_new) * np.eye(4) / 4
            error = np.max(np.abs(sigma_new - sigma))
            sigma_next = (1 - mixing) * sigma + mixing * sigma_new
            seconds = perf_counter() - start

            if verbose:
                print(f"{iteration+1:3d}  mu={mu*1e3: .6f} meV  error={error:.3e}  time={seconds:.1f}s", flush=True)
            if callback is not None:
                callback(dict(iteration=iteration + 1, sigma=sigma.copy(),
                              sigma_next=sigma_next.copy(), rho=rho.copy(), mu_hf=mu,
                              error=error, converged=error < tol, iteration_seconds=seconds))
            if error < tol or iteration == max_iter - 1:
                break
            sigma = sigma_next

    model.sigma = sigma
    model.rho = rho
    model.mu_hf = mu
