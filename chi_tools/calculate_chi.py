"""Static PH/PP bubbles from prepared states, normalized by N**2 without spin."""

import numpy as np
from scipy.special import expit


def calculate_chi_ph(model, eps=1e-10):
    """Return and store the particle-hole tensor from the prepared states at k."""
    return calculate_chi(model, eps=eps, channel="ph")[0]


def calculate_chi_pp(model, eps=1e-10):
    """Return and store the particle-particle tensor from prepared states at k and -k."""
    return calculate_chi(model, eps=eps, channel="pp")[1]


def calculate_chi(model, eps=1e-10, *, channel="both"):
    """Return (chi_ph, chi_pp), sharing occupations; channel selects both, ph, or pp."""
    if channel not in ("both", "ph", "pp"):
        raise ValueError("channel must be 'both', 'ph', or 'pp'.")
    do_ph, do_pp = channel != "pp", channel != "ph"
    e = getattr(model, "e_active", model.e)
    u = getattr(model, "u_active", model.u)
    if do_pp:
        e_inv = getattr(model, "e_active_inv", getattr(model, "e_inv", None))
        u_inv = getattr(model, "u_active_inv", getattr(model, "u_inv", None))
        if e_inv is None or u_inv is None:
            raise ValueError("Calculate inverse bands, then call energy_window() before PP.")

    nvalley = len(e)
    chi_ph = np.zeros((nvalley,) * 4) if do_ph else None
    chi_pp = np.zeros((nvalley,) * 4) if do_pp else None
    for k in range(len(e[0])):
        energies = [e[v][k] - model.mu for v in range(nvalley)]
        f = [expit(-model.beta * ev) for ev in energies]
        if do_pp:
            inverse = [e_inv[v][k] - model.mu for v in range(nvalley)]
            f_inv = [expit(-model.beta * ev) for ev in inverse]

        for a in range(nvalley):
            ua = u[a][k].T
            ea, fa = energies[a], f[a]
            for b in range(nvalley):
                if do_ph:
                    W = np.abs(ua.conj() @ u[b][k]) ** 2
                    numer = fa[:, None] - f[b][None, :]
                    denom = energies[b][None, :] - ea[:, None]
                    factor = np.empty_like(denom)
                    near = np.abs(denom) < eps
                    np.divide(numer, denom, out=factor, where=~near)
                    ni, _ = np.where(near)
                    factor[near] = model.beta * fa[ni] * (1 - fa[ni])
                    chi_ph[a, b, b, a] += np.sum(factor * W)

                if do_pp:
                    W = np.abs(ua @ u_inv[b][k]) ** 2
                    fb = f_inv[b]
                    numer = 1.0 - fa[:, None] - fb[None, :]
                    denom = -ea[:, None] - inverse[b][None, :]
                    factor = np.empty_like(denom)
                    near = np.abs(denom) < eps
                    np.divide(numer, denom, out=factor, where=~near)
                    ni, nj = np.where(near)
                    factor[near] = -0.5 * model.beta * (
                        fa[ni] * (1 - fa[ni]) + fb[nj] * (1 - fb[nj])
                    )
                    chi_pp[a, b, b, a] += np.sum(factor * W)

    if do_ph:
        chi_ph /= model.N**2
        model.chi_ph = chi_ph
    if do_pp:
        chi_pp /= model.N**2
        model.chi_pp = chi_pp
    return chi_ph, chi_pp
