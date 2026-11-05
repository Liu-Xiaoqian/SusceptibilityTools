"""Leading-instability classification for two-valley, spin-degenerate bubbles."""

import numpy as np
from .vertex import ph4_t2m, pp4_t2m


def kernel(self, vertices, channels="bare"):
    """Return density, magnetic, and optionally singlet and triplet kernels."""
    if channels not in ("bare", "dressed"):
        raise ValueError("channels must be 'bare' or 'dressed'.")

    chi_ph = ph4_t2m(self.chi_ph)
    Kd = chi_ph @ vertices['Vd']          # Density PH channel.
    Km = -chi_ph @ vertices['Vm']         # Magnetic PH channel.
    if channels == "bare":
        return {'Kd': Kd, 'Km': Km}

    chi_pp = pp4_t2m(self.chi_pp)
    Ks = -0.5 * (chi_pp @ vertices['Vs'])  # Spin-singlet PP channel.
    Kt = 0.5 * (chi_pp @ vertices['Vt'])   # Spin-triplet PP channel.
    return {'Kd': Kd, 'Km': Km, 'Ks': Ks, 'Kt': Kt}


def classify_instability(kernels, channels="bare", tol=1e-12):
    """Return the phase codes from explicit density, magnetic, and pairing kernels."""
    # Bare: 0 = FL, 1 = Stoner, 2 = IVC.
    # Dressed: 0 = FL, 1 = d-Stoner, 2 = d-IVC, 3 = m-Stoner,
    # 4 = m-IVC, 5 = s-PDW, 6 = s-Cooper, 7 = t-PDW, 8 = t-Cooper.
    if channels not in ("bare", "dressed"):
        raise ValueError("channels must be 'bare' or 'dressed'.")
    if channels == "bare":
        Kd, Km = np.broadcast_arrays(kernels['Kd'], kernels['Km'])
    else:
        Kd, Km, Ks, Kt = np.broadcast_arrays(
            kernels['Kd'], kernels['Km'], kernels['Ks'], kernels['Kt'])
    phase = np.zeros(Kd.shape[:-2], dtype=int)

    def eigmax(K):
        """Return the largest real eigenvalue and its right eigenvector."""
        w, v = np.linalg.eig(K)
        i = np.argmax(w.real)
        return w[i].real, v[:, i]

    def intervalley(v):
        """Test whether off-diagonal valley components dominate the mode."""
        # Valley basis: (00), (01), (10), (11).
        return abs(v[1])**2 + abs(v[2])**2 > abs(v[0])**2 + abs(v[3])**2

    for ind in np.ndindex(phase.shape):
        ld, vd = eigmax(Kd[ind])
        lm, vm = eigmax(Km[ind])
        if channels == "bare":
            eigenvalues = np.array([ld, lm])
        else:
            ls, vs = eigmax(Ks[ind])
            lt, vt = eigmax(Kt[ind])
            eigenvalues = np.array([ld, lm, ls, lt])

        largest = eigenvalues.max()
        if largest < 1 - tol:
            continue

        # Ties prefer density, magnetic, singlet, then triplet.
        channel = np.flatnonzero(eigenvalues >= largest - tol)[0]
        if channel == 0:  # Density: Stoner or IVC.
            phase[ind] = 2 if intervalley(vd) else 1
        elif channel == 1:  # Magnetic: Stoner or IVC.
            if channels == "bare":
                phase[ind] = 2 if intervalley(vm) else 1
            else:
                phase[ind] = 4 if intervalley(vm) else 3
        elif channel == 2:  # Singlet: PDW or Cooper.
            phase[ind] = 6 if intervalley(vs) else 5
        elif channel == 3:  # Triplet: PDW or Cooper.
            phase[ind] = 8 if intervalley(vt) else 7

    return phase
