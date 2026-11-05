"""Adapted from linearized-rpa, spinless branch; see PROVENANCE.md."""

import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from ..constants import s0, sx, sy, kB
from ..vertex import split_vertex
from ..phase_diagram import kernel

# lattice constant
la = 2.46  # Angstrom
a0 = la / np.sqrt(3)
dc = 3.35

class RhombohedralMultilayer:
    """
    Twisted k coordinates are fractional moire coordinates; untwisted k is
    Cartesian in inverse Angstrom.
    """
    split_vertex = split_vertex
    kernel = kernel

    def __init__(
        self,
        V: float = 0,
        nlayer: int = 2,
        theta: float = 0,
        Qcut: int = 3,
        reverse: bool = True,
        twist: bool = True,
        
        vf=2.6 * np.sqrt(3) / 2 * la,
        vppsigma=0.339,
        gamma3=0.28,
        gamma4=-0.140,
    ) -> None:
        """Initialize layer, twist, and hopping parameters."""
        self.V = V
        self.nlayer = int(nlayer)
        self.theta = theta
        self.Qcut = int(Qcut)
        self.reverse = bool(reverse)
        self.twist = bool(twist)

        self.vf = vf
        self.vppsigma = vppsigma
        self.v3 = gamma3 * np.sqrt(3) / 2 * la
        self.v4 = gamma4 * np.sqrt(3) / 2 * la

        if self.twist:
            if np.isclose(self.theta, 0.0):
                raise ValueError("twist=True requires theta != 0.")
            self.lm = la / (2 * np.sin(np.deg2rad(self.theta) / 2))
            self.cell = self.lm * np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2]])
            self.kD = self._moire_len()
        else:
            if not np.isclose(self.theta, 0.0):
                raise ValueError("twist=False describes the untwisted system, so use theta=0.")
            self.Qcut = 0
            self.lm = la
            self.cell = la * np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2]])
            self.kD = 0.0

        self.icell = 2 * np.pi * np.linalg.inv(self.cell).T
        self.Nqvec = (2 * self.Qcut + 1) ** 2
        self.Ndim = 2 * self.Nqvec * self.nlayer
        self.qvecs = np.mgrid[
            -self.Qcut:self.Qcut + 1,
            -self.Qcut:self.Qcut + 1,
        ].reshape(2, self.Nqvec)
        self._system_in()
        self._tunneling()

    def _tunneling(self): # fit from PhysRevResearch.1.013001
        """Relaxed-bilayer fit: theta in degrees, tunneling in eV."""
        if not self.twist:
            self.u_AA = self.u_AB = 0.0
            return
        theta = abs(float(self.theta))
        if not np.isfinite(theta) or not 0.42 <= theta <= 5.0:
            raise ValueError("Fit requires 0.42 <= |theta| <= 5 degrees.")
    
        self.u_AA = 0.07881664 / (1.0 + (0.46655647 / theta)**1.81002046)
        self.u_AB = 0.08455992 + 0.02428948 / (1.0 + (theta / 1.04079245)**2.86625337)

    def _moire_len(self):
        """Return the separation of the rotated Dirac points in inverse Angstrom."""
        th = np.deg2rad(self.theta)
        K = (4 * np.pi) / (3 * la) * np.array([1.0, 0.0])
        rot1 = np.array([
            [np.cos(th / 2), -np.sin(th / 2)],
            [np.sin(th / 2),  np.cos(th / 2)],
        ])
        rot2 = np.array([
            [ np.cos(th / 2), np.sin(th / 2)],
            [-np.sin(th / 2), np.cos(th / 2)],
        ])
        return np.linalg.norm(rot2 @ K - rot1 @ K)

    def _system_in(self):
        """Set layer orientations and ABC-ABC or ABC-CBA stacking."""
        slg = self.nlayer // 2 if self.twist else self.nlayer
        ang = [0] * slg + [1] * (self.nlayer - slg)
        seq1 = ['A', 'B', 'C'] * (slg // 3) + ['A', 'B', 'C'][:slg % 3]
        seq2 = ['A', 'B', 'C'] * ((self.nlayer - slg) // 3) + ['A', 'B', 'C'][:(self.nlayer - slg) % 3]
        seq2 = seq2[::-1] if self.reverse else seq2
        self.ang = ang
        self.seq = seq1 + seq2

    @staticmethod
    def _stacking_chirality(bottom, top):
        """Return the chirality of two neighboring layer labels."""
        pair = bottom + top
        if pair in ['AB', 'BC', 'CA']:
            return 1
        if pair in ['BA', 'CB', 'AC']:
            return -1
        if pair in ['AA', 'BB', 'CC']:
            return 0
        raise ValueError(
            f"ERROR in stacking_chirality: Unsupported stacking sequence '{pair}'. "
            "Supported sequences: AB, BC, CA, BA, CB, AC, AA, BB, CC."
        )

    def hamiltonian(self, km, valley):
        """Return the continuum Hamiltonian in eV for momentum km and valley +/-1."""
        tau = int(valley)
        if tau not in (-1, +1):
            raise ValueError("valley must be +1 or -1.")

        km = np.asarray(km, dtype=float)
        # Centered displacement potential:
        # U_l = -(V/1000) * (l - (N_layer - 1)/2)  [eV]
        E_field = lambda ilayer: -(self.V / 1000.0) * (ilayer - 0.5 * (self.nlayer - 1))

        if self.twist:
            b1m = np.array([-0.5, -np.sqrt(3) / 2]) * np.sqrt(3) * self.kD
            b2m = np.array([+1.0, 0.0]) * np.sqrt(3) * self.kD
            K1 = np.array([np.sqrt(3) / 2, +0.5]) * self.kD
            K2 = np.array([np.sqrt(3) / 2, -0.5]) * self.kD
            Ks = (K2, K1)

            def get_k(ilayer, q):
                """Return the Cartesian momentum for this layer and plane wave."""
                K = Ks[self.ang[ilayer]]
                return -tau * K + (km[0] + q[0]) * b1m + (km[1] + q[1]) * b2m
        else:
            def get_k(ilayer, q):
                """Return the Cartesian momentum for this layer and plane wave."""
                return km

        H = np.zeros((self.Ndim, self.Ndim), dtype=complex)
        idx = lambda layer, iq: 2 * (layer * self.Nqvec + iq)

        phi = tau * 2 * np.pi / 3
        T1 = np.array([[self.u_AA, self.u_AB], [self.u_AB, self.u_AA]], dtype=complex)
        T2 = np.array([[self.u_AA, self.u_AB * np.exp(-1j * phi)],
                       [self.u_AB * np.exp(+1j * phi), self.u_AA]], dtype=complex)
        T3 = np.array([[self.u_AA, self.u_AB * np.exp(+1j * phi)],
                       [self.u_AB * np.exp(-1j * phi), self.u_AA]], dtype=complex)
        qmap = {(int(q[0]), int(q[1])): iq for iq, q in enumerate(self.qvecs.T)}

        # Intralayer Dirac blocks.
        for ilayer in range(self.nlayer):
            for iq, q in enumerate(self.qvecs.T):
                i = idx(ilayer, iq)
                k = get_k(ilayer, q)
                H[i:i + 2, i:i + 2] = (
                    self.vf * (tau * k[0] * sx + k[1] * sy) + E_field(ilayer) * s0
                )

        # Same-angle interlayer hopping inside each untwisted multilayer block.
        for ilayer in range(self.nlayer - 1):
            if self.ang[ilayer] != self.ang[ilayer + 1]:
                continue
            chirality = self._stacking_chirality(self.seq[ilayer], self.seq[ilayer + 1])
            for iq, q in enumerate(self.qvecs.T):
                i = idx(ilayer, iq)
                j = idx(ilayer + 1, iq)
                k = get_k(ilayer, q)
                kminus = -(tau * k[0] - 1j * k[1])
                kplus = -(tau * k[0] + 1j * k[1])

                if chirality == 1:
                    T = np.array([[self.v4 * kminus, self.v3 * kplus],
                                  [self.vppsigma, self.v4 * kminus]], dtype=complex)
                elif chirality == -1:
                    T = np.array([[self.v4 * kplus, self.vppsigma],
                                  [self.v3 * kminus, self.v4 * kplus]], dtype=complex)
                else:
                    T = self.vppsigma * s0

                H[i:i + 2, j:j + 2] = T
                H[j:j + 2, i:i + 2] = T.conj().T

        # BM tunneling at the twisted interface.
        for ilayer in range(self.nlayer - 1):
            if self.ang[ilayer] == self.ang[ilayer + 1]:
                continue
            for iq, q in enumerate(self.qvecs.T):
                i = idx(ilayer, iq)
                for dq, T in (((0, 0), T1), ((-tau, 0), T2), ((-tau, -tau), T3)):
                    jq = qmap.get((int(q[0] + dq[0]), int(q[1] + dq[1])))
                    if jq is None:
                        continue
                    j = idx(ilayer + 1, jq)
                    H[i:i + 2, j:j + 2] = T
                    H[j:j + 2, i:i + 2] = T.conj().T

        return H

    def init_mesh(self, N=4000, kmax=0.06, cutoff=0.025):
        """Create the momentum mesh and its integration weights."""
        kn = np.fft.fftfreq(N)

        if self.twist:
            kx, ky = np.meshgrid(kn, kn)
            self.k = np.stack([kx, ky], axis=-1).reshape(-1, 2).T
        else:
            mesh = kn[np.abs(kn) <= cutoff]
            x, y = np.meshgrid(mesh, mesh)
            k = np.array([x, y]).T.dot(self.icell)
            i, j = np.where(np.linalg.norm(k, axis=2) <= kmax)
            self.k = k[i, j].T

        self.nk = self.k.shape[1]
        self.N = N
        self.factor = 1e4 / abs(np.linalg.det(self.cell)) / self.N / self.N

    def calculate_bandstructure(self):
        """Calculate and store the full Hamiltonians, energies, and eigenvectors at k."""
        h = np.empty((2, self.nk, self.Ndim, self.Ndim), dtype=np.complex128)
        for t, tau in enumerate([-1, +1]):
            for ki, k in enumerate(self.k.T):
                h[t, ki] = self.hamiltonian(k, valley=tau)
        e, u = np.linalg.eigh(h)
        self.h = h
        self.e = e
        self.u = u

    def calculate_inverse_bandstructure(self):
        """Calculate and store the full Hamiltonians, energies, and eigenvectors at -k."""
        h_inv = np.empty((2, self.nk, self.Ndim, self.Ndim), dtype=np.complex128)
        for t, tau in enumerate([-1, +1]):
            for ki, k in enumerate(-self.k.T):
                h_inv[t, ki] = self.hamiltonian(k, valley=tau)
        e_inv, u_inv = np.linalg.eigh(h_inv)
        self.h_inv = h_inv
        self.e_inv = e_inv
        self.u_inv = u_inv

    def energy_window(self, cutoff=600):
        """Prepare states within mu0 +/- cutoff (meV) and return bounds in eV."""
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError("cutoff must be finite and nonnegative in meV.")
        self.mu0 = np.median(self.e)
        self.energy_cutoff = cutoff
        self.energy_bounds = self.mu0 + np.array([-cutoff, cutoff]) / 1000
        lo, hi = self.energy_bounds
        self.band_mask = (self.e >= lo) & (self.e <= hi)
        self.e_active = [[self.e[v, k, keep] for k, keep in enumerate(masks)]
                         for v, masks in enumerate(self.band_mask)]
        self.u_active = [[self.u[v, k][:, keep] for k, keep in enumerate(masks)]
                         for v, masks in enumerate(self.band_mask)]
        self.e_active_inv = self.u_active_inv = None
        self.band_mask_inv = None
        if hasattr(self, "e_inv"):
            self.band_mask_inv = (self.e_inv >= lo) & (self.e_inv <= hi)
            self.e_active_inv = [[self.e_inv[v, k, keep] for k, keep in enumerate(masks)]
                                 for v, masks in enumerate(self.band_mask_inv)]
            self.u_active_inv = [[self.u_inv[v, k][:, keep] for k, keep in enumerate(masks)]
                                 for v, masks in enumerate(self.band_mask_inv)]
        return self.energy_bounds.copy()

    def calculate_mu(self, occ: float = 0, T: float = 0.4, e=None, tol: float = 1e-12):
        """Set and return mu in eV for density occ (10^12 cm^-2) and temperature T (K)."""
        if not np.isfinite(T) or T <= 0:
            raise ValueError("T must be finite and positive.")

        if e is None:
            e = self.e
        e = np.asarray(e)
        if e.size == 0 or not np.isfinite(e).all():
            raise ValueError("Energies must be finite and nonempty.")
        margin = max(1.0, 40 * kB * T)
        bounds = [e.min() - margin, e.max() + margin]

        beta = 1 / (kB * T)
        self.beta = beta
        self.occ = occ

        def func(mu):
            """Return the excess density relative to the requested filling."""
            return 2 * self.factor * np.sum(expit(-beta * (e - mu)) - 0.5) - occ

        self.mu = brentq(func, *bounds, xtol=tol)

        return self.mu
