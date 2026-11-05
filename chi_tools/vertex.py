"""Spin-valley interaction tensors and bare or dressed channel vertices."""

import numpy as np

def tensor(V0, V1=None, J=0.0):
    """Return U4 in (up valley0, up valley1, down valley0, down valley1)."""
    if V1 is None:
        V1 = V0

    U4 = np.zeros((4, 4, 4, 4), dtype=float)

    def add_density_pair(p, q, x):
        """Add the antisymmetrized density interaction between two flavors."""
        U4[p, q, q, p] += x
        U4[q, p, p, q] += x
        U4[p, p, q, q] -= x
        U4[q, q, p, p] -= x

    # V0: same valley, opposite spin
    add_density_pair(0, 2, V0)   # up0, dn0
    add_density_pair(1, 3, V0)   # up1, dn1

    # V1: different valley, all spin combinations
    add_density_pair(0, 1, V1)   # up0, up1
    add_density_pair(0, 3, V1)   # up0, dn1
    add_density_pair(2, 1, V1)   # dn0, up1
    add_density_pair(2, 3, V1)   # dn0, dn1

    if J != 0.0:
        # Hund coupling / exchange

        a = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        b = np.array([0, 2, 1, 3, 0, 2, 1, 3])
        c = np.array([3, 3, 2, 2, 1, 1, 0, 0])
        d = np.array([3, 1, 2, 0, 3, 1, 2, 0])
        U4[a, b, c, d] += J

        a = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        b = np.array([1, 3, 0, 2, 1, 3, 0, 2])
        c = np.array([3, 3, 2, 2, 1, 1, 0, 0])
        d = np.array([2, 0, 3, 1, 2, 0, 3, 1])
        U4[a, b, c, d] -= J

        a = np.array([0, 1, 2, 3])
        b = np.array([0, 1, 2, 3])
        c = np.array([1, 0, 3, 2])
        d = np.array([1, 0, 3, 2])
        U4[a, b, c, d] += 2 * J

        a = np.array([0, 1, 2, 3])
        b = np.array([1, 0, 3, 2])
        c = np.array([1, 0, 3, 2])
        d = np.array([0, 1, 2, 3])
        U4[a, b, c, d] -= 2 * J

    return U4

def perm(T, p):
    """Permute the trailing tensor axes while preserving batch axes."""
    return T.transpose(
        *range(T.ndim - len(p)),
        *(T.ndim - len(p) + i for i in p)
    )


def ph_t2m(T):
    """Convert a spin-valley PH tensor into a matrix."""
    return perm(T, (0,1,3,2)).reshape(*T.shape[:-4], 16, 16)


def ph_m2t(M):
    """Convert a spin-valley PH matrix back into a tensor."""
    return perm(M.reshape(*M.shape[:-2], 4,4,4,4), (0,1,3,2))


def pp_t2m(T):
    """Convert a spin-valley PP tensor into a matrix."""
    return perm(T, (2,0,1,3)).reshape(*T.shape[:-4], 16, 16)


def pp_m2t(M):
    """Convert a spin-valley PP matrix back into a tensor."""
    return perm(M.reshape(*M.shape[:-2], 4,4,4,4), (1,2,0,3))


def ph4_t2m(T):
    """Return the valley PH matrix, preserving any leading parameter axes."""
    T = np.asarray(T)
    if T.shape[-2:] == (4, 4):
        return T
    if T.shape[-4:] != (2, 2, 2, 2):
        raise ValueError("PH susceptibility must end in (4, 4) or (2, 2, 2, 2).")
    return perm(T, (0,1,3,2)).reshape(*T.shape[:-4], 4, 4)


def ph4_m2t(M):
    """Convert a valley PH matrix back into a four-index tensor."""
    return perm(M.reshape(*M.shape[:-2], 2,2,2,2), (0,1,3,2))


def pp4_t2m(T):
    """Return the valley PP matrix, preserving any leading parameter axes."""
    T = np.asarray(T)
    if T.shape[-2:] == (4, 4):
        return T
    if T.shape[-4:] != (2, 2, 2, 2):
        raise ValueError("PP susceptibility must end in (4, 4) or (2, 2, 2, 2).")
    return perm(T, (2,0,1,3)).reshape(*T.shape[:-4], 4, 4)


def pp4_m2t(M):
    """Convert a valley PP matrix back into a four-index tensor."""
    return perm(M.reshape(*M.shape[:-2], 2,2,2,2), (1,2,0,3))


def split_vertex(self, U4, vertex="bare"):
    """Return bare or dressed channel vertices using stored bubbles for dressing."""
    if vertex not in ("bare", "dressed"):
        raise ValueError("vertex must be 'bare' or 'dressed'.")

    def U4_to_8(T):
        """Separate spin and valley indices, grouping the four spin axes first."""
        return perm(
            T.reshape(*T.shape[:-4], 2,2,2,2,2,2,2,2),
            (0,2,4,6,1,3,5,7)
        )

    def embed_chi(chi):
        """Embed the valley bubble in the spin-valley tensor."""
        chi = np.asarray(chi)
        chi8 = np.zeros((*chi.shape[:-4], 2,2,2,2,2,2,2,2), dtype=chi.dtype)

        for s in range(2):
            chi8[..., s, s, s, s, :, :, :, :] = chi
            chi8[..., s, 1-s, 1-s, s, :, :, :, :] = chi

        return perm(chi8, (0,4,1,5,2,6,3,7)).reshape(*chi.shape[:-4], 4,4,4,4)

    U_8 = U4_to_8(U4)

    U_d = U_8[0,0,1,1] + U_8[0,0,0,0]
    U_m = U_8[0,0,1,1] - U_8[0,0,0,0]
    
    U_s = U_8[0,0,1,1] - U_8[0,1,1,0]
    U_t = U_8[0,0,1,1] + U_8[0,1,1,0]

    out = {
        "Vd": ph4_t2m(U_d),
        "Vm": ph4_t2m(U_m),
        "Vs": pp4_t2m(U_s),
        "Vt": pp4_t2m(U_t),
    }

    if vertex == "bare":
        return out

    U_ph = ph_t2m(U4)
    U_pp = pp_t2m(U4)
    I = np.eye(16)

    chi_ph_mat = ph_t2m(embed_chi(ph4_m2t(ph4_t2m(self.chi_ph))))
    # Explicit matrix RHS broadcasting works with both NumPy 1.x and 2.x.
    rhs_ph = np.broadcast_to(U_ph, chi_ph_mat.shape)
    Phi_ph = ph_m2t(np.linalg.solve(I - U_ph @ chi_ph_mat, rhs_ph) - U_ph)

    vPhi_ph = -perm(Phi_ph, (2,1,0,3))

    chi_pp_mat = pp_t2m(embed_chi(pp4_m2t(pp4_t2m(self.chi_pp))))
    rhs_pp = np.broadcast_to(U_pp, chi_pp_mat.shape)
    Phi_pp = pp_m2t(np.linalg.solve(I - 0.5 * U_pp @ chi_pp_mat, rhs_pp) - U_pp)

    V8_pair = U4_to_8(U4 + Phi_ph + vPhi_ph)

    Vs4 = V8_pair[...,0,0,1,1,:,:,:,:] - V8_pair[...,0,1,1,0,:,:,:,:]
    Vt4 = V8_pair[...,0,0,1,1,:,:,:,:] + V8_pair[...,0,1,1,0,:,:,:,:]

    V8_ph = U4_to_8(U4 + Phi_pp + vPhi_ph)

    Vd4 = V8_ph[...,0,0,1,1,:,:,:,:] + V8_ph[...,0,0,0,0,:,:,:,:]
    Vm4 = V8_ph[...,0,0,1,1,:,:,:,:] - V8_ph[...,0,0,0,0,:,:,:,:]

    out.update({
        "Vd": ph4_t2m(Vd4),
        "Vm": ph4_t2m(Vm4),
        "Vs": pp4_t2m(Vs4),
        "Vt": pp4_t2m(Vt4),
    })

    return out
