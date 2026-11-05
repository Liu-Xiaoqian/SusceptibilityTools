"""Small demonstration; increase the mesh and check convergence for research."""

import numpy as np

from chi_tools.models import RhombohedralMultilayer
from chi_tools.calculate_chi import calculate_chi
from chi_tools.vertex import tensor
from chi_tools.phase_diagram import classify_instability
from chi_tools.hartree_fock import hartree_fock, calculate_hf_band


def main():
    model = RhombohedralMultilayer(nlayer=3, V=20, theta=0, twist=False)
    model.init_mesh(N=200)
    model.calculate_bandstructure()
    model.calculate_inverse_bandstructure()
    mu = model.calculate_mu(occ=0, T=4.2)
    bounds = model.energy_window(cutoff=600)
    ph, pp = calculate_chi(model)
    U4 = tensor(16, J=-4)
    vertices = model.split_vertex(U4, vertex="bare")
    kernels = model.kernel(vertices, channels="bare")
    phase = classify_instability(kernels, channels="bare")
    print(f"mu = {mu:.8f} eV; nk = {model.nk}")
    print(f"mu0 = {model.mu0:.8f} eV; energy window = {bounds} eV")
    print("Bare phase code (0=FL, 1=Stoner, 2=IVC):", int(phase))

    # The zero-interaction limit verifies the HF workflow without claiming
    # convergence for a particular interacting physical state.
    hartree_fock(model, tensor(0), T=4.2, occ=0,
                      sigma_seed=np.zeros((4, 4)), verbose=False)
    calculate_hf_band(model)
    print(f"Noninteracting HF mu = {model.mu_hf:.8f} eV")
    print("HF band shape:", model.e_hf.shape)


if __name__ == "__main__":
    main()
