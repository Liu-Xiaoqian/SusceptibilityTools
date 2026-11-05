# SusceptibilityTools

A Python package for electronic susceptibilities, interaction-driven
instabilities, and Hartree–Fock calculations. Import it as `chi_tools`.

## Features

- Particle–hole and particle–particle susceptibilities with energy-window selection.
- Bare and dressed interaction vertices and instability classification.
- Full-band Hartree–Fock for spin/valley polarization and intervalley coherence.
- A rhombohedral multilayer model, with support for user-supplied band data.

## Install

Requires Python 3.10+, NumPy, and SciPy.

From the package directory:

```bash
python -m pip install -e .
```

## Usage

```python
import numpy as np
from chi_tools.models import RhombohedralMultilayer
from chi_tools.calculate_chi import calculate_chi
from chi_tools.vertex import tensor
from chi_tools.phase_diagram import classify_instability
from chi_tools.constants import s0, sx, sz
from chi_tools.hartree_fock import hartree_fock, calculate_hf_band

model = RhombohedralMultilayer(nlayer=3, V=20, twist=False)
model.init_mesh(N=4000)
model.calculate_bandstructure()
model.calculate_inverse_bandstructure()
model.calculate_mu(occ=0, T=4.2)
model.energy_window(cutoff=600)  # mu0 +/- 600 meV; susceptibility only.
calculate_chi(model)           # Sets model.chi_ph and model.chi_pp.

U4 = tensor(V0=16, J=-4)
vertices = model.split_vertex(U4, vertex="bare")
kernels = model.kernel(vertices, channels="bare")
phase = classify_instability(kernels, channels="bare")  # 0: FL, 1: Stoner, 2: IVC.

sigma_seed = 1e-3 * np.kron(s0, sx)  # Density IVC seed in eV.
hartree_fock(model, U4, T=4.2, occ=0, sigma_seed=sigma_seed, verbose=False)
calculate_hf_band(model)       # Sets model.e_hf using model.sigma.
```

Use `vertex="dressed"` and `channels="dressed"` to include pairing channels;
phase codes are listed in `phase_diagram.py`. Check mesh and iteration
convergence before interpreting results.

Choose a small initial self-energy according to the order being investigated:

| Initial order | Example seed |
| --- | --- |
| Valley polarization | `1e-3 * np.kron(s0, sz)` |
| Spin polarization | `1e-3 * np.kron(sz, s0)` |
| Opposite spin polarization in the two valleys | `1e-3 * np.kron(sz, sz)` |
| Density IVC | `1e-3 * np.kron(s0, sx)` |
| Magnetic IVC | `1e-3 * np.kron(sz, sx)` |

The coefficient `1e-3` is 1 meV. The seed sets the initial condition, not a
constraint on subsequent HF updates.

HF stores `sigma`, `rho`, and `mu_hf` on the model;
it returns nothing. `calculate_hf_band` stores energies in `e_hf`, with shape
`(nk, 4 * nband)`. Call it after each solve when bands are needed.
The bare bands and `model.mu` remain unchanged. HF always uses full bands,
independently of the susceptibility window.

To continue a calculation, use `model.sigma` as the seed:

```python
hartree_fock(model, U4, T=4.2, occ=0, sigma_seed=model.sigma)
```

For a later session, save `model.sigma` with `np.save` and load it as the seed
after preparing the same model and parameters.

## Conventions

Energies, interactions, and `sigma_seed` are in eV; `V` and `cutoff` are in meV;
`T` is in kelvin; `occ` is in units of `10^12 cm^-2`.
The window is centered on the full-spectrum half-filling chemical potential at
zero temperature. HF seeds are Hermitian `(4, 4)` matrices ordered as
`(up valley0, up valley1, down valley0, down valley1)`.

Custom models supply `e` with shape `(2, nk, nband)` and `u` with shape
`(2, nk, norbital, nband)`, plus `N`, `mu`, and `beta` for susceptibility.
PP also needs `e_inv` and `u_inv` at `-k`. HF needs `nk`, `factor`, and complete
band eigenbases (`nband == norbital`). Sums use momentum weight `1/N**2`;
interactions must follow the same cell normalization.

## Structure

```text
chi_tools/
├── models/rhombohedral_multilayer.py  # Hamiltonian, bands, filling, energy window
├── constants.py                     # Physical constants and Pauli matrices
├── calculate_chi.py                 # PH and PP bubbles
├── vertex.py                        # Interaction tensors and channel vertices
├── phase_diagram.py                 # Kernels and instability classification
└── hartree_fock.py                  # HF iteration and band calculation
```

Run the example with `python examples/basic_workflow.py` and tests with
`python -m unittest discover -s tests -v`.
See [PROVENANCE.md](PROVENANCE.md) for source attribution.
