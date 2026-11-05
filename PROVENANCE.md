# Source attribution

The Hamiltonian, momentum mesh, and bare susceptibility formulas are adapted
from [linearized-rpa, spinless branch](https://github.com/yaroslav-zhumagulov/linearized-rpa/tree/273eeaaeb803baeab0f5016ac30f684750231eda/lrpa),
commit `273eeaaeb803baeab0f5016ac30f684750231eda`.
The Hamiltonian source is `models/rhombohedral_multilayer.py`.
Chemical-potential calculation is adapted from `chemical_potential.py` and
implemented as a model method.

The HF iteration structure follows the occupation and self-energy updates in
[hartree_fock_sym](https://github.com/yaroslav-zhumagulov/linearized-rpa/blob/main/lrpa/hartree_fock.py),
with full-matrix diagonalization to retain intervalley coherence.

SusceptibilityTools includes its own constants and depends only on NumPy and
SciPy at runtime. It does not import `lrpa`.
The included continuum model retains the angle-dependent tunneling fit and
standard complex phases. Energy-window selection is explicit and preserves
the full spectra for filling calculations.

The interaction, response, and Hartree-Fock conventions are documented in
[README.md](README.md).

# Release metadata

The package version is `0.1.0.dev0`. Author and license metadata are not set
in `pyproject.toml`.
