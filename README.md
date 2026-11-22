# ephnegf — coupled electron–phonon NEGF from scratch (v0.4.0, stage 4)

Self-consistent Born approximation (SCBA) for electrons **and** phonons out of equilibrium,
with block-tridiagonal recursive Green's functions (RGF) for both subsystems.

## Layout
```
ephnegf/
  constants.py   Fermi/Bose, unit conversions (hbar = e = 1, eV)
  leads.py       Sancho–Rubio surface GFs, non-orthogonal (A = zS − H), electrons & phonons
  rgf.py         block-tridiagonal RGF with energy-dependent off-diagonal blocks (zS − H);
                 diagonal blocks of G^R, G^<, G^> (explicit Σ^> recursion)
  ballistic.py   Landauer transmission from diagonal blocks only
  io_sisl.py     SIESTA/TranSIESTA (.TSHS) and any sisl Hamiltonian -> BlockSystem;
                 automatic block-tridiagonal partition (BFS layering), transverse k
  convolve.py    FFT convolutions on the common grid ω_n = n·dE, Kramers–Kronig
  system.py      BlockSystem (H or K), Lead, Coupling, 1D chain builders
  scba.py        CoupledSCBA solver + observables (currents, heat flows, site-resolved heating)
tests/test_core.py      RGF vs dense, leads vs analytic, conservation, stage-2 regression
tests/test_nonorth.py   overlap S: leads, RGF, T=1, SCBA conservation; sisl GNR vs dense + integer channels
examples/chain_heating.py          6-site wire: IETS + heating profile + energy balance
examples/sisl_gnr_transmission.py  non-orthogonal ZGNR via sisl: bands + T(E), pristine vs defect
legacy/          stage-1 and stage-2 single-file prototypes (reference implementations)
```

## Physics / approximations
* Local (Holstein-type) coupling `H_eph = Σ_o g_o n_o u_a(o)`; local SCBA is Φ-derivable ⇒
  particle and energy conserving (verified to ~1e-13 / 1e-9).
* Phonons: full D^R, D^<, D^> with phonon leads at their own temperatures (`T_ph_L`, `T_ph_R`);
  occupations are *computed*, no phenomenological bath.
* Re Π^R off by default (adiabatic part already in DFT force constants); Re Σ^R kept.
* G^> is built from Σ^> with the same recursion, so the infinitesimal η does not act as a
  fictitious bath (this matters for conservation at the 1e-6 level).

## Install / run
```
pip install -e ".[siesta,plot,test]"
python -m pytest tests            # or: python tests/test_core.py; python tests/test_nonorth.py
python examples/sisl_gnr_transmission.py
python examples/chain_heating.py
```

## TranSIESTA front end
```python
from ephnegf.io_sisl import from_siesta
el = from_siesta("device.TSHS", "elecL.TSHS", "elecR.TSHS", axis=2,
                 left_atoms=range(0, 8), right_atoms=range(56, 64), k=(0, 0, 0))
T = ephnegf.transmission(el, E)
```
The device must contain one electrode principal layer at each end (TranSIESTA convention),
with nsc = 1 along the transport axis. Non-orthogonal overlaps are handled exactly.
Note: in a non-orthogonal basis the local e-ph self-energy is still conserving, but
`n_el` (diagonal of G^<) is not a Mulliken population.

## Roadmap
1. ✅ 1D single level, Einstein phonon (legacy/negf_eph_1d.py)
2. ✅ phonon NEGF + phonon leads, coupled (legacy/negf_eph_1d_stage2.py)
3. ✅ package: block RGF, Sancho–Rubio leads, multi-site devices
4. ✅ non-orthogonal basis (S matrix) + sisl/TranSIESTA reader, Landauer T(E) (this version)
5. ☐ Wannier90 `_hr.dat` + phonopy force constants + e-ph matrices (EPW / finite differences)
6. ☐ non-local (block) self-energies, k-point sampling for 2D/3D leads
7. ☐ performance: Numba/CuPy kernels, MPI over energies, Anderson mixing
8. ☐ Hartree / Poisson feedback, thermoelectric and phonon-drag outputs
