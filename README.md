# ephnegf — coupled electron–phonon NEGF from scratch (v0.5.0, stage 5)

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
  units.py       real-unit conversions (eV/Å², amu -> eV²; dH/dx -> internal coupling)
  phonons.py     phonopy FORCE_CONSTANTS reader, ASR, dynamical matrix, electrode layer
                 blocks, phonon BlockSystem with automatic atom-level BTD partition
  coupling.py    general e-ph matrices M^λ (finite differences), block-local cutting
                 (electron blocks × phonon groups), Holstein as special case
  io_sisl.py     SIESTA/TranSIESTA (.TSHS) and any sisl Hamiltonian -> BlockSystem;
                 automatic block-tridiagonal partition (BFS layering), transverse k
  convolve.py    FFT convolutions on the common grid ω_n = n·dE, Kramers–Kronig
  system.py      BlockSystem (H or K), Lead, Coupling, 1D chain builders
  scba.py        CoupledSCBA solver + observables (currents, heat flows, site-resolved heating)
tests/test_core.py      RGF vs dense, leads vs analytic, conservation, stage-2 regression
tests/test_phonons_eph.py  units/phonon band edge, finite-difference M vs analytic, non-local SCBA conservation
tests/test_nonorth.py   overlap S: leads, RGF, T=1, SCBA conservation; sisl GNR vs dense + integer channels
examples/chain_heating.py          6-site wire: IETS + heating profile + energy balance
examples/carbon_chain_ssh.py       C chain in real units: FC phonons + finite-difference SSH coupling
examples/sisl_gnr_transmission.py  non-orthogonal ZGNR via sisl: bands + T(E), pristine vs defect
legacy/          stage-1 and stage-2 single-file prototypes (reference implementations)
```

## Physics / approximations
* General coupling `H_eph = Σ_λ M^λ u_λ`, treated block-locally: Σ kept inside each electron
  RGF block, D_λλ' inside each phonon group (atom or whole phonon block). The functional is
  Φ-derivable ⇒ particle and energy conserving for any partition (verified to ~1e-15 / 1e-9).
  Parts of M linking different electron blocks are dropped (`EPCoupling.dropped_fraction`);
  coarser blocks reduce this at higher cost. Holstein = special case (regression-tested).
* Overlap derivatives ∂S/∂u are not yet included in M.
* Phonons: full D^R, D^<, D^> with phonon leads at their own temperatures (`T_ph_L`, `T_ph_R`);
  occupations are *computed*, no phenomenological bath.
* Re Π^R off by default (adiabatic part already in DFT force constants); Re Σ^R kept.
* G^> is built from Σ^> with the same recursion, so the infinitesimal η does not act as a
  fictitious bath (this matters for conservation at the 1e-6 level).

## Install / run
```
pip install -e ".[siesta,plot,test]"
python -m pytest tests            # or: python tests/test_core.py; python tests/test_phonons_eph.py  units/phonon band edge, finite-difference M vs analytic, non-local SCBA conservation
tests/test_nonorth.py
python examples/carbon_chain_ssh.py       C chain in real units: FC phonons + finite-difference SSH coupling
examples/sisl_gnr_transmission.py
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
4. ✅ non-orthogonal basis (S matrix) + sisl/TranSIESTA reader, Landauer T(E)
5. ✅ force constants (phonopy) + finite-difference e-ph matrices, block-local non-local SCBA (this version)
5b. ☐ SIESTA displaced-geometry workflow (±δ TSHS -> M), Wannier90 `_hr.dat`, EPW g(R_e,R_p)
6. ☐ off-diagonal (inter-block) self-energies, k-point sampling for 2D/3D leads
7. ☐ performance: Numba/CuPy kernels, MPI over energies, Anderson mixing
8. ☐ Hartree / Poisson feedback, thermoelectric and phonon-drag outputs
