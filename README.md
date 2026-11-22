# ephnegf — coupled electron–phonon NEGF from scratch (v0.3.0, stage 3)

Self-consistent Born approximation (SCBA) for electrons **and** phonons out of equilibrium,
with block-tridiagonal recursive Green's functions (RGF) for both subsystems.

## Layout
```
ephnegf/
  constants.py   Fermi/Bose, unit conversions (hbar = e = 1, eV)
  leads.py       Sancho–Rubio surface GFs (electrons: z=E+iη, phonons: z=ω²+iη), self-energies
  rgf.py         block-tridiagonal RGF: diagonal blocks of G^R, G^<, G^> (explicit Σ^> recursion)
  convolve.py    FFT convolutions on the common grid ω_n = n·dE, Kramers–Kronig
  system.py      BlockSystem (H or K), Lead, Coupling, 1D chain builders
  scba.py        CoupledSCBA solver + observables (currents, heat flows, site-resolved heating)
tests/test_core.py      6 tests (RGF vs dense, leads vs analytic, conservation, regression)
examples/chain_heating.py  6-site wire: IETS + heating profile + energy balance
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

## Run
```
python tests/test_core.py
python examples/chain_heating.py
```

## Roadmap
1. ✅ 1D single level, Einstein phonon (legacy/negf_eph_1d.py)
2. ✅ phonon NEGF + phonon leads, coupled (legacy/negf_eph_1d_stage2.py)
3. ✅ package: block RGF, Sancho–Rubio leads, multi-site devices (this version)
4. ☐ non-orthogonal basis (S matrix) + sisl reader for SIESTA/TranSIESTA H, S, Σ_lead
5. ☐ Wannier90 `_hr.dat` + phonopy force constants + e-ph matrices (EPW / finite differences)
6. ☐ non-local (block) self-energies, k-point sampling for 2D/3D leads
7. ☐ performance: Numba/CuPy kernels, MPI over energies, Anderson mixing
8. ☐ Hartree / Poisson feedback, thermoelectric and phonon-drag outputs
