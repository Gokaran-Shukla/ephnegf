"""Stage 5 demo: monatomic carbon chain in REAL units.

  phonons : force constants 30 eV/A^2, m = 12 amu  -> band top ~0.20 eV (1650 cm^-1)
  electrons: t(d) = 2.5 eV + 4 eV/A * (bond stretch)   (SSH-type, bond-modulated hopping)
  e-ph    : M^lambda by finite differences of H(x)  (exactly what will be done with
            SIESTA Hamiltonians at +/- displaced geometries)
  SCBA    : block-local, electron blocks = phonon groups = 4 atoms
Run:  python examples/carbon_chain_ssh.py     (~3 min)
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ephnegf as ep
from ephnegf.system import Lead, from_dense_blocks, chain_blocks
from ephnegf.coupling import from_dense, phonon_groups, eph_finite_difference
from ephnegf.units import EV_TO_CM1
from test_phonons_eph import ssh_H, chain_phonon_system, T0, MASS

n, bs = 14, 4
lead = lambda: Lead(np.array([[0.0]]), np.array([[-T0]]), np.array([[-T0]]))
el = from_dense_blocks(ssh_H(n), chain_blocks(n, bs), lead(), lead())
ph = chain_phonon_system(n, bs)
d = 1e-3
M = np.array([eph_finite_difference(ssh_H(n, d * np.eye(n)[l]), ssh_H(n, -d * np.eye(n)[l]), d, MASS)
              for l in range(n)])
cp = from_dense(M, el, ph, phonon_groups(ph, "block"))
print(f"electron blocks {el.sizes}, phonon blocks {ph.sizes}, "
      f"dropped |M|^2 fraction (inter-block bonds) = {cp.dropped_fraction:.2f}")
s = ep.Settings(Emin=-5.5, Emax=5.5, dE=2e-3, wmax=0.22, T=50.0)
solver = ep.CoupledSCBA(el, ph, cp, s)

Vs = np.arange(0.0, 0.601, 0.02)
t0, res, guess = time.time(), [], None
for V in Vs:
    r = solver.solve(V, guess); guess = r.pop("guess"); res.append(r)
print(f"{len(Vs)} biases in {time.time()-t0:.0f} s")
for r in res[::10]:
    c = abs(r["I_L"] + r["I_R"]) / abs(r["I_L"]) if r["V"] > 0 else 0
    print(f"V={r['V']:.2f}  I={r['I_uA']:.3f} uA  cons={c:.1e}  P_e={r['P_e_uW']*1e3:.4f} nW  "
          f"heat->leads={-(r['J_L_uW']+r['J_R_uW'])*1e3:.4f} nW")

I = np.array([r["I_uA"] for r in res]); G = np.gradient(I, Vs)
G0 = 2 * ep.constants.E_OVER_H_UA * 1.0        # ballistic uniform chain: T = 1 per spin
inv = np.argsort(ph.order)
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
w = solver.w
dos0 = res[0]["phonon_dos"].sum(1)
ax[0, 0].plot(w * EV_TO_CM1, dos0, c="C0")
ax[0, 0].set(xlabel=r"$\omega$ (cm$^{-1}$)", ylabel="device phonon DOS",
             title="(a) phonons from force constants (real units)")
ax[0, 1].plot(Vs, (G - G0) / G0 * 100, "o-", ms=3, c="C3")
ax[0, 1].axvline(w[-1] * 0 + 0.2045, ls=":", c="k", lw=1)
ax[0, 1].set(xlabel="V (V)", ylabel=r"$(G-G_0)/G_0$ (%)", title="(b) inelastic conductance drop")
for j, r in enumerate(res[5::5]):
    ax[1, 0].plot(np.arange(n), r["N_dof"][inv], "o-", ms=3, c=plt.cm.viridis(j / 5),
                  label=f"V = {r['V']:.1f} V")
ax[1, 0].set(xlabel="atom", ylabel="phonon occupation", title="(c) heating profile")
ax[1, 0].legend(fontsize=7)
ax[1, 1].plot(Vs, [r["P_e_uW"] * 1e3 for r in res], c="C0", lw=3, alpha=0.5,
              label="power lost by electrons")
ax[1, 1].plot(Vs, [-(r["J_L_uW"] + r["J_R_uW"]) * 1e3 for r in res], "--", c="C3",
              label="heat into phonon leads")
ax[1, 1].set(xlabel="V (V)", ylabel="power (nW)", title="(d) energy balance")
ax[1, 1].legend(fontsize=8)
fig.suptitle(f"C chain, {n} atoms: k = 30 eV/Å², t = 2.5 eV, ∂t/∂d = 4 eV/Å, T = 50 K", fontsize=10)
fig.tight_layout()
fig.savefig("fig7_carbon_chain.png", dpi=150)
print("Saved fig7_carbon_chain.png")
