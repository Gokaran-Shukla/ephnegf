"""Stage 3 demo: 6-site atomic wire, every site carries a local vibration coupled to
its neighbours and (at the ends) to acoustic phonon leads.

Outputs: IETS, site-resolved phonon heating profile along the wire, energy balance.
Run:  python examples/chain_heating.py      (~2-3 min)
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ephnegf as ep

N = 6
omega0, M = 0.10, 0.10
el = ep.chain_electrons(np.zeros(N), t_dev=1.0, t_lead=2.0, tc=1.0)
ph = ep.chain_phonons(omega0, N, k_dev=2e-3, k_lead=0.0064, k_c=2e-3)
cp = ep.chain_coupling(N, M, omega0)
solver = ep.CoupledSCBA(el, ph, cp, ep.Settings(T=10.0))
solver0 = ep.CoupledSCBA(el, ph, ep.chain_coupling(N, 0.0, omega0), ep.Settings(T=10.0))

Vs = np.arange(0.0, 0.3001, 0.004)
t0 = time.time()
res, guess = [], None
for V in Vs:
    r = solver.solve(V, guess)
    guess = r.pop("guess")
    res.append(r)
bal = [solver0.solve(V) for V in Vs]
print(f"{len(Vs)} bias points, {time.time()-t0:.1f} s, "
      f"<iters> = {np.mean([r['iters'] for r in res]):.1f}")
print("\n   V     |I_L+I_R|/|I_L|   P_e (nW)   heat to leads (nW)   N(site 1..6)")
for r in res[::15]:
    cons = abs(r["I_L"] + r["I_R"]) / abs(r["I_L"]) if r["V"] > 0 else 0.0
    print(f"  {r['V']:.2f}   {cons:10.1e}   {r['P_e_uW']*1e3:9.4f}   "
          f"{-(r['J_L_uW']+r['J_R_uW'])*1e3:9.4f}       "
          + " ".join(f"{n:.4f}" for n in r["N_dof"]))

I = np.array([r["I_uA"] for r in res]); I0 = np.array([r["I_uA"] for r in bal])
G, G0 = np.gradient(I, Vs), np.gradient(I0, Vs)
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
ax[0, 0].plot(Vs, G0, c="C0", label="ballistic (M = 0)")
ax[0, 0].plot(Vs, G, c="C3", label="coupled e–ph SCBA")
ax[0, 0].set(xlabel="V (V)", ylabel="dI/dV (µA/V)", title="(a) differential conductance")
ax[0, 0].legend(fontsize=8)
ax[0, 1].plot(Vs, (G - G0) / G0 * 100, c="C3")
ax[0, 1].set(xlabel="V (V)", ylabel=r"$(G - G_{\rm ball})/G_{\rm ball}$ (%)",
             title="(b) inelastic correction: conductance drop above ħω")
for a in ax[0]:
    a.axvspan(np.sqrt(omega0**2 - 4e-3), np.sqrt(omega0**2 + 4e-3), color="0.9", zorder=0)
sites = np.arange(1, N + 1)
cmap = plt.get_cmap("viridis")
for j, r in enumerate(res[::15]):
    ax[1, 0].plot(sites, r["N_dof"], "o-", c=cmap(j / 5), label=f"V = {r['V']:.2f} V")
ax[1, 0].set(xlabel="site", ylabel="phonon occupation", title="(c) heating profile along the wire")
ax[1, 0].legend(fontsize=8)
ax[1, 1].plot(Vs, [r["P_e_uW"] * 1e3 for r in res], c="C0", lw=3, alpha=0.5,
              label="power lost by electrons")
ax[1, 1].plot(Vs, [-(r["J_L_uW"] + r["J_R_uW"]) * 1e3 for r in res], "--", c="C3",
              label="heat into phonon leads")
ax[1, 1].set(xlabel="V (V)", ylabel="power (nW)", title="(d) energy balance")
ax[1, 1].legend(fontsize=8)
fig.suptitle(f"{N}-site wire, ħω₀ = {omega0} eV (band shaded), M = {M} eV, T = 10 K", fontsize=10)
fig.tight_layout()
fig.savefig("fig5_wire_heating.png", dpi=150)
print("\nSaved fig5_wire_heating.png")
