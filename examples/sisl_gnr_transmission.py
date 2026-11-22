"""Stage 4 demo: non-orthogonal tight-binding zigzag graphene nanoribbon via sisl.

Exactly the same path is used for TranSIESTA output:
    el = from_siesta("device.TSHS", "elecL.TSHS", "elecR.TSHS", axis=2,
                     left_atoms=..., right_atoms=...)
Run:  python examples/sisl_gnr_transmission.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sisl
import ephnegf as ep
from ephnegf.io_sisl import from_sisl

W, ncell = 6, 12
g = sisl.geom.zgnr(W)
H = sisl.Hamiltonian(g, orthogonal=False)
H.construct([[0.1, 1.44], [(0.0, 1.0), (-2.7, 0.11)]])    # (H, S) on-site, nearest neighbour


def device(defect):
    Hd = H.tile(ncell, 0)
    Hd.set_nsc([1, 1, 1])
    if defect:
        for ia in Hd.geometry.close(Hd.geometry.center(), R=1.5)[:1]:
            Hd[ia, ia] = (20.0, 1.0)                      # vacancy-like on-site shift
    lo = list(range(H.no))
    ro = list(range(Hd.no - H.no, Hd.no))
    return from_sisl(Hd, H, H, axis=0, left_orbs=lo, right_orbs=ro)


E = np.linspace(-3.0, 3.0, 1201)
el0, el1 = device(False), device(True)
print(f"device: {el0.n} orbitals -> {len(el0.Hd)} BTD blocks, sizes {el0.sizes}")
T0, T0b = ep.transmission(el0, E, return_both=True)
T1 = ep.transmission(el1, E)
print(f"max |T_L - T_R| (pristine) = {np.max(np.abs(T0 - T0b)):.1e}")

k = np.linspace(-0.5, 0.5, 401)
bands = np.array([H.eigh(k=[kk, 0, 0]) for kk in k])

fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), gridspec_kw=dict(width_ratios=[1, 1.4]), sharey=True)
ax[0].plot(k, bands, c="C0", lw=1)
ax[0].set(xlabel=r"$k$ (2π/a)", ylabel="E (eV)", title=f"(a) {W}-ZGNR bands (non-orthogonal TB)",
          ylim=(E[0], E[-1]))
ax[1].plot(T0, E, c="C0", label="pristine")
ax[1].plot(T1, E, c="C3", label="one vacancy-like site")
ax[1].set(xlabel="T(E)", title="(b) Landauer transmission (block RGF)")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig6_gnr_transmission.png", dpi=150)
print("Saved fig6_gnr_transmission.png")
