#!/usr/bin/env python3
"""
negf_eph_1d.py
==============
Self-consistent Born approximation (SCBA) NEGF for a 1D tight-binding junction
with local Einstein phonons (Holstein coupling  M n_i (b_i + b_i^dag)).

  left lead (1D chain) -- device (N sites, eps0, t_dev) -- right lead (1D chain)

Stages implemented
  * ballistic Landauer (M = 0)
  * Born (non-self-consistent, Sigma[G0])           -> shows current NON-conservation
  * SCBA with equilibrium phonons (N = N_Bose(T))
  * SCBA with non-equilibrium phonons: mode occupation N_i solved self-consistently
    from emission/absorption rates + weak damping gamma_bath to an external bath
    (= weak-damping, delta-spectral limit of the phonon NEGF; full D^<> with
    phonon leads is the next stage)

Units / conventions
  energies in eV, hbar = 1, e = 1;  G< = i f A,  G> = -i (1-f) A
  Sigma_a< = i f_a Gamma_a,  Sigma_a> = -i (1-f_a) Gamma_a
  Particle current INTO the device from lead a:
      I_a = s * int dE/2pi Tr[Sigma_a< G> - Sigma_a> G<]
  Hartree (static polaron) term of the Holstein model is omitted.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.signal import hilbert

KB = 8.617333262e-5          # eV/K
E_OVER_H_UA = 38.7404585     # (e/h)*1 eV  in microampere
EV2_OVER_HBAR_UW = 243.4135  # (1 eV)^2/hbar in microwatt


def fermi(E, mu, T):
    return 0.5 * (1.0 - np.tanh(0.5 * (E - mu) / (KB * T)))


def bose(w, T):
    return 1.0 / np.expm1(w / (KB * T))


def lead_surface_g(E, t, eta):
    """Retarded surface GF of a semi-infinite 1D chain (onsite 0, hopping t)."""
    z = E + 1j * eta
    root = np.sqrt(z * z - 4 * t * t)
    g = (z - root) / (2 * t * t)
    other = (z + root) / (2 * t * t)
    return np.where(g.imag <= 0, g, other)


def shift(X, n):
    """Y(E) = X(E - n*dE) on the uniform grid (zero fill at the edges)."""
    if n == 0:
        return X.copy()
    Y = np.zeros_like(X)
    if n > 0:
        Y[n:] = X[:-n]
    else:
        Y[:n] = X[-n:]
    return Y


def kramers_kronig_real(im):
    """Re f = -H[Im f] for a retarded function (zero-padded FFT Hilbert transform)."""
    nE = im.shape[0]
    pad = np.zeros((2 * nE,) + im.shape[1:])
    pad[nE // 2: nE // 2 + nE] = im
    h = np.imag(hilbert(pad, axis=0))[nE // 2: nE // 2 + nE]
    return -h


@dataclass
class Params:
    N: int = 1                 # device sites (all carry a local Einstein mode)
    eps0: float = 1.0          # device on-site energy (eV)
    t_dev: float = 1.0         # device hopping (eV)
    t_lead: float = 2.0        # lead hopping (eV)  -> band [-4, 4] eV
    tc: float = 0.6            # device-lead coupling (eV)
    omega0: float = 0.10       # phonon energy (eV)
    M: float = 0.15            # e-ph coupling (eV)
    T: float = 10.0            # temperature (K)
    gamma_bath: float = 2e-4   # phonon damping to external bath (eV), noneq mode
    Emin: float = -4.5
    Emax: float = 4.5
    dE: float = 5e-4           # adjusted so omega0/dE is an integer
    eta: float = 1e-7
    spin: int = 2
    mix: float = 0.5
    tol: float = 1e-10
    maxit: int = 400
    real_part: bool = True     # include Re Sigma^R via Kramers-Kronig
    phonons: str = "equilibrium"   # "equilibrium" | "nonequilibrium"
    born_only: bool = False    # non-self-consistent first Born approximation


class Junction:
    def __init__(self, p: Params):
        self.p = p
        self.nw = max(1, int(round(p.omega0 / p.dE)))
        self.dE = p.omega0 / self.nw
        self.E = np.arange(p.Emin, p.Emax + 0.5 * self.dE, self.dE)
        nE, N = len(self.E), p.N
        self.idx = np.arange(N)
        H = np.diag(np.full(N, p.eps0)) - p.t_dev * (np.eye(N, k=1) + np.eye(N, k=-1))
        g = lead_surface_g(self.E, p.t_lead, p.eta)
        self.SL = np.zeros((nE, N, N), complex)
        self.SR = np.zeros((nE, N, N), complex)
        self.SL[:, 0, 0] = p.tc ** 2 * g
        self.SR[:, N - 1, N - 1] = p.tc ** 2 * g
        self.GamL = 1j * (self.SL - self.SL.conj().transpose(0, 2, 1))
        self.GamR = 1j * (self.SR - self.SR.conj().transpose(0, 2, 1))
        self.A0 = (self.E + 1j * p.eta)[:, None, None] * np.eye(N) - H

    # ---- phonon pieces -------------------------------------------------
    def _diag(self, d):
        S = np.zeros((len(self.E), self.p.N, self.p.N), complex)
        S[:, self.idx, self.idx] = d
        return S

    def sigma_ph(self, Gl, Gg, Nph):
        p, nw = self.p, self.nw
        gl = np.diagonal(Gl, axis1=1, axis2=2)
        gg = np.diagonal(Gg, axis1=1, axis2=2)
        M2 = p.M ** 2
        sl = M2 * (Nph * shift(gl, +nw) + (Nph + 1) * shift(gl, -nw))   # G<(E-w), G<(E+w)
        sg = M2 * (Nph * shift(gg, -nw) + (Nph + 1) * shift(gg, +nw))   # G>(E+w), G>(E-w)
        im_r = -0.5 * np.real(1j * (sg - sl))                            # Im Sigma^R = -Gamma/2
        re_r = kramers_kronig_real(im_r) if p.real_part else 0.0
        return self._diag(re_r + 1j * im_r), self._diag(sl), self._diag(sg)

    def rates(self, Gl, Gg):
        """Phonon emission / absorption kernels K_em, K_abs per site (eV = hbar*rate)."""
        p, nw = self.p, self.nw
        n = np.real(-1j * np.diagonal(Gl, axis1=1, axis2=2))
        nb = np.real(1j * np.diagonal(Gg, axis1=1, axis2=2))
        pref = p.spin * p.M ** 2 / (2 * np.pi) * self.dE
        K_em = pref * np.sum(n * shift(nb, +nw), axis=0)    # e: E -> E-w, emits
        K_abs = pref * np.sum(n * shift(nb, -nw), axis=0)   # e: E -> E+w, absorbs
        return K_em, K_abs

    # ---- solver ---------------------------------------------------------
    def solve(self, V):
        p, E, N = self.p, self.E, self.p.N
        fL = fermi(E, +0.5 * V, p.T)[:, None, None]
        fR = fermi(E, -0.5 * V, p.T)[:, None, None]
        NB = bose(p.omega0, p.T)
        Nph = np.full(N, NB)
        z = np.zeros((len(E), N, N), complex)
        Sr, Sl, Sg = z, z, z
        hist = []
        coupled = p.M != 0.0
        for it in range(p.maxit):
            Gr = np.linalg.inv(self.A0 - self.SL - self.SR - Sr)
            Ga = Gr.conj().transpose(0, 2, 1)
            SigL = 1j * (fL * self.GamL + fR * self.GamR) + Sl
            SigG = -1j * ((1 - fL) * self.GamL + (1 - fR) * self.GamR) + Sg
            Gl = Gr @ SigL @ Ga
            Gg = Gr @ SigG @ Ga
            if not coupled or (p.born_only and it == 1):
                break
            if p.phonons == "nonequilibrium":
                K_em, K_abs = self.rates(Gl, Gg)
                N_new = (K_em + p.gamma_bath * NB) / (K_abs - K_em + p.gamma_bath)
                Nph = N_new if it == 0 else (1 - p.mix) * Nph + p.mix * N_new
            Sr_n, Sl_n, Sg_n = self.sigma_ph(Gl, Gg, Nph)
            err = np.max(np.abs(Sl_n - Sl)) / (np.max(np.abs(Sl_n)) + 1e-300)
            hist.append(err)
            a = 1.0 if (it == 0 or p.born_only) else p.mix
            Sr = (1 - a) * Sr + a * Sr_n
            Sl = (1 - a) * Sl + a * Sl_n
            Sg = (1 - a) * Sg + a * Sg_n
            if it > 0 and err < p.tol:
                # one final G with converged Sigma
                Gr = np.linalg.inv(self.A0 - self.SL - self.SR - Sr)
                Ga = Gr.conj().transpose(0, 2, 1)
                SigL = 1j * (fL * self.GamL + fR * self.GamR) + Sl
                SigG = -1j * ((1 - fL) * self.GamL + (1 - fR) * self.GamR) + Sg
                Gl, Gg = Gr @ SigL @ Ga, Gr @ SigG @ Ga
                break
        return self._observables(V, fL, fR, Gr, Gl, Gg, Nph, NB, hist)

    def _observables(self, V, fL, fR, Gr, Gl, Gg, Nph, NB, hist):
        p, E, dE = self.p, self.E, self.dE

        def X(f, Gam):
            Sl_a, Sg_a = 1j * f * Gam, -1j * (1 - f) * Gam
            return np.real(np.trace(Sl_a @ Gg - Sg_a @ Gl, axis1=1, axis2=2))

        XL, XR = X(fL, self.GamL), X(fR, self.GamR)
        I_L = p.spin * np.sum(XL) * dE / (2 * np.pi)          # e/hbar units
        I_R = p.spin * np.sum(XR) * dE / (2 * np.pi)
        P_e = p.spin * np.sum(E * (XL + XR)) * dE / (2 * np.pi)  # power into electrons from leads
        if p.M != 0.0:
            K_em, K_abs = self.rates(Gl, Gg)
            P_ph = float(np.sum(p.omega0 * ((Nph + 1) * K_em - Nph * K_abs)))
        else:
            P_ph = 0.0
        A = -2 * np.imag(np.trace(Gr, axis1=1, axis2=2))
        return dict(V=V, I_uA=I_L * 2 * np.pi * E_OVER_H_UA, I_L=I_L, I_R=I_R,
                    P_e_uW=P_e * EV2_OVER_HBAR_UW, P_ph_uW=P_ph * EV2_OVER_HBAR_UW,
                    Nph=Nph.copy(), NB=NB, A=A, iters=len(hist), hist=hist)


def bias_sweep(p, Vs):
    J = Junction(p)
    out = [J.solve(V) for V in Vs]
    return J, out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Params()
    Vs = np.arange(0.0, 0.2501, 0.002)
    runs = {
        "ballistic (M=0)": replace(base, M=0.0),
        "SCBA, equilibrium phonon": replace(base, phonons="equilibrium"),
        "SCBA, non-equilibrium phonon": replace(base, phonons="nonequilibrium"),
    }
    res = {}
    for name, p in runs.items():
        J, r = bias_sweep(p, Vs)
        res[name] = r
        print(f"{name:32s} done  (dE = {J.dE*1e3:.3f} meV, nE = {len(J.E)})")

    # Born vs SCBA conservation
    Vc = np.arange(0.02, 0.2501, 0.02)
    _, born = bias_sweep(replace(base, born_only=True), Vc)
    _, scba = bias_sweep(base, Vc)

    def viol(r):
        return np.array([abs(x["I_L"] + x["I_R"]) / abs(x["I_L"]) for x in r])

    print("\n  V(V)   |I_L+I_R|/|I_L| Born    SCBA      P_e(uW)   P_ph(uW)  [SCBA eq]")
    for vb, vs, x in zip(viol(born), viol(scba), scba):
        print(f"  {x['V']:.2f}   {vb:12.3e}        {vs:9.2e}  {x['P_e_uW']:9.4f} {x['P_ph_uW']:9.4f}")
    print("\nNon-eq phonon energy balance:")
    for x in res["SCBA, non-equilibrium phonon"][::25]:
        print(f"  V={x['V']:.3f}  N={x['Nph'][0]:.4f}  P_e={x['P_e_uW']:.5f} uW  "
              f"P_ph={x['P_ph_uW']:.5f} uW  iters={x['iters']}")

    # ---- Figure 1: transport + IETS + heating ----
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    for (name, r), c in zip(res.items(), ("C0", "C1", "C2")):
        I = np.array([x["I_uA"] for x in r])
        G = np.gradient(I, Vs)
        d2 = np.gradient(G, Vs)
        ax[0, 0].plot(Vs, I, c=c, label=name)
        ax[0, 1].plot(Vs, G, c=c, label=name)
        if "M=0" not in name:
            ax[1, 0].plot(Vs, d2 / G, c=c, label=name)
    ax[0, 0].set(xlabel="V (V)", ylabel="I (µA)", title="(a) I–V")
    ax[0, 1].set(xlabel="V (V)", ylabel="dI/dV (µA/V)", title="(b) differential conductance")
    ax[1, 0].set(xlabel="V (V)", ylabel=r"$(d^2I/dV^2)/(dI/dV)$ (1/V)", title="(c) IETS")
    for a in (ax[0, 1], ax[1, 0]):
        a.axvline(base.omega0, ls=":", c="k", lw=1)
    ax[0, 0].legend(fontsize=8)
    Nn = np.array([x["Nph"][0] for x in res["SCBA, non-equilibrium phonon"]])
    ax[1, 1].plot(Vs, Nn, c="C2", label="self-consistent N(V)")
    ax[1, 1].axhline(bose(base.omega0, base.T), ls="--", c="k", lw=1, label="Bose N(T)")
    ax[1, 1].axvline(base.omega0, ls=":", c="k", lw=1)
    ax[1, 1].set(xlabel="V (V)", ylabel="phonon occupation N", title="(d) current-induced heating")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle(rf"Single level $\varepsilon_0$={base.eps0} eV, $\hbar\omega_0$={base.omega0} eV, "
                 rf"M={base.M} eV, T={base.T} K", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig1_transport_iets.png", dpi=150)

    # ---- Figure 2: sidebands + conservation + convergence ----
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ps = replace(base, eps0=0.0, tc=0.3, omega0=0.2, M=0.2)
    for M, ls in ((0.0, "--"), (0.2, "-")):
        J = Junction(replace(ps, M=M))
        r = J.solve(0.0)
        m = (J.E > -1) & (J.E < 1)
        ax[0].plot(J.E[m], r["A"][m], ls, label=f"M = {M} eV")
    ax[0].set(xlabel="E (eV)", ylabel="A(E)", title=r"(a) phonon sidebands ($\hbar\omega_0$=0.2 eV)")
    ax[0].legend()
    ax[1].semilogy(Vc, viol(born), "o-", label="first Born (non-SC)")
    ax[1].semilogy(Vc, np.maximum(viol(scba), 1e-16), "s-", label="SCBA")
    ax[1].set(xlabel="V (V)", ylabel=r"$|I_L+I_R|/|I_L|$", title="(b) current conservation")
    ax[1].legend()
    J = Junction(base)
    r = J.solve(0.2)
    ax[2].semilogy(r["hist"], "o-", ms=3)
    ax[2].set(xlabel="SCBA iteration", ylabel=r"max $|\Delta\Sigma^<|/|\Sigma^<|$",
              title="(c) SCBA convergence (V = 0.2 V)")
    fig.tight_layout()
    fig.savefig("fig2_sidebands_conservation.png", dpi=150)
    print("\nSaved fig1_transport_iets.png, fig2_sidebands_conservation.png")
