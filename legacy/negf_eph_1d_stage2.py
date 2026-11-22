#!/usr/bin/env python3
"""
negf_eph_1d_stage2.py
=====================
Stage 2: fully coupled electron + phonon NEGF (SCBA) with phonon leads.

Electrons : 1D TB leads -- device (N sites) -- 1D TB leads           (as in stage 1)
Phonons   : 1D harmonic-chain leads -- device atoms (N) -- 1D harmonic-chain leads
Coupling  : H_eph = sum_i g_i n_i u_i   (u_i mass-weighted displacement, hbar = 1)
            Einstein-mode limit: g = M * sqrt(2*omega0)  (M = stage-1 coupling)

Self-consistent loop (no fixed phonon occupation, no phenomenological bath):
    G  -> Pi_e^<>(w)  = -i s (g g^T) o int dE/2pi G^<>(E) o G^><(E-w)^T
       -> D^R = [w^2 - K - Pi_L - Pi_R - Pi_e^R]^-1 ,  D^<> = D^R (Pi_leads^<> + Pi_e^<>) D^A
       -> Sigma^<>(E) = i (g g^T) o int_0^inf dw/2pi [D^<>(w) o G^<>(E-w) + D^><(w)^T o G^<>(E+w)]
       -> Sigma^R via Kramers-Kronig -> G ...
All energy convolutions done with FFTs on a common grid (w_n = n dE).

Bosonic conventions: Pi_a^< = -i N_a Lambda_a, Pi_a^> = -i (N_a+1) Lambda_a,
                     Lambda_a = i (Pi_a^R - Pi_a^A)
Heat current INTO the device phonons from bath a (lead L, R, or electrons e):
    J_a = - int_0^inf dw/2pi  w Re Tr[Pi_a^< D^> - Pi_a^> D^<]
Re Pi_e^R is omitted by default (adiabatic renormalisation is already in DFT force
constants -> avoids double counting); Re Sigma^R for electrons is kept.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.signal import fftconvolve
from negf_eph_1d import (fermi, bose, lead_surface_g, kramers_kronig_real,
                         KB, E_OVER_H_UA, EV2_OVER_HBAR_UW, Params as Stage1Params,
                         Junction as Stage1Junction)


# ------------------------------------------------------------------ helpers
def conv_fwd(G, K):
    """out[m] = sum_{n>=1} K[n] G[m-n]   (G: (nE,), K: (nmax+1,), K[0] = 0)"""
    return fftconvolve(G, K, mode="full")[: len(G)]


def conv_bwd(G, K):
    """out[m] = sum_{n>=1} K[n] G[m+n]"""
    nmax = len(K) - 1
    return fftconvolve(G, K[::-1], mode="full")[nmax: nmax + len(G)]


def corr(a, b, nmax):
    """c[n] = sum_m a[m] b[m-n],  n = 0..nmax"""
    nE = len(a)
    full = fftconvolve(a, b[::-1], mode="full")
    return full[nE - 1: nE - 1 + nmax + 1]


def phonon_lead_selfenergy(w, k_lead, k_c, eta=1e-16):
    """Retarded self-energy of a semi-infinite harmonic chain (spring k_lead,
    mass-weighted, units eV^2) attached by spring k_c. Valid for w > 0."""
    x = w ** 2 - 2 * k_lead
    g = lead_surface_g(x, k_lead, eta)       # same algebra as the TB chain
    return k_c ** 2 * g


@dataclass
class Params2:
    # electrons (as stage 1)
    N: int = 1
    eps0: float = 1.0
    t_dev: float = 1.0
    t_lead: float = 2.0
    tc: float = 0.6
    T: float = 10.0
    Emin: float = -4.5
    Emax: float = 4.5
    dE: float = 5e-4
    eta: float = 1e-7
    spin: int = 2
    # phonons
    omega0: float = 0.10       # device mode frequency incl. springs to leads: K_ii = w0^2
    k_dev: float = 0.0         # springs between device atoms (eV^2)
    k_lead: float = 0.0064     # phonon-lead spring -> band top 2*sqrt(k) = 0.16 eV
    k_c: float = 8e-4          # device-lead spring -> mode linewidth ~ 2 meV
    wmax: float = 0.20         # phonon grid cutoff (eV)
    eta_ph: float = 1e-6
    # coupling
    M: float = 0.15            # Einstein-equivalent coupling (eV);  g = M sqrt(2 w0)
    # numerics
    mix: float = 0.5
    tol: float = 1e-9
    maxit: int = 300
    re_sigma: bool = True
    re_pi: bool = False


class CoupledJunction:
    def __init__(self, p: Params2):
        self.p = p
        N = p.N
        # --- electron part: reuse stage-1 construction
        s1 = Stage1Params(N=p.N, eps0=p.eps0, t_dev=p.t_dev, t_lead=p.t_lead, tc=p.tc,
                          T=p.T, Emin=p.Emin, Emax=p.Emax, dE=p.dE, eta=p.eta,
                          omega0=p.dE, M=0.0)
        self.el = Stage1Junction(s1)
        self.E, self.dE = self.el.E, self.el.dE
        # --- phonon grid w_n = n dE, n = 0..nmax  (n = 0 kept as a zero row)
        self.nmax = int(round(p.wmax / self.dE))
        self.w = np.arange(self.nmax + 1) * self.dE
        wp = self.w[1:]
        K = np.diag(np.full(N, p.omega0 ** 2))
        for i in range(N - 1):
            K[i, i] += p.k_dev; K[i + 1, i + 1] += p.k_dev
            K[i, i + 1] -= p.k_dev; K[i + 1, i] -= p.k_dev
        # omega0 is the frequency of the device mode with its springs to the
        # leads included (K_ii = w0^2), so the IETS feature sits at ~omega0.
        self.K = K
        nw = len(self.w)
        self.PiL = np.zeros((nw, N, N), complex)
        self.PiR = np.zeros((nw, N, N), complex)
        sl = phonon_lead_selfenergy(wp, p.k_lead, p.k_c)
        self.PiL[1:, 0, 0] = sl
        self.PiR[1:, N - 1, N - 1] = sl
        self.LamL = 1j * (self.PiL - self.PiL.conj().transpose(0, 2, 1))
        self.LamR = 1j * (self.PiR - self.PiR.conj().transpose(0, 2, 1))
        W = (self.w + 1j * p.eta_ph) ** 2
        self.B0 = W[:, None, None] * np.eye(N) - K
        self.Nbose = np.zeros(nw)
        self.Nbose[1:] = bose(wp, p.T)
        self.g = np.full(N, p.M * np.sqrt(2 * p.omega0))
        self.gg = np.outer(self.g, self.g)

    # ---------------------------------------------------------- phonon side
    def phonon_greens(self, Pi_r, Pi_l, Pi_g):
        DR = np.linalg.inv(self.B0 - self.PiL - self.PiR - Pi_r)
        DR[0] = 0.0
        DA = DR.conj().transpose(0, 2, 1)
        N = self.Nbose[:, None, None]
        Pl = -1j * N * (self.LamL + self.LamR) + Pi_l
        Pg = -1j * (N + 1) * (self.LamL + self.LamR) + Pi_g
        return DR, DR @ Pl @ DA, DR @ Pg @ DA

    def pi_electron(self, Gl, Gg):
        p, N, nmax = self.p, self.p.N, self.nmax
        nw = nmax + 1
        Pl = np.zeros((nw, N, N), complex)
        Pg = np.zeros((nw, N, N), complex)
        for i in range(N):
            for j in range(N):
                pref = -1j * p.spin * self.gg[i, j] * self.dE / (2 * np.pi)
                Pl[:, i, j] = pref * corr(Gl[:, i, j], Gg[:, j, i], nmax)
                Pg[:, i, j] = pref * corr(Gg[:, i, j], Gl[:, j, i], nmax)
        Pl[0] = Pg[0] = 0.0
        Pr = 0.5 * (Pg - Pl)                      # anti-Hermitian part: (Pi^R - Pi^A)/2
        if p.re_pi:
            im = np.imag(Pr)
            Pr = Pr + kramers_kronig_real(im)     # crude; off by default
        return Pr, Pl, Pg

    # -------------------------------------------------------- electron side
    def sigma_electron(self, Gl, Gg, Dl, Dg):
        p, N = self.p, self.p.N
        nE = len(self.E)
        Sl = np.zeros((nE, N, N), complex)
        Sg = np.zeros((nE, N, N), complex)
        c = 1j * self.dE / (2 * np.pi)
        for i in range(N):
            for j in range(N):
                gij = self.gg[i, j]
                # D^<(-w)_ij = D^>(w)_ji
                Sl[:, i, j] = c * gij * (conv_fwd(Gl[:, i, j], Dl[:, i, j]) +
                                         conv_bwd(Gl[:, i, j], Dg[:, j, i]))
                Sg[:, i, j] = c * gij * (conv_fwd(Gg[:, i, j], Dg[:, i, j]) +
                                         conv_bwd(Gg[:, i, j], Dl[:, j, i]))
        Sr = 0.5 * (Sg - Sl)                      # (Sigma^R - Sigma^A)/2
        if p.re_sigma:
            Sr = Sr + kramers_kronig_real(np.imag(Sr))
        return Sr, Sl, Sg

    def electron_greens(self, Sr, Sl, Sg, fL, fR):
        el = self.el
        Gr = np.linalg.inv(el.A0 - el.SL - el.SR - Sr)
        Ga = Gr.conj().transpose(0, 2, 1)
        SigL = 1j * (fL * el.GamL + fR * el.GamR) + Sl
        SigG = -1j * ((1 - fL) * el.GamL + (1 - fR) * el.GamR) + Sg
        return Gr, Gr @ SigL @ Ga, Gr @ SigG @ Ga

    # --------------------------------------------------------------- solve
    def solve(self, V, Sigma0=None):
        p, E, N = self.p, self.E, self.p.N
        fL = fermi(E, +0.5 * V, p.T)[:, None, None]
        fR = fermi(E, -0.5 * V, p.T)[:, None, None]
        z = np.zeros((len(E), N, N), complex)
        Sr, Sl, Sg = Sigma0 if Sigma0 is not None else (z, z, z)
        hist = []
        for it in range(p.maxit):
            Gr, Gl, Gg = self.electron_greens(Sr, Sl, Sg, fL, fR)
            Pr, Pl, Pg = self.pi_electron(Gl, Gg)
            DR, Dl, Dg = self.phonon_greens(Pr, Pl, Pg)
            Sr_n, Sl_n, Sg_n = self.sigma_electron(Gl, Gg, Dl, Dg)
            err = np.max(np.abs(Sl_n - Sl)) / (np.max(np.abs(Sl_n)) + 1e-300)
            hist.append(err)
            a = 1.0 if (it == 0 and Sigma0 is None) else p.mix
            Sr, Sl, Sg = [(1 - a) * o + a * n for o, n in ((Sr, Sr_n), (Sl, Sl_n), (Sg, Sg_n))]
            if it > 0 and err < p.tol:
                break
        # final consistent pass
        Gr, Gl, Gg = self.electron_greens(Sr, Sl, Sg, fL, fR)
        Pr, Pl, Pg = self.pi_electron(Gl, Gg)
        DR, Dl, Dg = self.phonon_greens(Pr, Pl, Pg)
        obs = self._observables(V, fL, fR, Gr, Gl, Gg, DR, Dl, Dg, Pl, Pg, hist)
        obs["Sigma"] = (Sr, Sl, Sg)
        return obs

    def _observables(self, V, fL, fR, Gr, Gl, Gg, DR, Dl, Dg, Pl_e, Pg_e, hist):
        p, el, E, dE, w = self.p, self.el, self.E, self.dE, self.w

        def X(f, Gam):
            return np.real(np.trace(1j * f * Gam @ Gg + 1j * (1 - f) * Gam @ Gl,
                                    axis1=1, axis2=2))

        XL, XR = X(fL, el.GamL), X(fR, el.GamR)
        I_L = p.spin * np.sum(XL) * dE / (2 * np.pi)
        I_R = p.spin * np.sum(XR) * dE / (2 * np.pi)
        P_e = p.spin * np.sum(E * (XL + XR)) * dE / (2 * np.pi)

        Nb = self.Nbose[:, None, None]

        def J(Pl_a, Pg_a):
            integrand = np.real(np.trace(Pl_a @ Dg - Pg_a @ Dl, axis1=1, axis2=2))
            return -np.sum(w * integrand) * dE / (2 * np.pi)

        J_L = J(-1j * Nb * self.LamL, -1j * (Nb + 1) * self.LamL)
        J_R = J(-1j * Nb * self.LamR, -1j * (Nb + 1) * self.LamR)
        J_e = J(Pl_e, Pg_e)
        # effective occupation of the device mode at its spectral peak
        il = np.real(1j * np.trace(Dl, axis1=1, axis2=2))
        ig = np.real(1j * np.trace(Dg, axis1=1, axis2=2))
        Bw = ig - il
        k = np.argmax(Bw[1:]) + 1
        N_eff = il[k] / Bw[k]
        A = -2 * np.imag(np.trace(Gr, axis1=1, axis2=2))
        Bph = -2 * w * np.imag(np.trace(DR, axis1=1, axis2=2)) / np.pi  # phonon DOS
        u = EV2_OVER_HBAR_UW
        return dict(V=V, I_uA=I_L * 2 * np.pi * E_OVER_H_UA, I_L=I_L, I_R=I_R,
                    P_e_uW=P_e * u, J_L_uW=J_L * u, J_R_uW=J_R * u, J_e_uW=J_e * u,
                    N_eff=N_eff, w_peak=w[k], NB_peak=self.Nbose[k],
                    A=A, Bph=Bph, iters=len(hist), hist=hist)


# ------------------------------------------------------- ballistic phonon test
def ballistic_phonon_test(k=0.0064, dw=2e-6, Ts=np.logspace(0, 2.7, 40)):
    """Uniform harmonic chain: T(w) = 1 in the band, kappa -> kappa_0 at low T."""
    w = np.arange(1, int(2 * np.sqrt(k) / dw) + 200) * dw
    Pi = phonon_lead_selfenergy(w, k, k)
    D = 1.0 / ((w + 1e-12j) ** 2 - 2 * k - 2 * Pi)
    Lam = -2 * np.imag(Pi)
    Tw = Lam * np.abs(D) ** 2 * Lam
    kap = []
    for T in Ts:
        x = w / (KB * T)
        dNdT = x * np.exp(-np.clip(x, None, 700)) / (-np.expm1(-x)) ** 2 / T
        kap.append(np.sum(w * Tw * dNdT) * dw / (2 * np.pi))
    kap0 = np.pi ** 2 * KB ** 2 * Ts / (3 * 2 * np.pi)          # pi^2 kB^2 T / 3h, hbar = 1
    return w, Tw, Ts, np.array(kap) / kap0


if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- test 1: ballistic phonon transport
    w_t, Tw, Ts, ratio = ballistic_phonon_test()
    print(f"[ballistic phonons] T(w) in band: min {Tw[w_t < 0.155].min():.6f}, "
          f"max {Tw[w_t < 0.155].max():.6f};  kappa/kappa0 at 1 K = {ratio[0]:.5f}")

    # ---------------- test 2: coupled e-ph bias sweep
    p = Params2()
    Jn = CoupledJunction(p)
    print(f"grid: nE = {len(Jn.E)}, n_w = {len(Jn.w)}, dE = {Jn.dE*1e3:.2f} meV")
    Vs = np.arange(0.0, 0.2501, 0.002)
    t0 = time.time()
    res, S0 = [], None
    for V in Vs:
        r = Jn.solve(V, Sigma0=S0)          # warm start from previous bias
        S0 = r.pop("Sigma")
        res.append(r)
    print(f"coupled sweep: {time.time()-t0:.1f} s, iterations/bias ~ "
          f"{np.mean([r['iters'] for r in res]):.1f}")

    print("\n  V      |I_L+I_R|/|I_L|   P_e(uW)    J_e->ph(uW)  -(J_L+J_R)(uW)  N_eff   N_Bose")
    for r in res[::25]:
        cons = abs(r["I_L"] + r["I_R"]) / abs(r["I_L"]) if r["V"] > 0 else 0.0
        print(f"  {r['V']:.3f}  {cons:12.2e}   {r['P_e_uW']:9.5f}  {r['J_e_uW']:11.5f}  "
              f"{-(r['J_L_uW']+r['J_R_uW']):13.5f}  {r['N_eff']:.4f}  {r['NB_peak']:.2e}")

    # reference: stage-1 SCBA with equilibrium Einstein phonon, same M
    J1 = Stage1Junction(Stage1Params(M=p.M, omega0=p.omega0, T=p.T, dE=p.dE))
    ref = [J1.solve(V) for V in Vs]
    J0 = Stage1Junction(Stage1Params(M=0.0, T=p.T, dE=p.dE))
    bal = [J0.solve(V) for V in Vs]

    # ---------------- figure 3: phonon validation + phonon spectra
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(w_t * 1e3, Tw, c="C0")
    ax[0].set(xlabel=r"$\hbar\omega$ (meV)", ylabel=r"$\mathcal{T}_{ph}(\omega)$",
              title="(a) ballistic phonon transmission, uniform chain", ylim=(0, 1.15))
    ax[1].semilogx(Ts, ratio, "o-", ms=3, c="C0")
    ax[1].axhline(1, ls="--", c="k", lw=1)
    ax[1].set(xlabel="T (K)", ylabel=r"$\kappa/\kappa_0$",
              title=r"(b) thermal conductance vs quantum $\kappa_0=\pi^2k_B^2T/3h$")
    m = (Jn.w > 0.085) & (Jn.w < 0.115)
    Bph0 = -2 * Jn.w * np.imag(np.trace(Jn.phonon_greens(
        *(np.zeros_like(Jn.PiL),) * 3)[0], axis1=1, axis2=2)) / np.pi
    ax[2].plot(Jn.w[m] * 1e3, Bph0[m], "--", c="k", label="phonon leads only")
    for r, c in ((res[0], "C0"), (res[100], "C1")):
        ax[2].plot(Jn.w[m] * 1e3, r["Bph"][m], c=c, label=f"+ electrons, V = {r['V']:.2f} V")
    ax[2].set(xlabel=r"$\hbar\omega$ (meV)", ylabel="device phonon DOS",
              title="(c) mode broadened by e–h pairs")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig3_phonon_validation.png", dpi=150)

    # ---------------- figure 4: coupled transport
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    curves = (("ballistic (M=0)", [r["I_uA"] for r in bal], "C0"),
              ("stage 1: SCBA, equilibrium Einstein phonon", [r["I_uA"] for r in ref], "C1"),
              ("stage 2: coupled e–ph NEGF + phonon leads", [r["I_uA"] for r in res], "C3"))
    for name, I, c in curves:
        I = np.array(I)
        G = np.gradient(I, Vs)
        ax[0, 0].plot(Vs, G, c=c, label=name)
        if "M=0" not in name:
            ax[0, 1].plot(Vs, np.gradient(G, Vs) / G, c=c, label=name)
    ax[0, 0].set(xlabel="V (V)", ylabel="dI/dV (µA/V)", title="(a) differential conductance")
    ax[0, 1].set(xlabel="V (V)", ylabel=r"$(d^2I/dV^2)/(dI/dV)$ (1/V)", title="(b) IETS")
    ax[0, 0].legend(fontsize=7)
    for a in ax[0]:
        a.axvline(p.omega0, ls=":", c="k", lw=1)
    ax[1, 0].plot(Vs, [r["N_eff"] for r in res], c="C3", label=r"$N_{\rm eff}$ at mode peak")
    ax[1, 0].axhline(res[0]["NB_peak"], ls="--", c="k", lw=1, label="Bose N(T)")
    ax[1, 0].axvline(p.omega0, ls=":", c="k", lw=1)
    ax[1, 0].set(xlabel="V (V)", ylabel="phonon occupation",
                 title="(c) heating of the device mode (no fitted bath)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(Vs, [r["P_e_uW"] * 1e3 for r in res], c="C0", lw=3, alpha=0.5,
                  label="power lost by electrons")
    ax[1, 1].plot(Vs, [-(r["J_L_uW"] + r["J_R_uW"]) * 1e3 for r in res], "--", c="C3",
                  label="heat into phonon leads")
    ax[1, 1].set(xlabel="V (V)", ylabel="power (nW)", title="(d) energy balance")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle(rf"$\varepsilon_0$={p.eps0} eV, $\hbar\omega_0$={p.omega0} eV, M={p.M} eV, "
                 rf"T={p.T} K, $k_c$={p.k_c} eV$^2$", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig4_coupled_transport.png", dpi=150)
    print("\nSaved fig3_phonon_validation.png, fig4_coupled_transport.png")
