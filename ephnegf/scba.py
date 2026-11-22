"""Self-consistent Born approximation for coupled electron + phonon NEGF.

Electrons and phonons are both block-tridiagonal and solved with RGF; the e-ph
self-energies are LOCAL (diagonal in orbitals / phonon dofs). The local SCBA is
Phi-derivable, hence particle- and energy-conserving.

    Sigma_o^<>(E) = i g_o^2 int_0^inf dw/2pi [ D_aa^<>(w) G_oo^<>(E-w) + D_aa^><(w) G_oo^<>(E+w) ]
    Pi_a^<>(w)    = -i s sum_{o->a} g_o^2 int dE/2pi  G_oo^<>(E) G_oo^><(E-w)
    Sigma^R = (Sigma^> - Sigma^<)/2 + i*0 + Re via Kramers-Kronig
    Pi^R    = (Pi^>    - Pi^<)/2        (Re Pi^R off by default: already in DFT force constants)
"""
from dataclasses import dataclass
import numpy as np
from .constants import fermi, bose, E_OVER_H_UA, EV2_OVER_HBAR_UW
from .leads import lead_selfenergy, broadening
from .rgf import rgf_general
from .convolve import conv_fwd, conv_bwd, corr, kramers_kronig_real
from .system import BlockSystem, Coupling


@dataclass
class Settings:
    T: float = 10.0               # electron-lead temperature (K)
    T_ph_L: float = None          # phonon-lead temperatures (default: T)
    T_ph_R: float = None
    Emin: float = -4.5
    Emax: float = 4.5
    dE: float = 5e-4              # common grid spacing; phonon grid w_n = n dE
    wmax: float = 0.20
    eta: float = 1e-7
    eta_ph: float = 1e-6
    spin: int = 2
    mix: float = 0.5
    tol: float = 1e-9
    maxit: int = 300
    re_sigma: bool = True
    re_pi: bool = False


def _blocks_from_diag(d, offsets):
    """(nX, n) diagonal values -> list of (nX, b, b) diagonal blocks."""
    out = []
    for i in range(len(offsets) - 1):
        b = offsets[i + 1] - offsets[i]
        B = np.zeros((d.shape[0], b, b), complex)
        B[:, np.arange(b), np.arange(b)] = d[:, offsets[i]:offsets[i + 1]]
        out.append(B)
    return out


def _diag_from_blocks(blocks):
    return np.concatenate([np.diagonal(B, axis1=1, axis2=2) for B in blocks], axis=1)


class CoupledSCBA:
    def __init__(self, el: BlockSystem, ph: BlockSystem, cp: Coupling, s: Settings = Settings()):
        self.el, self.ph, self.cp, self.s = el, ph, cp, s
        self.E = np.arange(s.Emin, s.Emax + 0.5 * s.dE, s.dE)
        self.dE = s.dE
        self.nmax = int(round(s.wmax / s.dE))
        self.w = np.arange(self.nmax + 1) * s.dE
        self.eo, self.po = el.offsets, ph.offsets
        self.g2 = np.asarray(cp.g, float) ** 2
        self.dof = np.asarray(cp.dof, int)
        # --- electron leads (edge blocks)
        zE = self.E + 1j * s.eta
        self.SL = lead_selfenergy(zE, el.left)
        self.SR = lead_selfenergy(zE, el.right)
        self.GamL, self.GamR = broadening(self.SL), broadening(self.SR)
        self.Ad0_el, self.U_el, self.L_el = el.a_blocks(zE)
        self.Ad0_el[0] = self.Ad0_el[0] - self.SL
        self.Ad0_el[-1] = self.Ad0_el[-1] - self.SR
        # --- phonon leads (edge blocks), w > 0 only
        wp = self.w[1:]
        zW = (wp + 1j * s.eta_ph) ** 2
        zlead = wp ** 2 + 1e-16j
        self.PL = lead_selfenergy(zlead, ph.left)
        self.PR = lead_selfenergy(zlead, ph.right)
        self.LamL, self.LamR = broadening(self.PL), broadening(self.PR)
        self.Ad0_ph, self.U_ph, self.L_ph = ph.a_blocks(zW)
        self.Ad0_ph[0] = self.Ad0_ph[0] - self.PL
        self.Ad0_ph[-1] = self.Ad0_ph[-1] - self.PR
        TphL = s.T if s.T_ph_L is None else s.T_ph_L
        TphR = s.T if s.T_ph_R is None else s.T_ph_R
        self.NphL = bose(wp, TphL)[:, None, None]
        self.NphR = bose(wp, TphR)[:, None, None]

    # ------------------------------------------------------------ pieces
    def electron_greens(self, Sr, Sl, Sg, fL, fR):
        """Sr, Sl, Sg: (nE, n_orb) local self-energies."""
        Srb = _blocks_from_diag(Sr, self.eo)
        Ad = [A - S for A, S in zip(self.Ad0_el, Srb)]
        Slb = _blocks_from_diag(Sl, self.eo)
        Sgb = _blocks_from_diag(Sg, self.eo)
        fL, fR = fL[:, None, None], fR[:, None, None]
        Slb[0] = Slb[0] + 1j * fL * self.GamL
        Slb[-1] = Slb[-1] + 1j * fR * self.GamR
        Sgb[0] = Sgb[0] - 1j * (1 - fL) * self.GamL
        Sgb[-1] = Sgb[-1] - 1j * (1 - fR) * self.GamR
        return rgf_general(Ad, self.U_el, self.L_el, Slb, Sgb)

    def phonon_greens(self, Pr, Pl, Pg):
        """Pr, Pl, Pg: (nw, n_dof) local polarisations (w = 0 row ignored)."""
        Prb = _blocks_from_diag(Pr[1:], self.po)
        Ad = [A - P for A, P in zip(self.Ad0_ph, Prb)]
        Plb = _blocks_from_diag(Pl[1:], self.po)
        Pgb = _blocks_from_diag(Pg[1:], self.po)
        Plb[0] = Plb[0] - 1j * self.NphL * self.LamL
        Plb[-1] = Plb[-1] - 1j * self.NphR * self.LamR
        Pgb[0] = Pgb[0] - 1j * (self.NphL + 1) * self.LamL
        Pgb[-1] = Pgb[-1] - 1j * (self.NphR + 1) * self.LamR
        DR, Dl, Dg = rgf_general(Ad, self.U_ph, self.L_ph, Plb, Pgb)
        pad = lambda L: [np.concatenate([np.zeros((1,) + B.shape[1:], complex), B]) for B in L]
        return pad(DR), pad(Dl), pad(Dg)

    def polarisation(self, gl, gg):
        """gl, gg: (nE, n_orb) diagonal G^<, G^>.  Returns Pi^R, Pi^<, Pi^> (nw, n_dof)."""
        c = -1j * self.s.spin * self.dE / (2 * np.pi) * self.g2
        pl_o = c * corr(gl, gg, self.nmax)
        pg_o = c * corr(gg, gl, self.nmax)
        nd = self.ph.n
        Pl = np.zeros((self.nmax + 1, nd), complex)
        Pg = np.zeros_like(Pl)
        for o, a in enumerate(self.dof):
            Pl[:, a] += pl_o[:, o]
            Pg[:, a] += pg_o[:, o]
        Pl[0] = Pg[0] = 0.0
        Pr = 0.5 * (Pg - Pl)
        if self.s.re_pi:
            Pr = Pr + kramers_kronig_real(np.imag(Pr))
        return Pr, Pl, Pg

    def self_energy(self, gl, gg, dl, dg):
        """gl, gg (nE, n_orb); dl, dg (nw, n_dof).  Returns Sigma^R, Sigma^<, Sigma^>."""
        c = 1j * self.dE / (2 * np.pi) * self.g2
        Dl, Dg = dl[:, self.dof], dg[:, self.dof]
        Sl = c * (conv_fwd(gl, Dl) + conv_bwd(gl, Dg))
        Sg = c * (conv_fwd(gg, Dg) + conv_bwd(gg, Dl))
        Sr = 0.5 * (Sg - Sl)
        if self.s.re_sigma:
            Sr = Sr + kramers_kronig_real(np.imag(Sr))
        return Sr, Sl, Sg

    # ------------------------------------------------------------- solve
    def solve(self, V, guess=None):
        s, E = self.s, self.E
        fL, fR = fermi(E, +0.5 * V, s.T), fermi(E, -0.5 * V, s.T)
        n_orb = self.el.n
        if guess is None:
            Sr = np.zeros((len(E), n_orb), complex); Sl = Sr.copy(); Sg = Sr.copy()
        else:
            Sr, Sl, Sg = guess
        hist = []
        for it in range(s.maxit):
            GR, Gl, Gg = self.electron_greens(Sr, Sl, Sg, fL, fR)
            gl, gg = _diag_from_blocks(Gl), _diag_from_blocks(Gg)
            Pr, Pl, Pg = self.polarisation(gl, gg)
            DR, Dl, Dg = self.phonon_greens(Pr, Pl, Pg)
            Sr_n, Sl_n, Sg_n = self.self_energy(gl, gg, _diag_from_blocks(Dl), _diag_from_blocks(Dg))
            err = np.max(np.abs(Sl_n - Sl)) / (np.max(np.abs(Sl_n)) + 1e-300)
            hist.append(err)
            a = 1.0 if (it == 0 and guess is None) else s.mix
            Sr, Sl, Sg = [(1 - a) * o + a * n for o, n in ((Sr, Sr_n), (Sl, Sl_n), (Sg, Sg_n))]
            if it > 0 and err < s.tol:
                break
        GR, Gl, Gg = self.electron_greens(Sr, Sl, Sg, fL, fR)
        gl, gg = _diag_from_blocks(Gl), _diag_from_blocks(Gg)
        Pr, Pl, Pg = self.polarisation(gl, gg)
        DR, Dl, Dg = self.phonon_greens(Pr, Pl, Pg)
        out = self.observables(V, fL, fR, GR, Gl, Gg, DR, Dl, Dg, Pl, Pg)
        out.update(iters=len(hist), hist=hist, guess=(Sr, Sl, Sg))
        return out

    # ------------------------------------------------------- observables
    def observables(self, V, fL, fR, GR, Gl, Gg, DR, Dl, Dg, Pl, Pg):
        s, E, dE, w = self.s, self.E, self.dE, self.w
        tr = lambda X: np.real(np.trace(X, axis1=1, axis2=2))

        def X(f, Gam, Gl_b, Gg_b):
            f = f[:, None, None]
            return tr(1j * f * Gam @ Gg_b + 1j * (1 - f) * Gam @ Gl_b)

        XL = X(fL, self.GamL, Gl[0], Gg[0])
        XR = X(fR, self.GamR, Gl[-1], Gg[-1])
        k = s.spin * dE / (2 * np.pi)
        I_L, I_R = k * XL.sum(), k * XR.sum()
        P_e = k * np.sum(E * (XL + XR))

        def Jph(Lam, N, Dl_b, Dg_b):
            Pl_a, Pg_a = -1j * N * Lam, -1j * (N + 1) * Lam
            return -np.sum(w[1:] * tr(Pl_a @ Dg_b[1:] - Pg_a @ Dl_b[1:])) * dE / (2 * np.pi)

        J_L = Jph(self.LamL, self.NphL, Dl[0], Dg[0])
        J_R = Jph(self.LamR, self.NphR, Dl[-1], Dg[-1])
        dl, dg = _diag_from_blocks(Dl), _diag_from_blocks(Dg)
        J_e = -np.sum(w[:, None] * np.real(Pl * dg - Pg * dl)) * dE / (2 * np.pi)
        # mode-energy-weighted phonon occupation per dof
        occ = np.real(1j * dl)
        spec = np.real(1j * (dg - dl))
        N_dof = np.sum(w[:, None] * occ, 0) / np.maximum(np.sum(w[:, None] * spec, 0), 1e-300)
        n_el = np.sum(np.real(-1j * _diag_from_blocks(Gl)), 0) * dE / (2 * np.pi)
        A = -2 * np.imag(_diag_from_blocks(GR)).sum(1)
        u = EV2_OVER_HBAR_UW
        return dict(V=V, I_uA=I_L * 2 * np.pi * E_OVER_H_UA, I_L=I_L, I_R=I_R,
                    P_e_uW=P_e * u, J_L_uW=J_L * u, J_R_uW=J_R * u, J_e_uW=J_e * u,
                    N_dof=N_dof, n_el=n_el, A=A,
                    phonon_dos=-2 * w[:, None] * np.imag(_diag_from_blocks(DR)) / np.pi)
