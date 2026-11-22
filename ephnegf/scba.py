"""Self-consistent Born approximation for coupled electron + phonon NEGF.

Electrons (H, S) and phonons (K) are block-tridiagonal and solved with RGF.
The e-ph coupling H_eph = sum_l M^l u_l is treated block-locally (see coupling.py):

  Sigma_b^<>(E) = i int_0^inf dw/2pi sum_{l,l' in g} M^l [ D_{ll'}^<>(w) G_bb^<>(E-w)
                                                        + D_{l'l}^><(w) G_bb^<>(E+w) ] M^l'
  Pi_{ll'}^<>(w) = -i s sum_b int dE/2pi Tr[ M^l G_bb^<>(E) M^l' G_bb^><(E-w) ]
  Sigma^R = Lambda - i Gamma/2,  Gamma = i(Sigma^> - Sigma^<),  Lambda = Hilbert(Gamma)/2
  Pi^R    = (Pi^> - Pi^<)/2        (Hermitian part off by default: in DFT force constants)

The local Holstein model of stages 1-3 is the special case b = single orbital,
g = single dof (ephnegf.system.Coupling is converted automatically).
"""
from dataclasses import dataclass
import numpy as np
from .constants import fermi, bose, E_OVER_H_UA, EV2_OVER_HBAR_UW
from .leads import lead_selfenergy, broadening
from .rgf import rgf_general
from .convolve import conv_fwd, conv_bwd, corr, kramers_kronig_real
from .system import BlockSystem, Coupling
from .coupling import EPCoupling, from_holstein


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


def _hermitian_from_gamma(Gam):
    """Lambda = (1/2pi) P int Gamma(E')/(E-E') dE'  (elementwise Hilbert transform)."""
    return -0.5 * (kramers_kronig_real(np.real(Gam)) + 1j * kramers_kronig_real(np.imag(Gam)))


def _diag(blocks):
    return np.concatenate([np.diagonal(B, axis1=1, axis2=2) for B in blocks], axis=1)


class CoupledSCBA:
    def __init__(self, el: BlockSystem, ph: BlockSystem, cp, s: Settings = Settings()):
        self.el, self.ph, self.s = el, ph, s
        self.cp = from_holstein(cp, el, ph) if isinstance(cp, Coupling) else cp
        self.E = np.arange(s.Emin, s.Emax + 0.5 * s.dE, s.dE)
        self.dE = s.dE
        self.nmax = int(round(s.wmax / s.dE))
        self.w = np.arange(self.nmax + 1) * s.dE
        self.esz, self.psz = el.sizes, ph.sizes
        poff = ph.offsets
        self.gloc = []                       # group -> (phonon block, local indices)
        for g in self.cp.groups:
            p = int(np.searchsorted(poff, g[0], side="right") - 1)
            if np.any(g >= poff[p + 1]) or np.any(g < poff[p]):
                raise ValueError("a phonon group spans two phonon blocks")
            self.gloc.append((p, np.asarray(g) - poff[p]))
        zE = self.E + 1j * s.eta
        self.SL, self.SR = lead_selfenergy(zE, el.left), lead_selfenergy(zE, el.right)
        self.GamL, self.GamR = broadening(self.SL), broadening(self.SR)
        self.Ad0_el, self.U_el, self.L_el = el.a_blocks(zE)
        self.Ad0_el[0] = self.Ad0_el[0] - self.SL
        self.Ad0_el[-1] = self.Ad0_el[-1] - self.SR
        wp = self.w[1:]
        zW = (wp + 1j * s.eta_ph) ** 2
        zlead = wp ** 2 + 1e-16j
        self.PL, self.PR = lead_selfenergy(zlead, ph.left), lead_selfenergy(zlead, ph.right)
        self.LamL, self.LamR = broadening(self.PL), broadening(self.PR)
        self.Ad0_ph, self.U_ph, self.L_ph = ph.a_blocks(zW)
        self.Ad0_ph[0] = self.Ad0_ph[0] - self.PL
        self.Ad0_ph[-1] = self.Ad0_ph[-1] - self.PR
        TphL = s.T if s.T_ph_L is None else s.T_ph_L
        TphR = s.T if s.T_ph_R is None else s.T_ph_R
        self.NphL = bose(wp, TphL)[:, None, None]
        self.NphR = bose(wp, TphR)[:, None, None]

    def _zero_el(self):
        return [np.zeros((len(self.E), b, b), complex) for b in self.esz]

    def _zero_ph(self):
        return [np.zeros((self.nmax + 1, b, b), complex) for b in self.psz]

    # ------------------------------------------------------- Green's fns
    def electron_greens(self, Sr, Sl, Sg, fL, fR):
        Ad = [A - S for A, S in zip(self.Ad0_el, Sr)]
        Slb, Sgb = list(Sl), list(Sg)
        fL, fR = fL[:, None, None], fR[:, None, None]
        Slb[0] = Slb[0] + 1j * fL * self.GamL
        Slb[-1] = Slb[-1] + 1j * fR * self.GamR
        Sgb[0] = Sgb[0] - 1j * (1 - fL) * self.GamL
        Sgb[-1] = Sgb[-1] - 1j * (1 - fR) * self.GamR
        return rgf_general(Ad, self.U_el, self.L_el, Slb, Sgb)

    def phonon_greens(self, Pr, Pl, Pg):
        Ad = [A - P[1:] for A, P in zip(self.Ad0_ph, Pr)]
        Plb = [P[1:] for P in Pl]
        Pgb = [P[1:] for P in Pg]
        Plb[0] = Plb[0] - 1j * self.NphL * self.LamL
        Plb[-1] = Plb[-1] - 1j * self.NphR * self.LamR
        Pgb[0] = Pgb[0] - 1j * (self.NphL + 1) * self.LamL
        Pgb[-1] = Pgb[-1] - 1j * (self.NphR + 1) * self.LamR
        DR, Dl, Dg = rgf_general(Ad, self.U_ph, self.L_ph, Plb, Pgb)
        pad = lambda L: [np.concatenate([np.zeros((1,) + B.shape[1:], complex), B]) for B in L]
        return pad(DR), pad(Dl), pad(Dg)

    # -------------------------------------------------- self-energies
    def polarisation(self, Gl, Gg):
        c = -1j * self.s.spin * self.dE / (2 * np.pi)
        Pl, Pg = self._zero_ph(), self._zero_ph()
        for b, gi, M in self.cp.terms:
            p, loc = self.gloc[gi]
            ix = np.ix_(np.arange(self.nmax + 1), loc, loc)
            Al = np.einsum("lij,Ejk->Elik", M, Gl[b])          # M^l G^<
            Ag = np.einsum("lij,Ejk->Elik", M, Gg[b])          # M^l G^>
            # Tr[A_l(E) B_l'(E-w)] = sum_ik A_l,ik(E) B_l',ki(E-w)
            Pl[p][ix] += c * corr(Al[:, :, None], np.swapaxes(Ag, -1, -2)[:, None],
                                  self.nmax).sum((-1, -2))
            Pg[p][ix] += c * corr(Ag[:, :, None], np.swapaxes(Al, -1, -2)[:, None],
                                  self.nmax).sum((-1, -2))
        Pr = []
        for p in range(len(Pl)):
            Pl[p][0] = Pg[p][0] = 0.0
            Pr_p = 0.5 * (Pg[p] - Pl[p])
            if self.s.re_pi:
                Pr_p = Pr_p + _hermitian_from_gamma(1j * (Pg[p] - Pl[p]))
            Pr.append(Pr_p)
        return Pr, Pl, Pg

    def self_energy(self, Gl, Gg, Dl, Dg):
        c = 1j * self.dE / (2 * np.pi)
        Sl, Sg = self._zero_el(), self._zero_el()
        for b, gi, M in self.cp.terms:
            p, loc = self.gloc[gi]
            dl = Dl[p][:, loc][:, :, loc][..., None, None]          # (nw, ng, ng, 1, 1)
            dg = Dg[p][:, loc][:, :, loc][..., None, None]
            dlT, dgT = np.swapaxes(dl, 1, 2), np.swapaxes(dg, 1, 2)
            gl, gg = Gl[b][:, None, None], Gg[b][:, None, None]
            Cl = conv_fwd(gl, dl) + conv_bwd(gl, dgT)
            Cg = conv_fwd(gg, dg) + conv_bwd(gg, dlT)
            Sl[b] += c * np.einsum("lij,Elmjk,mkn->Ein", M, Cl, M, optimize=True)
            Sg[b] += c * np.einsum("lij,Elmjk,mkn->Ein", M, Cg, M, optimize=True)
        Sr = []
        for b in range(len(Sl)):
            Gam = 1j * (Sg[b] - Sl[b])
            S = -0.5j * Gam
            if self.s.re_sigma:
                S = S + _hermitian_from_gamma(Gam)
            Sr.append(S)
        return Sr, Sl, Sg

    # ------------------------------------------------------------- solve
    def solve(self, V, guess=None):
        s, E = self.s, self.E
        fL, fR = fermi(E, +0.5 * V, s.T), fermi(E, -0.5 * V, s.T)
        if guess is None:
            Sr, Sl, Sg = self._zero_el(), self._zero_el(), self._zero_el()
        else:
            Sr, Sl, Sg = guess
        hist = []
        for it in range(s.maxit):
            GR, Gl, Gg = self.electron_greens(Sr, Sl, Sg, fL, fR)
            Pr, Pl, Pg = self.polarisation(Gl, Gg)
            DR, Dl, Dg = self.phonon_greens(Pr, Pl, Pg)
            Sr_n, Sl_n, Sg_n = self.self_energy(Gl, Gg, Dl, Dg)
            num = max(np.max(np.abs(a - b)) for a, b in zip(Sl_n, Sl))
            den = max(np.max(np.abs(a)) for a in Sl_n) + 1e-300
            hist.append(num / den)
            a = 1.0 if (it == 0 and guess is None) else s.mix
            mixl = lambda O, N: [(1 - a) * o + a * n for o, n in zip(O, N)]
            Sr, Sl, Sg = mixl(Sr, Sr_n), mixl(Sl, Sl_n), mixl(Sg, Sg_n)
            if it > 0 and hist[-1] < s.tol:
                break
        GR, Gl, Gg = self.electron_greens(Sr, Sl, Sg, fL, fR)
        Pr, Pl, Pg = self.polarisation(Gl, Gg)
        DR, Dl, Dg = self.phonon_greens(Pr, Pl, Pg)
        out = self.observables(V, fL, fR, GR, Gl, Gg, DR, Dl, Dg, Pl, Pg)
        out.update(iters=len(hist), hist=hist, guess=(Sr, Sl, Sg))
        return out

    # ------------------------------------------------------- observables
    def observables(self, V, fL, fR, GR, Gl, Gg, DR, Dl, Dg, Pl, Pg):
        s, E, dE, w = self.s, self.E, self.dE, self.w
        tr = lambda X: np.real(np.trace(X, axis1=-2, axis2=-1))

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
        J_e = -sum(np.sum(w * tr(a @ dg - b @ dl)) for a, b, dl, dg in zip(Pl, Pg, Dl, Dg)) \
            * dE / (2 * np.pi)
        dl, dg = _diag(Dl), _diag(Dg)
        occ = np.real(1j * dl)
        spec = np.real(1j * (dg - dl))
        N_dof = np.sum(w[:, None] * occ, 0) / np.maximum(np.sum(w[:, None] * spec, 0), 1e-300)
        n_el = np.sum(np.real(-1j * _diag(Gl)), 0) * dE / (2 * np.pi)
        A = -2 * np.imag(_diag(GR)).sum(1)
        u = EV2_OVER_HBAR_UW
        return dict(V=V, I_uA=I_L * 2 * np.pi * E_OVER_H_UA, I_L=I_L, I_R=I_R,
                    P_e_uW=P_e * u, J_L_uW=J_L * u, J_R_uW=J_R * u, J_e_uW=J_e * u,
                    N_dof=N_dof, n_el=n_el, A=A,
                    phonon_dos=-2 * w[:, None] * np.imag(_diag(DR)) / np.pi)
