"""General electron-phonon coupling  H_eph = sum_lambda M^lambda u_lambda.

Block-local SCBA (Luisier-type): the electron self-energy is kept inside each
electron RGF block, the phonon propagator inside each phonon *group*
(an atom's dofs, or a whole phonon block). The resulting functional
    Phi = sum_{b,g} sum_{l,l' in g} D_{l l'} Tr_b[M^l G_bb M^l' G_bb]
is Phi-derivable => particle and energy conserving for any choice of blocks/groups.
Parts of M connecting different electron blocks are dropped (reported by
`dropped_fraction`); use coarser blocks to reduce them.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from .units import XZP


@dataclass
class EPCoupling:
    groups: List[np.ndarray]                       # phonon dof groups (RGF row positions)
    terms: List[Tuple[int, int, np.ndarray]]       # (electron block, group, M (ng, b, b))
    dropped_fraction: float = 0.0


def _inv(order, n):
    inv = np.empty(n, int)
    inv[np.asarray(order)] = np.arange(n)
    return inv


def phonon_groups(ph, mode="atom"):
    """'atom': each atom's dofs; 'block': every phonon RGF block; 'dof': single dofs."""
    off = ph.offsets
    if mode == "block":
        return [np.arange(off[i], off[i + 1]) for i in range(len(off) - 1)]
    if mode == "dof":
        return [np.array([i]) for i in range(ph.n)]
    dof = getattr(ph, "dof", 1)
    order = getattr(ph, "order", np.arange(ph.n))
    inv = _inv(order, ph.n)
    return [inv[np.arange(a * dof, (a + 1) * dof)] for a in range(ph.n // dof)]


def from_holstein(cp, el, ph):
    """Local Holstein coupling (ephnegf.system.Coupling) -> EPCoupling (exact)."""
    groups = [np.array([a]) for a in range(ph.n)]
    pinv = _inv(getattr(ph, "order", np.arange(ph.n)), ph.n)
    einv = _inv(getattr(el, "order", np.arange(el.n)), el.n)
    eoff = el.offsets
    acc = {}
    for o0, (g, a0) in enumerate(zip(cp.g, cp.dof)):
        o, a = int(einv[o0]), int(pinv[a0])
        b = int(np.searchsorted(eoff, o, side="right") - 1)
        nb = eoff[b + 1] - eoff[b]
        key = (b, int(a))
        if key not in acc:
            acc[key] = np.zeros((1, nb, nb), complex)
        acc[key][0, o - eoff[b], o - eoff[b]] += g
    return EPCoupling(groups, [(b, a, M) for (b, a), M in acc.items()])


def from_dense(M, el, ph, groups, tol=1e-12):
    """M: (n_dof, n_orb, n_orb) coupling matrices in ORIGINAL device orbital / dof order
    (internal units, see units.py). Cut into electron blocks and phonon groups."""
    eorder = getattr(el, "order", np.arange(el.n))
    porder = getattr(ph, "order", np.arange(ph.n))
    Mr = M[porder][:, eorder][:, :, eorder]              # RGF ordering
    eoff = el.offsets
    total = np.sum(np.abs(Mr) ** 2)
    kept = 0.0
    terms = []
    for gi, g in enumerate(groups):
        for b in range(len(eoff) - 1):
            s = slice(eoff[b], eoff[b + 1])
            Mb = Mr[g][:, s, s]
            if np.max(np.abs(Mb)) > tol:
                terms.append((b, gi, Mb.astype(complex)))
                kept += np.sum(np.abs(Mb) ** 2)
    frac = 1.0 - kept / total if total > 0 else 0.0
    return EPCoupling(groups, terms, dropped_fraction=frac)


def eph_finite_difference(H_plus, H_minus, delta, mass):
    """M^lambda (internal units) from Hamiltonians at +/- delta (A) displacement of one
    Cartesian dof of an atom of mass `mass` (amu):  dH/dx * XZP / sqrt(m)."""
    return (np.asarray(H_plus) - np.asarray(H_minus)) / (2 * delta) * XZP / np.sqrt(mass)
