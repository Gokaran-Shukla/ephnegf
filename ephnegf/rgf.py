"""Block-tridiagonal recursive Green's function (RGF), batched over energy.

Solves  A G^R = 1  with  A = z - H - Sigma^R  block tridiagonal, and
        G^<  = G^R Sigma^< G^A   with Sigma^< block DIAGONAL (leads + local scattering),
returning only the diagonal blocks of G^R, G^< (and G^> = G^< + G^R - G^A).
Cost O(nE * nblocks * b^3) instead of O(nE * N^3).

Conventions: Hc[i] = H_{i,i+1} (coupling of block i to block i+1 in the
'Hamiltonian' -- for phonons this is the dynamical-matrix coupling K_{i,i+1}).
Algorithm: Lake et al., JAP 81, 7845 (1997); Svizhenko et al., JAP 91, 2343 (2002).
"""
import numpy as np


def _dag(X):
    return np.conj(np.swapaxes(X, -1, -2))


def _lesser(gR, Hc, Sl, GR):
    """Forward/backward sweep for a 'lesser-type' function given retarded parts."""
    nb = len(gR)
    gl = [None] * nb
    gl[0] = gR[0] @ Sl[0] @ _dag(gR[0])
    for i in range(1, nb):
        T = Hc[i - 1]
        gl[i] = gR[i] @ (Sl[i] + T.conj().T @ gl[i - 1] @ T) @ _dag(gR[i])
    Gl = [None] * nb
    Gl[-1] = gl[-1]
    for i in range(nb - 2, -1, -1):
        T, Td = Hc[i], Hc[i].conj().T
        g, gd, g_l = gR[i], _dag(gR[i]), gl[i]
        Gl[i] = (g_l + g @ T @ Gl[i + 1] @ Td @ gd
                 + g @ T @ GR[i + 1] @ Td @ g_l + g_l @ T @ _dag(GR[i + 1]) @ Td @ gd)
    return Gl


def rgf_diagonal(Ad, Hc, Sl, Sg=None):
    """
    Ad : list of (nE, b_i, b_i)  diagonal blocks of A = z - H - Sigma^R
    Hc : list of (b_i, b_{i+1})   upper off-diagonal blocks of H (A_{i,i+1} = -Hc[i])
    Sl : list of (nE, b_i, b_i)  diagonal blocks of Sigma^<
    Sg : list of diagonal blocks of Sigma^>. If given, G^> = G^R Sigma^> G^A is
         computed with the same recursion (exactly consistent with Sigma^>, i.e. the
         infinitesimal eta does NOT leak in as a fictitious bath). If None,
         G^> = G^< + G^R - G^A (includes the eta contribution).
    returns GR, Gl, Gg : lists of diagonal blocks
    """
    nb = len(Ad)
    gR = [None] * nb
    gR[0] = np.linalg.inv(Ad[0])
    for i in range(1, nb):
        T = Hc[i - 1]
        gR[i] = np.linalg.inv(Ad[i] - T.conj().T @ gR[i - 1] @ T)
    GR = [None] * nb
    GR[-1] = gR[-1]
    for i in range(nb - 2, -1, -1):
        T = Hc[i]
        GR[i] = gR[i] + gR[i] @ T @ GR[i + 1] @ T.conj().T @ gR[i]
    Gl = _lesser(gR, Hc, Sl, GR)
    if Sg is None:
        Gg = [Gl[i] + GR[i] - _dag(GR[i]) for i in range(nb)]
    else:
        Gg = _lesser(gR, Hc, Sg, GR)
    return GR, Gl, Gg
