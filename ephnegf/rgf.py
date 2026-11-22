"""Block-tridiagonal recursive Green's function (RGF), batched over energy.

Solves  A G^R = 1  for block-tridiagonal  A = z S - H - Sigma^R  and
        G^<> = G^R Sigma^<> G^A  with Sigma^<> block DIAGONAL (leads + local scattering),
returning the diagonal blocks of G^R, G^<, G^>.   Cost O(nE * nblocks * b^3).

Off-diagonal blocks of A may be energy dependent (non-orthogonal basis):
    U_i = A_{i,i+1} = z S_{i,i+1} - H_{i,i+1}
    L_i = A_{i+1,i} = z S_{i,i+1}^+ - H_{i,i+1}^+
Each U_i / L_i is either a constant (b_i, b_{i+1}) matrix or a batch (nE, ., .).
Algorithm: Lake et al., JAP 81, 7845 (1997); Svizhenko et al., JAP 91, 2343 (2002).
"""
import numpy as np


def _dag(X):
    return np.conj(np.swapaxes(X, -1, -2))


def _retarded(Ad, U, L):
    nb = len(Ad)
    gR = [None] * nb
    gR[0] = np.linalg.inv(Ad[0])
    for i in range(1, nb):
        gR[i] = np.linalg.inv(Ad[i] - L[i - 1] @ gR[i - 1] @ U[i - 1])
    GR = [None] * nb
    GR[-1] = gR[-1]
    for i in range(nb - 2, -1, -1):
        GR[i] = gR[i] + gR[i] @ U[i] @ GR[i + 1] @ L[i] @ gR[i]
    return gR, GR


def _lesser(gR, GR, U, L, Sl):
    nb = len(gR)
    gl = [None] * nb
    gl[0] = gR[0] @ Sl[0] @ _dag(gR[0])
    for i in range(1, nb):
        Li = L[i - 1]
        gl[i] = gR[i] @ (Sl[i] + Li @ gl[i - 1] @ _dag(Li)) @ _dag(gR[i])
    Gl = [None] * nb
    Gl[-1] = gl[-1]
    for i in range(nb - 2, -1, -1):
        g, gd, g_l = gR[i], _dag(gR[i]), gl[i]
        Ui, Li = U[i], L[i]
        Gl[i] = (g_l + g @ Ui @ Gl[i + 1] @ _dag(Ui) @ gd
                 + g @ Ui @ GR[i + 1] @ Li @ g_l
                 + g_l @ _dag(Li) @ _dag(GR[i + 1]) @ _dag(Ui) @ gd)
    return Gl


def rgf_general(Ad, U, L, Sl, Sg=None):
    """
    Ad : list of (nE, b_i, b_i)   diagonal blocks of A
    U, L : lists of upper / lower off-diagonal blocks of A (constant or (nE, ., .))
    Sl, Sg : lists of diagonal blocks of Sigma^<, Sigma^>.
        If Sg is given, G^> is built with the same recursion (exactly consistent with
        Sigma^>: the infinitesimal eta does not act as a fictitious bath).
        Otherwise G^> = G^< + G^R - G^A.
    """
    gR, GR = _retarded(Ad, U, L)
    Gl = _lesser(gR, GR, U, L, Sl)
    if Sg is None:
        Gg = [Gl[i] + GR[i] - _dag(GR[i]) for i in range(len(Ad))]
    else:
        Gg = _lesser(gR, GR, U, L, Sg)
    return GR, Gl, Gg


def rgf_diagonal(Ad, Hc, Sl, Sg=None):
    """Orthogonal basis: A_{i,i+1} = -H_{i,i+1}, A_{i+1,i} = -H_{i,i+1}^+."""
    U = [-T for T in Hc]
    L = [-T.conj().T for T in Hc]
    return rgf_general(Ad, U, L, Sl, Sg)
