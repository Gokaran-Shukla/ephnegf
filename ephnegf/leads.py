"""Semi-infinite lead surface Green's functions (Sancho-Rubio decimation), batched over energy.

Works with a non-orthogonal basis: every quantity is expressed through
    A00(z) = z S00 - H00,   A01(z) = z S01 - H01,   A10(z) = z S01^+ - H01^+
where H01 / S01 couple a principal layer to the next layer FURTHER AWAY from the
device (so the same routine serves left and right leads).
Electrons: z = E + i eta.  Phonons: H -> K (dynamical matrix), S = 1, z = w^2 + i eta.
"""
import numpy as np


def _dag(X):
    return np.conj(np.swapaxes(X, -1, -2))


def _amat(z, H, S):
    """z S - H for a batch of z; S=None means S = 0 (off-diagonal) handled by caller."""
    return z[:, None, None] * S - H


def _decimate(A00, A01, A10, tol, maxit):
    es = A00.copy()
    e = A00.copy()
    a, b = A01.copy(), A10.copy()
    for _ in range(maxit):
        g = np.linalg.inv(e)
        agb = a @ g @ b
        bga = b @ g @ a
        es = es - agb
        e = e - agb - bga
        a = -(a @ g @ a)
        b = -(b @ g @ b)
        if np.max(np.abs(a)) < tol and np.max(np.abs(b)) < tol:
            break
    return np.linalg.inv(es)


def _blocks(z, H00, H01, S00, S01):
    n = H00.shape[0]
    S00 = np.eye(n) if S00 is None else S00
    S01 = np.zeros_like(H01) if S01 is None else S01
    A00 = _amat(z, H00, S00).astype(complex)
    A01 = _amat(z, H01, S01).astype(complex)
    A10 = (z[:, None, None] * S01.conj().T - H01.conj().T).astype(complex)
    return A00, A01, A10


def sancho_rubio(z, H00, H01, S00=None, S01=None, tol=1e-12, maxit=200,
                 res_tol=1e-7, delta=1e-5):
    """Surface GF  g_s = [A00 - A01 g_s A10]^-1  of a semi-infinite lead whose
    layers continue along H01 / S01.

    Decimation can lose precision at isolated high-symmetry energies when eta is
    tiny (e.g. band centre of a chain). Each point is checked against the Dyson
    equation; failed points are replaced by the average of z +/- delta (O(delta^2)).
    """
    z = np.atleast_1d(np.asarray(z, complex))
    A00, A01, A10 = _blocks(z, H00, H01, S00, S01)
    gs = _decimate(A00, A01, A10, tol, maxit)
    res = gs - np.linalg.inv(A00 - A01 @ gs @ A10)
    scale = np.maximum(1.0, np.max(np.abs(gs), axis=(1, 2)))
    bad = np.max(np.abs(res), axis=(1, 2)) > res_tol * scale
    if np.any(bad):
        acc = 0
        for d in (+delta, -delta):
            Ab = _blocks(z[bad] + d, H00, H01, S00, S01)
            acc = acc + _decimate(*Ab, tol, maxit)
        gs[bad] = 0.5 * acc
    return gs


def lead_selfenergy(z, lead):
    """Sigma(z) = A_dl g_s A_ld  on the device edge block, with
    A_dl = z tauS - tau  (device edge -> first lead layer),  A_ld = z tauS^+ - tau^+."""
    z = np.atleast_1d(np.asarray(z, complex))
    gs = sancho_rubio(z, lead.H00, lead.H01, lead.S00, lead.S01)
    tS = np.zeros_like(lead.tau) if lead.tauS is None else lead.tauS
    A_dl = z[:, None, None] * tS - lead.tau
    A_ld = z[:, None, None] * tS.conj().T - lead.tau.conj().T
    return A_dl @ gs @ A_ld


def broadening(S):
    """Gamma = i (S - S^dagger)."""
    return 1j * (S - _dag(S))
