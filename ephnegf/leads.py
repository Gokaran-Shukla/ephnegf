"""Semi-infinite lead surface Green's functions (Sancho-Rubio decimation), batched over energy.

The same routine serves electrons (z = E + i eta, blocks H00/H01) and phonons
(z = w^2 + i eta, blocks K00/K01 of the mass-weighted dynamical matrix).
"""
import numpy as np


def sancho_rubio(z, H00, H01, tol=1e-12, maxit=200, res_tol=1e-7, delta=1e-5):
    """Surface GF of a semi-infinite lead extending to the RIGHT:
        ... H00 | H01 -> next cell.
    z: (nz,) complex; H00, H01: (n, n).  Returns g_s (nz, n, n).
    For a lead extending to the LEFT pass H01 -> H01^dagger.

    Decimation loses precision at isolated high-symmetry energies when eta is tiny
    (e.g. band centre of a chain). Each point is checked against the Dyson
    equation g = [z - H00 - H01 g H01^+]^-1; failed points are replaced by the
    average of z +/- delta (error O(delta^2)).
    """
    z = np.asarray(z, complex)
    gs = _decimate(z, H00, H01, tol, maxit)
    n = H00.shape[0]
    res = gs - np.linalg.inv(z[:, None, None] * np.eye(n) - H00 - H01 @ gs @ H01.conj().T)
    bad = np.max(np.abs(res), axis=(1, 2)) > res_tol * np.maximum(1.0, np.max(np.abs(gs), axis=(1, 2)))
    if np.any(bad):
        zb = z[bad]
        gs[bad] = 0.5 * (_decimate(zb + delta, H00, H01, tol, maxit) +
                         _decimate(zb - delta, H00, H01, tol, maxit))
    return gs


def _decimate(z, H00, H01, tol, maxit):
    n = H00.shape[0]
    I = np.eye(n)
    zI = z[:, None, None] * I
    a = np.broadcast_to(H01, (len(z), n, n)).astype(complex).copy()
    b = np.broadcast_to(H01.conj().T, (len(z), n, n)).astype(complex).copy()
    es = np.broadcast_to(H00, (len(z), n, n)).astype(complex).copy()
    e = es.copy()
    for _ in range(maxit):
        g = np.linalg.inv(zI - e)
        agb = a @ g @ b
        bga = b @ g @ a
        es = es + agb
        e = e + agb + bga
        a = a @ g @ a
        b = b @ g @ b
        if np.max(np.abs(a)) < tol and np.max(np.abs(b)) < tol:
            break
    return np.linalg.inv(zI - es)


def lead_selfenergy(z, H00, H01, tau, side):
    """Self-energy on the device edge block.
    tau: coupling matrix device-edge-block -> first lead cell (n_dev_edge, n_lead).
    side: 'L' (lead to the left) or 'R' (lead to the right).
    Sigma = tau g_s tau^dagger.
    """
    if side == "R":
        gs = sancho_rubio(z, H00, H01)
    else:
        gs = sancho_rubio(z, H00, H01.conj().T)
    return tau @ gs @ tau.conj().T


def broadening(S):
    """Gamma = i (S - S^dagger)."""
    return 1j * (S - np.conj(np.swapaxes(S, -1, -2)))
