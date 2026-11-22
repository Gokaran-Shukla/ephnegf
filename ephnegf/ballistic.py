"""Ballistic (Landauer) quantities from block-tridiagonal RGF."""
import numpy as np
from .leads import lead_selfenergy, broadening
from .rgf import rgf_general


def transmission(el, E, eta=1e-7, return_both=False):
    """T(E) = Tr[Gamma_L (G Gamma_R G^+)_{00}]  -- needs only diagonal blocks.
    Works for orthogonal and non-orthogonal (S) bases.
    With return_both=True also returns T computed from the right lead (consistency check)."""
    E = np.atleast_1d(np.asarray(E, float))
    z = E + 1j * eta
    SL, SR = lead_selfenergy(z, el.left), lead_selfenergy(z, el.right)
    GL, GR_ = broadening(SL), broadening(SR)
    Ad, U, L = el.a_blocks(z)
    Ad[0] = Ad[0] - SL
    Ad[-1] = Ad[-1] - SR
    zero = [np.zeros_like(A) for A in Ad]
    S_R = list(zero); S_R[-1] = GR_.astype(complex)
    _, Gl, _ = rgf_general(Ad, U, L, S_R)
    T = np.real(np.trace(GL @ Gl[0], axis1=1, axis2=2))
    if not return_both:
        return T
    S_L = list(zero); S_L[0] = GL.astype(complex)
    _, Gl2, _ = rgf_general(Ad, U, L, S_L)
    T2 = np.real(np.trace(GR_ @ Gl2[-1], axis1=1, axis2=2))
    return T, T2
