"""FFT convolutions on the common grid (w_n = n dE) and Kramers-Kronig.
All arrays have the energy / frequency index on axis 0; trailing axes are batched."""
import numpy as np
from scipy.signal import fftconvolve, hilbert


def conv_fwd(G, K):
    """out[m] = sum_{n>=0} K[n] G[m-n]"""
    return fftconvolve(G, K, mode="full", axes=0)[: G.shape[0]]


def conv_bwd(G, K):
    """out[m] = sum_{n>=0} K[n] G[m+n]"""
    nmax = K.shape[0] - 1
    return fftconvolve(G, K[::-1], mode="full", axes=0)[nmax: nmax + G.shape[0]]


def corr(a, b, nmax):
    """c[n] = sum_m a[m] b[m-n],  n = 0..nmax"""
    nE = a.shape[0]
    return fftconvolve(a, b[::-1], mode="full", axes=0)[nE - 1: nE + nmax]


def kramers_kronig_real(im):
    """Re f = -H[Im f] for a retarded function (zero-padded FFT Hilbert transform)."""
    nE = im.shape[0]
    pad = np.zeros((2 * nE,) + im.shape[1:])
    pad[nE // 2: nE // 2 + nE] = im
    return -np.imag(hilbert(pad, axis=0))[nE // 2: nE // 2 + nE]
