"""Physical constants, distribution functions and unit conversions (hbar = e = 1, eV)."""
import numpy as np

KB = 8.617333262e-5           # eV/K
E_OVER_H_UA = 38.7404585      # (e/h) * 1 eV  -> microampere
EV2_OVER_HBAR_UW = 243.4135   # (1 eV)^2 / hbar -> microwatt


def fermi(E, mu, T):
    return 0.5 * (1.0 - np.tanh(0.5 * (E - mu) / (KB * T)))


def bose(w, T):
    w = np.asarray(w, float)
    out = np.zeros_like(w)
    m = w > 0
    out[m] = 1.0 / np.expm1(w[m] / (KB * T))
    return out
