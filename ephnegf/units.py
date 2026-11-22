"""Unit conventions for real-material input.

Internal phonon units: mass-weighted dynamical matrix K in eV^2 so that w (eV) solves
det(w^2 - K) = 0; displacement field u normalised with hbar = 1 in eV units.

    (hbar w)^2 [eV^2] = Phi [eV/A^2] / m [amu] * HB2
    x [A]             = u * XZP / sqrt(m [amu])          (physical displacement)
    g [internal]      = dH/dx [eV/A] * XZP / sqrt(m)     (coupling to u)
Einstein check: g = M sqrt(2 w), M = dH/dx * x_zp, x_zp = XZP / sqrt(2 m w).
"""
import numpy as np

HBAR = 1.054571817e-34
AMU = 1.66053906660e-27
EV = 1.602176634e-19
HB2 = HBAR ** 2 / (AMU * 1e-20) / EV           # 4.1801e-3 (eV);  w^2[eV^2] = (Phi/m)[eV/A^2/amu] * HB2[eV]
XZP = np.sqrt(HB2)                              # 0.064654 eV^{1/2} A
EV_TO_CM1 = 8065.54429
EV_TO_THZ = 241.799050
