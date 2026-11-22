"""Stage 5 tests: real-unit phonons from force constants, finite-difference e-ph
matrices, block-local SCBA with non-local coupling (conservation)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import ephnegf as ep
from ephnegf.units import HB2, XZP
from ephnegf.phonons import chain_fc, dynamical_matrix, layer_blocks
from ephnegf.system import Lead, from_dense_blocks, chain_blocks
from ephnegf.coupling import from_dense, phonon_groups, eph_finite_difference

K_SPRING, MASS, T0, BETA = 30.0, 12.0, 2.5, 4.0     # eV/A^2, amu, eV, eV/A


def chain_phonon_system(n, size):
    Ks = dynamical_matrix(chain_fc(5, K_SPRING), [MASS] * 5)
    B = layer_blocks(Ks, 1)
    K = dynamical_matrix(chain_fc(n, K_SPRING), [MASS] * n)
    L = Lead(B[0], B[-1], B[-1]); R = Lead(B[0], B[+1], B[+1])
    return from_dense_blocks(K, chain_blocks(n, size), L, R)


def ssh_H(n, x=None):
    x = np.zeros(n) if x is None else x
    H = np.zeros((n, n))
    for i in range(n - 1):
        t = T0 + BETA * (x[i + 1] - x[i])          # bond stretching weakens/strengthens t
        H[i, i + 1] = H[i + 1, i] = -t
    return H


def test_units_phonon_band_edge():
    """Ballistic phonon transmission of a uniform chain: T = 1 below
    w_max = 2 sqrt(k/m * HB2) and 0 above (checks unit conversion + phonon leads)."""
    ph = chain_phonon_system(6, 2)
    wmax = 2 * np.sqrt(K_SPRING / MASS * HB2)
    w = np.array([0.2, 0.5, 0.9, 0.98, 1.05, 1.2]) * wmax
    T = ep.transmission(ph, w ** 2, eta=1e-14)
    assert np.allclose(T[:4], 1.0, atol=1e-6) and np.all(T[4:] < 1e-6)
    assert abs(wmax - 0.2045) < 1e-3          # ~1650 cm^-1 for a C chain with 30 eV/A^2


def test_finite_difference_ssh():
    n, d = 6, 1e-3
    M = np.array([eph_finite_difference(ssh_H(n, d * np.eye(n)[l]), ssh_H(n, -d * np.eye(n)[l]),
                                        d, MASS) for l in range(n)])
    Ma = np.zeros_like(M); c = XZP / np.sqrt(MASS)
    for i in range(n - 1):            # dH_{i,i+1}/dx_i = +BETA, /dx_{i+1} = -BETA
        Ma[i, i, i + 1] = Ma[i, i + 1, i] = BETA * c
        Ma[i + 1, i, i + 1] = Ma[i + 1, i + 1, i] = -BETA * c
    assert np.allclose(M, Ma, atol=1e-10)


def test_nonlocal_scba_conservation():
    n = 8
    el = from_dense_blocks(ssh_H(n), chain_blocks(n, 3),
                           Lead(np.array([[0.0]]), np.array([[-T0]]), np.array([[-T0]])),
                           Lead(np.array([[0.0]]), np.array([[-T0]]), np.array([[-T0]])))
    ph = chain_phonon_system(n, 3)
    d = 1e-3
    M = np.array([eph_finite_difference(ssh_H(n, d * np.eye(n)[l]), ssh_H(n, -d * np.eye(n)[l]),
                                        d, MASS) for l in range(n)])
    cp = from_dense(M, el, ph, phonon_groups(ph, "block"))
    assert 0 < cp.dropped_fraction < 0.6
    s = ep.Settings(Emin=-5.5, Emax=5.5, dE=2e-3, wmax=0.22, T=50.0)
    r = ep.CoupledSCBA(el, ph, cp, s).solve(0.5)
    assert abs(r["I_L"] + r["I_R"]) / abs(r["I_L"]) < 1e-9
    assert abs(r["J_e_uW"] + r["J_L_uW"] + r["J_R_uW"]) / abs(r["J_e_uW"]) < 1e-8
    assert abs(r["P_e_uW"] - r["J_e_uW"]) / abs(r["J_e_uW"]) < 1e-4


if __name__ == "__main__":
    test_units_phonon_band_edge(); print("phonon units / band edge ok")
    test_finite_difference_ssh(); print("finite-difference e-ph ok")
    test_nonlocal_scba_conservation(); print("non-local block SCBA conservation ok")
