"""Core numerical tests:  python -m pytest tests  (or  python tests/test_core.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ephnegf.leads import sancho_rubio
from ephnegf.rgf import rgf_diagonal


def test_sancho_rubio_1d_chain():
    t = 2.0
    E = np.linspace(-5, 5, 401) + 1e-9j
    gs = sancho_rubio(E, np.array([[0.0]]), np.array([[-t]]))[:, 0, 0]
    root = np.sqrt(E * E - 4 * t * t)
    ga = (E - root) / (2 * t * t)
    ga = np.where(ga.imag <= 0, ga, (E + root) / (2 * t * t))
    assert np.max(np.abs(gs - ga)) < 1e-6


def test_rgf_vs_dense():
    rng = np.random.default_rng(1)
    sizes = [2, 3, 2, 2, 3]
    nE = 7
    N = sum(sizes)
    off = np.cumsum([0] + sizes)
    H = np.zeros((N, N), complex)
    for i, b in enumerate(sizes):
        X = rng.normal(size=(b, b)) + 1j * rng.normal(size=(b, b))
        H[off[i]:off[i + 1], off[i]:off[i + 1]] = X + X.conj().T
        if i < len(sizes) - 1:
            H[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = rng.normal(size=(b, sizes[i + 1]))
    H = np.triu(H) + np.triu(H, 1).conj().T
    E = np.linspace(-1, 1, nE)
    Sr, Sl, Ad, Hc = [], [], [], []
    A_full = np.zeros((nE, N, N), complex)
    Sl_full = np.zeros((nE, N, N), complex)
    for i, b in enumerate(sizes):
        Y = rng.normal(size=(nE, b, b)) + 1j * rng.normal(size=(nE, b, b))
        Gam = Y @ np.conj(np.swapaxes(Y, 1, 2)) * 0.1           # positive broadening
        sr = -0.5j * Gam
        sl = 1j * 0.3 * Gam
        sl_ = slice(off[i], off[i + 1])
        Ad.append(E[:, None, None] * np.eye(b) - H[sl_, sl_] - sr)
        Sl.append(sl)
        A_full[:, sl_, sl_] = Ad[-1]
        Sl_full[:, sl_, sl_] = sl
        if i < len(sizes) - 1:
            Hc.append(H[sl_, off[i + 1]:off[i + 2]])
            A_full[:, sl_, off[i + 1]:off[i + 2]] = -Hc[-1]
            A_full[:, off[i + 1]:off[i + 2], sl_] = -Hc[-1].conj().T
    GR, Gl, Gg = rgf_diagonal(Ad, Hc, Sl)
    GRd = np.linalg.inv(A_full)
    Gld = GRd @ Sl_full @ np.conj(np.swapaxes(GRd, 1, 2))
    for i in range(len(sizes)):
        sl_ = slice(off[i], off[i + 1])
        assert np.allclose(GR[i], GRd[:, sl_, sl_], atol=1e-10)
        assert np.allclose(Gl[i], Gld[:, sl_, sl_], atol=1e-10)


if __name__ == "__main__":
    test_sancho_rubio_1d_chain(); print("sancho_rubio ok")
    test_rgf_vs_dense(); print("rgf ok")


def test_rgf_greater_with_sigma_greater():
    """G^> from explicit Sigma^> recursion equals dense G^R Sigma^> G^A."""
    rng = np.random.default_rng(3)
    nE, b, nb = 4, 2, 4
    Ad, Hc, Sl, Sg = [], [], [], []
    N = b * nb
    A = np.zeros((nE, N, N), complex); SG = np.zeros_like(A)
    for i in range(nb):
        X = rng.normal(size=(b, b)); H = X + X.T
        Y = rng.normal(size=(nE, b, b)) + 1j * rng.normal(size=(nE, b, b))
        Gam = 0.1 * Y @ np.conj(np.swapaxes(Y, 1, 2))
        Ad.append(0.3 * np.eye(b) - H + 0.5j * Gam + 1e-3j * np.eye(b))
        Sl.append(0.4j * Gam); Sg.append(-0.6j * Gam)
        s = slice(i * b, (i + 1) * b)
        A[:, s, s] = Ad[-1]; SG[:, s, s] = Sg[-1]
        if i < nb - 1:
            Hc.append(rng.normal(size=(b, b)))
            A[:, s, (i + 1) * b:(i + 2) * b] = -Hc[-1]
            A[:, (i + 1) * b:(i + 2) * b, s] = -Hc[-1].T
    GR, Gl, Gg = rgf_diagonal(Ad, Hc, Sl, Sg)
    GRd = np.linalg.inv(A)
    Ggd = GRd @ SG @ np.conj(np.swapaxes(GRd, 1, 2))
    for i in range(nb):
        s = slice(i * b, (i + 1) * b)
        assert np.allclose(Gg[i], Ggd[:, s, s], atol=1e-10)


def test_scba_conservation_and_legacy_regression():
    """Single-site device: reproduces the stage-2 reference values (dense legacy
    code, analytic leads) and conserves particle current and energy."""
    import ephnegf as ep
    el = ep.chain_electrons([1.0]); ph = ep.chain_phonons(0.1, 1)
    cp = ep.chain_coupling(1, 0.15, 0.1)
    r = ep.CoupledSCBA(el, ph, cp, ep.Settings()).solve(0.2)
    assert abs(r["I_uA"] - 1.8656877216115784) / 1.8656877216 < 1e-6       # legacy stage 2
    assert abs(r["I_L"] + r["I_R"]) / abs(r["I_L"]) < 1e-10
    assert abs(r["P_e_uW"] - r["J_e_uW"]) / r["J_e_uW"] < 1e-6
    assert abs(r["J_e_uW"] + r["J_L_uW"] + r["J_R_uW"]) / r["J_e_uW"] < 1e-8


def test_ballistic_phonon_chain():
    """Uniform harmonic chain through the package leads: T(w) = 1 in the band."""
    from ephnegf.leads import lead_selfenergy, broadening
    k = 0.0064
    w = np.linspace(0.005, 0.155, 300)
    z = w ** 2 + 1e-16j
    H00, H01, tau = np.array([[2 * k]]), np.array([[-k]]), np.array([[-k]])
    PL = lead_selfenergy(z, H00, H01, tau, "L")[:, 0, 0]
    PR = lead_selfenergy(z, H00, H01, tau, "R")[:, 0, 0]
    D = 1 / (z - 2 * k - PL - PR)
    T = (-2 * PL.imag) * np.abs(D) ** 2 * (-2 * PR.imag)
    assert np.allclose(T, 1.0, atol=1e-8)


if __name__ == "__main__":
    test_rgf_greater_with_sigma_greater(); print("rgf G> ok")
    test_ballistic_phonon_chain(); print("ballistic phonon ok")
    test_scba_conservation_and_legacy_regression(); print("SCBA conservation + regression ok")
