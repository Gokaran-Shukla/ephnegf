"""Stage 4 tests: non-orthogonal basis, Landauer transmission, sisl front end."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import ephnegf as ep
from ephnegf.system import Lead, BlockSystem
from ephnegf.leads import sancho_rubio
from ephnegf.rgf import rgf_general


def _chain(n, eps=0.0, t=-1.0, s=0.1, tc=None, sc=None):
    tc = t if tc is None else tc
    sc = s if sc is None else sc
    lead = lambda: Lead(np.array([[eps]]), np.array([[t]]), np.array([[tc]]),
                        np.array([[1.0]]), np.array([[s]]), np.array([[sc]]))
    Hd = [np.array([[eps]]) for _ in range(n)]
    Hc = [np.array([[t]]) for _ in range(n - 1)]
    Sd = [np.array([[1.0]]) for _ in range(n)]
    Sc = [np.array([[s]]) for _ in range(n - 1)]
    return BlockSystem(Hd, Hc, lead(), lead(), Sd, Sc)


def test_sancho_rubio_nonorthogonal_chain():
    eps, t, s = 0.2, -1.0, 0.1
    z = np.linspace(-3, 3, 301) + 1e-9j
    gs = sancho_rubio(z, np.array([[eps]]), np.array([[t]]),
                      np.array([[1.0]]), np.array([[s]]))[:, 0, 0]
    a, c = z - eps, z * s - t
    r = np.sqrt(a * a - 4 * c * c)
    g1, g2 = (a - r) / (2 * c * c), (a + r) / (2 * c * c)
    ga = np.where(g1.imag <= 0, g1, g2)
    assert np.max(np.abs(gs - ga)) < 1e-6


def test_rgf_general_vs_dense():
    """Energy-dependent off-diagonal blocks (z S - H)."""
    rng = np.random.default_rng(7)
    sizes, nE = [2, 3, 2, 3], 5
    N = sum(sizes); off = np.cumsum([0] + sizes)
    H = rng.normal(size=(N, N)); H = H + H.T
    S = 0.1 * rng.normal(size=(N, N)); S = S + S.T + 3 * np.eye(N)
    mask = np.zeros((N, N), bool)
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if abs(i - j) <= 1:
                mask[off[i]:off[i + 1], off[j]:off[j + 1]] = True
    H, S = H * mask, S * mask
    z = np.linspace(-1, 1, nE) + 0.05j
    A = z[:, None, None] * S - H
    Ad, U, L, Sl, Sg = [], [], [], [], []
    SL_full = np.zeros((nE, N, N), complex); SG_full = SL_full.copy()
    for i, b in enumerate(sizes):
        sl = slice(off[i], off[i + 1])
        Y = rng.normal(size=(nE, b, b)) + 1j * rng.normal(size=(nE, b, b))
        Gam = 0.1 * Y @ np.conj(np.swapaxes(Y, 1, 2))
        A[:, sl, sl] += 0.5j * Gam
        Ad.append(A[:, sl, sl].copy())
        Sl.append(0.3j * Gam); Sg.append(-0.7j * Gam)
        SL_full[:, sl, sl] = Sl[-1]; SG_full[:, sl, sl] = Sg[-1]
        if i < len(sizes) - 1:
            nx = slice(off[i + 1], off[i + 2])
            U.append(A[:, sl, nx].copy()); L.append(A[:, nx, sl].copy())
    GR, Gl, Gg = rgf_general(Ad, U, L, Sl, Sg)
    GRd = np.linalg.inv(A)
    GAd = np.conj(np.swapaxes(GRd, 1, 2))
    for i in range(len(sizes)):
        sl = slice(off[i], off[i + 1])
        assert np.allclose(GR[i], GRd[:, sl, sl], atol=1e-10)
        assert np.allclose(Gl[i], (GRd @ SL_full @ GAd)[:, sl, sl], atol=1e-10)
        assert np.allclose(Gg[i], (GRd @ SG_full @ GAd)[:, sl, sl], atol=1e-10)


def test_nonorthogonal_pristine_chain_T1():
    eps, t, s = 0.0, -1.0, 0.1
    el = _chain(5, eps, t, s)
    lo, hi = (eps + 2 * t) / (1 + 2 * s), (eps - 2 * t) / (1 - 2 * s)   # band edges
    E = np.linspace(lo + 0.02, hi - 0.02, 200)
    T, T2 = ep.transmission(el, E, return_both=True)
    assert np.allclose(T, 1.0, atol=1e-6) and np.allclose(T, T2, atol=1e-8)
    Eout = np.array([lo - 0.3, hi + 0.3])
    assert np.all(ep.transmission(el, Eout) < 1e-6)


def _gnr(W=4, ncell=8, defect=False):
    import sisl
    g = sisl.geom.zgnr(W)
    H = sisl.Hamiltonian(g, orthogonal=False)
    H.construct([[0.1, 1.44], [(0.0, 1.0), (-2.7, 0.11)]])
    Hd = H.tile(ncell, 0)
    Hd.set_nsc([1, 1, 1])
    if defect:
        mid = Hd.geometry.close(Hd.geometry.center(), R=1.5)[:2]
        for ia in mid:
            Hd[ia, ia] = (3.0, 1.0)
    no = H.no
    return H, Hd, list(range(no)), list(range(Hd.no - no, Hd.no))


def test_sisl_gnr_vs_dense():
    """sisl front end + automatic BTD partition reproduce brute-force dense Landauer."""
    from ephnegf.io_sisl import from_sisl
    from ephnegf.leads import lead_selfenergy, broadening
    H, Hd, lo, ro = _gnr(defect=True)
    el = from_sisl(Hd, H, H, axis=0, left_orbs=lo, right_orbs=ro)
    assert len(el.Hd) > 3
    E = np.linspace(-1.5, 1.5, 7)
    T = ep.transmission(el, E)
    # dense reference
    z = E + 1e-7j
    Hk, Sk = Hd.Hk(format="array"), Hd.Sk(format="array")
    A = z[:, None, None] * Sk - Hk
    SL = lead_selfenergy(z, el.left); SR = lead_selfenergy(z, el.right)
    order = el.order
    A = A[:, order][:, :, order]
    nL, nR = SL.shape[1], SR.shape[1]
    A[:, :nL, :nL] -= SL; A[:, -nR:, -nR:] -= SR
    G = np.linalg.inv(A)
    GamL, GamR = broadening(SL), broadening(SR)
    G1N = G[:, :nL, -nR:]
    Td = np.real(np.trace(GamL @ G1N @ GamR @ np.conj(np.swapaxes(G1N, 1, 2)), axis1=1, axis2=2))
    assert np.allclose(T, Td, atol=1e-8)


def test_sisl_gnr_pristine_integer_channels():
    from ephnegf.io_sisl import from_sisl
    H, Hd, lo, ro = _gnr(defect=False)
    el = from_sisl(Hd, H, H, axis=0, left_orbs=lo, right_orbs=ro)
    E = np.array([-2.0, -1.0, -0.4, 0.6, 1.2, 2.2])
    T = ep.transmission(el, E)
    # channel count from the electrode bands: crossings of E over the BZ / 2
    k = np.linspace(-0.5, 0.5, 4001)
    eig = np.array([H.eigh(k=[kk, 0, 0]) for kk in k])
    nch = [np.sum(np.abs(np.diff(np.sign(eig - e), axis=0)) > 0) // 2 for e in E]
    assert np.allclose(T, nch, atol=1e-5), (T, nch)


def test_scba_nonorthogonal_conservation():
    """Coupled SCBA in a non-orthogonal basis stays particle- and energy-conserving."""
    el = _chain(3, eps=1.0, t=-1.0, s=0.1, tc=-0.6, sc=0.05)
    el.left.H00 = el.right.H00 = np.array([[0.0]])
    el.left.H01 = el.right.H01 = np.array([[-2.0]])
    el.left.S01 = el.right.S01 = np.array([[0.05]])
    ph = ep.chain_phonons(0.1, 3, k_dev=1e-3)
    cp = ep.chain_coupling(3, 0.1, 0.1)
    r = ep.CoupledSCBA(el, ph, cp, ep.Settings(Emin=-5, Emax=5)).solve(0.2)
    assert abs(r["I_L"] + r["I_R"]) / abs(r["I_L"]) < 1e-9
    assert abs(r["P_e_uW"] - r["J_e_uW"]) / r["J_e_uW"] < 1e-5
    assert abs(r["J_e_uW"] + r["J_L_uW"] + r["J_R_uW"]) / r["J_e_uW"] < 1e-8


if __name__ == "__main__":
    test_scba_nonorthogonal_conservation(); print("SCBA non-orthogonal conservation ok")
    test_sancho_rubio_nonorthogonal_chain(); print("non-orthogonal lead ok")
    test_rgf_general_vs_dense(); print("rgf_general (zS - H) ok")
    test_nonorthogonal_pristine_chain_T1(); print("non-orthogonal chain T = 1 ok")
    test_sisl_gnr_vs_dense(); print("sisl GNR vs dense ok")
    test_sisl_gnr_pristine_integer_channels(); print("sisl GNR integer channels ok")
