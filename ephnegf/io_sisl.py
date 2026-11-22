"""Front end for SIESTA / TranSIESTA (and any sisl Hamiltonian) via sisl.

Typical TranSIESTA use:
    el = from_siesta("device.TSHS", "elecL.TSHS", "elecR.TSHS", axis=2,
                     left_atoms=range(0, 8), right_atoms=range(56, 64))
    T  = ephnegf.ballistic.transmission(el, E)

Conventions (as in TranSIESTA): the device region CONTAINS one copy of each
electrode principal layer at its ends (left_atoms / right_atoms), in the same
orbital order as the electrode file. The device is partitioned automatically into
a block-tridiagonal form by a breadth-first layering of its coupling graph.
"""
import numpy as np
from .system import BlockSystem, Lead


def _lattice(geom):
    return geom.lattice if hasattr(geom, "lattice") else geom.sc


def layer_blocks(H, axis, k=(0.0, 0.0, 0.0), spin=0):
    """Fourier-sum a periodic sisl Hamiltonian over the directions transverse to
    `axis` (Bloch vector k in reduced units) and return principal-layer blocks
        {offset: (H_block, S_block)}  for offsets -1, 0, +1 along `axis`.
    Raises if couplings reach beyond nearest layers (tile the electrode first)."""
    k = np.asarray(k, float)
    no = H.no
    lat = _lattice(H.geometry)
    sc_off = np.asarray(lat.sc_off)
    Hc = H.tocsr(spin)
    Sc = None if H.orthogonal else H.tocsr(H.S_idx)
    blocks = {o: [np.zeros((no, no), complex), np.zeros((no, no), complex)] for o in (-1, 0, 1)}
    for isc, off in enumerate(sc_off):
        Hb = Hc[:, isc * no:(isc + 1) * no].toarray()
        Sb = None if Sc is None else Sc[:, isc * no:(isc + 1) * no].toarray()
        if not np.any(Hb) and (Sb is None or not np.any(Sb)):
            continue
        a = int(off[axis])
        if abs(a) > 1:
            raise ValueError("couplings beyond the next principal layer: tile the electrode "
                             "along the transport axis")
        perp = off.astype(float).copy(); perp[axis] = 0.0
        ph = np.exp(2j * np.pi * np.dot(k, perp))
        blocks[a][0] += ph * Hb
        if Sb is not None:
            blocks[a][1] += ph * Sb
    if H.orthogonal:
        blocks[0][1] = np.eye(no, dtype=complex)
    return {o: (b[0], b[1]) for o, b in blocks.items()}


def lead_from_sisl(H, axis, side, k=(0.0, 0.0, 0.0), spin=0, tau_rows=None):
    """Lead object from a sisl electrode Hamiltonian (periodic along `axis`).
    side = 'L' (lead continues towards -axis) or 'R' (towards +axis).
    tau_rows: (b_edge, no_elec) embedding of the electrode orbitals in the device
    edge block (default: the edge block IS one electrode layer)."""
    B = layer_blocks(H, axis, k, spin)
    away = -1 if side == "L" else +1
    H00, S00 = B[0]
    H01, S01 = B[away]
    tau, tauS = H01, S01
    if tau_rows is not None:
        tau, tauS = tau_rows @ tau, tau_rows @ tauS
    ortho = H.orthogonal
    return Lead(H00=H00, H01=H01, tau=tau,
                S00=None if ortho else S00, S01=None if ortho else S01,
                tauS=None if ortho else tauS)


def partition_btd(A, first, last):
    """Breadth-first layering of the coupling graph of matrix A (n x n, nonzero
    pattern), starting from orbital list `first`. All layers from the first one that
    touches `last` onward are merged so that `last` sits in the final block.
    Returns list of index arrays (blocks)."""
    n = A.shape[0]
    adj = (np.abs(A) > 1e-12)
    np.fill_diagonal(adj, False)
    assigned = np.zeros(n, bool)
    cur = np.asarray(list(first))
    assigned[cur] = True
    layers = [cur]
    last = set(int(i) for i in last)
    while assigned.sum() < n:
        nb = np.where(adj[cur].any(0) & ~assigned)[0]
        if len(nb) == 0:
            raise ValueError("device graph is disconnected from the left electrode")
        assigned[nb] = True
        layers.append(nb)
        cur = nb
    first_last = next(i for i, L in enumerate(layers) if last & set(L.tolist()))
    if first_last == 0:
        raise ValueError("left and right electrode layers couple directly; enlarge the device")
    return layers[:first_last] + [np.concatenate(layers[first_last:])]


def device_from_matrices(Hk, Sk, left_orbs, right_orbs, lead_L, lead_R_raw, ortho):
    """Assemble a BlockSystem from dense device matrices (at fixed transverse k)."""
    blocks = partition_btd(np.abs(Hk) + (0 if Sk is None else np.abs(Sk)), left_orbs, right_orbs)
    # enforce right orbitals in the order given by the user (= electrode order)
    rset = set(int(i) for i in right_orbs)
    rest = np.array([i for i in blocks[-1] if int(i) not in rset], int)
    blocks[-1] = np.concatenate([rest, np.asarray(right_orbs, int)])
    Hd = [Hk[np.ix_(b, b)] for b in blocks]
    Hc = [Hk[np.ix_(blocks[i], blocks[i + 1])] for i in range(len(blocks) - 1)]
    Sd = Sc = None
    if not ortho:
        Sd = [Sk[np.ix_(b, b)] for b in blocks]
        Sc = [Sk[np.ix_(blocks[i], blocks[i + 1])] for i in range(len(blocks) - 1)]
    nR = len(right_orbs)
    rows = np.zeros((len(blocks[-1]), nR))
    rows[len(rest):, :] = np.eye(nR)
    lead_R = Lead(lead_R_raw.H00, lead_R_raw.H01, rows @ lead_R_raw.tau,
                  lead_R_raw.S00, lead_R_raw.S01,
                  None if lead_R_raw.tauS is None else rows @ lead_R_raw.tauS)
    sys_ = BlockSystem(Hd, Hc, lead_L, lead_R, Sd, Sc)
    sys_.order = np.concatenate(blocks)      # device orbital index of each RGF row
    return sys_


def from_sisl(Hdev, HelL, HelR, axis, left_orbs, right_orbs, k=(0.0, 0.0, 0.0), spin=0):
    """Build the electron BlockSystem from sisl Hamiltonians.
    Hdev must not be periodic along `axis` (nsc = 1 there, as in TranSIESTA)."""
    if _lattice(Hdev.geometry).nsc[axis] != 1:
        raise ValueError("device Hamiltonian must have nsc = 1 along the transport axis")
    Hk = Hdev.Hk(k=k, spin=spin, format="array") if Hdev.spin.is_polarized \
        else Hdev.Hk(k=k, format="array")
    Sk = None if Hdev.orthogonal else Hdev.Sk(k=k, format="array")
    lead_L = lead_from_sisl(HelL, axis, "L", k, spin)
    lead_R = lead_from_sisl(HelR, axis, "R", k, spin)
    if len(left_orbs) != HelL.no or len(right_orbs) != HelR.no:
        raise ValueError("electrode orbital count does not match left/right device orbitals")
    return device_from_matrices(Hk, Sk, list(left_orbs), list(right_orbs),
                                lead_L, lead_R, Hdev.orthogonal)


def from_siesta(device_tshs, elecL_tshs, elecR_tshs, axis, left_atoms, right_atoms,
                k=(0.0, 0.0, 0.0), spin=0):
    """Read TranSIESTA / SIESTA .TSHS files and build the electron BlockSystem.
    left_atoms / right_atoms: device atom indices of the electrode copies
    (TranSIESTA's 'electrode-position'), in electrode order."""
    import sisl
    Hd = sisl.get_sile(device_tshs).read_hamiltonian()
    HL = sisl.get_sile(elecL_tshs).read_hamiltonian()
    HR = sisl.get_sile(elecR_tshs).read_hamiltonian()
    lo = Hd.geometry.a2o(np.asarray(list(left_atoms)), all=True)
    ro = Hd.geometry.a2o(np.asarray(list(right_atoms)), all=True)
    return from_sisl(Hd, HL, HR, axis, lo, ro, k, spin)
