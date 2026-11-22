"""Phonon front end: force constants -> mass-weighted dynamical matrix -> BlockSystem.

Device convention (as for electrons): the device contains one copy of each electrode
principal layer at its ends; electrodes are described by K00 / K01 extracted from a
supercell of >= 3 principal layers ordered layer by layer.
"""
import numpy as np
from .system import BlockSystem, Lead
from .units import HB2
from .io_sisl import partition_btd


# --------------------------------------------------------------- readers
def read_phonopy_fc(path):
    """phonopy FORCE_CONSTANTS (full, text) -> (N, N, 3, 3) array in eV/A^2."""
    with open(path) as f:
        head = f.readline().split()
        n1, n2 = int(head[0]), int(head[-1])
        if n1 != n2:
            raise ValueError("compact FORCE_CONSTANTS: write full format (FC_FULL = .TRUE.)")
        fc = np.zeros((n1, n2, 3, 3))
        for _ in range(n1 * n2):
            i, j = (int(x) - 1 for x in f.readline().split()[:2])
            for a in range(3):
                fc[i, j, a] = [float(x) for x in f.readline().split()]
    return fc


def enforce_asr(fc):
    """Acoustic sum rule: Phi_ii = -sum_{j != i} Phi_ij (also works for (3N,3N) with d=3)."""
    fc = fc.copy()
    if fc.ndim == 4:
        for i in range(fc.shape[0]):
            fc[i, i] = 0.0
            fc[i, i] = -fc[i].sum(0)
    else:
        raise ValueError("pass (N, N, 3, 3)")
    return fc


def dynamical_matrix(fc, masses, axes=(0, 1, 2)):
    """K [eV^2] from Phi [eV/A^2] and masses [amu]. fc: (N,N,3,3) or (N,N) for 1 dof/atom.
    axes selects Cartesian components kept (e.g. (0,) for a longitudinal 1D model)."""
    m = np.asarray(masses, float)
    if fc.ndim == 2:
        return fc / np.sqrt(np.outer(m, m)) * HB2
    N, d = fc.shape[0], len(axes)
    sub = fc[:, :, list(axes)][:, :, :, list(axes)]
    K = sub.transpose(0, 2, 1, 3).reshape(N * d, N * d)
    mm = np.repeat(m, d)
    return K / np.sqrt(np.outer(mm, mm)) * HB2


def layer_blocks(K_super, n_layer):
    """Principal-layer blocks from a supercell dynamical matrix whose dofs are ordered
    layer by layer (>= 3 layers). Returns {-1: K_{m,m-1}, 0: K_mm, +1: K_{m,m+1}}."""
    nl = K_super.shape[0] // n_layer
    if nl < 3:
        raise ValueError("need >= 3 principal layers in the supercell")
    m = nl // 2
    s = lambda i: slice(i * n_layer, (i + 1) * n_layer)
    return {-1: K_super[s(m), s(m - 1)], 0: K_super[s(m), s(m)], +1: K_super[s(m), s(m + 1)]}


# ------------------------------------------------------- device assembly
def phonons_from_K(K, first_atoms, last_atoms, leadL, leadR, dof=3, min_block=0):
    """Build the phonon BlockSystem.
    K            : device dynamical matrix (eV^2), dofs atom-major (atom i -> i*dof..)
    first/last_atoms : device atoms that are the electrode-layer copies (electrode order)
    leadL/leadR  : dicts from layer_blocks for the left / right electrodes
    min_block    : merge BFS layers until interior blocks have >= min_block atoms
    Returns system with .order (device dof index of each RGF row) and .atom_order."""
    na = K.shape[0] // dof
    Ka = np.abs(K).reshape(na, dof, na, dof).max(axis=(1, 3))
    ablocks = partition_btd(Ka, first_atoms, last_atoms)
    ablocks = coarsen(ablocks, min_block)
    lset = set(int(a) for a in last_atoms)
    rest = [a for a in ablocks[-1] if int(a) not in lset]
    ablocks[-1] = np.array(rest + list(last_atoms), int)
    ablocks[0] = np.array(list(first_atoms), int)
    dofs = [np.concatenate([np.arange(a * dof, (a + 1) * dof) for a in b]) for b in ablocks]
    Kd = [K[np.ix_(b, b)] for b in dofs]
    Kc = [K[np.ix_(dofs[i], dofs[i + 1])] for i in range(len(dofs) - 1)]
    L = Lead(H00=leadL[0], H01=leadL[-1], tau=leadL[-1])
    nR = leadR[0].shape[0]
    rows = np.zeros((len(dofs[-1]), nR)); rows[len(dofs[-1]) - nR:] = np.eye(nR)
    R = Lead(H00=leadR[0], H01=leadR[+1], tau=rows @ leadR[+1])
    ph = BlockSystem(Kd, Kc, L, R)
    ph.order = np.concatenate(dofs)
    ph.atom_order = np.concatenate(ablocks)
    ph.dof = dof
    return ph


def coarsen(blocks, min_block):
    """Merge consecutive interior layers until each has >= min_block entries
    (first and last blocks are kept as they are; a short remainder joins the
    previous interior block)."""
    blocks = [np.asarray(b, int) for b in blocks]
    if min_block <= 1 or len(blocks) <= 3:
        return blocks
    inner, cur = [], []
    for b in blocks[1:-1]:
        cur.extend(b.tolist())
        if len(cur) >= min_block:
            inner.append(np.array(cur, int)); cur = []
    if cur:
        if inner:
            inner[-1] = np.concatenate([inner[-1], np.array(cur, int)])
        else:
            inner.append(np.array(cur, int))
    return [blocks[0]] + inner + [blocks[-1]]


def chain_fc(n, k_spring):
    """Nearest-neighbour 1D chain force constants (n, n) in eV/A^2 (1 dof per atom),
    with the ASR satisfied for an infinite chain (interior diagonal 2k)."""
    fc = 2 * k_spring * np.eye(n) - k_spring * (np.eye(n, k=1) + np.eye(n, k=-1))
    return fc
