"""Block-tridiagonal descriptions of the electron and phonon subsystems + builders."""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Lead:
    H00: np.ndarray            # principal-layer block (H or K)
    H01: np.ndarray            # coupling to the next layer, pointing AWAY from the device
    tau: np.ndarray            # coupling device edge block -> first lead layer (b_edge, n_lead)
    S00: Optional[np.ndarray] = None   # overlaps (non-orthogonal basis); None = orthogonal
    S01: Optional[np.ndarray] = None
    tauS: Optional[np.ndarray] = None


@dataclass
class BlockSystem:
    """Generic block-tridiagonal operator (electron H or phonon dynamical matrix K)."""
    Hd: List[np.ndarray]            # diagonal blocks
    Hc: List[np.ndarray]            # upper off-diagonal blocks H_{i,i+1}
    left: Lead
    right: Lead
    Sd: Optional[List[np.ndarray]] = None   # overlap blocks (None = orthogonal basis)
    Sc: Optional[List[np.ndarray]] = None

    @property
    def orthogonal(self):
        return self.Sd is None

    def a_blocks(self, z):
        """Diagonal blocks z S_ii - H_ii (no leads) and off-diagonal U_i, L_i of A = zS - H."""
        z = np.asarray(z, complex)
        if self.orthogonal:
            Ad = [z[:, None, None] * np.eye(h.shape[0]) - h for h in self.Hd]
            U = [-T for T in self.Hc]
            L = [-T.conj().T for T in self.Hc]
        else:
            Ad = [z[:, None, None] * S - h for S, h in zip(self.Sd, self.Hd)]
            U = [z[:, None, None] * S - T for S, T in zip(self.Sc, self.Hc)]
            L = [z[:, None, None] * S.conj().T - T.conj().T for S, T in zip(self.Sc, self.Hc)]
        return Ad, U, L

    @property
    def sizes(self):
        return [h.shape[0] for h in self.Hd]

    @property
    def offsets(self):
        return np.cumsum([0] + self.sizes)

    @property
    def n(self):
        return int(sum(self.sizes))


@dataclass
class Coupling:
    """Local (Holstein-type) e-ph coupling  H_eph = sum_o g_o n_o u_{a(o)}.
    g    : (n_orb,) coupling of each electron orbital (eV^{3/2}, mass-weighted u, hbar=1)
    dof  : (n_orb,) index of the phonon degree of freedom each orbital couples to
    Einstein-mode equivalence: g = M sqrt(2 w0).
    """
    g: np.ndarray
    dof: np.ndarray


# ------------------------------------------------------------------ builders
def chain_electrons(eps, t_dev=1.0, t_lead=2.0, tc=0.6):
    """1D tight-binding chain, one orbital per block, 1D chain leads."""
    eps = np.atleast_1d(np.asarray(eps, float))
    Hd = [np.array([[e]]) for e in eps]
    Hc = [np.array([[-t_dev]]) for _ in range(len(eps) - 1)]
    lead = lambda: Lead(np.array([[0.0]]), np.array([[-t_lead]]), np.array([[-tc]]))
    return BlockSystem(Hd, Hc, lead(), lead())


def chain_phonons(omega0, n, k_dev=0.0, k_lead=0.0064, k_c=8e-4):
    """1D mass-weighted harmonic chain (units eV^2).  Device atoms have K_ii = omega0^2
    (frequency of the clamped device mode incl. its springs), coupled by -k_dev to
    neighbours and by -k_c to acoustic chain leads (band top 2 sqrt(k_lead))."""
    Hd = [np.array([[omega0 ** 2]]) for _ in range(n)]
    Hc = [np.array([[-k_dev]]) for _ in range(n - 1)]
    lead = lambda: Lead(np.array([[2 * k_lead]]), np.array([[-k_lead]]), np.array([[-k_c]]))
    return BlockSystem(Hd, Hc, lead(), lead())


def chain_coupling(n, M, omega0):
    return Coupling(g=np.full(n, M * np.sqrt(2 * omega0)), dof=np.arange(n))
