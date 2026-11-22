"""ephnegf -- coupled electron-phonon NEGF (SCBA) built from scratch.

Stage 3: block-tridiagonal RGF for electrons and phonons, Sancho-Rubio leads,
local (Holstein-type) e-ph self-energies, particle/energy-conserving SCBA.
"""
from .system import BlockSystem, Lead, Coupling, chain_electrons, chain_phonons, chain_coupling
from .scba import CoupledSCBA, Settings
from .leads import sancho_rubio, lead_selfenergy
from .rgf import rgf_diagonal

__version__ = "0.3.0"
