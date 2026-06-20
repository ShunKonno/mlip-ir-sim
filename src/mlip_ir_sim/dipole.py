"""Dipole moment tracking for IR spectroscopy via MD.

Partial charges
---------------
For MLIP models that predict forces but not electron density (e.g.
MACE-OFF23), we assign partial charges once using a GAFF-like rule-based
atom-typing scheme.  Bonds are detected from covalent-radius thresholds;
each atom is assigned a charge based on its element and bonding environment
(functional group).  The charges are then normalised to the desired total
charge (0 for neutral molecules).

This approach is fast, robust, and gives physically correct dipole
fluctuations for typical organic molecules.

DipoleTracker
-------------
Once charges are assigned, the total dipole μ(t) = Σ_i q_i r_i(t) is
tracked with PBC-continuous position unwrapping to avoid discontinuous
jumps when atoms cross periodic boundaries.
"""
from __future__ import annotations

import numpy as np

# Covalent radii in Å (used for bond detection)
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "S": 1.05, "Cl": 0.99, "Br": 1.14,
    "I": 1.33, "P": 1.07,
}
_BOND_SCALE = 1.25  # multiply (r_cov_i + r_cov_j) by this to get bond threshold


def _find_bonds(atoms) -> list[tuple[int, int]]:
    """Return (i, j) bond pairs based on covalent radius thresholds."""
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    n = len(atoms)
    bonds: list[tuple[int, int]] = []
    radii = np.array([_COVALENT_RADII.get(s, 1.0) for s in syms])

    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(pos[i] - pos[j]))
            threshold = _BOND_SCALE * (radii[i] + radii[j])
            if r < threshold:
                bonds.append((i, j))
    return bonds


def compute_gaff_charges(atoms, total_charge: float = 0.0) -> np.ndarray:
    """Assign GAFF-inspired partial charges for neutral HCNOFS organic molecules.

    Charges are based on element and local bonding environment (functional
    group), then renormalised to ``total_charge``.

    Parameters
    ----------
    atoms : ase.Atoms
        Molecule (single molecule; for a super-cell use ``n_mol`` tiling).
    total_charge : float
        Target net charge in elementary charge units.

    Returns
    -------
    ndarray, shape (n_atoms,)
        Partial charges in units of e.
    """
    syms = atoms.get_chemical_symbols()
    n = len(atoms)

    bonds = _find_bonds(atoms)
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    charges = np.zeros(n)
    for i in range(n):
        s = syms[i]
        nb_syms = [syms[j] for j in neighbors[i]]

        if s == "H":
            if "O" in nb_syms or "N" in nb_syms:
                charges[i] = +0.44   # protic H (O-H, N-H)
            else:
                charges[i] = +0.06   # aliphatic C-H

        elif s == "O":
            n_bonds_i = len(neighbors[i])
            if "H" in nb_syms:
                charges[i] = -0.60   # hydroxyl / carboxylic acid -OH
            elif n_bonds_i == 1:
                charges[i] = -0.56   # carbonyl =O
            elif n_bonds_i == 2:
                charges[i] = -0.40   # ether / ester bridge O

        elif s == "C":
            o_count = nb_syms.count("O")
            h_count = nb_syms.count("H")
            if o_count >= 2:
                charges[i] = +0.70   # carboxyl / ester C(=O)O
            elif o_count == 1:
                charges[i] = +0.40   # C bonded to one O (e.g. aldehyde C)
            else:
                # Aliphatic: slight negative per H (C-H inductive effect)
                charges[i] = -0.06 * h_count   # CH3: -0.18, CH2: -0.12, CH: -0.06

        elif s == "N":
            h_count = nb_syms.count("H")
            charges[i] = -0.40 + 0.10 * h_count  # amine N, adjust for NH

        elif s == "S":
            charges[i] = -0.20

        elif s in ("F", "Cl", "Br", "I"):
            charges[i] = -0.15

    # Renormalise to target total charge by distributing the residual
    # uniformly across all carbon atoms (most numerous and flexible)
    deficit = total_charge - charges.sum()
    c_idx = [i for i, s in enumerate(syms) if s == "C"]
    if c_idx:
        charges[c_idx] += deficit / len(c_idx)
    else:
        charges += deficit / n  # fallback: distribute over all atoms

    return charges


class DipoleTracker:
    """Track μ(t) = Σ_i q_i r_i(t) with PBC-continuous position unwrapping.

    Parameters
    ----------
    atoms : ase.Atoms
        Initial configuration of the simulation cell (multiple molecules OK).
    charges : ndarray or None
        Per-atom partial charges (e).  If *None*, GAFF-like charges are
        computed for the first molecule and tiled across the cell.
    n_mol : int or None
        Number of identical molecules in the cell (for charge tiling).
    """

    def __init__(self, atoms, charges=None, n_mol: int | None = None):
        if charges is None:
            n_total = len(atoms)
            if n_mol is not None and n_mol > 1:
                n_per_mol = n_total // n_mol
                mol_charges = compute_gaff_charges(atoms[:n_per_mol])
                charges = np.tile(mol_charges, n_mol)
                remainder = n_total - n_mol * n_per_mol
                if remainder:
                    charges = np.concatenate(
                        [charges, compute_gaff_charges(atoms[n_mol * n_per_mol:])]
                    )
            else:
                charges = compute_gaff_charges(atoms)

        if len(charges) != len(atoms):
            raise ValueError(
                f"charges length ({len(charges)}) ≠ n_atoms ({len(atoms)})"
            )
        self.charges = np.asarray(charges, dtype=float)
        self._prev_scaled = atoms.get_scaled_positions().copy()
        self._unwrapped = atoms.get_positions().copy()

    def update(self, atoms, charges: np.ndarray | None = None) -> np.ndarray:
        """Advance one step; return total dipole μ (e·Å, shape (3,)).

        Parameters
        ----------
        atoms : ase.Atoms
            Current configuration (used for positions and cell).
        charges : ndarray or None
            Per-atom charges for this step.  When provided (e.g. from
            AIMNet2's dynamic charge output), these override the stored
            ``self.charges``, capturing the electronic polarisation response
            ∂q/∂r that fixed-charge models miss.  When None (default), the
            charges stored at initialisation are used.
        """
        s_new = atoms.get_scaled_positions()
        ds = s_new - self._prev_scaled
        ds -= np.round(ds)  # wrap Δs to (−0.5, 0.5] to detect PBC crossings
        self._unwrapped += ds @ atoms.get_cell()
        self._prev_scaled = s_new.copy()
        q = self.charges if charges is None else np.asarray(charges, dtype=float)
        return q @ self._unwrapped  # shape (3,)
