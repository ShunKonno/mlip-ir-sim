"""One-shot GFN2-xTB partial charges at the equilibrated geometry.

Called once after equilibration (Stage E): each molecule is extracted from
the periodic cell in its *actual* equilibrated conformation, and GFN2-xTB
(semi-empirical tight-binding, via tblite) yields Mulliken partial charges.
These replace the rule-based GAFF-like charges for the production dipole
μ(t) = Σ q_i r_i(t), so the charges reflect the real conformer geometry
instead of generic functional-group values.

The charges remain *fixed* during production (no charge flux), so peak
intensities are still approximate — but magnitudes are now quantum-derived.
"""
from __future__ import annotations

import numpy as np

_BOHR_PER_ANG = 1.8897259886


def _unwrap_molecule(spos_blk, cell):
    """Reassemble one molecule into a continuous (non-PBC-split) geometry.

    A long chain can span more than half the cell along a short lattice
    vector, so unwrapping relative to a single reference atom fails.  Instead
    we grow the molecule along its bond graph: every atom is placed by minimum
    image relative to an *already-placed bonded neighbour*, and a single bond
    is always far shorter than half the cell, so each step is unambiguous.
    """
    n = len(spos_blk)
    pos = np.zeros((n, 3))
    placed = np.zeros(n, dtype=bool)
    pos[0] = spos_blk[0] @ cell
    placed[0] = True

    # Bond threshold in Å (covalent, generous): C–C/C–O/C–H all < 1.8 Å.
    bond_max = 1.9
    frontier = [0]
    while frontier:
        i = frontier.pop()
        for j in range(n):
            if placed[j]:
                continue
            ds = spos_blk[j] - spos_blk[i]
            ds -= np.round(ds)              # minimum image of the j–i vector
            rij = ds @ cell
            if np.linalg.norm(rij) < bond_max:
                pos[j] = pos[i] + rij
                placed[j] = True
                frontier.append(j)
    if not placed.all():
        raise RuntimeError(
            f"Molecule unwrap incomplete: {(~placed).sum()} atom(s) had no "
            "bonded neighbour within 1.9 Å (broken connectivity?)."
        )
    return pos


def wrap_molecules(atoms, n_mol: int,
                   prev_centroids: np.ndarray | None = None):
    """Return a copy with every molecule made whole and wrapped by centroid.

    Plain per-atom wrapping (``atoms.wrap()``) splits any molecule that
    straddles a periodic boundary — half its atoms jump to the opposite face,
    so bonds appear stretched across the whole box.  Here each molecule is
    first reassembled along its bond graph (so it is contiguous), then the
    *whole* molecule is shifted by an integer number of lattice vectors so its
    centroid lies inside the cell.  Molecules stay intact and inside the box,
    which is what a visualiser (OVITO/VMD) expects.

    Parameters
    ----------
    prev_centroids : ndarray, shape (n_mol, 3) or None
        Cartesian centroids from the previous frame.  When provided, each
        molecule is placed at the periodic image *closest* to its previous
        centroid rather than always folding into [0, 1).  This eliminates the
        sudden full-cell jumps that appear when a centroid crosses a boundary
        — an artefact that the visualiser would otherwise show as a molecule
        teleporting across the box.

    Returns
    -------
    new_atoms : ase.Atoms
        Copy with wrapped positions (MD state is not touched).
    new_centroids : ndarray, shape (n_mol, 3)
        Cartesian centroids after wrapping (pass as ``prev_centroids`` on the
        next frame to maintain continuity).
    """
    n_at = len(atoms) // n_mol
    spos = atoms.get_scaled_positions()
    cell = atoms.get_cell()[:]
    inv_cell = np.linalg.inv(cell)
    new_pos = np.empty((len(atoms), 3))
    new_centroids = np.empty((n_mol, 3))

    for m in range(n_mol):
        blk = slice(m * n_at, (m + 1) * n_at)
        pos = _unwrap_molecule(spos[blk], cell)   # whole, may poke outside
        centroid = pos.mean(axis=0)

        if prev_centroids is not None:
            # Choose the periodic image of the centroid nearest to the
            # previous frame's centroid (minimum-image convention in
            # fractional space) → no discontinuous jumps in the trajectory.
            dc = centroid - prev_centroids[m]
            dc_frac = dc @ inv_cell
            dc_frac -= np.round(dc_frac)
            centroid = prev_centroids[m] + dc_frac @ cell
            shift = centroid - pos.mean(axis=0)
            new_pos[blk] = pos + shift
        else:
            # First frame: just fold centroid into [0, 1).
            shift = np.floor(centroid @ inv_cell) @ cell
            new_pos[blk] = pos - shift
            centroid = centroid - shift

        new_centroids[m] = centroid

    out = atoms.copy()
    out.set_positions(new_pos)
    return out, new_centroids


def compute_xtb_charges_isolated(atoms, n_mol: int) -> np.ndarray:
    """GFN2-xTB charges computed in a fresh subprocess (OpenMP-safe).

    tblite and PyTorch each bundle their own OpenMP runtime (libomp).  Once
    torch's libomp is initialised — which happens during the MACE MD — loading
    tblite's copy in the *same* process aborts (``OMP: Error #15``) or
    segfaults on macOS.  Running the charge calculation in a brand-new Python
    interpreter that never imports torch sidesteps the conflict completely.

    Same inputs/outputs as :func:`compute_xtb_charges`; use this from any
    process that has already loaded torch.
    """
    import os
    import subprocess
    import sys
    import tempfile

    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    worker = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {src_dir!r})\n"
        "from ase import Atoms\n"
        "from mlip_ir_sim.charges import compute_xtb_charges\n"
        "d = np.load(sys.argv[1])\n"
        "a = Atoms(numbers=d['numbers'], positions=d['positions'], "
        "cell=d['cell'], pbc=True)\n"
        "np.save(sys.argv[2], compute_xtb_charges(a, int(d['n_mol'])))\n"
    )
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.npz")
        out = os.path.join(td, "out.npy")
        np.savez(inp,
                 numbers=atoms.get_atomic_numbers(),
                 positions=atoms.get_positions(),
                 cell=np.array(atoms.get_cell()),
                 n_mol=n_mol)
        proc = subprocess.run([sys.executable, "-c", worker, inp, out],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(
                "xTB charge subprocess failed (returncode "
                f"{proc.returncode}):\n{proc.stderr[-2000:]}")
        return np.load(out)


def compute_xtb_charges(atoms, n_mol: int) -> np.ndarray:
    """GFN2-xTB Mulliken charges for every molecule in the cell.

    Molecules must be stored as contiguous equal-size blocks (the builder
    guarantees this).  Each molecule is unwrapped along its bond graph so a
    chain that crosses a periodic boundary is reassembled correctly before
    the single-point calculation.

    Run via :func:`compute_xtb_charges_isolated` from any process that has
    already imported torch (e.g. during MD) — see that function for why.

    Returns
    -------
    ndarray, shape (n_atoms,)
        Partial charges in e, each molecule renormalised to net 0.
    """
    from tblite.interface import Calculator

    n_at = len(atoms) // n_mol
    numbers = atoms.get_atomic_numbers()
    spos = atoms.get_scaled_positions()
    cell = atoms.get_cell()[:]
    charges = np.empty(len(atoms))

    for m in range(n_mol):
        blk = slice(m * n_at, (m + 1) * n_at)
        pos = _unwrap_molecule(spos[blk], cell)

        calc = Calculator("GFN2-xTB", numbers[blk], pos * _BOHR_PER_ANG)
        calc.set("verbosity", 0)
        q = calc.singlepoint().get("charges")
        q -= q.sum() / n_at  # exact neutrality per molecule
        charges[blk] = q

    return charges
