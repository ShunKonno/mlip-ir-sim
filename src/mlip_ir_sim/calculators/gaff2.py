"""GAFF2 classical force field calculator via OpenMM + AmberTools.

Requirements (available after setup):
  - antechamber / parmchk2 / tleap  (AmberTools, found at /opt/miniconda3/bin/)
  - parmed                           (pip install parmed)
  - openmm                           (pip install openmm)

Atom-ordering note
------------------
antechamber produces a specific ordering within each molecule:
  18 C (terminal → carboxyl), 2 O (carbonyl, hydroxyl), 36 H

When the input comes from a CIF crystal structure, the ASE atom ordering
within each molecule differs.  _mol_perm() detects the GAFF2 atom types
from the bond graph and computes the permutation needed to map between the
two orderings, so the calculator accepts any consistent per-molecule ordering.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

_AMBER_DIRS = ['/opt/miniconda3/bin', '/opt/homebrew/bin', '/usr/local/bin']


def _find(name: str) -> str:
    hit = shutil.which(name)
    if hit:
        return hit
    for d in _AMBER_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError(
        f"'{name}' not found.  Install AmberTools:\n"
        "    conda install -c conda-forge ambertools\n"
        f"Searched: PATH + {_AMBER_DIRS}"
    )


def _parameterise(cache_dir: Path) -> tuple[str, str]:
    """Run antechamber → parmchk2 → tleap; return (prmtop, inpcrd) paths."""
    prmtop = cache_dir / 'stearic.prmtop'
    inpcrd = cache_dir / 'stearic.inpcrd'
    if prmtop.exists() and inpcrd.exists():
        return str(prmtop), str(inpcrd)

    cache_dir.mkdir(parents=True, exist_ok=True)
    antechamber = _find('antechamber')
    parmchk2    = _find('parmchk2')
    tleap       = _find('tleap')
    obabel      = shutil.which('obabel') or '/opt/homebrew/bin/obabel'

    amber_bin  = str(Path(antechamber).parent)
    amber_home = str(Path(antechamber).parent.parent)
    env = os.environ.copy()
    env['AMBERHOME'] = amber_home
    env['PATH'] = amber_bin + ':' + env.get('PATH', '')

    smi_file = cache_dir / 'stearic.smi'
    smi_file.write_text('CCCCCCCCCCCCCCCCCC(=O)O stearic\n')
    mol2_raw = cache_dir / 'stearic_raw.mol2'
    subprocess.run(
        [obabel, str(smi_file), '--gen3d', '-omol2', '-O', str(mol2_raw)],
        check=True, capture_output=True,
    )

    mol2_gaff = cache_dir / 'stearic_gaff2.mol2'
    subprocess.run(
        [antechamber,
         '-i', str(mol2_raw),  '-fi', 'mol2',
         '-o', str(mol2_gaff), '-fo', 'mol2',
         '-c', 'bcc', '-nc', '0', '-m', '1',
         '-at', 'gaff2', '-s', '2', '-pf', 'y'],
        check=True, capture_output=True, cwd=str(cache_dir), env=env,
    )

    frcmod = cache_dir / 'stearic.frcmod'
    subprocess.run(
        [parmchk2,
         '-i', str(mol2_gaff), '-f', 'mol2',
         '-o', str(frcmod), '-s', '2'],
        check=True, capture_output=True, cwd=str(cache_dir), env=env,
    )

    leap_in = cache_dir / 'leap.in'
    leap_in.write_text(
        f"source leaprc.gaff2\n"
        f"mol = loadmol2 {mol2_gaff}\n"
        f"loadamberparams {frcmod}\n"
        f"saveamberparm mol {prmtop} {inpcrd}\n"
        f"quit\n"
    )
    result = subprocess.run(
        [tleap, '-f', str(leap_in)],
        capture_output=True, text=True, cwd=str(cache_dir), env=env,
    )
    if not prmtop.exists():
        raise RuntimeError(
            f"tleap failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    print(f"[GAFF2] Parameterisation done → {prmtop}")
    return str(prmtop), str(inpcrd)


def _mol_perm(atoms_mol, cell: np.ndarray) -> np.ndarray:
    """Compute the permutation that maps this molecule's atom ordering to GAFF2 ordering.

    GAFF2 ordering (from antechamber, stearic acid):
      [0..17]  18 C's: terminal methyl first, carboxyl C last
      [18]     carbonyl O  (=O)
      [19]     hydroxyl O  (-OH)
      [20..54] 35 H's on C: grouped by bonded C, terminal methyl first
      [55]     hydroxyl H

    Returns perm such that  gaff2_positions = cif_positions[inv_perm],
    and  cif_forces = gaff2_forces[perm].
    i.e., perm[cif_i] = gaff2_j.
    """
    from mlip_ir_sim.charges import _unwrap_molecule

    n = len(atoms_mol)
    nums = atoms_mol.get_atomic_numbers()
    spos = atoms_mol.get_scaled_positions(wrap=False)
    pos  = _unwrap_molecule(spos, cell)

    # Element-pair bond cutoffs (Å).  Tighter than 1.9 Å to avoid spurious
    # bonds: CIF H positions can be displaced (X-ray riding-model, 0.96–1.18 Å)
    # and geminal/vicinal H…C distances can reach ~1.9 Å.
    _CUTOFFS = {
        (1, 1): 0.0,   # H-H: never bonded
        (1, 6): 1.4,   # C-H  (CIF range 0.79–1.18 Å, vicinal C…H ~2.2 Å)
        (1, 8): 1.2,   # O-H  (X-ray ~0.82–0.96 Å)
        (6, 6): 1.7,   # C-C  (single 1.54, double 1.34 Å)
        (6, 8): 1.6,   # C-O  (single 1.43, double 1.23 Å)
        (8, 8): 0.0,   # O-O  (not present in stearic acid)
    }

    def _bmax(zi, zj):
        return _CUTOFFS.get((min(zi, zj), max(zi, zj)), 1.9)

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cut = _bmax(int(nums[i]), int(nums[j]))
            if cut > 0 and np.linalg.norm(pos[i] - pos[j]) < cut:
                adj[i].append(j)
                adj[j].append(i)

    # Carboxyl C: the unique C bonded to exactly 2 O atoms
    carboxyl_c = next(
        i for i in range(n)
        if nums[i] == 6 and sum(1 for j in adj[i] if nums[j] == 8) == 2
    )

    # BFS along C–C bonds from carboxyl C to enumerate the chain
    c_chain: list[int] = [carboxyl_c]
    visited: set[int] = {carboxyl_c}
    current = carboxyl_c
    while True:
        nxt = next((j for j in adj[current] if nums[j] == 6 and j not in visited), None)
        if nxt is None:
            break
        c_chain.append(nxt)
        visited.add(nxt)
        current = nxt
    # c_chain: [carboxyl_C, ..., terminal_methyl_C]

    # Identify the two O atoms on carboxyl C
    o_bonds = [j for j in adj[carboxyl_c] if nums[j] == 8]
    if len(o_bonds) != 2:
        raise RuntimeError(f"Expected 2 O on carboxyl C, found {len(o_bonds)}")
    has_h = lambda o: any(nums[j] == 1 for j in adj[o])
    carbonyl_o = next(o for o in o_bonds if not has_h(o))
    hydroxyl_o = next(o for o in o_bonds if     has_h(o))
    hydroxyl_h = next(j for j in adj[hydroxyl_o] if nums[j] == 1)

    # Build perm[cif_i] = gaff2_j
    perm = np.full(n, -1, dtype=int)

    # C chain: GAFF2[0] = terminal methyl, GAFF2[17] = carboxyl
    for gaff_j, c in enumerate(reversed(c_chain)):
        perm[c] = gaff_j

    perm[carbonyl_o] = 18
    perm[hydroxyl_o] = 19

    # H's on each C (terminal first), then hydroxyl H
    gaff_h = 20
    for c in reversed(c_chain):             # terminal → carboxyl
        for h in sorted(j for j in adj[c] if nums[j] == 1):
            perm[h] = gaff_h
            gaff_h += 1
    perm[hydroxyl_h] = 55

    if (perm == -1).any():
        raise RuntimeError("Permutation incomplete — check molecule connectivity.")
    if len(set(perm)) != n:
        raise RuntimeError("Permutation has duplicates — check molecule connectivity.")
    return perm


class GAFF2Calculator:
    """OpenMM/GAFF2 ASE-compatible calculator for a periodic stearic acid cell.

    Accepts any consistent per-molecule atom ordering (random-packing XYZ or
    CIF crystal structure).  Internally remaps atoms to the GAFF2 topology
    ordering before calling OpenMM, then remaps forces back.

    Parameters
    ----------
    atoms : ase.Atoms
        Full periodic cell (n_mol × 56 atoms).
    n_mol : int
        Number of molecules.
    cache_dir : str or None
        Where to store (and reuse) the AMBER topology files.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms, n_mol: int, cache_dir: str | None = None):
        import parmed as pmd
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit

        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[3] / 'outputs' / 'gaff2_params'
        else:
            cache_dir = Path(cache_dir)

        print("[GAFF2] Parameterising stearic acid (antechamber / tleap)…")
        prmtop, inpcrd = _parameterise(cache_dir)

        n_per_mol = len(atoms) // n_mol
        self._n_mol = n_mol
        self._n_per_mol = n_per_mol

        # ── Per-molecule permutation: input ordering → GAFF2 ordering ─────
        # Compute from the first molecule's bond graph.
        # perm[input_i] = gaff2_j within one molecule.
        mol0 = atoms[:n_per_mol]
        cell = atoms.get_cell()[:]
        single_perm = _mol_perm(mol0, cell)

        # Check if ordering is already GAFF2 (identity permutation)
        is_identity = np.all(single_perm == np.arange(n_per_mol))
        if is_identity:
            self._perm     = None
            self._inv_perm = None
            print("[GAFF2] Atom ordering matches GAFF2 — no remapping needed.")
        else:
            # Build full-system permutation across all n_mol molecules
            single_inv = np.argsort(single_perm)
            full_perm     = np.empty(len(atoms), dtype=int)
            full_inv_perm = np.empty(len(atoms), dtype=int)
            for m in range(n_mol):
                off = m * n_per_mol
                full_perm    [off:off+n_per_mol] = off + single_perm
                full_inv_perm[off:off+n_per_mol] = off + single_inv
            self._perm     = full_perm       # input_i → gaff2_j
            self._inv_perm = full_inv_perm   # gaff2_j → input_i
            print(f"[GAFF2] CIF atom ordering detected — permutation applied "
                  f"({n_mol} molecules × {n_per_mol} atoms).")

        # ── Replicate single-molecule topology n_mol times ────────────────
        single = pmd.load_file(prmtop, inpcrd)
        multi  = single * n_mol if n_mol > 1 else single

        # Set initial cell (orthorhombic; NPT may update it later)
        multi.box = [cell[0, 0], cell[1, 1], cell[2, 2], 90.0, 90.0, 90.0]

        # ── OpenMM System: GAFF2 bonded + PME electrostatics + LJ ─────────
        # Cutoff must be < half the shortest periodic box dimension.
        a, b, c = cell[0, 0], cell[1, 1], cell[2, 2]
        max_cut_nm = min(a, b, c) * 0.45 / 10.0  # 90 % of half-box, Å → nm
        cutoff_nm = min(1.0, max_cut_nm)
        print(f"[GAFF2] PME cutoff: {cutoff_nm:.3f} nm "
              f"(shortest box edge: {min(a, b, c):.2f} Å)")
        system = multi.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=cutoff_nm * unit.nanometer,
            constraints=None,
            rigidWater=False,
        )

        integrator = mm.VerletIntegrator(0.0005 * unit.picoseconds)
        self._ctx = mm.Context(system, integrator,
                               mm.Platform.getPlatformByName("CPU"))

        # Set initial positions
        pos0 = atoms.get_positions()
        if self._inv_perm is not None:
            pos0 = pos0[self._inv_perm]   # input order → GAFF2 order
        self._ctx.setPositions(pos0 * 0.1)

        print(f"[GAFF2] OpenMM context ready: {n_mol} molecules, "
              f"{len(atoms)} atoms, PME cutoff {cutoff_nm:.3f} nm.")

    # ── ASE Calculator interface ──────────────────────────────────────────

    def get_potential_energy(self, atoms) -> float:
        return self._eval(atoms)[0]

    def get_forces(self, atoms) -> np.ndarray:
        return self._eval(atoms)[1]

    def get_stress(self, atoms) -> np.ndarray:
        """6-component Voigt stress [xx,yy,zz,yz,xz,xy] in eV/Å³.

        Molecular-centroid virial: for each molecule subtract its centroid before
        computing Σ (r_i - R_mol) ⊗ F_i.  This eliminates the drift of absolute
        atomic positions that makes the naive Σ r_i⊗F_i wildly wrong for PME
        systems (long-range electrostatic stress is still approximate, but the
        result is physically bounded and correct in sign).
        """
        _, F = self._eval(atoms)
        cell  = atoms.get_cell()[:]
        # wrap all atoms into [0, cell) so centroids are well-defined
        pos   = atoms.get_scaled_positions() @ cell   # shape (N, 3)
        V     = atoms.get_volume()
        n     = self._n_per_mol
        s     = np.zeros((3, 3))
        for m in range(self._n_mol):
            r = pos[m*n:(m+1)*n]          # (n_per, 3) – wrapped into cell
            f = F[m*n:(m+1)*n]            # forces in input ordering
            dr = r - r.mean(axis=0)       # centroid-relative positions
            s += np.einsum('ia,ib->ab', dr, f)
        s /= V
        return np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])

    def _eval(self, atoms) -> tuple[float, np.ndarray]:
        import openmm.unit as unit

        # Update cell (NPT may change it)
        cell = atoms.get_cell()[:]
        self._ctx.setPeriodicBoxVectors(
            [cell[0, 0]*0.1, 0.0,          0.0         ],
            [0.0,          cell[1,1]*0.1,   0.0         ],
            [0.0,          0.0,          cell[2,2]*0.1  ],
        )

        pos = atoms.get_positions()
        if self._inv_perm is not None:
            pos = pos[self._inv_perm]          # input → GAFF2 order
        self._ctx.setPositions(pos * 0.1)      # Å → nm

        state = self._ctx.getState(getEnergy=True, getForces=True)
        E_kJ = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        F_kJ_nm = np.array(
            state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
        )
        # kJ/mol/nm → eV/Å  (÷ 964.85)
        F_eV_A = F_kJ_nm / 964.85

        if self._perm is not None:
            F_eV_A = F_eV_A[self._perm]        # GAFF2 → input order

        return E_kJ / 96.485, F_eV_A
