"""GAFF2 classical force field calculator via OpenMM + AmberTools.

Requirements
------------
- antechamber / parmchk2 / tleap  (AmberTools — conda install -c conda-forge ambertools)
- obabel                           (Open Babel  — conda install -c conda-forge openbabel)
- parmed                           (pip install parmed)
- openmm                           (pip install openmm)

Usage
-----
Pass the full periodic cell (all molecules), the molecule count, a short
name used for cache files, and an optional cache directory::

    from mlip_ir_sim.calculators.gaff2 import GAFF2Calculator
    calc = GAFF2Calculator(atoms, n_mol=32, mol_name="my_molecule")
    atoms.calc = calc

Atom-ordering note
------------------
Parameterisation is derived from the *first molecule* in the ``atoms``
object (atoms[:n_per_mol]).  antechamber and tleap preserve the mol2
atom order, so the resulting AMBER topology has the same atom ordering
as the ASE ``atoms`` object — no internal permutation is applied.

The cached topology files are stored under ``cache_dir/{mol_name}.*``
and reused on subsequent runs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

_AMBER_DIRS = ['/opt/miniconda3/bin', '/opt/homebrew/bin', '/usr/local/bin']


def _find(name: str) -> str:
    """Find an executable in PATH or well-known AmberTools install locations."""
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


def _parameterise(mol_atoms, mol_name: str, cache_dir: Path) -> str:
    """Derive GAFF2 parameters for *mol_atoms* and return the prmtop path.

    Parameters
    ----------
    mol_atoms : ase.Atoms
        A single molecule with explicit H atoms and no periodic boundary
        conditions.  The atom ordering is preserved in the output topology.
    mol_name : str
        Short identifier used for all cached file names
        (e.g. ``"stearic_acid"``).
    cache_dir : Path
        Directory for topology cache files.  Reused on subsequent calls.

    Returns
    -------
    str
        Path to the ``.prmtop`` file.
    """
    import ase.io

    prmtop = cache_dir / f'{mol_name}.prmtop'
    inpcrd = cache_dir / f'{mol_name}.inpcrd'
    if prmtop.exists() and inpcrd.exists():
        return str(prmtop)

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

    # Write molecule to XYZ, convert to mol2 (obabel preserves atom order).
    # --gen3d is NOT used: the input already has optimised 3D coordinates.
    mol_copy = mol_atoms.copy()
    mol_copy.pbc = False
    xyz_path = cache_dir / f'{mol_name}_input.xyz'
    mol2_raw = cache_dir / f'{mol_name}_raw.mol2'
    ase.io.write(str(xyz_path), mol_copy)
    subprocess.run(
        [obabel, str(xyz_path), '-omol2', '-O', str(mol2_raw)],
        check=True, capture_output=True,
    )

    # antechamber: assign GAFF2 atom types (preserves input atom ordering)
    mol2_gaff = cache_dir / f'{mol_name}_gaff2.mol2'
    subprocess.run(
        [antechamber,
         '-i', str(mol2_raw),  '-fi', 'mol2',
         '-o', str(mol2_gaff), '-fo', 'mol2',
         '-c', 'bcc', '-nc', '0', '-m', '1',
         '-at', 'gaff2', '-s', '2', '-pf', 'y'],
        check=True, capture_output=True, cwd=str(cache_dir), env=env,
    )

    frcmod = cache_dir / f'{mol_name}.frcmod'
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
    return str(prmtop)


class GAFF2Calculator:
    """OpenMM/GAFF2 ASE-compatible periodic-cell calculator.

    Parameterises the molecule from the first molecule in *atoms* and
    replicates the topology to fill the full cell.  Any organic molecule
    with explicit H atoms is supported.

    Parameters
    ----------
    atoms : ase.Atoms
        Full periodic cell (``n_mol`` molecules, all atoms explicit).
    n_mol : int
        Number of molecules in the cell.
    mol_name : str
        Short identifier for topology cache files (e.g. ``"stearic_acid"``).
    cache_dir : str or None
        Directory for AMBER topology cache.  Defaults to
        ``<package_root>/outputs/gaff2_params``.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        atoms,
        n_mol: int,
        mol_name: str = "molecule",
        cache_dir: str | None = None,
    ):
        import parmed as pmd
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit

        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[3] / 'outputs' / 'gaff2_params'
        else:
            cache_dir = Path(cache_dir)

        n_per_mol = len(atoms) // n_mol
        self._n_mol     = n_mol
        self._n_per_mol = n_per_mol

        print(f"[GAFF2] Parameterising {mol_name} ({n_per_mol} atoms/molecule)…")
        mol0 = atoms[:n_per_mol].copy()
        prmtop = _parameterise(mol0, mol_name, cache_dir)

        # ── Replicate single-molecule topology n_mol times ────────────────
        single = pmd.load_file(prmtop)
        multi  = single * n_mol if n_mol > 1 else single

        cell = atoms.get_cell()[:]
        multi.box = [cell[0, 0], cell[1, 1], cell[2, 2], 90.0, 90.0, 90.0]

        # ── OpenMM System: GAFF2 bonded + PME electrostatics + LJ ─────────
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
        self._ctx.setPositions(atoms.get_positions() * 0.1)

        print(f"[GAFF2] OpenMM context ready: {n_mol} molecules × "
              f"{n_per_mol} atoms, PME cutoff {cutoff_nm:.3f} nm.")

    # ── ASE Calculator interface ──────────────────────────────────────────

    def get_potential_energy(self, atoms) -> float:
        return self._eval(atoms)[0]

    def get_forces(self, atoms) -> np.ndarray:
        return self._eval(atoms)[1]

    def get_stress(self, atoms) -> np.ndarray:
        """6-component Voigt stress [xx,yy,zz,yz,xz,xy] in eV/Å³.

        Molecular-centroid virial to avoid drift artefacts in PME systems.
        """
        _, F = self._eval(atoms)
        cell = atoms.get_cell()[:]
        pos  = atoms.get_scaled_positions() @ cell
        V    = atoms.get_volume()
        n    = self._n_per_mol
        s    = np.zeros((3, 3))
        for m in range(self._n_mol):
            r  = pos[m*n:(m+1)*n]
            f  = F[m*n:(m+1)*n]
            dr = r - r.mean(axis=0)
            s += np.einsum('ia,ib->ab', dr, f)
        s /= V
        return np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])

    def _eval(self, atoms) -> tuple[float, np.ndarray]:
        import openmm.unit as unit

        cell = atoms.get_cell()[:]
        self._ctx.setPeriodicBoxVectors(
            [cell[0, 0]*0.1, 0.0,          0.0         ],
            [0.0,          cell[1,1]*0.1,   0.0         ],
            [0.0,          0.0,          cell[2,2]*0.1  ],
        )
        self._ctx.setPositions(atoms.get_positions() * 0.1)

        state = self._ctx.getState(getEnergy=True, getForces=True)
        E_kJ = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        F_kJ_nm = np.array(
            state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
        )
        return E_kJ / 96.485, F_kJ_nm / 964.85  # kJ/mol → eV, kJ/mol/nm → eV/Å
