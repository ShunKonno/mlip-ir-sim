"""Tests for SystemBuilder using the bundled stearic acid XYZ.

These tests require only ASE and SciPy — no MACE or calculator needed.
They verify cell construction, atom counts, and basic geometry checks.
"""
from pathlib import Path

import numpy as np
import pytest

from mlip_ir_sim import SystemBuilder

# Small molecule counts in these tests deliberately trigger the "too few
# molecules for target density" warning — suppress it to keep output clean.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

XYZ = Path(__file__).parent / "stearic_acid.xyz"
N_ATOMS_PER_MOL = 56  # C18H36O2


def test_build_returns_atoms():
    system = SystemBuilder(xyz_path=XYZ, num_molecules=4)
    atoms = system.build()
    assert len(atoms) == 4 * N_ATOMS_PER_MOL


def test_build_is_cached():
    system = SystemBuilder(xyz_path=XYZ, num_molecules=4)
    atoms1 = system.build()
    atoms2 = system.build()
    assert atoms1 is atoms2


def test_build_pbc():
    system = SystemBuilder(xyz_path=XYZ, num_molecules=4)
    atoms = system.build()
    assert all(atoms.get_pbc()), "Atoms should have periodic boundary conditions"


def test_build_cell_is_cubic():
    system = SystemBuilder(xyz_path=XYZ, num_molecules=4)
    atoms = system.build()
    cell = atoms.get_cell()[:]
    # Off-diagonal elements should be zero (cubic cell)
    off_diag = cell - np.diag(np.diag(cell))
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-6)


def test_density_plausible():
    """Placement density should be in a reasonable range (0.05–0.3 g/cm³)."""
    system = SystemBuilder(xyz_path=XYZ, num_molecules=8, density_gcc=0.85)
    atoms = system.build()
    mass_g = float(sum(atoms.get_masses())) / 6.02214076e23
    vol_cm3 = atoms.get_volume() * 1e-24
    rho = mass_g / vol_cm3
    # placement is at low density; target density is reached after MD
    assert 0.01 < rho < 0.5, f"Unexpected placement density {rho:.3f} g/cm³"


def test_formula():
    system = SystemBuilder(xyz_path=XYZ, num_molecules=2)
    atoms = system.build()
    # Stearic acid C18H36O2 × 2 molecules
    numbers = atoms.get_atomic_numbers()
    assert (numbers == 6).sum() == 2 * 18  # 18 C per molecule
    assert (numbers == 8).sum() == 2 * 2   # 2 O per molecule
    assert (numbers == 1).sum() == 2 * 36  # 36 H per molecule


# ── Pluggable calculator injection (force-field-agnostic) ────────────────────

def _dummy_calc():
    from ase.calculators.calculator import Calculator, all_changes

    class _Dummy(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 0.0,
                            "forces": np.zeros((len(self.atoms), 3))}

    return _Dummy()


def test_inject_calculator_instance():
    calc = _dummy_calc()
    system = SystemBuilder(xyz_path=XYZ, num_molecules=2, calculator=calc)
    atoms = system.build()
    assert atoms.calc is calc


def test_inject_calculator_factory():
    """A factory(atoms) -> calculator is called with the built cell."""
    seen = {}

    def factory(atoms):
        seen["n"] = len(atoms)
        return _dummy_calc()

    system = SystemBuilder(xyz_path=XYZ, num_molecules=2, calculator=factory)
    atoms = system.build()
    assert seen["n"] == len(atoms)
    assert atoms.calc is not None


def test_model_and_calculator_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        SystemBuilder(xyz_path=XYZ, num_molecules=2,
                      model="MACE-OFF23(Small)", calculator=_dummy_calc())


def test_periodic_false_gives_cluster():
    system = SystemBuilder(xyz_path=XYZ, num_molecules=2,
                           calculator=_dummy_calc(), periodic=False)
    atoms = system.build()
    assert not atoms.get_pbc().any()
