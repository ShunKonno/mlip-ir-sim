"""Force-field-agnostic integration tests for IRSpectrumSimulator.

These use tiny custom ASE calculators (no MACE / no heavy deps) to prove the
pipeline adapts to whatever the calculator provides:

* forces-only, periodic       → NVT + NVE (NPT auto-skipped, no stress)
* forces + per-atom charges    → dynamic per-step dipole
* non-periodic cluster         → densify / NPT auto-skipped

The systems and trajectories are deliberately minimal so the suite stays fast.
"""
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from mlip_ir_sim import IRSpectrum, IRSpectrumSimulator, SystemBuilder

XYZ = Path(__file__).parent / "stearic_acid.xyz"

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# Very short MD so the integration tests run in a couple of seconds.
FAST = dict(
    temperature=300.0,
    eq_time_ps=0.002,
    prod_time_ps=0.01,
    timestep_fs=1.0,
    cold_eq_ps=0.002,
    warmup_ps=0.002,
    compress_ps=0.002,
    logfile=None,
)


class _ForcesOnly(Calculator):
    """Cheapest possible potential: zero energy/forces, no stress, no charges."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0,
                        "forces": np.zeros((len(self.atoms), 3))}


class _WithCharges(Calculator):
    """Forces-only potential that also outputs per-atom charges (AIMNet-style)."""

    implemented_properties = ["energy", "forces", "charges"]

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        q = np.zeros(n)
        q[0::2], q[1::2] = 0.01, -0.01   # neutral, tiny fluctuating dipole
        self.results = {"energy": 0.0, "forces": np.zeros((n, 3)), "charges": q}


def test_forces_only_periodic_runs():
    """A forces-only periodic calculator → NVT+NVE, NPT auto-skipped."""
    system = SystemBuilder(xyz_path=XYZ, num_molecules=2,
                           calculator=_ForcesOnly())
    sim = IRSpectrumSimulator(system)
    # npt_ps > 0 on purpose: a stress-less calculator would crash NPTBerendsen
    # if the skip did not happen, so a clean return proves the auto-skip.
    spec = sim.run(npt_ps=1.0, charge_method="gaff", **FAST)
    assert isinstance(spec, IRSpectrum)
    assert spec.frequencies.shape == spec.intensities.shape


def test_dynamic_charges_runs():
    """A charge-outputting calculator → dynamic per-step dipole path."""
    system = SystemBuilder(xyz_path=XYZ, num_molecules=2,
                           calculator=_WithCharges())
    sim = IRSpectrumSimulator(system)
    spec = sim.run(npt_ps=0.0, **FAST)
    assert isinstance(spec, IRSpectrum)


def test_non_periodic_cluster_runs():
    """A non-periodic cluster → densify / NPT auto-skipped."""
    system = SystemBuilder(xyz_path=XYZ, num_molecules=2,
                           calculator=_ForcesOnly(), periodic=False)
    atoms = system.build()
    assert not atoms.get_pbc().any()
    sim = IRSpectrumSimulator(system)
    spec = sim.run(npt_ps=1.0, charge_method="gaff", **FAST)
    assert isinstance(spec, IRSpectrum)
