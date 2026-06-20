"""Safety wrappers for MD integration with potentially divergent force fields.

`ForceCappedCalculator` wraps an inner ASE calculator and clamps per-atom
force magnitudes to a maximum value, preventing NaN from severely overlapping
configurations during early equilibration / compression stages.

Used during Stages A–D of the simulator pipeline; removed for NVE production
so that the true MACE forces drive the dipole dynamics.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class ForceCappedCalculator(Calculator):
    """Wrap an ASE calculator and clamp |F_i| ≤ F_max per atom.

    Analogous to GROMACS ``mdrun -emstep`` / LAMMPS ``fix nve/limit``:
    bounds the force magnitude so the leapfrog integrator cannot blow up
    when starting from a high-energy configuration.

    Parameters
    ----------
    inner_calc : ase.calculators.calculator.Calculator
        The underlying force field (e.g. MACECalculator).
    F_max : float
        Maximum per-atom force magnitude in eV/Å.  Forces above this are
        rescaled (direction preserved).  50 eV/Å is roughly 5× a typical
        bonded equilibrium force gradient.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, inner_calc, F_max: float = 50.0):
        super().__init__()
        self.inner = inner_calc
        self.F_max = float(F_max)

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        E = float(self.inner.get_potential_energy(atoms))
        F = np.asarray(self.inner.get_forces(atoms), dtype=float)

        f_norm = np.linalg.norm(F, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.F_max / np.clip(f_norm, 1e-10, None))
        F_capped = F * scale

        self.results = {"energy": E, "forces": F_capped}
        # Pass the true (uncapped) stress through so an NPT barostat can run
        # even while forces are capped.  Note: with active capping the stress
        # is not strictly consistent with the capped forces, so the pipeline
        # removes the cap before the NPT stage.
        try:
            self.results["stress"] = np.asarray(self.inner.get_stress(atoms),
                                                dtype=float)
        except Exception:
            pass
