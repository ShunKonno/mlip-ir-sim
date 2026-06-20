"""AIMNet2 calculator wrapper for IR spectroscopy simulations.

AIMNet2 is a message-passing ML potential that jointly predicts
energies, forces, and per-atom partial charges at each step, making
dynamic-charge dipole tracking possible without extra QM calculations.

Supported models (alias → ωB97M-D3 or B973c-D3 level DFT):
    "aimnet2"          — ωB97M-D3/def2-TZVP  (default, best for organics)
    "aimnet2-b973c"    — B973c-D3 (faster, slightly less accurate)
    "aimnet2-2025"     — B973c-D3 2025 retrain (newest)

Models are downloaded automatically on first use to ~/.cache/aimnet.

Note: AIMNet2 is a molecular model — it does not use periodic images.
For crystal simulations, the supercell is treated as a finite cluster.
Intramolecular forces are accurate; long-range periodic electrostatics
are not captured (use MACE-OFF23 for crystal-phase force accuracy).
"""
from __future__ import annotations


def build_aimnet2_calculator(atoms, model: str = "aimnet2", device: str = "cpu"):
    """Return an AIMNet2ASE calculator attached to *atoms*.

    Parameters
    ----------
    atoms : ase.Atoms
        System (PBC will be disabled — AIMNet2 is a molecular model).
    model : str
        AIMNet2 model alias or path.  Default: ``"aimnet2"``
        (ωB97M-D3/def2-TZVP).
    device : str
        Torch device string: ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """
    try:
        from aimnet.calculators.aimnet2ase import AIMNet2ASE
        from aimnet.calculators.calculator import AIMNet2Calculator
    except ImportError as e:
        raise ImportError(
            "AIMNet2 not found.  Install with:  pip install aimnet"
        ) from e

    # AIMNet2 is a gas-phase molecular model: disable PBC so the calculator
    # does not attempt to compute neighbours across periodic images.
    atoms.pbc = False

    # AIMNet2 model weights are float64; MPS (Apple Silicon GPU) does not
    # support float64 → fall back to CPU automatically.
    if device == "mps":
        print("[AIMNet2] WARNING: MPS does not support float64 — using CPU instead.")
        device = "cpu"

    base = AIMNet2Calculator(model=model, device=device)
    calc = AIMNet2ASE(base_calc=base, charge=0, mult=1)
    print(f"[AIMNet2] Calculator ready: model={model!r}, device={device}")
    print(f"[AIMNet2] Properties: {calc.implemented_properties}")
    return calc
