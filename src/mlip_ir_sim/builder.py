"""Simulation cell builder.

Reads a single-molecule XYZ (random packing) or a CIF (crystal), builds a
periodic cell, and attaches a force field / MLIP.  The calculator is fully
pluggable: pass any ASE calculator instance (or a ``factory(atoms) -> calc``
callable), or give a ``model`` string to use the built-in MACE shortcut.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SystemConfig:
    xyz_path: Path | None
    num_molecules: int
    model: str | None = None
    mode: Literal["bulk"] = "bulk"
    density_gcc: float = 0.2  # g/cm³ — initial placement density
    device: str = "cpu"
    dtype: str = "float32"
    cif_path: Path | None = None       # crystal-structure input (overrides xyz)
    supercell: tuple[int, int, int] | None = None  # None → auto from cutoff
    periodic: bool = True              # False → finite cluster (molecular MLIPs)


class SystemBuilder:
    """Build a simulation cell from a single-molecule XYZ or a crystal CIF.

    The calculator is force-field-agnostic.  Provide exactly one of:

    * ``calculator`` — any ASE calculator instance (MACE, AIMNet2, an OpenMM
      classical FF, a custom potential, …), or a ``factory(atoms) -> calc``
      callable for calculators that must size themselves to the built cell.
    * ``model`` — a MACE model name (built-in convenience shortcut).

    Parameters
    ----------
    xyz_path : str or Path
        Path to the optimised single-molecule XYZ geometry (random packing).
    num_molecules : int
        Number of molecules to place in the simulation cell.
    model : str or None
        MACE model identifier, e.g. ``"MACE-OFF23(Small)"``.  Mutually
        exclusive with ``calculator``.  If both are *None*, no calculator is
        attached (build geometry only).
    calculator : ase Calculator, callable, or None
        A ready calculator instance, or a ``factory(atoms) -> calculator``.
        Takes precedence over ``model``.
    density_gcc : float
        Initial packing density in g/cm³.  Use a lower value if placement
        fails; the system will equilibrate to the physical density.
    device : str
        PyTorch device for the MACE shortcut: ``"cpu"``, ``"mps"``, ``"cuda"``,
        or ``"auto"``.  Ignored when ``calculator`` is supplied.
    periodic : bool
        If *False*, build a finite (non-periodic) cluster — appropriate for
        molecular MLIPs that do not use periodic images.  The simulator then
        skips the densification / NPT stages automatically.  Crystal (CIF)
        input is always periodic.
    """

    def __init__(
        self,
        xyz_path: str | Path | None = None,
        num_molecules: int = 0,
        model: str | None = None,
        calculator=None,
        density_gcc: float = 0.5,
        device: str = "cpu",
        dtype: str = "float32",
        cif_path: str | Path | None = None,
        supercell: tuple[int, int, int] | None = None,
        periodic: bool = True,
    ):
        if xyz_path is None and cif_path is None:
            raise ValueError("Provide either xyz_path (random packing) or cif_path (crystal).")
        if model is not None and calculator is not None:
            raise ValueError("Provide either `model` or `calculator`, not both.")
        self.config = SystemConfig(
            xyz_path=Path(xyz_path) if xyz_path is not None else None,
            num_molecules=int(num_molecules),
            model=model,
            density_gcc=density_gcc,
            device=device,
            dtype=dtype,
            cif_path=Path(cif_path) if cif_path is not None else None,
            supercell=tuple(supercell) if supercell is not None else None,
            periodic=bool(periodic),
        )
        self._calculator = calculator
        self.is_crystal = cif_path is not None
        self.num_molecules_actual: int | None = None  # set by build()
        self._atoms = None  # cached result

    def _attach_calculator(self, atoms) -> None:
        """Attach the calculator: injected instance/factory, or MACE ``model``.

        ``calculator`` (if given) takes precedence and may be either a ready
        ASE calculator instance or a ``factory(atoms) -> calculator`` callable
        (use a factory for calculators that must size themselves to the cell,
        e.g. a periodic classical force field).
        """
        from ase.calculators.calculator import Calculator as _ASECalc

        if self._calculator is not None:
            if isinstance(self._calculator, _ASECalc):
                calc = self._calculator
            elif callable(self._calculator):
                calc = self._calculator(atoms)
            else:
                calc = self._calculator  # duck-typed calculator instance
            atoms.calc = calc
            print(f"[builder] Attached calculator: {type(calc).__name__}")
        elif self.config.model is not None:
            calc = _make_calculator(self.config.model, self.config.device,
                                    self.config.dtype)
            atoms.calc = calc
            actual_device = getattr(calc, "device", self.config.device)
            print(f"[builder] Attached calculator: {self.config.model} on {actual_device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _min_molecules_for_density(mol_mass_amu, L_min, density_gcc):
        """Smallest molecule count whose cubic box at `density_gcc` has L ≥ L_min.

        For a cubic PBC box, L = (n·M / (N_A·ρ·1e-24))^(1/3).  Requiring
        L ≥ L_min (so an extended molecule never overlaps its own periodic
        image) gives n ≥ N_A·ρ·1e-24·L_min³ / M.
        """
        N_A = 6.02214076e23
        return int(np.ceil(N_A * density_gcc * 1e-24 * L_min ** 3 / mol_mass_amu))

    def build(self):
        """Build and return the :class:`ase.Atoms` simulation cell.

        Molecules are placed at *low* density in a generous cubic box (so RSA
        succeeds and no molecule overlaps its own periodic image).  The
        simulator then equilibrates under NPT, letting the box contract to the
        physical density predicted by the potential.  ``density_gcc`` is only an
        initial-placement target / sanity reference here, not the final density.

        The cell is built once and cached.
        """
        if self._atoms is not None:
            return self._atoms

        import ase
        import ase.io

        cfg = self.config

        if cfg.cif_path is not None:
            atoms = self._build_from_cif()
            self._attach_calculator(atoms)
            self._atoms = atoms
            return atoms

        mol = ase.io.read(str(cfg.xyz_path))
        n_mol = cfg.num_molecules
        self.num_molecules_actual = n_mol

        mol_mass_amu = sum(mol.get_masses())          # g/mol
        N_A = 6.02214076e23

        # ---- Periodic-image safety: box must exceed the molecular span ----
        mol_pos = mol.get_positions()
        L_mol = float(np.max(mol_pos.max(axis=0) - mol_pos.min(axis=0)))
        L_min = L_mol * 1.2

        # If NPT contracts the box to ~density_gcc, will L still exceed L_min?
        n_min = self._min_molecules_for_density(mol_mass_amu, L_min, cfg.density_gcc)
        print(
            f"[builder] {n_mol} molecules × {mol.get_chemical_formula()}, "
            f"M={mol_mass_amu:.1f} u, molecular span ≈ {L_mol:.1f} Å"
        )
        if n_mol < n_min:
            warnings.warn(
                f"n_mol={n_mol} is too small to reach ρ≈{cfg.density_gcc} g/cm³ in a "
                f"cubic box without the molecule overlapping its periodic image "
                f"(needs L ≥ {L_min:.1f} Å → at least {n_min} molecules).  "
                f"The box cannot contract below L={L_min:.1f} Å, so the achievable "
                f"density will be capped below the target.  "
                f"Recommend --n_mol {n_min} or more.",
                stacklevel=2,
            )

        # ---- Initial (build) box: low density so RSA places cleanly ----
        # NPT will contract this toward the physical density during MD.
        build_density = min(0.18, cfg.density_gcc * 0.5)
        vol_build = n_mol * mol_mass_amu / (N_A * build_density * 1e-24)
        L = max(vol_build ** (1.0 / 3.0), L_min)

        print(
            f"[builder] Initial placement at ρ={build_density:.3f} g/cm³ (L={L:.2f} Å); "
            f"NPT will contract toward the physical density during MD."
        )

        # ---- Random placement with overlap detection ----
        np.random.seed(42)
        atoms = self._place_random(mol, n_mol, L)

        # No pre-relaxation: RSA at low density guarantees a minimum
        # inter-molecular atom separation (min_dist), so molecules are rigid and
        # MACE-safe.  A point-wise numpy relaxation would tear rigid molecules
        # apart (inter-molecular forces move atoms independently).

        # ---- Periodicity (False → finite cluster for molecular MLIPs) ----
        atoms.pbc = cfg.periodic

        # ---- Attach calculator (any ASE calculator, or the MACE shortcut) ----
        self._attach_calculator(atoms)

        self._atoms = atoms
        return atoms

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_from_cif(self):
        """Build a crystal supercell from a CIF file (experimental structure).

        The CIF is expanded to the full unit cell by ASE (space-group symmetry).
        Atoms arrive site-major (all images of site 1, then site 2, …), so we

        1. detect bonds with PBC (covalent radii × 1.25, matching the dipole
           module's bond rule),
        2. group atoms into molecules via connected components,
        3. unwrap each molecule across periodic boundaries (make it whole),
        4. reorder atoms so each molecule is a contiguous block with an
           identical internal site sequence (required by the charge tiling in
           DipoleTracker and the H-bond probe),
        5. repeat into a supercell (``supercell=None`` → smallest repeat with
           every lattice vector ≥ 9 Å, twice the MACE-OFF cutoff).

        Starting from the experimental crystal means the H-bond network and
        chain packing are correct from step 0 — no densification or annealing
        is needed, only thermal equilibration.
        """
        import ase
        import ase.io
        from ase.neighborlist import neighbor_list

        from .dipole import _BOND_SCALE, _COVALENT_RADII

        cfg = self.config
        unit = ase.io.read(str(cfg.cif_path))
        syms = unit.get_chemical_symbols()

        # ---- 1. PBC-aware bonds: d < 1.25·(r_i + r_j) ----
        cutoffs = [_BOND_SCALE * _COVALENT_RADII.get(s, 1.0) for s in syms]
        ii, jj, shifts = neighbor_list("ijS", unit, cutoffs)

        # ---- 2./3. Connected components + unwrap (BFS carrying cell shifts) ----
        nbrs: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(len(unit))]
        for a, b, S in zip(ii, jj, shifts):
            nbrs[a].append((b, S))

        pos = unit.get_positions()
        cell = unit.get_cell()[:]
        unwrapped = pos.copy()
        mol_id = np.full(len(unit), -1, dtype=int)
        n_found = 0
        for seed in range(len(unit)):
            if mol_id[seed] >= 0:
                continue
            mol_id[seed] = n_found
            stack = [seed]
            while stack:
                a = stack.pop()
                for b, S in nbrs[a]:
                    if mol_id[b] >= 0:
                        continue
                    mol_id[b] = n_found
                    # neighbour image position relative to a's unwrapped position
                    unwrapped[b] = unwrapped[a] + (pos[b] + S @ cell - pos[a])
                    stack.append(b)
            n_found += 1

        # ---- 4. Reorder: molecule-contiguous, ascending original index ----
        order = np.lexsort((np.arange(len(unit)), mol_id))
        new_syms = [syms[k] for k in order]
        new_pos = unwrapped[order]
        n_per_mol = len(unit) // n_found
        sizes = np.bincount(mol_id)
        if not np.all(sizes == n_per_mol):
            raise ValueError(
                f"CIF molecules have unequal sizes {sorted(set(sizes))}: bond "
                f"detection failed or the structure is not a single-component "
                f"molecular crystal."
            )
        ref_seq = new_syms[:n_per_mol]
        for m in range(1, n_found):
            if new_syms[m * n_per_mol:(m + 1) * n_per_mol] != ref_seq:
                raise ValueError(
                    f"Molecule {m} has a different atom-site sequence; cannot "
                    f"tile per-molecule charges.  Check the CIF for disorder."
                )

        atoms = ase.Atoms(symbols=new_syms, positions=new_pos,
                          cell=unit.get_cell(), pbc=True)

        # ---- 5. Supercell ----
        if cfg.supercell is not None:
            reps = cfg.supercell
        else:
            min_len = 9.0  # Å — ≥ 2 × MACE-OFF cutoff (4.5 Å) per lattice vector
            reps = tuple(max(1, int(np.ceil(min_len / l)))
                         for l in atoms.cell.lengths())
        atoms = atoms.repeat(reps)  # cell-copy-major → molecules stay contiguous

        n_mol_total = n_found * int(np.prod(reps))
        self.num_molecules_actual = n_mol_total

        mass_g = float(sum(atoms.get_masses())) / 6.02214076e23
        rho = mass_g / (atoms.get_volume() * 1e-24)
        a_l, b_l, c_l = atoms.cell.lengths()
        print(
            f"[builder] Crystal from {cfg.cif_path.name}: {n_found} molecules/cell "
            f"× {n_per_mol} atoms, supercell {reps[0]}×{reps[1]}×{reps[2]} → "
            f"{n_mol_total} molecules, {len(atoms)} atoms\n"
            f"[builder] Cell {a_l:.2f}×{b_l:.2f}×{c_l:.2f} Å, "
            f"ρ = {rho:.4f} g/cm³ (experimental crystal density)"
        )
        return atoms

    @staticmethod
    def _place_random(mol, n_mol: int, L: float, min_dist: float = 2.5,
                      max_tries: int = 2000):
        """Random sequential addition of n_mol molecules into a cubic box of size L.

        Caller is expected to pass the (low-density) BUILD box size; subsequent
        compression to the target density is done by the simulator during MD
        (Stage B in IRSpectrumSimulator.run).

        If RSA cannot fit a molecule after max_tries, a best-effort fallback
        searches 500 candidate positions and chooses the one with the maximum
        minimum atom-atom distance, avoiding near-zero separations that would
        produce NaN energies in MACE.
        """
        from scipy.spatial.transform import Rotation
        import ase

        mol_pos0 = mol.get_positions() - mol.get_center_of_mass()

        # ---- Choose placement box (conservative 20% packing fraction) ----
        N_A = 6.02214076e23
        mol_mass_amu = sum(mol.get_masses())
        V_mol = mol_mass_amu / (N_A * 1.0 * 1e-24)   # Å³, ~1 g/cm³ as proxy
        L_rsa = (n_mol * V_mol / 0.20) ** (1.0 / 3.0)
        L_place = max(L, L_rsa)

        if L_place > L * 1.01:
            print(
                f"[builder] RSA placement at L={L_place:.1f} Å "
                f"(target density needs L={L:.1f} Å); "
                f"will compress after placement."
            )

        placed_positions: list[np.ndarray] = []
        placed_syms: list[str] = []

        for i_mol in range(n_mol):
            for attempt in range(max_tries):
                center = np.random.uniform(0.0, L_place, 3)
                R = Rotation.random().as_matrix()
                new_pos = mol_pos0 @ R.T + center

                if placed_positions:
                    existing = np.vstack(placed_positions)
                    delta = new_pos[:, None, :] - existing[None, :, :]
                    delta -= L_place * np.round(delta / L_place)
                    dists = np.sqrt((delta ** 2).sum(axis=-1))
                    if dists.min() < min_dist:
                        continue

                placed_positions.append(new_pos)
                placed_syms.extend(mol.get_chemical_symbols())
                break
            else:
                # Best-effort fallback: find the position with max min-distance
                # (much safer than purely random, avoids near-zero separations)
                best_pos, best_d = None, -1.0
                existing = np.vstack(placed_positions) if placed_positions else None
                for _ in range(500):
                    c = np.random.uniform(0.0, L_place, 3)
                    R = Rotation.random().as_matrix()
                    pos = mol_pos0 @ R.T + c
                    if existing is not None:
                        delta = pos[:, None, :] - existing[None, :, :]
                        delta -= L_place * np.round(delta / L_place)
                        d = np.sqrt((delta ** 2).sum(-1)).min()
                    else:
                        d = np.inf
                    if d > best_d:
                        best_d, best_pos = d, pos
                warnings.warn(
                    f"Could not place molecule {i_mol + 1}/{n_mol} after "
                    f"{max_tries} tries (best gap = {best_d:.2f} Å).",
                    stacklevel=3,
                )
                placed_positions.append(best_pos)
                placed_syms.extend(mol.get_chemical_symbols())

        all_pos = np.vstack(placed_positions)
        atoms = ase.Atoms(
            symbols=placed_syms,
            positions=all_pos,
            cell=[L_place, L_place, L_place],
            pbc=True,
        )

        # ---- Compress uniformly to target box ----
        if L_place > L:
            atoms.set_cell([L, L, L], scale_atoms=True)

        return atoms


# ------------------------------------------------------------------
# Calculator factory
# ------------------------------------------------------------------

_MPS_SHIM_INSTALLED = False


def _enable_mps_float32_shim() -> None:
    """Redirect ``Tensor.double()`` to float32 for MPS tensors.

    MACE's model forward unconditionally calls ``.double()`` on the energy
    accumulation (mace/modules/models.py), but Apple's MPS backend cannot
    create float64 tensors at all.  Since the MD pipeline already runs in
    float32, redirecting ``.double()`` to float32 *for MPS tensors only* lets
    the forward pass complete; verified to reproduce CPU energies/forces to
    float32 precision (ΔE = 0, Δ|F| < 1e-5 eV/Å).  CPU/CUDA tensors are
    untouched.
    """
    global _MPS_SHIM_INSTALLED
    if _MPS_SHIM_INSTALLED:
        return
    import torch

    _orig_double = torch.Tensor.double

    def _double_shim(self):
        if self.device.type == "mps":
            return self.to(torch.float32)
        return _orig_double(self)

    torch.Tensor.double = _double_shim
    _MPS_SHIM_INSTALLED = True


def _resolve_device(device: str, dtype: str) -> tuple[str, str]:
    """Resolve a requested device to an available one, fixing dtype as needed.

    Accepts ``"cpu"``, ``"cuda"``, ``"mps"`` (Apple-Silicon GPU), or ``"auto"``
    (prefer mps → cuda → cpu).  PyTorch's MPS backend does not support float64,
    so float32 is forced there, ``PYTORCH_ENABLE_MPS_FALLBACK`` is set, and a
    shim redirects MACE's internal ``.double()`` calls to float32 on MPS.
    """
    import torch

    d = device.lower()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()

    if d == "auto":
        d = "mps" if mps_ok else ("cuda" if cuda_ok else "cpu")

    if d in ("mps", "gpu", "apple"):
        if mps_ok:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            _enable_mps_float32_shim()
            if dtype == "float64":
                warnings.warn("MPS does not support float64; using float32.", stacklevel=2)
            return "mps", "float32"
        warnings.warn("MPS not available; falling back to CPU.", stacklevel=2)
        return "cpu", dtype

    if d == "cuda":
        if cuda_ok:
            return "cuda", dtype
        warnings.warn("CUDA not available; falling back to CPU.", stacklevel=2)
        return "cpu", dtype

    return "cpu", dtype


def _relocate_calc_to_device(calc, device: str):
    """Move an already-loaded MACE calculator's models onto ``device``.

    Used for MPS: the model file is stored in float64 and
    ``torch.load(map_location="mps")`` fails (MPS has no float64).  We instead
    load on CPU as float32, then move the float32 model here.
    """
    import torch

    dev = torch.device(device)
    for m in getattr(calc, "models", []):
        m.to(dev)
    calc.device = dev
    return calc


def _make_calculator(model_name: str, device: str = "cpu", dtype: str = "float32"):
    """Instantiate a MACE ASE calculator from a model name string."""
    name = model_name.strip()
    device, dtype = _resolve_device(device, dtype)

    # MACE-OFF weights are float64; torch.load(map_location="mps") fails because
    # MPS lacks float64.  Load on CPU (converted to float32), then relocate.
    load_device = "cpu" if device == "mps" else device

    if "mace-off" in name.lower() or "mace_off" in name.lower():
        from mace.calculators import mace_off
        # Normalise: "MACE-OFF23(Small)" → "small", bare "small"/"medium"/"large" pass through
        _off_map = {
            "mace-off23(small)": "small",
            "mace-off23(medium)": "medium",
            "mace-off23(large)": "large",
        }
        key = _off_map.get(name.lower(), name.lower())
        calc = mace_off(model=key, device=load_device, default_dtype=dtype)
    elif "mace_mp" in name.lower() or "mace-mp" in name.lower():
        from mace.calculators import mace_mp
        calc = mace_mp(model=name, device=load_device, default_dtype=dtype)
    else:
        raise ValueError(
            f"Unrecognised model name: '{model_name}'.  "
            "Expected 'MACE-OFF23(Small/Medium/Large)', 'small', or 'mace_mp'."
        )

    if device != load_device:
        _relocate_calc_to_device(calc, device)
    return calc
