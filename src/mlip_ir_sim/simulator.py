"""MD-based IR spectrum simulator (densify-then-NPT, physically-grounded).

Pipeline
--------
A. Cold equilibration  (T=10 K, force-capped)             — absorb initial forces
B. Warm-up ramp        (T 10→target K, force-capped)      — gentle heating
C. Densification       (gradual box contraction → ρ_target, force-capped)
                                                          — bring the dilute
                                                            placement into the
                                                            liquid regime
D. NPT refinement      (target T & P=1 bar, true forces)  — relax to the density
                                                            the potential predicts
E. NVT equilibration   (target T, fixed final density)    — thermalise
F. NVE production       (true forces, dipole ACF → spectrum)

Why both C and D?  A 1-bar barostat cannot condense a dilute gas (the driving
force P_target − P ≈ 1 bar is negligible on the atomic scale), so we first
contract the box to the requested density, then let NPT fine-tune to the
potential's true equilibrium density — where the barostat works because a
near-incompressible liquid produces a strong pressure response.

Stages A–C keep |F_i| ≤ force_cap via :class:`ForceCappedCalculator` to prevent
integrator divergence; the cap is removed for D–F so the barostat sees the true
stress and production uses the true forces.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np


# ── Checkpoint helpers (module-level so they survive import) ─────────────────

_STAGE_ORDER = ['A', 'B', 'C', 'D', 'G', 'H', 'E', 'F']


def _ckpt_save(path: str, stage: str, atoms, step: int = 0,
               dipoles=None, tracker=None, charge_method: str = 'gaff') -> None:
    """Atomically write a checkpoint file.

    Uses a write-then-rename pattern so an interrupted write never leaves a
    corrupted checkpoint (rename is atomic on POSIX filesystems).
    """
    data: dict = {
        'stage': stage,
        'step': step,
        'positions': atoms.get_positions().copy(),
        'velocities': atoms.get_velocities().copy(),
        'cell': np.array(atoms.get_cell()),
        'dipoles': dipoles[:step].copy() if dipoles is not None else None,
        'charge_method': charge_method,
    }
    if tracker is not None:
        data['tracker_unwrapped'] = tracker._unwrapped.copy()
        data['tracker_prev_scaled'] = tracker._prev_scaled.copy()
        data['charges'] = tracker.charges.copy()
    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    os.replace(tmp, path)


def _ckpt_load(path: str) -> dict | None:
    """Load a checkpoint; return None if absent or corrupted."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    except Exception as exc:
        print(f"[simulator] Checkpoint corrupted ({exc}), starting fresh.")
        return None


class IRSpectrumSimulator:
    """Orchestrate cold-eq → warm-up → densify → NPT → NVT → NVE production.

    Parameters
    ----------
    system : SystemBuilder
        A configured :class:`~.builder.SystemBuilder` instance, built at low
        density.  This simulator contracts the box to ``density_gcc`` and then
        refines to the physical density via NPT.
    """

    def __init__(self, system):
        self.system = system

    def run(
        self,
        *,
        temperature: float,
        pressure: float = 1.0,             # bar — NPT target pressure (Stage D)
        eq_time_ps: float,
        prod_time_ps: float,
        timestep_fs: float = 0.5,
        langevin_friction_per_fs: float = 0.01,
        logfile: str = "-",
        loginterval: int = 50,
        n_mol: int | None = None,
        fwhm_cm1: float = 10.0,
        # ── multi-stage parameters ──
        cold_eq_ps: float = 0.05,
        warmup_ps: float = 0.2,
        compress_ps: float = 2.0,
        npt_ps: float = 2.0,
        force_cap: float = 50.0,
        anneal_T: float = 0.0,
        anneal_ps: float = 0.0,
        cool_ps: float = 0.0,
        traj_path: str | None = None,
        traj_interval: int = 100,
        checkpoint_path: str | None = None,
        checkpoint_interval: int = 1000,
        charge_method: str = "xtb",
    ) -> "IRSpectrum":  # noqa: F821
        """Run the densify-then-NPT MD pipeline and return the IR spectrum.

        Parameters
        ----------
        temperature, pressure : float
            Target temperature (K) and NPT pressure (bar).
        eq_time_ps : float
            Stage E duration (NVT equilibration at the final density).
        prod_time_ps : float
            Stage F duration (NVE production for the dipole ACF).
        cold_eq_ps, warmup_ps : float
            Stage A / B durations.
        compress_ps : float
            Stage C duration (gradual contraction to ρ_target).
        npt_ps : float
            Stage D duration (NPT refinement toward the physical density).
        force_cap : float
            Per-atom force cap in eV/Å during Stages A–C.  Removed for D–F.
        anneal_T : float
            Melt/anneal temperature (K).  If greater than ``temperature``, the
            warm-up/densify/NPT stages run at this elevated temperature so
            molecules can diffuse and explore configurational space.  Stage G
            then holds at ``anneal_T`` and Stage H slow-cools to ``temperature``
            to reach a well-equilibrated low-temperature structure.
            Annealing is disabled when ``anneal_T ≤ temperature`` or ``anneal_ps``
            is 0 (the pipeline then behaves exactly as before).
        anneal_ps : float
            Stage G duration: NPT hold at ``anneal_T``.
        cool_ps : float
            Stage H duration: NPT slow-cool ``anneal_T`` → ``temperature``.
        traj_path : str or None
            If given, write an Extended-XYZ trajectory (OVITO-readable, with
            cell/Lattice info so the contraction and NPT are visible) across all
            stages.  A frame is saved every ``traj_interval`` MD steps.
        traj_interval : int
            Steps between saved frames (per stage).  Default 100.
        checkpoint_path : str or None
            Path to the checkpoint file.  On each run, if this file exists the
            simulation resumes from it, skipping completed stages and
            continuing Stage F from the saved step.  After each stage and every
            ``checkpoint_interval`` steps during Stage F, the file is
            atomically overwritten.  Pass *None* (default) to disable.
        checkpoint_interval : int
            Steps between Stage F checkpoint saves.  Default 1000.
        charge_method : str
            Partial charges for the production dipole.  ``"xtb"`` (default):
            one GFN2-xTB single point per molecule at the equilibrated
            geometry (after Stage E) — quantum-derived, conformer-specific.
            ``"gaff"``: rule-based functional-group charges (no extra
            dependency).  Falls back to gaff with a warning if tblite is not
            installed.
        """
        import ase.units as u
        from ase.io import write as ase_write
        from ase.md.langevin import Langevin
        from ase.md.nptberendsen import NPTBerendsen
        from ase.md.verlet import VelocityVerlet
        from ase.md.velocitydistribution import (
            MaxwellBoltzmannDistribution,
            Stationary,
            ZeroRotation,
        )

        from .charges import wrap_molecules
        from .dipole import DipoleTracker
        from .safety import ForceCappedCalculator
        from .spectrum import dipole_acf, ir_spectrum_from_acf

        atoms = self.system.build()
        true_calc = atoms.calc  # uncapped calculator (MACE, TBLite, OpenMM, …)

        # ── Checkpoint: load prior state if available ─────────────────────
        ckpt = _ckpt_load(checkpoint_path) if checkpoint_path else None
        if ckpt is not None and len(ckpt['positions']) != len(atoms):
            print(f"[simulator] Checkpoint atom count ({len(ckpt['positions'])}) "
                  f"≠ current system ({len(atoms)}) — starting fresh.")
            ckpt = None
        if ckpt is not None:
            atoms.set_positions(ckpt['positions'])
            atoms.set_velocities(ckpt['velocities'])
            atoms.set_cell(ckpt['cell'], scale_atoms=False)
            print(f"[simulator] Checkpoint loaded: stage='{ckpt['stage']}', "
                  f"step={ckpt.get('step', 0)}")

        def _done(stage: str) -> bool:
            """True if this stage is complete and should be skipped."""
            if ckpt is None or stage == 'F':
                return False
            try:
                return _STAGE_ORDER.index(stage) <= _STAGE_ORDER.index(ckpt['stage'])
            except ValueError:
                return False

        def _save(stage: str) -> None:
            if checkpoint_path:
                _ckpt_save(checkpoint_path, stage, atoms)

        # ── Trajectory writer (Extended XYZ, OVITO-readable) ──────────────
        if traj_path is not None:
            traj_path = str(traj_path)
            os.makedirs(os.path.dirname(traj_path) or ".", exist_ok=True)
            if ckpt is None and os.path.exists(traj_path):
                os.remove(traj_path)  # start fresh (keep frames when resuming)

        # Stateful centroid tracker: each frame is placed at the image closest
        # to the previous frame → no sudden full-cell jumps in the trajectory.
        _prev_traj_centroids: list[np.ndarray | None] = [None]

        def _snapshot():
            # Copy drops the calculator → writes geometry + cell only (no
            # recomputation), keeping the write cheap and OVITO-friendly.
            # Wrap *per molecule* with inter-frame continuity so no molecule
            # appears to teleport when its centroid crosses a cell boundary.
            # Falls back to plain per-atom wrap if molecule unwrap fails
            # (e.g. during a large-step NPT expansion with classical FF).
            if _wrap_ok:
                try:
                    snap, new_cent = wrap_molecules(
                        atoms, n_mol, prev_centroids=_prev_traj_centroids[0])
                    _prev_traj_centroids[0] = new_cent
                except RuntimeError:
                    snap = atoms.copy()
                    snap.wrap()
            else:
                snap = atoms.copy()
                snap.wrap()
            ase_write(traj_path, snap, append=True, format="extxyz")

        def _attach_writer(dyn):
            if traj_path is not None:
                dyn.attach(_snapshot, interval=traj_interval)

        if n_mol is None:
            n_mol = getattr(self.system, "num_molecules_actual", None)
        if n_mol is None:
            n_mol = getattr(getattr(self.system, "config", None), "num_molecules", None)
        # Per-molecule trajectory wrapping needs equal-size contiguous blocks.
        _wrap_ok = (n_mol is not None and n_mol > 0 and len(atoms) % n_mol == 0)
        is_crystal = bool(getattr(self.system, "is_crystal", False))
        mass_g = float(sum(atoms.get_masses())) / 6.02214076e23  # total mass in g

        def _density_gcc():
            return mass_g / (atoms.get_volume() * 1e-24)

        # Target box length from the requested density (Stage-C contraction goal)
        density_target = float(getattr(self.system.config, "density_gcc", 0.85))
        L_target = (mass_g / (density_target * 1e-24)) ** (1.0 / 3.0)

        # Annealing: run at elevated temperature so molecules can diffuse and
        # sample a wider configurational space, then slow-cool to the production
        # temperature.  When disabled, T_melt
        # collapses to `temperature` and Stages G/H are skipped (old behaviour).
        annealing = anneal_T > temperature and anneal_ps > 0.0
        T_melt = anneal_T if annealing else temperature

        dt = timestep_fs * u.fs
        n_cold = max(1, int(round(cold_eq_ps * 1e3 / timestep_fs)))
        n_warm = max(1, int(round(warmup_ps * 1e3 / timestep_fs)))
        n_comp = max(1, int(round(compress_ps * 1e3 / timestep_fs)))
        n_npt  = max(0, int(round(npt_ps * 1e3 / timestep_fs)))
        n_anneal = max(1, int(round(anneal_ps * 1e3 / timestep_fs))) if annealing else 0
        n_cool   = max(1, int(round(cool_ps * 1e3 / timestep_fs))) if annealing else 0
        n_eq   = max(1, int(round(eq_time_ps * 1e3 / timestep_fs)))
        n_prod = max(1, int(round(prod_time_ps * 1e3 / timestep_fs)))

        L0 = float(atoms.cell[0, 0])
        if is_crystal and annealing:
            print("[simulator] WARNING: annealing above the melting point will "
                  "destroy the crystal order you started from (CIF input).  "
                  "Set anneal_ps=0 unless melting is intended.")
        densify_line = (
            "  Stage C  Densify   : skipped (crystal start at experimental density)\n"
            if is_crystal else
            f"  Stage C  Densify   : {n_comp} steps ({compress_ps} ps)  "
            f"L {L0:.1f}→{L_target:.1f} Å (ρ {_density_gcc():.3f}→{density_target:.3f})\n"
        )
        anneal_lines = (
            f"  Stage G  Anneal    : {n_anneal} steps ({anneal_ps} ps)  NPT hold @ {T_melt} K\n"
            f"  Stage H  Cool      : {n_cool} steps ({cool_ps} ps)  NPT {T_melt}→{temperature} K\n"
            if annealing else
            "  Stage G/H Anneal   : (disabled — anneal_T ≤ T or anneal_ps=0)\n"
        )
        print(
            f"\n[simulator] {'Crystal-start' if is_crystal else 'Densify-then-NPT'} MD  "
            f"(dt={timestep_fs} fs, "
            f"target T={temperature} K, melt T={T_melt} K, P={pressure} bar)\n"
            f"  Stage A  Cold eq   : {n_cold} steps ({cold_eq_ps} ps)  @ 10 K\n"
            f"  Stage B  Warm-up   : {n_warm} steps ({warmup_ps} ps)  T 10→{T_melt} K\n"
            + densify_line +
            f"  Stage D  NPT       : {'skipped (--npt 0)' if n_npt == 0 else f'{n_npt} steps ({npt_ps} ps)  refine to physical ρ @ {T_melt} K'}\n"
            + anneal_lines +
            f"  Stage E  NVT eq    : {n_eq} steps ({eq_time_ps} ps)  @ {temperature} K\n"
            f"  Stage F  Production: {n_prod} steps ({prod_time_ps} ps)  NVE\n"
            f"  Total atoms        : {len(atoms)}\n"
            f"  Force cap          : {force_cap} eV/Å  (Stages A–C)\n"
        )

        # ── Initialise velocities at 10 K (cold start, only for fresh runs) ──
        if ckpt is None:
            MaxwellBoltzmannDistribution(atoms, temperature_K=10.0,
                                         rng=np.random.default_rng(0))
            Stationary(atoms)
            ZeroRotation(atoms)

        friction = langevin_friction_per_fs / u.fs

        # ── Wrap MACE in a force cap for the protective stages A–C ─────────
        if true_calc is not None:
            atoms.calc = ForceCappedCalculator(true_calc, F_max=force_cap)

        # ──────────────────────── Stage A: Cold eq ───────────────────────
        if _done('A'):
            print("[simulator] Stage A  skipped (checkpoint)")
        else:
            t0 = time.time()
            print(f"[simulator] Stage A  Cold equilibration at 10 K ({n_cold} steps)…")
            dyn = Langevin(atoms, timestep=dt, temperature_K=10.0,
                           friction=0.1 / u.fs, fixcm=False,
                           logfile=logfile, loginterval=loginterval)
            _attach_writer(dyn)
            dyn.run(n_cold)
            print(f"[simulator] Stage A done in {time.time() - t0:.1f} s")
            _save('A')

        # ──────────────────────── Stage B: Warm-up ramp ──────────────────
        if _done('B'):
            print("[simulator] Stage B  skipped (checkpoint)")
        else:
            t0 = time.time()
            print(f"[simulator] Stage B  Warm-up 10→{T_melt} K ({n_warm} steps)…")
            dyn = Langevin(atoms, timestep=dt, temperature_K=10.0,
                           friction=0.05 / u.fs, fixcm=False,
                           logfile=logfile, loginterval=loginterval)
            _attach_writer(dyn)
            ramp_every = max(1, n_warm // 100)
            for step in range(n_warm):
                if step % ramp_every == 0:
                    T_now = 10.0 + (T_melt - 10.0) * (step + 1) / n_warm
                    dyn.set_temperature(temperature_K=T_now)
                dyn.run(1)
            print(f"[simulator] Stage B done in {time.time() - t0:.1f} s")
            _save('B')

        # ──────────────────────── Stage C: Densification ─────────────────
        # Gradual geometric box contraction L0 → L_target (only if contracting).
        # Crystal start: the cell is already at the experimental density (and is
        # generally non-cubic), so geometric contraction is skipped entirely.
        if _done('C'):
            print("[simulator] Stage C  skipped (checkpoint)")
        else:
            t0 = time.time()
            if is_crystal:
                print(f"[simulator] Stage C  skipped (crystal at experimental "
                      f"ρ={_density_gcc():.3f} g/cm³).")
            elif L_target < L0:
                print(f"[simulator] Stage C  Densify L {L0:.2f}→{L_target:.2f} Å "
                      f"({n_comp} steps)…")
                per_step_scale = (L_target / L0) ** (1.0 / n_comp)
                dyn = Langevin(atoms, timestep=dt, temperature_K=T_melt,
                               friction=friction, fixcm=False,
                               logfile=logfile, loginterval=loginterval)
                _attach_writer(dyn)
                comp_report = max(1, n_comp // 10)
                for step in range(n_comp):
                    atoms.set_cell(atoms.get_cell() * per_step_scale, scale_atoms=True)
                    dyn.run(1)
                    if (step + 1) % comp_report == 0:
                        print(f"  contract step {step+1}/{n_comp}  "
                              f"L={atoms.cell[0,0]:.2f} Å  ρ={_density_gcc():.3f} g/cm³")
                atoms.set_cell([L_target, L_target, L_target], scale_atoms=True)
                print(f"[simulator] Stage C done in {time.time() - t0:.1f} s  "
                      f"→ ρ={_density_gcc():.3f} g/cm³, L={atoms.cell[0,0]:.2f} Å")
            else:
                print(f"[simulator] Stage C  skipped (placement L={L0:.1f} Å already "
                      f"≤ target L={L_target:.1f} Å).")
            _save('C')

        # ──── Remove force cap before NPT (barostat needs true stress) ────
        if true_calc is not None:
            atoms.calc = true_calc

        # ──────────────────────── Stage D: NPT refinement ────────────────
        # Crystal: anisotropic barostat (each lattice vector relaxes on its own
        # stress component — an orthorhombic cell must not be forced to scale
        # isotropically).  Random packing: isotropic, as before.
        from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
        npt_cls = Inhomogeneous_NPTBerendsen if is_crystal else NPTBerendsen
        if n_npt == 0:
            print("[simulator] Stage D  skipped (npt_ps=0)")
            _save('D')
        elif _done('D'):
            print("[simulator] Stage D  skipped (checkpoint)")
        else:
            t0 = time.time()
            print(f"[simulator] Stage D  NPT at {T_melt} K, {pressure} bar "
                  f"({n_npt} steps, {'anisotropic' if is_crystal else 'isotropic'})…  "
                  f"ρ_start={_density_gcc():.3f} g/cm³")
            dyn = npt_cls(
                atoms, timestep=dt,
                temperature_K=T_melt,
                pressure_au=pressure * u.bar,
                compressibility_au=4.57e-5 / u.bar,   # ~organic liquid
                taut=100 * u.fs, taup=1000 * u.fs,
                fixcm=False, logfile=logfile, loginterval=loginterval,
            )
            _attach_writer(dyn)
            npt_report = max(1, n_npt // 10)
            for step in range(n_npt):
                dyn.run(1)
                if (step + 1) % npt_report == 0:
                    print(f"  NPT step {step+1}/{n_npt}  "
                          f"L={atoms.cell[0,0]:.2f} Å  ρ={_density_gcc():.3f} g/cm³")
            print(f"[simulator] Stage D done in {time.time() - t0:.1f} s  "
                  f"→ ρ={_density_gcc():.3f} g/cm³, L={atoms.cell[0,0]:.2f} Å")
            _save('D')

        # ──────────────────── Stage G: Anneal hold (melt) ────────────────
        # NPT hold above the target temperature so the system can diffuse and
        # sample a wider configurational space before slow-cooling to T.
        if annealing:
            if _done('G'):
                print("[simulator] Stage G  skipped (checkpoint)")
            else:
                t0 = time.time()
                print(f"[simulator] Stage G  Anneal hold at {T_melt} K, NPT "
                      f"({n_anneal} steps)…")
                dyn = NPTBerendsen(
                    atoms, timestep=dt,
                    temperature_K=T_melt,
                    pressure_au=pressure * u.bar,
                    compressibility_au=4.57e-5 / u.bar,
                    taut=100 * u.fs, taup=1000 * u.fs,
                    fixcm=False, logfile=logfile, loginterval=loginterval,
                )
                _attach_writer(dyn)
                anneal_report = max(1, n_anneal // 10)
                for step in range(n_anneal):
                    dyn.run(1)
                    if (step + 1) % anneal_report == 0:
                        print(f"  anneal step {step+1}/{n_anneal}  "
                              f"ρ={_density_gcc():.3f} g/cm³")
                print(f"[simulator] Stage G done in {time.time() - t0:.1f} s  "
                      f"→ ρ={_density_gcc():.3f} g/cm³")
                _save('G')

            # ─────────────────── Stage H: Slow-cool to T ─────────────────
            if _done('H'):
                print("[simulator] Stage H  skipped (checkpoint)")
            else:
                t0 = time.time()
                print(f"[simulator] Stage H  Slow-cool {T_melt}→{temperature} K, NPT "
                      f"({n_cool} steps)…")
                dyn = NPTBerendsen(
                    atoms, timestep=dt,
                    temperature_K=T_melt,
                    pressure_au=pressure * u.bar,
                    compressibility_au=4.57e-5 / u.bar,
                    taut=100 * u.fs, taup=1000 * u.fs,
                    fixcm=False, logfile=logfile, loginterval=loginterval,
                )
                _attach_writer(dyn)
                cool_every = max(1, n_cool // 100)
                cool_report = max(1, n_cool // 10)
                for step in range(n_cool):
                    if step % cool_every == 0:
                        T_now = T_melt + (temperature - T_melt) * (step + 1) / n_cool
                        dyn.set_temperature(temperature_K=T_now)
                    dyn.run(1)
                    if (step + 1) % cool_report == 0:
                        print(f"  cool step {step+1}/{n_cool}  "
                              f"ρ={_density_gcc():.3f} g/cm³")
                print(f"[simulator] Stage H done in {time.time() - t0:.1f} s  "
                      f"→ ρ={_density_gcc():.3f} g/cm³")
                _save('H')

        # ──────────────────────── Stage E: NVT eq ─────────────────────────
        if _done('E'):
            print("[simulator] Stage E  skipped (checkpoint)")
            rho_final = _density_gcc()
        else:
            t0 = time.time()
            print(f"[simulator] Stage E  NVT equilibration at {temperature} K ({n_eq} steps)…")
            dyn = Langevin(atoms, timestep=dt, temperature_K=temperature,
                           friction=friction, fixcm=False,
                           logfile=logfile, loginterval=loginterval)
            _attach_writer(dyn)
            dyn.run(n_eq)
            rho_final = _density_gcc()
            print(f"[simulator] Stage E done in {time.time() - t0:.1f} s  "
                  f"→ ρ={rho_final:.3f} g/cm³")
            _save('E')

        # ── Resuming Stage F?  (charge method must match, or the stored
        #    dipoles were computed with different charges → restart F) ──────
        resuming_f = (ckpt is not None and ckpt['stage'] == 'F')
        if resuming_f and ckpt.get('charge_method', 'gaff') != charge_method:
            print(f"[simulator] Charge method changed "
                  f"('{ckpt.get('charge_method', 'gaff')}' → '{charge_method}') — "
                  f"restarting Stage F from step 0 (equilibration is kept).")
            resuming_f = False

        # ── Partial charges for the production dipole ─────────────────────
        # xtb: one GFN2-xTB single point per molecule at the *equilibrated*
        # geometry — conformer-specific quantum charges.  Computed once here,
        # fixed during production, and carried through checkpoints so a
        # resumed run reuses the identical charges (no dipole discontinuity).
        charges = None
        if resuming_f and ckpt.get('charges') is not None:
            charges = ckpt['charges']
            print("[simulator] Charges restored from checkpoint.")
        elif charge_method == "xtb":
            # Runs in a fresh subprocess (torch + tblite cannot share one
            # OpenMP runtime).  Any failure degrades to GAFF rather than
            # killing a multi-hour run.
            try:
                from .charges import compute_xtb_charges_isolated
                t0 = time.time()
                print(f"[simulator] GFN2-xTB charges at the equilibrated geometry "
                      f"({n_mol} molecules, isolated subprocess)…")
                charges = compute_xtb_charges_isolated(atoms, n_mol)
                print(f"[simulator] xTB charges done in {time.time() - t0:.1f} s  "
                      f"q ∈ [{charges.min():+.3f}, {charges.max():+.3f}] e")
            except Exception as exc:
                print(f"[simulator] WARNING: xTB charge step failed ({exc})\n"
                      "            Falling back to rule-based GAFF charges.")
                charge_method = "gaff"
                charges = None

        # ── Prepare dipole tracker after equilibration ────────────────────
        tracker = DipoleTracker(atoms, charges=charges, n_mol=n_mol)

        # ── Stage F partial-resume: restore tracker state from checkpoint ─
        dipoles = np.empty((n_prod, 3), dtype=float)
        f_start = 0
        if resuming_f:
            f_start = ckpt.get('step', 0)
            saved_dip = ckpt.get('dipoles')
            if saved_dip is not None and f_start > 0:
                dipoles[:f_start] = saved_dip
            if ckpt.get('tracker_unwrapped') is not None:
                tracker._unwrapped = ckpt['tracker_unwrapped'].copy()
                tracker._prev_scaled = ckpt['tracker_prev_scaled'].copy()
            print(f"[simulator] Stage F  resuming from step {f_start}/{n_prod}")

        # ──────────────────────── Stage F: NVE production ────────────────
        dyn_prod = VelocityVerlet(atoms, timestep=dt,
                                  logfile=logfile, loginterval=loginterval)
        _attach_writer(dyn_prod)

        # Dynamic charges: if the calculator exposes per-atom charges (e.g.
        # AIMNet2), use them at every step instead of the fixed charges stored
        # in the tracker.  This captures ∂q/∂r (electronic polarisation
        # response) at essentially no extra cost since charges are computed
        # together with forces in a single forward pass.
        _calc_has_charges = ("charges" in getattr(
            getattr(atoms, "calc", None), "implemented_properties", []))
        if _calc_has_charges:
            print("[simulator] Dynamic charges detected (AIMNet2) — using "
                  "per-step charges for dipole tracking.")

        t0 = time.time()
        if f_start == 0:
            print(f"[simulator] Stage F  NVE production ({n_prod} steps)…")
        report_every = max(1, n_prod // 10)

        for step in range(f_start, n_prod):
            dyn_prod.run(1)
            step_charges = atoms.get_charges() if _calc_has_charges else None
            dipoles[step] = tracker.update(atoms, charges=step_charges)
            if checkpoint_path and (step + 1) % checkpoint_interval == 0:
                _ckpt_save(checkpoint_path, 'F', atoms, step=step + 1,
                           dipoles=dipoles, tracker=tracker,
                           charge_method=charge_method)
            if (step + 1) % report_every == 0:
                elapsed = time.time() - t0
                steps_done = step + 1 - f_start
                remaining = elapsed / steps_done * (n_prod - step - 1) if steps_done > 0 else 0
                print(f"  step {step+1}/{n_prod}  "
                      f"elapsed {elapsed:.0f}s  ETA {remaining:.0f}s")

        print(f"[simulator] Stage F done in {time.time() - t0:.1f} s")

        # ── Compute spectrum ──────────────────────────────────────────────
        dt_s = timestep_fs * 1e-15
        print("[simulator] Computing ACF and IR spectrum…")
        acf = dipole_acf(dipoles)
        spectrum = ir_spectrum_from_acf(acf, dt_s, temperature=temperature,
                                        fwhm_cm1=fwhm_cm1,
                                        quantum_correction="schofield")
        spectrum.metadata["final_density_gcc"] = rho_final

        freq_res = spectrum.metadata.get("freq_resolution_cm1", float("nan"))
        print(f"[simulator] Done.  Frequency resolution ≈ {freq_res:.2f} cm⁻¹  "
              f"(final ρ={rho_final:.3f} g/cm³)")

        return spectrum
