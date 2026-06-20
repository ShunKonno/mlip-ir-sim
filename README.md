# mlip-ir-sim

Python library for simulating **transmission IR spectra** of organic molecules using machine-learning interatomic potentials (MLIPs) and molecular dynamics.

The pipeline drives **any ASE-compatible force field or MLIP** with an ASE MD engine, records the total dipole-moment trajectory, Fourier-transforms its autocorrelation function into a broadened IR absorption spectrum, and returns a customisable `IRSpectrum` result object.

---

## Features

- **Force-field-agnostic** — plug in any ASE calculator; the pipeline reads its `implemented_properties` and adapts automatically
- Random-packing, crystal-start (CIF), or non-periodic cluster cell builder
- Multi-stage MD (cold eq → warm-up → densification → NPT → NVT → NVE production)
- Dipole ACF → transmission IR spectrum with Schofield / detailed-balance quantum correction
- `IRSpectrum` result object with matplotlib plotting and CSV / JSON serialisation
- Checkpoint / resume support for long HPC runs

### How the pipeline adapts to your calculator

The simulator inspects the attached calculator and chooses the right path — no configuration needed:

| Your calculator provides… | Pipeline behaviour |
|---|---|
| **forces only** (most classical FFs, forces-only MLIPs) | NVT + NVE; charges assigned for the dipole (`charge_method`) |
| **forces + stress** (MACE-OFF/-MP, periodic FFs) | adds the NPT barostat (densification + optional annealing) |
| **forces + per-atom charges** (AIMNet2-style) | dynamic per-step dipole (captures ∂q/∂r); no charge assignment |
| **non-periodic** (`periodic=False`, molecular MLIPs) | finite cluster: densification / NPT skipped |

---

## Installation

```bash
pip install mlip-ir-sim
```

Core runtime dependencies: ASE, NumPy, SciPy, matplotlib.

### Calculator back-ends (choose at least one)

| Back-end | Install |
|---|---|
| MACE-OFF23 / MACE-MP | `pip install "mlip-ir-sim[mace]"` |
| GFN2-xTB partial charges | `pip install "mlip-ir-sim[xtb]"` |

---

## Quick start

```python
from mlip_ir_sim import SystemBuilder, IRSpectrumSimulator

# 1. Build a simulation cell from a single-molecule XYZ file
system = SystemBuilder(
    xyz_path="molecule.xyz",
    num_molecules=30,
    model="MACE-OFF23(Small)",
    density_gcc=0.85,
    device="cpu",          # "cpu" | "cuda" | "mps" | "auto"
)

# 2. Run the MD pipeline
sim = IRSpectrumSimulator(system)
spectrum = sim.run(
    temperature=300.0,     # K
    eq_time_ps=5.0,        # NVT equilibration duration
    prod_time_ps=50.0,     # NVE production duration (longer → finer resolution)
    fwhm_cm1=10.0,         # Gaussian peak broadening
)

# 3. Plot — returns (fig, ax) for further customisation
fig, ax = spectrum.plot(title="Simulated IR")
ax.set_xlim(4000, 400)    # zoom to fingerprint + functional-group region
fig.savefig("ir_spectrum.png", dpi=150)

# 4. Save spectrum data
spectrum.save("ir_spectrum.csv")                    # two-column CSV
spectrum.save("ir_spectrum.json", format="json")    # JSON with metadata
```

### Plugging in any calculator

Pass `calculator=` with a ready ASE calculator instance, or a `factory(atoms) -> calculator` callable for potentials that must size themselves to the built cell (e.g. a periodic classical force field):

```python
# (a) any ASE calculator instance
from some_mlip import MyCalculator
system = SystemBuilder(xyz_path="molecule.xyz", num_molecules=30,
                       calculator=MyCalculator())

# (b) a factory that receives the built Atoms object
def make_calc(atoms):
    return ClassicalForceField(atoms, ...)   # e.g. an OpenMM-backed calculator

system = SystemBuilder(xyz_path="molecule.xyz", num_molecules=30,
                       calculator=make_calc)

# (c) molecular MLIP with no periodic images → finite cluster
system = SystemBuilder(xyz_path="molecule.xyz", num_molecules=30,
                       calculator=MolecularMLIP(), periodic=False)
```

`model=` and `calculator=` are mutually exclusive: `model` is just a built-in shortcut for MACE.

### Crystal-start simulation (from CIF)

```python
system = SystemBuilder(
    cif_path="structure.cif",
    model="MACE-OFF23(Small)",
    device="cpu",
)

sim = IRSpectrumSimulator(system)
spectrum = sim.run(temperature=300.0, eq_time_ps=5.0, prod_time_ps=50.0)
```

---

## API reference

### `SystemBuilder`

```python
SystemBuilder(
    xyz_path: str | Path | None = None,      # single-molecule XYZ (random packing)
    num_molecules: int = 0,
    model: str | None = None,                 # MACE shortcut, e.g. "MACE-OFF23(Small)"
    calculator=None,                          # any ASE calculator, or factory(atoms)->calc
    density_gcc: float = 0.85,                # target density (g/cm³)
    device: str = "cpu",                      # "cpu" | "cuda" | "mps" | "auto" (MACE only)
    dtype: str = "float32",
    cif_path: str | Path | None = None,       # crystal-start (alternative to xyz_path)
    supercell: tuple[int, int, int] | None = None,
    periodic: bool = True,                    # False → finite cluster (molecular MLIPs)
)
```

Provide **either** `model` (MACE shortcut) **or** `calculator` (any ASE calculator).
Call `.build()` to obtain the ASE `Atoms` object (result is cached).

### `IRSpectrumSimulator.run()`

| Parameter | Default | Description |
|---|---|---|
| `temperature` | — | Target temperature (K) |
| `pressure` | `1.0` | NPT pressure (bar) |
| `eq_time_ps` | — | NVT equilibration duration (ps) |
| `prod_time_ps` | — | NVE production duration (ps) — determines frequency resolution |
| `timestep_fs` | `0.5` | MD timestep (fs) |
| `fwhm_cm1` | `10.0` | Gaussian peak width (cm⁻¹) |
| `quantum_correction` | `"schofield"` | `"schofield"` / `"detailed_balance"` / `"none"` |
| `charge_method` | `"auto"` | dipole charges when the calculator has none: `"auto"` (xtb→gaff), `"xtb"` (GFN2), `"gaff"` (rule-based) |
| `traj_path` | `None` | Path for Extended-XYZ trajectory (OVITO-readable) |
| `checkpoint_path` | `None` | Checkpoint file for pause/resume |
| `anneal_T` | `0.0` | Melt temperature for annealing stage (K) |
| `anneal_ps` | `0.0` | NPT hold at `anneal_T` (ps) |
| `cool_ps` | `0.0` | Slow-cool `anneal_T → temperature` (ps) |

Returns an `IRSpectrum`.

### `IRSpectrum`

#### Plotting

```python
fig, ax = spectrum.plot(
    ax=None,               # existing Axes, or None to create a new figure
    label="Simulated",
    color="steelblue",
    figsize=(10, 4),
    xlim=(4500, 200),      # IR convention: high ν on the left
    ylim=(-0.02, 1.12),
    # any extra kwargs are forwarded to ax.plot()
)

# Overlay two spectra
fig, ax = spectrum.compare(
    other,
    labels=("Simulated", "Experimental"),
    colors=("steelblue", "firebrick"),
)
```

#### Saving and loading

```python
spectrum.save("out.csv")                      # CSV (default)
spectrum.save("out.json", format="json")      # JSON (includes metadata)
spectrum.save_csv("out.csv")
spectrum.save_json("out.json")

loaded = IRSpectrum.load("out.csv")           # auto-detect from extension
loaded = IRSpectrum.load("out.json")
loaded = IRSpectrum.load_csv("out.csv")
loaded = IRSpectrum.load_json("out.json")
```

`spectrum.metadata` is a dict containing simulation parameters
(temperature, timestep, frequency resolution, quantum correction, final density, etc.).

---

## Frequency resolution

The intrinsic frequency resolution scales with production time:

```
Δν [cm⁻¹] = 1 / (N_steps × dt_s × c_cm)
```

| `prod_time_ps` (dt = 0.5 fs) | Δν |
|---|---|
| 10 ps | ≈ 3.3 cm⁻¹ |
| 33 ps | ≈ 1.0 cm⁻¹ |
| 50 ps | ≈ 0.7 cm⁻¹ |

Zero-padding (applied internally) refines the display grid but does not improve the intrinsic resolution.

---

## License

MIT
