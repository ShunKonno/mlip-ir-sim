# mlip-ir-sim

**MLIP-based Infrared Spectroscopy Simulator**

`mlip-ir-sim` is a Python library to compute **“pure” bulk IR absorption spectra (equivalent to transmission measurements)** from molecular dynamics (MD) trajectories driven by machine-learning interatomic potentials (MLIPs), and to compare them with experimental data.  
The primary output is a **transmission-equivalent (bulk) IR spectrum** derived from dipole fluctuations in MD.  
Optionally, the transmission-equivalent spectrum can be **converted to an ATR-like spectrum**, so you can inspect and compare **both** (Transmission and ATR) side-by-side.

---

## Background and design principles

From an MD trajectory, one can track the total dipole moment \(\mu(t)\) of the system. By computing its autocorrelation function (ACF) and Fourier transforming it (Wiener–Khinchin theorem), we obtain a **fundamental bulk IR spectrum** that corresponds most directly to **transmission-like absorption**.

ATR (attenuated total reflection), while experimentally convenient, can show **wavelength-dependent distortions** (e.g., via penetration depth and Fresnel effects). To enable direct comparison with ATR-FTIR datasets, this project provides an **optional post-processing step** that maps the bulk (transmission-equivalent) spectrum to an ATR-like observable.

---

## Key features (planned)

- **Automated initialization and equilibration**
  - Load an optimized single-molecule structure (XYZ) and an MLIP model (e.g., MACE)
  - Build a periodic cell with the requested number of molecules and equilibrate (e.g., NPT)
- **Dipole moment tracking**
  - Run a production MD trajectory and record energies, forces, and \(\mu(t)\)
- **Pure bulk IR via Wiener–Khinchin**
  - ACF of \(\mu(t)\) → FFT-based spectral estimation
- **Transmission → ATR conversion (optional)**
  - Apply a wavelength-dependent transformation to approximate ATR-FTIR observables

---

## Tech stack (planned)

- **Molecular simulation**: `ase`
- **MLIP calculator**: `mace-torch` (or a compatible ASE calculator)
- **Numerics & signal processing**: `numpy`, `scipy`
- **Plotting**: `matplotlib` (optional)

---

## Usage (proposed API)

This is a target API shown in the README; implementations will be added incrementally.

```python
from mlip_ir_sim import SystemBuilder, IRSpectrumSimulator
from mlip_ir_sim.corrections import apply_atr_correction

# 1) Build a periodic system from an XYZ and an MLIP model
system = SystemBuilder(xyz_path="data/sample_mol.xyz", num_molecules=32, model="mace_mp")

# 2) Run equilibration + production and compute a pure bulk IR spectrum
simulator = IRSpectrumSimulator(system)
pure_ir = simulator.run(temperature=300, pressure=1.0, eq_time_ps=5.0, prod_time_ps=20.0)

# 3) Optionally convert the transmission-equivalent spectrum to an ATR-like spectrum
atr = apply_atr_correction(pure_ir, refractive_index_crystal=2.4)  # e.g., Diamond ATR

# Save and plot (examples)
pure_ir.plot(show_experimental="data/sample_exp_transmission.csv")
atr.plot(show_experimental="data/sample_exp_atr.csv")
```

---

## Directory structure

```
mlip-ir-sim/
  README.md
  requirements.txt
  .gitignore
  src/
    mlip_ir_sim/
      __init__.py
      builder.py
      simulator.py
      spectrum.py
      corrections.py
  notebooks/
    demo_simulation.ipynb
  data/
    sample_mol.xyz
    sample_exp_transmission.csv
    sample_exp_atr.csv
```

---

## Development notes (first milestones)

- Load an XYZ and replicate/place molecules into a PBC cell
- Run short MD with an arbitrary ASE calculator (a simple LJ toy system is fine to start)
- Record \(\mu(t)\) → ACF → FFT and obtain a “reasonable-looking” spectrum (units/prefactors to be made rigorous later)

