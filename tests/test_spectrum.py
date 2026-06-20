"""Fast unit tests for IRSpectrum — no MACE or heavy dependencies required."""
import json
from pathlib import Path

import numpy as np
import pytest

from mlip_ir_sim import IRSpectrum

FREQ = np.arange(200.0, 4501.0, 1.0)
INTS = np.abs(np.sin(FREQ / 500.0))
META = {"temperature_K": 300.0, "fwhm_cm1": 10.0, "spectrum_type": "Transmission"}


@pytest.fixture
def spectrum():
    return IRSpectrum(frequencies=FREQ.copy(), intensities=INTS.copy(), metadata=META.copy())


# ── save / load ──────────────────────────────────────────────────────────────

def test_csv_roundtrip(spectrum, tmp_path):
    p = tmp_path / "spec.csv"
    spectrum.save(p)
    loaded = IRSpectrum.load(p)
    np.testing.assert_allclose(loaded.frequencies, spectrum.frequencies)
    np.testing.assert_allclose(loaded.intensities, spectrum.intensities)


def test_json_roundtrip(spectrum, tmp_path):
    p = tmp_path / "spec.json"
    spectrum.save(p, format="json")
    loaded = IRSpectrum.load(p)
    np.testing.assert_allclose(loaded.frequencies, spectrum.frequencies)
    np.testing.assert_allclose(loaded.intensities, spectrum.intensities)
    assert loaded.metadata["temperature_K"] == 300.0


def test_json_contains_metadata(spectrum, tmp_path):
    p = tmp_path / "spec.json"
    spectrum.save_json(p)
    data = json.loads(p.read_text())
    assert "wavenumber_cm1" in data
    assert "absorbance" in data
    assert data["metadata"]["spectrum_type"] == "Transmission"


def test_load_autodetect_csv(spectrum, tmp_path):
    p = tmp_path / "auto.csv"
    spectrum.save_csv(p)
    loaded = IRSpectrum.load(p)
    assert len(loaded.frequencies) == len(spectrum.frequencies)


def test_load_autodetect_json(spectrum, tmp_path):
    p = tmp_path / "auto.json"
    spectrum.save_json(p)
    loaded = IRSpectrum.load(p)
    assert len(loaded.frequencies) == len(spectrum.frequencies)


def test_save_unknown_format_raises(spectrum, tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        spectrum.save(tmp_path / "spec.txt", format="txt")


# ── plotting ─────────────────────────────────────────────────────────────────

def test_plot_returns_fig_ax(spectrum):
    import matplotlib
    matplotlib.use("Agg")
    result = spectrum.plot()
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert hasattr(fig, "savefig")
    assert hasattr(ax, "set_xlabel")


def test_plot_with_existing_ax(spectrum):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    returned_fig, returned_ax = spectrum.plot(ax=ax)
    assert returned_ax is ax
    assert returned_fig is fig
    plt.close(fig)


def test_compare_returns_fig_ax(spectrum):
    import matplotlib
    matplotlib.use("Agg")
    other = IRSpectrum(frequencies=FREQ.copy(), intensities=INTS * 0.8, metadata={})
    fig, ax = spectrum.compare(other, labels=("A", "B"))
    assert hasattr(fig, "savefig")
    plt_lines = ax.get_lines()
    assert len(plt_lines) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)
