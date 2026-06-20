"""Dipole-ACF → IR spectrum pipeline.

Theory
------
The IR absorption coefficient of an isotropic system is

    α(ω) ∝ ω · (1 − e^{−βℏω}) · J(ω),     J(ω) = ∫ ⟨M(0)·M(t)⟩ e^{−iωt} dt,

where M(t) is the total system dipole moment.  J(ω) is the dipole spectral
density (the real cosine transform of the autocorrelation function), and the
prefactor ω·(1 − e^{−βℏω}) is the *detailed-balance* quantum correction.  In
the classical limit βℏω ≪ 1 it reduces to the familiar ω² lineshape, while at
high frequency it suppresses the over-weighting that a bare ω² factor causes.

J(ω) is obtained as the real part of the FFT of the (windowed) ACF; negative
values arising from finite-trajectory windowing are clipped to zero (a power
spectral density is non-negative).  The ACF itself is computed via the
Wiener–Khinchin theorem (FFT-based, O(N log N)).

Frequency resolution Δν = 1/(N·Δt·c): a 1 cm⁻¹ resolution requires a
production trajectory of ≈ 33.4 ps.  Zero-padding only refines the display
grid, not the intrinsic resolution.
"""
from __future__ import annotations

import numpy as np

from .utils import C_LIGHT, absorption_prefactor, schofield_factor


def dipole_acf(dipoles: np.ndarray) -> np.ndarray:
    """Compute the dipole autocorrelation function via Wiener–Khinchin.

    Parameters
    ----------
    dipoles : ndarray, shape (N, 3)
        Dipole moment trajectory in arbitrary units (e·Å recommended).

    Returns
    -------
    acf : ndarray, shape (N,)
        Normalised (value=1 at t=0) unbiased ACF summed over x, y, z.
    """
    dipoles = np.asarray(dipoles, dtype=float)
    if dipoles.ndim == 1:
        dipoles = dipoles[:, None]
    N = len(dipoles)

    # Remove mean (DC component — no contribution to vibrational spectrum)
    dipoles = dipoles - dipoles.mean(axis=0)

    acf = np.zeros(N)
    n_pad = 2 * N  # zero-padding for linear (not circular) correlation

    for alpha in range(dipoles.shape[1]):
        mu = dipoles[:, alpha]
        F = np.fft.rfft(mu, n=n_pad)
        power = np.abs(F) ** 2
        c = np.fft.irfft(power, n=n_pad)[:N].real
        # Unbiased normalisation: divide by number of contributing pairs
        c /= np.arange(N, 0, -1, dtype=float)
        acf += c

    # Normalise so that ACF[0] = 1
    if acf[0] > 0:
        acf /= acf[0]
    return acf


def _gaussian_broaden(
    freqs: np.ndarray,
    intensities: np.ndarray,
    fwhm_cm1: float,
    out_freqs: np.ndarray,
) -> np.ndarray:
    """Sum Gaussians centred at each (freq, intensity) point onto out_freqs."""
    sigma = fwhm_cm1 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    diff = out_freqs[:, None] - freqs[None, :]          # (n_out, n_in)
    broadened = np.exp(-0.5 * (diff / sigma) ** 2) @ intensities  # (n_out,)
    if broadened.max() > 0:
        broadened /= broadened.max()
    return broadened


def ir_spectrum_from_acf(
    acf: np.ndarray,
    dt_s: float,
    temperature: float = 300.0,
    fwhm_cm1: float = 10.0,
    quantum_correction: str = "schofield",
) -> "IRSpectrum":  # noqa: F821 – forward ref resolved at runtime
    """Compute the IR absorption spectrum from the dipole ACF.

    Parameters
    ----------
    acf : ndarray, shape (N,)
        Dipole autocorrelation function (output of :func:`dipole_acf`).
    dt_s : float
        MD timestep in seconds.
    temperature : float
        Simulation temperature in K (used for quantum correction).
    fwhm_cm1 : float
        Gaussian peak width (FWHM) in cm⁻¹ applied after FFT.
        Larger values → smoother spectrum.  Default: 10 cm⁻¹.
    quantum_correction : str
        Quantum correction applied to J_cl(ω) before output.

        ``"schofield"`` (default) — applies both the detailed-balance factor
        ω·(1−e^{−βℏω}) AND the Schofield (ZPE) factor βℏω/(1−e^{−βℏω}).
        Their product is β·ℏ·ω², equivalent to the classical ω²·J_cl formula
        but grounded in the quantum derivation.  Recommended for classical MD
        (MACE, GAFF2, xTB): corrects the systematic underestimation of
        high-frequency (C-H, C=O) mode amplitudes from equipartition.

        ``"detailed_balance"`` — applies only ω·(1−e^{−βℏω}).  In the
        classical limit this reduces to ω², but at high frequency (βℏω >> 1)
        it becomes ≈ ω, suppressing C-H stretches relative to fingerprint
        modes.  Not generally recommended; kept for backward-compatibility.

        ``"none"`` — no prefactor; raw J_cl(ω) output (useful for debugging).

    Returns
    -------
    IRSpectrum
        Frequencies (cm⁻¹) and normalised intensities on a 1 cm⁻¹ grid.
    """
    from .results import IRSpectrum

    N = len(acf)

    # Blackman window to reduce spectral leakage from finite trajectory
    window = np.blackman(N)
    acf_w = acf * window

    # Zero-pad to the next power of 2 beyond 8×N for fine frequency grid
    n_target = 8 * N
    n_pad = 1
    while n_pad < n_target:
        n_pad <<= 1

    # Real cosine transform of the ACF → dipole spectral density J(ω).
    # Re(FFT) is the cosine transform; clip unphysical negative values that
    # arise from finite-trajectory windowing (a spectral density is ≥ 0).
    F = np.fft.rfft(acf_w, n=n_pad)
    psd = np.clip(F.real, 0.0, None)

    # Frequency axis
    freqs_hz = np.fft.rfftfreq(n_pad, d=dt_s)
    freqs_cm1 = freqs_hz / (C_LIGHT * 100.0)

    # Quantum correction prefactor(s) applied to J_cl(ω).
    # "schofield" (default): detailed_balance × Schofield = β·ℏ·ω² — correct
    # treatment for classical MD where ZPE is not sampled by equipartition.
    # "detailed_balance": ω·(1−e^{−βℏω}) only — suppresses C-H at high ν.
    if quantum_correction == "schofield":
        psd *= absorption_prefactor(freqs_cm1, temperature)
        psd *= schofield_factor(freqs_cm1, temperature)
    elif quantum_correction == "detailed_balance":
        psd *= absorption_prefactor(freqs_cm1, temperature)
    elif quantum_correction != "none":
        raise ValueError(f"Unknown quantum_correction={quantum_correction!r}; "
                         "choose 'schofield', 'detailed_balance', or 'none'.")

    # Trim to standard mid-IR range (200–4500 cm⁻¹)
    mask = (freqs_cm1 > 200.0) & (freqs_cm1 < 4500.0)
    freqs_raw = freqs_cm1[mask]
    psd_raw = psd[mask]
    if psd_raw.max() > 0:
        psd_raw /= psd_raw.max()

    # Gaussian broadening: sum Gaussians onto uniform 1 cm⁻¹ output grid
    out_freqs = np.arange(200.0, 4501.0, 1.0)
    psd_out = _gaussian_broaden(freqs_raw, psd_raw, fwhm_cm1, out_freqs)

    return IRSpectrum(
        frequencies=out_freqs,
        intensities=psd_out,
        metadata={
            "spectrum_type": "Transmission",
            "temperature_K": temperature,
            "dt_s": dt_s,
            "n_frames": N,
            "freq_resolution_cm1": float(1.0 / (N * dt_s * C_LIGHT * 100.0)),
            "fwhm_cm1": fwhm_cm1,
            "quantum_correction": quantum_correction,
        },
    )
