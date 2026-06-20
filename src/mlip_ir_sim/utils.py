from __future__ import annotations

import numpy as np
from scipy import constants

# Physical constants
C_LIGHT = constants.c  # m/s
H_PLANCK = constants.h  # J·s
K_BOLTZMANN = constants.k  # J/K
E_CHARGE = constants.e  # C
HBAR = constants.hbar  # J·s


def fs_to_s(t_fs: float) -> float:
    return t_fs * 1e-15


def cm1_to_hz(nu_cm1: float | np.ndarray) -> float | np.ndarray:
    return nu_cm1 * 100.0 * C_LIGHT


def eA_to_Cm(mu_eA: float | np.ndarray) -> float | np.ndarray:
    return mu_eA * E_CHARGE * 1e-10


def absorption_prefactor(nu_cm1: np.ndarray, temperature_K: float) -> np.ndarray:
    """Detailed-balance IR-absorption prefactor  ω·(1 − e^{−βℏω}).

    The quantum IR absorption coefficient from the dipole autocorrelation
    function is

        α(ω) ∝ ω · (1 − e^{−βℏω}) · J(ω),

    where J(ω) = FT[⟨M(0)·M(t)⟩] is the (classical) dipole spectral density.
    The factor ω·(1 − e^{−βℏω}) enforces detailed balance and, in the
    classical limit βℏω ≪ 1, reduces to ω·βℏω ∝ ω² — i.e. the familiar ω²
    lineshape — while suppressing the spurious over-weighting of
    high-frequency modes that a bare ω² factor produces.

    Returned values are unnormalised (shape only); the spectrum is rescaled
    to unit maximum downstream, so absolute prefactors are irrelevant.
    """
    omega = 2.0 * np.pi * cm1_to_hz(nu_cm1)
    beta = 1.0 / (K_BOLTZMANN * temperature_K)
    x = beta * HBAR * omega
    return omega * (1.0 - np.exp(-x))


def schofield_factor(nu_cm1: np.ndarray, temperature_K: float) -> np.ndarray:
    """Schofield (harmonic quantum) correction  βℏω / (1 − e^{−βℏω}).

    Converts the classical dipole spectral density J_cl(ω) into the quantum
    spectral density J_QM(ω) ≈ Q(ω)·J_cl(ω) (exact for harmonic systems):

        Q(ω) = βℏω / (1 − e^{−βℏω})

    Physical interpretation: classical MD underestimates high-frequency mode
    amplitudes because equipartition assigns energy kT per mode, while the
    quantum ground state has zero-point energy ℏω/2 >> kT (e.g. C-H stretch
    at 3000 cm⁻¹, T=298 K: ZPE = 0.186 eV vs kT = 0.026 eV, ratio ≈ 7).
    Q corrects this underestimation; it is large where βℏω >> 1.

    Note: absorption_prefactor × schofield_factor = β·ℏ·ω², i.e. the two
    corrections together reproduce the classical ω²·J_cl formula, which is
    both the classical limit and the full quantum-corrected result in the
    harmonic approximation.

    At low frequencies (βℏω → 0), Q → 1 (no correction needed).
    """
    omega = 2.0 * np.pi * cm1_to_hz(nu_cm1)
    beta = 1.0 / (K_BOLTZMANN * temperature_K)
    x = beta * HBAR * omega
    # Avoid divide-by-zero at ω=0 and numerical overflow for large x
    safe_x = np.where(np.abs(x) < 1e-6, 1e-6, x)
    return np.where(np.abs(x) < 1e-6, 1.0, safe_x / (1.0 - np.exp(-safe_x)))
