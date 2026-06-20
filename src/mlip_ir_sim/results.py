from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class IRSpectrum:
    """Transmission IR spectrum computed from an MLIP-MD simulation.

    Attributes
    ----------
    frequencies : ndarray
        Wavenumbers in cm⁻¹ on a uniform 1 cm⁻¹ grid (200–4500 cm⁻¹).
    intensities : ndarray
        Normalised absorbance in [0, 1] (arbitrary units).
    metadata : dict
        Simulation parameters stored at creation time (temperature, timestep,
        frequency resolution, quantum correction type, etc.).
    """

    frequencies: np.ndarray
    intensities: np.ndarray
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        ax=None,
        label: str = "Simulated",
        title: str | None = None,
        color: str = "steelblue",
        figsize: tuple[float, float] = (10, 4),
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        **line_kwargs,
    ) -> tuple:
        """Plot the spectrum on a matplotlib Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes. A new figure is created when *None*.
        label : str
            Legend label for this spectrum.
        title : str, optional
            Axes title.
        color : str
            Line colour passed to ``ax.plot``.
        figsize : tuple of float
            ``(width, height)`` in inches, used only when creating a new figure.
        xlim : tuple of float, optional
            ``(xmax, xmin)`` wavenumber limits (IR convention: high ν on the
            left). Defaults to ``(4500, 200)``.
        ylim : tuple of float, optional
            ``(ymin, ymax)`` absorbance limits. Defaults to ``(-0.02, 1.12)``.
        **line_kwargs
            Extra keyword arguments forwarded to ``ax.plot`` (e.g. ``lw``,
            ``ls``, ``alpha``).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.plot(self.frequencies, self.intensities, color=color,
                label=label, lw=line_kwargs.pop("lw", 1.5), **line_kwargs)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Absorbance (a.u.)", fontsize=12)

        _xlim = xlim or (
            min(4500.0, float(self.frequencies.max())),
            max(200.0, float(self.frequencies.min())),
        )
        ax.set_xlim(*_xlim)
        ax.set_ylim(*(ylim or (-0.02, 1.12)))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        if title:
            ax.set_title(title, fontsize=13)
        if created:
            fig.tight_layout()

        return fig, ax

    def compare(
        self,
        other: IRSpectrum,
        *,
        labels: tuple[str, str] = ("Spectrum A", "Spectrum B"),
        colors: tuple[str, str] = ("steelblue", "firebrick"),
        title: str | None = None,
        figsize: tuple[float, float] = (10, 4),
    ) -> tuple:
        """Overlay *self* and *other* on the same axes.

        Parameters
        ----------
        other : IRSpectrum
            Second spectrum to overlay.
        labels : tuple of str
            Legend labels for self and other.
        colors : tuple of str
            Line colours for self and other.
        title : str, optional
            Axes title.
        figsize : tuple of float
            Figure size in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        self.plot(ax=ax, label=labels[0], color=colors[0])
        other.plot(ax=ax, label=labels[1], color=colors[1])
        if title:
            ax.set_title(title, fontsize=13)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Serialisation — save
    # ------------------------------------------------------------------

    def save(self, path: str | Path, format: Literal["csv", "json"] = "csv") -> None:
        """Save the spectrum to *path*.

        Parameters
        ----------
        path : str or Path
            Destination file path. The directory is created if it does not exist.
        format : {'csv', 'json'}
            ``'csv'`` writes a two-column file (wavenumber, absorbance).
            ``'json'`` writes frequency array, intensity array, and metadata.
        """
        if format == "csv":
            self.save_csv(path)
        elif format == "json":
            self.save_json(path)
        else:
            raise ValueError(f"Unknown format {format!r}; choose 'csv' or 'json'.")

    def save_csv(self, path: str | Path) -> None:
        """Save (wavenumber, absorbance) pairs as a two-column CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            path,
            np.column_stack([self.frequencies, self.intensities]),
            delimiter=",",
            header="wavenumber_cm-1,absorbance",
            comments="",
        )

    def save_json(self, path: str | Path) -> None:
        """Save frequencies, intensities, and metadata as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "wavenumber_cm1": self.frequencies.tolist(),
            "absorbance": self.intensities.tolist(),
            "metadata": {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in self.metadata.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Serialisation — load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> IRSpectrum:
        """Load a spectrum from *path*, auto-detecting format from the extension.

        ``.json`` → :meth:`load_json`, everything else → :meth:`load_csv`.
        """
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.load_json(path)
        return cls.load_csv(path)

    @classmethod
    def load_csv(cls, path: str | Path) -> IRSpectrum:
        """Load a spectrum from a two-column CSV file."""
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        return cls(frequencies=data[:, 0], intensities=data[:, 1])

    @classmethod
    def load_json(cls, path: str | Path) -> IRSpectrum:
        """Load a spectrum from a JSON file written by :meth:`save_json`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            frequencies=np.array(payload["wavenumber_cm1"]),
            intensities=np.array(payload["absorbance"]),
            metadata=payload.get("metadata", {}),
        )
