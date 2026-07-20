"""Shared, stable array contracts used across package modules."""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class SpectralCurve:
    """A finite one-dimensional curve on a vacuum-Angstrom wavelength axis."""

    wavelength_angstrom: np.ndarray
    values: np.ndarray
    name: str

    def __post_init__(self) -> None:
        wave = np.asarray(self.wavelength_angstrom, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        if wave.ndim != 1 or values.ndim != 1 or wave.shape != values.shape:
            raise ValueError(f"{self.name}: wavelength and value arrays must be equal-length 1-D arrays")
        if len(wave) < 2 or not np.all(np.isfinite(wave)) or not np.all(np.isfinite(values)):
            raise ValueError(f"{self.name}: curve must contain at least two finite samples")
        if not np.all(np.diff(wave) > 0):
            raise ValueError(f"{self.name}: wavelengths must be strictly increasing")
        object.__setattr__(self, "wavelength_angstrom", wave)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class Spectrum:
    """A CALSPEC-like S-lambda spectrum in cgs per vacuum Angstrom."""

    wavelength_angstrom: np.ndarray
    flux_lambda: np.ndarray
    name: str

    def __post_init__(self) -> None:
        wave = np.asarray(self.wavelength_angstrom, dtype=np.float64)
        flux = np.asarray(self.flux_lambda, dtype=np.float64)
        if wave.ndim != 1 or flux.ndim != 1 or wave.shape != flux.shape:
            raise ValueError(f"{self.name}: wavelength and flux arrays must be equal-length 1-D arrays")
        if len(wave) < 2 or not np.all(np.isfinite(wave)) or not np.all(np.isfinite(flux)):
            raise ValueError(f"{self.name}: spectrum must contain at least two finite samples")
        if not np.all(np.diff(wave) > 0):
            raise ValueError(f"{self.name}: wavelengths must be strictly increasing")
        object.__setattr__(self, "wavelength_angstrom", wave)
        object.__setattr__(self, "flux_lambda", flux)


@dataclass(frozen=True)
class SelectedData:
    """Validated observation arrays after filter and quality selection."""

    row_id: np.ndarray
    target_index: np.ndarray
    instrument_index: np.ndarray
    airmass: np.ndarray
    observed_magnitude: np.ndarray
    magnitude_uncertainty: np.ndarray
    flux: np.ndarray
    flux_uncertainty: np.ndarray
    snr: np.ndarray
    target_names: Sequence[str]
    instrument_names: Sequence[str]
    images: Sequence[str]
    sites: Sequence[str]
    summary: Dict[str, object]

    def __post_init__(self) -> None:
        arrays = [
            self.row_id, self.target_index, self.instrument_index, self.airmass,
            self.observed_magnitude, self.magnitude_uncertainty, self.flux,
            self.flux_uncertainty, self.snr,
        ]
        lengths = {len(np.asarray(item)) for item in arrays}
        if len(lengths) != 1:
            raise ValueError("selected observation arrays must have identical lengths")

