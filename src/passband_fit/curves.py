"""Numerical helpers for spectral curves on vacuum-Angstrom grids.

This module deliberately contains no file I/O.  Loaders are responsible for
converting wavelength coordinates to vacuum Angstroms and response columns to
dimensionless fractions before constructing :class:`SpectralCurve` objects.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .contracts import SpectralCurve, Spectrum


def validate_wavelength_grid(wavelength_angstrom: np.ndarray) -> np.ndarray:
    """Return a validated, one-dimensional float64 vacuum-Angstrom grid."""

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("wavelength grid must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(wavelength)):
        raise ValueError("wavelength grid contains non-finite values")
    if np.any(wavelength <= 0.0):
        raise ValueError("vacuum wavelengths must be positive")
    if not np.all(np.diff(wavelength) > 0.0):
        raise ValueError("wavelength grid must be strictly increasing")
    return wavelength


def interpolate_values(
    source_wavelength_angstrom: np.ndarray,
    source_values: np.ndarray,
    target_wavelength_angstrom: np.ndarray,
    *,
    left: float = 0.0,
    right: float = 0.0,
) -> np.ndarray:
    """Linearly interpolate finite values onto a vacuum-Angstrom grid.

    The default zero fill is appropriate for throughput curves.  It is not an
    authorization to extrapolate spectra; use :func:`interpolate_spectrum`
    with a support mask for CALSPEC-like data.
    """

    source_wavelength = validate_wavelength_grid(source_wavelength_angstrom)
    target_wavelength = validate_wavelength_grid(target_wavelength_angstrom)
    values = np.asarray(source_values, dtype=np.float64)
    if values.ndim != 1 or values.shape != source_wavelength.shape:
        raise ValueError("source values and wavelengths must be equal-length one-dimensional arrays")
    if not np.all(np.isfinite(values)):
        raise ValueError("source values contain non-finite entries")
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError("interpolation fill values must be finite")
    return np.asarray(
        np.interp(target_wavelength, source_wavelength, values, left=left, right=right),
        dtype=np.float64,
    )


def interpolate_curve(
    curve: SpectralCurve,
    target_wavelength_angstrom: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Interpolate a dimensionless curve, using one fill value at both ends."""

    return interpolate_values(
        curve.wavelength_angstrom,
        curve.values,
        target_wavelength_angstrom,
        left=fill_value,
        right=fill_value,
    )


def clip_dimensionless_response(
    values: np.ndarray,
    *,
    name: str = "response",
    upper: Optional[float] = None,
) -> np.ndarray:
    """Clip small nonphysical response values after validating finiteness.

    Pass ``upper=1`` for physical transmission, reflectivity, or QE inputs.
    A fitted effective passband or wavelength warp need not be capped at one.
    """

    response = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(response)):
        raise ValueError(f"{name} contains non-finite values")
    if upper is not None and (not np.isfinite(upper) or upper <= 0.0):
        raise ValueError("upper bound must be finite and positive")
    return np.clip(response, 0.0, upper)


def trapezoid_weights(wavelength_angstrom: np.ndarray) -> np.ndarray:
    """Return float64 weights whose dot product performs trapezoid integration."""

    wavelength = validate_wavelength_grid(wavelength_angstrom)
    delta = np.diff(wavelength)
    weights = np.empty_like(wavelength)
    weights[0] = 0.5 * delta[0]
    weights[-1] = 0.5 * delta[-1]
    if wavelength.size > 2:
        weights[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    return weights


def trapezoid_integral(
    values: np.ndarray,
    wavelength_angstrom: Optional[np.ndarray] = None,
    *,
    weights: Optional[np.ndarray] = None,
    axis: int = -1,
) -> np.ndarray:
    """Integrate one or more arrays along ``axis`` using trapezoid weights."""

    if (wavelength_angstrom is None) == (weights is None):
        raise ValueError("provide exactly one of wavelength_angstrom or weights")
    values_array = np.asarray(values, dtype=np.float64)
    integration_weights = (
        trapezoid_weights(wavelength_angstrom)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if integration_weights.ndim != 1 or not np.all(np.isfinite(integration_weights)):
        raise ValueError("trapezoid weights must be a finite one-dimensional array")
    normalized_axis = np.core.numeric.normalize_axis_index(axis, values_array.ndim)
    if values_array.shape[normalized_axis] != integration_weights.size:
        raise ValueError("integration axis length does not match trapezoid weights")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("integrand contains non-finite values")
    return np.tensordot(values_array, integration_weights, axes=(normalized_axis, 0))


def extinction_magnitude_to_transmission(
    extinction_mag_per_airmass: np.ndarray,
    *,
    clip_negative: bool = True,
) -> np.ndarray:
    """Convert extinction ``k_lambda`` in mag/airmass to unit-airmass transmission.

    ``T(lambda, 1) = 10**(-0.4 * k_lambda)``.  Negative tabulation artifacts
    are clipped by default; strict callers can instead request an exception.
    """

    extinction = np.asarray(extinction_mag_per_airmass, dtype=np.float64)
    if not np.all(np.isfinite(extinction)):
        raise ValueError("atmospheric extinction contains non-finite values")
    if np.any(extinction < 0.0):
        if not clip_negative:
            raise ValueError("atmospheric extinction cannot be negative")
        extinction = np.maximum(extinction, 0.0)
    transmission = np.power(10.0, -0.4 * extinction)
    return np.asarray(np.clip(transmission, 0.0, 1.0), dtype=np.float64)


def atmosphere_at_airmass(unit_airmass_transmission: np.ndarray, airmass: float) -> np.ndarray:
    """Evaluate a unit-airmass transmission curve as ``T_1**airmass``."""

    transmission = np.asarray(unit_airmass_transmission, dtype=np.float64)
    if not np.all(np.isfinite(transmission)):
        raise ValueError("unit-airmass transmission contains non-finite values")
    if np.any((transmission < 0.0) | (transmission > 1.0)):
        raise ValueError("unit-airmass transmission must lie in [0, 1]")
    if not np.isfinite(airmass) or airmass <= 0.0:
        raise ValueError("airmass must be finite and positive")
    return np.asarray(np.power(transmission, float(airmass)), dtype=np.float64)


def combine_nominal_components(
    filter_transmission: np.ndarray,
    aluminum_squared: np.ndarray,
    quantum_efficiency: np.ndarray,
) -> np.ndarray:
    """Form ``filter * aluminum_squared * QE`` without re-squaring aluminum."""

    components = [
        clip_dimensionless_response(filter_transmission, name="filter transmission"),
        clip_dimensionless_response(aluminum_squared, name="aluminum-squared reflectivity"),
        clip_dimensionless_response(quantum_efficiency, name="quantum efficiency"),
    ]
    if len({component.shape for component in components}) != 1:
        raise ValueError("nominal passband components must have identical shapes")
    if components[0].ndim != 1:
        raise ValueError("nominal passband components must be one-dimensional")
    combined = components[0] * components[1] * components[2]
    if not np.any(combined > 0.0):
        raise ValueError("combined nominal passband has no positive throughput")
    return np.asarray(combined, dtype=np.float64)


def passband_support_mask(passband: np.ndarray, relative_threshold: float = 1.0e-6) -> np.ndarray:
    """Return samples at or above a fraction of a passband's positive maximum."""

    response = np.asarray(passband, dtype=np.float64)
    if response.ndim != 1 or not np.all(np.isfinite(response)):
        raise ValueError("passband must be a finite one-dimensional array")
    if np.any(response < 0.0):
        raise ValueError("passband cannot contain negative values")
    if not np.isfinite(relative_threshold) or not 0.0 < relative_threshold < 1.0:
        raise ValueError("relative support threshold must lie strictly between zero and one")
    peak = float(np.max(response))
    if peak <= 0.0:
        raise ValueError("passband has no positive throughput")
    return response >= relative_threshold * peak


def spectrum_covers_passband(
    spectrum_wavelength_angstrom: np.ndarray,
    grid_wavelength_angstrom: np.ndarray,
    passband: np.ndarray,
    *,
    relative_threshold: float = 1.0e-6,
) -> bool:
    """Return whether a spectrum spans all non-negligible passband support."""

    spectrum_wavelength = validate_wavelength_grid(spectrum_wavelength_angstrom)
    required_minimum, required_maximum = passband_support_interval(
        grid_wavelength_angstrom,
        passband,
        relative_threshold=relative_threshold,
    )
    return bool(
        spectrum_wavelength[0] <= required_minimum
        and spectrum_wavelength[-1] >= required_maximum
    )


def passband_support_interval(
    grid_wavelength_angstrom: np.ndarray,
    passband: np.ndarray,
    *,
    relative_threshold: float = 1.0e-6,
) -> tuple[float, float]:
    """Return the inclusive wavelength interval of non-negligible response."""

    grid = validate_wavelength_grid(grid_wavelength_angstrom)
    support = passband_support_mask(passband, relative_threshold)
    if support.shape != grid.shape:
        raise ValueError("passband and grid must have identical shapes")
    support_wavelength = grid[support]
    return float(support_wavelength[0]), float(support_wavelength[-1])


def spectrum_coverage_error(
    spectrum: Spectrum,
    grid_wavelength_angstrom: np.ndarray,
    passband: np.ndarray,
    *,
    relative_threshold: float = 1.0e-6,
) -> Optional[str]:
    """Describe missing blue/red coverage, or return ``None`` when covered."""

    required_minimum, required_maximum = passband_support_interval(
        grid_wavelength_angstrom,
        passband,
        relative_threshold=relative_threshold,
    )
    available_minimum = float(spectrum.wavelength_angstrom[0])
    available_maximum = float(spectrum.wavelength_angstrom[-1])
    missing = []
    if available_minimum > required_minimum:
        missing.append(
            f"blue tail by {available_minimum - required_minimum:.1f} Angstrom"
        )
    if available_maximum < required_maximum:
        missing.append(
            f"red tail by {required_maximum - available_maximum:.1f} Angstrom"
        )
    if not missing:
        return None
    return (
        f"{spectrum.name}: available {available_minimum:.1f}--{available_maximum:.1f} "
        f"Angstrom; required {required_minimum:.1f}--{required_maximum:.1f} Angstrom; "
        f"missing {' and '.join(missing)}"
    )


def interpolate_spectrum(
    spectrum: Spectrum,
    target_wavelength_angstrom: np.ndarray,
    *,
    passband: Optional[np.ndarray] = None,
    relative_threshold: float = 1.0e-6,
) -> np.ndarray:
    """Interpolate a spectrum without extrapolating material passband support.

    If ``passband`` is omitted the spectrum must cover the entire target grid.
    Values outside the spectrum are filled with zero only after that coverage
    requirement has been satisfied, so they cannot contribute to an integral.
    """

    grid = validate_wavelength_grid(target_wavelength_angstrom)
    if passband is None:
        covered = spectrum.wavelength_angstrom[0] <= grid[0] and spectrum.wavelength_angstrom[-1] >= grid[-1]
    else:
        covered = spectrum_covers_passband(
            spectrum.wavelength_angstrom,
            grid,
            passband,
            relative_threshold=relative_threshold,
        )
    if not covered:
        if passband is None:
            detail = (
                f"{spectrum.name}: available {spectrum.wavelength_angstrom[0]:.1f}--"
                f"{spectrum.wavelength_angstrom[-1]:.1f} Angstrom; required "
                f"{grid[0]:.1f}--{grid[-1]:.1f} Angstrom"
            )
        else:
            detail = spectrum_coverage_error(
                spectrum,
                grid,
                passband,
                relative_threshold=relative_threshold,
            )
        raise ValueError(
            f"spectrum does not cover the full required wavelength range: {detail}"
        )
    return interpolate_values(
        spectrum.wavelength_angstrom,
        spectrum.flux_lambda,
        grid,
        left=0.0,
        right=0.0,
    )


# Concise aliases useful to callers constructing atmosphere curves.
extinction_to_transmission = extinction_magnitude_to_transmission
transmission_at_airmass = atmosphere_at_airmass
