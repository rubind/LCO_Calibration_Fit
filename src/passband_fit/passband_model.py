"""Pure array transformations for nominal and fitted effective passbands."""

from __future__ import annotations

import numpy as np

from .curves import (
    atmosphere_at_airmass,
    combine_nominal_components,
    passband_support_mask,
    validate_wavelength_grid,
)


def remap_passband(
    wavelength_angstrom: np.ndarray,
    nominal_passband: np.ndarray,
    *,
    pivot_wavelength_angstrom: float,
    delta_wavelength_angstrom: float = 0.0,
    width_scale: float = 1.0,
    require_nonzero: bool = True,
) -> np.ndarray:
    """Shift and broaden a passband using the corrected output-frame convention.

    The nominal response is sampled at

    ``pivot + (lambda - pivot - delta) / width``.

    Consequently a positive ``delta`` moves the output response redward by
    exactly that many Angstroms relative to the width-only result, while a
    width greater than one broadens it around the pivot.
    """

    wavelength = validate_wavelength_grid(wavelength_angstrom)
    nominal = np.asarray(nominal_passband, dtype=np.float64)
    if nominal.shape != wavelength.shape or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal passband must be finite and match the wavelength grid")
    if np.any(nominal < 0.0) or not np.any(nominal > 0.0):
        raise ValueError("nominal passband must be nonnegative with positive support")
    if not np.isfinite(pivot_wavelength_angstrom) or pivot_wavelength_angstrom <= 0.0:
        raise ValueError("pivot wavelength must be finite and positive")
    if not np.isfinite(delta_wavelength_angstrom):
        raise ValueError("wavelength shift must be finite")
    if not np.isfinite(width_scale) or width_scale <= 0.0:
        raise ValueError("width scale must be finite and positive")

    source_wavelength = pivot_wavelength_angstrom + (
        wavelength - pivot_wavelength_angstrom - delta_wavelength_angstrom
    ) / width_scale
    remapped = np.asarray(
        np.interp(source_wavelength, wavelength, nominal, left=0.0, right=0.0),
        dtype=np.float64,
    )
    if require_nonzero and not np.any(remapped > 0.0):
        raise ValueError("remapped passband lies entirely outside the wavelength grid")
    return remapped


def exponential_warp(
    wavelength_angstrom: np.ndarray,
    *,
    pivot_wavelength_angstrom: float,
    warp_parameter: float = 0.0,
    warp_scale_angstrom: float,
    maximum_absolute_exponent: float = 100.0,
) -> np.ndarray:
    """Return the pivoted warp ``exp[-a (lambda-pivot)/scale]``.

    Positive ``a`` suppresses the red side relative to the blue side.  The
    explicit exponent guard turns numerically unusable sampler proposals into
    clean errors rather than overflow or near-singular response curves.
    """

    wavelength = validate_wavelength_grid(wavelength_angstrom)
    scalars = (
        pivot_wavelength_angstrom,
        warp_parameter,
        warp_scale_angstrom,
        maximum_absolute_exponent,
    )
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("warp parameters must be finite")
    if pivot_wavelength_angstrom <= 0.0 or warp_scale_angstrom <= 0.0:
        raise ValueError("warp pivot and scale must be positive")
    if maximum_absolute_exponent <= 0.0:
        raise ValueError("maximum_absolute_exponent must be positive")
    exponent = -warp_parameter * (
        wavelength - pivot_wavelength_angstrom
    ) / warp_scale_angstrom
    if np.max(np.abs(exponent)) > maximum_absolute_exponent:
        raise ValueError("exponential warp exceeds configured exponent guard")
    return np.asarray(np.exp(exponent), dtype=np.float64)


def instrument_passband(
    wavelength_angstrom: np.ndarray,
    nominal_passband: np.ndarray,
    *,
    pivot_wavelength_angstrom: float,
    delta_wavelength_angstrom: float = 0.0,
    width_scale: float = 1.0,
    warp_parameter: float = 0.0,
    warp_scale_angstrom: float,
    maximum_absolute_exponent: float = 100.0,
) -> np.ndarray:
    """Construct a remapped, exponentially warped atmosphere-free response."""

    remapped = remap_passband(
        wavelength_angstrom,
        nominal_passband,
        pivot_wavelength_angstrom=pivot_wavelength_angstrom,
        delta_wavelength_angstrom=delta_wavelength_angstrom,
        width_scale=width_scale,
    )
    warp = exponential_warp(
        wavelength_angstrom,
        pivot_wavelength_angstrom=pivot_wavelength_angstrom,
        warp_parameter=warp_parameter,
        warp_scale_angstrom=warp_scale_angstrom,
        maximum_absolute_exponent=maximum_absolute_exponent,
    )
    result = remapped * warp
    if not np.all(np.isfinite(result)) or not np.any(result > 0.0):
        raise ValueError("instrument passband is numerically invalid")
    return np.asarray(result, dtype=np.float64)


def observation_passband(
    atmosphere_free_passband: np.ndarray,
    unit_airmass_transmission: np.ndarray,
    airmass: float,
) -> np.ndarray:
    """Multiply an instrument passband by atmosphere at one observation's X."""

    instrument = np.asarray(atmosphere_free_passband, dtype=np.float64)
    atmosphere = atmosphere_at_airmass(unit_airmass_transmission, airmass)
    if instrument.ndim != 1 or instrument.shape != atmosphere.shape:
        raise ValueError("instrument and atmosphere passbands must be matching one-dimensional arrays")
    if not np.all(np.isfinite(instrument)) or np.any(instrument < 0.0):
        raise ValueError("instrument passband must be finite and nonnegative")
    result = instrument * atmosphere
    if not np.any(result > 0.0):
        raise ValueError("observation passband has no positive support")
    return np.asarray(result, dtype=np.float64)


def support_reaches_grid_boundary(
    passband: np.ndarray,
    *,
    relative_threshold: float = 1.0e-6,
) -> bool:
    """Return whether non-negligible response touches either grid endpoint."""

    support = passband_support_mask(passband, relative_threshold)
    return bool(support[0] or support[-1])


# Public aliases matching terminology used elsewhere in the package plan.
build_nominal_passband = combine_nominal_components
remap_nominal_passband = remap_passband
modified_instrument_passband = instrument_passband
effective_observation_passband = observation_passband

