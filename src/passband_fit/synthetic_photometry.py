"""Photon-counting synthetic photometry on vacuum-Angstrom grids."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .curves import trapezoid_integral, trapezoid_weights, validate_wavelength_grid

# Exact SI speed of light converted to Angstrom/s and the conventional AB zero.
SPEED_OF_LIGHT_ANGSTROM_PER_SECOND = 2.99792458e18
AB_ZERO_FNU_CGS = 3631.0e-23  # erg s^-1 cm^-2 Hz^-1


def ab_reference_flux_lambda(wavelength_angstrom: np.ndarray) -> np.ndarray:
    """Return the 3631-Jy AB reference spectrum in cgs per Angstrom.

    With wavelength expressed in Angstroms, using ``c`` in Angstrom/s in
    ``f_lambda = f_nu c / lambda**2`` directly yields per-Angstrom density.
    """

    wavelength = validate_wavelength_grid(wavelength_angstrom)
    return np.asarray(
        AB_ZERO_FNU_CGS * SPEED_OF_LIGHT_ANGSTROM_PER_SECOND / wavelength**2,
        dtype=np.float64,
    )


def photon_counting_integral(
    wavelength_angstrom: np.ndarray,
    passband: np.ndarray,
    flux_lambda: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute ``integral lambda P(lambda) S_lambda(lambda) dlambda``."""

    wavelength = validate_wavelength_grid(wavelength_angstrom)
    response = np.asarray(passband, dtype=np.float64)
    flux = np.asarray(flux_lambda, dtype=np.float64)
    if response.shape != wavelength.shape or flux.shape != wavelength.shape:
        raise ValueError("wavelength, passband, and flux arrays must have identical one-dimensional shapes")
    if not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise ValueError("passband must be finite and nonnegative")
    if not np.all(np.isfinite(flux)):
        raise ValueError("spectrum contains non-finite fluxes")
    integration_weights = trapezoid_weights(wavelength) if weights is None else np.asarray(weights, dtype=np.float64)
    integral = float(
        trapezoid_integral(wavelength * response * flux, weights=integration_weights)
    )
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("photon-counting integral must be finite and positive")
    return integral


def magnitude_from_reference(
    wavelength_angstrom: np.ndarray,
    source_flux_lambda: np.ndarray,
    reference_flux_lambda: np.ndarray,
    observation_passband: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute a photon-counting magnitude against an explicit reference.

    The identical observation passband is used in numerator and denominator;
    it may therefore already include atmosphere at the observation's airmass.
    """

    numerator = photon_counting_integral(
        wavelength_angstrom, observation_passband, source_flux_lambda, weights=weights
    )
    denominator = photon_counting_integral(
        wavelength_angstrom, observation_passband, reference_flux_lambda, weights=weights
    )
    return float(-2.5 * np.log10(numerator / denominator))


def vega_magnitude(
    wavelength_angstrom: np.ndarray,
    source_flux_lambda: np.ndarray,
    vega_flux_lambda: np.ndarray,
    observation_passband: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute a Vega magnitude, defining the supplied Vega spectrum as zero."""

    return magnitude_from_reference(
        wavelength_angstrom,
        source_flux_lambda,
        vega_flux_lambda,
        observation_passband,
        weights=weights,
    )


def ab_magnitude(
    wavelength_angstrom: np.ndarray,
    source_flux_lambda: np.ndarray,
    observation_passband: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute an AB magnitude using the 3631-Jy per-Angstrom reference."""

    reference = ab_reference_flux_lambda(wavelength_angstrom)
    return magnitude_from_reference(
        wavelength_angstrom,
        source_flux_lambda,
        reference,
        observation_passband,
        weights=weights,
    )


def synthetic_magnitude(
    wavelength_angstrom: np.ndarray,
    source_flux_lambda: np.ndarray,
    observation_passband: np.ndarray,
    *,
    magnitude_system: str,
    vega_flux_lambda: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Dispatch to the configured ``'vega'`` or ``'ab'`` convention."""

    system = magnitude_system.strip().lower()
    if system == "ab":
        return ab_magnitude(
            wavelength_angstrom, source_flux_lambda, observation_passband, weights=weights
        )
    if system == "vega":
        if vega_flux_lambda is None:
            raise ValueError("vega_flux_lambda is required for Vega magnitudes")
        return vega_magnitude(
            wavelength_angstrom,
            source_flux_lambda,
            vega_flux_lambda,
            observation_passband,
            weights=weights,
        )
    raise ValueError("magnitude_system must be either 'vega' or 'ab'")


def photon_weighted_mean_wavelength(
    wavelength_angstrom: np.ndarray,
    passband: np.ndarray,
    flux_lambda: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute ``integral lambda^2 P S / integral lambda P S`` in Angstroms."""

    wavelength = validate_wavelength_grid(wavelength_angstrom)
    response = np.asarray(passband, dtype=np.float64)
    flux = np.asarray(flux_lambda, dtype=np.float64)
    if response.shape != wavelength.shape or flux.shape != wavelength.shape:
        raise ValueError("wavelength, passband, and flux arrays must have identical shapes")
    integration_weights = trapezoid_weights(wavelength) if weights is None else np.asarray(weights, dtype=np.float64)
    denominator = photon_counting_integral(
        wavelength, response, flux, weights=integration_weights
    )
    numerator = float(
        trapezoid_integral(wavelength**2 * response * flux, weights=integration_weights)
    )
    if not np.isfinite(numerator) or numerator <= 0.0:
        raise ValueError("mean-wavelength numerator must be finite and positive")
    return numerator / denominator


def grid_convergence_delta_magnitude(
    coarse_wavelength_angstrom: np.ndarray,
    coarse_source_flux_lambda: np.ndarray,
    coarse_passband: np.ndarray,
    coarse_reference_flux_lambda: np.ndarray,
    fine_wavelength_angstrom: np.ndarray,
    fine_source_flux_lambda: np.ndarray,
    fine_passband: np.ndarray,
    fine_reference_flux_lambda: np.ndarray,
) -> float:
    """Return ``abs(m_coarse - m_fine)`` for independently sampled grids."""

    coarse_magnitude = magnitude_from_reference(
        coarse_wavelength_angstrom,
        coarse_source_flux_lambda,
        coarse_reference_flux_lambda,
        coarse_passband,
    )
    fine_magnitude = magnitude_from_reference(
        fine_wavelength_angstrom,
        fine_source_flux_lambda,
        fine_reference_flux_lambda,
        fine_passband,
    )
    return float(abs(coarse_magnitude - fine_magnitude))


def check_grid_convergence(
    coarse_wavelength_angstrom: np.ndarray,
    coarse_source_flux_lambda: np.ndarray,
    coarse_passband: np.ndarray,
    coarse_reference_flux_lambda: np.ndarray,
    fine_wavelength_angstrom: np.ndarray,
    fine_source_flux_lambda: np.ndarray,
    fine_passband: np.ndarray,
    fine_reference_flux_lambda: np.ndarray,
    *,
    tolerance_mag: float = 5.0e-4,
) -> bool:
    """Return whether coarse/fine synthetic magnitudes agree within tolerance."""

    if not np.isfinite(tolerance_mag) or tolerance_mag <= 0.0:
        raise ValueError("tolerance_mag must be finite and positive")
    delta = grid_convergence_delta_magnitude(
        coarse_wavelength_angstrom,
        coarse_source_flux_lambda,
        coarse_passband,
        coarse_reference_flux_lambda,
        fine_wavelength_angstrom,
        fine_source_flux_lambda,
        fine_passband,
        fine_reference_flux_lambda,
    )
    return delta <= tolerance_mag


# Backward-friendly concise aliases.
photon_integral = photon_counting_integral
mean_wavelength = photon_weighted_mean_wavelength

