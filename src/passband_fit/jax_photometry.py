"""Differentiable, full-grid photon-counting photometry implemented in JAX.

The functions in this module perform the same wavelength-grid trapezoid sums
as the NumPy kernel.  They do not use an emulator, band moments, resampling, or
any other approximation.  Inputs are promoted to float64 and all public
functions are compatible with :func:`jax.jit` and :func:`jax.grad` for valid
physical parameters.

File loading, unit conversion, and host-side validation remain the
responsibility of the existing I/O and preparation layers.  Vega and AB modes
share the same numerical operation once the appropriate reference ``S_lambda``
array has been supplied.
"""

from __future__ import annotations

from typing import Optional

from jax import config as _jax_config

# Synthetic photometry needs substantially better precision than JAX's default
# float32 mode.  This must be set before importing jax.numpy.
_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp


Array = jax.Array

SPEED_OF_LIGHT_ANGSTROM_PER_SECOND = 2.99792458e18
AB_ZERO_FNU_CGS = 3631.0e-23  # erg s^-1 cm^-2 Hz^-1


def _float64(value: object) -> Array:
    return jnp.asarray(value, dtype=jnp.float64)


def trapezoid_weights(wavelength_angstrom: object) -> Array:
    """Return full-grid trapezoid weights for an increasing wavelength axis."""

    wavelength = _float64(wavelength_angstrom)
    spacing = jnp.diff(wavelength)
    interior = 0.5 * (spacing[:-1] + spacing[1:])
    return jnp.concatenate(
        (0.5 * spacing[:1], interior, 0.5 * spacing[-1:]), axis=0
    )


def ab_reference_flux_lambda(wavelength_angstrom: object) -> Array:
    """Return the 3631-Jy AB reference in cgs per vacuum Angstrom."""

    wavelength = _float64(wavelength_angstrom)
    return (
        AB_ZERO_FNU_CGS
        * SPEED_OF_LIGHT_ANGSTROM_PER_SECOND
        / wavelength**2
    )


def remap_passband(
    wavelength_angstrom: object,
    nominal_passband: object,
    pivot_wavelength_angstrom: object,
    delta_wavelength_angstrom: object = 0.0,
    width_scale: object = 1.0,
) -> Array:
    """Apply the exact output-frame shift/width wavelength remapping.

    The nominal passband is evaluated at

    ``pivot + (wavelength - pivot - delta) / width``.

    Thus positive ``delta`` moves the output passband redward by exactly that
    many Angstroms relative to its width-only position.  ``jnp.interp`` matches
    the NumPy kernel's linear interpolation and zero fill outside the grid.
    Invalid nonpositive widths produce NaNs rather than a misleading curve.
    """

    wavelength = _float64(wavelength_angstrom)
    nominal = _float64(nominal_passband)
    pivot = _float64(pivot_wavelength_angstrom)
    delta = _float64(delta_wavelength_angstrom)
    width = _float64(width_scale)
    source_wavelength = pivot + (wavelength - pivot - delta) / width
    remapped = jnp.interp(
        source_wavelength,
        wavelength,
        nominal,
        left=0.0,
        right=0.0,
    )
    return jnp.where(width > 0.0, remapped, jnp.nan)


def exponential_warp(
    wavelength_angstrom: object,
    pivot_wavelength_angstrom: object,
    warp_parameter: object,
    warp_scale_angstrom: object,
    maximum_absolute_exponent: object = 100.0,
) -> Array:
    """Return ``exp[-a (wavelength-pivot)/scale]`` with a finite guard.

    For valid parameters this is exactly the NumPy warp.  Invalid scales or an
    exponent beyond the configured numerical guard produce NaNs, which a JAX
    log-posterior can map to negative infinity without Python exceptions.
    """

    wavelength = _float64(wavelength_angstrom)
    pivot = _float64(pivot_wavelength_angstrom)
    warp = _float64(warp_parameter)
    scale = _float64(warp_scale_angstrom)
    guard = _float64(maximum_absolute_exponent)
    exponent = -warp * (wavelength - pivot) / scale
    valid = (scale > 0.0) & (guard > 0.0) & (jnp.max(jnp.abs(exponent)) <= guard)
    return jnp.where(valid, jnp.exp(exponent), jnp.full_like(wavelength, jnp.nan))


def instrument_passband(
    wavelength_angstrom: object,
    nominal_passband: object,
    pivot_wavelength_angstrom: object,
    delta_wavelength_angstrom: object = 0.0,
    width_scale: object = 1.0,
    warp_parameter: object = 0.0,
    warp_scale_angstrom: object = 1.0,
    maximum_absolute_exponent: object = 100.0,
) -> Array:
    """Construct one differentiable atmosphere-free instrument passband."""

    remapped = remap_passband(
        wavelength_angstrom,
        nominal_passband,
        pivot_wavelength_angstrom,
        delta_wavelength_angstrom,
        width_scale,
    )
    warp = exponential_warp(
        wavelength_angstrom,
        pivot_wavelength_angstrom,
        warp_parameter,
        warp_scale_angstrom,
        maximum_absolute_exponent,
    )
    return remapped * warp


def instrument_passbands(
    wavelength_angstrom: object,
    nominal_passband: object,
    pivot_wavelength_angstrom: object,
    delta_wavelength_angstrom: object,
    width_scale: object,
    warp_parameter: object,
    warp_scale_angstrom: object,
    maximum_absolute_exponent: object = 100.0,
) -> Array:
    """Vectorize :func:`instrument_passband` over instrument parameters."""

    shifts = _float64(delta_wavelength_angstrom)
    widths = _float64(width_scale)
    warps = _float64(warp_parameter)

    def one(shift: Array, width: Array, warp: Array) -> Array:
        return instrument_passband(
            wavelength_angstrom,
            nominal_passband,
            pivot_wavelength_angstrom,
            shift,
            width,
            warp,
            warp_scale_angstrom,
            maximum_absolute_exponent,
        )

    return jax.vmap(one)(shifts, widths, warps)


def atmosphere_at_airmass(
    unit_airmass_transmission: object, airmass: object
) -> Array:
    """Evaluate ``T(lambda, X) = T(lambda, 1)**X`` without grid reduction."""

    transmission = _float64(unit_airmass_transmission)
    x_value = _float64(airmass)
    evaluated = transmission**x_value[..., None] if x_value.ndim else transmission**x_value
    valid_transmission = jnp.all((transmission >= 0.0) & (transmission <= 1.0))
    valid_airmass = jnp.all(x_value > 0.0)
    return jnp.where(valid_transmission & valid_airmass, evaluated, jnp.nan)


def observation_passband(
    atmosphere_free_passband: object,
    unit_airmass_transmission: object,
    airmass: object,
) -> Array:
    """Multiply one atmosphere-free passband by atmosphere at one airmass."""

    return _float64(atmosphere_free_passband) * atmosphere_at_airmass(
        unit_airmass_transmission, airmass
    )


def photon_counting_integral(
    wavelength_angstrom: object,
    passband: object,
    flux_lambda: object,
    weights: Optional[object] = None,
) -> Array:
    """Compute the full-grid ``sum weights*lambda*P*S_lambda`` integral.

    Leading dimensions are broadcast normally; only the last dimension is the
    wavelength axis.  No samples are dropped or summarized.
    """

    wavelength = _float64(wavelength_angstrom)
    response = _float64(passband)
    flux = _float64(flux_lambda)
    integration_weights = (
        trapezoid_weights(wavelength) if weights is None else _float64(weights)
    )
    return jnp.sum(
        integration_weights * wavelength * response * flux,
        axis=-1,
    )


def magnitude_from_reference(
    wavelength_angstrom: object,
    source_flux_lambda: object,
    reference_flux_lambda: object,
    observation_passband: object,
    weights: Optional[object] = None,
) -> Array:
    """Compute photon-counting magnitudes with one passband in both integrals."""

    numerator = photon_counting_integral(
        wavelength_angstrom,
        observation_passband,
        source_flux_lambda,
        weights,
    )
    denominator = photon_counting_integral(
        wavelength_angstrom,
        observation_passband,
        reference_flux_lambda,
        weights,
    )
    return -2.5 * jnp.log10(numerator / denominator)


def vega_magnitude(
    wavelength_angstrom: object,
    source_flux_lambda: object,
    vega_flux_lambda: object,
    observation_passband: object,
    weights: Optional[object] = None,
) -> Array:
    """Compute a Vega magnitude, making the supplied Vega spectrum zero."""

    return magnitude_from_reference(
        wavelength_angstrom,
        source_flux_lambda,
        vega_flux_lambda,
        observation_passband,
        weights,
    )


def ab_magnitude(
    wavelength_angstrom: object,
    source_flux_lambda: object,
    observation_passband: object,
    weights: Optional[object] = None,
) -> Array:
    """Compute an AB magnitude against the exact 3631-Jy per-A reference."""

    return magnitude_from_reference(
        wavelength_angstrom,
        source_flux_lambda,
        ab_reference_flux_lambda(wavelength_angstrom),
        observation_passband,
        weights,
    )


def photon_weighted_mean_wavelength(
    wavelength_angstrom: object,
    passband: object,
    flux_lambda: object,
    weights: Optional[object] = None,
) -> Array:
    """Compute ``integral lambda^2 P S / integral lambda P S`` in Angstroms."""

    wavelength = _float64(wavelength_angstrom)
    response = _float64(passband)
    flux = _float64(flux_lambda)
    integration_weights = (
        trapezoid_weights(wavelength) if weights is None else _float64(weights)
    )
    denominator = photon_counting_integral(
        wavelength, response, flux, integration_weights
    )
    numerator = jnp.sum(
        integration_weights * wavelength**2 * response * flux,
        axis=-1,
    )
    return numerator / denominator


def observation_magnitudes(
    wavelength_angstrom: object,
    nominal_passband: object,
    spectra_flux_lambda: object,
    reference_flux_lambda: object,
    unit_airmass_transmission: object,
    instrument_index: object,
    target_index: object,
    airmass: object,
    delta_wavelength_angstrom: object,
    width_scale: object,
    warp_parameter: object,
    pivot_wavelength_angstrom: object,
    warp_scale_angstrom: object,
    weights: Optional[object] = None,
    maximum_absolute_exponent: object = 100.0,
) -> Array:
    """Evaluate all observation magnitudes with the exact full-grid kernel.

    ``spectra_flux_lambda`` has shape ``(n_target, n_wavelength)``.  Instrument
    parameter arrays have shape ``(n_instrument,)`` and observation index/
    airmass arrays have shape ``(n_observation,)``.  The reference is a single
    Vega or AB ``S_lambda`` array.  The return shape is ``(n_observation,)``.
    """

    wavelength = _float64(wavelength_angstrom)
    all_spectra = _float64(spectra_flux_lambda)
    reference = _float64(reference_flux_lambda)
    instrument_ids = jnp.asarray(instrument_index, dtype=jnp.int32)
    target_ids = jnp.asarray(target_index, dtype=jnp.int32)
    x_values = _float64(airmass)
    curves = instrument_passbands(
        wavelength,
        nominal_passband,
        pivot_wavelength_angstrom,
        delta_wavelength_angstrom,
        width_scale,
        warp_parameter,
        warp_scale_angstrom,
        maximum_absolute_exponent,
    )
    atmosphere = atmosphere_at_airmass(unit_airmass_transmission, x_values)
    effective = curves[instrument_ids] * atmosphere
    sources = all_spectra[target_ids]
    return magnitude_from_reference(
        wavelength,
        sources,
        reference,
        effective,
        weights,
    )


# Compact aliases matching the NumPy kernel terminology.
photon_integral = photon_counting_integral
mean_wavelength = photon_weighted_mean_wavelength

