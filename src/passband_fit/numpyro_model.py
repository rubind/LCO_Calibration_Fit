"""Differentiable hierarchical posterior used by NumPyro NUTS.

The potential evaluates the wavelength-remapped passband and the complete
photon-counting numerator and denominator integrals for every likelihood
evaluation.  It deliberately contains no passband emulator, precomputed
linearization, or color-term approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from .hierarchical_model import ModelPriors, ParameterLayout
from .jax_photometry import instrument_passbands, photon_counting_integral


Array = jax.Array


@dataclass(frozen=True)
class JaxPosteriorData:
    """Immutable numerical inputs to the JAX posterior."""

    wavelength_angstrom: np.ndarray
    trapezoid_weights: np.ndarray
    filter_transmission: np.ndarray
    aluminum_squared: np.ndarray
    quantum_efficiency: np.ndarray
    atmosphere_by_observation: np.ndarray
    spectra_by_target: np.ndarray
    reference_flux: np.ndarray
    target_index: np.ndarray
    instrument_index: np.ndarray
    observed_magnitude: np.ndarray
    magnitude_uncertainty: np.ndarray
    pivot_wavelength_angstrom: float
    warp_scale_angstrom: float
    shift_abs_max_angstrom: float
    width_min: float
    width_max: float
    warp_abs_max: float
    response_threshold: float


def _normal_logpdf(value: Array, mean: float, sigma: float) -> Array:
    standardized = (value - mean) / sigma
    return jnp.sum(-0.5 * standardized**2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * pi))


def _log_halfnormal_in_log_coordinate(
    log_value: Array, scale: float
) -> Array:
    """Return ``log p(exp(x)) + x`` for a half-normal physical prior."""

    physical = jnp.exp(log_value)
    return (
        0.5 * jnp.log(2.0 / pi)
        - jnp.log(scale)
        - 0.5 * (physical / scale) ** 2
        + log_value
    )


def _unpack_theta_jax(
    theta: Array, layout: ParameterLayout
) -> tuple[dict[str, Array], Array]:
    """Apply the existing non-centered vector layout with JAX operations."""

    values: dict[str, Array] = {}
    log_prior = jnp.asarray(0.0, dtype=jnp.float64)
    for family in layout.enabled_families:
        mu_index, log_tau_index, z_slice = layout.family_indices(family)
        mu = theta[mu_index]
        log_tau = theta[log_tau_index]
        tau = jnp.exp(log_tau)
        z = theta[z_slice]
        values[family] = mu + tau * z
        values[f"mu_{family}"] = mu
        values[f"tau_{family}"] = tau
    if "eta" not in values:
        values["eta"] = jnp.zeros(layout.n_instruments, dtype=jnp.float64)
    if "a" not in values:
        values["a"] = jnp.zeros(layout.n_instruments, dtype=jnp.float64)
    if layout.n_targets:
        log_sigma_star_index, _ = layout.star_indices()
        values["sigma_star"] = jnp.exp(theta[log_sigma_star_index])
    else:
        values["sigma_star"] = jnp.asarray(0.0, dtype=jnp.float64)
    values["width"] = jnp.exp(values["eta"])
    values["sigma_extra"] = jnp.exp(theta[-1])
    return values, log_prior


def build_potential_fn(
    data: JaxPosteriorData,
    layout: ParameterLayout,
    priors: ModelPriors,
) -> Callable[[Array], Array]:
    """Build a negative log posterior for ``numpyro.infer.NUTS``.

    The returned function accepts the package's stable, unconstrained sampled
    vector.  NumPyro therefore samples the same non-centered coordinates that
    appear in persisted parameter tables.
    """

    wave = jnp.asarray(data.wavelength_angstrom, dtype=jnp.float64)
    integration_weight = jnp.asarray(data.trapezoid_weights, dtype=jnp.float64)
    filter_transmission = jnp.asarray(data.filter_transmission, dtype=jnp.float64)
    fixed_instrument_response = jnp.asarray(
        data.aluminum_squared * data.quantum_efficiency, dtype=jnp.float64
    )
    atmosphere = jnp.asarray(data.atmosphere_by_observation, dtype=jnp.float64)
    spectra = jnp.asarray(data.spectra_by_target, dtype=jnp.float64)
    reference = jnp.asarray(data.reference_flux, dtype=jnp.float64)
    target_index = jnp.asarray(data.target_index, dtype=jnp.int32)
    instrument_index = jnp.asarray(data.instrument_index, dtype=jnp.int32)
    observed = jnp.asarray(data.observed_magnitude, dtype=jnp.float64)
    uncertainty = jnp.asarray(data.magnitude_uncertainty, dtype=jnp.float64)
    pivot = jnp.asarray(data.pivot_wavelength_angstrom, dtype=jnp.float64)
    warp_scale = jnp.asarray(data.warp_scale_angstrom, dtype=jnp.float64)

    def potential(theta: Array) -> Array:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        parameters, log_prior = _unpack_theta_jax(theta, layout)

        for family in layout.enabled_families:
            mu_index, log_tau_index, z_slice = layout.family_indices(family)
            mu_prior = getattr(priors, f"mu_{family}")
            tau_prior = getattr(priors, f"tau_{family}")
            log_prior = log_prior + _normal_logpdf(
                theta[mu_index], mu_prior.mean, mu_prior.sigma
            )
            log_prior = log_prior + _log_halfnormal_in_log_coordinate(
                theta[log_tau_index], tau_prior.scale
            )
            log_prior = log_prior + _normal_logpdf(theta[z_slice], 0.0, 1.0)
        if layout.n_targets:
            log_sigma_star_index, _ = layout.star_indices()
            log_prior = log_prior + _log_halfnormal_in_log_coordinate(
                theta[log_sigma_star_index], priors.sigma_star.scale
            )
        log_prior = log_prior + _log_halfnormal_in_log_coordinate(
            theta[-1], priors.sigma_extra.scale
        )

        shifts = parameters["delta_lambda"]
        widths = parameters["width"]
        warps = parameters["a"]

        instrument_curves = instrument_passbands(
            wave,
            filter_transmission,
            pivot,
            shifts,
            widths,
            warps,
            warp_scale,
        )
        instrument_curves = instrument_curves * fixed_instrument_response[None, :]
        effective = instrument_curves[instrument_index] * atmosphere
        source_flux = spectra[target_index]

        # Full numerator and denominator trapezoid integrals are evaluated by
        # the shared JAX kernel at every NUTS state.
        numerator = photon_counting_integral(
            wave, effective, source_flux, integration_weight
        )
        denominator = photon_counting_integral(
            wave, effective, reference, integration_weight
        )
        synthetic_magnitude = -2.5 * jnp.log10(numerator / denominator)
        prediction = synthetic_magnitude + parameters["Z"][instrument_index]

        variance = uncertainty**2 + parameters["sigma_extra"] ** 2
        residual = observed - prediction
        precision = 1.0 / variance
        quadratic = jnp.sum(residual**2 * precision)
        log_determinant = jnp.sum(jnp.log(variance))
        if layout.n_targets:
            precision_sum = jnp.bincount(
                target_index, weights=precision, length=layout.n_targets
            )
            weighted_residual_sum = jnp.bincount(
                target_index,
                weights=residual * precision,
                length=layout.n_targets,
            )
            sigma_star_squared = parameters["sigma_star"] ** 2
            denominator_star = 1.0 + sigma_star_squared * precision_sum
            quadratic = quadratic - jnp.sum(
                sigma_star_squared * weighted_residual_sum**2 / denominator_star
            )
            log_determinant = log_determinant + jnp.sum(jnp.log(denominator_star))
        log_likelihood = -0.5 * (
            quadratic + log_determinant + observed.size * jnp.log(2.0 * pi)
        )

        curve_max = jnp.max(effective, axis=1)
        boundary_limit = data.response_threshold * curve_max
        boundary_clear = jnp.all(effective[:, 0] <= boundary_limit) & jnp.all(
            effective[:, -1] <= boundary_limit
        )
        valid = (
            jnp.all(jnp.isfinite(theta))
            & jnp.all(jnp.abs(shifts) <= data.shift_abs_max_angstrom)
            & jnp.all(widths >= data.width_min)
            & jnp.all(widths <= data.width_max)
            & jnp.all(jnp.abs(warps) <= data.warp_abs_max)
            & jnp.all(curve_max > 0.0)
            & boundary_clear
            & jnp.all(numerator > 0.0)
            & jnp.all(denominator > 0.0)
            & jnp.all(jnp.isfinite(synthetic_magnitude))
            & jnp.isfinite(log_prior)
            & jnp.isfinite(log_likelihood)
        )
        return jnp.where(valid, -(log_prior + log_likelihood), jnp.inf)

    return potential
