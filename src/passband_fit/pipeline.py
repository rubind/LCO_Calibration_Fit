"""End-to-end preparation, inference, persistence, and reporting pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import FitConfig
from .contracts import SelectedData
from .curves import (
    combine_nominal_components,
    interpolate_curve,
    interpolate_spectrum,
    spectrum_coverage_error,
    trapezoid_weights,
)
from .diagnostics import (
    binned_residual_trend,
    information_criteria,
    parameter_summary,
    posterior_predictive_draws,
    posterior_predictive_p_values,
    posterior_predictive_statistics,
    posterior_matrices,
    residual_metrics,
    transform_samples,
)
from .hierarchical_model import HierarchicalPosterior, ModelPriors, ParameterLayout
from .io import (
    load_atmosphere_curves,
    load_curve,
    load_filter_curve,
    load_selected_data,
    load_target_mapping,
    load_target_spectra,
    load_vega_spectrum,
)
from .output import build_manifest, write_csv, write_json, write_manifest, write_resolved_config
from .passband_model import instrument_passband, support_reaches_grid_boundary
from .population_passband import generate_population_mean_passband_products
from .plotting import (
    fitted_passband_panel_title,
    format_quantile_summary,
    format_zeropoint_conventions,
    passband_ensemble_filename,
    passband_model_label,
    plot_acceptance_fractions,
    plot_corner,
    plot_forest,
    plot_log_probability_trace,
    plot_passband,
    plot_passband_ensemble,
    plot_residual_comparison,
    plot_trace,
)
from .numpyro_model import JaxPosteriorData, build_potential_fn
from .numpyro_sampler import NumpyroResult, NumpyroSettings, run_numpyro_nuts
from .synthetic_photometry import (
    ab_magnitude,
    ab_reference_flux_lambda,
    photon_weighted_mean_wavelength,
    synthetic_magnitude,
)


@dataclass
class PreparedFit:
    """All arrays needed by a file-I/O-free likelihood evaluation."""

    config: FitConfig
    selected: SelectedData
    wavelength_angstrom: np.ndarray
    weights: np.ndarray
    filter_transmission: np.ndarray
    aluminum_squared: np.ndarray
    quantum_efficiency: np.ndarray
    nominal_passband: np.ndarray
    atmosphere_site_names: Tuple[str, ...]
    instrument_site_names: Tuple[str, ...]
    unit_airmass_transmission: np.ndarray
    unit_airmass_transmission_by_site: np.ndarray
    unit_airmass_transmission_by_instrument: np.ndarray
    spectra: np.ndarray
    reference_flux: np.ndarray
    vega_flux: np.ndarray
    atmosphere_by_observation: np.ndarray
    coverage_excluded_targets: Tuple[str, ...]
    grid_convergence_max_delta_mag: float

    def instrument_curves(self, parameters: Mapping[str, object]) -> np.ndarray:
        """Build atmosphere-free curves with fixed aluminum and CCD QE."""

        shifts = np.asarray(parameters["delta_lambda"], dtype=float)
        widths = np.asarray(parameters["width"], dtype=float)
        warps = np.asarray(parameters["warp"], dtype=float)
        count = len(self.selected.instrument_names)
        if any(array.shape != (count,) for array in (shifts, widths, warps)):
            raise ValueError("instrument parameter arrays have the wrong shape")
        bounds = self.config.bounds
        if np.any(np.abs(shifts) > bounds.shift_abs_max_angstrom):
            raise ValueError("wavelength shift exceeds configured hard bound")
        if np.any((widths < bounds.width_min) | (widths > bounds.width_max)):
            raise ValueError("width exceeds configured hard bound")
        if np.any(np.abs(warps) > bounds.warp_abs_max):
            raise ValueError("warp exceeds configured hard bound")
        if self.config.model.passband_mode == "fixed":
            fixed_curve = (
                self.filter_transmission
                * self.aluminum_squared
                * self.quantum_efficiency
            )
            return np.broadcast_to(
                fixed_curve[None, :],
                (count, len(self.wavelength_angstrom)),
            ).copy()
        curves = np.empty((count, len(self.wavelength_angstrom)))
        for index in range(count):
            curves[index] = instrument_passband(
                self.wavelength_angstrom,
                self.filter_transmission,
                pivot_wavelength_angstrom=self.config.model.pivot_wavelength_angstrom,
                delta_wavelength_angstrom=float(shifts[index]),
                width_scale=float(widths[index]),
                warp_parameter=float(warps[index]),
                warp_scale_angstrom=self.config.model.warp_scale_angstrom,
            )
            curves[index] *= self.aluminum_squared * self.quantum_efficiency
        return curves

    def evaluator(self, parameters: Mapping[str, object]) -> np.ndarray:
        """Return one synthetic magnitude per selected observation."""

        curves = self.instrument_curves(parameters)
        effective = curves[self.selected.instrument_index] * self.atmosphere_by_observation
        peak = np.max(effective, axis=1)
        endpoint_limit = self.config.grid.response_threshold * peak
        if np.any(effective[:, 0] >= endpoint_limit) or np.any(
            effective[:, -1] >= endpoint_limit
        ):
            raise ValueError("complete effective passband reaches the wavelength-grid boundary")
        source = self.spectra[self.selected.target_index]
        weighted = effective * (self.weights * self.wavelength_angstrom)[None, :]
        numerator = np.sum(weighted * source, axis=1)
        denominator = np.sum(weighted * self.reference_flux[None, :], axis=1)
        if np.any(numerator <= 0) or np.any(denominator <= 0):
            raise ValueError("synthetic photon integral is nonpositive")
        result = -2.5 * np.log10(numerator / denominator)
        if not np.all(np.isfinite(result)):
            raise ValueError("synthetic magnitude is nonfinite")
        return result

    def mean_wavelengths(self, parameters: Mapping[str, object]) -> np.ndarray:
        """Return observation-specific photon-weighted wavelengths."""

        curves = self.instrument_curves(parameters)
        result = np.empty(len(self.selected.row_id))
        for row in range(len(result)):
            passband = curves[self.selected.instrument_index[row]] * self.atmosphere_by_observation[row]
            result[row] = photon_weighted_mean_wavelength(
                self.wavelength_angstrom,
                passband,
                self.spectra[self.selected.target_index[row]],
                weights=self.weights,
            )
        return result


def _subset_selected(
    selected: SelectedData,
    keep: np.ndarray,
    excluded: Sequence[str],
    *,
    fit_passband_shape: bool = True,
) -> SelectedData:
    """Subset rows and rebuild compact target/instrument indices."""

    keep = np.asarray(keep, dtype=bool)
    row_targets = np.asarray(selected.target_names, dtype=object)[selected.target_index[keep]]
    row_instruments = np.asarray(selected.instrument_names, dtype=object)[
        selected.instrument_index[keep]
    ]
    target_names = tuple(dict.fromkeys(str(item) for item in row_targets))
    instrument_names = tuple(dict.fromkeys(str(item) for item in row_instruments))
    target_lookup = {name: index for index, name in enumerate(target_names)}
    instrument_lookup = {name: index for index, name in enumerate(instrument_names)}
    target_index = np.asarray([target_lookup[str(item)] for item in row_targets], dtype=int)
    instrument_index = np.asarray(
        [instrument_lookup[str(item)] for item in row_instruments], dtype=int
    )
    summary = dict(selected.summary)
    distinct_targets = {
        name: int(len(set(str(item) for item in row_targets[row_instruments == name])))
        for name in instrument_names
    }
    leverage_warnings = []
    for name in instrument_names:
        row_count = int(np.count_nonzero(row_instruments == name))
        if fit_passband_shape and distinct_targets[name] < 2:
            leverage_warnings.append(
                f"{name} has only {distinct_targets[name]} distinct target; shape parameters "
                "are prior/hierarchy dominated"
            )
        elif row_count < 4:
            leverage_warnings.append(
                f"{name} has only {row_count} observations; instrument parameters are weakly constrained"
            )
    summary.update(
        {
            "rejected_insufficient_coverage": int(np.count_nonzero(~keep)),
            "coverage_excluded_targets": list(excluded),
            "final_used": int(np.count_nonzero(keep)),
            "observations_per_target": {
                name: int(np.count_nonzero(row_targets == name)) for name in target_names
            },
            "observations_per_instrument": {
                name: int(np.count_nonzero(row_instruments == name)) for name in instrument_names
            },
            "distinct_targets_per_instrument": distinct_targets,
            "leverage_warnings": leverage_warnings,
        }
    )
    return SelectedData(
        row_id=selected.row_id[keep],
        target_index=target_index,
        instrument_index=instrument_index,
        airmass=selected.airmass[keep],
        observed_magnitude=selected.observed_magnitude[keep],
        magnitude_uncertainty=selected.magnitude_uncertainty[keep],
        flux=selected.flux[keep],
        flux_uncertainty=selected.flux_uncertainty[keep],
        snr=selected.snr[keep],
        target_names=target_names,
        instrument_names=instrument_names,
        images=tuple(np.asarray(selected.images, dtype=object)[keep]),
        sites=tuple(np.asarray(selected.sites, dtype=object)[keep]),
        summary=summary,
    )


def _component_arrays(
    config: FitConfig,
    wave: np.ndarray,
    atmosphere_curves: Mapping[str, object],
) -> Tuple[np.ndarray, ...]:
    filter_curve = load_filter_curve(config)
    ccd_curve = load_curve(config.inputs.ccd, "CCD QE")
    aluminum_curve = load_curve(config.inputs.aluminum_squared, "aluminum squared")
    filter_values = interpolate_curve(filter_curve, wave)
    aluminum = interpolate_curve(aluminum_curve, wave)
    qe = interpolate_curve(ccd_curve, wave)
    atmospheres = {
        name: interpolate_curve(curve, wave)
        for name, curve in atmosphere_curves.items()
    }
    nominal = combine_nominal_components(filter_values, aluminum, qe)
    return filter_values, aluminum, qe, atmospheres, nominal


def _effective_coverage_envelope(
    config: FitConfig,
    wave: np.ndarray,
    filter_transmission: np.ndarray,
    aluminum_squared: np.ndarray,
    quantum_efficiency: np.ndarray,
    least_extincted_atmosphere: np.ndarray,
) -> np.ndarray:
    """Union of complete effective responses at transform-bound corners."""

    shifts = (
        (0.0,)
        if config.model.passband_mode == "fixed"
        else (-config.bounds.shift_abs_max_angstrom, config.bounds.shift_abs_max_angstrom)
    )
    widths = (
        (1.0,)
        if config.model.passband_mode in {"fixed", "shift_only"}
        else (config.bounds.width_min, config.bounds.width_max)
    )
    warps = (
        (-config.bounds.warp_abs_max, config.bounds.warp_abs_max)
        if config.model.passband_mode == "shift_width_warp"
        else (0.0,)
    )
    fixed_response = aluminum_squared * quantum_efficiency
    nominal_effective = (
        filter_transmission * fixed_response * least_extincted_atmosphere
    )
    envelope = nominal_effective / np.max(nominal_effective)
    for shift in shifts:
        for width in widths:
            for warp in warps:
                curve = instrument_passband(
                    wave,
                    filter_transmission,
                    pivot_wavelength_angstrom=config.model.pivot_wavelength_angstrom,
                    delta_wavelength_angstrom=shift,
                    width_scale=width,
                    warp_parameter=warp,
                    warp_scale_angstrom=config.model.warp_scale_angstrom,
                )
                effective = curve * fixed_response * least_extincted_atmosphere
                envelope = np.maximum(envelope, effective / np.max(effective))
    if support_reaches_grid_boundary(
        envelope, relative_threshold=config.grid.response_threshold
    ):
        raise ValueError(
            "configured wavelength grid does not contain the hard-bound passband envelope"
        )
    return envelope


def _validate_component_coverage(
    config: FitConfig,
    atmosphere_curves: Mapping[str, object],
) -> None:
    """Allow zero fill only beyond physically negligible atmosphere/QE edges."""

    threshold = config.grid.response_threshold
    component_edge_threshold = max(threshold, 1.0e-3)
    for site_name, atmosphere_curve in atmosphere_curves.items():
        if atmosphere_curve.values[0] > component_edge_threshold:
            raise ValueError(
                f"atmosphere blue-edge coverage failure for site {site_name}; "
                "zero fill is allowed only after the unit-airmass transmission "
                "is negligible. "
                f"Transmission is {atmosphere_curve.values[0]:.6g} at the first "
                f"tabulated wavelength "
                f"{atmosphere_curve.wavelength_angstrom[0]:.1f} Angstrom, above "
                f"the component-edge threshold {component_edge_threshold:g}. "
                "Extend the physical atmosphere model farther blueward."
            )

    ccd_curve = load_curve(config.inputs.ccd, "CCD QE")
    endpoint_limit = component_edge_threshold * float(np.max(ccd_curve.values))
    if ccd_curve.values[-1] > endpoint_limit:
        raise ValueError(
            "CCD QE red-edge coverage failure; zero fill is allowed only after "
            f"the QE is negligible. QE is {ccd_curve.values[-1]:.6g} at the last "
            f"tabulated wavelength {ccd_curve.wavelength_angstrom[-1]:.1f} "
            f"Angstrom, above {component_edge_threshold:g} of its peak. Extend the QE curve."
        )


def _convergence_check(
    config: FitConfig,
    target_spectra: Mapping[str, object],
    vega: object,
    coarse_wave: np.ndarray,
    coarse_nominal: np.ndarray,
    valid_targets: Sequence[str],
    atmosphere_curves: Mapping[str, object],
    site_names: Sequence[str],
) -> float:
    """Compare nominal magnitudes on the configured and half-step grids."""

    half_step = config.grid.spacing_angstrom / 2
    fine_wave = np.arange(
        config.grid.minimum_angstrom,
        config.grid.maximum_angstrom + half_step / 2,
        half_step,
    )
    _, _, _, fine_atmospheres, fine_nominal = _component_arrays(
        config, fine_wave, atmosphere_curves
    )
    _, _, _, coarse_atmospheres, _ = _component_arrays(
        config, coarse_wave, atmosphere_curves
    )
    x_value = 1.3
    fine_weights = trapezoid_weights(fine_wave)
    coarse_weights = trapezoid_weights(coarse_wave)
    deltas = []
    for site_name in site_names:
        fine_passband = fine_nominal * fine_atmospheres[site_name] ** x_value
        coarse_passband = coarse_nominal * coarse_atmospheres[site_name] ** x_value
        if config.model.magnitude_system == "vega":
            fine_reference = interpolate_spectrum(
                vega,
                fine_wave,
                passband=fine_passband,
                relative_threshold=config.grid.response_threshold,
            )
            coarse_reference = interpolate_spectrum(
                vega,
                coarse_wave,
                passband=coarse_passband,
                relative_threshold=config.grid.response_threshold,
            )
        else:
            fine_reference = coarse_reference = None
        for name in valid_targets:
            spectrum = target_spectra[name]
            fine_source = interpolate_spectrum(
                spectrum,
                fine_wave,
                passband=fine_passband,
                relative_threshold=config.grid.response_threshold,
            )
            coarse_source = interpolate_spectrum(
                spectrum,
                coarse_wave,
                passband=coarse_passband,
                relative_threshold=config.grid.response_threshold,
            )
            fine_mag = synthetic_magnitude(
                fine_wave,
                fine_source,
                fine_passband,
                magnitude_system=config.model.magnitude_system,
                vega_flux_lambda=fine_reference,
                weights=fine_weights,
            )
            coarse_mag = synthetic_magnitude(
                coarse_wave,
                coarse_source,
                coarse_passband,
                magnitude_system=config.model.magnitude_system,
                vega_flux_lambda=coarse_reference,
                weights=coarse_weights,
            )
            deltas.append(abs(fine_mag - coarse_mag))
    maximum = float(max(deltas, default=0.0))
    if maximum > config.grid.convergence_tolerance_mag:
        raise ValueError(
            f"grid convergence failed: {maximum:.6g} mag exceeds "
            f"{config.grid.convergence_tolerance_mag:.6g} mag"
        )
    return maximum


def prepare_fit(config: FitConfig) -> PreparedFit:
    """Load all real inputs and fail if any reference spectrum lacks coverage."""

    initial = load_selected_data(config)
    prefix_to_site, atmosphere_curves = load_atmosphere_curves(
        config, initial.sites
    )
    observation_site_names = tuple(
        prefix_to_site[str(prefix).lower()] for prefix in initial.sites
    )
    atmosphere_site_names = tuple(dict.fromkeys(observation_site_names))
    wave = np.arange(
        config.grid.minimum_angstrom,
        config.grid.maximum_angstrom + config.grid.spacing_angstrom / 2,
        config.grid.spacing_angstrom,
    )
    filter_values, aluminum, qe, atmospheres, nominal = _component_arrays(
        config, wave, atmosphere_curves
    )
    _validate_component_coverage(config, atmosphere_curves)
    atmosphere_by_observation = np.asarray(
        [
            atmospheres[site_name] ** float(airmass)
            for site_name, airmass in zip(observation_site_names, initial.airmass)
        ],
        dtype=np.float64,
    )
    least_extincted_atmosphere = np.max(atmosphere_by_observation, axis=0)
    coverage_envelope = _effective_coverage_envelope(
        config,
        wave,
        filter_values,
        aluminum,
        qe,
        least_extincted_atmosphere,
    )
    target_spectra = load_target_spectra(config, initial.target_names)
    vega = load_vega_spectrum(config)
    coverage_failures = []
    detail = spectrum_coverage_error(
        vega,
        wave,
        coverage_envelope,
        relative_threshold=config.grid.response_threshold,
    )
    if detail is not None:
        coverage_failures.append(f"Vega reference — {detail}")
    for name in initial.target_names:
        detail = spectrum_coverage_error(
            target_spectra[name],
            wave,
            coverage_envelope,
            relative_threshold=config.grid.response_threshold,
        )
        if detail is not None:
            coverage_failures.append(f"target {name} — {detail}")
    if coverage_failures:
        raise ValueError(
            "reference spectrum coverage failure; extrapolation is disabled. "
            "Extend the listed stellar spectra through the required passband tails:\n- "
            + "\n- ".join(coverage_failures)
        )
    valid_targets = list(initial.target_names)
    excluded: Tuple[str, ...] = ()
    keep = np.ones(len(initial.row_id), dtype=bool)
    selected = _subset_selected(
        initial,
        keep,
        excluded,
        fit_passband_shape=config.model.passband_mode != "fixed",
    )
    instrument_site_names = []
    for instrument_index in range(len(selected.instrument_names)):
        rows = selected.instrument_index == instrument_index
        names = tuple(
            dict.fromkeys(np.asarray(observation_site_names, dtype=object)[rows])
        )
        if len(names) != 1:
            raise ValueError(
                f"instrument {selected.instrument_names[instrument_index]} maps "
                f"to multiple atmosphere sites: {', '.join(names)}"
            )
        instrument_site_names.append(str(names[0]))
    instrument_site_names_tuple = tuple(instrument_site_names)
    atmosphere_by_instrument = np.asarray(
        [atmospheres[name] for name in instrument_site_names_tuple],
        dtype=np.float64,
    )
    # Retain one representative curve for backward-compatible summaries. The
    # reported population zeropoint conversion averages the per-instrument
    # conversions themselves, and the likelihood uses each observation's site.
    representative_atmosphere = np.mean(atmosphere_by_instrument, axis=0)
    spectra = np.asarray(
        [
            interpolate_spectrum(
                target_spectra[name],
                wave,
                passband=coverage_envelope,
                relative_threshold=config.grid.response_threshold,
            )
            for name in selected.target_names
        ]
    )
    vega_flux = interpolate_spectrum(
        vega,
        wave,
        passband=coverage_envelope,
        relative_threshold=config.grid.response_threshold,
    )
    reference = (
        vega_flux
        if config.model.magnitude_system == "vega"
        else ab_reference_flux_lambda(wave)
    )
    convergence = _convergence_check(
        config,
        target_spectra,
        vega,
        wave,
        nominal,
        selected.target_names,
        atmosphere_curves,
        atmosphere_site_names,
    )
    return PreparedFit(
        config=config,
        selected=selected,
        wavelength_angstrom=wave,
        weights=trapezoid_weights(wave),
        filter_transmission=filter_values,
        aluminum_squared=aluminum,
        quantum_efficiency=qe,
        nominal_passband=nominal,
        atmosphere_site_names=atmosphere_site_names,
        instrument_site_names=instrument_site_names_tuple,
        unit_airmass_transmission=representative_atmosphere,
        unit_airmass_transmission_by_site=np.asarray(
            [atmospheres[name] for name in atmosphere_site_names],
            dtype=np.float64,
        ),
        unit_airmass_transmission_by_instrument=atmosphere_by_instrument,
        spectra=spectra,
        reference_flux=reference,
        vega_flux=vega_flux,
        atmosphere_by_observation=atmosphere_by_observation,
        coverage_excluded_targets=excluded,
        grid_convergence_max_delta_mag=convergence,
    )


def nominal_parameters(prepared: PreparedFit) -> Dict[str, object]:
    count = len(prepared.selected.instrument_names)
    return {
        "delta_lambda": np.zeros(count),
        "width": np.ones(count),
        "warp": np.zeros(count),
        "zeropoint": np.zeros(count),
        "sigma_extra": 0.0,
    }


def fit_baseline(prepared: PreparedFit) -> Dict[str, object]:
    """Maximum-likelihood nominal-passband zeropoints and common scatter."""

    synthetic = prepared.evaluator(nominal_parameters(prepared))
    selected = prepared.selected
    count = len(selected.instrument_names)
    initial_z = np.asarray(
        [
            np.median(
                (selected.observed_magnitude - synthetic)[selected.instrument_index == index]
            )
            for index in range(count)
        ]
    )
    initial = np.r_[initial_z, np.log(0.01)]

    def objective(vector: np.ndarray) -> float:
        sigma = float(np.exp(vector[-1]))
        variance = selected.magnitude_uncertainty**2 + sigma**2
        residual = (
            selected.observed_magnitude
            - synthetic
            - vector[:-1][selected.instrument_index]
        )
        value = 0.5 * np.sum(residual**2 / variance + np.log(2 * np.pi * variance))
        return float(value) if np.isfinite(value) else 1.0e300

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(None, None)] * count + [(-20.0, 0.0)],
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"baseline optimization failed: {result.message}")
    prediction = synthetic + result.x[:-1][selected.instrument_index]
    residual = selected.observed_magnitude - prediction
    return {
        "zeropoint": result.x[:-1],
        "sigma_extra": float(np.exp(result.x[-1])),
        "synthetic_magnitude": synthetic,
        "prediction": prediction,
        "residual": residual,
        "maximum_log_likelihood": float(-result.fun),
        "metrics": residual_metrics(residual, selected.magnitude_uncertainty),
        "criteria": information_criteria(float(-result.fun), count + 1, len(residual)),
    }


def model_priors(config: FitConfig) -> ModelPriors:
    """Translate public configuration priors to inference priors."""

    return ModelPriors.from_mapping(
        {
            "shift_population_mean": config.priors.shift_population_mean,
            "shift_population_scale": config.priors.shift_population_scale,
            "log_width_population_mean": config.priors.log_width_population_mean,
            "log_width_population_scale": config.priors.log_width_population_scale,
            "warp_population_mean": config.priors.warp_population_mean,
            "warp_population_scale": config.priors.warp_population_scale,
            "zeropoint_population_mean": config.priors.zeropoint_population_mean,
            "zeropoint_population_scale": config.priors.zeropoint_population_scale,
            "star_correlated_scatter": config.priors.star_correlated_scatter,
            "extra_scatter": config.priors.extra_scatter,
        }
    )


def build_posterior(
    prepared: PreparedFit,
) -> Tuple[ParameterLayout, HierarchicalPosterior]:
    layout = ParameterLayout(
        prepared.config.model.passband_mode,
        len(prepared.selected.instrument_names),
        prepared.selected.instrument_names,
        n_targets=len(prepared.selected.target_names),
        target_names=prepared.selected.target_names,
    )
    posterior = HierarchicalPosterior(
        layout,
        prepared.selected.instrument_index,
        prepared.selected.observed_magnitude,
        prepared.selected.magnitude_uncertainty,
        prepared.evaluator,
        model_priors(prepared.config),
        target_index=prepared.selected.target_index,
    )
    return layout, posterior


def initial_vector_from_baseline(
    layout: ParameterLayout,
    priors: ModelPriors,
    baseline: Mapping[str, object],
) -> np.ndarray:
    """Seed zeropoint hierarchy and extra scatter from the nominal MLE."""

    vector = layout.initial_vector(priors)
    zeropoints = np.asarray(baseline["zeropoint"], dtype=float)
    mu_index, log_tau_index, z_slice = layout.family_indices("Z")
    mu = float(np.mean(zeropoints))
    tau = max(float(np.std(zeropoints)), 1.0e-3)
    vector[mu_index] = mu
    vector[log_tau_index] = np.log(tau)
    vector[z_slice] = (zeropoints - mu) / tau
    vector[-1] = np.log(max(float(baseline["sigma_extra"]), 1.0e-6))
    return vector


def _safe_label(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not result:
        raise ValueError("run label contains no safe path characters")
    return result


def create_run_directory(config: FitConfig, run_label: Optional[str] = None) -> Path:
    label = (
        run_label
        or config.output.run_label
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    destination = (
        config.output.directory
        / _safe_label(str(config.selected_filter))
        / _safe_label(label)
    )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _conditional_star_draws(
    posterior: HierarchicalPosterior,
    samples: np.ndarray,
    indices: np.ndarray,
    random_seed: int,
) -> np.ndarray:
    """Reconstruct marginalized star effects for selected posterior draws."""

    generator = np.random.default_rng(random_seed)
    return np.asarray(
        [
            posterior.sample_conditional_star_offsets(samples[index], generator)
            for index in indices
        ],
        dtype=np.float64,
    )


def posterior_boundary_diagnostics(
    samples: np.ndarray,
    layout: ParameterLayout,
    config: FitConfig,
    passband_draws: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Summarize posterior proximity to numerical parameter/grid boundaries."""

    physical = [layout.unpack(row) for row in np.asarray(samples)]
    result: Dict[str, object] = {"proximity_fraction": 0.1, "parameters": {}}

    def symmetric(name: str, values: np.ndarray, bound: float) -> None:
        result["parameters"][name] = {
            "hard_min": -float(bound),
            "hard_max": float(bound),
            "posterior_min": float(np.min(values)),
            "posterior_max": float(np.max(values)),
            "fraction_in_outer_10_percent": float(
                np.mean(np.abs(values) >= 0.9 * bound)
            ),
        }

    if "delta_lambda" in layout.enabled_families:
        symmetric(
            "delta_lambda_angstrom",
            np.asarray([draw["delta_lambda"] for draw in physical]),
            config.bounds.shift_abs_max_angstrom,
        )
    if "eta" in layout.enabled_families:
        widths = np.asarray([draw["width"] for draw in physical])
        width_span = config.bounds.width_max - config.bounds.width_min
        edge_distance = np.minimum(
            widths - config.bounds.width_min,
            config.bounds.width_max - widths,
        )
        result["parameters"]["width"] = {
            "hard_min": float(config.bounds.width_min),
            "hard_max": float(config.bounds.width_max),
            "posterior_min": float(np.min(widths)),
            "posterior_max": float(np.max(widths)),
            "fraction_in_outer_10_percent": float(
                np.mean(edge_distance <= 0.1 * width_span)
            ),
        }
    if "a" in layout.enabled_families:
        symmetric(
            "warp",
            np.asarray([draw["warp"] for draw in physical]),
            config.bounds.warp_abs_max,
        )
    if passband_draws is not None:
        curves = np.asarray(passband_draws, dtype=np.float64)
        peak = np.max(curves, axis=-1)
        endpoint_ratio = np.maximum(curves[..., 0], curves[..., -1]) / peak
        result["wavelength_grid"] = {
            "response_threshold": float(config.grid.response_threshold),
            "maximum_endpoint_to_peak_ratio": float(np.max(endpoint_ratio)),
            "fraction_above_one_tenth_threshold": float(
                np.mean(endpoint_ratio >= 0.1 * config.grid.response_threshold)
            ),
        }
    parameter_warning = any(
        details["fraction_in_outer_10_percent"] > 0.01
        for details in result["parameters"].values()
    )
    grid_warning = bool(
        passband_draws is not None
        and result["wavelength_grid"]["fraction_above_one_tenth_threshold"] > 0.01
    )
    result["warning"] = parameter_warning or grid_warning
    return result


def _write_products(
    run_dir: Path,
    prepared: PreparedFit,
    baseline: Mapping[str, object],
    layout: ParameterLayout,
    posterior: HierarchicalPosterior,
    result: NumpyroResult,
    settings: NumpyroSettings,
) -> Dict[str, object]:
    """Write posterior tables, passbands, residuals, and core plots."""

    flat, flat_logp = result.flattened()
    if flat_logp is None:
        raise RuntimeError("NumPyro did not return potential energy/log probability")
    np.savez_compressed(
        run_dir / "flattened_samples.npz",
        samples=flat,
        log_probability=flat_logp,
        parameter_names=np.asarray(layout.parameter_names, dtype="U"),
    )
    sampled_summary = parameter_summary(flat, layout.parameter_names, flat_logp)
    sampled_summary.to_csv(run_dir / "parameter_summary_sampled.csv")
    derived = transform_samples(
        flat, layout.derived_vector, len(layout.derived_parameter_names)
    )
    physical_summary = parameter_summary(
        derived, layout.derived_parameter_names, flat_logp
    )
    reconstruction_count = min(1000, len(flat))
    reconstruction_indices = np.linspace(
        0, len(flat) - 1, reconstruction_count, dtype=int
    )
    reconstructed_star_draws = _conditional_star_draws(
        posterior,
        flat,
        reconstruction_indices,
        settings.seed + 1701,
    )
    if layout.n_targets:
        reconstructed_names = tuple(
            f"star_offset[{name}]" for name in prepared.selected.target_names
        )
        reconstructed_summary = parameter_summary(
            reconstructed_star_draws,
            reconstructed_names,
            flat_logp[reconstruction_indices],
        )
        # These are conditional random draws, not deterministic transforms of
        # the marginal state at the highest saved marginal log probability.
        reconstructed_summary["maximum_posterior"] = np.nan
        physical_summary = pd.concat([physical_summary, reconstructed_summary])
    physical_summary.to_csv(run_dir / "parameter_summary_physical.csv")
    covariance, correlation = posterior_matrices(flat, layout.parameter_names)
    covariance.to_csv(run_dir / "posterior_covariance.csv")
    correlation.to_csv(run_dir / "posterior_correlation.csv")

    prediction_draw_count = min(200, len(flat))
    prediction_indices = np.linspace(
        0, len(flat) - 1, prediction_draw_count, dtype=int
    )
    prediction_star_by_target = _conditional_star_draws(
        posterior, flat, prediction_indices, settings.seed + 1702
    )
    prediction_without_star_draws = np.asarray(
        [posterior.predict_without_star_offsets(flat[index]) for index in prediction_indices]
    )
    star_offset_draws = prediction_star_by_target[:, prepared.selected.target_index]
    prediction_draws = prediction_without_star_draws + star_offset_draws
    synthetic_draws = np.asarray(
        [
            prepared.evaluator(layout.unpack(flat[index]))
            for index in prediction_indices
        ]
    )
    mean_wavelength_draws = np.asarray(
        [
            prepared.mean_wavelengths(layout.unpack(flat[index]))
            for index in prediction_indices
        ]
    )
    prediction_quantiles = np.percentile(prediction_draws, [16, 50, 84], axis=0)
    synthetic_quantiles = np.percentile(synthetic_draws, [16, 50, 84], axis=0)
    wavelength_quantiles = np.percentile(
        mean_wavelength_draws, [16, 50, 84], axis=0
    )
    star_offset_quantiles = np.percentile(star_offset_draws, [16, 50, 84], axis=0)
    prediction_without_star_quantiles = np.percentile(
        prediction_without_star_draws, [16, 50, 84], axis=0
    )
    fitted_prediction = prediction_quantiles[1]
    fitted_prediction_without_star = prediction_without_star_quantiles[1]
    fitted_synthetic = synthetic_quantiles[1]
    fitted_residual = prepared.selected.observed_magnitude - fitted_prediction
    fitted_residual_without_star = (
        prepared.selected.observed_magnitude - fitted_prediction_without_star
    )
    mu_z_index, _, _ = layout.family_indices("Z")
    ab_minus_vega_by_instrument = np.asarray(
        [
            ab_magnitude(
                prepared.wavelength_angstrom,
                prepared.vega_flux,
                prepared.nominal_passband
                * prepared.unit_airmass_transmission_by_instrument[index],
                weights=prepared.weights,
            )
            for index in range(len(prepared.selected.instrument_names))
        ]
    )
    ab_minus_vega = float(np.mean(ab_minus_vega_by_instrument))
    mean_zeropoint_annotation = format_zeropoint_conventions(
        flat[:, mu_z_index],
        prepared.config.model.magnitude_system,
        ab_minus_vega,
    )
    star_dispersion_annotation = None
    if layout.n_targets:
        log_sigma_star_index, _ = layout.star_indices()
        star_dispersion_annotation = format_quantile_summary(
            "star-offset population dispersion σ_star",
            np.exp(flat[:, log_sigma_star_index]),
            unit="mag",
        )
    write_json(
        run_dir / "zeropoint_conventions.json",
        {
            "configured_magnitude_system": prepared.config.model.magnitude_system,
            "ab_minus_vega_magnitude": ab_minus_vega,
            "ab_minus_vega_magnitude_by_instrument": {
                name: float(ab_minus_vega_by_instrument[index])
                for index, name in enumerate(prepared.selected.instrument_names)
            },
            "conversion_passband": (
                "equally instrument-weighted mean of nominal site-specific "
                "unit-airmass AB-minus-Vega conversions"
            ),
            "zeropoint_definition": "fitted additive offset; internal magnitude bookkeeping excluded",
        },
    )
    nominal_wave = prepared.mean_wavelengths(nominal_parameters(prepared))
    fitted_wave = wavelength_quantiles[1]
    table = pd.DataFrame(
        {
            "source_row_id": prepared.selected.row_id,
            "image": prepared.selected.images,
            "target": np.asarray(prepared.selected.target_names)[prepared.selected.target_index],
            "instrument": np.asarray(prepared.selected.instrument_names)[
                prepared.selected.instrument_index
            ],
            "site": prepared.selected.sites,
            "airmass": prepared.selected.airmass,
            "flux": prepared.selected.flux,
            "flux_uncertainty": prepared.selected.flux_uncertainty,
            "snr": prepared.selected.snr,
            "observed_magnitude": prepared.selected.observed_magnitude,
            "magnitude_uncertainty": prepared.selected.magnitude_uncertainty,
            "nominal_synthetic_magnitude": baseline["synthetic_magnitude"],
            "nominal_prediction": baseline["prediction"],
            "nominal_residual": baseline["residual"],
            "nominal_mean_wavelength_angstrom": nominal_wave,
            "fitted_synthetic_magnitude": fitted_synthetic,
            "fitted_synthetic_magnitude_q16": synthetic_quantiles[0],
            "fitted_synthetic_magnitude_q84": synthetic_quantiles[2],
            "fitted_star_offset": star_offset_quantiles[1],
            "fitted_star_offset_q16": star_offset_quantiles[0],
            "fitted_star_offset_q84": star_offset_quantiles[2],
            "fitted_prediction_without_star_offset": fitted_prediction_without_star,
            "fitted_prediction_without_star_offset_q16": prediction_without_star_quantiles[0],
            "fitted_prediction_without_star_offset_q84": prediction_without_star_quantiles[2],
            "fitted_residual_without_star_offset": fitted_residual_without_star,
            "fitted_prediction": fitted_prediction,
            "fitted_prediction_q16": prediction_quantiles[0],
            "fitted_prediction_q84": prediction_quantiles[2],
            "fitted_residual": fitted_residual,
            "fitted_mean_wavelength_angstrom": fitted_wave,
            "fitted_mean_wavelength_q16_angstrom": wavelength_quantiles[0],
            "fitted_mean_wavelength_q84_angstrom": wavelength_quantiles[2],
        }
    )
    write_csv(run_dir / "residuals.csv", table)
    nominal_curve_columns = {
        "wavelength_angstrom": prepared.wavelength_angstrom,
        "filter_transmission": prepared.filter_transmission,
        "aluminum_squared": prepared.aluminum_squared,
        "quantum_efficiency": prepared.quantum_efficiency,
        # Retained for compatibility with summary scripts; this is the
        # equally instrument-weighted representative atmosphere.
        "unit_airmass_transmission": prepared.unit_airmass_transmission,
        "nominal_combined_passband": prepared.nominal_passband,
    }
    for site_index, site_name in enumerate(prepared.atmosphere_site_names):
        nominal_curve_columns[
            f"unit_airmass_transmission_{site_name}"
        ] = prepared.unit_airmass_transmission_by_site[site_index]
    write_csv(run_dir / "nominal_curves.csv", pd.DataFrame(nominal_curve_columns))

    draw_count = min(prepared.config.plotting.posterior_draws, len(flat))
    draw_indices = np.linspace(0, len(flat) - 1, draw_count, dtype=int)
    passband_draws = np.empty(
        (
            draw_count,
            len(prepared.selected.instrument_names),
            len(prepared.wavelength_angstrom),
        )
    )
    physical_draws = []
    for output_index, sample_index in enumerate(draw_indices):
        draw = layout.unpack(flat[sample_index])
        physical_draws.append(draw)
        passband_draws[output_index] = prepared.instrument_curves(draw)
    forest_star_draws = _conditional_star_draws(
        posterior, flat, draw_indices, settings.seed + 1703
    )
    quantiles = np.percentile(passband_draws, [16, 50, 84], axis=0)
    np.savez_compressed(
        run_dir / "posterior_passbands.npz",
        wavelength_angstrom=prepared.wavelength_angstrom,
        q16=quantiles[0],
        median=quantiles[1],
        q84=quantiles[2],
        instrument_names=np.asarray(prepared.selected.instrument_names, dtype="U"),
    )
    least_extincted_atmosphere = np.empty_like(
        prepared.unit_airmass_transmission_by_instrument
    )
    for instrument_index in range(len(prepared.selected.instrument_names)):
        rows = prepared.selected.instrument_index == instrument_index
        least_extincted_atmosphere[instrument_index] = (
            prepared.unit_airmass_transmission_by_instrument[instrument_index]
            ** float(np.min(prepared.selected.airmass[rows]))
        )
    boundary_diagnostics = posterior_boundary_diagnostics(
        flat[draw_indices],
        layout,
        prepared.config,
        passband_draws * least_extincted_atmosphere[None, :, :],
    )
    write_json(run_dir / "boundary_diagnostics.json", boundary_diagnostics)

    plots = run_dir / "plots"
    plot_trace(result.chain, layout.parameter_names, plots / "trace.png")
    plot_log_probability_trace(result.log_probability, plots / "log_probability.png")
    plot_acceptance_fractions(
        result.diagnostics.acceptance_probability_by_chain, plots / "acceptance.png"
    )
    plot_residual_comparison(
        nominal_wave,
        baseline["residual"],
        fitted_wave,
        fitted_residual,
        groups=prepared.selected.instrument_index,
        group_names=prepared.selected.instrument_names,
        filter_name=str(prepared.config.selected_filter),
        output_path=plots / "residuals_by_instrument.png",
    )
    plot_residual_comparison(
        nominal_wave,
        baseline["residual"],
        fitted_wave,
        fitted_residual,
        groups=prepared.selected.target_index,
        group_names=prepared.selected.target_names,
        intermediate_x=fitted_wave,
        intermediate_residual=fitted_residual_without_star,
        panel_titles=(
            "Nominal passband",
            fitted_passband_panel_title(prepared.config.model.passband_mode),
            "Full fitted model, after star offset",
        ),
        panel_annotations=(
            None,
            mean_zeropoint_annotation,
            star_dispersion_annotation,
        ),
        distinct_group_markers=True,
        filter_name=str(prepared.config.selected_filter),
        output_path=plots / "residuals_by_target.png",
    )
    for label, values in (
        ("airmass", prepared.selected.airmass),
        ("observed_magnitude", prepared.selected.observed_magnitude),
        ("snr", prepared.selected.snr),
        ("instrument_index", prepared.selected.instrument_index),
        ("target_index", prepared.selected.target_index),
    ):
        plot_residual_comparison(
            values,
            baseline["residual"],
            values,
            fitted_residual,
            x_label=label.replace("_", " "),
            filter_name=str(prepared.config.selected_filter),
            output_path=plots / f"residuals_vs_{label}.png",
        )
    plot_passband_ensemble(
        prepared.wavelength_angstrom,
        quantiles[1],
        prepared.selected.instrument_names,
        nominal=prepared.nominal_passband,
        title=(
            f"{prepared.config.selected_filter} passbands — model: "
            f"{passband_model_label(prepared.config.model.passband_mode)}"
        ),
        output_path=plots
        / passband_ensemble_filename(
            str(prepared.config.selected_filter),
            prepared.config.model.passband_mode,
        ),
    )
    generate_population_mean_passband_products(run_dir)
    for instrument, name in enumerate(prepared.selected.instrument_names):
        safe_name = _safe_label(name)
        plot_passband(
            prepared.wavelength_angstrom,
            prepared.nominal_passband,
            quantiles[1, instrument],
            quantiles[0, instrument],
            quantiles[2, instrument],
            effective_curves={
                "median at X=1": quantiles[1, instrument]
                * prepared.unit_airmass_transmission_by_instrument[instrument],
                "median at X=2": quantiles[1, instrument]
                * prepared.unit_airmass_transmission_by_instrument[instrument] ** 2,
            },
            title=name,
            output_path=plots / f"passband_{safe_name}.png",
        )

    family_keys = []
    if "delta_lambda" in layout.enabled_families:
        family_keys.append(("delta_lambda", "mu_delta_lambda", "tau_delta_lambda"))
    if "eta" in layout.enabled_families:
        family_keys.append(("width", "mu_eta", "tau_eta"))
    if "a" in layout.enabled_families:
        family_keys.append(("warp", "mu_a", "tau_a"))
    family_keys.append(("zeropoint", "mu_Z", "tau_Z"))
    for family, mu_key, tau_key in family_keys:
        values = np.asarray([draw[family] for draw in physical_draws])
        low, center, high = np.percentile(values, [16, 50, 84], axis=0)
        plot_forest(
            prepared.selected.instrument_names,
            center,
            low,
            high,
            population_mean=float(np.median([draw[mu_key] for draw in physical_draws])),
            population_dispersion=float(np.median([draw[tau_key] for draw in physical_draws])),
            parameter_label=family,
            output_path=plots / f"forest_{family}.png",
        )
    if layout.n_targets:
        low, center, high = np.percentile(forest_star_draws, [16, 50, 84], axis=0)
        plot_forest(
            prepared.selected.target_names,
            center,
            low,
            high,
            population_mean=0.0,
            population_dispersion=float(
                np.median([draw["sigma_star"] for draw in physical_draws])
            ),
            parameter_label="star correlated offset (mag)",
            output_path=plots / "forest_star_offset.png",
        )

    # A smoke chain is too short for meaningful corner contours.
    if result.chain.shape[0] >= max(50, 2 * layout.ndim):
        corner_indices = np.linspace(0, len(flat) - 1, min(len(flat), 5000), dtype=int)
        corner_samples = flat[corner_indices]
        hyper_indices = [
            index
            for index, name in enumerate(layout.parameter_names)
            if name.startswith("mu_")
            or name.startswith("log_tau_")
            or name == "log_sigma_star"
            or name == "log_sigma_extra"
        ]
        plot_corner(
            corner_samples,
            layout.parameter_names,
            subset=hyper_indices,
            output_path=plots / "corner_hyperparameters.png",
        )
        for instrument, name in enumerate(prepared.selected.instrument_names):
            subset = [
                index
                for index, parameter_name in enumerate(layout.parameter_names)
                if f"[{name}]" in parameter_name
            ]
            subset.append(len(layout.parameter_names) - 1)
            plot_corner(
                corner_samples,
                layout.parameter_names,
                subset=subset,
                output_path=plots / f"corner_{_safe_label(name)}.png",
            )
    else:
        write_json(
            run_dir / "plots" / "corner_skipped.json",
            {
                "reason": "chain too short for meaningful corner contours",
                "production_steps": result.chain.shape[0],
                "minimum_steps": max(50, 2 * layout.ndim),
            },
        )

    # Posterior predictive residual distributions and trend checks.
    predictive_draw_count = min(200, len(flat))
    predictive_indices = np.linspace(
        0, len(flat) - 1, predictive_draw_count, dtype=int
    )
    predictive_star_by_target = _conditional_star_draws(
        posterior, flat, predictive_indices, settings.seed + 1704
    )
    model_means = np.asarray(
        [
            posterior.predict_without_star_offsets(flat[index])
            + predictive_star_by_target[row][prepared.selected.target_index]
            for row, index in enumerate(predictive_indices)
        ]
    )
    extra_scatter = np.asarray(
        [layout.unpack(flat[index])["sigma_extra"] for index in predictive_indices]
    )
    replicated = posterior_predictive_draws(
        model_means,
        prepared.selected.magnitude_uncertainty,
        extra_scatter,
        random_seed=settings.seed,
    )
    predictive_residuals = replicated - model_means
    observed_residual = prepared.selected.observed_magnitude - fitted_prediction
    ppc_statistics = posterior_predictive_statistics(
        predictive_residuals, x=fitted_wave
    )
    write_csv(run_dir / "posterior_predictive_statistics.csv", ppc_statistics)
    ppc_values = {
        "wavelength": posterior_predictive_p_values(
            observed_residual, predictive_residuals, x=fitted_wave
        ),
        "airmass": posterior_predictive_p_values(
            observed_residual, predictive_residuals, x=prepared.selected.airmass
        ),
        "by_instrument": {},
        "by_target": {},
    }
    for index, name in enumerate(prepared.selected.instrument_names):
        selected_rows = prepared.selected.instrument_index == index
        ppc_values["by_instrument"][name] = posterior_predictive_p_values(
            observed_residual[selected_rows],
            predictive_residuals[:, selected_rows],
        )
    for index, name in enumerate(prepared.selected.target_names):
        selected_rows = prepared.selected.target_index == index
        ppc_values["by_target"][name] = posterior_predictive_p_values(
            observed_residual[selected_rows],
            predictive_residuals[:, selected_rows],
        )
    write_json(run_dir / "posterior_predictive_p_values.json", ppc_values)
    write_csv(
        run_dir / "binned_residual_trend_nominal.csv",
        binned_residual_trend(nominal_wave, baseline["residual"]),
    )
    write_csv(
        run_dir / "binned_residual_trend_fitted.csv",
        binned_residual_trend(fitted_wave, fitted_residual),
    )

    fitted_metrics = residual_metrics(
        fitted_residual, prepared.selected.magnitude_uncertainty
    )
    likelihood_indices = np.linspace(0, len(flat) - 1, min(len(flat), 5000), dtype=int)
    candidate_likelihoods = np.asarray(
        [posterior.log_likelihood(flat[index]) for index in likelihood_indices]
    )
    maximum_likelihood = float(np.max(candidate_likelihoods))
    comparison = {
        "baseline": {
            "maximum_log_likelihood": baseline["maximum_log_likelihood"],
            "metrics": baseline["metrics"],
            "criteria": baseline["criteria"],
        },
        "fitted": {
            "maximum_searched_sample_log_likelihood": maximum_likelihood,
            "likelihood_samples_searched": int(len(likelihood_indices)),
            "metrics": fitted_metrics,
            "criteria": information_criteria(
                maximum_likelihood, layout.ndim, len(fitted_residual)
            ),
        },
        "aic_bic_note": (
            "Hierarchical AIC/BIC are exploratory and use the best posterior "
            "draw's data likelihood."
        ),
    }
    write_json(run_dir / "model_comparison.json", comparison)
    return {
        "flat_samples": len(flat),
        "fitted_metrics": fitted_metrics,
        "model_comparison": comparison,
        "boundary_diagnostics": boundary_diagnostics,
    }


def _manifest_input_paths(prepared: PreparedFit) -> Sequence[Path]:
    """Return every scientific input used by this selected-filter run."""

    config = prepared.config
    mapping = load_target_mapping(config.inputs.target_mapping)
    paths = [
        config.inputs.observations.path,
        config.inputs.target_mapping,
        config.inputs.vega_spectrum,
        config.inputs.filter_curves[str(config.selected_filter)].path,
        config.inputs.ccd.path,
        config.inputs.aluminum_squared.path,
    ]
    if config.inputs.atmosphere is not None:
        paths.append(config.inputs.atmosphere.path)
    else:
        if config.inputs.site_mapping is None:
            raise ValueError("site-specific atmosphere configuration lacks site_mapping")
        paths.append(config.inputs.site_mapping)
        paths.extend(curve.path for curve in config.inputs.atmospheres.values())
    for target in prepared.selected.target_names:
        path = Path(mapping[target])
        if not path.is_absolute():
            path = config.inputs.spectra_directory / path
        paths.append(path)
    return paths


def run_fit(
    config: FitConfig,
    run_label: Optional[str] = None,
    resume_directory: Optional[Path] = None,
    sampler_override: Optional[NumpyroSettings] = None,
) -> Path:
    """Prepare, sample with NumPyro NUTS, and persist one complete filter run."""

    if resume_directory is not None:
        raise ValueError(
            "NumPyro runs cannot resume an emcee chain.h5 directory; start a new "
            "run with --config and --filter"
        )

    settings = sampler_override or NumpyroSettings(
        num_warmup=config.sampler.num_warmup,
        num_samples=config.sampler.num_samples,
        num_chains=config.sampler.num_chains,
        thinning=config.sampler.thinning,
        seed=config.sampler.random_seed,
        target_accept_prob=config.sampler.target_accept_prob,
        max_tree_depth=config.sampler.max_tree_depth,
        chain_method=config.sampler.chain_method,
        progress_bar=config.sampler.progress,
        dense_mass=config.sampler.dense_mass,
    )
    settings.validate()

    # NumPyro must set virtual CPU devices before the first JAX array/device
    # operation. This gives Apple Silicon one independently scheduled device
    # per parallel chain without Python multiprocessing.
    if settings.chain_method == "parallel" and settings.num_chains > 1:
        import numpyro

        numpyro.set_host_device_count(settings.num_chains)

    prepared = prepare_fit(config)
    layout, posterior = build_posterior(prepared)
    run_kind = "smoke" if sampler_override is not None else "production"
    run_dir = create_run_directory(config, run_label)
    write_resolved_config(run_dir / "resolved_config.yaml", config)
    write_json(run_dir / "selection_summary.json", prepared.selected.summary)
    write_json(
        run_dir / "index_mappings.json",
        {
            "targets": {
                index: name for index, name in enumerate(prepared.selected.target_names)
            },
            "instruments": {
                index: name
                for index, name in enumerate(prepared.selected.instrument_names)
            },
            "instrument_atmosphere_sites": {
                name: prepared.instrument_site_names[index]
                for index, name in enumerate(prepared.selected.instrument_names)
            },
            "image_prefix_atmosphere_sites": {
                prefix: site_name
                for prefix, site_name in sorted(
                    {
                        str(prefix): prepared.instrument_site_names[instrument_index]
                        for prefix, instrument_index in zip(
                            prepared.selected.sites,
                            prepared.selected.instrument_index,
                        )
                    }.items()
                )
            },
        },
    )
    manifest = build_manifest(
        config,
        layout.parameter_names,
        input_paths=_manifest_input_paths(prepared),
        status="running",
        extra={
            "ndim": layout.ndim,
            "n_observations": len(prepared.selected.row_id),
            "grid_convergence_max_delta_mag": prepared.grid_convergence_max_delta_mag,
            "coverage_excluded_targets": prepared.coverage_excluded_targets,
            "run_kind": run_kind,
            "convergence_claim": False,
            "effective_sampler_settings": asdict(settings),
        },
    )
    write_manifest(run_dir / "manifest.json", manifest)

    baseline = fit_baseline(prepared)
    posterior_data = JaxPosteriorData(
        wavelength_angstrom=prepared.wavelength_angstrom,
        trapezoid_weights=prepared.weights,
        filter_transmission=prepared.filter_transmission,
        aluminum_squared=prepared.aluminum_squared,
        quantum_efficiency=prepared.quantum_efficiency,
        atmosphere_by_observation=prepared.atmosphere_by_observation,
        spectra_by_target=prepared.spectra,
        reference_flux=prepared.reference_flux,
        target_index=prepared.selected.target_index,
        instrument_index=prepared.selected.instrument_index,
        observed_magnitude=prepared.selected.observed_magnitude,
        magnitude_uncertainty=prepared.selected.magnitude_uncertainty,
        pivot_wavelength_angstrom=config.model.pivot_wavelength_angstrom,
        warp_scale_angstrom=config.model.warp_scale_angstrom,
        shift_abs_max_angstrom=config.bounds.shift_abs_max_angstrom,
        width_min=config.bounds.width_min,
        width_max=config.bounds.width_max,
        warp_abs_max=config.bounds.warp_abs_max,
        response_threshold=config.grid.response_threshold,
    )
    potential_fn = build_potential_fn(posterior_data, layout, posterior.priors)
    center = initial_vector_from_baseline(layout, model_priors(config), baseline)
    generator = np.random.default_rng(settings.seed)
    starts = np.repeat(center[None, :], settings.num_chains, axis=0)
    if settings.num_chains > 1:
        starts += generator.normal(size=starts.shape) * (
            1.0e-3 * np.maximum(1.0, np.abs(center))[None, :]
        )
    result = run_numpyro_nuts(
        potential_fn,
        starts,
        settings,
        parameter_names=layout.parameter_names,
    )
    if result.log_probability is None:
        raise RuntimeError("NumPyro did not persist potential energy")
    chain_payload = {
        "chain": result.chain,
        "log_probability": result.log_probability,
        "parameter_names": np.asarray(layout.parameter_names, dtype="U"),
    }
    chain_payload.update(
        {f"extra_{name}": value for name, value in result.extra_fields.items()}
    )
    np.savez_compressed(run_dir / "chain.npz", **chain_payload)
    write_json(run_dir / "diagnostics.json", result.diagnostics.as_dict())
    write_csv(
        run_dir / "diagnostics_by_parameter.csv",
        pd.DataFrame(
            {
                "parameter": layout.parameter_names,
                "r_hat": result.diagnostics.rhat,
                "effective_sample_size": result.diagnostics.effective_sample_size,
            }
        ),
    )
    products = _write_products(
        run_dir, prepared, baseline, layout, posterior, result, settings
    )
    summary = {
        "selected_filter": config.selected_filter,
        "n_observations": len(prepared.selected.row_id),
        "n_targets": len(prepared.selected.target_names),
        "n_instruments": len(prepared.selected.instrument_names),
        "coverage_excluded_targets": list(prepared.coverage_excluded_targets),
        "grid_convergence_max_delta_mag": prepared.grid_convergence_max_delta_mag,
        "diagnostics": result.diagnostics.as_dict(),
        "run_kind": manifest.get("run_kind", run_kind),
        "convergence_claim": False,
        "effective_sampler_settings": asdict(settings),
        "products": products,
    }
    write_json(run_dir / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["actual_chain"] = {
        "chains": int(result.chain.shape[1]),
        "parameters": int(result.chain.shape[2]),
        "posterior_draws_per_chain": int(result.chain.shape[0]),
    }
    manifest["artifacts"] = sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
    )
    write_manifest(run_dir / "manifest.json", manifest)
    return run_dir
