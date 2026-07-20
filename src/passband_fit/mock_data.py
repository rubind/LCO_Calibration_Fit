"""Small deterministic mock problems for integration and recovery tests."""

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple

import numpy as np

from .curves import atmosphere_at_airmass, trapezoid_weights
from .hierarchical_model import ModelPriors, ParameterLayout
from .passband_model import instrument_passband, observation_passband
from .synthetic_photometry import synthetic_magnitude


@dataclass(frozen=True)
class MockProblem:
    """An in-memory hierarchical inference fixture with known parameters."""

    layout: ParameterLayout
    theta_truth: np.ndarray
    instrument_index: np.ndarray
    target_index: np.ndarray
    airmass: np.ndarray
    observed_magnitude: np.ndarray
    magnitude_uncertainty: np.ndarray
    evaluator: Callable[[Mapping[str, object]], np.ndarray]
    truth: Mapping[str, object]


def _set_family(
    theta: np.ndarray, layout: ParameterLayout, family: str, values: np.ndarray, tau: float
) -> None:
    mu_index, log_tau_index, z_slice = layout.family_indices(family)
    mu = float(np.mean(values))
    theta[mu_index] = mu
    theta[log_tau_index] = np.log(tau)
    theta[z_slice] = (values - mu) / tau


def make_mock_problem(
    mode: str = "shift_only",
    magnitude_system: str = "vega",
    n_instruments: int = 2,
    n_targets: int = 4,
    seed: int = 1729,
) -> MockProblem:
    """Create a compact fixed-seed problem covering color and airmass leverage."""

    if magnitude_system not in {"vega", "ab"}:
        raise ValueError("magnitude_system must be 'vega' or 'ab'")
    if n_instruments < 2 or n_targets < 2:
        raise ValueError("mock problem requires at least two instruments and targets")
    rng = np.random.default_rng(seed)
    wave = np.arange(3800.0, 7200.1, 20.0)
    weights = trapezoid_weights(wave)
    nominal = np.exp(-0.5 * ((wave - 5400.0) / 520.0) ** 2)
    t1 = np.exp(-0.025 * (wave / 5500.0) ** -1.2)
    slopes = np.linspace(-3.2, 1.0, n_targets)
    spectra = np.asarray([(wave / 5500.0) ** slope for slope in slopes])
    vega = (wave / 5500.0) ** -2.0

    combinations = [
        (instrument, target, airmass)
        for instrument in range(n_instruments)
        for target in range(n_targets)
        for airmass in (1.0, 1.6)
    ]
    instrument_index = np.asarray([item[0] for item in combinations], dtype=int)
    target_index = np.asarray([item[1] for item in combinations], dtype=int)
    airmass = np.asarray([item[2] for item in combinations], dtype=float)
    layout = ParameterLayout(
        mode,
        n_instruments,
        tuple(f"mock-{index}" for index in range(n_instruments)),
        n_targets=n_targets,
        target_names=tuple(f"star-{index}" for index in range(n_targets)),
    )

    def evaluator(parameters: Mapping[str, object]) -> np.ndarray:
        shifts = np.asarray(parameters["delta_lambda"], dtype=float)
        widths = np.asarray(parameters["width"], dtype=float)
        warps = np.asarray(parameters["warp"], dtype=float)
        instrument_curves = [
            instrument_passband(
                wave,
                nominal,
                pivot_wavelength_angstrom=5400.0,
                delta_wavelength_angstrom=shifts[index],
                width_scale=widths[index],
                warp_parameter=warps[index],
                warp_scale_angstrom=1100.0,
            )
            for index in range(n_instruments)
        ]
        result = np.empty(len(combinations), dtype=float)
        for row, (instrument, target, x_value) in enumerate(combinations):
            effective = observation_passband(
                instrument_curves[instrument], t1, x_value
            )
            result[row] = synthetic_magnitude(
                wave,
                spectra[target],
                effective,
                magnitude_system=magnitude_system,
                vega_flux_lambda=vega if magnitude_system == "vega" else None,
                weights=weights,
            )
        return result

    theta = layout.initial_vector(ModelPriors())
    shifts = np.linspace(-18.0, 18.0, n_instruments)
    _set_family(theta, layout, "delta_lambda", shifts, 18.0)
    widths = np.linspace(0.985, 1.015, n_instruments)
    if "eta" in layout.enabled_families:
        _set_family(theta, layout, "eta", np.log(widths), 0.02)
    warps = np.linspace(-0.08, 0.08, n_instruments)
    if "a" in layout.enabled_families:
        _set_family(theta, layout, "a", warps, 0.08)
    zeropoints = np.linspace(-0.12, 0.12, n_instruments)
    _set_family(theta, layout, "Z", zeropoints, 0.12)
    log_sigma_star_index, _ = layout.star_indices()
    theta[log_sigma_star_index] = np.log(0.008)
    theta[-1] = np.log(0.012)
    physical = layout.unpack(theta)
    star_offsets = 0.008 * np.linspace(-1.0, 1.0, n_targets)
    measurement_sigma = np.full(len(combinations), 0.01)
    prediction = (
        evaluator(physical)
        + zeropoints[instrument_index]
        + star_offsets[target_index]
    )
    observed = prediction + rng.normal(
        scale=np.sqrt(measurement_sigma**2 + float(physical["sigma_extra"]) ** 2)
    )
    truth: Dict[str, object] = dict(physical)
    truth["star_offset"] = star_offsets
    truth["prediction"] = prediction
    return MockProblem(
        layout=layout,
        theta_truth=theta,
        instrument_index=instrument_index,
        target_index=target_index,
        airmass=airmass,
        observed_magnitude=observed,
        magnitude_uncertainty=measurement_sigma,
        evaluator=evaluator,
        truth=truth,
    )
