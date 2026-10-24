from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from passband_fit.config import load_config
from passband_fit.io import load_curve, load_site_mapping
from passband_fit.hierarchical_model import HierarchicalPosterior
from passband_fit.mock_data import make_mock_problem
from passband_fit.numpyro_model import JaxPosteriorData, build_potential_fn
from passband_fit.pipeline import (
    build_posterior,
    fit_baseline,
    posterior_boundary_diagnostics,
    prepare_fit,
)


@pytest.mark.parametrize(
    "mode", ["fixed", "shift_only", "shift_width", "shift_width_warp", "shift_width_leak"]
)
@pytest.mark.parametrize("magnitude_system", ["vega", "ab"])
def test_mock_modes_and_magnitude_systems_are_finite(mode, magnitude_system):
    problem = make_mock_problem(mode, magnitude_system, seed=8)
    posterior = HierarchicalPosterior(
        problem.layout,
        problem.instrument_index,
        problem.observed_magnitude,
        problem.magnitude_uncertainty,
        problem.evaluator,
        target_index=problem.target_index,
    )
    assert np.isfinite(posterior(problem.theta_truth))
    assert np.all(np.isfinite(problem.evaluator(problem.truth)))
    assert problem.layout.unpack(problem.theta_truth)["width"].shape == (2,)


def test_mock_truth_predicts_better_than_wrong_shift():
    problem = make_mock_problem("shift_only", "vega", seed=22)
    truth_prediction = problem.evaluator(problem.truth) + np.asarray(
        problem.truth["zeropoint"]
    )[problem.instrument_index] + np.asarray(problem.truth["star_offset"])[
        problem.target_index
    ]
    wrong = dict(problem.truth)
    wrong["delta_lambda"] = -np.asarray(problem.truth["delta_lambda"])
    wrong_prediction = problem.evaluator(wrong) + np.asarray(
        problem.truth["zeropoint"]
    )[problem.instrument_index] + np.asarray(problem.truth["star_offset"])[
        problem.target_index
    ]
    truth_rms = np.sqrt(np.mean((problem.observed_magnitude - truth_prediction) ** 2))
    wrong_rms = np.sqrt(np.mean((problem.observed_magnitude - wrong_prediction) ** 2))
    assert truth_rms < wrong_rms


def test_real_gp_preparation_and_baseline():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "example_config.yaml", selected_filter="gp")
    prepared = prepare_fit(config)
    layout, posterior = build_posterior(prepared)
    baseline = fit_baseline(prepared)
    assert len(prepared.selected.row_id) == prepared.selected.summary["final_used"]
    assert len(prepared.selected.instrument_names) >= 1
    assert len(prepared.selected.target_names) >= 1
    assert prepared.grid_convergence_max_delta_mag < 5e-4
    assert np.isfinite(baseline["maximum_log_likelihood"])
    assert np.isfinite(posterior(layout.initial_vector(posterior.priors)))
    prefix_to_site = load_site_mapping(config.inputs.site_mapping)
    atmosphere_by_site = dict(
        zip(
            prepared.atmosphere_site_names,
            prepared.unit_airmass_transmission_by_site,
        )
    )
    for row, (prefix, airmass) in enumerate(
        zip(prepared.selected.sites, prepared.selected.airmass)
    ):
        expected = atmosphere_by_site[prefix_to_site[prefix]] ** airmass
        np.testing.assert_allclose(prepared.atmosphere_by_observation[row], expected)


@pytest.mark.parametrize("mode", ["fixed", "shift_only"])
def test_real_gp_jax_potential_matches_full_numpy_likelihood(mode):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "example_config.yaml", selected_filter="gp")
    config = replace(config, model=replace(config.model, passband_mode=mode))
    prepared = prepare_fit(config)
    layout, posterior = build_posterior(prepared)
    data = JaxPosteriorData(
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
    potential = build_potential_fn(data, layout, posterior.priors)
    theta = layout.initial_vector(posterior.priors)
    assert float(-potential(theta)) == pytest.approx(posterior(theta), abs=2.0e-10)


def test_fixed_boundary_diagnostics_omit_shape_parameters():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "I.yaml", selected_filter="I")
    config = replace(config, model=replace(config.model, passband_mode="fixed"))
    problem = make_mock_problem("fixed", "vega", seed=6)
    diagnostic = posterior_boundary_diagnostics(
        np.repeat(problem.theta_truth[None, :], 3, axis=0),
        problem.layout,
        config,
    )
    assert diagnostic["parameters"] == {}
    assert diagnostic["warning"] is False


def test_rejects_atmosphere_with_non_negligible_blue_endpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "I_2.yaml", selected_filter="I")
    site_name = "CTIO"
    curve_config = config.inputs.atmospheres[site_name]
    raw_curve = load_curve(curve_config, f"{site_name} atmosphere")
    truncated_path = tmp_path / "truncated_atmosphere.txt"
    keep = raw_curve.wavelength_angstrom >= 3300.0
    np.savetxt(
        truncated_path,
        np.column_stack((raw_curve.wavelength_angstrom[keep], raw_curve.values[keep])),
    )
    atmosphere = replace(
        curve_config,
        path=truncated_path,
        wavelength_unit="angstrom",
    )
    atmospheres = dict(config.inputs.atmospheres)
    atmospheres[site_name] = atmosphere
    config = replace(
        config,
        inputs=replace(config.inputs, atmospheres=atmospheres),
    )
    with pytest.raises(ValueError, match="atmosphere blue-edge coverage failure"):
        prepare_fit(config)


def test_i_rejects_qe_table_ending_at_non_negligible_response(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "I.yaml", selected_filter="I")
    raw = np.loadtxt(config.inputs.ccd.path)
    truncated_path = tmp_path / "truncated_qe.txt"
    np.savetxt(truncated_path, raw[raw[:, 0] <= 9000.0])
    ccd = replace(config.inputs.ccd, path=truncated_path)
    config = replace(config, inputs=replace(config.inputs, ccd=ccd))
    with pytest.raises(ValueError, match="CCD QE red-edge coverage failure"):
        prepare_fit(config)


def test_filter_remapping_keeps_ccd_qe_cutoff_fixed():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "I_2.yaml", selected_filter="I")
    prepared = prepare_fit(config)
    parameters = {
        "delta_lambda": np.full(len(prepared.selected.instrument_names), 500.0),
        "width": np.full(len(prepared.selected.instrument_names), 1.2),
        "warp": np.zeros(len(prepared.selected.instrument_names)),
        "zeropoint": np.zeros(len(prepared.selected.instrument_names)),
        "sigma_extra": 0.0,
    }
    curves = prepared.instrument_curves(parameters)
    assert np.all(curves[:, prepared.quantum_efficiency == 0.0] == 0.0)


def test_boundary_diagnostics_reports_near_shift_guard():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "I.yaml", selected_filter="I")
    problem = make_mock_problem("shift_only", "vega", seed=4)
    theta = problem.theta_truth.copy()
    mu, log_tau, z_slice = problem.layout.family_indices("delta_lambda")
    theta[mu] = 950.0
    theta[log_tau] = np.log(1.0)
    theta[z_slice] = 0.0
    diagnostic = posterior_boundary_diagnostics(
        np.repeat(theta[None, :], 3, axis=0), problem.layout, config
    )
    shift = diagnostic["parameters"]["delta_lambda_angstrom"]
    assert shift["fraction_in_outer_10_percent"] == pytest.approx(1.0)
    assert diagnostic["warning"] is True
