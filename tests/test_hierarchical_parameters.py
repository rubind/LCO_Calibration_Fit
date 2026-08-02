import numpy as np
import pytest
from scipy.stats import halfnorm, multivariate_normal
from types import SimpleNamespace

from passband_fit.hierarchical_model import (
    HierarchicalPosterior,
    ModelPriors,
    ParameterLayout,
    log_prior,
)


@pytest.mark.parametrize(
    "mode,n_families",
    [
        ("fixed", 1),
        ("shift_only", 2),
        ("shift_width", 3),
        ("shift_width_warp", 4),
    ],
)
def test_layout_dimension_and_stable_names(mode, n_families):
    layout = ParameterLayout(mode, 3, ("a", "b", "c"))
    assert layout.ndim == n_families * 5 + 1
    expected_first_family = "Z" if mode == "fixed" else "delta_lambda"
    assert layout.parameter_names[0:2] == (
        f"mu_{expected_first_family}",
        f"log_tau_{expected_first_family}",
    )
    assert layout.parameter_names[-1] == "log_sigma_extra"
    assert len(layout.parameter_names) == len(set(layout.parameter_names))


def test_noncentered_transform_and_disabled_parameters():
    layout = ParameterLayout("shift_only", 2, ("one", "two"))
    theta = layout.initial_vector()
    mu_index, log_tau_index, z_slice = layout.family_indices("delta_lambda")
    theta[mu_index] = 4.0
    theta[log_tau_index] = np.log(2.0)
    theta[z_slice] = (-1.0, 1.5)
    z_mu, z_log_tau, z_z_slice = layout.family_indices("Z")
    theta[z_mu] = 0.2
    theta[z_log_tau] = np.log(0.1)
    theta[z_z_slice] = (1.0, -2.0)
    theta[-1] = np.log(0.03)

    parameters = layout.unpack(theta)
    np.testing.assert_allclose(parameters["delta_lambda"], [2.0, 7.0])
    np.testing.assert_allclose(parameters["zeropoint"], [0.3, 0.0], atol=1.0e-15)
    np.testing.assert_allclose(parameters["width"], 1.0)
    np.testing.assert_allclose(parameters["warp"], 0.0)
    assert parameters["sigma_extra"] == pytest.approx(0.03)


def test_fixed_mode_has_only_zeropoint_hierarchy_and_scatter_parameters():
    layout = ParameterLayout(
        "fixed",
        2,
        ("one", "two"),
        n_targets=1,
        target_names=("star",),
    )
    assert layout.enabled_families == ("Z",)
    assert layout.parameter_names == (
        "mu_Z",
        "log_tau_Z",
        "z_Z[one]",
        "z_Z[two]",
        "log_sigma_star",
        "log_sigma_extra",
    )
    parameters = layout.unpack(layout.initial_vector())
    np.testing.assert_array_equal(parameters["delta_lambda"], [0.0, 0.0])
    np.testing.assert_array_equal(parameters["width"], [1.0, 1.0])
    np.testing.assert_array_equal(parameters["warp"], [0.0, 0.0])


def test_width_derived_vector_reports_eta_and_physical_width():
    layout = ParameterLayout("shift_width", 1, ("camera",))
    theta = layout.initial_vector()
    mu, log_tau, z = layout.family_indices("eta")
    theta[mu] = np.log(1.1)
    theta[log_tau] = np.log(0.02)
    theta[z] = 0.0
    values = layout.derived_vector(theta)
    paired = dict(zip(layout.derived_parameter_names, values))
    assert paired["eta[camera]"] == pytest.approx(np.log(1.1))
    assert paired["width[camera]"] == pytest.approx(1.1)


def test_log_halfnormal_prior_includes_log_coordinate_jacobian():
    layout = ParameterLayout("shift_only", 1)
    priors = ModelPriors()
    first = layout.initial_vector(priors)
    second = first.copy()
    _, scale_index, _ = layout.family_indices("delta_lambda")
    first[scale_index] = np.log(5.0)
    second[scale_index] = np.log(10.0)

    actual_difference = log_prior(second, layout, priors) - log_prior(first, layout, priors)
    expected_difference = (
        halfnorm.logpdf(10.0, scale=priors.tau_delta_lambda.scale)
        + np.log(10.0)
        - halfnorm.logpdf(5.0, scale=priors.tau_delta_lambda.scale)
        - np.log(5.0)
    )
    assert actual_difference == pytest.approx(expected_difference)


def test_prior_adapter_accepts_public_config_names():
    normal = lambda mean, sigma: SimpleNamespace(mean=mean, sigma=sigma)
    half = lambda scale: SimpleNamespace(scale=scale)
    config = SimpleNamespace(
        shift_population_mean=normal(1.0, 2.0),
        shift_population_scale=half(3.0),
        log_width_population_mean=normal(0.0, 0.1),
        log_width_population_scale=half(0.2),
        warp_population_mean=normal(0.0, 0.3),
        warp_population_scale=half(0.4),
        zeropoint_population_mean=normal(0.0, 0.5),
        zeropoint_population_scale=half(0.6),
        star_correlated_scatter=half(0.08),
        extra_scatter=half(0.7),
    )
    priors = ModelPriors.from_mapping(config)
    assert priors.mu_delta_lambda.mean == 1.0
    assert priors.tau_delta_lambda.scale == 3.0
    assert priors.sigma_star.scale == 0.08
    assert priors.sigma_extra.scale == 0.7


def test_marginalized_star_offsets_and_covariance():
    layout = ParameterLayout(
        "shift_only",
        1,
        ("camera",),
        n_targets=2,
        target_names=("alpha", "beta"),
    )
    theta = layout.initial_vector()
    log_sigma_star, z_star = layout.star_indices()
    theta[log_sigma_star] = np.log(0.02)
    assert z_star.start == z_star.stop
    theta[-1] = np.log(0.01)
    target_index = np.array([0, 0, 1])
    posterior = HierarchicalPosterior(
        layout,
        [0, 0, 0],
        [10.03, 11.03, 11.99],
        [0.004, 0.005, 0.006],
        lambda parameters: np.array([10.0, 11.0, 12.0]),
        target_index=target_index,
    )

    base = np.asarray(layout.unpack(theta)["zeropoint"])[0]
    base_prediction = np.array([10.0 + base, 11.0 + base, 12.0 + base])
    conditional_mean, conditional_variance = posterior.conditional_star_distribution(theta)
    variance = np.array([0.004, 0.005, 0.006]) ** 2 + 0.01**2
    residual = np.array([10.03, 11.03, 11.99]) - base_prediction
    for target in range(2):
        selected = target_index == target
        precision_sum = np.sum(1.0 / variance[selected])
        weighted_sum = np.sum(residual[selected] / variance[selected])
        denominator = 1.0 + 0.02**2 * precision_sum
        assert conditional_mean[target] == pytest.approx(0.02**2 * weighted_sum / denominator)
        assert conditional_variance[target] == pytest.approx(0.02**2 / denominator)
    np.testing.assert_allclose(
        posterior.predict(theta), base_prediction + conditional_mean[target_index]
    )
    covariance = posterior.marginal_covariance(theta)
    expected_diagonal = np.array([0.004, 0.005, 0.006]) ** 2 + 0.01**2 + 0.02**2
    np.testing.assert_allclose(np.diag(covariance), expected_diagonal)
    assert covariance[0, 1] == pytest.approx(0.02**2)
    assert covariance[0, 2] == pytest.approx(0.0)
    assert "sigma_star" in layout.derived_parameter_names
    assert "star_offset[alpha]" not in layout.derived_parameter_names

    expected_log_likelihood = multivariate_normal.logpdf(
        np.array([10.03, 11.03, 11.99]),
        mean=base_prediction,
        cov=covariance,
    )
    assert posterior.log_likelihood(theta) == pytest.approx(expected_log_likelihood)


def test_target_indices_are_required_and_validated():
    layout = ParameterLayout("shift_only", 1, n_targets=1)
    args = (layout, [0], [1.0], [0.1], lambda parameters: np.array([1.0]))
    with pytest.raises(ValueError, match="target_index is required"):
        HierarchicalPosterior(*args)
    with pytest.raises(ValueError, match="outside the parameter layout"):
        HierarchicalPosterior(*args, target_index=[1])


def test_full_normalized_likelihood_and_zeropoint_sign():
    layout = ParameterLayout("shift_only", 2)
    theta = layout.initial_vector()
    _, _, z_slice = layout.family_indices("Z")
    # Set mu_Z=0, tau_Z=0.1, yielding +0.1 and -0.2 mag offsets.
    mu_index, log_tau_index, _ = layout.family_indices("Z")
    theta[mu_index] = 0.0
    theta[log_tau_index] = np.log(0.1)
    theta[z_slice] = (1.0, -2.0)
    theta[-1] = np.log(0.05)
    instrument = np.array([0, 1, 0])
    synthetic = np.array([10.0, 11.0, 12.0])
    observed = synthetic + np.array([0.1, -0.2, 0.1]) + np.array([0.02, -0.01, 0.03])
    uncertainty = np.array([0.04, 0.04, 0.06])
    posterior = HierarchicalPosterior(
        layout,
        instrument,
        observed,
        uncertainty,
        lambda parameters: synthetic,
    )
    predicted = posterior.predict(theta)
    np.testing.assert_allclose(predicted, synthetic + np.array([0.1, -0.2, 0.1]))
    residual = observed - predicted
    variance = uncertainty**2 + 0.05**2
    expected = -0.5 * np.sum(residual**2 / variance + np.log(2 * np.pi * variance))
    assert posterior.log_likelihood(theta) == pytest.approx(expected)


def test_invalid_evaluator_output_returns_negative_infinity():
    layout = ParameterLayout("shift_only", 1)
    posterior = HierarchicalPosterior(
        layout,
        [0],
        [1.0],
        [0.1],
        lambda parameters: np.array([np.nan]),
    )
    assert posterior(layout.initial_vector()) == -np.inf


def test_layout_rejects_unknown_mode_and_wrong_shape():
    with pytest.raises(ValueError):
        ParameterLayout("unknown", 2)
    layout = ParameterLayout("shift_only", 2)
    with pytest.raises(ValueError):
        layout.unpack(np.zeros(layout.ndim + 1))
