import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from passband_fit.cli import _stored_effective_atmospheres
from passband_fit.diagnostics import (
    binned_residual_trend,
    information_criteria,
    parameter_summary,
    posterior_matrices,
    posterior_predictive_draws,
    posterior_predictive_p_values,
    residual_metrics,
)
from passband_fit.plotting import (
    _corner_quantile_title,
    corner_parameter_label,
    fitted_passband_panel_title,
    format_quantile_summary,
    format_zeropoint_conventions,
    passband_ensemble_filename,
    plot_acceptance_fractions,
    plot_corner,
    plot_passband,
    plot_passband_ensemble,
    plot_residual_comparison,
    plot_trace,
    residuals_by_target_filename,
)


def test_parameter_summary_and_matrices():
    samples = np.array([[1.0, 5.0], [2.0, 7.0], [3.0, 9.0]])
    summary = parameter_summary(samples, ["a", "b"], [-3.0, -1.0, -2.0])
    assert summary.loc["a", "median"] == pytest.approx(2.0)
    assert summary.loc["a", "maximum_posterior"] == pytest.approx(2.0)
    covariance, correlation = posterior_matrices(samples, ["a", "b"])
    assert covariance.shape == (2, 2)
    assert correlation.loc["a", "b"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("filter_name", "mode", "expected"),
    [
        ("B", "fixed", "passbands_all_instruments_B_fixed.png"),
        ("B", "shift_only", "passbands_all_instruments_B_shift_only.png"),
        ("rp", "shift_width", "passbands_all_instruments_rp_shift_width.png"),
        (
            "z s",
            "shift_width_warp",
            "passbands_all_instruments_z-s_shift_width_warp.png",
        ),
    ],
)
def test_passband_ensemble_filename_includes_filter_and_model(
    filter_name, mode, expected
):
    assert passband_ensemble_filename(filter_name, mode) == expected


@pytest.mark.parametrize(
    ("filter_name", "mode", "expected"),
    [
        ("B", "fixed", "residuals_by_target_B_fixed.png"),
        ("V", "shift_only", "residuals_by_target_V_shift.png"),
        ("rp", "shift_width", "residuals_by_target_rp_shift_width.png"),
        (
            "z s",
            "shift_width_warp",
            "residuals_by_target_z-s_shift_width_warp.png",
        ),
    ],
)
def test_residuals_by_target_filename_includes_filter_and_model(
    filter_name, mode, expected
):
    assert residuals_by_target_filename(filter_name, mode) == expected


def test_residual_metrics_and_information_criteria():
    residual = np.array([1.0, -1.0, 2.0])
    result = residual_metrics(residual, [1.0, 2.0, 1.0])
    assert result["n"] == 3
    assert result["rms"] == pytest.approx(np.sqrt(2.0))
    assert result["weighted_rms"] != result["rms"]
    criteria = information_criteria(-10.0, 3, 100)
    assert criteria["aic"] == pytest.approx(26.0)
    assert criteria["bic"] == pytest.approx(3 * np.log(100) + 20)


def test_posterior_predictive_helpers_are_seeded_and_dimensioned():
    means = np.zeros((100, 4))
    measurement = np.repeat(0.1, 4)
    extra = np.repeat(0.2, 100)
    first = posterior_predictive_draws(means, measurement, extra, random_seed=22)
    second = posterior_predictive_draws(means, measurement, extra, random_seed=22)
    np.testing.assert_allclose(first, second)
    assert np.std(first) == pytest.approx(np.hypot(0.1, 0.2), rel=0.15)
    p_values = posterior_predictive_p_values(np.zeros(4), first, x=np.arange(4.0))
    assert set(p_values) == {"mean", "rms", "std", "slope"}
    assert all(0 <= value <= 1 for value in p_values.values())


def test_binned_residual_trend():
    result = binned_residual_trend([0, 1, 2, 3], [0.0, 0.2, 0.4, 0.6], bins=[0, 2, 4])
    assert result["n"].tolist() == [2, 2]
    np.testing.assert_allclose(result["mean_residual"], [0.1, 0.5])


def test_plotting_primitives_save_files(tmp_path):
    trace_path = tmp_path / "trace.png"
    residual_path = tmp_path / "residual.png"
    passband_path = tmp_path / "passband.png"
    acceptance_path = tmp_path / "acceptance.png"
    plot_trace(np.zeros((3, 4, 2)), ["a", "b"], trace_path)
    plot_residual_comparison(
        [1, 2], [0.1, -0.1], [1.1, 2.1], [0.05, -0.05], output_path=residual_path
    )
    plot_passband(
        [4000, 5000], [0, 1], [0, 0.9], [0, 0.8], [0, 1.0], output_path=passband_path
    )
    plot_acceptance_fractions([0.2, 0.3, 0.4], acceptance_path)
    for path in (trace_path, residual_path, passband_path, acceptance_path):
        assert path.exists() and path.stat().st_size > 0


def test_corner_titles_scale_precision_with_uncertainty(tmp_path):
    # These quantiles are exactly 0.0277 +/- 0.0023. Three percent of the
    # uncertainty is closest to 1e-4, so four decimal places are retained.
    samples = np.linspace(0.0254, 0.0300, 101)
    title = _corner_quantile_title("mu_Z", samples)
    assert "0.0277" in title
    assert "0.0016" in title
    assert "-0.0000" not in title
    assert "+0.0000" not in title
    output = tmp_path / "corner.png"
    figure = plot_corner(samples[:, None], ["mu_Z"], output_path=output)
    assert "0.0277" in figure.axes[0].get_title()
    assert figure.axes[0].get_xlabel() == (
        "mu_Z\npopulation mean zeropoint [mag]"
    )
    assert output.exists() and output.stat().st_size > 0


@pytest.mark.parametrize(
    ("name", "description"),
    [
        ("mu_delta_lambda", "population mean shift [Å]"),
        ("log_tau_Z", "ln(zeropoint Σ / mag)"),
        (
            "z_eta[lsc1m004-fa03]",
            "standardized instrument ln-width deviation",
        ),
        ("log_sigma_star", "ln(star-residual Σ / mag)"),
    ],
)
def test_corner_parameter_labels_preserve_name_and_explain_meaning(
    name, description
):
    assert corner_parameter_label(name) == f"{name}\n{description}"


def test_corner_parameter_label_leaves_unknown_name_unchanged():
    assert corner_parameter_label("custom_parameter") == "custom_parameter"


def test_zeropoint_convention_annotation_preserves_additive_offset_sign():
    samples = np.linspace(0.0254, 0.0300, 101)
    ab_minus_vega = 0.125

    from_vega = format_zeropoint_conventions(samples, "vega", ab_minus_vega)
    expected_from_vega = (
        format_quantile_summary(
            "mean zeropoint μ_Z (Vega=0)", samples, unit="mag"
        )
        + "\n"
        + format_quantile_summary(
            "mean zeropoint μ_Z (AB)", samples - ab_minus_vega, unit="mag"
        )
    )
    assert from_vega == expected_from_vega

    from_ab = format_zeropoint_conventions(samples, "AB", ab_minus_vega)
    expected_from_ab = (
        format_quantile_summary(
            "mean zeropoint μ_Z (Vega=0)",
            samples + ab_minus_vega,
            unit="mag",
        )
        + "\n"
        + format_quantile_summary(
            "mean zeropoint μ_Z (AB)", samples, unit="mag"
        )
    )
    assert from_ab == expected_from_ab


def test_three_panel_residual_titles_include_filter_and_rms():
    figure = plot_residual_comparison(
        [1, 2],
        [0.1, -0.1],
        [1.1, 2.1],
        [0.01, -0.01],
        intermediate_x=[1.1, 2.1],
        intermediate_residual=[0.05, -0.05],
        filter_name="V",
        panel_annotations=(
            None,
            "mean zeropoint μ_Z = 0.0123 mag",
            "star-offset population dispersion σ_star = 0.0045 mag",
        ),
    )
    assert len(figure.axes) == 3
    assert all("V —" in axis.get_title() for axis in figure.axes)
    assert all("RMS =" in axis.get_title() for axis in figure.axes)
    assert "100.0 mmag" in figure.axes[0].get_title()
    assert "50.0 mmag" in figure.axes[1].get_title()
    assert "10.0 mmag" in figure.axes[2].get_title()
    assert "mean zeropoint μ_Z = 0.0123 mag" in figure.axes[1].get_title()
    assert "mean zeropoint μ_Z" not in figure.axes[2].get_title()
    assert "star-offset population dispersion σ_star" not in figure.axes[1].get_title()
    assert (
        "star-offset population dispersion σ_star = 0.0045 mag"
        in figure.axes[2].get_title()
    )


def test_residual_figure_scale_reduces_both_dimensions():
    regular = plot_residual_comparison(
        [1, 2], [0.1, -0.1], [1, 2], [0.05, -0.05]
    )
    compact = plot_residual_comparison(
        [1, 2],
        [0.1, -0.1],
        [1, 2],
        [0.05, -0.05],
        figure_scale=0.8,
    )
    np.testing.assert_allclose(
        compact.get_size_inches(), 0.8 * regular.get_size_inches()
    )


def test_residual_title_fontsize_can_be_reduced_independently():
    figure = plot_residual_comparison(
        [1, 2],
        [0.1, -0.1],
        [1, 2],
        [0.05, -0.05],
        title_fontsize=9.0,
    )
    assert all(axis.title.get_fontsize() == pytest.approx(9.0) for axis in figure.axes)


@pytest.mark.parametrize(
    ("mode", "label"),
    (
        ("fixed", "fixed (no passband modification)"),
        ("shift_only", "shift"),
        ("shift_width", "shift + width"),
        ("shift_width_warp", "shift + width + warp"),
    ),
)
def test_fitted_passband_panel_title_names_model(mode, label):
    title = fitted_passband_panel_title(mode)
    if mode == "fixed":
        assert title == "Fixed nominal passband + zeropoints,\nbefore star offset"
        return
    assert f"Fitted passband ({label}) + zeropoints" in title
    assert "before star offset" in title


def test_grouped_residuals_can_use_distinct_filled_markers():
    groups = np.array([0, 1, 2])
    figure = plot_residual_comparison(
        [1, 2, 3],
        [0.1, 0.0, -0.1],
        [1, 2, 3],
        [0.05, 0.0, -0.05],
        groups=groups,
        group_names=("star-a", "star-b", "star-c"),
        distinct_group_markers=True,
    )
    collections = figure.axes[0].collections
    marker_paths = {
        tuple(np.round(collection.get_paths()[0].vertices, 6).ravel())
        for collection in collections
    }
    assert len(marker_paths) == 3
    assert all(collection.get_sizes()[0] == pytest.approx(18.0) for collection in collections)


def test_nominal_ensemble_curve_is_drawn_above_instruments():
    figure = plot_passband_ensemble(
        [4000, 5000],
        np.array([[0.1, 0.9], [0.2, 0.8]]),
        ["camera-a", "camera-b"],
        nominal=[0.15, 0.85],
        instrument_atmospheric_transmission=[[0.4, 0.9], [0.5, 0.8]],
        nominal_atmospheric_transmission=[0.45, 0.85],
        atmospheric_airmass=1.3,
        title="V passbands — model: shift + width",
    )
    lines = figure.axes[0].lines
    nominal = next(line for line in lines if line.get_label() == "nominal")
    instruments = [line for line in lines if line.get_label() != "nominal"]
    assert all(nominal.get_zorder() > line.get_zorder() for line in instruments)
    assert figure.axes[0].get_title() == "V passbands — model: shift + width"
    assert len(figure.axes) == 1
    np.testing.assert_allclose(nominal.get_ydata(), [0.15, 0.85] * np.array([0.45, 0.85]))
    np.testing.assert_allclose(instruments[0].get_ydata(), [0.04, 0.81])
    np.testing.assert_allclose(instruments[1].get_ydata(), [0.1, 0.64])
    assert figure.axes[0].get_ylabel() == "effective throughput at X=1.3"


def test_stored_atmospheres_map_instruments_and_average_sites_after_scaling():
    curves = pd.DataFrame(
        {
            "unit_airmass_transmission": [0.1, 0.1],
            "unit_airmass_transmission_CTIO": [0.5, 0.8],
            "unit_airmass_transmission_Sutherland": [0.7, 0.9],
        }
    )
    expected = np.mean(np.array([[0.5, 0.8], [0.7, 0.9]]) ** 1.3, axis=0)
    instruments, mean = _stored_effective_atmospheres(
        curves,
        ["camera-a", "camera-b"],
        {
            "instrument_atmosphere_sites": {
                "camera-a": "CTIO",
                "camera-b": "Sutherland",
            }
        },
        1.3,
    )
    np.testing.assert_allclose(instruments[0], np.array([0.5, 0.8]) ** 1.3)
    np.testing.assert_allclose(instruments[1], np.array([0.7, 0.9]) ** 1.3)
    np.testing.assert_allclose(mean, expected)
