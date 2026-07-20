"""Small plotting primitives for required science and MCMC diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


def _finish_figure(
    figure: plt.Figure,
    output_path: Optional[Union[Path, str]],
    tight_rect: Optional[Sequence[float]] = None,
) -> plt.Figure:
    figure.tight_layout(rect=tight_rect)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
    return figure


def plot_trace(
    chain: np.ndarray,
    parameter_names: Sequence[str],
    output_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot every chain trace for every sampled parameter."""

    values = np.asarray(chain, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(parameter_names):
        raise ValueError("chain must have shape (draws, chains, len(parameter_names))")
    figure, axes = plt.subplots(
        values.shape[2], 1, figsize=(10, max(2.0, 1.6 * values.shape[2])), sharex=True,
        squeeze=False,
    )
    for index, name in enumerate(parameter_names):
        axes[index, 0].plot(values[:, :, index], color="black", alpha=0.18, linewidth=0.5)
        axes[index, 0].set_ylabel(name)
    axes[-1, 0].set_xlabel("posterior draw")
    return _finish_figure(figure, output_path)


def plot_log_probability_trace(
    log_probability: np.ndarray, output_path: Optional[Union[Path, str]] = None
) -> plt.Figure:
    """Plot the log-posterior trajectory for all chains."""

    values = np.asarray(log_probability, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("log_probability must have shape (draws, chains)")
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(values, color="black", alpha=0.2, linewidth=0.6)
    axis.set(xlabel="ensemble step", ylabel="log posterior")
    return _finish_figure(figure, output_path)


def plot_acceptance_fractions(
    acceptance_fraction: Sequence[float],
    output_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot per-chain mean acceptance probabilities and their median."""

    values = np.asarray(acceptance_fraction, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("acceptance_fraction must be a nonempty finite vector")
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(np.arange(len(values)), values, color="tab:blue")
    axis.axhline(np.median(values), color="tab:red", linestyle="--", label="median")
    axis.set(xlabel="chain", ylabel="mean acceptance probability", ylim=(0, 1))
    axis.legend()
    return _finish_figure(figure, output_path)


def _grouped_scatter(
    axis: plt.Axes,
    x: np.ndarray,
    residual: np.ndarray,
    groups: Optional[np.ndarray],
    group_names: Optional[Sequence[str]],
) -> None:
    if groups is None:
        axis.scatter(x, residual, s=18, alpha=0.75)
        return
    unique = np.unique(groups)
    for group in unique:
        selected = groups == group
        label = str(group)
        if group_names is not None:
            integer = int(group)
            if integer < 0 or integer >= len(group_names):
                raise ValueError("group index is outside group_names")
            label = str(group_names[integer])
        axis.scatter(x[selected], residual[selected], s=18, alpha=0.75, label=label)
    if len(unique) <= 15:
        axis.legend(fontsize="small", ncol=2)


def plot_residual_comparison(
    x_original: Sequence[float],
    residual_original: Sequence[float],
    x_fitted: Sequence[float],
    residual_fitted: Sequence[float],
    groups: Optional[Sequence[int]] = None,
    group_names: Optional[Sequence[str]] = None,
    x_label: str = "photon-weighted mean wavelength (Angstrom)",
    output_path: Optional[Union[Path, str]] = None,
    intermediate_x: Optional[Sequence[float]] = None,
    intermediate_residual: Optional[Sequence[float]] = None,
    panel_titles: Optional[Sequence[str]] = None,
    filter_name: Optional[str] = None,
    panel_annotations: Optional[Sequence[Optional[str]]] = None,
) -> plt.Figure:
    """Plot two or three stages of observed-minus-model residuals."""

    arrays = [np.asarray(item, dtype=np.float64) for item in (
        x_original, residual_original, x_fitted, residual_fitted
    )]
    has_intermediate = intermediate_x is not None or intermediate_residual is not None
    if has_intermediate:
        if intermediate_x is None or intermediate_residual is None:
            raise ValueError("intermediate_x and intermediate_residual must be supplied together")
        arrays.extend(
            (
                np.asarray(intermediate_x, dtype=np.float64),
                np.asarray(intermediate_residual, dtype=np.float64),
            )
        )
    lengths = {len(item) for item in arrays}
    if len(lengths) != 1 or any(item.ndim != 1 for item in arrays):
        raise ValueError("residual comparison arrays must be equal-length vectors")
    group_array = None if groups is None else np.asarray(groups)
    if group_array is not None and group_array.shape != arrays[0].shape:
        raise ValueError("groups must have one entry per residual")
    if has_intermediate:
        panels = ((arrays[0], arrays[1]), (arrays[4], arrays[5]), (arrays[2], arrays[3]))
        default_titles = (
            "Nominal passband",
            "Fitted before star offset",
            "Full fitted model",
        )
    else:
        panels = ((arrays[0], arrays[1]), (arrays[2], arrays[3]))
        default_titles = ("Nominal passband", "Fitted passband")
    titles = tuple(panel_titles) if panel_titles is not None else default_titles
    if len(titles) != len(panels):
        raise ValueError("panel_titles must contain one title per residual panel")
    annotations = (
        tuple(panel_annotations)
        if panel_annotations is not None
        else (None,) * len(panels)
    )
    if len(annotations) != len(panels):
        raise ValueError("panel_annotations must contain one entry per residual panel")
    annotation_line_count = max(
        (annotation.count("\n") + 1 for annotation in annotations if annotation),
        default=0,
    )
    figure_height = 4.0 + 0.45 * max(0, annotation_line_count - 1)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6 * len(panels), figure_height),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for axis, (x_values, residual_values), title, annotation in zip(
        axes, panels, titles, annotations
    ):
        _grouped_scatter(axis, x_values, residual_values, group_array, group_names)
        axis.axhline(0.0, color="black", linewidth=0.8)
        if filter_name is not None:
            rms_mmag = 1000.0 * np.sqrt(np.mean(residual_values**2))
            title = f"{filter_name} — {title}\nRMS = {rms_mmag:.1f} mmag"
        if annotation:
            title = f"{title}\n{annotation}"
        axis.set(xlabel=x_label, title=title)
    axes[0].set_ylabel("observed - model magnitude")
    # Reserve a small explicit top margin because multi-line axes titles can
    # otherwise be clipped by some Matplotlib versions when saved tightly.
    return _finish_figure(figure, output_path, tight_rect=(0.0, 0.0, 1.0, 0.94))


def plot_passband(
    wavelength_angstrom: Sequence[float],
    nominal: Sequence[float],
    posterior_median: Sequence[float],
    credible_low: Sequence[float],
    credible_high: Sequence[float],
    effective_curves: Optional[Mapping[str, Sequence[float]]] = None,
    title: Optional[str] = None,
    output_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Plot nominal, fitted median/68% interval, and optional effective curves."""

    wave = np.asarray(wavelength_angstrom, dtype=np.float64)
    curves = [np.asarray(value, dtype=np.float64) for value in (
        nominal, posterior_median, credible_low, credible_high
    )]
    if wave.ndim != 1 or any(item.shape != wave.shape for item in curves):
        raise ValueError("all passband curves must match the one-dimensional wavelength grid")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(wave, curves[0], linestyle="--", color="black", label="nominal")
    axis.plot(wave, curves[1], color="tab:blue", label="posterior median")
    axis.fill_between(wave, curves[2], curves[3], color="tab:blue", alpha=0.25, label="68% interval")
    for label, curve in (effective_curves or {}).items():
        values = np.asarray(curve, dtype=np.float64)
        if values.shape != wave.shape:
            raise ValueError("effective passband curves must match wavelength grid")
        axis.plot(wave, values, alpha=0.75, label=str(label))
    axis.set(xlabel="wavelength (Angstrom)", ylabel="relative throughput", title=title)
    axis.legend()
    return _finish_figure(figure, output_path)


def plot_passband_ensemble(
    wavelength_angstrom: Sequence[float],
    median_curves: np.ndarray,
    instrument_names: Sequence[str],
    nominal: Optional[Sequence[float]] = None,
    output_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Compare posterior median instrumental passbands on one axis."""

    wave = np.asarray(wavelength_angstrom, dtype=np.float64)
    curves = np.asarray(median_curves, dtype=np.float64)
    if curves.shape != (len(instrument_names), len(wave)):
        raise ValueError("median_curves must have shape (instruments, wavelengths)")
    figure, axis = plt.subplots(figsize=(10, 5))
    if nominal is not None:
        nominal_values = np.asarray(nominal, dtype=np.float64)
        if nominal_values.shape != wave.shape:
            raise ValueError("nominal must match wavelength grid")
        axis.plot(
            wave,
            nominal_values,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="nominal",
            zorder=10,
        )
    for curve, name in zip(curves, instrument_names):
        axis.plot(wave, curve, alpha=0.8, label=str(name), zorder=2)
    axis.set(xlabel="wavelength (Angstrom)", ylabel="relative throughput")
    axis.legend(fontsize="small", ncol=2)
    return _finish_figure(figure, output_path)


def plot_forest(
    instrument_names: Sequence[str],
    median: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    population_mean: Optional[float] = None,
    population_dispersion: Optional[float] = None,
    parameter_label: str = "parameter",
    output_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Forest plot for instrument effects and an inferred population interval."""

    center, low, high = (np.asarray(item, dtype=np.float64) for item in (median, lower, upper))
    expected = (len(instrument_names),)
    if center.shape != expected or low.shape != expected or high.shape != expected:
        raise ValueError("forest arrays must contain one value per instrument")
    if np.any(low > center) or np.any(high < center):
        raise ValueError("credible limits must bracket medians")
    y = np.arange(len(instrument_names))
    figure, axis = plt.subplots(figsize=(8, max(4, 0.42 * len(y))))
    axis.errorbar(center, y, xerr=(center - low, high - center), fmt="o", color="tab:blue")
    if population_mean is not None:
        axis.axvline(population_mean, color="tab:red", label="population mean")
        if population_dispersion is not None:
            axis.axvspan(
                population_mean - population_dispersion,
                population_mean + population_dispersion,
                color="tab:red",
                alpha=0.15,
                label="population mean +/- dispersion",
            )
        axis.legend()
    axis.set(yticks=y, yticklabels=list(instrument_names), xlabel=parameter_label)
    axis.invert_yaxis()
    return _finish_figure(figure, output_path)


def plot_corner(
    samples: np.ndarray,
    parameter_names: Sequence[str],
    subset: Optional[Sequence[int]] = None,
    truths: Optional[Sequence[float]] = None,
    output_path: Optional[Union[Path, str]] = None,
) -> plt.Figure:
    """Create a corner plot with uncertainty-scaled numerical precision.

    Title resolution is chosen near three percent of the smaller 16--84
    percentile uncertainty.  This retains useful significant figures and
    prevents finite uncertainties from being displayed as ``0.00``.
    """

    import corner

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(parameter_names):
        raise ValueError("samples must have shape (draws, len(parameter_names))")
    indices = np.arange(values.shape[1]) if subset is None else np.asarray(subset, dtype=int)
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= values.shape[1]):
        raise ValueError("subset contains invalid parameter indices")
    labels = [parameter_names[index] for index in indices]
    selected_truths = None
    if truths is not None:
        truth_values = np.asarray(truths, dtype=np.float64)
        if truth_values.shape != (values.shape[1],):
            raise ValueError("truths must contain one value per parameter")
        selected_truths = truth_values[indices]
    selected_values = values[:, indices]
    titles = [
        _corner_quantile_title(label, selected_values[:, column])
        for column, label in enumerate(labels)
    ]
    figure = corner.corner(
        selected_values,
        labels=labels,
        titles=titles,
        truths=selected_truths,
        show_titles=True,
        title_fmt=None,
    )
    return _finish_figure(figure, output_path)


def _corner_quantile_title(
    label: str,
    samples: Sequence[float],
    precision_fraction: float = 0.03,
) -> str:
    """Format median and asymmetric errors at uncertainty-aware precision."""

    return format_quantile_summary(label, samples, precision_fraction)


def format_quantile_summary(
    label: str,
    samples: Sequence[float],
    precision_fraction: float = 0.03,
    unit: Optional[str] = None,
) -> str:
    """Format a median and asymmetric 16--84% uncertainty for plot text."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("corner-title samples must be a nonempty finite vector")
    if not np.isfinite(precision_fraction) or not 0.0 < precision_fraction < 1.0:
        raise ValueError("precision_fraction must lie between zero and one")
    low, median, high = np.percentile(values, [16.0, 50.0, 84.0])
    minus = float(median - low)
    plus = float(high - median)
    positive_errors = [error for error in (minus, plus) if error > 0.0]
    if positive_errors:
        uncertainty_scale = min(positive_errors)
    else:
        uncertainty_scale = max(float(np.std(values)), np.finfo(float).eps)
    target_resolution = precision_fraction * uncertainty_scale
    resolution_exponent = int(np.round(np.log10(target_resolution)))
    resolution = 10.0**resolution_exponent

    def formatted(value: float) -> str:
        rounded = np.round(value / resolution) * resolution
        if resolution_exponent < 0 and -resolution_exponent <= 12:
            return f"{rounded:.{-resolution_exponent}f}"
        if resolution_exponent >= 0:
            return f"{rounded:.0f}"
        return f"{rounded:.6e}"

    result = (
        f"{label} = "
        rf"${{{formatted(float(median))}}}_{{-{formatted(minus)}}}"
        rf"^{{+{formatted(plus)}}}$"
    )
    if unit:
        result += f" {unit}"
    return result


def format_zeropoint_conventions(
    mu_z_samples: Sequence[float],
    configured_system: str,
    ab_minus_vega_magnitude: float,
) -> str:
    """Format the additive mean offset in Vega-zero and AB conventions."""

    samples = np.asarray(mu_z_samples, dtype=np.float64)
    if samples.ndim != 1 or len(samples) == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("mu_z_samples must be a nonempty finite vector")
    if not np.isfinite(ab_minus_vega_magnitude):
        raise ValueError("AB-minus-Vega magnitude must be finite")
    system = configured_system.strip().lower()
    if system == "vega":
        vega_samples = samples
        ab_samples = samples - ab_minus_vega_magnitude
    elif system == "ab":
        ab_samples = samples
        vega_samples = samples + ab_minus_vega_magnitude
    else:
        raise ValueError("configured_system must be 'vega' or 'ab'")
    vega_text = format_quantile_summary(
        "mean zeropoint μ_Z (Vega=0)", vega_samples, unit="mag"
    )
    ab_text = format_quantile_summary(
        "mean zeropoint μ_Z (AB)", ab_samples, unit="mag"
    )
    return f"{vega_text}\n{ab_text}"
