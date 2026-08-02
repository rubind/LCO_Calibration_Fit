"""Small plotting primitives for required science and MCMC diagnostics."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from .passband_model import instrument_passband


FILLED_GROUP_MARKERS = (
    "o",
    "s",
    "^",
    "v",
    "D",
    "P",
    "X",
    "h",
    "p",
    "<",
    ">",
    "8",
    "d",
    "H",
)


def passband_model_label(passband_mode: str) -> str:
    """Return a concise, readable label for a configured passband model."""

    labels = {
        "fixed": "fixed (no passband modification)",
        "shift_only": "shift",
        "shift_width": "shift + width",
        "shift_width_warp": "shift + width + warp",
    }
    try:
        return labels[passband_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported passband mode: {passband_mode!r}") from exc


def passband_ensemble_filename(filter_name: str, passband_mode: str) -> str:
    """Return the filter- and model-specific ensemble-passband PNG name."""

    passband_model_label(passband_mode)  # Validate the canonical mode.
    safe_filter = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(filter_name)).strip(".-")
    if not safe_filter:
        raise ValueError("filter name contains no safe filename characters")
    return f"passbands_all_instruments_{safe_filter}_{passband_mode}.png"


def population_passband_filename(filter_name: str, passband_mode: str) -> str:
    """Return the filter- and model-specific population-passband PNG name."""

    passband_model_label(passband_mode)
    safe_filter = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(filter_name)).strip(".-")
    if not safe_filter:
        raise ValueError("filter name contains no safe filename characters")
    return f"population_mean_passband_{safe_filter}_{passband_mode}.png"


def fitted_passband_panel_title(passband_mode: str) -> str:
    """Return a readable residual-panel title for a configured passband model."""

    if passband_mode == "fixed":
        return "Fixed nominal passband + zeropoints,\nbefore star offset"
    model_label = passband_model_label(passband_mode)
    return (
        f"Fitted passband ({model_label}) + zeropoints,\n"
        "before star offset"
    )


def corner_parameter_label(parameter_name: str) -> str:
    """Add a concise physical interpretation while preserving the exact name."""

    descriptions = {
        "mu_delta_lambda": "population mean shift [Å]",
        "log_tau_delta_lambda": "ln(shift Σ / Å)",
        "mu_eta": "population mean ln(width scale)",
        "log_tau_eta": "ln(ln-width Σ)",
        "mu_a": "population mean warp",
        "log_tau_a": "ln(warp Σ)",
        "mu_Z": "population mean zeropoint [mag]",
        "log_tau_Z": "ln(zeropoint Σ / mag)",
        "log_sigma_star": "ln(star-residual Σ / mag)",
        "log_sigma_extra": "ln(point-scatter Σ / mag)",
    }
    name = str(parameter_name)
    description = descriptions.get(name)
    if description is None:
        standardized = re.fullmatch(
            r"z_(delta_lambda|eta|a|Z)\[(.+)\]",
            name,
        )
        if standardized is not None:
            family_descriptions = {
                "delta_lambda": "standardized instrument shift deviation",
                "eta": "standardized instrument ln-width deviation",
                "a": "standardized instrument warp deviation",
                "Z": "standardized instrument zeropoint deviation",
            }
            description = family_descriptions[standardized.group(1)]
    return name if description is None else f"{name}\n{description}"


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
    distinct_group_markers: bool,
) -> None:
    if groups is None:
        axis.scatter(x, residual, s=18, alpha=0.75)
        return
    unique = np.unique(groups)
    if distinct_group_markers and len(unique) > len(FILLED_GROUP_MARKERS):
        raise ValueError(
            f"distinct markers support at most {len(FILLED_GROUP_MARKERS)} groups"
        )
    for marker_index, group in enumerate(unique):
        selected = groups == group
        label = str(group)
        if group_names is not None:
            integer = int(group)
            if integer < 0 or integer >= len(group_names):
                raise ValueError("group index is outside group_names")
            label = str(group_names[integer])
        marker = FILLED_GROUP_MARKERS[marker_index] if distinct_group_markers else "o"
        axis.scatter(
            x[selected],
            residual[selected],
            s=18,
            marker=marker,
            alpha=0.75,
            label=label,
        )
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
    distinct_group_markers: bool = False,
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
    display_line_counts = []
    for title, annotation in zip(titles, annotations):
        line_count = title.count("\n") + 1
        if filter_name is not None:
            line_count += 1  # RMS line
        if annotation:
            line_count += annotation.count("\n") + 1
        display_line_counts.append(line_count)
    figure_height = 4.0 + 0.45 * max(0, max(display_line_counts) - 3)
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
        _grouped_scatter(
            axis,
            x_values,
            residual_values,
            group_array,
            group_names,
            distinct_group_markers,
        )
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


def population_mean_passband_quantiles(
    wavelength_angstrom: Sequence[float],
    filter_transmission: Sequence[float],
    aluminum_squared: Sequence[float],
    quantum_efficiency: Sequence[float],
    samples: np.ndarray,
    parameter_names: Sequence[str],
    *,
    passband_mode: str,
    pivot_wavelength_angstrom: float,
    warp_scale_angstrom: float,
    maximum_draws: int,
) -> np.ndarray:
    """Return pointwise 16/50/84% curves using population-mean parameters only.

    Each atmosphere-free curve contains the transformed filter, fixed
    aluminum-squared reflectivity, and fixed CCD QE. Every draw is divided by
    its own peak, removing the grey normalization that is degenerate with the
    additive zeropoint. Population dispersions and instrument deviations are
    deliberately ignored.
    """

    passband_model_label(passband_mode)
    wave = np.asarray(wavelength_angstrom, dtype=np.float64)
    components = [
        np.asarray(value, dtype=np.float64)
        for value in (
            filter_transmission,
            aluminum_squared,
            quantum_efficiency,
        )
    ]
    if wave.ndim != 1 or len(wave) < 2:
        raise ValueError("wavelength must be a vector with at least two entries")
    if (
        not np.all(np.isfinite(wave))
        or not np.all(np.diff(wave) > 0.0)
        or any(value.shape != wave.shape for value in components)
        or any(not np.all(np.isfinite(value)) for value in components)
        or any(np.any(value < 0.0) for value in components)
    ):
        raise ValueError("wavelength and passband components are invalid")
    posterior = np.asarray(samples, dtype=np.float64)
    if posterior.ndim == 3:
        posterior = posterior.reshape(-1, posterior.shape[-1])
    if (
        posterior.ndim != 2
        or posterior.shape[1] != len(parameter_names)
        or not np.all(np.isfinite(posterior))
    ):
        raise ValueError("samples must match the finite sampled-parameter matrix")
    if maximum_draws <= 0:
        raise ValueError("maximum_draws must be positive")

    required = []
    if passband_mode != "fixed":
        required.append("mu_delta_lambda")
    if passband_mode in {"shift_width", "shift_width_warp"}:
        required.append("mu_eta")
    if passband_mode == "shift_width_warp":
        required.append("mu_a")
    missing = [name for name in required if name not in parameter_names]
    if missing:
        raise ValueError(
            "samples lack population-mean parameter(s): " + ", ".join(missing)
        )

    fixed_response = components[1] * components[2]

    def normalized_curve(shift: float, width: float, warp: float) -> np.ndarray:
        if passband_mode == "fixed":
            curve = components[0] * fixed_response
        else:
            curve = instrument_passband(
                wave,
                components[0],
                pivot_wavelength_angstrom=pivot_wavelength_angstrom,
                delta_wavelength_angstrom=shift,
                width_scale=width,
                warp_parameter=warp,
                warp_scale_angstrom=warp_scale_angstrom,
            )
            curve = curve * fixed_response
        peak = float(np.max(curve))
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError("population passband has a nonpositive peak")
        return curve / peak

    if passband_mode == "fixed":
        curve = normalized_curve(0.0, 1.0, 0.0)
        return np.stack((curve, curve, curve))

    lookup = {name: parameter_names.index(name) for name in required}
    indices = np.linspace(
        0,
        len(posterior) - 1,
        min(maximum_draws, len(posterior)),
        dtype=np.int64,
    )
    curves = np.empty((len(indices), len(wave)), dtype=np.float64)
    for output_index, sample_index in enumerate(indices):
        draw = posterior[sample_index]
        shift = float(draw[lookup["mu_delta_lambda"]])
        width = (
            float(np.exp(draw[lookup["mu_eta"]]))
            if "mu_eta" in lookup
            else 1.0
        )
        warp = float(draw[lookup["mu_a"]]) if "mu_a" in lookup else 0.0
        curves[output_index] = normalized_curve(shift, width, warp)
    return np.percentile(curves, [16.0, 50.0, 84.0], axis=0)


def plot_population_mean_passband(
    wavelength_angstrom: Sequence[float],
    nominal: Sequence[float],
    quantiles: np.ndarray,
    *,
    title: str,
    output_path: Optional[Union[Path, str]] = None,
    response_threshold: float = 1.0e-3,
) -> plt.Figure:
    """Plot the normalized population-mean passband and its 68% posterior band."""

    wave = np.asarray(wavelength_angstrom, dtype=np.float64)
    nominal_curve = np.asarray(nominal, dtype=np.float64)
    credible = np.asarray(quantiles, dtype=np.float64)
    if (
        wave.ndim != 1
        or nominal_curve.shape != wave.shape
        or credible.shape != (3, len(wave))
        or not np.all(np.isfinite(nominal_curve))
        or not np.all(np.isfinite(credible))
        or np.any(nominal_curve < 0.0)
        or np.any(credible < 0.0)
        or np.any(credible[0] > credible[1])
        or np.any(credible[1] > credible[2])
    ):
        raise ValueError("population-passband plot arrays are invalid")
    if not np.isfinite(response_threshold) or not 0.0 < response_threshold < 1.0:
        raise ValueError("response_threshold must lie between zero and one")
    nominal_peak = float(np.max(nominal_curve))
    if nominal_peak <= 0.0:
        raise ValueError("nominal passband has a nonpositive peak")
    normalized_nominal = nominal_curve / nominal_peak

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        wave,
        normalized_nominal,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="nominal",
        zorder=3,
    )
    axis.fill_between(
        wave,
        credible[0],
        credible[2],
        color="tab:blue",
        alpha=0.28,
        label="population-mean 68% interval",
        zorder=1,
    )
    axis.plot(
        wave,
        credible[1],
        color="tab:blue",
        linewidth=1.6,
        label="population-mean posterior median",
        zorder=2,
    )
    support = np.maximum(normalized_nominal, credible[2]) >= response_threshold
    if np.any(support):
        selected = np.flatnonzero(support)
        padding = max(5, int(np.ceil(0.04 * (selected[-1] - selected[0] + 1))))
        lower_index = max(0, int(selected[0]) - padding)
        upper_index = min(len(wave) - 1, int(selected[-1]) + padding)
        axis.set_xlim(wave[lower_index], wave[upper_index])
    axis.set_ylim(-0.02, 1.05)
    axis.set(
        xlabel="vacuum wavelength (Å)",
        ylabel="relative atmosphere-free throughput (unit peak)",
        title=title,
    )
    axis.grid(color="0.90", linewidth=0.6)
    axis.legend()
    axis.text(
        0.01,
        0.015,
        "Posterior population means only; population dispersions and "
        "instrument offsets excluded.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize="small",
        color="0.25",
    )
    return _finish_figure(figure, output_path)


def plot_passband_ensemble(
    wavelength_angstrom: Sequence[float],
    median_curves: np.ndarray,
    instrument_names: Sequence[str],
    nominal: Optional[Sequence[float]] = None,
    output_path: Optional[Union[Path, str]] = None,
    title: Optional[str] = None,
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
    axis.set(
        xlabel="wavelength (Angstrom)",
        ylabel="relative throughput",
        title=title,
    )
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
    names = [parameter_names[index] for index in indices]
    labels = [corner_parameter_label(name) for name in names]
    selected_truths = None
    if truths is not None:
        truth_values = np.asarray(truths, dtype=np.float64)
        if truth_values.shape != (values.shape[1],):
            raise ValueError("truths must contain one value per parameter")
        selected_truths = truth_values[indices]
    selected_values = values[:, indices]
    titles = [
        _corner_quantile_title(name, selected_values[:, column])
        for column, name in enumerate(names)
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

    vega_samples = convert_zeropoint_samples(
        mu_z_samples,
        configured_system,
        "vega",
        ab_minus_vega_magnitude,
    )
    ab_samples = convert_zeropoint_samples(
        mu_z_samples,
        configured_system,
        "ab",
        ab_minus_vega_magnitude,
    )
    vega_text = format_quantile_summary(
        "mean zeropoint μ_Z (Vega=0)", vega_samples, unit="mag"
    )
    ab_text = format_quantile_summary(
        "mean zeropoint μ_Z (AB)", ab_samples, unit="mag"
    )
    return f"{vega_text}\n{ab_text}"


def convert_zeropoint_samples(
    mu_z_samples: Sequence[float],
    configured_system: str,
    desired_system: str,
    ab_minus_vega_magnitude: float,
) -> np.ndarray:
    """Convert additive zeropoint samples between Vega-zero and AB conventions."""

    samples = np.asarray(mu_z_samples, dtype=np.float64)
    if samples.ndim != 1 or len(samples) == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("mu_z_samples must be a nonempty finite vector")
    if not np.isfinite(ab_minus_vega_magnitude):
        raise ValueError("AB-minus-Vega magnitude must be finite")
    source = configured_system.strip().lower()
    destination = desired_system.strip().lower()
    if source not in {"vega", "ab"}:
        raise ValueError("configured_system must be 'vega' or 'ab'")
    if destination not in {"vega", "ab"}:
        raise ValueError("desired_system must be 'vega' or 'ab'")
    if source == destination:
        return samples.copy()
    if source == "vega":
        return samples - ab_minus_vega_magnitude
    return samples + ab_minus_vega_magnitude
