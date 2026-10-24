"""Command-line interface for validation, fitting, summaries, and plots."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "passband-fit-matplotlib")
)

import h5py
import numpy as np
import pandas as pd

from .config import load_config
from .curves import interpolate_spectrum, trapezoid_weights
from .io import load_calspec_spectrum, load_vega_spectrum
from .output import sha256_file, write_json
from .pipeline import build_posterior, fit_baseline, prepare_fit, run_fit
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
    plot_passband_ensemble,
    plot_residual_comparison,
    plot_trace,
    residuals_by_target_filename,
)
from .numpyro_sampler import NumpyroSettings
from .synthetic_photometry import ab_magnitude


LOGGER = logging.getLogger("passband_fit")


def _stored_effective_atmospheres(
    curves: pd.DataFrame,
    instrument_names: Sequence[str],
    index_mappings: dict,
    airmass: float = 1.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-instrument and equal-site-mean persisted atmospheres.

    Runs predating site-specific atmospheres contain only the legacy
    representative curve; it is then used for every instrument and for the
    nominal curve so regenerated plots remain faithful to those old runs.
    """

    prefix = "unit_airmass_transmission_"
    site_columns = [name for name in curves.columns if name.startswith(prefix)]
    if not site_columns:
        column = "unit_airmass_transmission"
        if column not in curves:
            raise ValueError(
                "nominal_curves.csv lacks unit_airmass_transmission"
            )
        atmosphere = curves[column].to_numpy(dtype=float) ** float(airmass)
        return (
            np.broadcast_to(atmosphere, (len(instrument_names), len(atmosphere))).copy(),
            atmosphere,
        )

    atmosphere_by_site = {
        name[len(prefix) :]: curves[name].to_numpy(dtype=float) ** float(airmass)
        for name in site_columns
    }
    instrument_sites = index_mappings.get("instrument_atmosphere_sites", {})
    missing_instruments = [
        name for name in instrument_names if name not in instrument_sites
    ]
    if missing_instruments:
        raise ValueError(
            "index_mappings.json lacks atmosphere sites for instrument(s): "
            + ", ".join(missing_instruments)
        )
    missing_sites = sorted(
        {instrument_sites[name] for name in instrument_names}
        - set(atmosphere_by_site)
    )
    if missing_sites:
        raise ValueError(
            "nominal_curves.csv lacks atmosphere curve(s) for site(s): "
            + ", ".join(missing_sites)
        )
    instrument_atmospheres = np.asarray(
        [atmosphere_by_site[instrument_sites[name]] for name in instrument_names]
    )
    mean_atmosphere = np.mean(list(atmosphere_by_site.values()), axis=0)
    return instrument_atmospheres, mean_atmosphere


def _saved_ab_minus_vega(run_dir: Path, curves: pd.DataFrame) -> tuple[str, float]:
    """Read or reconstruct the nominal unit-airmass AB-minus-Vega offset."""

    saved = run_dir / "zeropoint_conventions.json"
    if saved.is_file():
        values = json.loads(saved.read_text())
        return (
            str(values["configured_magnitude_system"]),
            float(values["ab_minus_vega_magnitude"]),
        )
    config = load_config(run_dir / "resolved_config.yaml")
    wave = curves["wavelength_angstrom"].to_numpy(dtype=float)
    passband = (
        curves["nominal_combined_passband"].to_numpy(dtype=float)
        * curves["unit_airmass_transmission"].to_numpy(dtype=float)
    )
    configured_vega = config.inputs.vega_spectrum
    if configured_vega.is_file():
        vega = load_vega_spectrum(config)
    else:
        local_candidates = [
            parent / "CALSPEC_spectra" / configured_vega.name
            for parent in run_dir.resolve().parents
        ]
        local_vega = next((path for path in local_candidates if path.is_file()), None)
        if local_vega is None:
            # Retain the standard input error when no portable copy is available.
            vega = load_vega_spectrum(config)
        else:
            manifest = json.loads((run_dir / "manifest.json").read_text())
            expected_hashes = {
                digest
                for path, digest in manifest.get("input_sha256", {}).items()
                if Path(path).name == configured_vega.name
            }
            if expected_hashes and sha256_file(local_vega) not in expected_hashes:
                raise ValueError(
                    "local Vega spectrum does not match the spectrum hashed in the run: "
                    f"{local_vega}"
                )
            vega = load_calspec_spectrum(local_vega, name="alpha_lyr")
    vega_flux = interpolate_spectrum(
        vega,
        wave,
        passband=passband,
        relative_threshold=config.grid.response_threshold,
    )
    ab_minus_vega = ab_magnitude(
        wave,
        vega_flux,
        passband,
        weights=trapezoid_weights(wave),
    )
    write_json(
        saved,
        {
            "configured_magnitude_system": config.model.magnitude_system,
            "ab_minus_vega_magnitude": ab_minus_vega,
            "conversion_passband": (
                "nominal combined passband with the representative site "
                "atmosphere at unit airmass"
                if config.model.atmosphere_mode == "site_specific"
                else "nominal combined passband at unit airmass"
            ),
            "zeropoint_definition": "fitted additive offset; internal magnitude bookkeeping excluded",
        },
    )
    return config.model.magnitude_system, ab_minus_vega


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passband-fit",
        description="Fit one LCO effective passband against CALSPEC standards.",
    )
    parser.add_argument("--verbose", action="store_true", help="enable informational logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="load and validate without sampling")
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--filter", required=True)

    fit = subparsers.add_parser("fit", help="create a NumPyro NUTS run")
    fit.add_argument("--config", type=Path)
    fit.add_argument("--filter")
    fit.add_argument("--run-label")
    fit.add_argument("--seed", type=int, help="Override sampler.random_seed in the saved run configuration")
    fit.add_argument(
        "--resume",
        type=Path,
        help="legacy option; rejected because emcee chains cannot become NumPyro runs",
    )
    fit.add_argument(
        "--smoke",
        action="store_true",
        help="run 2 short NumPyro chains; not a convergence run",
    )

    summarize = subparsers.add_parser("summarize", help="print a persisted run summary")
    summarize.add_argument("--run", required=True, type=Path)

    plot = subparsers.add_parser("plot", help="regenerate core plots from persisted artifacts")
    plot.add_argument("--run", required=True, type=Path)
    return parser


def _validation_payload(config_path: Path, selected_filter: str) -> dict:
    config = load_config(config_path, selected_filter=selected_filter)
    prepared = prepare_fit(config)
    baseline = fit_baseline(prepared)
    layout, posterior = build_posterior(prepared)
    center = layout.initial_vector(posterior.priors)
    return {
        "status": "valid",
        "selected_filter": selected_filter,
        "selection": prepared.selected.summary,
        "n_targets": len(prepared.selected.target_names),
        "n_instruments": len(prepared.selected.instrument_names),
        "parameter_dimension": layout.ndim,
        "coverage_excluded_targets": list(prepared.coverage_excluded_targets),
        "grid_convergence_max_delta_mag": prepared.grid_convergence_max_delta_mag,
        "baseline": {
            "maximum_log_likelihood": baseline["maximum_log_likelihood"],
            "sigma_extra": baseline["sigma_extra"],
            "metrics": baseline["metrics"],
        },
        "initial_log_posterior": posterior(center),
    }


def _smoke_settings(config: object, ndim: int) -> NumpyroSettings:
    del ndim
    return NumpyroSettings(
        num_warmup=20,
        num_samples=30,
        num_chains=2,
        thinning=1,
        seed=config.sampler.random_seed,
        target_accept_prob=config.sampler.target_accept_prob,
        max_tree_depth=min(config.sampler.max_tree_depth, 7),
        chain_method=config.sampler.chain_method,
        progress_bar=False,
        dense_mass=False,
    )


def _regenerate_plots(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    names = manifest.get("parameter_names", [])
    chain_path = run_dir / "chain.npz"
    if chain_path.is_file():
        stored = np.load(chain_path)
        chain = np.asarray(stored["chain"])
        log_probability = np.asarray(stored["log_probability"])
        diagnostics = json.loads((run_dir / "diagnostics.json").read_text())
        acceptance = np.asarray(
            diagnostics["acceptance_probability_by_chain"], dtype=float
        )
    else:
        # Read-only compatibility for plots from pre-NumPyro emcee runs.
        with h5py.File(run_dir / "chain.h5", "r") as handle:
            group = handle["mcmc"]
            chain = np.asarray(group["chain"])
            log_probability = np.asarray(group["log_prob"])
            acceptance = np.asarray(group["accepted"]) / max(1, chain.shape[0])
    plots = run_dir / "plots"
    plot_trace(chain, names, plots / "trace.png")
    plot_log_probability_trace(log_probability, plots / "log_probability.png")
    plot_acceptance_fractions(acceptance, plots / "acceptance.png")
    residuals = pd.read_csv(run_dir / "residuals.csv")
    curves = pd.read_csv(run_dir / "nominal_curves.csv")
    run_config = load_config(run_dir / "resolved_config.yaml")
    filter_name = str(manifest.get("selected_filter", "unknown filter"))
    generate_population_mean_passband_products(run_dir)
    mean_zeropoint_annotation = None
    if "mu_Z" in names:
        mu_z_samples = chain[:, :, names.index("mu_Z")].reshape(-1)
        magnitude_system, ab_minus_vega = _saved_ab_minus_vega(run_dir, curves)
        mean_zeropoint_annotation = format_zeropoint_conventions(
            mu_z_samples,
            magnitude_system,
            ab_minus_vega,
        )
    star_dispersion_annotation = None
    if "log_sigma_star" in names:
        sigma_star_samples = np.exp(
            chain[:, :, names.index("log_sigma_star")].reshape(-1)
        )
        star_dispersion_annotation = format_quantile_summary(
            "star-offset population dispersion σ_star",
            sigma_star_samples,
            unit="mag",
        )
    instruments = tuple(dict.fromkeys(residuals["instrument"].astype(str)))
    if run_config.model.passband_mode == "shift_width_leak":
        physical = pd.read_csv(run_dir / "parameter_summary_physical.csv").set_index("parameter")
        scales = physical.loc[[f"leak_scale[{name}]" for name in instruments]]
        plot_forest(instruments, scales["median"].to_numpy(), scales["q16"].to_numpy(), scales["q84"].to_numpy(),
                    parameter_label="leak scale (0–1)", output_path=plots / "forest_leak_scale.png")
    lookup = {name: index for index, name in enumerate(instruments)}
    groups = residuals["instrument"].map(lookup).to_numpy()
    target_residual_path = plots / residuals_by_target_filename(
        filter_name, run_config.model.passband_mode
    )
    plot_residual_comparison(
        residuals["nominal_mean_wavelength_angstrom"],
        residuals["nominal_residual"],
        residuals["fitted_mean_wavelength_angstrom"],
        residuals["fitted_residual"],
        groups=groups,
        group_names=instruments,
        filter_name=filter_name,
        output_path=plots / "residuals_by_instrument.png",
    )
    targets = tuple(dict.fromkeys(residuals["target"].astype(str)))
    target_lookup = {name: index for index, name in enumerate(targets)}
    target_groups = residuals["target"].map(target_lookup).to_numpy()
    if "fitted_residual_without_star_offset" in residuals:
        residual_without_star = residuals["fitted_residual_without_star_offset"]
    elif "fitted_star_offset" in residuals:
        # Compatibility for runs written before the exact pre-star residual
        # was persisted. Medians of the two components provide a close plot-only
        # reconstruction without rerunning sampling.
        residual_without_star = (
            residuals["fitted_residual"] + residuals["fitted_star_offset"]
        )
    else:
        residual_without_star = None
    target_plot_kwargs = {}
    if residual_without_star is not None:
        target_plot_kwargs = {
            "intermediate_x": residuals["fitted_mean_wavelength_angstrom"],
            "intermediate_residual": residual_without_star,
            "panel_titles": (
                "Nominal passband",
                fitted_passband_panel_title(run_config.model.passband_mode),
                "Full fitted model, after star offset",
            ),
            "panel_annotations": (
                None,
                mean_zeropoint_annotation,
                star_dispersion_annotation,
            ),
        }
    elif mean_zeropoint_annotation is not None or star_dispersion_annotation is not None:
        fitted_annotations = "\n".join(
            annotation
            for annotation in (
                mean_zeropoint_annotation,
                star_dispersion_annotation,
            )
            if annotation is not None
        )
        target_plot_kwargs = {
            "panel_annotations": (None, fitted_annotations),
        }
    plot_residual_comparison(
        residuals["nominal_mean_wavelength_angstrom"],
        residuals["nominal_residual"],
        residuals["fitted_mean_wavelength_angstrom"],
        residuals["fitted_residual"],
        groups=target_groups,
        group_names=targets,
        distinct_group_markers=True,
        figure_scale=0.8,
        title_fontsize=9.0,
        filter_name=filter_name,
        output_path=target_residual_path,
        **target_plot_kwargs,
    )
    legacy_target_residual_plot = plots / "residuals_by_target.png"
    if legacy_target_residual_plot.is_file():
        legacy_target_residual_plot.unlink()
    passbands = np.load(run_dir / "posterior_passbands.npz")
    passband_instruments = passbands["instrument_names"].astype(str)
    index_mappings = json.loads((run_dir / "index_mappings.json").read_text())
    instrument_atmospheres, mean_atmosphere = _stored_effective_atmospheres(
        curves,
        passband_instruments,
        index_mappings,
        1.3,
    )
    plot_passband_ensemble(
        passbands["wavelength_angstrom"],
        passbands["median"],
        passband_instruments,
        nominal=curves["nominal_combined_passband"],
        instrument_atmospheric_transmission=instrument_atmospheres,
        nominal_atmospheric_transmission=mean_atmosphere,
        atmospheric_airmass=1.3,
        title=(
            f"{filter_name} passbands — model: "
            f"{passband_model_label(run_config.model.passband_mode)}"
        ),
        output_path=plots
        / passband_ensemble_filename(filter_name, run_config.model.passband_mode),
    )
    legacy_passband_plot = plots / "passbands_all_instruments.png"
    if legacy_passband_plot.is_file():
        legacy_passband_plot.unlink()
    flattened_path = run_dir / "flattened_samples.npz"
    if flattened_path.is_file() and names:
        flattened = np.load(flattened_path)
        samples = np.asarray(flattened["samples"], dtype=float)
        draw_indices = np.linspace(
            0, len(samples) - 1, min(len(samples), 5000), dtype=int
        )
        corner_samples = samples[draw_indices]
        hyper_indices = [
            index
            for index, name in enumerate(names)
            if name.startswith("mu_")
            or name.startswith("log_tau_")
            or name == "log_sigma_star"
            or name == "log_sigma_extra"
        ]
        plot_corner(
            corner_samples,
            names,
            subset=hyper_indices,
            output_path=plots / "corner_hyperparameters.png",
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        if args.command == "validate":
            print(json.dumps(_validation_payload(args.config, args.filter), indent=2))
            return 0
        if args.command == "fit":
            if args.resume is not None:
                resolved = args.resume / "resolved_config.yaml"
                config = load_config(resolved, selected_filter=args.filter)
            else:
                if args.config is None or args.filter is None:
                    parser.error("new fits require --config and --filter")
                config = load_config(args.config, selected_filter=args.filter)
            if args.seed is not None:
                if not 0 <= args.seed < 2**32:
                    parser.error("--seed must be between 0 and 2**32-1")
                config = replace(config, sampler=replace(config.sampler, random_seed=args.seed))
            override = None
            if args.smoke:
                prepared = prepare_fit(config)
                layout, _ = build_posterior(prepared)
                override = _smoke_settings(config, layout.ndim)
            run_dir = run_fit(
                config,
                run_label=args.run_label,
                resume_directory=args.resume,
                sampler_override=override,
            )
            print(run_dir)
            return 0
        if args.command == "summarize":
            summary_path = args.run / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"missing run summary: {summary_path}")
            print(summary_path.read_text(), end="")
            return 0
        if args.command == "plot":
            _regenerate_plots(args.run)
            print(args.run / "plots")
            return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
