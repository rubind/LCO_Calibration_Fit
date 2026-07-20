"""Extend CALSPEC FITS spectra with best-matching ASCII stellar templates.

The measured CALSPEC samples are never modified.  For each requested blue or
red extension, every usable template is compared with the adjacent CALSPEC
overlap after fitting a very smooth multiplicative correction in log flux.
The lowest robust residual selects the template.  A short cosine blend to a
locally fitted CALSPEC continuum makes the join continuous outside the trusted
CALSPEC interval.

Synthetic rows are explicitly flagged in the output FITS table.  Their error
and exposure columns are not presented as measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d


FITS_SUFFIXES = {".fits", ".fit", ".fts"}
DEFAULT_PATTERNS = ("*.txt", "*.dat", "*.ascii", "*.csv", "*.spec")


@dataclass(frozen=True)
class TemplateSpectrum:
    path: Path
    wavelength_angstrom: np.ndarray
    flux: np.ndarray


@dataclass(frozen=True)
class TemplateMatch:
    side: str
    template_path: str
    mad_mmag: float
    overlap_minimum_angstrom: float
    overlap_maximum_angstrom: float
    overlap_points: int
    continuum_degree: int
    continuum_coefficients: Tuple[float, ...]
    correction_coordinate_scale_angstrom: float


@dataclass(frozen=True)
class ExtensionResult:
    wavelength_angstrom: np.ndarray
    flux: np.ndarray
    model_extension: np.ndarray
    matches: Mapping[str, TemplateMatch]
    candidate_rows: Tuple[Mapping[str, object], ...]
    original_minimum_angstrom: float
    original_maximum_angstrom: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wavelength_factor(unit: str) -> float:
    normalized = unit.strip().lower()
    factors = {
        "angstrom": 1.0,
        "angstroms": 1.0,
        "aa": 1.0,
        "nm": 10.0,
        "micron": 1.0e4,
        "microns": 1.0e4,
        "um": 1.0e4,
    }
    if normalized not in factors:
        raise ValueError(f"unsupported template wavelength unit {unit!r}")
    return factors[normalized]


def _air_to_vacuum(wavelength_air_angstrom: np.ndarray) -> np.ndarray:
    if np.any(wavelength_air_angstrom < 2000.0):
        raise ValueError("air-to-vacuum conversion is supported only above 2000 Angstrom")
    sigma_squared = (1.0e4 / wavelength_air_angstrom) ** 2
    refractive_index = 1.0 + 1.0e-8 * (
        8342.13
        + 2406030.0 / (130.0 - sigma_squared)
        + 15997.0 / (38.9 - sigma_squared)
    )
    return wavelength_air_angstrom * refractive_index


def _column_key(value: str) -> object:
    try:
        return int(value)
    except ValueError:
        return value


def load_ascii_template(
    path: Path,
    *,
    wavelength_column: str = "0",
    flux_column: str = "1",
    wavelength_unit: str = "angstrom",
    wavelength_convention: str = "vacuum",
    delimiter: Optional[str] = None,
) -> TemplateSpectrum:
    """Load one numeric or named-column ASCII template spectrum."""

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"template does not exist: {path}")
    wave_key = _column_key(str(wavelength_column))
    flux_key = _column_key(str(flux_column))
    actual_delimiter = "," if delimiter is None and path.suffix.lower() == ".csv" else delimiter
    try:
        if isinstance(wave_key, int) and isinstance(flux_key, int):
            data = np.genfromtxt(
                path,
                comments="#",
                delimiter=actual_delimiter,
                invalid_raise=False,
                dtype=np.float64,
            )
            if data.ndim == 1:
                data = data[None, :]
            if data.ndim != 2 or max(wave_key, flux_key) >= data.shape[1]:
                raise ValueError("requested numeric columns are absent")
            wave = data[:, wave_key]
            flux = data[:, flux_key]
        else:
            table = pd.read_csv(
                path,
                comment="#",
                sep=actual_delimiter if actual_delimiter is not None else r"\s+",
                engine="python",
            )
            wave = pd.to_numeric(table[wave_key], errors="coerce").to_numpy(dtype=float)
            flux = pd.to_numeric(table[flux_key], errors="coerce").to_numpy(dtype=float)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        raise ValueError(f"could not parse template {path}: {exc}") from exc

    finite_positive = np.isfinite(wave) & np.isfinite(flux) & (wave > 0.0) & (flux > 0.0)
    wave = np.asarray(wave[finite_positive], dtype=np.float64)
    flux = np.asarray(flux[finite_positive], dtype=np.float64)
    if len(wave) < 10:
        raise ValueError(f"template {path} has fewer than 10 positive finite samples")
    wave *= _wavelength_factor(wavelength_unit)
    convention = wavelength_convention.strip().lower()
    if convention == "air":
        wave = _air_to_vacuum(wave)
    elif convention != "vacuum":
        raise ValueError("template wavelength convention must be 'air' or 'vacuum'")
    order = np.argsort(wave, kind="stable")
    wave, flux = wave[order], flux[order]
    unique, inverse, counts = np.unique(wave, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        summed = np.zeros(len(unique), dtype=np.float64)
        np.add.at(summed, inverse, flux)
        flux = summed / counts
        wave = unique
    if not np.all(np.diff(wave) > 0.0):
        raise ValueError(f"template {path} does not have an increasing wavelength grid")
    return TemplateSpectrum(path.resolve(), wave, flux)


def discover_templates(
    directory: Path,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
) -> Tuple[Path, ...]:
    """Return unique matching template files recursively in stable order."""

    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"template directory does not exist: {root}")
    found = {path.resolve() for pattern in patterns for path in root.rglob(pattern) if path.is_file()}
    if not found:
        raise ValueError(
            f"no template files under {root} matched: {', '.join(patterns)}"
        )
    return tuple(sorted(found, key=lambda item: str(item)))


def _read_calspec(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    path = Path(path)
    if path.suffix.lower() not in FITS_SUFFIXES or not path.is_file():
        raise ValueError(f"CALSPEC input must be an existing FITS file: {path}")
    try:
        with fits.open(path, memmap=False) as hdul:
            sci_index = hdul.index_of("SCI") if "SCI" in hdul else 1
            hdu = hdul[sci_index]
            names = {name.upper(): name for name in hdu.columns.names}
            if "WAVELENGTH" not in names or "FLUX" not in names:
                raise ValueError("SCI table requires WAVELENGTH and FLUX columns")
            wave = np.asarray(hdu.data[names["WAVELENGTH"]], dtype=np.float64)
            flux = np.asarray(hdu.data[names["FLUX"]], dtype=np.float64)
            wave_unit = str(hdu.columns[names["WAVELENGTH"]].unit or "").strip().lower()
            flux_unit = str(hdu.columns[names["FLUX"]].unit or "").strip().lower()
    except (OSError, IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"could not read CALSPEC FITS file {path}: {exc}") from exc
    if wave_unit not in {"angstrom", "angstroms", "ang", "aa"}:
        raise ValueError(f"unsupported CALSPEC wavelength unit {wave_unit!r}")
    if flux_unit not in {
        "flam",
        "erg/s/cm2/a",
        "erg/s/cm^2/a",
        "erg s-1 cm-2 angstrom-1",
    }:
        raise ValueError(f"unsupported CALSPEC flux unit {flux_unit!r}")
    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0.0)
    if not np.all(valid):
        raise ValueError("CALSPEC wavelengths must be positive and wavelength/flux finite")
    order = np.argsort(wave, kind="stable")
    wave, flux = wave[order], flux[order]
    if np.any(np.diff(wave) <= 0.0):
        raise ValueError("CALSPEC wavelengths must be strictly increasing and unique")
    return wave, flux, sci_index


def _robust_polynomial_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(x), dtype=bool)
    coefficients = np.polynomial.polynomial.polyfit(x, y, degree)
    for _ in range(8):
        residual = y - np.polynomial.polynomial.polyval(x, coefficients)
        center = float(np.median(residual[keep]))
        scale = 1.4826 * float(np.median(np.abs(residual[keep] - center)))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            break
        updated = np.abs(residual - center) <= 4.0 * scale
        if np.count_nonzero(updated) < max(degree + 2, 10) or np.array_equal(updated, keep):
            break
        keep = updated
        coefficients = np.polynomial.polynomial.polyfit(x[keep], y[keep], degree)
    return np.asarray(coefficients, dtype=np.float64), keep


def _match_template(
    side: str,
    calspec_wave: np.ndarray,
    calspec_flux: np.ndarray,
    template: TemplateSpectrum,
    target_wavelength: float,
    *,
    overlap_width_angstrom: float,
    continuum_degree: int,
    smooth_width_angstrom: float,
    minimum_overlap_points: int,
) -> TemplateMatch:
    edge = float(calspec_wave[0] if side == "blue" else calspec_wave[-1])
    if side == "blue":
        if template.wavelength_angstrom[0] > target_wavelength:
            raise ValueError("template does not reach requested blue wavelength")
        selected = (
            (calspec_wave >= edge)
            & (calspec_wave <= edge + overlap_width_angstrom)
            & (calspec_wave >= template.wavelength_angstrom[0])
            & (calspec_wave <= template.wavelength_angstrom[-1])
        )
    else:
        if template.wavelength_angstrom[-1] < target_wavelength:
            raise ValueError("template does not reach requested red wavelength")
        selected = (
            (calspec_wave <= edge)
            & (calspec_wave >= edge - overlap_width_angstrom)
            & (calspec_wave >= template.wavelength_angstrom[0])
            & (calspec_wave <= template.wavelength_angstrom[-1])
        )
    wave = calspec_wave[selected]
    observed = calspec_flux[selected]
    if len(wave) < minimum_overlap_points:
        raise ValueError(
            f"only {len(wave)} overlap samples; need {minimum_overlap_points}"
        )
    template_flux = np.interp(wave, template.wavelength_angstrom, template.flux)
    valid = np.isfinite(template_flux) & (template_flux > 0.0) & (observed > 0.0)
    wave, observed, template_flux = wave[valid], observed[valid], template_flux[valid]
    if len(wave) < minimum_overlap_points:
        raise ValueError("insufficient positive overlap flux")
    log_ratio = np.log(observed) - np.log(template_flux)
    if smooth_width_angstrom > 0.0:
        spacing = float(np.median(np.diff(wave)))
        sigma_samples = smooth_width_angstrom / max(spacing, np.finfo(float).eps)
        if sigma_samples >= 0.5:
            log_ratio = gaussian_filter1d(log_ratio, sigma_samples, mode="nearest")
    x = (wave - edge) / overlap_width_angstrom
    coefficients, keep = _robust_polynomial_fit(x, log_ratio, continuum_degree)
    residual = log_ratio - np.polynomial.polynomial.polyval(x, coefficients)
    # Rank with the raw median absolute deviation in magnitude residuals.  Do
    # not convert it to an RMS-equivalent sigma: the robust statistic and its
    # units should remain explicit when noisy CALSPEC pixels are present.
    residual_magnitude = -(2.5 / np.log(10.0)) * residual
    centered = residual_magnitude - np.median(residual_magnitude)
    mad_mmag = float(1000.0 * np.median(np.abs(centered)))
    return TemplateMatch(
        side=side,
        template_path=str(template.path),
        mad_mmag=mad_mmag,
        overlap_minimum_angstrom=float(wave[0]),
        overlap_maximum_angstrom=float(wave[-1]),
        overlap_points=int(np.count_nonzero(keep)),
        continuum_degree=continuum_degree,
        continuum_coefficients=tuple(float(value) for value in coefficients),
        correction_coordinate_scale_angstrom=float(overlap_width_angstrom),
    )


def _extension_grid(edge: float, target: float, spacing: float, side: str) -> np.ndarray:
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("output spacing must be positive and finite")
    if side == "red":
        values = np.arange(edge + spacing, target + spacing / 2.0, spacing)
        if len(values) == 0 or values[-1] < target - spacing * 1.0e-6:
            values = np.r_[values, target]
        elif values[-1] > target:
            values[-1] = target
    else:
        descending = np.arange(edge - spacing, target - spacing / 2.0, -spacing)
        if len(descending) == 0 or descending[-1] > target + spacing * 1.0e-6:
            descending = np.r_[descending, target]
        elif descending[-1] < target:
            descending[-1] = target
        values = descending[::-1]
    return np.asarray(np.unique(values), dtype=np.float64)


def _edge_spacing(wavelength: np.ndarray, side: str) -> float:
    differences = np.diff(wavelength)
    local = differences[: min(50, len(differences))] if side == "blue" else differences[-50:]
    return float(np.median(local))


def _local_continuum(
    side: str,
    calspec_wave: np.ndarray,
    calspec_flux: np.ndarray,
    extension_wave: np.ndarray,
    width_angstrom: float,
) -> np.ndarray:
    edge = float(calspec_wave[0] if side == "blue" else calspec_wave[-1])
    if side == "blue":
        selected = calspec_wave <= edge + width_angstrom
        edge_flux = calspec_flux[0]
    else:
        selected = calspec_wave >= edge - width_angstrom
        edge_flux = calspec_flux[-1]
    if edge_flux <= 0.0:
        raise ValueError(
            f"{side} CALSPEC boundary flux is nonpositive; choose a trusted splice boundary"
        )
    wave = calspec_wave[selected]
    selected_flux = calspec_flux[selected]
    positive = selected_flux > 0.0
    wave = wave[positive]
    selected_flux = selected_flux[positive]
    if len(wave) < 5:
        raise ValueError(f"{side} CALSPEC edge has too few positive samples for blending")
    log_flux = np.log(selected_flux)
    x = (wave - edge) / width_angstrom
    coefficients, _ = _robust_polynomial_fit(x, log_flux, degree=1)
    predicted_edge = float(np.exp(np.polynomial.polynomial.polyval(0.0, coefficients)))
    predicted = np.exp(
        np.polynomial.polynomial.polyval(
            (extension_wave - edge) / width_angstrom,
            coefficients,
        )
    )
    return predicted * (edge_flux / predicted_edge)


def _evaluate_extension(
    side: str,
    match: TemplateMatch,
    template: TemplateSpectrum,
    calspec_wave: np.ndarray,
    calspec_flux: np.ndarray,
    target_wavelength: float,
    *,
    output_spacing_angstrom: Optional[float],
    blend_width_angstrom: float,
) -> Tuple[np.ndarray, np.ndarray]:
    edge = float(calspec_wave[0] if side == "blue" else calspec_wave[-1])
    spacing = output_spacing_angstrom or _edge_spacing(calspec_wave, side)
    extension_wave = _extension_grid(edge, target_wavelength, spacing, side)
    raw_template = np.interp(
        extension_wave,
        template.wavelength_angstrom,
        template.flux,
    )
    x = (extension_wave - edge) / match.correction_coordinate_scale_angstrom
    correction = np.exp(
        np.polynomial.polynomial.polyval(x, match.continuum_coefficients)
    )
    extension_flux = raw_template * correction
    if blend_width_angstrom > 0.0:
        local = _local_continuum(
            side,
            calspec_wave,
            calspec_flux,
            extension_wave,
            max(4.0 * blend_width_angstrom, 50.0),
        )
        distance = np.abs(extension_wave - edge)
        phase = np.clip(distance / blend_width_angstrom, 0.0, 1.0)
        template_weight = 0.5 - 0.5 * np.cos(np.pi * phase)
        extension_flux = (1.0 - template_weight) * local + template_weight * extension_flux
    if not np.all(np.isfinite(extension_flux)) or np.any(extension_flux <= 0.0):
        raise ValueError(f"{side} extension produced nonpositive or nonfinite flux")
    return extension_wave, extension_flux


def extend_spectrum(
    calspec_wave: np.ndarray,
    calspec_flux: np.ndarray,
    templates: Sequence[TemplateSpectrum],
    *,
    minimum_wavelength_angstrom: Optional[float] = None,
    maximum_wavelength_angstrom: Optional[float] = None,
    overlap_width_angstrom: float = 1000.0,
    continuum_degree: int = 2,
    smooth_width_angstrom: float = 30.0,
    blend_width_angstrom: float = 25.0,
    output_spacing_angstrom: Optional[float] = None,
    minimum_overlap_points: int = 30,
) -> ExtensionResult:
    """Select templates and extend one or both ends of a CALSPEC spectrum."""

    wave = np.asarray(calspec_wave, dtype=np.float64)
    flux = np.asarray(calspec_flux, dtype=np.float64)
    if wave.ndim != 1 or flux.shape != wave.shape or len(wave) < 10:
        raise ValueError("CALSPEC wavelength and flux must be equal-length vectors")
    if not np.all(np.isfinite(wave)) or not np.all(np.isfinite(flux)):
        raise ValueError("CALSPEC spectrum contains nonfinite values")
    if np.any(np.diff(wave) <= 0.0):
        raise ValueError("CALSPEC wavelengths must be strictly increasing")
    if not templates:
        raise ValueError("at least one template spectrum is required")
    if overlap_width_angstrom <= 0.0 or smooth_width_angstrom < 0.0:
        raise ValueError("overlap width must be positive and smoothing width nonnegative")
    if blend_width_angstrom < 0.0:
        raise ValueError("blend width must be nonnegative")
    if continuum_degree not in {0, 1, 2, 3}:
        raise ValueError("continuum degree must be 0, 1, 2, or 3")
    if minimum_overlap_points < continuum_degree + 2:
        raise ValueError("minimum overlap points is too small for continuum degree")

    requested: Dict[str, float] = {}
    if minimum_wavelength_angstrom is not None:
        if minimum_wavelength_angstrom >= wave[0]:
            raise ValueError("minimum wavelength must lie blueward of CALSPEC coverage")
        requested["blue"] = float(minimum_wavelength_angstrom)
    if maximum_wavelength_angstrom is not None:
        if maximum_wavelength_angstrom <= wave[-1]:
            raise ValueError("maximum wavelength must lie redward of CALSPEC coverage")
        requested["red"] = float(maximum_wavelength_angstrom)
    if not requested:
        raise ValueError("request at least one blue or red extension wavelength")

    matches: Dict[str, TemplateMatch] = {}
    candidate_rows: List[Mapping[str, object]] = []
    template_lookup = {str(template.path): template for template in templates}
    for side, target in requested.items():
        successful: List[TemplateMatch] = []
        for template in templates:
            try:
                match = _match_template(
                    side,
                    wave,
                    flux,
                    template,
                    target,
                    overlap_width_angstrom=overlap_width_angstrom,
                    continuum_degree=continuum_degree,
                    smooth_width_angstrom=smooth_width_angstrom,
                    minimum_overlap_points=minimum_overlap_points,
                )
            except ValueError as exc:
                candidate_rows.append(
                    {
                        "side": side,
                        "template_path": str(template.path),
                        "status": "rejected",
                        "reason": str(exc),
                        "mad_mmag": None,
                    }
                )
                continue
            successful.append(match)
            candidate_rows.append(
                {
                    "side": side,
                    "template_path": str(template.path),
                    "status": "candidate",
                    "reason": "",
                    "mad_mmag": match.mad_mmag,
                    "overlap_points": match.overlap_points,
                    "overlap_minimum_angstrom": match.overlap_minimum_angstrom,
                    "overlap_maximum_angstrom": match.overlap_maximum_angstrom,
                }
            )
        if not successful:
            raise ValueError(
                f"no template can provide the requested {side} extension to {target:.1f} Angstrom"
            )
        matches[side] = min(successful, key=lambda item: item.mad_mmag)

    blue_wave = blue_flux = np.asarray([], dtype=np.float64)
    red_wave = red_flux = np.asarray([], dtype=np.float64)
    if "blue" in matches:
        match = matches["blue"]
        blue_wave, blue_flux = _evaluate_extension(
            "blue",
            match,
            template_lookup[match.template_path],
            wave,
            flux,
            requested["blue"],
            output_spacing_angstrom=output_spacing_angstrom,
            blend_width_angstrom=blend_width_angstrom,
        )
    if "red" in matches:
        match = matches["red"]
        red_wave, red_flux = _evaluate_extension(
            "red",
            match,
            template_lookup[match.template_path],
            wave,
            flux,
            requested["red"],
            output_spacing_angstrom=output_spacing_angstrom,
            blend_width_angstrom=blend_width_angstrom,
        )
    combined_wave = np.r_[blue_wave, wave, red_wave]
    combined_flux = np.r_[blue_flux, flux, red_flux]
    model_extension = np.r_[
        np.ones(len(blue_wave), dtype=bool),
        np.zeros(len(wave), dtype=bool),
        np.ones(len(red_wave), dtype=bool),
    ]
    if np.any(np.diff(combined_wave) <= 0.0):
        raise RuntimeError("combined extension wavelength grid is not strictly increasing")
    return ExtensionResult(
        wavelength_angstrom=combined_wave,
        flux=combined_flux,
        model_extension=model_extension,
        matches=matches,
        candidate_rows=tuple(candidate_rows),
        original_minimum_angstrom=float(wave[0]),
        original_maximum_angstrom=float(wave[-1]),
    )


def _fill_for_column(name: str, original: np.ndarray, count: int, spacing: float) -> np.ndarray:
    upper = name.upper()
    shape = (count,) + original.shape[1:]
    if upper in {"STATERROR", "SYSERROR"}:
        return np.full(shape, np.nan, dtype=original.dtype)
    if upper == "FWHM":
        return np.full(shape, spacing, dtype=original.dtype)
    if upper == "DATAQUAL":
        return np.full(shape, 2, dtype=original.dtype)
    if upper == "TOTEXP":
        return np.zeros(shape, dtype=original.dtype)
    if np.issubdtype(original.dtype, np.floating):
        return np.full(shape, np.nan, dtype=original.dtype)
    if np.issubdtype(original.dtype, np.integer):
        return np.zeros(shape, dtype=original.dtype)
    return np.full(shape, "", dtype=original.dtype)


def write_extended_fits(
    input_path: Path,
    output_path: Path,
    result: ExtensionResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Preserve the CALSPEC HDU structure and add explicitly flagged model rows."""

    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(input_path, memmap=False) as hdul:
        sci_index = hdul.index_of("SCI") if "SCI" in hdul else 1
        old_hdu = hdul[sci_index]
        old_data = old_hdu.data
        names = {name.upper(): name for name in old_hdu.columns.names}
        original_wave = np.asarray(old_data[names["WAVELENGTH"]], dtype=float)
        order = np.argsort(original_wave, kind="stable")
        blue_count = int(np.count_nonzero(result.wavelength_angstrom < result.original_minimum_angstrom))
        red_count = int(np.count_nonzero(result.wavelength_angstrom > result.original_maximum_angstrom))
        spacing = float(np.median(np.diff(result.wavelength_angstrom)))
        columns = []
        for definition in old_hdu.columns:
            original = np.asarray(old_data[definition.name])[order]
            if definition.name.upper() == "WAVELENGTH":
                values = result.wavelength_angstrom
            elif definition.name.upper() == "FLUX":
                values = result.flux
            else:
                blue = _fill_for_column(definition.name, original, blue_count, spacing)
                red = _fill_for_column(definition.name, original, red_count, spacing)
                values = np.concatenate((blue, original, red), axis=0)
            columns.append(
                fits.Column(
                    name=definition.name,
                    format=definition.format,
                    unit=definition.unit,
                    dim=definition.dim,
                    null=definition.null,
                    bscale=definition.bscale,
                    bzero=definition.bzero,
                    disp=definition.disp,
                    array=values,
                )
            )
        columns.append(
            fits.Column(name="MODEL_EXT", format="L", array=result.model_extension)
        )
        new_hdu = fits.BinTableHDU.from_columns(
            columns,
            header=old_hdu.header.copy(),
            name=old_hdu.name,
        )
        new_hdu.header["EXTENDED"] = (True, "Contains template-model extension rows")
        new_hdu.header["ORIGWMIN"] = (result.original_minimum_angstrom, "Original minimum wavelength")
        new_hdu.header["ORIGWMAX"] = (result.original_maximum_angstrom, "Original maximum wavelength")
        for side, match in result.matches.items():
            prefix = "B" if side == "blue" else "R"
            new_hdu.header[f"{prefix}TMPL"] = (Path(match.template_path).name[:68], f"{side} template")
            new_hdu.header[f"{prefix}MAD"] = (match.mad_mmag, f"{side} overlap MAD mmag")
        new_hdu.header.add_history("Original CALSPEC rows are unchanged; MODEL_EXT marks synthetic rows.")
        new_hdu.header.add_history("Extensions use a template times a smooth log-flux polynomial correction.")
        hdul[sci_index] = new_hdu
        hdul[0].header.add_history("Spectrum extended outside CALSPEC coverage; see SCI MODEL_EXT.")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".fits", dir=str(output_path.parent)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            hdul.writeto(temporary, overwrite=True, checksum=True)
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    return output_path


def write_diagnostics(
    output_path: Path,
    input_path: Path,
    templates: Sequence[TemplateSpectrum],
    result: ExtensionResult,
    settings: Mapping[str, object],
) -> Tuple[Path, Path, Path]:
    """Write JSON provenance, candidate ranking CSV, and a splice plot."""

    output_path = Path(output_path)
    prefix = output_path.with_suffix("")
    json_path = prefix.with_name(prefix.name + "_extension.json")
    csv_path = prefix.with_name(prefix.name + "_template_ranking.csv")
    plot_path = prefix.with_name(prefix.name + "_extension.png")
    payload = {
        "input_calspec": str(Path(input_path).resolve()),
        "input_calspec_sha256": _sha256(Path(input_path)),
        "output_fits": str(output_path.resolve()),
        "original_range_angstrom": [
            result.original_minimum_angstrom,
            result.original_maximum_angstrom,
        ],
        "extended_range_angstrom": [
            float(result.wavelength_angstrom[0]),
            float(result.wavelength_angstrom[-1]),
        ],
        "model_extension_rows": int(np.count_nonzero(result.model_extension)),
        "matches": {side: asdict(match) for side, match in result.matches.items()},
        "settings": dict(settings),
        "template_sha256": {str(item.path): _sha256(item.path) for item in templates},
        "warning": (
            "MODEL_EXT rows are synthetic template-based values, not CALSPEC measurements; "
            "their STATERROR/SYSERROR fields are NaN and DATAQUAL is 2."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ranking = pd.DataFrame(result.candidate_rows)
    if not ranking.empty:
        ranking = ranking.sort_values(["side", "mad_mmag"], na_position="last")
    ranking.to_csv(csv_path, index=False)

    figure = make_diagnostic_figure(templates, result)
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return json_path, csv_path, plot_path


def make_diagnostic_figure(
    templates: Sequence[TemplateSpectrum],
    result: ExtensionResult,
) -> plt.Figure:
    """Plot the full splice, selected-template overlap, and candidate scores."""

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    original = ~result.model_extension
    original_wave = result.wavelength_angstrom[original]
    original_flux = result.flux[original]
    axes[0].plot(
        original_wave,
        original_flux,
        color="black",
        linewidth=1.0,
        label="CALSPEC",
    )
    axes[0].plot(
        result.wavelength_angstrom[result.model_extension],
        result.flux[result.model_extension],
        color="tab:blue",
        linewidth=1.2,
        label="template extension",
    )
    axes[0].set(ylabel="flux (FLAM)", yscale="log")
    axes[0].legend()

    template_lookup = {str(template.path): template for template in templates}
    side_colors = {"blue": "tab:purple", "red": "tab:orange"}
    for side, match in result.matches.items():
        template = template_lookup[match.template_path]
        selected = (
            (original_wave >= match.overlap_minimum_angstrom)
            & (original_wave <= match.overlap_maximum_angstrom)
            & (original_flux > 0.0)
        )
        overlap_wave = original_wave[selected]
        overlap_flux = original_flux[selected]
        edge = (
            result.original_minimum_angstrom
            if side == "blue"
            else result.original_maximum_angstrom
        )
        template_flux = np.interp(
            overlap_wave,
            template.wavelength_angstrom,
            template.flux,
        )
        coordinate = (overlap_wave - edge) / match.correction_coordinate_scale_angstrom
        corrected_template = template_flux * np.exp(
            np.polynomial.polynomial.polyval(
                coordinate,
                match.continuum_coefficients,
            )
        )
        color = side_colors[side]
        axes[1].plot(
            overlap_wave,
            overlap_flux,
            color="black",
            linewidth=1.0,
            alpha=0.85,
            label=f"CALSPEC ({side} overlap)",
        )
        axes[1].plot(
            overlap_wave,
            corrected_template,
            color=color,
            linewidth=1.0,
            linestyle="--",
            label=(
                f"{side} corrected template: {Path(match.template_path).name} "
                f"({match.mad_mmag:.2f} mmag MAD)"
            ),
        )
    axes[1].set(
        xlabel="vacuum wavelength (Angstrom)",
        ylabel="flux (FLAM)",
        yscale="log",
    )
    axes[1].legend(fontsize="small")

    ranking = pd.DataFrame(result.candidate_rows)
    if not ranking.empty:
        ranking = ranking.sort_values(["side", "mad_mmag"], na_position="last")
    for side, match in result.matches.items():
        selected = ranking[(ranking["side"] == side) & (ranking["status"] == "candidate")]
        axes[2].scatter(
            np.arange(len(selected)),
            selected["mad_mmag"],
            s=24,
            label=f"{side} candidates",
        )
        best_index = int(np.argmin(selected["mad_mmag"].to_numpy()))
        axes[2].scatter(best_index, match.mad_mmag, marker="*", s=100)
    axes[2].set(xlabel="candidate rank order", ylabel="overlap MAD (mmag)")
    if result.matches:
        axes[2].legend()
    figure.tight_layout()
    return figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extend a CALSPEC FITS spectrum with the best-matching ASCII stellar template. "
            "Original CALSPEC samples are preserved exactly."
        )
    )
    parser.add_argument("calspec", type=Path, help="input CALSPEC FITS file")
    parser.add_argument("template_directory", type=Path, help="directory searched recursively")
    parser.add_argument("output", type=Path, help="new extended FITS file")
    parser.add_argument("--min-wavelength", type=float, help="requested blue limit in vacuum Angstrom")
    parser.add_argument("--max-wavelength", type=float, help="requested red limit in vacuum Angstrom")
    parser.add_argument("--pattern", action="append", dest="patterns", help="template glob; repeatable")
    parser.add_argument("--template-wave-column", default="0", help="zero-based index or column name")
    parser.add_argument("--template-flux-column", default="1", help="zero-based index or column name")
    parser.add_argument(
        "--template-wavelength-unit",
        default="angstrom",
        choices=("angstrom", "nm", "micron"),
    )
    parser.add_argument(
        "--template-wavelength-convention",
        default="vacuum",
        choices=("vacuum", "air"),
    )
    parser.add_argument("--delimiter", help="template delimiter; default whitespace or CSV comma")
    parser.add_argument("--overlap-width", type=float, default=1000.0, help="edge overlap in Angstrom")
    parser.add_argument("--continuum-degree", type=int, default=2, choices=(0, 1, 2, 3))
    parser.add_argument("--smooth-width", type=float, default=30.0, help="matching smoothing in Angstrom")
    parser.add_argument("--blend-width", type=float, default=25.0, help="continuous edge blend in Angstrom")
    parser.add_argument("--output-spacing", type=float, help="extension sampling in Angstrom")
    parser.add_argument("--minimum-overlap-points", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_from_args(args: argparse.Namespace) -> Path:
    wave, flux, _ = _read_calspec(args.calspec)
    paths = discover_templates(args.template_directory, args.patterns or DEFAULT_PATTERNS)
    templates = []
    failures = []
    for path in paths:
        try:
            templates.append(
                load_ascii_template(
                    path,
                    wavelength_column=args.template_wave_column,
                    flux_column=args.template_flux_column,
                    wavelength_unit=args.template_wavelength_unit,
                    wavelength_convention=args.template_wavelength_convention,
                    delimiter=args.delimiter,
                )
            )
        except ValueError as exc:
            failures.append(str(exc))
    if not templates:
        raise ValueError("no usable templates:\n- " + "\n- ".join(failures))
    settings = {
        "minimum_wavelength_angstrom": args.min_wavelength,
        "maximum_wavelength_angstrom": args.max_wavelength,
        "overlap_width_angstrom": args.overlap_width,
        "continuum_degree": args.continuum_degree,
        "smooth_width_angstrom": args.smooth_width,
        "blend_width_angstrom": args.blend_width,
        "output_spacing_angstrom": args.output_spacing,
        "minimum_overlap_points": args.minimum_overlap_points,
        "template_wavelength_unit": args.template_wavelength_unit,
        "template_wavelength_convention": args.template_wavelength_convention,
        "template_parse_failures": failures,
    }
    result = extend_spectrum(
        wave,
        flux,
        templates,
        minimum_wavelength_angstrom=args.min_wavelength,
        maximum_wavelength_angstrom=args.max_wavelength,
        overlap_width_angstrom=args.overlap_width,
        continuum_degree=args.continuum_degree,
        smooth_width_angstrom=args.smooth_width,
        blend_width_angstrom=args.blend_width,
        output_spacing_angstrom=args.output_spacing,
        minimum_overlap_points=args.minimum_overlap_points,
    )
    output = write_extended_fits(args.calspec, args.output, result, overwrite=args.overwrite)
    write_diagnostics(output, args.calspec, templates, result, settings)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = run_from_args(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
