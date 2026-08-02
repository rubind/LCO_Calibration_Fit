"""Readers and validation for observations, throughput curves, and CALSPEC data."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from .config import (
    CurveConfig,
    FitConfig,
    ObservationInputConfig,
    PathLike,
    SelectionConfig,
)
from .contracts import SelectedData, SpectralCurve, Spectrum


LOGGER = logging.getLogger(__name__)


class InputError(ValueError):
    """Raised when an input file cannot satisfy the documented data contract."""


def _first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return line.rstrip("\n")
    raise InputError(f"input table is empty: {path}")


def load_observation_table(config: ObservationInputConfig) -> pd.DataFrame:
    """Read an observation table and attach an immutable zero-based source row id."""

    path = config.path
    if not path.is_file():
        raise InputError(f"observation table does not exist: {path}")
    try:
        if config.format == "whitespace":
            first = _first_nonempty_line(path)
            if first.lstrip().startswith("#"):
                names = first.lstrip()[1:].strip().split()
                table = pd.read_csv(path, sep=r"\s+", comment="#", names=names, engine="python")
            else:
                table = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
        elif config.format == "csv":
            table = pd.read_csv(path, comment="#")
        elif config.format == "tsv":
            table = pd.read_csv(path, sep="\t", comment="#")
        else:  # Protected by configuration validation, retained for direct construction.
            raise InputError(f"unsupported observation format: {config.format}")
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise InputError(f"could not read observation table {path}: {exc}") from exc
    required = [
        config.columns.target,
        config.columns.filter,
        config.columns.image,
        config.columns.airmass,
        config.columns.flux,
        config.columns.flux_uncertainty,
    ]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise InputError(f"observation table is missing column(s): {', '.join(missing)}")
    if table.empty:
        raise InputError(f"observation table has no data rows: {path}")
    result = table.copy()
    result.insert(0, "_source_row_id", np.arange(len(result), dtype=np.int64))
    return result


def _as_finite_numeric(series: pd.Series, label: str, row_ids: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    bad = ~np.isfinite(numeric)
    if np.any(bad):
        rows = ", ".join(str(int(item)) for item in row_ids[bad][:10])
        suffix = "..." if np.count_nonzero(bad) > 10 else ""
        raise InputError(f"nonfinite or nonnumeric {label} in source row(s) {rows}{suffix}")
    return numeric


def _parse_instruments(images: Sequence[str], pattern: str) -> Tuple[str, ...]:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise InputError(f"invalid instrument_regex: {exc}") from exc
    instruments = []
    for image in images:
        match = compiled.search(Path(str(image)).name)
        if match is None or "instrument" not in match.groupdict() or not match.group("instrument"):
            raise InputError(f"cannot parse instrument from image name {image!r}")
        instruments.append(match.group("instrument"))
    return tuple(instruments)


def select_observations(
    table: pd.DataFrame,
    observation_config: ObservationInputConfig,
    selection: SelectionConfig,
    selected_filter: str,
) -> SelectedData:
    """Select one filter, apply ordered quality cuts, and derive indexed arrays.

    Rejection classes are mutually exclusive and applied in this order:
    nonpositive flux, nonpositive uncertainty, and S/N at or below the threshold.
    Other malformed selected-filter numeric data fail validation.
    """

    columns = observation_config.columns
    if "_source_row_id" not in table:
        source_ids = np.arange(len(table), dtype=np.int64)
    else:
        source_ids = pd.to_numeric(table["_source_row_id"], errors="raise").to_numpy(dtype=np.int64)
    filter_values = table[columns.filter].astype(str).to_numpy()
    match = filter_values == str(selected_filter)
    matching = table.loc[match].copy()
    matching_ids = source_ids[match]
    if matching.empty:
        raise InputError(f"no observations match selected filter {selected_filter!r}")

    for label, name in (("target", columns.target), ("image", columns.image)):
        if matching[name].isna().any() or np.any(matching[name].astype(str).str.len().to_numpy() == 0):
            bad = matching_ids[matching[name].isna().to_numpy()]
            raise InputError(f"missing {label} in selected observations; source rows {bad.tolist()}")

    airmass = _as_finite_numeric(matching[columns.airmass], "airmass", matching_ids)
    flux = _as_finite_numeric(matching[columns.flux], "flux", matching_ids)
    uncertainty = _as_finite_numeric(
        matching[columns.flux_uncertainty], "flux uncertainty", matching_ids
    )
    if np.any(airmass <= 0):
        rows = matching_ids[airmass <= 0]
        raise InputError(f"airmass must be positive; source rows {rows[:10].tolist()}")

    reject_flux = flux <= 0
    remaining = ~reject_flux
    reject_uncertainty = remaining & (uncertainty <= 0)
    remaining &= ~reject_uncertainty
    snr = np.full(len(matching), np.nan, dtype=np.float64)
    snr[remaining] = flux[remaining] / uncertainty[remaining]
    reject_snr = remaining & (snr <= float(selection.snr_min))
    keep = remaining & ~reject_snr
    if not np.any(keep):
        raise InputError(f"no observations remain for filter {selected_filter!r} after quality cuts")

    used = matching.loc[keep]
    used_ids = matching_ids[keep]
    used_flux = flux[keep]
    used_uncertainty = uncertainty[keep]
    used_airmass = airmass[keep]
    used_snr = snr[keep]
    target_values = tuple(used[columns.target].astype(str).tolist())
    image_values = tuple(used[columns.image].astype(str).tolist())
    instrument_values = _parse_instruments(image_values, observation_config.instrument_regex)
    site_values = tuple(item[: observation_config.site_prefix_length] for item in instrument_values)

    target_index, target_names_array = pd.factorize(np.asarray(target_values, dtype=object), sort=False)
    instrument_index, instrument_names_array = pd.factorize(
        np.asarray(instrument_values, dtype=object), sort=False
    )
    target_names = tuple(str(item) for item in target_names_array.tolist())
    instrument_names = tuple(str(item) for item in instrument_names_array.tolist())
    magnitude = 24.0 - 2.5 * np.log10(used_flux)
    magnitude_uncertainty = (2.5 / np.log(10.0)) * used_uncertainty / used_flux

    target_counts = pd.Series(target_values, dtype=str).value_counts(sort=False)
    instrument_counts = pd.Series(instrument_values, dtype=str).value_counts(sort=False)
    summary: Dict[str, object] = {
        "total_rows": int(len(table)),
        "matching_filter": int(len(matching)),
        "rejected_nonpositive_flux": int(np.count_nonzero(reject_flux)),
        "rejected_nonpositive_uncertainty": int(np.count_nonzero(reject_uncertainty)),
        "rejected_snr": int(np.count_nonzero(reject_snr)),
        "final_used": int(np.count_nonzero(keep)),
        "observations_per_target": {str(key): int(value) for key, value in target_counts.items()},
        "observations_per_instrument": {
            str(key): int(value) for key, value in instrument_counts.items()
        },
    }
    return SelectedData(
        row_id=used_ids,
        target_index=np.asarray(target_index, dtype=np.int64),
        instrument_index=np.asarray(instrument_index, dtype=np.int64),
        airmass=used_airmass,
        observed_magnitude=magnitude,
        magnitude_uncertainty=magnitude_uncertainty,
        flux=used_flux,
        flux_uncertainty=used_uncertainty,
        snr=used_snr,
        target_names=target_names,
        instrument_names=instrument_names,
        images=image_values,
        sites=site_values,
        summary=summary,
    )


def load_selected_data(config: FitConfig, selected_filter: Optional[str] = None) -> SelectedData:
    """Load and select observations using a resolved :class:`FitConfig`."""

    filter_name = selected_filter if selected_filter is not None else config.selected_filter
    if filter_name is None:
        raise InputError("a selected filter is required")
    return select_observations(
        load_observation_table(config.inputs.observations),
        config.inputs.observations,
        config.selection,
        str(filter_name),
    )


def _read_two_column_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise InputError(f"curve file does not exist: {path}")
    wavelengths = []
    values = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, original in enumerate(handle, start=1):
                line = original.split("#", 1)[0].strip()
                if not line:
                    continue
                fields = line.replace(",", " ").split()
                if len(fields) < 2:
                    continue
                try:
                    wavelength = float(fields[0])
                except ValueError:
                    continue  # A textual header is allowed.
                try:
                    value = float(fields[1])
                except ValueError as exc:
                    raise InputError(f"invalid curve value at {path}:{line_number}") from exc
                wavelengths.append(wavelength)
                values.append(value)
    except (OSError, UnicodeError) as exc:
        raise InputError(f"could not read curve {path}: {exc}") from exc
    if len(wavelengths) < 2:
        raise InputError(f"curve must have at least two numeric rows: {path}")
    wave = np.asarray(wavelengths, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(wave)) or not np.all(np.isfinite(vals)):
        raise InputError(f"curve contains nonfinite data: {path}")
    return wave, vals


def _wavelength_to_angstrom(wavelength: np.ndarray, unit: str) -> np.ndarray:
    factors = {"angstrom": 1.0, "angstroms": 1.0, "aa": 1.0, "nm": 10.0,
               "micron": 1.0e4, "um": 1.0e4}
    try:
        return wavelength * factors[unit.lower()]
    except KeyError as exc:
        raise InputError(f"unsupported wavelength unit: {unit}") from exc


def _air_to_vacuum(wavelength_air: np.ndarray) -> np.ndarray:
    if np.any(wavelength_air < 2000.0):
        raise InputError("air-to-vacuum conversion is only supported above 2000 Angstrom")
    sigma_squared = (1.0e4 / wavelength_air) ** 2
    refractive_index = 1.0 + 1.0e-8 * (
        8342.13 + 2406030.0 / (130.0 - sigma_squared) + 15997.0 / (38.9 - sigma_squared)
    )
    return wavelength_air * refractive_index


def _deduplicate(
    wave: np.ndarray, values: np.ndarray, policy: str, name: str
) -> Tuple[np.ndarray, np.ndarray]:
    unique, inverse, counts = np.unique(wave, return_inverse=True, return_counts=True)
    if np.all(counts == 1):
        return wave, values
    if policy == "error":
        duplicates = unique[counts > 1]
        raise InputError(f"{name}: duplicate wavelength(s), including {duplicates[0]:g}")
    if policy == "mean":
        sums = np.zeros(len(unique), dtype=np.float64)
        np.add.at(sums, inverse, values)
        return unique, sums / counts
    indices = np.zeros(len(unique), dtype=np.int64)
    if policy == "first":
        for index in range(len(values) - 1, -1, -1):
            indices[inverse[index]] = index
    elif policy == "last":
        for index in range(len(values)):
            indices[inverse[index]] = index
    else:
        raise InputError(f"{name}: unsupported duplicate policy {policy}")
    return unique, values[indices]


def load_curve(config: CurveConfig, name: str = "curve") -> SpectralCurve:
    """Load a generic two-column curve and convert it to vacuum-Angstrom fractions."""

    wave, raw_values = _read_two_column_file(config.path)
    if config.sentinel_values:
        sentinel = np.zeros(len(raw_values), dtype=bool)
        for value in config.sentinel_values:
            sentinel |= np.isclose(raw_values, value, rtol=0.0, atol=1.0e-12)
        if np.any(sentinel):
            LOGGER.info("%s: removed %d configured sentinel samples", name, np.count_nonzero(sentinel))
            wave, raw_values = wave[~sentinel], raw_values[~sentinel]
    if len(wave) < 2:
        raise InputError(f"{name}: fewer than two samples remain after sentinel removal")

    small_negative = (raw_values < 0) & (raw_values >= -config.negative_tolerance)
    bad_negative = raw_values < -config.negative_tolerance
    if np.any(bad_negative):
        raise InputError(
            f"{name}: value {np.min(raw_values):g} is below negative tolerance "
            f"{-config.negative_tolerance:g}"
        )
    if np.any(small_negative):
        LOGGER.warning(
            "%s: clipped %d negative values (minimum %.8g) to zero",
            name, np.count_nonzero(small_negative), np.min(raw_values),
        )
        raw_values = raw_values.copy()
        raw_values[small_negative] = 0.0

    wave = _wavelength_to_angstrom(wave, config.wavelength_unit)
    if config.wavelength_convention == "air":
        wave = _air_to_vacuum(wave)
    if config.value_semantics == "fraction":
        values = raw_values
    elif config.value_semantics == "percent":
        values = raw_values / 100.0
    elif config.value_semantics == "extinction_mag_per_airmass":
        values = 10.0 ** (-0.4 * raw_values)
    else:
        raise InputError(f"{name}: unsupported value semantics {config.value_semantics}")
    if np.any(values < 0) or np.any(values > 1.0 + 1.0e-12):
        raise InputError(f"{name}: converted throughput values must lie in [0, 1]")
    values = np.clip(values, 0.0, 1.0)
    order = np.argsort(wave, kind="stable")
    wave, values = _deduplicate(wave[order], values[order], config.duplicate_policy, name)
    return SpectralCurve(wave, values, name)


def load_site_mapping(path: PathLike) -> Dict[str, str]:
    """Load an exact image-prefix to observatory-name mapping.

    The plain-text format has one whitespace-separated ``prefix site_name``
    pair per line. Blank lines and lines beginning with ``#`` are ignored.
    Site names may contain spaces, although the supplied mapping uses compact
    names matching the atmosphere configuration keys.
    """

    mapping_path = Path(path)
    if not mapping_path.is_file():
        raise InputError(f"site mapping does not exist: {mapping_path}")
    result: Dict[str, str] = {}
    try:
        with mapping_path.open("r", encoding="utf-8") as handle:
            for line_number, original in enumerate(handle, start=1):
                line = original.split("#", 1)[0].strip()
                if not line:
                    continue
                fields = line.split(None, 1)
                if len(fields) != 2 or not fields[0] or not fields[1].strip():
                    raise InputError(
                        f"invalid site mapping at {mapping_path}:{line_number}; "
                        "expected 'prefix site_name'"
                    )
                prefix, site_name = fields[0].lower(), fields[1].strip()
                if prefix in result:
                    raise InputError(
                        f"duplicate site prefix {prefix!r} in {mapping_path}"
                    )
                result[prefix] = site_name
    except (OSError, UnicodeError) as exc:
        raise InputError(f"could not read site mapping {mapping_path}: {exc}") from exc
    if not result:
        raise InputError(f"site mapping is empty: {mapping_path}")
    return result


def load_atmosphere_curves(
    config: FitConfig,
    required_prefixes: Sequence[str] = (),
) -> Tuple[Dict[str, str], Dict[str, SpectralCurve]]:
    """Load configured unit-airmass curves and resolve observation prefixes.

    Returns ``(prefix_to_site_name, curves_by_site_name)``. Legacy
    ``fixed_kpno`` configurations are represented by the synthetic site name
    ``fixed`` so old persisted runs remain readable.
    """

    if config.model.atmosphere_mode == "fixed_kpno":
        if config.inputs.atmosphere is None:  # Protected by config validation.
            raise InputError("fixed_kpno configuration has no atmosphere curve")
        curve = load_curve(
            config.inputs.atmosphere, "fixed unit-airmass atmosphere"
        )
        return (
            {str(prefix).lower(): "fixed" for prefix in required_prefixes},
            {"fixed": curve},
        )

    if config.inputs.site_mapping is None:  # Protected by config validation.
        raise InputError("site-specific configuration has no site mapping")
    prefix_to_site = load_site_mapping(config.inputs.site_mapping)
    required = tuple(dict.fromkeys(str(prefix).lower() for prefix in required_prefixes))
    missing_prefixes = [prefix for prefix in required if prefix not in prefix_to_site]
    if missing_prefixes:
        raise InputError(
            "site mapping lacks observed image prefix(es): "
            + ", ".join(missing_prefixes)
        )
    mapped_sites = set(prefix_to_site.values())
    missing_curves = sorted(mapped_sites - set(config.inputs.atmospheres))
    if missing_curves:
        raise InputError(
            "no atmosphere curve configured for mapped site(s): "
            + ", ".join(missing_curves)
        )
    curves = {
        site_name: load_curve(curve_config, f"{site_name} unit-airmass atmosphere")
        for site_name, curve_config in config.inputs.atmospheres.items()
    }
    return prefix_to_site, curves


def load_filter_curve(config: FitConfig, selected_filter: Optional[str] = None) -> SpectralCurve:
    """Load the configured curve for one exact filter label."""

    filter_name = selected_filter if selected_filter is not None else config.selected_filter
    if filter_name is None:
        raise InputError("a selected filter is required")
    try:
        curve_config = config.inputs.filter_curves[str(filter_name)]
    except KeyError as exc:
        raise InputError(f"no curve configured for filter {filter_name!r}") from exc
    return load_curve(curve_config, name=f"filter_{filter_name}")


_LEGACY_MAPPING_LINE = re.compile(
    r"^\s*['\"](?P<target>[^'\"]+)['\"]\s*:\s*['\"](?P<filename>[^'\"]+)['\"]\s*,?\s*$"
)


def load_target_mapping(path: PathLike) -> Dict[str, str]:
    """Load exact target aliases from YAML/JSON or legacy one-mapping-per-line text."""

    mapping_path = Path(path)
    if not mapping_path.is_file():
        raise InputError(f"target mapping does not exist: {mapping_path}")
    try:
        text = mapping_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise InputError(f"could not read target mapping {mapping_path}: {exc}") from exc
    loaded: object = None
    if mapping_path.suffix.lower() in {".yaml", ".yml", ".json"}:
        try:
            loaded = json.loads(text) if mapping_path.suffix.lower() == ".json" else yaml.safe_load(text)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise InputError(f"invalid target mapping {mapping_path}: {exc}") from exc
    if isinstance(loaded, Mapping):
        result = {str(key): str(value) for key, value in loaded.items()}
    else:
        result: Dict[str, str] = {}
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = _LEGACY_MAPPING_LINE.match(line)
            if match is None:
                raise InputError(f"invalid legacy target mapping line {mapping_path}:{line_number}")
            target, filename = match.group("target"), match.group("filename")
            if target in result:
                raise InputError(f"duplicate target alias {target!r} in {mapping_path}")
            result[target] = filename
    if not result:
        raise InputError(f"target mapping is empty: {mapping_path}")
    if any(not key or not value for key, value in result.items()):
        raise InputError(f"target mapping contains an empty alias or filename: {mapping_path}")
    return result


def _sorted_spectrum(wave: np.ndarray, flux: np.ndarray, name: str) -> Spectrum:
    if wave.shape != flux.shape or wave.ndim != 1:
        raise InputError(f"{name}: spectrum columns must be equal-length vectors")
    finite = np.isfinite(wave) & np.isfinite(flux)
    if not np.all(finite):
        raise InputError(f"{name}: spectrum contains nonfinite wavelength or flux")
    order = np.argsort(wave, kind="stable")
    wave, flux = wave[order], flux[order]
    if np.any(np.diff(wave) == 0):
        raise InputError(f"{name}: duplicate wavelengths are not allowed")
    return Spectrum(wave, flux, name)


def load_calspec_spectrum(path: PathLike, name: Optional[str] = None) -> Spectrum:
    """Load a CALSPEC FITS SCI table in Angstrom/FLAM, or a cgs two-column text file."""

    spectrum_path = Path(path)
    if not spectrum_path.is_file():
        raise InputError(f"spectrum does not exist: {spectrum_path}")
    spectrum_name = name or spectrum_path.stem
    if spectrum_path.suffix.lower() in {".fits", ".fit", ".fts"}:
        try:
            with fits.open(spectrum_path, memmap=False) as hdul:
                hdu = hdul["SCI"] if "SCI" in hdul else hdul[1]
                column_names = {item.upper(): item for item in hdu.columns.names}
                if "WAVELENGTH" not in column_names or "FLUX" not in column_names:
                    raise InputError(f"{spectrum_path}: FITS table requires WAVELENGTH and FLUX columns")
                wave_col = column_names["WAVELENGTH"]
                flux_col = column_names["FLUX"]
                wave = np.asarray(hdu.data[wave_col], dtype=np.float64)
                flux = np.asarray(hdu.data[flux_col], dtype=np.float64)
                wave_unit = str(hdu.columns[wave_col].unit or "").strip().lower()
                flux_unit = str(hdu.columns[flux_col].unit or "").strip().lower()
        except (OSError, IndexError, KeyError, TypeError) as exc:
            raise InputError(f"could not read CALSPEC FITS spectrum {spectrum_path}: {exc}") from exc
        if wave_unit not in {"angstrom", "angstroms", "ang", "aa"}:
            raise InputError(f"{spectrum_path}: unsupported CALSPEC wavelength unit {wave_unit!r}")
        if flux_unit not in {"flam", "erg/s/cm2/a", "erg/s/cm^2/a", "erg s-1 cm-2 angstrom-1"}:
            raise InputError(f"{spectrum_path}: unsupported CALSPEC flux unit {flux_unit!r}")
        return _sorted_spectrum(wave, flux, spectrum_name)
    wave, flux = _read_two_column_file(spectrum_path)
    return _sorted_spectrum(wave, flux, spectrum_name)


def load_target_spectra(config: FitConfig, target_names: Sequence[str]) -> Dict[str, Spectrum]:
    """Load spectra for exact selected-data target aliases."""

    mapping = load_target_mapping(config.inputs.target_mapping)
    missing = [str(target) for target in target_names if str(target) not in mapping]
    if missing:
        raise InputError("target mapping lacks exact alias(es): " + ", ".join(missing))
    spectra: Dict[str, Spectrum] = {}
    base = config.inputs.spectra_directory.resolve(strict=False)
    for target in target_names:
        exact = str(target)
        spectrum_path = Path(mapping[exact])
        if not spectrum_path.is_absolute():
            spectrum_path = base / spectrum_path
        spectra[exact] = load_calspec_spectrum(spectrum_path, name=exact)
    return spectra


def load_vega_spectrum(config: FitConfig) -> Spectrum:
    """Load the configured CALSPEC Vega reference spectrum."""

    return load_calspec_spectrum(config.inputs.vega_spectrum, name="alpha_lyr")
