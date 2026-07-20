"""Strict, versioned configuration loading for passband fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import yaml


PathLike = Union[str, Path]


class ConfigError(ValueError):
    """Raised when a configuration file violates the public schema."""


@dataclass(frozen=True)
class ObservationColumns:
    target: str = "star"
    filter: str = "filter"
    image: str = "image"
    airmass: str = "airmass"
    flux: str = "flux"
    flux_uncertainty: str = "flux_unc"


@dataclass(frozen=True)
class ObservationInputConfig:
    path: Path
    format: str = "whitespace"
    columns: ObservationColumns = field(default_factory=ObservationColumns)
    instrument_regex: str = r"^(?P<instrument>.+?)-\d{8}(?:-|$)"
    site_prefix_length: int = 3


@dataclass(frozen=True)
class CurveConfig:
    path: Path
    wavelength_unit: str
    value_semantics: str
    wavelength_convention: str = "vacuum"
    duplicate_policy: str = "error"
    negative_tolerance: float = 0.0
    sentinel_values: Tuple[float, ...] = ()


@dataclass(frozen=True)
class InputsConfig:
    observations: ObservationInputConfig
    filter_curves: Mapping[str, CurveConfig]
    ccd: CurveConfig
    aluminum_squared: CurveConfig
    atmosphere: CurveConfig
    spectra_directory: Path
    target_mapping: Path
    vega_spectrum: Path


@dataclass(frozen=True)
class SelectionConfig:
    snr_min: float = 10.0
    require_positive_flux: bool = True


@dataclass(frozen=True)
class GridConfig:
    minimum_angstrom: float = 3000.0
    maximum_angstrom: float = 11000.0
    spacing_angstrom: float = 2.0
    response_threshold: float = 1.0e-6
    convergence_tolerance_mag: float = 5.0e-4


@dataclass(frozen=True)
class ModelConfig:
    magnitude_system: str = "vega"
    passband_mode: str = "shift_only"
    pivot_wavelength_angstrom: float = 4800.0
    warp_scale_angstrom: float = 1200.0
    atmosphere_mode: str = "fixed_kpno"
    airmass_round_decimals: Optional[int] = None


@dataclass(frozen=True)
class BoundsConfig:
    shift_abs_max_angstrom: float = 250.0
    width_min: float = 0.7
    width_max: float = 1.3
    warp_abs_max: float = 2.0


@dataclass(frozen=True)
class NormalPrior:
    mean: float
    sigma: float


@dataclass(frozen=True)
class HalfNormalPrior:
    scale: float


@dataclass(frozen=True)
class PriorsConfig:
    shift_population_mean: NormalPrior = field(default_factory=lambda: NormalPrior(0.0, 50.0))
    shift_population_scale: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(30.0))
    log_width_population_mean: NormalPrior = field(default_factory=lambda: NormalPrior(0.0, 0.05))
    log_width_population_scale: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.03))
    warp_population_mean: NormalPrior = field(default_factory=lambda: NormalPrior(0.0, 0.5))
    warp_population_scale: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.3))
    zeropoint_population_mean: NormalPrior = field(default_factory=lambda: NormalPrior(0.0, 1.0))
    zeropoint_population_scale: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.2))
    star_correlated_scatter: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.03))
    extra_scatter: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.05))


@dataclass(frozen=True)
class SamplerConfig:
    num_warmup: int = 1000
    num_samples: int = 5000
    num_chains: int = 4
    thinning: int = 1
    random_seed: int = 12345
    chain_method: str = "parallel"
    progress: bool = True
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = False


@dataclass(frozen=True)
class PlottingConfig:
    posterior_draws: int = 500
    formats: Tuple[str, ...] = ("png",)


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    run_label: Optional[str] = None


@dataclass(frozen=True)
class FitConfig:
    schema_version: int
    selected_filter: Optional[str]
    inputs: InputsConfig
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    priors: PriorsConfig = field(default_factory=PriorsConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(Path("runs")))


def _mapping(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    return dict(value)


def _reject_unknown(values: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(unknown)}")


def _path(value: Any, base: Path, where: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"{where} must be a nonempty path")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = base / result
    return result.resolve(strict=False)


def _curve(value: Any, base: Path, where: str) -> CurveConfig:
    raw = _mapping(value, where)
    _reject_unknown(
        raw,
        ["path", "wavelength_unit", "value_semantics", "wavelength_convention",
         "duplicate_policy", "negative_tolerance", "sentinel_values"],
        where,
    )
    for required in ("path", "wavelength_unit", "value_semantics"):
        if required not in raw:
            raise ConfigError(f"missing required key {where}.{required}")
    convention = str(raw.get("wavelength_convention", "vacuum")).lower()
    if convention not in {"air", "vacuum"}:
        raise ConfigError(f"{where}.wavelength_convention must be 'air' or 'vacuum'")
    unit = str(raw["wavelength_unit"]).lower()
    if unit not in {"angstrom", "angstroms", "aa", "nm", "micron", "um"}:
        raise ConfigError(f"unsupported wavelength unit in {where}: {unit}")
    semantics = str(raw["value_semantics"]).lower()
    if semantics not in {"fraction", "percent", "extinction_mag_per_airmass"}:
        raise ConfigError(f"unsupported value semantics in {where}: {semantics}")
    duplicates = str(raw.get("duplicate_policy", "error")).lower()
    if duplicates not in {"error", "mean", "first", "last"}:
        raise ConfigError(f"unsupported duplicate_policy in {where}: {duplicates}")
    tolerance = float(raw.get("negative_tolerance", 0.0))
    if tolerance < 0:
        raise ConfigError(f"{where}.negative_tolerance must be nonnegative")
    sentinels = tuple(float(item) for item in raw.get("sentinel_values", ()))
    return CurveConfig(
        path=_path(raw["path"], base, f"{where}.path"),
        wavelength_unit=unit,
        value_semantics=semantics,
        wavelength_convention=convention,
        duplicate_policy=duplicates,
        negative_tolerance=tolerance,
        sentinel_values=sentinels,
    )


def _observations(value: Any, base: Path) -> ObservationInputConfig:
    raw = _mapping(value, "inputs.observations")
    _reject_unknown(raw, ["path", "format", "columns", "instrument_regex", "site_prefix_length"],
                    "inputs.observations")
    if "path" not in raw:
        raise ConfigError("missing required key inputs.observations.path")
    columns_raw = _mapping(raw.get("columns", {}), "inputs.observations.columns")
    column_names = ["target", "filter", "image", "airmass", "flux", "flux_uncertainty"]
    _reject_unknown(columns_raw, column_names, "inputs.observations.columns")
    columns = ObservationColumns(**{key: str(value) for key, value in columns_raw.items()})
    table_format = str(raw.get("format", "whitespace")).lower()
    if table_format not in {"whitespace", "csv", "tsv"}:
        raise ConfigError("inputs.observations.format must be whitespace, csv, or tsv")
    prefix_length = int(raw.get("site_prefix_length", 3))
    if prefix_length <= 0:
        raise ConfigError("inputs.observations.site_prefix_length must be positive")
    return ObservationInputConfig(
        path=_path(raw["path"], base, "inputs.observations.path"),
        format=table_format,
        columns=columns,
        instrument_regex=str(raw.get("instrument_regex", ObservationInputConfig.instrument_regex)),
        site_prefix_length=prefix_length,
    )


def _normal(value: Any, default: NormalPrior, where: str) -> NormalPrior:
    if value is None:
        return default
    raw = _mapping(value, where)
    _reject_unknown(raw, ["mean", "sigma"], where)
    result = NormalPrior(float(raw.get("mean", default.mean)), float(raw.get("sigma", default.sigma)))
    if not math.isfinite(result.mean) or not math.isfinite(result.sigma) or result.sigma <= 0:
        raise ConfigError(f"{where} mean must be finite and sigma positive/finite")
    return result


def _half_normal(value: Any, default: HalfNormalPrior, where: str) -> HalfNormalPrior:
    if value is None:
        return default
    raw = _mapping(value, where)
    _reject_unknown(raw, ["scale"], where)
    result = HalfNormalPrior(float(raw.get("scale", default.scale)))
    if not math.isfinite(result.scale) or result.scale <= 0:
        raise ConfigError(f"{where}.scale must be positive and finite")
    return result


def load_config(path: PathLike, selected_filter: Optional[str] = None) -> FitConfig:
    """Load and strictly validate YAML, resolving paths relative to its location."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw_loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"could not load configuration {config_path}: {exc}") from exc
    raw = _mapping(raw_loaded, "configuration root")
    root_keys = ["schema_version", "selected_filter", "inputs", "selection", "grid", "model",
                 "bounds", "priors", "sampler", "plotting", "output"]
    _reject_unknown(raw, root_keys, "configuration root")
    if int(raw.get("schema_version", -1)) != 1:
        raise ConfigError("schema_version must be 1")
    if "inputs" not in raw:
        raise ConfigError("missing required key inputs")
    base = config_path.parent

    inputs_raw = _mapping(raw["inputs"], "inputs")
    input_keys = ["observations", "filter_curves", "ccd", "aluminum_squared", "atmosphere",
                  "spectra_directory", "target_mapping", "vega_spectrum"]
    _reject_unknown(inputs_raw, input_keys, "inputs")
    missing = [key for key in input_keys if key not in inputs_raw]
    if missing:
        raise ConfigError(f"missing required input key(s): {', '.join(missing)}")
    filters_raw = _mapping(inputs_raw["filter_curves"], "inputs.filter_curves")
    if not filters_raw:
        raise ConfigError("inputs.filter_curves must not be empty")
    filter_curves = {str(key): _curve(value, base, f"inputs.filter_curves.{key}")
                     for key, value in filters_raw.items()}
    inputs = InputsConfig(
        observations=_observations(inputs_raw["observations"], base),
        filter_curves=filter_curves,
        ccd=_curve(inputs_raw["ccd"], base, "inputs.ccd"),
        aluminum_squared=_curve(inputs_raw["aluminum_squared"], base, "inputs.aluminum_squared"),
        atmosphere=_curve(inputs_raw["atmosphere"], base, "inputs.atmosphere"),
        spectra_directory=_path(inputs_raw["spectra_directory"], base, "inputs.spectra_directory"),
        target_mapping=_path(inputs_raw["target_mapping"], base, "inputs.target_mapping"),
        vega_spectrum=_path(inputs_raw["vega_spectrum"], base, "inputs.vega_spectrum"),
    )

    selection_raw = _mapping(raw.get("selection", {}), "selection")
    _reject_unknown(selection_raw, ["snr_min", "require_positive_flux"], "selection")
    selection = SelectionConfig(**selection_raw)
    if not math.isfinite(selection.snr_min) or selection.snr_min < 0:
        raise ConfigError("selection.snr_min must be finite and nonnegative")
    if not selection.require_positive_flux:
        raise ConfigError("selection.require_positive_flux=false is unsupported because magnitudes require positive flux")

    grid_raw = _mapping(raw.get("grid", {}), "grid")
    _reject_unknown(grid_raw, ["minimum_angstrom", "maximum_angstrom", "spacing_angstrom",
                               "response_threshold", "convergence_tolerance_mag"], "grid")
    grid = GridConfig(**grid_raw)
    if not all(math.isfinite(value) for value in (
        grid.minimum_angstrom, grid.maximum_angstrom, grid.spacing_angstrom,
        grid.response_threshold, grid.convergence_tolerance_mag,
    )):
        raise ConfigError("grid numeric values must be finite")
    if not (grid.minimum_angstrom < grid.maximum_angstrom and grid.spacing_angstrom > 0):
        raise ConfigError("grid limits must increase and spacing_angstrom must be positive")
    if not 0 < grid.response_threshold < 1:
        raise ConfigError("grid.response_threshold must lie between zero and one")
    if grid.convergence_tolerance_mag <= 0:
        raise ConfigError("grid.convergence_tolerance_mag must be positive")

    model_raw = _mapping(raw.get("model", {}), "model")
    _reject_unknown(model_raw, ["magnitude_system", "passband_mode", "pivot_wavelength_angstrom",
                                "warp_scale_angstrom", "atmosphere_mode", "airmass_round_decimals"],
                    "model")
    model = ModelConfig(**model_raw)
    if model.magnitude_system not in {"vega", "ab"}:
        raise ConfigError("model.magnitude_system must be vega or ab")
    if model.passband_mode not in {"shift_only", "shift_width", "shift_width_warp"}:
        raise ConfigError("unsupported model.passband_mode")
    if not math.isfinite(model.pivot_wavelength_angstrom) or model.pivot_wavelength_angstrom <= 0:
        raise ConfigError("model.pivot_wavelength_angstrom must be positive and finite")
    if not math.isfinite(model.warp_scale_angstrom) or model.warp_scale_angstrom <= 0:
        raise ConfigError("model.warp_scale_angstrom must be positive and finite")
    if model.atmosphere_mode != "fixed_kpno":
        raise ConfigError("only model.atmosphere_mode=fixed_kpno is implemented")
    if model.airmass_round_decimals is not None:
        raise ConfigError("airmass rounding is not implemented; use null for exact airmasses")

    bounds_raw = _mapping(raw.get("bounds", {}), "bounds")
    _reject_unknown(bounds_raw, ["shift_abs_max_angstrom", "width_min", "width_max", "warp_abs_max"],
                    "bounds")
    bounds = BoundsConfig(**bounds_raw)
    if not all(math.isfinite(value) for value in (
        bounds.shift_abs_max_angstrom, bounds.width_min, bounds.width_max,
        bounds.warp_abs_max,
    )):
        raise ConfigError("bound values must be finite")
    if bounds.shift_abs_max_angstrom <= 0 or bounds.warp_abs_max <= 0:
        raise ConfigError("shift and warp absolute bounds must be positive")
    if not 0 < bounds.width_min < bounds.width_max:
        raise ConfigError("bounds must satisfy 0 < width_min < width_max")

    defaults = PriorsConfig()
    priors_raw = _mapping(raw.get("priors", {}), "priors")
    prior_keys = list(asdict(defaults))
    _reject_unknown(priors_raw, prior_keys, "priors")
    priors = PriorsConfig(
        shift_population_mean=_normal(priors_raw.get("shift_population_mean"), defaults.shift_population_mean,
                                      "priors.shift_population_mean"),
        shift_population_scale=_half_normal(priors_raw.get("shift_population_scale"), defaults.shift_population_scale,
                                            "priors.shift_population_scale"),
        log_width_population_mean=_normal(priors_raw.get("log_width_population_mean"), defaults.log_width_population_mean,
                                          "priors.log_width_population_mean"),
        log_width_population_scale=_half_normal(priors_raw.get("log_width_population_scale"), defaults.log_width_population_scale,
                                                "priors.log_width_population_scale"),
        warp_population_mean=_normal(priors_raw.get("warp_population_mean"), defaults.warp_population_mean,
                                     "priors.warp_population_mean"),
        warp_population_scale=_half_normal(priors_raw.get("warp_population_scale"), defaults.warp_population_scale,
                                           "priors.warp_population_scale"),
        zeropoint_population_mean=_normal(priors_raw.get("zeropoint_population_mean"), defaults.zeropoint_population_mean,
                                          "priors.zeropoint_population_mean"),
        zeropoint_population_scale=_half_normal(priors_raw.get("zeropoint_population_scale"), defaults.zeropoint_population_scale,
                                                "priors.zeropoint_population_scale"),
        star_correlated_scatter=_half_normal(priors_raw.get("star_correlated_scatter"), defaults.star_correlated_scatter,
                                             "priors.star_correlated_scatter"),
        extra_scatter=_half_normal(priors_raw.get("extra_scatter"), defaults.extra_scatter,
                                   "priors.extra_scatter"),
    )

    sampler_raw = _mapping(raw.get("sampler", {}), "sampler")
    sampler_keys = list(asdict(SamplerConfig()))
    _reject_unknown(sampler_raw, sampler_keys, "sampler")
    sampler = SamplerConfig(**sampler_raw)
    integer_fields = {
        "num_warmup": sampler.num_warmup,
        "num_samples": sampler.num_samples,
        "num_chains": sampler.num_chains,
        "thinning": sampler.thinning,
        "random_seed": sampler.random_seed,
        "max_tree_depth": sampler.max_tree_depth,
    }
    invalid_integers = [
        name for name, value in integer_fields.items()
        if not isinstance(value, int) or isinstance(value, bool)
    ]
    if invalid_integers:
        raise ConfigError(
            "sampler integer field(s) must be integers: " + ", ".join(invalid_integers)
        )
    if sampler.num_warmup < 0:
        raise ConfigError("sampler.num_warmup must be nonnegative")
    if sampler.num_samples <= 0 or sampler.num_chains <= 0 or sampler.thinning <= 0:
        raise ConfigError("sampler samples, chains, and thinning must be positive")
    if sampler.random_seed < 0 or sampler.random_seed > 2**32 - 1:
        raise ConfigError("sampler.random_seed must lie in [0, 2**32 - 1]")
    if sampler.chain_method not in {"parallel", "sequential", "vectorized"}:
        raise ConfigError(
            "sampler.chain_method must be parallel, sequential, or vectorized"
        )
    if not isinstance(sampler.progress, bool):
        raise ConfigError("sampler.progress must be a boolean")
    if (not isinstance(sampler.target_accept_prob, (int, float))
            or isinstance(sampler.target_accept_prob, bool)
            or not math.isfinite(sampler.target_accept_prob)
            or not 0.0 < sampler.target_accept_prob < 1.0):
        raise ConfigError("sampler.target_accept_prob must be finite and lie between zero and one")
    if sampler.max_tree_depth <= 0:
        raise ConfigError("sampler.max_tree_depth must be positive")
    if not isinstance(sampler.dense_mass, bool):
        raise ConfigError("sampler.dense_mass must be a boolean")

    plotting_raw = _mapping(raw.get("plotting", {}), "plotting")
    _reject_unknown(plotting_raw, ["posterior_draws", "formats"], "plotting")
    if "formats" in plotting_raw:
        plotting_raw["formats"] = tuple(str(item) for item in plotting_raw["formats"])
    plotting = PlottingConfig(**plotting_raw)
    if plotting.posterior_draws <= 0 or not plotting.formats:
        raise ConfigError("plotting posterior_draws and formats must be nonempty/positive")
    if any(item.lower() != "png" for item in plotting.formats):
        raise ConfigError("only PNG plotting output is currently supported")

    output_raw = _mapping(raw.get("output", {}), "output")
    _reject_unknown(output_raw, ["directory", "run_label"], "output")
    output = OutputConfig(
        directory=_path(output_raw.get("directory", "runs"), base, "output.directory"),
        run_label=None if output_raw.get("run_label") is None else str(output_raw["run_label"]),
    )

    effective_filter = selected_filter if selected_filter is not None else raw.get("selected_filter")
    if effective_filter is not None:
        effective_filter = str(effective_filter)
        if effective_filter not in filter_curves:
            raise ConfigError(f"selected filter {effective_filter!r} has no configured curve")
    return FitConfig(
        schema_version=1,
        selected_filter=effective_filter,
        inputs=inputs,
        selection=selection,
        grid=grid,
        model=model,
        bounds=bounds,
        priors=priors,
        sampler=sampler,
        plotting=plotting,
        output=output,
    )


def config_to_dict(config: FitConfig, paths_as_strings: bool = True) -> Dict[str, Any]:
    """Return a serialization-safe representation of a resolved configuration."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value) if paths_as_strings else value
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(asdict(config))
