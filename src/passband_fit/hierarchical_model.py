"""Hierarchical parameterization and posterior for passband fitting.

The sampler uses non-centered instrument effects and analytically marginalizes
the per-star Gaussian effects. Positive population scales, correlated star
scatter, and common extra scatter are sampled in log space; their priors are
stated on the physical scale and include the corresponding log-transform
Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, pi
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


VALID_MODES = ("fixed", "shift_only", "shift_width", "shift_width_warp")
Array = np.ndarray
SyntheticMagnitudeEvaluator = Callable[[Mapping[str, object]], Array]


@dataclass(frozen=True)
class NormalPrior:
    """A normal prior parameterized by its physical mean and standard deviation."""

    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("normal-prior mean must be finite and sigma must be positive")


@dataclass(frozen=True)
class HalfNormalPrior:
    """A zero-location half-normal prior on a positive physical scale."""

    scale: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("half-normal scale must be finite and positive")


@dataclass(frozen=True)
class ModelPriors:
    """Configurable priors for all supported hierarchical families."""

    mu_delta_lambda: NormalPrior = field(default_factory=lambda: NormalPrior(0.0, 50.0))
    tau_delta_lambda: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(30.0))
    mu_eta: NormalPrior = field(default_factory=lambda: NormalPrior(0.0, 0.05))
    tau_eta: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.03))
    mu_a: NormalPrior = field(default_factory=lambda: NormalPrior(0.0, 0.5))
    tau_a: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.3))
    mu_Z: NormalPrior = field(default_factory=lambda: NormalPrior(0.0, 1.0))
    tau_Z: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.2))
    sigma_star: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.03))
    sigma_extra: HalfNormalPrior = field(default_factory=lambda: HalfNormalPrior(0.05))

    @classmethod
    def from_mapping(cls, values: Optional[object]) -> "ModelPriors":
        """Construct priors from mappings or the package ``PriorsConfig``.

        Both inference names (``mu_delta_lambda``) and public configuration
        names (``shift_population_mean``) are accepted without importing the
        configuration module and creating a dependency cycle.
        """

        if values is None:
            return cls()
        defaults = cls()
        aliases = {
            "mu_delta_lambda": "shift_population_mean",
            "tau_delta_lambda": "shift_population_scale",
            "mu_eta": "log_width_population_mean",
            "tau_eta": "log_width_population_scale",
            "mu_a": "warp_population_mean",
            "tau_a": "warp_population_scale",
            "mu_Z": "zeropoint_population_mean",
            "tau_Z": "zeropoint_population_scale",
            "sigma_star": "star_correlated_scatter",
            "sigma_extra": "extra_scatter",
        }
        normal_names = ("mu_delta_lambda", "mu_eta", "mu_a", "mu_Z")
        half_names = (
            "tau_delta_lambda",
            "tau_eta",
            "tau_a",
            "tau_Z",
            "sigma_star",
            "sigma_extra",
        )
        kwargs: Dict[str, object] = {}

        def configured(name: str) -> object:
            if isinstance(values, Mapping):
                return values.get(name, values.get(aliases[name], getattr(defaults, name)))
            return getattr(values, name, getattr(values, aliases[name], getattr(defaults, name)))

        for name in normal_names:
            raw = configured(name)
            if isinstance(raw, NormalPrior):
                kwargs[name] = raw
            elif isinstance(raw, Mapping):
                kwargs[name] = NormalPrior(float(raw["mean"]), float(raw["sigma"]))
            elif hasattr(raw, "mean") and hasattr(raw, "sigma"):
                kwargs[name] = NormalPrior(float(raw.mean), float(raw.sigma))
            else:
                raise TypeError(f"prior {name!r} must be a mapping or NormalPrior")
        for name in half_names:
            raw = configured(name)
            if isinstance(raw, HalfNormalPrior):
                kwargs[name] = raw
            elif isinstance(raw, Mapping):
                kwargs[name] = HalfNormalPrior(float(raw["scale"]))
            elif hasattr(raw, "scale"):
                kwargs[name] = HalfNormalPrior(float(raw.scale))
            else:
                raise TypeError(f"prior {name!r} must be a mapping or HalfNormalPrior")
        return cls(**kwargs)


@dataclass(frozen=True)
class ParameterLayout:
    """Stable sampled-vector layout for one passband model mode.

    Every enabled instrument hierarchy occupies ``mu, log_tau, z[0:J]``.
    When targets are configured, ``log_sigma_star`` follows the instrument
    families. The Gaussian star offsets are marginalized analytically rather
    than sampled. ``log_sigma_extra`` is always last.
    """

    mode: str
    n_instruments: int
    instrument_names: Optional[Sequence[str]] = None
    n_targets: int = 0
    target_names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"unsupported passband mode {self.mode!r}; choose from {VALID_MODES}")
        if int(self.n_instruments) != self.n_instruments or self.n_instruments < 1:
            raise ValueError("n_instruments must be a positive integer")
        if self.instrument_names is None:
            names = tuple(f"instrument_{index}" for index in range(self.n_instruments))
        else:
            names = tuple(str(item) for item in self.instrument_names)
            if len(names) != self.n_instruments:
                raise ValueError("instrument_names length must equal n_instruments")
            if len(set(names)) != len(names):
                raise ValueError("instrument_names must be unique")
        object.__setattr__(self, "instrument_names", names)
        if int(self.n_targets) != self.n_targets or self.n_targets < 0:
            raise ValueError("n_targets must be a nonnegative integer")
        if self.target_names is None:
            targets = tuple(f"target_{index}" for index in range(self.n_targets))
        else:
            targets = tuple(str(item) for item in self.target_names)
            if len(targets) != self.n_targets:
                raise ValueError("target_names length must equal n_targets")
            if len(set(targets)) != len(targets):
                raise ValueError("target_names must be unique")
        object.__setattr__(self, "target_names", targets)

    @property
    def enabled_families(self) -> Tuple[str, ...]:
        families = []
        if self.mode != "fixed":
            families.append("delta_lambda")
        if self.mode in ("shift_width", "shift_width_warp"):
            families.append("eta")
        if self.mode == "shift_width_warp":
            families.append("a")
        families.append("Z")
        return tuple(families)

    @property
    def ndim(self) -> int:
        star_dimension = 1 if self.n_targets else 0
        return len(self.enabled_families) * (self.n_instruments + 2) + star_dimension + 1

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        result = []
        for family in self.enabled_families:
            result.extend((f"mu_{family}", f"log_tau_{family}"))
            result.extend(f"z_{family}[{name}]" for name in self.instrument_names)
        if self.n_targets:
            result.append("log_sigma_star")
        result.append("log_sigma_extra")
        return tuple(result)

    @property
    def sampled_names(self) -> Tuple[str, ...]:
        """Alias used by output and plotting code."""

        return self.parameter_names

    def family_slice(self, family: str) -> slice:
        """Return the slice containing a family's standard-normal deviations."""

        if family not in self.enabled_families:
            raise KeyError(f"family {family!r} is disabled in mode {self.mode!r}")
        offset = self.enabled_families.index(family) * (self.n_instruments + 2)
        return slice(offset + 2, offset + 2 + self.n_instruments)

    def family_indices(self, family: str) -> Tuple[int, int, slice]:
        """Return sampled indices for ``mu``, ``log_tau`` and ``z``."""

        z_slice = self.family_slice(family)
        return z_slice.start - 2, z_slice.start - 1, z_slice

    def star_indices(self) -> Tuple[int, slice]:
        """Return the ``log_sigma_star`` index and an empty compatibility slice."""

        if not self.n_targets:
            raise KeyError("star-correlated scatter is disabled when n_targets is zero")
        index = len(self.enabled_families) * (self.n_instruments + 2)
        return index, slice(index + 1, index + 1)

    def unpack(self, theta: Sequence[float]) -> Dict[str, object]:
        """Transform a sampled vector into physical hyper/instrument parameters."""

        vector = np.asarray(theta, dtype=np.float64)
        if vector.shape != (self.ndim,):
            raise ValueError(f"theta must have shape ({self.ndim},), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("theta must contain only finite values")

        result: Dict[str, object] = {}
        for family in self.enabled_families:
            mu_index, log_tau_index, z_slice = self.family_indices(family)
            log_tau = float(vector[log_tau_index])
            tau = _safe_exp(log_tau, f"tau_{family}")
            mu = float(vector[mu_index])
            z = vector[z_slice].copy()
            values = mu + tau * z
            if not np.all(np.isfinite(values)):
                raise ValueError(f"derived {family} parameters are nonfinite")
            result[f"mu_{family}"] = mu
            result[f"log_tau_{family}"] = log_tau
            result[f"tau_{family}"] = tau
            result[f"z_{family}"] = z
            result[family] = values

        if self.n_targets:
            log_sigma_star_index, _ = self.star_indices()
            log_sigma_star = float(vector[log_sigma_star_index])
            sigma_star = _safe_exp(log_sigma_star, "sigma_star")
            result["log_sigma_star"] = log_sigma_star
            result["sigma_star"] = sigma_star
        else:
            result["log_sigma_star"] = -np.inf
            result["sigma_star"] = 0.0

        log_sigma_extra = float(vector[-1])
        result["log_sigma_extra"] = log_sigma_extra
        result["sigma_extra"] = _safe_exp(log_sigma_extra, "sigma_extra")
        if "delta_lambda" in result:
            result["delta_lambda"] = np.asarray(result["delta_lambda"])
        else:
            result["delta_lambda"] = np.zeros(self.n_instruments)
        if "eta" in result:
            eta = np.asarray(result["eta"])
            if np.any(eta > 700.0):
                raise ValueError("derived widths overflow")
            width = np.exp(eta)
            if not np.all(np.isfinite(width)) or np.any(width <= 0):
                raise ValueError("derived widths must be positive and finite")
            result["width"] = width
        else:
            result["eta"] = np.zeros(self.n_instruments)
            result["width"] = np.ones(self.n_instruments)
        if "a" in result:
            result["warp"] = np.asarray(result["a"])
        else:
            result["a"] = np.zeros(self.n_instruments)
            result["warp"] = np.zeros(self.n_instruments)
        result["zeropoint"] = np.asarray(result["Z"])
        result["instrument_names"] = self.instrument_names
        result["target_names"] = self.target_names
        return result

    @property
    def derived_parameter_names(self) -> Tuple[str, ...]:
        """Names matching :meth:`derived_vector`, including physical scales."""

        names = []
        for family in self.enabled_families:
            names.extend((f"mu_{family}", f"tau_{family}"))
            names.extend(f"{family}[{name}]" for name in self.instrument_names)
            if family == "eta":
                names.extend(f"width[{name}]" for name in self.instrument_names)
        if self.n_targets:
            names.append("sigma_star")
        names.append("sigma_extra")
        return tuple(names)

    def derived_vector(self, theta: Sequence[float]) -> Array:
        """Return a flat physical-parameter vector for summaries and tables."""

        unpacked = self.unpack(theta)
        values = []
        for family in self.enabled_families:
            values.extend((unpacked[f"mu_{family}"], unpacked[f"tau_{family}"]))
            values.extend(np.asarray(unpacked[family], dtype=float))
            if family == "eta":
                values.extend(np.asarray(unpacked["width"], dtype=float))
        if self.n_targets:
            values.append(unpacked["sigma_star"])
        values.append(unpacked["sigma_extra"])
        return np.asarray(values, dtype=np.float64)

    def initial_vector(self, priors: Optional[ModelPriors] = None) -> Array:
        """Construct a finite, nominal non-centered starting vector."""

        priors = priors or ModelPriors()
        vector = np.zeros(self.ndim, dtype=np.float64)
        for family in self.enabled_families:
            mu_index, log_tau_index, _ = self.family_indices(family)
            mu_prior = getattr(priors, f"mu_{family}")
            tau_prior = getattr(priors, f"tau_{family}")
            vector[mu_index] = mu_prior.mean
            # The half-normal median is about 0.67449 scale and stays away from zero.
            vector[log_tau_index] = np.log(0.67448975 * tau_prior.scale)
        if self.n_targets:
            log_sigma_star_index, _ = self.star_indices()
            vector[log_sigma_star_index] = np.log(0.67448975 * priors.sigma_star.scale)
        vector[-1] = np.log(0.67448975 * priors.sigma_extra.scale)
        return vector


def _safe_exp(value: float, label: str) -> float:
    if not np.isfinite(value) or value > 700.0:
        raise ValueError(f"{label} is nonfinite or overflows")
    result = float(np.exp(value))
    if result <= 0.0 or not np.isfinite(result):
        raise ValueError(f"{label} must be positive and finite")
    return result


def _normal_logpdf(value: Union[float, Array], prior: NormalPrior) -> float:
    z = (np.asarray(value) - prior.mean) / prior.sigma
    return float(np.sum(-0.5 * z * z - log(prior.sigma) - 0.5 * log(2.0 * pi)))


def _standard_normal_logpdf(value: Array) -> float:
    return float(np.sum(-0.5 * value * value - 0.5 * log(2.0 * pi)))


def _log_halfnormal_at_log_value(log_value: float, prior: HalfNormalPrior) -> float:
    """Density in log-coordinate: log p(exp(x)) + x."""

    try:
        physical_value = _safe_exp(float(log_value), "positive scale")
    except ValueError:
        return -np.inf
    log_density_physical = (
        0.5 * log(2.0 / pi)
        - log(prior.scale)
        - 0.5 * (physical_value / prior.scale) ** 2
    )
    return float(log_density_physical + log_value)


def log_prior(
    theta: Sequence[float],
    layout: ParameterLayout,
    priors: Optional[ModelPriors] = None,
) -> float:
    """Evaluate the normalized log prior in sampled coordinates."""

    priors = priors or ModelPriors()
    vector = np.asarray(theta, dtype=np.float64)
    if vector.shape != (layout.ndim,) or not np.all(np.isfinite(vector)):
        return -np.inf
    total = 0.0
    for family in layout.enabled_families:
        mu_index, log_tau_index, z_slice = layout.family_indices(family)
        total += _normal_logpdf(vector[mu_index], getattr(priors, f"mu_{family}"))
        total += _log_halfnormal_at_log_value(
            vector[log_tau_index], getattr(priors, f"tau_{family}")
        )
        total += _standard_normal_logpdf(vector[z_slice])
    if layout.n_targets:
        log_sigma_star_index, _ = layout.star_indices()
        total += _log_halfnormal_at_log_value(
            vector[log_sigma_star_index], priors.sigma_star
        )
    total += _log_halfnormal_at_log_value(vector[-1], priors.sigma_extra)
    return float(total) if np.isfinite(total) else -np.inf


class HierarchicalPosterior:
    """Normalized heteroscedastic Gaussian posterior with an injected evaluator.

    ``synthetic_magnitude_evaluator`` receives the mapping returned by
    :meth:`ParameterLayout.unpack` and must return one finite synthetic
    magnitude per observation.  Each configured target has a shared additive
    residual with population width ``sigma_star``. The offsets are integrated
    out exactly, producing diagonal-plus-rank-one within-star covariance
    blocks without dense matrix operations.
    """

    def __init__(
        self,
        layout: ParameterLayout,
        instrument_index: Sequence[int],
        observed_magnitude: Sequence[float],
        magnitude_uncertainty: Sequence[float],
        synthetic_magnitude_evaluator: SyntheticMagnitudeEvaluator,
        priors: Optional[object] = None,
        target_index: Optional[Sequence[int]] = None,
    ) -> None:
        self.layout = layout
        self.instrument_index = np.asarray(instrument_index, dtype=np.int64)
        self.observed_magnitude = np.asarray(observed_magnitude, dtype=np.float64)
        self.magnitude_uncertainty = np.asarray(magnitude_uncertainty, dtype=np.float64)
        if not (
            self.instrument_index.ndim
            == self.observed_magnitude.ndim
            == self.magnitude_uncertainty.ndim
            == 1
        ):
            raise ValueError("observation arrays must be one-dimensional")
        if not (
            len(self.instrument_index)
            == len(self.observed_magnitude)
            == len(self.magnitude_uncertainty)
        ):
            raise ValueError("observation arrays must have equal lengths")
        if len(self.observed_magnitude) == 0:
            raise ValueError("at least one observation is required")
        if np.any(self.instrument_index < 0) or np.any(
            self.instrument_index >= layout.n_instruments
        ):
            raise ValueError("instrument indices fall outside the parameter layout")
        if not np.all(np.isfinite(self.observed_magnitude)):
            raise ValueError("observed magnitudes must be finite")
        if not np.all(np.isfinite(self.magnitude_uncertainty)) or np.any(
            self.magnitude_uncertainty <= 0
        ):
            raise ValueError("magnitude uncertainties must be finite and positive")
        if not callable(synthetic_magnitude_evaluator):
            raise TypeError("synthetic_magnitude_evaluator must be callable")
        if target_index is None:
            if layout.n_targets:
                raise ValueError("target_index is required when n_targets is nonzero")
            self.target_index = np.zeros(len(self.observed_magnitude), dtype=np.int64)
        else:
            self.target_index = np.asarray(target_index, dtype=np.int64)
            if self.target_index.shape != self.observed_magnitude.shape:
                raise ValueError("target_index must have one entry per observation")
            if layout.n_targets == 0:
                raise ValueError("target_index requires a layout with configured targets")
            if np.any(self.target_index < 0) or np.any(
                self.target_index >= layout.n_targets
            ):
                raise ValueError("target indices fall outside the parameter layout")
        self.synthetic_magnitude_evaluator = synthetic_magnitude_evaluator
        if isinstance(priors, ModelPriors):
            self.priors = priors
        else:
            self.priors = ModelPriors.from_mapping(priors)

    @property
    def n_observations(self) -> int:
        return len(self.observed_magnitude)

    def predict_without_star_offsets(self, theta: Sequence[float]) -> Array:
        """Return the observation mean before the marginalized star offsets."""

        parameters = self.layout.unpack(theta)
        synthetic = np.asarray(
            self.synthetic_magnitude_evaluator(parameters), dtype=np.float64
        )
        if synthetic.shape != self.observed_magnitude.shape:
            raise ValueError(
                "synthetic evaluator must return one magnitude per observation; "
                f"expected {self.observed_magnitude.shape}, got {synthetic.shape}"
            )
        if not np.all(np.isfinite(synthetic)):
            raise ValueError("synthetic evaluator returned nonfinite magnitudes")
        zeropoints = np.asarray(parameters["zeropoint"], dtype=np.float64)
        return synthetic + zeropoints[self.instrument_index]

    def conditional_star_distribution(
        self, theta: Sequence[float]
    ) -> Tuple[Array, Array]:
        """Return conditional means and variances of the marginalized offsets."""

        if not self.layout.n_targets:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        parameters = self.layout.unpack(theta)
        base_prediction = self.predict_without_star_offsets(theta)
        variance = self.magnitude_uncertainty**2 + float(parameters["sigma_extra"]) ** 2
        precision = 1.0 / variance
        residual = self.observed_magnitude - base_prediction
        precision_sum = np.bincount(
            self.target_index, weights=precision, minlength=self.layout.n_targets
        )
        weighted_residual_sum = np.bincount(
            self.target_index,
            weights=residual * precision,
            minlength=self.layout.n_targets,
        )
        sigma_star_squared = float(parameters["sigma_star"]) ** 2
        denominator = 1.0 + sigma_star_squared * precision_sum
        conditional_variance = sigma_star_squared / denominator
        conditional_mean = (
            sigma_star_squared * weighted_residual_sum / denominator
        )
        return conditional_mean, conditional_variance

    def sample_conditional_star_offsets(
        self, theta: Sequence[float], generator: np.random.Generator
    ) -> Array:
        """Draw physical star offsets conditional on one marginal posterior draw."""

        if not isinstance(generator, np.random.Generator):
            raise TypeError("generator must be a numpy.random.Generator")
        mean, variance = self.conditional_star_distribution(theta)
        return mean + generator.normal(size=len(mean)) * np.sqrt(variance)

    def predict(
        self,
        theta: Sequence[float],
        star_offsets: Optional[Sequence[float]] = None,
    ) -> Array:
        """Return the mean using conditional means or supplied physical offsets."""

        prediction = self.predict_without_star_offsets(theta)
        if not self.layout.n_targets:
            return prediction
        if star_offsets is None:
            offsets, _ = self.conditional_star_distribution(theta)
        else:
            offsets = np.asarray(star_offsets, dtype=np.float64)
            if offsets.shape != (self.layout.n_targets,) or not np.all(np.isfinite(offsets)):
                raise ValueError("star_offsets must contain one finite value per target")
        return prediction + offsets[self.target_index]

    def marginal_covariance(self, theta: Sequence[float]) -> Array:
        """Return the covariance after analytically marginalizing star offsets.

        This is primarily a diagnostic expression of the model. Sampling uses
        an equivalent diagonal-plus-rank-one expression without dense matrices.
        """

        parameters = self.layout.unpack(theta)
        diagonal = self.magnitude_uncertainty**2 + float(parameters["sigma_extra"]) ** 2
        covariance = np.diag(diagonal)
        if self.layout.n_targets:
            same_star = self.target_index[:, None] == self.target_index[None, :]
            covariance = covariance + float(parameters["sigma_star"]) ** 2 * same_star
        return covariance

    def log_likelihood(self, theta: Sequence[float]) -> float:
        """Evaluate the full normalized Gaussian log likelihood."""

        try:
            parameters = self.layout.unpack(theta)
            predicted = self.predict_without_star_offsets(theta)
        except (FloatingPointError, OverflowError, ValueError):
            return -np.inf
        sigma_extra = float(parameters["sigma_extra"])
        variance = self.magnitude_uncertainty**2 + sigma_extra**2
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            return -np.inf
        residual = self.observed_magnitude - predicted
        precision = 1.0 / variance
        quadratic = np.sum(residual * residual * precision)
        log_determinant = np.sum(np.log(variance))
        if self.layout.n_targets:
            precision_sum = np.bincount(
                self.target_index, weights=precision, minlength=self.layout.n_targets
            )
            weighted_residual_sum = np.bincount(
                self.target_index,
                weights=residual * precision,
                minlength=self.layout.n_targets,
            )
            sigma_star_squared = float(parameters["sigma_star"]) ** 2
            denominator = 1.0 + sigma_star_squared * precision_sum
            quadratic -= np.sum(
                sigma_star_squared * weighted_residual_sum**2 / denominator
            )
            log_determinant += np.sum(np.log(denominator))
        result = -0.5 * (
            quadratic + log_determinant + self.n_observations * log(2.0 * pi)
        )
        return float(result) if np.isfinite(result) else -np.inf

    def log_prior(self, theta: Sequence[float]) -> float:
        return log_prior(theta, self.layout, self.priors)

    def __call__(self, theta: Sequence[float]) -> float:
        prior_value = self.log_prior(theta)
        if not np.isfinite(prior_value):
            return -np.inf
        likelihood_value = self.log_likelihood(theta)
        if not np.isfinite(likelihood_value):
            return -np.inf
        return float(prior_value + likelihood_value)
