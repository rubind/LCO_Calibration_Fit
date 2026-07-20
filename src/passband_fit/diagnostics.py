"""Posterior summaries, residual metrics, model criteria, and predictive checks."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _sample_matrix(samples: np.ndarray, names: Sequence[str]) -> np.ndarray:
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("samples must have shape (draws, parameters)")
    if matrix.shape[0] < 1 or matrix.shape[1] != len(names):
        raise ValueError("sample dimensions do not match the supplied parameter names")
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("samples must be finite")
    return matrix


def parameter_summary(
    samples: np.ndarray,
    names: Sequence[str],
    log_probability: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Summarize scalar posterior samples in a machine-readable table."""

    matrix = _sample_matrix(samples, names)
    quantiles = np.percentile(matrix, [16.0, 50.0, 84.0], axis=0)
    result = pd.DataFrame(
        {
            "parameter": tuple(names),
            "median": quantiles[1],
            "q16": quantiles[0],
            "q84": quantiles[2],
            "mean": np.mean(matrix, axis=0),
            "std": np.std(matrix, axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1]),
        }
    )
    result["maximum_posterior"] = np.nan
    if log_probability is not None:
        logp = np.asarray(log_probability, dtype=np.float64)
        if logp.shape != (matrix.shape[0],):
            raise ValueError("log_probability must have one entry per posterior draw")
        finite = np.isfinite(logp)
        if not np.any(finite):
            raise ValueError("log_probability contains no finite values")
        best = int(np.nanargmax(np.where(finite, logp, np.nan)))
        result["maximum_posterior"] = matrix[best]
    return result.set_index("parameter")


def transform_samples(samples: np.ndarray, transform: object, output_dimension: int) -> np.ndarray:
    """Apply a vector transform (for example ``layout.derived_vector``) by draw."""

    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim != 2 or output_dimension < 1 or not callable(transform):
        raise ValueError("samples must be 2-D, transform callable, and output_dimension positive")
    result = np.empty((len(matrix), output_dimension), dtype=np.float64)
    for index, row in enumerate(matrix):
        transformed = np.asarray(transform(row), dtype=np.float64)
        if transformed.shape != (output_dimension,) or not np.all(np.isfinite(transformed)):
            raise ValueError("transform returned an invalid derived vector")
        result[index] = transformed
    return result


def posterior_matrices(
    samples: np.ndarray, names: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return labeled posterior covariance and correlation matrices."""

    matrix = _sample_matrix(samples, names)
    if len(matrix) < 2:
        raise ValueError("at least two posterior draws are required for covariance")
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    standard_deviation = np.sqrt(np.diag(covariance))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0,
    )
    labels = list(names)
    return (
        pd.DataFrame(covariance, index=labels, columns=labels),
        pd.DataFrame(correlation, index=labels, columns=labels),
    )


def residual_metrics(
    residuals: Sequence[float], uncertainties: Optional[Sequence[float]] = None
) -> Dict[str, float]:
    """Compute RMS, weighted RMS, median, and Gaussian-scaled MAD."""

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("residuals must be a nonempty finite one-dimensional array")
    rms = float(np.sqrt(np.mean(values**2)))
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    if uncertainties is None:
        weighted_rms = rms
    else:
        sigma = np.asarray(uncertainties, dtype=np.float64)
        if sigma.shape != values.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("uncertainties must match residuals and be finite and positive")
        weights = 1.0 / sigma**2
        weighted_rms = float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))
    return {
        "n": int(len(values)),
        "rms": rms,
        "weighted_rms": weighted_rms,
        "median": median,
        "mad": mad,
        "mean": float(np.mean(values)),
    }


def information_criteria(
    maximum_log_likelihood: float, n_parameters: int, n_observations: int
) -> Dict[str, float]:
    """Compute AIC and BIC from a clearly identified maximum likelihood value."""

    if not np.isfinite(maximum_log_likelihood):
        raise ValueError("maximum_log_likelihood must be finite")
    if n_parameters < 0 or n_observations < 1:
        raise ValueError("n_parameters must be nonnegative and n_observations positive")
    return {
        "maximum_log_likelihood": float(maximum_log_likelihood),
        "n_parameters": int(n_parameters),
        "n_observations": int(n_observations),
        "aic": float(2 * n_parameters - 2 * maximum_log_likelihood),
        "bic": float(n_parameters * np.log(n_observations) - 2 * maximum_log_likelihood),
    }


def posterior_predictive_draws(
    model_mean_samples: np.ndarray,
    measurement_uncertainty: Sequence[float],
    sigma_extra_samples: Sequence[float],
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate replicated observed magnitudes for posterior model means."""

    means = np.asarray(model_mean_samples, dtype=np.float64)
    measurement = np.asarray(measurement_uncertainty, dtype=np.float64)
    extra = np.asarray(sigma_extra_samples, dtype=np.float64)
    if means.ndim != 2 or not np.all(np.isfinite(means)):
        raise ValueError("model_mean_samples must be a finite (draws, observations) array")
    if measurement.shape != (means.shape[1],) or np.any(measurement <= 0) or not np.all(
        np.isfinite(measurement)
    ):
        raise ValueError("measurement_uncertainty must be positive with one value per observation")
    if extra.shape != (means.shape[0],) or np.any(extra <= 0) or not np.all(np.isfinite(extra)):
        raise ValueError("sigma_extra_samples must be positive with one value per draw")
    sigma = np.sqrt(measurement[None, :] ** 2 + extra[:, None] ** 2)
    return means + np.random.default_rng(random_seed).normal(size=means.shape) * sigma


def linear_trend(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the ordinary least-squares residual slope, or zero for constant x."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.shape != y_values.shape or x_values.ndim != 1 or len(x_values) < 2:
        raise ValueError("x and y must be equal-length one-dimensional arrays")
    centered = x_values - np.mean(x_values)
    denominator = float(np.sum(centered**2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(centered * (y_values - np.mean(y_values))) / denominator)


def posterior_predictive_statistics(
    predictive_residuals: np.ndarray,
    x: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Compute per-draw scatter, location, and optional residual trend."""

    residuals = np.asarray(predictive_residuals, dtype=np.float64)
    if residuals.ndim != 2 or not np.all(np.isfinite(residuals)):
        raise ValueError("predictive_residuals must be a finite (draws, observations) array")
    result: Dict[str, np.ndarray] = {
        "mean": np.mean(residuals, axis=1),
        "rms": np.sqrt(np.mean(residuals**2, axis=1)),
        "std": np.std(residuals, axis=1, ddof=1) if residuals.shape[1] > 1 else np.zeros(len(residuals)),
    }
    if x is not None:
        x_values = np.asarray(x, dtype=np.float64)
        if x_values.shape != (residuals.shape[1],) or not np.all(np.isfinite(x_values)):
            raise ValueError("x must contain one finite value per observation")
        result["slope"] = np.asarray([linear_trend(x_values, row) for row in residuals])
    return pd.DataFrame(result)


def posterior_predictive_p_values(
    observed_residuals: Sequence[float],
    predictive_residuals: np.ndarray,
    x: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Return two-sided posterior-predictive tail areas for simple statistics."""

    observed = np.asarray(observed_residuals, dtype=np.float64)
    predictive = np.asarray(predictive_residuals, dtype=np.float64)
    if predictive.ndim != 2 or observed.shape != (predictive.shape[1],):
        raise ValueError("observed and predictive residual dimensions are incompatible")
    simulated = posterior_predictive_statistics(predictive, x=x)
    observed_statistics = {
        "mean": float(np.mean(observed)),
        "rms": float(np.sqrt(np.mean(observed**2))),
        "std": float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0,
    }
    if x is not None:
        observed_statistics["slope"] = linear_trend(x, observed)
    result = {}
    for name, value in observed_statistics.items():
        distribution = simulated[name].to_numpy()
        lower = np.mean(distribution <= value)
        upper = np.mean(distribution >= value)
        result[name] = float(min(1.0, 2.0 * min(lower, upper)))
    return result


def binned_residual_trend(
    x: Sequence[float],
    residuals: Sequence[float],
    bins: Union[int, Sequence[float]] = 10,
) -> pd.DataFrame:
    """Summarize mean residual and standard error in fixed x bins."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(residuals, dtype=np.float64)
    if x_values.shape != y_values.shape or x_values.ndim != 1 or len(x_values) == 0:
        raise ValueError("x and residuals must be equal-length nonempty one-dimensional arrays")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and residuals must be finite")
    if np.isscalar(bins):
        count = int(bins)
        if count < 1:
            raise ValueError("bins must be positive")
        low, high = float(np.min(x_values)), float(np.max(x_values))
        if low == high:
            edges = np.array([low - 0.5, high + 0.5])
        else:
            edges = np.linspace(low, high, count + 1)
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("bin edges must be a strictly increasing one-dimensional array")
    bin_index = np.digitize(x_values, edges[1:-1], right=False)
    rows = []
    for index in range(len(edges) - 1):
        selected = y_values[bin_index == index]
        if not len(selected):
            continue
        rows.append(
            {
                "bin_left": edges[index],
                "bin_right": edges[index + 1],
                "x_center": float(np.mean(x_values[bin_index == index])),
                "mean_residual": float(np.mean(selected)),
                "standard_error": (
                    float(np.std(selected, ddof=1) / np.sqrt(len(selected)))
                    if len(selected) > 1
                    else np.nan
                ),
                "n": int(len(selected)),
            }
        )
    return pd.DataFrame(rows)
