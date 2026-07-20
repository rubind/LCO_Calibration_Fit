"""Optional NumPyro NUTS adapter for an unconstrained vector potential.

The adapter is deliberately independent of the emcee implementation.  The
``potential_fn`` must be JAX differentiable and return the negative, possibly
unnormalized, log density for one unconstrained ``theta`` vector.

NumPyro normally exposes samples in ``(chains, draws, ...)`` order.  Results
here use ``(draws, chains, ndim)`` so downstream code can consume the same
leading axes as the legacy ensemble sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


PotentialFunction = Callable[[Any], Any]
VALID_CHAIN_METHODS = ("parallel", "sequential", "vectorized")


@dataclass(frozen=True)
class NumpyroSettings:
    """Configuration for a NumPyro NUTS run.

    ``num_samples`` is the number of retained draws per chain.  When
    ``thinning > 1``, the adapter asks NumPyro for additional transitions so
    that the returned chain still contains exactly ``num_samples`` draws.
    """

    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    thinning: int = 1
    seed: int = 0
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    chain_method: str = "parallel"
    progress_bar: bool = True
    dense_mass: bool = False

    def validate(self) -> None:
        if self.num_warmup < 0:
            raise ValueError("num_warmup must be nonnegative")
        if self.num_samples < 1 or self.num_chains < 1:
            raise ValueError("num_samples and num_chains must be positive")
        if self.thinning < 1:
            raise ValueError("thinning must be positive")
        if not isinstance(self.seed, (int, np.integer)) or not 0 <= int(self.seed) < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not np.isfinite(self.target_accept_prob) or not 0 < self.target_accept_prob < 1:
            raise ValueError("target_accept_prob must lie strictly between zero and one")
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be positive")
        if self.chain_method not in VALID_CHAIN_METHODS:
            raise ValueError(
                f"chain_method must be one of {VALID_CHAIN_METHODS}, got {self.chain_method!r}"
            )
        if not isinstance(self.dense_mass, bool):
            raise ValueError("dense_mass must be a boolean")


@dataclass(frozen=True)
class NumpyroDiagnostics:
    """Per-parameter split-chain diagnostics and basic NUTS health metrics."""

    parameter_names: Tuple[str, ...]
    rhat: np.ndarray
    effective_sample_size: np.ndarray
    mean_acceptance_probability: float
    acceptance_probability_by_chain: np.ndarray
    n_divergent: int
    divergent_by_chain: np.ndarray
    maximum_tree_steps: int
    warnings: Tuple[str, ...]
    requested_chain_method: str
    effective_chain_method: str
    backend: str
    local_device_count: int

    @property
    def rhat_by_parameter(self) -> Dict[str, float]:
        return dict(zip(self.parameter_names, self.rhat.tolist()))

    @property
    def ess_by_parameter(self) -> Dict[str, float]:
        return dict(zip(self.parameter_names, self.effective_sample_size.tolist()))

    def as_dict(self) -> Dict[str, object]:
        return {
            "rhat": self.rhat_by_parameter,
            "effective_sample_size": self.ess_by_parameter,
            "mean_acceptance_probability": self.mean_acceptance_probability,
            "acceptance_probability_by_chain": self.acceptance_probability_by_chain.tolist(),
            "n_divergent": self.n_divergent,
            "divergent_by_chain": self.divergent_by_chain.tolist(),
            "maximum_tree_steps": self.maximum_tree_steps,
            "warnings": list(self.warnings),
            "requested_chain_method": self.requested_chain_method,
            "effective_chain_method": self.effective_chain_method,
            "backend": self.backend,
            "local_device_count": self.local_device_count,
        }


@dataclass(frozen=True)
class NumpyroResult:
    """Materialized NUTS samples in legacy-compatible axis order."""

    chain: np.ndarray
    log_probability: Optional[np.ndarray]
    parameter_names: Tuple[str, ...]
    diagnostics: NumpyroDiagnostics
    extra_fields: Mapping[str, np.ndarray]

    def flattened(self, discard: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Flatten retained draw and chain axes while preserving alignment."""

        if discard < 0 or discard >= self.chain.shape[0]:
            raise ValueError("discard must retain at least one draw")
        selected = self.chain[discard:]
        flat_chain = selected.reshape((-1, selected.shape[-1]))
        if self.log_probability is None:
            flat_log_probability = None
        else:
            flat_log_probability = self.log_probability[discard:].reshape(-1)
        return flat_chain, flat_log_probability


def _parameter_names(ndim: int, names: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if names is None:
        return tuple(f"theta[{index}]" for index in range(ndim))
    result = tuple(str(name) for name in names)
    if len(result) != ndim:
        raise ValueError("parameter_names length must match initial theta dimension")
    if len(set(result)) != len(result):
        raise ValueError("parameter_names must be unique")
    return result


def _initial_params(initial_theta: Sequence[float], num_chains: int) -> Tuple[np.ndarray, int]:
    values = np.asarray(initial_theta, dtype=np.float64)
    if values.ndim == 1:
        if len(values) < 1:
            raise ValueError("initial_theta must contain at least one parameter")
        ndim = len(values)
        result = values if num_chains == 1 else np.repeat(values[None, :], num_chains, axis=0)
    elif values.ndim == 2:
        if values.shape[0] != num_chains or values.shape[1] < 1:
            raise ValueError(
                "two-dimensional initial_theta must have shape (num_chains, ndim)"
            )
        ndim = values.shape[1]
        result = values[0] if num_chains == 1 else values
    else:
        raise ValueError("initial_theta must have shape (ndim,) or (num_chains, ndim)")
    if not np.all(np.isfinite(result)):
        raise ValueError("initial_theta must contain only finite values")
    return np.asarray(result, dtype=np.float64), ndim


def _configure_chain_method(
    settings: NumpyroSettings, jax: Any
) -> Tuple[str, str, int, Tuple[str, ...]]:
    """Select parallel or safe sequential execution after JAX startup."""

    warnings = []
    backend = str(jax.default_backend())
    local_devices = int(jax.local_device_count())
    effective_method = settings.chain_method
    if settings.chain_method == "parallel" and settings.num_chains > local_devices:
        effective_method = "sequential"
        warnings.append(
            f"requested {settings.num_chains} parallel chains but only {local_devices} "
            f"{backend} device(s) are active; using sequential chains. To use virtual "
            "CPU devices, call this adapter before any other JAX computation."
        )
    return effective_method, backend, local_devices, tuple(warnings)


def _validate_initial_potential(
    potential_fn: PotentialFunction, initial: np.ndarray, num_chains: int, jax: Any
) -> None:
    import jax.numpy as jnp

    rows = initial[None, :] if num_chains == 1 else initial
    for index, row in enumerate(rows):
        try:
            value = np.asarray(jax.device_get(potential_fn(jnp.asarray(row))))
        except Exception as exc:
            raise ValueError(f"potential_fn failed at initial theta for chain {index}: {exc}") from exc
        if value.shape != () or not np.isfinite(float(value)):
            raise ValueError(
                f"potential_fn must return one finite scalar at initial theta; chain {index} "
                f"returned shape {value.shape}"
            )


def _diagnostics(
    grouped_chain: np.ndarray,
    parameter_names: Tuple[str, ...],
    grouped_extra: Mapping[str, np.ndarray],
    settings: NumpyroSettings,
    effective_chain_method: str,
    backend: str,
    local_device_count: int,
    configuration_warnings: Tuple[str, ...],
) -> NumpyroDiagnostics:
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    warnings = list(configuration_warnings)
    if settings.num_chains >= 2:
        try:
            rhat = np.asarray(split_gelman_rubin(grouped_chain), dtype=np.float64)
        except (AssertionError, FloatingPointError, ValueError):
            rhat = np.full(grouped_chain.shape[-1], np.nan)
            warnings.append("Rhat could not be estimated")
    else:
        rhat = np.full(grouped_chain.shape[-1], np.nan)
        warnings.append("Rhat requires at least two chains and is reported as NaN")
    try:
        ess = np.asarray(effective_sample_size(grouped_chain), dtype=np.float64)
    except (AssertionError, FloatingPointError, ValueError):
        ess = np.full(grouped_chain.shape[-1], np.nan)
        warnings.append("effective sample size could not be estimated")
    if np.any(np.isfinite(rhat) & (rhat > 1.01)):
        warnings.append("one or more parameters have Rhat greater than 1.01")
    if np.any(~np.isfinite(ess)):
        warnings.append("one or more effective sample-size estimates are nonfinite")

    acceptance = np.asarray(grouped_extra["accept_prob"], dtype=np.float64)
    divergences = np.asarray(grouped_extra["diverging"], dtype=bool)
    tree_steps = np.asarray(grouped_extra["num_steps"], dtype=np.int64)
    n_divergent = int(np.count_nonzero(divergences))
    maximum_tree_steps = int(np.max(tree_steps))
    if n_divergent:
        warnings.append(f"NUTS reported {n_divergent} divergent transition(s)")
    if maximum_tree_steps >= 2**settings.max_tree_depth - 1:
        warnings.append("one or more transitions reached the configured maximum tree depth")

    return NumpyroDiagnostics(
        parameter_names=parameter_names,
        rhat=rhat,
        effective_sample_size=ess,
        mean_acceptance_probability=float(np.mean(acceptance)),
        acceptance_probability_by_chain=np.mean(acceptance, axis=1),
        n_divergent=n_divergent,
        divergent_by_chain=np.count_nonzero(divergences, axis=1),
        maximum_tree_steps=maximum_tree_steps,
        warnings=tuple(warnings),
        requested_chain_method=settings.chain_method,
        effective_chain_method=effective_chain_method,
        backend=backend,
        local_device_count=local_device_count,
    )


def run_numpyro_nuts(
    potential_fn: PotentialFunction,
    initial_theta: Sequence[float],
    settings: Optional[NumpyroSettings] = None,
    parameter_names: Optional[Sequence[str]] = None,
) -> NumpyroResult:
    """Run NUTS for a JAX potential over one unconstrained theta vector.

    A one-dimensional ``initial_theta`` is replicated across chains.  Distinct
    starts can be supplied with shape ``(num_chains, ndim)``.  The returned log
    probability is ``-potential_energy`` and therefore has the same unknown
    additive normalization as ``potential_fn``.

    For CPU-parallel chains, call this function before any other JAX computation
    so ``numpyro.set_host_device_count`` can create virtual host devices.  If JAX
    has already initialized with too few devices (common in notebooks and on
    Apple Silicon), the adapter falls back to sequential chains and records a
    warning rather than failing inside ``pmap``.
    """

    if not callable(potential_fn):
        raise TypeError("potential_fn must be callable")
    settings = settings or NumpyroSettings()
    settings.validate()
    initial, ndim = _initial_params(initial_theta, settings.num_chains)
    names = _parameter_names(ndim, parameter_names)

    try:
        import numpyro

        if settings.chain_method == "parallel" and settings.num_chains > 1:
            # Deliberately before importing JAX in this module.
            numpyro.set_host_device_count(settings.num_chains)
        import jax
        import jax.numpy as jnp
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:  # pragma: no cover - exercised only in optional environments
        raise ImportError(
            "NumPyro sampling requires compatible 'jax' and 'numpyro' installations"
        ) from exc

    effective_method, backend, local_devices, method_warnings = _configure_chain_method(
        settings, jax
    )
    _validate_initial_potential(potential_fn, initial, settings.num_chains, jax)
    kernel = NUTS(
        potential_fn=potential_fn,
        target_accept_prob=settings.target_accept_prob,
        max_tree_depth=settings.max_tree_depth,
        dense_mass=settings.dense_mass,
    )
    # NumPyro treats num_samples as transitions before its thinning collector.
    # Multiplying here makes settings.num_samples the stable retained-draw count.
    mcmc = MCMC(
        kernel,
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples * settings.thinning,
        num_chains=settings.num_chains,
        thinning=settings.thinning,
        chain_method=effective_method,
        progress_bar=settings.progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(int(settings.seed)),
        init_params=jnp.asarray(initial),
        extra_fields=("potential_energy", "accept_prob", "diverging", "num_steps"),
    )
    grouped_chain = np.asarray(
        jax.device_get(mcmc.get_samples(group_by_chain=True)), dtype=np.float64
    )
    if grouped_chain.shape != (settings.num_chains, settings.num_samples, ndim):
        raise RuntimeError(
            "NumPyro returned an unexpected sample shape: "
            f"{grouped_chain.shape}, expected "
            f"({settings.num_chains}, {settings.num_samples}, {ndim})"
        )
    grouped_extra = {
        name: np.asarray(jax.device_get(value))
        for name, value in mcmc.get_extra_fields(group_by_chain=True).items()
    }
    diagnostics = _diagnostics(
        grouped_chain,
        names,
        grouped_extra,
        settings,
        effective_method,
        backend,
        local_devices,
        method_warnings,
    )
    chain = np.transpose(grouped_chain, (1, 0, 2))
    potential_energy = grouped_extra.get("potential_energy")
    log_probability = (
        None
        if potential_energy is None
        else -np.transpose(np.asarray(potential_energy, dtype=np.float64), (1, 0))
    )
    extra_fields = {
        name: np.transpose(value, (1, 0)) for name, value in grouped_extra.items()
    }
    return NumpyroResult(
        chain=chain,
        log_probability=log_probability,
        parameter_names=names,
        diagnostics=diagnostics,
        extra_fields=extra_fields,
    )
