"""Compatibility exports for the NumPyro sampler implementation.

New code may import :mod:`passband_fit.numpyro_sampler` directly.  The former
emcee implementation and its HDF5 restart backend were removed when inference
was migrated to NumPyro NUTS.
"""

from .numpyro_sampler import (
    NumpyroDiagnostics,
    NumpyroResult,
    NumpyroSettings,
    run_numpyro_nuts,
)

SamplerSettings = NumpyroSettings
SamplerResult = NumpyroResult
run_sampler = run_numpyro_nuts

__all__ = [
    "NumpyroDiagnostics",
    "NumpyroResult",
    "NumpyroSettings",
    "SamplerResult",
    "SamplerSettings",
    "run_numpyro_nuts",
    "run_sampler",
]
