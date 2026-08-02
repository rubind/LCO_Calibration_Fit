"""Small, format-stable output and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .config import FitConfig, PathLike, config_to_dict


MANIFEST_SCHEMA_VERSION = 1


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_json(path: PathLike, value: Any) -> Path:
    """Write JSON atomically with stable key ordering."""

    destination = Path(path)
    return _atomic_text(destination, json.dumps(_plain(value), indent=2, sort_keys=True) + "\n")


def write_yaml(path: PathLike, value: Any) -> Path:
    """Write safe YAML atomically."""

    destination = Path(path)
    text = yaml.safe_dump(_plain(value), sort_keys=False, default_flow_style=False)
    return _atomic_text(destination, text)


def write_csv(path: PathLike, value: Union[pd.DataFrame, Iterable[Mapping[str, Any]]]) -> Path:
    """Write a dataframe or sequence of row mappings atomically as CSV."""

    destination = Path(path)
    table = value if isinstance(value, pd.DataFrame) else pd.DataFrame(list(value))
    return _atomic_text(destination, table.to_csv(index=False))


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA256 digest without reading the entire file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a value after canonical JSON serialization."""

    payload = json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_input_paths(config: FitConfig) -> Sequence[Path]:
    paths = [
        config.inputs.observations.path,
        config.inputs.target_mapping,
        config.inputs.vega_spectrum,
        config.inputs.ccd.path,
        config.inputs.aluminum_squared.path,
    ]
    if config.inputs.atmosphere is not None:
        paths.append(config.inputs.atmosphere.path)
    else:
        if config.inputs.site_mapping is not None:
            paths.append(config.inputs.site_mapping)
        paths.extend(curve.path for curve in config.inputs.atmospheres.values())
    paths.extend(curve.path for curve in config.inputs.filter_curves.values())
    return paths


def build_manifest(
    config: FitConfig,
    parameter_names: Sequence[str] = (),
    input_paths: Optional[Sequence[PathLike]] = None,
    status: str = "created",
    run_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the versioned, JSON-safe provenance portion of a run manifest."""

    resolved = config_to_dict(config)
    paths = [Path(item) for item in (input_paths or _default_input_paths(config))]
    input_hashes: Dict[str, str] = {}
    for path in paths:
        absolute = path.resolve(strict=False)
        if not absolute.is_file():
            raise FileNotFoundError(f"cannot hash missing manifest input: {absolute}")
        input_hashes[str(absolute)] = sha256_file(absolute)
    now = datetime.now(timezone.utc)
    manifest: Dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id or now.strftime("%Y%m%dT%H%M%SZ"),
        "status": str(status),
        "created_utc": now.isoformat(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "selected_filter": config.selected_filter,
        "config_sha256": sha256_json(resolved),
        "input_sha256": input_hashes,
        "parameter_names": [str(item) for item in parameter_names],
        "sampler": {
            "engine": "numpyro_nuts",
            "num_warmup": config.sampler.num_warmup,
            "num_samples": config.sampler.num_samples,
            "num_chains": config.sampler.num_chains,
            "random_seed": config.sampler.random_seed,
            "thinning": config.sampler.thinning,
            "chain_method": config.sampler.chain_method,
            "target_accept_prob": config.sampler.target_accept_prob,
            "max_tree_depth": config.sampler.max_tree_depth,
            "dense_mass": config.sampler.dense_mass,
        },
        "platform": platform.platform(),
    }
    if extra:
        overlap = set(manifest) & set(extra)
        if overlap:
            raise ValueError("manifest extra cannot replace core key(s): " + ", ".join(sorted(overlap)))
        manifest.update(_plain(extra))
    return manifest


def write_manifest(path: PathLike, manifest: Mapping[str, Any]) -> Path:
    """Validate the basic manifest keys and atomically write JSON."""

    required = {"manifest_schema_version", "run_id", "status", "config_sha256", "input_sha256"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError("manifest is missing required key(s): " + ", ".join(missing))
    return write_json(path, manifest)


def write_resolved_config(path: PathLike, config: FitConfig) -> Path:
    """Write the fully resolved configuration as YAML."""

    return write_yaml(path, config_to_dict(config))
