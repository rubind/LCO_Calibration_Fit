from pathlib import Path

import pytest
import yaml

from passband_fit.config import ConfigError, config_to_dict, load_config
from passband_fit.output import build_manifest, write_json, write_manifest, write_resolved_config


ROOT = Path(__file__).resolve().parents[1]


def minimal_config():
    curve = {
        "path": "curve.txt",
        "wavelength_unit": "angstrom",
        "value_semantics": "fraction",
    }
    return {
        "schema_version": 1,
        "selected_filter": "gp",
        "inputs": {
            "observations": {"path": "observations.txt"},
            "filter_curves": {"gp": dict(curve)},
            "ccd": dict(curve),
            "aluminum_squared": dict(curve),
            "atmosphere": dict(curve),
            "spectra_directory": "spectra",
            "target_mapping": "mapping.txt",
            "vega_spectrum": "spectra/vega.fits",
        },
        "output": {"directory": "runs"},
    }


def write_config(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_example_config_loads_and_is_real_rooted():
    config = load_config(ROOT / "example_config.yaml")
    assert config.selected_filter == "gp"
    assert config.model.magnitude_system == "vega"
    assert config.model.passband_mode == "shift_only"
    assert config.grid.spacing_angstrom == 2.0
    assert config.grid.convergence_tolerance_mag == 0.0005
    assert config.inputs.observations.path == ROOT / "all_phot_filtered.txt"
    assert config.inputs.vega_spectrum.name == "alpha_lyr_stis_012.fits"
    assert config.sampler.num_warmup == 1000
    assert config.sampler.num_samples == 5000
    assert config.sampler.num_chains == 4
    assert config.sampler.chain_method == "parallel"
    assert config.sampler.target_accept_prob == pytest.approx(0.8)
    assert config.sampler.max_tree_depth == 10
    assert config.sampler.dense_mass is False
    assert config.priors.star_correlated_scatter.scale == pytest.approx(0.03)


def test_all_standalone_filter_configs_use_numpyro_sampler_schema():
    expected_keys = {
        "num_warmup",
        "num_samples",
        "num_chains",
        "thinning",
        "random_seed",
        "chain_method",
        "progress",
        "target_accept_prob",
        "max_tree_depth",
        "dense_mass",
    }
    paths = [ROOT / "example_config.yaml", *sorted((ROOT / "configs").glob("*.yaml"))]
    canonical = {"U.yaml", "B.yaml", "V.yaml", "R.yaml", "I.yaml", "up.yaml", "rp.yaml", "ip.yaml", "zs.yaml"}
    assert canonical <= {path.name for path in paths}
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(raw["sampler"]) == expected_keys, path
        config = load_config(path)
        assert config.sampler.num_chains == 4
        assert config.sampler.chain_method == "parallel"
        assert config.priors.star_correlated_scatter.scale == pytest.approx(0.03)


def test_numpyro_sampler_settings_round_trip(tmp_path):
    raw = minimal_config()
    raw["sampler"] = {
        "num_warmup": 25,
        "num_samples": 100,
        "num_chains": 2,
        "thinning": 3,
        "random_seed": 8675309,
        "chain_method": "vectorized",
        "progress": False,
        "target_accept_prob": 0.9,
        "max_tree_depth": 12,
        "dense_mass": True,
    }
    config = load_config(write_config(tmp_path, raw))
    assert config.sampler.num_warmup == 25
    assert config.sampler.num_samples == 100
    assert config.sampler.num_chains == 2
    assert config.sampler.thinning == 3
    assert config.sampler.random_seed == 8675309
    assert config.sampler.chain_method == "vectorized"
    assert config.sampler.progress is False
    assert config.sampler.target_accept_prob == pytest.approx(0.9)
    assert config.sampler.max_tree_depth == 12
    assert config.sampler.dense_mass is True
    assert config_to_dict(config)["sampler"] == raw["sampler"]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("num_warmup", -1, "num_warmup"),
        ("num_warmup", 1.5, "integer field"),
        ("num_samples", 0, "samples, chains, and thinning"),
        ("num_chains", 0, "samples, chains, and thinning"),
        ("thinning", 0, "samples, chains, and thinning"),
        ("random_seed", -1, "random_seed"),
        ("random_seed", 2**32, "random_seed"),
        ("chain_method", "threads", "chain_method"),
        ("progress", "true", "progress"),
        ("target_accept_prob", 0.0, "target_accept_prob"),
        ("target_accept_prob", 1.0, "target_accept_prob"),
        ("target_accept_prob", float("nan"), "target_accept_prob"),
        ("max_tree_depth", 0, "max_tree_depth"),
        ("dense_mass", "false", "dense_mass"),
    ],
)
def test_invalid_numpyro_sampler_settings_are_rejected(tmp_path, field, value, match):
    raw = minimal_config()
    raw["sampler"] = {field: value}
    with pytest.raises(ConfigError, match=match):
        load_config(write_config(tmp_path, raw))


def test_emcee_sampler_keys_are_rejected(tmp_path):
    raw = minimal_config()
    raw["sampler"] = {"walkers": 32}
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_config(tmp_path, raw))


def test_paths_resolve_relative_to_config_and_cli_filter_wins(tmp_path):
    raw = minimal_config()
    raw["inputs"]["filter_curves"]["rp"] = dict(raw["inputs"]["filter_curves"]["gp"])
    config = load_config(write_config(tmp_path, raw), selected_filter="rp")
    assert config.selected_filter == "rp"
    assert config.inputs.observations.path == tmp_path / "observations.txt"
    assert config.output.directory == tmp_path / "runs"


@pytest.mark.parametrize(
    "location,key",
    [("root", "surprise"), ("inputs", "site_map"), ("curve", "scale")],
)
def test_unknown_keys_are_errors(tmp_path, location, key):
    raw = minimal_config()
    if location == "root":
        raw[key] = 1
    elif location == "inputs":
        raw["inputs"][key] = 1
    else:
        raw["inputs"]["ccd"][key] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_config(tmp_path, raw))


def test_selected_filter_requires_a_configured_curve(tmp_path):
    with pytest.raises(ConfigError, match="has no configured curve"):
        load_config(write_config(tmp_path, minimal_config()), selected_filter="rp")


def test_resolved_config_and_basic_manifest_outputs(tmp_path):
    raw = minimal_config()
    config_path = write_config(tmp_path, raw)
    for relative in ["observations.txt", "curve.txt", "mapping.txt", "spectra/vega.fits"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    config = load_config(config_path)
    resolved = config_to_dict(config)
    assert Path(resolved["inputs"]["ccd"]["path"]).is_absolute()
    resolved_path = write_resolved_config(tmp_path / "out" / "resolved.yaml", config)
    assert yaml.safe_load(resolved_path.read_text())["schema_version"] == 1
    manifest = build_manifest(config, parameter_names=["mu_shift"], run_id="test-run")
    assert manifest["parameter_names"] == ["mu_shift"]
    assert len(manifest["config_sha256"]) == 64
    manifest_path = write_manifest(tmp_path / "out" / "manifest.json", manifest)
    assert "test-run" in manifest_path.read_text()
    json_path = write_json(tmp_path / "out" / "summary.json", {"value": 2})
    assert json_path.read_text().endswith("\n")
