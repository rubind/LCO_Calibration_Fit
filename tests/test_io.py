from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from passband_fit.config import (
    CurveConfig,
    ObservationColumns,
    ObservationInputConfig,
    SelectionConfig,
    load_config,
)
from passband_fit.io import (
    InputError,
    load_atmosphere_curves,
    load_calspec_spectrum,
    load_curve,
    load_observation_table,
    load_site_mapping,
    load_selected_data,
    load_target_mapping,
    load_target_spectra,
    select_observations,
)


ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT_PATTERN = r"^(?P<instrument>.+?)-\d{8}(?:-|$)"


def observation_config(path=Path("unused")):
    return ObservationInputConfig(
        path=path,
        columns=ObservationColumns(),
        instrument_regex=INSTRUMENT_PATTERN,
        site_prefix_length=3,
    )


def test_real_observation_table_and_gp_selection():
    config = load_config(ROOT / "example_config.yaml")
    table = load_observation_table(config.inputs.observations)
    selected = load_selected_data(config)
    assert len(table) == selected.summary["total_rows"]
    assert selected.summary["matching_filter"] > 0
    assert selected.summary["final_used"] == len(selected.row_id)
    assert selected.summary["matching_filter"] == sum(
        selected.summary[name]
        for name in (
            "rejected_nonpositive_flux",
            "rejected_nonpositive_uncertainty",
            "rejected_snr",
            "final_used",
        )
    )
    assert "lsc1m004-fa03" in selected.instrument_names
    index = selected.instrument_names.index("lsc1m004-fa03")
    assert np.all(np.asarray(selected.sites)[selected.instrument_index == index] == "lsc")


def test_site_mapping_selects_the_supplied_unit_airmass_curves():
    config = load_config(ROOT / "example_config.yaml")
    mapping = load_site_mapping(ROOT / "site_mapping.txt")
    assert mapping == {
        "tfn": "Tenerife-ish",
        "cpt": "Sutherland",
        "elp": "McDonald",
        "coj": "SidingSpring",
        "lsc": "CTIO",
    }
    resolved, curves = load_atmosphere_curves(config, mapping)
    assert resolved == mapping
    assert set(curves) == set(mapping.values())
    for curve in curves.values():
        assert curve.wavelength_angstrom[0] == pytest.approx(2900.0)
        assert curve.values[0] == 0.0
        assert np.all((curve.values >= 0.0) & (curve.values <= 1.0))


def test_site_mapping_rejects_an_unmapped_observed_prefix():
    config = load_config(ROOT / "example_config.yaml")
    with pytest.raises(InputError, match="lacks observed image prefix"):
        load_atmosphere_curves(config, ["xxx"])


def test_selection_accounting_magnitudes_and_exact_aliases():
    table = pd.DataFrame(
        {
            "_source_row_id": [10, 11, 12, 13, 14, 15],
            "star": ["Exact_A", "Exact_A", "Exact_B", "Exact B", "Other", "Exact_A"],
            "filter": ["gp", "gp", "gp", "gp", "rp", "gp"],
            "image": [
                "lsc1m004-fa03-20260707-0001.fits",
                "lsc1m004-fa03-20260707-0002.fits",
                "cpt1m013-fa01-20260707-0003.fits",
                "cpt1m013-fa01-20260707-e91-stacked.fits",
                "bad",
                "coj1m003-fa19-20260708-0004.fits",
            ],
            "airmass": [1.1, 1.2, 1.3, 1.4, 0.0, 1.5],
            "flux": [100.0, -1.0, 10.0, 100.0, np.nan, 100.0],
            "flux_unc": [10.0, 1.0, 0.0, 5.0, np.nan, 1.0],
        }
    )
    selected = select_observations(table, observation_config(), SelectionConfig(10.0), "gp")
    assert selected.summary == {
        "total_rows": 6,
        "matching_filter": 5,
        "rejected_nonpositive_flux": 1,
        "rejected_nonpositive_uncertainty": 1,
        "rejected_snr": 1,
        "final_used": 2,
        "observations_per_target": {"Exact B": 1, "Exact_A": 1},
        "observations_per_instrument": {"cpt1m013-fa01": 1, "coj1m003-fa19": 1},
    }
    np.testing.assert_array_equal(selected.row_id, [13, 15])
    np.testing.assert_allclose(selected.observed_magnitude, [19.0, 19.0])
    np.testing.assert_allclose(selected.magnitude_uncertainty, (2.5 / np.log(10)) * np.array([0.05, 0.01]))
    assert selected.target_names == ("Exact B", "Exact_A")
    assert selected.instrument_names == ("cpt1m013-fa01", "coj1m003-fa19")
    assert selected.sites == ("cpt", "coj")


def test_selected_nonfinite_numeric_data_fail_but_other_filter_does_not():
    base = pd.DataFrame(
        {
            "star": ["A", "B"],
            "filter": ["gp", "rp"],
            "image": ["lsc1m004-fa03-20260707-x.fits", "bad"],
            "airmass": [1.1, np.nan],
            "flux": [100.0, np.nan],
            "flux_unc": [1.0, np.nan],
        }
    )
    selected = select_observations(base, observation_config(), SelectionConfig(), "gp")
    assert selected.summary["final_used"] == 1
    base.loc[0, "flux"] = np.nan
    with pytest.raises(InputError, match="nonfinite or nonnumeric flux"):
        select_observations(base, observation_config(), SelectionConfig(), "gp")


def test_curve_conversions_sort_clip_sentinel_and_duplicate_mean(tmp_path, caplog):
    path = tmp_path / "curve.txt"
    path.write_text("lambda value\n500 0.5\n400 -0.00001\n500 0.7\n600 1.0\n", encoding="utf-8")
    curve = load_curve(
        CurveConfig(path, "nm", "fraction", duplicate_policy="mean", negative_tolerance=1e-4),
        "filter_gp",
    )
    np.testing.assert_allclose(curve.wavelength_angstrom, [4000.0, 5000.0, 6000.0])
    np.testing.assert_allclose(curve.values, [0.0, 0.6, 1.0])
    assert "clipped 1 negative" in caplog.text

    percent = tmp_path / "percent.txt"
    percent.write_text("4000 25\n5000 75\n", encoding="utf-8")
    np.testing.assert_allclose(load_curve(CurveConfig(percent, "angstrom", "percent")).values, [0.25, 0.75])

    extinction = tmp_path / "extinction.txt"
    extinction.write_text("2700 10\n3200 1\n5000 0\n", encoding="utf-8")
    converted = load_curve(
        CurveConfig(extinction, "angstrom", "extinction_mag_per_airmass", sentinel_values=(10.0,)),
        "atmosphere",
    )
    np.testing.assert_allclose(converted.values, [10 ** -0.4, 1.0])


def test_curve_duplicate_error_and_air_to_vacuum(tmp_path):
    path = tmp_path / "curve.txt"
    path.write_text("5000 0.2\n5000 0.3\n", encoding="utf-8")
    with pytest.raises(InputError, match="duplicate wavelength"):
        load_curve(CurveConfig(path, "angstrom", "fraction"))
    path.write_text("5000 0.2\n6000 0.3\n", encoding="utf-8")
    air_curve = load_curve(CurveConfig(path, "angstrom", "fraction", wavelength_convention="air"))
    assert np.all(air_curve.wavelength_angstrom > np.array([5000.0, 6000.0]))


def test_legacy_mapping_and_real_calspec_fits():
    mapping = load_target_mapping(ROOT / "star_spectra_mapping.txt")
    assert mapping["BD-11_3759"] == "bd11d3759_stis_003_extended_2um.fits"
    assert mapping["Vega"] == "alpha_lyr_stis_012.fits"
    spectrum = load_calspec_spectrum(ROOT / "CALSPEC_spectra/alpha_lyr_stis_012.fits")
    assert spectrum.name == "alpha_lyr_stis_012"
    assert spectrum.wavelength_angstrom[0] < 1000.0
    assert spectrum.wavelength_angstrom[-1] > 2.0e6
    assert np.all(np.diff(spectrum.wavelength_angstrom) > 0)


def test_target_spectra_use_exact_cleaned_table_aliases():
    config = load_config(ROOT / "example_config.yaml")
    spectra = load_target_spectra(config, ["BD-11_3759", "P330-E"])
    assert set(spectra) == {"BD-11_3759", "P330-E"}
    with pytest.raises(InputError, match="lacks exact alias"):
        load_target_spectra(config, ["bd-11_3759"])
