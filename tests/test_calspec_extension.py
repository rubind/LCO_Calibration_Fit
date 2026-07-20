from pathlib import Path

import numpy as np
import pytest
import matplotlib.pyplot as plt
from astropy.io import fits

from passband_fit.calspec_extension import (
    TemplateSpectrum,
    _read_calspec,
    extend_spectrum,
    load_ascii_template,
    make_diagnostic_figure,
    write_diagnostics,
    write_extended_fits,
)


def _template_flux(wavelength: np.ndarray) -> np.ndarray:
    continuum = (wavelength / 5000.0) ** -2.1
    features = (
        1.0
        - 0.16 * np.exp(-0.5 * ((wavelength - 5150.0) / 35.0) ** 2)
        - 0.09 * np.exp(-0.5 * ((wavelength - 5750.0) / 55.0) ** 2)
        - 0.12 * np.exp(-0.5 * ((wavelength - 6250.0) / 45.0) ** 2)
    )
    return continuum * features


def _correction(wavelength: np.ndarray) -> np.ndarray:
    x = (wavelength - 6000.0) / 1000.0
    return np.exp(0.3 + 0.10 * x + 0.04 * x**2)


def _write_calspec(path: Path, wave: np.ndarray, flux: np.ndarray) -> None:
    columns = [
        fits.Column(name="WAVELENGTH", format="1D", unit="ANGSTROMS", array=wave),
        fits.Column(name="FLUX", format="1E", unit="FLAM", array=flux.astype(np.float32)),
        fits.Column(
            name="STATERROR",
            format="1E",
            unit="FLAM",
            array=np.full(len(wave), 0.01, dtype=np.float32),
        ),
        fits.Column(
            name="SYSERROR",
            format="1E",
            unit="FLAM",
            array=np.full(len(wave), 0.02, dtype=np.float32),
        ),
        fits.Column(
            name="FWHM",
            format="1E",
            unit="ANGSTROMS",
            array=np.full(len(wave), 10.0, dtype=np.float32),
        ),
        fits.Column(name="DATAQUAL", format="1I", array=np.zeros(len(wave), dtype=np.int16)),
        fits.Column(name="TOTEXP", format="1E", unit="SEC", array=np.ones(len(wave))),
    ]
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="SCI")]
    ).writeto(path)


def test_best_template_extension_recovers_synthetic_tail_and_preserves_calspec(tmp_path):
    template_wave = np.arange(3500.0, 7000.1, 2.0)
    good_flux = _template_flux(template_wave)
    wrong_flux = (template_wave / 5000.0) ** -1.0 * (
        1.0 - 0.15 * np.exp(-0.5 * ((template_wave - 5400.0) / 30.0) ** 2)
    )
    calspec_wave = np.arange(4000.0, 6000.1, 10.0)
    calspec_flux = _template_flux(calspec_wave) * _correction(calspec_wave)
    templates = [
        TemplateSpectrum(tmp_path / "wrong.txt", template_wave, wrong_flux),
        TemplateSpectrum(tmp_path / "good.txt", template_wave, good_flux),
    ]
    result = extend_spectrum(
        calspec_wave,
        calspec_flux,
        templates,
        maximum_wavelength_angstrom=6500.0,
        overlap_width_angstrom=1000.0,
        continuum_degree=2,
        smooth_width_angstrom=10.0,
        blend_width_angstrom=20.0,
        output_spacing_angstrom=10.0,
        minimum_overlap_points=50,
    )
    assert Path(result.matches["red"].template_path).name == "good.txt"
    original = ~result.model_extension
    np.testing.assert_array_equal(result.wavelength_angstrom[original], calspec_wave)
    np.testing.assert_array_equal(result.flux[original], calspec_flux)
    assert result.wavelength_angstrom[-1] == 6500.0
    far_extension = result.wavelength_angstrom >= 6030.0
    expected = _template_flux(result.wavelength_angstrom[far_extension]) * _correction(
        result.wavelength_angstrom[far_extension]
    )
    np.testing.assert_allclose(result.flux[far_extension], expected, rtol=0.015)

    noisy_flux = calspec_flux.copy()
    noisy_flux[-25] *= 20.0
    noisy_flux[-55] *= 0.05
    noisy_result = extend_spectrum(
        calspec_wave,
        noisy_flux,
        templates,
        maximum_wavelength_angstrom=6500.0,
        overlap_width_angstrom=1000.0,
        continuum_degree=2,
        smooth_width_angstrom=10.0,
        blend_width_angstrom=20.0,
        output_spacing_angstrom=10.0,
        minimum_overlap_points=50,
    )
    assert Path(noisy_result.matches["red"].template_path).name == "good.txt"
    assert noisy_result.matches["red"].mad_mmag == pytest.approx(
        result.matches["red"].mad_mmag,
        abs=0.2,
    )


def test_ascii_units_and_air_conversion_are_supported(tmp_path):
    path = tmp_path / "template.txt"
    wave_nm = np.arange(400.0, 701.0, 1.0)
    flux = (wave_nm / 500.0) ** -2
    np.savetxt(path, np.c_[wave_nm, flux], header="wave flux")
    spectrum = load_ascii_template(path, wavelength_unit="nm")
    assert spectrum.wavelength_angstrom[0] == pytest.approx(4000.0)
    assert spectrum.wavelength_angstrom[-1] == pytest.approx(7000.0)
    air = load_ascii_template(
        path,
        wavelength_unit="nm",
        wavelength_convention="air",
    )
    assert np.all(air.wavelength_angstrom > spectrum.wavelength_angstrom)


def test_extended_fits_flags_model_rows_and_leaves_measurements_unchanged(tmp_path):
    calspec_path = tmp_path / "input.fits"
    output_path = tmp_path / "extended.fits"
    template_path = tmp_path / "good.txt"
    template_wave = np.arange(3500.0, 7000.1, 2.0)
    np.savetxt(template_path, np.c_[template_wave, _template_flux(template_wave)])
    calspec_wave = np.arange(4000.0, 6000.1, 10.0)
    calspec_flux = _template_flux(calspec_wave) * _correction(calspec_wave)
    _write_calspec(calspec_path, calspec_wave, calspec_flux)
    loaded_wave, loaded_flux, _ = _read_calspec(calspec_path)
    template = load_ascii_template(template_path)
    result = extend_spectrum(
        loaded_wave,
        loaded_flux,
        [template],
        maximum_wavelength_angstrom=6200.0,
        output_spacing_angstrom=10.0,
        minimum_overlap_points=30,
    )
    write_extended_fits(calspec_path, output_path, result)
    with fits.open(output_path) as hdul:
        data = hdul["SCI"].data
        assert "MODEL_EXT" in data.names
        model = np.asarray(data["MODEL_EXT"], dtype=bool)
        assert np.count_nonzero(model) == 20
        np.testing.assert_array_equal(data["WAVELENGTH"][~model], loaded_wave)
        np.testing.assert_array_equal(data["FLUX"][~model], loaded_flux)
        assert np.all(data["DATAQUAL"][model] == 2)
        assert np.all(np.isnan(data["STATERROR"][model]))
        assert hdul["SCI"].header["EXTENDED"]
        assert "RMAD" in hdul["SCI"].header
    with pytest.raises(FileExistsError):
        write_extended_fits(calspec_path, output_path, result)

    diagnostics = write_diagnostics(
        output_path,
        calspec_path,
        [template],
        result,
        {"test": True},
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in diagnostics)
    ranking_text = diagnostics[1].read_text(encoding="utf-8")
    assert "mad_mmag" in ranking_text
    assert "score_mmag" not in ranking_text
    figure = make_diagnostic_figure([template], result)
    assert len(figure.axes) == 3
    overlap_axis = figure.axes[1]
    assert len(overlap_axis.lines) == 2
    assert "CALSPEC" in overlap_axis.lines[0].get_label()
    assert "corrected template" in overlap_axis.lines[1].get_label()
    plt.close(figure)


def test_extension_fails_when_no_template_reaches_requested_tail(tmp_path):
    wave = np.arange(4000.0, 6000.1, 10.0)
    flux = np.ones_like(wave)
    template = TemplateSpectrum(tmp_path / "short.txt", wave.copy(), flux.copy())
    with pytest.raises(ValueError, match="no template can provide"):
        extend_spectrum(
            wave,
            flux,
            [template],
            maximum_wavelength_angstrom=6500.0,
        )
