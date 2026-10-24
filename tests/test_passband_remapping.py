import numpy as np
import pytest

from passband_fit.passband_model import (
    exponential_warp,
    instrument_passband,
    remap_passband,
    support_reaches_grid_boundary,
)


def _centroid(wavelength, response):
    return np.sum(wavelength * response) / np.sum(response)


def _sigma(wavelength, response):
    center = _centroid(wavelength, response)
    return np.sqrt(np.sum((wavelength - center) ** 2 * response) / np.sum(response))


def test_combined_width_and_shift_have_correct_output_frame_convention() -> None:
    wavelength = np.arange(3000.0, 8000.1, 0.5)
    pivot = 5000.0
    nominal_center = 5300.0
    nominal_sigma = 120.0
    nominal = np.exp(-0.5 * ((wavelength - nominal_center) / nominal_sigma) ** 2)
    width = 1.4
    delta = 75.0
    remapped = remap_passband(
        wavelength,
        nominal,
        pivot_wavelength_angstrom=pivot,
        delta_wavelength_angstrom=delta,
        width_scale=width,
    )
    expected_center = pivot + width * (nominal_center - pivot) + delta
    assert _centroid(wavelength, remapped) == pytest.approx(expected_center, abs=0.02)
    assert _sigma(wavelength, remapped) == pytest.approx(width * nominal_sigma, abs=0.02)


def test_positive_shift_is_red_and_width_greater_than_one_is_broader() -> None:
    wavelength = np.arange(3500.0, 6500.1, 1.0)
    nominal = np.exp(-0.5 * ((wavelength - 5000.0) / 150.0) ** 2)
    red = remap_passband(
        wavelength,
        nominal,
        pivot_wavelength_angstrom=5000.0,
        delta_wavelength_angstrom=50.0,
    )
    broad = remap_passband(
        wavelength,
        nominal,
        pivot_wavelength_angstrom=5000.0,
        width_scale=1.2,
    )
    assert _centroid(wavelength, red) > _centroid(wavelength, nominal)
    assert _sigma(wavelength, broad) > _sigma(wavelength, nominal)


def test_positive_warp_suppresses_red_relative_to_blue_and_is_one_at_pivot() -> None:
    wavelength = np.array([4000.0, 5000.0, 6000.0])
    warp = exponential_warp(
        wavelength,
        pivot_wavelength_angstrom=5000.0,
        warp_parameter=0.4,
        warp_scale_angstrom=1000.0,
    )
    assert warp[0] > warp[1] > warp[2]
    assert warp[1] == pytest.approx(1.0)


def test_warp_and_width_guards_reject_invalid_proposals() -> None:
    wavelength = np.linspace(4000.0, 6000.0, 101)
    nominal = np.ones_like(wavelength)
    with pytest.raises(ValueError, match="width scale"):
        remap_passband(
            wavelength,
            nominal,
            pivot_wavelength_angstrom=5000.0,
            width_scale=0.0,
        )
    with pytest.raises(ValueError, match="exponent guard"):
        exponential_warp(
            wavelength,
            pivot_wavelength_angstrom=5000.0,
            warp_parameter=1000.0,
            warp_scale_angstrom=1.0,
        )


def test_instrument_passband_and_boundary_detection() -> None:
    wavelength = np.linspace(4000.0, 6000.0, 1001)
    nominal = np.exp(-0.5 * ((wavelength - 5000.0) / 200.0) ** 2)
    fitted = instrument_passband(
        wavelength,
        nominal,
        pivot_wavelength_angstrom=5000.0,
        delta_wavelength_angstrom=20.0,
        width_scale=1.05,
        warp_parameter=0.1,
        warp_scale_angstrom=1000.0,
    )
    assert np.all(np.isfinite(fitted))
    assert np.all(fitted >= 0.0)
    assert not support_reaches_grid_boundary(fitted, relative_threshold=1.0e-4)
    assert support_reaches_grid_boundary(np.ones_like(wavelength))


@pytest.mark.parametrize("warp", [-0.7, 0.0, 0.8])
def test_warp_conserves_remapped_filter_area_and_magnitude(warp):
    from passband_fit.synthetic_photometry import vega_magnitude

    wavelength = np.linspace(3500., 7000., 2001) ** 1.0001
    nominal = np.exp(-0.5 * ((wavelength - 5100.) / 240.) ** 2)
    kwargs = dict(pivot_wavelength_angstrom=5000., delta_wavelength_angstrom=50., width_scale=1.15)
    remapped = remap_passband(wavelength, nominal, **kwargs)
    fitted = instrument_passband(wavelength, nominal, **kwargs, warp_parameter=warp, warp_scale_angstrom=1000.)
    np.testing.assert_allclose(np.trapz(fitted, wavelength), np.trapz(remapped, wavelength), rtol=2e-15)
    old_response = remapped * exponential_warp(
        wavelength, pivot_wavelength_angstrom=5000., warp_parameter=warp, warp_scale_angstrom=1000.)
    fixed_response = (wavelength / 5000.) ** -0.3
    source = (wavelength / 5000.) ** -1.5
    reference = (wavelength / 5000.) ** -3.
    assert vega_magnitude(wavelength, source, reference, fitted * fixed_response) == pytest.approx(
        vega_magnitude(wavelength, source, reference, old_response * fixed_response), abs=2e-14)

