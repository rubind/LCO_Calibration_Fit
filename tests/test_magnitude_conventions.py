import numpy as np
import pytest

from passband_fit.synthetic_photometry import (
    AB_ZERO_FNU_CGS,
    SPEED_OF_LIGHT_ANGSTROM_PER_SECOND,
    ab_magnitude,
    ab_reference_flux_lambda,
    check_grid_convergence,
    grid_convergence_delta_magnitude,
    photon_counting_integral,
    photon_weighted_mean_wavelength,
    synthetic_magnitude,
    vega_magnitude,
)


def _grid_and_passband(step: float = 1.0):
    wavelength = np.arange(4000.0, 7000.0 + 0.5 * step, step)
    passband = np.exp(-0.5 * ((wavelength - 5500.0) / 500.0) ** 2)
    return wavelength, passband


def test_ab_reference_has_correct_per_angstrom_units() -> None:
    wavelength = np.array([5000.0, 5500.0])
    reference = ab_reference_flux_lambda(wavelength)
    expected_at_5500 = 3631.0e-23 * 2.99792458e18 / 5500.0**2
    assert reference[1] == pytest.approx(expected_at_5500, rel=1.0e-15)
    assert reference[1] == pytest.approx(3.5985e-9, rel=2.0e-4)


def test_constant_3631_jy_spectrum_is_zero_ab_for_any_passband() -> None:
    wavelength, passband = _grid_and_passband()
    flat_fnu_flux_lambda = (
        AB_ZERO_FNU_CGS * SPEED_OF_LIGHT_ANGSTROM_PER_SECOND / wavelength**2
    )
    assert ab_magnitude(wavelength, flat_fnu_flux_lambda, passband) == pytest.approx(
        0.0, abs=1.0e-14
    )


def test_vega_is_exactly_zero_and_passband_scale_cancels() -> None:
    wavelength, passband = _grid_and_passband()
    vega = 2.0e-9 * (wavelength / 5500.0) ** -3.0
    source = 1.7 * vega * (wavelength / 5500.0) ** 0.4
    assert vega_magnitude(wavelength, vega, vega, passband) == pytest.approx(0.0)
    first = vega_magnitude(wavelength, source, vega, passband)
    second = vega_magnitude(wavelength, source, vega, 1234.5 * passband)
    assert first == pytest.approx(second, abs=2.0e-14)


def test_same_atmospheric_passband_is_used_in_numerator_and_denominator() -> None:
    wavelength, passband = _grid_and_passband()
    vega = np.ones_like(wavelength)
    atmosphere = np.exp(-0.8 * (wavelength - wavelength.min()) / wavelength.ptp())
    observation_passband = passband * atmosphere
    # A source identical to the reference remains zero even through chromatic atmosphere.
    assert vega_magnitude(
        wavelength, vega, vega, observation_passband
    ) == pytest.approx(0.0, abs=1.0e-14)


def test_photon_factor_and_mean_wavelength() -> None:
    wavelength = np.linspace(4000.0, 6000.0, 1001)
    response = np.ones_like(wavelength)
    flux = 1.0 / wavelength
    # lambda * S_lambda is constant, so photon-weighted mean is interval midpoint.
    integral = photon_counting_integral(wavelength, response, flux)
    assert integral == pytest.approx(2000.0, rel=1.0e-14)
    assert photon_weighted_mean_wavelength(
        wavelength, response, flux
    ) == pytest.approx(5000.0, rel=1.0e-14)


def test_dispatch_rejects_unknown_system_and_requires_vega() -> None:
    wavelength, passband = _grid_and_passband(10.0)
    source = np.ones_like(wavelength)
    with pytest.raises(ValueError, match="vega_flux_lambda"):
        synthetic_magnitude(
            wavelength, source, passband, magnitude_system="vega"
        )
    with pytest.raises(ValueError, match="either 'vega' or 'ab'"):
        synthetic_magnitude(
            wavelength, source, passband, magnitude_system="st"
        )


def test_grid_convergence_helper() -> None:
    coarse_wave, coarse_passband = _grid_and_passband(10.0)
    fine_wave, fine_passband = _grid_and_passband(1.0)
    coarse_source = (coarse_wave / 5500.0) ** -2.3
    fine_source = (fine_wave / 5500.0) ** -2.3
    coarse_ref = np.ones_like(coarse_wave)
    fine_ref = np.ones_like(fine_wave)
    delta = grid_convergence_delta_magnitude(
        coarse_wave,
        coarse_source,
        coarse_passband,
        coarse_ref,
        fine_wave,
        fine_source,
        fine_passband,
        fine_ref,
    )
    assert delta < 1.0e-6
    assert check_grid_convergence(
        coarse_wave,
        coarse_source,
        coarse_passband,
        coarse_ref,
        fine_wave,
        fine_source,
        fine_passband,
        fine_ref,
        tolerance_mag=1.0e-6,
    )

