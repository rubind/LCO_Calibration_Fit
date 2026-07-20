import numpy as np
import pytest

from passband_fit.contracts import SpectralCurve, Spectrum
from passband_fit.curves import (
    combine_nominal_components,
    interpolate_curve,
    interpolate_spectrum,
    passband_support_mask,
    passband_support_interval,
    spectrum_coverage_error,
    spectrum_covers_passband,
    trapezoid_integral,
    trapezoid_weights,
)


def test_interpolation_is_float64_linear_and_zero_outside() -> None:
    curve = SpectralCurve(
        np.array([4000, 5000, 6000], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int16),
        "triangle",
    )
    target = np.array([3500, 4500, 5500, 6500], dtype=np.int32)
    result = interpolate_curve(curve, target)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [0.0, 0.5, 0.5, 0.0])


def test_trapezoid_weights_integrate_nonuniform_linear_function_exactly() -> None:
    wavelength = np.array([1.0, 1.5, 2.75, 5.0])
    values = 3.0 * wavelength + 2.0
    weights = trapezoid_weights(wavelength)
    expected = 1.5 * (5.0**2 - 1.0**2) + 2.0 * (5.0 - 1.0)
    assert trapezoid_integral(values, weights=weights) == pytest.approx(expected)
    stacked = np.stack([values, 2.0 * values])
    np.testing.assert_allclose(
        trapezoid_integral(stacked, wavelength, axis=1), [expected, 2.0 * expected]
    )


def test_nominal_combination_uses_aluminum_squared_as_supplied() -> None:
    filt = np.array([0.5, 0.8])
    aluminum_squared = np.array([0.64, 0.81])
    qe = np.array([0.7, 0.9])
    combined = combine_nominal_components(filt, aluminum_squared, qe)
    np.testing.assert_allclose(combined, filt * aluminum_squared * qe)
    assert not np.allclose(combined, filt * aluminum_squared**2 * qe)


def test_passband_support_and_spectrum_coverage() -> None:
    grid = np.arange(4000.0, 6001.0, 100.0)
    passband = np.exp(-0.5 * ((grid - 5000.0) / 200.0) ** 2)
    support = passband_support_mask(passband, 1.0e-3)
    spectrum_wave = np.arange(grid[support][0], grid[support][-1] + 1.0, 10.0)
    assert spectrum_covers_passband(
        spectrum_wave, grid, passband, relative_threshold=1.0e-3
    )
    assert not spectrum_covers_passband(
        spectrum_wave[1:], grid, passband, relative_threshold=1.0e-3
    )
    assert passband_support_interval(
        grid, passband, relative_threshold=1.0e-3
    ) == pytest.approx((4300.0, 5700.0))


def test_spectrum_coverage_error_reports_required_range_and_missing_tails() -> None:
    grid = np.arange(4000.0, 6001.0, 100.0)
    passband = np.zeros_like(grid)
    passband[3:-3] = 1.0
    spectrum = Spectrum(
        np.arange(4400.0, 5501.0, 25.0),
        np.ones(45),
        "short-reference",
    )
    detail = spectrum_coverage_error(spectrum, grid, passband)
    assert detail is not None
    assert "available 4400.0--5500.0 Angstrom" in detail
    assert "required 4300.0--5700.0 Angstrom" in detail
    assert "blue tail by 100.0 Angstrom" in detail
    assert "red tail by 200.0 Angstrom" in detail


def test_spectrum_interpolation_never_extrapolates_material_support() -> None:
    grid = np.arange(4000.0, 6001.0, 100.0)
    passband = np.zeros_like(grid)
    passband[5:-5] = 1.0
    spectrum = Spectrum(
        np.arange(4400.0, 5601.0, 25.0),
        np.ones(49),
        "covered",
    )
    interpolated = interpolate_spectrum(spectrum, grid, passband=passband)
    assert np.all(interpolated[passband > 0.0] == 1.0)
    short = Spectrum(np.arange(4600.0, 5501.0, 25.0), np.ones(37), "short")
    with pytest.raises(ValueError, match="does not cover"):
        interpolate_spectrum(short, grid, passband=passband)


def test_invalid_grids_and_negative_passband_fail() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        trapezoid_weights(np.array([4000.0, 4100.0, 4050.0]))
    with pytest.raises(ValueError, match="negative"):
        passband_support_mask(np.array([0.0, -0.1, 1.0]))
