import jax
import jax.numpy as jnp
import numpy as np
import pytest

from passband_fit import jax_photometry as jphot
from passband_fit.curves import atmosphere_at_airmass as numpy_atmosphere
from passband_fit.curves import trapezoid_weights as numpy_weights
from passband_fit.passband_model import instrument_passband as numpy_instrument_passband
from passband_fit.synthetic_photometry import (
    ab_magnitude as numpy_ab_magnitude,
    ab_reference_flux_lambda as numpy_ab_reference,
    photon_counting_integral as numpy_photon_integral,
    photon_weighted_mean_wavelength as numpy_mean_wavelength,
    vega_magnitude as numpy_vega_magnitude,
)


def _fixture(step: float = 2.0):
    wavelength = np.arange(3800.0, 7200.0 + step / 2.0, step, dtype=np.float64)
    nominal = np.exp(-0.5 * ((wavelength - 5350.0) / 510.0) ** 2)
    source = 2.3e-9 * (wavelength / 5500.0) ** -1.7
    vega = 3.1e-9 * (wavelength / 5500.0) ** -3.1
    extinction = 0.08 * (wavelength / 5500.0) ** -1.25
    unit_atmosphere = 10.0 ** (-0.4 * extinction)
    return wavelength, nominal, source, vega, unit_atmosphere


def test_x64_is_enabled_and_trapezoid_weights_match_numpy() -> None:
    wavelength = np.array([4000.0, 4001.5, 4010.0, 4050.0])
    weights = jphot.trapezoid_weights(wavelength)
    assert jax.config.jax_enable_x64
    assert weights.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(weights), numpy_weights(wavelength), rtol=0, atol=0)


def test_passband_remap_warp_and_atmosphere_match_numpy() -> None:
    wavelength, nominal, _, _, t1 = _fixture()
    kwargs = dict(
        pivot_wavelength_angstrom=4800.0,
        delta_wavelength_angstrom=37.5,
        width_scale=1.027,
        warp_parameter=0.16,
        warp_scale_angstrom=1200.0,
    )
    expected = numpy_instrument_passband(wavelength, nominal, **kwargs)
    actual = jphot.instrument_passband(
        wavelength,
        nominal,
        kwargs["pivot_wavelength_angstrom"],
        kwargs["delta_wavelength_angstrom"],
        kwargs["width_scale"],
        kwargs["warp_parameter"],
        kwargs["warp_scale_angstrom"],
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(
        np.asarray(jphot.atmosphere_at_airmass(t1, 1.43)),
        numpy_atmosphere(t1, 1.43),
        rtol=2e-15,
        atol=2e-15,
    )


def test_full_grid_integral_magnitudes_and_mean_match_numpy() -> None:
    wavelength, nominal, source, vega, t1 = _fixture()
    response = numpy_instrument_passband(
        wavelength,
        nominal,
        pivot_wavelength_angstrom=4800.0,
        delta_wavelength_angstrom=-21.0,
        width_scale=0.986,
        warp_parameter=-0.11,
        warp_scale_angstrom=1200.0,
    ) * numpy_atmosphere(t1, 1.27)
    weights = numpy_weights(wavelength)
    assert float(jphot.photon_counting_integral(wavelength, response, source, weights)) == pytest.approx(
        numpy_photon_integral(wavelength, response, source, weights=weights), rel=2e-15
    )
    assert float(jphot.vega_magnitude(wavelength, source, vega, response, weights)) == pytest.approx(
        numpy_vega_magnitude(wavelength, source, vega, response, weights=weights), abs=2e-14
    )
    assert float(jphot.ab_magnitude(wavelength, source, response, weights)) == pytest.approx(
        numpy_ab_magnitude(wavelength, source, response, weights=weights), abs=2e-14
    )
    assert float(jphot.photon_weighted_mean_wavelength(wavelength, response, source, weights)) == pytest.approx(
        numpy_mean_wavelength(wavelength, response, source, weights=weights), rel=2e-15
    )


def test_ab_reference_and_scale_invariance_match_numpy() -> None:
    wavelength, nominal, source, _, _ = _fixture(5.0)
    np.testing.assert_allclose(
        np.asarray(jphot.ab_reference_flux_lambda(wavelength)),
        numpy_ab_reference(wavelength),
        rtol=0,
        atol=0,
    )
    first = jphot.ab_magnitude(wavelength, source, nominal)
    second = jphot.ab_magnitude(wavelength, source, 917.0 * nominal)
    assert float(first) == pytest.approx(float(second), abs=2e-14)


def test_vectorized_observation_evaluator_matches_numpy_loop_and_jit() -> None:
    wavelength, nominal, _, vega, t1 = _fixture(4.0)
    spectra = np.asarray(
        [
            2.0e-9 * (wavelength / 5500.0) ** -3.0,
            1.1e-9 * (wavelength / 5500.0) ** -0.4,
            0.7e-9 * (wavelength / 5500.0) ** 1.2,
        ]
    )
    instrument_index = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    target_index = np.array([0, 2, 0, 1, 2], dtype=np.int64)
    airmass = np.array([1.0, 1.4, 1.1, 1.7, 1.25])
    shifts = np.array([-18.0, 24.0])
    widths = np.array([0.99, 1.03])
    warps = np.array([-0.08, 0.12])
    weights = numpy_weights(wavelength)

    function = jax.jit(jphot.observation_magnitudes)
    actual = function(
        wavelength,
        nominal,
        spectra,
        vega,
        t1,
        instrument_index,
        target_index,
        airmass,
        shifts,
        widths,
        warps,
        4800.0,
        1200.0,
        weights,
    )
    expected = []
    for instrument, target, x_value in zip(instrument_index, target_index, airmass):
        curve = numpy_instrument_passband(
            wavelength,
            nominal,
            pivot_wavelength_angstrom=4800.0,
            delta_wavelength_angstrom=shifts[instrument],
            width_scale=widths[instrument],
            warp_parameter=warps[instrument],
            warp_scale_angstrom=1200.0,
        )
        effective = curve * numpy_atmosphere(t1, x_value)
        expected.append(
            numpy_vega_magnitude(
                wavelength,
                spectra[target],
                vega,
                effective,
                weights=weights,
            )
        )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=3e-13)


def test_gradients_are_finite_and_agree_with_central_differences() -> None:
    wavelength, nominal, source, vega, t1 = _fixture(2.0)
    weights = jphot.trapezoid_weights(wavelength)

    def magnitude(parameters):
        shift, log_width, warp = parameters
        curve = jphot.instrument_passband(
            wavelength,
            nominal,
            4800.0,
            shift,
            jnp.exp(log_width),
            warp,
            1200.0,
        )
        effective = curve * jphot.atmosphere_at_airmass(t1, 1.31)
        return jphot.vega_magnitude(wavelength, source, vega, effective, weights)

    point = jnp.array([31.37, jnp.log(1.019), 0.137], dtype=jnp.float64)
    gradient = np.asarray(jax.jit(jax.grad(magnitude))(point))
    assert np.all(np.isfinite(gradient))

    step = np.array([1.0e-3, 1.0e-6, 1.0e-6])
    central = np.empty(3)
    point_numpy = np.asarray(point)
    for index in range(3):
        offset = np.zeros(3)
        offset[index] = step[index]
        central[index] = (
            float(magnitude(jnp.asarray(point_numpy + offset)))
            - float(magnitude(jnp.asarray(point_numpy - offset)))
        ) / (2.0 * step[index])
    np.testing.assert_allclose(gradient, central, rtol=3e-6, atol=3e-9)


def test_vectorized_evaluator_gradient_has_expected_shapes() -> None:
    wavelength, nominal, source, vega, t1 = _fixture(5.0)
    spectra = np.stack([source, source * (wavelength / 5500.0) ** 1.5])
    instrument_index = np.array([0, 1, 0, 1])
    target_index = np.array([0, 0, 1, 1])
    airmass = np.array([1.0, 1.2, 1.4, 1.6])
    widths = jnp.array([1.0, 1.0])
    warps = jnp.array([0.0, 0.0])

    def total(shifts):
        return jnp.sum(
            jphot.observation_magnitudes(
                wavelength,
                nominal,
                spectra,
                vega,
                t1,
                instrument_index,
                target_index,
                airmass,
                shifts,
                widths,
                warps,
                4800.0,
                1200.0,
            )
        )

    derivative = jax.grad(total)(jnp.array([-10.3, 13.7]))
    assert derivative.shape == (2,)
    assert jnp.all(jnp.isfinite(derivative))

