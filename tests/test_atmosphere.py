import numpy as np
import pytest

from passband_fit.curves import (
    atmosphere_at_airmass,
    extinction_magnitude_to_transmission,
)
from passband_fit.passband_model import observation_passband


def test_extinction_magnitudes_convert_to_unit_airmass_transmission() -> None:
    extinction = np.array([0.0, 0.1, 1.0])
    transmission = extinction_magnitude_to_transmission(extinction)
    np.testing.assert_allclose(transmission, 10.0 ** (-0.4 * extinction))


def test_atmosphere_scales_as_unit_transmission_to_airmass_power() -> None:
    transmission_one = np.array([1.0, 0.9, 0.5, 0.0])
    transmission_two = atmosphere_at_airmass(transmission_one, 2.0)
    np.testing.assert_allclose(transmission_two, transmission_one**2)


def test_negative_extinction_clips_by_default_and_can_be_strict() -> None:
    result = extinction_magnitude_to_transmission(np.array([-1.0e-6, 0.2]))
    assert result[0] == 1.0
    with pytest.raises(ValueError, match="cannot be negative"):
        extinction_magnitude_to_transmission(
            np.array([-1.0e-6, 0.2]), clip_negative=False
        )


def test_invalid_transmission_and_airmass_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        atmosphere_at_airmass(np.array([0.8, 1.01]), 1.2)
    with pytest.raises(ValueError, match="positive"):
        atmosphere_at_airmass(np.array([0.8, 0.9]), 0.0)


def test_observation_passband_multiplies_atmosphere_once() -> None:
    instrument = np.array([0.2, 0.8, 1.0])
    transmission_one = np.array([0.9, 0.7, 0.5])
    result = observation_passband(instrument, transmission_one, 1.5)
    np.testing.assert_allclose(result, instrument * transmission_one**1.5)

