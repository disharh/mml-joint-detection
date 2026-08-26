import numpy as np
import pytest

from mml.populations.gw_populations import (
    GWPopulation,
    GWParams,
)


EXPECTED_KEYS = {
    "mass_1_source",
    "mass_2_source",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "ra",
    "dec",
    "psi",
    "geocent_time",
    "phase",
}


# ---------------------------------------------------------------------------
# GWParams
# ---------------------------------------------------------------------------

def test_gw_params_creation():

    params = GWParams(
        mass_1_source=30.0,
        mass_2_source=20.0,
        a_1=0.5,
        a_2=0.3,
        tilt_1=1.0,
        tilt_2=0.5,
        phi_12=2.0,
        phi_jl=1.5,
        theta_jn=0.8,
        ra=1.0,
        dec=0.5,
        psi=0.7,
        geocent_time=1000000000.0,
        phase=2.5,
    )

    assert params.mass_1_source == 30.0
    assert params.mass_2_source == 20.0
    assert params.a_1 == 0.5
    assert params.ra == 1.0


def test_gw_params_to_dict():

    params = GWParams(
        mass_1_source=30.0,
        mass_2_source=20.0,
        a_1=0.5,
        a_2=0.3,
        tilt_1=1.0,
        tilt_2=0.5,
        phi_12=2.0,
        phi_jl=1.5,
        theta_jn=0.8,
        ra=1.0,
        dec=0.5,
        psi=0.7,
        geocent_time=1000000000.0,
        phase=2.5,
    )

    result = params.to_dict()

    assert isinstance(result, dict)
    assert set(result.keys()) == EXPECTED_KEYS | {"luminosity_distance"}

    assert result["mass_1_source"] == 30.0


# ---------------------------------------------------------------------------
# GWPopulation
# ---------------------------------------------------------------------------

def test_gw_population_initialisation():

    population = GWPopulation()

    assert population.event_type == "BBH"
    assert population.spin_zero is False
    assert population.spin_precession is True
    assert population.cbc is not None


def test_gw_population_sample_scalar():

    population = GWPopulation()

    params = population.sample(size=1)

    assert isinstance(params, GWParams)

    for key in EXPECTED_KEYS:
        assert np.isscalar(getattr(params, key))


def test_gw_population_sample_population():

    population = GWPopulation()

    size = 5

    params = population.sample(size=size)

    assert isinstance(params, GWParams)

    for key in EXPECTED_KEYS:
        value = getattr(params, key)

        assert isinstance(value, np.ndarray)
        assert value.shape == (size,)


def test_gw_population_sample_finite():

    population = GWPopulation()

    params = population.sample(size=5)

    for key in EXPECTED_KEYS:
        value = getattr(params, key)

        assert np.all(np.isfinite(value))


# ---------------------------------------------------------------------------
# Morse phase
# ---------------------------------------------------------------------------
def test_compute_morse_phase():

    hessian = (
        np.array([0.2, 0.2, 2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.2, 2.0, 2.0]),
    )

    morse = GWPopulation.compute_morse_phase(hessian)

    # Minimum: det(A) > 0 and trace(A) > 0
    assert np.isclose(morse[0], 0.0)

    # Saddle: det(A) < 0
    assert np.isclose(morse[1], np.pi / 2)

    # Maximum: det(A) > 0 and trace(A) < 0
    assert np.isclose(morse[2], np.pi)


# ---------------------------------------------------------------------------
# Effective parameters
# ---------------------------------------------------------------------------

def test_compute_effective_params():

    gw_params = {
        "luminosity_distance": 1000.0,
        "geocent_time": 100.0,
        "phase": 1.0,
        "ra": 0.5,
        "dec": 0.3,
    }

    magnifications = np.array([1.0, 4.0])
    delays = np.array([0.0, 2.0])

    hessian = (
        np.array([0.2, 0.2]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.2, 0.2]),
    )

    x_image = np.array([0.0, 1.0])
    y_image = np.array([0.0, 1.0])

    result = GWPopulation.compute_effective_params(
        gw_params=gw_params,
        magnifications=magnifications,
        delays=delays,
        hessian=hessian,
        x_image=x_image,
        y_image=y_image,
        x_gw=0.0,
        y_gw=0.0,
    )

    assert np.allclose(
        result["effective_luminosity_distance"],
        [1000.0, 500.0],
    )

    assert np.allclose(
        result["effective_geocent_time"],
        [100.0, 102.0],
    )

    assert np.allclose(
        result["effective_phase"],
        [1.0, 1.0],
    )


def test_compute_effective_params_requires_luminosity_distance():

    gw_params = {
        "geocent_time": 100.0,
        "phase": 1.0,
        "ra": 0.5,
        "dec": 0.3,
    }

    with pytest.raises(KeyError):

        GWPopulation.compute_effective_params(
            gw_params=gw_params,
            magnifications=np.array([1.0]),
            delays=np.array([0.0]),
            hessian=(
                np.array([0.2]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.2]),
            ),
            x_image=np.array([0.0]),
            y_image=np.array([0.0]),
            x_gw=0.0,
            y_gw=0.0,
        )