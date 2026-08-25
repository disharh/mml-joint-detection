# tests/test_populations/test_source_populations.py

import numpy as np
import pytest

from mml.populations import Source, SourceParams

def test_source_params_creation():

    params = SourceParams(
        m_VIS_Euclid=24.0,
        log10_mStar=10.5,
        Re_maj_source=0.8,
        z_source=1.2,
        q_source=0.7,
        n_sersic_source=2.5,
        log_p_source=-3.0,
        theta_light_source=1.0,
        e1_source=0.1,
        e2_source=0.2,
    )

    assert params.m_VIS_Euclid == 24.0
    assert params.log10_mStar == 10.5
    assert params.Re_maj_source == 0.8
    assert params.z_source == 1.2
    assert params.q_source == 0.7
    assert params.n_sersic_source == 2.5
    assert params.log_p_source == -3.0
    assert params.theta_light_source == 1.0
    assert params.e1_source == 0.1
    assert params.e2_source == 0.2


def test_source_params_to_dict():

    params = SourceParams(
        m_VIS_Euclid=24.0,
        log10_mStar=10.5,
        Re_maj_source=0.8,
        z_source=1.2,
        q_source=0.7,
        n_sersic_source=2.5,
        log_p_source=-3.0,
        theta_light_source=1.0,
        e1_source=0.1,
        e2_source=0.2,
    )

    result = params.to_dict()

    assert isinstance(result, dict)

    assert set(result.keys()) == {
        "m_VIS_Euclid",
        "log10_mStar",
        "Re_maj_source",
        "z_source",
        "q_source",
        "n_sersic_source",
        "log_p_source",
        "theta_light_source",
        "e1_source",
        "e2_source",
    }

    assert result["m_VIS_Euclid"] == 24.0
    assert result["z_source"] == 1.2


def test_source_params_array_values():

    size = 5

    params = SourceParams(
        m_VIS_Euclid=np.ones(size),
        log10_mStar=np.ones(size),
        Re_maj_source=np.ones(size),
        z_source=np.ones(size),
        q_source=np.ones(size),
        n_sersic_source=np.ones(size),
        log_p_source=np.ones(size),
        theta_light_source=np.ones(size),
        e1_source=np.ones(size),
        e2_source=np.ones(size),
    )

    for value in params.to_dict().values():
        assert isinstance(value, np.ndarray)
        assert value.shape == (size,)

EXPECTED_KEYS = {
    "m_VIS_Euclid",
    "log10_mStar",
    "Re_maj_source",
    "z_source",
    "q_source",
    "n_sersic_source",
    "log_p_source",
    "theta_light_source",
    "e1_source",
    "e2_source",
}


def test_source_sample_returns_expected_parameters():

    source = Source()

    params = source.sample(size=5)

    assert set(params.to_dict().keys()) == EXPECTED_KEYS


def test_source_sample_population_shape():

    source = Source()

    size = 10
    params = source.sample(size=size)

    for key, value in params.to_dict().items():
        assert np.asarray(value).shape == (size,)


def test_source_sample_scalar():

    source = Source()

    params = source.sample(size=1)

    for key, value in params.to_dict().items():
        assert np.isscalar(value)


def test_source_sample_returns_finite_values():

    source = Source()

    params = source.sample(size=10)

    for key, value in params.to_dict().items():
        assert np.all(np.isfinite(value))


def test_source_invalid_model():

    with pytest.raises(ValueError):
        Source(model="invalid_model")