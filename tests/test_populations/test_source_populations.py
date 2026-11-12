"""Tests for source-galaxy population sampling.

The trained denmarf files are deliberately not required by this test suite:
the DensityEstimate loader is mocked so the tests remain fast and portable.
"""

from pathlib import Path

import numpy as np
import pytest

import mml.populations.source_populations as source_module
from mml.populations import Source, SourceParams


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


class FakeDensityEstimate:
    """Small deterministic stand-in for denmarf.DensityEstimate."""

    loaded_filename = None

    @classmethod
    def from_file(cls, filename):
        cls.loaded_filename = filename
        return cls()

    def sample(self, size):
        # Columns correspond to the six trained source-population features.
        return np.tile(
            np.array([24.0, 10.5, 0.8, 1.2, 0.7, 2.5]),
            (size, 1),
        )

    def score_samples(self, samples):
        return np.full(len(samples), -3.0)


@pytest.fixture
def fake_density_estimate(monkeypatch):
    monkeypatch.setattr(
        source_module, "DensityEstimate", FakeDensityEstimate
    )
    FakeDensityEstimate.loaded_filename = None
    return FakeDensityEstimate


@pytest.fixture
def model_file(tmp_path):
    """Create a placeholder model path; loading itself is mocked."""
    path = tmp_path / "fake_source_model.pkl"
    path.touch()
    return path


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
    assert set(result) == EXPECTED_KEYS
    assert result["m_VIS_Euclid"] == 24.0
    assert result["z_source"] == 1.2


def test_source_params_to_lenstronomy():
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

    assert params.to_lenstronomy() == {
        "re_source": 0.8,
        "nsersic_source": 2.5,
        "e1_source": 0.1,
        "e2_source": 0.2,
    }


def test_source_params_array_values():
    size = 5
    params = SourceParams(**{key: np.ones(size) for key in EXPECTED_KEYS})

    for value in params.to_dict().values():
        assert isinstance(value, np.ndarray)
        assert value.shape == (size,)


@pytest.mark.parametrize(
    ("model", "filename"),
    [
        (
            "mstar_weighted",
            "trained_de_weighted_mass_b32_h128_e1000_sizen.pkl",
        ),
        (
            "non_weighted",
            "trained_de_b32_h128_e1000_bdef_wbounds.pkl",
        ),
    ],
)
def test_default_model_path_is_inside_package_data(model, filename):
    path = Source._default_model_path(model)

    assert path.name == filename
    assert path.parent == (
        Path(source_module.__file__).resolve().parents[1] / "data/trained_models"
    )


def test_source_uses_explicit_model_path(
    fake_density_estimate, model_file
):
    source = Source(model_path=model_file)

    assert source.model_path == model_file.resolve()
    assert FakeDensityEstimate.loaded_filename == str(model_file.resolve())


def test_source_uses_default_model_path(
    fake_density_estimate, monkeypatch, tmp_path
):
    expected = tmp_path / Source.DEFAULT_MODELS["non_weighted"]
    expected.touch()
    monkeypatch.setattr(
        Source, "_default_model_path", classmethod(lambda cls, model: expected)
    )

    source = Source()

    assert source.model_path == expected.resolve()
    assert FakeDensityEstimate.loaded_filename == str(expected.resolve())


def test_source_missing_model_raises_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist.pkl"

    with pytest.raises(FileNotFoundError, match="Source population model not found"):
        Source(model_path=missing)


def test_source_invalid_model_raises_value_error():
    with pytest.raises(ValueError, match="Invalid model"):
        Source(model="invalid_model")


def test_source_sample_returns_expected_parameters(
    fake_density_estimate, model_file
):
    source = Source(model_path=model_file, rng=np.random.default_rng(1234))
    params = source.sample(size=5)

    assert set(params.to_dict()) == EXPECTED_KEYS


def test_source_sample_population_shape(fake_density_estimate, model_file):
    source = Source(model_path=model_file, rng=np.random.default_rng(1234))
    size = 10
    params = source.sample(size=size)

    for value in params.to_dict().values():
        assert np.asarray(value).shape == (size,)


def test_source_sample_size_one_returns_scalars(
    fake_density_estimate, model_file
):
    source = Source(model_path=model_file, rng=np.random.default_rng(1234))
    params = source.sample(size=1)

    for value in params.to_dict().values():
        assert np.isscalar(value)


def test_source_sample_returns_finite_values(
    fake_density_estimate, model_file
):
    source = Source(model_path=model_file, rng=np.random.default_rng(1234))
    params = source.sample(size=10)

    for value in params.to_dict().values():
        assert np.all(np.isfinite(value))


def test_source_sample_reproducible_for_same_seed(
    fake_density_estimate, model_file
):
    source1 = Source(model_path=model_file, rng=np.random.default_rng(42))
    source2 = Source(model_path=model_file, rng=np.random.default_rng(42))

    params1 = source1.sample(size=8)
    params2 = source2.sample(size=8)

    for key in EXPECTED_KEYS:
        np.testing.assert_array_equal(
            np.asarray(params1.to_dict()[key]),
            np.asarray(params2.to_dict()[key]),
        )


@pytest.mark.parametrize("size", [0, -1])
def test_source_sample_rejects_nonpositive_size(
    fake_density_estimate, model_file, size
):
    source = Source(model_path=model_file)

    with pytest.raises(ValueError, match="size must be >= 1"):
        source.sample(size=size)


def test_source_sample_rejects_non_integer_size(
    fake_density_estimate, model_file
):
    source = Source(model_path=model_file)

    with pytest.raises(TypeError, match="size must be an integer"):
        source.sample(size=2.5)
