import numpy as np
import pytest

from mml.lensing import einstein_radius, einstein_radius_vec


def test_einstein_radius_is_positive():
    """Einstein radius should be positive for a valid lensing configuration."""

    theta_e = einstein_radius(
        sigma=200,
        z_lens=0.5,
        z_source=1.0,
    )

    assert theta_e > 0


def test_einstein_radius_expected_value():
    """Check Einstein radius against a known reference value."""

    theta_e = einstein_radius(
        sigma=200,
        z_lens=0.5,
        z_source=1.0,
    )

    assert np.isclose(theta_e, 0.49234, rtol=1e-4)


def test_einstein_radius_invalid_redshift():
    """Source must lie behind the lens."""

    with pytest.raises(ValueError):
        einstein_radius(
            sigma=200,
            z_lens=1.0,
            z_source=0.5,
        )


def test_einstein_radius_vectorized():
    """Vectorized implementation should return one value per input."""

    sigma = np.array([150, 200, 250])
    z_lens = np.array([0.3, 0.5, 0.7])
    z_source = np.array([1.0, 1.2, 1.5])

    theta_e = einstein_radius_vec(
        sigma=sigma,
        z_lens=z_lens,
        z_source=z_source,
    )

    assert theta_e.shape == sigma.shape
    assert np.all(theta_e > 0)


def test_einstein_radius_vectorized_invalid_redshift():
    """Vectorized implementation should reject invalid lensing configurations."""

    sigma = np.array([200, 200])
    z_lens = np.array([0.5, 1.2])
    z_source = np.array([1.0, 1.0])

    with pytest.raises(ValueError):
        einstein_radius_vec(
            sigma=sigma,
            z_lens=z_lens,
            z_source=z_source,
        )