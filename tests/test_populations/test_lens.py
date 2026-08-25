import numpy as np

from mml.populations import (
    phi_loc,
    cvdf_fit,
    phi_ratio,
    dVdz,
    pi_l,
    pi_l_weighted,
)


def test_phi_loc_positive():
    """The local velocity-dispersion function should be positive."""

    sigma = np.array([60.0, 100.0, 200.0, 400.0])

    result = phi_loc(sigma)

    assert np.all(result > 0)


def test_cvdf_fit_returns_finite_values():
    """The CVDF fit should return finite values."""

    log10_sigma = np.array([np.log10(60.0), np.log10(200.0), np.log10(500.0)])
    z = 0.5

    result = cvdf_fit(log10_sigma, z)

    assert np.all(np.isfinite(result))


def test_phi_ratio_at_zero_redshift():
    """
    At z=0, phi(sigma,z) / phi(sigma,0) should equal 1.
    """

    sigma = np.array([60.0, 100.0, 200.0, 400.0])

    result = phi_ratio(sigma, 0.0)

    np.testing.assert_allclose(result, 1.0)


def test_dVdz_positive():
    """The comoving volume element should be positive for z > 0."""

    z = np.array([0.1, 0.5, 1.0, 2.0])

    result = dVdz(z)

    assert np.all(result > 0)


def test_pi_l_positive():
    """The lens population density should be positive."""

    result = pi_l(200.0, 0.5)

    assert result > 0


def test_pi_l_weighted_positive():
    """The lensing-weighted population density should be positive."""

    result = pi_l_weighted(200.0, 0.5)

    assert result > 0


def test_weighted_population_is_sigma4_times_unweighted():
    """Check the definition of the lensing-weighted population."""

    sigma = 200.0
    z = 0.5

    unweighted = pi_l(sigma, z)
    weighted = pi_l_weighted(sigma, z)

    np.testing.assert_allclose(
        weighted,
        sigma**4 * unweighted,
    )