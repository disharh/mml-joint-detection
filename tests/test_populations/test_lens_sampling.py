import numpy as np

from mml.populations import (
    sample_sigmaz,
    sample_ellipticity_theta,
    sample_slope_gamma,
    sample_shear,
    sample_lens_position,
)


def test_sample_sigmaz_shape():

    rng = np.random.default_rng(1234)

    sigma, z = sample_sigmaz(
        size=10,
        rng=rng,
    )

    assert sigma.shape == (10,)
    assert z.shape == (10,)


def test_sample_sigmaz_bounds():

    rng = np.random.default_rng(1234)

    sigma, z = sample_sigmaz(
        size=100,
        rng=rng,
    )

    assert np.all(sigma >= 60)
    assert np.all(sigma <= 600)

    assert np.all(z >= 0)
    assert np.all(z <= 3)


def test_sample_sigmaz_reproducible():

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    sigma1, z1 = sample_sigmaz(
        size=10,
        rng=rng1,
    )

    sigma2, z2 = sample_sigmaz(
        size=10,
        rng=rng2,
    )

    np.testing.assert_allclose(sigma1, sigma2)
    np.testing.assert_allclose(z1, z2)


def test_sample_slope_gamma_shape():

    rng = np.random.default_rng(1234)

    gamma = sample_slope_gamma(
        size=20,
        rng=rng,
    )

    assert gamma.shape == (20,)


def test_sample_shear_shape():

    rng = np.random.default_rng(1234)

    gamma_ext, phi_ext = sample_shear(
        size=20,
        rng=rng,
    )

    assert gamma_ext.shape == (20,)
    assert phi_ext.shape == (20,)


def test_sample_shear_bounds():

    rng = np.random.default_rng(1234)

    gamma_ext, phi_ext = sample_shear(
        size=100,
        rng=rng,
    )

    assert np.all(gamma_ext >= 0)
    assert np.all(phi_ext >= 0)
    assert np.all(phi_ext <= np.pi)


def test_sample_lens_position_shape():

    rng = np.random.default_rng(1234)

    dx, dy = sample_lens_position(
        size=20,
        rng=rng,
    )

    assert dx.shape == (20,)
    assert dy.shape == (20,)


def test_sample_lens_position_bounds():

    width = 0.05

    rng = np.random.default_rng(1234)

    dx, dy = sample_lens_position(
        size=100,
        lenspos_width=width,
        rng=rng,
    )

    assert np.all(dx >= -width)
    assert np.all(dx <= width)

    assert np.all(dy >= -width)
    assert np.all(dy <= width)


def test_sample_ellipticity_theta_shape():

    rng = np.random.default_rng(1234)

    sigma = np.full(20, 200.0)

    (
        ell_light,
        theta_light,
        ell_mass,
        theta_mass,
    ) = sample_ellipticity_theta(
        sigma=sigma,
        size=20,
        rng=rng,
    )

    assert ell_light.shape == (20,)
    assert theta_light.shape == (20,)
    assert ell_mass.shape == (20,)
    assert theta_mass.shape == (20,)


def test_sample_ellipticity_bounds():

    rng = np.random.default_rng(1234)

    sigma = np.full(100, 200.0)

    (
        ell_light,
        theta_light,
        ell_mass,
        theta_mass,
    ) = sample_ellipticity_theta(
        sigma=sigma,
        size=100,
        rng=rng,
    )

    assert np.all(ell_light >= 0)
    assert np.all(ell_light <= 0.8)

    assert np.all(ell_mass >= 0)
    assert np.all(ell_mass <= 0.8)

    assert np.all(theta_light >= 0)
    assert np.all(theta_light <= np.pi)


def test_shared_light_and_mass_ellipticity():

    rng = np.random.default_rng(1234)

    sigma = np.full(20, 200.0)

    (
        ell_light,
        theta_light,
        ell_mass,
        theta_mass,
    ) = sample_ellipticity_theta(
        sigma=sigma,
        size=20,
        separate_ellipticity=False,
        rng=rng,
    )

    np.testing.assert_array_equal(
        ell_light,
        ell_mass,
    )

    np.testing.assert_array_equal(
        theta_light,
        theta_mass,
    )