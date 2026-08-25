import numpy as np
import pytest

from mml.populations.lens_populations import Lens, LensParams


# ---------------------------------------------------------------------------
# LensParams
# ---------------------------------------------------------------------------

def make_lens_params():
    """Create a simple valid LensParams instance for testing."""

    return LensParams(
        sigma_lens=200.0,
        z_lens=0.5,
        q_lens=0.8,
        ell_mass_lens=0.2,
        theta_mass_lens=0.4,
        ell_light_lens=0.15,
        theta_light_lens=0.5,
        mag_lens=20.0,
        re_lens=1.2,
        x_lens=0.01,
        y_lens=-0.02,
        e1_lens=0.1,
        e2_lens=0.05,
        gamma=2.0,
        gamma1=0.01,
        gamma2=0.02,
    )


def test_lens_params_to_dict():

    params = make_lens_params()

    result = params.to_dict()

    assert isinstance(result, dict)

    expected_fields = {
        "sigma_lens",
        "z_lens",
        "q_lens",
        "ell_mass_lens",
        "theta_mass_lens",
        "ell_light_lens",
        "theta_light_lens",
        "mag_lens",
        "re_lens",
        "x_lens",
        "y_lens",
        "e1_lens",
        "e2_lens",
        "gamma",
        "gamma1",
        "gamma2",
        "theta_ein",
    }

    assert set(result) == expected_fields


def test_lens_params_to_lenstronomy():

    params = make_lens_params()
    params.theta_ein = 1.0

    kwargs = params.to_lenstronomy()

    assert isinstance(kwargs, list)
    assert len(kwargs) == 2

    assert kwargs[0]["theta_E"] == 1.0
    assert kwargs[0]["gamma"] == params.gamma
    assert kwargs[0]["e1"] == params.e1_lens
    assert kwargs[0]["e2"] == params.e2_lens
    assert kwargs[0]["center_x"] == params.x_lens
    assert kwargs[0]["center_y"] == params.y_lens

    assert kwargs[1]["gamma1"] == params.gamma1
    assert kwargs[1]["gamma2"] == params.gamma2


def test_lens_params_to_lenstronomy_requires_theta_ein():

    params = make_lens_params()

    with pytest.raises(ValueError, match="theta_ein"):

        params.to_lenstronomy()


# ---------------------------------------------------------------------------
# Lens initialisation
# ---------------------------------------------------------------------------

def test_lens_default_initialisation():

    lens = Lens()

    assert lens.sigmazfn == "cond_on_zs"
    assert lens.separate_ellipticity is True
    assert lens.lenspos_width == 0.05
    assert lens.shear_scale == 0.05
    assert lens.gamma_mean == 2.0
    assert lens.gamma_sigma == 0.2
    assert isinstance(lens.rng, np.random.Generator)


@pytest.mark.parametrize(
    "method",
    [
        "ewoud",
        "ler",
        "cond_on_zs",
    ],
)
def test_lens_valid_sigmaz_methods(method):

    lens = Lens(sigmazfn=method)

    assert lens.sigmazfn == method


def test_lens_invalid_sigmaz_method():

    with pytest.raises(ValueError, match="Invalid sigmazfn"):

        Lens(sigmazfn="invalid")


# ---------------------------------------------------------------------------
# Lens sampling
# ---------------------------------------------------------------------------

def test_lens_sample_returns_lens_params():

    rng = np.random.default_rng(1234)

    lens = Lens(
        sigmazfn="cond_on_zs",
        rng=rng,
    )

    params = lens.sample(
        size=1,
        zs=1.0,
    )

    assert isinstance(params, LensParams)


def test_lens_sample_scalar_output():

    rng = np.random.default_rng(1234)

    lens = Lens(
        sigmazfn="cond_on_zs",
        rng=rng,
    )

    params = lens.sample(
        size=1,
        zs=1.0,
    )

    assert np.isscalar(params.sigma_lens)
    assert np.isscalar(params.z_lens)
    assert np.isscalar(params.q_lens)
    assert np.isscalar(params.ell_mass_lens)
    assert np.isscalar(params.theta_mass_lens)
    assert np.isscalar(params.ell_light_lens)
    assert np.isscalar(params.theta_light_lens)
    assert np.isscalar(params.mag_lens)
    assert np.isscalar(params.re_lens)
    assert np.isscalar(params.x_lens)
    assert np.isscalar(params.y_lens)
    assert np.isscalar(params.e1_lens)
    assert np.isscalar(params.e2_lens)
    assert np.isscalar(params.gamma)
    assert np.isscalar(params.gamma1)
    assert np.isscalar(params.gamma2)


def test_lens_sample_population_shape():

    rng = np.random.default_rng(1234)

    lens = Lens(
        sigmazfn="cond_on_zs",
        rng=rng,
    )

    size = 20

    params = lens.sample(
        size=size,
        zs=1.0,
    )

    assert isinstance(params, LensParams)

    assert params.sigma_lens.shape == (size,)
    assert params.z_lens.shape == (size,)
    assert params.q_lens.shape == (size,)

    assert params.ell_mass_lens.shape == (size,)
    assert params.theta_mass_lens.shape == (size,)

    assert params.ell_light_lens.shape == (size,)
    assert params.theta_light_lens.shape == (size,)

    assert params.mag_lens.shape == (size,)
    assert params.re_lens.shape == (size,)

    assert params.x_lens.shape == (size,)
    assert params.y_lens.shape == (size,)

    assert params.e1_lens.shape == (size,)
    assert params.e2_lens.shape == (size,)

    assert params.gamma.shape == (size,)
    assert params.gamma1.shape == (size,)
    assert params.gamma2.shape == (size,)


# ---------------------------------------------------------------------------
# Physical / parameter bounds
# ---------------------------------------------------------------------------

def test_lens_sample_bounds():

    rng = np.random.default_rng(1234)

    lens = Lens(
        sigmazfn="cond_on_zs",
        rng=rng,
    )

    params = lens.sample(
        size=100,
        zs=1.0,
    )

    assert np.all(params.sigma_lens >= 60)
    assert np.all(params.sigma_lens <= 600)

    assert np.all(params.z_lens >= 0)
    assert np.all(params.z_lens < 1.0)

    assert np.all(params.ell_light_lens >= 0)
    assert np.all(params.ell_light_lens <= 0.8)

    assert np.all(params.ell_mass_lens >= 0)
    assert np.all(params.ell_mass_lens <= 0.8)

    assert np.all(params.q_lens >= 0.2)
    assert np.all(params.q_lens <= 1.0)

    assert np.all(params.theta_light_lens >= 0)
    assert np.all(params.theta_light_lens <= np.pi)

    assert np.all(params.x_lens >= -0.05)
    assert np.all(params.x_lens <= 0.05)

    assert np.all(params.y_lens >= -0.05)
    assert np.all(params.y_lens <= 0.05)


def test_lens_sample_positive_effective_radius():

    rng = np.random.default_rng(1234)

    lens = Lens(
        sigmazfn="cond_on_zs",
        rng=rng,
    )

    params = lens.sample(
        size=100,
        zs=1.0,
    )

    assert np.all(np.isfinite(params.re_lens))
    assert np.all(params.re_lens > 0)


# ---------------------------------------------------------------------------
# Shared mass/light ellipticity
# ---------------------------------------------------------------------------

def test_lens_shared_ellipticity():

    rng = np.random.default_rng(1234)

    lens = Lens(
        sigmazfn="cond_on_zs",
        separate_ellipticity=False,
        rng=rng,
    )

    params = lens.sample(
        size=20,
        zs=1.0,
    )

    np.testing.assert_allclose(
        params.ell_mass_lens,
        params.ell_light_lens,
    )

    np.testing.assert_allclose(
        params.theta_mass_lens,
        params.theta_light_lens,
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_lens_sampling_reproducible():

    lens1 = Lens(
        sigmazfn="cond_on_zs",
        rng=np.random.default_rng(1234),
    )

    lens2 = Lens(
        sigmazfn="cond_on_zs",
        rng=np.random.default_rng(1234),
    )

    p1 = lens1.sample(
        size=20,
        zs=1.0,
    )

    p2 = lens2.sample(
        size=20,
        zs=1.0,
    )

    np.testing.assert_allclose(
        p1.sigma_lens,
        p2.sigma_lens,
    )

    np.testing.assert_allclose(
        p1.z_lens,
        p2.z_lens,
    )

    np.testing.assert_allclose(
        p1.ell_light_lens,
        p2.ell_light_lens,
    )

    np.testing.assert_allclose(
        p1.ell_mass_lens,
        p2.ell_mass_lens,
    )

    np.testing.assert_allclose(
        p1.gamma,
        p2.gamma,
    )

    np.testing.assert_allclose(
        p1.x_lens,
        p2.x_lens,
    )


# ---------------------------------------------------------------------------
# Conditional redshift requirement
# ---------------------------------------------------------------------------

def test_conditional_sampling_requires_source_redshift():

    lens = Lens(
        sigmazfn="cond_on_zs",
        rng=np.random.default_rng(1234),
    )

    with pytest.raises(ValueError, match="zs"):

        lens.sample(size=1)


# ---------------------------------------------------------------------------
# Custom sampler settings
# ---------------------------------------------------------------------------

def test_custom_lens_sampling_settings():

    lens = Lens(
        sigmazfn="cond_on_zs",
        separate_ellipticity=False,
        lenspos_width=0.1,
        shear_scale=0.1,
        gamma_mean=2.1,
        gamma_sigma=0.1,
        rng=np.random.default_rng(1234),
    )

    assert lens.separate_ellipticity is False
    assert lens.lenspos_width == 0.1
    assert lens.shear_scale == 0.1
    assert lens.gamma_mean == 2.1
    assert lens.gamma_sigma == 0.1

def test_lens_conditional_sampling_respects_size(tmp_path):

    lens = Lens(
        sigmazfn="cond_on_zs",
        rng=np.random.default_rng(1234),
    )

    params = lens.sample(
        size=10,
        zs=1.0,
    )

    assert isinstance(params, LensParams)

    assert params.sigma_lens.shape == (10,)
    assert params.z_lens.shape == (10,)
    assert params.q_lens.shape == (10,)
    assert params.mag_lens.shape == (10,)
    assert params.re_lens.shape == (10,)