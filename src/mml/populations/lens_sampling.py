"""
Lens galaxy population sampling utilities.

This module contains sampling functions for lens-galaxy parameters
- Velocity dispersion and redshift (joint)
- Lens light and mass ellipticities and orientations
- External shear magnitude and orientation
- Lens offsets around the image centre

The population models from lens.py are used.
The default cosmology is Planck18.
"""

from pathlib import Path

import numpy as np
import scipy.stats as scs
from numpy.polynomial import chebyshev as cheb
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from ler.lens_galaxy_population import LensGalaxyParameterDistribution

from mml.utils import (
    get_cache_dir,
    uniform_sampler_from_2dpdf,
    x2u,
)

from .lens import pi_l, pi_l_weighted, kcormeanstd
from mml.populations import conditional_sigma_z


# ---------------------------------------------------------------------------
# Sampling velocity dispersion and lens redshift
# ---------------------------------------------------------------------------

def sample_sigmaz(
    size=1,
    weighted=False,
    tables_dir=None,
    rng=None,
):
    """
    Sample lens velocity dispersion and redshift.

    Samples from either the standard lens population distribution
    pi_l or the lensing-weighted distribution pi_l_weighted.

    Parameters
    ----------
    size : int, optional
        Number of samples.

    weighted : bool, optional
        If True, sample from the lensing-weighted distribution.

    tables_dir : path-like or None, optional
        Directory containing the Chebyshev inverse-CDF tables.
        If None, ``mml/tables`` is used.

    rng : numpy.random.Generator or None, optional
        Random-number generator. If None, a new generator is created.

    Returns
    -------
    sigma : ndarray
        Velocity dispersions [km/s].

    z : ndarray
        Lens redshifts.

    Notes
    -----
    If the required Chebyshev tables do not exist, they are
    generated automatically.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Use a package-level tables directory by default.
    if tables_dir is None:
        tables_dir = get_cache_dir("mml") / "lens_population"

    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if weighted:
        table_file = tables_dir / "tables_sigmaz_weighted.npz"
        pdf_func = pi_l_weighted
    else:
        table_file = tables_dir / "tables_sigmaz.npz"
        pdf_func = pi_l

    # Build tables if they do not already exist.
    if not table_file.exists():
        print("Building Chebyshev inverse-CDF tables!")

        cg_getx, cg_getz, lims = uniform_sampler_from_2dpdf(
            pdf_func,
            [[60, 600], [0.0, 3.0]],
            res_cg=[100, 120],
        )

        np.savez_compressed(
            table_file,
            cg_getx=cg_getx,
            cg_getz=cg_getz,
            lims=lims,
        )

    with np.load(table_file) as f:
        cg_getx = f["cg_getx"]
        cg_getz = f["cg_getz"]
        lims = f["lims"]

    u1 = rng.random(size)
    u2 = rng.random(size)

    sigma = cheb.chebval(
        x2u(u1, 0, 1),
        cg_getx,
    )

    z = cheb.chebval2d(
        x2u(u2, 0, 1),
        x2u(sigma, *lims[0]),
        cg_getz,
    )

    return sigma, z


def sample_sigmaz_ler(size=1):
    """
    Sample lens velocity dispersions and redshifts using LeR.

    Parameters
    ----------
    size : int, optional
        Number of samples.

    Returns
    -------
    sigma : ndarray
        Velocity dispersions [km/s].

    zl : ndarray
        Lens redshifts.
    """

    # Path to the LeR interpolator.
    json_path = Path(
        "interpolator_json",
        "source_redshift",
        "source_redshift_0.json",
    )

    create_new_interpolator = not json_path.exists()

    lens_param_samplers = {
        "velocity_dispersion": "velocity_dispersion_ewoud",
    }

    lens_param_samplers_params = {
        "velocity_dispersion": {
            "sigma_min": 60,
            "sigma_max": 600,
        }
    }

    lens = LensGalaxyParameterDistribution(
        z_min=0.0,
        z_max=3.0,
        cosmology=cosmo,
        lens_param_samplers=lens_param_samplers,
        lens_param_samplers_params=lens_param_samplers_params,
        create_new_interpolator=create_new_interpolator,
    )

    params = lens.sample_all_routine_epl_shear_intrinsic(size=size)

    sigma = params["sigma"]
    zl = params["zl"]

    return sigma, zl


# ---------------------------------------------------------------------------
# Lens ellipticity and orientation
# ---------------------------------------------------------------------------

def sample_ellipticity_theta(
    sigma,
    size,
    separate_ellipticity=True,
    rng=None
):
    """
    Sample lens light and mass ellipticities and orientations.

    Parameters
    ----------
    sigma : float or ndarray
        Velocity dispersion(s) [km/s].

    size : int
        Number of samples.

    separate_ellipticity : bool, optional
        If True, lens mass ellipticity and orientation are
        allowed to differ from the lens light.

    Returns
    -------
    ell_light : ndarray
        Lens light ellipticity.

    theta_light : ndarray
        Lens light orientation [radians].

    ell_mass : ndarray
        Lens mass ellipticity.

    theta_mass : ndarray
        Lens mass orientation [radians].
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = np.asarray(sigma)

    if sigma.size == 1:
        sigma = np.full(size, sigma)
    elif sigma.size != size:
        raise ValueError(
            "sigma must be a scalar or an array of length 'size'"
        )

    # Lens light ellipticity.
    scale = 0.378 - 5.72e-4 * sigma

    u_random = (
        rng.random(size)
        * scs.rayleigh(scale=scale).cdf(0.8)
    )

    ell_light = scs.rayleigh(scale=scale).ppf(u_random)

    # Lens light orientation.
    theta_light = rng.uniform(0, np.pi, size=size)

    # Lens mass ellipticity.
    if not separate_ellipticity:
        ell_mass = ell_light.copy()
        theta_mass = theta_light.copy()

    else:
        scale_mass = 0.2

        a = (0.0 - ell_light) / scale_mass
        b = (0.8 - ell_light) / scale_mass

        ell_mass = scs.truncnorm.rvs(
            a,
            b,
            loc=ell_light,
            scale=scale_mass,
            size=size,
            random_state=rng,
        )

        theta_mass = scs.norm.rvs(
            loc=theta_light,
            scale=34 / 180 * np.pi,
            size=size,
            random_state=rng,
        )

    return (
        ell_light,
        theta_light,
        ell_mass,
        theta_mass,
    )


# ---------------------------------------------------------------------------
# EPL slope
# ---------------------------------------------------------------------------

def sample_slope_gamma(size=1, mean=2.0, sigma=0.2, rng=None):

    """
    Sample the power-law slope gamma_m.

    Parameters
    ----------
    size : int, optional
        Number of samples.

    mean : float, optional
        Mean slope.

    sigma : float, optional
        Standard deviation.

    Returns
    -------
    ndarray
        Sampled power-law slopes.
    """

    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(mean, sigma, size=size)


# ---------------------------------------------------------------------------
# External shear
# ---------------------------------------------------------------------------


def sample_shear(size=1, scale=0.05, rng=None):

    """
    Sample external shear magnitude and orientation.

    Parameters
    ----------
    size : int, optional
        Number of samples.

    scale : float, optional
        Rayleigh scale parameter for the shear magnitude.

    Returns
    -------
    gamma_ext : ndarray
        External shear magnitude.

    phi_ext : ndarray
        External shear orientation [radians].
    """

    if rng is None:
        rng = np.random.default_rng()

    gamma_ext = rng.rayleigh(scale, size=size)
    phi_ext = rng.uniform(0, np.pi, size=size)

    return gamma_ext, phi_ext


# ---------------------------------------------------------------------------
# Lens position
# ---------------------------------------------------------------------------

def sample_lens_position(
    size=1,
    lenspos_width=0.05,
    rng=None,
):
    """
    Sample lens offsets around the image centre.

    Parameters
    ----------
    size : int, optional
        Number of samples.

    lenspos_width : float, optional
        Maximum absolute offset.

    Returns
    -------
    dx : ndarray
        x offsets.

    dy : ndarray
        y offsets.
    """
    if rng is None:
        rng = np.random.default_rng()

    random_positions = rng.random((size, 2))

    dx = (
        2 * random_positions[:, 0] - 1
    ) * lenspos_width

    dy = (
        2 * random_positions[:, 1] - 1
    ) * lenspos_width

    return dx, dy


# Fundamental plane (from Wempe+ 2024)

def sample_FP(sigma, z, ell, apply_kcorr=False, model_mean=None, model_std=None, cosmo=cosmo, rng=None):
    """
    Sample lens galaxy properties (Mr, re) from the r-band Fundamental Plane (FP)

    Parameters
    ----------
    sigma : float or ndarray
        Velocity dispersion [km/s].
    z : float or ndarray
        Redshift of the lens galaxy.
    ell : float or ndarray
        Light Ellipticity
    apply_kcorr : bool, optional
        Whether to apply k-correction. Default = False.
    model_mean, model_std : sklearn models, optional
        Pretrained regressors required if apply_kcorr=True.

    Returns
    -------
    Mr : ndarray
        Absolute r-band magnitude.
    re : ndarray
        Effective radius (in arcsecs).
    k_corr : ndarray
        K-correction needed for app mag calculation.
    """
    if rng is None:
        rng = np.random.default_rng()

    is_scalar = np.isscalar(sigma) and np.isscalar(z) and np.isscalar(ell)

    sigma = np.atleast_1d(sigma)
    z = np.atleast_1d(z)
    ell = np.atleast_1d(ell)

    size = sigma.shape[0]

    # FP parameters (r-band, rest-frame) (from Bernardi 2003)
    σ_μ = 0.610
    μ_s = 19.87
    R_s = 0.490
    σ_R = 0.241
    V_s = 2.200
    σ_V = 0.111
    ρ_Rμ = 0.760
    ρ_Vμ = 0.000
    ρ_RV = 0.543

    # Log velocity dispersion
    V = np.log10(sigma)

    mean = np.array([μ_s, R_s])
    slope = np.array([σ_μ * ρ_Vμ, σ_R * ρ_RV])

    means = mean + ((V - V_s) / σ_V)[:, None] * slope

    mu_real = means[:, 0]
    re_real = means[:, 1]

    cov = np.array([
        [σ_μ**2 * (1 - ρ_Vμ**2), σ_R * σ_μ * (ρ_Rμ - ρ_RV * ρ_Vμ)],
        [σ_R * σ_μ * (ρ_Rμ - ρ_RV * ρ_Vμ), σ_R**2 * (1 - ρ_RV**2)]
    ])

    eig, w = np.linalg.eig(cov)
    v_uniform = rng.random((size, 2))  # uniforms for mu and logR
    multivar_norm_given_cov = v_uniform @ np.diag(np.sqrt(eig)) @ w.T
    mu = mu_real + multivar_norm_given_cov[:, 0]
    re = re_real + multivar_norm_given_cov[:, 1]

    # Convert to observed magnitude
    Dl = cosmo.luminosity_distance(z)  # luminosity distance
    m_obs = (
        mu
        - 5 * np.log10((10**re * (cosmo.h / 0.7) * u.kpc / Dl).to_value(1) / (1 * u.arcsec).to_value(u.rad))
        - 2.5 * np.log10(2 * np.pi)
    )
    Mr = m_obs - 5 * np.log10((Dl / (10 * u.pc)).to_value(1))  # Absolute r-band magnitude

    # Optional k-correction (to be added later, I dont have the millenium simulation stuff worked out yet)
    if apply_kcorr:
        if model_mean is None or model_std is None:
            raise ValueError("Need model_mean and model_std for k-correction")
        kc_mean, kc_std = kcormeanstd(z, Mr, model_mean, model_std, size=size)
        u_kc = rng.random(size)
        k_corr = scs.norm.ppf(u_kc, loc=kc_mean, scale=kc_std)
        # Mr += k_corr
    else:
        k_corr = np.zeros(size)

    re = 10**re * (cosmo.h / 0.7) * (u.kpc / cosmo.angular_diameter_distance(z) * u.rad).to_value(u.arcsec)
    re /= np.sqrt(1 - ell)  # To convert from circular to major axis effective radius (the FP is fitted as circularised radii)

    if is_scalar:
        return Mr[0], re[0], k_corr[0]
    return Mr, re, k_corr


