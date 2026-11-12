"""
Lens galaxy population models.

This module contains:
    - The local velocity-dispersion function (VDF)
    - The redshift-dependent cumulative VDF
    - The redshift evolution of the VDF
    - The lens population probability distributions
    - K-corrections (Work in progress)

The population model follows the prescriptions used in:
    Bernardi et al. (2003)
    Torrey et al. (2015)
    Wempe et al. (2024)

The default cosmology is Planck18.
"""


import numpy as np
import scipy.stats as scs
import numdifftools as nd
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy import constants as c
from scipy.special import gamma
from sklearn.preprocessing import PolynomialFeatures

# ---------------------------------------------------------------------------
# Local velocity dispersion function
# ---------------------------------------------------------------------------

def phi_loc(sigma):
    """
    Local (z ~ 0) velocity dispersion function.

    Parameters
    ----------
    sigma : float or array-like
        Velocity dispersion [km/s].

    Returns
    -------
    float or ndarray
        Differential number density at z=0,
        in units of Mpc^-3 per km/s.

    Notes
    -----
    The functional form is

        φ(σ, z=0) =
            φ* (σ/σ*)^α exp[-(σ/σ*)^β]
            β / Γ(α/β) / σ

    with parameters from Bernardi et al. (2003) /
    Wempe et al. (2024).
    """

    alpha = 0.94
    beta = 1.85

    phi_star = 2.099e-2 * (cosmo.h / 0.7) ** 3
    sigma_star = 113.78  # km/s

    phi = (
        phi_star
        * (sigma / sigma_star) ** alpha
        * np.exp(-(sigma / sigma_star) ** beta)
        * beta
        / gamma(alpha / beta)
        / sigma
    )

    return phi


# ---------------------------------------------------------------------------
# Redshift-dependent cumulative VDF
# ---------------------------------------------------------------------------

def cvdf_fit(log10_sigma, z):
    """
    Redshift-dependent cumulative velocity-dispersion-function fit.

    Parameters
    ----------
    log10_sigma : float or array-like
        log10(sigma / km/s).

    z : float or array-like
        Redshift.

    Returns
    -------
    float or ndarray
        log10 of the cumulative VDF, log10[Phi(>sigma, z)].

    Notes
    -----
    The fit is

        f(log sigma, z)
            = c0(z)
            + c1(z) m*
            + c2(z) m*^2
            - exp(m*)

    where

        m* = log10(sigma) - c3(z)

    and each coefficient has the form

        ci(z) = ai + bi z + ci z^2.
    """

    coeff_matrix = np.array([
        [7.39149763, 5.72940031, -1.12055245],
        [-6.86339338, -5.27327109, 1.10411386],
        [2.85208259, 1.25569600, -0.28663846],
        [0.06703215, -0.04868317, 0.00764841],
    ])

    coeffs = [
        row[0] + row[1] * z + row[2] * z**2
        for row in coeff_matrix
    ]

    m_star = log10_sigma - coeffs[3]

    return (
        coeffs[0]
        + coeffs[1] * m_star
        + coeffs[2] * m_star**2
        - np.exp(m_star)
    )


# ---------------------------------------------------------------------------
# Redshift evolution of the VDF
# ---------------------------------------------------------------------------

def phi_ratio(sigma, z):
    """
    Ratio of the differential VDF at redshift z to its z=0 value.

    Parameters
    ----------
    sigma : float or array-like
        Velocity dispersion [km/s].

    z : float or array-like
        Redshift.

    Returns
    -------
    float or ndarray
        phi(sigma, z) / phi(sigma, 0).
    """

    d_cvdf = nd.Derivative(lambda x: cvdf_fit(x, z))
    d_cvdf0 = nd.Derivative(lambda x: cvdf_fit(x, 0))

    log_sigma = np.log10(sigma)

    phi_z = (
        10 ** cvdf_fit(log_sigma, z)
        / sigma
        * d_cvdf(log_sigma)
    )

    phi_0 = (
        10 ** cvdf_fit(log_sigma, 0)
        / sigma
        * d_cvdf0(log_sigma)
    )

    return phi_z / phi_0


# ---------------------------------------------------------------------------
# Cosmological volume element
# ---------------------------------------------------------------------------

def dVdz(z):
    """
    Differential comoving volume element per unit redshift.

    Parameters
    ----------
    z : float or array-like
        Redshift.

    Returns
    -------
    float or ndarray
        Comoving volume element dV/dz [Mpc^3].

    Notes
    -----
    The calculation is performed over the full sky using
    the Planck18 cosmology.
    """

    D_A = cosmo.angular_diameter_distance(z).to_value(u.Mpc)
    D_H = (c.c / cosmo.H0).to_value(u.Mpc)

    return (
        4
        * np.pi
        * D_H
        * ((1 + z) ** 2 * D_A**2 / cosmo.efunc(z))
    )


# ---------------------------------------------------------------------------
# Lens population distributions
# ---------------------------------------------------------------------------

def pi_l(sigma, z):
    """
    Differential lens population distribution.

    Parameters
    ----------
    sigma : float or array-like
        Velocity dispersion [km/s].

    z : float or array-like
        Redshift.

    Returns
    -------
    float or ndarray
        Differential number density per sigma per redshift.

    Notes
    -----
    The distribution is

        pi_l(sigma, z)
            = phi_loc(sigma)
              * phi_ratio(sigma, z)
              * dV/dz.
    """

    return phi_loc(sigma) * phi_ratio(sigma, z) * dVdz(z)


def pi_l_weighted(sigma, z):
    """
    Lensing-weighted lens population distribution.

    Parameters
    ----------
    sigma : float or array-like
        Velocity dispersion [km/s].

    z : float or array-like
        Redshift.

    Returns
    -------
    float or ndarray
        Lensing-weighted population density.

    Notes
    -----
    The weighting is

        pi_l_weighted(sigma, z)
            = sigma^4 pi_l(sigma, z)

    motivated by the sigma^4 dependence of the lensing
    cross-section for SIS/SIE-type lenses.
    """

    return sigma**4 * pi_l(sigma, z)


##Lens light profile

# k-correction needed for apparent magnitude (from Wempe+ 2024)
# still need to add a default model_mean and model_std -> comes from millenium gal simulations 

#dir_tables = Path(__file__).parent.parent / 'tables'  - put apt paths

#model_mean = load_pickle(str(dir_tables / "mean.sm"))
#model_std = load_pickle(str(dir_tables / "std.sm"))

def kcormeanstd(z, Mr_, model_mean, model_std, size=1):
    """
    Compute k-correction mean and std for a given z and Mr.

    Parameters
    ----------
    z : float or ndarray
        Redshift of lens galaxy.
    Mr_ : float or ndarray
        Absolute magnitude in r-band.
    model_mean : sklearn-like regressor
        Pretrained polynomial regression model for mean k-correction.
    model_std : sklearn-like regressor
        Pretrained polynomial regression model for std k-correction.
    size : int
        Number of samples to generate.

    Returns
    -------
    mean : float or ndarray
        Predicted mean k-correction.
    std : float or ndarray
        Predicted std of k-correction.
    """
    z = np.full(size, z) if np.isscalar(z) else np.asarray(z)
    Mr_ = np.full(size, Mr_) if np.isscalar(Mr_) else np.asarray(Mr_)

    polynomial_features = PolynomialFeatures(degree=4)
    Mr = np.clip(Mr_, -25, -15)  # restrict magnitude range

    x = np.vstack([np.log(1 + z), Mr]).T
    xp = polynomial_features.fit_transform(x)

    pred_mean = model_mean.predict(xp)
    pred_mean[x[:, 0] < 0.1] = 0
    pred_std = model_std.predict(xp)

    return pred_mean, np.abs(pred_std)