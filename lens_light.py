import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u, constants as c
from scipy.special import gamma
import scipy.stats as scs
from sklearn.preprocessing import PolynomialFeatures
from lens_mass import *
from lenstronomy.Util import param_util

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

# Fundamental plane (from Wempe+ 2024)

#V = np.log10(sigma)
# R band:
#σ_μ = 0.610; μ_s = 19.87; R_s = 0.490; σ_R = 0.241; V_s = 2.200
#σ_V = 0.111; ρ_Rμ = 0.760; ρ_Vμ = 0.000; ρ_RV = 0.543

# the FP is sampled from in the galaxy's rest frame. Slight deviation from Goldstein. => he means that the Q correction isnt required
# Not 100% correct but close enough I think. A bit more conservative.

#means = np.array([μ_s, R_s])+(V-V_s)/σ_V*np.array([σ_μ*ρ_Vμ, σ_R*ρ_RV]) 
#cov = np.array([[σ_μ**2*(1-ρ_Vμ**2), σ_R*σ_V*(ρ_Rμ-ρ_RV*ρ_Vμ)], # why is it σ_R*σ_V here & not σ_R*σ_μ   - i think typo                          
#                [σ_R*σ_V*(ρ_Rμ-ρ_RV*ρ_Vμ), σ_R**2*(1-ρ_RV**2)]])

#I am gonna make a corrected version of the FP sampling

def sample_FP(sigma, z, ell, apply_kcorr=False, model_mean=None, model_std=None, size=1):
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
    size : int
        Number of samples to generate.

    Returns
    -------
    Mr : ndarray
        Absolute r-band magnitude.
    re : ndarray
        Effective radius (arcsec).
    k_corr : ndarray
        K-correction needed for app mag calculation.
    """
    sigma = np.full(size, sigma) if np.isscalar(sigma) else np.asarray(sigma)
    z = np.full(size, z) if np.isscalar(z) else np.asarray(z)
    ell = np.full(size, ell) if np.isscalar(ell) else np.asarray(ell)

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

    means = np.array([μ_s, R_s]) + (V - V_s) / σ_V * np.array([σ_μ * ρ_Vμ, σ_R * ρ_RV])
    mu_real, re_real = means.T if means.ndim > 1 else means

    cov = np.array([
        [σ_μ**2 * (1 - ρ_Vμ**2), σ_R * σ_μ * (ρ_Rμ - ρ_RV * ρ_Vμ)],
        [σ_R * σ_μ * (ρ_Rμ - ρ_RV * ρ_Vμ), σ_R**2 * (1 - ρ_RV**2)]
    ])

    eig, w = np.linalg.eig(cov)
    v_uniform = np.random.rand(size, 2)  # uniforms for mu and logR
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
        u_kc = np.random.rand(size)
        k_corr = scs_norm.ppf(u_kc, loc=kc_mean, scale=kc_std)
        # Mr += k_corr
    else:
        k_corr = np.zeros(size)

    re = 10**re * (cosmo.h / 0.7) * (u.kpc / cosmo.angular_diameter_distance(z) * u.rad).to_value(u.arcsec)
    re /= np.sqrt(1 - ell)  # To convert from circular to major axis effective radius (the FP is fitted as circularised radii)

    return Mr, re, k_corr


def sample_lens_params(size=1, sigmazfn='ler'):
    if sigmazfn=='ewoud':
        sigma_lens, z_lens = sample_sigmaz(size)
    elif sigmazfn=='ler':
        sigma_lens, z_lens = sample_sigmaz_ler(size)
    else:
        raise ValueError("Invalid sigmazfn! Choose between 'ewoud' and 'ler'.")

    ell_light_lens, theta_light_lens, ell_mass_lens, theta_mass_lens = sample_ellipticity_theta(
        sigma=sigma_lens,
        size=size
    )

    gamma_slope = sample_slope_gamma(size)

    gamma_shear, phi_shear = sample_shear(size)

    x_lens, y_lens = sample_lens_position(size)

    Mr, re, k_corr = sample_FP(sigma_lens, z_lens, ell_light_lens)

    mag_lens = Mr + 5 * np.log10(
        (cosmo.luminosity_distance(z_lens) / (10 * u.pc)).to_value(1)
    ) + k_corr

    e1, e2 = param_util.phi_q2_ellipticity(
        phi= theta_mass_lens,
        q= 1 - ell_mass_lens
    )

    gamma1, gamma2 = param_util.shear_polar2cartesian(
        phi=phi_shear,
        gamma=gamma_shear
    )

    lens_params = {
        'sigma_lens' : sigma_lens,
        'z_lens': z_lens,
        'q_lens': 1 - ell_mass_lens,
        'ell_mass_lens' : ell_mass_lens,
        'theta_mass_lens': theta_mass_lens,
        'ell_light_lens' : ell_mass_lens,
        'theta_light_lens': theta_mass_lens,
        'mag_lens' : mag_lens,
        'x_lens' : x_lens,
        'y_lens' : y_lens,
        're_lens' : re,
        'e1_lens' : e1,
        'e2_lens' : e2,
        'gamma': gamma_slope,
        'gamma1': gamma1,
        'gamma2': gamma2
    }

    if size == 1:
        lens_params = {k: float(v[0]) for k, v in lens_params.items()}

    return lens_params