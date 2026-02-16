import numpy as np
import os
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u, constants as c
from scipy.special import gamma
import numdifftools as nd
import scipy.stats as scs
from chebtools import *
from pathlib import Path
from numpy.polynomial import chebyshev as cheb
from ler.lens_galaxy_population import LensGalaxyParameterDistribution

## Assumes the Planck18 cosmology (Planck Collaboration 2020).

# Local velocity dispersion function (VDF) (Bernardi 2003/Wempe+ 2024)

def phi_loc(sigma):
    """
    Local (z ~ 0) velocity dispersion function, Schechter-like form.
    
    Parameters
    ----------
    sigma : float or array
        Velocity dispersion [km/s].
    
    Returns
    -------
    phi_loc : float or array
        Differential number density at z=0, in units of Mpc^-3 (per km/s).
    
    Equation
    --------
    φ(σ, z=0) = φ* * (σ/σ*)^α * exp[-(σ/σ*)^β] * β / Γ(α/β) * (1/σ)
    
    where:
        α = 0.94 (low-σ slope)
        β = 1.85 (cutoff slope)
        σ* = 113.78 km/s (characteristic velocity dispersion)
        φ* = 2.099e-2 * (h/0.7)^3 Mpc^-3 (normalization)
    """
    alpha = 0.94
    beta = 1.85
    phi_star = 2.099e-2 * (cosmo.h / 0.7) ** 3  # normalization [Mpc^-3]
    sigma_star = 113.78  # km/s

    phi = (phi_star * (sigma / sigma_star) ** alpha *
           np.exp(-(sigma / sigma_star) ** beta) *
           beta / gamma(alpha / beta) / sigma)
    return phi


# Redshift-dependent cumulative VDF fit (Torrey+ 2015/Wempe+ 2024)

def cvdf_fit(log10_sigma, z):
    """
    Fit to cumulative velocity dispersion function (CVDF).
    
    Parameters
    ----------
    log10_sigma : float
        log10(σ / km/s).
    z : float
        Redshift.
    
    Returns
    -------
    f : float
        log10[ Φ(>σ, z) ], where Φ is the cumulative VDF.
    
    Equation
    --------
    f(logσ, z) = c0(z) + c1(z) * m* + c2(z) * m*^2 - exp(m*)
    
    where:
        m* = log10σ - c3(z)
        ci(z) = ai + bi z + ci z^2   (redshift-dependent coefficients)
    """
    
    coeff_matrix = np.array([
        [ 7.39149763,  5.72940031, -1.12055245],  # c0
        [-6.86339338, -5.27327109,  1.10411386],  # c1
        [ 2.85208259,  1.25569600, -0.28663846],  # c2
        [ 0.06703215, -0.04868317,  0.00764841]   # c3
    ])
    coeffs = [row[0] + row[1]*z + row[2]*z**2 for row in coeff_matrix]
    m_star = log10_sigma - coeffs[3]
    return coeffs[0] + coeffs[1]*m_star + coeffs[2]*m_star**2 - np.exp(m_star)


# Evolution factor φ(σ, z) / φ(σ, 0) (Wempe+ 2024)

def phi_ratio(sigma, z):
    """
    Ratio of differential VDF at redshift z to z=0.
    
    Parameters
    ----------
    sigma : float
        Velocity dispersion [km/s].
    z : float
        Redshift.
    
    Returns
    -------
    ratio : float
        φ(σ, z) / φ(σ, 0)
    
    Equation
    --------
    φ(σ, z) = [10^{f(logσ, z)} / σ] * (∂f/∂logσ)(z)
    ratio = φ(σ, z) / φ(σ, 0)
    """
    d_cvdf = nd.Derivative(lambda x: cvdf_fit(x, z))
    d_cvdf0 = nd.Derivative(lambda x: cvdf_fit(x, 0))

    log_sigma = np.log10(sigma)

    phi_z = (10 ** cvdf_fit(log_sigma, z) / sigma *
         d_cvdf(log_sigma))

    phi_0 = (10 ** cvdf_fit(log_sigma, 0) / sigma *
         d_cvdf0(log_sigma))

    return phi_z / phi_0


# Comoving volume element dV/dz (Wempe+ 2024)

def dVdz(z):
    """
    Differential comoving volume element per unit redshift, over full sky.
    
    Parameters
    ----------
    z : float
        Redshift.
    
    Returns
    -------
    dVdz : float
        Comoving volume element dV/dz [Mpc^3].
    
    Equation
    --------
    dV/dz = 4π D_H * ( (1+z)^2 * D_A(z)^2 / E(z) )
    
    where:
        D_H = c / H0   (Hubble distance)
        D_A(z) = angular diameter distance
        E(z) = H(z)/H0 = cosmo.efunc(z)
    """
    D_A = cosmo.angular_diameter_distance(z).to_value(u.Mpc)
    D_H = (c.c / cosmo.H0).to_value(u.Mpc)
    return 4 * np.pi * D_H * ((1 + z)**2 * D_A**2 / cosmo.efunc(z))


# Full redshift-dependent VDF (Wempe+ 2024)

def pi_l(sigma, z):
    """
    Differential velocity dispersion function at redshift z.
    
    Parameters
    ----------
    sigma : float
        Velocity dispersion [km/s].
    z : float
        Redshift.
    
    Returns
    -------
    pi_l : float
        Differential number density (per σ per z).
    
    Equation
    --------
    pi_l(σ, z) = φ_loc(σ) * [φ(σ, z)/φ(σ, 0)] * (dV/dz)
    """
    return phi_loc(sigma) * phi_ratio(sigma, z) * dVdz(z)


# Lensing-weighted distribution (Wempe+ 2024)

def pi_l_weighted(sigma, z):
    """
    Lensing-weighted velocity dispersion function.
    
    Parameters
    ----------
    sigma : float
        Velocity dispersion [km/s].
    z : float
        Redshift.
    
    Returns
    -------
    pi_w : float
        Weighted number density.
    
    Equation
    --------
    pi_w(σ, z) = σ^4 * pi(σ, z)
    
    Motivation
    ----------
    In SIS/SIE lens models, the lensing cross-section ∝ σ^4.
    So this weighting gives the effective distribution of lensing galaxies.
    """
    return sigma**4 * pi_l(sigma, z)


def sample_sigmaz(size=1, weighted=False, tables_dir=None, rng=None):
    """
    Sample velocity dispersion σ and redshift z for the lens galaxy from the 2D joint distribution pi_l / pi_l_weighted

    This function automatically:
    - builds Chebyshev inverse-CDF tables if missing
    - loads tables if they already exist
    - returns samples from π_l(σ,z) or σ^4 π_l(σ,z)

    Parameters
    ----------
    size : int
        Number of samples.
    weighted : bool
        If True, sample from lensing-weighted population.
    tables_dir : Path or str or None
        Where Chebyshev tables are stored.
    rng : np.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    sigma : ndarray
    z : ndarray
    """


    if rng is None:
        rng = np.random.default_rng()

    if tables_dir is None:
        tables_dir = Path(__file__).parent.parent / "tables"
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(exist_ok=True)

    if weighted:
        table_file = tables_dir / "tables_sigmaz_weighted.npz"
        pdf_func = pi_l_weighted
    else:
        table_file = tables_dir / "tables_sigmaz.npz"
        pdf_func = pi_l

    # Build tables if needed
    if not table_file.exists():
        print("Building Chebyshev inverse-CDF tables!")

        cg_getx, cg_getz, lims = uniform_sampler_from_2dpdf(
            pdf_func,
            [[60, 600], [0., 3.]],   # σ,z limits
            res_cg=[60, 80] #res_cg=[300,400] : problematic memory allocation
        )

        np.savez_compressed(
            table_file,
            cg_getx=cg_getx,
            cg_getz=cg_getz,
            lims=lims
        )

    f = np.load(table_file)
    cg_getx = f["cg_getx"]
    cg_getz = f["cg_getz"]
    lims = f["lims"]

    u1 = rng.random(size)
    u2 = rng.random(size)

    sigma = cheb.chebval(x2u(u1, 0, 1), cg_getx)

    z = cheb.chebval2d(
        x2u(u2, 0, 1),
        x2u(sigma, *lims[0]),
        cg_getz
    )

    return sigma, z

def sample_sigmaz_ler(size=1):
    """
    Sample lens velocity dispersions and redshifts using LeR.

    Parameters
    ----------
    size : int
        Number of samples to generate (default 1000).

    Returns
    -------
    sigma : np.ndarray
        Velocity dispersions (km/s).
    zl : np.ndarray
        Lens redshifts.

    Notes
    -----
    Creates a new interpolator only if 
    './interpolator_json/source_redshift/source_redshift_0.json' does not exist.
    """
     
    # Path to check for existing interpolator
    json_path = "./interpolator_json/source_redshift/source_redshift_0.json"
    create_new_interpolator = not os.path.exists(json_path)

    # Define samplers
    lens_param_samplers = dict(
        velocity_dispersion="velocity_dispersion_ewoud"
    )

    lens_param_samplers_params = {
        "velocity_dispersion": {
            "sigma_min": 60,    # km/s
            "sigma_max": 600    # km/s
        }
    }

    # Create LensGalaxyParameterDistribution
    lens = LensGalaxyParameterDistribution(
        z_min=0.0,
        z_max=3.0,
        lens_param_samplers=lens_param_samplers,
        lens_param_samplers_params=lens_param_samplers_params,
        create_new_interpolator=create_new_interpolator
    )

    # Sample the parameters
    params = lens.sample_all_routine_epl_shear_intrinsic(size=size)
    sigma = params["sigma"]
    zl = params["zl"]

    return sigma, zl

# Ellipticities (Follows Wempe+ 2024)

def sample_ellipticity_theta(sigma, size, separate_ellipticity=True):
    """
    Sample lens light and lens mass ellipticities and orientation angles.

    Parameters
    ----------
    sigma : float or ndarray of shape (size,)
        Velocity dispersion(s) [km/s].
    size : int
        Number of samples to generate.
    separate_ellipticity : bool
        Whether to allow lens mass ellipticity to differ from light. (True by default)

    Returns
    -------
    ell_light : ndarray of shape (size,)
    theta_light : ndarray of shape (size,)
    ell_mass : ndarray of shape (size,)
    theta_mass : ndarray of shape (size,)
    """
    sigma = np.asarray(sigma)
    if sigma.size == 1:
        sigma = np.full(size, sigma)
    elif sigma.size != size:
        raise ValueError("sigma must be a scalar or an array of length 'size'")

    # --- Lens light ellipticity ---
    s = 0.378 - 5.72e-4 * sigma
    u = np.random.rand(size) * scs.rayleigh(scale=s).cdf(0.8)
    ell_light = scs.rayleigh(scale=s).ppf(u)

    # --- Orientation angle for light ---
    theta_light = np.random.rand(size) * np.pi # [0, pi] is enough to cover all ellipses due to the definition

    # --- Lens mass ellipticity ---
    if not separate_ellipticity:
        # Light traces mass (or should I say mass traces light)
        ell_mass = ell_light.copy()
        theta_mass = theta_light.copy()
    else:
        # Truncated normal for mass ellipticity
        scale = 0.2
        a = (0.0 - ell_light) / scale
        b = (0.8 - ell_light) / scale
        # Add Scatter to light ellipticities
        ell_mass = scs.truncnorm.rvs(a, b, loc=ell_light, scale=scale, size=size)
        # Gaussian scatter for orientation 
        theta_mass = scs.norm.rvs(loc=theta_light, scale=34/180*np.pi, size=size)

    return ell_light, theta_light, ell_mass, theta_mass

# Slope of EPL

def sample_slope_gamma(size=1, mean=2.0, sigma=0.2):
    """
    Sample power-law slope γ_m.

    Returns
    -------
    gamma_m : ndarray
    """
    return np.random.normal(mean, sigma, size=size)   #check if scs computation is faster?

# He's actually using a truncated normal distribution i.e from -4,4 :
# if self.pemd:
#       a, b = (1.2-2)/0.2, (2.8-2)/0.2 # We truncate it at these values because there are some numerical issues in the extreme cases. It should not have any impact. 
#       res[8] = scs.truncnorm.ppf(v[8], a, b, loc=2, scale=0.2)


# External shear

def sample_shear(size=1, scale=0.05):
    """
    Sample external shear parameters γ_ext, phi_ext.

    Parameters
    ----------
    scale : float
        Rayleigh scale parameter for shear magnitude.

    Returns
    -------
    gamma_ext : ndarray 
    phi_ext : ndarray (angle in radians)
    """
    gamma_ext = np.random.rayleigh(scale, size=size)   #check if scs computation is faster?
    phi_ext = np.random.uniform(0, np.pi, size=size)  
    return gamma_ext, phi_ext

#Lens center (From Wempe+ 2024)

def sample_lens_position(size=1, lenspos_width=0.05):
    """
    Sample small lens offsets around the center.

    Parameters
    ----------
    size : int
        Number of samples to generate.
    lenspos_width : float
        Maximum offset in physical units.

    Returns
    -------
    dx, dy : ndarray
        Arrays of shape (size,) with sampled offsets.
    """
    u = np.random.rand(size, 2)  # shape (size, 2)
    dx = (2*u[:, 0] - 1) * lenspos_width
    dy = (2*u[:, 1] - 1) * lenspos_width
    return dx, dy


##Use this later for lens simulation
# --- Lens center ---
#cx0, cy0 = 0.0, 0.0

# --- Sample unit-cube random numbers for lens position ---
#u_pos = np.random.rand(2)
#dx, dy = prior_lenspos(u_pos)
#lens_center_x = cx0 + dx
#lens_center_y = cy0 + dy


## Einstein radius
def einstein_radius(sigma, z_lens, z_source, cosmology=cosmo):
    """
    Compute Einstein radius θ_E for a lens at z_lens with velocity dispersion σ,
    and a source at z_source.

    Parameters
    ----------
    sigma : float
        Velocity dispersion [km/s].
    z_lens : float
        Lens redshift.
    z_source : float
        Source redshift (must be > z_lens).
    cosmology : astropy.cosmology, optional
        Cosmology to use (default: Planck18).

    Returns
    -------
    theta_E : float
        Einstein radius [arcsec].
    """
    # Convert velocity dispersion to astropy quantity
    sigma = (sigma * u.km / u.s)

    # Angular diameter distances
    D_l = cosmology.angular_diameter_distance(z_lens)
    D_s = cosmology.angular_diameter_distance(z_source)
    D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source)

    # Einstein radius in radians
    theta_E_rad = 4 * np.pi * (sigma / c)**2 * (D_ls / D_s)

    # Convert to arcseconds
    theta_E_arcsec = theta_E_rad.to(u.arcsec)

    return theta_E_arcsec.value

