import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, simpson
from tqdm import tqdm
from lens_mass import pi_l

def precompute_M_z(z_min=0, z_max=3, z_n=2000,
                   sigma_min=60, sigma_max=600, sigma_n=2000,
                   cache_file='Mz_grid.npz'):

    if os.path.isfile(cache_file):
        data = np.load(cache_file)
        return data['z_grid'], data['M_grid']

    z_grid = np.linspace(z_min, z_max, z_n)
    sigma_grid = np.linspace(sigma_min, sigma_max, sigma_n)

    M_grid = np.zeros_like(z_grid)

    for i, z in enumerate(tqdm(z_grid, desc="Computing M(z)")):
        pi_vals = pi_l(sigma_grid, z)
        M_grid[i] = simpson(pi_vals, sigma_grid)

    np.savez(cache_file,
             z_grid=z_grid,
             M_grid=M_grid)

    return z_grid, M_grid



def build_cdf(x, pdf):

    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    pdf = np.maximum(pdf, 0)

    cdf = cumulative_trapezoid(pdf, x, initial=0)
    cdf = np.maximum.accumulate(cdf)

    if cdf[-1] <= 0:
        raise ValueError("PDF integrates to zero.")

    cdf /= cdf[-1]

    return cdf



# Sample z_l | z_s

def sample_z_l(z_source, z_grid, M_grid):

    z_eff = min(z_source, z_grid[-1])

    mask = z_grid < z_eff

    z_sub = z_grid[mask]
    M_sub = M_grid[mask]

    if len(z_sub) < 2:
        return z_grid[0]

    cdf = build_cdf(z_sub, M_sub)

    inv_cdf = interp1d(
        cdf,
        z_sub,
        bounds_error=False,
        fill_value=(z_sub[0], z_sub[-1])
    )

    return float(inv_cdf(np.random.rand()))

# Sample sigma | z_l

def sample_sigma_given_zl(z_l,
                          sigma_min=60,
                          sigma_max=600,
                          sigma_n=2000):

    sigma_grid = np.linspace(
        sigma_min,
        sigma_max,
        sigma_n
    )

    pdf = pi_l(sigma_grid, z_l)

    cdf = build_cdf(sigma_grid, pdf)

    inv_cdf = interp1d(
        cdf,
        sigma_grid,
        bounds_error=False,
        fill_value=(sigma_grid[0], sigma_grid[-1])
    )

    return float(inv_cdf(np.random.rand()))


def sample_sigma_zl_given_zs(z_source,
                   cache_file='Mz_grid.npz',
                   sigma_min=60,
                   sigma_max=600,
                   sigma_n=2000):
    """
    Sample lens redshift z_l and velocity dispersion sigma
    conditional on source redshift(s) z_source.

    Parameters
    ----------
    z_source : float or array_like
        Source redshift(s).

    cache_file : str
        Cached M(z) file.

    sigma_min, sigma_max : float
        Sigma grid limits.

    sigma_n : int
        Number of sigma grid points.

    Returns
    -------
        z_l, sigma : ndarray, ndarray
    """

    # Detect scalar input
    scalar_input = np.isscalar(z_source)

    z_source = np.atleast_1d(z_source).astype(float)
    n_samples = len(z_source)

    # Load/build M(z)
    z_grid, M_grid = precompute_M_z(
        cache_file=cache_file
    )

    z_l = np.empty(n_samples)
    sigma = np.empty(n_samples)

    for i, zs in enumerate(z_source):

        # Handle very small source redshift
        if zs <= z_grid[0]:
            z_l[i] = z_grid[0]
            sigma[i] = sigma_min
            continue

        # Sample lens redshift
        zl = sample_z_l(
            zs,
            z_grid,
            M_grid
        )

        # Sample velocity dispersion
        sig = sample_sigma_given_zl(
            zl,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_n=sigma_n
        )

        z_l[i] = zl
        sigma[i] = sig



    return sigma, z_l