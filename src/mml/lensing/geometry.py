"""
Basic gravitational-lensing geometry.

This module contains calculations related to lensing geometry,
including the Einstein radius.

The default cosmology is Planck18.
"""

import numpy as np

from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy import constants as c


def einstein_radius(
    sigma,
    z_lens,
    z_source,
    cosmology=cosmo,
):
    """
    Compute the Einstein radius for a SIS lens.

    Parameters
    ----------
    sigma : float
        Lens velocity dispersion [km/s].

    z_lens : float
        Lens redshift.

    z_source : float
        Source redshift. Must be greater than z_lens.

    cosmology : astropy.cosmology.Cosmology, optional
        Cosmology used to calculate angular-diameter distances.
        Defaults to Planck18.

    Returns
    -------
    float
        Einstein radius [arcsec].

    Raises
    ------
    ValueError
        If z_source <= z_lens.
    """

    if z_source <= z_lens:
        raise ValueError(
            "z_source must be greater than z_lens for lensing."
        )

    # Angular-diameter distances.
    D_l = cosmology.angular_diameter_distance(z_lens)
    D_s = cosmology.angular_diameter_distance(z_source)
    D_ls = cosmology.angular_diameter_distance_z1z2(
        z_lens,
        z_source,
    )

    # Einstein radius in radians.
    theta_E_rad = (
        4
        * np.pi
        * (
            sigma
            / c.c.to_value(u.km / u.s)
        ) ** 2
        * (D_ls / D_s)
        * (1 * u.radian)
    )

    # Convert to arcseconds.
    theta_E_arcsec = theta_E_rad.to_value(u.arcsec)

    return float(theta_E_arcsec)


def einstein_radius_vec(
    sigma,
    z_lens,
    z_source,
    cosmology=cosmo,
):
    """
    Compute Einstein radii for arrays of lenses and sources.

    Parameters
    ----------
    sigma : float or array-like
        Lens velocity dispersion(s) [km/s].

    z_lens : float or array-like
        Lens redshift(s).

    z_source : float or array-like
        Source redshift(s). Every source redshift must be
        greater than the corresponding lens redshift.

    cosmology : astropy.cosmology.Cosmology, optional
        Cosmology used to calculate angular-diameter distances.
        Defaults to Planck18.

    Returns
    -------
    float or ndarray
        Einstein radius/radii [arcsec].

    Raises
    ------
    ValueError
        If any element satisfies z_source <= z_lens.
    """

    sigma = np.asarray(sigma)
    z_lens = np.asarray(z_lens)
    z_source = np.asarray(z_source)

    # Validate the lensing configuration.
    if np.any(z_source <= z_lens):
        raise ValueError(
            "All elements must satisfy z_source > z_lens."
        )

    # Angular-diameter distances.
    D_l = cosmology.angular_diameter_distance(z_lens)
    D_s = cosmology.angular_diameter_distance(z_source)
    D_ls = cosmology.angular_diameter_distance_z1z2(
        z_lens,
        z_source,
    )

    # Einstein radius in radians.
    theta_E_rad = (
        4
        * np.pi
        * (
            sigma
            / c.c.to_value(u.km / u.s)
        ) ** 2
        * (D_ls / D_s)
        * (1 * u.radian)
    )

    # Convert to arcseconds.
    theta_E_arcsec = theta_E_rad.to_value(u.arcsec)

    return theta_E_arcsec