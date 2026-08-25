"""
High-level lens galaxy population sampling.

This module provides:

    LensParams
        Dataclass containing all sampled lens parameters.

    Lens
        High-level lens population sampler combining the individual
        sampling routines from ``lens_sampling`` and
        ``conditional_sigma_z``.

The lower-level population models live in ``lens.py`` and the
individual sampling functions live in ``lens_sampling.py``.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from lenstronomy.Util import param_util

from .lens_sampling import (
    sample_sigmaz,
    sample_sigmaz_ler,
    sample_ellipticity_theta,
    sample_slope_gamma,
    sample_shear,
    sample_lens_position,
    sample_FP,
)
from . import conditional_sigma_z


# ---------------------------------------------------------------------------
# Lens parameters
# ---------------------------------------------------------------------------

@dataclass
class LensParams:
    """
    Parameters describing a sampled lens galaxy.

    Parameters may be either scalars (for a single lens) or numpy
    arrays (for multiple sampled lenses).
    """

    sigma_lens: float | np.ndarray
    z_lens: float | np.ndarray

    q_lens: float | np.ndarray

    ell_mass_lens: float | np.ndarray
    theta_mass_lens: float | np.ndarray

    ell_light_lens: float | np.ndarray
    theta_light_lens: float | np.ndarray

    mag_lens: float | np.ndarray
    re_lens: float | np.ndarray

    x_lens: float | np.ndarray
    y_lens: float | np.ndarray

    e1_lens: float | np.ndarray
    e2_lens: float | np.ndarray

    gamma: float | np.ndarray

    gamma1: float | np.ndarray
    gamma2: float | np.ndarray

    theta_ein: Optional[float] = None

    def to_dict(self):
        """
        Convert lens parameters to a dictionary.

        Useful when writing YAML files.
        """
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    def to_lenstronomy(self):
        """
        Convert lens parameters to lenstronomy kwargs.
        """
        if self.theta_ein is None:
            raise ValueError("theta_ein must be set before converting to lenstronomy kwargs.")

        return [
            {
                "theta_E": self.theta_ein,
                "gamma": self.gamma,
                "e1": self.e1_lens,
                "e2": self.e2_lens,
                "center_x": self.x_lens,
                "center_y": self.y_lens,
            },
            {
                "gamma1": self.gamma1,
                "gamma2": self.gamma2,
            },
        ]


class Lens:
    """
    High-level sampler for lens-galaxy parameters.

    This class combines the lower-level sampling functions into a
    complete lens-galaxy population sampler.

    Parameters
    ----------
    sigmazfn : {"ewoud", "ler", "cond_on_zs"}
        Method used to sample lens velocity dispersion and redshift.

    separate_ellipticity : bool
        Whether lens mass and light ellipticities/orientations are
        sampled independently.

    lenspos_width : float
        Maximum absolute lens-centre offset.

    shear_scale : float
        Rayleigh scale parameter for external shear.

    gamma_mean : float
        Mean EPL slope.

    gamma_sigma : float
        Standard deviation of EPL slope.

    rng : numpy.random.Generator, optional
        Random-number generator.
    """

    def __init__(
        self,
        sigmazfn="cond_on_zs",
        separate_ellipticity=True,
        lenspos_width=0.05,
        shear_scale=0.05,
        gamma_mean=2.0,
        gamma_sigma=0.2,
        rng=None,
    ):

        valid_methods = {
            "ewoud",
            "ler",
            "cond_on_zs",
        }

        if sigmazfn not in valid_methods:
            raise ValueError(
                f"Invalid sigmazfn='{sigmazfn}'. "
                f"Choose from {sorted(valid_methods)}."
            )

        self.sigmazfn = sigmazfn
        self.separate_ellipticity = separate_ellipticity
        self.lenspos_width = lenspos_width
        self.shear_scale = shear_scale
        self.gamma_mean = gamma_mean
        self.gamma_sigma = gamma_sigma

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

    def _sample_sigmaz(self, size=1, zs=None):
        """Sample lens velocity dispersion and redshift."""

        if self.sigmazfn == "ewoud":

            return sample_sigmaz(
                size=size,
                rng=self.rng,
            )

        if self.sigmazfn == "ler":

            return sample_sigmaz_ler(
                size=size,
            )

        if self.sigmazfn == "cond_on_zs":

            if zs is None:
                raise ValueError(
                    "zs must be provided when sigmazfn='cond_on_zs'."
                )

            return conditional_sigma_z.sample_sigma_zl_given_zs(
                z_source=zs,
                size=size,
                rng=self.rng
            )

        raise RuntimeError(
            f"Unknown sigmazfn: {self.sigmazfn}"
        )

    def sample(self, size=1, zs=None):
        """
        Sample a complete population of lens galaxies.

        Parameters
        ----------
        size : int
            Number of lenses to sample.

        zs : float or array-like, optional
            Source redshift(s).

            Required when ``sigmazfn='cond_on_zs'``.

        Returns
        -------
        LensParams
            Sampled lens parameters.
        """

        sigma_lens, z_lens = self._sample_sigmaz(size=size,zs=zs)

        (
            ell_light_lens,
            theta_light_lens,
            ell_mass_lens,
            theta_mass_lens,
        ) = sample_ellipticity_theta(
            sigma=sigma_lens,
            size=size,
            separate_ellipticity=self.separate_ellipticity,
            rng=self.rng,
        )

        gamma = sample_slope_gamma(size=size,mean=self.gamma_mean,sigma=self.gamma_sigma,rng=self.rng)

        gamma_shear, phi_shear = sample_shear(size=size,scale=self.shear_scale,rng=self.rng)

        x_lens, y_lens = sample_lens_position(size=size,lenspos_width=self.lenspos_width,rng=self.rng)

        Mr, re_lens, k_corr = sample_FP(sigma=sigma_lens,z=z_lens,ell=ell_light_lens,rng=self.rng)

        mag_lens = (Mr + 5 * np.log10((cosmo.luminosity_distance(z_lens)/ (10 * u.pc)).to_value(1))+ k_corr)

        e1_lens, e2_lens = param_util.phi_q2_ellipticity(phi=theta_mass_lens,q=1 - ell_mass_lens)

        gamma1, gamma2 = param_util.shear_polar2cartesian(phi=phi_shear,gamma=gamma_shear)

        if size == 1:
            return LensParams(
                sigma_lens=float(sigma_lens[0]),
                z_lens=float(z_lens[0]),
                q_lens=float(1 - ell_mass_lens[0]),
                ell_mass_lens=float(ell_mass_lens[0]),
                theta_mass_lens=float(theta_mass_lens[0]),
                ell_light_lens=float(ell_light_lens[0]),
                theta_light_lens=float(theta_light_lens[0]),
                mag_lens=float(mag_lens[0]),
                re_lens=float(re_lens[0]),
                x_lens=float(x_lens[0]),
                y_lens=float(y_lens[0]),
                e1_lens=float(e1_lens[0]),
                e2_lens=float(e2_lens[0]),
                gamma=float(gamma[0]),
                gamma1=float(gamma1[0]),
                gamma2=float(gamma2[0]),
            )

        return LensParams(
            sigma_lens=sigma_lens,
            z_lens=z_lens,
            q_lens=1 - ell_mass_lens,
            ell_mass_lens=ell_mass_lens,
            theta_mass_lens=theta_mass_lens,
            ell_light_lens=ell_light_lens,
            theta_light_lens=theta_light_lens,
            mag_lens=mag_lens,
            re_lens=re_lens,
            x_lens=x_lens,
            y_lens=y_lens,
            e1_lens=e1_lens,
            e2_lens=e2_lens,
            gamma=gamma,
            gamma1=gamma1,
            gamma2=gamma2,
        )

