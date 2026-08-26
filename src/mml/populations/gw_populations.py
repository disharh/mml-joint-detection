"""
High-level gravitational-wave source population sampling and lensing.

This module provides:

    GWParams
        Intrinsic and extrinsic parameters of a sampled BBH.

    LensedGWParams
        Effective GW parameters for the images of a lensed BBH.

    GWPopulation
        BBH population sampler and gravitational-wave lensing utilities.
"""

from dataclasses import dataclass

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import (
    LensEquationSolver,
)
from ler.gw_source_population import CBCSourceParameterDistribution


# ---------------------------------------------------------------------------
# GW parameters
# ---------------------------------------------------------------------------

@dataclass
class GWParams:
    """Parameters describing a sampled BBH gravitational-wave source."""

    mass_1_source: float | np.ndarray
    mass_2_source: float | np.ndarray

    a_1: float | np.ndarray
    a_2: float | np.ndarray

    tilt_1: float | np.ndarray
    tilt_2: float | np.ndarray

    phi_12: float | np.ndarray
    phi_jl: float | np.ndarray

    theta_jn: float | np.ndarray

    ra: float | np.ndarray
    dec: float | np.ndarray

    psi: float | np.ndarray

    geocent_time: float | np.ndarray
    phase: float | np.ndarray

    luminosity_distance: float | np.ndarray | None = None

    def to_dict(self):
        """Convert GW parameters to a dictionary."""

        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


# ---------------------------------------------------------------------------
# Lensed GW parameters
# ---------------------------------------------------------------------------

@dataclass
class LensedGWParams:
    """Effective GW parameters for the images of a lensed BBH."""

    effective_luminosity_distance: np.ndarray
    effective_geocent_time: np.ndarray
    effective_phase: np.ndarray
    effective_ra: np.ndarray
    effective_dec: np.ndarray

    magnifications: np.ndarray
    time_delays: np.ndarray

    x_image: np.ndarray
    y_image: np.ndarray

    n_images: int


# ---------------------------------------------------------------------------
# GW population
# ---------------------------------------------------------------------------

class GWPopulation:
    """
    High-level BBH population sampler and GW lensing utility.

    Parameters
    ----------
    event_type : str
        Event type passed to LeR. Default is ``"BBH"``.

    spin_zero : bool
        Whether the LeR population assumes zero spins.

    spin_precession : bool
        Whether precessing spins are sampled.

    rng : numpy.random.Generator, optional
        Random-number generator.
    """

    SELECTED_KEYS = [
        "mass_1_source",
        "mass_2_source",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "theta_jn",
        "ra",
        "dec",
        "psi",
        "geocent_time",
        "phase",
    ]

    def __init__(
        self,
        event_type="BBH",
        spin_zero=False,
        spin_precession=True,
        rng=None,
    ):
        self.event_type = event_type
        self.spin_zero = spin_zero
        self.spin_precession = spin_precession
        self.rng = np.random.default_rng() if rng is None else rng

        self.cbc = CBCSourceParameterDistribution(
            event_type=event_type,
            spin_zero=spin_zero,
            spin_precession=spin_precession,
        )

    def sample(self, size=1):
        """
        Sample BBH GW parameters using LeR.

        Parameters
        ----------
        size : int
            Number of GW sources to sample.

        Returns
        -------
        GWParams
            Sampled GW parameters.
        """

        params = self.cbc.sample_gw_parameters(size=size)

        sampled = {
            key: params[key]
            for key in self.SELECTED_KEYS
        }

        if size == 1:
            return GWParams(
                **{
                    key: float(np.asarray(value)[0])
                    for key, value in sampled.items()
                }
            )

        return GWParams(
            **{
                key: np.asarray(value)
                for key, value in sampled.items()
            }
        )

    # ------------------------------------------------------------------
    # GW lensing
    # ------------------------------------------------------------------

    @staticmethod
    def _morse_phase(hessian):
        """Compute the Morse phase for each lensed image."""

        f_xx, f_xy, f_yx, f_yy = hessian

        A_xx = 1 - f_xx
        A_xy = -f_xy
        A_yx = -f_yx
        A_yy = 1 - f_yy

        det_A = A_xx * A_yy - A_xy * A_yx
        trace_A = A_xx + A_yy

        morse = np.zeros_like(det_A)

        morse[det_A < 0] = np.pi / 2
        morse[(det_A > 0) & (trace_A < 0)] = np.pi

        return morse

    def lens(
        self,
        gw,
        lens,
        x_gw,
        y_gw,
        z_source,
    ):
        """
        Lens a GW source using a sampled lens galaxy.

        Parameters
        ----------
        gw : GWParams
            Sampled GW parameters.

        lens : LensParams
            Sampled lens parameters.

        x_gw, y_gw : float
            GW source position in arcseconds.

        z_source : float
            Source redshift.

        Returns
        -------
        LensedGWParams
            Effective GW parameters for all lensed images.
        """

        if gw.luminosity_distance is None:
            print("Assuming z_source = z_gw to calculate GW luminosity distance..")
            gw.luminosity_distance = cosmo.luminosity_distance([z_source]).value

        kwargs_lens = lens.to_lenstronomy()

        lens_model = LensModel(
            lens_model_list=["EPL_NUMBA", "SHEAR"],
            cosmo=cosmo,
        )

        solver = LensEquationSolver(lens_model)

        x_image, y_image = solver.image_position_from_source(
            x_gw,
            y_gw,
            kwargs_lens,
            solver="analytical",
        )

        n_images = len(x_image)

        if n_images == 0:
            raise RuntimeError(
                "Lens equation solver produced no images."
            )

        if n_images > 5:
            raise RuntimeError(
                f"Lens equation solver produced {n_images} images."
            )

        magnifications = lens_model.magnification(
            x=x_image,
            y=y_image,
            kwargs=kwargs_lens,
        )

        lens_model_cosmo = LensModel(
            lens_model_list=["EPL_NUMBA", "SHEAR"],
            cosmo=cosmo,
            z_lens=lens.z_lens,
            z_source=z_source,
        )

        delays = lens_model_cosmo.arrival_time(
            x_image=x_image,
            y_image=y_image,
            kwargs_lens=kwargs_lens,
        )

        delays = (delays - delays.min()) * 86400.0

        hessian = lens_model.hessian(
            x_image,
            y_image,
            kwargs_lens,
        )

        morse_phase = self._morse_phase(hessian)

        arcsec_to_rad = 1.0 / 206265.0

        dx = (
            np.asarray(x_image) - x_gw
        ) * arcsec_to_rad

        dy = (
            np.asarray(y_image) - y_gw
        ) * arcsec_to_rad

        mu = np.abs(magnifications)

        effective_luminosity_distance = (
            gw.luminosity_distance / np.sqrt(mu)
        ).reshape(-1)

        effective_geocent_time = (
            gw.geocent_time + delays
        ).reshape(-1)

        effective_phase = (
            gw.phase - morse_phase
        ).reshape(-1)

        effective_ra = (
            gw.ra + dx / np.cos(gw.dec)
        ).reshape(-1)

        effective_dec = (
            gw.dec + dy
        ).reshape(-1)

        return LensedGWParams(
            effective_luminosity_distance=effective_luminosity_distance,
            effective_geocent_time=effective_geocent_time,
            effective_phase=effective_phase,
            effective_ra=effective_ra,
            effective_dec=effective_dec,
            magnifications=np.asarray(magnifications),
            time_delays=np.asarray(delays),
            x_image=np.asarray(x_image),
            y_image=np.asarray(y_image),
            n_images=n_images,
        )