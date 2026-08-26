"""
High-level gravitational-wave source population sampling.

This module provides:

    GWParams
        Dataclass containing sampled GW parameters.

    GWPopulation
        High-level BBH population sampler using LeR, together with
        utilities for applying lensing effects to GW parameters.
"""

from dataclasses import dataclass
import numpy as np

from ler.gw_source_population import CBCSourceParameterDistribution


# ---------------------------------------------------------------------------
# GW parameters
# ---------------------------------------------------------------------------

@dataclass
class GWParams:
    """
    Parameters describing a sampled BBH gravitational-wave source.

    Parameters may be either scalars (for a single event) or numpy
    arrays (for multiple sampled events).
    """

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

    # These are needed when computing the effective parameters.
    # They are not sampled by sample(), but can be attached later.
    luminosity_distance: float | np.ndarray | None = None

    def to_dict(self):
        """Convert GW parameters to a dictionary."""

        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


# ---------------------------------------------------------------------------
# GW population
# ---------------------------------------------------------------------------

class GWPopulation:
    """
    High-level BBH gravitational-wave population sampler.

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

        Note
        ----
        The current LeR sampling interface used here does not expose
        a NumPy Generator argument, so this is stored for future
        reproducibility support but is not currently passed to LeR.
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

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

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

    @staticmethod
    def compute_morse_phase(hessian):
        """
        Compute the Morse phase for each lensed image.

        Parameters
        ----------
        hessian : array-like
            Hessian components ``(f_xx, f_xy, f_yx, f_yy)``.

        Returns
        -------
        ndarray
            Morse phase for each image.
        """

        f_xx, f_xy, f_yx, f_yy = hessian

        A_xx = 1 - f_xx
        A_xy = -f_xy
        A_yx = -f_yx
        A_yy = 1 - f_yy

        detA = A_xx * A_yy - A_xy * A_yx
        traceA = A_xx + A_yy

        morse = np.zeros_like(detA)

        morse[(detA < 0)] = np.pi / 2.0
        morse[(detA > 0) & (traceA < 0)] = np.pi
        morse[(detA > 0) & (traceA > 0)] = 0.0

        return morse

    @staticmethod
    def compute_effective_params(
        gw_params,
        magnifications,
        delays,
        hessian,
        x_image,
        y_image,
        x_gw,
        y_gw,
    ):
        """
        Compute effective GW parameters for each lensed image.

        Parameters
        ----------
        gw_params : GWParams or dict
            Intrinsic/extrinsic GW parameters. Must contain
            ``luminosity_distance``, ``geocent_time``, ``phase``,
            ``ra`` and ``dec``.

        magnifications : array-like
            Lensing magnifications.

        delays : array-like
            Time delays in seconds.

        hessian : array-like
            Lens-model Hessian components.

        x_image, y_image : array-like
            Image positions in arcseconds.

        x_gw, y_gw : float
            GW source position in arcseconds.

        Returns
        -------
        GWParams or dict
            GW parameters with effective parameters added.
        """

        if isinstance(gw_params, GWParams):
            params = gw_params.to_dict()
            return_dataclass = True
        else:
            params = gw_params.copy()
            return_dataclass = False

        morse_phase = GWPopulation.compute_morse_phase(hessian)

        dL = params["luminosity_distance"]
        t0 = params["geocent_time"]
        phi = params["phase"]
        ra = params["ra"]
        dec = params["dec"]

        mu = np.abs(magnifications)
        dt = delays

        arcsec_to_rad = 1.0 / 206265.0

        dx = (np.asarray(x_image) - x_gw) * arcsec_to_rad
        dy = (np.asarray(y_image) - y_gw) * arcsec_to_rad

        cosdec = np.cos(dec)

        params["effective_luminosity_distance"] = (
            dL / np.sqrt(mu)
        ).reshape(-1)

        params["effective_geocent_time"] = (
            t0 + dt
        ).reshape(-1)

        params["effective_phase"] = (
            phi - morse_phase
        ).reshape(-1)

        params["effective_ra"] = (
            ra + dx / cosdec
        ).reshape(-1)

        params["effective_dec"] = (
            dec + dy
        ).reshape(-1)

        if return_dataclass:
            return params

        return params