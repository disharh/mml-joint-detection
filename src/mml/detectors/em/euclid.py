"""
Euclid EM detectability.

Calculate the Euclid S/N and observability of a lensed source galaxy.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import gammaincinv, gamma

import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Util import param_util

from mml.populations.lens_populations import LensParams
from mml.populations.source_populations import SourceParams
from mml.populations.positions import BBHPositionSample


# ---------------------------------------------------------------------------
# Euclid configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EuclidConfig:
    """Fixed configuration for Euclid imaging."""

    exp_time: float = 3 * 565
    background_rms: float = 9.494 * np.sqrt(3) / (3 * 565)

    num_pix: int = 50
    delta_pix: float = 0.1
    fwhm: float = 0.17
    zeropoint: float = 25.1209

    supersampling_factor: int = 5
    psf_truncation: int = 5

    # Observability thresholds
    theta_ein_threshold: float = 0.33
    score_threshold: float = 70
    source_size_threshold: float = 0.56
    pixel_snr_threshold: float = 1.5
    source_snr_threshold: float = 5.0

    # Sérsic profiles
    lens_nsersic: float = 4.0
    source_nsersic: float = 1.0

    # Point-source approximation
    point_source_fraction: float = 5.0


EUCLID = EuclidConfig()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class EuclidResult:
    """Euclid detectability result for one lensed system."""

    snr_lens: float
    snr_source: float
    magnification: float
    score: float

    resolved: bool
    source_size_ok: bool
    score_ok: bool
    source_snr_ok: bool
    observable: bool

    image_no_noise: np.ndarray | None = None
    image_with_noise: np.ndarray | None = None
    source_image: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Lenstronomy setup
# ---------------------------------------------------------------------------


def _data_kwargs(config):
    return sim_util.data_configure_simple(
        config.num_pix,
        config.delta_pix,
        config.exp_time,
        config.background_rms,
    )


def _psf_kwargs(config):
    return {
        "psf_type": "GAUSSIAN",
        "fwhm": config.fwhm,
        "pixel_size": config.delta_pix,
        "truncation": config.psf_truncation,
    }


@lru_cache(maxsize=None)
def _get_image_model(
    lens_model_list,
    lens_light_model_list=None,
    source_model_list=None,
    point_source_model_list=None,
    config=EUCLID,
):
    data = ImageData(**_data_kwargs(config))
    psf = PSF(**_psf_kwargs(config))

    lens_model = LensModel(list(lens_model_list))

    lens_light = (
        LightModel(list(lens_light_model_list))
        if lens_light_model_list
        else None
    )

    source_model = (
        LightModel(list(source_model_list))
        if source_model_list
        else None
    )

    point_source = (
        PointSource(
            list(point_source_model_list),
            fixed_magnification_list=[True],
            kwargs_lens_eqn_solver={"solver": "analytical"},
        )
        if point_source_model_list
        else None
    )

    return ImageModel(
        data,
        psf,
        lens_model_class=lens_model,
        lens_light_model_class=lens_light,
        source_model_class=source_model,
        point_source_class=point_source,
        kwargs_numerics={
            "supersampling_factor": config.supersampling_factor,
            "supersampling_convolution": True,
        },
    )


# ---------------------------------------------------------------------------
# Sérsic utilities
# ---------------------------------------------------------------------------


def _magnitude_to_flux(magnitude, config):
    """Convert apparent magnitude to Lenstronomy flux."""
    return 10 ** (-(magnitude - config.zeropoint) / 2.5)


def _normalise_sersic(kwargs):
    """Convert Sérsic amplitude to integrated flux."""
    kwargs = kwargs.copy()

    n = kwargs["n_sersic"]
    r = kwargs["R_sersic"]

    bn = gammaincinv(2 * n, 0.5)

    factor = (
        r**2
        * 2
        * np.pi
        * n
        * np.exp(bn)
        / bn ** (2 * n)
        * gamma(2 * n)
    )

    if "e1" in kwargs:
        _, q = param_util.ellipticity2phi_q(
            kwargs["e1"],
            kwargs["e2"],
        )
        factor /= max(q, 1 / q)

    kwargs["amp"] /= factor

    return kwargs


# ---------------------------------------------------------------------------
# Light models
# ---------------------------------------------------------------------------


def _lens_light(lens, config):
    """Build the Euclid lens-galaxy light model."""

    e1, e2 = param_util.phi_q2_ellipticity(
        lens.theta_light_lens,
        1 - lens.ell_light_lens,
    )

    kwargs = {
        "amp": _magnitude_to_flux(lens.mag_lens, config),
        "R_sersic": lens.re_lens,
        "n_sersic": config.lens_nsersic,
        "center_x": lens.x_lens,
        "center_y": lens.y_lens,
        "e1": e1,
        "e2": e2,
    }

    return _normalise_sersic(kwargs)


def _source_light(source, position, config):
    """Build the Euclid source-galaxy light model."""

    kwargs = {
        "amp": _magnitude_to_flux(
            source.m_VIS_Euclid,
            config,
        ),
        "R_sersic": source.Re_maj_source,
        "n_sersic": source.n_sersic_source,
        "center_x": position.x_source,
        "center_y": position.y_source,
        "e1": source.e1_source,
        "e2": source.e2_source,
    }

    return _normalise_sersic(kwargs)


def _point_source(source, position, config):
    """Build a point-source approximation of the galaxy."""

    return {
        "ra_source": position.x_source,
        "dec_source": position.y_source,
        "source_amp": _magnitude_to_flux(
            source.m_VIS_Euclid,
            config,
        ),
    }


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------


def _render_lens(lens, kwargs_lens, config):
    model = _get_image_model(
        ("EPL_NUMBA", "SHEAR"),
        lens_light_model_list=("SERSIC_ELLIPSE",),
        config=config,
    )

    model.reset_point_source_cache()

    return model.image(
        kwargs_lens=kwargs_lens,
        kwargs_lens_light=[_lens_light(lens, config)],
    )


def _render_source(
    source,
    position,
    kwargs_lens,
    config,
    point_source,
):
    if point_source:
        model = _get_image_model(
            ("EPL_NUMBA", "SHEAR"),
            point_source_model_list=("SOURCE_POSITION",),
            config=config,
        )

        model.reset_point_source_cache()

        return model.image(
            kwargs_lens=kwargs_lens,
            kwargs_ps=[_point_source(source, position, config)],
        )

    model = _get_image_model(
        ("EPL_NUMBA", "SHEAR"),
        source_model_list=("SERSIC_ELLIPSE",),
        config=config,
    )

    model.reset_point_source_cache()

    return model.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=[
            _source_light(source, position, config)
        ],
    )


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------


def calculate_snr(
    lens: LensParams,
    source: SourceParams,
    position: BBHPositionSample,
    *,
    config: EuclidConfig = EUCLID,
    return_images: bool = False,
    approximate_point_source: bool | None = None,
) -> EuclidResult:
    """
    Calculate Euclid detectability for a lensed source galaxy.
    """

    kwargs_lens = lens.to_lenstronomy()

    # Basic observability cuts
    resolved = lens.theta_ein > config.theta_ein_threshold

    source_size_ok = (
        source.Re_maj_source * source.q_source
        < config.source_size_threshold * lens.theta_ein
    )

    # Choose point-source approximation automatically
    if approximate_point_source is None:
        approximate_point_source = (
            source.Re_maj_source
            < config.delta_pix / config.point_source_fraction
        )

    # Render lens and source
    image_lens = _render_lens(
        lens,
        kwargs_lens,
        config,
    )

    image_source = _render_source(
        source,
        position,
        kwargs_lens,
        config,
        approximate_point_source,
    )

    image_total = image_lens + image_source

    # S/N
    denominator = (
        (config.background_rms * config.exp_time) ** 2
        + image_total * config.exp_time
    )

    snr_lens = np.sqrt(
        ((image_lens * config.exp_time) ** 2 / denominator).sum()
    )

    snr_source = np.sqrt(
        ((image_source * config.exp_time) ** 2 / denominator).sum()
    )
    source_snr_ok = (snr_source >= config.source_snr_threshold)

    # Lensing magnification
    unlensed_flux = _magnitude_to_flux(
        source.m_VIS_Euclid,
        config,
    )

    magnification = image_source.sum() / unlensed_flux

    # Pixel detection score
    noise = np.sqrt(
        config.background_rms**2
        + image_total / config.exp_time
    )

    score = np.sum(
        image_source / noise > config.pixel_snr_threshold
    )

    score_ok = score > config.score_threshold

    observable = (
        resolved
        and source_size_ok
        and score_ok
        and source_snr_ok
    )

    # Optional images
    image_no_noise = None
    image_with_noise = None

    if return_images:
        image_no_noise = image_total

        image_with_noise = (
            image_total
            + image_util.add_background(
                image_total,
                config.background_rms,
            )
            + image_util.add_poisson(
                image_total,
                exp_time=config.exp_time,
            )
        )

    return EuclidResult(
        snr_lens=snr_lens,
        snr_source=snr_source,
        magnification=magnification,
        score=score,
        resolved=resolved,
        source_size_ok=source_size_ok,
        score_ok=score_ok,
        source_snr_ok=source_snr_ok,
        observable=observable,
        image_no_noise=image_no_noise,
        image_with_noise=image_with_noise,
        source_image=image_source if return_images else None,
    )