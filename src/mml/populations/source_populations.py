"""
High-level source galaxy population sampling.

This module provides:

    SourceParams
        Dataclass containing sampled source-galaxy parameters.

    Source
        High-level source-galaxy population sampler based on a trained
        DensityEstimate / MAF model.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from denmarf import DensityEstimate
from lenstronomy.Util import param_util


# ---------------------------------------------------------------------------
# Source parameters
# ---------------------------------------------------------------------------

@dataclass
class SourceParams:
    """
    Parameters describing a sampled source galaxy.

    Parameters may be either scalars (for a single source) or numpy
    arrays (for multiple sampled sources).
    """

    m_VIS_Euclid: float | np.ndarray
    log10_mStar: float | np.ndarray
    Re_maj_source: float | np.ndarray
    z_source: float | np.ndarray
    q_source: float | np.ndarray
    n_sersic_source: float | np.ndarray
    log_p_source: float | np.ndarray

    theta_light_source: float | np.ndarray
    e1_source: float | np.ndarray
    e2_source: float | np.ndarray

    def to_dict(self):
        """
        Convert source parameters to a dictionary.

        Useful when writing YAML files.
        """

        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    def to_lenstronomy(self):
        """
        Convert source parameters to lenstronomy source kwargs.
        """

        return {
            "re_source": self.Re_maj_source,
            "nsersic_source": self.n_sersic_source,
            "e1_source": self.e1_source,
            "e2_source": self.e2_source,
        }


# ---------------------------------------------------------------------------
# Source population sampler
# ---------------------------------------------------------------------------

class Source:
    """
    High-level source-galaxy population sampler.

    Parameters
    ----------
    model : {"mstar_weighted", "non_weighted"}
        Trained DensityEstimate model to use.

    model_path : str or Path, optional
        Explicit path to the trained DensityEstimate model.

        If supplied, this takes precedence over ``model``.

    rng : numpy.random.Generator, optional
        Random-number generator used for source-galaxy angle sampling.
    """

    DEFAULT_MODELS = {
        "mstar_weighted":
            "trained_de_weighted_mass_b32_h128_e1000_sizen.pkl",

        "non_weighted":
            "trained_de_b32_h128_e1000_bdef_wbounds.pkl",
    }

    def __init__(
        self,
        model="non_weighted",
        model_path=None,
        rng=None,
    ):

        if model not in self.DEFAULT_MODELS:
            raise ValueError(
                f"Invalid model='{model}'. "
                f"Choose from {sorted(self.DEFAULT_MODELS)}."
            )

        self.model = model

        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng

        if model_path is None:
            model_path = self._default_model_path(model)

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Source population model not found: "
                f"{self.model_path}"
            )

        self.de = DensityEstimate.from_file(
            filename=self.model_path
        )

    @staticmethod
    def _default_model_path(model):
        """
        Return the default trained-model path.

        Adjust the project-root calculation here if your actual
        repository layout differs.
        """

        package_root = Path(__file__).resolve().parents[2]

        return (
            package_root
            / "trained_models"
            / Source.DEFAULT_MODELS[model]
        )

    def sample(self, size=1):
        """
        Sample source-galaxy parameters.

        Parameters
        ----------
        size : int
            Number of source galaxies to sample.

        Returns
        -------
        SourceParams
            Sampled source-galaxy parameters.
        """

        if not isinstance(size, (int, np.integer)):
            raise TypeError(
                "size must be an integer."
            )

        if size < 1:
            raise ValueError(
                "size must be >= 1."
            )
        xgen = self.de.sample(size)

        log_p = self.de.score_samples(xgen)

        theta_light_source = self.rng.uniform(
            0.0,
            np.pi,
            size=size,
        )

        e1_source, e2_source = param_util.phi_q2_ellipticity(
            theta_light_source,
            xgen[:, 4],
        )

        params = SourceParams(
            m_VIS_Euclid=xgen[:, 0],
            log10_mStar=xgen[:, 1],
            Re_maj_source=xgen[:, 2],
            z_source=xgen[:, 3],
            q_source=xgen[:, 4],
            n_sersic_source=xgen[:, 5],
            log_p_source=log_p,
            theta_light_source=theta_light_source,
            e1_source=e1_source,
            e2_source=e2_source,
        )

        if size == 1:
            return SourceParams(
                m_VIS_Euclid=float(params.m_VIS_Euclid[0]),
                log10_mStar=float(params.log10_mStar[0]),
                Re_maj_source=float(params.Re_maj_source[0]),
                z_source=float(params.z_source[0]),
                q_source=float(params.q_source[0]),
                n_sersic_source=float(params.n_sersic_source[0]),
                log_p_source=float(params.log_p_source[0]),
                theta_light_source=float(
                    params.theta_light_source[0]
                ),
                e1_source=float(params.e1_source[0]),
                e2_source=float(params.e2_source[0]),
            )

        return params