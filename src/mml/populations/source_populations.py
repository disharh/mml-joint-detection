"""
High-level source-galaxy population sampling using trained denmarf models.

By default, trained model files are expected in ``src/mml/data/``.
An explicit ``model_path`` can be supplied to use a model elsewhere.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from denmarf import DensityEstimate
from lenstronomy.Util import param_util


@dataclass
class SourceParams:
    """Parameters describing one or more sampled source galaxies."""

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
        """Return the parameters as a dictionary (e.g. for YAML output)."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    def to_lenstronomy(self):
        """Return the source-light parameters expected by lenstronomy."""
        return {
            "re_source": self.Re_maj_source,
            "nsersic_source": self.n_sersic_source,
            "e1_source": self.e1_source,
            "e2_source": self.e2_source,
        }


class Source:
    """
    Sample source galaxies from a trained DensityEstimate/MAF model.

    Parameters
    ----------
    model : {"mstar_weighted", "non_weighted"}
        Selects one of the default trained models.
    model_path : str or pathlib.Path, optional
        Explicit model-file path. Takes precedence over the default model
        associated with ``model``.
    rng : numpy.random.Generator, optional
        Random generator used for sampling source position angles.
    """

    DEFAULT_MODELS = {
        "mstar_weighted": "trained_de_weighted_mass_b32_h128_e1000_sizen.pkl",
        "non_weighted": "trained_de_b32_h128_e1000_bdef_wbounds.pkl",
    }

    def __init__(self, model="non_weighted", model_path=None, rng=None):
        if model not in self.DEFAULT_MODELS:
            raise ValueError(
                f"Invalid model={model!r}. "
                f"Choose from {sorted(self.DEFAULT_MODELS)}."
            )

        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng()

        if model_path is None:
            model_path = self._default_model_path(model)

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            available = sorted(
                p.name for p in self.model_path.parent.glob("*.pkl")
            ) if self.model_path.parent.is_dir() else []
            raise FileNotFoundError(
                f"Source population model not found: {self.model_path}. "
                "Place the trained model in src/mml/data/ or pass "
                "model_path explicitly."
                + (f" Available .pkl files: {available}" if available else "")
            )

        self.de = DensityEstimate.from_file(filename=str(self.model_path))

    @classmethod
    def _default_model_path(cls, model):
        """Resolve a default model relative to the installed mml package."""
        data_dir = Path(__file__).resolve().parents[1] / "data/trained_models/"
        return data_dir / cls.DEFAULT_MODELS[model]

    def sample(self, size=1):
        """
        Draw source-galaxy samples.

        Returns scalar-valued ``SourceParams`` for ``size=1`` and
        array-valued parameters otherwise.
        """
        if not isinstance(size, (int, np.integer)):
            raise TypeError("size must be an integer.")
        if size < 1:
            raise ValueError("size must be >= 1.")

        xgen = self.de.sample(size)
        log_p = self.de.score_samples(xgen)

        theta = self.rng.uniform(0.0, np.pi, size=size)
        e1, e2 = param_util.phi_q2_ellipticity(theta, xgen[:, 4])

        params = SourceParams(
            m_VIS_Euclid=xgen[:, 0],
            log10_mStar=xgen[:, 1],
            Re_maj_source=xgen[:, 2],
            z_source=xgen[:, 3],
            q_source=xgen[:, 4],
            n_sersic_source=xgen[:, 5],
            log_p_source=log_p,
            theta_light_source=theta,
            e1_source=e1,
            e2_source=e2,
        )

        if size == 1:
            return SourceParams(**{
                key: float(np.asarray(value)[0])
                for key, value in params.to_dict().items()
            })
        return params
