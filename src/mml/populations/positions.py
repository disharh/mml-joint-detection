from dataclasses import dataclass

import numpy as np

from mml.lensing.caustics import get_caustics
from mml.lensing.polygon import sample_polygon_single
from mml.lensing.sersic import sample_sersic_cart


@dataclass
class BBHPositionSample:
    """Sampled source-plane positions of a BBH and its host galaxy."""

    x_gw: float
    y_gw: float
    caustic_area: float
    x_source: float
    y_source: float

    def to_dict(self):
        return {
            "x_gw": self.x_gw,
            "y_gw": self.y_gw,
            "caustic_area": self.caustic_area,
            "x_source": self.x_source,
            "y_source": self.y_source,
        }


class BBHPositionSampler:
    """
    Sample BBH and source-galaxy positions in the source plane.

    The BBH is sampled uniformly within the appropriate caustic
    region for a specified image multiplicity. The host galaxy is
    then sampled from a Sersic profile centred on the BBH position.
    """

    def __init__(self, rng=None):
        """
        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator.
        """
        self.rng = (
            np.random.default_rng()
            if rng is None
            else rng
        )

    def sample(
        self,
        kwargs_lens,
        kwargs_source,
        num_detected_gws,
    ):
        """
        Sample one BBH and source-galaxy position.
        """
        poly_to_sample = get_caustics(
            kwargs_lens,
            num_detected_gws,
        )

        u_gw = self.rng.uniform(0.0, 1.0, size=2)
        u_gal = self.rng.uniform(0.0, 1.0, size=2)

        (x_gw, y_gw), area = sample_polygon_single(
            poly_to_sample,
            u_gw,
        )

        x_source, y_source = sample_sersic_cart(
            u_gal,
            kwargs_source["re_source"],
            kwargs_source["nsersic_source"],
            kwargs_source["e1_source"],
            kwargs_source["e2_source"],
            x_gw,
            y_gw,
        )

        return BBHPositionSample(
            x_gw=float(x_gw),
            y_gw=float(y_gw),
            caustic_area=float(area),
            x_source=float(x_source),
            y_source=float(y_source),
        )

    def sample_population(
        self,
        kwargs_lens,
        kwargs_source,
        num_detected_gws,
        size,
    ):
        """
        Sample multiple BBH/source-galaxy positions.
        """
        samples = [
            self.sample(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                num_detected_gws=num_detected_gws,
            )
            for _ in range(size)
        ]

        return {
            key: np.array(
                [sample.to_dict()[key] for sample in samples]
            )
            for key in samples[0].to_dict()
        }