from dataclasses import dataclass

import numpy as np

from mml.lensing.caustics import get_caustics, region_min_images
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
    # Minimum number of lensed images guaranteed by the region the BBH was
    # drawn from (2 = double, 3 = triple, 4 = quad). ``caustic_area`` is the
    # area of *that* region. Not part of ``to_dict`` (kept for backwards
    # compatibility); the simulation driver stores it explicitly.
    min_lensed_images: int | None = None

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
        min_lensed_images=None,
        *,
        num_detected_gws=None,
    ):
        """
        Sample one BBH and source-galaxy position.

        Parameters
        ----------
        min_lensed_images : int or str
            Source-plane region to draw the BBH from: 2/"double", 3/"triple"
            or 4/"quad". The BBH then produces *at least* this many lensed
            images. This is a lensing-geometry choice and says nothing about
            how many of those images are detectable.
        num_detected_gws : int, optional
            Deprecated alias of ``min_lensed_images``.
        """
        if min_lensed_images is None:
            min_lensed_images = num_detected_gws
        if min_lensed_images is None:
            raise TypeError("min_lensed_images must be given.")
        min_lensed_images = region_min_images(min_lensed_images)

        poly_to_sample = get_caustics(
            kwargs_lens,
            min_lensed_images,
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
            min_lensed_images=min_lensed_images,
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