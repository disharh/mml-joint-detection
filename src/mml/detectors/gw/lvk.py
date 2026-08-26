"""LVK gravitational-wave detectability."""

from dataclasses import dataclass

import numpy as np
from gwsnr import GWSNR

from mml.populations.gw_populations import (
    GWParams,
    LensedGWParams,
)


@dataclass
class LVKResult:
    """Result of the LVK detectability calculation."""

    snr_net: np.ndarray
    snr_H1: np.ndarray
    snr_L1: np.ndarray
    snr_V1: np.ndarray

    detected: bool
    detected_indices: np.ndarray
    n_detected: int


class LVKDetector:
    """
    LVK detector network.

    Parameters
    ----------
    waveform_approximant : str
        Waveform approximant used by GWSNR.

    snr_threshold : float
        Network SNR threshold.

    detector_snr_threshold : float
        Minimum individual-detector SNR.

    num_detected_images : int
        Number of detected lensed images required.
    """

    def __init__(
        self,
        waveform_approximant="IMRPhenomXPHM",
        snr_threshold=7.0,
        detector_snr_threshold=4.0,
        num_detected_images=2,
    ):
        self.snr_threshold = snr_threshold
        self.detector_snr_threshold = detector_snr_threshold
        self.num_detected_images = num_detected_images

        self.snr_calc = GWSNR(
            waveform_approximant=waveform_approximant
        )

    @staticmethod
    def _scalar(value):
        """Return a scalar from a scalar or single-element array."""

        return np.atleast_1d(value)[0]

    def calculate_snr(
        self,
        gw: GWParams,
        lensed: LensedGWParams,
        z_source: float,
    ) -> LVKResult:
        """
        Calculate LVK SNRs for all lensed images.

        Parameters
        ----------
        gw : GWParams
            Intrinsic GW parameters.

        lensed : LensedGWParams
            Lensing-modified GW parameters.

        z_source : float
            Source redshift used to convert source-frame masses
            to detector-frame masses.

        Returns
        -------
        LVKResult
            SNRs and detection result.
        """

        mass_1 = self._scalar(gw.mass_1_source) * (1 + z_source)
        mass_2 = self._scalar(gw.mass_2_source) * (1 + z_source)

        theta_jn = self._scalar(gw.theta_jn)
        psi = self._scalar(gw.psi)

        a_1 = self._scalar(gw.a_1)
        a_2 = self._scalar(gw.a_2)

        tilt_1 = self._scalar(gw.tilt_1)
        tilt_2 = self._scalar(gw.tilt_2)

        phi_12 = self._scalar(gw.phi_12)
        phi_jl = self._scalar(gw.phi_jl)

        snr_net = []
        snr_H1 = []
        snr_L1 = []
        snr_V1 = []

        for i in range(lensed.n_images):

            result = self.snr_calc.optimal_snr(
                mass_1=mass_1,
                mass_2=mass_2,
                luminosity_distance=(
                    lensed.effective_luminosity_distance[i]
                ),
                theta_jn=theta_jn,
                psi=psi,
                phase=lensed.effective_phase[i],
                geocent_time=lensed.effective_geocent_time[i],
                ra=lensed.effective_ra[i],
                dec=lensed.effective_dec[i],
                a_1=a_1,
                a_2=a_2,
                tilt_1=tilt_1,
                tilt_2=tilt_2,
                phi_12=phi_12,
                phi_jl=phi_jl,
            )

            snr_net.append(result["optimal_snr_net"][0])
            snr_H1.append(result["optimal_snr_H1"][0])
            snr_L1.append(result["optimal_snr_L1"][0])
            snr_V1.append(result["optimal_snr_V1"][0])

        snr_net = np.asarray(snr_net)
        snr_H1 = np.asarray(snr_H1)
        snr_L1 = np.asarray(snr_L1)
        snr_V1 = np.asarray(snr_V1)

        detected_mask = (
            (snr_net >= self.snr_threshold)
            & (snr_H1 >= self.detector_snr_threshold)
            & (snr_L1 >= self.detector_snr_threshold)
            & (snr_V1 >= self.detector_snr_threshold)
        )

        detected_indices = np.where(detected_mask)[0]

        if len(detected_indices) > self.num_detected_images:
            order = np.argsort(
                snr_net[detected_indices]
            )[::-1]

            detected_indices = detected_indices[
                order[: self.num_detected_images]
            ]

        detected = (
            len(detected_indices)
            >= self.num_detected_images
        )

        return LVKResult(
            snr_net=snr_net,
            snr_H1=snr_H1,
            snr_L1=snr_L1,
            snr_V1=snr_V1,
            detected=detected,
            detected_indices=detected_indices,
            n_detected=len(detected_indices),
        )