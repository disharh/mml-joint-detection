"""LVK gravitational-wave detectability, including detector duty cycles."""

from dataclasses import dataclass
import numpy as np

from gwsnr import GWSNR

from mml.populations.gw_populations import GWParams, LensedGWParams


@dataclass
class LVKResult:
    """SNRs, detector availability, and detection information for all images."""

    # SNRs calculated assuming the full H1-L1-V1 network
    snr_net_ideal: np.ndarray

    # SNRs calculated using only detectors operating at each image's arrival time
    snr_net: np.ndarray

    # Individual detector SNRs (always retained, even when a detector is offline)
    snr_H1: np.ndarray
    snr_L1: np.ndarray
    snr_V1: np.ndarray

    # Whether each detector was operating when each image arrived
    detector_available: dict

    # Detection information
    detected_mask: np.ndarray
    detected_indices: np.ndarray
    n_detected: int
    detected: bool

    # Effective parameters for every lensed image
    image_parameters: list


class LVKDetector:
    """Calculate lensed GW image SNRs and determine detectability."""

    def __init__(
        self,
        waveform_approximant="IMRPhenomXPHM",
        snr_threshold=8.0,
        detector_snr_threshold=4.0,
        num_detected_images=2,
        min_detectors=2,
        duty_cycles=None,
        random_seed=None,
    ):
        """
        Parameters
        ----------
        waveform_approximant : str
            Waveform approximant passed to GWSNR.

        snr_threshold : float
            Minimum network SNR for an image to be detectable.

        detector_snr_threshold : float
            Minimum individual-detector SNR.

        num_detected_images : int
            Minimum number of detectable lensed images required
            for the event to be considered detected.

        min_detectors : int
            Minimum number of operating detectors whose individual
            SNR exceeds detector_snr_threshold.

        duty_cycles : dict or None
            Operating probability of each detector.
            Example: {"H1": 0.7, "L1": 0.7, "V1": 0.75}.

            If None, all 3 detectors are assumed to be operating 100% probability.

        random_seed : int or None
            Seed for reproducible duty-cycle sampling.
        """

        self.snr_threshold = snr_threshold
        self.detector_snr_threshold = detector_snr_threshold
        self.num_detected_images = num_detected_images
        self.min_detectors = min_detectors

        self.detectors = ("H1", "L1", "V1")

        if duty_cycles is None:
            duty_cycles = {ifo: 1.0 for ifo in self.detectors}

        self.duty_cycles = {}

        for ifo in self.detectors:
            probability = duty_cycles.get(ifo, 1.0)

            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Duty cycle for {ifo} must be between 0 and 1."
                )

            self.duty_cycles[ifo] = probability

        if not 1 <= min_detectors <= len(self.detectors):
            raise ValueError(
                f"min_detectors must be between 1 and {len(self.detectors)}."
            )

        if num_detected_images < 1:
            raise ValueError("num_detected_images must be at least 1.")

        self.rng = np.random.default_rng(random_seed)

        self.snr_calc = GWSNR(
            waveform_approximant=waveform_approximant
        )

    @staticmethod
    def _scalar(value):
        """Extract a scalar from a scalar or single-element array."""
        return np.asarray(value).reshape(-1)[0].item()

    def _sample_detector_availability(self, n_images):
        """
        Sample whether each detector is operating for every image.

        Each detector/image combination is sampled independently.
        """

        availability = {}

        for ifo in self.detectors:
            availability[ifo] = (
                self.rng.random(n_images) < self.duty_cycles[ifo]
            )

        return availability

    def calculate_snr(
        self,
        gw: GWParams,
        lensed: LensedGWParams,
        z_source: float,
    ) -> LVKResult:
        """
        Calculate SNRs for every lensed image and determine detectability.

        All image SNRs and effective parameters are retained, including
        those of images that do not pass the detection criteria.
        """

        # --------------------------------------------------
        # 1. Source parameters
        # --------------------------------------------------

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

        # --------------------------------------------------
        # 2. Calculate SNRs for every image
        # --------------------------------------------------

        snr_net_ideal = []
        snr_H1 = []
        snr_L1 = []
        snr_V1 = []

        image_parameters = []

        for i in range(lensed.n_images):

            # Effective parameters of this lensed image
            luminosity_distance = lensed.effective_luminosity_distance[i]
            phase = lensed.effective_phase[i]
            geocent_time = lensed.effective_geocent_time[i]
            ra = lensed.effective_ra[i]
            dec = lensed.effective_dec[i]

            # Calculate optimal SNRs using the full detector network
            result = self.snr_calc.optimal_snr(
                mass_1=mass_1,
                mass_2=mass_2,
                luminosity_distance=luminosity_distance,
                theta_jn=theta_jn,
                psi=psi,
                phase=phase,
                geocent_time=geocent_time,
                ra=ra,
                dec=dec,
                a_1=a_1,
                a_2=a_2,
                tilt_1=tilt_1,
                tilt_2=tilt_2,
                phi_12=phi_12,
                phi_jl=phi_jl,
            )

            # Store all SNRs, irrespective of detectability
            snr_net_ideal.append(result["optimal_snr_net"][0])
            snr_H1.append(result["optimal_snr_H1"][0])
            snr_L1.append(result["optimal_snr_L1"][0])
            snr_V1.append(result["optimal_snr_V1"][0])

            # Store the effective parameters of every image
            image_parameters.append({
                "image_index": i,
                "luminosity_distance": luminosity_distance,
                "phase": phase,
                "geocent_time": geocent_time,
                "ra": ra,
                "dec": dec,
            })

        snr_net_ideal = np.asarray(snr_net_ideal)
        snr_H1 = np.asarray(snr_H1)
        snr_L1 = np.asarray(snr_L1)
        snr_V1 = np.asarray(snr_V1)

        # --------------------------------------------------
        # 3. Sample detector availability
        # --------------------------------------------------

        detector_available = self._sample_detector_availability(
            lensed.n_images
        )

        # --------------------------------------------------
        # 4. Calculate network SNR using operating detectors
        # --------------------------------------------------

        # An unavailable detector does not contribute to the
        # network SNR, but its calculated individual SNR is retained.

        snr_net = np.zeros(lensed.n_images)

        for i in range(lensed.n_images):

            active_snrs = []

            if detector_available["H1"][i]:
                active_snrs.append(snr_H1[i])

            if detector_available["L1"][i]:
                active_snrs.append(snr_L1[i])

            if detector_available["V1"][i]:
                active_snrs.append(snr_V1[i])

            if active_snrs:
                snr_net[i] = np.sqrt(np.sum(np.square(active_snrs)))

        # --------------------------------------------------
        # 5. Apply detection criteria to every image
        # --------------------------------------------------

        # Count operating detectors with individual SNR above threshold.
        n_detectors_above_threshold = np.zeros(
            lensed.n_images, dtype=int
        )

        for ifo, snr in (
            ("H1", snr_H1),
            ("L1", snr_L1),
            ("V1", snr_V1),
        ):
            n_detectors_above_threshold += (
                detector_available[ifo]
                & (snr >= self.detector_snr_threshold)
            )

        # An image is detectable if:
        #   1. Network SNR exceeds the threshold.
        #   2. At least min_detectors are operating and individually
        #      exceed the detector SNR threshold.

        detected_mask = (
            (snr_net >= self.snr_threshold)
            & (n_detectors_above_threshold >= self.min_detectors)
        )

        # Keep ALL detectable images.
        detected_indices = np.where(detected_mask)[0]

        # The event is detected if at least the requested number
        # of images are detectable.
        n_detected = len(detected_indices)
        detected = n_detected >= self.num_detected_images

        # --------------------------------------------------
        # 6. Return results for every image
        # --------------------------------------------------

        return LVKResult(
            snr_net_ideal=snr_net_ideal,
            snr_net=snr_net,
            snr_H1=snr_H1,
            snr_L1=snr_L1,
            snr_V1=snr_V1,
            detector_available=detector_available,
            detected_mask=detected_mask,
            detected_indices=detected_indices,
            n_detected=n_detected,
            detected=detected,
            image_parameters=image_parameters,
        )