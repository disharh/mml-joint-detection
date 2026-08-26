import numpy as np

from mml.detectors.gw.lvk import LVKDetector, LVKResult
from mml.populations.gw_populations import (
    GWParams,
    LensedGWParams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gw():
    return GWParams(
        mass_1_source=30.0,
        mass_2_source=20.0,
        a_1=0.5,
        a_2=0.3,
        tilt_1=0.4,
        tilt_2=0.6,
        phi_12=1.0,
        phi_jl=2.0,
        theta_jn=1.2,
        ra=1.0,
        dec=0.5,
        psi=0.7,
        geocent_time=100.0,
        phase=0.2,
        luminosity_distance=1000.0,
    )


def make_lensed(n_images=3):
    return LensedGWParams(
        effective_luminosity_distance=np.full(
            n_images,
            1000.0,
        ),
        effective_geocent_time=np.arange(
            n_images,
            dtype=float,
        ),
        effective_phase=np.zeros(n_images),
        effective_ra=np.ones(n_images),
        effective_dec=np.ones(n_images),

        magnifications=np.ones(n_images),
        time_delays=np.arange(
            n_images,
            dtype=float,
        ),

        x_image=np.arange(
            n_images,
            dtype=float,
        ),
        y_image=np.arange(
            n_images,
            dtype=float,
        ),

        n_images=n_images,
    )


class FakeGWSNR:
    """Return predefined detector SNRs."""

    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def optimal_snr(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.results)


def snr_result(net, h1, l1, v1):
    return {
        "optimal_snr_net": np.array([net]),
        "optimal_snr_H1": np.array([h1]),
        "optimal_snr_L1": np.array([l1]),
        "optimal_snr_V1": np.array([v1]),
    }


# ---------------------------------------------------------------------------
# Basic output
# ---------------------------------------------------------------------------

def test_calculate_snr_returns_result(monkeypatch):
    fake = FakeGWSNR([
        snr_result(10, 5, 5, 5),
        snr_result(12, 6, 6, 6),
    ])

    monkeypatch.setattr(
        "mml.detectors.gw.lvk.GWSNR",
        lambda **kwargs: fake,
    )

    detector = LVKDetector(
        snr_threshold=7,
        detector_snr_threshold=4,
        num_detected_images=2,
    )

    result = detector.calculate_snr(
        gw=make_gw(),
        lensed=make_lensed(2),
        z_source=1.0,
    )

    assert isinstance(result, LVKResult)

    assert result.snr_net.shape == (2,)
    assert result.snr_H1.shape == (2,)
    assert result.snr_L1.shape == (2,)
    assert result.snr_V1.shape == (2,)


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

def test_two_detected_images_means_detection(monkeypatch):
    fake = FakeGWSNR([
        snr_result(10, 5, 5, 5),
        snr_result(12, 6, 6, 6),
    ])

    monkeypatch.setattr(
        "mml.detectors.gw.lvk.GWSNR",
        lambda **kwargs: fake,
    )

    detector = LVKDetector(
        snr_threshold=7,
        detector_snr_threshold=4,
        num_detected_images=2,
    )

    result = detector.calculate_snr(
        gw=make_gw(),
        lensed=make_lensed(2),
        z_source=1.0,
    )

    assert result.detected
    assert result.n_detected == 2
    assert np.array_equal(
        result.detected_indices,
        np.array([0, 1]),
    )


def test_insufficient_detected_images(monkeypatch):
    fake = FakeGWSNR([
        snr_result(10, 5, 5, 5),
        snr_result(5, 5, 5, 5),
    ])

    monkeypatch.setattr(
        "mml.detectors.gw.lvk.GWSNR",
        lambda **kwargs: fake,
    )

    detector = LVKDetector(
        snr_threshold=7,
        detector_snr_threshold=4,
        num_detected_images=2,
    )

    result = detector.calculate_snr(
        gw=make_gw(),
        lensed=make_lensed(2),
        z_source=1.0,
    )

    assert not result.detected
    assert result.n_detected == 1
    assert np.array_equal(
        result.detected_indices,
        np.array([0]),
    )


def test_individual_detector_threshold_is_applied(monkeypatch):
    fake = FakeGWSNR([
        # Network SNR passes, but H1 fails.
        snr_result(10, 3, 5, 5),
        snr_result(12, 6, 6, 6),
    ])

    monkeypatch.setattr(
        "mml.detectors.gw.lvk.GWSNR",
        lambda **kwargs: fake,
    )

    detector = LVKDetector(
        snr_threshold=7,
        detector_snr_threshold=4,
        num_detected_images=2,
    )

    result = detector.calculate_snr(
        gw=make_gw(),
        lensed=make_lensed(2),
        z_source=1.0,
    )

    assert not result.detected
    assert result.n_detected == 1
    assert np.array_equal(
        result.detected_indices,
        np.array([1]),
    )


# ---------------------------------------------------------------------------
# Selection of loudest images
# ---------------------------------------------------------------------------

def test_loudest_detected_images_are_selected(monkeypatch):
    fake = FakeGWSNR([
        snr_result(15, 5, 5, 5),
        snr_result(25, 6, 6, 6),
        snr_result(10, 5, 5, 5),
    ])

    monkeypatch.setattr(
        "mml.detectors.gw.lvk.GWSNR",
        lambda **kwargs: fake,
    )

    detector = LVKDetector(
        snr_threshold=7,
        detector_snr_threshold=4,
        num_detected_images=2,
    )

    result = detector.calculate_snr(
        gw=make_gw(),
        lensed=make_lensed(3),
        z_source=1.0,
    )

    assert result.detected
    assert result.n_detected == 2

    # Images 1 and 0 have the largest network SNRs.
    assert np.array_equal(
        result.detected_indices,
        np.array([1, 0]),
    )


# ---------------------------------------------------------------------------
# Source-frame → detector-frame masses
# ---------------------------------------------------------------------------

def test_source_masses_are_redshifted(monkeypatch):
    fake = FakeGWSNR([
        snr_result(10, 5, 5, 5),
    ])

    monkeypatch.setattr(
        "mml.detectors.gw.lvk.GWSNR",
        lambda **kwargs: fake,
    )

    detector = LVKDetector()

    detector.calculate_snr(
        gw=make_gw(),
        lensed=make_lensed(1),
        z_source=2.0,
    )

    call = fake.calls[0]

    assert call["mass_1"] == 90.0
    assert call["mass_2"] == 60.0