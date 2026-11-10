import numpy as np
import pytest

from mml.detectors.gw.lvk import LVKDetector, LVKResult
from mml.populations.gw_populations import GWParams, LensedGWParams


# ============================================================
# Test data
# ============================================================

def make_gw():
    """Create a single representative BBH source."""

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
    """Create effective parameters for multiple lensed images."""

    return LensedGWParams(
        effective_luminosity_distance=np.full(n_images, 1000.0),
        effective_geocent_time=100.0 + np.arange(n_images) * 100.0,
        effective_phase=np.zeros(n_images),
        effective_ra=np.full(n_images, 1.0),
        effective_dec=np.full(n_images, 0.5),
        magnifications=np.ones(n_images),
        time_delays=np.arange(n_images) * 100.0,
        x_image=np.arange(n_images, dtype=float),
        y_image=np.arange(n_images, dtype=float),
        n_images=n_images,
    )


def make_snr_result(net=15.0, h1=10.0, l1=8.0, v1=5.0):
    """
    Mimic the dictionary returned by GWSNR.optimal_snr().

    The real implementation accesses the first element of each
    returned array, so the mock returns single-element arrays.
    """

    return {
        "optimal_snr_net": np.array([net]),
        "optimal_snr_H1": np.array([h1]),
        "optimal_snr_L1": np.array([l1]),
        "optimal_snr_V1": np.array([v1]),
    }


def make_snr_results(n_images, net=15.0, h1=10.0, l1=8.0, v1=5.0):
    """Create one mocked GWSNR result per image."""

    return [
        make_snr_result(net=net, h1=h1, l1=l1, v1=v1)
        for _ in range(n_images)
    ]


# ============================================================
# Fake GWSNR
# ============================================================

class FakeGWSNR:
    """
    Lightweight replacement for GWSNR.

    One FakeGWSNR instance is reused for all images, just as
    one real GWSNR instance is reused by LVKDetector.
    """

    def __init__(self, results):
        self.results = results
        self.calls = []

    def optimal_snr(self, **kwargs):
        self.calls.append(kwargs)

        index = len(self.calls) - 1

        if index >= len(self.results):
            raise AssertionError(
                "optimal_snr() was called more times than expected."
            )

        return self.results[index]


def run_detector(
    monkeypatch,
    n_images=3,
    detector_kwargs=None,
    results=None,
    z_source=0.0,
):
    """
    Run LVKDetector.calculate_snr() with GWSNR mocked.

    Crucially, GWSNR is patched before LVKDetector is instantiated.
    The factory is expected to be called exactly once.
    """

    if detector_kwargs is None:
        detector_kwargs = {}

    if results is None:
        results = make_snr_results(n_images)

    fake_gwsnr = FakeGWSNR(results)
    initialization_calls = []

    def fake_gwsnr_factory(**kwargs):
        initialization_calls.append(kwargs)
        return fake_gwsnr

    # Patch before constructing LVKDetector.
    monkeypatch.setattr(
        "mml.detectors.gw.lvk.GWSNR",
        fake_gwsnr_factory,
    )

    detector = LVKDetector(**detector_kwargs)

    # Check that the detector uses the mocked instance.
    assert detector.snr_calc is fake_gwsnr

    result = detector.calculate_snr(
        gw=make_gw(),
        lensed=make_lensed(n_images),
        z_source=z_source,
    )

    return result, fake_gwsnr, initialization_calls


# ============================================================
# GWSNR initialisation and reuse
# ============================================================

def test_gwsnr_is_initialised_only_once(monkeypatch):
    """GWSNR should be initialised once, not once per image."""

    n_images = 5

    result, fake, initialization_calls = run_detector(
        monkeypatch,
        n_images=n_images,
    )

    assert len(initialization_calls) == 1
    assert len(fake.calls) == n_images

    assert result.n_detected == n_images


def test_gwsnr_receives_waveform_approximant(monkeypatch):
    _, _, initialization_calls = run_detector(
        monkeypatch,
        detector_kwargs={
            "waveform_approximant": "IMRPhenomXPHM",
        },
        n_images=1,
    )

    assert len(initialization_calls) == 1
    assert initialization_calls[0]["waveform_approximant"] == (
        "IMRPhenomXPHM"
    )


def test_gwsnr_called_once_per_image(monkeypatch):
    n_images = 4

    _, fake, initialization_calls = run_detector(
        monkeypatch,
        n_images=n_images,
    )

    assert len(initialization_calls) == 1
    assert len(fake.calls) == n_images


# ============================================================
# Basic output tests
# ============================================================

def test_calculate_snr_returns_lvk_result(monkeypatch):
    n_images = 3

    result, _, _ = run_detector(
        monkeypatch,
        n_images=n_images,
    )

    assert isinstance(result, LVKResult)

    assert result.snr_net_ideal.shape == (n_images,)
    assert result.snr_net.shape == (n_images,)
    assert result.snr_H1.shape == (n_images,)
    assert result.snr_L1.shape == (n_images,)
    assert result.snr_V1.shape == (n_images,)

    assert result.detected_mask.shape == (n_images,)
    assert result.detected_indices.ndim == 1


def test_all_image_snrs_are_stored(monkeypatch):
    results = [
        make_snr_result(net=15, h1=10, l1=8, v1=5),
        make_snr_result(net=9, h1=6, l1=5, v1=3),
        make_snr_result(net=3, h1=2, l1=1, v1=1),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=3,
        results=results,
    )

    np.testing.assert_allclose(
        result.snr_net_ideal,
        [15, 9, 3],
    )
    np.testing.assert_allclose(
        result.snr_H1,
        [10, 6, 2],
    )
    np.testing.assert_allclose(
        result.snr_L1,
        [8, 5, 1],
    )
    np.testing.assert_allclose(
        result.snr_V1,
        [5, 3, 1],
    )


def test_image_parameters_are_stored_for_every_image(monkeypatch):
    n_images = 3

    result, _, _ = run_detector(
        monkeypatch,
        n_images=n_images,
    )

    assert len(result.image_parameters) == n_images

    for i, params in enumerate(result.image_parameters):
        assert params["image_index"] == i
        assert params["luminosity_distance"] == 1000.0
        assert params["phase"] == 0.0
        assert params["geocent_time"] == 100.0 + 100.0 * i
        assert params["ra"] == 1.0
        assert params["dec"] == 0.5


# ============================================================
# Source parameter handling
# ============================================================

def test_source_masses_are_redshifted(monkeypatch):
    """Source-frame masses should be converted to detector-frame masses."""

    _, fake, _ = run_detector(
        monkeypatch,
        n_images=1,
        z_source=2.0,
    )

    assert len(fake.calls) == 1

    call = fake.calls[0]

    assert call["mass_1"] == pytest.approx(90.0)
    assert call["mass_2"] == pytest.approx(60.0)


def test_source_parameters_are_passed_to_gwsnr(monkeypatch):
    gw = make_gw()

    _, fake, _ = run_detector(
        monkeypatch,
        n_images=1,
        z_source=0.0,
    )

    call = fake.calls[0]

    assert call["a_1"] == gw.a_1
    assert call["a_2"] == gw.a_2
    assert call["tilt_1"] == gw.tilt_1
    assert call["tilt_2"] == gw.tilt_2
    assert call["phi_12"] == gw.phi_12
    assert call["phi_jl"] == gw.phi_jl
    assert call["theta_jn"] == gw.theta_jn
    assert call["psi"] == gw.psi


def test_effective_image_parameters_are_passed_to_gwsnr(monkeypatch):
    n_images = 3

    _, fake, _ = run_detector(
        monkeypatch,
        n_images=n_images,
    )

    assert len(fake.calls) == n_images

    for i, call in enumerate(fake.calls):
        assert call["luminosity_distance"] == 1000.0
        assert call["phase"] == 0.0
        assert call["geocent_time"] == 100.0 + 100.0 * i
        assert call["ra"] == 1.0
        assert call["dec"] == 0.5


# ============================================================
# Network SNR calculation
# ============================================================

def test_network_snr_uses_all_detectors_when_online(monkeypatch):
    results = [
        make_snr_result(net=15, h1=10, l1=8, v1=5),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=1,
        results=results,
    )

    expected = np.sqrt(10.0**2 + 8.0**2 + 5.0**2)

    np.testing.assert_allclose(
        result.snr_net,
        [expected],
    )

    np.testing.assert_allclose(
        result.snr_net_ideal,
        [15.0],
    )


def test_network_snr_uses_only_operating_detectors(monkeypatch):
    results = [
        make_snr_result(net=15, h1=10, l1=8, v1=5),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=1,
        detector_kwargs={
            "duty_cycles": {
                "H1": 1.0,
                "L1": 1.0,
                "V1": 0.0,
            },
        },
        results=results,
    )

    expected = np.sqrt(10.0**2 + 8.0**2)

    np.testing.assert_allclose(
        result.snr_net,
        [expected],
    )

    assert result.detector_available["H1"][0]
    assert result.detector_available["L1"][0]
    assert not result.detector_available["V1"][0]


def test_offline_detector_snr_is_still_stored(monkeypatch):
    results = [
        make_snr_result(net=15, h1=10, l1=8, v1=5),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=1,
        detector_kwargs={
            "duty_cycles": {
                "H1": 1.0,
                "L1": 0.0,
                "V1": 0.0,
            },
        },
        results=results,
    )

    assert result.snr_L1[0] == 8.0
    assert result.snr_V1[0] == 5.0

    assert not result.detector_available["L1"][0]
    assert not result.detector_available["V1"][0]

    np.testing.assert_allclose(
        result.snr_net,
        [10.0],
    )


# ============================================================
# Image detection criteria
# ============================================================

def test_image_detected_when_network_and_individual_thresholds_pass(
    monkeypatch,
):
    results = [
        make_snr_result(net=15, h1=6, l1=6, v1=2),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=1,
    )

    # The default fake values pass both the network and
    # individual detector requirements.
    assert result.detected_mask[0]


def test_image_not_detected_when_network_snr_is_too_low(monkeypatch):
    results = [
        make_snr_result(net=5, h1=3, l1=3, v1=1),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=1,
        results=results,
    )

    assert not result.detected_mask[0]


def test_image_not_detected_when_too_few_detectors_pass(
    monkeypatch,
):
    results = [
        make_snr_result(net=15, h1=3, l1=8, v1=3),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=1,
        results=results,
    )

    # Only L1 passes the individual threshold of 4.
    assert not result.detected_mask[0]


def test_minimum_number_of_detectors_can_be_three(monkeypatch):
    results = [
        make_snr_result(net=15, h1=10, l1=8, v1=5),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=1,
        detector_kwargs={
            "min_detectors": 3,
        },
        results=results,
    )

    assert result.detected_mask[0]


def test_three_detectors_required_but_only_two_pass(monkeypatch):
    results = [
        make_snr_result(net=15, h1=10, l1=8, v1=2),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=1,
        detector_kwargs={
            "min_detectors": 3,
        },
        results=results,
    )

    assert not result.detected_mask[0]


# ============================================================
# Number of detectable images
# ============================================================

def test_two_detectable_images_means_event_detected(monkeypatch):
    results = [
        make_snr_result(net=15, h1=10, l1=8, v1=5),
        make_snr_result(net=12, h1=8, l1=7, v1=5),
        make_snr_result(net=3, h1=2, l1=1, v1=1),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=3,
        detector_kwargs={
            "num_detected_images": 2,
        },
        results=results,
    )

    assert result.n_detected == 2
    assert result.detected is True

    np.testing.assert_array_equal(
        result.detected_indices,
        [0, 1],
    )


def test_event_not_detected_when_too_few_images_pass(monkeypatch):
    results = [
        make_snr_result(net=15, h1=10, l1=8, v1=5),
        make_snr_result(net=3, h1=2, l1=1, v1=1),
        make_snr_result(net=2, h1=1, l1=1, v1=1),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=3,
        detector_kwargs={
            "num_detected_images": 2,
        },
        results=results,
    )

    assert result.n_detected == 1
    assert result.detected is False

    np.testing.assert_array_equal(
        result.detected_indices,
        [0],
    )


def test_all_detectable_images_are_retained(monkeypatch):
    """Do not truncate the detected images to num_detected_images."""

    results = [
        make_snr_result(net=15, h1=10, l1=8, v1=5),
        make_snr_result(net=12, h1=9, l1=7, v1=5),
        make_snr_result(net=10, h1=8, l1=6, v1=4),
    ]

    result, _, _ = run_detector(
        monkeypatch,
        n_images=3,
        detector_kwargs={
            "num_detected_images": 2,
        },
        results=results,
    )

    assert result.n_detected == 3
    assert result.detected is True

    np.testing.assert_array_equal(
        result.detected_indices,
        [0, 1, 2],
    )


def test_three_detectable_images_can_be_required(monkeypatch):
    result, _, _ = run_detector(
        monkeypatch,
        n_images=3,
        detector_kwargs={
            "num_detected_images": 3,
        },
    )

    assert result.n_detected == 3
    assert result.detected is True


# ============================================================
# Duty-cycle tests
# ============================================================

def test_duty_cycle_zero_always_offline(monkeypatch):
    n_images = 3

    result, fake, initialization_calls = run_detector(
        monkeypatch,
        n_images=n_images,
        detector_kwargs={
            "duty_cycles": {
                "H1": 0.0,
                "L1": 0.0,
                "V1": 0.0,
            },
        },
    )

    # One GWSNR initialisation, reused for all images.
    assert len(initialization_calls) == 1
    assert len(fake.calls) == n_images

    for ifo in ("H1", "L1", "V1"):
        np.testing.assert_array_equal(
            result.detector_available[ifo],
            np.zeros(n_images, dtype=bool),
        )

    # Ideal and individual SNRs are still retained.
    np.testing.assert_allclose(result.snr_net_ideal, 15.0)
    np.testing.assert_allclose(result.snr_H1, 10.0)
    np.testing.assert_allclose(result.snr_L1, 8.0)
    np.testing.assert_allclose(result.snr_V1, 5.0)

    # No operating detectors means zero effective network SNR.
    np.testing.assert_array_equal(
        result.snr_net,
        np.zeros(n_images),
    )

    assert result.n_detected == 0
    assert result.detected is False
    assert not np.any(result.detected_mask)
    assert result.detected_indices.size == 0


def test_duty_cycle_one_always_online(monkeypatch):
    result, _, _ = run_detector(
        monkeypatch,
        n_images=3,
        detector_kwargs={
            "duty_cycles": {
                "H1": 1.0,
                "L1": 1.0,
                "V1": 1.0,
            },
        },
    )

    for ifo in ("H1", "L1", "V1"):
        assert np.all(result.detector_available[ifo])


def test_unspecified_detector_duty_cycle_defaults_to_one(monkeypatch):
    result, _, _ = run_detector(
        monkeypatch,
        n_images=3,
        detector_kwargs={
            "duty_cycles": {
                "H1": 0.0,
            },
        },
    )

    assert not np.any(result.detector_available["H1"])
    assert np.all(result.detector_available["L1"])
    assert np.all(result.detector_available["V1"])


def test_duty_cycle_sampling_is_reproducible(monkeypatch):
    duty_cycles = {
        "H1": 0.5,
        "L1": 0.7,
        "V1": 0.3,
    }

    result_1, _, _ = run_detector(
        monkeypatch,
        n_images=30,
        detector_kwargs={
            "duty_cycles": duty_cycles,
            "random_seed": 42,
        },
    )

    result_2, _, _ = run_detector(
        monkeypatch,
        n_images=30,
        detector_kwargs={
            "duty_cycles": duty_cycles,
            "random_seed": 42,
        },
    )

    for ifo in ("H1", "L1", "V1"):
        np.testing.assert_array_equal(
            result_1.detector_available[ifo],
            result_2.detector_available[ifo],
        )


# ============================================================
# Input validation
# ============================================================

@pytest.mark.parametrize(
    "duty_cycles",
    [
        {"H1": 1.2},
        {"H1": -0.1},
        {"L1": 1.5},
        {"V1": -0.5},
    ],
)
def test_invalid_duty_cycle_raises_value_error(duty_cycles):
    with pytest.raises(ValueError):
        LVKDetector(duty_cycles=duty_cycles)


@pytest.mark.parametrize("min_detectors", [0, 4, -1])
def test_invalid_min_detectors_raises_value_error(min_detectors):
    with pytest.raises(ValueError):
        LVKDetector(min_detectors=min_detectors)


@pytest.mark.parametrize("num_detected_images", [0, -1])
def test_invalid_num_detected_images_raises_value_error(
    num_detected_images,
):
    with pytest.raises(ValueError):
        LVKDetector(num_detected_images=num_detected_images)