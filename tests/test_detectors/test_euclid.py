# tests/test_detectors/test_euclid.py

import numpy as np

from mml.detectors.em.euclid import (
    EUCLID,
    EuclidConfig,
    EuclidResult,
    calculate_snr,
)


def test_euclid_config():
    assert EUCLID.num_pix == 50
    assert EUCLID.delta_pix == 0.1
    assert EUCLID.fwhm == 0.17
    assert EUCLID.zeropoint == 25.1209
    assert EUCLID.theta_ein_threshold == 0.33


def test_euclid_config_is_frozen():
    try:
        EUCLID.num_pix = 100
        assert False
    except Exception:
        pass


def test_calculate_snr(lens, source, position):
    result = calculate_snr(
        lens,
        source,
        position,
    )

    assert isinstance(result, EuclidResult)

    assert np.isfinite(result.snr_lens)
    assert np.isfinite(result.snr_source)
    assert np.isfinite(result.magnification)
    assert np.isfinite(result.score)

    assert isinstance(result.resolved, (bool, np.bool_))
    assert isinstance(result.source_size_ok, (bool, np.bool_))
    assert isinstance(result.score_ok, (bool, np.bool_))
    assert isinstance(result.observable, (bool, np.bool_))


def test_einstein_radius_cut(lens, source, position):
    lens.theta_ein = 0.2

    result = calculate_snr(
        lens,
        source,
        position,
    )

    assert not result.resolved
    assert not result.observable


def test_source_size_cut(lens, source, position):
    source.Re_maj_source = 1.0

    result = calculate_snr(
        lens,
        source,
        position,
    )

    assert not result.source_size_ok
    assert not result.observable


def test_return_images(lens, source, position):
    result = calculate_snr(
        lens,
        source,
        position,
        return_images=True,
    )

    assert result.image_no_noise is not None
    assert result.image_with_noise is not None
    assert result.source_image is not None

    assert result.image_no_noise.shape == (
        EUCLID.num_pix,
        EUCLID.num_pix,
    )


def test_no_images_by_default(lens, source, position):
    result = calculate_snr(
        lens,
        source,
        position,
    )

    assert result.image_no_noise is None
    assert result.image_with_noise is None
    assert result.source_image is None


def test_observable_logic(lens, source, position):
    result = calculate_snr(
        lens,
        source,
        position,
    )

    assert result.observable == (
        result.resolved
        and result.source_size_ok
        and result.score_ok
    )