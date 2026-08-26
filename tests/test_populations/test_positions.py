import numpy as np

from mml.populations import (
    BBHPositionSample,
    BBHPositionSampler,
)


def lens_kwargs():
    return [
        {
            "theta_E": 1.0,
            "gamma": 2.0,
            "e1": 0.1,
            "e2": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        },
        {
            "gamma1": 0.02,
            "gamma2": 0.01,
        },
    ]


def source_kwargs():
    return {
        "re_source": 0.5,
        "nsersic_source": 2.0,
        "e1_source": 0.1,
        "e2_source": 0.05,
    }


def test_position_sampler_creation():
    sampler = BBHPositionSampler(
        rng=np.random.default_rng(42)
    )

    assert sampler is not None


def test_position_sample():
    sampler = BBHPositionSampler(
        rng=np.random.default_rng(42)
    )

    sample = sampler.sample(
        kwargs_lens=lens_kwargs(),
        kwargs_source=source_kwargs(),
        num_detected_gws=4,
    )

    assert isinstance(sample, BBHPositionSample)

    assert np.isfinite(sample.x_gw)
    assert np.isfinite(sample.y_gw)
    assert np.isfinite(sample.caustic_area)
    assert np.isfinite(sample.x_source)
    assert np.isfinite(sample.y_source)

    assert sample.caustic_area > 0


def test_position_sample_to_dict():
    sampler = BBHPositionSampler(
        rng=np.random.default_rng(42)
    )

    sample = sampler.sample(
        lens_kwargs(),
        source_kwargs(),
        4,
    )

    data = sample.to_dict()

    assert set(data) == {
        "x_gw",
        "y_gw",
        "caustic_area",
        "x_source",
        "y_source",
    }


def test_position_population():
    sampler = BBHPositionSampler(
        rng=np.random.default_rng(42)
    )

    samples = sampler.sample_population(
        kwargs_lens=lens_kwargs(),
        kwargs_source=source_kwargs(),
        num_detected_gws=4,
        size=20,
    )

    assert set(samples) == {
        "x_gw",
        "y_gw",
        "caustic_area",
        "x_source",
        "y_source",
    }

    for values in samples.values():
        assert values.shape == (20,)
        assert np.all(np.isfinite(values))