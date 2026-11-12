import numpy as np
import pytest

from mml.lensing.caustics import get_caustics


@pytest.fixture
def kwargs_lens():
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


@pytest.mark.parametrize("number_images", [2, 3, 4])
def test_get_caustics(kwargs_lens, number_images):
    caustics = get_caustics(
        kwargs_lens,
        number_images,
    )

    assert isinstance(caustics, np.ndarray)
    assert caustics.shape[0] == 2
    assert caustics.shape[1] > 2


def test_get_caustics_invalid_number(kwargs_lens):
    with pytest.raises(ValueError):
        get_caustics(kwargs_lens, 5)