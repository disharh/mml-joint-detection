import numpy as np
from mml.lensing.polygon import  sample_polygon_single


def test_sample_polygon_single_inside_square():
    polygon = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    (x, y), area = sample_polygon_single(
        polygon,
        np.array([0.5, 0.5]),
    )

    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert np.isclose(area, 1.0)
