import numpy as np

from mml.utils.numerics import (
    solvequadeq,
    cart2pol,
    pol2cart,
    polyarea,
)


def test_solvequadeq():
    roots = solvequadeq(1.0, -3.0, 2.0)

    assert np.isclose(roots[0], 1.0) or np.isclose(roots[0], 2.0)
    assert np.isclose(roots[1], 1.0) or np.isclose(roots[1], 2.0)
    assert np.allclose(np.sort(roots), [1.0, 2.0])

def test_cart2pol_pol2cart_roundtrip():
    xy = np.array([1.0, 1.0])

    r, theta = cart2pol(xy)
    recovered = pol2cart((r, theta))

    assert np.allclose(recovered, xy)


def test_polyarea_square():
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    area = polyarea(x, y)

    assert np.isclose(area, 1.0)