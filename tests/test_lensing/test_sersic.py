import numpy as np

from mml.lensing.sersic import sample_sersic_cart


def test_sample_sersic_cart_returns_scalar():
    x, y = sample_sersic_cart(
        np.array([0.5, 0.5]),
        re=0.5,
        n=2.0,
        e1=0.1,
        e2=0.05,
        center_x=1.0,
        center_y=-1.0,
    )

    assert np.isfinite(x)
    assert np.isfinite(y)


def test_sample_sersic_cart_vectorized():
    rng = np.random.default_rng(42)

    u = rng.uniform(0, 1, size=(100, 2))

    x, y = sample_sersic_cart(
        u.T,
        re=0.5,
        n=2.0,
        e1=0.1,
        e2=0.05,
        center_x=1.0,
        center_y=-1.0,
    )

    assert x.shape == (100,)
    assert y.shape == (100,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))