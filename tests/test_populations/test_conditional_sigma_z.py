import numpy as np
import pytest

from mml.populations.conditional_sigma_z import (
    build_cdf,
    precompute_M_z,
    sample_z_l,
    sample_sigma_given_zl,
    sample_sigma_zl_given_zs,
)


# ---------------------------------------------------------------------------
# build_cdf
# ---------------------------------------------------------------------------

def test_build_cdf_is_normalized():

    x = np.linspace(0, 1, 100)
    pdf = np.ones_like(x)

    cdf = build_cdf(x, pdf)

    assert cdf.shape == x.shape
    assert np.isclose(cdf[0], 0.0)
    assert np.isclose(cdf[-1], 1.0)


def test_build_cdf_is_monotonic():

    x = np.linspace(0, 1, 100)
    pdf = np.exp(-x)

    cdf = build_cdf(x, pdf)

    assert np.all(np.diff(cdf) >= 0)


def test_build_cdf_rejects_zero_pdf():

    x = np.linspace(0, 1, 10)
    pdf = np.zeros_like(x)

    with pytest.raises(ValueError):
        build_cdf(x, pdf)


# ---------------------------------------------------------------------------
# M(z) precomputation
# ---------------------------------------------------------------------------

def test_precompute_M_z(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    z_grid, M_grid = precompute_M_z(
        z_min=0.0,
        z_max=1.0,
        z_n=20,
        sigma_min=60,
        sigma_max=600,
        sigma_n=20,
        cache_file=cache_file,
    )

    assert z_grid.shape == (20,)
    assert M_grid.shape == (20,)

    assert np.all(np.isfinite(z_grid))
    assert np.all(np.isfinite(M_grid))

    assert np.all(M_grid >= 0)

    assert cache_file.exists()


def test_precompute_M_z_uses_cache(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    z1, M1 = precompute_M_z(
        z_min=0.0,
        z_max=1.0,
        z_n=10,
        sigma_min=60,
        sigma_max=600,
        sigma_n=10,
        cache_file=cache_file,
    )

    z2, M2 = precompute_M_z(
        z_min=0.0,
        z_max=1.0,
        z_n=10,
        sigma_min=60,
        sigma_max=600,
        sigma_n=10,
        cache_file=cache_file,
    )

    assert np.array_equal(z1, z2)
    assert np.array_equal(M1, M2)


# ---------------------------------------------------------------------------
# z_l | z_s
# ---------------------------------------------------------------------------

def test_sample_z_l_bounds():

    z_grid = np.linspace(0, 3, 100)
    M_grid = np.exp(z_grid)

    rng = np.random.default_rng(1234)

    zl = sample_z_l(
        z_source=1.0,
        z_grid=z_grid,
        M_grid=M_grid,
        rng=rng,
    )

    assert 0 <= zl <= 1.0


def test_sample_z_l_is_below_source():

    z_grid = np.linspace(0, 3, 100)
    M_grid = np.exp(z_grid)

    rng = np.random.default_rng(1234)

    samples = np.array([
        sample_z_l(
            z_source=1.0,
            z_grid=z_grid,
            M_grid=M_grid,
            rng=rng,
        )
        for _ in range(100)
    ])

    assert np.all(samples >= 0)
    assert np.all(samples <= 1.0)


def test_sample_z_l_reproducible():

    z_grid = np.linspace(0, 3, 100)
    M_grid = np.exp(z_grid)

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    samples1 = np.array([
        sample_z_l(
            1.0,
            z_grid,
            M_grid,
            rng=rng1,
        )
        for _ in range(20)
    ])

    samples2 = np.array([
        sample_z_l(
            1.0,
            z_grid,
            M_grid,
            rng=rng2,
        )
        for _ in range(20)
    ])

    assert np.array_equal(samples1, samples2)


# ---------------------------------------------------------------------------
# sigma | z_l
# ---------------------------------------------------------------------------

def test_sample_sigma_given_zl_bounds():

    rng = np.random.default_rng(1234)

    samples = np.array([
        sample_sigma_given_zl(
            z_l=0.5,
            sigma_min=60,
            sigma_max=600,
            sigma_n=100,
            rng=rng,
        )
        for _ in range(100)
    ])

    assert np.all(samples >= 60)
    assert np.all(samples <= 600)


def test_sample_sigma_given_zl_returns_finite():

    rng = np.random.default_rng(1234)

    sigma = sample_sigma_given_zl(
        z_l=0.5,
        sigma_min=60,
        sigma_max=600,
        sigma_n=100,
        rng=rng,
    )

    assert np.isfinite(sigma)


def test_sample_sigma_given_zl_reproducible():

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)

    samples1 = np.array([
        sample_sigma_given_zl(
            z_l=0.5,
            sigma_min=60,
            sigma_max=600,
            sigma_n=100,
            rng=rng1,
        )
        for _ in range(20)
    ])

    samples2 = np.array([
        sample_sigma_given_zl(
            z_l=0.5,
            sigma_min=60,
            sigma_max=600,
            sigma_n=100,
            rng=rng2,
        )
        for _ in range(20)
    ])

    assert np.array_equal(samples1, samples2)


# ---------------------------------------------------------------------------
# Full conditional sampler
# ---------------------------------------------------------------------------

def test_sample_sigma_zl_given_zs_scalar_source_size(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    rng = np.random.default_rng(1234)

    sigma, zl = sample_sigma_zl_given_zs(
        z_source=1.0,
        size=10,
        cache_file=cache_file,
        sigma_min=60,
        sigma_max=600,
        sigma_n=100,
        rng=rng,
    )

    assert sigma.shape == (10,)
    assert zl.shape == (10,)

    assert np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(zl))


def test_sample_sigma_zl_given_zs_array_source(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    rng = np.random.default_rng(1234)

    z_source = np.array([
        0.5,
        1.0,
        1.5,
    ])

    sigma, zl = sample_sigma_zl_given_zs(
        z_source=z_source,
        cache_file=cache_file,
        sigma_min=60,
        sigma_max=600,
        sigma_n=100,
        rng=rng,
    )

    assert sigma.shape == (3,)
    assert zl.shape == (3,)


def test_sample_sigma_zl_given_zs_scalar_source_defaults_to_one(
    tmp_path,
):

    cache_file = tmp_path / "Mz_grid.npz"

    rng = np.random.default_rng(1234)

    sigma, zl = sample_sigma_zl_given_zs(
        z_source=1.0,
        cache_file=cache_file,
        sigma_min=60,
        sigma_max=600,
        sigma_n=100,
        rng=rng,
    )

    assert sigma.shape == (1,)
    assert zl.shape == (1,)


def test_sample_sigma_zl_given_zs_size_mismatch(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    with pytest.raises(ValueError, match="size must match"):
        sample_sigma_zl_given_zs(
            z_source=np.array([0.5, 1.0, 1.5]),
            size=2,
            cache_file=cache_file,
            sigma_n=50,
        )


def test_sample_sigma_zl_given_zs_invalid_size(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    with pytest.raises(ValueError, match="size must be >= 1"):
        sample_sigma_zl_given_zs(
            z_source=1.0,
            size=0,
            cache_file=cache_file,
            sigma_n=50,
        )


def test_sample_sigma_zl_given_zs_bounds(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    rng = np.random.default_rng(1234)

    z_source = np.array([
        0.5,
        1.0,
        1.5,
    ])

    sigma, zl = sample_sigma_zl_given_zs(
        z_source=z_source,
        cache_file=cache_file,
        sigma_min=60,
        sigma_max=600,
        sigma_n=100,
        rng=rng,
    )

    assert np.all(sigma >= 60)
    assert np.all(sigma <= 600)

    assert np.all(zl >= 0)
    assert np.all(zl <= z_source)


def test_sample_sigma_zl_given_zs_scalar_source_bounds(
    tmp_path,
):

    cache_file = tmp_path / "Mz_grid.npz"

    rng = np.random.default_rng(1234)

    sigma, zl = sample_sigma_zl_given_zs(
        z_source=1.0,
        size=100,
        cache_file=cache_file,
        sigma_min=60,
        sigma_max=600,
        sigma_n=100,
        rng=rng,
    )

    assert np.all(sigma >= 60)
    assert np.all(sigma <= 600)

    assert np.all(zl >= 0)
    assert np.all(zl <= 1.0)


def test_sample_sigma_zl_given_zs_finite(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    rng = np.random.default_rng(1234)

    sigma, zl = sample_sigma_zl_given_zs(
        z_source=np.array([
            0.5,
            1.0,
            2.0,
        ]),
        cache_file=cache_file,
        sigma_n=100,
        rng=rng,
    )

    assert np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(zl))


def test_sample_sigma_zl_given_zs_reproducible(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    sigma1, zl1 = sample_sigma_zl_given_zs(
        z_source=1.0,
        size=10,
        cache_file=cache_file,
        sigma_n=100,
        rng=np.random.default_rng(1234),
    )

    sigma2, zl2 = sample_sigma_zl_given_zs(
        z_source=1.0,
        size=10,
        cache_file=cache_file,
        sigma_n=100,
        rng=np.random.default_rng(1234),
    )

    assert np.array_equal(sigma1, sigma2)
    assert np.array_equal(zl1, zl2)


def test_low_source_redshift(tmp_path):

    cache_file = tmp_path / "Mz_grid.npz"

    sigma, zl = sample_sigma_zl_given_zs(
        z_source=0.0,
        size=5,
        cache_file=cache_file,
        sigma_n=50,
    )

    assert sigma.shape == (5,)
    assert zl.shape == (5,)

    assert np.all(sigma == 60)
    assert np.all(zl == 0)
