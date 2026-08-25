from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, simpson
from tqdm import tqdm

from mml.utils import get_cache_dir
from mml.populations.lens import pi_l

def precompute_M_z(
    z_min=0,
    z_max=3,
    z_n=2000,
    sigma_min=60,
    sigma_max=600,
    sigma_n=2000,
    cache_file=None,
):
    """
    Precompute the redshift-dependent normalisation

        M(z) = integral pi_l(sigma, z) d sigma

    used when sampling z_l | z_s.
    """

    if cache_file is None:
        cache_file = (
            get_cache_dir("mml")
            / "lens_population"
            / "Mz_grid.npz"
        )

    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        with np.load(cache_file) as data:
            return data["z_grid"], data["M_grid"]

    z_grid = np.linspace(z_min, z_max, z_n)
    sigma_grid = np.linspace(
        sigma_min,
        sigma_max,
        sigma_n,
    )

    M_grid = np.zeros_like(z_grid)

    for i, z in enumerate(
        tqdm(z_grid, desc="Computing M(z)")
    ):
        pi_vals = pi_l(sigma_grid, z)
        M_grid[i] = simpson(
            pi_vals,
            x=sigma_grid,
        )

    np.savez(
        cache_file,
        z_grid=z_grid,
        M_grid=M_grid,
    )

    return z_grid, M_grid


def build_cdf(x, pdf):
    """
    Construct a normalised cumulative distribution from a PDF.
    """

    pdf = np.nan_to_num(
        pdf,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    pdf = np.maximum(pdf, 0)

    cdf = cumulative_trapezoid(
        pdf,
        x,
        initial=0,
    )

    cdf = np.maximum.accumulate(cdf)

    if cdf[-1] <= 0:
        raise ValueError("PDF integrates to zero.")

    cdf /= cdf[-1]

    return cdf


# z_l | z_s

def sample_z_l(
    z_source,
    z_grid,
    M_grid,
    rng=None,
):
    """
    Sample lens redshift conditional on a source redshift.

    Parameters
    ----------
    z_source : float
        Source redshift.

    z_grid, M_grid : ndarray
        Precomputed M(z) grid.

    rng : numpy.random.Generator, optional
        Random-number generator.

    Returns
    -------
    float
        Sampled lens redshift.
    """

    if rng is None:
        rng = np.random.default_rng()

    z_eff = min(
        z_source,
        z_grid[-1],
    )

    mask = z_grid < z_eff

    z_sub = z_grid[mask]
    M_sub = M_grid[mask]

    if len(z_sub) < 2:
        return float(z_grid[0])

    cdf = build_cdf(
        z_sub,
        M_sub,
    )

    inv_cdf = interp1d(
        cdf,
        z_sub,
        bounds_error=False,
        fill_value=(
            z_sub[0],
            z_sub[-1],
        ),
    )

    return float(
        inv_cdf(rng.random())
    )

# sigma | z_l

def sample_sigma_given_zl(
    z_l,
    sigma_min=60,
    sigma_max=600,
    sigma_n=2000,
    rng=None,
):
    """
    Sample lens velocity dispersion conditional on lens redshift.
    """

    if rng is None:
        rng = np.random.default_rng()

    sigma_grid = np.linspace(
        sigma_min,
        sigma_max,
        sigma_n,
    )

    pdf = pi_l(
        sigma_grid,
        z_l,
    )

    cdf = build_cdf(
        sigma_grid,
        pdf,
    )

    inv_cdf = interp1d(
        cdf,
        sigma_grid,
        bounds_error=False,
        fill_value=(
            sigma_grid[0],
            sigma_grid[-1],
        ),
    )

    return float(
        inv_cdf(rng.random())
    )

# Full conditional sampler

def sample_sigma_zl_given_zs(
    z_source,
    size=None,
    cache_file="Mz_grid.npz",
    sigma_min=60,
    sigma_max=600,
    sigma_n=2000,
    rng=None,
):
    """
    Sample lens velocity dispersion and redshift conditional on
    source redshift(s).

    Parameters
    ----------
    z_source : float or array_like
        Source redshift(s).

        If a scalar is supplied, it is repeated ``size`` times.

        If an array is supplied, one lens is sampled for each
        source redshift.

    size : int, optional
        Number of samples to generate when ``z_source`` is scalar.

        Required when sampling more than one lens from a single
        source redshift.

        If omitted, a scalar ``z_source`` produces one sample.

    cache_file : str or Path
        Cached M(z) file.

    sigma_min, sigma_max : float
        Velocity-dispersion limits.

    sigma_n : int
        Number of points in the sigma grid.

    rng : numpy.random.Generator, optional
        Random-number generator.

    Returns
    -------
    sigma : ndarray
        Lens velocity dispersions [km/s].

    z_l : ndarray
        Lens redshifts.

    Notes
    -----
    A scalar ``z_source`` with ``size > 1`` is interpreted as
    multiple independent lenses behind sources at the same
    redshift.
    """

    if rng is None:
        rng = np.random.default_rng()

    z_source = np.asarray(
        z_source,
        dtype=float,
    )

    if z_source.ndim == 0:

        # Scalar source redshift.
        if size is None:
            size = 1

        if size < 1:
            raise ValueError(
                "size must be >= 1."
            )

        z_source = np.full(
            size,
            float(z_source),
        )

    else:

        z_source = np.atleast_1d(
            z_source
        )

        if size is not None and size != len(z_source):
            raise ValueError(
                "When z_source is an array, "
                "size must match len(z_source)."
            )

        size = len(z_source)

    z_grid, M_grid = precompute_M_z(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_n=sigma_n,
        cache_file=cache_file,
    )

    sigma = np.empty(size)
    z_l = np.empty(size)

    # Sample each lens
    for i, zs in enumerate(z_source):

        # Handle source redshifts at/below the lower
        # edge of the precomputed grid.
        if zs <= z_grid[0]:

            z_l[i] = z_grid[0]
            sigma[i] = sigma_min
            continue

        # Sample lens redshift.
        z_l[i] = sample_z_l(
            z_source=zs,
            z_grid=z_grid,
            M_grid=M_grid,
            rng=rng,
        )

        # Sample velocity dispersion conditional on z_l.
        sigma[i] = sample_sigma_given_zl(
            z_l=z_l[i],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_n=sigma_n,
            rng=rng,
        )

    return sigma, z_l