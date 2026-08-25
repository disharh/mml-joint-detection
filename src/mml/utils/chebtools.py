"""
Chebyshev-based numerical tools.

This module provides utilities for constructing polynomial
approximations to probability distributions and their inverse
cumulative distribution functions.

The implementation is adapted from Ewoud's code and is used
within the MML lens-population sampling framework.
"""

import numpy as np
from numba import njit
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import brentq


def uniform_sampler_from_2dpdf(
    pdf,
    lims,
    res_cg=None,
):
    """
    Construct Chebyshev representations for inverse-transform
    sampling from a 2D probability density function.

    Parameters
    ----------
    pdf : callable
        Two-dimensional probability density function.

    lims : array-like
        Limits of the two variables:

            [[xmin, xmax],
             [ymin, ymax]]

    res_cg : array-like of int or None, optional
        Resolution of the Chebyshev grids. If None, defaults
        to [140, 140].

    Returns
    -------
    cg_getx : ndarray
        Chebyshev coefficients representing the inverse CDF
        of the marginal distribution of the first variable.

    cg_getz : ndarray
        Chebyshev coefficients representing the conditional
        inverse CDF of the second variable.

    lims : array-like
        Input limits, returned unchanged.
    """

    if res_cg is None:
        res_cg = [140, 140]

    print("Calculating first tables")

    chebgrid = chebinterpolate_2d(
        lambda x, z: pdf(
            u2x(x, *lims[0]),
            u2x(z, *lims[1]),
        ),
        res_cg,
    )

    # Integrate over the second dimension to obtain
    # the marginal distribution of the first variable.
    intdz = cheb.chebint(
        chebgrid,
        axis=1,
        k=0,
        lbnd=-1,
    )

    marginalx = cheb.chebval(
        1,
        intdz.T,
        tensor=False,
    )

    marginalxcdf = cheb.chebint(
        marginalx,
        lbnd=-1,
    )

    marginalxcdf /= cheb.chebval(
        1,
        marginalxcdf,
    )

    # Inverse CDF for the first variable.
    getx = lambda v: brentq(
        lambda x: (
            cheb.chebval(
                x2u(x, *lims[0]),
                marginalxcdf,
            )
            - v
        ),
        *lims[0],
    )

    print("Calculating second tables")

    cg_getx = cheb.chebinterpolate(
        lambda u: np.vectorize(getx)(
            u2x(u, 0, 1)
        ),
        res_cg[0] * 3,
    )

    # Conditional CDF of the second variable.
    def get_z(v, x):

        c_x = cheb.chebval(
            x2u(x, *lims[0]),
            marginalx,
        )

        F_x__z = cheb.chebval(
            x2u(x, *lims[0]),
            intdz / c_x,
            tensor=False,
        )

        z = brentq(
            lambda z: (
                cheb.chebval(
                    x2u(z, *lims[1]),
                    F_x__z,
                )
                - v
            ),
            *lims[1],
        )

        return z

    print("Calculating third tables")

    cg_getz = chebinterpolate_2d(
        np.vectorize(
            lambda v, x: get_z(
                u2x(v, 0, 1),
                u2x(x, *lims[0]),
            )
        ),
        res_cg,
    )

    return cg_getx, cg_getz, lims


def chebinterpolate_2d(
    func,
    deg,
    *args,
):
    """
    Construct 2D Chebyshev coefficients approximating a function.

    The function domain must be [-1, 1] in both dimensions.

    Parameters
    ----------
    func : callable
        Function to approximate.

    deg : array-like of int
        Polynomial degree in each dimension.

    *args
        Additional arguments passed to ``func``.

    Returns
    -------
    ndarray
        Two-dimensional array of Chebyshev coefficients.
    """

    deg = np.asarray(deg)

    if (
        deg.ndim != 1
        or deg.size != 2
        or deg.dtype.kind not in "iu"
    ):
        raise TypeError(
            "deg must be an array-like of 2 integers"
        )

    if np.any(deg < 0):
        raise ValueError(
            "expected deg >= 0"
        )

    deg = deg[::-1]
    order = deg + 1

    xcheb = cheb.chebpts1(order[0])
    ycheb = cheb.chebpts1(order[1])

    xgr, ygr = np.meshgrid(
        xcheb,
        ycheb,
        indexing="ij",
    )

    yfunc = func(
        xgr,
        ygr,
        *args,
    )

    m = cheb.chebvander2d(
        xgr,
        ygr,
        deg,
    )

    c = np.tensordot(
        m,
        yfunc,
        axes=[np.arange(deg.size)] * 2,
    ).reshape(tuple(order))

    c *= 4 / order[0] / order[1]

    c[0] /= 2
    c[:, 0] /= 2

    return c


def ppf_from_1d_pdf(
    pdf,
    lims,
    res=200,
    ret_all=False,
):
    """
    Construct Chebyshev representations of a 1D PDF, CDF,
    and PPF.

    Parameters
    ----------
    pdf : callable
        One-dimensional probability density function.

    lims : array-like
        Lower and upper limits of the variable.

    res : int, optional
        Chebyshev resolution.

    ret_all : bool, optional
        If True, return the PDF, CDF, and PPF coefficients.
        If False, return only the PPF coefficients.

    Returns
    -------
    ndarray or tuple of ndarray
        Chebyshev coefficients.
    """

    chebpdf = cheb.chebinterpolate(
        lambda x: pdf(
            u2x(x, *lims)
        ),
        res,
    )

    chebcdf = cheb.chebint(
        chebpdf,
        lbnd=-1,
    )

    marginal = cheb.chebval(
        1,
        chebcdf,
    )

    print("marginal distr, ", marginal)

    chebpdf /= marginal
    chebcdf /= marginal

    @np.vectorize
    def get_x(v):

        x = u2x(
            brentq(
                lambda u: (
                    cheb.chebval(
                        u,
                        chebcdf,
                    )
                    - v
                ),
                -1,
                1,
            ),
            *lims,
        )

        return x

    chebppf = cheb.chebinterpolate(
        lambda u: get_x(
            u2x(u, 0, 1)
        ),
        res,
    )

    if ret_all:
        return chebpdf, chebcdf, chebppf

    return chebppf


@njit(
    fastmath=True,
    cache=True,
)
def x2u(
    x,
    xmin,
    xmax,
):
    """
    Map a variable from [xmin, xmax] to [-1, 1].
    """

    return (
        (x - xmin)
        / (xmax - xmin)
        * 2
        - 1
    )


@njit(
    fastmath=True,
    cache=True,
)
def u2x(
    u,
    xmin,
    xmax,
):
    """
    Map a variable from [-1, 1] to [xmin, xmax].
    """

    return (
        (u + 1)
        * (xmax - xmin)
        / 2
        + xmin
    )