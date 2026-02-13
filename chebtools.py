### Adapted from Ewoud's code
## Tools to implement inverse transform sampling in 2D using Chebyshev polynomials for (conditional)CDFs

import numpy as np
from numba import njit
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import brentq

def uniform_sampler_from_2dpdf(pdf, lims, res_cg=[140, 140]):
    """
    Builds the chebyshev coefficients of the 2d PPF of pdf
    :param pdf: Function to inverse-transform sample from
    :param lims: Limits of the function
    :param res_cg: Resolution of the Chebyshev grid
    :return:
    """
    # intdz
    print("Calculating first tables")
    chebgrid = chebinterpolate_2d(lambda x, z: pdf(u2x(x, *lims[0]), u2x(z, *lims[1])), res_cg)
    intdz = cheb.chebint(chebgrid, axis=1, k=0, lbnd=-1)
    marginalx = cheb.chebval(1, intdz.T, tensor=False)
    marginalxcdf = cheb.chebint(marginalx, lbnd=-1)
    marginalxcdf /= cheb.chebval(1, marginalxcdf)


    getx = lambda v: brentq(lambda x: cheb.chebval(x2u(x, *lims[0]), marginalxcdf) - v, *lims[0])
    print("Calculating second tables")
    cg_getx = cheb.chebinterpolate(lambda u: np.vectorize(getx)(u2x(u, 0, 1)), res_cg[0] * 3)

    def get_z(v, x):
        c_x = cheb.chebval(x2u(x, *lims[0]), marginalx)

        F_x__z = cheb.chebval(x2u(x, *lims[0]), intdz / c_x, tensor=False)

        z = brentq(lambda z: cheb.chebval(x2u(z, *lims[1]), F_x__z) - v, *lims[1])
        return z

    print("Calculating third tables")
    cg_getz = chebinterpolate_2d(np.vectorize(lambda v, x: get_z(u2x(v, 0, 1), u2x(x, *lims[0]))), res_cg)
    return cg_getx, cg_getz, lims


def chebinterpolate_2d(func, deg, *args):
    """
    Builds the 2d chebyshev coefficients that approximate the function func (func's domain must be from -1 to 1)
    :param func: Function to approximate
    :param deg: Degree (n_x, n_y)
    :param args: Additional arguments to pass to the function
    :return:
    """
    deg = np.asarray(deg)

    # check arguments.
    if deg.ndim != 1 or deg.size != 2 or deg.dtype.kind not in 'iu':
        raise TypeError("deg must be an array-like of 2 integers")
    if any(deg < 0):
        raise ValueError("expected deg >= 0")

    deg = deg[::-1]
    order = deg + 1
    xcheb = cheb.chebpts1(order[0])
    ycheb = cheb.chebpts1(order[1])
    xgr, ygr = np.meshgrid(xcheb, ycheb, indexing='ij')
    yfunc = func(xgr, ygr, *args)
    m = cheb.chebvander2d(xgr, ygr, deg)
    c = np.tensordot(m, yfunc, axes=[np.arange(deg.size)] * 2).reshape(tuple(order))
    c *= 4 / order[0] / order[1]
    c[0] /= 2
    c[:, 0] /= 2
    return c


def ppf_from_1d_pdf(pdf, lims, res=200, ret_all=False):
    """
    Builds the chebyshev coefficients for the pdf, cdf and ppf of an arbitrary 1-dimensional PDF
    :param pdf: The PDF function
    :param lims: The parameter limits
    :param res: The resolution of the chebyshev grid
    :param ret_all: Whether to return (pdf, cdf, ppf) or just the ppf
    :return:
    """
    chebpdf = cheb.chebinterpolate(lambda x: pdf(u2x(x, *lims)), res)
    #if fix_yint:
        #chebpdf[0] -= cheb.chebval(-1, chebpdf)
    chebcdf = cheb.chebint(chebpdf, lbnd=-1)
    marginal = cheb.chebval(1, chebcdf)
    print('marginal distr, ', marginal)
    chebpdf /= marginal
    chebcdf /= marginal
    #chebcdf /= marginal
    @np.vectorize
    def get_x(v):
        x = u2x(brentq(lambda u: cheb.chebval(u, chebcdf)-v, -1, 1), *lims)
        return x
    chebppf = cheb.chebinterpolate(lambda u: get_x(u2x(u,0,1)), res)
    return (chebpdf, chebcdf, chebppf) if ret_all else chebppf


@njit(fastmath=True, cache=True)
def x2u(x,xmin,xmax):
    """Convenience function for usage with chebyshev."""
    return (x - xmin) / (xmax - xmin) * 2 - 1


@njit(fastmath=True, cache=True)
def u2x(u, xmin, xmax):
    return (u + 1) * (xmax - xmin) / 2 + xmin