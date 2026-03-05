## Copied from Ewoud's code

import numpy as np
from lenstronomy.Util import param_util
from scipy import stats as scs
from scipy.optimize import brentq
from scipy.special._ufuncs import gammaincinv

from numerics import cart2pol, pol2cart

px = lambda x: 1/2*(np.pi+2*x*np.sqrt(1-x**2)+2*np.arcsin(x))/np.pi
ppfx = lambda p: brentq(lambda x: (px(x)-p), -1, 1)
ppfy = lambda x, p: np.sqrt(1-x**2)*(2*p-1)


def sample_sersic_cart(u, re, n, e1, e2, center_x, center_y):
    u = np.asarray(u)
    onedim = u.ndim==1
    if onedim:
        u = u[:, None]
    x = np.array([ppfx(x) for x in u[0]])
    y = ppfy(x, u[1])
    ur = np.sqrt(x**2+y**2)
    uth = np.arctan2(y, x)/(2*np.pi)

    bn = gammaincinv(2*n, 0.5)
    h = re/bn**n

    r = scs.gengamma.ppf(ur, a=n*(1), c=1/n, scale=h)
    t = uth*2*np.pi
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    x, y = r*np.cos(t), q*r*np.sin(t)
    xr = np.cos(phi)*x-np.sin(phi)*y
    yr = np.sin(phi)*x+np.cos(phi)*y
    if onedim:
        return float(np.squeeze(xr+center_x)), float(np.squeeze(yr+center_y))
    else:
        return xr+center_x, yr+center_y



def cart_to_sersic_us(xcaus, ycaus, re, n, e1, e2, center_x, center_y):
    # Is the inverse of sample_sersic_cart.
    xc = xcaus-center_x
    yc = ycaus-center_y
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    xrb = np.cos(-phi)*xc-np.sin(-phi)*yc
    yrb = np.sin(-phi)*xc+np.cos(-phi)*yc
    r, theta = cart2pol((xrb, yrb/q))

    bn = gammaincinv(2*n, 0.5)
    h = re/bn**n
    ur = scs.gengamma.cdf(r, a=n*(1), c=1/n, scale=h)
    uth = theta/(2*np.pi)
    # print(uth)
    xw, yw = pol2cart((ur, uth*2*np.pi))
    # return xw, yw
    cdfs = np.empty((2, len(xw)))
    cdfs[0, :] = cdfx = px(xw)
    cdfs[1, :] = cdfy = 1/2+1/2*yw/np.sqrt(1-xw**2)
    return cdfs