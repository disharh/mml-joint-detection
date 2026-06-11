## Numerical computation utility copied from Ewoud

from numba import njit
import numpy as np


def brentq_nojit(f, xa, xb, xtol=2e-14, rtol=16*np.finfo(float).eps, maxiter=100, args=(), verbose=False):
    """
    A numba-compatible implementation of brentq (largely copied from scipy)
    :param f: function to optimize
    :param xa: left bound
    :param xb: right bound
    :param xtol: x-coord root tolerance
    :param rtol: x-coord relative tolerance
    :param maxiter: maximum num of iterations
    :param args: additional arguments to pass to function in the form f(x, args)
    :param verbose: do some more diagnostic output
    :return: 
    """
    xpre=xa
    xcur=xb
    xblk=0.
    fblk=0.
    spre=0.
    scur=0.
    fpre = f(xpre, args)
    fcur = f(xcur, args)
    funcalls = 2
    if fpre*fcur>0:
        raise ValueError('Signs are not different')
    if fpre == 0:
        if verbose:
            print(funcalls)
        return xpre
    if fcur == 0:
        if verbose:
            print(funcalls)
        return xcur
    iterations = 0
    for i in range(maxiter):
        iterations += 1
        if fpre*fcur < 0:
            xblk = xpre
            fblk = fpre
            spres = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            
            fpre = fcur
            fcur = fblk
            fblk = fpre
            
        delta = (xtol + rtol*abs(xcur))/2
        sbis = (xblk - xcur)/2
        if fcur == 0 or abs(sbis) < delta:
            if verbose:
                print(funcalls)
            return xcur

        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else:
                dpre = (fpre - fcur)/(xpre - xcur)
                dblk = (fblk - fcur)/(xblk - xcur)
                stry = -fcur*(fblk*dblk - fpre*dpre)\
                    /(dblk*dpre*(fblk - fpre))
            
            if 2*abs(stry) < min(abs(spre), 3*abs(sbis) - delta):
                spre = scur
                scur = stry
            else:
                spre = sbis
                scur = sbis
        else:
            spre = sbis
            scur = sbis

        xpre = xcur; fpre = fcur;
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = f(xcur, args)
        funcalls += 1
    
    if verbose:
        print(funcalls)
    return xcur
brentq = njit(cache=True)(brentq_nojit)
brentq_inline = njit(cache=True, inline='always')(brentq_nojit)