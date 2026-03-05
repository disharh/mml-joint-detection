# Modified numerics.py from Ewoud's code

import numpy as np
from numba import njit, float64, guvectorize

@njit(cache=True, error_model='numpy', fastmath=True)
def solvequadeq_single(a, b, c):
    """Solve a*x^2 + b*x + c = 0 for scalar inputs, numerically stable"""
    if a == 0:
        return -c/b, -c/b + 1e-8
    if b == 0:
        return (-c/a)**0.5, -(-c/a)**0.5
    sD = (b**2 - 4*a*c)**0.5
    x1 = (-b - np.sign(b)*sD)/(2*a)
    x2 = 2*c/(-b - np.sign(b)*sD)
    return x1, x2

@njit(cache=True, error_model='numpy', fastmath=True)
def solvequadeq_arr(a, b, c):
    """Solve a*x^2 + b*x + c = 0 for array inputs"""
    sD = np.sqrt(b**2 - 4*a*c)
    x1 = (-b - np.sign(b)*sD)/(2*a)
    x2 = 2*c/(-b - np.sign(b)*sD)
    # handle a==0 or b==0
    x1 = np.where(b != 0, np.where(a != 0, x1, -c/b), -np.sqrt(-c/a))
    x2 = np.where(b != 0, np.where(a != 0, x2, -c/b+1e-8), +np.sqrt(-c/a))
    return x1, x2

def solvequadeq(a, b, c):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) or isinstance(c, np.ndarray):
        return solvequadeq_arr(a, b, c)
    else:
        return solvequadeq_single(a, b, c)
    
@njit(cache=True, fastmath=True)
def rotmat(th):
    """2D rotation matrix for angle th"""
    return np.array([[np.cos(th), np.sin(th)],
                     [-np.sin(th), np.cos(th)]])
@njit(cache=True)
def cdot(a, b):
    """Dot product of complex numbers"""
    return a.real*b.real + a.imag*b.imag

@njit(cache=True)
def ps(x, p):
    """Signed power: |x|^p * sign(x)"""
    return np.abs(x)**p * np.sign(x)

@njit(fastmath=True, cache=True)
def cart2pol(x):
    """Convert [x, y] -> (r, theta)"""
    return np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0]) % (2*np.pi)

@njit(fastmath=True, cache=True)
def pol2cart(x):
    """Convert (r, theta) -> (x, y)"""
    return x[0]*np.cos(x[1]), x[0]*np.sin(x[1])

@njit(fastmath=True, cache=True)
def polyarea(x, y):
    """Area of polygon with vertices (x, y)"""
    return 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
