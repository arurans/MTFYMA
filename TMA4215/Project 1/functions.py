import numpy as np
from numba import njit as superspeed

@superspeed
def f1(x):
    return x**2-2

@superspeed
def Df1(x):
    return 2*x

@superspeed
def f2(x):
    return np.exp(x)

@superspeed
def Df2(x):
    return np.exp(x)

#@superspeed
def g(z):
    return z**5-1

#@superspeed
def Dg(z):
    return 5*z**4